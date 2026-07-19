"""
search_engine.py — Multithreaded, combinatorial search across the two SIR
timeline indices (2002 baseline vs 2025 current roll).

Key properties:
  1. Multithreaded: a ThreadPoolExecutor fires the query at BOTH indices
     concurrently (one thread per timeline) and merges results.
  2. Combinatorial search: EPIC No / Door No / Voter Name / Relation Name
     can each be supplied independently, and the query builder combines
     whichever subset is given — pure OR across fields, or AND if the
     caller asks for stricter matching, or "at least N of these fields
     must match" (a true combinatorics-style match_count control).
  3. Prefix / autocomplete: 3+ letters of a name return matches, powered
     by the edge_ngram field built in es_sync.py.
  4. Fuzzy tolerance: handles OCR / transliteration spelling drift common
     in scanned electoral rolls.

RANKING (house-number / any single-field search)
---------------------------------------------------
Every field clause is a tier of sub-clauses, in descending priority:
    EXACT (term on the lowercase-normalized `.raw` keyword field, heavily
           boosted via cfg.exact_match_boost) > SEARCH (plain tokenized
           match) > PREFIX (edge_ngram, for partial/autocomplete input) >
           FUZZY (Levenshtein tolerance, run against a non-ngram field so
           "closest value" actually means closest to the whole value, not
           closest to an arbitrary n-gram fragment).
This guarantees an exact (or very close) match rises to the top of a
result set regardless of how BM25 scores the broader recall clauses below
it — important when a single-field search (e.g. house/door number alone)
can pull back hundreds of loosely-related hits via prefix/fuzzy recall.

NOTE: this mapping change (the `.raw` normalizer) requires the ES indices
to be rebuilt — run `python main.py sync --timeline both --force` (or
`es_sync.py --force`) after deploying this, or the old index still has the
previous mapping and exact-match boosting won't work correctly.

NAME + RELATION NAME (AND, not OR)
-------------------------------------
When BOTH voter_name and relation_name are supplied, they are combined
into a single `bool.must` sub-clause — i.e. a record has to satisfy BOTH
to be returned at all — regardless of the overall `combine_mode` used for
other fields (age/gender/part_no/etc.). An unqualified name match alone,
or an unqualified relation-name match alone, isn't a reliable identity
signal; the pair together is. If only one of the two is supplied, normal
single-field recall behaviour applies unchanged.

VOTER ID (EPIC) SEARCH
-------------------------
EPIC search, when it's the ONLY field supplied (the common "look up this
one voter" case), is a LOOKUP rather than a relevance-ranked recall net —
see SIRSearchEngine._epic_lookup(): try an exact match first; only if that
comes back empty, fall back to the single closest fuzzy candidate, and
only if it clears cfg.epic_fuzzy_min_score; otherwise return nothing at
all rather than an arbitrary "least-bad" guess. If epic_no is combined
with other fields in one combinatorial query, it still uses a plain exact
term clause (unchanged) — the exact/fuzzy/nothing tri-state logic is
sequential decision-making that doesn't compose cleanly into a single
OR/AND/MIN_N query alongside independent field clauses; flag if you also
want fuzzy-EPIC blended into combined queries and we can look at that
separately.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Literal

from elasticsearch import Elasticsearch
from rapidfuzz import fuzz

from config import SIRConfig
from name_variants import generate_variants

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("search_engine")


@dataclass
class SearchQuery:
    """All fields optional — combinatorics engine works with any subset."""
    epic_no: Optional[str] = None
    door_no: Optional[str] = None
    voter_name: Optional[str] = None
    relation_name: Optional[str] = None
    constituency: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    part_no: Optional[str] = None
    # combine_mode:
    #   "OR"      -> match ANY supplied field (broadest recall)
    #   "AND"     -> match ALL supplied fields (strict)
    #   "MIN_N"   -> at least `min_match` of the supplied fields must match
    # NOTE: voter_name + relation_name supplied together are ALWAYS ANDed
    # together as a single combined clause regardless of this setting -
    # see module docstring.
    combine_mode: Literal["OR", "AND", "MIN_N"] = "OR"
    min_match: int = 2   # used only when combine_mode == "MIN_N"


class SIRQueryBuilder:
    """Builds an Elasticsearch bool query out of a SearchQuery, handling
    exact/prefix/fuzzy tiering, prefix search, and fuzzy tolerance."""

    def __init__(self, cfg: SIRConfig):
        self.cfg = cfg

    def _single_value_clause(self, field: str, value: str, boost: float = 1.0) -> List[Dict[str, Any]]:
        """Exact / search / prefix / fuzzy clause tier for one literal
        value - see module docstring for why each tier exists and how
        they're prioritized relative to each other."""
        raw_field = f"{field}.raw"
        value_lower = value.lower()

        clauses: List[Dict[str, Any]] = [
            # EXACT - unanalyzed, lowercase-normalized keyword match. Heavily
            # boosted so an exact hit always outranks everything below it.
            {"term": {raw_field: {"value": value_lower, "boost": boost * self.cfg.exact_match_boost}}},
            # SEARCH - plain tokenized match (search_analyzer: lowercase only).
            {"match": {field: {"query": value, "analyzer": "search_analyzer", "boost": boost}}},
        ]
        if len(value) >= self.cfg.prefix_min_chars:
            # PREFIX - matches the edge_ngram field built at index time, for
            # partial/autocomplete input.
            clauses.append({"match": {field: {"query": value, "boost": boost}}})

        # FUZZY - Levenshtein tolerance for OCR/typo drift. Run against a
        # NON-ngram field: `.std` (plain text) for names, `.raw` (keyword)
        # for door_no - fuzzy-matching against edge_ngram tokens doesn't
        # behave like whole-value similarity, which is what "closest
        # matching house number" actually means.
        if field in ("voter_name", "relation_name"):
            clauses.append({
                "match": {
                    f"{field}.std": {
                        "query": value, "fuzziness": self.cfg.fuzziness,
                        "prefix_length": 1, "boost": boost * 0.4,
                    }
                }
            })
        else:
            clauses.append({
                "fuzzy": {
                    raw_field: {
                        "value": value_lower, "fuzziness": self.cfg.fuzziness,
                        "prefix_length": 1, "boost": boost * 0.4,
                    }
                }
            })
        return clauses

    def _name_clause(self, field: str, value: str, expand_variants: bool = False) -> Dict[str, Any]:
        """
        Uses match_phrase_prefix-like behaviour via our edge_ngram field for
        short partial input (>= prefix_min_chars), and falls back to a fuzzy
        match for full names/typos, with an exact-match tier on top (see
        _single_value_clause).

        When expand_variants=True (voter_name / relation_name only), also
        generates spelling-transliteration and nickname alternates via
        name_variants.generate_variants() and searches all of them in the
        same query, so e.g. typing "Ravindra" also surfaces records stored
        as "Raveendra" or "Ravi". The as-typed value keeps a higher boost
        so exact/close matches still rank above nickname guesses.
        """
        value = value.strip()
        should: List[Dict[str, Any]] = []

        if expand_variants:
            variants = generate_variants(value)
        else:
            variants = [value]

        for i, v in enumerate(variants):
            boost = 1.0 if i == 0 else 0.5   # as-typed value ranks above generated variants
            should.extend(self._single_value_clause(field, v, boost=boost))

        return {"bool": {"should": should, "minimum_should_match": 1}}

    def _epic_exact_clause(self, value: str) -> Dict[str, Any]:
        """Plain exact-only EPIC clause, used when epic_no is combined with
        OTHER fields in one combinatorial query. For an EPIC-only query,
        SIRSearchEngine._epic_lookup() is used instead (exact -> fuzzy
        fallback -> nothing; see module docstring)."""
        return {"term": {"epic_no": value.strip().lower()}}

    def build(self, q: SearchQuery) -> Dict[str, Any]:
        field_clauses: List[Dict[str, Any]] = []

        if q.epic_no:
            field_clauses.append(self._epic_exact_clause(q.epic_no))
        if q.door_no:
            field_clauses.append(self._name_clause("door_no", q.door_no))
        if q.constituency:
            field_clauses.append(self._name_clause("constituency", q.constituency))

        # voter_name + relation_name together => must BOTH match (see module
        # docstring) - treated as one combined clause regardless of the
        # combine_mode used for other fields.
        if q.voter_name and q.relation_name:
            field_clauses.append({
                "bool": {"must": [
                    self._name_clause("voter_name", q.voter_name, expand_variants=True),
                    self._name_clause("relation_name", q.relation_name, expand_variants=True),
                ]}
            })
        elif q.voter_name:
            field_clauses.append(self._name_clause("voter_name", q.voter_name, expand_variants=True))
        elif q.relation_name:
            field_clauses.append(self._name_clause("relation_name", q.relation_name, expand_variants=True))

        if q.age is not None:
            field_clauses.append({"term": {"age": q.age}})
        if q.gender:
            field_clauses.append({"term": {"gender": q.gender.upper()}})
        if q.part_no:
            field_clauses.append({"term": {"part_no": str(q.part_no)}})

        if not field_clauses:
            raise ValueError("At least one search field must be supplied.")

        if q.combine_mode == "AND":
            body = {"bool": {"must": field_clauses}}
        elif q.combine_mode == "MIN_N":
            body = {"bool": {"should": field_clauses,
                              "minimum_should_match": min(q.min_match, len(field_clauses))}}
        else:  # OR
            body = {"bool": {"should": field_clauses, "minimum_should_match": 1}}

        return {"query": body, "size": self.cfg.result_size_per_index}


class SIRSearchEngine:
    """
    Spawns one worker thread per timeline index and searches them
    concurrently, then merges + ranks + tags results with their
    source timeline (2002 / 2025) so the caller can see both sides
    of the SIR comparison in a single call.
    """

    def __init__(self, cfg: SIRConfig):
        self.cfg = cfg
        self.es = Elasticsearch(cfg.es_hosts, basic_auth=(cfg.es_username, cfg.es_password)
                                 if cfg.es_username else None,
                                 request_timeout=cfg.es_request_timeout,
                                 retry_on_timeout=True, max_retries=3)
        self.qb = SIRQueryBuilder(cfg)

    def _search_index(self, index_name: str, timeline_label: str, body: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            resp = self.es.search(index=index_name, body=body)
        except Exception as e:
            log.error(f"Search failed on index '{index_name}': {e}")
            return []
        hits = []
        for h in resp["hits"]["hits"]:
            rec = h["_source"]
            rec["_score"] = h["_score"]
            rec["_timeline"] = timeline_label
            hits.append(rec)
        return hits

    def _search_both(self, body: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Fires the same query body at both timeline indices concurrently."""
        targets = [
            (self.cfg.es_index_2025, "2025"),
            (self.cfg.es_index_2002, "2002"),
        ]
        results: Dict[str, List[Dict[str, Any]]] = {"2025": [], "2002": []}
        with ThreadPoolExecutor(max_workers=max(self.cfg.max_search_workers, len(targets))) as pool:
            future_map = {
                pool.submit(self._search_index, idx, label, body): label
                for idx, label in targets
            }
            for future in as_completed(future_map):
                label = future_map[future]
                results[label] = future.result()
        return results

    def _epic_lookup(self, epic_value: str) -> List[Dict[str, Any]]:
        """Voter ID search is a LOOKUP, not a relevance-ranked recall net:
        try an exact match first across both timelines; only if that comes
        back completely empty, fall back to the single closest fuzzy
        candidate - and only if it's actually close, so a wildly different
        EPIC is never shown just because it's the "least-bad" option.
        Returns [] if nothing reasonable is found.

        Elasticsearch is used here purely for CANDIDATE RETRIEVAL (its
        fuzzy query narrows ~1M+ documents down to a handful efficiently);
        the actual accept/reject decision uses rapidfuzz directly on the
        real EPIC strings (cfg.epic_fuzzy_min_similarity, 0-100 scale) since
        ES's own _score isn't a portable scale to threshold against."""
        value = epic_value.strip().lower()

        exact_body = {"query": {"term": {"epic_no": value}}, "size": self.cfg.result_size_per_index}
        exact = self._search_both(exact_body)
        exact_hits = exact["2025"] + exact["2002"]
        if exact_hits:
            exact_hits.sort(key=lambda r: r.get("_score", 0), reverse=True)
            return exact_hits

        fuzzy_body = {
            "query": {"fuzzy": {"epic_no": {"value": value, "fuzziness": self.cfg.fuzziness, "prefix_length": 1}}},
            "size": self.cfg.result_size_per_index,
        }
        fuzzy = self._search_both(fuzzy_body)
        fuzzy_hits = fuzzy["2025"] + fuzzy["2002"]
        if not fuzzy_hits:
            return []

        for hit in fuzzy_hits:
            hit["_epic_similarity"] = fuzz.ratio(value, str(hit.get("epic_no", "")).lower())
        fuzzy_hits.sort(key=lambda r: r["_epic_similarity"], reverse=True)
        top = fuzzy_hits[0]
        if top["_epic_similarity"] < self.cfg.epic_fuzzy_min_similarity:
            return []   # nothing close enough to be worth showing as "the closest match"
        return [top]

    def search(self, query: SearchQuery) -> Dict[str, Any]:
        only_epic = bool(query.epic_no) and not any([
            query.door_no, query.voter_name, query.relation_name, query.constituency,
            query.age is not None, query.gender, query.part_no,
        ])

        if only_epic:
            merged = self._epic_lookup(query.epic_no)
            return {
                "query": asdict(query),
                "count_2025": sum(1 for r in merged if r.get("_timeline") == "2025"),
                "count_2002": sum(1 for r in merged if r.get("_timeline") == "2002"),
                "merged_results": merged,
                "name_variants_tried": [],
                "relation_variants_tried": [],
            }

        body = self.qb.build(query)
        results = self._search_both(body)
        merged = self._merge_and_rank(results["2025"], results["2002"])
        return {
            "query": asdict(query),
            "count_2025": len(results["2025"]),
            "count_2002": len(results["2002"]),
            "merged_results": merged,
            "name_variants_tried": generate_variants(query.voter_name) if query.voter_name else [],
            "relation_variants_tried": generate_variants(query.relation_name) if query.relation_name else [],
        }

    @staticmethod
    def _merge_and_rank(hits_2025: List[Dict[str, Any]], hits_2002: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        combined = hits_2025 + hits_2002
        combined.sort(key=lambda r: r.get("_score", 0), reverse=True)
        return combined


# ---------------------------------------------------------------------------
# quick manual test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = SIRConfig()
    engine = SIRSearchEngine(cfg)

    # Example 1: pure prefix name search (3 letters)
    q1 = SearchQuery(voter_name="Sur", combine_mode="OR")
    print(engine.search(q1))

    # Example 2: house-number-only search - exact/closest door number should
    # rank first even among a large loosely-matching result set.
    q2 = SearchQuery(door_no="12-45")
    print(engine.search(q2))

    # Example 3: name + relation name together - AND enforced automatically.
    q3 = SearchQuery(voter_name="Suresh Rao", relation_name="Ganapathi Rao")
    print(engine.search(q3))

    # Example 4: EPIC-only lookup - exact, else closest, else nothing.
    q4 = SearchQuery(epic_no="KA1234567")
    print(engine.search(q4))

    # Example 5: combinatorial — EPIC OR (name AND door_no)
    q5 = SearchQuery(epic_no="KA1234567", voter_name="Suresh Rao",
                      door_no="12-45", combine_mode="MIN_N", min_match=2)
    print(engine.search(q5))
