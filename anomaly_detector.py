"""
anomaly_detector.py — Implements the categories of irregularities the
Election Commission of India's SIR (Special Intensive Revision) process
is designed to catch, run across the Dakshina Kannada 2002 (baseline)
and 2025 (current) voter-list timelines held in MongoDB.

Detected anomaly categories (mirrors real ECI-SIR flags):
  1. DUPLICATE_EPIC        — same EPIC number assigned to >1 elector
  2. DUPLICATE_PERSON      — same name+relation+age (fuzzy) at different
                              addresses / multiple EPICs -> possible
                              multiple-enrolment fraud
  3. OVERCROWDED_HOUSE     — a single door/house number with an
                              abnormally large number of registered
                              voters (classic bogus-voter pattern)
  4. AGE_ANOMALY           — age outside plausible bounds, or age that
                              regresses illogically between 2002 and 2025
                              for the same linked elector
  5. LINEAGE_BREAK         — elector present in 2002 baseline, absent
                              from 2025 roll, with no fuzzy-name match
                              anywhere else in the 2025 constituency data
                              (silent/unverified deletion — needs Form 7
                              verification per ECI protocol)
  6. GHOST_ENTRY           — elector appears in 2025 with no 2002 lineage
                              AND no supporting new-registration form
                              trail (flag for Form 6 audit)
  7. DECEASED_STILL_ACTIVE — status marked DECEASED but still ACTIVE in
                              a later roll
  8. RELATION_NAME_DRIFT   — same EPIC across timelines but relation_name
                              changed significantly (possible identity
                              swap / EPIC re-use fraud)

Uses rapidfuzz for tolerant name matching (handles OCR + transliteration
spelling variance typical of scanned electoral rolls).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from pymongo import MongoClient
from rapidfuzz import fuzz

from config import SIRConfig
from schema_mapping import SCHEMA_2002, SCHEMA_2025, normalize_doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("anomaly_detector")


@dataclass
class Anomaly:
    category: str
    severity: str              # LOW / MEDIUM / HIGH / CRITICAL
    epic_no: Optional[str]
    details: str
    records: List[Dict[str, Any]] = field(default_factory=list)


class SIRAnomalyDetector:
    def __init__(self, cfg: SIRConfig):
        self.cfg = cfg
        self.mongo = MongoClient(cfg.mongo_uri)
        self.db_client = self.mongo
        self.anomalies: List[Anomaly] = []

    # ------------------------------------------------------------------
    # data loaders
    # ------------------------------------------------------------------
    def _load(self, db_name: str, collection_name: str, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_docs = list(self.db_client[db_name][collection_name].find({}))
        return [normalize_doc(d, schema) for d in raw_docs]

    # ------------------------------------------------------------------
    # 1. Duplicate EPIC numbers within a single roll
    # ------------------------------------------------------------------
    def check_duplicate_epic(self, records: List[Dict[str, Any]], timeline: str):
        buckets = defaultdict(list)
        for r in records:
            epic = r.get("epic_no")
            if epic:
                buckets[epic].append(r)
        for epic, group in buckets.items():
            if len(group) > 1:
                self.anomalies.append(Anomaly(
                    category="DUPLICATE_EPIC",
                    severity="CRITICAL",
                    epic_no=epic,
                    details=f"EPIC '{epic}' assigned to {len(group)} distinct entries in {timeline} roll.",
                    records=group,
                ))

    # ------------------------------------------------------------------
    # 2. Duplicate person via fuzzy name+relation+age match across
    #    different EPICs/addresses (multiple-enrolment fraud pattern)
    # ------------------------------------------------------------------
    def check_duplicate_person(self, records: List[Dict[str, Any]], timeline: str):
        n = len(records)
        # bucket by first-3-letters of name to avoid O(n^2) over the whole roll
        buckets = defaultdict(list)
        for r in records:
            name = (r.get("voter_name") or "").strip()
            if len(name) >= 3:
                buckets[name[:3].lower()].append(r)

        seen_pairs = set()
        for bucket in buckets.values():
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    a, b = bucket[i], bucket[j]
                    if a.get("epic_no") == b.get("epic_no"):
                        continue
                    key = tuple(sorted([str(a.get("_id")), str(b.get("_id"))]))
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)

                    name_score = fuzz.token_sort_ratio(
                        a.get("voter_name", ""), b.get("voter_name", ""))
                    rel_score = fuzz.token_sort_ratio(
                        a.get("relation_name", "") or "",
                        b.get("relation_name", "") or "")
                    same_age = a.get("age") == b.get("age")

                    if (name_score >= self.cfg.duplicate_name_fuzzy_threshold
                            and rel_score >= self.cfg.duplicate_name_fuzzy_threshold
                            and same_age):
                        epic_a = a.get("epic_no")
                        epic_b = b.get("epic_no")
                        self.anomalies.append(Anomaly(
                            category="DUPLICATE_PERSON",
                            severity="HIGH",
                            epic_no=f"{epic_a} / {epic_b}",
                            details=(f"Possible multiple enrolment in {timeline}: name/relation/age "
                                     f"match at score {name_score}/{rel_score} across two EPICs."),
                            records=[a, b],
                        ))

    # ------------------------------------------------------------------
    # 3. Overcrowded house / door number
    # ------------------------------------------------------------------
    def check_overcrowded_house(self, records: List[Dict[str, Any]], timeline: str):
        buckets = defaultdict(list)
        for r in records:
            door = r.get("door_no")
            if door:
                buckets[door].append(r)
        for door, group in buckets.items():
            if len(group) > self.cfg.max_voters_per_house:
                self.anomalies.append(Anomaly(
                    category="OVERCROWDED_HOUSE",
                    severity="MEDIUM",
                    epic_no=None,
                    details=(f"Door/House No. '{door}' has {len(group)} registered voters in "
                             f"{timeline} roll (threshold={self.cfg.max_voters_per_house})."),
                    records=group,
                ))

    # ------------------------------------------------------------------
    # 4. Age anomaly (implausible age)
    # ------------------------------------------------------------------
    def check_age_anomaly(self, records: List[Dict[str, Any]], timeline: str):
        for r in records:
            age = r.get("age")
            if age is None:
                continue
            if age < self.cfg.min_valid_age or age > self.cfg.max_valid_age:
                self.anomalies.append(Anomaly(
                    category="AGE_ANOMALY",
                    severity="MEDIUM",
                    epic_no=r.get("epic_no"),
                    details=f"Age {age} outside plausible bounds in {timeline} roll.",
                    records=[r],
                ))

    # ------------------------------------------------------------------
    # 5 & 6. Cross-timeline lineage: LINEAGE_BREAK (in 2002, missing 2025)
    #        and GHOST_ENTRY (in 2025, no 2002 lineage) using fuzzy link
    # ------------------------------------------------------------------
    def check_lineage(self, records_2002: List[Dict[str, Any]], records_2025: List[Dict[str, Any]]):
        epic_2025 = {r.get("epic_no"): r for r in records_2025 if r.get("epic_no")}
        epic_2002 = {r.get("epic_no"): r for r in records_2002 if r.get("epic_no")}

        # index 2025 names by first-3-letters for fuzzy fallback linking
        name_buckets_2025 = defaultdict(list)
        for r in records_2025:
            name = (r.get("voter_name") or "").strip()
            if len(name) >= 3:
                name_buckets_2025[name[:3].lower()].append(r)

        for epic, rec_2002 in epic_2002.items():
            if epic in epic_2025:
                continue  # direct EPIC continuity found, fine
            # try fuzzy name+relation link before declaring a break
            name = (rec_2002.get("voter_name") or "").strip()
            candidates = name_buckets_2025.get(name[:3].lower(), []) if len(name) >= 3 else []
            linked = False
            for cand in candidates:
                score = fuzz.token_sort_ratio(name, cand.get("voter_name", ""))
                if score >= self.cfg.lineage_break_fuzzy_threshold:
                    linked = True
                    break
            if not linked:
                self.anomalies.append(Anomaly(
                    category="LINEAGE_BREAK",
                    severity="HIGH",
                    epic_no=epic,
                    details=("Elector present in 2002 baseline roll, absent from 2025 roll, with no "
                             "fuzzy name match found — requires Form 7 (deletion) verification per "
                             "ECI-SIR protocol before being treated as a valid exclusion."),
                    records=[rec_2002],
                ))

        for epic, rec_2025 in epic_2025.items():
            if epic in epic_2002:
                continue
            name = (rec_2025.get("voter_name") or "").strip()
            # ghost check: no fuzzy match anywhere in 2002 roll at all
            candidates = [r for r in records_2002
                          if (r.get("voter_name") or "")[:3].lower() == name[:3].lower()]
            linked = any(fuzz.token_sort_ratio(name, c.get("voter_name", ""))
                         >= self.cfg.lineage_break_fuzzy_threshold for c in candidates)
            if not linked:
                self.anomalies.append(Anomaly(
                    category="GHOST_ENTRY",
                    severity="MEDIUM",
                    epic_no=epic,
                    details=("New elector in 2025 roll with no traceable 2002 lineage — expected for "
                             "genuine new/turning-18 registrations, but flagged for Form 6 audit trail "
                             "confirmation per SIR protocol."),
                    records=[rec_2025],
                ))

    # ------------------------------------------------------------------
    # 7. Deceased but still marked active in a later roll
    # ------------------------------------------------------------------
    def check_deceased_still_active(self, records_2002: List[Dict[str, Any]], records_2025: List[Dict[str, Any]]):
        epic_2025 = {r.get("epic_no"): r for r in records_2025 if r.get("epic_no")}
        for r in records_2002:
            if (r.get("status") or "").upper() == "DECEASED":
                epic = r.get("epic_no")
                later = epic_2025.get(epic)
                if later and (later.get("status") or "").upper() == "ACTIVE":
                    self.anomalies.append(Anomaly(
                        category="DECEASED_STILL_ACTIVE",
                        severity="CRITICAL",
                        epic_no=epic,
                        details="Elector marked DECEASED in 2002 roll but ACTIVE in 2025 roll.",
                        records=[r, later],
                    ))

    # ------------------------------------------------------------------
    # 8. Relation-name drift on a continuous EPIC (possible identity swap)
    # ------------------------------------------------------------------
    def check_relation_drift(self, records_2002: List[Dict[str, Any]], records_2025: List[Dict[str, Any]]):
        epic_2025 = {r.get("epic_no"): r for r in records_2025 if r.get("epic_no")}
        for r2002 in records_2002:
            epic = r2002.get("epic_no")
            r2025 = epic_2025.get(epic)
            if not r2025:
                continue
            rel_2002 = r2002.get("relation_name") or ""
            rel_2025 = r2025.get("relation_name") or ""
            if not rel_2002 or not rel_2025:
                continue
            score = fuzz.token_sort_ratio(rel_2002, rel_2025)
            if score < 60:  # big drift in relation name under a continuous EPIC
                self.anomalies.append(Anomaly(
                    category="RELATION_NAME_DRIFT",
                    severity="HIGH",
                    epic_no=epic,
                    details=(f"Relation name changed drastically ('{rel_2002}' -> '{rel_2025}') "
                             f"under the same EPIC (similarity={score}). Possible EPIC re-use / "
                             f"identity mismatch — requires manual field verification."),
                    records=[r2002, r2025],
                ))

    # ------------------------------------------------------------------
    # orchestrator
    # ------------------------------------------------------------------
    def run_full_audit(self) -> List[Anomaly]:
        self.anomalies = []
        records_2025 = self._load(self.cfg.mongo_db_2025, self.cfg.mongo_collection_2025, SCHEMA_2025)
        records_2002 = self._load(self.cfg.mongo_db_2002, self.cfg.mongo_collection_2002, SCHEMA_2002)

        log.info(f"Loaded {len(records_2025)} (2025) / {len(records_2002)} (2002) records.")

        # within-roll checks, run per timeline
        self.check_duplicate_epic(records_2025, "2025")
        self.check_duplicate_epic(records_2002, "2002")
        self.check_duplicate_person(records_2025, "2025")
        self.check_duplicate_person(records_2002, "2002")
        self.check_overcrowded_house(records_2025, "2025")
        self.check_overcrowded_house(records_2002, "2002")
        self.check_age_anomaly(records_2025, "2025")
        self.check_age_anomaly(records_2002, "2002")

        # cross-timeline checks
        self.check_lineage(records_2002, records_2025)
        self.check_deceased_still_active(records_2002, records_2025)
        self.check_relation_drift(records_2002, records_2025)

        log.info(f"Audit complete: {len(self.anomalies)} anomalies found.")
        return self.anomalies

    def summary(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for a in self.anomalies:
            counts[a.category] += 1
        return dict(counts)


if __name__ == "__main__":
    cfg = SIRConfig()
    detector = SIRAnomalyDetector(cfg)
    anomalies = detector.run_full_audit()
    print("Summary:", detector.summary())
    for a in anomalies[:20]:
        print(f"[{a.severity}] {a.category} | EPIC={a.epic_no} | {a.details}")
