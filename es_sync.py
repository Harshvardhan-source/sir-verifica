"""
es_sync.py — Pushes both MongoDB voter-list timelines (2002 baseline, 2025 current)
into two separate Elasticsearch indices, with an edge_ngram analyzer on
name/relation fields so that typing the first 3 letters of a name returns
matches. Records are normalized from their real (differing) schemas via
schema_mapping.normalize_doc() before indexing.

Run once (or on a schedule) to keep ES in sync with Mongo:
    python es_sync.py --timeline 2025
    python es_sync.py --timeline 2002
    python es_sync.py --timeline both
"""

import argparse
import logging
from typing import Iterable, Dict, Any

from pymongo import MongoClient
from elasticsearch import Elasticsearch, helpers

from config import SIRConfig
from schema_mapping import SCHEMA_2002, SCHEMA_2025, normalize_doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("es_sync")


def build_index_mapping() -> Dict[str, Any]:
    return {
        "settings": {
            "analysis": {
                "filter": {
                    "edge_ngram_filter": {
                        "type": "edge_ngram",
                        "min_gram": 3,
                        "max_gram": 20,
                    }
                },
                "normalizer": {
                    # applied to keyword/.raw fields so an exact-match query
                    # doesn't miss a real match purely over casing (e.g. a
                    # stored "Suresh Rao" vs a search for "suresh rao") -
                    # see search_engine.py's exact-match clause.
                    "lowercase_normalizer": {
                        "type": "custom",
                        "filter": ["lowercase"],
                    }
                },
                "analyzer": {
                    "prefix_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "edge_ngram_filter"],
                    },
                    "search_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    },
                }
            },
            "number_of_shards": 1,
            "number_of_replicas": 0,
            # Disabled during bulk load - ES refreshing/merging every second
            # WHILE you're bulk-indexing 1.5M+ docs adds real overhead right
            # when you can least afford it. sync_timeline() explicitly calls
            # indices.refresh() once at the end and restores this to "1s".
            "refresh_interval": "-1",
        },
        "mappings": {
            "properties": {
                "epic_no": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "door_no": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {"raw": {"type": "keyword", "normalizer": "lowercase_normalizer"}},
                },
                "voter_name": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                        "std": {"type": "text"},
                    },
                },
                "relation_name": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {
                        "raw": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                        "std": {"type": "text"},
                    },
                },
                "relation_type": {"type": "keyword"},
                "age": {"type": "integer"},
                "gender": {"type": "keyword"},
                "part_no": {"type": "keyword"},
                "serial_no": {"type": "keyword"},
                "constituency": {
                    "type": "text",
                    "analyzer": "prefix_analyzer",
                    "search_analyzer": "search_analyzer",
                    "fields": {"raw": {"type": "keyword", "normalizer": "lowercase_normalizer"}},
                },
                "status": {"type": "keyword"},
                "religion": {"type": "keyword"},
                "community": {"type": "keyword"},
                "timeline_year": {"type": "keyword"},
                "mongo_id": {"type": "keyword"},
            }
        },
    }


def create_index(es: Elasticsearch, index_name: str, force: bool = False):
    if es.indices.exists(index=index_name):
        if not force:
            log.info(f"Index '{index_name}' already exists — skipping creation.")
            return
        es.indices.delete(index=index_name)
        log.info(f"Deleted existing index '{index_name}' (force=True).")
    es.indices.create(index=index_name, body=build_index_mapping())
    log.info(f"Created index '{index_name}'.")


def mongo_docs_to_es_actions(docs: Iterable[Dict[str, Any]], index_name: str,
                              timeline_year: str, schema: Dict[str, Any]):
    for doc in docs:
        canon = normalize_doc(doc, schema)
        canon["timeline_year"] = timeline_year
        yield {
            "_index": index_name,
            "_id": canon["mongo_id"],
            "_source": canon,
        }


def _with_progress(actions, total: int, log_every: int = 25000):
    """Wraps the action generator to log progress periodically -- useful
    when syncing very large collections (hundreds of thousands+ docs)."""
    count = 0
    for action in actions:
        count += 1
        if count % log_every == 0:
            pct = (count / total * 100) if total else 0
            log.info(f"  ... {count}/{total} documents processed ({pct:.1f}%)")
        yield action
    log.info(f"  ... {count}/{total} documents processed (done)")


def sync_timeline(cfg: SIRConfig, timeline: str, force_reindex: bool = False):
    """timeline: '2025' or '2002'"""
    mongo = MongoClient(cfg.mongo_uri)
    es = Elasticsearch(cfg.es_hosts, basic_auth=(cfg.es_username, cfg.es_password)
                        if cfg.es_username else None,
                        request_timeout=cfg.es_request_timeout,
                        retry_on_timeout=True, max_retries=3)

    if timeline == "2025":
        coll = mongo[cfg.mongo_db_2025][cfg.mongo_collection_2025]
        index_name = cfg.es_index_2025
        schema = SCHEMA_2025
    elif timeline == "2002":
        coll = mongo[cfg.mongo_db_2002][cfg.mongo_collection_2002]
        index_name = cfg.es_index_2002
        schema = SCHEMA_2002
    else:
        raise ValueError("timeline must be '2025' or '2002'")

    create_index(es, index_name, force=force_reindex)

    total = coll.count_documents({})
    log.info(f"Syncing {total} documents from Mongo '{coll.database.name}.{coll.name}' -> ES '{index_name}'")

    cursor = coll.find({}, batch_size=1000)  # no_cursor_timeout not permitted on shared/free Atlas tiers
    actions = _with_progress(
        mongo_docs_to_es_actions(cursor, index_name, timeline, schema),
        total=total,
    )
    try:
        success, errors = helpers.bulk(es, actions, chunk_size=cfg.es_bulk_chunk_size,
                                        raise_on_error=False, stats_only=False)
        if errors:
            log.warning(f"{len(errors)} documents failed to index. First error: {errors[0]}")
        log.info(f"Indexed {success} documents into '{index_name}'.")
    finally:
        # Always try to restore normal near-real-time refresh, even if the
        # bulk load failed/timed out partway through - otherwise the index
        # is left with refresh disabled (recent writes won't show up in
        # search) indefinitely. If THIS call also fails (e.g. the node is
        # genuinely down), it's logged rather than masking the real error.
        try:
            es.indices.put_settings(index=index_name, settings={"refresh_interval": "1s"})
            es.indices.refresh(index=index_name)
            log.info(f"Refresh interval restored to '1s' for '{index_name}'.")
        except Exception as e:
            log.warning(f"Could not restore refresh_interval on '{index_name}': {e}. "
                        f"Fix manually: PUT {index_name}/_settings {{'refresh_interval': '1s'}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync MongoDB voter lists into Elasticsearch")
    parser.add_argument("--timeline", choices=["2025", "2002", "both"], default="both")
    parser.add_argument("--force", action="store_true", help="Drop and recreate index")
    args = parser.parse_args()

    cfg = SIRConfig()
    timelines = ["2025", "2002"] if args.timeline == "both" else [args.timeline]
    for t in timelines:
        sync_timeline(cfg, t, force_reindex=args.force)