"""
main.py — unified CLI for the Dakshina Kannada SIR Verification System.

Usage:
    python main.py sync --timeline both --force
    python main.py search --name "Sur" --door "12-45"
    python main.py search --epic "KA1234567"
    python main.py audit
"""

import argparse
import json

from config import SIRConfig
from es_sync import sync_timeline
from search_engine import SIRSearchEngine, SearchQuery
from anomaly_detector import SIRAnomalyDetector


def cmd_sync(args, cfg: SIRConfig):
    timelines = ["2025", "2002"] if args.timeline == "both" else [args.timeline]
    for t in timelines:
        sync_timeline(cfg, t, force_reindex=args.force)


def cmd_search(args, cfg: SIRConfig):
    engine = SIRSearchEngine(cfg)
    q = SearchQuery(
        epic_no=args.epic,
        door_no=args.door,
        voter_name=args.name,
        relation_name=args.relation,
        constituency=args.constituency,
        age=args.age,
        gender=args.gender,
        part_no=args.part,
        combine_mode=args.mode,
        min_match=args.min_match,
    )
    result = engine.search(q)
    print(json.dumps(result, indent=2, default=str))


def cmd_audit(args, cfg: SIRConfig):
    detector = SIRAnomalyDetector(cfg)
    anomalies = detector.run_full_audit()
    print(json.dumps(detector.summary(), indent=2))
    if args.verbose:
        for a in anomalies:
            print(f"[{a.severity}] {a.category} | EPIC={a.epic_no} | {a.details}")


def main():
    parser = argparse.ArgumentParser(description="Dakshina Kannada SIR Verification System")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sync = sub.add_parser("sync", help="Sync MongoDB -> Elasticsearch")
    p_sync.add_argument("--timeline", choices=["2025", "2002", "both"], default="both")
    p_sync.add_argument("--force", action="store_true")
    p_sync.set_defaults(func=cmd_sync)

    p_search = sub.add_parser("search", help="Combinatorial multithreaded search")
    p_search.add_argument("--epic")
    p_search.add_argument("--door")
    p_search.add_argument("--name")
    p_search.add_argument("--relation")
    p_search.add_argument("--constituency")
    p_search.add_argument("--age", type=int)
    p_search.add_argument("--gender")
    p_search.add_argument("--part")
    p_search.add_argument("--mode", choices=["OR", "AND", "MIN_N"], default="OR")
    p_search.add_argument("--min-match", dest="min_match", type=int, default=2)
    p_search.set_defaults(func=cmd_search)

    p_audit = sub.add_parser("audit", help="Run full ECI-SIR anomaly audit")
    p_audit.add_argument("--verbose", action="store_true")
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    cfg = SIRConfig()
    args.func(args, cfg)


if __name__ == "__main__":
    main()
