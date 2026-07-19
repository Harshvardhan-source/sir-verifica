"""
SIR (Special Intensive Revision) Voter Verification System
Modeled on Election Commission of India's SIR protocol.
Constituency: Dakshina Kannada

config.py — central configuration for Mongo + Elasticsearch connections,
index names, and tunable thresholds used by the anomaly detector.

Field-name mapping now lives in schema_mapping.py, since the 2002 and
2025 collections use genuinely different schemas.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SIRConfig:
    # ---- MongoDB ----
    mongo_uri: str = os.environ.get("SIR_MONGO_URI", "")
    # ^ No hardcoded fallback on purpose - this file goes into git. Set
    # SIR_MONGO_URI as a real environment variable locally, and as a
    # Render "Environment Variable" (encrypted at rest, not committed) for
    # deployment - never paste a live connection string into this file.

    # 2002 baseline roll
    mongo_db_2002: str = "SurveyDataBase"
    mongo_collection_2002: str = "DK_2002_new2"

    # 2025 current roll
    mongo_db_2025: str = "SurveyDataBase"
    # TODO: confirm the actual collection name for the 2025 roll in the
    # 'DK' database (only the DB name was provided) -- update if different.
    mongo_collection_2025: str = "DK"

    # ---- Elasticsearch ----
    # SIR_ES_HOSTS supports one or more comma-separated URLs. Defaults to
    # localhost for local dev - MUST be set in production (Render has no
    # "localhost:9200"; see deployment notes for hosting options).
    es_hosts: List[str] = field(default_factory=lambda: [
        h.strip() for h in os.environ.get("SIR_ES_HOSTS", "http://localhost:9200").split(",") if h.strip()
    ])
    es_index_2025: str = "voters_2025"
    es_index_2002: str = "voters_2002"
    es_username: str = os.environ.get("SIR_ES_USERNAME") or None
    es_password: str = os.environ.get("SIR_ES_PASSWORD") or None
    es_bulk_chunk_size: int = 500      # docs per bulk request. Lowered from 2000 after a real
                                        # sync timed out at 2000/request on a local ES instance -
                                        # if you have a beefier cluster you can raise this back up,
                                        # but a slow node will time out on a too-large chunk
                                        # regardless of es_request_timeout below, since retries
                                        # just resend the same oversized request.
    es_request_timeout: int = 120      # seconds; raised from 60 - gives slower/local ES nodes
                                        # more headroom per bulk request before giving up

    # ---- Search behaviour ----
    prefix_min_chars: int = 3          # min chars before name-prefix search kicks in
    fuzziness: str = "AUTO"            # ES fuzziness for OCR/spelling variance
    max_search_workers: int = 4        # thread pool size (>=2, one per timeline index)
    result_size_per_index: int = 100
    exact_match_boost: float = 50.0    # multiplier applied to an exact (raw keyword) match so
                                        # it always ranks above prefix/fuzzy hits regardless of
                                        # BM25 on the analyzed fields - see search_engine.py
    epic_fuzzy_min_similarity: int = 85  # rapidfuzz score (0-100, same scale as
                                        # duplicate_name_fuzzy_threshold below) a
                                        # fuzzy EPIC candidate must clear to be
                                        # shown as "the closest match" when no
                                        # exact EPIC exists. ES is used only to
                                        # retrieve candidates here - the actual
                                        # accept/reject decision is rapidfuzz on
                                        # the real EPIC strings, since ES's own
                                        # _score isn't a portable scale to
                                        # threshold against directly.

    # ---- Anomaly detection thresholds (ECI-SIR style) ----
    max_voters_per_house: int = 15     # flag houses with more voters than this
    min_valid_age: int = 18
    max_valid_age: int = 115
    duplicate_name_fuzzy_threshold: int = 90   # rapidfuzz score (0-100) to treat as "same person"
    lineage_break_fuzzy_threshold: int = 85    # threshold for 2002<->2025 name-linking without EPIC match