"""
schema_mapping.py — the 2002 and 2025 collections use genuinely different
field names (different OCR/digitization pipelines produced them), so this
module maps both into one canonical shape used everywhere downstream
(ES indexing, search, anomaly detection).

Canonical keys produced by normalize_doc():
    epic_no, door_no, voter_name, relation_name, relation_type,
    age, gender, part_no, serial_no, constituency, status,
    religion, community, mongo_id

Both source schemas carry OCR spell-check metadata (Name_Changed /
Name_Corrected, and the relative-name equivalent). When a name was
flagged as corrected, normalize_doc() prefers the corrected spelling —
that's the more reliable value for matching across timelines.
"""

import math
from typing import Optional, Dict, Any


def _clean_value(v):
    """Elasticsearch's bulk API requires strict JSON -- NaN/Infinity (which
    Python's own json module tolerates as a non-standard extension) will be
    rejected. Some source records have NaN ages left over from an OCR/pandas
    pipeline step; convert those to None here."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _corrected(doc: Dict[str, Any], raw_key: str, changed_key: Optional[str],
               corrected_key: Optional[str]) -> str:
    raw = (doc.get(raw_key) or "").strip() if raw_key else ""
    if changed_key and corrected_key and doc.get(changed_key):
        corrected = (doc.get(corrected_key) or "").strip()
        if corrected:
            return corrected
    return raw


# ---------------------------------------------------------------------------
# 2002 baseline roll — collection 'DK_2002_new2' in DB 'SurveyDataBase'
# Verified against a live sample doc on 2026-07-15 via check_connection.py --
# this collection uses the bilingual/underscored field style, e.g.
# 'voter_id_/_epic_no', 'house_/_flat_no', 'voter_name_(english)'.
# DO NOT swap this with SCHEMA_2025 below -- that exact regression already
# happened once and silently nulled out both indices without erroring.
# ---------------------------------------------------------------------------
SCHEMA_2002 = {
    "epic_no": "voter_id_/_epic_no",
    # door_no/part_no: this dataset was OCR'd across ~600+ PDF pages in
    # separate batches, and the exact key punctuation drifts between
    # batches (e.g. 'house_/_flat_no' vs 'house_/flat_no', with/without
    # the underscore before a Kannada parenthetical). A single hardcoded
    # key silently nulls out the field for whichever batches don't match.
    # Lists here are tried in order, first key present wins.
    "door_no": ["house_/_flat_no", "house_/flat_no", "house_flat_no"],
    "name_raw": "voter_name_(english)",
    "name_changed": "Name_Changed",
    "name_corrected": "Name_Corrected",
    "relation_name_raw": "relative_name_(english)",
    "relation_name_changed": "RelativeName_Changed",
    "relation_name_corrected": "RelativeName_Corrected",
    "relation_type": "relationship",
    "age": "age",
    "gender": "gender",
    "part_no": ["part_no_(ಭಾಗ)", "part_no(ಭಾಗ)", "part_no"],
    "serial_no": "serial_no",
    "constituency": "constituency",
    "status": None,                # not present in this collection
    "religion": "Religion",
    "community": "Community",
}

# ---------------------------------------------------------------------------
# 2025 current roll — collection 'DK' in DB 'SurveyDataBase'
# Verified against a live sample doc on 2026-07-15 via check_connection.py --
# this collection uses plain field names, e.g. 'voterid', 'address', 'name'.
# DO NOT swap this with SCHEMA_2002 above -- that exact regression already
# happened once and silently nulled out both indices without erroring.
# ---------------------------------------------------------------------------
SCHEMA_2025 = {
    "epic_no": "voterid",
    "door_no": "address",
    "name_raw": "name",
    "name_changed": "Name_Changed",
    "name_corrected": "Name_Corrected",
    "relation_name_raw": "relative_name",
    "relation_name_changed": "Relative Name_Changed",
    "relation_name_corrected": "Relative Name_Corrected",
    "relation_type": "relation_type",
    "age": "age",
    "gender": "gender",
    "part_no": "part_no",
    "serial_no": None,             # not present in this collection
    "constituency": "assembly_constituency_name",
    "status": None,                # not present in this collection
    # Fixed 2026-07-18: this was wrongly set to None. Your classifier
    # pipeline (mongo_voter_classifier.py) populates these on both
    # collections -- confirmed present in a live sample doc.
    "religion": "Religion",
    "community": "Community",
}


def normalize_doc(doc: Dict[str, Any], schema: Dict[str, Optional[str]]) -> Dict[str, Any]:
    name = _corrected(doc, schema["name_raw"], schema["name_changed"], schema["name_corrected"])
    relation_name = _corrected(doc, schema["relation_name_raw"],
                                schema["relation_name_changed"], schema["relation_name_corrected"])

    def g(key: str):
        """Looks up a canonical field. schema[key] may be a single field
        name, or a list of candidate field names tried in order (first
        one actually present in this doc wins) -- see door_no/part_no
        above for why."""
        field = schema.get(key)
        if not field:
            return None
        candidates = field if isinstance(field, list) else [field]
        for cand in candidates:
            if cand in doc and doc.get(cand) not in (None, ""):
                return doc.get(cand)
        return None

    epic = g("epic_no")
    canon = {
        "epic_no": (str(epic).strip() if epic else None) or None,
        "door_no": g("door_no"),
        "voter_name": name or None,
        "relation_name": relation_name or None,
        "relation_type": g("relation_type"),
        "age": g("age"),
        "gender": g("gender"),
        "part_no": g("part_no"),
        "serial_no": g("serial_no"),
        "constituency": g("constituency") or "Dakshina Kannada",
        "status": g("status") or "ACTIVE",   # neither collection has a status field yet
        "religion": g("religion"),
        "community": g("community"),
        "mongo_id": str(doc.get("_id")),
    }
    return {k: _clean_value(v) for k, v in canon.items()}