"""
name_variants.py — generates plausible alternate spellings of an Indian
(primarily Kannada/Tulu-region) name, to widen recall when a reviewer
searches OCR'd, hand-transliterated electoral roll data.

This exists for the INTERACTIVE SEARCH BOX only (api_server.py /api/search,
used by search_engine.py). It is NOT wired into anomaly_detector.py's
automated 2002<->2025 fuzzy lineage-linking. Widening what a human reviewer
*sees* on a search is low-stakes -- they're picking from a results list
either way. Widening what the system automatically treats as "the same
voter across timelines" on a nickname guess is a much higher-stakes
decision (it directly feeds LINEAGE_BREAK / DUPLICATE_PERSON flags) and
should be a deliberate, separately-reviewed choice, not a side effect of
a search UX improvement.

Two independent, and independently unreliable, sources of variants:

1. TRANSLITERATION_RULES -- systematic vowel-length / consonant-aspiration
   substitutions that show up when the same Kannada sound is rendered into
   English by different typists or OCR passes (e.g. "Ravindra" <->
   "Raveendra" <-> "Ravindar"). These are mechanical and reasonably safe --
   worst case they generate a variant that matches nothing.

2. NICKNAMES -- a curated short-form dictionary (e.g. "Ravindra" -> "Ravi").
   THIS LIST IS A STARTING POINT, NOT AN AUTHORITY. Nicknames are local
   convention, not something derivable from spelling, and getting an entry
   wrong has real consequences here: a false nickname mapping can surface
   an irrelevant person in search results; a missing one just means the
   reviewer has to try the short form themselves (same as today). Please
   review/correct/extend this dict for your own constituency's naming
   conventions -- it was seeded with common patterns, not verified against
   your actual voter data.
"""

import re
from typing import List

# ---------------------------------------------------------------------------
# 1. Transliteration substitution rules
# Applied on a lowercased copy of the name. Each (find, replace) pair is
# tried once per generation pass; results are deduplicated. Order-sensitive
# rules (like the -endra/-endar swap) are listed as explicit whole-suffix
# pairs rather than relying on the shorter substrings, to avoid mangling
# unrelated parts of the name.
# ---------------------------------------------------------------------------
_SUFFIX_RULES = [
    ("endra", "endar"), ("endar", "endra"),
    ("indra", "indar"), ("indar", "indra"),
    ("achar", "acharya"), ("acharya", "achar"),
]

_SUBSTRING_RULES = [
    ("ee", "i"),
    ("oo", "u"),
    ("dh", "d"),
    ("th", "t"),
    ("ph", "f"),
    ("sh", "s"),
    ("v", "w"),
]


def _transliteration_variants(name_lower: str) -> List[str]:
    variants = set()

    for find, replace in _SUFFIX_RULES:
        if name_lower.endswith(find):
            variants.add(name_lower[: -len(find)] + replace)

    for find, replace in _SUBSTRING_RULES:
        if find in name_lower:
            variants.add(name_lower.replace(find, replace))
        # also try the reverse direction opportunistically
        if replace in name_lower:
            variants.add(name_lower.replace(replace, find))

    variants.discard(name_lower)
    return sorted(variants)


# ---------------------------------------------------------------------------
# 2. Nickname / short-form dictionary — SEED DATA, REVIEW BEFORE TRUSTING.
# Keys and values are lowercase. Stored one-directional (full -> short);
# generate_variants() expands both directions at lookup time.
# ---------------------------------------------------------------------------
NICKNAMES = {
    "ravindra": ["ravi"],
    "ravindran": ["ravi"],
    "surendra": ["suri"],
    "narendra": ["naren"],
    "rajendra": ["raju", "raja"],
    "devendra": ["deva"],
    "virendra": ["viru"],
    "dharmendra": ["dharma"],
    "yashwanth": ["yash"],
    "yashwant": ["yash"],
    "krishnamurthy": ["krishna", "murthy"],
    "krishnappa": ["krishna"],
    "ramakrishna": ["ram", "krishna"],
    "gopalakrishna": ["gopal", "krishna"],
    "venkataramana": ["venkat", "venky"],
    "venkataraman": ["venkat", "venky"],
    "venkatesh": ["venkat", "venky"],
    "lakshminarayana": ["lakshman", "narayan"],
    "sathyanarayana": ["sathya"],
    "chandrashekhar": ["chandra"],
    "chandrasekhara": ["chandra"],
    "vishwanatha": ["vishu"],
    "vishwanath": ["vishu"],
    "manjunatha": ["manju"],
    "manjunath": ["manju"],
    "puttaswamy": ["putta"],
    "basavaraju": ["basu"],
    "subramanya": ["subbu"],
    "subramaniam": ["subbu"],
    "narasimha": ["simha"],
    "jayarama": ["jaya"],
    "jayaram": ["jaya"],
    "mahalakshmi": ["lakshmi"],
}

# reverse-index built once at import time so short -> full lookups are cheap
_REVERSE_NICKNAMES: dict = {}
for _full, _shorts in NICKNAMES.items():
    for _short in _shorts:
        _REVERSE_NICKNAMES.setdefault(_short, []).append(_full)


def _nickname_variants(name_lower: str) -> List[str]:
    variants = set(NICKNAMES.get(name_lower, []))
    variants.update(_REVERSE_NICKNAMES.get(name_lower, []))
    variants.discard(name_lower)
    return sorted(variants)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def generate_variants(name: str, max_variants: int = 6) -> List[str]:
    """
    Returns a list starting with the original (trimmed, as-typed) name,
    followed by up to `max_variants` generated alternates. Only whole
    tokens are varied (e.g. for "Ravindra Rao" only "Ravindra" gets
    expanded) since surnames/relation-name components don't follow the
    same nickname conventions.
    """
    name = (name or "").strip()
    if not name:
        return []

    first_token = name.split()[0]
    rest = name[len(first_token):]
    lower = first_token.lower()

    candidates = []
    seen = {lower}
    for v in _nickname_variants(lower) + _transliteration_variants(lower):
        if v not in seen:
            seen.add(v)
            candidates.append(v)

    variants = [name]
    for v in candidates[:max_variants]:
        variants.append(v + rest)
    return variants