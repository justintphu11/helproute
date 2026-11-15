"""High-recall crisis safety detector for HelpRoute."""

import re

PATTERNS = {
    "suicide_self_harm": [r"kill myself", r"end it all", r"want to die", r"suicid", r"self[ -]?harm", r"hurt myself"],
    "abuse_violence": [r"domestic violence", r"elder abuse", r"child abuse", r"\babusive\b", r"\bassault", r"hits me", r"beating me"],
    "emergency_services": [r"need (the )?police", r"call 911", r"emergency shelter tonight", r"immediate danger", r"unsafe tonight"],
}


def detect(text: str) -> dict:
    value = (text or "").lower()
    matches = {category: [p for p in patterns if re.search(p, value)] for category, patterns in PATTERNS.items()}
    matches = {k: v for k, v in matches.items() if v}
    return {"crisis": int(bool(matches)), "categories": list(matches), "matched_phrases": [p for values in matches.values() for p in values]}
