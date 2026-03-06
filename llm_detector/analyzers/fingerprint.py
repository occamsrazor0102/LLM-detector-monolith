"""Intrinsic fingerprint detection -- LLM-preferred vocabulary."""

import re

FINGERPRINT_WORDS = [
    'delve', 'utilize', 'comprehensive', 'streamline', 'leverage', 'robust',
    'facilitate', 'innovative', 'synergy', 'paradigm', 'holistic', 'nuanced',
    'multifaceted', 'spearhead', 'underscore', 'pivotal', 'landscape',
    'cutting-edge', 'actionable', 'seamlessly', 'noteworthy', 'meticulous',
    'endeavor', 'paramount', 'aforementioned', 'furthermore', 'henceforth',
]

_FINGERPRINT_RE = re.compile(
    r'\b(?:' + '|'.join(re.escape(w) for w in FINGERPRINT_WORDS) + r')\b',
    re.IGNORECASE
)


def run_fingerprint(text):
    """Detect LLM fingerprint words. Returns (score, hit_count, rate)."""
    word_count = len(text.split())
    matches = _FINGERPRINT_RE.findall(text)
    hits = len(matches)
    rate = hits / max(word_count / 1000, 1)
    score = min(rate / 5.0, 1.0)
    return score, hits, rate
