"""Preamble detection -- catches LLM output artifacts like 'Sure, here is...'"""

import re

PREAMBLE_PATTERNS = [
    (r"(?i)^\s*[\"']?(got it|sure thing|absolutely|certainly|of course)[.!,\s]", "assistant_ack", "CRITICAL"),
    (r"(?i)^\s*[\"']?here(?:'s| is| are)\s+(your|the|a)\s+(final|updated|revised|complete|rewritten|prompt|task|evaluation)", "artifact_delivery", "CRITICAL"),
    (r"(?i)^\s*[\"']?below is\s+(a\s+)?(rewritten|revised|updated|the|your)", "artifact_delivery", "CRITICAL"),
    (r"(?i)(copy[- ]?paste|ready to use|plug[- ]and[- ]play)", "copy_paste_instruction", "MEDIUM"),
    (r"(?i)(failure[- ]inducing|designed to (test|challenge|trip|catch|induce))", "meta_design", "CRITICAL"),
    (r"(?i)^\s*[\"']?(I'?ve |I have |I'?ll |let me )(created?|drafted?|prepared?|written|designed|built|put together)", "first_person_creation", "CRITICAL"),
    (r"(?i)(natural workplace style|sounds? like a real|human[- ]issued|reads? like a human)", "style_masking", "HIGH"),
    (r"(?i)notes on what I (fixed|changed|cleaned|updated|revised)", "editorial_meta", "HIGH"),
]


def run_preamble(text):
    """Detect LLM preamble artifacts. Returns (score, severity, hits)."""
    first_500 = text[:500]
    hits = []
    severity = 'NONE'

    for pat, name, sev in PREAMBLE_PATTERNS:
        search_text = first_500 if name in ('assistant_ack', 'artifact_delivery', 'first_person_creation') else text
        if re.search(pat, search_text):
            hits.append((name, sev))
            if sev == 'CRITICAL':
                severity = 'CRITICAL'
            elif sev == 'HIGH' and severity not in ('CRITICAL',):
                severity = 'HIGH'
            elif sev == 'MEDIUM' and severity == 'NONE':
                severity = 'MEDIUM'

    score = {'CRITICAL': 0.99, 'HIGH': 0.75, 'MEDIUM': 0.50, 'NONE': 0.0}[severity]
    return score, severity, hits
