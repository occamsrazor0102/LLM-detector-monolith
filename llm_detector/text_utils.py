"""Shared text utilities used across multiple modules."""

import re
from llm_detector.compat import HAS_SPACY

if HAS_SPACY:
    from llm_detector.compat import _nlp

# Top-50 English function words (closed class, highly stable across registers)
ENGLISH_FUNCTION_WORDS = frozenset([
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'am', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'shall', 'should', 'may', 'might', 'can', 'could', 'must',
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
    'into', 'about', 'between', 'through', 'after', 'before',
    'and', 'or', 'but', 'not', 'if', 'that', 'this', 'it', 'he', 'she',
    'they', 'we', 'i', 'you', 'my', 'your', 'his', 'her', 'its', 'our',
    'their', 'who', 'which', 'what', 'there',
])


def get_sentences(text):
    """Segment text into sentences using spacy sentencizer or regex fallback."""
    if HAS_SPACY:
        doc = _nlp(text)
        return [s.text for s in doc.sents]
    else:
        sents = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sents if s.strip()]
