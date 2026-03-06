"""Tests for individual analyzer modules."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector.compat import HAS_SEMANTIC, HAS_PERPLEXITY
from llm_detector.analyzers.semantic_resonance import run_semantic_resonance
from llm_detector.analyzers.perplexity import run_perplexity
from tests.conftest import AI_TEXT, HUMAN_TEXT, CLINICAL_TEXT

PASSED = 0
FAILED = 0


def check(label, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {label}")
    else:
        FAILED += 1
        print(f"  [FAIL] {label}  -- {detail}")


def test_semantic_resonance():
    print("\n-- SEMANTIC RESONANCE --")

    short = "Hello world."
    r_short = run_semantic_resonance(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_SEMANTIC:
        r_ai = run_semantic_resonance(AI_TEXT)
        check("AI text: semantic_ai_score > 0", r_ai['semantic_ai_score'] > 0,
              f"got {r_ai['semantic_ai_score']}")
        check("AI text: semantic_delta > 0", r_ai['semantic_delta'] > 0,
              f"got {r_ai['semantic_delta']}")
        check("AI text: has determination", r_ai['determination'] is not None,
              f"got {r_ai['determination']}, delta={r_ai['semantic_delta']}")

        r_human = run_semantic_resonance(HUMAN_TEXT)
        check("Human text: lower ai_score", r_human['semantic_ai_score'] < r_ai['semantic_ai_score'],
              f"human={r_human['semantic_ai_score']}, ai={r_ai['semantic_ai_score']}")
    else:
        print("  (sentence-transformers not installed -- skipping model tests)")
        check("Unavailable: ai_score=0", r_short['semantic_ai_score'] == 0.0)


def test_perplexity():
    print("\n-- PERPLEXITY SCORING --")

    short = "Hello world."
    r_short = run_perplexity(short)
    check("Short text: no determination", r_short['determination'] is None)

    if HAS_PERPLEXITY:
        r_normal = run_perplexity(CLINICAL_TEXT)
        check("Normal text: perplexity > 0", r_normal['perplexity'] > 0,
              f"got {r_normal['perplexity']}")
        check("Normal text: has reason", len(r_normal.get('reason', '')) > 0)
    else:
        print("  (transformers/torch not installed -- skipping model tests)")
        check("Unavailable: perplexity=0", r_short['perplexity'] == 0.0)


def test_feature_flags():
    print("\n-- FEATURE AVAILABILITY FLAGS --")
    check("HAS_SEMANTIC is bool", isinstance(HAS_SEMANTIC, bool))
    check("HAS_PERPLEXITY is bool", isinstance(HAS_PERPLEXITY, bool))
    print(f"    HAS_SEMANTIC={HAS_SEMANTIC}, HAS_PERPLEXITY={HAS_PERPLEXITY}")


if __name__ == '__main__':
    print("=" * 70)
    print("Analyzer Tests")
    print("=" * 70)

    test_feature_flags()
    test_semantic_resonance()
    test_perplexity()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    if FAILED > 0:
        sys.exit(1)
