"""Tests for windowed scoring and the windowed channel."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_detector_monolith import score_windows
from llm_detector_monolith import score_windowed
from tests.conftest import AI_TEXT, HUMAN_TEXT

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


SHORT_TEXT = "This is a short sentence. And another one."

# Long AI-style text with many formulaic sentences
LONG_AI_TEXT = (
    "This comprehensive analysis provides a thorough examination of the key factors. "
    "Furthermore, it is essential to note that the implementation ensures alignment. "
    "Additionally, this approach demonstrates the critical importance of evaluation. "
    "Moreover, the systematic assessment reveals significant opportunities for growth. "
    "In conclusion, these findings underscore the transformative potential of this framework. "
    "It is important to recognize that evidence-based strategies drive optimal outcomes. "
    "Consequently, the integration of these methodologies yields substantial improvements. "
    "To that end, we must consider the multifaceted nature of this challenge. "
    "Ultimately, this holistic perspective enables more effective decision making. "
    "In summary, the convergence of these factors creates a compelling case for action."
)

LONG_HUMAN_TEXT = (
    "so I was looking at the logs yesterday and found something weird. "
    "turns out the parser was choking on timestamps with milliseconds. "
    "I hacked together a fix but it's kinda ugly tbh. "
    "the regex now handles both formats which is nice. "
    "oh and I also noticed the memory usage spikes around midnight. "
    "probably the cron job running the full backup or something. "
    "anyway I'll clean up the code tomorrow when I have more time. "
    "Dave said he'd review it but he's been super busy with the migration. "
    "the whole thing is a mess honestly but it works for now. "
    "I'll write proper tests once we have the new CI pipeline set up."
)


def test_short_text_empty_windows():
    print("\n-- Short text returns empty windows --")
    result = score_windows(SHORT_TEXT)
    check("n_windows == 0", result['n_windows'] == 0,
          f"got {result['n_windows']}")
    check("windows list empty", len(result['windows']) == 0)
    check("max_window_score == 0", result['max_window_score'] == 0.0)
    check("mixed_signal is False", result['mixed_signal'] is False)


def test_ai_text_produces_scores():
    print("\n-- AI text produces window scores --")
    result = score_windows(LONG_AI_TEXT)
    check("n_windows > 0 for long AI text", result['n_windows'] > 0,
          f"got {result['n_windows']}")
    check("max_window_score > 0", result['max_window_score'] > 0.0,
          f"got {result['max_window_score']}")


def test_human_text_low_scores():
    print("\n-- Human text produces low window scores --")
    result = score_windows(LONG_HUMAN_TEXT)
    if result['n_windows'] > 0:
        check("human text max_window_score < AI text",
              result['max_window_score'] < score_windows(LONG_AI_TEXT)['max_window_score'],
              f"human={result['max_window_score']}")
    else:
        check("human text has no windows (short)", True)


def test_hot_span():
    print("\n-- Hot span counting --")
    result = score_windows(LONG_AI_TEXT)
    check("hot_span_length is int", isinstance(result['hot_span_length'], int))
    check("hot_span_length >= 0", result['hot_span_length'] >= 0)


def test_channel_none_returns_green():
    print("\n-- Channel with None returns GREEN --")
    ch = score_windowed(None)
    check("severity == GREEN", ch.severity == 'GREEN',
          f"got {ch.severity}")
    check("score == 0.0", ch.score == 0.0,
          f"got {ch.score}")


def test_channel_empty_windows():
    print("\n-- Channel with empty window result returns GREEN --")
    ch = score_windowed(window_result={'n_windows': 0, 'max_window_score': 0.0,
                                        'mean_window_score': 0.0, 'window_variance': 0.0,
                                        'hot_span_length': 0, 'mixed_signal': False})
    check("severity == GREEN for empty windows", ch.severity == 'GREEN')


def test_channel_high_hot_span():
    print("\n-- Channel with high hot span --")
    ch = score_windowed(window_result={
        'max_window_score': 0.65,
        'mean_window_score': 0.50,
        'window_variance': 0.01,
        'hot_span_length': 4,
        'n_windows': 5,
        'mixed_signal': False,
    })
    check("high hot span produces RED", ch.severity == 'RED',
          f"got {ch.severity}")


if __name__ == '__main__':
    print("=" * 70)
    print("  WINDOWED SCORING TESTS")
    print("=" * 70)

    test_short_text_empty_windows()
    test_ai_text_produces_scores()
    test_human_text_low_scores()
    test_hot_span()
    test_channel_none_returns_green()
    test_channel_empty_windows()
    test_channel_high_hot_span()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 70}")
    sys.exit(1 if FAILED else 0)
