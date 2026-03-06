"""Windowed scoring -- detect mixed human+AI content via per-window analysis.

Ref: M4GT-Bench (Wang et al. 2024) -- mixed detection as separate task.
"""

import re
import statistics

from llm_detector.text_utils import ENGLISH_FUNCTION_WORDS, get_sentences
from llm_detector.analyzers.self_similarity import _FORMULAIC_PATTERNS, _TRANSITION, _POWER_ADJ


def score_windows(text, window_size=5, stride=2):
    """Score text in overlapping sentence windows.

    Returns dict with per-window scores, max/mean/variance, hot span, mixed signal.
    """
    sentences = get_sentences(text)
    if len(sentences) < window_size:
        return {
            'windows': [],
            'max_window_score': 0.0,
            'mean_window_score': 0.0,
            'window_variance': 0.0,
            'hot_span_length': 0,
            'n_windows': 0,
            'mixed_signal': False,
        }

    windows = []
    for start in range(0, len(sentences) - window_size + 1, stride):
        end = start + window_size
        window_text = ' '.join(sentences[start:end])
        window_words = window_text.split()
        n_w = max(len(window_words), 1)

        formulaic_count = sum(
            len(re.findall(pat, window_text, re.I))
            for pat, _weight in _FORMULAIC_PATTERNS
        )
        formulaic_density = formulaic_count / (n_w / 100)

        trans_hits = len(_TRANSITION.findall(window_text))
        trans_density = trans_hits / (n_w / 100)

        power_hits = len(_POWER_ADJ.findall(window_text))
        power_density = power_hits / (n_w / 100)

        fw = sum(1 for w in window_words if w.lower() in ENGLISH_FUNCTION_WORDS)
        fw_ratio = fw / n_w

        w_sent_lengths = [len(s.split()) for s in sentences[start:end] if s.strip()]
        if len(w_sent_lengths) >= 2:
            w_mean = statistics.mean(w_sent_lengths)
            w_std = statistics.stdev(w_sent_lengths)
            w_cv = w_std / max(w_mean, 1)
        else:
            w_cv = 0.5

        ai_indicators = 0.0
        if formulaic_density > 2.0:
            ai_indicators += min(formulaic_density / 5.0, 0.3)
        if trans_density > 3.0:
            ai_indicators += min(trans_density / 8.0, 0.2)
        if power_density > 1.5:
            ai_indicators += min(power_density / 4.0, 0.2)
        if w_cv < 0.25 and len(w_sent_lengths) >= 3:
            ai_indicators += 0.15
        if fw_ratio < 0.12:
            ai_indicators += 0.15

        window_score = min(ai_indicators, 1.0)

        windows.append({
            'start': start,
            'end': end,
            'score': round(window_score, 3),
            'formulaic': round(formulaic_density, 2),
            'transitions': round(trans_density, 2),
            'sent_cv': round(w_cv, 3),
        })

    scores = [w['score'] for w in windows]
    max_score = max(scores) if scores else 0.0
    mean_score = statistics.mean(scores) if scores else 0.0
    variance = statistics.variance(scores) if len(scores) >= 2 else 0.0

    hot_threshold = 0.30
    hot_span = 0
    current_span = 0
    for s in scores:
        if s >= hot_threshold:
            current_span += 1
            hot_span = max(hot_span, current_span)
        else:
            current_span = 0

    mixed_signal = variance >= 0.02 and max_score >= 0.30 and mean_score < 0.50

    return {
        'windows': windows,
        'max_window_score': round(max_score, 3),
        'mean_window_score': round(mean_score, 3),
        'window_variance': round(variance, 4),
        'hot_span_length': hot_span,
        'n_windows': len(windows),
        'mixed_signal': mixed_signal,
    }
