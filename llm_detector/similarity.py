"""Cross-submission similarity analysis."""

import re
import math
from collections import defaultdict


def _word_shingles(text, k=3):
    words = re.findall(r'\w+', text.lower())
    if len(words) < k:
        return {tuple(words)} if words else set()
    return set(tuple(words[i:i+k]) for i in range(len(words) - k + 1))


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


_STRUCT_FEATURES = [
    # Original v0.53 features
    'prompt_signature_composite', 'prompt_signature_cfd', 'prompt_signature_mfsr',
    'prompt_signature_must_rate', 'instruction_density_idi',
    'voice_dissonance_spec_score', 'voice_dissonance_voice_score',
    # v0.58+ additions
    'self_similarity_nssi_score', 'self_similarity_comp_ratio',
    'self_similarity_hapax_ratio', 'self_similarity_sent_length_cv',
    'window_max_score', 'window_mean_score',
    'stylo_fw_ratio', 'stylo_ttr', 'stylo_sent_dispersion',
]


def _structural_similarity(r1, r2):
    diff_sq = sum((r1.get(f, 0) - r2.get(f, 0)) ** 2 for f in _STRUCT_FEATURES)
    return 1.0 / (1.0 + math.sqrt(diff_sq))


def analyze_similarity(results, text_map, jaccard_threshold=0.40, struct_threshold=0.90):
    """Analyze cross-submission similarity within occupation groups."""
    by_occ = defaultdict(list)
    for r in results:
        occ = r.get('occupation', '(unknown)')
        by_occ[occ].append(r)

    shingle_cache = {}
    for tid, text in text_map.items():
        shingle_cache[tid] = _word_shingles(text)

    flagged_pairs = []

    for occ, group in by_occ.items():
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                r_a, r_b = group[i], group[j]
                att_a = r_a.get('attempter', '').strip().lower()
                att_b = r_b.get('attempter', '').strip().lower()

                if att_a and att_b and att_a == att_b:
                    continue

                tid_a = r_a.get('task_id', '')
                tid_b = r_b.get('task_id', '')

                jac = _jaccard(
                    shingle_cache.get(tid_a, set()),
                    shingle_cache.get(tid_b, set())
                )
                struct = _structural_similarity(r_a, r_b)

                flags = []
                if jac >= jaccard_threshold:
                    flags.append('text')
                if struct >= struct_threshold:
                    flags.append('structural')

                if flags:
                    flagged_pairs.append({
                        'id_a': tid_a,
                        'id_b': tid_b,
                        'attempter_a': r_a.get('attempter', ''),
                        'attempter_b': r_b.get('attempter', ''),
                        'occupation': occ,
                        'jaccard': jac,
                        'structural': struct,
                        'flag_type': '+'.join(flags),
                        'det_a': r_a['determination'],
                        'det_b': r_b['determination'],
                    })

    flagged_pairs.sort(key=lambda p: p['jaccard'], reverse=True)
    return flagged_pairs


def print_similarity_report(pairs):
    """Print cross-submission similarity findings."""
    if not pairs:
        print("\n  No cross-attempter similarity clusters detected.")
        return

    print(f"\n{'='*90}")
    print(f"  SIMILARITY CLUSTERS: {len(pairs)} flagged pairs")
    print(f"{'='*90}")

    for p in pairs:
        icon = 'RED' if p['jaccard'] >= 0.70 else 'AMBER' if p['jaccard'] >= 0.50 else 'YELLOW'
        print(f"\n  [{icon}] Jaccard={p['jaccard']:.2f}  Struct={p['structural']:.2f}  [{p['flag_type']}]")
        print(f"     {p['id_a'][:15]:15s} ({p['attempter_a'] or '?':20s}) [{p['det_a']}]")
        print(f"     {p['id_b'][:15]:15s} ({p['attempter_b'] or '?':20s}) [{p['det_b']}]")
        print(f"     Occupation: {p['occupation'][:50]}")
