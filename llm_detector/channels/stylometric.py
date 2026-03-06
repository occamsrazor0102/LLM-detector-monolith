"""Channel 2: Stylometric signals (generic_aigt primary).

Combines NSSI, semantic resonance, perplexity, and fingerprints.
"""

from llm_detector.channels import ChannelResult


def score_stylometric(fingerprint_score, self_sim, voice_dis=None, semantic=None, ppl=None):
    """Score stylometric channel. Returns ChannelResult."""
    sub = {}
    score = 0.0
    severity = 'GREEN'
    parts = []

    # Fingerprints: supporting-only
    if fingerprint_score > 0:
        sub['fingerprints'] = fingerprint_score

    # NSSI: primary stylometric signal
    if self_sim and self_sim.get('determination'):
        nssi_det = self_sim['determination']
        nssi_score = self_sim.get('nssi_score', 0)
        nssi_signals = self_sim.get('nssi_signals', 0)
        sub['nssi_score'] = nssi_score
        sub['nssi_signals'] = nssi_signals

        if nssi_det == 'RED':
            score = max(score, min(0.85, self_sim.get('confidence', 0.80)))
            severity = 'RED'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(RED)")
        elif nssi_det == 'AMBER':
            score = max(score, min(0.65, self_sim.get('confidence', 0.60)))
            severity = 'AMBER'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(AMBER)")
        elif nssi_det == 'YELLOW':
            score = max(score, min(0.40, self_sim.get('confidence', 0.30)))
            severity = 'YELLOW'
            parts.append(f"NSSI={nssi_score:.2f}/{nssi_signals}sig(YELLOW)")

    # Semantic resonance: supporting signal
    if semantic and semantic.get('determination'):
        sem_det = semantic['determination']
        sem_delta = semantic.get('semantic_delta', 0)
        sub['semantic_ai_score'] = semantic.get('semantic_ai_mean', 0)
        sub['semantic_delta'] = sem_delta

        if sem_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"Sem=AMBER(delta={sem_delta:.2f},boost)")
            else:
                score = max(score, semantic.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"Sem=AMBER(delta={sem_delta:.2f})")
        elif sem_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"Sem=YELLOW(delta={sem_delta:.2f},supporting)")
            else:
                score = max(score, semantic.get('confidence', 0.30))
                severity = 'YELLOW'
                parts.append(f"Sem=YELLOW(delta={sem_delta:.2f})")

    # Perplexity: supporting signal
    if ppl and ppl.get('determination'):
        ppl_det = ppl['determination']
        ppl_val = ppl.get('perplexity', 0)
        sub['perplexity'] = ppl_val

        if ppl_det == 'AMBER':
            if severity in ('RED', 'AMBER'):
                score = min(score + 0.10, 1.0)
                parts.append(f"PPL={ppl_val:.0f}(AMBER,boost)")
            else:
                score = max(score, ppl.get('confidence', 0.55))
                severity = max(severity, 'AMBER',
                               key=lambda s: ChannelResult.SEV_ORDER.get(s, 0))
                parts.append(f"PPL={ppl_val:.0f}(AMBER)")
        elif ppl_det == 'YELLOW':
            if severity != 'GREEN':
                score = min(score + 0.05, 1.0)
                parts.append(f"PPL={ppl_val:.0f}(YELLOW,supporting)")
            else:
                score = max(score, ppl.get('confidence', 0.30))
                severity = 'YELLOW'
                parts.append(f"PPL={ppl_val:.0f}(YELLOW)")

    # Fingerprints add supporting weight if any stylometric signal is active
    if fingerprint_score >= 0.30 and severity != 'GREEN':
        score = min(score + 0.10, 1.0)
        parts.append(f"fingerprint={fingerprint_score:.2f}(supporting)")

    explanation = f"Stylometry: {', '.join(parts)}" if parts else 'Stylometry: no signals'

    return ChannelResult(
        'stylometry', score, severity, explanation,
        mode_eligibility=['generic_aigt'],
        sub_signals=sub,
    )
