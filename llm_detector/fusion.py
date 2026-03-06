"""Evidence fusion -- combines channel results into final determination."""

from llm_detector.channels import ChannelResult
from llm_detector.channels.prompt_structure import score_prompt_structure
from llm_detector.channels.stylometric import score_stylometric
from llm_detector.channels.continuation import score_continuation
from llm_detector.channels.windowed import score_windowed


def _detect_mode(prompt_sig, instr_density, self_sim, word_count):
    """Auto-detect whether text is a task prompt or generic AI text."""
    prompt_signal = 0.0
    if prompt_sig['composite'] >= 0.15:
        prompt_signal += prompt_sig['composite']
    if instr_density and instr_density.get('idi', 0) >= 5:
        prompt_signal += 0.3
    if prompt_sig.get('framing_completeness', 0) >= 2:
        prompt_signal += 0.2

    generic_signal = 0.0
    if self_sim and self_sim.get('nssi_signals', 0) >= 3:
        generic_signal += 0.4
    if word_count >= 400:
        generic_signal += 0.2

    if prompt_signal > generic_signal + 0.1:
        return 'task_prompt'
    elif generic_signal > prompt_signal + 0.1:
        return 'generic_aigt'
    else:
        return 'task_prompt'


def determine(preamble_score, preamble_severity, prompt_sig, voice_dis,
              instr_density=None, word_count=0,
              self_sim=None, cont_result=None, lang_gate=None, norm_report=None,
              mode='auto', fingerprint_score=0.0, semantic=None, ppl=None, **kwargs):
    """Evidence fusion with channel-based corroboration.

    Returns (determination, reason, confidence, channel_details).
    """
    # Mode detection
    if mode == 'auto':
        mode = _detect_mode(prompt_sig, instr_density, self_sim, word_count)

    # Score all channels
    ch_prompt = score_prompt_structure(preamble_score, preamble_severity, prompt_sig, voice_dis, instr_density, word_count)
    ch_style = score_stylometric(fingerprint_score, self_sim, voice_dis, semantic=semantic, ppl=ppl)
    ch_cont = score_continuation(cont_result)
    ch_window = score_windowed(window_result=kwargs.get('window_result'))

    channels = [ch_prompt, ch_style, ch_cont, ch_window]
    channel_details = {
        'mode': mode,
        'channels': {ch.channel: {
            'score': ch.score, 'severity': ch.severity,
            'explanation': ch.explanation, 'mode_eligible': mode in ch.mode_eligibility,
        } for ch in channels},
    }

    # Fairness severity cap
    severity_cap = None
    if lang_gate and lang_gate.get('support_level') == 'UNSUPPORTED':
        severity_cap = 'YELLOW'
    elif lang_gate and lang_gate.get('support_level') == 'REVIEW':
        severity_cap = 'AMBER'

    def _apply_cap(det, reason, conf):
        if severity_cap is None:
            return det, reason, conf
        sev_order = {'GREEN': 0, 'YELLOW': 1, 'REVIEW': 1, 'AMBER': 2, 'RED': 3}
        if sev_order.get(det, 0) > sev_order.get(severity_cap, 3):
            gate_reason = lang_gate.get('reason', 'language support gate')
            return severity_cap, f"{reason} [capped from {det}: {gate_reason}]", min(conf, 0.40)
        return det, reason, conf

    # L0 CRITICAL: instant RED
    if ch_prompt.sub_signals.get('preamble') == 0.99 and preamble_severity == 'CRITICAL':
        det, reason, conf = _apply_cap('RED', ch_prompt.explanation, 0.99)
        return det, reason, conf, channel_details

    # Mode-aware channel filtering
    if mode == 'task_prompt':
        primary_channels = [ch for ch in channels if 'task_prompt' in ch.mode_eligibility]
        supporting_channels = [ch for ch in channels if 'task_prompt' not in ch.mode_eligibility]
    else:
        primary_channels = channels
        supporting_channels = []

    # Evidence fusion
    all_active = sorted(
        [ch for ch in channels if ch.severity != 'GREEN'],
        key=lambda c: c.sev_level, reverse=True,
    )
    primary_active = sorted(
        [ch for ch in primary_channels if ch.severity != 'GREEN'],
        key=lambda c: c.sev_level, reverse=True,
    )
    support_active = [ch for ch in supporting_channels if ch.severity != 'GREEN']

    n_red = sum(1 for ch in all_active if ch.severity == 'RED')
    n_amber_plus = sum(1 for ch in all_active if ch.sev_level >= 2)
    n_yellow_plus = sum(1 for ch in all_active if ch.sev_level >= 1)
    n_primary_red = sum(1 for ch in primary_active if ch.severity == 'RED')
    n_primary_amber = sum(1 for ch in primary_active if ch.sev_level >= 2)
    n_primary_yellow_plus = sum(1 for ch in primary_active if ch.sev_level >= 1)

    top_explanations = [ch.explanation for ch in all_active[:3]]
    combined_reason = ' + '.join(top_explanations) if top_explanations else 'No significant signals'
    top_score = max((ch.score for ch in all_active), default=0.0)

    # RED: strong primary + supporting, or two AMBER+ channels
    if n_primary_red >= 1 and n_yellow_plus >= 2:
        det, reason, conf = _apply_cap('RED', combined_reason, top_score)
        return det, reason, conf, channel_details

    if n_primary_amber >= 2:
        det, reason, conf = _apply_cap('RED', combined_reason, min(top_score, 0.85))
        return det, reason, conf, channel_details

    if mode == 'task_prompt' and n_primary_red >= 1 and n_yellow_plus == 1:
        det, reason, conf = _apply_cap('AMBER', f"{combined_reason} [single-channel, demoted from RED]", min(top_score, 0.75))
        return det, reason, conf, channel_details

    if mode == 'generic_aigt' and n_red >= 1:
        if n_yellow_plus >= 2:
            det, reason, conf = _apply_cap('RED', combined_reason, top_score)
        else:
            det, reason, conf = _apply_cap('RED', f"{combined_reason} [single-channel]", min(top_score, 0.75))
        return det, reason, conf, channel_details

    # AMBER: one channel at AMBER, or two at YELLOW+
    if n_primary_amber >= 1:
        det, reason, conf = _apply_cap('AMBER', combined_reason, min(top_score, 0.70))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.60), channel_details
        return det, reason, conf, channel_details

    if mode == 'task_prompt':
        convergence_count = n_primary_yellow_plus + min(1, len(support_active))
    else:
        convergence_count = n_yellow_plus

    if convergence_count >= 2:
        det, reason, conf = _apply_cap('AMBER', f"{combined_reason} [multi-channel convergence]", min(top_score, 0.60))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.55), channel_details
        return det, reason, conf, channel_details

    # Supporting channels at AMBER in task_prompt mode
    if mode == 'task_prompt' and any(ch.sev_level >= 2 for ch in support_active):
        support_expl = [ch.explanation for ch in support_active if ch.sev_level >= 2]
        det, reason, conf = _apply_cap('AMBER', f"{' + '.join(support_expl)} [supporting channel]", 0.55)
        return det, reason, conf, channel_details

    # YELLOW: one channel at YELLOW+
    if n_yellow_plus >= 1:
        det, reason, conf = _apply_cap('YELLOW', combined_reason, min(top_score, 0.45))
        if ch_window.sub_signals.get('mixed_signal') and ch_window.severity != 'GREEN':
            return 'MIXED', f"{reason} [windowed variance suggests hybrid text]", min(conf, 0.50), channel_details
        return det, reason, conf, channel_details

    # Obfuscation delta
    if norm_report and norm_report.get('obfuscation_delta', 0) >= 0.05:
        delta = norm_report['obfuscation_delta']
        det, reason, conf = _apply_cap('YELLOW', f"Text normalization delta ({delta:.1%}) suggests obfuscation", 0.35)
        return det, reason, conf, channel_details

    # REVIEW: any channel has non-zero score
    any_signal = any(ch.score > 0.05 for ch in channels)
    if any_signal:
        weak_parts = [ch.explanation for ch in channels if ch.score > 0.05]
        return 'REVIEW', f"Weak signals below threshold: {' + '.join(weak_parts[:2])}", 0.10, channel_details

    # GREEN
    return 'GREEN', 'No significant signals', 0.0, channel_details
