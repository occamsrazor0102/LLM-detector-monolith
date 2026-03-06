"""DNA-GPT Divergent Continuation Analysis via LLM API.

Truncates candidate text, regenerates continuations via LLM API,
measures n-gram overlap (BScore) between original and regenerated.
Ref: Yang et al. (2024) "DNA-GPT" (ICLR 2024)
"""

import re
import statistics


def _dna_ngrams(tokens, n):
    """Generate n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def _dna_bscore(original_tokens, regenerated_tokens, ns=(2, 3, 4), weights=(0.25, 0.50, 0.25)):
    """Compute DNA-GPT BScore: weighted n-gram overlap."""
    scores = []
    for n, w in zip(ns, weights):
        orig_ng = set(_dna_ngrams(original_tokens, n))
        regen_ng = set(_dna_ngrams(regenerated_tokens, n))
        if not orig_ng or not regen_ng:
            scores.append(0.0)
            continue
        overlap = len(orig_ng & regen_ng)
        precision = overlap / len(regen_ng) if regen_ng else 0
        recall = overlap / len(orig_ng) if orig_ng else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores.append(f1 * w)
    return sum(scores)


def _dna_truncate_text(text, ratio=0.5):
    """Truncate text at sentence boundary. Returns (prefix, continuation)."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if len(sentences) < 4:
        words = text.split()
        mid = int(len(words) * ratio)
        return ' '.join(words[:mid]), ' '.join(words[mid:])
    cut = max(2, int(len(sentences) * ratio))
    return ' '.join(sentences[:cut]), ' '.join(sentences[cut:])


def _dna_call_anthropic(prefix, continuation_length, api_key,
                        model='claude-sonnet-4-20250514', n_samples=3, temperature=0.7):
    """Generate continuations using Anthropic API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic  (required for continuation analysis with Anthropic)")
    client = anthropic.Anthropic(api_key=api_key)
    continuations = []
    max_tokens = min(max(continuation_length * 2, 200), 4096)
    for _ in range(n_samples):
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, temperature=temperature,
            messages=[{"role": "user",
                       "content": f"Continue the following text naturally, maintaining the same style, tone, and topic. Do not add any preamble or meta-commentary — just continue writing:\n\n{prefix}"}]
        )
        continuations.append(msg.content[0].text if msg.content else "")
    return continuations


DNA_GPT_STORED_PROMPT_ID = 'pmpt_69a8ff3fd48081938b2de58954245ebf0f4f01733906fee0'


def _dna_call_openai(prefix, continuation_length, api_key,
                     model='gpt-4o-mini', n_samples=3, temperature=0.7):
    """Generate continuations using OpenAI Responses API with stored prompt."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai  (required for continuation analysis with OpenAI)")
    client = openai.OpenAI(api_key=api_key)
    continuations = []
    max_tokens = min(max(continuation_length * 2, 200), 4096)
    for _ in range(n_samples):
        resp = client.responses.create(
            model=model,
            max_output_tokens=max_tokens,
            temperature=temperature,
            instructions={
                "type": "stored_prompt",
                "id": DNA_GPT_STORED_PROMPT_ID,
            },
            input=prefix,
        )
        continuations.append(resp.output_text or "")
    return continuations


def run_continuation_api(text, api_key=None, provider='anthropic', model=None,
                         truncation_ratio=0.5, n_samples=3, temperature=0.7):
    """DNA-GPT divergent continuation analysis via LLM API."""
    word_count = len(text.split())

    if word_count < 150:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: insufficient text',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    if not api_key:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: no API key provided',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    prefix, original_continuation = _dna_truncate_text(text, truncation_ratio)
    if len(original_continuation.split()) < 30:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: continuation too short after truncation',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    orig_tokens = original_continuation.lower().split()
    continuation_word_count = len(orig_tokens)

    if model is None:
        model = 'claude-sonnet-4-20250514' if provider == 'anthropic' else 'gpt-4o-mini'

    try:
        if provider == 'anthropic':
            continuations = _dna_call_anthropic(prefix, continuation_word_count, api_key,
                                                model, n_samples, temperature)
        elif provider == 'openai':
            continuations = _dna_call_openai(prefix, continuation_word_count, api_key,
                                             model, n_samples, temperature)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")
    except Exception as e:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': f'DNA-GPT: API call failed ({e})',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    sample_scores = []
    for regen_text in continuations:
        regen_tokens = regen_text.lower().split()
        if len(regen_tokens) < 10:
            continue
        regen_tokens = regen_tokens[:int(len(orig_tokens) * 1.5)]
        bs = _dna_bscore(orig_tokens, regen_tokens)
        sample_scores.append(round(bs, 4))

    if not sample_scores:
        return {'bscore': 0.0, 'bscore_samples': [], 'determination': None,
                'confidence': 0.0, 'reason': 'DNA-GPT: all regenerations failed or too short',
                'n_samples': 0, 'truncation_ratio': truncation_ratio, 'word_count': word_count}

    bscore = statistics.mean(sample_scores)
    bscore_max = max(sample_scores)

    if bscore >= 0.20 and bscore_max >= 0.22:
        det, conf = 'RED', min(0.90, 0.60 + bscore)
        reason = f"DNA-GPT: high continuation overlap (BScore={bscore:.3f}, max={bscore_max:.3f})"
    elif bscore >= 0.12:
        det, conf = 'AMBER', min(0.70, 0.40 + bscore)
        reason = f"DNA-GPT: elevated continuation overlap (BScore={bscore:.3f})"
    elif bscore >= 0.08:
        det, conf = 'YELLOW', min(0.40, 0.20 + bscore)
        reason = f"DNA-GPT: moderate continuation overlap (BScore={bscore:.3f})"
    else:
        det, conf = 'GREEN', max(0.0, 0.10 - bscore)
        reason = f"DNA-GPT: low continuation overlap (BScore={bscore:.3f}) -- likely human"

    return {
        'bscore': round(bscore, 4), 'bscore_max': round(bscore_max, 4),
        'bscore_samples': sample_scores, 'determination': det,
        'confidence': round(conf, 4), 'reason': reason,
        'n_samples': len(sample_scores), 'truncation_ratio': truncation_ratio,
        'continuation_words': continuation_word_count, 'word_count': word_count,
    }
