"""Local perplexity scoring via distilgpt2.

AI text has low perplexity (< 20); human text typically > 35.
Ref: GLTR (Gehrmann et al. 2019), DetectGPT (Mitchell et al. 2023)
"""

from llm_detector.compat import HAS_PERPLEXITY, _PPL_MODEL, _PPL_TOKENIZER

if HAS_PERPLEXITY:
    import torch as _torch


def run_perplexity(text):
    """Calculate token-level perplexity using distilgpt2.

    Returns dict with perplexity, determination, and confidence.
    """
    if not HAS_PERPLEXITY:
        return {
            'perplexity': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Perplexity scoring unavailable (transformers/torch not installed)',
        }

    words = text.split()
    if len(words) < 50:
        return {
            'perplexity': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Perplexity: text too short',
        }

    encodings = _PPL_TOKENIZER(text, return_tensors='pt', truncation=True,
                                max_length=1024)
    input_ids = encodings.input_ids

    if input_ids.size(1) < 10:
        return {
            'perplexity': 0.0,
            'determination': None,
            'confidence': 0.0,
            'reason': 'Perplexity: too few tokens after encoding',
        }

    with _torch.no_grad():
        outputs = _PPL_MODEL(input_ids, labels=input_ids)
        loss = outputs.loss

    ppl = _torch.exp(loss).item()

    if ppl <= 15.0:
        det = 'AMBER'
        conf = min(0.65, (20.0 - ppl) / 20.0)
        reason = f"Low perplexity ({ppl:.1f}): highly predictable text"
    elif ppl <= 25.0:
        det = 'YELLOW'
        conf = min(0.35, (30.0 - ppl) / 30.0)
        reason = f"Moderate perplexity ({ppl:.1f}): somewhat predictable"
    else:
        det = None
        conf = 0.0
        reason = f"Normal perplexity ({ppl:.1f}): consistent with human text"

    return {
        'perplexity': round(ppl, 2),
        'determination': det,
        'confidence': conf,
        'reason': reason,
    }
