# LLM Authorship Signal Analyzer for Human Data Pipelines

A stylometric detection pipeline that identifies LLM-generated or LLM-assisted task prompts submitted through human data collection workflows. Designed for quality assurance in benchmark construction (GDPval-style evaluation tasks), clinical education assessments, and any pipeline where humans are expected to author original prompts but may submit LLM-generated content instead.

## The Problem

Human data pipelines — where workers author task prompts, evaluation scenarios, or assessment items — are vulnerable to a specific failure mode: a contributor uses an LLM to generate their submission rather than writing it themselves. This degrades data quality because LLM-generated prompts exhibit systematic biases in structure, vocabulary, and specification patterns that contaminate the resulting benchmark.

Standard AI-text detectors (GPTZero, Originality.ai, etc.) are trained on prose and perform poorly on task prompts, which are inherently instructional and specification-heavy. This tool is purpose-built for that domain.

## How It Works

The pipeline analyzes text across multiple independent layers, each targeting a different authorship signal. No single layer is definitive — the system combines evidence across layers using priority-based aggregation and multi-layer convergence logic.

### Detection Layers

All detection logic lives in `llm_detector_monolith.py` (v0.65):

| Layer | Description |
|-------|-------------|
| **Preamble** | Catches LLM output artifacts: assistant acknowledgments, artifact delivery frames, first-person creation claims, meta-design language, chain-of-thought leakage (`<think>` tags, reasoning-model self-correction phrases) |
| **Fingerprint** | 32-word tiered lexicon of LLM-preferred vocabulary (diagnostic only, not standalone trigger) |
| **Prompt Signature** | Structural patterns of LLM-generated prompts: Constraint Frame Density, Must-Frame Saturation Rate, meta-evaluation design language |
| **Voice Dissonance** | Measures contradiction between casual voice markers and technical specification density |
| **Instruction Density** | Counts formal-exhaustive specification patterns: imperatives, conditionals, binary specs |
| **Semantic Resonance** | Cosine similarity of sentence embeddings against AI/human archetype centroids |
| **Token Cohesiveness (TOCSIN)** | Semantic distance under random word deletion — AI text maintains higher cohesion (Ma & Wang, EMNLP 2024) |
| **Self-Similarity** | N-gram Self-Similarity Index (NSSI) — 13 signals including structural compression delta (s13) |
| **Continuation (API)** | DNA-GPT divergent continuation analysis via Anthropic/OpenAI API |
| **Continuation (Local)** | Multi-truncation DNA-GPT proxy: backoff n-gram LM at gamma=0.3/0.5/0.7 with stability scoring |
| **Perplexity** | distilgpt2-based perplexity with DivEye surprisal variance, volatility decay, and compression-perplexity divergence (Carlini et al. 2021) |
| **Windowed** | Sentence-window scoring with FW trajectory, compression profile, and CUSUM changepoint detection |
| **Surprisal Trajectory** | Windowed analysis of per-token loss stationarity across the text |

### Scoring Channels

Signals are organized into four independent scoring channels:

| Channel | Primary Layers |
|---------|----------------|
| **Prompt Structure** | Preamble, Prompt Signature, Voice Dissonance, Instruction Density |
| **Stylometric** | Self-Similarity (NSSI), Semantic Resonance, TOCSIN, Perplexity, Fingerprint |
| **Continuation** | Multi-truncation Continuation (API or Local), NCD matrix, Surprisal improvement curve |
| **Windowed** | Sentence-window scoring, FW trajectory CV, Compression trajectory, CUSUM changepoint |

### Determination Levels

| Level | Meaning | Action |
|-------|---------|--------|
| RED | Strong evidence of LLM generation | Flag for review, likely reject |
| AMBER | Substantial evidence, high confidence | Flag for manual review |
| YELLOW | Minor signals or convergence pattern | Note for awareness, may be legitimate |
| GREEN | No significant signals detected | Pass |

## File Structure

```
llm_detector_monolith.py       # All detection logic in a single module
tests/                         # Test suite (10 test files, 220+ tests)
run_detector                   # Thin CLI launcher
pyproject.toml                 # Package metadata & dependencies
llm_detector.spec              # PyInstaller build spec
```

## Installation

```bash
pip install openpyxl pandas
# Optional (improves sentence segmentation):
pip install spacy
# Optional (semantic resonance + TOCSIN):
pip install sentence-transformers scikit-learn
# Optional (perplexity scoring + surprisal variance):
pip install transformers torch
# Optional (robust Unicode normalization):
pip install ftfy
# Optional (PDF input):
pip install pypdf
# Optional (DNA-GPT API continuation):
pip install anthropic  # or: pip install openai
```

## Usage

### Single Text Analysis

```bash
python llm_detector_monolith.py --text "Your prompt text here"
# or
./run_detector --text "Your prompt text here"
```

### Desktop GUI

```bash
python llm_detector_monolith.py --gui
```

### File Mode (XLSX/CSV/PDF)

```bash
python llm_detector_monolith.py input.xlsx --sheet "Sheet1" --prompt-col "prompt"
python llm_detector_monolith.py input.csv --prompt-col "content"
python llm_detector_monolith.py document.pdf
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--text` | Analyze a single text string |
| `--gui` | Launch desktop GUI mode |
| `--sheet` | Sheet name for XLSX files |
| `--prompt-col` | Column name containing prompts (default: "prompt") |
| `--verbose`, `-v` | Show all layer details for every result |
| `--output`, `-o` | Output CSV path |
| `--attempter` | Filter by attempter name (substring match) |
| `--no-similarity` | Skip cross-submission similarity analysis |
| `--similarity-threshold` | Jaccard threshold for similarity flagging (default: 0.40) |
| `--no-layer3` | Skip continuation analysis entirely |
| `--api-key` | API key for DNA-GPT continuation analysis |
| `--provider` | LLM provider: `anthropic` or `openai` (default: anthropic) |
| `--mode` | Detection mode: `task_prompt`, `generic_aigt`, or `auto` (default: auto) |
| `--collect PATH` | Append scored results to JSONL for baseline accumulation |
| `--analyze-baselines JSONL` | Compute percentile distributions from accumulated data |
| `--calibrate JSONL` | Build calibration table from labeled baselines |
| `--cal-table JSON` | Path to calibration table JSON |

### Python API

```python
from llm_detector_monolith import analyze_prompt

result = analyze_prompt(
    text="You are a board-certified pharmacist. Analyze the following...",
    task_id="task_001",
    occupation="pharmacist",
    attempter="worker_42",
)

print(result['determination'])       # RED / AMBER / YELLOW / GREEN
print(result['reason'])              # Primary signal description
print(result['confidence'])          # 0.0 - 1.0

# Channel details and agreement
print(result['channel_details'])     # Per-channel severity/score/explanation
print(result['channel_details']['channel_agreement'])  # AGREE / DISAGREE

# Layer-level diagnostics
print(result['voice_dissonance_vsd'])            # Voice-Specification Dissonance
print(result['prompt_signature_composite'])      # Prompt signature composite
print(result['instruction_density_idi'])         # Instruction Density Index

# Perplexity + compression-perplexity divergence (requires transformers + torch)
print(result['perplexity_value'])                # Mean perplexity
print(result['perplexity_surprisal_variance'])   # Token-level surprisal variance
print(result['perplexity_volatility_decay'])     # First/second half variance ratio
print(result['perplexity_zlib_normalized_ppl'])  # zlib-normalized perplexity
print(result['perplexity_comp_ppl_ratio'])       # Compression-perplexity ratio

# Continuation (multi-truncation)
print(result['continuation_composite'])          # DNA-GPT local composite
print(result['continuation_composite_stability'])# Stability across gamma=0.3/0.5/0.7
print(result['continuation_ncd_matrix_mean'])    # Multi-segment NCD mean
print(result['continuation_improvement_rate'])   # Surprisal improvement rate

# Windowed analysis
print(result['window_fw_trajectory_cv'])         # Function word trajectory CV
print(result['window_comp_trajectory_cv'])       # Compression trajectory CV
print(result['window_changepoint'])              # CUSUM changepoint (or None)

# Surprisal trajectory
print(result['surprisal_stationarity'])          # Loss stationarity score
print(result['surprisal_trajectory_cv'])         # Trajectory coefficient of variation

# Token cohesiveness (TOCSIN)
print(result['tocsin_cohesiveness'])             # Semantic cohesion under deletion

# Conformal calibration
print(result['calibrated_confidence'])           # Calibrated confidence
print(result['confidence_quantile'])             # Conformal quantile

# Cross-submission similarity (batch mode)
from llm_detector_monolith import analyze_similarity
results = [analyze_prompt(t['prompt'], ...) for t in tasks]
text_map = {r['task_id']: t['prompt'] for r, t in zip(results, tasks)}
flags = analyze_similarity(results, text_map)
```

## Testing

```bash
python tests/test_pipeline.py
python tests/test_analyzers.py
python tests/test_continuation_local.py
python tests/test_continuation_multi.py
python tests/test_compressibility.py
python tests/test_fusion.py
python tests/test_normalize.py
python tests/test_calibration.py
python tests/test_lexicon.py
python tests/test_windowed.py
python tests/test_preamble_cot.py
```

## Design Principles

**Density over presence.** Individual prompt-engineering patterns (role-setting, format directives) are expected in human-authored prompts. The signal is the *density* and formulaic stacking of multiple categories — not any single pattern.

**No single-layer vetoes.** Every layer can be defeated individually. The convergence floor ensures that when multiple layers whisper, the system still listens.

**Voice gate preserves specificity.** Voice Dissonance requires actual casual voice absence, not just specification presence. A human writing formal text naturally varies more than an LLM generating sterile specifications.

**Diagnostic layers inform but don't trigger.** Fingerprint analysis participates in convergence and similarity analysis but doesn't fire standalone signals — the false positive rate on individual vocabulary items is too high. New v0.65 features (TOCSIN, surprisal trajectory, NCD matrix, compression trajectory, changepoint) emit diagnostics only until calibrated.

**Supporting signals support.** Perplexity, surprisal variance, and volatility decay act as supporting signals only — they boost confidence when other channels already fire, but never promote severity on their own.

**Audit trail by default.** Every determination includes the primary signal, all supporting signals, and full layer-level diagnostics. Nothing is hidden from the reviewer.

## License

MIT
