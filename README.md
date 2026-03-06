# LLM Authorship Signal Analyzer for Human Data Pipelines

A stylometric detection pipeline that identifies LLM-generated or LLM-assisted task prompts submitted through human data collection workflows. Designed for quality assurance in benchmark construction (GDPval-style evaluation tasks), clinical education assessments, and any pipeline where humans are expected to author original prompts but may submit LLM-generated content instead.

## The Problem

Human data pipelines — where workers author task prompts, evaluation scenarios, or assessment items — are vulnerable to a specific failure mode: a contributor uses an LLM to generate their submission rather than writing it themselves. This degrades data quality because LLM-generated prompts exhibit systematic biases in structure, vocabulary, and specification patterns that contaminate the resulting benchmark.

Standard AI-text detectors (GPTZero, Originality.ai, etc.) are trained on prose and perform poorly on task prompts, which are inherently instructional and specification-heavy. This tool is purpose-built for that domain.

## How It Works

The pipeline analyzes text across multiple independent layers, each targeting a different authorship signal. No single layer is definitive — the system combines evidence across layers using priority-based aggregation and multi-layer convergence logic.

### Detection Layers

| Layer | Module | Description |
|-------|--------|-------------|
| **Preamble** | `analyzers/preamble.py` | Catches LLM output artifacts: assistant acknowledgments, artifact delivery frames, first-person creation claims, meta-design language |
| **Fingerprint** | `analyzers/fingerprint.py` | 120-word tiered lexicon of LLM-preferred vocabulary (diagnostic only, not standalone trigger) |
| **Prompt Signature** | `analyzers/prompt_signature.py` | Structural patterns of LLM-generated prompts: Constraint Frame Density, Must-Frame Saturation Rate, meta-evaluation design language |
| **Voice Dissonance** | `analyzers/voice_dissonance.py` | Measures contradiction between casual voice markers and technical specification density |
| **Instruction Density** | `analyzers/instruction_density.py` | Counts formal-exhaustive specification patterns: imperatives, conditionals, binary specs |
| **Semantic Resonance** | `analyzers/semantic_resonance.py` | Cosine similarity of sentence embeddings against AI/human archetype centroids |
| **Self-Similarity** | `analyzers/self_similarity.py` | N-gram Self-Similarity Index (NSSI) for detecting formulaic LLM patterns |
| **Continuation (API)** | `analyzers/continuation_api.py` | DNA-GPT divergent continuation analysis via Anthropic/OpenAI API |
| **Continuation (Local)** | `analyzers/continuation_local.py` | Zero-LLM DNA-GPT proxy using backoff n-gram language model |
| **Perplexity** | `analyzers/perplexity.py` | distilgpt2-based perplexity scoring |

### Scoring Channels

Signals are organized into four independent scoring channels:

| Channel | Module | Primary Layers |
|---------|--------|----------------|
| **Prompt Structure** | `channels/prompt_structure.py` | Preamble, Prompt Signature, Voice Dissonance, Instruction Density |
| **Stylometric** | `channels/stylometric.py` | Self-Similarity, Semantic Resonance, Perplexity, Fingerprint |
| **Continuation** | `channels/continuation.py` | Continuation API or Continuation Local |
| **Windowed** | `channels/windowed.py` | Sentence-window scoring |

### Determination Levels

| Level | Meaning | Action |
|-------|---------|--------|
| RED | Strong evidence of LLM generation | Flag for review, likely reject |
| AMBER | Substantial evidence, high confidence | Flag for manual review |
| YELLOW | Minor signals or convergence pattern | Note for awareness, may be legitimate |
| GREEN | No significant signals detected | Pass |

## Package Structure

```
llm_detector/                  # Main package
    __init__.py                # Version, public API re-exports
    __main__.py                # python -m llm_detector entry point
    compat.py                  # Feature flags (HAS_SPACY, HAS_FTFY, etc.)
    text_utils.py              # Shared utilities
    normalize.py               # Text normalization
    language_gate.py           # Language/fairness support check
    pipeline.py                # Full pipeline orchestration
    fusion.py                  # Evidence fusion across channels
    calibration.py             # Conformal calibration
    baselines.py               # Baseline collection and analysis
    similarity.py              # Cross-submission similarity
    io.py                      # File I/O (XLSX, CSV, PDF)
    cli.py                     # Command-line interface
    gui.py                     # Desktop GUI

    analyzers/                 # One module per detection layer
        preamble.py
        fingerprint.py
        prompt_signature.py
        voice_dissonance.py
        instruction_density.py
        semantic_resonance.py
        self_similarity.py
        continuation_api.py
        continuation_local.py
        perplexity.py
        stylometry.py
        windowing.py

    channels/                  # Channel scoring
        prompt_structure.py
        stylometric.py
        continuation.py
        windowed.py

    lexicon/                   # Externalized detection vocabulary
        packs.py               # LexiconPack definitions & scoring engine
        integration.py         # Enhanced layer wrappers

tests/                         # Test suite
run_detector                   # Thin CLI launcher
```

## Installation

```bash
pip install openpyxl pandas
# Optional (improves sentence segmentation):
pip install spacy
# Optional (semantic resonance layer):
pip install sentence-transformers scikit-learn
# Optional (perplexity scoring):
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
python -m llm_detector --text "Your prompt text here"
# or
./run_detector --text "Your prompt text here"
```

### Desktop GUI

```bash
python -m llm_detector --gui
```

### File Mode (XLSX/CSV/PDF)

```bash
python -m llm_detector input.xlsx --sheet "Sheet1" --prompt-col "prompt"
python -m llm_detector input.csv --prompt-col "content"
python -m llm_detector document.pdf
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
from llm_detector import analyze_prompt

result = analyze_prompt(
    text="You are a board-certified pharmacist. Analyze the following...",
    task_id="task_001",
    occupation="pharmacist",
    attempter="worker_42",
)

print(result['determination'])       # RED / AMBER / YELLOW / GREEN
print(result['reason'])              # Primary signal description
print(result['confidence'])          # 0.0 - 1.0
print(result['supporting_signals'])  # List of other signals that fired

# Layer-level diagnostics
print(result['voice_dissonance_vsd'])            # Voice-Specification Dissonance
print(result['prompt_signature_composite'])      # Prompt signature composite
print(result['instruction_density_idi'])         # Instruction Density Index

# Cross-submission similarity (batch mode)
from llm_detector import analyze_similarity
results = [analyze_prompt(t['prompt'], ...) for t in tasks]
text_map = {r['task_id']: t['prompt'] for r, t in zip(results, tasks)}
flags = analyze_similarity(results, text_map)
```

## Testing

```bash
python tests/test_pipeline.py
python tests/test_analyzers.py
python tests/test_continuation_local.py
python tests/test_fusion.py
python tests/test_normalize.py
```

## Design Principles

**Density over presence.** Individual prompt-engineering patterns (role-setting, format directives) are expected in human-authored prompts. The signal is the *density* and formulaic stacking of multiple categories — not any single pattern.

**No single-layer vetoes.** Every layer can be defeated individually. The convergence floor ensures that when multiple layers whisper, the system still listens.

**Voice gate preserves specificity.** Voice Dissonance requires actual casual voice absence, not just specification presence. A human writing formal text naturally varies more than an LLM generating sterile specifications.

**Diagnostic layers inform but don't trigger.** Fingerprint analysis participates in convergence and similarity analysis but doesn't fire standalone signals — the false positive rate on individual vocabulary items is too high.

**Audit trail by default.** Every determination includes the primary signal, all supporting signals, and full layer-level diagnostics. Nothing is hidden from the reviewer.

## License

MIT
