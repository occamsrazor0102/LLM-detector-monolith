"""
Feature detection and optional dependency management.

Centralizes all try/except ImportError blocks so other modules can check
availability flags without repeating import logic.
"""

import re

# ── tkinter ──────────────────────────────────────────────────────────────────
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

# ── spaCy: lightweight sentencizer ──────────────────────────────────────────
try:
    import spacy
    from spacy.lang.en import English
    _nlp = English()
    _nlp.add_pipe("sentencizer")
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("INFO: spacy not installed. Sentence segmentation will use regex fallback.")
except Exception as e:
    HAS_SPACY = False
    print(f"INFO: spacy sentencizer setup failed ({e}). Using regex fallback.")

# ── ftfy: robust text encoding repair ───────────────────────────────────────
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# ── sentence-transformers: semantic vector analysis ─────────────────────────
HAS_SEMANTIC = False
_EMBEDDER = None
_AI_CENTROIDS = None
_HUMAN_CENTROIDS = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    import numpy as np

    _EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')

    _AI_ARCHETYPES = [
        "As an AI language model, I cannot provide personal opinions.",
        "Here is a comprehensive breakdown of the key factors to consider.",
        "To address this challenge, we must consider multiple perspectives.",
        "This thorough analysis demonstrates the critical importance of the topic.",
        "Furthermore, it is essential to note that this approach ensures alignment.",
        "In conclusion, by leveraging these strategies we can achieve optimal results.",
    ]
    _HUMAN_ARCHETYPES = [
        "honestly idk maybe try restarting it lol",
        "so I went ahead and just hacked together a quick script",
        "tbh the whole thing is kinda janky but it works",
        "yeah no that's totally wrong, here's what actually happened",
        "I messed around with it for a bit and got something working",
    ]
    _AI_CENTROIDS = _EMBEDDER.encode(_AI_ARCHETYPES)
    _HUMAN_CENTROIDS = _EMBEDDER.encode(_HUMAN_ARCHETYPES)
    HAS_SEMANTIC = True
except ImportError:
    pass
except Exception as e:
    print(f"INFO: sentence-transformers setup failed ({e}). Semantic layer disabled.")

# ── transformers: local perplexity scoring ──────────────────────────────────
HAS_PERPLEXITY = False
_PPL_MODEL = None
_PPL_TOKENIZER = None

try:
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import torch as _torch

    _PPL_MODEL_ID = 'distilgpt2'
    _PPL_MODEL = GPT2LMHeadModel.from_pretrained(_PPL_MODEL_ID)
    _PPL_TOKENIZER = GPT2TokenizerFast.from_pretrained(_PPL_MODEL_ID)
    _PPL_MODEL.eval()
    HAS_PERPLEXITY = True
except ImportError:
    pass
except Exception as e:
    print(f"INFO: transformers/torch setup failed ({e}). Perplexity scoring disabled.")

# ── pypdf: PDF text extraction ──────────────────────────────────────────────
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
