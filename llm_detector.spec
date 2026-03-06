# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for LLM Detector Pipeline.

Build with:
    pyinstaller llm_detector.spec

Or for a single-file executable:
    pyinstaller llm_detector.spec --onefile
"""

import sys
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

# Collect all submodules of the package
hiddenimports = collect_submodules('llm_detector')

# Optional deps — include if installed, skip gracefully if not
for mod in ['anthropic', 'openai', 'pypdf', 'spacy', 'ftfy',
            'sentence_transformers', 'sklearn', 'transformers', 'torch']:
    try:
        __import__(mod)
        hiddenimports += collect_submodules(mod)
    except ImportError:
        pass

a = Analysis(
    ['llm_detector/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='llm-detector',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # windowed mode — GUI opens by default
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='llm-detector',
)
