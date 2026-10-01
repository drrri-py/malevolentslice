# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec file for MalevolentSlice Studio
Standalone Executable Packaging Specification.
"""

import sys
import os
from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Collect all customtkinter assets, themes, and fonts
ctk_datas, ctk_binaries, ctk_hiddenimports = collect_all('customtkinter')

# Core application datas
added_datas = [
    ('src/malevolentslice/assets', 'malevolentslice/assets'),
    ('src/malevolentslice/assets', 'assets'),
] + ctk_datas

added_binaries = [] + ctk_binaries

added_hiddenimports = [
    'malevolentslice',
    'malevolentslice.gui',
    'malevolentslice.gui.app',
    'malevolentslice.pipeline',
    'malevolentslice.core.vad_engine',
    'malevolentslice.core.streamer',
    'malevolentslice.core.segmenter',
    'malevolentslice.core.exporter',
    'malevolentslice.core.noise_gate',
    'malevolentslice.core.transcriber',
    'malevolentslice.utils.path_resolver',
    'malevolentslice.utils.memory',
    'malevolentslice.utils.logger',
    'onnxruntime',
    'soundfile',
    'scipy',
    'scipy.signal',
    'scipy.io.wavfile',
    'faster_whisper',
    'ctranslate2',
    'tokenizers',
    'huggingface_hub',
    'psutil',
    'click',
    'rich',
    'darkdetect',
    'PIL',
    'PIL.Image',
] + ctk_hiddenimports

a = Analysis(
    ['src/malevolentslice/gui/app.py'],
    pathex=['src'],
    binaries=added_binaries,
    datas=added_datas,
    hiddenimports=added_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter.test', 'unittest', 'test'],
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
    name='MalevolentSliceStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MalevolentSliceStudio',
)

if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='MalevolentSliceStudio.app',
        icon=None,
        bundle_identifier='com.fidriyani.malevolentslice',
    )
