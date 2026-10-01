import os
import sys
import logging
import urllib.request
from typing import Optional, List, Dict, Any

logger = logging.getLogger("malevolentslice")

# Online Silero VAD download sources
SILERO_VAD_URLS = {
    "v5": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
    "v4": "https://raw.githubusercontent.com/snakers4/silero-vad/v4.0/files/silero_vad.onnx"
}

def is_frozen_bundle() -> bool:
    """Returns True if the application is running inside a PyInstaller frozen bundle."""
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


def get_base_dir() -> str:
    """
    Returns the root directory for assets and resources.
    In a PyInstaller bundle, this resolves to sys._MEIPASS.
    In development or normal Python package mode, this points to the package root.
    """
    if is_frozen_bundle():
        return getattr(sys, "_MEIPASS")
    # malevolentslice package directory: src/malevolentslice/utils/../.. -> malevolentslice
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_user_cache_dir() -> str:
    """Returns the dedicated user cache directory (~/.cache/malevolentslice)."""
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "malevolentslice")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_models_cache_dir() -> str:
    """Returns the cache directory for AI models (~/.cache/malevolentslice/models)."""
    models_dir = os.path.join(get_user_cache_dir(), "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def resolve_asset_path(filename: str) -> Optional[str]:
    """
    Resolves the absolute path of a bundled asset file across PyInstaller bundles,
    package directories, development repository root, and user cache.
    """
    candidates = []

    # 1. PyInstaller MEIPASS locations
    if is_frozen_bundle():
        meipass = getattr(sys, "_MEIPASS")
        candidates.append(os.path.join(meipass, "malevolentslice", "assets", filename))
        candidates.append(os.path.join(meipass, "assets", filename))
        candidates.append(os.path.join(meipass, filename))

    # 2. Package-internal assets directory (src/malevolentslice/assets)
    pkg_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", filename))
    candidates.append(pkg_assets)

    # 3. Development root repository assets directory (../../assets)
    dev_root_assets = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", filename))
    candidates.append(dev_root_assets)

    # 4. User cache directory
    cache_path = os.path.join(get_user_cache_dir(), filename)
    candidates.append(cache_path)
    models_cache_path = os.path.join(get_models_cache_dir(), filename)
    candidates.append(models_cache_path)

    for path in candidates:
        if os.path.isfile(path):
            return os.path.abspath(path)

    return None


def get_available_vad_models() -> List[Dict[str, str]]:
    """
    Returns list of discovered VAD models with friendly display names and file paths.
    """
    models = []
    found_keys = set()

    # Known defaults
    v5_path = resolve_asset_path("silero_vad_v5.onnx")
    if v5_path:
        models.append({"name": "Silero VAD v5 (Latest, Recommended)", "path": v5_path, "version": "v5"})
        found_keys.add("v5")

    v4_path = resolve_asset_path("silero_vad_v4.onnx") or resolve_asset_path("silero_vad.onnx")
    if v4_path:
        models.append({"name": "Silero VAD v4 (Legacy)", "path": v4_path, "version": "v4"})
        found_keys.add("v4")

    # If neither found, add placeholder entry for v5 download
    if not models:
        models.append({"name": "Silero VAD v5 (Download on Start)", "path": "v5", "version": "v5"})

    return models


def resolve_or_download_vad_model(model_choice: Optional[str] = None) -> str:
    """
    Resolves the VAD model path based on a choice name or file path.
    Downloads to user cache if required.
    """
    if model_choice and os.path.isfile(model_choice):
        return os.path.abspath(model_choice)

    # Check if choice is v4
    is_v4 = False
    if model_choice and ("v4" in model_choice.lower() or "legacy" in model_choice.lower()):
        is_v4 = True

    target_filename = "silero_vad_v4.onnx" if is_v4 else "silero_vad_v5.onnx"
    resolved = resolve_asset_path(target_filename)
    if resolved and os.path.isfile(resolved):
        return resolved

    # Fallback to default silero_vad.onnx
    fallback = resolve_asset_path("silero_vad.onnx")
    if fallback and os.path.isfile(fallback):
        return fallback

    # Need to download
    download_key = "v4" if is_v4 else "v5"
    download_url = SILERO_VAD_URLS[download_key]
    dest_path = os.path.join(get_models_cache_dir(), target_filename)

    logger.info(f"Downloading Silero VAD ({download_key}) from {download_url} to {dest_path}...")
    urllib.request.urlretrieve(download_url, dest_path)
    return dest_path
