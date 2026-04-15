"""
Lazy installer — ensures dependencies are available, installs on first use.

Handles: ffmpeg, whisper-cpp (brew), whisper-cpp models (HuggingFace download),
whisperx (pip into server's venv).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Model URLs on HuggingFace
_HF_MODEL_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Map friendly names to GGML filenames
WHISPER_CPP_MODELS = {
    "base": "ggml-base.bin",
    "base.en": "ggml-base.en.bin",
    "small": "ggml-small.bin",
    "small.en": "ggml-small.en.bin",
    "medium": "ggml-medium.bin",
    "medium.en": "ggml-medium.en.bin",
    "large-v2": "ggml-large-v2.bin",
    "large-v3": "ggml-large-v3.bin",
    "large-v3-turbo": "ggml-large-v3-turbo.bin",
}

# Default model directory
_MODELS_DIR = Path.home() / ".cache" / "whisper-cpp-models"


def _log(msg: str):
    print(f"[installer] {msg}", file=sys.stderr, flush=True)


def ensure_ffmpeg() -> bool:
    """Ensure ffmpeg and ffprobe are available. Install via brew if not."""
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True

    _log("ffmpeg not found, installing via brew...")
    try:
        subprocess.run(
            ["brew", "install", "ffmpeg"],
            capture_output=True, text=True, check=True, timeout=600,
        )
        _log("ffmpeg installed successfully")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        _log(f"Failed to install ffmpeg: {e}")
        raise RuntimeError(
            "ffmpeg is required but could not be installed. "
            "Install manually: brew install ffmpeg"
        )


def ensure_whisper_cpp() -> str:
    """
    Ensure whisper-cpp CLI is available. Install via brew if not.
    Returns path to the whisper-cli binary.
    Note: brew package is 'whisper-cpp' but binary is 'whisper-cli'.
    """
    # Check common binary names
    for name in ("whisper-cli", "whisper-cpp"):
        path = shutil.which(name)
        if path:
            return path
    # Check brew prefix directly
    brew_bin = "/opt/homebrew/bin/whisper-cli"
    if os.path.exists(brew_bin):
        return brew_bin

    _log("whisper-cpp not found, installing via brew...")
    try:
        subprocess.run(
            ["brew", "install", "whisper-cpp"],
            capture_output=True, text=True, check=True, timeout=600,
        )
        _log("whisper-cpp installed successfully")
        # After install, check again
        for name in ("whisper-cli", "whisper-cpp"):
            path = shutil.which(name)
            if path:
                return path
        if os.path.exists(brew_bin):
            return brew_bin
        raise RuntimeError("whisper-cpp installed but binary not found")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        _log(f"Failed to install whisper-cpp: {e}")
        raise RuntimeError(
            "whisper-cpp could not be installed. "
            "Install manually: brew install whisper-cpp"
        )


def ensure_whisper_cpp_model(model_name: str = "base") -> str:
    """
    Ensure a whisper.cpp GGML model is downloaded.
    Downloads from HuggingFace if not present.
    Returns absolute path to the model file.
    """
    if model_name not in WHISPER_CPP_MODELS:
        available = ", ".join(sorted(WHISPER_CPP_MODELS.keys()))
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    filename = WHISPER_CPP_MODELS[model_name]
    model_path = _MODELS_DIR / filename

    if model_path.exists():
        return str(model_path)

    # Download from HuggingFace
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"{_HF_MODEL_BASE}/{filename}"
    _log(f"Downloading model {model_name} ({filename}) from HuggingFace...")

    try:
        subprocess.run(
            ["curl", "-fSL", "--progress-bar", "-o", str(model_path), url],
            check=True, timeout=1800,  # 30 min for large models
        )
        _log(f"Model downloaded: {model_path}")
        return str(model_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # Clean up partial download
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model {model_name}: {e}")


def ensure_whisperx() -> bool:
    """
    Ensure whisperx Python package is available in the server's venv.
    Installs via pip if not found.
    """
    try:
        import importlib.util
        if importlib.util.find_spec("whisperx"):
            return True
    except (ImportError, ModuleNotFoundError):
        pass

    _log("whisperx not found, installing via pip (this may take a few minutes)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "whisperx"],
            capture_output=True, text=True, check=True, timeout=600,
        )
        _log("whisperx installed successfully")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        _log(f"Failed to install whisperx: {e}")
        raise RuntimeError(
            "whisperx could not be installed. "
            "Try manually: pip install whisperx"
        )


def convert_to_wav(input_path: str, output_path: str) -> str:
    """Convert any audio file to WAV 16kHz mono (required by whisper.cpp)."""
    ensure_ffmpeg()
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            output_path,
        ],
        capture_output=True, check=True, timeout=300,
    )
    return output_path
