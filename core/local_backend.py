"""
Local Whisper backend — transcription using whisperx-mlx on Apple Silicon.

Runs Whisper via MLX (GPU accelerated) with optional speaker diarization
via PyAnnote. Requires `pip install whisperx-mlx` and HuggingFace token
for diarization models.
"""

import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional


DEFAULT_MODEL = "large-v3"


def _check_whisperx_available() -> bool:
    """Check if whisperx-mlx CLI is available."""
    try:
        result = subprocess.run(
            ["whisperx", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def transcribe_local(
    audio_path: str,
    language: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    diarize: bool = False,
    hf_token: Optional[str] = None,
) -> Dict:
    """
    Transcribe audio using whisperx-mlx (local, GPU-accelerated on Apple Silicon).

    Args:
        audio_path: Path to audio file.
        language: ISO-639-1 code (None = auto-detect).
        model: Whisper model name (e.g., "large-v3", "base", "small").
        diarize: Enable speaker diarization (requires HF_TOKEN).
        hf_token: HuggingFace token for PyAnnote diarization models.

    Returns:
        Dict with: model, language, duration, segments, full_text.
    """
    if not _check_whisperx_available():
        raise RuntimeError(
            "whisperx-mlx is not installed. Install with: pip install whisperx-mlx[mlx]"
        )

    with tempfile.TemporaryDirectory() as output_dir:
        cmd = [
            "whisperx",
            audio_path,
            "--model", model,
            "--output_dir", output_dir,
            "--output_format", "json",
        ]

        if language:
            cmd.extend(["--language", language])

        if diarize:
            if not hf_token:
                raise ValueError(
                    "HuggingFace token required for diarization. "
                    "Set HF_TOKEN in .env or pass hf_token parameter."
                )
            cmd.extend(["--diarize", "--hf_token", hf_token])

        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=1800,  # 30 min timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"whisperx-mlx failed: {result.stderr}")

        # Find output JSON file
        json_files = list(Path(output_dir).glob("*.json"))
        if not json_files:
            raise RuntimeError(
                f"whisperx-mlx produced no output. stderr: {result.stderr}"
            )

        with open(json_files[0], "r", encoding="utf-8") as f:
            whisperx_output = json.load(f)

        return _parse_whisperx_output(whisperx_output, model, diarize)


def _parse_whisperx_output(output: dict, model: str, diarize: bool) -> Dict:
    """Parse whisperx-mlx JSON output into our standard format."""
    segments = []

    for seg in output.get("segments", []):
        segment = {
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "").strip(),
        }
        if diarize and "speaker" in seg:
            segment["speaker"] = seg["speaker"]
        segments.append(segment)

    # Calculate duration from last segment
    duration = 0
    if segments:
        last = segments[-1]
        duration = last.get("end", 0)

    full_text = " ".join(s["text"] for s in segments if s["text"])

    return {
        "model": model,
        "backend": "local",
        "language": output.get("language", ""),
        "duration": duration,
        "segments": segments,
        "full_text": full_text,
    }
