"""
Local transcription backends.

Two options:
- whisper.cpp: Fast, GPU Metal on Apple Silicon, no diarization
- whisperx: Slower (CPU on Mac), but with speaker diarization via PyAnnote
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

from .installer import (
    ensure_whisper_cpp,
    ensure_whisper_cpp_model,
    ensure_whisperx,
    convert_to_wav,
)


# --- whisper.cpp backend ---

def transcribe_whisper_cpp(
    audio_path: str,
    language: Optional[str] = None,
    model: str = "base",
) -> Dict:
    """
    Transcribe using whisper.cpp (GPU Metal on Apple Silicon).

    Args:
        audio_path: Path to audio file (any format — auto-converted to WAV).
        language: ISO-639-1 code (None = auto-detect).
        model: Model name: "base", "small", "medium", "large-v3", "large-v3-turbo".

    Returns:
        Standard result dict with segments, full_text, etc.
    """
    whisper_bin = ensure_whisper_cpp()
    model_path = ensure_whisper_cpp_model(model)

    # whisper.cpp requires WAV 16kHz mono
    with tempfile.TemporaryDirectory(prefix="wcpp_") as tmp:
        wav_path = str(Path(tmp) / "input.wav")
        convert_to_wav(audio_path, wav_path)

        # Run whisper.cpp — outputs JSON to stdout with -oj flag
        json_output_path = str(Path(tmp) / "input.json")
        cmd = [
            whisper_bin,
            "-m", model_path,
            "-f", wav_path,
            "-oj",  # output JSON
            "-of", str(Path(tmp) / "input"),  # output file prefix (produces input.json)
        ]

        if language:
            cmd.extend(["-l", language])

        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=1800,
        )

        if result.returncode != 0:
            raise RuntimeError(f"whisper-cpp failed: {result.stderr}")

        # Parse JSON output
        if not Path(json_output_path).exists():
            # Try alternate output location
            alt_paths = list(Path(tmp).glob("*.json"))
            if alt_paths:
                json_output_path = str(alt_paths[0])
            else:
                raise RuntimeError(
                    f"whisper-cpp produced no JSON output. "
                    f"stdout: {result.stdout[:500]}, stderr: {result.stderr[:500]}"
                )

        with open(json_output_path, "r", encoding="utf-8") as f:
            output = json.load(f)

        return _parse_whisper_cpp_output(output, model)


def _parse_whisper_cpp_output(output: dict, model: str) -> Dict:
    """Parse whisper.cpp JSON into standard format."""
    segments = []

    # whisper.cpp JSON has "transcription" array
    raw_segments = output.get("transcription", output.get("segments", []))

    for seg in raw_segments:
        # whisper.cpp uses "timestamps" dict or direct start/end
        if "timestamps" in seg:
            ts = seg["timestamps"]
            start = _parse_timestamp(ts.get("from", "00:00:00.000"))
            end = _parse_timestamp(ts.get("to", "00:00:00.000"))
        else:
            start = seg.get("start", 0)
            end = seg.get("end", 0)

        # Convert from ms to seconds if needed
        if isinstance(start, int) and start > 1000:
            start = start / 1000.0
            end = end / 1000.0

        text = seg.get("text", "").strip()
        if text:
            segments.append({"start": start, "end": end, "text": text})

    duration = segments[-1]["end"] if segments else 0
    full_text = " ".join(s["text"] for s in segments)

    return {
        "model": f"whisper.cpp/{model}",
        "backend": "local",
        "language": output.get("result", {}).get("language", ""),
        "duration": duration,
        "segments": segments,
        "full_text": full_text,
    }


def _parse_timestamp(ts: str) -> float:
    """Parse whisper.cpp timestamp 'HH:MM:SS.mmm' to seconds."""
    try:
        parts = ts.split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except (ValueError, IndexError):
        return 0.0


# --- whisperx backend (with diarization) ---

def transcribe_whisperx(
    audio_path: str,
    language: Optional[str] = None,
    model: str = "large-v3",
    hf_token: Optional[str] = None,
) -> Dict:
    """
    Transcribe using whisperx with speaker diarization.

    Args:
        audio_path: Path to audio file.
        language: ISO-639-1 code (None = auto-detect).
        model: Whisper model name (e.g., "large-v3", "base").
        hf_token: HuggingFace token for PyAnnote diarization models.

    Returns:
        Standard result dict with segments (including speaker labels), full_text.
    """
    ensure_whisperx()

    if not hf_token:
        raise ValueError(
            "HuggingFace token required for diarization. "
            "Set HF_TOKEN in .env file. Get token at: https://huggingface.co/settings/tokens"
        )

    with tempfile.TemporaryDirectory(prefix="wxdiarize_") as output_dir:
        cmd = [
            "whisperx",
            audio_path,
            "--model", model,
            "--diarize",
            "--hf_token", hf_token,
            "--output_dir", output_dir,
            "--output_format", "json",
            "--device", "cpu",
            "--compute_type", "int8",
        ]

        if language:
            cmd.extend(["--language", language])

        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=3600,  # 1 hour for large files + model download
        )

        if result.returncode != 0:
            raise RuntimeError(f"whisperx failed: {result.stderr[:1000]}")

        json_files = list(Path(output_dir).glob("*.json"))
        if not json_files:
            raise RuntimeError(
                f"whisperx produced no output. stderr: {result.stderr[:500]}"
            )

        with open(json_files[0], "r", encoding="utf-8") as f:
            output = json.load(f)

        return _parse_whisperx_output(output, model)


def _parse_whisperx_output(output: dict, model: str) -> Dict:
    """Parse whisperx JSON with diarization into standard format."""
    segments = []

    for seg in output.get("segments", []):
        segment = {
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", "").strip(),
        }
        if "speaker" in seg:
            segment["speaker"] = seg["speaker"]
        if segment["text"]:
            segments.append(segment)

    duration = segments[-1]["end"] if segments else 0
    full_text = " ".join(s["text"] for s in segments)

    return {
        "model": f"whisperx/{model}",
        "backend": "local-diarize",
        "language": output.get("language", ""),
        "duration": duration,
        "segments": segments,
        "full_text": full_text,
    }
