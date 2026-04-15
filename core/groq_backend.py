"""
Groq Whisper backend — cloud transcription via Groq API.

Features: auto-chunking for files >25MB, retry with exponential backoff,
rate limiting (3s between requests for Groq's 20 RPM limit).
"""

import os
import time
import tempfile
from typing import Dict, Optional

from groq import Groq

from .chunker import needs_chunking, split_audio, merge_segments

RATE_LIMIT_DELAY = 3  # seconds between API calls
DEFAULT_MODEL = "whisper-large-v3-turbo"


def _transcribe_single(
    client: Groq,
    audio_path: str,
    model: str,
    language: Optional[str],
    max_retries: int = 3,
) -> Dict:
    """Transcribe a single audio file (must be under 25MB)."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            with open(audio_path, "rb") as f:
                kwargs = {
                    "file": f,
                    "model": model,
                    "response_format": "verbose_json",
                }
                if language:
                    kwargs["language"] = language

                transcription = client.audio.transcriptions.create(**kwargs)

            segments = []
            if hasattr(transcription, "segments") and transcription.segments:
                for seg in transcription.segments:
                    if isinstance(seg, dict):
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "")
                    else:
                        start = seg.start
                        end = seg.end
                        text = seg.text
                    segments.append({"start": start, "end": end, "text": text})

            return {
                "model": model,
                "language": getattr(transcription, "language", language),
                "duration": getattr(transcription, "duration", None),
                "segments": segments,
                "full_text": transcription.text,
            }

        except Exception as e:
            last_error = e
            if attempt < max_retries:
                backoff = RATE_LIMIT_DELAY * (2 ** (attempt - 1))
                time.sleep(backoff)

    raise RuntimeError(f"Groq transcription failed after {max_retries} attempts: {last_error}")


def transcribe_groq(
    audio_path: str,
    api_key: str,
    language: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Dict:
    """
    Transcribe audio file using Groq Whisper API.
    Automatically chunks files >25MB.

    Args:
        audio_path: Path to audio file.
        api_key: Groq API key.
        language: ISO-639-1 code (empty/None = auto-detect).
        model: Whisper model name.

    Returns:
        Dict with: model, language, duration, segments, full_text.
    """
    client = Groq(api_key=api_key, timeout=300.0)

    if not needs_chunking(audio_path):
        return _transcribe_single(client, audio_path, model, language)

    # Large file — split and transcribe chunks
    with tempfile.TemporaryDirectory() as chunk_dir:
        chunks = split_audio(audio_path, chunk_dir)
        chunk_results = []

        for i, chunk in enumerate(chunks):
            result = _transcribe_single(client, chunk["path"], model, language)
            chunk_results.append({"result": result, "offset": chunk["offset"]})

            # Rate limiting between chunks
            if i < len(chunks) - 1:
                time.sleep(RATE_LIMIT_DELAY)

        return merge_segments(chunk_results)
