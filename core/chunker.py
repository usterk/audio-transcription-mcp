"""
Audio chunker — splits large files for Groq Whisper API.

Groq free tier: 25MB per file. This module splits larger files into
~20MB chunks (64kbps mono MP3), then merges transcription results
with corrected timestamps.
"""

import json
import os
import subprocess
from typing import Dict, List


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB (Groq free tier)
TARGET_CHUNK_SIZE = 20 * 1024 * 1024  # 20MB target (safety margin)
TARGET_BITRATE = 64000  # 64kbps mono


def get_audio_info(file_path: str) -> Dict:
    """Get audio duration, bitrate, and size using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        file_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    fmt = info.get("format", {})
    return {
        "duration": float(fmt.get("duration", 0)),
        "bitrate": int(fmt.get("bit_rate", TARGET_BITRATE)),
        "size": int(fmt.get("size", 0)),
    }


def needs_chunking(file_path: str) -> bool:
    """Check if file exceeds Groq's size limit."""
    return os.path.getsize(file_path) > MAX_FILE_SIZE


def split_audio(
    file_path: str,
    chunk_dir: str,
    max_chunk_bytes: int = TARGET_CHUNK_SIZE,
) -> List[Dict]:
    """
    Split audio into chunks that fit within Groq's file size limit.
    Re-encodes to 64kbps mono MP3 for predictable chunk sizes.

    Returns:
        List of dicts: [{"path": str, "offset": float}, ...]
    """
    info = get_audio_info(file_path)
    duration = info["duration"]

    chunk_duration = (max_chunk_bytes * 8) / TARGET_BITRATE
    chunk_duration = min(chunk_duration, 1200)  # max 20 min per chunk
    chunk_duration = max(60, chunk_duration)  # min 1 min per chunk

    chunks = []
    offset = 0.0
    chunk_index = 0

    while offset < duration:
        chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_index:03d}.mp3")
        cmd = [
            "ffmpeg", "-y",
            "-i", file_path,
            "-ss", str(offset),
            "-t", str(chunk_duration),
            "-ac", "1",
            "-b:a", "64k",
            chunk_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 0:
            chunk_size = os.path.getsize(chunk_path)

            # Re-encode at lower bitrate if still over limit
            if chunk_size > MAX_FILE_SIZE:
                tmp_path = chunk_path + ".tmp.mp3"
                cmd_resplit = [
                    "ffmpeg", "-y",
                    "-i", chunk_path,
                    "-ac", "1", "-b:a", "48k",
                    tmp_path,
                ]
                subprocess.run(cmd_resplit, capture_output=True, check=True)
                os.replace(tmp_path, chunk_path)

            chunks.append({"path": chunk_path, "offset": offset})

        offset += chunk_duration
        chunk_index += 1

    return chunks


def merge_segments(chunk_results: List[Dict]) -> Dict:
    """
    Merge transcription results from multiple chunks.
    Corrects timestamps by adding each chunk's offset.

    Args:
        chunk_results: List of (result_dict, offset_seconds) tuples as dicts
            with keys: "result" and "offset"

    Returns:
        Merged transcription result dict.
    """
    all_segments = []
    all_texts = []
    total_duration = 0.0
    detected_language = None
    model = None

    for item in chunk_results:
        result = item["result"]
        offset = item["offset"]

        if detected_language is None:
            detected_language = result.get("language")
        if model is None:
            model = result.get("model")

        for segment in result.get("segments", []):
            adjusted = segment.copy()
            adjusted["start"] = segment["start"] + offset
            if "end" in segment:
                adjusted["end"] = segment["end"] + offset
            all_segments.append(adjusted)

        all_texts.append(result.get("full_text", ""))

        if result.get("duration"):
            total_duration += result["duration"]

    return {
        "language": detected_language,
        "duration": total_duration,
        "model": model,
        "segments": all_segments,
        "full_text": " ".join(all_texts),
    }
