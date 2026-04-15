"""
Source resolver — detects input type and prepares audio for transcription.

Handles: YouTube URLs, remote file URLs, local audio files, local video files.
"""

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".webm", ".opus"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}

YOUTUBE_PATTERNS = [
    "youtube.com/watch",
    "youtu.be/",
    "youtube.com/embed",
    "youtube.com/v/",
]


@dataclass
class ResolvedSource:
    audio_path: str
    source_type: str  # "youtube_transcript", "youtube_audio", "remote_url", "local_audio", "local_video"
    transcript_data: Optional[dict] = None  # Only for youtube_transcript fast path
    cleanup_paths: Optional[list] = None  # Temp files to clean up
    original_source: str = ""


def is_youtube_url(source: str) -> bool:
    return any(pattern in source for pattern in YOUTUBE_PATTERNS)


def is_remote_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in ("http", "https") and not is_youtube_url(source)


def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)",
        r"youtube\.com\/embed\/([^&\n?#]+)",
        r"youtube\.com\/v\/([^&\n?#]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if len(url) == 11 and "/" not in url:
        return url
    return None


def _get_youtube_transcript(video_url: str, languages: Optional[list] = None) -> dict:
    """Download transcript via YouTube Transcript API (fast path, no audio needed)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from: {video_url}")

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    found = None
    if languages:
        try:
            found = transcript_list.find_transcript(languages)
        except NoTranscriptFound:
            found = next(iter(transcript_list))
    else:
        for lang in ["en", "pl"]:
            try:
                found = transcript_list.find_transcript([lang])
                break
            except NoTranscriptFound:
                continue
        if not found:
            found = next(iter(transcript_list))

    transcript_data = found.fetch()
    snippets = list(transcript_data)

    return {
        "source": video_url,
        "source_type": "youtube",
        "video_id": video_id,
        "language": found.language_code,
        "is_auto_generated": found.is_generated,
        "backend": "youtube_api",
        "segments": [
            {"start": s.start, "end": s.start + s.duration, "text": s.text}
            for s in snippets
        ],
        "full_text": " ".join(s.text for s in snippets),
    }


def _download_youtube_audio(video_url: str, output_path: str) -> str:
    """Download audio from YouTube via yt-dlp (64kbps mono MP3)."""
    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--postprocessor-args", "ffmpeg:-ac 1 -b:a 64k",
        "-o", output_path,
        video_url,
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)

    if os.path.exists(output_path):
        return output_path

    base = os.path.splitext(output_path)[0]
    for ext in [".mp3", ".m4a", ".webm", ".opus", ".wav"]:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate

    raise RuntimeError(f"Downloaded audio not found at: {output_path}")


def _download_remote_file(url: str, output_path: str) -> str:
    """Download a remote audio/video file."""
    cmd = ["curl", "-fsSL", "-o", output_path, url]
    subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    if not os.path.exists(output_path):
        raise RuntimeError(f"Download failed: {url}")
    return output_path


def _extract_audio_from_video(video_path: str, output_path: str) -> str:
    """Extract audio from video using ffmpeg (64kbps mono MP3)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-b:a", "64k",
        "-f", "mp3",
        output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=600)
    return output_path


def _get_file_type(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    elif ext in VIDEO_EXTENSIONS:
        return "video"
    return "unknown"


def resolve_source(
    source: str,
    temp_dir: str,
    force_audio_download: bool = False,
    languages: Optional[list] = None,
) -> ResolvedSource:
    """
    Resolve source to an audio file path ready for transcription.

    Args:
        source: YouTube URL, remote URL, or local file path.
        temp_dir: Temporary directory for downloads.
        force_audio_download: Skip YouTube transcript fast path (needed for diarization).
        languages: Preferred languages for YouTube transcript API.

    Returns:
        ResolvedSource with audio_path and metadata.
    """
    cleanup_paths = []

    # YouTube URL
    if is_youtube_url(source):
        # Fast path: try YouTube transcript API first (unless forced to download audio)
        if not force_audio_download:
            try:
                transcript = _get_youtube_transcript(source, languages)
                return ResolvedSource(
                    audio_path="",
                    source_type="youtube_transcript",
                    transcript_data=transcript,
                    original_source=source,
                )
            except Exception:
                pass  # Fall through to audio download

        # Slow path: download audio
        audio_path = os.path.join(temp_dir, "youtube_audio.mp3")
        actual_path = _download_youtube_audio(source, audio_path)
        cleanup_paths.append(actual_path)
        return ResolvedSource(
            audio_path=actual_path,
            source_type="youtube_audio",
            cleanup_paths=cleanup_paths,
            original_source=source,
        )

    # Remote URL
    if is_remote_url(source):
        parsed = urlparse(source)
        filename = Path(parsed.path).name or "downloaded_file"
        download_path = os.path.join(temp_dir, filename)
        _download_remote_file(source, download_path)
        cleanup_paths.append(download_path)

        file_type = _get_file_type(download_path)
        if file_type == "video":
            audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
            _extract_audio_from_video(download_path, audio_path)
            cleanup_paths.append(audio_path)
            return ResolvedSource(
                audio_path=audio_path,
                source_type="remote_url",
                cleanup_paths=cleanup_paths,
                original_source=source,
            )
        return ResolvedSource(
            audio_path=download_path,
            source_type="remote_url",
            cleanup_paths=cleanup_paths,
            original_source=source,
        )

    # Local file
    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    file_type = _get_file_type(source)
    if file_type == "video":
        audio_path = os.path.join(temp_dir, "extracted_audio.mp3")
        _extract_audio_from_video(source, audio_path)
        cleanup_paths.append(audio_path)
        return ResolvedSource(
            audio_path=audio_path,
            source_type="local_video",
            cleanup_paths=cleanup_paths,
            original_source=source,
        )
    elif file_type == "audio":
        return ResolvedSource(
            audio_path=source,
            source_type="local_audio",
            original_source=source,
        )
    else:
        raise ValueError(
            f"Unsupported file type: {Path(source).suffix}. "
            f"Supported audio: {', '.join(sorted(AUDIO_EXTENSIONS))} | "
            f"Supported video: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )
