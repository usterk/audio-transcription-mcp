# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mcp[cli]>=1.0.0",
#     "groq>=0.4.0",
#     "youtube-transcript-api>=1.0.0",
#     "yt-dlp>=2024.0.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
"""
Audio Transcription MCP Server

Single tool: `transcribe` — handles YouTube URLs, remote URLs,
local audio/video files. Two backends: Groq (cloud, default)
and whisperx-mlx (local, with optional speaker diarization).
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load .env from server directory
load_dotenv(Path(__file__).parent / ".env")

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.source_resolver import resolve_source
from core.groq_backend import transcribe_groq
from core.local_backend import transcribe_local
from core.formatter import save_output

mcp = FastMCP(
    "audio-transcription",
    instructions="Transcribe audio/video from local files, YouTube, or URLs. "
    "Supports Groq Whisper (cloud) and whisperx-mlx (local with diarization).",
)


@mcp.tool()
def transcribe(
    source: str,
    backend: str = "groq",
    language: str = "",
    diarize: bool = False,
    model: str = "",
    output_dir: str = "",
) -> dict:
    """Transcribe audio from a local file, YouTube URL, or remote file URL.

    Args:
        source: Absolute path to audio/video file, YouTube URL, or URL to audio file.
        backend: "groq" (cloud, default) or "local" (whisperx-mlx on Apple Silicon).
        language: ISO-639-1 language code (e.g., "pl", "en"). Empty = auto-detect.
        diarize: Enable speaker diarization. Only works with backend="local".
        model: Whisper model. Groq: "whisper-large-v3-turbo" (default), "whisper-large-v3",
               "distil-whisper-english". Local: "large-v3" (default), "base", "small".
        output_dir: Directory for output files. Default: same directory as source file.

    Returns:
        Dict with json_path, txt_path, summary (duration, language, segments count).
    """
    # Validate backend
    if backend not in ("groq", "local"):
        return {"error": f"Unknown backend: {backend}. Use 'groq' or 'local'."}

    # Diarize only with local backend
    if diarize and backend != "local":
        return {"error": "Diarization requires backend='local'. Groq does not support speaker diarization."}

    # Force audio download for YouTube if using local backend or diarization
    force_audio = backend == "local"

    # Parse language
    lang = language.strip() if language else None
    languages = [lang] if lang else None

    with tempfile.TemporaryDirectory(prefix="mcp_transcribe_") as temp_dir:
        # Step 1: Resolve source
        try:
            resolved = resolve_source(
                source,
                temp_dir=temp_dir,
                force_audio_download=force_audio,
                languages=languages,
            )
        except Exception as e:
            return {"error": f"Source resolution failed: {e}"}

        # Step 2: Transcribe
        try:
            if resolved.source_type == "youtube_transcript":
                # Fast path — transcript already available from YouTube API
                result = resolved.transcript_data
            elif backend == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    return {"error": "GROQ_API_KEY not set. Add it to .env file."}
                groq_model = model or "whisper-large-v3-turbo"
                result = transcribe_groq(
                    resolved.audio_path,
                    api_key=api_key,
                    language=lang,
                    model=groq_model,
                )
                result["backend"] = "groq"
            elif backend == "local":
                local_model = model or "large-v3"
                hf_token = os.getenv("HF_TOKEN")
                result = transcribe_local(
                    resolved.audio_path,
                    language=lang,
                    model=local_model,
                    diarize=diarize,
                    hf_token=hf_token,
                )
                result["backend"] = "local"
            else:
                return {"error": f"Unknown backend: {backend}"}
        except Exception as e:
            return {"error": f"Transcription failed: {e}"}

        # Step 3: Save output files
        try:
            metadata = {
                "source": resolved.original_source or source,
                "source_type": resolved.source_type,
                "backend": result.get("backend", backend),
                "model": result.get("model", model),
                "diarize": diarize,
            }

            paths = save_output(
                result=result,
                metadata=metadata,
                source=source,
                output_dir=output_dir or None,
            )
        except Exception as e:
            return {"error": f"Failed to save output: {e}"}

        # Step 4: Return summary
        segments = result.get("segments", [])
        duration = result.get("duration", 0)
        detected_lang = result.get("language", "")

        summary_parts = []
        if duration:
            minutes = duration / 60
            summary_parts.append(f"{minutes:.1f} min")
        if detected_lang:
            summary_parts.append(f"lang={detected_lang}")
        summary_parts.append(f"{len(segments)} segments")
        summary_parts.append(f"backend={result.get('backend', backend)}")

        return {
            "json_path": paths["json_path"],
            "txt_path": paths["txt_path"],
            "summary": ", ".join(summary_parts),
            "duration": duration,
            "language": detected_lang,
            "segments_count": len(segments),
        }


if __name__ == "__main__":
    mcp.run()
