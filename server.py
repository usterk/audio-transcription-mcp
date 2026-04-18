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
local audio/video files. Three backends:
- groq: Cloud Whisper API (fast, default)
- local: whisper.cpp with Metal GPU (fast local, no diarization)
- local-diarize: whisperx with PyAnnote (slower, speaker diarization)

Dependencies install automatically on first use.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP, Context

# Load .env from server directory
load_dotenv(Path(__file__).parent / ".env")

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from core.source_resolver import resolve_source
from core.groq_backend import transcribe_groq
from core.local_backend import transcribe_whisper_cpp, transcribe_whisperx
from core.formatter import save_output

VALID_BACKENDS = ("groq", "local", "local-diarize")

mcp = FastMCP(
    "audio-transcription",
    instructions=(
        "Transcribe audio/video from local files, YouTube, or URLs. "
        "Three backends: groq (cloud, fast, default), local (whisper.cpp, GPU Metal), "
        "local-diarize (whisperx, speaker identification). "
        "Dependencies auto-install on first use. First run may take several minutes. "
        "IMPORTANT: When the user asks to transcribe without providing a specific file path or URL, "
        "ask them to provide the path or URL. If they attached a file, use the file's local path. "
        "If they mention a file by name, search for it in their working directory or home folder. "
        "The source parameter accepts: absolute file paths (e.g. /Users/name/audio.mp3), "
        "YouTube URLs, or remote file URLs."
    ),
)


@mcp.tool()
async def transcribe(
    source: str,
    backend: str = "groq",
    language: str = "",
    model: str = "",
    output_dir: str = "",
    ctx: Context = None,
) -> dict:
    """Transcribe audio from a local file, YouTube URL, or remote file URL.

    Args:
        source: Absolute path to audio/video file, YouTube URL, or URL to audio file.
        backend: "groq" (cloud, fast, default), "local" (whisper.cpp, GPU Metal, fast),
                 or "local-diarize" (whisperx, speaker identification, slower).
        language: ISO-639-1 language code (e.g., "pl", "en"). Empty = auto-detect.
        model: Whisper model override. Groq default: "whisper-large-v3-turbo".
               Local default: "base" (fast) — options: base, small, medium, large-v3, large-v3-turbo.
               Local-diarize default: "large-v3".
        output_dir: REQUIRED for URLs. Directory for output files. For local files defaults
               to same directory as source. For YouTube/remote URLs you MUST provide this -
               use the current working directory of the project.

    Returns:
        Dict with json_path, txt_path, summary (duration, language, segments count).
        First run with local/local-diarize backends may take extra time for dependency installation.
    """
    async def log(msg: str):
        if ctx:
            await ctx.info(msg)

    async def progress(step: float, total: float, msg: str = ""):
        if ctx:
            await ctx.report_progress(step, total, msg)

    if backend not in VALID_BACKENDS:
        return {"error": f"Unknown backend: {backend}. Use: {', '.join(VALID_BACKENDS)}"}

    # For local backends, always download audio (skip YouTube transcript fast path)
    force_audio = backend in ("local", "local-diarize")
    lang = language.strip() if language else None
    languages = [lang] if lang else None

    with tempfile.TemporaryDirectory(prefix="mcp_transcribe_") as temp_dir:
        # Step 1: Resolve source
        await progress(1, 5, "Resolving source...")
        await log(f"Resolving source: {source}")
        try:
            resolved = await asyncio.to_thread(
                resolve_source,
                source,
                temp_dir=temp_dir,
                force_audio_download=force_audio,
                languages=languages,
            )
        except Exception as e:
            return {"error": f"Source resolution failed: {e}"}

        await log(f"Source type: {resolved.source_type}")

        # Step 2: Transcribe
        await progress(2, 5, "Transcribing audio...")
        try:
            if resolved.source_type == "youtube_transcript":
                await log("Using YouTube transcript API (fast path)")
                result = resolved.transcript_data

            elif backend == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    return {"error": "GROQ_API_KEY not set. Add it to .env file."}
                groq_model = model or "whisper-large-v3-turbo"
                await log(f"Transcribing with Groq ({groq_model})...")
                result = await asyncio.to_thread(
                    transcribe_groq,
                    resolved.audio_path,
                    api_key=api_key,
                    language=lang,
                    model=groq_model,
                )

            elif backend == "local":
                local_model = model or "base"
                await log(f"Transcribing with whisper.cpp ({local_model})...")
                await log("Installing dependencies if needed (first run may take a few minutes)...")
                result = await asyncio.to_thread(
                    transcribe_whisper_cpp,
                    resolved.audio_path,
                    language=lang,
                    model=local_model,
                )

            elif backend == "local-diarize":
                wx_model = model or "large-v3"
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    return {
                        "error": "HF_TOKEN not set. Required for speaker diarization. "
                        "Get token at https://huggingface.co/settings/tokens and add to .env file."
                    }
                await log(f"Transcribing with whisperx ({wx_model}) + diarization...")
                await log("Installing dependencies if needed (first run may take several minutes)...")
                result = await asyncio.to_thread(
                    transcribe_whisperx,
                    resolved.audio_path,
                    language=lang,
                    model=wx_model,
                    hf_token=hf_token,
                )

            else:
                return {"error": f"Unknown backend: {backend}"}

        except Exception as e:
            return {"error": f"Transcription failed: {e}"}

        await log(f"Transcription complete: {len(result.get('segments', []))} segments")

        # Step 3: Save output files
        await progress(3, 5, "Saving output files...")
        try:
            metadata = {
                "source": resolved.original_source or source,
                "source_type": resolved.source_type,
                "backend": result.get("backend", backend),
                "model": result.get("model", model),
                "diarize": backend == "local-diarize",
            }

            paths = await asyncio.to_thread(
                save_output,
                result=result,
                metadata=metadata,
                source=source,
                output_dir=output_dir or None,
            )
        except Exception as e:
            return {"error": f"Failed to save output: {e}"}

        # Step 4: Return summary
        await progress(5, 5, "Done!")
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
