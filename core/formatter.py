"""
Output formatter — generates JSON and TXT transcription files.

TXT format:
- Each segment on its own line
- Gaps >2s between segments = extra blank line
- With diarization: "Speaker N: text"
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional


SILENCE_GAP_THRESHOLD = 2.0  # seconds — gap triggers blank line in TXT


def format_json(result: dict, metadata: dict) -> dict:
    """Build full JSON output combining transcription result with metadata."""
    output = {**metadata}
    output["language"] = result.get("language", "")
    output["duration"] = result.get("duration", 0)
    output["diarize"] = metadata.get("diarize", False)
    output["segments"] = result.get("segments", [])
    output["full_text"] = result.get("full_text", "")
    return output


def format_txt(result: dict, diarize: bool = False) -> str:
    """
    Build formatted TXT output.

    - Each segment = one line
    - Gap > 2s = extra blank line (visual paragraph break)
    - diarize=True: prefix each line with "Speaker N: "
    """
    segments = result.get("segments", [])
    if not segments:
        return result.get("full_text", "")

    lines = []
    prev_end = None

    for seg in segments:
        start = seg.get("start", 0)
        text = seg.get("text", "").strip()
        if not text:
            continue

        # Insert blank line for silence gaps
        if prev_end is not None and (start - prev_end) > SILENCE_GAP_THRESHOLD:
            lines.append("")

        if diarize and "speaker" in seg:
            lines.append(f"{seg['speaker']}: {text}")
        else:
            lines.append(text)

        prev_end = seg.get("end", start)

    return "\n".join(lines)


def _derive_output_name(source: str) -> str:
    """Derive output filename from source (without extension)."""
    if "youtube.com" in source or "youtu.be" in source:
        # Use video ID if available
        import re
        match = re.search(r"(?:v=|youtu\.be/)([^&\n?#]+)", source)
        if match:
            return match.group(1)
    path = Path(source)
    if path.exists():
        return path.stem
    # Fallback for URLs
    return Path(source.split("?")[0].split("#")[0]).stem or "transcript"


def save_output(
    result: dict,
    metadata: dict,
    source: str,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Save transcription as JSON + TXT files.

    Args:
        result: Transcription result with segments and full_text.
        metadata: Source metadata (source, backend, model, etc.).
        source: Original source string (for deriving filename).
        output_dir: Directory for output files. Defaults to source file's directory.

    Returns:
        Dict with "json_path" and "txt_path".
    """
    # Determine output directory
    if output_dir:
        out_dir = Path(output_dir)
    elif os.path.exists(source):
        out_dir = Path(source).parent
    else:
        raise ValueError(
            f"output_dir is required for non-local sources (URLs). "
            f"Source: {source}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = _derive_output_name(source)

    # Save JSON
    json_data = format_json(result, metadata)
    json_path = out_dir / f"{base_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Save TXT
    diarize = metadata.get("diarize", False)
    txt_content = format_txt(result, diarize=diarize)
    txt_path = out_dir / f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt_content)
        f.write("\n")

    return {"json_path": str(json_path), "txt_path": str(txt_path)}
