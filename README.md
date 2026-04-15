# Audio Transcription MCP Server

MCP server for audio/video transcription with three backends:

| Backend | Engine | Speed | Diarization | Requires |
|---------|--------|-------|-------------|----------|
| `groq` (default) | Groq Whisper API | Fast (cloud) | No | `GROQ_API_KEY` |
| `local` | whisper.cpp (Metal GPU) | Fast (local) | No | Auto-installs via brew |
| `local-diarize` | whisperx + PyAnnote | Slow (CPU) | Yes | `HF_TOKEN` + auto-installs |

## Supported sources

- Local audio files: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.webm`, `.opus`
- Local video files: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`
- YouTube URLs (transcript API fast path + Whisper fallback)
- Remote file URLs

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/usterk/audio-transcription-mcp.git
cd audio-transcription-mcp
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (and optionally HF_TOKEN for diarization)
```

### 2. Register in Claude Code

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "audio-transcription": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/audio-transcription-mcp", "server.py"]
    }
  }
}
```

### 3. Register in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "audio-transcription": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/audio-transcription-mcp", "server.py"]
    }
  }
}
```

## Usage

The server exposes a single `transcribe` tool:

```
transcribe(
  source: str,          # File path, YouTube URL, or remote URL
  backend: str = "groq", # "groq", "local", or "local-diarize"
  language: str = "",    # ISO-639-1 code (e.g., "pl", "en"), empty = auto
  model: str = "",       # Model override (see defaults below)
  output_dir: str = "",  # Output directory (required for URLs)
)
```

### Model defaults

| Backend | Default model | Options |
|---------|--------------|---------|
| `groq` | `whisper-large-v3-turbo` | `whisper-large-v3`, `distil-whisper-english` |
| `local` | `base` | `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `local-diarize` | `large-v3` | Any Whisper model name |

### Output

Returns paths to two files saved next to the source (or in `output_dir`):

- **`.json`** — full transcription with segments, timestamps, metadata
- **`.txt`** — formatted text with line breaks between segments, blank lines for pauses >2s

With `local-diarize`, segments include speaker labels:
```
Speaker 1: Hello, welcome to the podcast.
Speaker 2: Thanks for having me.
```

## Lazy installation

Local backends install dependencies automatically on first use:

- `local`: `brew install whisper-cpp` + model download from HuggingFace (~142MB for base)
- `local-diarize`: `pip install whisperx` (~4GB with PyTorch) + PyAnnote models

First run takes extra time. Subsequent runs use cached dependencies.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) for Metal GPU acceleration
- [uv](https://docs.astral.sh/uv/) for dependency management
- [ffmpeg](https://formulae.brew.sh/formula/ffmpeg) (auto-installed if missing)
- Python 3.11-3.13
