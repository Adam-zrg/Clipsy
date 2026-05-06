# Clipsy Video Highlighter

## Requirements

- Python 3.11+
- ffmpeg + ffprobe in PATH
- AI API key with image input

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn python-multipart openai python-dotenv
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# open http://localhost:8000
```

## How it works

1. **Frame extraction** — ffmpeg pulls one frame every 2 seconds
2. **Scene analysis** — GLM-4V-Flash rates each frame (score 1–10) and describes it
3. **Cut planning** — GLM-4-Flash selects the best segments to match your prompt
4. **Cut + merge** — ffmpeg cuts clips and merges them with an audio fade-out
