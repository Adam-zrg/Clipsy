# GLM Video Highlighter

A local tool that takes a video file and a natural language prompt and outputs a trimmed highlight reel using Zhipu AI GLM-4V for scene analysis.

## Requirements

- Python 3.11+
- ffmpeg + ffprobe in PATH
- A Zhipu AI API key (https://open.bigmodel.cn)

## Setup

```bash
cd glm-highlighter
pip install fastapi uvicorn python-multipart zhipuai python-dotenv
cp .env.example .env   # then add your key
uvicorn main:app --reload
# open http://localhost:8000
```

## How it works

1. **Frame extraction** — ffmpeg pulls one frame every 2 seconds
2. **Scene analysis** — GLM-4V-Flash rates each frame (score 1–10) and describes it
3. **Cut planning** — GLM-4-Flash selects the best segments to match your prompt
4. **Cut + merge** — ffmpeg cuts clips and merges them with an audio fade-out

## Pipeline models

| Step | Model | Purpose |
|------|-------|---------|
| Scene analysis | `glm-4v-flash` | Vision — cheapest multimodal |
| Cut planning | `glm-4-flash` | Text — generates segment JSON |

## API routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| POST | `/analyze` | Upload video + prompt → analysis + cut plan |
| POST | `/render` | Cut plan → rendered MP4 |
| GET | `/status/{job_id}` | Poll render progress |
| GET | `/download/{job_id}` | Download final MP4 |
| GET | `/video/{job_id}` | Stream original video (for preview) |
