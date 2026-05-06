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
uvicorn main:app --reload
# open http://localhost:8000
```

## How it works

1. **Frame extraction** — ffmpeg pulls one frame every 2 seconds
2. **Scene analysis** — GLM-4V-Flash rates each frame (score 1–10) and describes it
3. **Cut planning** — GLM-4-Flash selects the best segments to match your prompt
4. **Cut + merge** — ffmpeg cuts clips and merges them with an audio fade-out
