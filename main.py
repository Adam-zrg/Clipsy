import os
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Clipsy — Video Editor")

TEMP_DIR = Path("temp")
OUTPUTS_DIR = Path("outputs")
TEMP_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

jobs: dict = {}

DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_BASE_URL = "https://taotoken.net/api/v1"


def _resolve_credentials(api_key: str, model: str, base_url: str) -> tuple[str, str, str]:
    """Use values from request; fall back to env vars if empty."""
    resolved_key = api_key.strip() or os.getenv("MOONSHOT_API_KEY", "")
    resolved_model = model.strip() or DEFAULT_MODEL
    resolved_url = base_url.strip() or DEFAULT_BASE_URL
    if not resolved_key:
        raise HTTPException(
            400,
            "API key required — set it in the app settings or add MOONSHOT_API_KEY to .env",
        )
    return resolved_key, resolved_model, resolved_url


class RenderRequest(BaseModel):
    job_id: str
    cut_plan: dict
    api_key: Optional[str] = ""
    model: Optional[str] = ""
    base_url: Optional[str] = ""


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    prompt: str = Form(...),
    api_key: str = Form(""),
    model: str = Form(""),
    base_url: str = Form(""),
    frames_per_min: int = Form(12),
    batch_size: int = Form(20),
    resolution: str = Form("640x360"),
):
    resolved_key, resolved_model, resolved_url = _resolve_credentials(api_key, model, base_url)

    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / (video.filename or "upload.mp4")
    content = await video.read()
    with open(video_path, "wb") as f:
        f.write(content)

    jobs[job_id] = {
        "status": "extracting",
        "video_path": str(video_path),
        "prompt": prompt,
        "model": resolved_model,
        "base_url": resolved_url,
        # Store key only for render reuse; never expose it in /status
        "_api_key": resolved_key,
        "progress": 0,
    }

    try:
        import time as _time
        from pipeline.extractor import extract_frames, get_video_duration
        from pipeline.analyzer import analyze_frames
        from pipeline.planner import plan_cuts

        _t0 = _time.perf_counter()

        video_duration = get_video_duration(str(video_path))
        jobs[job_id]["video_duration_sec"] = video_duration
        print(f"[timing] video_info     : {_time.perf_counter()-_t0:.2f}s")

        _t = _time.perf_counter()
        frames = extract_frames(str(video_path), job_id, frames_per_min, resolution)
        print(f"[timing] extract_frames : {_time.perf_counter()-_t:.2f}s  ({len(frames)} frames)")
        jobs[job_id]["status"] = "analyzing"
        jobs[job_id]["frame_count"] = len(frames)

        _t = _time.perf_counter()
        analysis, total_tokens = await analyze_frames(
            frames, job_id, resolved_key, resolved_model, resolved_url, batch_size
        )
        print(f"[timing] analyze_frames : {_time.perf_counter()-_t:.2f}s  ({total_tokens} tokens)")
        jobs[job_id]["status"] = "planning"

        _t = _time.perf_counter()
        cut_plan = await plan_cuts(
            analysis, prompt, job_id, resolved_key, resolved_model, resolved_url
        )
        print(f"[timing] plan_cuts      : {_time.perf_counter()-_t:.2f}s")
        print(f"[timing] TOTAL          : {_time.perf_counter()-_t0:.2f}s")

        estimated_cost = round(total_tokens * 0.001 / 1000, 5)

        jobs[job_id].update(
            {
                "status": "analyzed",
                "analysis": analysis,
                "cut_plan": cut_plan,
                "total_tokens": total_tokens,
                "estimated_cost_usd": estimated_cost,
            }
        )

        return {
            "job_id": job_id,
            "analysis": analysis,
            "cut_plan": cut_plan,
            "video_duration_sec": video_duration,
            "estimated_cost_usd": estimated_cost,
            "frame_count": len(frames),
            "model": resolved_model,
        }

    except HTTPException:
        raise
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)
        raise HTTPException(500, str(exc)) from exc


@app.post("/render")
async def render(req: RenderRequest, background_tasks: BackgroundTasks):
    job_id = req.job_id
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    from pipeline.planner import validate_cut_plan

    video_duration = jobs[job_id].get("video_duration_sec")
    errors = validate_cut_plan(req.cut_plan, video_duration)
    if errors:
        raise HTTPException(422, {"validation_errors": errors})

    # Prefer credentials from render request; fall back to what was used for analyze
    resolved_key, resolved_model, resolved_url = _resolve_credentials(
        req.api_key or jobs[job_id].get("_api_key", ""),
        req.model or jobs[job_id].get("model", ""),
        req.base_url or jobs[job_id].get("base_url", ""),
    )

    jobs[job_id]["status"] = "rendering"
    jobs[job_id]["render_progress"] = 0
    jobs[job_id].pop("error", None)

    background_tasks.add_task(_do_render, job_id, req.cut_plan)
    return {"job_id": job_id, "status": "rendering"}


def _do_render(job_id: str, cut_plan: dict) -> None:
    try:
        from pipeline.cutter import cut_and_merge

        video_path = jobs[job_id]["video_path"]
        output_name = f"{job_id[:8]}_montage"
        output_path = cut_and_merge(video_path, cut_plan["segments"], output_name, job_id)

        jobs[job_id]["output_path"] = output_path
        jobs[job_id]["status"] = "done"
        jobs[job_id]["render_progress"] = 100
        jobs[job_id]["download_url"] = f"/download/{job_id}"
    except Exception as exc:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(exc)


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "status": job.get("status"),
        "render_progress": job.get("render_progress", 0),
        "download_url": job.get("download_url"),
        "error": job.get("error"),
        "estimated_cost_usd": job.get("estimated_cost_usd"),
        "video_duration_sec": job.get("video_duration_sec"),
        "frame_count": job.get("frame_count"),
        "model": job.get("model"),
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in jobs or "output_path" not in jobs[job_id]:
        raise HTTPException(404, "Output not ready")
    path = jobs[job_id]["output_path"]
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"highlight_{job_id[:8]}.mp4",
    )


@app.get("/video/{job_id}")
async def serve_original(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return FileResponse(jobs[job_id]["video_path"])
