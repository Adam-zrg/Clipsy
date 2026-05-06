import re
import subprocess
from pathlib import Path

TEMP_DIR = Path("temp")


def get_video_duration(video_path: str) -> float:
    # ffmpeg -i écrit les infos sur stderr ; on parse la ligne Duration
    result = subprocess.run(
        ["ffmpeg", "-i", video_path],
        capture_output=True,
        text=True,
    )
    output = result.stderr + result.stdout
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
    if not match:
        raise RuntimeError(
            f"Could not read video duration (ffmpeg output):\n{output[:500]}"
        )
    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return h * 3600 + m * 60 + s


def extract_frames(video_path: str, job_id: str, frames_per_min: int = 12, resolution: str = "640x360") -> list[dict]:
    frames_dir = TEMP_DIR / job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    fps = frames_per_min / 60
    interval_sec = 60 / frames_per_min
    pattern = str(frames_dir / "frame_%04d.jpg")

    if resolution == "original":
        vf = f"fps={fps}"
    else:
        w, h = resolution.split("x")
        vf = f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease"

    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", vf,
            pattern, "-y",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr[-1000:]}")

    frames = []
    for frame_file in sorted(frames_dir.glob("frame_*.jpg")):
        frame_index = int(frame_file.stem.split("_")[1]) - 1
        frames.append({
            "frame_path": str(frame_file),
            "timestamp_sec": float(frame_index * interval_sec),
            "frame_index": frame_index,
        })

    return frames
