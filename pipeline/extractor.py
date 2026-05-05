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
            f"Impossible de lire la durée vidéo (ffmpeg output):\n{output[:500]}"
        )
    h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return h * 3600 + m * 60 + s


def extract_frames(video_path: str, job_id: str) -> list[dict]:
    frames_dir = TEMP_DIR / job_id / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(frames_dir / "frame_%04d.jpg")

    result = subprocess.run(
        [
            "ffmpeg", "-i", video_path,
            "-vf", "fps=0.5",
            pattern, "-y",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed:\n{result.stderr[-1000:]}")

    frames = []
    for frame_file in sorted(frames_dir.glob("frame_*.jpg")):
        stem = frame_file.stem           # "frame_0001"
        file_num = int(stem.split("_")[1])  # 1-based
        frame_index = file_num - 1
        timestamp_sec = float(frame_index * 2)
        frames.append({
            "frame_path": str(frame_file),
            "timestamp_sec": timestamp_sec,
            "frame_index": frame_index,
        })

    return frames
