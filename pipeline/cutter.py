import subprocess
from pathlib import Path

TEMP_DIR = Path("temp")
OUTPUTS_DIR = Path("outputs")


def _run_ffmpeg(args: list[str], label: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg {label} failed:\n{result.stderr}")


def cut_and_merge(
    video_path: str,
    segments: list[dict],
    output_name: str,
    job_id: str,
) -> str:
    job_tmp = TEMP_DIR / job_id
    job_tmp.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    clip_paths: list[Path] = []

    for i, seg in enumerate(segments):
        start = float(seg["start_sec"])
        end = float(seg["end_sec"])
        duration = end - start
        clip_path = job_tmp / f"clip_{i:03d}.mp4"

        _run_ffmpeg(
            [
                "ffmpeg",
                "-ss", str(start),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                str(clip_path),
                "-y",
            ],
            f"clip {i}",
        )
        clip_paths.append(clip_path)

    # Build concat list
    concat_file = job_tmp / "concat.txt"
    lines = [f"file '{p.resolve()}'" for p in clip_paths]
    concat_file.write_text("\n".join(lines), encoding="utf-8")

    output_path = OUTPUTS_DIR / f"{output_name}.mp4"

    # Total duration for audio fade-out calculation
    total_duration = sum(
        float(s["end_sec"]) - float(s["start_sec"]) for s in segments
    )
    fade_start = max(0.0, total_duration - 1.5)

    _run_ffmpeg(
        [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-af", f"afade=t=out:st={fade_start:.2f}:d=1.5",
            str(output_path),
            "-y",
        ],
        "merge",
    )

    return str(output_path)
