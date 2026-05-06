"""
Test API max batch size against the configured endpoint.
Usage:
    python test_api.py --key YOUR_API_KEY [--base-url URL] [--model MODEL] [--max-batch 30]
Results are saved to test_results.json.
"""
import asyncio
import base64
import json
import time
import argparse
from datetime import datetime
from pathlib import Path


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def call_vision(client, image_path: str, model: str) -> tuple[float, int, str]:
    b64 = encode_image(image_path)
    start = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }],
        )
        latency = time.perf_counter() - start
        tokens = response.usage.total_tokens if response.usage else 0
        return latency, tokens, "ok"
    except Exception as e:
        latency = time.perf_counter() - start
        return latency, 0, f"error: {e}"


def pad_frames(frames: list[str], n: int) -> list[str]:
    return [frames[i % len(frames)] for i in range(n)]


async def test_batch(client, frames: list[str], model: str, batch_size: int) -> dict:
    batch = pad_frames(frames, batch_size)
    start = time.perf_counter()
    results = await asyncio.gather(*[call_vision(client, f, model) for f in batch])
    wall = time.perf_counter() - start

    latencies = [r[0] for r in results]
    errors = [r[2] for r in results if r[2] != "ok"]
    ok_count = batch_size - len(errors)

    print(f"  batch={batch_size:>3}  wall={wall:.2f}s  "
          f"avg_lat={sum(latencies)/len(latencies):.2f}s  "
          f"ok={ok_count}  errors={len(errors)}", end="")
    if errors:
        print(f"  ← {errors[0][:80]}", end="")
    print()

    return {
        "batch_size": batch_size,
        "wall_sec": round(wall, 2),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2),
        "min_latency_sec": round(min(latencies), 2),
        "max_latency_sec": round(max(latencies), 2),
        "ok": ok_count,
        "errors": len(errors),
        "error_sample": errors[0] if errors else None,
    }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", required=True, help="API key")
    parser.add_argument("--base-url", default="https://taotoken.net/api/v1")
    parser.add_argument("--model", default="kimi-k2.6")
    parser.add_argument("--max-batch", type=int, default=20)
    parser.add_argument("--frames-dir", default=None)
    parser.add_argument("--output", default="test_results.json")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.key, base_url=args.base_url)

    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        temp = Path("temp")
        candidates = [d / "frames" for d in temp.iterdir() if (d / "frames").exists()]
        if not candidates:
            print("No frames found. Run an analysis first or pass --frames-dir.")
            return
        frames_dir = sorted(candidates)[-1]

    frames = sorted(str(f) for f in frames_dir.glob("frame_*.jpg"))
    print(f"Frames   : {len(frames)} from {frames_dir}")
    print(f"Endpoint : {args.base_url}")
    print(f"Model    : {args.model}")

    print(f"\n── Max batch test (ramp 1 → {args.max_batch}) ──")
    batch_results = []
    last_ok = 1
    for size in [6, 9, 12, 15, 18, 20]:
        if size > args.max_batch:
            break
        result = await test_batch(client, frames, args.model, size)
        batch_results.append(result)
        if result["errors"] == 0:
            last_ok = size
        else:
            print(f"  → First failure at batch={size}. Safe limit: {last_ok}")
            break
        await asyncio.sleep(1)
    else:
        print(f"  → No errors up to batch={last_ok}.")

    output = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "endpoint": args.base_url,
        "model": args.model,
        "safe_batch_limit": last_ok,
        "batches": batch_results,
    }
    out_path = Path(args.output)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path.resolve()}")


asyncio.run(main())
