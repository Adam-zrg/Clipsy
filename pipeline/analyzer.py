import asyncio
import base64
import json
import re
import time
from pathlib import Path

TEMP_DIR = Path("temp")

SYSTEM_PROMPT = (
    "You are a video analyst. Describe this video frame concisely in 1-2 sentences. "
    "Rate its editing relevance from 1-10 based on visual clarity, content, action, or narrative value. "
    'Respond ONLY as JSON: {"description": "...", "score": N, "tags": [...]}'
)

STRICT_PROMPT = (
    "Respond ONLY with a single JSON object. No markdown, no explanation. "
    'Format: {"description": "one sentence", "score": 5, "tags": ["tag1"]}'
)


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_from_text(text: str) -> dict:
    text = text.strip()
    # strip markdown fences (```json ... ``` or ``` ... ```)
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        print(f"[analyzer] JSON parse failed, raw response: {text[:200]!r}")
        raise


async def _call_vision_async(client, frame: dict, prompt: str, model: str) -> tuple[str, int]:
    b64 = _encode_image(frame["frame_path"])
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return content, tokens


async def _analyze_frame_async(client, frame: dict, model: str) -> dict:
    idx = frame["frame_index"]
    t0 = time.perf_counter()
    try:
        content, tokens = await _call_vision_async(client, frame, SYSTEM_PROMPT, model)
        elapsed = time.perf_counter() - t0
        parse_ok = True
        try:
            data = _parse_json_from_text(content)
        except (json.JSONDecodeError, AttributeError):
            parse_ok = False
            data = {"description": content[:120] if content else "Parse error", "score": 5, "tags": []}

        print(f"[analyzer] frame {idx:3d} | {elapsed:.2f}s | {tokens} tokens | parse={'ok' if parse_ok else 'FALLBACK'}")
        return {
            "timestamp_sec": frame["timestamp_sec"],
            "frame_index": idx,
            "description": str(data.get("description", "")),
            "score": int(data.get("score", 5)),
            "tags": list(data.get("tags", [])),
            "tokens_used": tokens,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"[analyzer] frame {idx:3d} | {elapsed:.2f}s | ERROR: {e}")
        return {
            "timestamp_sec": frame["timestamp_sec"],
            "frame_index": idx,
            "description": "Analysis failed",
            "score": 5,
            "tags": [],
            "tokens_used": 0,
        }


async def analyze_frames(
    frames: list[dict],
    job_id: str,
    api_key: str,
    model: str,
    base_url: str,
    batch_size: int = 20,
) -> tuple[list[dict], int]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    results: list[dict] = []
    total_tokens = 0

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(frames) + batch_size - 1) // batch_size
        print(f"[analyzer] batch {batch_num}/{total_batches} — {len(batch)} frames starting at frame {batch[0]['frame_index']}")
        t_batch = time.perf_counter()
        batch_results = await asyncio.gather(
            *[_analyze_frame_async(client, f, model) for f in batch]
        )
        batch_tokens = sum(r["tokens_used"] for r in batch_results)
        print(f"[analyzer] batch {batch_num}/{total_batches} done in {time.perf_counter()-t_batch:.2f}s | {batch_tokens} tokens")
        results.extend(batch_results)
        total_tokens += batch_tokens

        if i + batch_size < len(frames):
            await asyncio.sleep(0.5)

    analysis_path = TEMP_DIR / job_id / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    await client.close()
    return results, total_tokens
