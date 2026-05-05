import asyncio
import base64
import json
import os
import re
from pathlib import Path

TEMP_DIR = Path("temp")

SYSTEM_PROMPT = (
    "You are a video analyst. Describe this video frame concisely in 1-2 sentences. "
    "Rate its 'highlight value' from 1-10 based on visual interest, action, emotion, or importance. "
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _call_vision(client, frame: dict, prompt: str, model: str) -> tuple[str, int]:
    b64 = _encode_image(frame["frame_path"])
    response = client.chat.completions.create(
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


def _analyze_frame_sync(client, frame: dict, model: str) -> dict:
    try:
        content, tokens = _call_vision(client, frame, SYSTEM_PROMPT, model)
        try:
            data = _parse_json_from_text(content)
        except (json.JSONDecodeError, AttributeError):
            try:
                content2, tokens2 = _call_vision(client, frame, STRICT_PROMPT, model)
                tokens += tokens2
                data = _parse_json_from_text(content2)
            except Exception:
                data = {"description": content[:120] if content else "Parse error", "score": 5, "tags": []}

        return {
            "timestamp_sec": frame["timestamp_sec"],
            "frame_index": frame["frame_index"],
            "description": str(data.get("description", "")),
            "score": int(data.get("score", 5)),
            "tags": list(data.get("tags", [])),
            "tokens_used": tokens,
        }
    except Exception as e:
        print(f"[analyzer] frame {frame['frame_index']} error: {e}")
        return {
            "timestamp_sec": frame["timestamp_sec"],
            "frame_index": frame["frame_index"],
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
) -> tuple[list[dict], int]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    results: list[dict] = []
    total_tokens = 0
    batch_size = 5

    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        batch_results = await asyncio.gather(
            *[asyncio.to_thread(_analyze_frame_sync, client, f, model) for f in batch]
        )
        results.extend(batch_results)
        total_tokens += sum(r["tokens_used"] for r in batch_results)

        if i + batch_size < len(frames):
            await asyncio.sleep(0.5)

    analysis_path = TEMP_DIR / job_id / "analysis.json"
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results, total_tokens
