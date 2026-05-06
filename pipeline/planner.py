import asyncio
import json
import re
import time
from pathlib import Path

TEMP_DIR = Path("temp")

SYSTEM_PROMPT_TEMPLATE = """You are a professional video editor AI.
Given scene analysis data and a user editing prompt, create a precise cut plan that strictly follows the user's instructions.

Return ONLY a single valid JSON object — no markdown fences, no extra text.
Required format:
{{
  "reasoning": "brief explanation of selection choices",
  "total_duration_sec": <number>,
  "segments": [
    {{"start_sec": <number>, "end_sec": <number>, "reason": "<why this segment>"}},
    ...
  ]
}}

Hard constraints you MUST respect:
- Target total duration: {target_sec} seconds (sum of all segments ≈ this value)
- Minimum segment length: 2 seconds (no micro-cuts)
- Maximum 12 segments (keep it coherent)
- Segments must be in chronological order
- No overlapping segments
- Follow the user's instructions literally. Only prioritize high-relevance frames if the user explicitly asks for highlights or a highlight reel.
"""


def _parse_duration_from_prompt(prompt: str) -> int:
    p = prompt.lower()
    m = re.search(r"(\d+)\s*min(?:ute)?s?", p)
    if m:
        return int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*s(?:ec(?:ond)?s?)?(?:\b)", p)
    if m:
        return int(m.group(1))
    return 30


def _validate_cut_plan(plan: dict, video_duration: float | None = None) -> list[str]:
    errors = []
    segs = plan.get("segments", [])
    if not segs:
        errors.append("No segments in plan")
        return errors

    prev_end = -1.0
    for i, s in enumerate(segs):
        start = s.get("start_sec", 0)
        end = s.get("end_sec", 0)
        length = end - start
        if length < 2:
            errors.append(f"Segment {i} is too short ({length:.1f}s < 2s)")
        if start < prev_end:
            errors.append(f"Segment {i} overlaps previous (start={start}, prev_end={prev_end})")
        if video_duration and end > video_duration:
            errors.append(f"Segment {i} end ({end}s) exceeds video duration ({video_duration}s)")
        prev_end = end

    if len(segs) > 12:
        errors.append(f"Too many segments ({len(segs)} > 12)")
    return errors


def _parse_plan_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


async def _call_text_async(client, system_prompt: str, user_content: str, model: str) -> tuple[str, int]:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens if response.usage else 0
    return content, tokens


async def plan_cuts(
    analysis: list[dict],
    prompt: str,
    job_id: str,
    api_key: str,
    model: str,
    base_url: str,
) -> dict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    target_sec = _parse_duration_from_prompt(prompt)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(target_sec=target_sec)

    user_content = (
        f"User request: {prompt}\n\n"
        f"Scene analysis ({len(analysis)} frames):\n"
        + json.dumps(analysis, indent=2, ensure_ascii=False)
        + f"\n\nExecute the user's editing instructions. Target duration: {target_sec} seconds."
    )

    input_chars = len(user_content) + len(system_prompt)
    print(f"[planner] input size: {input_chars} chars (~{input_chars//4} tokens est.) | {len(analysis)} frames | target {target_sec}s")
    t0 = time.perf_counter()
    content, tokens = await _call_text_async(client, system_prompt, user_content, model)
    print(f"[planner] API call done in {time.perf_counter()-t0:.2f}s | {tokens} tokens | response {len(content)} chars")

    parse_ok = True
    try:
        plan = _parse_plan_json(content)
    except (json.JSONDecodeError, AttributeError):
        parse_ok = False
        print(f"[planner] JSON parse failed, retrying. Raw response: {content[:300]!r}")
        retry_system = "Return ONLY the JSON object, no other text:\n" + system_prompt
        t1 = time.perf_counter()
        content2, tokens2 = await _call_text_async(client, retry_system, user_content, model)
        print(f"[planner] retry done in {time.perf_counter()-t1:.2f}s | {tokens2} tokens")
        plan = _parse_plan_json(content2)

    print(f"[planner] plan parsed ok={parse_ok} | {len(plan.get('segments', []))} segments")

    await client.close()

    plan_path = TEMP_DIR / job_id / "cut_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    return plan


def validate_cut_plan(plan: dict, video_duration: float | None = None) -> list[str]:
    return _validate_cut_plan(plan, video_duration)
