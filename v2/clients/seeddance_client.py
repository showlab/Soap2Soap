"""
Seed Dance (BytePlus) client — image-to-video via seedance-1-5-pro.
Auth: BYTEPLUS_API_KEY environment variable.

API docs: https://docs.byteplus.com/en/docs/ModelArk/1366799
"""
from __future__ import annotations
import asyncio
import os

SEEDDANCE_MODEL = "seedance-1-5-pro-251215"
BYTEPLUS_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"

_VALID_DURATIONS = [4, 8]


def _snap_duration(seconds: int) -> int:
    return min(_VALID_DURATIONS, key=lambda v: abs(v - seconds))


def _get_client():
    from byteplussdkarkruntime import AsyncArk
    api_key = os.environ.get("BYTEPLUS_API_KEY")
    if not api_key:
        raise EnvironmentError("BYTEPLUS_API_KEY is not set.")
    return AsyncArk(base_url=BYTEPLUS_BASE_URL, api_key=api_key)


def _image_to_base64_url(image_path: str, max_kb: int = 800) -> str:
    """Convert image to base64 data URL, resizing if needed to stay under max_kb."""
    from PIL import Image
    import io, base64
    img = Image.open(image_path).convert("RGB")
    # Resize if image would produce a too-large base64
    quality = 85
    for _ in range(5):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        size_kb = buf.tell() // 1024
        if size_kb <= max_kb:
            break
        # Halve dimensions and retry
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.LANCZOS)
        quality = 85
    b64 = base64.b64encode(buf.getvalue()).decode()
    print(f"     Image encoded: {size_kb}KB → base64")
    return f"data:image/jpeg;base64,{b64}"


async def _generate_async(
    image_path: str,
    prompt: str,
    output_path: str,
    duration: int,
    aspect_ratio: str,
) -> bool:
    import httpx

    snap_dur = _snap_duration(duration)

    print(f"  🎬 Seed Dance: generating {snap_dur}s video (from {duration}s shot)...")
    image_url = _image_to_base64_url(image_path)

    client = _get_client()
    content = [
        {
            "type": "text",
            "text": (
                f"{prompt} "
                f"--resolution 1080p "
                f"--ratio {aspect_ratio} "
                f"--duration {snap_dur} "
                f"--camerafixed false"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": image_url},
            "role": "first_frame",
        },
    ]

    result = await client.content_generation.tasks.create(
        model=SEEDDANCE_MODEL,
        content=content,
    )
    task_id = result.id
    print(f"     Task submitted: {task_id}")

    poll = 0
    while True:
        poll += 1
        await asyncio.sleep(5)
        print(f"     Waiting... ({poll * 5}s)")
        status = await client.content_generation.tasks.get(task_id=task_id)
        if status.status == "succeeded":
            video_url = status.content.video_url
            break
        elif status.status in ("failed", "cancelled"):
            raise RuntimeError(f"Task {status.status}: {getattr(status, 'error', 'unknown')}")

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as http:
        resp = await http.get(video_url)
        resp.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(resp.content)
    print(f"  ✅ Seed Dance saved: {output_path} ({snap_dur}s, {len(resp.content)//1024}KB)")
    return True


def generate_video_seeddance(
    image_path: str,
    prompt: str,
    output_path: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
) -> bool:
    """
    Synchronous entry point. Uses a fresh event loop each call to avoid reuse issues.
    No retries — each call costs money.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                _generate_async(image_path, prompt, output_path, duration, aspect_ratio)
            )
        finally:
            loop.close()
    except Exception as e:
        print(f"  ❌ Seed Dance failed: {e}")
        return False
