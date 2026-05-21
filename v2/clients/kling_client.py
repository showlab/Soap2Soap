"""
Kling (可灵) client — image-to-video generation via Kling AI API.
Auth: JWT HS256 signed with KLING_ACCESS_KEY + KLING_SECRET_KEY
"""
from __future__ import annotations
import base64
import os
import time
import requests

KLING_API_BASE = "https://api.klingai.com"
KLING_MODEL = "kling-v1-6"

# Kling supports 5 or 10 seconds
_VALID_DURATIONS = [5, 10]


def _snap_duration(seconds: int) -> str:
    """Snap to nearest valid Kling duration (5 or 10s), returned as string."""
    snapped = min(_VALID_DURATIONS, key=lambda v: abs(v - seconds))
    return str(snapped)


def _get_jwt_token() -> str:
    """Build a short-lived JWT using KLING_ACCESS_KEY + KLING_SECRET_KEY."""
    try:
        import jwt as pyjwt
    except ImportError:
        raise ImportError("PyJWT is required for Kling client: pip install PyJWT")

    access_key = os.environ.get("KLING_ACCESS_KEY")
    secret_key = os.environ.get("KLING_SECRET_KEY")
    if not access_key or not secret_key:
        raise EnvironmentError(
            "KLING_ACCESS_KEY and KLING_SECRET_KEY must be set."
        )

    now = int(time.time())
    payload = {
        "iss": access_key,
        "exp": now + 1800,  # 30 min
        "nbf": now - 5,
    }
    return pyjwt.encode(payload, secret_key, algorithm="HS256")


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_jwt_token()}",
        "Content-Type": "application/json",
    }


def generate_video_kling(
    image_path: str,
    prompt: str,
    output_path: str,
    duration: int = 5,
    aspect_ratio: str = "16:9",
) -> bool:
    """
    Generate a video clip via Kling image-to-video API.
    No retries — each call costs money.
    """
    kling_duration = _snap_duration(duration)
    print(f"  🎬 Kling: generating {kling_duration}s video (from {duration}s shot)...")

    # Encode image as base64
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model_name": KLING_MODEL,
        "image": f"data:image/png;base64,{image_b64}",
        "prompt": prompt,
        "duration": kling_duration,
        "aspect_ratio": aspect_ratio,
        "cfg_scale": 0.5,
    }

    try:
        # Submit task
        resp = requests.post(
            f"{KLING_API_BASE}/v1/videos/image2video",
            headers=_headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        if result.get("code") != 0:
            print(f"  ❌ Kling submit failed: {result.get('message', result)}")
            return False

        task_id = result["data"]["task_id"]
        print(f"     Task submitted: {task_id}")

        # Poll until done
        poll = 0
        while True:
            poll += 1
            time.sleep(10)
            print(f"     Waiting... ({poll * 10}s)")

            status_resp = requests.get(
                f"{KLING_API_BASE}/v1/videos/image2video/{task_id}",
                headers=_headers(),
                timeout=30,
            )
            status_resp.raise_for_status()
            status = status_resp.json()

            if status.get("code") != 0:
                print(f"  ❌ Kling poll error: {status.get('message', status)}")
                return False

            task_status = status["data"]["task_status"]
            if task_status == "succeed":
                videos = status["data"]["task_result"]["videos"]
                if not videos:
                    print("  ⚠️  Kling: no videos in result")
                    return False
                video_url = videos[0]["url"]
                break
            elif task_status in ("failed", "error"):
                reason = status["data"].get("task_status_msg", "unknown")
                print(f"  ❌ Kling task failed: {reason}")
                return False
            # else: processing/submitted → keep polling

        # Download video
        dl = requests.get(video_url, timeout=120)
        dl.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(dl.content)
        print(f"  ✅ Kling video saved: {output_path} ({kling_duration}s)")
        return True

    except Exception as e:
        print(f"  ❌ Kling failed: {e}")
        return False
