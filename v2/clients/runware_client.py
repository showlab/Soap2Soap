"""
Runware image generation client — GPT Image 2 (openai:gpt-image@2).
Auth: RUNWARE_API_KEY environment variable.
REST endpoint: https://api.runware.ai/v1
"""
from __future__ import annotations
import base64
import io
import os
import uuid
from typing import List, Optional

import requests
from PIL import Image

RUNWARE_API_URL = "https://api.runware.ai/v1"
RUNWARE_MODEL = "openai:gpt-image@2"

# Aspect ratio → (width, height); multiples of 16, area 655k–8.3M
# 1280×720 chosen for 16:9 — works reliably with reference images (1536×864 causes 504)
_SIZE_MAP = {
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "1:1":  (1024, 1024),
    "4:3":  (1360, 1024),
    "3:4":  (1024, 1360),
}
_DEFAULT_SIZE = (1280, 720)


def _api_key() -> str:
    key = os.environ.get("RUNWARE_API_KEY")
    if not key:
        raise EnvironmentError("RUNWARE_API_KEY is not set.")
    return key


def _pil_to_data_uri(img: Image.Image) -> str:
    """Encode PIL image as JPEG data URI."""
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def generate_keyframe(
    prompt: str,
    reference_images: List[Image.Image],
    aspect_ratio: str = "16:9",
    save_path: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Image.Image]:
    """
    Generate a keyframe using Runware GPT Image 2.
    Reference images are passed as JPEG data URIs.
    """
    api_key = _api_key()
    width, height = _SIZE_MAP.get(aspect_ratio, _DEFAULT_SIZE)

    task: dict = {
        "taskType": "imageInference",
        "taskUUID": str(uuid.uuid4()),
        "model": RUNWARE_MODEL,
        "positivePrompt": prompt,
        "width": width,
        "height": height,
        "outputType": "URL",
        "outputFormat": "PNG",
        "numberResults": 1,
    }

    if reference_images:
        task["referenceImages"] = [_pil_to_data_uri(img) for img in reference_images]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                RUNWARE_API_URL,
                json=[task],
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()

            results = data.get("data", [])
            if not results:
                print(f"  ❌ Attempt {attempt}: empty response data")
                continue

            image_url = results[0].get("imageURL")
            if not image_url:
                print(f"  ❌ Attempt {attempt}: no imageURL in response — {results[0]}")
                continue

            # Download the image
            img_resp = requests.get(image_url, timeout=60)
            img_resp.raise_for_status()
            img = Image.open(io.BytesIO(img_resp.content))
            if save_path:
                img.save(save_path)
                print(f"  ✅ Keyframe saved (Runware GPT-Image-2): {save_path}")
            return img

        except requests.HTTPError as e:
            print(f"  ❌ Attempt {attempt}: HTTP {e.response.status_code} — {e.response.text[:200]}")
        except Exception as exc:
            print(f"  ❌ Attempt {attempt}: {exc}")

    return None
