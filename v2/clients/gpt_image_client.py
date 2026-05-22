"""
GPT Image client — keyframe generation via OpenAI gpt-image-1.
Auth: OPENAI_API_KEY environment variable.

Supports:
  - Text-to-image (no reference images)
  - Multi-image reference via base64-encoded images in prompt context
"""
from __future__ import annotations
import base64
import io
import os
from typing import List, Optional

from PIL import Image

GPT_IMAGE_MODEL = "gpt-image-1"

# Aspect ratio → OpenAI size mapping
_SIZE_MAP = {
    "16:9":  "1536x1024",
    "9:16":  "1024x1536",
    "1:1":   "1024x1024",
    "4:3":   "1536x1024",   # closest landscape
    "3:4":   "1024x1536",   # closest portrait
}
_DEFAULT_SIZE = "1536x1024"


def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _pil_to_b64(img: Image.Image, max_kb: int = 800) -> str:
    """Compress and encode PIL image to base64 JPEG."""
    img = img.convert("RGB")
    quality = 85
    for _ in range(5):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() // 1024 <= max_kb:
            break
        w, h = img.size
        img = img.resize((w * 3 // 4, h * 3 // 4), Image.LANCZOS)
    return base64.b64encode(buf.getvalue()).decode()


def _build_prompt_with_refs(prompt: str, reference_images: List[Image.Image]) -> list:
    """
    Build OpenAI message content with reference images + text prompt.
    Returns a content list for use in responses.create or chat.completions.
    """
    content = []
    for i, ref_img in enumerate(reference_images, 1):
        b64 = _pil_to_b64(ref_img)
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}",
        })
    content.append({"type": "input_text", "text": prompt})
    return content


def generate_keyframe(
    prompt: str,
    reference_images: List[Image.Image],
    aspect_ratio: str = "16:9",
    save_path: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Image.Image]:
    """
    Generate a keyframe using GPT Image 1.
    - No reference images: images.generate (text-to-image)
    - With reference images: responses.create with image input → image output
    """
    client = _get_client()
    size = _SIZE_MAP.get(aspect_ratio, _DEFAULT_SIZE)

    for attempt in range(1, max_retries + 1):
        try:
            if not reference_images:
                # Pure text-to-image
                response = client.images.generate(
                    model=GPT_IMAGE_MODEL,
                    prompt=prompt,
                    size=size,
                    quality="high",
                    n=1,
                    response_format="b64_json",
                )
                b64_data = response.data[0].b64_json
            else:
                # Multi-image reference via responses API
                content = _build_prompt_with_refs(prompt, reference_images)
                response = client.responses.create(
                    model=GPT_IMAGE_MODEL,
                    input=content,
                    tools=[{"type": "image_generation", "size": size, "quality": "high"}],
                )
                # Extract generated image from tool output
                b64_data = None
                for item in response.output:
                    if getattr(item, "type", None) == "image_generation_call":
                        b64_data = item.result
                        break
                if b64_data is None:
                    print(f"  ❌ Attempt {attempt}: no image in response output")
                    continue

            img_bytes = base64.b64decode(b64_data)
            img = Image.open(io.BytesIO(img_bytes))
            if save_path:
                img.save(save_path)
                print(f"  ✅ Keyframe saved (GPT Image): {save_path}")
            return img

        except Exception as e:
            print(f"  ❌ GPT Image attempt {attempt} failed: {e}")
            if attempt < max_retries:
                import time
                time.sleep(5)

    return None
