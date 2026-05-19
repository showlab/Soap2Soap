"""
Imagen client — text-to-image (character references) and
image-to-image (shot keyframes with consistency references).

For text-to-image:   uses Imagen 3 (imagen-3.0-generate-002)
For image-to-image:  uses Gemini 3 Pro Image (gemini-2.0-flash-preview-image-generation)
                     because Imagen 3 does not support multi-image reference input.
"""
from __future__ import annotations
import os
import io
from typing import List, Optional, Tuple
from PIL import Image
from google import genai
from google.genai import types

IMAGEN_MODEL = "imagen-3.0-generate-002"
GEMINI_IMAGE_MODEL = "gemini-2.0-flash-preview-image-generation"


def _client() -> genai.Client:
    api_key = os.environ.get("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("GENAI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def generate_character_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    save_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Text-to-image via Imagen 3 for generating character reference sheets.
    Returns a PIL Image, and optionally saves it to disk.
    """
    client = _client()
    try:
        response = client.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                safety_filter_level="BLOCK_MEDIUM_AND_ABOVE",
            ),
        )
        if not response.generated_images:
            print(f"  ⚠️  Imagen returned no image for prompt (len={len(prompt)})")
            return None
        img_bytes = response.generated_images[0].image.image_bytes
        img = Image.open(io.BytesIO(img_bytes))
        if save_path:
            img.save(save_path)
            print(f"  ✅ Character image saved: {save_path}")
        return img
    except Exception as e:
        print(f"  ❌ Imagen generation failed: {e}")
        return None


def generate_keyframe(
    prompt: str,
    reference_images: List[Image.Image],
    aspect_ratio: str = "16:9",
    save_path: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Image.Image]:
    """
    Image-to-image via Gemini for shot keyframes.
    Sends (prompt + reference images) to Gemini image generation model.
    Automatically safety-rewrites and retries on EMPTY_PARTS / PROHIBITED_CONTENT.
    """
    from v2.clients.gemini_client import safety_rewrite

    client = _client()
    current_prompt = prompt

    # Cap prompt length
    MAX_CHARS = 4000
    if len(current_prompt) > MAX_CHARS:
        print(f"  ⚠️  Prompt {len(current_prompt)} chars → truncating to {MAX_CHARS}")
        current_prompt = current_prompt[:MAX_CHARS]

    for attempt in range(1, max_retries + 1):
        try:
            contents: list = [current_prompt] + reference_images

            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                ),
            )

            # Parse response
            candidates = getattr(response, "candidates", [])
            if not candidates:
                print(f"  ❌ Attempt {attempt}: no candidates")
                _log_empty_response(response)
                if attempt < max_retries:
                    current_prompt = safety_rewrite(current_prompt)
                continue

            candidate = candidates[0]
            finish = str(getattr(candidate, "finish_reason", ""))
            if finish in ("SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY"):
                print(f"  ❌ Attempt {attempt}: {finish} — safety rewriting prompt")
                if attempt < max_retries:
                    current_prompt = safety_rewrite(current_prompt)
                continue

            parts = getattr(getattr(candidate, "content", None), "parts", None)
            if not parts:
                print(f"  ❌ Attempt {attempt}: EMPTY_PARTS — safety rewriting prompt")
                _log_empty_response(response)
                if attempt < max_retries:
                    current_prompt = safety_rewrite(current_prompt)
                continue

            for part in parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    img = Image.open(io.BytesIO(part.inline_data.data))
                    if save_path:
                        img.save(save_path)
                        print(f"  ✅ Keyframe saved: {save_path}")
                    return img

            print(f"  ⚠️  Attempt {attempt}: no image part found")

        except Exception as e:
            print(f"  ❌ Attempt {attempt}: exception — {e}")
            if attempt < max_retries:
                import time; time.sleep(5)

    return None


def _log_empty_response(response) -> None:
    """Log raw response info when generation fails — disambiguates quota vs filter."""
    candidates = getattr(response, "candidates", [])
    if candidates:
        c = candidates[0]
        print(f"     finish_reason : {getattr(c, 'finish_reason', 'N/A')}")
        print(f"     safety_ratings: {getattr(c, 'safety_ratings', 'N/A')}")
    print(f"     prompt_feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
