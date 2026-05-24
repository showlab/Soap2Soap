"""
Image generation client.

Character reference sheets: Imagen 3 (text-to-image), falls back to Gemini image model.
Shot keyframes:             Gemini image model with multi-image reference support.
"""
from __future__ import annotations
import io
import os
import time
from typing import List, Optional

from PIL import Image
from google import genai
from google.genai import types

# Imagen 3 — best quality text-to-image, no multi-image-reference support
IMAGEN_MODEL = "imagen-3.0-generate-002"

# Gemini image generation — supports multi-image input (reference consistency)
# Use the model confirmed working with a standard GENAI_API_KEY
# Image generation model (supports multi-image reference input)
GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image-preview"


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
    Generate a character reference sheet.
    Tries Imagen 3 first; falls back to Gemini image model if Imagen is unavailable.
    """
    client = _client()

    # ── Imagen 3 attempt ──────────────────────────────────────────────────────
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
        if response.generated_images:
            img = Image.open(io.BytesIO(response.generated_images[0].image.image_bytes))
            if save_path:
                img.save(save_path)
                print(f"  ✅ Character image saved (Imagen 3): {save_path}")
            return img
    except Exception as e:
        msg = str(e)
        if "NOT_FOUND" in msg or "not supported" in msg:
            print("  ⚠️  Imagen 3 unavailable — falling back to Gemini image model")
        else:
            print(f"  ⚠️  Imagen 3 error ({msg}) — falling back to Gemini image model")

    # ── Gemini image fallback (no reference images for character sheet) ────────
    return generate_keyframe(prompt=prompt, reference_images=[], save_path=save_path)


def generate_keyframe(
    prompt: str,
    reference_images: List[Image.Image],
    aspect_ratio: str = "16:9",
    save_path: Optional[str] = None,
    max_retries: int = 3,
) -> Optional[Image.Image]:
    """
    Generate a shot keyframe via Gemini image generation.
    Accepts an ordered list of reference PIL images (character refs + previous frame).
    Automatically safety-rewrites and retries on PROHIBITED_CONTENT / EMPTY_PARTS.
    """
    from v2.clients.gemini_client import safety_rewrite

    client = _client()
    current_prompt = prompt

    MAX_CHARS = 4000
    if len(current_prompt) > MAX_CHARS:
        current_prompt = current_prompt[:MAX_CHARS]

    for attempt in range(1, max_retries + 1):
        try:
            contents: list = [current_prompt] + reference_images
            response = client.models.generate_content(
                model=GEMINI_IMAGE_MODEL,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )

            candidates = getattr(response, "candidates", [])
            if not candidates:
                print(f"  ❌ Attempt {attempt}: no candidates")
                _log_response(response)
                if attempt < max_retries:
                    current_prompt = safety_rewrite(current_prompt)
                continue

            candidate = candidates[0]
            finish = str(getattr(candidate, "finish_reason", ""))
            if finish in ("SAFETY", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "RECITATION"):
                print(f"  ❌ Attempt {attempt}: {finish} → safety rewriting")
                if attempt < max_retries:
                    current_prompt = safety_rewrite(current_prompt)
                continue

            parts = getattr(getattr(candidate, "content", None), "parts", None)
            if not parts:
                print(f"  ❌ Attempt {attempt}: EMPTY_PARTS → safety rewriting")
                _log_response(response)
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

            print(f"  ⚠️  Attempt {attempt}: no inline image in parts")

        except Exception as exc:
            print(f"  ❌ Attempt {attempt}: {exc}")
            if attempt < max_retries:
                time.sleep(5)

    return None


def generate_keyframe_with_model(
    model: str,
    prompt: str,
    reference_images: List[Image.Image],
    aspect_ratio: str = "16:9",
    save_path: Optional[str] = None,
) -> Optional[Image.Image]:
    """
    Unified keyframe generation entry point.
    model: "gemini" (default) or "gpt-image" (Runware GPT Image 2)
    """
    if model == "gpt-image":
        from v2.clients.runware_client import generate_keyframe as runware_keyframe
        return runware_keyframe(prompt, reference_images, aspect_ratio, save_path)
    return generate_keyframe(prompt, reference_images, aspect_ratio, save_path)


def _log_response(response) -> None:
    """Print raw response metadata to distinguish quota exhaustion from content filters."""
    candidates = getattr(response, "candidates", [])
    if candidates:
        c = candidates[0]
        print(f"     finish_reason : {getattr(c, 'finish_reason', 'N/A')}")
        print(f"     safety_ratings: {getattr(c, 'safety_ratings', 'N/A')}")
    print(f"     prompt_feedback: {getattr(response, 'prompt_feedback', 'N/A')}")
