"""
Step 2 — Character Reference Image Generation + Design Sheet.

1. Generate individual reference image per character (Imagen 3 → Gemini fallback).
2. Merge all character images into a single unified Design Sheet via Gemini.
   The design sheet becomes the primary consistency reference for all shots.
"""
from __future__ import annotations
import os
from typing import List, TYPE_CHECKING

from PIL import Image
from v2.clients.imagen_client import generate_character_image, generate_keyframe
from v2.prompts.shot_compiler import STYLE_PREFIXES
from v2.prompts.design_sheet import get_design_sheet_prompt

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


CHARACTER_T2I_TEMPLATE = """{style_prefix}

Character reference sheet. Full-body portrait on neutral light background.
Designed for visual consistency across animated shots.

{description}

Pose: Standing, facing camera, neutral expression, arms relaxed.
Background: Clean white or very light grey, no props.
Lighting: Soft even lighting, no harsh shadows.
Detail level: High — capture all clothing details, accessories, hair style precisely.
"""


def _generate_design_sheet(
    char_images: List[str],
    style: str,
    save_path: str,
) -> bool:
    """
    Merge all character reference images into one unified design sheet.
    Uses Gemini image generation (multi-image input → single output).
    """
    if not char_images:
        return False

    pil_images = []
    for path in char_images:
        try:
            pil_images.append(Image.open(path))
        except Exception:
            pass

    if not pil_images:
        return False

    prompt = get_design_sheet_prompt(style=style)
    print(f"  Generating design sheet from {len(pil_images)} character image(s)...")
    img = generate_keyframe(
        prompt=prompt,
        reference_images=pil_images,
        aspect_ratio="16:9",
        save_path=save_path,
    )
    return img is not None


def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print("STEP 2 — Character Reference Images + Design Sheet")
    print("=" * 60)

    style_prefix = STYLE_PREFIXES.get(state.style, STYLE_PREFIXES["realistic"])

    # ── 2a: Individual character reference images ─────────────────────────────
    for char in state.characters:
        save_path = os.path.join(state.output_dir, f"char_{char.id.replace('@', '')}.png")

        if os.path.exists(save_path):
            print(f"  ⏭️  {char.id} — already exists, skipping")
            char.image_path = save_path
            entity = state.reference_store.get_entity(char.id)
            if entity:
                entity.image_path = save_path
            continue

        print(f"  Generating reference image for {char.id} ({char.name})...")
        prompt = CHARACTER_T2I_TEMPLATE.format(
            style_prefix=style_prefix,
            description=char.description,
        )
        img = generate_character_image(prompt, aspect_ratio="1:1", save_path=save_path)
        if img:
            char.image_path = save_path
            entity = state.reference_store.get_entity(char.id)
            if entity:
                entity.image_path = save_path
        else:
            print(f"  ⚠️  Failed to generate reference for {char.id} — will use text-only")

    generated = sum(1 for c in state.characters if c.image_path)
    print(f"  {generated}/{len(state.characters)} character reference images ready")

    # ── 2b: Design sheet (unified character collage) ──────────────────────────
    sheet_path = os.path.join(state.output_dir, "design_sheet.png")

    if os.path.exists(sheet_path):
        print(f"  ⏭️  Design sheet already exists: {sheet_path}")
        state.design_sheet_path = sheet_path
    else:
        char_images = [c.image_path for c in state.characters if c.image_path]
        if len(char_images) >= 1:
            ok = _generate_design_sheet(char_images, state.style, sheet_path)
            if ok:
                state.design_sheet_path = sheet_path
                print(f"  ✅ Design sheet saved: {sheet_path}")
            else:
                print(f"  ⚠️  Design sheet generation failed — will use individual refs")
        else:
            print(f"  ⚠️  No character images available for design sheet")

    return state
