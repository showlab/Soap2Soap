"""
Step 2 — Character Reference Image Generation.
Uses Imagen 3 to generate a clean reference sheet for each character.
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from v2.clients.imagen_client import generate_character_image
from v2.prompts.shot_compiler import STYLE_PREFIXES

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


def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print("STEP 2 — Character Reference Image Generation")
    print("=" * 60)

    style_prefix = STYLE_PREFIXES.get(state.style, STYLE_PREFIXES["realistic"])

    for char in state.characters:
        save_path = os.path.join(state.output_dir, f"char_{char.id.replace('@', '')}.png")

        if os.path.exists(save_path):
            print(f"  ⏭️  {char.id} — already exists, skipping")
            char.image_path = save_path
            # Update reference store
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
    return state
