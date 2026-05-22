"""
ReferenceResolver — replaces @character_XX tokens with descriptions and
assembles the reference image list for each API call.

Ported from pai_v1_backend/app/model/prompts/reference_resolver.py
and adapted for local-file (path-based) workflow.
"""
from __future__ import annotations
from typing import List, Optional, Tuple, TYPE_CHECKING
from PIL import Image

if TYPE_CHECKING:
    from v2.core.reference_store import ReferenceStore


def resolve_references(
    base_prompt: str,
    reference_store: "ReferenceStore",
    previous_image_path: Optional[str] = None,
    inject_consistency_instruction: bool = True,
) -> Tuple[str, List[Image.Image]]:
    """
    Build the final prompt and the ordered reference image list for an API call.

    Args:
        base_prompt: Shot prompt containing @character_XX tokens.
        reference_store: Populated store with character images.
        previous_image_path: Path to the previous shot's keyframe (consistency mode).
        inject_consistency_instruction: Add "maintain consistency" language.

    Returns:
        (final_prompt, pil_images)
        pil_images order: character refs first, then previous-shot image last.
    """
    # 1. Find which entities are mentioned
    mentioned = []
    for entity_id, entity in reference_store.entities.items():
        if entity_id in base_prompt:
            mentioned.append(entity)
    mentioned.sort(key=lambda e: e.image_index)

    # 2. Load character reference images
    ref_images: List[Image.Image] = []
    loaded_entities = []
    for entity in mentioned:
        if entity.image_path:
            try:
                ref_images.append(Image.open(entity.image_path))
                loaded_entities.append(entity)
            except Exception:
                pass  # Skip missing images gracefully

    # 3. Build legend string  e.g. "Image 1 (Rose), Image 2 (Jack)"
    legend_parts = []
    for i, entity in enumerate(loaded_entities):
        name = reference_store.get_entity_name(entity.entity_id)
        legend_parts.append(f"Image {i + 1} ({name})")

    # 4. Optionally append previous-shot image for consistency
    prev_img_idx: Optional[int] = None
    if previous_image_path:
        try:
            ref_images.append(Image.open(previous_image_path))
            prev_img_idx = len(ref_images)
            legend_parts.append(f"Image {prev_img_idx} (Previous Shot)")
        except Exception:
            pass

    legend_str = ""
    if legend_parts:
        legend_str = "Reference images: " + ", ".join(legend_parts) + ". "

    # 5. Replace @tokens with "Name from Image N"
    resolved = base_prompt
    for i, entity in enumerate(loaded_entities):
        name = reference_store.get_entity_name(entity.entity_id)
        resolved = resolved.replace(entity.entity_id, f"{name} from Image {i + 1}")

    # 6. Consistency instruction
    consistency_note = ""
    if inject_consistency_instruction and prev_img_idx:
        consistency_note = (
            f"Maintain EXACT visual consistency with the environment, lighting, "
            f"and character appearance from Image {prev_img_idx} (Previous Shot). "
        )

    final_prompt = f"{legend_str}{consistency_note}Shot Description: {resolved}"
    return final_prompt, ref_images
