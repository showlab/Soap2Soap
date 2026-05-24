"""
Narrative Grid prompt — generates a 2×2 grid of 4 shots simultaneously.
Ported from pai_v1_backend/app/model/prompts/narrative_grid.py
"""
from typing import List, Dict, Optional

NARRATIVE_GRID_TEMPLATE = """Generate a single image containing a 2×2 grid of 4 cinematic frames.

Maintain consistent character design using the provided reference images.
{transition_note}

No borders. No text in the frames.

════════════════════════════════════════════════════════════════
REFERENCE IMAGES
════════════════════════════════════════════════════════════════
{reference_section}

════════════════════════════════════════════════════════════════
CHARACTERS
════════════════════════════════════════════════════════════════
{character_section}

════════════════════════════════════════════════════════════════
ENVIRONMENT
════════════════════════════════════════════════════════════════
{environment_section}

════════════════════════════════════════════════════════════════
4 SHOTS (2×2 grid, reading order: top-left, top-right, bottom-left, bottom-right)
════════════════════════════════════════════════════════════════
{shot_lines}

════════════════════════════════════════════════════════════════
STYLE
════════════════════════════════════════════════════════════════
{style_keywords}

════════════════════════════════════════════════════════════════
RULES
════════════════════════════════════════════════════════════════
• All characters must match their design sheet appearance in every frame
• Same character = same face, hair, clothing across ALL frames
• Consistent lighting and visual style across all 4 frames
• Previous grids establish the visual canon — maintain continuity
"""


def compile_narrative_grid_prompt(
    shots: List[Dict],
    style: str = "disney",
    characters: Optional[List[Dict]] = None,
    environment: Optional[str] = None,
    previous_grids: int = 0,
    source_frame_grid: bool = False,
) -> str:
    """
    Build a 2×2 grid generation prompt.

    Args:
        shots: Exactly 4 shot dicts with key 't2i_prompt' (pad if fewer).
        style: Target visual style.
        characters: List of {"name": str, "description": str}.
        environment: Shared environment description.
        previous_grids: How many previous grids exist (for reference labeling).
        source_frame_grid: If True, the last reference image is a 2×2 grid of
            source-video frames used as layout/composition reference.
    """
    if len(shots) != 4:
        raise ValueError(f"narrative_grid requires exactly 4 shots, got {len(shots)}")

    style_keywords = {
        "disney":    "Disney 3D animated movie style. Vibrant colors, expressive characters, polished CG render.",
        "realistic": "Photorealistic cinematic style. Shot on 35mm film.",
        "anime":     "Japanese anime style. Clean linework, vivid colors.",
        "clay":      "Claymation stop-motion style. Visible clay texture.",
        "lego":      "LEGO brick animation style. Bright primary colors.",
        "family_guy":"American adult animated TV style. Flat colors, clean outlines.",
    }.get(style, style)

    # Reference section
    if source_frame_grid:
        reference_section = (
            "• Images 1-N: Character reference sheet(s)\n"
            "• LAST image: A 2×2 grid of REAL source-video frames showing the original "
            "compositions, framing, and scene layouts for each of the 4 panels (TL/TR/BL/BR). "
            "Use this last image as the LAYOUT REFERENCE — mimic the camera angle, framing, "
            "and composition of each cell while rendering everything in the target visual style."
        )
    elif previous_grids == 0:
        reference_section = "• Image 1: Character design sheet (all characters shown together)"
    else:
        lines = ["• Image 1: Character design sheet"]
        for i in range(previous_grids):
            lines.append(f"• Image {i+2}: Previous narrative grid {i+1}")
        lines.append("\nMaintain consistency with ALL reference images.")
        reference_section = "\n".join(lines)

    # Character section
    if characters:
        char_lines = [f"• {c['name']}: {c['description'][:120]}" for c in characters]
        character_section = "\n".join(char_lines)
        character_section += "\n\nMatch each character exactly to their design sheet appearance."
    else:
        character_section = "Match characters to the design sheet reference."

    # Environment
    environment_section = environment or "Consistent environment across all frames."

    # Shot lines
    shot_lines = ""
    for i, shot in enumerate(shots):
        desc = shot.get("t2i_prompt") or shot.get("description") or ""
        pad = " (PADDING — duplicate of previous)" if shot.get("padded") else ""
        shot_lines += f"{i+1}. {desc}{pad}\n\n"

    return NARRATIVE_GRID_TEMPLATE.format(
        transition_note="",
        reference_section=reference_section,
        character_section=character_section,
        environment_section=environment_section,
        shot_lines=shot_lines.strip(),
        style_keywords=style_keywords,
    )
