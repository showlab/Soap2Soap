"""
Zoom-In prompt — extracts a single shot from a 2×2 grid via camera crop.
Ported from pai_v1_backend/app/model/prompts/zoom_in.py
"""

GRID_POSITIONS = ["TOP-LEFT", "TOP-RIGHT", "BOTTOM-LEFT", "BOTTOM-RIGHT"]

ZOOM_IN_TEMPLATE = """ZOOM-IN — {POSITION} GRID CELL (CAMERA CROP ONLY)

Generate a single cinematic image that is a zoomed-in view of the {POSITION} frame
from the provided 2×2 reference grid image.

CONTENT DESCRIPTION:
{CONTENT_DESCRIPTION}

This is a CAMERA ZOOM + CROP only.
NOT a new scene, NOT a redesign, NOT a reinterpretation.

HARD CONSTRAINTS — preserve EXACTLY:
• Environment layout and spatial relationships
• Geometry and perspective
• Lighting direction, intensity, and color temperature
• Art style and rendering style
• All characters and their appearance
• Same moment in time, same world state

CAMERA INSTRUCTIONS:
• Same camera position and angle as {POSITION} frame
• Same perspective lines and horizon line
• Reduced field of view only (longer focal length effect)
• Must look like a cropped and zoomed version — not a new composition

No borders. No text. No style drift.
"""


def compile_zoom_in_prompt(position: str, content_description: str = "") -> str:
    """
    Args:
        position: One of GRID_POSITIONS
        content_description: Shot's t2i_prompt for context
    """
    if position not in GRID_POSITIONS:
        position = GRID_POSITIONS[0]
    return ZOOM_IN_TEMPLATE.format(
        POSITION=position,
        CONTENT_DESCRIPTION=content_description[:300] if content_description else "As shown in the grid.",
    )
