"""
Design Sheet prompt — merges all character reference images into one unified sheet.
Ported from pai_v1_backend.
"""

DESIGN_SHEET_PROMPT = """Create a single unified character design sheet for an animated production.

The sheet shows ALL provided characters side-by-side with clear labels.

REQUIREMENTS:
- Show all characters as full-body standing poses on a neutral background
- Each character labeled with their name/ID below them
- All characters rendered in the SAME visual style: {style}
- Consistent lighting across all characters (soft studio lighting)
- Clean white or very light grey background
- No scene props — characters only
- Characters arranged horizontally from left to right

Use the provided reference images as the EXACT visual basis for each character's
appearance (face, hair, clothing, build). Do NOT invent new appearances.

This design sheet will be used as a consistency reference for all subsequent
video frame generation — accuracy and clarity are critical.
"""


def get_design_sheet_prompt(style: str = "disney") -> str:
    return DESIGN_SHEET_PROMPT.format(style=style)
