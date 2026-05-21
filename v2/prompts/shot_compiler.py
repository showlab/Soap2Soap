"""
Compiles t2i and i2v prompts for each shot, then rewrites them into the target style.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.core.schema import Shot

STYLE_PREFIXES = {
    "realistic":      "Photorealistic cinematic style. Shot on 35mm film.",
    "disney":         "Disney 3D animated movie style. Vibrant colors, expressive characters, polished CG render.",
    "pixar":          "Pixar 3D animated movie style. Warm soft lighting, subsurface skin glow, richly detailed environments, emotionally expressive characters with realistic proportions.",
    "anime":          "Japanese anime style. Clean linework, vivid colors, cinematic composition.",
    "japanese_anime": "Japanese manga/anime style. Clean lines, dynamic composition, expressive faces.",
    "clay":           "Claymation stop-motion style. Visible clay texture, warm handcrafted look.",
    "lego":           "LEGO brick animation style. Blocky figures, bright primary colors, plastic sheen.",
    "family_guy":     "American adult animated TV style. Flat colors, clean outlines, comedic proportions.",
}

# Detailed style rewrite instructions per style
STYLE_REWRITE_GUIDES = {
    "realistic": "Keep the prompt as-is. It is already in photorealistic cinematic language.",
    "disney": (
        "Rewrite using Disney 3D animated movie language. "
        "Characters become expressive 3D-animated Disney figures with rounded features, big eyes, and smooth polished CG surfaces. "
        "Environments become vibrant, colorful, and painterly. "
        "Use words like: vibrant, expressive, animated, polished CG, Disney style, lush, storybook."
    ),
    "pixar": (
        "Rewrite using Pixar 3D animated movie language. "
        "Characters become Pixar-style 3D figures: realistic body proportions, soft subsurface skin glow, richly detailed clothing with fabric texture, highly expressive faces with nuanced emotions. "
        "Environments are warmly lit, physically detailed, and cinematic — like a Pixar feature film frame. "
        "Use words like: Pixar 3D animation, subsurface scattering, warm soft lighting, cinematic depth of field, photorealistic textures, expressive, emotionally rich, detailed environment."
    ),
    "anime": (
        "Rewrite using Japanese anime visual language. "
        "Characters become anime-style with clean linework, large expressive eyes, and stylized proportions. "
        "Environments use flat, vivid colors with detailed linework. "
        "Use words like: anime style, clean linework, cel-shaded, vivid, dynamic pose, expressive."
    ),
    "japanese_anime": (
        "Rewrite using Japanese manga/anime language. "
        "Characters have exaggerated proportions, expressive faces, dynamic poses. "
        "Use words like: manga style, speed lines, expressive, dynamic, cel-shaded, bold outlines."
    ),
    "clay": (
        "Rewrite using claymation stop-motion language. "
        "Characters become clay figures with visible fingerprint texture, slightly imperfect shapes, and a handcrafted warmth. "
        "Environments are made of clay, fabric, and foam. "
        "Use words like: clay figure, stop-motion, claymation, handcrafted, textured clay, warm lighting."
    ),
    "lego": (
        "Rewrite using LEGO animation language. "
        "Characters become LEGO minifigures: blocky cylindrical head, claw hands, snap-on hair/helmet, printed face and torso, no bendable knees. "
        "Clothing becomes printed LEGO torso decorations or snap-on pieces. "
        "Environments are built from colorful LEGO bricks and plates. "
        "Use words like: LEGO minifigure, brick-built, plastic sheen, printed decoration, stud-top, bright primary colors, LEGO diorama."
    ),
    "family_guy": (
        "Rewrite using Family Guy 2D animation language. "
        "Characters become flat-colored cartoon figures with thick black outlines and comedic proportions. "
        "Environments use simple flat backgrounds with clean shapes. "
        "Use words like: cartoon style, flat colors, thick outlines, comedic, 2D animated, simple background."
    ),
}

_STYLE_REWRITE_PROMPT = """Rewrite the following image/video generation prompt to match the target visual style.

TARGET STYLE: {style_name}
STYLE GUIDE: {style_guide}

ORIGINAL PROMPT:
{prompt}

Rules:
- Rewrite ALL character and environment descriptions using the target style language
- Keep ALL @character_XX tokens exactly as written — do NOT replace or remove them
- Keep camera parameters (shot size, lens mm, camera angle, depth of field, composition)
- Keep dialogue references if present
- Output ONLY the rewritten prompt text, no explanation, no markdown, no prefix
"""


def _rewrite_for_style(prompt: str, style: str) -> str:
    """Call Gemini to rewrite prompt into the target style. Falls back to original on error."""
    if style == "realistic":
        return prompt  # no rewrite needed

    from v2.clients.gemini_client import text_generate
    guide = STYLE_REWRITE_GUIDES.get(style, STYLE_REWRITE_GUIDES["realistic"])
    rewrite_prompt = _STYLE_REWRITE_PROMPT.format(
        style_name=style.upper(),
        style_guide=guide,
        prompt=prompt,
    )
    try:
        result = text_generate(rewrite_prompt)
        return result.strip() if result and result.strip() else prompt
    except Exception:
        return prompt


def compile_t2i_prompt(shot: "Shot", style: str = "realistic") -> str:
    """
    Build and style-rewrite the image generation prompt for a shot.
    Step 1: assemble raw content description from shot fields.
    Step 2: call Gemini to rewrite into target style language.
    """
    # Assemble raw content
    lines = []
    if shot.t2i_prompt:
        lines.append(shot.t2i_prompt)
    if shot.environment_description:
        lines.append(f"Environment: {shot.environment_description}")
    if shot.lighting_setup:
        lines.append(f"Lighting: {shot.lighting_setup}")
    if shot.color_grading:
        lines.append(f"Color grading: {shot.color_grading}")
    if shot.shot_size:
        lines.append(f"Shot size: {shot.shot_size}")
    if shot.camera_angle:
        lines.append(f"Camera angle: {shot.camera_angle}")
    if shot.focal_length:
        lines.append(f"Lens: {shot.focal_length}")
    if shot.depth_of_field:
        lines.append(f"Depth of field: {shot.depth_of_field}")
    if shot.mood_atmosphere:
        lines.append(f"Mood: {shot.mood_atmosphere}")
    if shot.composition:
        lines.append(f"Composition: {shot.composition}")

    raw = "\n".join(lines)
    return _rewrite_for_style(raw, style)


def compile_i2v_prompt(shot: "Shot", style: str = "realistic") -> str:
    """
    Build and style-rewrite the video generation prompt for a shot.
    Dialogue is preserved in the original language (not rewritten).
    """
    base = shot.i2v_prompt or shot.t2i_prompt or "A cinematic shot."

    motion_note = ""
    if shot.subject_movement:
        motion_note = f" Action: {shot.subject_movement}."
    if shot.camera_movement:
        motion_note += f" Camera: {shot.camera_movement}."

    # Rewrite only the motion/scene description, not the dialogue
    raw_scene = f"{base}{motion_note}"
    styled_scene = _rewrite_for_style(raw_scene, style)

    # Preserve dialogue in original language
    dialogue_note = ""
    if shot.dialogue:
        lines = "; ".join(f'{d.speaker_id} says: "{d.text}"' for d in shot.dialogue)
        dialogue_note = f" Spoken dialogue (preserve original language exactly): {lines}."

    return f"{styled_scene}{dialogue_note}"
