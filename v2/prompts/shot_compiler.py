"""
Compiles t2i and i2v prompts for each shot.
Ported from pai_v1_backend/app/model/prompts/prompt_compiler.py and adapted.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.core.schema import Shot

STYLE_PREFIXES = {
    "realistic": "Photorealistic cinematic style. Shot on 35mm film.",
    "disney":    "Disney 3D animated movie style. Vibrant colors, expressive characters, polished CG render.",
    "anime":     "Japanese anime style. Clean linework, vivid colors, cinematic composition.",
    "japanese_anime": "Japanese manga/anime style. Clean lines, dynamic composition, expressive faces.",
    "clay":      "Claymation stop-motion style. Visible clay texture, warm handcrafted look.",
    "lego":      "LEGO brick animation style. Blocky figures, bright primary colors, plastic sheen.",
    "family_guy": "American adult animated TV style. Flat colors, clean outlines, comedic proportions.",
}


def compile_t2i_prompt(shot: "Shot", style: str = "realistic") -> str:
    """
    Build the image generation prompt for a shot's first frame.
    Structured like pai_v1_backend: style prefix + scene fields + character tokens.
    """
    style_prefix = STYLE_PREFIXES.get(style, STYLE_PREFIXES["realistic"])
    lines = [f"Style: {style_prefix}"]

    if shot.t2i_prompt:
        lines.append(f"Scene: {shot.t2i_prompt}")
    if shot.environment_description:
        lines.append(f"Environment: {shot.environment_description}")
    if shot.lighting_setup:
        lines.append(f"Lighting: {shot.lighting_setup}")
    if shot.color_grading:
        lines.append(f"Color: {shot.color_grading}")
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

    return "\n".join(lines)


def compile_i2v_prompt(shot: "Shot", style: str = "realistic") -> str:
    """
    Build the video generation prompt for the full shot action.
    Includes dialogue injection if present.
    """
    style_prefix = STYLE_PREFIXES.get(style, STYLE_PREFIXES["realistic"])
    base = shot.i2v_prompt or shot.t2i_prompt or "A cinematic shot."

    # Inject dialogue naturally
    dialogue_note = ""
    if shot.dialogue:
        lines = "; ".join(
            f'{d.speaker_id} says: "{d.text}"' for d in shot.dialogue
        )
        dialogue_note = f" Dialogue: {lines}."

    motion_note = ""
    if shot.subject_movement:
        motion_note = f" Action: {shot.subject_movement}."
    if shot.camera_movement:
        motion_note += f" Camera: {shot.camera_movement}."

    return f"Style: {style_prefix}. {base}{motion_note}{dialogue_note}"
