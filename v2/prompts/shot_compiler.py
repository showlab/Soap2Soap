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

# t2i: full creative rewrite, used for keyframe image generation
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

# i2v: light style prefix only — must stay faithful to the original scene description
_STYLE_I2V_REWRITE_PROMPT = """Add a brief visual style qualifier to the following video generation prompt.

TARGET STYLE: {style_name}
STYLE QUALIFIER: {style_qualifier}

ORIGINAL PROMPT (characters already described in natural language):
{prompt}

Rules:
- Prepend a SHORT style label (e.g. "Claymation stop-motion animation." or "Pixar 3D animated film.")
- Keep the original action and scene description EXACTLY — do NOT expand, rephrase, or embellish
- Do NOT invent new details, objects, or character behaviors not in the original
- Keep all character names (natural language) as-is
- Keep camera notes as-is
- Output ONLY the final prompt, no explanation, 1-3 sentences max
"""

# Short style qualifiers for i2v (just a label, not a full guide)
_I2V_STYLE_QUALIFIERS = {
    "clay":           "Claymation stop-motion animation. Handcrafted clay figures, visible texture.",
    "pixar":          "Pixar 3D animated film. Warm lighting, expressive characters.",
    "disney":         "Disney 3D animated film. Vibrant colors, polished CG.",
    "anime":          "Japanese anime style. Clean linework, expressive.",
    "japanese_anime": "Japanese manga/anime style. Dynamic, expressive.",
    "lego":           "LEGO brick animation. Blocky minifigures, plastic sheen.",
    "family_guy":     "Family Guy 2D cartoon. Flat colors, thick outlines.",
    "realistic":      "",
}


def _rewrite_for_style(prompt: str, style: str) -> str:
    """Full creative rewrite for t2i (keyframe) prompts."""
    if style == "realistic":
        return prompt

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


def _rewrite_i2v_for_style(prompt: str, style: str) -> str:
    """Light style prefix rewrite for i2v prompts — faithful to original content."""
    if style == "realistic":
        return prompt

    qualifier = _I2V_STYLE_QUALIFIERS.get(style, "")
    if not qualifier:
        return prompt

    from v2.clients.gemini_client import text_generate
    rewrite_prompt = _STYLE_I2V_REWRITE_PROMPT.format(
        style_name=style.upper(),
        style_qualifier=qualifier,
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


def _detect_language(texts: list[str]) -> str:
    """Return 'zh' if any text contains Chinese characters, else 'en'."""
    for t in texts:
        if any('一' <= c <= '鿿' for c in t):
            return 'zh'
    return 'en'


def _build_char_alias(description: str) -> str:
    """
    Derive a short natural-language alias from a character description.
    e.g. "Age: 30s. Male. Clothing: black zip-up jacket..." → "man in black jacket"
    """
    import re
    desc = description or ""

    # Gender
    if "Female" in desc:
        gender = "woman" if "Age: 3" in desc or "Age: 4" in desc or "Age: 5" in desc else "girl"
        if re.search(r"Age:\s*\d?[5-9][-–]", desc) or "Age: 5" in desc or "Age: 6" in desc:
            gender = "girl"
    elif "Male" in desc:
        gender = "man"
        if re.search(r"Age:\s*[5-9]\b|\bAge:\s*\d\b", desc):
            gender = "boy"
    else:
        gender = "person"

    # First clothing item (color + item)
    clothing_match = re.search(r"Clothing:\s*([^.]+)", desc)
    clothing = ""
    if clothing_match:
        raw = clothing_match.group(1).strip()
        # split on common connectors, take first item only
        first_item = re.split(r",| and | with | over | worn", raw)[0].strip()
        # keep up to 5 words to stay concise
        words = first_item.split()[:5]
        clothing = " ".join(words).rstrip(".,;")

    if clothing:
        return f"the {gender} in {clothing.lower()}"
    return f"the {gender}"


def build_char_alias_map(characters) -> dict:
    """Build {char_id: alias} map from a list of Character objects."""
    return {c.id: _build_char_alias(c.description) for c in characters}


def resolve_char_tokens(text: str, alias_map: dict) -> str:
    """Replace @character_XX tokens with natural-language aliases."""
    for char_id, alias in alias_map.items():
        text = text.replace(char_id, alias)
    return text


def compile_i2v_prompt(shot: "Shot", style: str = "realistic", dialogue_lang: str = "auto",
                       char_alias_map: dict = None) -> str:
    """
    Build and style-rewrite the video generation prompt for a shot.

    dialogue_lang:
      "auto" — detect from dialogue text (Chinese chars → zh, else en)
      "zh"   — always use Chinese dialogue instruction
      "en"   — always use English dialogue instruction
    """
    import re
    base = shot.i2v_prompt or shot.t2i_prompt or "A cinematic shot."
    # Strip any pre-existing dialogue tail from base (e.g., "台词：xxx说：..."),
    # since we re-append a standardized dialogue note below.
    base = re.sub(r"\s*台词：.*$", "", base).strip()

    motion_note = ""
    if shot.subject_movement:
        motion_note = f" Action: {shot.subject_movement}."
    if shot.camera_movement:
        motion_note += f" Camera: {shot.camera_movement}."

    raw_scene = f"{base}{motion_note}"

    # Resolve @character tokens BEFORE rewrite so Gemini sees natural-language names
    if char_alias_map:
        raw_scene = resolve_char_tokens(raw_scene, char_alias_map)

    # Light style rewrite: only adds a style label, stays faithful to original content
    styled_scene = _rewrite_i2v_for_style(raw_scene, style)

    # Off-screen / voiceover speakers — should NOT be lip-synced to visible characters
    OFF_SCREEN_SPEAKERS = {"旁白", "背景音", "街边群众"}

    dialogue_note = ""
    if shot.dialogue:
        if dialogue_lang == "auto":
            lang = _detect_language([d.text for d in shot.dialogue])
        else:
            lang = dialogue_lang

        alias = char_alias_map or {}
        on_screen = [d for d in shot.dialogue if d.speaker_id not in OFF_SCREEN_SPEAKERS]
        off_screen = [d for d in shot.dialogue if d.speaker_id in OFF_SCREEN_SPEAKERS]

        parts = []
        if off_screen:
            if lang == "zh":
                lines = "；".join(f'{d.speaker_id}："{d.text}"' for d in off_screen)
                parts.append(f"画外音（off-screen voiceover ONLY, no visible character speaks）：{lines}。")
            else:
                lines = "; ".join(f'{d.speaker_id}: "{d.text}"' for d in off_screen)
                parts.append(f"Off-screen voiceover (NOT spoken by any visible character): {lines}.")

        if on_screen:
            if lang == "zh":
                lines = "；".join(
                    f'{alias.get(d.speaker_id, d.speaker_id)}说："{d.text}"'
                    for d in on_screen
                )
                parts.append(f"对话内容（必须严格使用中文原文发音）：{lines}。")
            else:
                lines = "; ".join(
                    f'{alias.get(d.speaker_id, d.speaker_id)} says: "{d.text}"'
                    for d in on_screen
                )
                parts.append(f"Spoken dialogue (preserve original language exactly): {lines}.")

        dialogue_note = " " + " ".join(parts)

    return f"{styled_scene}{dialogue_note}"
