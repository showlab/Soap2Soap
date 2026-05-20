"""
Gemini prompt for video analysis.

Key design:
- scene_id is a SHARED location identifier across shots in the same environment
- t2i_prompt must be a RICH image-generation prompt using @character_XX tokens
- When SceneDetect cuts are provided, Gemini must describe each cut exactly as given
"""

VIDEO_ANALYSIS_PROMPT = """You are a professional cinematographer and script analyst.

Analyze this video carefully and extract a structured JSON with:
1. Characters (visual appearance only, NO identity assumptions)
2. Scenes (distinct physical locations / environments)
3. Shots as specified below

{transcript_section}

══════════════════════════════════════════════════════════
CRITICAL: scene_id RULES
══════════════════════════════════════════════════════════
scene_id = the PHYSICAL LOCATION / ENVIRONMENT, shared across shots in the same place.
• Same location → same scene_id (e.g. all battlefield shots → "scene_battlefield")
• New location → new scene_id
• Use short snake_case: "scene_ship_deck", "scene_interior_cabin", etc.

══════════════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
══════════════════════════════════════════════════════════

{{
  "aspect_ratio": "16:9",
  "characters": [
    {{
      "id": "@character_01",
      "name": "Character 1",
      "description": "Name: Character 1. Age: 30s. Male. Light skin. Hair: dark brown, short. Face: square jaw, stubble. Build: muscular. Clothing: dark blue long robe with gold trim, red flowing cape attached at shoulders, leather belt. Accessories: gold circular amulet on chest. Distinctive: glowing hand effects.",
      "scenes": ["scene_battlefield"]
    }}
  ],
  "shots": [
    {{
      "index": 0,
      "scene_id": "scene_battlefield",
      "time_range": "0.00s - 4.50s",
      "start_time": 0.0,
      "end_time": 4.5,
      "duration": 4.5,
      "setting_description": "Destroyed city battlefield, crumbled buildings, thick dust clouds",
      "environment_description": "Rubble-strewn ground with broken concrete, orange dust fills the air, distant fires burning, sky is dark smoky orange, low visibility",
      "lighting_setup": "Harsh overhead sunlight filtered through smoke, strong orange tint, deep shadows under debris",
      "color_grading": "Desaturated with heavy orange-brown push, high contrast shadows",
      "shot_size": "Medium shot",
      "camera_angle": "Eye-level",
      "camera_movement": "Slow push-in",
      "focal_length": "85mm",
      "depth_of_field": "Shallow, subject sharp, background blurred",
      "mood_atmosphere": "Tense, desperate, apocalyptic",
      "composition": "Subject center-left, rule of thirds, rubble framing right",
      "subject_movement": "@character_01 raises both hands as glowing orange energy circles form around them",
      "characters": ["@character_01"],
      "dialogue": [
        {{"speaker_id": "@character_01", "text": "I can't stop him alone.", "start_time": 2.1, "end_time": 3.4}}
      ],
      "t2i_prompt": "@character_01 (Name from Image 1) stands center-left in a destroyed city battlefield. @character_01 wears a dark blue long robe with gold trim and a flowing red cape. Both hands raised, glowing orange energy circles forming around them. Background: crumbled concrete buildings, thick orange dust clouds. Medium shot, 85mm, shallow depth of field, eye-level. Orange-tinted desaturated color grade.",
      "i2v_prompt": "@character_01 (Name from Image 1) slowly raises both hands as glowing orange energy circles expand around them. Slow push-in camera. Orange dust billows in background. Tense, desperate atmosphere. @character_01 says: 'I can't stop him alone.'"
    }}
  ]
}}

══════════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════════
- scene_id: SHARED by shots in the same physical location
- characters: Use @character_01, @character_02 etc. CONSISTENTLY across ALL shots
- character description: Be VERY detailed on clothing (each item: color, material, cut), accessories, distinctive features — this is used for visual consistency
- t2i_prompt REQUIREMENTS (CRITICAL):
  * Must use @character_XX tokens for EVERY character present (NOT their names)
  * Write it as a complete standalone image generation prompt
  * Include: character(s) with tokens, what they are wearing, their pose/action, environment details, camera setup, lighting
  * Minimum 3-4 sentences
  * Example pattern: "@character_01 [description of pose/action]. @character_01 wears [clothing details]. Background: [environment]. [Shot size], [lens], [lighting]."
- i2v_prompt: Full action/motion description. Use @character_XX tokens. Weave in dialogue naturally if present.
- dialogue: ONLY transcript lines within this shot's time range; copy EXACTLY; include start_time and end_time
- Names: do NOT invent — use "Character 1", "Character 2" etc. if unknown
"""

TRANSCRIPT_SECTION_TEMPLATE = """AUDIO TRANSCRIPT (Whisper — use for accurate dialogue):
--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---
Instructions: Match dialogue lines to shots by timestamp. Copy text EXACTLY. Identify speaker from on-screen presence.
"""

NO_TRANSCRIPT_SECTION = "(No audio transcript available)"

CUTS_SECTION_TEMPLATE = """
══════════════════════════════════════════════════════════
CAMERA CUTS — HARD CONSTRAINTS (detected by SceneDetect)
══════════════════════════════════════════════════════════
You MUST output EXACTLY {n_cuts} shots. Each cut below = exactly one shot.

{cut_lines}

MANDATORY RULES:
• Do NOT merge any two cuts — every cut is its own shot, even if < 1 second
• Do NOT split any cut into multiple shots
• Use the start_time and end_time EXACTLY as given (do not round or adjust)
• Output shot count must equal {n_cuts}
"""

NO_CUTS_SECTION = ""


def get_analysis_prompt(
    max_shots: int = 10,
    transcript: str = "",
    cuts: list = None,
) -> str:
    transcript_section = (
        TRANSCRIPT_SECTION_TEMPLATE.format(transcript=transcript)
        if transcript and transcript != "(no speech detected)"
        else NO_TRANSCRIPT_SECTION
    )

    # Build shot count instruction
    if cuts:
        shot_count_instruction = f"You must output EXACTLY {len(cuts)} shots — one per SceneDetect cut listed below."
        cuts_lines = "\n".join(
            f"  Shot {i+1}: start={s:.3f}s  end={e:.3f}s  duration={e-s:.2f}s"
            for i, (s, e) in enumerate(cuts)
        )
        cuts_section = CUTS_SECTION_TEMPLATE.format(
            n_cuts=len(cuts),
            cut_lines=cuts_lines,
        )
    else:
        shot_count_instruction = f"Up to {max_shots} shots, in chronological order, covering the full video."
        cuts_section = NO_CUTS_SECTION

    prompt = VIDEO_ANALYSIS_PROMPT.format(
        transcript_section=transcript_section,
    )

    # Replace the shot count placeholder
    prompt = prompt.replace(
        "3. Shots as specified below",
        f"3. {shot_count_instruction}"
    )

    return prompt + cuts_section
