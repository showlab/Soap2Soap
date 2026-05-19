"""
Gemini prompt for video analysis.

Key design: scene_id is a SHARED identifier for shots that take place in
the same physical location/environment. Multiple shots can share the same
scene_id. This is used later to group shots into the same consistency grid.

Accepts an optional Whisper transcript to improve dialogue accuracy.
"""

VIDEO_ANALYSIS_PROMPT = """You are a professional cinematographer and script analyst.

Analyze this video carefully and extract a structured JSON with:
1. Characters (visual appearance only, NO identity assumptions)
2. Scenes (distinct physical locations / environments)
3. Up to {max_shots} shots, in chronological order

{transcript_section}

══════════════════════════════════════════════════════════
CRITICAL: scene_id RULES
══════════════════════════════════════════════════════════
scene_id represents the PHYSICAL LOCATION / ENVIRONMENT — NOT the shot number.

Rules:
• Assign the SAME scene_id to ALL shots filmed in the same place.
  Example: if shots 1, 3, 5 are all on the ship's lower deck → all get scene_id "scene_ship_lower_deck"
• Assign a DIFFERENT scene_id when the location clearly changes.
  Example: shots 2, 4 cut to the ship's bridge → scene_id "scene_ship_bridge"
• scene_id must be a short snake_case label describing the location:
  "scene_ship_deck", "scene_interior_cabin", "scene_dining_room", etc.
• If uncertain whether a cut is a new location or just a new camera angle of
  the same place, use the SAME scene_id (same environment = same scene).

This is essential: shots with the same scene_id will be generated together
in the same consistency grid to ensure visual coherence.

══════════════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:
══════════════════════════════════════════════════════════

{{
  "aspect_ratio": "16:9",
  "characters": [
    {{
      "id": "@character_01",
      "name": "Character 1",
      "description": "Name: Character 1. [Age range]. [Gender]. [Skin tone]. [Hair: color, length, style]. [Face features]. [Build]. [Clothing: each item with color, fabric, cut in detail]. [Accessories]. [Distinctive features].",
      "scenes": ["scene_ship_lower_deck", "scene_ship_bridge"]
    }}
  ],
  "shots": [
    {{
      "index": 0,
      "scene_id": "scene_ship_lower_deck",
      "time_range": "0.00s - 4.50s",
      "start_time": 0.0,
      "end_time": 4.5,
      "duration": 4.5,
      "setting_description": "Interior lower deck of a large ship, wooden stairs, benches, passengers seated",
      "environment_description": "Detailed environment: materials, colors, lighting sources, spatial layout, atmosphere",
      "lighting_setup": "Strong backlight from stairway, soft fill from portholes",
      "color_grading": "Slightly desaturated, warm highlights, cool shadows",
      "shot_size": "Medium shot",
      "camera_angle": "Eye-level",
      "camera_movement": "Smooth tracking right",
      "focal_length": "35mm",
      "depth_of_field": "Deep focus",
      "mood_atmosphere": "Curious, slightly tense, class contrast",
      "composition": "Rule of thirds, subject left entering from right",
      "subject_movement": "@character_01 descends stairs and walks into room",
      "characters": ["@character_01"],
      "dialogue": [
        {{"speaker_id": "@character_01", "text": "Where am I?", "start_time": 2.1, "end_time": 3.0}}
      ],
      "t2i_prompt": "Concise visual description of the STATIC first frame, 2-3 sentences. Include @character_01 token.",
      "i2v_prompt": "Full cinematic description of the complete shot action/motion. Include character actions, camera movement, atmosphere. If dialogue: weave it in naturally (e.g. '@character_01 turns and says ...'). Include @character_01 token."
    }}
  ]
}}

══════════════════════════════════════════════════════════
RULES
══════════════════════════════════════════════════════════
- scene_id: SHARED by shots in the same physical location (see rules above)
- shots: cover the ENTIRE video chronologically with no time gaps
- Maximum {max_shots} shots total — merge very short cuts if needed
- characters: @character_01, @character_02 etc. used consistently across all shots
- t2i_prompt: STATIC first frame description (no motion words)
- i2v_prompt: FULL action/motion description for video generation
- dialogue: ONLY lines from the transcript within this shot's time range; copy EXACTLY; empty list if none
- Dialogue entries MUST include start_time and end_time from the transcript
- Clothing: be precise (colors, patterns, materials) — critical for visual consistency
- Names: do NOT invent — use "Character 1", "Character 2" if unknown
"""

TRANSCRIPT_SECTION_TEMPLATE = """AUDIO TRANSCRIPT (Whisper speech recognition — use for accurate dialogue):
--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Instructions:
1. Fill the dialogue field for each shot using ONLY transcript lines within that shot's time range
2. Enrich i2v_prompt with what characters are saying in that shot
3. Identify speaker from what you see (lip movement, on-screen presence)
"""

NO_TRANSCRIPT_SECTION = "(No audio transcript — infer dialogue from lip movement if possible)"

CUTS_SECTION_TEMPLATE = """CAMERA CUTS (detected by SceneDetect — these are HARD boundaries):
{cut_lines}

CRITICAL RULES when SceneDetect cuts are provided:
• Each cut above MUST become exactly ONE shot in your output.
• Do NOT merge two cuts into one shot, even if the scene looks similar.
• Do NOT split one cut into multiple shots.
• Use the provided start_time and end_time EXACTLY as given — do not adjust them.
• Total shots in your output must equal {n_cuts}.
"""

NO_CUTS_SECTION = "(No cut detection data — determine shot boundaries from visual analysis)"


def get_analysis_prompt(
    max_shots: int = 10,
    transcript: str = "",
    cuts: list = None,
) -> str:
    # Transcript section
    if transcript and transcript != "(no speech detected)":
        transcript_section = TRANSCRIPT_SECTION_TEMPLATE.format(transcript=transcript)
    else:
        transcript_section = NO_TRANSCRIPT_SECTION

    # Cuts section
    if cuts:
        cut_lines = "\n".join(
            f"  Shot {i+1}: start={s:.3f}s  end={e:.3f}s  duration={e-s:.2f}s"
            for i, (s, e) in enumerate(cuts)
        )
        cuts_section = CUTS_SECTION_TEMPLATE.format(
            cut_lines=cut_lines,
            n_cuts=len(cuts),
        )
    else:
        cuts_section = NO_CUTS_SECTION

    # Replace max_shots instruction when cuts are provided
    if cuts:
        max_shots_note = (
            f"You must output EXACTLY {len(cuts)} shots — one per SceneDetect cut above. "
            f"Do not output more or fewer."
        )
    else:
        max_shots_note = f"Up to {max_shots} shots (scene segments), in chronological order"

    prompt = VIDEO_ANALYSIS_PROMPT.replace(
        "2. Up to {max_shots} shots (scene segments), in chronological order",
        f"2. {max_shots_note}"
    )

    return prompt.format(
        max_shots=max_shots,
        transcript_section=transcript_section,
    ) + f"\n\n{cuts_section}"
