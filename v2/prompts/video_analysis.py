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


# ─────────────────────────────────────────────────────────────────────────────
# Per-shot analysis (new architecture: one clip per API call)
# ─────────────────────────────────────────────────────────────────────────────

CHARACTER_EXTRACTION_PROMPT = """You are analyzing a video to identify all distinct characters by visual appearance only.
Do NOT assume celebrity or fictional identity — use neutral names like "Character 1".

Return ONLY valid JSON, no markdown, no explanation:
{
  "characters": [
    {
      "id": "@character_01",
      "name": "Character 1",
      "description": "Age: 30s. Male. Light skin. Hair: dark brown, short. Face: square jaw, stubble. Build: muscular. Clothing: dark blue long robe with gold trim, red flowing cape attached at shoulders, leather belt. Accessories: gold circular amulet on chest. Distinctive: glowing hand effects."
    }
  ]
}

RULES:
- Use @character_01, @character_02, etc. as IDs
- Be VERY detailed on clothing: each item with color, material, cut
- Include all accessories and distinctive visual features
- NO identity assumptions — describe only what you see
"""

SHOT_ANALYSIS_PROMPT = """You are a professional cinematographer analyzing a single video shot.

SHOT INFO:
- Index: {index} of {total}
- Time range: {time_range}
- Duration: {duration:.2f}s

KNOWN CHARACTERS (use these IDs consistently):
{characters_json}

{transcript_section}

Analyze ONLY this shot and return detailed metadata.
Return ONLY valid JSON (one shot object), no markdown:
{{
  "scene_id": "scene_xxx",
  "setting_description": "one sentence describing the physical location",
  "environment_description": "detailed description of background, ground, sky, atmosphere, what fills the frame",
  "lighting_setup": "light direction, quality, color temperature, key/fill ratio",
  "color_grading": "color palette, saturation, contrast style",
  "shot_size": "Extreme close-up / Close-up / Medium close-up / Medium / Wide / Extreme wide",
  "camera_angle": "Eye-level / Low angle / High angle / Dutch angle",
  "camera_movement": "Static / Pan / Tilt / Dolly / Handheld / Push-in / Pull-out",
  "focal_length": "24mm / 35mm / 50mm / 85mm / 100mm etc.",
  "depth_of_field": "Deep / Shallow — describe what is in/out of focus",
  "mood_atmosphere": "2-3 emotional adjectives",
  "composition": "subject placement, framing elements, rule of thirds etc.",
  "subject_movement": "describe character actions using @character_XX tokens",
  "characters": ["@character_01"],
  "dialogue": [{{"speaker_id": "@character_01", "text": "exact words", "start_time": 0.0, "end_time": 1.0}}],
  "t2i_prompt": "@character_01 [pose/action in detail]. @character_01 wears [full clothing details]. Background: [environment]. [Shot size], [focal length], [lighting]. [Color grade].",
  "i2v_prompt": "@character_01 [motion description]. [Camera movement]. [Atmosphere]. Dialogue if any."
}}

RULES:
- Use @character_XX tokens for every character present (NEVER their names)
- t2i_prompt: minimum 4 sentences, full clothing + environment + camera + lighting
- i2v_prompt: full motion/action description, weave in dialogue naturally
- dialogue: only lines audible in THIS shot; copy text EXACTLY; include timestamps
- scene_id: use the same value for all shots in the same physical location
"""


def get_shot_analysis_prompt(
    index: int,
    total: int,
    start_time: float,
    end_time: float,
    characters: list,
    transcript_lines: list = None,
) -> str:
    import json as _json
    duration = end_time - start_time
    time_range = f"{start_time:.2f}s - {end_time:.2f}s"

    # Build characters JSON (id + description only)
    char_list = [{"id": c["id"], "name": c.get("name", ""), "description": c.get("description", "")} for c in characters]
    characters_json = _json.dumps(char_list, ensure_ascii=False, indent=2)

    # Build transcript section for this shot's time window
    if transcript_lines:
        relevant = [
            ln for ln in transcript_lines
            if ln.get("end", 0) >= start_time and ln.get("start", 0) <= end_time
        ]
        if relevant:
            lines_text = "\n".join(f"  [{ln['start']:.1f}s-{ln['end']:.1f}s] {ln['text']}" for ln in relevant)
            transcript_section = f"AUDIO TRANSCRIPT for this shot:\n{lines_text}"
        else:
            transcript_section = "(No dialogue in this shot)"
    else:
        transcript_section = "(No transcript available)"

    return SHOT_ANALYSIS_PROMPT.format(
        index=index,
        total=total,
        time_range=time_range,
        duration=duration,
        characters_json=characters_json,
        transcript_section=transcript_section,
    )


CHUNK_ANALYSIS_PROMPT = """You are a professional cinematographer and film editor analyzing a video segment.

SEGMENT INFO:
- This clip covers {start_time:.1f}s – {end_time:.1f}s of the original video (duration: {duration:.1f}s)
- Time offset: add {start_time:.1f}s to all timestamps you report

KNOWN CHARACTERS (use these IDs consistently):
{characters_json}

{transcript_section}

YOUR TASK:
Analyze this segment and select approximately {target_shots} KEY SHOTS that best represent the story.
Do NOT output one shot per camera cut — select narrative-significant moments only.

SELECTION RULES:
1. Every line of dialogue must appear in at least one shot — never drop dialogue
2. Shots with dialogue are MANDATORY — include all of them
3. Merge visually similar, same-scene, silent shots into one
4. Identify scene changes, character introductions, emotional beats
5. MAX GAP RULE (CRITICAL): The time gap between the end of one shot and the start of the next must NEVER exceed 10 seconds. If a gap would be larger, insert an additional shot to cover that period — even if the content is visually similar to adjacent shots
6. FULL COVERAGE: Your selected shots must collectively cover the entire segment from {start_time:.1f}s to {end_time:.1f}s with no gap larger than 10s

CRITICAL: scene_id RULES
- scene_id = the PHYSICAL LOCATION, shared across shots in the same place
- Same location → same scene_id (e.g. "scene_market_stall")
- Use short snake_case

Return ONLY valid JSON, no markdown:
{{
  "shots": [
    {{
      "index": 0,
      "scene_id": "scene_market",
      "time_range": "{start_time:.2f}s - {shot_end:.2f}s",
      "start_time": {start_time:.2f},
      "end_time": {shot_end:.2f},
      "duration": {duration:.2f},
      "narrative_role": "Opening — buyer enters market",
      "setting_description": "Outdoor market street with fruit stalls",
      "environment_description": "Busy street, colorful stalls, sunlight",
      "lighting_setup": "Natural daylight, warm tones",
      "color_grading": "Warm, saturated",
      "shot_size": "Wide",
      "camera_angle": "Eye-level",
      "camera_movement": "Static",
      "focal_length": "35mm",
      "depth_of_field": "Deep",
      "mood_atmosphere": "Lively, everyday",
      "composition": "Subject center, market fills background",
      "subject_movement": "@character_01 walks toward the stall",
      "characters": ["@character_01"],
      "dialogue": [{{"speaker_id": "@character_01", "text": "这瓜多少钱", "start_time": {start_time:.2f}, "end_time": {start_time_plus:.2f}}}],
      "t2i_prompt": "@character_01 [pose]. Background: [environment]. [Shot size], [lens], [lighting].",
      "i2v_prompt": "@character_01 [motion]. [Camera]. @character_01 says: '这瓜多少钱'."
    }}
  ]
}}

RULES:
- start_time and end_time must be ABSOLUTE timestamps (include the {start_time:.1f}s offset)
- t2i_prompt: min 3-4 sentences, use @character_XX tokens, include clothing + environment + camera
- i2v_prompt: full motion description, weave in all dialogue naturally
- dialogue: copy text EXACTLY, include absolute timestamps
"""


def get_chunk_analysis_prompt(
    start_time: float,
    end_time: float,
    characters: list,
    transcript_lines: list = None,
    target_shots: int = 10,
) -> str:
    import json as _json
    duration = end_time - start_time

    char_list = [{"id": c["id"], "name": c.get("name", ""), "description": c.get("description", "")} for c in characters]
    characters_json = _json.dumps(char_list, ensure_ascii=False, indent=2)

    if transcript_lines:
        relevant = [
            ln for ln in transcript_lines
            if ln.get("end", 0) >= start_time and ln.get("start", 0) <= end_time
        ]
        if relevant:
            lines_text = "\n".join(f"  [{ln['start']:.1f}s-{ln['end']:.1f}s] {ln['text']}" for ln in relevant)
            transcript_section = f"AUDIO TRANSCRIPT for this segment:\n{lines_text}"
        else:
            transcript_section = "(No dialogue in this segment)"
    else:
        transcript_section = "(No transcript available)"

    # Dummy values for the JSON example placeholders
    shot_end = start_time + min(duration, 10.0)
    start_time_plus = start_time + 2.0

    return CHUNK_ANALYSIS_PROMPT.format(
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        characters_json=characters_json,
        transcript_section=transcript_section,
        target_shots=target_shots,
        shot_end=shot_end,
        start_time_plus=start_time_plus,
    )


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
