"""
Gemini prompt for video analysis — extracts structured shot data from a video.
Accepts an optional Whisper transcript to improve dialogue accuracy and
enrich shot descriptions / i2v prompts with speech context.
"""

VIDEO_ANALYSIS_PROMPT = """You are a professional cinematographer and script analyst.

Analyze this video carefully. Extract a structured JSON description with:
1. Characters (visual appearance, NOT identity assumptions)
2. Up to {max_shots} shots (scene segments), in chronological order

{transcript_section}

OUTPUT FORMAT — return ONLY valid JSON, no markdown, no explanation:

{{
  "aspect_ratio": "16:9",
  "characters": [
    {{
      "id": "@character_01",
      "name": "Character 1",
      "description": "Name: Character 1. [Age range]. [Gender]. [Ethnicity/skin tone]. [Hair: color, length, style]. [Face: shape, notable features]. [Body: build, height impression]. [Clothing in this video: item, color, fabric, cut]. [Accessories]. [Distinctive features].",
      "scenes": ["shot_01", "shot_02"]
    }}
  ],
  "shots": [
    {{
      "index": 0,
      "scene_id": "shot_01",
      "time_range": "0.00s - 4.50s",
      "start_time": 0.0,
      "end_time": 4.5,
      "duration": 4.5,
      "setting_description": "Interior lower deck of a large ship, wooden stairs, benches",
      "environment_description": "Detailed scene environment: materials, colors, lighting sources, spatial layout",
      "lighting_setup": "Strong backlight from stairway, soft fill from portholes",
      "color_grading": "Slightly desaturated, warm highlights, cool shadows",
      "shot_size": "Medium shot",
      "camera_angle": "Eye-level",
      "camera_movement": "Smooth tracking right",
      "focal_length": "35mm",
      "depth_of_field": "Deep focus",
      "mood_atmosphere": "Curious, slightly tense, class contrast",
      "composition": "Rule of thirds, subject on left entering from right",
      "subject_movement": "@character_01 descends stairs and walks into room",
      "characters": ["@character_01"],
      "dialogue": [
        {{"speaker_id": "@character_01", "text": "Where am I?", "start_time": 2.1, "end_time": 3.0}}
      ],
      "t2i_prompt": "Concise visual description of the FIRST FRAME of this shot, 2-3 sentences. Include @character_01 token.",
      "i2v_prompt": "Full cinematic description of the shot action/movement for video generation. Include character actions, camera movement, atmosphere, and any spoken dialogue naturally woven in. Include @character_01 token."
    }}
  ]
}}

RULES:
- Use @character_01, @character_02 etc. consistently across shots
- Shots must cover the ENTIRE video chronologically, no gaps
- Maximum {max_shots} shots total — merge short segments if needed
- t2i_prompt: describe the STATIC first frame (no motion words)
- i2v_prompt: describe the FULL action/motion of the shot — if speech occurs in this shot, incorporate the spoken lines naturally into the motion description (e.g. "@character_01 turns to @character_02 and says 'Hello again'")
- dialogue: use ONLY lines from the transcript that fall within this shot's time range; copy the text EXACTLY as transcribed; empty list if none
- dialogue entries must include start_time and end_time from the transcript
- Be precise about clothing details (colors, patterns, materials) for consistency
- Do NOT invent names — use "Character 1", "Character 2" if unknown
"""

TRANSCRIPT_SECTION_TEMPLATE = """AUDIO TRANSCRIPT (from Whisper speech recognition, use this for accurate dialogue):
--- TRANSCRIPT START ---
{transcript}
--- TRANSCRIPT END ---

Use the transcript to:
1. Fill in the dialogue field for each shot (match by timestamp to the shot's time range)
2. Enrich the i2v_prompt with what characters are saying in that shot
3. Identify who is speaking based on what you see on screen
"""

NO_TRANSCRIPT_SECTION = """(No audio transcript available — infer dialogue from lip movement if possible)"""


def get_analysis_prompt(max_shots: int = 10, transcript: str = "") -> str:
    if transcript and transcript != "(no speech detected)":
        transcript_section = TRANSCRIPT_SECTION_TEMPLATE.format(transcript=transcript)
    else:
        transcript_section = NO_TRANSCRIPT_SECTION
    return VIDEO_ANALYSIS_PROMPT.format(
        max_shots=max_shots,
        transcript_section=transcript_section,
    )
