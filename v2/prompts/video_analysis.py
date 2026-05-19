"""
Gemini prompt for video analysis — extracts structured shot data from a video.
"""

VIDEO_ANALYSIS_PROMPT = """You are a professional cinematographer and script analyst.

Analyze this video carefully. Extract a structured JSON description with:
1. Characters (visual appearance, NOT identity assumptions)
2. Up to {max_shots} shots (scene segments), in chronological order

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
        {{"speaker_id": "@character_01", "text": "Where am I?"}}
      ],
      "t2i_prompt": "Concise visual description of the FIRST FRAME of this shot, 2-3 sentences. Include @character_01 token.",
      "i2v_prompt": "Full cinematic description of the shot action/movement for video generation. Include character actions, camera movement, atmosphere. Include @character_01 token."
    }}
  ]
}}

RULES:
- Use @character_01, @character_02 etc. consistently across shots
- Shots must cover the ENTIRE video chronologically, no gaps
- Maximum {max_shots} shots total — merge short segments if needed
- t2i_prompt: describe the STATIC first frame (no motion words)
- i2v_prompt: describe the FULL action/motion of the shot
- dialogue: only if characters are clearly speaking; empty list if none
- Be precise about clothing details (colors, patterns, materials) for consistency
- Do NOT invent names — use "Character 1", "Character 2" if unknown
"""


def get_analysis_prompt(max_shots: int = 10) -> str:
    return VIDEO_ANALYSIS_PROMPT.format(max_shots=max_shots)
