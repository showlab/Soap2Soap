"""
Shot planning prompt — re-plans raw SceneDetect shots into a concise narrative shot list.
"""

SHOT_PLANNING_PROMPT = """You are a film editor and story analyst. You have analyzed a {duration:.0f}-second video and produced {n_shots} raw shots from SceneDetect cuts.

Your task is to RE-PLAN these into approximately {target_shots} key shots that best represent the full story.

══════════════════════════════════════════════════════════
RAW SHOTS (from per-clip Gemini analysis):
══════════════════════════════════════════════════════════
{shots_json}

══════════════════════════════════════════════════════════
STRICT RULES
══════════════════════════════════════════════════════════

DIALOGUE RULES (highest priority):
1. Every line of dialogue MUST appear in exactly one planned shot's i2v_prompt and dialogue list — no line may be dropped
2. A shot that contains dialogue CANNOT be silently merged away — it must become the "anchor" shot that absorbs surrounding silent shots
3. If multiple consecutive shots all have dialogue, keep them as separate planned shots (do NOT merge shots with different dialogue)
4. The dialogue field of each planned shot must include ALL lines from its source shots, in chronological order

NARRATIVE RULES:
5. Identify the story's key narrative beats: opening, conflict introduction, escalation, climax, resolution — each must have at least one dedicated planned shot
6. Character first appearances must have a dedicated shot
7. Merging is ONLY allowed between shots that are: (a) in the same scene, (b) have no dialogue, (c) have similar visual content

OUTPUT RULES:
8. Target shot count: {target_shots} (±3 is acceptable; never go below {min_shots})
9. Each planned shot must cover a continuous time range (start_time to end_time)
10. source_shots must list all original shot indices that are merged into this planned shot
11. narrative_role must explain WHY this shot is kept (its story function)
12. t2i_prompt: write a rich standalone image generation prompt describing the KEY FRAME of this shot
13. i2v_prompt: describe the full action/motion across the entire merged time range; naturally weave in ALL dialogue

══════════════════════════════════════════════════════════
Return ONLY valid JSON, no markdown:
══════════════════════════════════════════════════════════
{{
  "planned_shots": [
    {{
      "index": 0,
      "source_shots": [1, 2, 3],
      "narrative_role": "Opening — establishes the market setting and introduces the buyer",
      "scene_id": "scene_market",
      "time_range": "0.00s - 11.40s",
      "start_time": 0.0,
      "end_time": 11.40,
      "duration": 11.40,
      "characters": ["@character_01"],
      "dialogue": [
        {{"speaker_id": "@character_01", "text": "有一个人前来买瓜", "start_time": 0.0, "end_time": 2.6}}
      ],
      "setting_description": "Outdoor market street with fruit stalls",
      "environment_description": "Busy street market, colorful fruit stands, sunlight",
      "lighting_setup": "Natural daylight, warm tones",
      "shot_size": "Wide",
      "camera_angle": "Eye-level",
      "camera_movement": "Static",
      "mood_atmosphere": "Lively, everyday",
      "t2i_prompt": "@character_01 walks into a busy outdoor market. Background: colorful fruit stands lining the street. Wide shot, natural daylight.",
      "i2v_prompt": "@character_01 enters the market street and approaches a watermelon stall. @character_01 says: '有一个人前来买瓜'. Camera static, lively street atmosphere."
    }}
  ]
}}
"""


def get_shot_planning_prompt(
    shots: list,
    video_duration: float,
) -> str:
    import json as _json

    n_shots = len(shots)
    # Target: ~10 shots per minute, min 6
    target_shots = max(6, round(video_duration / 60 * 10))
    min_shots = max(4, target_shots - 3)

    # Build compact shot summary for the prompt
    compact = []
    for s in shots:
        compact.append({
            "index": s.get("index", 0) + 1,
            "time_range": s.get("time_range", ""),
            "scene_id": s.get("scene_id", ""),
            "characters": s.get("characters", []),
            "dialogue": s.get("dialogue", []),
            "t2i_prompt": s.get("t2i_prompt", "")[:200],
            "i2v_prompt": s.get("i2v_prompt", "")[:200],
            "subject_movement": s.get("subject_movement", ""),
        })

    shots_json = _json.dumps(compact, ensure_ascii=False, indent=2)

    return SHOT_PLANNING_PROMPT.format(
        duration=video_duration,
        n_shots=n_shots,
        target_shots=target_shots,
        min_shots=min_shots,
        shots_json=shots_json,
    )
