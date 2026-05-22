"""
Camera grouping prompts — used by step3b to group shots into camera setups.
Ported from pai_v1_backend/app/model/prompts/camera_grouping.py
"""

GROUP_SHOTS_PROMPT = """You are an expert Director of Photography. Group the following shots into Camera Setups.

GROUPING RULES:
1. Group shots filmed from the SAME physical camera location and viewing angle.
2. Different focal lengths (wide vs close-up) CAN be in the same group if camera stays put.
3. Reverse angles = new group.
4. Camera tracking/dolly to new position = new group.
5. Small pans/tilts = same group.

DESCRIPTION (per group): Describe the camera viewpoint in detail:
- Specific location (e.g. "ship's lower deck stairway area, facing the crowd")
- Perspective & angle (e.g. "eye-level from the doorway")
- Viewing direction (e.g. "looking over @character_01's shoulder toward @character_02")
- Relative subject positions
- Key lighting from this angle

OUTPUT: Return ONLY valid JSON:
{{
  "groups": [
    {{
      "group_id": "group_01",
      "description": "...",
      "shot_indices": [0, 1, 2],
      "parent_group_ids": []
    }},
    {{
      "group_id": "group_02",
      "description": "...",
      "shot_indices": [3, 4],
      "parent_group_ids": ["group_01"]
    }}
  ]
}}

HIERARCHY RULES for parent_group_ids:
- Root groups (establishing shots, wide shots) have empty parent_group_ids.
- Close-up groups that are cut-ins of a wider shot should list the wider group as parent.
- Reverse angle groups are siblings (no parent relationship unless clearly a cut-in).
- Keep hierarchy shallow (max 2 levels).

SHOTS TO GROUP:
{shots_json}
"""


def get_camera_grouping_prompt(shots_summary: list) -> str:
    import json
    return GROUP_SHOTS_PROMPT.format(shots_json=json.dumps(shots_summary, indent=2))
