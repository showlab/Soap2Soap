"""
Step 3b — Camera Group Analysis (for camera_tree mode).

Uses Gemini to group shots by camera setup and establish a parent-child
hierarchy. The result is stored in state.camera_groups and used by
step4_keyframes in camera_tree mode.

Skipped in default and consistency modes.
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

from v2.clients.gemini_client import text_generate, extract_json
from v2.prompts.camera_grouping import get_camera_grouping_prompt

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


def run(state: "PipelineState") -> "PipelineState":
    if state.generation_mode != "camera_tree":
        return state

    print("\n" + "=" * 60)
    print("STEP 3b — Camera Group Analysis (Camera Tree Mode)")
    print("=" * 60)

    # Build compact shot summaries for the LLM
    shots_summary = [
        {
            "index": s.index,
            "shot_size": s.shot_size,
            "camera_angle": s.camera_angle,
            "camera_movement": s.camera_movement,
            "setting": s.setting_description[:80],
            "t2i_prompt": s.t2i_prompt[:120],
        }
        for s in state.shots
    ]

    prompt = get_camera_grouping_prompt(shots_summary)
    print(f"  Asking Gemini to group {len(state.shots)} shots by camera setup...")

    try:
        raw = text_generate(prompt)
        data = extract_json(raw)
        groups = data.get("groups", [])

        # Validate: every shot must appear in exactly one group
        indexed = {s.index for s in state.shots}
        covered = {idx for g in groups for idx in g.get("shot_indices", [])}
        missing = indexed - covered
        if missing:
            print(f"  ⚠️  {len(missing)} shot(s) not assigned to any group: {missing}")
            # Put uncovered shots in a catch-all group
            groups.append({
                "group_id": "group_fallback",
                "description": "Unassigned shots",
                "shot_indices": sorted(missing),
                "parent_group_ids": [],
            })

        state.camera_groups = groups
        print(f"  Identified {len(groups)} camera group(s):")
        for g in groups:
            parents = g.get("parent_group_ids", [])
            parent_str = f" (child of {parents})" if parents else " (root)"
            print(f"    {g['group_id']}: shots {g['shot_indices']}{parent_str}")

    except Exception as e:
        print(f"  ❌ Camera grouping failed: {e}")
        print(f"  ⚠️  Falling back to single-group (all shots in one group)")
        state.camera_groups = [{
            "group_id": "group_01",
            "description": "All shots",
            "shot_indices": [s.index for s in state.shots],
            "parent_group_ids": [],
        }]

    return state
