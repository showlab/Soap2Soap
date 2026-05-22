"""
Step 1b — Intelligent Shot Planning.

Takes all raw shots from Step 1 (one per SceneDetect cut) and asks Gemini
to re-plan them into a concise narrative shot list.

Rules enforced by the prompt:
- Every dialogue line is preserved in exactly one planned shot
- Shots with dialogue cannot be silently merged away
- Merging only between silent, same-scene, visually similar shots
- Narrative key beats (opening, conflict, climax, resolution) each get a shot
- Target: ~10 shots per minute of video
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

from v2.clients.gemini_client import text_generate, extract_json
from v2.core.schema import Dialogue, Shot
from v2.prompts.shot_planning import get_shot_planning_prompt
from v2.pipeline.step1_analyze import _normalize_scene_ids

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print("STEP 1b — Intelligent Shot Planning")
    print("=" * 60)

    if not state.shots:
        print("  ⚠️  No shots to plan — skipping")
        return state

    video_duration = state.shots[-1].end_time if state.shots else 60.0
    target = max(6, round(video_duration / 60 * 10))
    print(f"  {len(state.shots)} raw shots → target ~{target} planned shots "
          f"(video: {video_duration:.0f}s)")

    # Serialize shots for the prompt
    shots_data = []
    for s in state.shots:
        shots_data.append({
            "index": s.index,
            "time_range": s.time_range,
            "start_time": s.start_time,
            "end_time": s.end_time,
            "scene_id": s.scene_id,
            "characters": s.characters,
            "dialogue": [{"speaker_id": d.speaker_id, "text": d.text,
                          "start_time": d.start_time, "end_time": d.end_time}
                         for d in s.dialogue],
            "t2i_prompt": s.t2i_prompt,
            "i2v_prompt": s.i2v_prompt,
            "subject_movement": s.subject_movement,
            "setting_description": s.setting_description,
            "environment_description": s.environment_description,
            "lighting_setup": s.lighting_setup,
            "shot_size": s.shot_size,
            "camera_angle": s.camera_angle,
            "camera_movement": s.camera_movement,
            "mood_atmosphere": s.mood_atmosphere,
        })

    prompt = get_shot_planning_prompt(shots_data, video_duration)

    print(f"  Sending {len(state.shots)} shots to Gemini for planning...")
    try:
        raw = text_generate(prompt)
        data = extract_json(raw)
    except Exception as e:
        print(f"  ⚠️  Shot planning failed: {e} — keeping original shots")
        return state

    planned = data.get("planned_shots", [])
    if not planned:
        print("  ⚠️  Empty planned_shots — keeping original shots")
        return state

    # Build new Shot list from planned shots
    new_shots = []
    for p in planned:
        shot = Shot(
            index=p.get("index", len(new_shots)),
            scene_id=p.get("scene_id", "scene_01"),
            time_range=p.get("time_range", ""),
            start_time=float(p.get("start_time", 0)),
            end_time=float(p.get("end_time", 0)),
            duration=float(p.get("duration", 0)) or round(
                float(p.get("end_time", 0)) - float(p.get("start_time", 0)), 3),
            setting_description=p.get("setting_description", ""),
            environment_description=p.get("environment_description", ""),
            lighting_setup=p.get("lighting_setup", ""),
            shot_size=p.get("shot_size", ""),
            camera_angle=p.get("camera_angle", ""),
            camera_movement=p.get("camera_movement", ""),
            mood_atmosphere=p.get("mood_atmosphere", ""),
            characters=p.get("characters", []),
            dialogue=[
                Dialogue(
                    speaker_id=d["speaker_id"],
                    text=d["text"],
                    start_time=float(d.get("start_time", 0)),
                    end_time=float(d.get("end_time", 0)),
                )
                for d in p.get("dialogue", [])
            ],
            t2i_prompt=p.get("t2i_prompt", ""),
            i2v_prompt=p.get("i2v_prompt", ""),
        )
        src = p.get("source_shots", [])
        role = p.get("narrative_role", "")
        print(f"  Shot {shot.shot_id}: [{', '.join(str(x) for x in src)}] → "
              f"{shot.time_range}  {role[:60]}")
        new_shots.append(shot)

    new_shots = _normalize_scene_ids(new_shots)
    state.shots = new_shots
    print(f"\n  Planning complete: {len(state.shots)} shots "
          f"(was {len(shots_data)})")
    return state
