"""
Step 1 — Video Analysis.
Uses Gemini to understand the input video and extract:
  - Characters (visual description)
  - Shots (scenes, camera, dialogue, t2i/i2v prompts)
  - Aspect ratio
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from v2.clients.gemini_client import upload_video, analyze_video, extract_json
from v2.core.schema import Character, Dialogue, Shot, PipelineState
from v2.core.reference_store import ReferenceStore, ReferenceEntity
from v2.prompts.video_analysis import get_analysis_prompt

if TYPE_CHECKING:
    pass


def run(state: PipelineState) -> PipelineState:
    print("\n" + "=" * 60)
    print("STEP 1 — Video Analysis")
    print("=" * 60)

    # 1. Upload video
    video_uri = upload_video(state.video_path)

    # 2. Send to Gemini for analysis
    print("  Analyzing video with Gemini...")
    prompt = get_analysis_prompt(max_shots=state.max_shots)
    raw = analyze_video(video_uri, prompt)

    # 3. Parse JSON
    data = extract_json(raw)

    # 4. Aspect ratio
    state.aspect_ratio = data.get("aspect_ratio", "16:9")

    # 5. Build characters + reference store
    store = ReferenceStore()
    raw_chars = data.get("characters", [])
    for idx, c in enumerate(raw_chars):
        char = Character(
            id=c["id"],
            name=c.get("name", f"Character {idx + 1}"),
            description=c.get("description", ""),
        )
        state.characters.append(char)
        store.add_entity(ReferenceEntity(
            entity_id=char.id,
            entity_type="character",
            description=char.description,
            image_index=idx + 1,
        ))

    state.reference_store = store
    print(f"  Found {len(state.characters)} character(s): "
          f"{', '.join(c.id for c in state.characters)}")

    # 6. Build shots
    raw_shots = data.get("shots", [])[:state.max_shots]
    for s in raw_shots:
        shot = Shot(
            index=s.get("index", len(state.shots)),
            scene_id=s.get("scene_id", f"shot_{len(state.shots)+1:02d}"),
            time_range=s.get("time_range", ""),
            start_time=float(s.get("start_time", 0)),
            end_time=float(s.get("end_time", 0)),
            duration=float(s.get("duration", 0)),
            setting_description=s.get("setting_description", ""),
            environment_description=s.get("environment_description", ""),
            lighting_setup=s.get("lighting_setup", ""),
            color_grading=s.get("color_grading", ""),
            shot_size=s.get("shot_size", ""),
            camera_angle=s.get("camera_angle", ""),
            camera_movement=s.get("camera_movement", ""),
            focal_length=s.get("focal_length", ""),
            depth_of_field=s.get("depth_of_field", ""),
            mood_atmosphere=s.get("mood_atmosphere", ""),
            composition=s.get("composition", ""),
            subject_movement=s.get("subject_movement", ""),
            characters=s.get("characters", []),
            dialogue=[
                Dialogue(speaker_id=d["speaker_id"], text=d["text"])
                for d in s.get("dialogue", [])
            ],
            t2i_prompt=s.get("t2i_prompt", ""),
            i2v_prompt=s.get("i2v_prompt", ""),
        )
        state.shots.append(shot)

    print(f"  Extracted {len(state.shots)} shot(s)")
    return state
