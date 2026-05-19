"""
Step 1 — Video Analysis.
Uses Gemini to understand the input video and extract:
  - Characters (visual description)
  - Shots (scenes, camera, dialogue, t2i/i2v prompts)
  - Aspect ratio
"""
from __future__ import annotations
import os
import subprocess
from typing import TYPE_CHECKING

from v2.clients.gemini_client import upload_video, analyze_video, extract_json
from v2.core.schema import Character, Dialogue, Shot, PipelineState
from v2.core.reference_store import ReferenceStore, ReferenceEntity
from v2.prompts.video_analysis import get_analysis_prompt

if TYPE_CHECKING:
    pass


def _get_video_duration(video_path: str) -> float:
    """Get video duration in seconds via ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _parse_time_raw(value) -> float:
    """Parse a raw time value (string or float) to seconds, without MM:SS correction."""
    if isinstance(value, str):
        v = value.strip().rstrip("s")
        if ":" in v:
            parts = v.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(v)
    return float(value)


def _mmss_to_seconds(value: float) -> float:
    """Convert M.SS float (e.g. 1.09 = 1m09s) to pure seconds (69)."""
    minutes = int(value)
    seconds = round((value - minutes) * 100)
    return minutes * 60 + seconds


def _fix_timestamps(shots: list, video_duration: float) -> list:
    """
    Detect if Gemini returned timestamps in MM.SS float format instead of pure seconds,
    and correct them.

    Detection: if max(end_time) < video_duration * 0.5, times are probably MM.SS encoded.
    """
    if not shots or video_duration <= 0:
        return shots

    max_end = max(s.end_time for s in shots)
    if max_end < video_duration * 0.5:
        print(f"  ⚠️  Time format mismatch detected: max end={max_end:.2f}s but video={video_duration:.1f}s")
        print(f"       Correcting MM.SS → seconds (e.g. 1.09 → 69s)")
        for s in shots:
            s.start_time = _mmss_to_seconds(s.start_time)
            s.end_time = _mmss_to_seconds(s.end_time)
            s.duration = round(s.end_time - s.start_time, 3)
            s.time_range = f"{s.start_time:.2f}s - {s.end_time:.2f}s"

    return shots


def run(state: PipelineState, transcript: str = "") -> PipelineState:
    print("\n" + "=" * 60)
    print("STEP 1 — Video Analysis")
    print("=" * 60)

    # Get actual video duration for timestamp validation
    video_duration = _get_video_duration(state.video_path)
    print(f"  Video duration: {video_duration:.1f}s")

    if transcript and transcript != "(no speech detected)":
        print(f"  Using Whisper transcript ({len(transcript.splitlines())} lines)")
    else:
        print("  No transcript — Gemini will infer dialogue from visuals")

    # 1. Upload video
    video_uri = upload_video(state.video_path)

    # 2. Send to Gemini for analysis (with transcript injected)
    print("  Analyzing video with Gemini...")
    prompt = get_analysis_prompt(max_shots=state.max_shots, transcript=transcript)
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
        start = _parse_time_raw(s.get("start_time", 0))
        end = _parse_time_raw(s.get("end_time", 0))
        shot = Shot(
            index=s.get("index", len(state.shots)),
            scene_id=s.get("scene_id", f"shot_{len(state.shots)+1:02d}"),
            time_range=s.get("time_range", ""),
            start_time=start,
            end_time=end,
            duration=round(end - start, 3),
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
                Dialogue(
                    speaker_id=d["speaker_id"],
                    text=d["text"],
                    start_time=float(d.get("start_time", 0)),
                    end_time=float(d.get("end_time", 0)),
                )
                for d in s.get("dialogue", [])
            ],
            t2i_prompt=s.get("t2i_prompt", ""),
            i2v_prompt=s.get("i2v_prompt", ""),
        )
        state.shots.append(shot)

    # 7. Auto-correct MM.SS timestamp format if needed
    state.shots = _fix_timestamps(state.shots, video_duration)

    print(f"  Extracted {len(state.shots)} shot(s):")
    for s in state.shots:
        print(f"    Shot {s.shot_id}: {s.time_range} ({s.duration:.1f}s)")

    return state
