"""
Step 1 — Video Analysis.

Two-phase approach for accurate shot segmentation:

Phase A: SceneDetect finds precise camera cut timestamps (frame-accurate).
         These are passed to Gemini as HARD constraints — Gemini cannot
         merge or split them, only describe each segment.

Phase B: Gemini receives the video + cut timestamps + Whisper transcript,
         and fills in per-shot metadata (scene_id, characters, descriptions,
         t2i/i2v prompts, dialogue).

This ensures every camera cut becomes its own shot — no silent merging.
"""
from __future__ import annotations
import os
import subprocess
from typing import List, Tuple, TYPE_CHECKING

from v2.clients.gemini_client import upload_video, analyze_video, extract_json
from v2.core.schema import Character, Dialogue, Shot, PipelineState
from v2.core.reference_store import ReferenceStore, ReferenceEntity
from v2.prompts.video_analysis import get_analysis_prompt

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_video_duration(video_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def _detect_cuts(video_path: str) -> List[Tuple[float, float]]:
    """
    Use SceneDetect to find camera cut boundaries.
    Returns list of (start_sec, end_sec) tuples covering the full video.
    Falls back to [(0, duration)] if SceneDetect is unavailable.
    """
    try:
        from scenedetect import detect, AdaptiveDetector
        scenes = detect(video_path, AdaptiveDetector())
        cuts = []
        for start, end in scenes:
            # .seconds is the non-deprecated property; fall back to get_seconds()
            s = start.seconds if hasattr(start, 'seconds') else start.get_seconds()
            e = end.seconds if hasattr(end, 'seconds') else end.get_seconds()
            cuts.append((round(s, 3), round(e, 3)))
        return cuts
    except ImportError:
        print("  ⚠️  scenedetect not installed — skipping cut detection")
        return []
    except Exception as e:
        print(f"  ⚠️  SceneDetect error: {e} — skipping cut detection")
        return []


def _parse_time_raw(value) -> float:
    if isinstance(value, str):
        v = value.strip().rstrip("s")
        if ":" in v:
            parts = v.split(":")
            return float(parts[0]) * 60 + float(parts[1])
        return float(v)
    return float(value)


def _mmss_to_seconds(value: float) -> float:
    minutes = int(value)
    seconds = round((value - minutes) * 100)
    return minutes * 60 + seconds


def _fix_timestamps(shots: list, video_duration: float) -> list:
    """Correct MM.SS float timestamps (e.g. 1.09 → 69s) when Gemini uses that format."""
    if not shots or video_duration <= 0:
        return shots
    max_end = max(s.end_time for s in shots)
    if max_end < video_duration * 0.5:
        print(f"  ⚠️  Timestamp format mismatch (max={max_end:.2f}s vs video={video_duration:.1f}s) "
              f"— correcting MM.SS → seconds")
        for s in shots:
            s.start_time = _mmss_to_seconds(s.start_time)
            s.end_time = _mmss_to_seconds(s.end_time)
            s.duration = round(s.end_time - s.start_time, 3)
            s.time_range = f"{s.start_time:.2f}s - {s.end_time:.2f}s"
    return shots


def _enforce_cut_timestamps(shots: list, cuts: List[Tuple[float, float]]) -> list:
    """
    After Gemini analysis, snap shot boundaries to the SceneDetect cuts.
    If Gemini merged or split cuts incorrectly, this rebuilds the shot list
    to match SceneDetect's boundaries exactly, reusing Gemini metadata
    (scene_id, descriptions, etc.) by matching each cut to the closest
    Gemini shot.
    """
    if not cuts or not shots:
        return shots

    # Only enforce as many cuts as we have Gemini shots — respect max_shots cap
    cuts = cuts[:len(shots)]

    # Build a map: for each SceneDetect cut, find the Gemini shot that
    # overlaps with its midpoint most
    from v2.core.schema import Dialogue as Dlg
    new_shots = []
    for idx, (start, end) in enumerate(cuts):
        mid = (start + end) / 2
        # Find the Gemini shot whose time range contains this midpoint
        matched = None
        for s in shots:
            if s.start_time <= mid <= s.end_time:
                matched = s
                break
        # If no exact match, use the closest shot
        if matched is None:
            matched = min(shots, key=lambda s: abs((s.start_time + s.end_time) / 2 - mid))

        new_shot = Shot(
            index=idx,
            scene_id=matched.scene_id,
            time_range=f"{start:.2f}s - {end:.2f}s",
            start_time=start,
            end_time=end,
            duration=round(end - start, 3),
            setting_description=matched.setting_description,
            environment_description=matched.environment_description,
            lighting_setup=matched.lighting_setup,
            color_grading=matched.color_grading,
            shot_size=matched.shot_size,
            camera_angle=matched.camera_angle,
            camera_movement=matched.camera_movement,
            focal_length=matched.focal_length,
            depth_of_field=matched.depth_of_field,
            mood_atmosphere=matched.mood_atmosphere,
            composition=matched.composition,
            subject_movement=matched.subject_movement,
            characters=matched.characters,
            dialogue=[
                d for d in matched.dialogue
                if start <= d.start_time <= end or start <= d.end_time <= end
            ],
            t2i_prompt=matched.t2i_prompt,
            i2v_prompt=matched.i2v_prompt,
        )
        new_shots.append(new_shot)

    return new_shots


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(state: PipelineState, transcript: str = "") -> PipelineState:
    print("\n" + "=" * 60)
    print("STEP 1 — Video Analysis")
    print("=" * 60)

    video_duration = _get_video_duration(state.video_path)
    print(f"  Video duration: {video_duration:.1f}s")

    # ── Phase A: SceneDetect — find precise cut points ────────────────────────
    print("  Running SceneDetect for camera cut detection...")
    cuts = _detect_cuts(state.video_path)
    if cuts:
        print(f"  SceneDetect found {len(cuts)} cut(s):")
        for i, (s, e) in enumerate(cuts):
            print(f"    Cut {i+1}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")
    else:
        print("  SceneDetect unavailable — Gemini will determine shot boundaries")

    if transcript and transcript != "(no speech detected)":
        print(f"  Using Whisper transcript ({len(transcript.splitlines())} lines)")

    # ── Phase B: Gemini — describe each segment ───────────────────────────────
    video_uri = upload_video(state.video_path)

    # If SceneDetect found cuts, cap max_shots to actual cut count
    # (no point asking Gemini for more shots than cuts)
    effective_max = min(state.max_shots, len(cuts)) if cuts else state.max_shots

    print(f"  Analyzing video with Gemini (max_shots={effective_max})...")
    prompt = get_analysis_prompt(
        max_shots=effective_max,
        transcript=transcript,
        cuts=cuts,
    )
    raw = analyze_video(video_uri, prompt)
    data = extract_json(raw)

    state.aspect_ratio = data.get("aspect_ratio", "16:9")

    # Build characters
    store = ReferenceStore()
    for idx, c in enumerate(data.get("characters", [])):
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

    # Build shots from Gemini response
    raw_shots = data.get("shots", [])[:effective_max]
    for s in raw_shots:
        start = _parse_time_raw(s.get("start_time", 0))
        end = _parse_time_raw(s.get("end_time", 0))
        shot = Shot(
            index=s.get("index", len(state.shots)),
            scene_id=s.get("scene_id", f"scene_{len(state.shots)+1:02d}"),
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

    # Fix MM.SS timestamps if needed
    state.shots = _fix_timestamps(state.shots, video_duration)

    # ── Phase C: Enforce SceneDetect boundaries ───────────────────────────────
    if cuts:
        before = len(state.shots)
        state.shots = _enforce_cut_timestamps(state.shots, cuts)
        after = len(state.shots)
        if before != after:
            print(f"  Shot count adjusted: {before} → {after} (aligned to SceneDetect cuts)")

    print(f"  Final: {len(state.shots)} shot(s):")
    for s in state.shots:
        print(f"    Shot {s.shot_id}: {s.time_range} ({s.duration:.1f}s)  scene={s.scene_id}")

    return state
