"""
Step 1 — Video Analysis.

New per-shot architecture:
  Phase A: SceneDetect finds precise camera cut timestamps.
           Shots < 1s are filtered. If > 16 remain, user confirmation required.
  Phase B1: Upload full video → extract character list (consistent @character_XX IDs).
  Phase B2: For each cut, ffmpeg extracts a clip → upload → Gemini analyzes ONE shot.
           Each call returns detailed schema (t2i_prompt, i2v_prompt, environment, etc.)
  Phase C: Aggregate shots. Scene clustering is implicit via scene_id per shot.
"""
from __future__ import annotations
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, TYPE_CHECKING

from v2.clients.gemini_client import upload_video, analyze_video, text_generate, extract_json
from v2.core.schema import Character, Dialogue, Shot, PipelineState
from v2.core.reference_store import ReferenceStore, ReferenceEntity
from v2.prompts.video_analysis import (
    CHARACTER_EXTRACTION_PROMPT,
    get_shot_analysis_prompt,
    get_analysis_prompt,  # kept for fallback
)

if TYPE_CHECKING:
    pass

MAX_SHOTS_DEFAULT = 16
MIN_SHOT_DURATION = 1.0  # seconds — filter shots shorter than this


def _normalize_scene_ids(shots: list) -> list:
    """
    Normalize scene_id across all shots to fix zero-padding inconsistencies
    from separate Gemini calls (e.g. scene_001 vs scene_01 → both become scene_1).
    Also merges near-duplicate names (scene_ruins_01 / scene_ruins_1 → scene_ruins_1).
    """
    import re

    def canonical(sid: str) -> str:
        # Strip leading zeros from any numeric segment: scene_001 → scene_1
        return re.sub(r'(?<=[_\-])0+(\d)', r'\1', sid)

    mapping = {s.scene_id: canonical(s.scene_id) for s in shots}
    for shot in shots:
        shot.scene_id = mapping[shot.scene_id]
    return shots


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
    """SceneDetect → list of (start_sec, end_sec) tuples."""
    try:
        from scenedetect import detect, AdaptiveDetector
        scenes = detect(video_path, AdaptiveDetector())
        cuts = []
        for start, end in scenes:
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


def _ensure_720p(video_path: str) -> str:
    """
    Return a 720p version of the video for analysis.
    If the video is already ≤720p, returns the original path.
    Otherwise encodes a 720p copy next to the original and returns its path.
    """
    # Check current height
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "stream=height",
             "-of", "csv=p=0", video_path],
            capture_output=True, text=True,
        )
        height = int(result.stdout.strip().splitlines()[0])
    except Exception:
        return video_path

    if height <= 720:
        return video_path

    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_720p{ext}"
    if os.path.exists(out_path):
        print(f"  720p version exists: {out_path}")
        return out_path

    print(f"  Resizing {height}p → 720p for analysis: {out_path}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", "scale=-2:720",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "copy",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  720p saved: {out_path} ({size_mb:.1f} MB)")
        return out_path

    print(f"  ⚠️  Resize failed, using original")
    return video_path


def _extract_clip(video_path: str, start: float, end: float, out_path: str) -> bool:
    """Extract a video segment [start, end] using ffmpeg stream copy (fast)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


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


# ─────────────────────────────────────────────────────────────────────────────
# Phase B1 — Character extraction from full video
# ─────────────────────────────────────────────────────────────────────────────

def _extract_characters(video_uri: str) -> List[dict]:
    """Upload full video and extract character list with consistent IDs."""
    print("  Extracting characters from full video...")
    raw = analyze_video(video_uri, CHARACTER_EXTRACTION_PROMPT)
    try:
        data = extract_json(raw)
        chars = data.get("characters", [])
        print(f"  Found {len(chars)} character(s): {', '.join(c['id'] for c in chars)}")
        return chars
    except Exception as e:
        print(f"  ⚠️  Character extraction failed: {e} — continuing with empty character list")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Phase B2 — Per-shot analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_shot(
    clip_uri: str,
    index: int,
    total: int,
    start: float,
    end: float,
    characters: List[dict],
    transcript_lines: Optional[List[dict]] = None,
    max_retries: int = 3,
) -> Optional[dict]:
    """Analyze a single shot clip and return schema dict. Retries on empty response."""
    prompt = get_shot_analysis_prompt(
        index=index,
        total=total,
        start_time=start,
        end_time=end,
        characters=characters,
        transcript_lines=transcript_lines,
    )
    import time as _time
    for attempt in range(1, max_retries + 1):
        try:
            raw = analyze_video(clip_uri, prompt)
            data = extract_json(raw)
            return data
        except Exception as e:
            if attempt < max_retries:
                print(f"    ⚠️  Shot {index} attempt {attempt} failed: {e} — retrying...")
                _time.sleep(2)
            else:
                print(f"    ⚠️  Shot {index} analysis failed after {max_retries} attempts: {e}")
    return None


def _process_clip(
    shot_idx: int,
    total: int,
    start: float,
    end: float,
    video_path: str,
    clip_dir: str,
    characters: List[dict],
    transcript_lines: Optional[List[dict]],
) -> tuple:
    """
    Worker for parallel execution: extract clip → upload → analyze.
    Returns (shot_idx, start, end, shot_data).
    """
    clip_path = os.path.join(clip_dir, f"clip_{shot_idx:02d}.mp4")

    if os.path.exists(clip_path):
        print(f"  [{shot_idx}/{total}] Clip exists, skipping extraction → uploading...")
    else:
        ok = _extract_clip(video_path, start, end, clip_path)
        if not ok:
            print(f"    ⚠️  [{shot_idx}/{total}] Clip extraction failed")
            return shot_idx, start, end, None

    clip_uri = upload_video(clip_path)
    print(f"  [{shot_idx}/{total}] Uploaded {start:.2f}s-{end:.2f}s → analyzing...")
    shot_data = _analyze_shot(
        clip_uri=clip_uri,
        index=shot_idx,
        total=total,
        start=start,
        end=end,
        characters=characters,
        transcript_lines=transcript_lines,
    )
    t2i_len = len(shot_data.get("t2i_prompt", "")) if shot_data else 0
    print(f"    ✅ Shot {shot_idx}: scene={shot_data.get('scene_id','?') if shot_data else '?'}  t2i={t2i_len}c")
    return shot_idx, start, end, shot_data


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(
    state: PipelineState,
    transcript: str = "",
    transcript_lines: Optional[List[dict]] = None,
    yes: bool = False,
) -> PipelineState:
    print("\n" + "=" * 60)
    print("STEP 1 — Video Analysis")
    print("=" * 60)

    video_duration = _get_video_duration(state.video_path)
    print(f"  Video duration: {video_duration:.1f}s")

    # ── Phase A: SceneDetect ─────────────────────────────────────────────────
    print("  Running SceneDetect for camera cut detection...")
    cuts = _detect_cuts(state.video_path)

    if cuts:
        # Filter cuts shorter than MIN_SHOT_DURATION
        before_filter = len(cuts)
        cuts = [(s, e) for s, e in cuts if e - s >= MIN_SHOT_DURATION]
        filtered = before_filter - len(cuts)
        if filtered:
            print(f"  Filtered {filtered} cut(s) < {MIN_SHOT_DURATION}s")

        print(f"  SceneDetect: {len(cuts)} cut(s) after filtering:")
        for i, (s, e) in enumerate(cuts):
            print(f"    Cut {i+1}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")

        # Cap to max_shots
        if len(cuts) > state.max_shots:
            # If more than 16 and user hasn't confirmed, ask
            if len(cuts) > MAX_SHOTS_DEFAULT and not yes:
                print(f"\n  ⚠️  {len(cuts)} shots detected (> {MAX_SHOTS_DEFAULT}).")
                try:
                    answer = input(f"  Continue with all {len(cuts)} shots? [y/N]: ").strip().lower()
                    if answer != "y":
                        cuts = cuts[:MAX_SHOTS_DEFAULT]
                        print(f"  Capped to {MAX_SHOTS_DEFAULT} shots")
                except EOFError:
                    # Non-interactive: cap automatically
                    cuts = cuts[:state.max_shots]
                    print(f"  Non-interactive mode — capped to {state.max_shots} shots")
            else:
                cuts = cuts[:state.max_shots]
                print(f"  Capped to {state.max_shots} shots (max_shots limit)")
    else:
        print("  SceneDetect unavailable — will use single full-video analysis")

    # ── Prepare 720p version for analysis (faster upload, sufficient quality) ─
    analysis_video_path = _ensure_720p(state.video_path)

    # ── Phase B1: Character extraction from full video ───────────────────────
    print("\n  Uploading full video for character extraction...")
    full_video_uri = upload_video(analysis_video_path)

    characters = _extract_characters(full_video_uri)

    # Register characters in state
    store = ReferenceStore()
    for idx, c in enumerate(characters):
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

    # ── Phase B2: Per-shot clip extraction + analysis (parallel) ────────────
    if cuts:
        MAX_WORKERS = 5  # concurrent uploads + API calls
        print(f"\n  Analyzing {len(cuts)} shots in parallel (max {MAX_WORKERS} workers)...")

        # Clips stored next to the input video for reuse across pipeline runs
        video_base = os.path.splitext(os.path.basename(state.video_path))[0]
        video_dir = os.path.dirname(os.path.abspath(state.video_path))
        clip_dir = os.path.join(video_dir, f"{video_base}_clips")
        os.makedirs(clip_dir, exist_ok=True)
        print(f"  Clip cache: {clip_dir}")

        # Submit all shots concurrently (extract from 720p source)
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for i, (start, end) in enumerate(cuts):
                shot_idx = i + 1
                f = pool.submit(
                    _process_clip,
                    shot_idx, len(cuts), start, end,
                    analysis_video_path, clip_dir, characters, transcript_lines,
                )
                futures[f] = i

        # Collect results in original order
        results = {}
        for f in futures:
            try:
                shot_idx, start, end, shot_data = f.result()
                results[futures[f]] = (shot_idx, start, end, shot_data)
            except Exception as e:
                i = futures[f]
                print(f"  ⚠️  Shot {i+1} worker error: {e}")

        for i in sorted(results):
            shot_idx, start, end, shot_data = results[i]
            if shot_data is None:
                shot_data = {}

            shot = Shot(
                index=i,
                scene_id=shot_data.get("scene_id", f"scene_{shot_idx:02d}"),
                time_range=f"{start:.2f}s - {end:.2f}s",
                start_time=start,
                end_time=end,
                duration=round(end - start, 3),
                setting_description=shot_data.get("setting_description", ""),
                environment_description=shot_data.get("environment_description", ""),
                lighting_setup=shot_data.get("lighting_setup", ""),
                color_grading=shot_data.get("color_grading", ""),
                shot_size=shot_data.get("shot_size", ""),
                camera_angle=shot_data.get("camera_angle", ""),
                camera_movement=shot_data.get("camera_movement", ""),
                focal_length=shot_data.get("focal_length", ""),
                depth_of_field=shot_data.get("depth_of_field", ""),
                mood_atmosphere=shot_data.get("mood_atmosphere", ""),
                composition=shot_data.get("composition", ""),
                subject_movement=shot_data.get("subject_movement", ""),
                characters=shot_data.get("characters", []),
                dialogue=[
                    Dialogue(
                        speaker_id=d["speaker_id"],
                        text=d["text"],
                        start_time=float(d.get("start_time", 0)),
                        end_time=float(d.get("end_time", 0)),
                    )
                    for d in shot_data.get("dialogue", [])
                ],
                t2i_prompt=shot_data.get("t2i_prompt", ""),
                i2v_prompt=shot_data.get("i2v_prompt", ""),
            )
            state.shots.append(shot)
            print(f"    ✅ Shot {shot_idx}: scene={shot.scene_id}  t2i={len(shot.t2i_prompt)}c")

        # Normalize scene_ids across all shots (fix zero-padding inconsistencies)
        state.shots = _normalize_scene_ids(state.shots)

    else:
        # Fallback: analyze full video in one call (old behavior)
        print("\n  Falling back to full-video single-call analysis...")
        effective_max = state.max_shots
        prompt = get_analysis_prompt(max_shots=effective_max, transcript=transcript)
        from v2.clients.gemini_client import analyze_video as _av
        raw = _av(full_video_uri, prompt)
        data = extract_json(raw)
        state.aspect_ratio = data.get("aspect_ratio", "16:9")

        # Re-register characters if not done yet (fallback path gets characters from here)
        if not state.characters:
            store = ReferenceStore()
            for idx, c in enumerate(data.get("characters", [])):
                char = Character(id=c["id"], name=c.get("name", f"Character {idx+1}"),
                                 description=c.get("description", ""))
                state.characters.append(char)
                store.add_entity(ReferenceEntity(
                    entity_id=char.id, entity_type="character",
                    description=char.description, image_index=idx + 1,
                ))
            state.reference_store = store

        for s in data.get("shots", [])[:effective_max]:
            start = _parse_time_raw(s.get("start_time", 0))
            end = _parse_time_raw(s.get("end_time", 0))
            shot = Shot(
                index=s.get("index", len(state.shots)),
                scene_id=s.get("scene_id", f"scene_{len(state.shots)+1:02d}"),
                time_range=s.get("time_range", ""),
                start_time=start, end_time=end,
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
                    Dialogue(speaker_id=d["speaker_id"], text=d["text"],
                             start_time=float(d.get("start_time", 0)),
                             end_time=float(d.get("end_time", 0)))
                    for d in s.get("dialogue", [])
                ],
                t2i_prompt=s.get("t2i_prompt", ""),
                i2v_prompt=s.get("i2v_prompt", ""),
            )
            state.shots.append(shot)

        state.shots = _fix_timestamps(state.shots, video_duration)

    print(f"\n  Final: {len(state.shots)} shot(s):")
    for s in state.shots:
        print(f"    Shot {s.shot_id}: {s.time_range} ({s.duration:.1f}s)  scene={s.scene_id}")

    return state
