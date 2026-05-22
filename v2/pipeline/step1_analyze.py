"""
Step 1 — Video Analysis (sliding-window approach).

Architecture:
  Phase A: Resize input to 720p for fast upload.
  Phase B1: Upload full video → extract character list (consistent @character_XX IDs).
  Phase B2: Split video into ~60s chunks → upload + analyze each chunk in parallel.
            Each chunk Gemini call returns ~10 narrative shots (not one per camera cut).
  Phase C: Merge all chunk results, fix time offsets, normalize scene_ids.
"""
from __future__ import annotations
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, TYPE_CHECKING

from v2.clients.gemini_client import upload_video, analyze_video, extract_json
from v2.core.schema import Character, Dialogue, Shot, PipelineState
from v2.core.reference_store import ReferenceStore, ReferenceEntity
from v2.prompts.video_analysis import (
    CHARACTER_EXTRACTION_PROMPT,
    get_chunk_analysis_prompt,
)

if TYPE_CHECKING:
    pass

CHUNK_DURATION = 60.0     # seconds per analysis window
MAX_WORKERS = 3            # parallel chunk uploads + analysis
SHOTS_PER_MINUTE = 10      # target shots per minute of video


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


def _ensure_720p(video_path: str) -> str:
    """Return a 720p version for analysis; encode if source is higher res."""
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
        "-c:a", "copy", out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  720p saved: {out_path} ({size_mb:.1f} MB)")
        return out_path
    print(f"  ⚠️  Resize failed, using original")
    return video_path


def _extract_chunk(video_path: str, start: float, end: float, out_path: str) -> bool:
    """Extract a video chunk, re-encoding for keyframe alignment."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and os.path.exists(out_path)


def _normalize_scene_ids(shots: list) -> list:
    """Strip leading zeros from numeric scene_id segments."""
    import re
    def canonical(sid: str) -> str:
        return re.sub(r'(?<=[_\-])0+(\d)', r'\1', sid)
    mapping = {s.scene_id: canonical(s.scene_id) for s in shots}
    for shot in shots:
        shot.scene_id = mapping[shot.scene_id]
    return shots


def _fix_timestamps(shots: list, video_duration: float) -> list:
    if not shots or video_duration <= 0:
        return shots
    max_end = max(s.end_time for s in shots)
    if max_end < video_duration * 0.5:
        print(f"  ⚠️  Timestamp mismatch — correcting MM.SS → seconds")
        for s in shots:
            def mm(v):
                m = int(v); return m * 60 + round((v - m) * 100, 3)
            s.start_time = mm(s.start_time)
            s.end_time = mm(s.end_time)
            s.duration = round(s.end_time - s.start_time, 3)
            s.time_range = f"{s.start_time:.2f}s - {s.end_time:.2f}s"
    return shots


# ─────────────────────────────────────────────────────────────────────────────
# Phase B1 — Character extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_characters(video_uri: str) -> List[dict]:
    print("  Extracting characters from full video...")
    raw = analyze_video(video_uri, CHARACTER_EXTRACTION_PROMPT)
    try:
        data = extract_json(raw)
        chars = data.get("characters", [])
        print(f"  Found {len(chars)} character(s): {', '.join(c['id'] for c in chars)}")
        return chars
    except Exception as e:
        print(f"  ⚠️  Character extraction failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Phase B2 — Sliding window chunk analysis
# ─────────────────────────────────────────────────────────────────────────────

def _analyze_chunk(
    chunk_idx: int,
    total_chunks: int,
    start: float,
    end: float,
    video_path: str,
    chunk_dir: str,
    characters: List[dict],
    transcript_lines: Optional[List[dict]],
    max_retries: int = 3,
) -> Tuple[int, float, float, List[dict]]:
    """Extract, upload and analyze one ~60s chunk. Returns (chunk_idx, start, end, shots)."""
    import time as _time

    chunk_path = os.path.join(chunk_dir, f"chunk_{chunk_idx:02d}.mp4")

    if os.path.exists(chunk_path):
        print(f"  [chunk {chunk_idx}/{total_chunks}] Exists, re-uploading...")
    else:
        ok = _extract_chunk(video_path, start, end, chunk_path)
        if not ok:
            print(f"  ⚠️  Chunk {chunk_idx} extraction failed")
            return chunk_idx, start, end, []

    duration = end - start
    target = max(3, round(duration / 60 * SHOTS_PER_MINUTE))
    prompt = get_chunk_analysis_prompt(
        start_time=start,
        end_time=end,
        characters=characters,
        transcript_lines=transcript_lines,
        target_shots=target,
    )

    clip_uri = upload_video(chunk_path)
    print(f"  [chunk {chunk_idx}/{total_chunks}] Uploaded {start:.1f}s-{end:.1f}s → analyzing (target {target} shots)...")

    for attempt in range(1, max_retries + 1):
        try:
            raw = analyze_video(clip_uri, prompt)
            data = extract_json(raw)
            shots = data.get("shots", [])
            print(f"  [chunk {chunk_idx}/{total_chunks}] ✅ {len(shots)} shots")
            return chunk_idx, start, end, shots
        except Exception as e:
            if attempt < max_retries:
                print(f"  [chunk {chunk_idx}/{total_chunks}] attempt {attempt} failed: {e} — retrying...")
                _time.sleep(3)
            else:
                print(f"  [chunk {chunk_idx}/{total_chunks}] ⚠️  failed after {max_retries} attempts: {e}")
    return chunk_idx, start, end, []


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
    print("STEP 1 — Video Analysis (sliding window)")
    print("=" * 60)

    video_duration = _get_video_duration(state.video_path)
    print(f"  Video duration: {video_duration:.1f}s")

    # ── Prepare 720p version ─────────────────────────────────────────────────
    analysis_video_path = _ensure_720p(state.video_path)

    # ── Phase B1: Character extraction ───────────────────────────────────────
    print("\n  Uploading full video for character extraction...")
    full_video_uri = upload_video(analysis_video_path)
    characters = _extract_characters(full_video_uri)

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

    # ── Phase B2: Sliding window chunk analysis ───────────────────────────────
    # Build chunk boundaries
    chunks = []
    t = 0.0
    while t < video_duration:
        end = min(t + CHUNK_DURATION, video_duration)
        chunks.append((t, end))
        t = end

    n_chunks = len(chunks)
    total_target = max(6, round(video_duration / 60 * SHOTS_PER_MINUTE))
    print(f"\n  Splitting into {n_chunks} chunk(s) of ~{CHUNK_DURATION:.0f}s")
    print(f"  Target: ~{total_target} shots total, {MAX_WORKERS} parallel workers")

    chunk_dir = os.path.join(
        os.path.dirname(os.path.abspath(state.video_path)),
        os.path.splitext(os.path.basename(state.video_path))[0] + "_chunks"
    )
    os.makedirs(chunk_dir, exist_ok=True)
    print(f"  Chunk cache: {chunk_dir}")

    # Submit all chunks in parallel
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, (start, end) in enumerate(chunks):
            f = pool.submit(
                _analyze_chunk,
                i + 1, n_chunks, start, end,
                analysis_video_path, chunk_dir, characters, transcript_lines,
            )
            futures[f] = i

    # Collect results in order
    chunk_results = {}
    for f in futures:
        try:
            chunk_idx, start, end, shots = f.result()
            chunk_results[futures[f]] = (chunk_idx, start, end, shots)
        except Exception as e:
            print(f"  ⚠️  Chunk worker error: {e}")

    # ── Phase C: Merge and build Shot list ────────────────────────────────────
    shot_index = 0
    for i in sorted(chunk_results):
        _, chunk_start, chunk_end, raw_shots = chunk_results[i]

        for s in raw_shots:
            # Parse times — Gemini should output absolute times but clamp just in case
            start = float(s.get("start_time", chunk_start))
            end = float(s.get("end_time", chunk_end))

            # If Gemini returned relative times (0-based), shift by chunk offset
            if end <= CHUNK_DURATION and chunk_start > 0:
                start += chunk_start
                end += chunk_start

            start = max(chunk_start, min(start, chunk_end))
            end = max(start + 0.1, min(end, chunk_end))

            shot = Shot(
                index=shot_index,
                scene_id=s.get("scene_id", f"scene_{shot_index+1:02d}"),
                time_range=f"{start:.2f}s - {end:.2f}s",
                start_time=round(start, 3),
                end_time=round(end, 3),
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
                        start_time=float(d.get("start_time", start)),
                        end_time=float(d.get("end_time", end)),
                    )
                    for d in s.get("dialogue", [])
                ],
                t2i_prompt=s.get("t2i_prompt", ""),
                i2v_prompt=s.get("i2v_prompt", ""),
            )
            state.shots.append(shot)
            shot_index += 1

    state.shots = _normalize_scene_ids(state.shots)

    print(f"\n  Final: {len(state.shots)} shot(s):")
    for s in state.shots:
        dlg = f" [{len(s.dialogue)} lines]" if s.dialogue else ""
        print(f"    Shot {s.shot_id}: {s.time_range} ({s.duration:.1f}s)  "
              f"scene={s.scene_id}{dlg}")

    return state
