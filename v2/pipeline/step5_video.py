"""
Step 5 — Video Generation.
dev_mode=True  → static 3-second clip per keyframe (fast, no API cost)
dev_mode=False → Seed Dance 2.0 (default) or Veo 3 image-to-video
Concurrent: I2V_WORKERS workers, up to I2V_MAX_RETRIES retries on failure.
"""
from __future__ import annotations
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from v2.clients.veo_client import generate_video as _veo_generate
from v2.clients.veo_client import generate_video_static_fallback
from v2.core.reference_resolver import resolve_references

if TYPE_CHECKING:
    from v2.core.schema import PipelineState

I2V_WORKERS = 5
I2V_MAX_RETRIES = 2

_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _call_model(video_model, shot, state, video_path, final_prompt) -> bool:
    """Single model call (no retry logic here)."""
    if state.dev_mode:
        return generate_video_static_fallback(shot.keyframe_path, video_path, duration=3)
    elif video_model == "seeddance":
        from v2.clients.seeddance_client import generate_video_seeddance
        return generate_video_seeddance(
            image_path=shot.keyframe_path,
            prompt=final_prompt,
            output_path=video_path,
            duration=int(shot.duration),
            aspect_ratio=state.aspect_ratio,
        )
    else:  # veo
        return _veo_generate(
            image_path=shot.keyframe_path,
            prompt=final_prompt,
            output_path=video_path,
            dev_mode=False,
            duration=int(shot.duration),
            aspect_ratio=state.aspect_ratio,
        )


def _process_shot(shot, state, total):
    """Generate video for a single shot with retries. Thread-safe."""
    video_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}_video.mp4")

    if os.path.exists(video_path):
        _safe_print(f"  ⏭️  Shot {shot.shot_id} — video exists, skipping")
        return shot.shot_id, video_path, "skipped"

    video_model = getattr(state, "video_model", "seeddance")
    mode_label = "STATIC" if state.dev_mode else video_model.upper()
    _safe_print(f"  🎬 [{shot.shot_id}/{total}] {mode_label}: Shot {shot.shot_id} ({shot.time_range})")

    final_prompt, _ = resolve_references(
        base_prompt=shot.i2v_prompt,
        reference_store=state.reference_store,
        previous_image_path=None,
        inject_consistency_instruction=False,
    )
    _safe_print(f"     Prompt ({len(final_prompt)}c): {final_prompt[:120]}...")

    max_attempts = 1 if state.dev_mode else (1 + I2V_MAX_RETRIES)
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            _safe_print(f"     Retry {attempt - 1}/{I2V_MAX_RETRIES} for Shot {shot.shot_id}...")
        ok = _call_model(video_model, shot, state, video_path, final_prompt)
        if ok:
            return shot.shot_id, video_path, "done"

    _safe_print(f"  ❌ Shot {shot.shot_id} failed after {max_attempts} attempt(s)")
    return shot.shot_id, None, "failed"


def run(state: "PipelineState") -> "PipelineState":
    video_model = getattr(state, "video_model", "seeddance")
    mode_label = "STATIC FALLBACK (dev)" if state.dev_mode else video_model.upper()
    workers = 1 if state.dev_mode else I2V_WORKERS

    print("\n" + "=" * 60)
    print(f"STEP 5 — Video Generation [{mode_label}] (workers={workers}, retries={I2V_MAX_RETRIES})")
    print("=" * 60)

    shots_to_process = []
    for shot in state.shots:
        if not shot.keyframe_path:
            print(f"  ⏭️  Shot {shot.shot_id} — no keyframe, skipping video")
            continue
        video_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}_video.mp4")
        if os.path.exists(video_path):
            shot.video_path = video_path
            shot.status = "done"
        else:
            shots_to_process.append(shot)

    already_done = sum(1 for s in state.shots if s.video_path)
    print(f"  {already_done} already done, {len(shots_to_process)} to generate")

    if not shots_to_process:
        print(f"  All videos already exist.")
        return state

    total = len(state.shots)
    shot_map = {s.shot_id: s for s in state.shots}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_shot, shot, state, total): shot.shot_id
            for shot in shots_to_process
        }
        for future in as_completed(futures):
            shot_id = futures[future]
            try:
                shot_id, video_path, status = future.result()
                shot = shot_map[shot_id]
                if video_path:
                    shot.video_path = video_path
                    shot.status = "done"
                else:
                    shot.status = "failed"
            except Exception as e:
                _safe_print(f"  ❌ Shot {shot_id} thread error: {e}")
                shot_map[shot_id].status = "failed"

    done = sum(1 for s in state.shots if s.video_path)
    failed = sum(1 for s in state.shots if s.status == "failed")
    print(f"\n  {done}/{len(state.shots)} videos generated ({failed} failed)")
    return state
