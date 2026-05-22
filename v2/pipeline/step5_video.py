"""
Step 5 — Video Generation.
dev_mode=True  → static 3-second clip per keyframe (fast, no API cost)
dev_mode=False → Veo 3 or Kling image-to-video (selected via state.video_model)
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

# Veo 3 concurrent workers (each call is ~60-120s, so 3 in parallel is safe)
VEO_MAX_WORKERS = 1  # Reduce to avoid 503 overload

_print_lock = threading.Lock()


def _safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


def _process_shot(shot, state, total):
    """Generate video for a single shot. Thread-safe."""
    video_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}_video.mp4")

    if os.path.exists(video_path):
        _safe_print(f"  ⏭️  Shot {shot.shot_id} — video exists, skipping")
        return shot.shot_id, video_path, "skipped"

    video_model = getattr(state, "video_model", "veo")
    mode_label = "STATIC" if state.dev_mode else video_model.upper()
    _safe_print(f"  🎬 [{shot.shot_id}/{total}] {mode_label}: Shot {shot.shot_id} ({shot.time_range})")

    final_prompt, _ = resolve_references(
        base_prompt=shot.i2v_prompt,
        reference_store=state.reference_store,
        previous_image_path=None,
        inject_consistency_instruction=False,
    )

    _safe_print(f"     Prompt ({len(final_prompt)}c): {final_prompt[:120]}...")

    if state.dev_mode:
        ok = generate_video_static_fallback(shot.keyframe_path, video_path, duration=3)
    elif video_model == "kling":
        from v2.clients.kling_client import generate_video_kling
        ok = generate_video_kling(
            image_path=shot.keyframe_path,
            prompt=final_prompt,
            output_path=video_path,
            duration=int(shot.duration),
            aspect_ratio=state.aspect_ratio,
        )
    else:  # veo (default)
        ok = _veo_generate(
            image_path=shot.keyframe_path,
            prompt=final_prompt,
            output_path=video_path,
            dev_mode=False,
            duration=int(shot.duration),
            aspect_ratio=state.aspect_ratio,
        )

    if ok:
        return shot.shot_id, video_path, "done"
    return shot.shot_id, None, "failed"


def run(state: "PipelineState") -> "PipelineState":
    mode_label = "STATIC FALLBACK (dev)" if state.dev_mode else "Veo 3"
    workers = 1 if state.dev_mode else VEO_MAX_WORKERS

    print("\n" + "=" * 60)
    print(f"STEP 5 — Video Generation [{mode_label}] (workers={workers})")
    print("=" * 60)

    # Separate shots that need processing from those already done
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

    if workers == 1:
        # Sequential: one shot at a time with inter-shot delay to avoid 503
        for shot in shots_to_process:
            shot_id, video_path, status = _process_shot(shot, state, total)
            if video_path:
                shot_map[shot_id].video_path = video_path
                shot_map[shot_id].status = "done"
            else:
                shot_map[shot_id].status = "failed"
                _safe_print(f"  ❌ Shot {shot_id} video failed")
            if shot != shots_to_process[-1]:
                import time
                _safe_print(f"  ⏳ Waiting 5s before next shot...")
                time.sleep(5)
    else:
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
                        _safe_print(f"  ❌ Shot {shot_id} video failed")
                except Exception as e:
                    _safe_print(f"  ❌ Shot {shot_id} thread error: {e}")
                    shot_map[shot_id].status = "failed"

    done = sum(1 for s in state.shots if s.video_path)
    failed = sum(1 for s in state.shots if s.status == "failed")
    print(f"\n  {done}/{len(state.shots)} videos generated ({failed} failed)")
    return state
