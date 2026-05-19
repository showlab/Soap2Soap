"""
Step 5 — Video Generation.
dev_mode=True  → static 3-second clip per keyframe (fast, no API cost)
dev_mode=False → Veo 3 image-to-video
"""
from __future__ import annotations
import os
from typing import TYPE_CHECKING

from v2.clients.veo_client import generate_video
from v2.core.reference_resolver import resolve_references

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


def run(state: "PipelineState") -> "PipelineState":
    mode_label = "STATIC FALLBACK (dev)" if state.dev_mode else "Veo 3"
    print("\n" + "=" * 60)
    print(f"STEP 5 — Video Generation [{mode_label}]")
    print("=" * 60)

    for shot in state.shots:
        if not shot.keyframe_path:
            print(f"  ⏭️  Shot {shot.shot_id} — no keyframe, skipping video")
            continue

        video_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}_video.mp4")

        if os.path.exists(video_path):
            print(f"  ⏭️  Shot {shot.shot_id} — video exists, skipping")
            shot.video_path = video_path
            continue

        print(f"  [{shot.shot_id}/{len(state.shots)}] {mode_label}: Shot {shot.shot_id}")

        # Resolve tokens in i2v prompt (text only — no images for Veo)
        final_prompt, _ = resolve_references(
            base_prompt=shot.i2v_prompt,
            reference_store=state.reference_store,
            previous_image_path=None,
            inject_consistency_instruction=False,
        )

        ok = generate_video(
            image_path=shot.keyframe_path,
            prompt=final_prompt,
            output_path=video_path,
            dev_mode=state.dev_mode,
            duration=max(3, int(shot.duration)),
            aspect_ratio=state.aspect_ratio,
        )

        if ok:
            shot.video_path = video_path
            shot.status = "done"
        else:
            print(f"  ❌ Shot {shot.shot_id} video failed")
            shot.status = "failed"

    done = sum(1 for s in state.shots if s.video_path)
    print(f"\n  {done}/{len(state.shots)} videos generated")
    return state
