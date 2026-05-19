"""
Step 4 — Keyframe Generation (Consistency Mode).
Uses Gemini image generation with:
  - Character reference images (from ReferenceStore)
  - Previous shot keyframe (consistency mode — ported from pai_v1_backend)
  - Style-compiled t2i prompt

Each shot N sees: [character refs...] + [previous keyframe] as reference images.
"""
from __future__ import annotations
import os
from typing import Optional, TYPE_CHECKING

from v2.clients.imagen_client import generate_keyframe
from v2.core.reference_resolver import resolve_references

if TYPE_CHECKING:
    from v2.core.schema import PipelineState, Shot


def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print("STEP 4 — Keyframe Generation (Consistency Mode)")
    print("=" * 60)

    previous_keyframe: Optional[str] = None

    for shot in state.shots:
        save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")

        if os.path.exists(save_path):
            print(f"  ⏭️  Shot {shot.shot_id} — keyframe exists, skipping")
            shot.keyframe_path = save_path
            previous_keyframe = save_path
            continue

        print(f"\n  [{shot.shot_id}/{len(state.shots)}] Generating keyframe for Shot {shot.shot_id}"
              f" ({shot.time_range})")

        # Resolve @tokens → descriptions + collect reference PIL images
        final_prompt, ref_images = resolve_references(
            base_prompt=shot.t2i_prompt,
            reference_store=state.reference_store,
            previous_image_path=previous_keyframe,
            inject_consistency_instruction=previous_keyframe is not None,
        )

        print(f"     prompt: {len(final_prompt)} chars  |  ref images: {len(ref_images)}")

        img = generate_keyframe(
            prompt=final_prompt,
            reference_images=ref_images,
            aspect_ratio=state.aspect_ratio.replace(":", "/"),  # "16:9" → "16/9"
            save_path=save_path,
        )

        if img:
            shot.keyframe_path = save_path
            shot.status = "keyframe_done"
            previous_keyframe = save_path
        else:
            print(f"  ❌ Shot {shot.shot_id} keyframe failed — skipping")
            shot.status = "failed"
            # Don't update previous_keyframe so next shot still references the last good one

    done = sum(1 for s in state.shots if s.keyframe_path)
    print(f"\n  {done}/{len(state.shots)} keyframes generated")
    return state
