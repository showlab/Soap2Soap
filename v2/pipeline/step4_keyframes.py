"""
Step 4 — Keyframe Generation.

Supports three modes (set via state.generation_mode):

  default      — Each shot generated independently with character refs only.
  consistency  — Shots generated 4-at-a-time as a 2×2 grid, then cropped +
                 zoomed. Forces style / environment / character consistency within
                 each group of 4. (Ported from pai_v1_backend consistency mode)
  camera_tree  — Shots grouped by camera setup (from step3b). Each group's first
                 shot uses parent group anchor as reference; subsequent shots in
                 the group use the group's first shot. Uses FramePool DAG scheduling.
                 (Ported from pai_v1_backend camera_tree mode)
"""
from __future__ import annotations
import io
import os
from typing import List, Optional, TYPE_CHECKING

from PIL import Image

from v2.clients.imagen_client import generate_keyframe
from v2.clients.gemini_client import safety_rewrite
from v2.core.reference_resolver import resolve_references

if TYPE_CHECKING:
    from v2.core.schema import PipelineState, Shot


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _base_ref_images(shot: "Shot", state: "PipelineState") -> List[Image.Image]:
    """
    Load the design sheet (preferred) or individual character reference images
    for the characters that appear in this shot.
    """
    images = []

    # Use design sheet as primary reference (all characters in one image)
    if state.design_sheet_path and os.path.exists(state.design_sheet_path):
        try:
            images.append(Image.open(state.design_sheet_path))
            return images
        except Exception:
            pass

    # Fallback: individual character images
    for char_id in shot.characters:
        entity = state.reference_store.get_entity(char_id)
        if entity and entity.image_path and os.path.exists(entity.image_path):
            try:
                images.append(Image.open(entity.image_path))
            except Exception:
                pass
    return images


def _generate_one(
    shot: "Shot",
    state: "PipelineState",
    extra_refs: Optional[List[Image.Image]] = None,
    extra_ref_label: str = "Previous Shot",
    consistency_note: str = "",
) -> bool:
    """
    Generate a single keyframe. extra_refs appended after character refs.
    Returns True on success.
    """
    save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")

    base_refs = _base_ref_images(shot, state)
    all_refs = base_refs + (extra_refs or [])

    # Build legend for extra refs
    n_char = len(base_refs)
    legend_parts = []
    if base_refs:
        legend_parts.append("Image 1 (Design Sheet)" if state.design_sheet_path else
                            f"Images 1-{n_char} (Characters)")
    if extra_refs:
        for i, _ in enumerate(extra_refs):
            legend_parts.append(f"Image {n_char + i + 1} ({extra_ref_label})")

    legend = ("Reference images: " + ", ".join(legend_parts) + ". ") if legend_parts else ""
    full_prompt = f"{legend}{consistency_note}Shot Description: {shot.t2i_prompt}"

    img = generate_keyframe(
        prompt=full_prompt,
        reference_images=all_refs,
        save_path=save_path,
    )
    if img:
        shot.keyframe_path = save_path
        shot.status = "keyframe_done"
        return True
    shot.status = "failed"
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: Default
# ─────────────────────────────────────────────────────────────────────────────

def _run_default(state: "PipelineState") -> "PipelineState":
    print("  Mode: DEFAULT (each shot independent)")
    for shot in state.shots:
        save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")
        if os.path.exists(save_path):
            print(f"  ⏭️  Shot {shot.shot_id} exists")
            shot.keyframe_path = save_path
            shot.status = "keyframe_done"
            continue
        print(f"  [{shot.shot_id}/{len(state.shots)}] Shot {shot.shot_id} ({shot.time_range})")
        _generate_one(shot, state)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: Consistency (2×2 grid)
# ─────────────────────────────────────────────────────────────────────────────

def _crop_grid(grid_img: Image.Image) -> List[Image.Image]:
    """Crop 2×2 grid image into 4 cells (TL, TR, BL, BR)."""
    w, h = grid_img.size
    hw, hh = w // 2, h // 2
    boxes = [(0, 0, hw, hh), (hw, 0, w, hh), (0, hh, hw, h), (hw, hh, w, h)]
    return [grid_img.crop(b) for b in boxes]


def _zoom_and_select(
    grid_img: Image.Image,
    cell_img: Image.Image,
    shot: "Shot",
    position: str,
    state: "PipelineState",
    n_variations: int = 2,
) -> Optional[Image.Image]:
    """
    Generate zoom-in variations from a grid cell and pick the best one via Gemini.
    Falls back to the raw crop if all variations fail.
    """
    from v2.clients.gemini_client import _client, _GEMINI_VISION_MODEL_FLASH
    from v2.prompts.zoom_in import compile_zoom_in_prompt
    import json, re

    zoom_prompt = compile_zoom_in_prompt(position, shot.t2i_prompt)
    variations: List[Image.Image] = []

    for _ in range(n_variations):
        var = generate_keyframe(
            prompt=zoom_prompt,
            reference_images=[grid_img],
        )
        if var:
            variations.append(var)

    if not variations:
        return cell_img  # fallback to raw crop

    if len(variations) == 1:
        return variations[0]

    # Ask Gemini to pick the best variation
    try:
        select_prompt = (
            f"You are reviewing {len(variations)} zoom-in variations of a video frame.\n"
            f"Shot description: {shot.t2i_prompt[:200]}\n"
            f"Image 1 is the reference crop. Images 2-{len(variations)+1} are the variations.\n"
            f"Which variation best matches the shot description and maintains visual quality?\n"
            f"Return JSON: {{\"best_index\": 1}} (1-based index among the variations, not including the crop)"
        )
        all_imgs = [cell_img] + variations
        client = _client()
        resp = client.models.generate_content(
            model=_GEMINI_VISION_MODEL_FLASH,
            contents=all_imgs + [select_prompt],
        )
        data = json.loads(re.search(r"\{.*\}", resp.text or "{}", re.DOTALL).group())
        idx = int(data.get("best_index", 1)) - 1
        if 0 <= idx < len(variations):
            return variations[idx]
    except Exception:
        pass

    return variations[0]


def _run_consistency(state: "PipelineState") -> "PipelineState":
    from v2.prompts.narrative_grid import compile_narrative_grid_prompt
    from v2.prompts.zoom_in import GRID_POSITIONS

    print("  Mode: CONSISTENCY (2×2 grid → crop → zoom)")

    GRID_SIZE = 4
    shots = state.shots
    previous_grids: List[Image.Image] = []

    # Build character list for prompt
    char_list = [
        {"name": c.name, "description": c.description[:120]}
        for c in state.characters
    ]

    # Partition shots into groups of 4 (pad last group if needed)
    groups = []
    for i in range(0, len(shots), GRID_SIZE):
        group = shots[i:i + GRID_SIZE]
        groups.append(group)

    for g_idx, group in enumerate(groups):
        # Check if all 4 shots in this group already have keyframes
        save_paths = [os.path.join(state.output_dir, f"shot_{s.shot_id}.png") for s in group]
        if all(os.path.exists(p) for p in save_paths):
            print(f"  ⏭️  Grid {g_idx+1}: all shots exist, skipping")
            for s, p in zip(group, save_paths):
                s.keyframe_path = p
                s.status = "keyframe_done"
            try:
                previous_grids.append(Image.open(save_paths[0]))  # use first as prev ref
            except Exception:
                pass
            continue

        # Pad group to 4 shots if needed
        padded_group = list(group)
        while len(padded_group) < GRID_SIZE:
            padded_group.append(type('PaddedShot', (), {
                't2i_prompt': group[-1].t2i_prompt,
                'padded': True,
            })())

        # Build grid prompt
        shot_dicts = [
            {"t2i_prompt": s.t2i_prompt, "padded": getattr(s, 'padded', False)}
            for s in padded_group
        ]

        # Reference images: design sheet + previous grids
        ref_imgs: List[Image.Image] = []
        if state.design_sheet_path and os.path.exists(state.design_sheet_path):
            try:
                ref_imgs.append(Image.open(state.design_sheet_path))
            except Exception:
                pass
        ref_imgs.extend(previous_grids)

        grid_prompt = compile_narrative_grid_prompt(
            shots=shot_dicts,
            style=state.style,
            characters=char_list,
            previous_grids=len(previous_grids),
        )

        print(f"\n  Grid {g_idx+1}/{len(groups)}: generating 2×2 for shots "
              f"{[s.shot_id for s in group]}...")

        grid_img = generate_keyframe(
            prompt=grid_prompt,
            reference_images=ref_imgs,
            aspect_ratio="16:9",
        )

        if not grid_img:
            print(f"  ❌ Grid {g_idx+1} generation failed — falling back to default for these shots")
            for shot in group:
                _generate_one(shot, state,
                               extra_refs=previous_grids[-1:] if previous_grids else None,
                               extra_ref_label="Previous Grid")
            continue

        previous_grids.append(grid_img)

        # Save grid for reference
        grid_save = os.path.join(state.output_dir, f"grid_{g_idx+1}.png")
        grid_img.save(grid_save)

        # Crop + zoom each cell
        cells = _crop_grid(grid_img)
        for cell_idx, (shot, cell_img) in enumerate(zip(group, cells)):
            if getattr(shot, 'padded', False):
                continue

            save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")
            position = GRID_POSITIONS[cell_idx]

            print(f"    Shot {shot.shot_id} ({position}): zoom + select...")
            final_img = _zoom_and_select(
                grid_img=grid_img,
                cell_img=cell_img,
                shot=shot,
                position=position,
                state=state,
            )
            if final_img:
                final_img.save(save_path)
                shot.keyframe_path = save_path
                shot.status = "keyframe_done"
                print(f"    ✅ Shot {shot.shot_id} saved")
            else:
                # Last resort: save raw crop
                cell_img.save(save_path)
                shot.keyframe_path = save_path
                shot.status = "keyframe_done"
                print(f"    ⚠️  Shot {shot.shot_id}: used raw crop")

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3: Camera Tree
# ─────────────────────────────────────────────────────────────────────────────

def _run_camera_tree(state: "PipelineState") -> "PipelineState":
    from v2.core.frame_pool import FramePool, build_frame_dependency_graph

    print("  Mode: CAMERA TREE (DAG dependency scheduling)")

    if not state.camera_groups:
        print("  ⚠️  No camera groups found — falling back to default mode")
        return _run_default(state)

    shot_map = {s.index: s for s in state.shots}
    frames, group_first_shot = build_frame_dependency_graph(state.camera_groups)

    def generator(frame, ref_paths: List[str]) -> Optional[str]:
        shot = shot_map.get(frame.shot_idx)
        if not shot:
            return None

        save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")
        if os.path.exists(save_path):
            print(f"  ⏭️  Shot {shot.shot_id} exists")
            shot.keyframe_path = save_path
            shot.status = "keyframe_done"
            return save_path

        # Base: design sheet / character refs
        base_refs = _base_ref_images(shot, state)

        # Extra: previous-shot ref (from dependency)
        extra_refs = []
        for path in ref_paths:
            try:
                extra_refs.append(Image.open(path))
            except Exception:
                pass

        all_refs = base_refs + extra_refs
        n_base = len(base_refs)

        # Build legend
        parts = []
        if base_refs:
            parts.append("Image 1 (Design Sheet)" if state.design_sheet_path else
                         f"Images 1-{n_base} (Characters)")
        if extra_refs:
            label = "Group Anchor" if frame.depends_on_anchors else "Previous Shot"
            for i in range(len(extra_refs)):
                parts.append(f"Image {n_base + i + 1} ({label})")

        legend = ("Reference images: " + ", ".join(parts) + ". ") if parts else ""
        consistency = ""
        if extra_refs:
            consistency = (
                f"Maintain EXACT visual consistency with "
                f"Image {n_base + 1} "
                f"({'Group Anchor' if frame.depends_on_anchors else 'Previous Shot'}). "
            )
        if frame.group_desc:
            consistency += f"Camera viewpoint: {frame.group_desc}. "

        full_prompt = f"{legend}{consistency}Shot Description: {shot.t2i_prompt}"

        print(f"  [{shot.shot_id}/{len(state.shots)}] Shot {shot.shot_id} "
              f"(group={frame.group_id}, depth={frame.depth}, refs={len(all_refs)})")

        img = generate_keyframe(
            prompt=full_prompt,
            reference_images=all_refs,
            save_path=save_path,
        )
        if img:
            shot.keyframe_path = save_path
            shot.status = "keyframe_done"
            return save_path
        shot.status = "failed"
        return None

    pool = FramePool(frames, group_first_shot)
    results = pool.run(generator)

    stats = pool.stats
    print(f"\n  Camera Tree complete: {stats.get('done', 0)} done, "
          f"{stats.get('failed', 0)} failed")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print(f"STEP 4 — Keyframe Generation [{state.generation_mode.upper()}]")
    print("=" * 60)

    mode = state.generation_mode

    if mode == "camera_tree":
        state = _run_camera_tree(state)
    elif mode == "consistency":
        state = _run_consistency(state)
    else:
        state = _run_default(state)

    done = sum(1 for s in state.shots if s.keyframe_path)
    print(f"\n  {done}/{len(state.shots)} keyframes generated")
    return state
