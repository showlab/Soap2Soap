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
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, TYPE_CHECKING

KF_WORKERS = 10  # concurrent keyframe generation / refinement threads

from PIL import Image

from v2.clients.imagen_client import generate_keyframe as _gemini_keyframe
from v2.clients.imagen_client import generate_keyframe_with_model as generate_keyframe_model


def _kf_generate(state, prompt, reference_images, aspect_ratio="16:9", save_path=None):
    """Route keyframe generation to the model selected in state.keyframe_model."""
    model = getattr(state, "keyframe_model", "gemini")
    return generate_keyframe_model(model, prompt, reference_images, aspect_ratio, save_path)


# Keep generate_keyframe as alias for callers that don't have state
generate_keyframe = _gemini_keyframe
from v2.clients.gemini_client import safety_rewrite, text_generate
from v2.core.reference_resolver import resolve_references


def _compress_grid_prompt(prompt: str, limit: int) -> str:
    """Use Gemini to compress a grid prompt to within `limit` characters."""
    compress_instruction = (
        f"The following is an image generation prompt for a 2×2 grid of cinematic frames. "
        f"It is too long. Compress it to under {limit} characters while preserving: "
        f"(1) the 4-panel grid structure, "
        f"(2) each panel's core scene description and character actions, "
        f"(3) the art style and consistency instructions. "
        f"Remove redundant wording and verbose phrasing. Output ONLY the compressed prompt, no explanation.\n\n"
        f"{prompt}"
    )
    try:
        compressed = text_generate(compress_instruction)
        # Hard truncate as safety net
        return compressed[:limit] if len(compressed) > limit else compressed
    except Exception as e:
        print(f"  ⚠️  Gemini compression failed ({e}) — hard truncating")
        return prompt[:limit]

if TYPE_CHECKING:
    from v2.core.schema import PipelineState, Shot


# ─────────────────────────────────────────────────────────────────────────────
# Shared helper
# ─────────────────────────────────────────────────────────────────────────────

def _base_ref_images(shot: "Shot", state: "PipelineState") -> List[Image.Image]:
    """
    Load individual character reference images for characters in this shot.
    Falls back to design sheet if no individual refs are available.
    """
    images = []

    # Prefer per-character refs (only those relevant to this shot)
    for char_id in getattr(shot, 'characters', []):
        entity = state.reference_store.get_entity(char_id)
        if entity and entity.image_path and os.path.exists(entity.image_path):
            try:
                images.append(Image.open(entity.image_path))
            except Exception:
                pass

    # Fallback: design sheet when no individual refs exist
    if not images and state.design_sheet_path and os.path.exists(state.design_sheet_path):
        try:
            images.append(Image.open(state.design_sheet_path))
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

    img = _kf_generate(state, prompt=full_prompt, reference_images=all_refs, save_path=save_path)
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
    print(f"  Mode: DEFAULT (each shot independent, {KF_WORKERS} workers)")
    total = len(state.shots)

    def _process(shot):
        save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")
        if os.path.exists(save_path):
            print(f"  ⏭️  Shot {shot.shot_id} exists")
            shot.keyframe_path = save_path
            shot.status = "keyframe_done"
            return
        print(f"  [{shot.shot_id}/{total}] Shot {shot.shot_id} ({shot.time_range})")
        _generate_one(shot, state)

    with ThreadPoolExecutor(max_workers=KF_WORKERS) as ex:
        futures = {ex.submit(_process, s): s for s in state.shots}
        for f in as_completed(futures):
            f.result()
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


def _refine_keyframe(
    cell_img: Image.Image,
    shot: "Shot",
    state: "PipelineState",
) -> Image.Image:
    """
    Refine a cropped Grid cell using character reference images.
    Input: raw crop + character refs for the characters in this shot.
    Output: refined keyframe with corrected character appearance.
    Falls back to raw crop on failure.
    """
    # Load character reference images for this shot's characters
    char_refs: List[Image.Image] = []
    for char_id in shot.characters:
        entity = state.reference_store.get_entity(char_id)
        if entity and entity.image_path and os.path.exists(entity.image_path):
            try:
                char_refs.append(Image.open(entity.image_path))
            except Exception:
                pass

    # Fallback: use design sheet if individual refs unavailable
    if not char_refs and state.design_sheet_path and os.path.exists(state.design_sheet_path):
        try:
            char_refs.append(Image.open(state.design_sheet_path))
        except Exception:
            pass

    if not char_refs:
        return cell_img  # no refs available, return raw crop

    n_chars = len(shot.characters)
    char_ids = ", ".join(shot.characters) if shot.characters else "the character(s)"
    ref_label = "character reference sheet" if len(char_refs) == 1 else f"{len(char_refs)} character reference sheets"

    refine_prompt = (
        f"Refine this keyframe using the provided {ref_label}.\n\n"
        f"Image 1 is the RAW KEYFRAME to refine.\n"
        f"Images 2-{len(char_refs)+1} are CHARACTER REFERENCE SHEETS for {char_ids} "
        f"(each sheet shows face close-up on the left and full-body outfit on the right).\n\n"
        f"Instructions:\n"
        f"- Keep the scene composition, background, camera angle, and action from Image 1\n"
        f"- Correct the character(s) appearance to precisely match the reference sheets: "
        f"face, hair color/style, skin tone, clothing (every item), accessories\n"
        f"- Maintain the same artistic style as Image 1\n"
        f"- Output a single refined image (not a collage)\n\n"
        f"Shot description: {shot.t2i_prompt[:300]}"
    )

    ref_images = [cell_img] + char_refs
    refined = _kf_generate(state, prompt=refine_prompt, reference_images=ref_images)
    return refined if refined else cell_img



def _partition_by_scene(shots: list, max_grid: int = 4) -> list:
    """
    Group shots by scene_id first, then split each scene into grids of max_grid.

    Shots from different scenes are NEVER placed in the same grid — doing so
    would cause Gemini to cross-contaminate environments and styles.

    Example (max_grid=4):
      Scene A: shots 0,1,2,3,4  →  [0,1,2,3], [4]
      Scene B: shots 5,6        →  [5,6]
      Scene A: shots 7,8        →  [7,8]   ← new group even though same scene_id
    """
    grids = []
    current_scene: str = None
    current_group: list = []

    for shot in shots:
        scene = getattr(shot, 'scene_id', None) or "unknown"
        if scene != current_scene or len(current_group) >= max_grid:
            if current_group:
                grids.append(current_group)
            current_group = [shot]
            current_scene = scene
        else:
            current_group.append(shot)

    if current_group:
        grids.append(current_group)

    return grids


def _process_grid_group(
    g_idx: int,
    n_groups: int,
    group: list,
    state: "PipelineState",
    char_list: list,
    prev_same_scene: List[Image.Image],
) -> Optional[Image.Image]:
    """Generate one 2×2 grid + refine 4 cells. Returns the raw grid image (or None)."""
    from v2.prompts.narrative_grid import compile_narrative_grid_prompt

    GRID_SIZE = 4
    scene_id = getattr(group[0], 'scene_id', 'unknown') or 'unknown'

    save_paths = [os.path.join(state.output_dir, f"shot_{s.shot_id}.png") for s in group]
    if all(os.path.exists(p) for p in save_paths):
        print(f"  ⏭️  Grid {g_idx+1} (scene={scene_id}): all shots exist, skipping")
        for s, p in zip(group, save_paths):
            s.keyframe_path = p
            s.status = "keyframe_done"
        try:
            return Image.open(save_paths[0])
        except Exception:
            return None

    padded_group = list(group)
    while len(padded_group) < GRID_SIZE:
        padded_group.append(type('PaddedShot', (), {
            't2i_prompt': group[-1].t2i_prompt,
            'padded': True,
        })())
    shot_dicts = [
        {"t2i_prompt": s.t2i_prompt, "padded": getattr(s, 'padded', False)}
        for s in padded_group
    ]

    # Collect characters appearing in this group
    chars_in_group, seen_ids = [], set()
    for s in group:
        for cid in getattr(s, 'characters', []):
            if cid not in seen_ids:
                seen_ids.add(cid)
                chars_in_group.append(cid)

    ref_imgs: List[Image.Image] = []
    for cid in chars_in_group:
        entity = state.reference_store.get_entity(cid)
        if entity and entity.image_path and os.path.exists(entity.image_path):
            try:
                ref_imgs.append(Image.open(entity.image_path))
            except Exception:
                pass

    if not ref_imgs and state.design_sheet_path and os.path.exists(state.design_sheet_path):
        try:
            ref_imgs.append(Image.open(state.design_sheet_path))
        except Exception:
            pass

    source_grid_img = None
    if getattr(state, "source_frame_grid", False):
        from v2.core.source_frames import extract_midpoint_frame, compose_2x2_grid
        try:
            frames = [
                extract_midpoint_frame(state.video_path, s.start_time, s.end_time)
                for s in group
            ]
            source_grid_img = compose_2x2_grid(frames)
            src_save = os.path.join(state.output_dir, f"source_grid_{g_idx+1}.png")
            source_grid_img.save(src_save)
            ref_imgs.append(source_grid_img)
        except Exception as e:
            print(f"    ⚠️  source frame extraction failed ({e}) — falling back to prev_scene")
            ref_imgs.extend(prev_same_scene)
    else:
        ref_imgs.extend(prev_same_scene)

    char_ids_label = ", ".join(chars_in_group) if chars_in_group else "none"
    if source_grid_img is not None:
        print(f"    Grid {g_idx+1}: refs chars=[{char_ids_label}], source_frame_grid=on")
    else:
        print(f"    Grid {g_idx+1}: refs chars=[{char_ids_label}], prev_scene_grids={len(prev_same_scene)}")

    grid_prompt = compile_narrative_grid_prompt(
        shots=shot_dicts,
        style=state.style,
        characters=char_list,
        environment=group[0].environment_description[:200] if group[0].environment_description else None,
        previous_grids=0 if source_grid_img is not None else len(prev_same_scene),
        source_frame_grid=source_grid_img is not None,
    )

    PROMPT_LIMIT = 4000
    if len(grid_prompt) > PROMPT_LIMIT:
        print(f"  ⚠️  Grid {g_idx+1} prompt {len(grid_prompt)}c > {PROMPT_LIMIT}c — compressing...")
        grid_prompt = _compress_grid_prompt(grid_prompt, PROMPT_LIMIT)
        print(f"  ✂️  Grid {g_idx+1} compressed to {len(grid_prompt)}c")

    print(f"  ▶ Grid {g_idx+1}/{n_groups} (scene='{scene_id}', shots {[s.shot_id for s in group]})...")
    grid_img = _kf_generate(state, prompt=grid_prompt, reference_images=ref_imgs, aspect_ratio="16:9")

    if not grid_img:
        print(f"  ❌ Grid {g_idx+1} failed — falling back to default for these shots")
        for shot in group:
            _generate_one(shot, state,
                          extra_refs=prev_same_scene[-1:] if prev_same_scene else None,
                          extra_ref_label="Previous Grid (same scene)")
        return None

    grid_save = os.path.join(state.output_dir, f"grid_{g_idx+1}_scene_{scene_id}.png")
    grid_img.save(grid_save)

    # Crop and refine each cell concurrently
    cells = _crop_grid(grid_img)
    real_pairs = [(s, c) for s, c in zip(group, cells) if not getattr(s, 'padded', False)]

    def _refine_and_save(shot, cell_img):
        save_path = os.path.join(state.output_dir, f"shot_{shot.shot_id}.png")
        print(f"    Shot {shot.shot_id}: refining with {len(shot.characters)} char(s)...")
        final_img = _refine_keyframe(cell_img, shot, state)
        final_img.save(save_path)
        shot.keyframe_path = save_path
        shot.status = "keyframe_done"
        print(f"    ✅ Shot {shot.shot_id} saved")

    with ThreadPoolExecutor(max_workers=KF_WORKERS) as ex:
        futures = [ex.submit(_refine_and_save, s, c) for s, c in real_pairs]
        for f in as_completed(futures):
            f.result()

    return grid_img


def _run_consistency(state: "PipelineState") -> "PipelineState":
    print("  Mode: CONSISTENCY (scene-aware 2×2 grid → crop → zoom)")

    GRID_SIZE = 4
    shots = state.shots
    char_list = [
        {"name": c.name, "description": c.description[:120]}
        for c in state.characters
    ]

    groups = _partition_by_scene(shots, max_grid=GRID_SIZE)
    print(f"  {len(shots)} shots → {len(groups)} grid group(s) across scenes:")
    for g in groups:
        scene = getattr(g[0], 'scene_id', '?')
        print(f"    Scene '{scene}': shots {[s.shot_id for s in g]}")

    use_source = getattr(state, "source_frame_grid", False)

    if use_source:
        # No prev-grid dependency → run all groups concurrently
        print(f"  Source-frame mode: running {len(groups)} grids CONCURRENTLY ({KF_WORKERS} workers)")
        with ThreadPoolExecutor(max_workers=KF_WORKERS) as ex:
            futures = [
                ex.submit(_process_grid_group, i, len(groups), g, state, char_list, [])
                for i, g in enumerate(groups)
            ]
            for f in as_completed(futures):
                f.result()
    else:
        # Sequential: each group uses prev_same_scene as reference
        scene_grids: dict = {}
        for g_idx, group in enumerate(groups):
            scene_id = getattr(group[0], 'scene_id', 'unknown') or 'unknown'
            prev_same_scene = scene_grids.get(scene_id, [])
            grid_img = _process_grid_group(g_idx, len(groups), group, state, char_list, prev_same_scene)
            if grid_img is not None:
                scene_grids.setdefault(scene_id, []).append(grid_img)

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

        img = _kf_generate(state, prompt=full_prompt, reference_images=all_refs, save_path=save_path)
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
