"""
Step 4b — Keyframe Inspection & Auto-Fix.

Two-pass inspection after keyframe generation:

Pass 1 — Individual frame review (per shot):
  Gemini checks each keyframe for:
  - Grid/collage badcase (generation failure producing multi-image sheet)
  - Style match (does it match the target style, e.g. Disney?)
  - Scene/environment match (does it match the shot description?)
  - Character appearance match (do characters look right?)
  If any check fails → re-generate the keyframe (max MAX_RETRIES times).

Pass 2 — Collective grid consistency review:
  All passing keyframes are arranged into a labeled grid (2×2, 3×3, 4×4).
  Gemini evaluates cross-shot consistency:
  - Visual style uniformity
  - Background / environment continuity
  - Character appearance / clothing consistency
  Shots flagged as inconsistent are re-generated (max MAX_RETRIES times).

MAX_RETRIES = 3 per shot, shared across both passes.
"""
from __future__ import annotations
import io
import json
import os
import math
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types

if TYPE_CHECKING:
    from v2.core.schema import PipelineState, Shot

MAX_RETRIES = 3
from v2.clients.gemini_client import VLM_MODEL as _GEMINI_VISION_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# Gemini helpers
# ─────────────────────────────────────────────────────────────────────────────

def _client() -> genai.Client:
    api_key = os.environ.get("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("GENAI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _ask_gemini(prompt: str, images: List[Image.Image]) -> str:
    client = _client()
    contents: list = images + [prompt]
    response = client.models.generate_content(
        model=_GEMINI_VISION_MODEL,
        contents=contents,
    )
    return response.text or ""


def _parse_json_response(text: str) -> dict:
    import re
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Pass 1 — Individual frame inspection
# ─────────────────────────────────────────────────────────────────────────────

PASS1_PROMPT = """You are a quality inspector for AI-generated keyframe images.

TARGET STYLE: {style}
EXPECTED SHOT DESCRIPTION: {description}
EXPECTED CHARACTERS: {characters}

Inspect this image and check ALL of the following:

1. GRID_BADCASE: Is this image a collage/grid of multiple smaller images stitched together (a contact sheet), rather than a single unified image?
2. STYLE_MATCH: Does the visual style match "{style}"? (e.g. if Disney: should be vibrant 3D-animated, not photorealistic or sketch)
3. SCENE_MATCH: Does the scene/environment in the image match the expected description?
4. CHARACTER_MATCH: Do the characters present look correct based on the description? (appearance, clothing, overall design)

Return ONLY valid JSON, no explanation:
{{
  "passed": true or false,
  "issues": {{
    "grid_badcase": true or false,
    "style_mismatch": true or false,
    "scene_mismatch": true or false,
    "character_mismatch": true or false
  }},
  "failed_checks": ["grid_badcase", "style_mismatch", ...],
  "description": "one sentence explaining what is wrong, or 'OK' if passed"
}}
"""


def inspect_single_frame(shot: "Shot", style: str) -> Tuple[bool, List[str], str]:
    """
    Run Pass 1 on a single shot's keyframe.
    Returns (passed, failed_checks, description).
    """
    if not shot.keyframe_path or not os.path.exists(shot.keyframe_path):
        return False, ["missing_image"], "Keyframe file not found"

    img = Image.open(shot.keyframe_path)

    char_desc = ", ".join(shot.characters) if shot.characters else "none specified"
    prompt = PASS1_PROMPT.format(
        style=style,
        description=shot.t2i_prompt[:300],
        characters=char_desc,
    )

    try:
        raw = _ask_gemini(prompt, [img])
        result = _parse_json_response(raw)
        passed = bool(result.get("passed", False))
        failed = result.get("failed_checks", [])
        desc = result.get("description", raw[:200])
        return passed, failed, desc
    except Exception as e:
        return False, ["inspection_error"], str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Pass 2 — Grid consistency inspection
# ─────────────────────────────────────────────────────────────────────────────

def _make_labeled_grid(shots: List["Shot"], cell_size: int = 300) -> Image.Image:
    """
    Arrange keyframe thumbnails into a labeled grid.
    Grid is square: 2×2 for ≤4 shots, 3×3 for ≤9, 4×4 for ≤16.
    Each cell is labeled with its shot number.
    """
    n = len(shots)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    label_h = 30
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + label_h)
    grid = Image.new("RGB", (grid_w, grid_h), color=(40, 40, 40))
    draw = ImageDraw.Draw(grid)

    for idx, shot in enumerate(shots):
        row, col = divmod(idx, cols)
        x = col * cell_size
        y = row * (cell_size + label_h)

        # Draw label background
        draw.rectangle([x, y, x + cell_size, y + label_h], fill=(60, 60, 60))
        draw.text((x + 6, y + 6), f"Shot {shot.shot_id}", fill=(255, 220, 80))

        # Draw thumbnail
        if shot.keyframe_path and os.path.exists(shot.keyframe_path):
            thumb = Image.open(shot.keyframe_path).convert("RGB")
            thumb.thumbnail((cell_size, cell_size))
            # Center-crop to exact cell_size
            tw, th = thumb.size
            paste_x = x + (cell_size - tw) // 2
            paste_y = y + label_h + (cell_size - th) // 2
            grid.paste(thumb, (paste_x, paste_y))
        else:
            draw.rectangle([x, y + label_h, x + cell_size, y + label_h + cell_size],
                           fill=(80, 20, 20))
            draw.text((x + 10, y + label_h + cell_size // 2), "MISSING", fill=(255, 80, 80))

    return grid


PASS2_PROMPT = """You are a quality inspector for a {style}-style animated video.

Below is a grid showing all {n} keyframes (labeled Shot 1 through Shot {n}).
Evaluate the CONSISTENCY across all shots:

1. STYLE: Are ALL frames rendered in the same "{style}" visual style? (e.g. all Disney 3D-animated, not a mix of styles)
2. ENVIRONMENT: Are background/environment elements visually consistent within scenes that should share the same location?
3. CHARACTER_APPEARANCE: Do the same characters look consistent (face, clothing, proportions) across all shots they appear in?

For each shot that FAILS any consistency check, list it with specific reasons.
A shot passes if it looks consistent with the majority of other shots.

Return ONLY valid JSON:
{{
  "overall_passed": true or false,
  "failed_shots": [
    {{
      "shot_id": 3,
      "issues": ["style_mismatch", "character_inconsistent", "environment_inconsistent"],
      "description": "Shot 3 uses photorealistic style instead of Disney 3D animation"
    }}
  ]
}}
"""


def inspect_grid(shots: List["Shot"], style: str) -> Tuple[bool, List[Dict]]:
    """
    Run Pass 2 grid inspection.
    Returns (overall_passed, list of {shot_id, issues, description}).
    """
    valid_shots = [s for s in shots if s.keyframe_path and os.path.exists(s.keyframe_path)]
    if len(valid_shots) < 2:
        return True, []

    grid_img = _make_labeled_grid(valid_shots)
    prompt = PASS2_PROMPT.format(style=style, n=len(valid_shots))

    try:
        raw = _ask_gemini(prompt, [grid_img])
        result = _parse_json_response(raw)
        overall = bool(result.get("overall_passed", True))
        failed = result.get("failed_shots", [])
        return overall, failed
    except Exception as e:
        print(f"  ⚠️  Grid inspection error: {e}")
        return True, []


# ─────────────────────────────────────────────────────────────────────────────
# Re-generation helper
# ─────────────────────────────────────────────────────────────────────────────

def _regenerate_keyframe(shot: "Shot", state: "PipelineState",
                         previous_keyframe: Optional[str]) -> bool:
    """Re-generate a single shot's keyframe using the existing pipeline logic."""
    from v2.clients.imagen_client import generate_keyframe
    from v2.core.reference_resolver import resolve_references

    final_prompt, ref_images = resolve_references(
        base_prompt=shot.t2i_prompt,
        reference_store=state.reference_store,
        previous_image_path=previous_keyframe,
        inject_consistency_instruction=previous_keyframe is not None,
    )
    img = generate_keyframe(
        prompt=final_prompt,
        reference_images=ref_images,
        save_path=shot.keyframe_path,
    )
    return img is not None


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(state: "PipelineState") -> "PipelineState":
    print("\n" + "=" * 60)
    print("STEP 4b — Keyframe Inspection & Auto-Fix")
    print("=" * 60)

    retry_counts: Dict[int, int] = {s.shot_id: 0 for s in state.shots}

    # ── PASS 1: Individual frame review ──────────────────────────────────────
    print("\n  ── Pass 1: Individual Frame Review ──")
    for shot in state.shots:
        if not shot.keyframe_path:
            continue

        for attempt in range(MAX_RETRIES + 1):
            passed, failed_checks, desc = inspect_single_frame(shot, state.style)

            if passed:
                print(f"  ✅ Shot {shot.shot_id}: OK")
                break

            print(f"  ❌ Shot {shot.shot_id} failed [{', '.join(failed_checks)}]: {desc}")

            if retry_counts[shot.shot_id] >= MAX_RETRIES:
                print(f"  ⚠️  Shot {shot.shot_id}: max retries reached, keeping current")
                break

            # Find previous valid keyframe for consistency reference
            prev = None
            for prev_shot in state.shots:
                if prev_shot.index < shot.index and prev_shot.keyframe_path:
                    prev = prev_shot.keyframe_path

            print(f"  🔄 Shot {shot.shot_id}: re-generating (attempt {retry_counts[shot.shot_id]+1}/{MAX_RETRIES})")
            retry_counts[shot.shot_id] += 1
            ok = _regenerate_keyframe(shot, state, prev)
            if not ok:
                print(f"  ⚠️  Shot {shot.shot_id}: re-generation failed")
                break

    # ── PASS 2: Grid consistency review ──────────────────────────────────────
    print("\n  ── Pass 2: Grid Consistency Review ──")

    # Save grid for inspection reference
    valid = [s for s in state.shots if s.keyframe_path and os.path.exists(s.keyframe_path)]
    if valid:
        grid_path = os.path.join(state.output_dir, "inspection_grid.png")
        grid_img = _make_labeled_grid(valid)
        grid_img.save(grid_path)
        print(f"  Grid saved: {grid_path} ({len(valid)} shots)")

    for grid_round in range(MAX_RETRIES):
        overall, failed_shots = inspect_grid(state.shots, state.style)

        if overall or not failed_shots:
            print(f"  ✅ Grid inspection passed")
            break

        print(f"  ❌ Grid round {grid_round+1}: {len(failed_shots)} shot(s) inconsistent")
        any_regen = False

        for item in failed_shots:
            shot_id = item.get("shot_id")
            issues = item.get("issues", [])
            desc = item.get("description", "")

            shot = next((s for s in state.shots if s.shot_id == shot_id), None)
            if not shot or not shot.keyframe_path:
                continue

            if retry_counts.get(shot_id, 0) >= MAX_RETRIES:
                print(f"  ⚠️  Shot {shot_id}: max retries reached, skipping")
                continue

            print(f"  🔄 Shot {shot_id} [{', '.join(issues)}]: {desc[:80]}")
            print(f"       Re-generating (total retries so far: {retry_counts[shot_id]})")

            prev = None
            for prev_shot in state.shots:
                if prev_shot.index < shot.index and prev_shot.keyframe_path:
                    prev = prev_shot.keyframe_path

            retry_counts[shot_id] = retry_counts.get(shot_id, 0) + 1
            ok = _regenerate_keyframe(shot, state, prev)
            if ok:
                any_regen = True
                print(f"  ✅ Shot {shot_id}: re-generated")
            else:
                print(f"  ⚠️  Shot {shot_id}: re-generation failed")

        if not any_regen:
            print(f"  ℹ️  No shots could be re-generated, stopping grid rounds")
            break

        # Update grid after re-generation
        valid = [s for s in state.shots if s.keyframe_path and os.path.exists(s.keyframe_path)]
        grid_img = _make_labeled_grid(valid)
        grid_img.save(grid_path)
        print(f"  Grid updated: {grid_path}")

    # Summary
    total_retries = sum(retry_counts.values())
    print(f"\n  Inspection complete — total re-generations: {total_retries}")
    return state
