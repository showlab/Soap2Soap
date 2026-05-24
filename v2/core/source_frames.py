"""
Source video frame extraction and 2×2 grid composition.

Used when --source-frame-grid is enabled: for each grid group, extract
the midpoint frame of each shot from the source video and compose them
into a 2×2 reference grid that guides AI grid generation.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from typing import List

from PIL import Image


def extract_midpoint_frame(video_path: str, start_time: float, end_time: float) -> Image.Image:
    """Extract a single frame from `video_path` at the midpoint of [start, end]."""
    midpoint = (start_time + end_time) / 2.0
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{midpoint:.3f}", "-i", video_path,
             "-frames:v", "1", "-q:v", "2", tmp_path],
            check=True, capture_output=True,
        )
        return Image.open(tmp_path).copy()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def compose_2x2_grid(images: List[Image.Image], target_cell_size=(640, 360)) -> Image.Image:
    """
    Compose up to 4 images into a 2×2 grid (TL, TR, BL, BR).
    Pads with the last image (or black) if fewer than 4 are provided.
    Each cell is resized to `target_cell_size` to ensure a uniform grid.
    """
    if not images:
        raise ValueError("compose_2x2_grid requires at least 1 image")

    padded = list(images[:4])
    while len(padded) < 4:
        padded.append(padded[-1].copy())

    cw, ch = target_cell_size
    cells = [img.convert("RGB").resize((cw, ch), Image.LANCZOS) for img in padded]

    grid = Image.new("RGB", (cw * 2, ch * 2))
    grid.paste(cells[0], (0, 0))      # TL
    grid.paste(cells[1], (cw, 0))     # TR
    grid.paste(cells[2], (0, ch))     # BL
    grid.paste(cells[3], (cw, ch))    # BR
    return grid
