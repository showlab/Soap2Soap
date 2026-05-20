"""
Step 6 — Video Merge.
Concatenates all shot clips into the final output video with ffmpeg.
"""
from __future__ import annotations
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


def run(state: "PipelineState") -> str:
    """Returns the path to the final merged video, or "" on failure."""
    print("\n" + "=" * 60)
    print("STEP 6 — Video Merge")
    print("=" * 60)

    clips = [s.video_path for s in state.shots if s.video_path]
    if not clips:
        print("  ❌ No video clips to merge")
        return ""

    print(f"  Merging {len(clips)} clips...")

    # Write concat list
    list_path = os.path.join(state.output_dir, "concat_list_v2.txt")
    with open(list_path, "w") as f:
        for clip in clips:
            f.write(f"file '{os.path.abspath(clip)}'\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(state.output_dir, f"final_output_v2_{timestamp}.mp4")

    # Re-encode with uniform 1280×720 to handle clips of different resolutions
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  ✅ Final video: {output_path} ({size_mb:.1f} MB)")
        os.remove(list_path)
        return output_path
    else:
        print(f"  ❌ ffmpeg merge failed:\n{result.stderr[-500:]}")
        return ""
