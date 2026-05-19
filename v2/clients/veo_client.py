"""
Veo client — image-to-video generation via Veo 3.
In dev_mode, produces a 3-second static-image clip with ffmpeg instead.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import time
from typing import Optional
from google import genai
from google.genai import types

VEO_MODEL = "veo-3.0-generate-preview"


def _client() -> genai.Client:
    api_key = os.environ.get("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("GENAI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def generate_video_static_fallback(
    image_path: str,
    output_path: str,
    duration: int = 3,
) -> bool:
    """
    Dev-mode fallback: encode a still image as a <duration>-second MP4.
    Uses ffmpeg: -loop 1 -t <duration> -pix_fmt yuv420p
    """
    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", image_path,
        "-c:v", "libx264",
        "-t", str(duration),
        "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dimensions
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ Static clip: {output_path} ({duration}s)")
        return True
    else:
        print(f"  ❌ ffmpeg static clip failed: {result.stderr[-300:]}")
        return False


def generate_video_veo(
    image_path: str,
    prompt: str,
    output_path: str,
    duration: int = 4,
    aspect_ratio: str = "16:9",
    max_retries: int = 3,
) -> bool:
    """
    Generate a video clip via Veo 3 using a keyframe image as the first frame.
    """
    client = _client()

    # Upload keyframe image as the first frame
    with open(image_path, "rb") as f:
        image_file = client.files.upload(
            file=f,
            config=types.UploadFileConfig(mime_type="image/png")
        )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  🎬 Veo attempt {attempt}: generating {duration}s video...")
            operation = client.models.generate_videos(
                model=VEO_MODEL,
                prompt=prompt,
                image=types.Image(
                    image_bytes=open(image_path, "rb").read(),
                    mime_type="image/png",
                ),
                config=types.GenerateVideoConfig(
                    aspect_ratio=aspect_ratio,
                    duration_seconds=duration,
                    number_of_videos=1,
                ),
            )

            poll = 0
            while not operation.done:
                poll += 1
                print(f"     Waiting... ({poll * 10}s)")
                time.sleep(10)
                operation = client.operations.get(operation)

            if not operation.response or not operation.response.generated_videos:
                print(f"  ⚠️  Veo attempt {attempt}: empty response")
                # Log raw operation info for diagnosis
                print(f"     name    : {getattr(operation, 'name', 'N/A')}")
                print(f"     metadata: {getattr(operation, 'metadata', 'N/A')}")
                print(f"     error   : {getattr(operation, 'error', 'N/A')}")
                print(f"     response: {operation.response}")
                continue

            video = operation.response.generated_videos[0]
            client.files.download(file=video.video)
            video.video.save(output_path)
            print(f"  ✅ Veo video saved: {output_path}")
            return True

        except Exception as e:
            print(f"  ❌ Veo attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(10)

    return False


def generate_video(
    image_path: str,
    prompt: str,
    output_path: str,
    dev_mode: bool = True,
    duration: int = 4,
    aspect_ratio: str = "16:9",
) -> bool:
    """
    Unified entry point.
    dev_mode=True  → static 3s clip (fast, no API cost)
    dev_mode=False → Veo 3 (real video)
    """
    if dev_mode:
        return generate_video_static_fallback(image_path, output_path, duration=3)
    return generate_video_veo(image_path, prompt, output_path, duration, aspect_ratio)
