"""
Gemini client — video understanding, text tasks, safety rewrite.
"""
from __future__ import annotations
import os
import json
import re
import time
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types


_GEMINI_VISION_MODEL_FLASH = "gemini-2.0-flash"


def _client() -> genai.Client:
    api_key = os.environ.get("GENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("GENAI_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def upload_video(video_path: str) -> str:
    """Upload a local video file to the Gemini File API and return the file URI."""
    client = _client()
    print(f"  Uploading video: {video_path}")
    with open(video_path, "rb") as f:
        video_file = client.files.upload(
            file=f,
            config=types.UploadFileConfig(mime_type="video/mp4")
        )
    # Wait until active
    while video_file.state.name == "PROCESSING":
        time.sleep(3)
        video_file = client.files.get(name=video_file.name)
    print(f"  Upload done: {video_file.uri}")
    return video_file.uri


def analyze_video(video_uri: str, prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Send a video (by URI) to Gemini with a prompt and return the text response."""
    client = _client()
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_uri(file_uri=video_uri, mime_type="video/mp4"),
            prompt,
        ],
    )
    return response.text


def text_generate(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Plain text generation."""
    client = _client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


def safety_rewrite(original_prompt: str) -> str:
    """
    Ask Gemini to rewrite a prompt that triggered content filters.
    Ported from pai_v1_backend safety_rewrite.py.
    """
    rewrite_prompt = f"""The following image generation prompt triggered a safety filter.

ORIGINAL PROMPT:
"{original_prompt}"

Rewrite this prompt to be safe and compliant while preserving the visual composition.
- Remove language that could trigger violence, NSFW, or sensitive-content filters.
- Keep all character descriptions, clothing, and scene details.
- Return ONLY the rewritten prompt text, no explanations, no JSON.
"""
    result = text_generate(rewrite_prompt)
    return result.strip()


def extract_json(text: str) -> dict:
    """Extract the first JSON object found in a text response."""
    # Try ```json ... ``` block first
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try raw { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in response:\n{text[:500]}")
