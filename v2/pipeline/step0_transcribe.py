"""
Step 0 — Audio Transcription via Whisper.

Extracts audio from the input video, runs OpenAI Whisper to get a
word-level / segment-level transcript with timestamps, and returns a
structured list of dialogue lines that Step 1 passes into Gemini.

Output format (list of dicts, sorted by start time):
  [{"start": 4.2, "end": 6.8, "text": "Hello Mr. Dawson"}, ...]
"""
from __future__ import annotations
import json
import os
import subprocess
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from v2.core.schema import PipelineState


def _extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract mono 16kHz WAV from video for Whisper."""
    if os.path.exists(audio_path):
        print(f"  ⏭️  Audio already exists: {audio_path}")
        return True
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        audio_path, "-loglevel", "error",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ Audio extracted: {audio_path}")
        return True
    print(f"  ❌ Audio extraction failed: {result.stderr[-200:]}")
    return False


def _run_whisper(audio_path: str, output_dir: str, language: str = "auto") -> List[Dict]:
    """
    Run Whisper on the audio file.
    Returns a list of segment dicts: [{"start": float, "end": float, "text": str}]
    """
    transcript_path = os.path.join(
        output_dir,
        os.path.splitext(os.path.basename(audio_path))[0] + ".json"
    )

    if os.path.exists(transcript_path):
        print(f"  ⏭️  Transcript already exists: {transcript_path}")
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("segments", [])

    print(f"  Running Whisper (model=medium)...")
    print(f"  Note: first run downloads ~1.4 GB model")

    lang_args = ["--language", language] if language != "auto" else []
    cmd = [
        "whisper", audio_path,
        "--model", "medium",
        "--output_format", "json",
        "--output_dir", output_dir,
        "--verbose", "False",
    ] + lang_args

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Whisper failed: {result.stderr[-300:]}")
        return []

    if os.path.exists(transcript_path):
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
        segments = data.get("segments", [])
        print(f"  ✅ Whisper transcribed {len(segments)} segment(s)")
        return segments

    print("  ❌ Whisper output file not found")
    return []


def segments_to_dialogue(segments: List[Dict]) -> List[Dict]:
    """
    Convert Whisper segments into clean dialogue lines.
    Merges very short pauses (< 0.5s gap) into single lines.
    Returns: [{"start": float, "end": float, "text": str}]
    """
    if not segments:
        return []

    lines = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0))
        end = float(seg.get("end", 0))

        # Merge with previous line if gap is tiny
        if lines and start - lines[-1]["end"] < 0.5:
            lines[-1]["text"] += " " + text
            lines[-1]["end"] = end
        else:
            lines.append({"start": start, "end": end, "text": text})

    return lines


def format_transcript_for_prompt(dialogue_lines: List[Dict]) -> str:
    """
    Format dialogue lines as a timestamped string for injection into the Gemini prompt.
    Example:
      [0.0s - 3.2s] Hello, is anyone here?
      [5.4s - 7.1s] I'm looking for Jack.
    """
    if not dialogue_lines:
        return "(no speech detected)"
    return "\n".join(
        f"[{d['start']:.1f}s - {d['end']:.1f}s] {d['text']}"
        for d in dialogue_lines
    )


def run(state: "PipelineState") -> List[Dict]:
    """
    Run Whisper transcription and return dialogue lines.
    Stores audio and transcript files in output_dir.
    Does NOT modify state — returns raw dialogue list for Step 1.
    """
    print("\n" + "=" * 60)
    print("STEP 0 — Audio Transcription (Whisper)")
    print("=" * 60)

    base = os.path.splitext(os.path.basename(state.video_path))[0]
    audio_path = os.path.join(state.output_dir, f"{base}.wav")

    # 1. Extract audio
    if not _extract_audio(state.video_path, audio_path):
        print("  ⚠️  Skipping transcription (audio extraction failed)")
        return []

    # 2. Run Whisper
    segments = _run_whisper(audio_path, state.output_dir)

    # 3. Convert to clean dialogue lines
    dialogue_lines = segments_to_dialogue(segments)
    print(f"  {len(dialogue_lines)} dialogue line(s) found:")
    for d in dialogue_lines:
        print(f"    [{d['start']:.1f}s-{d['end']:.1f}s] {d['text'][:80]}")

    return dialogue_lines
