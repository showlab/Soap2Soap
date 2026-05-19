"""
Soap2Soap V2 — Main Pipeline Orchestrator.

Video-to-Video generation:
  Video → Whisper transcript → Gemini analysis (with transcript)
        → Characters (Imagen) → Shot prompts
        → Keyframes (Gemini/Imagen, consistency mode) → Video (Veo3 or static) → Merge

Usage:
    python v2/pipeline.py <video_path> --style disney [--shots 10] [--real-video]
    python v2/pipeline.py <video_path> --style disney --no-whisper  # skip transcription
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime

# Ensure v2 package is importable from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v2.core.schema import PipelineState
from v2.pipeline import (
    step0_transcribe,
    step1_analyze,
    step2_characters,
    step3_compile,
    step3b_camera_groups,
    step4_keyframes,
    step4b_inspect,
    step5_video,
    step6_merge,
)


def run_pipeline(
    video_path: str,
    style: str = "disney",
    max_shots: int = 10,
    dev_mode: bool = True,
    output_dir: str = ".",
    skip_to_step: int = 1,
    use_whisper: bool = True,
    generation_mode: str = "consistency",
) -> str:
    """
    Run the full V2V pipeline. Returns path to final video.
    skip_to_step: resume from this step if prior outputs exist.
    """
    print("\n" + "=" * 70)
    print("🎬 Soap2Soap V2 — Video-to-Video Pipeline")
    print("=" * 70)
    print(f"  Input    : {video_path}")
    print(f"  Style    : {style}")
    print(f"  Max shots: {max_shots}")
    print(f"  Mode     : {generation_mode}")
    print(f"  Video    : {'Veo 3 (real)' if not dev_mode else 'static 3s fallback (dev mode)'}")
    print(f"  Whisper  : {'enabled' if use_whisper else 'disabled'}")
    print(f"  Output   : {output_dir}")
    print("=" * 70)

    state = PipelineState(
        video_path=video_path,
        style=style,
        max_shots=max_shots,
        dev_mode=dev_mode,
        output_dir=output_dir,
        generation_mode=generation_mode,
    )

    # Check for cached analysis
    cache_path = os.path.join(output_dir, "v2_analysis.json")
    if skip_to_step > 1 and os.path.exists(cache_path):
        print(f"\n  ↩️  Loading cached analysis from {cache_path}")
        state = _load_state(state, cache_path)
        # Re-apply timestamp correction on cached data too
        video_duration = step1_analyze._get_video_duration(video_path)
        state.shots = step1_analyze._fix_timestamps(state.shots, video_duration)
        for s in state.shots:
            print(f"    Shot {s.shot_id}: {s.time_range} ({s.duration:.1f}s)")
    else:
        # Step 0 — Whisper transcription (runs before Gemini analysis)
        transcript = ""
        if use_whisper:
            dialogue_lines = step0_transcribe.run(state)
            transcript = step0_transcribe.format_transcript_for_prompt(dialogue_lines)

        # Step 1 — Video Analysis (Gemini, with transcript injected)
        state = step1_analyze.run(state, transcript=transcript)
        _save_state(state, cache_path)

    # Step 2 — Character images
    state = step2_characters.run(state)

    # Step 3 — Compile prompts
    state = step3_compile.run(state)

    # Step 3b — Camera group analysis (camera_tree mode only)
    state = step3b_camera_groups.run(state)

    # Step 4 — Keyframes
    state = step4_keyframes.run(state)

    # Step 4b — Inspection & auto-fix
    state = step4b_inspect.run(state)

    # Step 5 — Video clips
    state = step5_video.run(state)

    # Step 6 — Merge
    final_video = step6_merge.run(state)

    # Summary
    summary = state.to_summary()
    summary["final_video"] = final_video
    print("\n" + "=" * 70)
    print("📊 Pipeline Summary")
    print("=" * 70)
    for k, v in summary.items():
        print(f"  {k:15}: {v}")
    print("=" * 70)

    return final_video


def _save_state(state: PipelineState, path: str):
    """Save analysis results to JSON cache."""
    data = {
        "video_path": state.video_path,
        "style": state.style,
        "aspect_ratio": state.aspect_ratio,
        "characters": [
            {"id": c.id, "name": c.name, "description": c.description}
            for c in state.characters
        ],
        "shots": [
            {
                "index": s.index,
                "scene_id": s.scene_id,
                "time_range": s.time_range,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "setting_description": s.setting_description,
                "environment_description": s.environment_description,
                "lighting_setup": s.lighting_setup,
                "color_grading": s.color_grading,
                "shot_size": s.shot_size,
                "camera_angle": s.camera_angle,
                "camera_movement": s.camera_movement,
                "focal_length": s.focal_length,
                "depth_of_field": s.depth_of_field,
                "mood_atmosphere": s.mood_atmosphere,
                "composition": s.composition,
                "subject_movement": s.subject_movement,
                "characters": s.characters,
                "dialogue": [{"speaker_id": d.speaker_id, "text": d.text} for d in s.dialogue],
                "t2i_prompt": s.t2i_prompt,
                "i2v_prompt": s.i2v_prompt,
            }
            for s in state.shots
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  💾 Analysis cached: {path}")


def _load_state(state: PipelineState, path: str) -> PipelineState:
    """Restore characters and shots from cache JSON."""
    from v2.core.schema import Character, Dialogue, Shot
    from v2.core.reference_store import ReferenceStore, ReferenceEntity

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    state.aspect_ratio = data.get("aspect_ratio", "16:9")
    store = ReferenceStore()

    for idx, c in enumerate(data.get("characters", [])):
        char = Character(id=c["id"], name=c["name"], description=c["description"])
        state.characters.append(char)
        store.add_entity(ReferenceEntity(
            entity_id=char.id,
            entity_type="character",
            description=char.description,
            image_index=idx + 1,
        ))

    state.reference_store = store

    for s in data.get("shots", []):
        shot = Shot(
            index=s["index"],
            scene_id=s["scene_id"],
            time_range=s["time_range"],
            start_time=s["start_time"],
            end_time=s["end_time"],
            duration=s["duration"],
            setting_description=s.get("setting_description", ""),
            environment_description=s.get("environment_description", ""),
            lighting_setup=s.get("lighting_setup", ""),
            color_grading=s.get("color_grading", ""),
            shot_size=s.get("shot_size", ""),
            camera_angle=s.get("camera_angle", ""),
            camera_movement=s.get("camera_movement", ""),
            focal_length=s.get("focal_length", ""),
            depth_of_field=s.get("depth_of_field", ""),
            mood_atmosphere=s.get("mood_atmosphere", ""),
            composition=s.get("composition", ""),
            subject_movement=s.get("subject_movement", ""),
            characters=s.get("characters", []),
            dialogue=[Dialogue(d["speaker_id"], d["text"]) for d in s.get("dialogue", [])],
            t2i_prompt=s.get("t2i_prompt", ""),
            i2v_prompt=s.get("i2v_prompt", ""),
        )
        state.shots.append(shot)

    return state


def main():
    parser = argparse.ArgumentParser(
        description="Soap2Soap V2 — Video-to-Video generation"
    )
    parser.add_argument("video", help="Input video path")
    parser.add_argument("--style", default="disney",
                        choices=["realistic", "disney", "anime", "japanese_anime",
                                 "clay", "lego", "family_guy"])
    parser.add_argument("--shots", type=int, default=10,
                        help="Max shots to generate (default: 10)")
    parser.add_argument("--real-video", action="store_true",
                        help="Use Veo 3 for video generation (default: static fallback)")
    parser.add_argument("--output-dir", default=".",
                        help="Output directory (default: current dir)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip Step 0+1 if v2_analysis.json already exists")
    parser.add_argument("--no-whisper", action="store_true",
                        help="Skip Whisper transcription (faster, less accurate dialogue)")
    parser.add_argument("--mode", default="consistency",
                        choices=["default", "consistency", "camera_tree"],
                        help="Keyframe generation mode (default: consistency)")
    args = parser.parse_args()

    final = run_pipeline(
        video_path=args.video,
        style=args.style,
        max_shots=args.shots,
        dev_mode=not args.real_video,
        output_dir=args.output_dir,
        skip_to_step=2 if args.resume else 1,
        use_whisper=not args.no_whisper,
        generation_mode=args.mode,
    )

    if final:
        print(f"\n🎉 Done! Final video: {final}")
    else:
        print("\n⚠️  Pipeline completed with errors — check individual shot files")
        sys.exit(1)


if __name__ == "__main__":
    main()
