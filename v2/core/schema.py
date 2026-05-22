"""
Core data models for Soap2Soap V2 pipeline.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Dialogue:
    speaker_id: str        # "@character_01"
    text: str
    start_time: float = 0.0  # from Whisper transcript
    end_time: float = 0.0


@dataclass
class Character:
    id: str            # "@character_01"
    name: str          # display name
    description: str   # full visual description for prompts
    image_path: Optional[str] = None   # local path to generated reference image


@dataclass
class Shot:
    index: int                          # 0-based
    scene_id: str                       # "scene_01"
    time_range: str                     # "0.00s - 4.53s"
    start_time: float
    end_time: float
    duration: float

    # Scene metadata
    setting_description: str = ""
    environment_description: str = ""
    lighting_setup: str = ""
    color_grading: str = ""
    shot_size: str = ""
    camera_angle: str = ""
    camera_movement: str = ""
    focal_length: str = ""
    depth_of_field: str = ""
    mood_atmosphere: str = ""
    composition: str = ""
    subject_movement: str = ""

    # Characters present in this shot
    characters: List[str] = field(default_factory=list)  # ["@character_01"]
    dialogue: List[Dialogue] = field(default_factory=list)

    # Compiled prompts (filled in step 3)
    t2i_prompt: str = ""
    i2v_prompt: str = ""

    # Generated assets (filled in steps 4-5)
    keyframe_path: Optional[str] = None
    video_path: Optional[str] = None
    status: str = "pending"   # "pending" | "done" | "failed"

    @property
    def shot_id(self) -> int:
        """1-based shot ID for filenames."""
        return self.index + 1


@dataclass
class PipelineState:
    video_path: str
    style: str                          # "disney" | "realistic" | ...
    aspect_ratio: str = "16:9"
    max_shots: int = 10
    dev_mode: bool = True               # If True: static-image fallback instead of Veo3

    # Generation mode for keyframes
    generation_mode: str = "consistency"  # "default" | "consistency" | "camera_tree"

    # Video generation model
    video_model: str = "veo"             # "veo" | "kling"

    characters: List[Character] = field(default_factory=list)
    shots: List[Shot] = field(default_factory=list)

    # Design sheet (unified character reference for all shots)
    design_sheet_path: Optional[str] = None

    # Camera tree groups (set by step3b)
    camera_groups: List[Dict[str, Any]] = field(default_factory=list)

    # Output dir (working directory)
    output_dir: str = "."

    def get_character(self, char_id: str) -> Optional[Character]:
        for c in self.characters:
            if c.id == char_id:
                return c
        return None

    def to_summary(self) -> Dict[str, Any]:
        return {
            "video": self.video_path,
            "style": self.style,
            "characters": len(self.characters),
            "shots": len(self.shots),
            "done": sum(1 for s in self.shots if s.status == "done"),
            "failed": sum(1 for s in self.shots if s.status == "failed"),
        }
