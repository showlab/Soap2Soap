#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Allocation Agent - Context Memory Management Center

Features:
1. Dynamically allocate structured context memory packages for each shot
2. Manage character identity, environment anchors, Visual DNA, narrative context
3. Serve as persistent shared memory layer, aligning all downstream generation

Usage:
    python agent_memory.py clip1_script.json              # Allocate memory for all shots
    python agent_memory.py clip1_script.json --shot 9     # Allocate memory for specific shot
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Style prompt definitions
STYLE_PROMPTS = {
    "realistic": """
**STYLE: REALISTIC CINEMATIC**
- Photorealistic cinematic style
- High-quality photography, film-like imagery
- Natural lighting, realistic textures
- True-to-life colors and details
""",

    "lego": """
**STYLE: LEGO ANIMATION**
- Everything made of LEGO bricks and pieces
- Plastic toy aesthetic with glossy surfaces
- Minifigure-style characters with articulated joints
- Vibrant colors, clean geometric shapes
- Studded surfaces, brick-built environments
- Stop-motion animation style
""",

    "disney": """
**STYLE: DISNEY/PIXAR 3D CGI ANIMATION**
- Disney/Pixar 3D computer-generated imagery style (like Toy Story, Finding Nemo, Frozen, Moana, Coco)
- High-quality 3D rendered characters with smooth, polished surfaces
- Expressive characters with large eyes, soft skin, and subsurface scattering
- Clean, rounded 3D shapes with friendly character designs
- Vibrant, saturated colors with glossy, plastic-like materials
- Professional 3D rendering with realistic lighting and shadows
- Soft ambient occlusion and depth of field
- Warm, magical atmosphere with polished, cinematic quality
- Modern 3D animation aesthetic with volumetric lighting

**NEGATIVE PROMPT - STRICTLY NO 2D ELEMENTS**:
- ABSOLUTELY FORBIDDEN: 2D hand-drawn style, line art, outlines, cel-shading
- NO flat colors, NO paper textures, NO sketchy lines
- NO manga/anime 2D style, NO comic book style
- NO hand-drawn appearance, NO pencil/ink lines
- MUST be 3D rendered with depth, volume, and realistic lighting
- NO 2D-only: All elements must have 3D geometry and rendering
""",

    "anime": """
**STYLE: JAPANESE ANIME**
- Japanese anime/manga art style
- Cel-shaded rendering with flat colors
- Characteristic anime facial features and expressions
- Dynamic action poses with speed lines
- Vibrant, saturated colors
- Clean line art with distinct outlines
""",

    "clay": """
**STYLE: CLAYMATION / STOP-MOTION**
- Claymation or plasticine animation style
- Handmade, molded clay aesthetic with visible fingerprints
- Soft, malleable textures and organic shapes
- Warm, earthy color palette
- Chunky, rounded character designs
- Visible thumbprints and tool marks
""",

    "japanese_anime": """
**STYLE: JAPANESE MANGA**
- Japanese manga illustration style
- Black ink outlines with screentone shading
- Detailed line art with cross-hatching
- Dramatic expressions and dynamic angles
- Monochrome or limited color palette
- Comic panel aesthetic with speech bubbles
""",

    "family_guy": """
**STYLE: FAMILY GUY AMERICAN CARTOON**
- Family Guy-style American satirical cartoon
- Exaggerated character proportions with oversized heads
- Bold, thick black outlines with flat colors
- Expressive and exaggerated facial expressions
- Satirical and humorous character designs
- Bright, saturated colors with simple shading
- Caricature-style anatomy with exaggerated features
- Stylized, simplified backgrounds
"""
}


class MemoryAllocationAgent:
    """Memory Allocation Agent - Allocate context memory for each shot"""

    def __init__(self, script_json="clip1_script.json",
                 character_mapping="character_mapping.json",
                 reference_dir="reference_images",
                 style="realistic"):
        """
        Initialize

        Args:
            script_json: Script JSON file path
            character_mapping: Character pairing configuration file path
            reference_dir: Reference images directory
            style: Generation style (realistic, lego, disney, anime, clay, japanese_anime)
        """
        self.script_json = script_json
        self.character_mapping_file = character_mapping
        self.reference_dir = reference_dir
        self.style = style

        # Get style prompt
        self.style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["realistic"])

        # Data storage
        self.script_data = None
        self.character_mappings = []
        self.memory_store = {}  # shot_id -> memory_package

        # Cached data
        self.major_scenes_map = {}  # major_scene_id -> scene_data
        self.character_roster = {}  # character_id -> character_data
        self.character_position_mapping = None  # Character position mapping (loaded from character_position_mapping.json)

    def load_script_data(self):
        """Load script data"""
        print(f"Reading: {self.script_json}")

        with open(self.script_json, 'r', encoding='utf-8') as f:
            self.script_data = json.load(f)

        # Extract major_scenes mapping
        major_scenes = self.script_data.get("major_scenes", {}).get("major_scenes", [])
        for ms in major_scenes:
            self.major_scenes_map[ms["scene_id"]] = ms

        # Extract character roster
        character_roster = self.script_data.get("character_roster", {})
        self.character_roster = character_roster

        print(f"✅ Script data loaded")
        print(f"   - {len(self.major_scenes_map)} major scenes")
        print(f"   - {len(self.character_roster.get('characters', []))} characters")

        return self.script_data

    def load_character_mappings(self):
        """Load character pairing configuration"""
        if not os.path.exists(self.character_mapping_file):
            print(f"⚠️  Character pairing file not found: {self.character_mapping_file}")
            return False

        print(f"Reading: {self.character_mapping_file}")
        with open(self.character_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.character_mappings = data.get('mappings', [])
        print(f"✅ Loaded {len(self.character_mappings)} character pairings")

        return True

    def load_character_position_mapping(self):
        """Load character position mapping file"""
        mapping_file = "character_position_mapping.json"
        if not os.path.exists(mapping_file):
            print(f"⚠️  Character position mapping file not found: {mapping_file}")
            self.character_position_mapping = {}
            return False

        print(f"Reading character position mapping: {mapping_file}")
        with open(mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.character_position_mapping = data
        main_count = len(data.get("main_characters", {}))
        supporting_count = len(data.get("supporting_characters", {}))
        print(f"✅ Loaded character position mapping: {main_count} main characters, {supporting_count} supporting characters")
        return True

    def get_character_collage_ref(self, char_id):
        """
        Get collage reference path by character ID (by priority)

        Priority: Main character collage > Supporting character collage > Clothing image

        Returns:
            Collage file path, or None if not found
        """
        if not self.character_position_mapping:
            return None

        # 1. First look for main character collage
        main_chars = self.character_position_mapping.get("main_characters", {})
        if char_id in main_chars:
            collage_file = main_chars[char_id].get("collage_file")
            if collage_file and os.path.exists(collage_file):
                return collage_file

        # 2. Look for supporting character collage
        supporting_chars = self.character_position_mapping.get("supporting_characters", {})
        if char_id in supporting_chars:
            collage_file = supporting_chars[char_id].get("collage_file")
            if collage_file and os.path.exists(collage_file):
                return collage_file

        # 3. Fallback to individual clothing image
        clean_id = char_id.replace('@', '')
        clothing_path = os.path.join(self.reference_dir, f"{clean_id}_clothing.png")
        if os.path.exists(clothing_path):
            return clothing_path

        return None

    def detect_characters_in_shot(self, shot_data):
        """
        Detect characters present in shot data

        Args:
            shot_data: Shot data dictionary

        Returns:
            List of detected character IDs
        """
        detected_characters = []

        # Collect text content
        text_to_check = ""
        if "subject_movement" in shot_data:
            text_to_check += shot_data["subject_movement"] + " "
        if "I2V Prompt" in shot_data:
            text_to_check += shot_data["I2V Prompt"] + " "
        if "Language_to_One_Shot_Prompt" in shot_data:
            text_to_check += shot_data["Language_to_One_Shot_Prompt"]

        # Detect all characters
        for mapping in self.character_mappings:
            char_id = mapping['video_character']
            char_name = mapping.get('video_character_name', '')

            # Check if present in text (only check char_id to avoid char_name mismatches)
            if char_id in text_to_check:
                detected_characters.append(char_id)

        return detected_characters

    def get_major_scene_for_shot(self, shot_data):
        """
        Get the major_scene that the shot belongs to

        Args:
            shot_data: Shot data dictionary

        Returns:
            major_scene_id or None
        """
        start_time = shot_data.get("start_time", 0.0)
        end_time = shot_data.get("end_time", 0.0)

        for scene_id, scene_data in self.major_scenes_map.items():
            ms_start = scene_data.get("start_time", 0.0)
            ms_end = scene_data.get("end_time", 0.0)

            if start_time <= ms_end and end_time >= ms_start:
                return scene_id

        return None

    def allocate_memory_for_shot(self, shot_data):
        """
        Allocate memory package for a single shot

        Args:
            shot_data: Shot data dictionary

        Returns:
            Memory package dictionary
        """
        shot_id = shot_data.get("scene_index")
        print(f"\n{'='*70}")
        print(f"📦 Allocating memory for Shot {shot_id}")
        print(f"{'='*70}")
        print(f"Style: {self.style.upper()}")

        # 1. Basic information
        memory_package = {
            "shot_id": shot_id,
            "time_range": shot_data.get("time_range", ""),
            "start_time": shot_data.get("start_time", 0.0),
            "end_time": shot_data.get("end_time", 0.0),
            "style": self.style,
            "style_prompt": self.style_prompt  # Add style prompt
        }

        # 2. Character identity allocation
        print("\n[1/5] Character identity allocation...")
        detected_characters = self.detect_characters_in_shot(shot_data)
        memory_package["characters"] = detected_characters

        # Character pairing mapping
        character_mappings = {}
        for char_id in detected_characters:
            mapping = self.get_character_mapping(char_id)
            if mapping:
                character_mappings[char_id] = {
                    "target_name": mapping.get('target_name', ''),
                    "target_face": mapping.get('target_face', ''),
                    "physical_attributes": mapping.get('physical_attributes', ''),
                    "clothing": mapping.get('clothing', '')
                }

        memory_package["character_mappings"] = character_mappings

        if not detected_characters:
            print(f"   ⚠️  No characters detected")

        # 3. Environment anchor management
        print("\n[2/5] Environment anchor management...")
        major_scene_id = self.get_major_scene_for_shot(shot_data)
        memory_package["major_scene"] = major_scene_id

        if major_scene_id:
            env_ref_path = os.path.join(self.reference_dir, f"{major_scene_id}_environment.png")
            if os.path.exists(env_ref_path):
                memory_package["environment_ref"] = env_ref_path
                print(f"   ✅ Environment reference: {os.path.basename(env_ref_path)}")
            else:
                memory_package["environment_ref"] = ""
                print(f"   ⚠️  Environment reference image not found: {env_ref_path}")
        else:
            memory_package["environment_ref"] = ""
            print(f"   ⚠️  No associated major_scene found")

        # ========== Smart reference image allocation (strictly limited to 6: 1 environment + ≤5 portraits) ==========
        print("\n[3/5] Smart reference image allocation...")

        # Read character classification info (main characters vs supporting characters)
        mapping_file = "character_mapping.json"
        main_characters = []
        supporting_characters = []

        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                main_characters = mapping_data.get("main_character_ids", [])
                supporting_characters = mapping_data.get("supporting_character_ids", [])

        # Classify characters in scene
        main_chars_in_scene = [c for c in detected_characters if c in main_characters]
        supporting_chars_in_scene = [c for c in detected_characters if c in supporting_characters]

        # Get scene wardrobe information
        scene_wardrobe = self.script_data.get("scene_wardrobe", {}).get("scene_wardrobes", {})
        scene_wardrobe_info = scene_wardrobe.get(major_scene_id, {}) if major_scene_id else {}
        character_wardrobe = scene_wardrobe_info.get("character_wardrobe", {})

        # Reference image allocation strategy (portrait limit 5)
        portrait_quota = 5  # Portrait-type image quota
        clothing_refs = []  # Clothing reference image list
        character_refs = [] # Character collage list

        # First priority: Main character clothing reference images
        for char_id in main_chars_in_scene:
            if portrait_quota <= 0:
                break

            clean_char_id = char_id.replace('@', '')
            clothing_path = os.path.join(
                self.reference_dir,
                f"{major_scene_id}_{clean_char_id}_clothing.png"
            )

            if os.path.exists(clothing_path):
                clothing_refs.append(clothing_path)
                portrait_quota -= 1
                print(f"   ✅ [Priority 1] Allocated main character clothing: {char_id}")
            else:
                print(f"   ⚠️  Main character clothing image not found, skipping: {char_id}")

        # Second priority: Supporting character clothing reference images
        for char_id in supporting_chars_in_scene:
            if portrait_quota <= 0:
                break

            clean_char_id = char_id.replace('@', '')
            clothing_path = os.path.join(
                self.reference_dir,
                f"{major_scene_id}_{clean_char_id}_clothing.png"
            )

            if os.path.exists(clothing_path):
                clothing_refs.append(clothing_path)
                portrait_quota -= 1
                print(f"   ✅ [Priority 2] Allocated supporting character clothing: {char_id}")
            else:
                print(f"   ⚠️  Supporting character clothing image not found, skipping: {char_id}")

        # Third priority: Character collages (use remaining quota)
        # Prioritize main character collages, then supporting character collages
        if portrait_quota > 0 and main_chars_in_scene:
            main_collage = "main_characters_1.png"
            if os.path.exists(main_collage):
                character_refs.append(main_collage)
                portrait_quota -= 1
                print(f"   ✅ [Priority 3] Allocated main character collage")
            else:
                print(f"   ⚠️  Main character collage not found, skipping")

        if portrait_quota > 0 and supporting_chars_in_scene:
            supporting_collage = "supporting_characters_1.png"
            if os.path.exists(supporting_collage):
                character_refs.append(supporting_collage)
                portrait_quota -= 1
                print(f"   ✅ [Priority 3] Allocated supporting character collage")
            else:
                print(f"   ⚠️  Supporting character collage not found, skipping")

        # Print allocation result statistics
        total_images = 1 + len(clothing_refs) + len(character_refs)
        portrait_count = len(clothing_refs) + len(character_refs)

        print(f"\n   📊 Reference image allocation results:")
        print(f"      Environment images: 1")
        print(f"      Clothing images: {len(clothing_refs)}")
        print(f"      Collage images: {len(character_refs)}")
        print(f"      Total: {total_images} (limit 6)")
        print(f"      Portrait-type: {portrait_count} (limit 5)")

        memory_package["clothing_refs"] = clothing_refs
        memory_package["character_refs"] = character_refs

        # 4. Visual DNA maintenance
        print("\n[4/5] Visual DNA maintenance...")
        visual_dna = {
            "lighting": shot_data.get("lighting_setup", ""),
            "color": shot_data.get("color_grading", ""),
            "mood": shot_data.get("mood_atmosphere", ""),
            "shot_size": shot_data.get("shot_size", ""),
            "camera_angle": shot_data.get("camera_angle", ""),
            "camera_height": shot_data.get("camera_height", ""),
            "focal_length": shot_data.get("focal_length", ""),
            "depth_of_field": shot_data.get("depth_of_field", ""),
        }

        memory_package["visual_dna"] = visual_dna
        print(f"   ✅ Lighting: {visual_dna['lighting'][:50]}...")
        print(f"   ✅ Color: {visual_dna['color'][:50]}...")
        print(f"   ✅ Mood: {visual_dna['mood'][:50]}...")
        print(f"   ✅ Shot size: {visual_dna['shot_size']}")

        # 5. Narrative context extraction
        print("\n[5/5] Narrative context extraction...")
        narrative = {
            "action": shot_data.get("subject_movement", ""),
            "camera_movement": shot_data.get("camera_movement", ""),
            "duration": shot_data.get("duration", 0.0),
            "i2v_prompt": shot_data.get("I2V Prompt", ""),
            "language_prompt": shot_data.get("Language_to_One_Shot_Prompt", "")
        }

        memory_package["narrative"] = narrative
        print(f"   ✅ Action: {narrative['action'][:60]}...")
        print(f"   ✅ Duration: {narrative['duration']}s")

        print(f"\n✅ Shot {shot_id} memory allocation complete")

        return memory_package

    def get_character_mapping(self, character_id):
        """Get character pairing information"""
        for mapping in self.character_mappings:
            if mapping['video_character'] == character_id:
                return mapping
        return None

    def get_character_shot_type(self, char_id, shot_data):
        """
        Get shot type for character in current scene

        Args:
            char_id: Character ID (currently unused, all characters share scene's shot type)
            shot_data: Shot data

        Returns:
            Shot type string: "CLOSE_UP", "MEDIUM_SHOT", "FULL_BODY_SHOT", or "UNKNOWN"
        """
        # TODO: Read face_type field from shot_data
        # Temporarily return UNKNOWN, will implement after scene analysis adds face_type field
        return "UNKNOWN"

    def allocate_all_memory(self):
        """Allocate memory for all shots"""
        print(f"\n{'='*70}")
        print("🎬 Starting memory allocation for all shots")
        print(f"{'='*70}")

        # Load character position mapping (for priority allocation)
        self.load_character_position_mapping()

        scenes = self.script_data.get("scenes", [])
        active_scenes = [s for s in scenes if not s.get("_disabled", False)]

        print(f"Found {len(active_scenes)} valid shots\n")

        for scene in active_scenes:
            shot_id = scene.get("scene_index")
            memory_package = self.allocate_memory_for_shot(scene)
            self.memory_store[shot_id] = memory_package

        print(f"\n{'='*70}")
        print(f"✅ Memory allocation complete!")
        print(f"   Total: {len(self.memory_store)} shots")
        print(f"{'='*70}")

        return self.memory_store

    def get_shot_memory(self, shot_id):
        """
        Get memory package for specific shot

        Args:
            shot_id: Shot ID (can be string, integer, or float)

        Returns:
            Memory package dictionary, or None if not found
        """
        # First try raw type
        if shot_id in self.memory_store:
            return self.memory_store[shot_id]

        # If not found, try converting to string and search
        shot_id_str = str(shot_id)
        if shot_id_str in self.memory_store:
            return self.memory_store[shot_id_str]

        # If numeric string, try converting to integer or float
        try:
            shot_id_int = int(shot_id)
            if shot_id_int in self.memory_store:
                return self.memory_store[shot_id_int]
        except (ValueError, TypeError):
            pass

        try:
            shot_id_float = float(shot_id)
            if shot_id_float in self.memory_store:
                return self.memory_store[shot_id_float]
        except (ValueError, TypeError):
            pass

        # Not found, return None
        return None

    def save_memory_allocation(self, output_file="memory_allocation.json"):
        """Save memory allocation to JSON file"""
        print(f"\nSaving memory allocation: {output_file}")

        output_data = {
            "metadata": {
                "script_json": self.script_json,
                "character_mapping": self.character_mapping_file,
                "reference_dir": self.reference_dir,
                "created_at": datetime.now().isoformat(),
                "total_shots": len(self.memory_store)
            },
            "memory_store": self.memory_store
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved: {output_file}")

    def load_memory_allocation(self, input_file="memory_allocation.json"):
        """Load memory allocation from JSON file"""
        if not os.path.exists(input_file):
            print(f"⚠️  Memory allocation file not found: {input_file}")
            return False

        print(f"Loading memory allocation: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.memory_store = data.get('memory_store', {})
        print(f"✅ Loaded memory allocation for {len(self.memory_store)} shots")

        return True


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Memory Allocation Agent - Allocate context memory for shots',
        epilog='''
Usage examples:
  python %(prog)s                           # Use default clip1_script.json
  python %(prog)s clip2.json                 # Specify other script file
  python %(prog)s clip1.json --shot 9        # Allocate memory for specific shot
        '''
    )

    parser.add_argument(
        'script',
        nargs='?',
        default='clip1_script.json',
        help='Script JSON file path (default: clip1_script.json)'
    )

    parser.add_argument('--mapping', default='character_mapping.json', help='Character pairing configuration file')
    parser.add_argument('--reference', default='reference_images', help='Reference images directory')
    parser.add_argument('--shot', type=int, help='Allocate memory for specific shot ID')
    parser.add_argument('--output', default='memory_allocation.json', help='Output JSON file path')
    parser.add_argument('--style',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       default='realistic',
                       help='Generation style: realistic=realistic (default), lego=LEGO, disney=Disney, anime=anime, clay=claymation, japanese_anime=Japanese manga, family_guy=Family Guy cartoon')

    args = parser.parse_args()

    try:
        # Initialize Agent
        agent = MemoryAllocationAgent(
            script_json=args.script,
            character_mapping=args.mapping,
            reference_dir=args.reference,
            style=args.style
        )

        # Load data
        agent.load_script_data()
        agent.load_character_mappings()

        # Execute allocation
        if args.shot:
            # Allocate for single shot
            scenes = agent.script_data.get("scenes", [])
            shot_data = None
            for scene in scenes:
                if scene.get("scene_index") == args.shot:
                    shot_data = scene
                    break

            if shot_data:
                memory = agent.allocate_memory_for_shot(shot_data)
                agent.memory_store[args.shot] = memory
            else:
                print(f"❌ Shot {args.shot} does not exist")
                return 1
        else:
            # Allocate for all shots
            agent.allocate_all_memory()

        # Save results
        agent.save_memory_allocation(args.output)

        print(f"\n{'='*70}")
        print("✅ Memory Allocation complete!")
        print(f"{'='*70}")
        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
