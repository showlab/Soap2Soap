#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene Reference Image Generation Agent

Features:
1. Generate environment reference images for each major_scene (pure environment, no people)
2. Generate clothing reference images for each character (showing clothing DNA)
3. Save to reference_images/ directory for use by subsequent Agents

Usage:
    python agent_reference.py
"""

import os
import sys
import json
from pathlib import Path

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from google import genai
    from google.genai import types
    from PIL import Image
except ImportError:
    print("Error: Required libraries google-genai and Pillow need to be installed")
    print("Please run: pip install google-genai Pillow")
    sys.exit(1)


# Import style prompts from agent_memory.py
try:
    from agent_memory import STYLE_PROMPTS
except ImportError:
    STYLE_PROMPTS = {}

class ReferenceAgent:
    """Scene Reference Image Generation Agent"""

    def __init__(self, script_json="clip1_script.json", output_dir="reference_images", style="realistic"):
        """
        Initialize

        Args:
            script_json: Script JSON file path
            output_dir: Reference image output directory
            style: Generation style (realistic, lego, disney, anime, clay, japanese_anime, family_guy)
        """
        self.script_json = script_json
        self.output_dir = output_dir
        self.style = style

        # Get style prompt
        self.style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS.get("realistic", ""))

        # Initialize Gemini client
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: Environment variable GENAI_API_KEY not found")

        self.client = genai.Client(api_key=api_key)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # ========== Retry Configuration ==========
        self.max_retries = 3  # Maximum number of retries
        self.retry_delay = 5  # Retry interval (seconds)

        # ========== Read Video Aspect Ratio Metadata ==========
        self.target_width = None
        self.target_height = None
        self.aspect_ratio = None

        # Load script_data to read video_metadata
        if os.path.exists(self.script_json):
            try:
                with open(self.script_json, 'r', encoding='utf-8') as f:
                    script_data = json.load(f)

                video_metadata = script_data.get("video_metadata", {})
                aspect_ratio_info = video_metadata.get("aspect_ratio", {})

                if aspect_ratio_info:
                    self.target_width = aspect_ratio_info.get("width", 1920)
                    self.target_height = aspect_ratio_info.get("height", 1080)
                    self.aspect_ratio = aspect_ratio_info.get("aspect_ratio", "16:9")

                    print(f"\n{'='*60}")
                    print(f"--> Loading aspect ratio information from video metadata:")
                    print(f"{'='*60}")
                    print(f"    Resolution: {self.target_width}x{self.target_height}")
                    print(f"    Aspect Ratio: {self.aspect_ratio}")
                    print(f"{'='*60}")
                else:
                    # If no metadata, use default value 16:9
                    print(f"\n⚠️  Video aspect ratio metadata not found, using default value 16:9")
                    self.target_width = 1920
                    self.target_height = 1080
                    self.aspect_ratio = "16:9"
            except Exception as e:
                print(f"\n⚠️  Failed to read video metadata: {e}, using default value 16:9")
                self.target_width = 1920
                self.target_height = 1080
                self.aspect_ratio = "16:9"
        else:
            print(f"\n⚠️  Script file does not exist, using default value 16:9")
            self.target_width = 1920
            self.target_height = 1080
            self.aspect_ratio = "16:9"

    def load_script_data(self):
        """Load script data"""
        print(f"Reading: {self.script_json}")

        with open(self.script_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def generate_environment_references(self, script_data):
        """Generate environment reference images for all major_scenes"""

        major_scenes = script_data.get("major_scenes", {}).get("major_scenes", [])

        print(f"\nFound {len(major_scenes)} major scenes")
        print("="*70)

        for scene in major_scenes:
            scene_id = scene.get("scene_id")
            env_desc = scene.get("environment_description", "")

            print(f"\nGenerating environment reference image: {scene_id}")
            print(f"  Description: {env_desc[:100]}...")

            self._generate_environment_image(scene_id, env_desc)

    def _generate_environment_image(self, scene_id, description):
        """Generate environment reference image for a single scene"""

        # Build prompt, inject style prompt
        prompt = f"""
Generate a HIGH-QUALITY ENVIRONMENT-ONLY reference image for a video scene.

{self.style_prompt}

**CRITICAL: ENVIRONMENT ONLY - NO CHARACTERS**
This image must contain ONLY the background environment (room, furniture, lighting, etc.).
DO NOT include any people, characters, faces, or bodies.

**SCENE DESCRIPTION**:
{description}

**IMAGE REQUIREMENTS**:
- FORMAT: {self.aspect_ratio} aspect ratio ({self.target_width}x{self.target_height})
- EMPTY ROOM: Show the room as if no one is there
- WIDE SHOT: Use a wide camera angle showing full room layout
- PROPER LIGHTING: Natural lighting as described
- REFERENCE QUALITY: High detail, accurate to description
- NO TEXT/GRAPHICS: No overlays, subtitles, or any visible text on image

**CRITICAL: TEXT ELIMINATION**:
- ABSOLUTELY NO text, subtitles, watermarks, or visible writing
- DO NOT render any descriptive text or labels from the scene description
- The image must be completely text-free
- Ignore any dialogue, quotes, or speech content in description - render ONLY visual environment

Generate a clean, high-quality environment image that captures the empty space described above.
"""

        output_file = os.path.join(self.output_dir, f"{scene_id}_environment.png")

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"  🔄 Attempt {attempt + 1}...")
                else:
                    print(f"  Calling Gemini to generate...")

                response = self.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                    )
                )

                # Check if response is valid
                if response is None:
                    print(f"  ⚠️  API returned empty response")
                    if attempt < self.max_retries - 1:
                        print(f"  🔄 Retrying in {self.retry_delay} seconds...")
                        import time
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        print(f"  ❌ Maximum retries reached, generation failed")
                        return

                # Save image
                saved = False
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if part.inline_data:
                                    image_bytes = part.inline_data.data
                                    import io
                                    image = Image.open(io.BytesIO(image_bytes))

                                    # Resize to standard dimensions (using dimensions read from video metadata)
                                    image = image.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
                                    image.save(output_file)

                                    saved = True
                                    print(f"  ✅ Saved: {output_file}")
                                    print(f"     Image size: {self.target_width}x{self.target_height}")
                                    print(f"     Target aspect ratio: {self.aspect_ratio}")
                                    break
                        if saved:
                            break

                if saved:
                    # Successfully generated, exit retry loop
                    break
                else:
                    print(f"  ⚠️  No valid image data found")
                    if attempt < self.max_retries - 1:
                        print(f"  🔄 Retrying in {self.retry_delay} seconds...")
                        import time
                        time.sleep(self.retry_delay)
                    else:
                        print(f"  ❌ Maximum retries reached, generation failed")

            except Exception as e:
                print(f"  ❌ Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"  🔄 Retrying in {self.retry_delay} seconds...")
                    import time
                    time.sleep(self.retry_delay)
                else:
                    print(f"  ❌ Maximum retries reached, giving up")
                    import traceback
                    print(f"  📋 Last error details: {traceback.format_exc()}")

    def generate_clothing_references(self, script_data, character_mapping_file="character_mapping.json", memory_allocation_file="memory_allocation.json"):
        """Generate clothing reference images for each character in each scene (by scene+character combination)"""

        # Load character pairing information
        character_mappings = []
        if os.path.exists(character_mapping_file):
            with open(character_mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                character_mappings = data.get('mappings', [])

        # Get scene wardrobe information
        scene_wardrobe = script_data.get("scene_wardrobe", {}).get("scene_wardrobes", {})

        # Get all major_scenes
        major_scenes = script_data.get("major_scenes", {}).get("major_scenes", [])

        if not character_mappings:
            print("\n⚠️  No character pairing information found, skipping clothing reference generation")
            print("   Please run first: python create_character_sheet.py")
            return

        if not major_scenes:
            print("\n⚠️  No major scene information found, skipping clothing reference generation")
            return

        print(f"\nFound {len(major_scenes)} major scenes")
        print(f"Found {len(character_mappings)} character pairings")
        print("="*70)

        # Determine whether to use copy mode (family_guy and lego styles)
        use_copy_mode = self.style in ['family_guy', 'lego']

        if use_copy_mode:
            print(f"\n📋 Style '{self.style}' uses copy reference image mode")
            print("   Will directly copy existing reference images, not calling Gemini for generation")
            print("="*70)

        # Iterate through each major_scene
        for scene in major_scenes:
            scene_id = scene.get("scene_id")

            # Get wardrobe information for this scene
            scene_wardrobe_info = scene_wardrobe.get(scene_id, {})
            character_wardrobe = scene_wardrobe_info.get("character_wardrobe", {})

            # Get the list of characters in this scene from scene_wardrobe's character_wardrobe
            characters_in_scene = list(character_wardrobe.keys())

            if not characters_in_scene:
                print(f"\n⚠️  Scene {scene_id} has no character information, skipping")
                continue

            print(f"\nScene {scene_id} contains characters: {', '.join(characters_in_scene)}")

            # Generate clothing reference images for each character in this scene
            for char_id in characters_in_scene:
                # Find the pairing information for this character (face reference image)
                matching_mapping = None
                for mapping in character_mappings:
                    if mapping['video_character'] == char_id:
                        matching_mapping = mapping
                        break

                if matching_mapping:
                    if use_copy_mode:
                        # family_guy and lego styles: directly copy reference images
                        print(f"\nCopying reference image: {scene_id} - {char_id}")
                        self._copy_reference_image(
                            scene_id,
                            char_id,
                            matching_mapping
                        )
                    else:
                        # Other styles: call Gemini to generate
                        print(f"\nGenerating scene clothing reference image: {scene_id} - {char_id}")
                        self._generate_scene_character_clothing_image(
                            scene_id,
                            char_id,
                            matching_mapping,
                            character_wardrobe
                        )
                else:
                    print(f"\n⚠️  Character {char_id} has no pairing information, skipping")

    def _copy_reference_image(self, scene_id, char_id, mapping):
        """
        Copy existing reference image as scene clothing reference image (for family_guy and lego styles only)

        Args:
            scene_id: Scene ID (e.g., major_scene_1)
            char_id: Character ID (e.g., @character_01)
            mapping: Character pairing information (used to get face reference image path)
        """
        import shutil

        target_face_path = mapping['target_face']
        clean_char_id = char_id.replace('@', '')

        # Output filename: {scene_id}_{clean_char_id}_clothing.png
        output_file = os.path.join(self.output_dir, f"{scene_id}_{clean_char_id}_clothing.png")

        try:
            # Check if source file exists
            if not os.path.exists(target_face_path):
                print(f"  ❌ Reference image does not exist: {target_face_path}")
                return

            # Copy file
            shutil.copy2(target_face_path, output_file)

            # Resize dimensions (keep consistent with generated images)
            image = Image.open(output_file)
            image = image.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
            image.save(output_file)

            print(f"  ✅ Copied: {os.path.basename(target_face_path)}")
            print(f"     → {output_file}")
            print(f"     Image size: {self.target_width}x{self.target_height}")
            print(f"     Target aspect ratio: {self.aspect_ratio}")

        except Exception as e:
            print(f"  ❌ Copy failed: {e}")

    def _generate_scene_character_clothing_image(self, scene_id, char_id, mapping, character_wardrobe):
        """
        Generate clothing reference image for a single character in a single scene

        Args:
            scene_id: Scene ID (e.g., major_scene_1)
            char_id: Character ID (e.g., @character_01)
            mapping: Character pairing information (used to get face reference image)
            character_wardrobe: Wardrobe information dictionary for this scene {char_id: {full_description: "..."}}
        """

        target_name = mapping['target_name']
        target_face_path = mapping['target_face']

        # Get the character's clothing description for this scene from scene_wardrobe
        char_wardrobe_info = character_wardrobe.get(char_id, {})
        clothing_desc = char_wardrobe_info.get('full_description',
                                              'Professional business attire')

        # Build prompt, inject style prompt
        prompt = f"""
Generate a FULL CHARACTER REFERENCE image showing a person wearing the specified clothing.

{self.style_prompt}

**SCENE**: {scene_id}
**CHARACTER INFORMATION**:
- Character Name: {target_name}
- Target Face Reference: will use {target_face_path}

**CLOTHING DESCRIPTION**:
{clothing_desc}

**IMAGE REQUIREMENTS**:
- FORMAT: {self.aspect_ratio} aspect ratio ({self.target_width}x{self.target_height})
- FULL BODY SHOT: Show the character from head to toe wearing the outfit
- FACE + CLOTHING: Must include BOTH the face wearing the clothing AND the full outfit details
- NEUTRAL BACKGROUND: Plain background that doesn't distract (white/gray/gradient)
- HIGH QUALITY: Detailed fabric textures, colors, and facial features
- REFERENCE STYLE: Like a character design sheet or movie poster

**CRITICAL**:
- Show the FULL CHARACTER with face and complete outfit
- Face must match the reference image provided
- Clothing must be clearly visible and accurate to description
- Professional photography style
- Accurate colors, textures, and facial features
- ABSOLUTELY NO text, labels, or visible writing on the image
- The image must be completely text-free

Generate a high-quality full-body character reference that shows exactly what this character looks like wearing this outfit.
"""

        # Output filename: {scene_id}_{clean_char_id}_clothing.png
        clean_char_id = char_id.replace('@', '')
        output_file = os.path.join(self.output_dir, f"{scene_id}_{clean_char_id}_clothing.png")

        # Prepare input content (prompt + foreign face reference image)
        input_contents = [prompt]

        # Load foreign face reference image
        if os.path.exists(target_face_path):
            try:
                face_image = Image.open(target_face_path)
                input_contents.append(face_image)
                print(f"  ✅ Loaded foreign face reference: {os.path.basename(target_face_path)}")
            except Exception as e:
                print(f"  ⚠️  Unable to load foreign face: {e}")

        # Retry loop
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    print(f"  🔄 Attempt {attempt + 1}...")
                else:
                    print(f"  Calling Gemini to generate...")

                response = self.client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=input_contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['IMAGE'],
                    )
                )

                # Check if response is valid
                if response is None:
                    print(f"  ⚠️  API returned empty response")
                    if attempt < self.max_retries - 1:
                        print(f"  🔄 Retrying in {self.retry_delay} seconds...")
                        import time
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        print(f"  ❌ Maximum retries reached, generation failed")
                        return

                # Save image
                saved = False
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if part.inline_data:
                                    image_bytes = part.inline_data.data
                                    import io
                                    image = Image.open(io.BytesIO(image_bytes))

                                    # Resize to standard dimensions (using dimensions read from video metadata)
                                    image = image.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
                                    image.save(output_file)

                                    saved = True
                                    print(f"  ✅ Saved: {output_file}")
                                    print(f"     Image size: {self.target_width}x{self.target_height}")
                                    print(f"     Target aspect ratio: {self.aspect_ratio}")
                                    break
                        if saved:
                            break

                if saved:
                    # Successfully generated, exit retry loop
                    break
                else:
                    print(f"  ⚠️  No valid image data found")
                    if attempt < self.max_retries - 1:
                        print(f"  🔄 Retrying in {self.retry_delay} seconds...")
                        import time
                        time.sleep(self.retry_delay)
                    else:
                        print(f"  ❌ Maximum retries reached, generation failed")

            except Exception as e:
                print(f"  ❌ Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    print(f"  🔄 Retrying in {self.retry_delay} seconds...")
                    import time
                    time.sleep(self.retry_delay)
                else:
                    print(f"  ❌ Maximum retries reached, giving up")
                    import traceback
                    print(f"  📋 Last error details: {traceback.format_exc()}")

    def generate_all_references(self):
        """Generate all reference images (environment + clothing)"""
        print("\n" + "="*70)
        print("📐 Scene Reference Image Generation Agent")
        print("="*70)
        print(f"Style: {self.style}")

        # Load script data
        script_data = self.load_script_data()

        # Generate environment reference images (for all styles)
        self.generate_environment_references(script_data)

        # Generate clothing reference images (for all styles, but family_guy and lego use copy mode)
        self.generate_clothing_references(script_data)

        print("\n" + "="*70)
        print("✅ Reference image generation completed!")
        print(f"   Output directory: {self.output_dir}/")
        print("="*70)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Scene Reference Image Generation Agent - Generate environment and clothing reference images',
        epilog='''
Usage examples:
  python %(prog)s                                      # Use default clip1_script.json
  python %(prog)s clip2.json                            # Specify other script file
  python %(prog)s clip3.json --output refs/             # Specify output directory
  python %(prog)s clip1.json --style disney             # Use Disney style
        '''
    )

    # Positional argument: script JSON file
    parser.add_argument(
        'script',
        nargs='?',
        default='clip1_script.json',
        help='Script JSON file path (default: clip1_script.json)'
    )

    parser.add_argument('--output', default='reference_images', help='Reference image output directory')

    parser.add_argument('--style',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       default='realistic',
                       help='Generation style: realistic=realistic (default), lego=Lego, disney=Disney, anime=anime, clay=clay, japanese_anime=Japanese anime, family_guy=Family Guy American cartoon')

    args = parser.parse_args()

    try:
        agent = ReferenceAgent(
            script_json=args.script,
            output_dir=args.output,
            style=args.style
        )

        agent.generate_all_references()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
