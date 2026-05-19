#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image/Video Generation Agent

Features:
1. Integrates existing image generation and video generation functionality
2. Injects foreign face references and clothing descriptions before generation
3. Uses target foreign faces for generation

Usage:
    python agent_generation.py clip1_script.json              # Generate all keyframes
    python agent_generation.py clip1_script.json --mode video # Generate all videos
    python agent_generation.py clip1_script.json --shot 9     # Generate specific shot
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from google import genai
    from google.genai import types
    from PIL import Image
except ImportError:
    print("Error: Required libraries google-genai and Pillow not installed")
    print("Please run: pip install google-genai Pillow")
    sys.exit(1)


class GenerationAgent:
    """Image/Video Generation Agent"""

    def __init__(self, script_json="clip1_script.json", character_mapping="character_mapping.json",
                 reference_dir="reference_images", memory_agent=None, style="realistic"):
        """
        Initialize

        Args:
            script_json: Script JSON file path
            character_mapping: Character mapping configuration file path
            reference_dir: Reference image directory
            memory_agent: MemoryAllocationAgent instance (optional)
            style: Generation style (realistic, lego, disney, anime, clay, japanese_anime)
        """
        self.script_json = script_json
        self.character_mapping_file = character_mapping
        self.reference_dir = reference_dir
        self.memory_agent = memory_agent
        self.style = style

        # Initialize Gemini client
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: Environment variable GENAI_API_KEY not found")

        self.client = genai.Client(api_key=api_key)

        # Load data
        self.script_data = None
        self.character_mappings = []
        self.shots_data = []

        # Generation configuration - will be dynamically read from video metadata
        self.target_width = None
        self.target_height = None
        self.aspect_ratio = None
        self.duration_seconds = 8

        # Save error information (for intelligent review agent)
        self.last_errors = {}

    def load_script_data(self):
        """Load script data"""
        print(f"Reading: {self.script_json}")

        with open(self.script_json, 'r', encoding='utf-8') as f:
            self.script_data = json.load(f)

        # ========== Stage 2: Read video aspect ratio metadata ==========
        video_metadata = self.script_data.get("video_metadata", {})
        aspect_ratio_info = video_metadata.get("aspect_ratio", {})

        if aspect_ratio_info:
            self.target_width = aspect_ratio_info.get("width", 1920)
            self.target_height = aspect_ratio_info.get("height", 1080)
            self.aspect_ratio = aspect_ratio_info.get("aspect_ratio", "16:9")

            print(f"\n{'='*60}")
            print(f"--> Loaded aspect ratio info from video metadata:")
            print(f"{'='*60}")
            print(f"    Resolution: {self.target_width}x{self.target_height}")
            print(f"    Aspect Ratio: {self.aspect_ratio}")
            print(f"{'='*60}")
        else:
            # If no metadata, use default values
            print(f"\n⚠️  Video aspect ratio metadata not found, using default 16:9")
            self.target_width = 1920
            self.target_height = 1080
            self.aspect_ratio = "16:9"

        # Extract scene data
        major_scenes = self.script_data.get("major_scenes", {}).get("major_scenes", [])

        # Create major_scene_map
        self.major_scene_map = {}
        for ms in major_scenes:
            self.major_scene_map[ms["scene_id"]] = ms

        # Convert to shots_data format
        self.shots_data = []
        for scene in self.script_data.get("scenes", []):
            if scene.get("_disabled", False):
                continue

            scene_id = scene.get("scene_index", 0)
            start_time = scene.get("start_time", 0.0)

            # Find the major_scene this belongs to
            current_major_scene_id = None
            for ms in major_scenes:
                if ms["start_time"] <= start_time <= ms["end_time"]:
                    current_major_scene_id = ms["scene_id"]
                    break

            json_content = {
                "lighting_setup": scene.get("lighting_setup", ""),
                "color_grading": scene.get("color_grading", ""),
                "composition": scene.get("composition", ""),
                "mood_atmosphere": scene.get("mood_atmosphere", ""),
                "shot_size": scene.get("shot_size", ""),
                "camera_angle": scene.get("camera_angle", ""),
                "camera_height": scene.get("camera_height", ""),
                "horizontal_angle": scene.get("horizontal_angle", ""),
                "focal_length": scene.get("focal_length", ""),
                "depth_of_field": scene.get("depth_of_field", ""),
                "tech_device": scene.get("tech_device", ""),
                "camera_movement": scene.get("camera_movement", ""),
                "subject_movement": scene.get("subject_movement", ""),
                "duration": scene.get("duration", 0.0),
                "t2i_prompt": scene.get("I2V Prompt", ""),
                "language_to_one_shot": scene.get("Language_to_One_Shot_Prompt", ""),
                "time_range": scene.get("time_range", ""),
                "start_time": start_time,
                "end_time": scene.get("end_time", 0.0),
                "major_scene_id": current_major_scene_id
            }

            self.shots_data.append({
                "id": scene_id,
                "json_content": json_content
            })

        print(f"✅ Loaded {len(self.shots_data)} shots")

        return self.script_data

    def load_character_mappings(self):
        """Load character mapping configuration"""
        if not os.path.exists(self.character_mapping_file):
            print(f"⚠️  Character mapping file not found: {self.character_mapping_file}")
            return False

        print(f"Reading: {self.character_mapping_file}")
        with open(self.character_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.character_mappings = data.get('mappings', [])
        print(f"✅ Loaded {len(self.character_mappings)} character mappings")

        return True

    def load_memory_allocation(self):
        """Load or create memory allocation"""
        if self.memory_agent is None:
            # If no memory_agent is passed in, create one
            from agent_memory import MemoryAllocationAgent
            self.memory_agent = MemoryAllocationAgent(
                script_json=self.script_json,
                character_mapping=self.character_mapping_file,
                reference_dir=self.reference_dir
            )

        # Try to load existing memory allocation
        if os.path.exists("memory_allocation.json"):
            print("Loading memory allocation...")
            success = self.memory_agent.load_memory_allocation()
            if success:
                return True

        # If no existing allocation, allocate new memory
        print("Allocating new memory...")
        self.memory_agent.load_script_data()
        self.memory_agent.load_character_mappings()
        self.memory_agent.allocate_all_memory()
        self.memory_agent.save_memory_allocation()

        return True

    def get_character_mapping(self, character_id):
        """
        Get mapping information by character ID

        Args:
            character_id: Character ID (e.g., @character_01)

        Returns:
            Mapping information dictionary, or None if not found
        """
        for mapping in self.character_mappings:
            if mapping['video_character'] == character_id:
                return mapping
        return None

    def build_enhanced_prompt(self, shot_data, character_refs, clothing_refs):
        """
        Build enhanced generation prompt

        Args:
            shot_data: Shot data
            character_refs: List of character reference image paths
            clothing_refs: List of clothing reference image paths

        Returns:
            Complete prompt string
        """
        content_json = shot_data["json_content"]
        major_scene_id = content_json.get("major_scene_id")

        # Basic JSON content
        prompt_str = json.dumps(content_json, indent=2)

        # Composition rules
        composition_rules = f"""

**CRITICAL COMPOSITION RULES (MUST FOLLOW)**:
0. **ASPECT RATIO REQUIREMENT**:
   - MUST generate image with EXACT aspect ratio: {self.aspect_ratio}
   - Target resolution: {self.target_width}x{self.target_height}
   - CRITICAL: All generated content MUST maintain this aspect ratio

1. **SINGLE SHOT ONLY - ABSOLUTE REQUIREMENT**:
   - Generate ONLY ONE continuous shot/frame
   - DO NOT generate multiple images arranged together
   - DO NOT create split-screen compositions
   - DO NOT create diptych, triptych, or grid layouts
   - DO NOT arrange multiple shots side-by-side or stacked
   - MUST be a single, unified scene

2. **NO TEXT/GRAPHICS - ABSOLUTE PROHIBITION**:
   - DO NOT include any text overlays, subtitles, titles, captions
   - DO NOT add watermarks, logos, or signatures
   - DO NOT include visible text anywhere in the image
   - DO NOT add graphics, arrows, or UI elements
   - DO NOT include dialogue bubbles or text boxes
   - DO NOT render any dialogue, speech, or conversation as visible text
   - The image must be completely text-free
   - IGNORE any dialogue quotes or speech content in descriptions - these are for context only, NOT to be rendered as text

3. **PURE CINEMATIC SCENE**:
   - Focus purely on the cinematic scene itself
   - Show only the environment, characters, lighting, and action
   - No artificial composition layouts or frames
"""

        # Detect shot type and add targeted composition guidance
        shot_size = content_json.get("shot_size", "")
        shot_size_lower = shot_size.lower()

        if any(term in shot_size_lower for term in ["extreme close-up", "ecu", "close-up", "cu", "medium close-up", "mcu"]):
            composition_guidance = f"""
**COMPOSITION GUIDANCE FOR CLOSE-UP SHOTS**:
- CRITICAL: The subject MUST occupy the ENTIRE {self.aspect_ratio} frame
- DO NOT leave empty space above head or below shoulders
- Frame the subject TIGHTLY from edge to edge
"""
        elif any(term in shot_size_lower for term in ["wide shot", "wide", "establishing", "extreme wide"]):
            composition_guidance = f"""
**COMPOSITION GUIDANCE FOR WIDE SHOTS**:
- CRITICAL: The scene MUST fill the ENTIRE {self.aspect_ratio} frame
- Show the full environment but keep subjects LARGE enough to be visible
- NO empty sky areas or blank floor space
"""
        else:
            composition_guidance = f"""
**COMPOSITION GUIDANCE**:
- CRITICAL: The frame MUST be completely filled in {self.aspect_ratio} format
- Subject(s) should occupy 70-90% of the frame
- NO empty or wasted space anywhere
"""

        # Character and clothing consistency guidance
        consistency_instruction = f"""

**CRITICAL CONSISTENCY INSTRUCTIONS**:
This shot belongs to Major Scene: {major_scene_id}

**CHARACTER FACE REFERENCES**:
You have been provided with reference images showing the target faces to use.
- Use these for: Exact facial features, hairstyles, skin tones
- CRITICAL: The generated characters MUST match these face references

**CLOTHING DNA COMPLIANCE**:
The scene wardrobe contains EXACT clothing DNA specifications. You MUST follow ALL dimensions:

**EXACT CLOTHING SPECIFICATIONS FOR THIS SCENE**:
"""

        # Inject clothing descriptions
        if major_scene_id:
            scene_wardrobe = self.script_data.get("scene_wardrobe", {}).get("scene_wardrobes", {})
            if major_scene_id in scene_wardrobe:
                scene_wardrobe_info = scene_wardrobe[major_scene_id]
                character_wardrobe_list = scene_wardrobe_info.get('character_wardrobe', {})

                for char_id, wardrobe_data in character_wardrobe_list.items():
                    full_desc = wardrobe_data.get('full_description', '')
                    if full_desc:
                        # Check if there is a character mapping
                        mapping = self.get_character_mapping(f"@{char_id}")
                        if mapping:
                            target_name = mapping.get('target_name', 'Unknown')
                            consistency_instruction += f"\n**{char_id} (Face: {target_name})**:\n{full_desc}\n"
                        else:
                            consistency_instruction += f"\n**{char_id}**:\n{full_desc}\n"

        consistency_instruction += """

**ZERO TOLERANCE FOR VARIATIONS**:
- All shots in this major scene MUST use identical clothing DNA
- NO frame-to-frame variations allowed
- Consistency is MANDATORY
"""

        prompt_str = composition_rules + composition_guidance + consistency_instruction + "\n\nSHOT SPECIFICS:\n" + prompt_str

        return prompt_str

    def detect_characters_in_shot(self, shot_data):
        """
        Detect characters appearing in shot

        Args:
            shot_data: Shot data

        Returns:
            List of detected character IDs
        """
        content_json = shot_data["json_content"]
        text_to_check = ""

        if "subject_movement" in content_json:
            text_to_check += content_json["subject_movement"] + " "
        if "t2i_prompt" in content_json:
            text_to_check += content_json["t2i_prompt"]

        detected_characters = []

        for mapping in self.character_mappings:
            char_id = mapping['video_character']
            char_name = mapping.get('video_character_name', '')

            # Check if appears in text
            if char_id in text_to_check or char_name in text_to_check:
                detected_characters.append(char_id)

        return detected_characters

    def generate_image_for_shot(self, shot_id):
        """
        Generate keyframe image for a single shot

        Args:
            shot_id: Shot ID

        Returns:
            True on success, False on failure
        """
        # Get memory package
        if self.memory_agent:
            memory_package = self.memory_agent.get_shot_memory(shot_id)
            if not memory_package:
                print(f"❌ Memory package for Shot {shot_id} does not exist")
                return False
            use_memory = True
        else:
            use_memory = False
            # Find shot data
            shot_data = None
            for shot in self.shots_data:
                if shot["id"] == shot_id:
                    shot_data = shot
                    break

            if not shot_data:
                print(f"❌ Shot {shot_id} does not exist")
                return False

        print(f"\n{'='*70}")
        print(f"Generating keyframe for Shot {shot_id}")
        print(f"{'='*70}")

        # Delete old image (to avoid detecting previously generated files)
        old_image_path = f"shot_{shot_id}.png"
        if os.path.exists(old_image_path):
            try:
                os.remove(old_image_path)
                print(f"🗑️  Deleted old image: {old_image_path}")
            except Exception as e:
                print(f"⚠️  Unable to delete old image: {e}")

        # Prepare reference images
        ref_images = []

        if use_memory:
            # Use memory package
            print(f"Generating using memory package")
            print(f"Scene: {memory_package.get('major_scene', 'N/A')}")
            print(f"Characters: {', '.join(memory_package.get('characters', []))}")

            # ========== Load reference images in new priority order ==========
            # 1️⃣ Environment reference image (first priority, single image)
            env_ref = memory_package.get("environment_ref", "")
            if env_ref and os.path.exists(env_ref):
                try:
                    ref_images.append(Image.open(env_ref))
                    print(f"  ✅ [1/1] Loaded environment reference: {os.path.basename(env_ref)}")
                except Exception as e:
                    print(f"  ⚠️  Unable to load environment reference: {e}")

            # 2️⃣ Clothing reference images (second priority, main characters → supporting characters)
            clothing_refs = memory_package.get("clothing_refs", [])
            for i, ref_path in enumerate(clothing_refs, 1):
                if os.path.exists(ref_path):
                    try:
                        ref_images.append(Image.open(ref_path))
                        print(f"  ✅ [{i}/{len(clothing_refs)}] Loaded clothing reference: {os.path.basename(ref_path)}")
                    except Exception as e:
                        print(f"  ⚠️  Unable to load clothing reference: {e}")

            # 3️⃣ Character montage (third priority, remaining quota)
            character_refs = memory_package.get("character_refs", [])
            for i, ref_path in enumerate(character_refs, 1):
                if os.path.exists(ref_path):
                    try:
                        ref_images.append(Image.open(ref_path))
                        print(f"  ✅ [{len(clothing_refs)+i}/{len(clothing_refs)+len(character_refs)}] Loaded character montage: {os.path.basename(ref_path)}")
                    except Exception as e:
                        print(f"  ⚠️  Unable to load character montage: {e}")

            # Print total statistics
            total_images = 1 + len(clothing_refs) + len(character_refs)
            portrait_count = len(clothing_refs) + len(character_refs)

            print(f"\n  📊 Image statistics for this generation:")
            print(f"     Environment: 1 image")
            print(f"     Clothing: {len(clothing_refs)} images")
            print(f"     Montage: {len(character_refs)} images")
            print(f"     Total: {total_images} images")
            print(f"     Portrait type: {portrait_count} images (limit 5)")

            # Get Visual DNA
            visual_dna = memory_package.get("visual_dna", {})

            # Extract character list from memory package
            detected_characters = memory_package.get("characters", [])

            # Build enhanced prompt
            prompt_str = self.build_enhanced_prompt_from_memory(
                shot_id, memory_package, ref_images
            )

            # When using memory package, reference images are already loaded, skip subsequent loading steps
            print(f"\nTotal reference images: {len(ref_images)} (loaded from memory package)")
            print(f"Prompt length: {len(prompt_str)} characters")

            # Truncate prompt if it exceeds the safe limit to avoid EMPTY_PARTS from Gemini
            MAX_PROMPT_CHARS = 4000
            if len(prompt_str) > MAX_PROMPT_CHARS:
                print(f"⚠️  Prompt exceeds {MAX_PROMPT_CHARS} chars, truncating to reduce EMPTY_PARTS risk")
                prompt_str = prompt_str[:MAX_PROMPT_CHARS]

        else:
            # Original logic (without using memory package)
            content_json = shot_data["json_content"]
            major_scene_id = content_json.get("major_scene_id")
            print(f"Scene: {major_scene_id if major_scene_id else 'N/A'}")

            # Detect characters
            detected_characters = self.detect_characters_in_shot(shot_data)
            print(f"Detected characters: {', '.join(detected_characters) if detected_characters else 'None'}")

            # 1. Load environment reference image
            if major_scene_id:
                env_ref_path = os.path.join(self.reference_dir, f"{major_scene_id}_environment.png")
                if os.path.exists(env_ref_path):
                    try:
                        ref_images.append(Image.open(env_ref_path))
                        print(f"✅ Loaded environment reference: {os.path.basename(env_ref_path)}")
                    except Exception as e:
                        print(f"⚠️  Unable to load environment reference: {e}")

            # 2. Load foreign face reference images
            for char_id in detected_characters:
                mapping = self.get_character_mapping(char_id)
                if mapping:
                    target_face_path = mapping.get('target_face')
                    if target_face_path and os.path.exists(target_face_path):
                        try:
                            ref_images.append(Image.open(target_face_path))
                            print(f"✅ Loaded foreign face: {mapping.get('target_name', 'Unknown')} ({os.path.basename(target_face_path)})")
                        except Exception as e:
                            print(f"⚠️  Unable to load foreign face: {e}")

            # 3. Load clothing reference images
            for char_id in detected_characters:
                clean_id = char_id.replace('@', '')
                clothing_ref_path = os.path.join(self.reference_dir, f"{clean_id}_clothing.png")
                if os.path.exists(clothing_ref_path):
                    try:
                        ref_images.append(Image.open(clothing_ref_path))
                        print(f"✅ Loaded clothing reference: {os.path.basename(clothing_ref_path)}")
                    except Exception as e:
                        print(f"⚠️  Unable to load clothing reference: {e}")

            print(f"Total reference images: {len(ref_images)}")

            # Build enhanced prompt (without using memory package)
            prompt_str = self.build_enhanced_prompt(shot_data, detected_characters, [])
            print(f"Prompt length: {len(prompt_str)} characters")

        # Prepare input
        input_contents = [prompt_str] + ref_images

        # Call model for generation
        try:
            # Note: generate_content() does not support aspect_ratio parameter
            # Aspect ratio control is completely implemented through strict rules in prompt
            response = self.client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=input_contents,
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                )
            )

            # Check response and diagnose errors
            saved = False

            # Detailed error diagnosis
            if not response.candidates or len(response.candidates) == 0:
                print(f"\n❌ Shot {shot_id} generation failed")
                self._print_detailed_error(response, shot_id, "NO_CANDIDATES")

                # Save error information for intelligent review agent
                self._save_error_info(shot_id, response, "NO_CANDIDATES")

                return False

            for candidate_idx, candidate in enumerate(response.candidates):
                # Check finish_reason
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason
                    finish_reason_str = str(finish_reason)

                    # Only these are true error states
                    error_finish_reasons = ["SAFETY", "RECITATION", "IMAGE_SAFETY", "MAX_TOKENS", "BLOCK_REASON_UNSPECIFIED"]

                    if finish_reason_str in error_finish_reasons:
                        if not saved:  # Only print error when not successful
                            print(f"\n❌ Shot {shot_id} generation failed")
                            self._print_detailed_error(response, shot_id, f"FINISH_REASON: {finish_reason}")
                            error_diagnosed = True

                            # Save error information for intelligent review agent
                            self._save_error_info(shot_id, response, f"FINISH_REASON: {finish_reason}")

                            return False  # Return failure directly, do not continue checking

                # Check if content exists
                if not hasattr(candidate, 'content'):
                    if not saved:
                        print(f"\n❌ Shot {shot_id} generation failed")
                        self._print_detailed_error(response, shot_id, "NO_CONTENT")

                        # Save error information for intelligent review agent
                        self._save_error_info(shot_id, response, "NO_CONTENT")

                    return False

                if not candidate.content:
                    if not saved:
                        print(f"\n❌ Shot {shot_id} generation failed")
                        self._print_detailed_error(response, shot_id, "EMPTY_CONTENT")

                        # Save error information for intelligent review agent
                        self._save_error_info(shot_id, response, "EMPTY_CONTENT")

                    return False

                # Check if parts exists
                if not hasattr(candidate.content, 'parts'):
                    if not saved:
                        print(f"\n❌ Shot {shot_id} generation failed")
                        self._print_detailed_error(response, shot_id, "NO_PARTS")

                        # Save error information for intelligent review agent
                        self._save_error_info(shot_id, response, "NO_PARTS")

                    return False

                if not candidate.content.parts:
                    if not saved:
                        print(f"\n❌ Shot {shot_id} generation failed")
                        self._print_detailed_error(response, shot_id, "EMPTY_PARTS")

                        # Save error information for intelligent review agent
                        self._save_error_info(shot_id, response, "EMPTY_PARTS")

                    return False

                # Process image data normally
                for part in candidate.content.parts:
                    if part.inline_data:
                        image_bytes = part.inline_data.data
                        import io
                        image = Image.open(io.BytesIO(image_bytes))

                        # Save directly without any processing
                        filename = f"shot_{shot_id}.png"
                        image.save(filename)

                        print(f"✅ Successfully saved: {filename}")
                        print(f"   Image size: {image.size}")
                        print(f"   Target aspect ratio: {self.aspect_ratio}")
                        saved = True
                        break
                if saved:
                    break

            if not saved:
                print(f"⚠️  Shot {shot_id} did not generate valid image")
                self._print_detailed_error(response, shot_id, "NO_VALID_IMAGE")
                return False

            return True

        except Exception as e:
            print(f"\n❌ Shot {shot_id} generation failed: {e}")
            import traceback
            traceback.print_exc()

            # Save error information for intelligent review agent
            self._save_error_info(shot_id, None, str(e), traceback.format_exc())

            # Try to get more information from the exception
            if "NoneType" in str(e) and "content.parts" in str(e):
                print(f"\n{'='*70}")
                print(f"💡 Error Analysis:")
                print(f"{'='*70}")
                print(f"Possible causes:")
                print(f"  1. Gemini safety review blocked generation (content involves violence, pornography, etc.)")
                print(f"  2. Gemini content policy violation (copyright, trademark issues)")
                print(f"  3. Prompt too long or format issue")
                print(f"  4. Reference images do not meet requirements")
                print(f"  5. API quota or rate limiting issue")
                print(f"\nSuggestions:")
                print(f"  - Check if script description contains sensitive content")
                print(f"  - Check if reference images are appropriate")
                print(f"  - Check if prompt length is reasonable")
                print(f"{'='*70}")

            return False

    def _print_detailed_error(self, response, shot_id, error_type):
        """
        Print detailed Gemini error information

        Args:
            response: Gemini API response object
            shot_id: Shot ID
            error_type: Error type identifier
        """
        print(f"\n{'='*70}")
        print(f"📋 Gemini Detailed Error Information - Shot {shot_id}")
        print(f"{'='*70}")
        print(f"Error type: {error_type}")

        # 1. Check prompt_feedback (contains reasons for being blocked)
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback = response.prompt_feedback
            print(f"\n🔍 Prompt Feedback:")

            # Check block reason
            if hasattr(feedback, 'block_reason') and feedback.block_reason:
                block_reason = feedback.block_reason
                print(f"  Block reason: {block_reason}")

                if block_reason == "SAFETY":
                    print(f"  ⚠️  Content violates safety policy")
                    print(f"  Possible reasons: violence, gore, pornography, hate speech, etc.")
                elif block_reason == "BLOCK_REASON_UNSPECIFIED":
                    print(f"  ⚠️  Content blocked (reason unspecified)")
                else:
                    print(f"  ⚠️  Block reason: {block_reason}")

            # Get safety ratings
            if hasattr(feedback, 'safety_ratings') and feedback.safety_ratings:
                print(f"\n🛡️  Safety ratings:")
                for rating in feedback.safety_ratings:
                    category = rating.category if hasattr(rating, 'category') else "UNKNOWN"
                    probability = rating.probability if hasattr(rating, 'probability') else "UNKNOWN"
                    print(f"  {category}: {probability}")

                    # Check if there is high risk
                    if 'HIGH' in str(probability) or 'MEDIUM' in str(probability):
                        print(f"    ⚠️  Detected {probability} risk content")

        # 2. Check candidates
        if hasattr(response, 'candidates') and response.candidates:
            print(f"\n📊 Candidates information:")
            print(f"  Candidate count: {len(response.candidates)}")

            for idx, candidate in enumerate(response.candidates):
                print(f"\n  Candidate #{idx + 1}:")

                # Check finish_reason
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason:
                    finish_reason = candidate.finish_reason
                    print(f"  Finish reason: {finish_reason}")

                    if finish_reason == "FINISH_REASON_UNSPECIFIED":
                        print(f"    ℹ️  Reason unspecified (possibly blocked by content policy)")
                    elif finish_reason == "RECITATION":
                        print(f"    ⚠️  Possibly involves copyrighted content (citing protected content)")
                    elif finish_reason == "SAFETY":
                        print(f"    ⚠️  Blocked for safety reasons")
                    elif finish_reason == "MAX_TOKENS":
                        print(f"    ⚠️  Generated content too long")
                    elif finish_reason == "IMAGE_SAFETY":
                        print(f"    ⚠️  Image safety review failed")

                # Check content status
                if hasattr(candidate, 'content'):
                    if candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            print(f"  Content Parts: {len(candidate.content.parts)} parts")
                        else:
                            print(f"  Content Parts: Empty or does not exist")
                    else:
                        print(f"  Content: None (content is empty)")
                else:
                    print(f"  Content: No content attribute")

                # Check candidate-level safety ratings
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    print(f"  Safety ratings:")
                    for rating in candidate.safety_ratings:
                        category = rating.category if hasattr(rating, 'category') else "UNKNOWN"
                        probability = rating.probability if hasattr(rating, 'probability') else "UNKNOWN"
                        print(f"    {category}: {probability}")

        # 3. Try to get response text (if any)
        if hasattr(response, 'text') and response.text:
            print(f"\n📝 Response text (first 500 characters):")
            print(f"  {response.text[:500]}")

        print(f"{'='*70}")
        print(f"💡 Suggestion: Modify prompt or reference images based on the above error information")
        print(f"{'='*70}")

    def _save_error_info(self, shot_id, response=None, error_type="", error_trace=""):
        """
        Save detailed error information (for intelligent review agent)

        Args:
            shot_id: Shot ID
            response: Gemini response object (if any)
            error_type: Error type
            error_trace: Error stack trace
        """
        error_info = {
            "shot_id": shot_id,
            "timestamp": datetime.now().isoformat(),
            "error_type": error_type,
            "trace": error_trace
        }

        # If response object exists, extract detailed information
        if response:
            error_info["response_details"] = self._extract_response_details(response)

        # Save to instance variable
        self.last_errors[shot_id] = error_info

    def _extract_response_details(self, response):
        """
        Extract detailed information from response object (for intelligent review agent)

        Args:
            response: Gemini response object

        Returns:
            Dictionary containing detailed information
        """
        details = {}

        # Extract prompt_feedback
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            feedback = response.prompt_feedback
            details["prompt_feedback"] = {
                "block_reason": str(feedback.block_reason) if hasattr(feedback, 'block_reason') else None,
                "safety_ratings": [
                    {
                        "category": str(r.category),
                        "probability": str(r.probability)
                    }
                    for r in feedback.safety_ratings
                ] if hasattr(feedback, 'safety_ratings') else []
            }

        # Extract candidates information
        if hasattr(response, 'candidates') and response.candidates:
            details["candidates"] = []
            for idx, candidate in enumerate(response.candidates):
                candidate_info = {
                    "index": idx,
                    "finish_reason": str(candidate.finish_reason) if hasattr(candidate, 'finish_reason') else None
                }

                # Check content and parts
                if hasattr(candidate, 'content'):
                    if candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            candidate_info["has_content"] = True
                            candidate_info["parts_count"] = len(candidate.content.parts)
                        else:
                            candidate_info["has_content"] = True
                            candidate_info["parts_empty"] = True
                    else:
                        candidate_info["has_content"] = False
                else:
                    candidate_info["has_content"] = False

                # safety_ratings
                if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                    candidate_info["safety_ratings"] = [
                        {
                            "category": str(r.category),
                            "probability": str(r.probability)
                        }
                        for r in candidate.safety_ratings
                    ]

                details["candidates"].append(candidate_info)

        return details

    def get_last_error_info(self, shot_id):
        """
        Get the last error information for specified shot (for intelligent review agent)

        Args:
            shot_id: Shot ID

        Returns:
            Error information dictionary, or None if not exists
        """
        return self.last_errors.get(shot_id)

    def generate_video_for_shot(self, shot_id, duration_seconds=None):
        """
        Generate video clip for a single shot (using Gemini Veo 3)

        Args:
            shot_id: Shot ID
            duration_seconds: Video duration in seconds, defaults to duration in shot data

        Returns:
            True on success, False on failure
        """
        # Get memory package
        if self.memory_agent:
            memory_package = self.memory_agent.get_shot_memory(shot_id)
            if not memory_package:
                print(f"❌ Memory package for Shot {shot_id} does not exist")
                return False
            use_memory = True
        else:
            print(f"❌ Video generation requires Memory Allocation Agent support")
            return False

        print(f"\n{'='*70}")
        print(f"Generating video clip for Shot {shot_id}")
        print(f"{'='*70}")

        # Get video duration
        if duration_seconds is None:
            duration_seconds = memory_package.get("narrative", {}).get("duration", 8)

        # Veo 3 API only supports integers 4, 6, 8 seconds
        # Convert float to nearest valid integer
        if isinstance(duration_seconds, float):
            # Round to nearest integer
            rounded_duration = round(duration_seconds)
            # Ensure within valid range (4, 6, 8)
            if rounded_duration <= 4:
                duration_seconds = 4
            elif rounded_duration <= 6:
                duration_seconds = 6
            else:
                duration_seconds = 8
            print(f"⚠️  Video duration adjusted to: {duration_seconds} seconds (original: {rounded_duration} seconds)")

        # Ensure it's an integer
        duration_seconds = int(duration_seconds)

        print(f"Video duration: {duration_seconds} seconds")
        print(f"Generating using memory package")
        print(f"Scene: {memory_package.get('major_scene', 'N/A')}")
        print(f"Characters: {', '.join(memory_package.get('characters', []))}")

        # Prepare reference images
        ref_images = []

        # 1. Load environment reference image
        env_ref = memory_package.get("environment_ref", "")
        if env_ref and os.path.exists(env_ref):
            try:
                ref_images.append(Image.open(env_ref))
                print(f"✅ Loaded environment reference: {os.path.basename(env_ref)}")
            except Exception as e:
                print(f"⚠️  Unable to load environment reference: {e}")

        # 2. Load foreign face reference images
        for char_ref in memory_package.get("character_refs", []):
            if os.path.exists(char_ref):
                try:
                    ref_images.append(Image.open(char_ref))
                    print(f"✅ Loaded foreign face: {os.path.basename(char_ref)}")
                except Exception as e:
                    print(f"⚠️  Unable to load foreign face: {e}")

        # 3. Load clothing reference images
        for clothing_ref in memory_package.get("clothing_refs", []):
            if os.path.exists(clothing_ref):
                try:
                    ref_images.append(Image.open(clothing_ref))
                    print(f"✅ Loaded clothing reference: {os.path.basename(clothing_ref)}")
                except Exception as e:
                    print(f"⚠️  Unable to load clothing reference: {e}")

        print(f"Total reference images: {len(ref_images)}")

        # Build video generation prompt
        narrative = memory_package.get("narrative", {})
        visual_dna = memory_package.get("visual_dna", {})
        character_mappings = memory_package.get("character_mappings", {})

        # Video generation prompt - emphasize action and camera movement
        video_prompt = f"""Generate a {duration_seconds} second cinematic video clip with the following specifications:

**NARRATIVE CONTENT**:
- Action/Subject Movement: {narrative.get('action', '')}
- Camera Movement: {narrative.get('camera_movement', '')}

**CHARACTER MAPPINGS**:
"""
        for char_id, char_info in character_mappings.items():
            target_name = char_info.get('target_name', 'Unknown')
            video_prompt += f"\n**{char_id}** → {target_name}\n"
            video_prompt += f"  - Clothing: {char_info.get('clothing', 'N/A')}\n"

        video_prompt += f"""
**VISUAL DNA**:
- Lighting: {visual_dna.get('lighting', '')}
- Color Grading: {visual_dna.get('color', '')}
- Mood/Atmosphere: {visual_dna.get('mood', '')}
- Shot Size: {visual_dna.get('shot_size', '')}
- Camera Angle: {visual_dna.get('camera_angle', '')}
- Camera Height: {visual_dna.get('camera_height', '')}
- Focal Length: {visual_dna.get('focal_length', '')}
- Depth of Field: {visual_dna.get('depth_of_field', '')}

**TECHNICAL SPECS**:
- Aspect Ratio: {self.aspect_ratio} ({self.target_width}x{self.target_height})
- Duration: {duration_seconds} seconds
- Frame Rate: 30 fps
- Style: Photorealistic cinematic video

**CRITICAL REQUIREMENTS**:
1. Use the reference images provided for character faces and clothing
2. Maintain visual consistency throughout the entire {duration_seconds} second clip
3. Smooth camera movement as specified
4. Natural character movements
5. High-quality photorealistic rendering
6. NO text, watermarks, or overlays
"""

        print(f"Video Prompt length: {len(video_prompt)} characters")

        # Prepare input
        input_contents = [video_prompt] + ref_images

        # Call Veo 3.1 to generate video
        try:
            print(f"\nCalling Gemini Veo 3.1 to generate video...")
            print(f"This may take a few minutes, please be patient...")

            # Read keyframe image for image-to-video generation
            image_path = f"shot_{shot_id}.png"
            input_image = None

            if os.path.exists(image_path):
                print(f"   Using keyframe image as video starting frame: {image_path}")
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                input_image = types.Image(image_bytes=image_bytes, mime_type="image/png")

            # Use correct GenerateVideosConfig
            config = types.GenerateVideosConfig(
                aspect_ratio=self.aspect_ratio,
                duration_seconds=duration_seconds,
            )

            # Use generate_videos API, correct model name is veo-3.1-generate-preview
            if input_image:
                # Image-to-video mode
                operation = self.client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=video_prompt,
                    image=input_image,
                    config=config,
                )
            else:
                # Text-to-video mode
                operation = self.client.models.generate_videos(
                    model="veo-3.1-generate-preview",
                    prompt=video_prompt,
                    config=config,
                )

            # Wait for async operation to complete
            import time
            poll_count = 0
            while not operation.done:
                poll_count += 1
                print(f"   Waiting for video generation... ({poll_count * 10} seconds)")
                time.sleep(10)
                operation = self.client.operations.get(operation)

            print(f"   Operation completed, waited {poll_count} polls")

            if not operation.response or not operation.response.generated_videos:
                print(f"⚠️  Shot {shot_id} did not generate valid video")
                return False

            # Save result - use official recommended method to download video
            generated_video = operation.response.generated_videos[0]
            self.client.files.download(file=generated_video.video)

            # Save video file
            filename = f"shot_{shot_id}_video.mp4"
            generated_video.video.save(filename)

            print(f"✅ Successfully saved video: {filename}")

            # Get video file size
            if hasattr(generated_video.video, 'video_bytes') and generated_video.video.video_bytes:
                file_size_mb = len(generated_video.video.video_bytes) / (1024 * 1024)
                print(f"   File size: {file_size_mb:.2f} MB")
            print(f"   Duration: {duration_seconds} seconds")

            return True

        except Exception as e:
            print(f"❌ Shot {shot_id} video generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def build_enhanced_prompt_from_memory(self, shot_id, memory_package, ref_images):
        """
        Build enhanced generation prompt from memory package

        Args:
            shot_id: Shot ID
            memory_package: Memory package
            ref_images: List of reference images

        Returns:
            Complete prompt string
        """
        # Basic composition rules
        composition_rules = f"""

**CRITICAL COMPOSITION RULES (MUST FOLLOW)**:
0. **ASPECT RATIO REQUIREMENT**:
   - MUST generate image with EXACT aspect ratio: {self.aspect_ratio}
   - Target resolution: {self.target_width}x{self.target_height}
   - CRITICAL: All generated content MUST maintain this aspect ratio

{memory_package.get('style_prompt', '')}

1. **SINGLE SHOT ONLY - ABSOLUTE REQUIREMENT**:
   - Generate ONLY ONE continuous shot/frame
   - DO NOT generate multiple images arranged together
   - DO NOT create split-screen compositions
   - DO NOT create diptych, triptych, or grid layouts
   - DO NOT arrange multiple shots side-by-side or stacked
   - MUST be a single, unified scene

2. **NO TEXT/GRAPHICS - ABSOLUTE PROHIBITION**:
   - DO NOT include any text overlays, subtitles, titles, captions
   - DO NOT add watermarks, logos, or signatures
   - DO NOT include visible text anywhere in the image
   - DO NOT add graphics, arrows, or UI elements
   - DO NOT include dialogue bubbles or text boxes
   - DO NOT render any dialogue, speech, or conversation as visible text
   - The image must be completely text-free
   - IGNORE any dialogue quotes or speech content in descriptions - these are for context only, NOT to be rendered as text

3. **PURE CINEMATIC SCENE**:
   - Focus purely on the cinematic scene itself
   - Show only the environment, characters, lighting, and action
   - No artificial composition layouts or frames
"""

        # Composition guidance
        shot_size = memory_package.get("visual_dna", {}).get("shot_size", "")
        shot_size_lower = shot_size.lower()

        if any(term in shot_size_lower for term in ["extreme close-up", "ecu", "close-up", "cu", "medium close-up", "mcu"]):
            composition_guidance = f"""
**COMPOSITION GUIDANCE FOR CLOSE-UP SHOTS**:
- CRITICAL: The subject MUST occupy the ENTIRE {self.aspect_ratio} frame
- DO NOT leave empty space above head or below shoulders
- Frame the subject TIGHTLY from edge to edge
"""
        elif any(term in shot_size_lower for term in ["wide shot", "wide", "establishing", "extreme wide"]):
            composition_guidance = f"""
**COMPOSITION GUIDANCE FOR WIDE SHOTS**:
- CRITICAL: The scene MUST fill the ENTIRE {self.aspect_ratio} frame
- Show the full environment but keep subjects LARGE enough to be visible
- NO empty sky areas or blank floor space
"""
        else:
            composition_guidance = f"""
**COMPOSITION GUIDANCE**:
- CRITICAL: The frame MUST be completely filled in {self.aspect_ratio} format
- Subject(s) should occupy 70-90% of the frame
- NO empty or wasted space anywhere
"""

        # Context consistency guidance
        consistency_instruction = f"""

**CRITICAL CONTEXT FROM MEMORY ALLOCATION**:
This shot belongs to Major Scene: {memory_package.get('major_scene', 'Unknown')}
You have been provided with {len(ref_images)} reference images.

**CHARACTERS IN THIS SHOT**:
{', '.join(memory_package.get('characters', []))}

**CHARACTER MAPPINGS**:
"""

        # Add character mapping information
        character_mappings = memory_package.get("character_mappings", {})
        for char_id, char_info in character_mappings.items():
            target_name = char_info.get('target_name', 'Unknown')
            consistency_instruction += f"\n**{char_id}**:\n"
            consistency_instruction += f"- Target Face: {target_name}\n"
            consistency_instruction += f"- Clothing: {char_info.get('clothing', 'N/A')}\n"

        # Visual DNA information
        visual_dna = memory_package.get("visual_dna", {})
        narrative = memory_package.get("narrative", {})

        # Extract key information
        subject_movement = narrative.get('action', '')
        language_prompt = narrative.get('language_prompt', '')
        camera_movement = narrative.get('camera_movement', '')

        # ========== Add: Use quality check feedback ==========
        generation_feedback = memory_package.get("generation_feedback", [])
        feedback_instruction = ""
        if generation_feedback:
            feedback_instruction = f"""

**⚠️ CRITICAL FEEDBACK FROM PREVIOUS QUALITY CHECK (MUST FIX)**:
The previous generation attempt failed quality inspection. Please address these issues:

"""
            for i, feedback in enumerate(generation_feedback, 1):
                feedback_instruction += f"{i}. {feedback}\n"

            feedback_instruction += f"""
**MANDATORY CORRECTIONS**:
- You MUST fix ALL the issues mentioned above
- Pay special attention to style consistency (2D vs 3D, cartoon vs realistic)
- Ensure character appearance matches the reference images exactly
- DO NOT repeat the same mistakes as before
- If feedback mentions "2D style" but reference shows "3D", generate in 3D style
- If feedback mentions "clothing mismatch", carefully match the clothing from reference images
"""

        consistency_instruction += f"""

**VISUAL DNA**:
- Lighting: {visual_dna.get('lighting', 'N/A')}
- Color: {visual_dna.get('color', 'N/A')}
- Mood: {visual_dna.get('mood', 'N/A')}
- Shot Size: {visual_dna.get('shot_size', 'N/A')}
- Camera Angle: {visual_dna.get('camera_angle', 'N/A')}

**ACTION AND COMPOSITION PRIORITY**:
1. **PRIMARY - Subject Movement & Positioning**:
   {subject_movement}

2. **SECONDARY - Detailed Scene Description**:
   {language_prompt}

3. **Camera Movement**:
   {camera_movement}

**CRITICAL GENERATION INSTRUCTIONS**:
- **PRIORITY 1**: Character positioning, body language, gestures, and facial expressions MUST follow "Subject Movement" above
- **PRIORITY 2**: Environmental details, lighting nuances, textures, and atmosphere should reference "Detailed Scene Description"
- If there's any conflict between movement and detail descriptions, **ALWAYS prioritize the Subject Movement (PRIORITY 1)**
- The character's pose, stance, and action are the most important elements to capture accurately
- Use the language prompt only for adding contextual details and refinements

**NEGATIVE PROMPT - TEXT ELIMINATION**:
- **CRITICAL**: DO NOT render any text, subtitles, or dialogue from the "Detailed Scene Description"
- All dialogue quotes, speech content, or conversation text in the description are for CONTEXT ONLY
- **ABSOLUTELY NO**: Subtitles, speech bubbles, text overlays, or any visible writing
- **EXTRACT ONLY**: Visual elements (actions, expressions, emotions, environment)
- **DISCARD ALL**: Text content, dialogue, quotes, or speech when rendering the image
- If description contains dialogue like "Hello!", RENDER ONLY the character speaking/expressing, NOT the text "Hello!"

**CRITICAL CONSISTENCY REQUIREMENTS**:
- Match the exact face from the character reference images
- Follow the clothing DNA specifications exactly
- Maintain the Visual DNA (lighting, color, mood)
- Execute the narrative action (Subject Movement) as the primary directive
- Apply the camera movement specified
- Use detailed scene description for environmental context and finer details
"""

        # Combine all parts
        prompt_str = composition_rules + composition_guidance + consistency_instruction

        return prompt_str

    def generate_all_images(self):
        """Generate all keyframe images"""
        print(f"\n{'='*70}")
        print("🎨 Starting to generate all keyframe images")
        print(f"{'='*70}")

        success_count = 0
        failed_count = 0

        for shot in self.shots_data:
            shot_id = shot["id"]

            if self.generate_image_for_shot(shot_id):
                success_count += 1
            else:
                failed_count += 1

            time.sleep(1)  # Avoid requesting too fast

        print(f"\n{'='*70}")
        print("✅ Keyframe generation completed!")
        print(f"   Successful: {success_count} shots")
        print(f"   Failed: {failed_count} shots")
        print(f"{'='*70}")

        return success_count, failed_count

    def generate_all_videos(self):
        """Generate all video clips"""
        print(f"\n{'='*70}")
        print("🎬 Starting to generate all video clips")
        print(f"{'='*70}")

        success_count = 0
        failed_count = 0

        for shot in self.shots_data:
            shot_id = shot["id"]

            if self.generate_video_for_shot(shot_id):
                success_count += 1
            else:
                failed_count += 1

            time.sleep(2)  # Video generation is slow, give more interval time

        print(f"\n{'='*70}")
        print("✅ Video generation completed!")
        print(f"   Successful: {success_count} shots")
        print(f"   Failed: {failed_count} shots")
        print(f"{'='*70}")

        return success_count, failed_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Image/Video Generation Agent - Integrates image and video generation functionality',
        epilog='''
Usage examples:
  python %(prog)s                           # Use default clip1_script.json
  python %(prog)s clip2.json                 # Specify other script file
  python %(prog)s clip1.json --shot 9        # Generate specific shot
  python %(prog)s clip1.json --mode video    # Generate in video mode
        '''
    )

    parser.add_argument(
        'script',
        nargs='?',
        default='clip1_script.json',
        help='Script JSON file path (default: clip1_script.json)'
    )

    parser.add_argument('--mapping', default='character_mapping.json', help='Character mapping configuration file')
    parser.add_argument('--reference', default='reference_images', help='Reference image directory')
    parser.add_argument('--shot', type=int, help='Generate specific shot ID')
    parser.add_argument('--mode', choices=['image', 'video'], default='image',
                       help='Generation mode: image=keyframe, video=video (default: image)')

    parser.add_argument('--style',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       default='realistic',
                       help='Generation style: realistic=realistic(default), lego=Lego, disney=Disney, anime=anime, clay=clay, japanese_anime=Japanese anime, family_guy=Family Guy American cartoon')

    args = parser.parse_args()

    try:
        # Initialize Agent
        agent = GenerationAgent(
            script_json=args.script,
            character_mapping=args.mapping,
            reference_dir=args.reference,
            style=args.style
        )

        # Load data
        agent.load_script_data()
        agent.load_character_mappings()

        # Load memory allocation
        agent.load_memory_allocation()

        # Execute generation
        if args.shot:
            # Generate single shot
            if args.mode == 'image':
                success = agent.generate_image_for_shot(args.shot)
                return 0 if success else 1
            else:  # video mode
                success = agent.generate_video_for_shot(args.shot)
                return 0 if success else 1
        else:
            # Generate all
            if args.mode == 'image':
                agent.generate_all_images()
            else:  # video mode
                agent.generate_all_videos()

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
