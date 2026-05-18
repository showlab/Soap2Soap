#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Character Sheet Image Generation Script - Interactive Pairing

Features:
1. Read clip1_script.json to get character information
2. Interactively let users assign foreign faces to each character
3. Use Gemini to generate character headshots
4. Generate annotated character collages
5. Save pairing configuration for subsequent Agent use

Usage:
    python create_character_sheet.py              # Interactive pairing
    python create_character_sheet.py --use-config  # Use existing configuration
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
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
except ImportError:
    print("Error: Pillow library is required")
    print("Please run: pip install Pillow")
    sys.exit(1)

# Optional face recognition library (for automatic face cropping)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Note: face_recognition library not installed, using center crop mode")


class CharacterSheetGenerator:
    """Character Sheet Collage Generator"""

    def __init__(self, script_json="clip1_script.json", foreign_faces_dir="foreign_faces", gemini_client=None, style="realistic", auto_generate_face=False):
        """
        Initialize

        Args:
            script_json: Script JSON file path
            foreign_faces_dir: Foreign face library directory
            gemini_client: Gemini client (for generating headshots)
            style: Generation style (all modes except lego use Gemini to extract faces)
            auto_generate_face: Whether to automatically use AI to generate face images (True=AI generation, False=manual upload)
        """
        self.script_json = script_json
        self.foreign_faces_dir = foreign_faces_dir
        self.character_mappings = []
        self.script_data = None  # Add script_data attribute for querying role_classification
        self.gemini_client = gemini_client  # Gemini client
        self.style = style  # Add style attribute
        self.auto_generate_face = auto_generate_face  # Add auto-generate face flag

        # Create storage directory for AI-generated faces
        if self.auto_generate_face:
            self.generated_faces_dir = "generated_faces"
            os.makedirs(self.generated_faces_dir, exist_ok=True)
            print(f"✅ Auto-generate face mode enabled, faces will be saved to: {self.generated_faces_dir}/")

    def load_script_data(self):
        """Load script data"""
        print(f"Reading script data: {self.script_json}")

        if not os.path.exists(self.script_json):
            print(f"Error: Script file not found {self.script_json}")
            return None

        with open(self.script_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Save script_data for subsequent role_classification queries
        self.script_data = data

        characters = data.get('character_roster', {}).get('characters', [])
        print(f"Found {len(characters)} character(s)\n")

        return data, characters

    def interactive_mapping(self, characters):
        """Interactive character pairing - supports manual upload or AI auto-generated faces"""

        print("="*70)
        print("Character Sheet Mapping - Interactive Configuration")
        print("="*70)

        # Display current mode
        if self.auto_generate_face:
            print("\n🤖 AUTO-GENERATE MODE: AI will generate face images based on character descriptions")
            print("   Generated faces will be saved to: generated_faces/\n")
        else:
            print("\n📤 MANUAL UPLOAD MODE: Please upload reference face images for each character\n")

        for idx, char in enumerate(characters, 1):
            char_id = char.get('character_id', '')
            char_name = char.get('names', {}).get('primary_name', '')
            aliases = char.get('names', {}).get('aliases', [])
            titles = char.get('names', {}).get('titles', [])

            # Display character information
            print(f"\n{'='*70}")
            print(f"Character {idx}/{len(characters)}: {char_id}")
            print(f"{'='*70}")
            print(f"  Primary Name: {char_name}")
            if aliases:
                print(f"  Aliases: {', '.join(aliases)}")
            if titles:
                print(f"  Titles: {', '.join(titles)}")
            print(f"  Physical Attributes: {char.get('physical_attributes', 'N/A')}")
            print(f"  Hair: {char.get('hair', 'N/A')}")
            print(f"  Clothing: {char.get('clothing_variations', [{}])[0].get('description', 'N/A')}")

            # Check if there are reference images
            ref_images_path = os.path.join('reference_images', char_id.replace('@', ''))
            has_reference = os.path.exists(ref_images_path)

            if has_reference:
                ref_files = [f for f in os.listdir(ref_images_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if ref_files:
                    print(f"  Reference Image: {ref_images_path}/{ref_files[0]}")

            # Choose different processing approach based on mode
            if self.auto_generate_face:
                # ====== AI Auto-Generate Mode ======
                print(f"\n🤖 AI will generate a face for this character...")

                # Generate filename and path
                clean_char_id = char_id.replace('@', '')
                target_name = char_name if char_name else f"character_{clean_char_id}"
                generated_face_path = os.path.join(self.generated_faces_dir, f"{clean_char_id}_{target_name}.png")

                # Prepare character information for generation
                character_info = {
                    'name': target_name,
                    'physical_attributes': char.get('physical_attributes', 'Adult'),
                    'hair': char.get('hair', 'Natural hair'),
                    'gender': self._infer_gender_from_attributes(char.get('physical_attributes', ''))
                }

                # Call AI to generate face
                success = self.generate_face_with_gemini(character_info, generated_face_path)

                if success:
                    face_path = generated_face_path
                    # No need to print again since generate_face_with_gemini already printed internally
                else:
                    print(f"  ❌ AI generation failed, please provide a manual image path")
                    # If AI generation fails, fall back to manual input
                    face_path = self._manual_face_input(char_id, char_name)
                    if face_path is None:
                        continue  # Skip this character
                    target_name = os.path.basename(face_path)
                    if '.' in target_name:
                        target_name = os.path.splitext(target_name)[0]
            else:
                # ====== Manual Upload Mode ======
                print(f"\nPlease upload the reference face image for this character that you want to appear in the generated video:")
                face_path = self._manual_face_input(char_id, char_name)
                if face_path is None:
                    continue  # Skip this character

                # Extract name from file path (e.g., foreign_faces/emma.png -> emma)
                target_name = os.path.basename(face_path)
                if '.' in target_name:
                    target_name = os.path.splitext(target_name)[0]

            # Save pairing information
            mapping = {
                'video_character': char_id,
                'video_character_name': char_name,
                'video_aliases': aliases,
                'video_titles': titles,
                'target_face': face_path,
                'target_name': target_name,
                'notes': '',  # No longer interactively input notes
                'physical_attributes': char.get('physical_attributes', ''),
                'hair': char.get('hair', ''),
                'face_source': 'ai_generated' if self.auto_generate_face else 'manual_upload'  # Record source
                # clothing field deleted - clothing information will be dynamically obtained from scene_wardrobe
            }

            self.character_mappings.append(mapping)

            print(f"\n✅ Character {char_id} mapped to: {target_name}")

    def _manual_face_input(self, char_id, char_name):
        """Helper function for manual input of face image path"""
        while True:
            face_path = input(f"  Reference image path (relative path, e.g., {self.foreign_faces_dir}/emma.png): ").strip()

            if not face_path:
                print("  ⚠️  Path cannot be empty, please re-enter")
                continue

            # Check if file exists
            if os.path.exists(face_path):
                return face_path
            else:
                print(f"  ⚠️  File not found: {face_path}")
                retry = input("  Continue with this path? (y/n): ").strip().lower()
                if retry == 'y':
                    return face_path
                else:
                    # Ask whether to skip
                    skip = input("  Skip this character? (y/n): ").strip().lower()
                    if skip == 'y':
                        return None

    def _infer_gender_from_attributes(self, attributes):
        """Infer gender from physical_attributes"""
        attributes_lower = attributes.lower()
        if any(word in attributes_lower for word in ['male', 'man', 'boy', 'he', 'his', 'mr.', 'sir']):
            if any(word in attributes_lower for word in ['female', 'woman', 'girl', 'she', 'her', 'ms.', 'mrs.']):
                return 'Unknown'  # Contains both gender words
            return 'Male'
        elif any(word in attributes_lower for word in ['female', 'woman', 'girl', 'she', 'her', 'ms.', 'mrs.']):
            return 'Female'
        else:
            return 'Unknown'

    def load_existing_config(self, config_file="character_mapping.json"):
        """Load existing pairing configuration"""
        if not os.path.exists(config_file):
            print(f"Config file not found: {config_file}")
            return False

        print(f"Reading configuration: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.character_mappings = data.get('mappings', [])
        print(f"✅ Loaded {len(self.character_mappings)} character mapping(s)\n")
        return True

    def save_config(self, config_file="character_mapping.json"):
        """Save pairing configuration"""
        config = {
            'mappings': self.character_mappings,
            'created_at': self._get_timestamp(),
            'total_characters': len(self.character_mappings)
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Configuration saved to: {config_file}")

    def generate_headshot_with_gemini(self, face_image_path, output_path):
        """
        Use Gemini to generate character headshot (all modes except lego)

        Args:
            face_image_path: Original face image path
            output_path: Output headshot path

        Returns:
            True on success, False on failure
        """
        # Only skip Gemini processing in lego mode
        if self.style == "lego":
            print(f"  ℹ️  Lego mode detected, skipping Gemini face extraction")
            return False

        if not self.gemini_client:
            print("  ⚠️  Gemini client not provided, using original image")
            return False

        try:
            from google import genai
            from google.genai import types

            # Load original face image as PIL Image object
            face_image = Image.open(face_image_path)

            # Build prompt
            prompt = """
Extract and generate a CLEAN HEADSHOT image containing ONLY the person's face.

**REQUIREMENTS**:
- Crop to show ONLY the face (from forehead to chin)
- DO NOT include neck, shoulders, or any body parts below the chin
- DO NOT include background elements - remove all backgrounds
- Square format output (face centered in square)
- Maintain exact facial features and expression from original
- High quality, sharp focus on the face
- NO text, NO watermarks, NO borders

**CRITICAL**:
- The output MUST be a FACE-ONLY image
- No shoulders, no torso, no body - just the face
- Crop from hairline/top of forehead to bottom of chin
- Remove everything except the face
"""

            print(f"  🎨 Calling Gemini to generate headshot (mode: {self.style})...")

            # Call Gemini (directly pass PIL Image object)
            response = self.gemini_client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[prompt, face_image],  # Directly pass PIL Image object
                config=types.GenerateContentConfig(
                    response_modalities=['IMAGE'],
                )
            )

            # Save generated image
            saved = False
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.inline_data:
                        image_bytes = part.inline_data.data
                        import io
                        generated_image = Image.open(io.BytesIO(image_bytes))

                        # Directly save Gemini-generated original image, no cropping or resize
                        generated_image.save(output_path)

                        saved = True
                        print(f"  ✅ Headshot generated: {output_path}")
                        break
                if saved:
                    break

            if not saved:
                print(f"  ⚠️  Gemini generation failed, using original image")
                print(f"  📋 Debug: Response candidates: {len(response.candidates) if hasattr(response, 'candidates') else 0}")
                return False

            return True

        except Exception as e:
            print(f"  ⚠️  Gemini generation error: {e}, using original image")
            import traceback
            print(f"  📋 Traceback: {traceback.format_exc()}")
            return False

    def generate_face_with_gemini(self, character_info, output_path):
        """
        Use Gemini to generate face image based on character description

        Args:
            character_info: Character information dictionary, containing physical_attributes, hair, etc.
            output_path: Output face image path

        Returns:
            True on success, False on failure
        """
        if not self.gemini_client:
            print("  ⚠️  Gemini client not available for face generation")
            return False

        try:
            from google.genai import types

            # Extract character information
            char_name = character_info.get('name', 'character')
            physical_attributes = character_info.get('physical_attributes', 'Adult')
            hair = character_info.get('hair', 'Natural hair')
            gender = character_info.get('gender', 'Unknown')

            # Build corresponding style prompts based on style
            style_prompts = {
                'disney': """
**STYLE: DISNEY/PIXAR 3D CGI ANIMATION**:
- 3D animated character in Disney/Pixar style
- Soft, rounded features with expressive eyes
- Vibrant, saturated colors
- Subsurface scattering on skin (translucent quality)
- Warm, appealing lighting
- Stylized but believable proportions
- Clean, polished 3D render look
- NO realistic photorealism - must look like a high-quality 3D animated film character
""",
                'lego': """
**STYLE: LEGO ANIMATION**:
- LEGO minifigure character face
- Simple, iconic LEGO facial features
- Plastic toy appearance with characteristic LEGO aesthetic
- Bold, primary colors
- Clean plastic surface with slight reflections
- Stylized, minimal facial details typical of LEGO figures
- Must look like an actual LEGO toy character
""",
                'anime': """
**STYLE: JAPANESE ANIME**:
- Anime/manga character face style
- Large, expressive eyes with highlights
- Stylized facial features
- Clean line art with cel-shading
- Vibrant anime color palette
- Soft gradients on skin
- Characteristic anime aesthetic
- 2D illustration style, NOT 3D
""",
                'japanese_anime': """
**STYLE: JAPANESE ANIME**:
- Anime/manga character face style
- Large, expressive eyes with highlights
- Stylized facial features
- Clean line art with cel-shading
- Vibrant anime color palette
- Soft gradients on skin
- Characteristic anime aesthetic
- 2D illustration style, NOT 3D
""",
                'clay': """
**STYLE: CLAYMATION / STOP-MOTION**:
- Clay/stop-motion animation character face
- Textured clay surface appearance
- Soft, rounded forms
- Visible clay fingerprints and texture
- Warm, handcrafted aesthetic
- Slight imperfections typical of clay animation
- Like characters from Wallace & Gromit or similar
""",
                'family_guy': """
**STYLE: FAMILY GUY AMERICAN ANIMATION**:
- Family Guy cartoon character face style
- Simple, flat 2D animation style
- Bold outlines
- Minimal facial detail
- Characteristic Family Guy aesthetic
- Bright, flat colors
- American adult animation style
""",
                'realistic': """
**STYLE: REALISTIC CINEMATIC**:
- Photorealistic human face
- Natural skin texture and pores
- Realistic lighting and shadows
- Cinematic portrait quality
- Natural colors and tones
- High-detail realistic rendering
- Like a professional portrait photograph
"""
            }

            # Get current style prompt
            style_prompt = style_prompts.get(self.style, style_prompts['realistic'])

            # Build prompt for face generation (add style prompt)
            prompt = f"""
Generate a HIGH-QUALITY portrait of a person's face based on the following description.

{style_prompt}

**CHARACTER DESCRIPTION**:
- Name: {char_name}
- Gender: {gender}
- Physical Attributes: {physical_attributes}
- Hair: {hair}

**REQUIREMENTS**:
- Generate a portrait showing ONLY the face and head (crop from forehead to chin)
- Face should be centered and clearly visible
- Square or portrait aspect ratio
- Neutral or slight smile expression
- Good lighting on the face
- Clean background (solid color or simple gradient)
- NO text, watermarks, or borders
- The face should be clearly visible

**CRITICAL**:
- Generate ONLY ONE face
- Make the face distinctive and memorable
- MUST match the specified art style above
"""

            print(f"  🎨 Generating face for {char_name} with Gemini (Style: {self.style})...")

            # Call Gemini to generate face (max 3 retries)
            max_retries = 3
            saved = False

            for retry in range(max_retries):
                if retry > 0:
                    print(f"  🔄 Retry {retry + 1}/{max_retries}...")

                try:
                    response = self.gemini_client.models.generate_content(
                        model="gemini-3-pro-image-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_modalities=['IMAGE'],
                        )
                    )

                    # Check if response is valid
                    if not response or not response.candidates:
                        print(f"  ⚠️  Empty response from Gemini, retrying...")
                        continue

                    # Iterate through candidate results
                    for candidate in response.candidates:
                        # Check if content and parts exist
                        if not candidate.content:
                            print(f"  ⚠️  No content in candidate, trying next...")
                            continue

                        if not candidate.content.parts:
                            print(f"  ⚠️  No parts in content, trying next...")
                            continue

                        for part in candidate.content.parts:
                            if part.inline_data:
                                image_bytes = part.inline_data.data
                                import io
                                generated_image = Image.open(io.BytesIO(image_bytes))
                                generated_image.save(output_path)

                                saved = True
                                print(f"  ✅ AI-generated face saved: {output_path}")
                                break
                        if saved:
                            break

                    if saved:
                        break  # Successfully generated, exit retry loop

                except Exception as e:
                    print(f"  ⚠️  Generation error: {e}")
                    if retry < max_retries - 1:
                        print(f"  🔄 Will retry...")
                        continue

            if not saved:
                print(f"  ❌ Failed to generate face after {max_retries} attempts")
                return False

            return True

        except Exception as e:
            print(f"  ⚠️  Face generation error: {e}")
            import traceback
            print(f"  📋 Traceback: {traceback.format_exc()}")
            return False

    def crop_face_region(self, image_path, padding_percent=0.1):
        """
        Crop face region from image (keep only face, not torso)

        Optimized mechanical cropping scheme:
        - Asymmetric padding (more on top, less on bottom)
        - Focus on face area, avoid including torso

        Args:
            image_path: Image path
            padding_percent: Padding percentage around face (default 10%)

        Returns:
            Cropped face image, or center crop of original if detection fails
        """
        img = Image.open(image_path)

        # Try using face_recognition library to detect face
        if FACE_RECOGNITION_AVAILABLE:
            try:
                # Detect face location
                face_locations = face_recognition.face_locations(np.array(img))
                if face_locations:
                    # Use first detected face
                    top, right, bottom, left = face_locations[0]

                    # Calculate face size
                    face_height = bottom - top
                    face_width = right - left

                    # Add padding (more on top, less on bottom to avoid including torso)
                    padding_h = int(face_height * padding_percent)
                    padding_w = int(face_width * padding_percent)

                    # Crop region (ensure not exceeding image boundaries)
                    img_width, img_height = img.size
                    crop_top = max(0, top - int(padding_h * 1.5))  # More padding on top, protect forehead and hair
                    crop_bottom = min(img_height, bottom + int(padding_h * 0.2))  # Less padding on bottom, avoid including torso
                    crop_left = max(0, left - padding_w)
                    crop_right = min(img_width, right + padding_w)

                    return img.crop((crop_left, crop_top, crop_right, crop_bottom))
            except Exception as e:
                print(f"  ⚠️  Face detection failed: {e}, using center crop")

        # Fallback: center crop (only take face area, not include torso)
        img_width, img_height = img.size

        # Crop center area (only take upper face area)
        crop_size = min(img_width, img_height) * 0.45  # Reduce to 45%, more focused on face
        crop_size = int(crop_size)

        left = (img_width - crop_size) // 2
        top = (img_height * 0.02)  # Start from 2% from top, more upward
        right = left + crop_size
        bottom = top + crop_size

        return img.crop((left, int(top), right, int(bottom)))

    def add_text_overlay(self, image, mapping):
        """
        Add name annotation at bottom of face image (not obscuring face)

        Args:
            image: PIL Image object
            mapping: Character pairing information

        Returns:
            Image with text added
        """
        img_copy = image.copy()

        # Add black bar at bottom of image for displaying name
        width, height = img_copy.size
        bottom_bar_height = height // 8  # Bottom takes 1/8

        # Create bottom black bar
        bottom_bar = Image.new('RGB', (width, bottom_bar_height), color='black')
        img_copy.paste(bottom_bar, (0, height - bottom_bar_height))

        # Draw text on bar
        draw = ImageDraw.Draw(img_copy)

        # Adjust font based on image size
        font_size = max(16, bottom_bar_height // 2)

        # Try loading font
        try:
            # Windows
            font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", font_size)
        except:
            try:
                # Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                # Default font
                font = ImageFont.load_default()

        # Determine name to display
        display_name = mapping.get('target_name', '').strip()
        if not display_name:
            display_name = mapping['video_character']

        # Draw name centered at bottom (white)
        text_bbox = draw.textbbox((0, 0), display_name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = (width - text_width) // 2
        text_y = height - bottom_bar_height + (bottom_bar_height - font_size) // 2

        draw.text((text_x, text_y), display_name, fill='white', font=font)

        return img_copy

    def group_characters_by_role(self):
        """
        Group characters by role_classification

        Returns:
            (main_characters, supporting_characters) two lists
        """
        main_characters = []
        supporting_characters = []

        for mapping in self.character_mappings:
            char_id = mapping['video_character']
            # Find character information from script_data to get role_classification
            role = "supporting"  # Default to supporting
            if hasattr(self, 'script_data') and self.script_data:
                characters = self.script_data.get('character_roster', {}).get('characters', [])
                for char in characters:
                    if char.get('character_id') == char_id:
                        role = char.get('role_classification', 'supporting')
                        break

            if role == "main":
                main_characters.append(mapping)
            else:
                supporting_characters.append(mapping)

        return main_characters, supporting_characters

    def create_collage(self, mappings, max_per_sheet=4, output_prefix="collage"):
        """
        Create character collage

        Args:
            mappings: Character pairing list
            max_per_sheet: Maximum number of characters per sheet (1-4)
            output_prefix: Output file prefix

        Returns:
            List of generated file paths
        """
        if not mappings:
            return []

        card_size = 600  # Size of each face card (square)
        generated_files = []

        # Group: max max_per_sheet per group
        for i in range(0, len(mappings), max_per_sheet):
            group = mappings[i:i+max_per_sheet]
            group_size = len(group)

            print(f"\nGenerating collage group {i//max_per_sheet + 1}: {group_size} character(s)")

            # Determine layout
            if group_size == 1:
                # 1 person: single vertical image
                cols, rows = 1, 1
            elif group_size == 2:
                # 2 people: vertical arrangement (vertical collage)
                cols, rows = 1, 2
            elif group_size == 3:
                # 3 people: vertical arrangement (vertical collage)
                cols, rows = 1, 3
            else:
                # 4 people: 2x2 grid
                cols, rows = 2, 2

            # Create canvas
            canvas_width = card_size * cols
            canvas_height = card_size * rows
            canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')

            # Create card for each character
            for idx, mapping in enumerate(group):
                # Calculate position
                col = idx % cols
                row = idx // cols
                x = col * card_size
                y = row * card_size

                print(f"  Processing character {idx+1}/{group_size}: {mapping.get('target_name', mapping['video_character'])}")

                # Load and process foreign face image
                target_face_path = mapping['target_face']
                if os.path.exists(target_face_path):
                    try:
                        # Create temporary headshot path
                        char_id = mapping['video_character'].replace('@', '')
                        temp_headshot_path = f"temp_headshot_{char_id}.png"

                        # 1. Try using Gemini to generate headshot (all modes except lego)
                        gemini_success = self.generate_headshot_with_gemini(target_face_path, temp_headshot_path)

                        if gemini_success and os.path.exists(temp_headshot_path):
                            # Use Gemini-generated headshot
                            face_img = Image.open(temp_headshot_path)
                            # Clean up temporary file
                            os.remove(temp_headshot_path)
                            print(f"    ✅ Using Gemini headshot")
                        else:
                            # lego mode or Gemini failed: use original image directly, no cropping
                            face_img = Image.open(target_face_path)
                            print(f"    ℹ️  Using original image (keeping original size)")

                        # 2. Maintain original aspect ratio, only adaptively resize to card_size
                        # Use thumbnail method to maintain aspect ratio
                        face_img.thumbnail((card_size, card_size), Image.Resampling.LANCZOS)

                        # 3. Create card with background (center aligned)
                        card = Image.new('RGB', (card_size, card_size), color='white')

                        # Calculate center position
                        img_width, img_height = face_img.size
                        paste_x = (card_size - img_width) // 2
                        paste_y = (card_size - img_height) // 2

                        # Paste to card center
                        card.paste(face_img, (paste_x, paste_y))

                        # 4. Paste to canvas (no text annotation, avoid obscuring image)
                        canvas.paste(card, (x, y))

                        # 5. Draw name on canvas (below card, not obscuring image)
                        # Prioritize target_name, if empty use video_character
                        char_name = mapping.get('target_name', '').strip()
                        if not char_name:
                            char_name = mapping['video_character']  # Use @character_id
                        self._draw_name_on_canvas(canvas, x, y, card_size, char_name, idx)

                        print(f"    ✅ Completed: {mapping.get('target_name', mapping['video_character'])}")

                    except Exception as e:
                        # Error handling
                        print(f"    ⚠️  Image processing failed: {e}")
                        self._draw_error_placeholder(canvas, x, y, card_size,
                                                      mapping.get('target_name', mapping['video_character']),
                                                      f"Image loading failed")
                else:
                    # Image does not exist
                    print(f"    ⚠️  Image not found: {target_face_path}")
                    self._draw_error_placeholder(canvas, x, y, card_size,
                                                    mapping.get('target_name', mapping['video_character']),
                                                    "Image not found")

            # Save collage
            output_file = f"{output_prefix}_{i//max_per_sheet + 1}.png"
            canvas.save(output_file)
            generated_files.append(output_file)

            print(f"  ✅ Collage saved: {output_file} ({cols}x{rows}, {canvas_width}x{canvas_height})")

        return generated_files

    def generate_character_sheet(self, output_file=None):
        """
        Generate character collage (grouped by main/supporting roles)

        New logic:
        1. Group by role_classification
        2. Prioritize generating main character collages
        3. Max 4 people/image, group if exceeded
        4. 1-3 people vertical, 4 people 2x2 layout

        Returns:
            True on success, False on failure
        """
        if not self.character_mappings:
            print("Error: No character mapping information")
            return False

        print(f"\n{'='*70}")
        print("🎭 Character Collage Generation - Grouped by Role")
        print(f"{'='*70}")

        if FACE_RECOGNITION_AVAILABLE:
            print("   - Face detection: Enabled")
        else:
            print("   - Face detection: Not installed, using center crop")

        # 1. Group by main/supporting roles
        main_characters, supporting_characters = self.group_characters_by_role()

        print(f"\nCharacter Statistics:")
        print(f"  - Main Characters: {len(main_characters)}")
        print(f"  - Supporting Characters: {len(supporting_characters)}")

        all_generated_files = []

        # 2. Prioritize generating main character collages
        if main_characters:
            print(f"\n{'='*70}")
            print("📸 Generating Main Character Collage")
            print(f"{'='*70}")
            main_files = self.create_collage(main_characters, max_per_sheet=4, output_prefix="main_characters")
            all_generated_files.extend(main_files)

        # 3. Generate supporting character collages
        if supporting_characters:
            print(f"\n{'='*70}")
            print("📸 Generating Supporting Character Collage")
            print(f"{'='*70}")
            supporting_files = self.create_collage(supporting_characters, max_per_sheet=4, output_prefix="supporting_characters")
            all_generated_files.extend(supporting_files)

        # 4. Save character position mapping (for subsequent Agent use)
        self._save_position_mapping(main_characters, supporting_characters)

        print(f"\n{'='*70}")
        print(f"✅ Character collage generation completed!")
        print(f"{'='*70}")
        print(f"\nGenerated Files:")
        for file in all_generated_files:
            print(f"  📄 {file}")
        print(f"\nTotal: {len(all_generated_files)} collage(s)")

        # 5. Save pairing configuration (containing character grouping information)
        self._save_enhanced_config(main_characters, supporting_characters)

        return True

    def _save_position_mapping(self, main_characters, supporting_characters):
        """Save character position mapping in collage"""
        position_mapping = {
            "main_characters": {},
            "supporting_characters": {}
        }

        # Main character position mapping (1-4 people per group)
        for i in range(0, len(main_characters), 4):
            group = main_characters[i:i+4]
            group_num = i // 4 + 1
            for idx, mapping in enumerate(group):
                char_id = mapping['video_character']
                position_mapping["main_characters"][char_id] = {
                    "collage_file": f"main_characters_{group_num}.png",
                    "position": idx,  # 0-3
                    "grid_position": self._get_grid_position(idx, len(group))
                }

        # Supporting character position mapping
        for i in range(0, len(supporting_characters), 4):
            group = supporting_characters[i:i+4]
            group_num = i // 4 + 1
            for idx, mapping in enumerate(group):
                char_id = mapping['video_character']
                position_mapping["supporting_characters"][char_id] = {
                    "collage_file": f"supporting_characters_{group_num}.png",
                    "position": idx,
                    "grid_position": self._get_grid_position(idx, len(group))
                }

        # Save to file
        with open("character_position_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(position_mapping, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Character position mapping saved: character_position_mapping.json")

    def _get_grid_position(self, index, total):
        """Get grid position (row, col)"""
        if total <= 3:
            # Vertical arrangement
            return (index, 0)
        else:
            # 2x2 grid
            return (index // 2, index % 2)

    def _save_enhanced_config(self, main_characters, supporting_characters):
        """Save enhanced pairing configuration (containing main/supporting character information)"""
        enhanced_config = {
            "mappings": self.character_mappings,
            "main_character_ids": [m['video_character'] for m in main_characters],
            "supporting_character_ids": [m['video_character'] for m in supporting_characters],
            "created_at": self._get_timestamp(),
            "total_characters": len(self.character_mappings)
        }

        with open("character_mapping.json", 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, indent=2, ensure_ascii=False)

        print(f"💾 Enhanced mapping config saved: character_mapping.json")

    def _draw_name_on_canvas(self, canvas, card_x, card_y, card_size, name, index):
        """
        Draw name on canvas (outside card, not obscuring image)

        Args:
            canvas: Canvas object
            card_x: Card x coordinate
            card_y: Card y coordinate
            card_size: Card size
            name: Character name
            index: Index (for calculating position)
        """
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(canvas)

        # Calculate name position (centered at bottom of card)
        text_x = card_x + card_size // 2
        text_y = card_y + card_size - 30  # 30 pixels from bottom

        # Adjust font based on card size
        try:
            # Windows
            font = ImageFont.truetype("C:\\Windows\\Fonts\\msyhbd.ttc", 24)  # Use bold, increase font size
        except:
            try:
                # Linux
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                # Default font
                font = ImageFont.load_default()

        # Calculate centered position
        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        final_x = text_x - text_width // 2

        # Draw white background rectangle (opaque)
        bg_x = final_x - 8
        bg_y = text_y - 3
        bg_width = text_width + 16
        bg_height = text_bbox[3] - text_bbox[1] + 6

        # Draw white background
        draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height],
                     fill='white', outline='gray')

        # Draw text (use dark color to ensure visibility)
        draw.text((final_x, text_y), name, fill='black', font=font)

    def _draw_error_placeholder(self, sheet, x, y, size, name, error_msg):
        """Draw error placeholder"""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(sheet)

        # Draw border
        draw.rectangle([x, y, x+size-2, y+size-2], outline='red', width=2)

        # Try loading font
        try:
            font = ImageFont.truetype("C:\\Windows\\Fonts\\msyh.ttc", 24)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()

        # Draw error message
        draw.text((x+20, y+size//2-40), name, fill='darkblue', font=font)
        draw.text((x+20, y+size//2), error_msg, fill='red', font=font)

    def _find_reference_image(self, char_id):
        """Find character reference image"""
        # Remove @ symbol
        clean_id = char_id.replace('@', '')

        # Try multiple possible paths
        possible_paths = [
            os.path.join('reference_images', clean_id, 'ref_01.png'),
            os.path.join('reference_images', clean_id, 'ref_1.png'),
            os.path.join('clip1_images', f'shot_*_{clean_id.replace("character_", "")}.png'),
        ]

        # For shot_*.png pattern, need actual search
        import glob
        script_dir = os.path.dirname(self.script_json)

        # Find keyframes in clip1_images directory where this character appears
        try:
            # Extract character number
            char_num = clean_id.replace('character_', '')
            # Find shots containing this character
            pattern = os.path.join(script_dir, 'clip1_images', f'shot_*{char_num}.png')
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        except:
            pass

        # Check reference_images directory
        ref_dir = os.path.join(os.path.dirname(self.script_json), 'reference_images', clean_id)
        if os.path.exists(ref_dir):
            files = [f for f in os.listdir(ref_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if files:
                return os.path.join(ref_dir, files[0])

        return None

    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Character Sheet Long Image Generation Tool - Interactive Character Pairing',
        epilog='''
Usage examples:
  python %(prog)s                                    # Manual upload mode (default)
  python %(prog)s --auto-generate-face               # AI auto-generate face mode
  python %(prog)s clip2.json                         # Specify other script file
  python %(prog)s clip2.json --use-config            # Use existing configuration, no interaction
  python %(prog)s clip3.json --auto-generate-face    # AI generate face
  python %(prog)s clip3.json --foreign-dir faces/ --output sheet.png  # Specify other options
        '''
    )

    # Positional argument: script JSON file
    parser.add_argument(
        'script',
        nargs='?',
        default='clip1_script.json',
        help='Script JSON file path (default: clip1_script.json)'
    )

    parser.add_argument('--foreign-dir', default='foreign_faces', help='Foreign face library directory')
    parser.add_argument('--output', default='character_sheet.png', help='Output long image file')
    parser.add_argument('--config', default='character_mapping.json', help='Pairing configuration file')
    parser.add_argument('--use-config', action='store_true', help='Use existing configuration, no interaction')
    parser.add_argument('--style', default='realistic',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       help='Generation style (all modes except lego use Gemini to extract faces)')
    parser.add_argument('--auto-generate-face', action='store_true',
                       help='Enable AI auto-generate face mode (use Gemini to automatically generate face images based on character descriptions, no manual upload required)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("🎬 Character Sheet Generator")
    print("="*70)
    print(f"Script file: {args.script}")

    # Display face acquisition mode
    if args.auto_generate_face:
        print("Face Mode: 🤖 AI Auto-Generate (Gemini)")
    else:
        print("Face Mode: 📤 Manual Upload")

    # Initialize Gemini client
    try:
        from google import genai
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            print("⚠️  Environment variable GENAI_API_KEY not found")
            print("   Will use mechanical cropping mode")
            gemini_client = None
            if args.auto_generate_face:
                print("❌ AI generate mode requires GENAI_API_KEY")
                return 1
        else:
            gemini_client = genai.Client(api_key=api_key)
            print("✅ Gemini client initialized")
    except Exception as e:
        print(f"⚠️  Gemini client initialization failed: {e}")
        print("   Will use mechanical cropping mode")
        gemini_client = None
        if args.auto_generate_face:
            print("❌ AI generate mode requires Gemini client")
            return 1

    generator = CharacterSheetGenerator(
        script_json=args.script,
        foreign_faces_dir=args.foreign_dir,
        gemini_client=gemini_client,  # Pass Gemini client
        style=args.style,  # Pass style parameter
        auto_generate_face=args.auto_generate_face  # Pass auto-generate face flag
    )

    # Load script data
    result = generator.load_script_data()
    if not result:
        print("❌ Failed to load script data")
        return 1

    data, characters = result

    # Check if configuration files already exist
    config_exists = os.path.exists(args.config)
    position_mapping_exists = os.path.exists("character_position_mapping.json")

    # If both configuration files exist, automatically use them
    if config_exists and position_mapping_exists:
        print(f"\n✅ Found existing configuration files:")
        print(f"   - {args.config}")
        print(f"   - character_position_mapping.json")
        print(f"   Using existing configuration (skip interactive mapping)\n")

        if not generator.load_existing_config(args.config):
            print("❌ Failed to load config, please remove the config files and try again")
            return 1
    # If --use-config parameter is specified
    elif args.use_config:
        if not generator.load_existing_config(args.config):
            print("❌ Failed to load config, please run without --use-config flag first")
            return 1
    # Otherwise, perform interactive pairing
    else:
        # Interactive pairing
        generator.interactive_mapping(characters)

        # Save configuration
        generator.save_config(args.config)

    # Generate long image
    success = generator.generate_character_sheet(args.output)

    if success:
        print("\n" + "="*70)
        print("✅ Completed!")
        print("="*70)
        print(f"\nGenerated Files:")
        print(f"  1. {args.output} - Character Sheet")
        print(f"  2. {args.config} - Mapping Configuration (for subsequent Agents)")
        print("\nNext Step:")
        print(f"  Run Master Agent: python agent_master.py")
        print("="*70)
        return 0
    else:
        print("\n❌ Failed to generate character sheet")
        return 1


if __name__ == "__main__":
    sys.exit(main())
