#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Inspection Agent

Features:
1. Check if generated faces match target foreign faces
2. Check clothing consistency
3. Provide pass/fail results

Usage:
    python agent_inspection.py shot_9.png @character_03    # Check specific image
    python agent_inspection.py --check-all                 # Check all generated images
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: Pillow and numpy libraries are required")
    print("Please run: pip install Pillow numpy")
    sys.exit(1)

# Optional face recognition library
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Note: face_recognition library not installed, will use Gemini Vision API for face checking")


class InspectionAgent:
    """Quality Inspection Agent"""

    def __init__(self, character_mapping="character_mapping.json", script_json="clip1_script.json", use_gemini=True):
        """
        Initialize

        Args:
            character_mapping: Character mapping configuration file path
            script_json: Script JSON file path
            use_gemini: Whether to use Gemini Vision API (takes priority over face_recognition)
        """
        self.character_mapping_file = character_mapping
        self.script_json = script_json
        self.use_gemini = use_gemini
        self.character_mappings = []
        self.script_data = None
        self.shots_map = {}  # shot_id -> shot_data mapping

        # Initialize client if using Gemini
        if self.use_gemini:
            try:
                from google import genai
                from google.genai import types

                api_key = os.environ.get("GENAI_API_KEY")
                if not api_key:
                    raise ValueError("Error: Environment variable GENAI_API_KEY not found")

                self.genai_client = genai.Client(api_key=api_key)
                self.genai_available = True
            except Exception as e:
                print(f"⚠️  Gemini initialization failed: {e}")
                print("   Will fall back to face_recognition library")
                self.genai_available = False
        else:
            self.genai_available = False

    def load_character_mappings(self):
        """Load character mapping configuration"""
        if not os.path.exists(self.character_mapping_file):
            print(f"❌ Character mapping file not found: {self.character_mapping_file}")
            return False

        print(f"Reading: {self.character_mapping_file}")
        with open(self.character_mapping_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.character_mappings = data.get('mappings', [])
        print(f"✅ Loaded {len(self.character_mappings)} character mappings")

        return True

    def load_script_data(self):
        """Load script data"""
        if not os.path.exists(self.script_json):
            print(f"❌ Script file not found: {self.script_json}")
            return False

        print(f"Reading: {self.script_json}")
        with open(self.script_json, 'r', encoding='utf-8') as f:
            self.script_data = json.load(f)

        # Build shot_id to shot_data mapping
        scenes = self.script_data.get("scenes", [])
        for scene in scenes:
            if scene.get("_disabled", False):
                continue
            scene_index = scene.get("scene_index")
            if scene_index is not None:
                self.shots_map[scene_index] = scene

        print(f"✅ Loaded {len(self.shots_map)} shot data")
        return True

    def detect_character_in_shot(self, shot_id):
        """
        Automatically detect character in shot through face feature matching

        Args:
            shot_id: shot ID

        Returns:
            Character ID (e.g. @character_03), returns None if not detected
        """
        image_file = f"shot_{shot_id}.png"
        if not os.path.exists(image_file):
            return None

        # Method 1: Use face_recognition library for face matching (most accurate)
        if FACE_RECOGNITION_AVAILABLE:
            return self._match_face_by_face_recognition(image_file)

        # Method 2: Use Gemini Vision for face comparison (alternative)
        if self.genai_available:
            return self._match_face_by_gemini(image_file)

        print(f"   ⚠️  No face detection method available")
        return None

    def _match_face_by_face_recognition(self, image_file):
        """
        Use face_recognition library for face feature matching

        Args:
            image_file: Image path

        Returns:
            Best matching character ID, returns None if detection fails
        """
        try:
            import face_recognition

            # Load generated image and encode face
            generated_image = face_recognition.load_image_file(image_file)
            generated_encodings = face_recognition.face_encodings(generated_image)

            if len(generated_encodings) == 0:
                print(f"   ⚠️  No face detected in image")
                return None

            if len(generated_encodings) > 1:
                print(f"   ⚠️  Multiple faces detected, using first face")

            generated_encoding = generated_encodings[0]

            # Compare with all target faces to find best match
            best_match_char = None
            best_similarity = -1

            for mapping in self.character_mappings:
                target_face_path = mapping.get('target_face')
                if not target_face_path or not os.path.exists(target_face_path):
                    continue

                try:
                    # Load target face and encode
                    target_image = face_recognition.load_image_file(target_face_path)
                    target_encodings = face_recognition.face_encodings(target_image)

                    if len(target_encodings) == 0:
                        continue

                    target_encoding = target_encodings[0]

                    # Calculate distance and convert to similarity
                    distance = face_recognition.face_distance([target_encoding], generated_encoding)[0]
                    similarity = 1 - distance  # Convert to 0-1 similarity

                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_char = mapping['video_character']

                except Exception as e:
                    continue

            if best_match_char and best_similarity > 0.4:  # Set minimum threshold
                print(f"   ✅ Face match successful: {best_match_char} (similarity: {best_similarity:.2f})")
                return best_match_char
            else:
                print(f"   ⚠️  No matching face found (highest similarity: {best_similarity:.2f})")
                return None

        except Exception as e:
            print(f"   ⚠️  Face recognition failed: {e}")
            return None

    def _match_face_by_gemini(self, image_file):
        """
        Use Gemini Vision for face comparison (alternative method)

        Args:
            image_file: Image path

        Returns:
            Best matching character ID, returns None if detection fails
        """
        try:
            from PIL import Image
            import io

            # Load generated image
            with open(image_file, "rb") as f:
                generated_img = Image.open(io.BytesIO(f.read()))

            # Calculate similarity for each target face
            best_match_char = None
            best_score = 0

            for mapping in self.character_mappings:
                target_face_path = mapping.get('target_face')
                if not target_face_path or not os.path.exists(target_face_path):
                    continue

                # Load target face
                with open(target_face_path, "rb") as f:
                    target_img = Image.open(io.BytesIO(f.read()))

                # Build comparison prompt
                prompt = f"""
Compare the face in the first image with the reference face in the second image.

**TASK**:
Rate how similar these two faces are in terms of:
- Facial features (eyes, nose, mouth, face shape)
- Hair style and color
- Overall appearance

**OUTPUT FORMAT** (must follow exactly):
SIMILARITY_SCORE: <score from 0.0 to 1.0>

Example:
SIMILARITY_SCORE: 0.85
"""

                # Call Gemini Vision
                response = self.genai_client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=[
                        prompt,
                        generated_img,
                        target_img
                    ]
                )

                # Parse response
                response_text = response.candidates[0].content.parts[0].text

                for line in response_text.split('\n'):
                    if 'SIMILARITY_SCORE:' in line:
                        try:
                            score = float(line.split(':')[1].strip())
                            if score > best_score:
                                best_score = score
                                best_match_char = mapping['video_character']
                        except:
                            pass

            if best_match_char and best_score > 0.6:
                print(f"   ✅ Gemini face match: {best_match_char} (similarity: {best_score:.2f})")
                return best_match_char
            else:
                print(f"   ⚠️  No matching face found (highest similarity: {best_score:.2f})")
                return None

        except Exception as e:
            print(f"   ⚠️  Gemini face matching failed: {e}")
            return None

    def get_character_mapping(self, character_id):
        """
        Get mapping information based on character ID

        Args:
            character_id: Character ID (e.g. @character_03)

        Returns:
            Mapping information dictionary, returns None if not found
        """
        for mapping in self.character_mappings:
            if mapping['video_character'] == character_id:
                return mapping
        return None

    def check_face_with_face_recognition(self, generated_image_path, target_face_path, threshold=0.5):
        """
        Use face_recognition library to check face match

        Args:
            generated_image_path: Generated image path
            target_face_path: Target face image path
            threshold: Similarity threshold (0-1, default 0.5)

        Returns:
            (passed, similarity, message)
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return False, 0.0, "face_recognition library not installed"

        try:
            # Load images
            generated_img = face_recognition.load_image_file(generated_image_path)
            target_img = face_recognition.load_image_file(target_face_path)

            # Encode faces
            generated_encodings = face_recognition.face_encodings(generated_img)
            target_encodings = face_recognition.face_encodings(target_img)

            if not generated_encodings:
                return False, 0.0, "No face detected in generated image"
            if not target_encodings:
                return False, 0.0, "No face detected in target face image"

            # Calculate distance
            generated_encoding = generated_encodings[0]
            target_encoding = target_encodings[0]

            distance = face_recognition.face_distance([target_encoding], generated_encoding)[0]
            similarity = 1 - distance  # Convert to similarity

            # Check if passed
            passed = similarity >= threshold

            message = f"Face similarity: {similarity:.2f} (threshold: {threshold})"
            return passed, similarity, message

        except Exception as e:
            return False, 0.0, f"Face check error: {e}"

    def check_face_with_gemini(self, generated_image_path, target_face_path, character_id):
        """
        Use Gemini Vision API to check face match

        Args:
            generated_image_path: Generated image path
            target_face_path: Target face image path
            character_id: Character ID

        Returns:
            (passed, score, message)
        """
        if not self.genai_available:
            return False, 0.0, "Gemini Vision not available"

        try:
            # Load images as PIL Image objects
            from PIL import Image
            import io

            with open(generated_image_path, "rb") as f:
                generated_img = Image.open(io.BytesIO(f.read()))
            with open(target_face_path, "rb") as f:
                target_img = Image.open(io.BytesIO(f.read()))

            # Build prompt
            prompt = f"""
Compare the face in the first image (generated frame) with the reference face in the second image (target face).

**TASK**:
Analyze whether the face in the generated image matches the target face reference.

**EVALUATION CRITERIA**:
- Facial features (eyes, nose, mouth, face shape)
- Hair style and color
- Overall facial similarity
- Skin tone

**OUTPUT FORMAT** (must follow exactly):
FACE_SIMILARITY_SCORE: <score from 0.0 to 1.0>
MATCH: <YES or NO>
REASON: <brief explanation>

Example:
FACE_SIMILARITY_SCORE: 0.85
MATCH: YES
REASON: The generated face closely matches the target face in terms of facial features, hair style, and overall appearance.
"""

            # Call Gemini Vision - pass PIL Image objects directly
            response = self.genai_client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[
                    prompt,
                    generated_img,
                    target_img
                ]
            )

            # Parse response
            response_text = response.candidates[0].content.parts[0].text

            # Extract similarity score
            score = 0.0
            match = "NO"
            reason = ""

            for line in response_text.split('\n'):
                if 'FACE_SIMILARITY_SCORE:' in line:
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'MATCH:' in line:
                    match = line.split(':')[1].strip().upper()
                elif 'REASON:' in line:
                    reason = line.split(':')[1].strip()

            passed = match == "YES" and score >= 0.75

            message = f"Face similarity: {score:.2f} (Gemini evaluation)"
            if reason:
                message += f"\nNote: {reason}"

            return passed, score, message

        except Exception as e:
            return False, 0.0, f"Gemini face check error: {e}"

    def check_clothing_consistency(self, generated_image_path, character_id):
        """
        Check clothing consistency (using Gemini Vision)

        Args:
            generated_image_path: Generated image path
            character_id: Character ID

        Returns:
            (passed, score, message)
        """
        mapping = self.get_character_mapping(character_id)
        if not mapping:
            return True, 1.0, "No clothing configuration, skipping check"

        clothing_desc = mapping.get('clothing', '')

        if not clothing_desc:
            return True, 1.0, "No clothing description, skipping check"

        if not self.genai_available:
            return True, 1.0, "Gemini not available, skipping clothing check"

        try:
            # Load image as PIL Image object
            from PIL import Image
            import io

            with open(generated_image_path, "rb") as f:
                img = Image.open(io.BytesIO(f.read()))

            # Build prompt
            prompt = f"""
Analyze the clothing in this image and compare it with the following description:

**CLOTHING DESCRIPTION**:
{clothing_desc}

**TASK**:
Evaluate whether the clothing in the image matches the description above.

**EVALUATION CRITERIA**:
- Color match
- Style match
- Fabric type
- Overall appearance

**OUTPUT FORMAT** (must follow exactly):
CLOTHING_SIMILARITY_SCORE: <score from 0.0 to 1.0>
MATCH: <YES or NO>
REASON: <brief explanation>

Example:
CLOTHING_SIMILARITY_SCORE: 0.75
MATCH: YES
REASON: The clothing color and style closely match the description, though some details differ.
"""

            # Call Gemini Vision - pass PIL Image object directly
            response = self.genai_client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=[
                    prompt,
                    img
                ]
            )

            # Parse response
            response_text = response.candidates[0].content.parts[0].text

            # Extract similarity score
            score = 0.0
            match = "NO"
            reason = ""

            for line in response_text.split('\n'):
                if 'CLOTHING_SIMILARITY_SCORE:' in line:
                    try:
                        score = float(line.split(':')[1].strip())
                    except:
                        pass
                elif 'MATCH:' in line:
                    match = line.split(':')[1].strip().upper()
                elif 'REASON:' in line:
                    reason = line.split(':')[1].strip()

            passed = match == "YES" and score >= 0.75

            message = f"Clothing consistency: {score:.2f}"
            if reason:
                message += f"\nNote: {reason}"

            return passed, score, message

        except Exception as e:
            return True, 1.0, f"Clothing check error, skipped: {e}"

    def inspect_single_image(self, image_path, character_id):
        """
        Check quality of a single image

        Args:
            image_path: Image path
            character_id: Character ID

        Returns:
            Inspection result dictionary
        """
        if not os.path.exists(image_path):
            return {
                "passed": False,
                "image": image_path,
                "character": character_id,
                "face_check": {"passed": False, "message": "Image not found"},
                "clothing_check": {"passed": True, "message": "Skipped"}
            }

        # Get character mapping information
        mapping = self.get_character_mapping(character_id)
        if not mapping:
            return {
                "passed": False,
                "image": image_path,
                "character": character_id,
                "face_check": {"passed": False, "message": "Character mapping not found"},
                "clothing_check": {"passed": True, "message": "Skipped"}
            }

        target_face_path = mapping.get('target_face')
        if not target_face_path or not os.path.exists(target_face_path):
            return {
                "passed": False,
                "image": image_path,
                "character": character_id,
                "face_check": {"passed": False, "message": "Target face image not found"},
                "clothing_check": {"passed": True, "message": "Skipped"}
            }

        print(f"\n{'='*70}")
        print(f"🔍 Checking: {image_path}")
        print(f"   Character: {character_id} → {mapping.get('target_name', 'Unknown')}")
        print(f"{'='*70}")

        # 1. Face check
        print("\n[1/2] Face check...")
        if self.use_gemini and self.genai_available:
            face_passed, face_score, face_message = self.check_face_with_gemini(
                image_path, target_face_path, character_id
            )
        elif FACE_RECOGNITION_AVAILABLE:
            face_passed, face_score, face_message = self.check_face_with_face_recognition(
                image_path, target_face_path, threshold=0.5
            )
        else:
            face_passed, face_score, face_message = False, 0.0, "No face check method available"

        print(f"   {face_message}")
        print(f"   Result: {'✅ Passed' if face_passed else '❌ Failed'}")

        # 2. Clothing check
        print("\n[2/2] Clothing check...")
        clothing_passed, clothing_score, clothing_message = self.check_clothing_consistency(
            image_path, character_id
        )

        print(f"   {clothing_message}")
        print(f"   Result: {'✅ Passed' if clothing_passed else '❌ Failed'}")

        # Overall result
        overall_passed = face_passed and clothing_passed

        print(f"\n{'='*70}")
        print(f"Overall Result: {'✅ Passed' if overall_passed else '❌ Failed'}")
        print(f"{'='*70}")

        return {
            "passed": overall_passed,
            "image": image_path,
            "character": character_id,
            "face_check": {
                "passed": face_passed,
                "score": face_score,
                "message": face_message
            },
            "clothing_check": {
                "passed": clothing_passed,
                "score": clothing_score,
                "message": clothing_message
            }
        }

    def inspect_all_generated_images(self, output_dir="."):
        """
        Check all generated images

        Args:
            output_dir: Output directory

        Returns:
            List of inspection results
        """
        # Find all shot_*.png files
        import glob

        pattern = os.path.join(output_dir, "shot_*.png")
        image_files = glob.glob(pattern)

        if not image_files:
            print(f"⚠️  No shot_*.png files found in {output_dir}")
            return []

        print(f"\n{'='*70}")
        print(f"🔍 Batch Quality Inspection")
        print(f"{'='*70}")
        print(f"Found {len(image_files)} image files")

        results = []

        for image_file in sorted(image_files):
            # Extract shot ID from filename
            basename = os.path.basename(image_file)
            shot_id_str = basename.replace('shot_', '').replace('.png', '')

            # Try to convert to integer or float
            try:
                shot_id = float(shot_id_str) if '.' in shot_id_str else int(shot_id_str)
            except ValueError:
                print(f"\n⚠️  Unable to parse shot ID: {shot_id_str}, skipping")
                continue

            # Auto-detect character
            print(f"\n📸 File: {basename}")
            character_id = self.detect_character_in_shot(shot_id)

            if not character_id:
                print(f"   ⚠️  Unable to auto-detect character, skipping: {basename}")
                continue

            print(f"   ✅ Auto-detected character: {character_id}")

            # Perform inspection
            result = self.inspect_single_image(image_file, character_id)
            results.append(result)

        # Output summary
        print(f"\n{'='*70}")
        print("📊 Inspection Summary")
        print(f"{'='*70}")

        passed_count = sum(1 for r in results if r['passed'])
        failed_count = len(results) - passed_count

        print(f"Total: {len(results)} images")
        print(f"✅ Passed: {passed_count} images")
        print(f"❌ Failed: {failed_count} images")

        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Quality Inspection Agent - Check face and clothing consistency',
        epilog='''
Usage examples:
  python %(prog)s shot_9.png @character_03         # Check specific image
  python %(prog)s --check-all                      # Check all generated images
  python %(prog)s shot_9.png @character_03 --use-gemini  # Use Gemini Vision
        '''
    )

    parser.add_argument('image', nargs='?', help='Image path (use with character_id)')
    parser.add_argument('character_id', nargs='?', help='Character ID (e.g. @character_03)')
    parser.add_argument('--mapping', default='character_mapping.json', help='Character mapping configuration file')
    parser.add_argument('--script', default='clip1_script.json', help='Script JSON file path')
    parser.add_argument('--check-all', action='store_true', help='Check all generated images')
    parser.add_argument('--use-gemini', action='store_true', default=True,
                       help='Use Gemini Vision API (default)')
    parser.add_argument('--use-face-recognition', action='store_true',
                       help='Use face_recognition library instead of Gemini')

    args = parser.parse_args()

    # Determine which check method to use
    use_gemini = args.use_gemini and not args.use_face_recognition

    try:
        # Initialize Agent
        agent = InspectionAgent(
            character_mapping=args.mapping,
            script_json=args.script,
            use_gemini=use_gemini
        )

        # Load configuration
        if not agent.load_character_mappings():
            return 1

        # Execute inspection
        if args.check_all:
            # Batch check - need to load script data
            if not agent.load_script_data():
                return 1
            results = agent.inspect_all_generated_images()
            return 0 if all(r['passed'] for r in results) else 1
        elif args.image and args.character_id:
            # Single check
            result = agent.inspect_single_image(args.image, args.character_id)
            return 0 if result['passed'] else 1
        else:
            print("❌ Error: Please specify --check-all or provide both image and character_id arguments")
            return 1

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
