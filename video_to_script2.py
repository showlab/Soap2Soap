import argparse
import os
import time
import json
import re
import subprocess
from google import genai
import cv2  # For getting video duration
from scenedetect import detect, AdaptiveDetector

# 1. Configure command line arguments (removed --input_script)
def parse_arguments():
    parser = argparse.ArgumentParser(description="Gemini Video Scene Labeling Tool")
    # --input_script removed because the model now extracts characters from video itself
    parser.add_argument("--video", required=True, help="Path to the video file")
    parser.add_argument("--transcript", required=True, help="Path to Whisper output JSON file")
    parser.add_argument("--output", required=True, help="Path to save the final script JSON")
    return parser.parse_args()

# 2. Helper function: Extract JSON
def extract_json_from_response(text):
    """
    Extract JSON content from Gemini response

    Supports multiple formats:
    1. ```json ... ``` code blocks
    2. ``` ... ``` code blocks
    3. Explanatory text followed by JSON
    4. Pure JSON text
    """
    # Method 1: Extract ```json ... ``` code blocks
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1)
        except json.JSONDecodeError:
            pass  # Continue trying other methods

    # Method 2: Extract ``` ... ``` code blocks
    match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        try:
            json.loads(match.group(1))
            return match.group(1)
        except json.JSONDecodeError:
            pass  # Continue trying other methods

    # Method 3: Find content between first { and last }
    # Handle cases with explanatory text before JSON
    first_brace = text.find('{')
    last_brace = text.rfind('}')

    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        extracted = text[first_brace:last_brace + 1]
        # Verify extracted content is valid JSON
        try:
            json.loads(extracted)
            return extracted
        except json.JSONDecodeError:
            pass  # Continue trying other methods

    # Method 4: Try to fix common JSON format issues
    # For example: double braces, extra commas, comments, etc.
    try:
        cleaned_text = text

        # 4.1 Fix double braces issue {{ → {
        # Only replace at JSON start and end to avoid breaking string content
        cleaned_text = cleaned_text.strip()
        if cleaned_text.startswith('{{') and cleaned_text.endswith('}}'):
            # Remove {{ from start and }} from end
            cleaned_text = '{' + cleaned_text[2:-2] + '}'
        elif cleaned_text.startswith('{{{') and cleaned_text.endswith('}}}'):
            # Handle triple braces case
            cleaned_text = '{{' + cleaned_text[3:-3] + '}}'

        # 4.2 Fix illegal number formats (leading zeros issue)
        # Convert 00.00, 01.5 etc. to 0.0, 1.5
        # Match number format after colon (may have leading zeros)
        def fix_leading_zeros(match):
            number = match.group(1)
            # Remove leading zeros, keep decimal point
            if '.' in number:
                # 00.00 -> 0.00, 01.5 -> 1.5
                integer_part = number.split('.')[0]
                decimal_part = number.split('.')[1]
                # Remove leading zeros from integer part
                integer_part = integer_part.lstrip('0') or '0'
                return f': {integer_part}.{decimal_part}'
            else:
                # Integer
                number_fixed = str(int(number))
                return f': {number_fixed}'

        # Match pattern: colon followed by number (may have leading zeros)
        cleaned_text = re.sub(r':\s*(0\d+\.?\d*)', fix_leading_zeros, cleaned_text)

        # 4.3 Remove possible comments
        lines = cleaned_text.split('\n')
        json_lines = []
        in_string = False
        for line in lines:
            # Simple comment removal logic
            if '//' in line and not in_string:
                line = line.split('//')[0]
            json_lines.append(line)
            in_string = '"' in line

        cleaned_text = '\n'.join(json_lines)

        # Try to parse directly
        json.loads(cleaned_text)
        return cleaned_text
    except:
        pass

    # Method 5: If all else fails, return original text
    return text

# 3. Helper function: Scene detection
def detect_scenes(video_path):
    """
    Detect scene transition points in video
    Return list of scenes, each containing (start_time, end_time)
    """
    print(f"--> Detecting scenes in video...")

    # Pass video path directly, don't use open_video()
    # detect() function handles video opening automatically, avoiding VideoStreamCv2 type issues
    scene_list = detect(video_path, AdaptiveDetector())

    # Convert to more usable format
    scenes = []
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds()
        end_time = scene[1].get_seconds()
        duration = end_time - start_time

        scenes.append({
            "scene_index": i + 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration
        })

        print(f"    Scene {i+1}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")

    print(f"    ✅ Detected {len(scenes)} scenes")
    return scenes

def merge_short_scenes(scenes, target_duration=8.0, min_duration=4.0):
    """
    Merge scenes that are too short into adjacent scenes to make average duration close to target_duration

    Args:
        scenes: Original scene list
        target_duration: Target average duration (seconds), default 8 seconds
        min_duration: Minimum duration threshold, scenes shorter than this will be considered for merging

    Returns:
        Merged scene list
    """
    print(f"\n--> Merging short scenes (target: {target_duration}s, min: {min_duration}s)...")

    if not scenes:
        return scenes

    merged_scenes = []
    i = 0

    while i < len(scenes):
        current_scene = scenes[i]
        current_duration = current_scene["duration"]

        # If current scene is too short, try merging with next scene
        if current_duration < min_duration and i + 1 < len(scenes):
            next_scene = scenes[i + 1]
            merged_duration = current_duration + next_scene["duration"]

            # Only merge if merged duration doesn't exceed target_duration * 1.5
            if merged_duration <= target_duration * 1.5:
                # Merge two scenes
                merged = {
                    "scene_index": current_scene["scene_index"],
                    "start_time": current_scene["start_time"],
                    "end_time": next_scene["end_time"],
                    "duration": merged_duration,
                    "merged_from": [
                        current_scene["scene_index"],
                        next_scene["scene_index"]
                    ]
                }
                merged_scenes.append(merged)

                print(f"    Merged scenes {current_scene['scene_index']} & {next_scene['scene_index']} "
                      f"({current_duration:.2f}s + {next_scene['duration']:.2f}s → {merged_duration:.2f}s)")

                i += 2  # Skip the next scene that was merged
                continue

        # Don't merge, add directly
        merged_scenes.append(current_scene)
        i += 1

    # Statistics
    original_count = len(scenes)
    merged_count = len(merged_scenes)
    reduction_rate = (1 - merged_count / original_count) * 100

    original_avg = sum(s["duration"] for s in scenes) / original_count
    merged_avg = sum(s["duration"] for s in merged_scenes) / merged_count

    print(f"    ✅ Scenes: {original_count} → {merged_count} (reduced by {reduction_rate:.1f}%)")
    print(f"    Original avg duration: {original_avg:.2f}s")
    print(f"    Merged avg duration: {merged_avg:.2f}s")
    print(f"    API calls saved: {original_count - merged_count}")

    return merged_scenes

# 4. Helper function: Read Whisper JSON and convert to text
def get_video_duration(video_path):
    """Get total video duration (seconds)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def detect_video_aspect_ratio(video_path):
    """
    Use ffprobe to detect video aspect ratio, simplified to landscape/portrait determination

    Args:
        video_path: Video file path

    Returns:
        Dictionary containing video aspect ratio information:
        {
            "width": Width,
            "height": Height,
            "aspect_ratio": "16:9" or "9:16" (only these two)
            "ratio_decimal": Aspect ratio in decimal form
        }
    """
    try:
        # Use ffprobe to get video stream information
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Get width and height
        width = int(data['streams'][0]['width'])
        height = int(data['streams'][0]['height'])

        # Determine landscape or portrait
        # Landscape: width >= height (including square)
        # Portrait: width < height
        is_landscape = width >= height

        if is_landscape:
            # Landscape → 16:9 (1920x1080)
            target_width = 1920
            target_height = 1080
            aspect_ratio = "16:9"
            ratio_decimal = 1.777778
            orientation = "Landscape"
        else:
            # Portrait → 9:16 (1080x1920)
            target_width = 1080
            target_height = 1920
            aspect_ratio = "9:16"
            ratio_decimal = 0.5625
            orientation = "Portrait"

        video_info = {
            "width": target_width,
            "height": target_height,
            "aspect_ratio": aspect_ratio,
            "ratio_decimal": ratio_decimal
        }

        print(f"\n{'='*60}")
        print(f"--> Detected video aspect ratio information:")
        print(f"{'='*60}")
        print(f"    Original resolution: {width}x{height}")
        print(f"    Video orientation: {orientation}")
        print(f"    Generated ratio: {aspect_ratio}")
        print(f"    Target resolution: {target_width}x{target_height}")
        print(f"{'='*60}")

        return video_info

    except subprocess.CalledProcessError as e:
        print(f"⚠️  ffprobe execution failed: {e}")
        print(f"    Using default ratio 16:9")
        return {
            "width": 1920,
            "height": 1080,
            "aspect_ratio": "16:9",
            "ratio_decimal": 1.777778
        }
    except Exception as e:
        print(f"⚠️  Failed to detect video aspect ratio: {e}")
        print(f"    Using default ratio 16:9")
        return {
            "width": 1920,
            "height": 1080,
            "aspect_ratio": "16:9",
            "ratio_decimal": 1.777778
        }

def filter_transcript_by_time(transcript_text, start_time, end_time):
    """Filter dialogue by time range"""
    lines = transcript_text.strip().split('\n')
    filtered_lines = []

    for line in lines:
        # Extract timestamp, e.g., "[0.00s -> 2.50s] Text"
        match = re.match(r'\[(\d+\.?\d*)s\s*->\s*(\d+\.?\d*)s\]\s*(.*)', line)
        if match:
            line_start = float(match.group(1))
            line_end = float(match.group(2))
            text = match.group(3)

            # If this time segment overlaps with current segment
            if line_start < end_time and line_end > start_time:
                filtered_lines.append(line)

    return '\n'.join(filtered_lines) if filtered_lines else "No dialogue in this segment"

def load_and_format_whisper(json_path):
    """
    Load Whisper-transcribed Chinese text

    Note: Using --task transcribe, output is original Chinese text
    Translation will be intelligently processed in subsequent steps using video context
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        formatted_text = ""
        # Compatible with Whisper native JSON format
        segments = data.get('text', '')
        segments_list = data.get('segments', [])

        if not segments_list and 'text' in data:
             return data['text']

        for seg in segments_list:
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            formatted_text += f"[{start:.2f}s -> {end:.2f}s] {text}\n"

        return formatted_text
    except Exception as e:
        print(f"Warning: Could not parse Whisper JSON ({e}). Using raw file content.")
        with open(json_path, 'r', encoding='utf-8') as f:
            return f.read()


def smart_translate_transcript(client, myfile, chinese_transcript):
    """
    Intelligently translate Chinese transcript text, handling omitted subjects

    Analyze through video context to correctly infer Chinese sentences with omitted subjects
    For example: "去吃饭" -> "He/She/I goes to eat" (determined based on characters in video)
    """
    print(f"\n{'='*60}")
    print(f"--> Smart Translation with Video Context")
    print(f"{'='*60}")

    prompt = f"""
You are translating Chinese dialogue from a video to English, with CRITICAL attention to context.

--- CHINESE TRANSCRIPT (WITH TIMESTAMPS) ---
{chinese_transcript}

--- YOUR TASK ---
Watch the video and translate each Chinese dialogue segment to English.

**CRITICAL RULE FOR CHINESE OMISSION SUBJECTS**:
Chinese frequently omits subjects (e.g., "去吃饭" = "[go] eat [rice]", "很高兴" = "[very] happy").
When translating to English, you MUST infer the correct subject based on:

1. **VISUAL CONTEXT**: Who is speaking in the video at that timestamp?
2. **CONVERSATION FLOW**: What was discussed before and after?
3. **CHARACTER IDENTITY**: Is the speaker the protagonist, supporting character, narrator?
4. **ACTION CONTEXT**: What are they doing in the scene?

**SUBJECT INFERENCE GUIDELINES**:
- If a male character is speaking: use "he"
- If a female character is speaking: use "she"
- If the speaker is referring to themselves: use "I"
- If speaking to someone: use "you"
- If describing others: use their names or "they"
- Preserve the original tone and emotion

**EXAMPLES**:
Chinese: "[0.5s -> 2.3s] 去吃饭吧"
Context: Female character speaking to male character
Translation: "[0.5s -> 2.3s] Let's go eat."

Chinese: "[2.5s -> 4.1s] 好的，没问题"
Context: Male character responding
Translation: "[2.5s -> 4.1s] Okay, no problem."

Chinese: "[5.0s -> 7.2s] 很高兴见到你"
Context: Woman speaking to another person
Translation: "[5.0s -> 7.2s] Very happy to meet you."

**OUTPUT FORMAT**:
Return the translated transcript in the SAME format as input:
[timestamp_start -> timestamp_end] English translation

Maintain ALL timestamps exactly as they appear in the Chinese transcript.
"""

    print("    Sending translation request with video context...")
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[myfile, prompt]
        )

        if not response.text:
            print("    ⚠️  No response for translation")
            return chinese_transcript  # Return original text

        translated_text = response.text.strip()
        print(f"    ✅ Translation completed")
        return translated_text

    except Exception as e:
        print(f"    ⚠️  Translation failed: {e}")
        print(f"    Using original Chinese transcript")
        return chinese_transcript  # Return original text on error


def extract_character_names(client, myfile, translated_transcript, character_roster):
    """
    Extract character names from translated transcript text and video

    Establish mapping from character_id to real names
    For example: @character_01 -> "Emma"
    """
    print(f"\n{'='*60}")
    print(f"--> Extracting Character Names from Dialogue")
    print(f"{'='*60}")

    prompt = f"""
You are analyzing a video's dialogue to extract ALL CHARACTER NAMES, TITLES, AND FORMS OF ADDRESS mentioned or used.

--- TRANSLATED TRANSCRIPT (ENGLISH) ---
{translated_transcript}

--- GLOBAL CHARACTER ROSTER ---
{json.dumps(character_roster, indent=2, ensure_ascii=False)}

--- YOUR TASK ---
Watch the video and extract ALL names, titles, and forms of address for each character from the dialogue.

**PRIORITY SYSTEM FOR CHARACTER NAMES**:
1. **NEVER REPLACE** names that came from on-screen labels (source: "on_screen_label")
2. **ADD NEW NAMES** from dialogue to the existing name collections
3. **CATEGORIZE** each name you find into the correct category

**INSTRUCTIONS**:

**A. Extract ALL Forms of Address from Dialogue**

For each character in the roster, find EVERY way they are referred to in dialogue:

1. **Direct Address**: When someone speaks TO them
   - "Hey, Emma, come here!" → "Emma" (alias)
   - "Ms. Smith, wait!" → "Ms. Smith" (title)
   - "Mom, can you help?" → "Mom" (familial)

2. **References**: When someone speaks ABOUT them
   - "Where did Sarah go?" → "Sarah" (alias)
   - "The Doctor will see you now" → "Doctor" (role/title)
   - "Your father is waiting" → "father" (familial reference)

3. **Self-Introduction**: When they introduce themselves
   - "Hi, I'm Michael Chen" → "Michael Chen" (primary_name)
   - "I'm Dr. Smith" → "Dr. Smith" (title)

4. **Third-Person References**: When referred to in third person
   - "Tell the Detective I'm ready" → "Detective" (role/title)

**B. Categorize Each Name You Find**

For each name you extract, determine which category it belongs to:

- **primary_name**: Full formal name (e.g., "Emma Smith", "Michael Chen")
- **aliases**: Nicknames, first names only, shortened versions (e.g., "Emma", "Em", "Emmy", "Mike")
- **titles**: Formal titles and honorifics (e.g., "Mr.", "Ms.", "Dr.", "Professor", "Detective", "Officer", "Mr. Smith", "Dr. Chen")
- **roles**: Professional or story roles (e.g., "Doctor", "Teacher", "Protagonist", "Antagonist", "Detective")
- **familial**: Family-based names (e.g., "Mom", "Dad", "Mother", "Father", "Sarah's Mom", "John's Father")

**C. Add to Existing Collections**

- Check what names each character ALREADY has (from on-screen labels)
- ONLY ADD new names from dialogue - DO NOT replace existing names
- If a name from dialogue already exists in a category, don't add it again
- Track the SOURCE as "dialogue" for all names extracted from dialogue

**OUTPUT FORMAT**:
Return a JSON object that ADDS TO the existing name collections:
{{
  "character_name_updates": {{
    "@character_01": {{
      "names_to_add": {{
        "primary_name": null,  // null means don't change existing
        "aliases": ["Em", "Emmy"],  // Add these if not already present
        "aliases_sources": {{"Em": "dialogue", "Emmy": "dialogue"}},
        "titles": ["Ms. Smith"],  // Add these if not already present
        "titles_sources": {{"Ms. Smith": "dialogue"}},
        "roles": [],  // No new roles from dialogue
        "roles_sources": {{}},
        "familial": ["Mom", "Sarah's Mom"],  // Add these
        "familial_sources": {{"Mom": "dialogue", "Sarah's Mom": "dialogue"}}
      }},
      "evidence": "Found in dialogue: 'Em' at [15.3s], 'Emmy' at [45.2s], 'Ms. Smith' at [67.8s], 'Mom' at [89.1s]"
    }},
    "@character_02": {{
      "names_to_add": {{
        "primary_name": "Michael Chen",  // Set this if it was missing
        "aliases": ["Mike"],
        "aliases_sources": {{"Mike": "dialogue"}},
        "titles": ["Dr. Chen", "Professor"],
        "titles_sources": {{"Dr. Chen": "dialogue", "Professor": "dialogue"}},
        "roles": [],
        "roles_sources": {{}},
        "familial": []
      }},
      "evidence": "Self-introduced as 'Michael Chen' at [25.8s], called 'Mike' at [90.2s], referred to as 'Dr. Chen' and 'Professor' in dialogue"
    }}
  }}
}}

**CRITICAL RULES**:
- **NEVER REPLACE** names that came from on-screen labels (source: "on_screen_label")
- **ONLY ADD** new names from dialogue - preserve existing names
- **EXTRACT EVERY FORM OF ADDRESS** - not just proper names, but titles, roles, and familial terms
- **CATEGORIZE CORRECTLY** - pay attention to the context to determine the right category
- **TRACK EVIDENCE** - list where in the dialogue each name was found

**CRITICAL OUTPUT RULES (MUST FOLLOW)**:
1. Output ONLY the JSON object above - nothing else
2. DO NOT include any explanations or text outside the JSON
3. Your response must start with '{{' and end with '}}'
4. NO markdown code blocks
5. Directly output the raw JSON only

**IMPORTANT**:
- Only use names that are ACTUALLY SPOKEN in the dialogue
- Do NOT invent names
- If uncertain, use descriptive placeholders
- Be precise about which character ID gets which name
"""

    print("    Extracting character names from dialogue and video...")
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[myfile, prompt]
        )

        if not response.text:
            print("    ⚠️  No response for name extraction")
            return {}

        cleaned_json = extract_json_from_response(response.text)

        try:
            name_updates_data = json.loads(cleaned_json)
            character_name_updates = name_updates_data.get("character_name_updates", {})

            print(f"    ✅ Extracted name updates for {len(character_name_updates)} characters")
            for char_id, update_info in character_name_updates.items():
                names_to_add = update_info.get("names_to_add", {})
                evidence = update_info.get("evidence", "")
                print(f"       {char_id}:")
                if names_to_add.get("aliases"):
                    print(f"          aliases: {names_to_add['aliases']}")
                if names_to_add.get("titles"):
                    print(f"          titles: {names_to_add['titles']}")
                if names_to_add.get("familial"):
                    print(f"          familial: {names_to_add['familial']}")
                print(f"          evidence: {evidence[:80]}...")

            return character_name_updates

        except json.JSONDecodeError as e:
            print(f"    ⚠️  Failed to parse name updates JSON: {e}")
            return {}

    except Exception as e:
        print(f"    ❌ Error extracting character names: {e}")
        return {}

def identify_all_characters(client, myfile, full_whisper_text, video_duration):
    """
    Phase 1: Global Character Identification
    Analyze the entire video to identify all characters and create profiles
    """
    print(f"\n{'='*60}")
    print(f"--> PHASE 1: Global Character Identification")
    print(f"{'='*60}")

    prompt = f"""
You are analyzing an entire video to identify ALL unique characters that appear throughout.

--- AUDIO TRANSCRIPT (FULL VIDEO) ---
The complete speech transcript:
{full_whisper_text}

--- YOUR TASK ---
Watch the entire video and create a comprehensive character roster. For each unique person you see:

1. Assign them a UNIQUE ID in the format @character_XX (XX is 01, 02, 03, etc.)

2. **EXTRACT ALL CHARACTER NAMES AND TITLES** (CRITICAL - HIGHEST PRIORITY):

   **A. From On-Screen Text Labels (FIRST APPEARANCE)**
   - When a character FIRST APPEARS in the video, look carefully for text labels/overlays near their face
   - Common formats: "Name", "Name - Role", "Name: Description", or similar text annotations
   - These on-screen text labels are the MOST RELIABLE source
   - Extract ALL information from on-screen text:
     - Primary name (full name if available)
     - Titles (Dr., Mr., Ms., Professor, Detective, etc.)
     - Roles (Protagonist, Antagonist, Doctor, Teacher, etc.)

   **B. Categorize the Names**
   For each character, collect names in these categories:
   - **primary_name**: Main/full name (e.g., "Emma Smith", "Michael Chen")
   - **aliases**: Nicknames, shortened versions, first name only (e.g., "Em", "Emmy", "Mike")
   - **titles**: Formal titles and honorifics (e.g., "Mr. Smith", "Dr. Chen", "Professor", "Detective")
   - **roles**: Character roles in story or profession (e.g., "Protagonist", "Doctor", "Teacher")
   - **familial**: Family-based names (e.g., "Mom", "Dad", "Sarah's Mom")

   **C. Track Sources**
   For EVERY name you collect, record where it came from:
   - "on_screen_label" - From text overlay in video
   - "dialogue" - From dialogue (if name is mentioned in this stage)
   - "id_fallback" - No name found

   **Example**:
   If you see on-screen text "Emma Smith - Protagonist", and later in dialogue hear "Em", "Ms. Smith", and "Mom":
   - primary_name: "Emma Smith" (source: on_screen_label)
   - aliases: ["Emma", "Em"] (sources: Emma=on_screen_label, Em=dialogue)
   - titles: ["Ms. Smith"] (source: dialogue)
   - roles: ["Protagonist"] (source: on_screen_label)
   - familial: ["Mom"] (source: dialogue)

3. Describe their identifying traits using the STRICT criteria below:
   - **Physical Attributes**: Gender, approximate age, body type, skin tone
   - **Hair**: Color, length, style (including changes throughout the video)
   - **Face**: Facial hair, glasses, distinctive marks, scars
   - **Clothing**: Document different outfits worn in different scenes (colors, patterns, style, layers)
   - **Accessories**: Hats, jewelry, bags, glasses
   - **Distinctive Features**: Any unique trait that separates them from others

4. List the scenes/time ranges where each character appears and what they wear in each

--- IMPORTANT RULES ---
- Assign IDs based on FIRST APPEARANCE in the video
- The same person must keep the SAME ID throughout the entire video, **EVEN IF THEIR CLOTHING CHANGES**
- Different people must have DIFFERENT IDs
- Use PERMANENT physical traits (face, hair, body) as primary identifiers
- Use CLOTHING as secondary identifier (document changes, but don't let it create a new ID)
- If a person changes clothes but has the same face/body/hair, use the SAME ID
- Only count REAL characters (ignore background extras)
- **PAY SPECIAL ATTENTION**: When a character first appears, PAUSE and look for text labels around them

--- OUTPUT FORMAT ---
Return a JSON object:
{{
  "characters": [
    {{
      "character_id": "@character_01",
      "names": {{
        "primary_name": "Emma Smith",
        "primary_source": "on_screen_label",
        "aliases": ["Emma", "Em", "Emmy"],
        "aliases_sources": {{"Emma": "on_screen_label", "Em": "dialogue", "Emmy": "dialogue"}},
        "titles": ["Ms. Smith", "Mrs. Smith"],
        "titles_sources": {{"Ms. Smith": "dialogue", "Mrs. Smith": "dialogue"}},
        "roles": ["Protagonist", "Detective"],
        "roles_sources": {{"Protagonist": "on_screen_label", "Detective": "dialogue"}},
        "familial": ["Mom", "Sarah's Mom"],
        "familial_sources": {{"Mom": "dialogue", "Sarah's Mom": "dialogue"}}
      }},
      "physical_attributes": "Gender, age, body type, skin tone",
      "hair": "Color, length, style",
      "face": "Facial features, glasses, marks",
      "clothing_variations": [
        {{"scene": "Scene 1", "description": "Light blue business suit, white blouse"}},
        {{"scene": "Scene 5", "description": "Black leather jacket, skirt"}}
      ],
      "accessories": "Glasses, jewelry, etc.",
      "distinctive_features": "Unique identifying traits",
      "first_appearance": "0.0s",
      "scenes": ["Scene 1", "Scene 3", "Scene 5"]
    }},
    {{
      "character_id": "@character_02",
      "names": {{
        "primary_name": "Michael",
        "primary_source": "on_screen_label",
        "aliases": ["Mike", "Mikey"],
        "aliases_sources": {{"Mike": "dialogue", "Mikey": "dialogue"}},
        "titles": ["Dr. Chen", "Professor Chen"],
        "titles_sources": {{"Dr. Chen": "on_screen_label", "Professor Chen": "dialogue"}},
        "roles": ["Supporting Character", "Doctor"],
        "roles_sources": {{"Supporting Character": "on_screen_label", "Doctor": "dialogue"}},
        "familial": [],
        "familial_sources": {{}}
      }},
      "physical_attributes": "Male, 30s, athletic build",
      "hair": "Platinum blonde, medium length, styled",
      "face": "Clean-shaven, no glasses",
      "clothing_variations": [
        {{"scene": "Scene 1", "description": "Patterned grey jacket, white t-shirt"}},
        {{"scene": "Scene 2", "description": "White suit, glasses"}}
      ],
      "accessories": "None",
      "distinctive_features": "Platinum blonde hair is key identifier",
      "first_appearance": "0.0s",
      "scenes": ["Scene 1", "Scene 2", "Scene 4"]
    }}
  ]
}}

**IMPORTANT NOTES ON CHARACTER NAMES**:

**names object structure:**
- **primary_name**: The character's main/full name (e.g., "Emma Smith", "Michael Chen")
- **primary_source**: Where the primary name came from ("on_screen_label" / "dialogue" / "id_fallback")
- **aliases**: Alternative names the character is called (nicknames, shortened versions, first name only)
  - Examples: "Em", "Emmy", "Mike", "Mikey"
- **titles**: Formal titles and honorifics
  - Examples: "Mr. Smith", "Ms. Smith", "Dr. Chen", "Professor Chen", "Detective", "Officer"
- **roles**: Character roles in the story or profession
  - Examples: "Protagonist", "Antagonist", "Doctor", "Teacher", "Detective"
- **familial**: Family-based names
  - Examples: "Mom", "Dad", "Sarah's Mom", "John's Father"

***_sources fields**: For each name category, track where each name came from:
- "on_screen_label" - From text overlay in video
- "dialogue" - From dialogue (spoken by someone)
- "id_fallback" - No name found

**IF NO ON-SCREEN TEXT FOUND**: Set primary_name to empty string "", primary_source to "id_fallback", but keep collecting names from dialogue in the next stage

**CRITICAL OUTPUT RULES (MUST FOLLOW)**:
1. Output ONLY the JSON object above - nothing else
2. DO NOT include any explanations or text outside the JSON
3. Your response must start with '{{' and end with '}}'
4. NO markdown code blocks
5. Directly output the raw JSON only
"""

    print("    Sending character identification request to Gemini...")
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[myfile, prompt]
        )

        if not response.text:
            print("    ⚠️  No response for character identification")
            return {}

        cleaned_json = extract_json_from_response(response.text)

        try:
            char_data = json.loads(cleaned_json)
            print(f"    ✅ Identified {len(char_data.get('characters', []))} unique characters")

            for char in char_data.get('characters', []):
                print(f"       - {char['character_id']}: {char.get('description', 'N/A')[:50]}...")

            return char_data
        except json.JSONDecodeError as e:
            print(f"    ⚠️  Failed to parse character JSON: {e}")
            return {}

    except Exception as e:
        print(f"    ❌ Error identifying characters: {e}")
        return {}


def detect_major_scenes(client, myfile, video_duration):
    """
    Phase 2: Major Scene Detection
    Identify major scene changes in video (environment/location changes)
    """
    print(f"\n{'='*60}")
    print(f"--> PHASE 2: Major Scene Detection")
    print(f"{'='*60}")

    prompt = f"""
You are analyzing a video to identify MAJOR SCENES (locations/environments).

--- YOUR TASK ---
Watch the entire video and identify all major scenes where the environment/setting changes.

A "MAJOR SCENE" is defined by:
- LOCATION CHANGE: Moving to a different place (e.g., office → home → restaurant)
- LIGHTING CHANGE: Significant shift in lighting style (e.g., daylight → indoor artificial)
- SETTING CHANGE: Different background environment

NOT a major scene:
- Camera angle changes within the same room
- Different shot sizes of the same location
- Minor camera movements

For each major scene, provide:
1. Unique ID (major_scene_01, major_scene_02, etc.)
2. Start and end time
3. Description of the location/setting
4. Dominant lighting style
5. **ENVIRONMENT-ONLY REFERENCE DESCRIPTION** (CRITICAL - NEW)

For EACH major scene, create a detailed description of the ENVIRONMENT ONLY (NO people, NO characters).

This description will be used to generate a clean background reference image.

**CRITICAL: ENVIRONMENT ONLY - NO CHARACTERS**
- Describe the EMPTY ROOM/SPACE
- Do NOT include any people, characters, or figures
- Remove all references to characters
- Focus ONLY on: room, furniture, lighting, decor, atmosphere

**INCLUDE IN ENVIRONMENT DESCRIPTION**:
- ✅ Room layout and architecture (size, shape, ceiling height)
- ✅ Walls (color, material, texture)
- ✅ Flooring (type, color, pattern, grain)
- ✅ Windows (size, placement, style, view through them)
- ✅ Doors (type, position, handle style)
- ✅ Furniture (type, placement, colors, materials, shapes)
- ✅ Lighting fixtures (type, position, color temperature, intensity)
- ✅ Decorative elements (artwork, plants, rugs, curtains, objects)
- ✅ Atmosphere (lighting quality, mood, time of day)

**EXCLUDE FROM ENVIRONMENT DESCRIPTION**:
- ❌ All people, characters, figures
- ❌ Character actions or movements
- ❌ Character clothing or faces
- ❌ Dialogue or speech

--- OUTPUT FORMAT ---
Return a JSON object:
{{
  "major_scenes": [
    {{
      "scene_id": "major_scene_01",
      "start_time": 0.0,
      "end_time": 65.5,
      "duration": 65.5,
      "location_type": "Luxury penthouse living room",
      "setting_description": "Modern minimalist room with floor-to-ceiling windows, circular sofa",
      "lighting_style": "Natural daylight from windows, cool ambient fill",
      "color_palette": "Cool greys, whites, blues",
      "environment_description": "A modern minimalist living room with floor-to-ceiling windows on the back wall offering a city view. The room features hardwood oak flooring with warm tone and horizontal grain patterns. A curved grey sectional sofa with textured fabric is positioned in the center facing the windows. The walls are painted in cool light grey with smooth matte finish. The ceiling height is approximately 10 feet with modern recessed lighting fixtures providing soft ambient illumination. A distinctive circular floor lamp with brass stand stands in the left foreground. White sheer curtains frame the windows. The overall atmosphere is bright and airy with soft natural daylight streaming through, creating gentle diffused lighting throughout the space."
    }},
    {{
      "scene_id": "major_scene_02",
      "start_time": 65.5,
      "end_time": 156.8,
      "duration": 91.3,
      "location_type": "Office meeting room",
      "setting_description": "Corporate boardroom with large table, window background",
      "lighting_style": "Artificial indoor lighting, warm ceiling lights",
      "color_palette": "Warm browns, golds, cream",
      "environment_description": "A corporate boardroom with a large polished wooden conference table in the center. The room features floor-to-ceiling glass windows on one wall showing an office view. The walls are painted in warm cream with wood paneling accents. The flooring is dark hardwood with reflective finish. Three modern pendant lights with warm white LEDs hang above the table. Executive leather chairs surround the table. The overall atmosphere is professional and well-lit with warm artificial lighting from the ceiling fixtures."
    }}
  ]
}}

**CRITICAL OUTPUT RULES (MUST FOLLOW)**:
1. Output ONLY a valid JSON object - nothing else
2. DO NOT include any explanations or text outside the JSON
3. Your response must start with '{{' (single brace) and end with '}}' (single brace)
4. NO markdown code blocks (no ```json ... ```)
5. Use standard JSON format (no trailing commas, all strings quoted)
6. DO NOT use double braces - use single braces only
7. **TIME FORMAT**: start_time, end_time, and duration MUST be numbers (like 0.0, 65.5), NOT strings (NOT "00:00", NOT "01:08")

**IMPORTANT**: The "environment_description" field MUST describe ONLY the empty room/space with NO characters.
"""

    print("    Sending major scene detection request to Gemini...")
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[myfile, prompt]
        )

        if not response.text:
            print("    ⚠️  No response for major scene detection")
            return {}

        # Debug: Save raw response
        print(f"\n    📋 Raw Gemini Response (first 500 chars):")
        print(f"    {response.text[:500]}")

        cleaned_json = extract_json_from_response(response.text)

        # Debug: Display cleaned JSON
        print(f"\n    📋 Cleaned JSON (first 500 chars):")
        print(f"    {cleaned_json[:500]}")

        try:
            scene_data = json.loads(cleaned_json)
            print(f"    ✅ Detected {len(scene_data.get('major_scenes', []))} major scenes")

            for scene in scene_data.get('major_scenes', []):
                start = parse_time_to_seconds(scene['start_time'])
                end = parse_time_to_seconds(scene['end_time'])
                print(f"       - {scene['scene_id']}: {scene.get('location_type', 'N/A')} ({start}s - {end}s)")
                # Check if environment_description exists
                if scene.get('environment_description'):
                    print(f"         ✓ Environment description generated ({len(scene['environment_description'])} chars)")
                else:
                    print(f"         ⚠️  WARNING: No environment_description found!")

            return scene_data
        except json.JSONDecodeError as e:
            print(f"    ⚠️  Failed to parse major scene JSON: {e}")
            # Save complete response to file for debugging
            debug_file = "major_scene_debug_response.txt"
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== RAW GEMINI RESPONSE ===\n")
                    f.write(response.text)
                    f.write(f"\n\n=== CLEANED JSON ===\n")
                    f.write(cleaned_json)
                    f.write(f"\n\n=== ERROR ===\n")
                    f.write(str(e))
                print(f"    💾 Saved debug response to: {debug_file}")
            except:
                pass
            return {}

    except Exception as e:
        print(f"    ❌ Error detecting major scenes: {e}")
        return {}


def classify_character_roles(client, myfile, character_roster, video_duration):
    """
    Classify main and supporting characters based on screen time and importance

    Args:
        client: Gemini client
        myfile: Video file
        character_roster: Character roster
        video_duration: Total video duration

    Returns:
        Updated character_roster with role_classification field added to each character
    """
    print(f"\n{'='*60}")
    print(f"--> Classifying Character Roles (Main/Supporting)")
    print(f"{'='*60}")

    prompt = f"""
You are analyzing a video to classify characters into MAIN CHARACTERS and SUPPORTING CHARACTERS.

--- CHARACTER ROSTER ---
{json.dumps(character_roster, indent=2, ensure_ascii=False)}

--- YOUR TASK ---
Watch the entire video and classify each character based on:

1. **Screen Time**:
   - Main characters: Appear in MULTIPLE scenes, LONG total duration (>20% of video)
   - Supporting characters: Appear in FEW scenes, SHORT total duration (<20% of video)

2. **Story Importance**:
   - Main characters: Central to the plot, drive the story forward
   - Supporting characters: Auxiliary roles, help or hinder main characters

3. **Dialogue/Interaction**:
   - Main characters: Frequent dialogue, active participation
   - Supporting characters: Minimal dialogue, passive presence

--- CLASSIFICATION CRITERIA ---
- **MAIN CHARACTER** (主角):
  - Appears in 3+ scenes OR appears in >25% of video duration
  - Has significant dialogue or drives the plot
  - Central to the story

- **SUPPORTING CHARACTER** (配角):
  - Appears in 1-2 scenes OR appears in <25% of video duration
  - Limited dialogue or auxiliary role
  - Not central to the story

--- OUTPUT FORMAT ---
Return a JSON object with the same structure as the input character_roster,
but ADD a "role_classification" field to each character:

{{
  "characters": [
    {{
      "character_id": "@character_01",
      ... all existing fields ...
      "role_classification": "main"  // or "supporting"
    }},
    ...
  ]
}}

IMPORTANT:
- Add "role_classification": "main" for main characters
- Add "role_classification": "supporting" for supporting characters
- Keep ALL existing fields unchanged
"""

    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[myfile, prompt]
        )

        response_text = response.text

        # Extract JSON
        cleaned_json = extract_json_from_response(response_text)

        if cleaned_json:
            updated_roster = json.loads(cleaned_json)
            characters = updated_roster.get("characters", [])

            main_count = 0
            supporting_count = 0

            for char in characters:
                role = char.get("role_classification", "supporting")
                char_id = char.get("character_id", "unknown")
                name = char.get("names", {}).get("primary_name", "Unknown")

                if role == "main":
                    main_count += 1
                    print(f"    ✅ MAIN: {char_id} ({name})")
                else:
                    supporting_count += 1
                    print(f"    📋 SUPPORTING: {char_id} ({name})")

            print(f"\n    Summary: {main_count} main, {supporting_count} supporting")
            return updated_roster
        else:
            print(f"    ⚠️  Could not extract JSON from response")
            return character_roster  # Return original roster

    except Exception as e:
        print(f"    ❌ Error classifying character roles: {e}")
        # If failed, default all characters to supporting
        for char in character_roster.get("characters", []):
            char["role_classification"] = "supporting"
        return character_roster


def build_scene_wardrobe(client, myfile, major_scenes, character_roster):
    """
    Phase 3: Build character wardrobe profiles for each major scene
    Ensure clothing descriptions are consistent within the same scene
    """
    print(f"\n{'='*60}")
    print(f"--> PHASE 3: Building Scene Wardrobe")
    print(f"{'='*60}")

    wardrobe_data = {"scene_wardrobes": {}}

    for major_scene in major_scenes.get("major_scenes", []):
        scene_id = major_scene["scene_id"]
        start_time = parse_time_to_seconds(major_scene["start_time"])
        end_time = parse_time_to_seconds(major_scene["end_time"])

        print(f"\n    Analyzing wardrobe for {scene_id} ({start_time}s - {end_time}s)...")

        prompt = f"""
You are analyzing a specific time segment of a video to create EXACT clothing descriptions for characters.

--- TIME SEGMENT ---
Scene ID: {scene_id}
Start: {start_time}s
End: {end_time}s

--- GLOBAL CHARACTER ROSTER ---
{json.dumps(character_roster, indent=2, ensure_ascii=False)}

--- YOUR TASK ---
Watch this time segment carefully and document what each character is wearing in THIS SCENE.

For each character present in this scene:
1. Use their EXACT character ID from the roster (e.g., @character_01)
2. Describe their outfit in DETAIL:
   - Tops (shirt, blouse, jacket, coat, etc.) - colors, materials, style
   - Bottoms (pants, skirt, etc.) - colors, style
   - Shoes (if visible) - color, type
   - Accessories (glasses, jewelry, hats, bags, etc.)
   - Overall style (professional, casual, elegant, etc.)

IMPORTANT CLOTHING DESCRIPTION RULES:
- Be SPECIFIC about colors (not just "light" - say "light blue", "cream", "pale grey")
- Describe materials if visible (cotton, silk, leather, wool)
- Note fit (fitted, loose, tailored)
- List ALL items the character is wearing
- Use CONSISTENT terminology throughout

**CRITICAL: CLOTHING DNA EXTRACTION**
For each clothing item, provide detailed specifications across 7 dimensions:

**1. COLOR SYSTEM** (MANDATORY - Use EXACT color identification):
- Primary Color: Pantone TCX code + HEX value + Common name
- Secondary Colors (if applicable): Same format
- Pattern Colors (if applicable): Same format
- Format: "Pantone 18-0303 TCX (#8B8C8E) - Warm Grey"
- NOTE: You MUST match the actual color from the video, considering the lighting conditions
- Choose the closest Pantone TCX (Fashion & Textile) color
- Provide HEX for digital accuracy

**2. FABRIC SYSTEM** (MANDATORY):
- Material: Type (cotton, silk, wool, leather, linen, synthetic blend)
- Weave: Plain, twill, satin, knit, woven
- Weight: g/m² estimate (light <150g, medium 150-250g, heavy >250g)
- Opacity: Opaque / Semi-transparent / Transparent
- Finish: Matte / Satin (shiny) / Metallic / Textured
- Stretch: None / 2-way / 4-way (estimate if visible)
- Texture: Smooth / Rough / Velvety / Grainy
- Drape: Structured / Flowing / Stiff

**3. CUT & FIT SYSTEM** (MANDATORY):
For Tops:
- Fit: Slim / Regular / Loose / Oversized
- Length: Crop / Waist / Hip / Thigh / Knee / Full
- Shoulder: Natural / Drop / Raglan / Set-in
- Sleeve: Sleeveless / Short / 3/4 / Long
- Collar: V-neck / Round / Collared / Lapel / None
- Closure: Buttons / Zipper / Pullover / Tie

For Bottoms:
- Fit: Skinny / Straight / Wide / Bootcut / Relaxed
- Length: Short / Capri / Ankle / Full
- Waist: High-rise / Mid-rise / Low-rise
- Hem: Straight / Tapered / Flared / Cuffed
- Pockets: Type and placement

**4. DETAILS SYSTEM** (MANDATORY):
- Buttons: Count, placement, color (Pantone+HEX), material, size
- Zipper: Placement, color, type (invisible/metal), length
- Pockets: Type (patch/slash/welt), placement, flap
- Stitching: Thread color, type (topstitch/overlock), spacing
- Any distinctive features: Pleats, cuffs, lining, etc.

**5. PATTERN SYSTEM** (if applicable):
- Type: Geometric / Floral / Abstract / Ethnic / Solid
- Size: Small / Medium / Large (estimate motif size)
- Arrangement: Regular / Irregular
- Direction: Horizontal / Vertical / Diagonal
- Density: Sparse / Medium / Dense
- Colors: List all pattern colors (Pantone+HEX)

**6. ACCESSORIES SYSTEM** (MANDATORY):
- Shoes: Type, color (Pantone+HEX), material, heel height
- Jewelry: Necklaces, earrings, bracelets, rings, watches (type, color, size)
- Bags: Type, color, size, material
- Any other visible accessories

**7. STYLING SYSTEM** (MANDATORY):
- Layering: Single / Double / Multiple
- Tuck: Tucked in / Half-tucked / Untucked
- Jacket state: Open / Buttoned (how many)
- Sleeve state: Rolled / Unrolled
- Overall style descriptor

--- OUTPUT FORMAT ---
Return a JSON object with detailed clothing DNA for each character:
{{
  "scene_id": "{scene_id}",
  "character_wardrobe": {{
    "@character_01": {{
      "top": {{
        "item_name": "Name of the top garment",
        "color": {{
          "primary": {{
            "pantone_tc": "Pantone XX-XXXX TCX",
            "hex": "#RRGGBB",
            "name": "Color Name",
            "description": "Brief color description"
          }},
          "secondary": {{
            "pantone_tc": "Pantone XX-XXXX TCX",
            "hex": "#RRGGBB",
            "name": "Color Name",
            "description": "Brief description"
          }}
        }},
        "fabric": {{
          "material": "Material type",
          "weave": "Weave type",
          "weight": "XXX g/m²",
          "opacity": "Opaque/Semi-transparent",
          "finish": "Matte/Satin/Metallic",
          "stretch": "None/2-way/4-way",
          "texture": "Texture description"
        }},
        "cut": {{
          "fit": "Slim/Regular/Loose",
          "length": "Length",
          "sleeve": "Sleeve type",
          "closure": "Closure type"
        }},
        "details": {{
          "buttons": "Description",
          "pockets": "Description"
        }},
        "pattern": {{
          "type": "Pattern type or 'Solid'",
          "colors": ["List of pattern colors"]
        }}
      }},
      "bottom": {{
        "item_name": "Name of bottom garment",
        "color": {{
          "primary": {{
            "pantone_tc": "Pantone XX-XXXX TCX",
            "hex": "#RRGGBB",
            "name": "Color Name"
          }}
        }},
        "fabric": {{
          "material": "Material",
          "weight": "XXX g/m²"
        }},
        "cut": {{
          "fit": "Fit type",
          "length": "Length",
          "waist": "Waist type"
        }}
      }},
      "shoes": {{
        "type": "Shoe type",
        "color": {{
          "pantone_tc": "Pantone XX-XXXX TCX",
          "hex": "#RRGGBB",
          "name": "Color Name"
        }},
        "material": "Material"
      }},
      "accessories": {{
        "jewelry": "Description",
        "other": "Other accessories"
      }},
      "styling": {{
        "layering": "Layering description",
        "overall": "Overall style"
      }},
      "full_description": "Complete natural language description integrating all DNA elements"
    }}
  }}
}}

**CRITICAL REQUIREMENTS**:
1. ALL color fields MUST include Pantone TCX + HEX
2. Be as detailed and specific as possible
3. Only describe what is VISIBLE in the video
4. If uncertain about a detail, note it as "likely" or "estimated"
5. The full_description should be a comprehensive natural language paragraph
6. Use EXACT character IDs from the roster

**CRITICAL OUTPUT RULES (MUST FOLLOW)**:
1. Output ONLY the JSON object above - nothing else
2. DO NOT include any explanations or text outside the JSON
3. Your response must start with '{{' and end with '}}'
4. NO markdown code blocks
5. Directly output the raw JSON only
"""

        try:
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=[myfile, prompt]
            )

            if not response.text:
                print(f"       ⚠️  No wardrobe data for {scene_id}")
                continue

            cleaned_json = extract_json_from_response(response.text)

            try:
                wardrobe = json.loads(cleaned_json)
                wardrobe_data["scene_wardrobes"][scene_id] = wardrobe
                print(f"       ✅ Wardrobe recorded for {len(wardrobe.get('character_wardrobe', {}))} characters")
            except json.JSONDecodeError as e:
                print(f"       ⚠️  Failed to parse wardrobe JSON for {scene_id}: {e}")

        except Exception as e:
            print(f"       ❌ Error building wardrobe for {scene_id}: {e}")

    return wardrobe_data


def parse_time_to_seconds(time_value):
    """
    Convert various time formats to seconds (float)

    Supported formats:
    - Number (int/float): Return directly
    - "00:00" format: minutes:seconds
    - "00:00:00" format: hours:minutes:seconds
    - "65.5" format: Numeric string
    """
    if isinstance(time_value, (int, float)):
        return float(time_value)

    if isinstance(time_value, str):
        # Check if contains colon (time format)
        if ':' in time_value:
            parts = time_value.split(':')
            if len(parts) == 2:
                # "00:00" format (minutes:seconds)
                minutes, seconds = parts
                return float(minutes) * 60 + float(seconds)
            elif len(parts) == 3:
                # "00:00:00" format (hours:minutes:seconds)
                hours, minutes, seconds = parts
                return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

        # Try to convert directly to number
        try:
            return float(time_value)
        except ValueError:
            pass

    raise ValueError(f"Unable to parse time format: {time_value}")


def get_major_scene_for_time(major_scenes, timestamp):
    """Get the major scene ID for a given timestamp"""
    for scene in major_scenes.get("major_scenes", []):
        # Use parse_time_to_seconds to handle various time formats
        start = parse_time_to_seconds(scene["start_time"])
        end = parse_time_to_seconds(scene["end_time"])

        if start <= timestamp <= end:
            return scene["scene_id"]
    return None


def get_character_display_name(character_id, character_roster):
    """
    Get the display name for a character

    If a real name was extracted, use the name; otherwise use character_id
    For example: @character_01 has name "Emma" -> Return "Emma (@character_01)"
    If no name -> Return "@character_01"
    """
    for char in character_roster.get("characters", []):
        if char.get("character_id") == character_id:
            char_name = char.get("character_name", "")
            name_source = char.get("name_source", "")

            # If there's a real name (not fallback), use the name
            if char_name and char_name != character_id and name_source != "id_fallback":
                return f"{char_name} ({character_id})"
            else:
                return character_id

    return character_id


def build_character_naming_guide(character_roster):
    """
    Build character naming guide

    Returns a string guiding how to reference characters in different situations
    """
    guide_lines = ["--- CHARACTER NAMING GUIDE (USE NAMES WHEN POSSIBLE) ---\n"]

    for char in character_roster.get("characters", []):
        char_id = char.get("character_id", "")
        char_name = char.get("character_name", "")
        name_source = char.get("name_source", "")
        evidence = char.get("name_evidence", "")

        if char_name and char_name != char_id and name_source != "id_fallback":
            guide_lines.append(f"{char_id}:")
            guide_lines.append(f"  Name: {char_name}")
            guide_lines.append(f"  Source: {name_source}")
            guide_lines.append(f"  Evidence: {evidence}")
            guide_lines.append(f"  ** PREFERRED: Use '{char_name}' in descriptions instead of '{char_id}' **\n")
        else:
            guide_lines.append(f"{char_id}:")
            guide_lines.append(f"  Name: Not mentioned in dialogue")
            guide_lines.append(f"  ** Use '{char_id}' (no name available) **\n")

    return "\n".join(guide_lines)


def main():
    args = parse_arguments()

    if "GENAI_API_KEY" not in os.environ:
        raise ValueError("Error: 'GENAI_API_KEY' environment variable is not set.")

    # Initialize client
    client = genai.Client(api_key=os.environ["GENAI_API_KEY"])

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video file not found: {args.video}")

    # ========== Stage 1: Detect video aspect ratio ==========
    video_aspect_ratio = detect_video_aspect_ratio(args.video)

    # Upload video to Gemini
    print(f"--> Uploading video to Gemini: {args.video}...")
    myfile = client.files.upload(file=args.video)

    while myfile.state == "PROCESSING":
        print("    Waiting for video processing...")
        time.sleep(5)
        myfile = client.files.get(name=myfile.name)

    if myfile.state == "FAILED":
        raise ValueError(f"Video processing failed: {myfile.name}")

    print(f"--> Video Ready.")

    # Load complete Whisper subtitles (original Chinese text)
    print(f"--> Loading Chinese transcript from: {args.transcript}")
    chinese_transcript = load_and_format_whisper(args.transcript)

    # Intelligently translate Chinese subtitles (handle omitted subjects)
    translated_transcript = smart_translate_transcript(client, myfile, chinese_transcript)
    full_whisper_text = translated_transcript  # Use translated version

    # Use scene detection instead of fixed time segmentation
    scenes = detect_scenes(args.video)

    # ========== Removed: No longer merge scenes ==========
    # Now directly use original segments, no longer perform shot merging and deduplication

    # ========== Phase 1: Global character identification ==========
    character_roster = identify_all_characters(client, myfile, full_whisper_text, 0)

    # ========== New phase: Extract all character forms of address ==========
    character_name_updates = extract_character_names(client, myfile, full_whisper_text, character_roster)

    # Merge forms of address extracted from dialogue into character profiles (add, don't replace)
    for char_info in character_roster.get("characters", []):
        char_id_key = char_info.get("character_id")  # @character_01

        # Initialize names object (if doesn't exist)
        if "names" not in char_info:
            char_info["names"] = {
                "primary_name": "",
                "primary_source": "id_fallback",
                "aliases": [],
                "aliases_sources": {},
                "titles": [],
                "titles_sources": {},
                "roles": [],
                "roles_sources": {},
                "familial": [],
                "familial_sources": {}
            }

        # If there are updates extracted from dialogue, merge them
        if char_id_key in character_name_updates:
            names_to_add = character_name_updates[char_id_key].get("names_to_add", {})
            evidence = character_name_updates[char_id_key].get("evidence", "")

            # Helper function: Safely add to list (avoid duplicates)
            def add_to_list(target_list, target_sources, new_items, new_sources, source_type):
                for item in (new_items or []):
                    if item and item not in target_list:
                        target_list.append(item)
                        if new_sources and item in new_sources:
                            target_sources[item] = source_type

            # Merge each category
            existing_names = char_info["names"]

            # primary_name: If not present, set from dialogue
            if names_to_add.get("primary_name") and not existing_names.get("primary_name"):
                existing_names["primary_name"] = names_to_add["primary_name"]
                existing_names["primary_source"] = "dialogue"

            # aliases: Add new aliases
            add_to_list(
                existing_names["aliases"],
                existing_names["aliases_sources"],
                names_to_add.get("aliases"),
                names_to_add.get("aliases_sources"),
                "dialogue"
            )

            # titles: Add new titles
            add_to_list(
                existing_names["titles"],
                existing_names["titles_sources"],
                names_to_add.get("titles"),
                names_to_add.get("titles_sources"),
                "dialogue"
            )

            # roles: Add new roles
            add_to_list(
                existing_names["roles"],
                existing_names["roles_sources"],
                names_to_add.get("roles"),
                names_to_add.get("roles_sources"),
                "dialogue"
            )

            # familial: Add new familial terms
            add_to_list(
                existing_names["familial"],
                existing_names["familial_sources"],
                names_to_add.get("familial"),
                names_to_add.get("familial_sources"),
                "dialogue"
            )

            # Record evidence
            char_info["name_evidence"] = evidence
            print(f"       ✅ {char_id_key}: Merged dialogue names")
            print(f"          Total aliases: {len(existing_names['aliases'])}")
            print(f"          Total titles: {len(existing_names['titles'])}")
            print(f"          Total familial: {len(existing_names['familial'])}")
        else:
            # No new names extracted from dialogue
            if not char_info["names"].get("primary_name"):
                char_info["names"]["primary_name"] = char_id_key
                char_info["names"]["primary_source"] = "id_fallback"
            print(f"       ℹ️  {char_id_key}: No new names from dialogue")

    print(f"\n✅ Character names integrated into roster (all aliases, titles, and familial terms merged)")

    # ========== Get total video duration ==========
    video_duration = get_video_duration(args.video)

    # ========== New phase: Main/supporting character classification ==========
    print(f"\n{'='*60}")
    print(f"--> Character Role Classification")
    print(f"{'='*60}")
    character_roster = classify_character_roles(client, myfile, character_roster, video_duration)

    # ========== Phase 2: Major scene detection ==========
    major_scenes = detect_major_scenes(client, myfile, 0)

    # ========== Phase 3: Build scene wardrobe profiles ==========
    scene_wardrobe = build_scene_wardrobe(client, myfile, major_scenes, character_roster)

    # Convert character list to string for easy use in subsequent prompts
    character_info_str = json.dumps(character_roster, indent=2, ensure_ascii=False)
    wardrobe_info_str = json.dumps(scene_wardrobe, indent=2, ensure_ascii=False)

    # Build character naming guide
    character_naming_guide = build_character_naming_guide(character_roster)

    print(f"\n--> Processing scenes with character and wardrobe consistency...")

    segments = []



    for scene_info in scenes:
        scene_index = scene_info["scene_index"]
        start_time = scene_info["start_time"]
        end_time = scene_info["end_time"]
        duration = scene_info["duration"]

        # Find the major scene this belongs to
        major_scene_id = get_major_scene_for_time(major_scenes, start_time)

        print(f"\n{'='*60}")
        print(f"--> Processing Shot {scene_index}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s)")
        print(f"    Major Scene: {major_scene_id}")
        print(f"{'='*60}")

        # Filter dialogue for current time segment
        segment_transcript = filter_transcript_by_time(full_whisper_text, start_time, end_time)
        print(f"    Transcript lines in this scene: {len(segment_transcript.split(chr(10)))}")

        # Build prompt for current scene
        segment_type = "SCENE (a continuous shot)"
        merged_instruction = ""

        prompt = f"""
You will analyze a video {segment_type} from {start_time:.2f}s to {end_time:.2f}s (duration: {duration:.2f}s).{merged_instruction}
Your task is to perform a deep cinematic analysis of THIS TIME SEGMENT.

--- TIME INFORMATION ---
Scene Start: {start_time:.2f}s
Scene End: {end_time:.2f}s
Scene Duration: {duration:.2f}s

--- MAJOR SCENE CONTEXT ---
This shot belongs to: {major_scene_id}

{character_naming_guide}

--- SCENE WARDROBE (CRITICAL: USE EXACTLY THESE DESCRIPTIONS!) ---
The following is the EXACT clothing for characters in this major scene:
{wardrobe_info_str}

IMPORTANT RULES FOR CLOTHING DESCRIPTIONS:
- You MUST use the EXACT clothing descriptions from the wardrobe above
- Do NOT modify, redescribe, or change the clothing details
- Copy the full_description word-for-word when referring to character clothing
- This ensures consistency across all shots in this major scene

--- GLOBAL CHARACTER ROSTER (IMPORTANT: USE THESE IDs!) ---
The following characters have been identified in the video. You MUST use these EXACT IDs when referring to them:
{character_info_str}

CRITICAL RULES FOR CHARACTER IDENTIFICATION:
1. **MATCH EXISTING IDS FIRST**: Before assigning any new ID, you MUST check if the person in this scene matches someone from the roster above
2. **USE ALL 7 CRITERIA**: Match characters using the STRICT criteria below:
   - Physical Attributes (gender, age, body type, skin tone) - PRIMARY
   - Hair (color, length, style) - PRIMARY
   - Face (facial hair, glasses, distinctive marks) - PRIMARY
   - Clothing (colors, patterns, style, layers) - SECONDARY (may change!)
   - Accessories (hats, jewelry, bags, glasses) - SECONDARY
   - Current Actions (what they're doing) - CONTEXT CLUE
   - Distinctive Features (unique traits) - KEY IDENTIFIER
3. **CLOTHING CHANGES ARE OK**: A person changing clothes is STILL the same person - look at face/hair/body
4. **CONSISTENCY IS MANDATORY**: If the same person appears in multiple scenes (even with different clothes), they MUST have the SAME ID
5. **CHECK CAREFULLY**: Use PERMANENT features (face, hair, body) as primary identifiers

--- AUDIO TRANSCRIPT (FROM WHISPER) ---
The following is the speech transcript for THIS SCENE, with timestamps:
{segment_transcript}

IMPORTANT: Use this transcript to understand what the characters are saying in this scene.
Include dialogue in your descriptions with timestamp annotations like:
- "Emma speaks: [1.2s] their dialogue here" (if name available)
- "@character_01 speaks: [1.2s] their dialogue here" (if no name)

--- CHARACTER IDENTIFICATION LOGIC ---
To correctly identify the characters in this scene, you must cross-reference the visual data with the GLOBAL CHARACTER ROSTER using the following STRICT criteria:
1. **Physical Attributes**: Gender, approximate age, body type, skin tone
2. **Hair**: Color, length, style
3. **Face**: Facial hair, glasses, distinctive marks
4. **Clothing**: Colors, patterns, style, layers (may vary - check other scenes!)
5. **Accessories**: Hats, jewelry, bags
6. **Current Actions**: What they are doing in this specific scene
7. **Distinctive Features**: Any unique trait that separates them from others

MATCHING PROCESS:
1. **FIRST**: Check the GLOBAL CHARACTER ROSTER above
2. **COMPARE**: Use the 7 criteria to match people in this scene with the roster
3. **PRIORITY**: Trust face/hair/body MORE than clothing (clothing can change!)
4. **USE EXISTING IDS**: If you find a match, use that character's ID or NAME
5. **NEW IDS ONLY**: If absolutely certain the person is NOT in the roster, assign a new ID (rare!)

--- NAMING CONVENTION INSTRUCTIONS (STRICT) ---
1. **PREFER CHARACTER NAMES**: When a character has a name extracted from dialogue, USE THE NAME in all descriptions
   - For example: Use "Emma" instead of "@character_01" when Emma's name is known
2. **FALLBACK TO IDs**: If a character has NO name, use their ID (e.g., "@character_01")
3. **SCOPE**: This rule applies to ALL output fields, specifically including "subject_movement" and "I2V Prompt".
4. DO NOT use generic nouns or create new IDs unless absolutely necessary.
   - WRONG: "The boy moves left." or "@character_99" (when it's actually @character_01)
   - CORRECT: "Emma moves left." (if name known)
   - CORRECT: "@character_01 moves left." (if no name available)

--- TASK PART 1: TECHNICAL CINEMATIC EXTRACTION ---
Analyze THIS TIME SEGMENT to extract specific cinematic parameters.
Provide concise, professional descriptions for each category:

1. **Lighting & Color**:
   - lighting_setup: Source type, direction, hardness (e.g., "Harsh sunlight", "Rim light", "Soft window light").
   - color_grading: Color tendency, contrast, LUT style (e.g., "Cool blues", "Teal and Orange", "Desaturated").

2. **Composition & Atmosphere**:
   - composition: Arrangement of elements (e.g., "Rule of thirds", "Center symmetry", "Leading lines").
   - mood_atmosphere: Abstract feeling and psychological suggestion (e.g., "Tense", "Epic", "Melancholic").

3. **Camera Geometry (3D Space)**:
   - shot_size: Subject size in frame (e.g., "Wide Shot", "Medium Close-up").
   - camera_angle: Vertical angle (e.g., "Low Angle", "High Angle", "Eye-level").
   - camera_height: Physical height from ground (e.g., "Waist-Level", "Ground-Level").
   - horizontal_angle: Angle relative to subject (e.g., "Frontal", "Three-Quarter", "Profile").

4. **Technical Specs (Optical Texture)**:
   - focal_length: Perspective feel (e.g., "80mm telephoto", "24mm wide").
   - depth_of_field: Background blur (e.g., "f/1.8 Shallow focus", "f/8 Deep focus").
   - tech_device: Camera/Lens metadata for texture (e.g., "IMAX MSM 9802", "Kodak Vision3 500T", "Anamorphic lens").

5. **Motion Dynamics**:
   - camera_movement: How the camera moves in this segment (e.g., "Static", "Tracking shot", "Handheld shake").
   - subject_movement: How the specific characters or environment move in THIS TIME SEGMENT.
     **MUST use strict IDs** and reference their speech/actions from the transcript.
     Include dialogue with timestamps like: "@character_01 turns head and speaks: [2.5s] 你好世界"

--- TASK PART 2: NARRATIVE I2V PROMPT ---
Write ONE single, highly detailed cinematic paragraph describing THIS TIME SEGMENT.
- It must integrate all the technical details extracted above into a cohesive narrative flow.
- Cover setting, environment, lighting, detailed character actions, and spatial dynamics.
- **CRITICAL**: Include relevant dialogue/spoken content from the transcript.
- **IMPORTANT**: Do NOT include timestamp annotations (like [2.5s]) in the I2V Prompt - just use the dialogue naturally.
- **CRITICAL**: Use the strict IDs (@character_01, @character_02) for all character references.
- Match what characters are saying (from transcript) with their visual appearance and actions in THIS SEGMENT.
- NO bullet points. One continuous prose block.

**CRITICAL COMPOSITION RULES**:
- **SINGLE SHOT ONLY**: Describe ONE continuous shot/frame. NEVER describe multiple shots, split screens, or composite layouts.
- **NO SPLIT COMPOSITIONS**: Do NOT describe split-screen, diptych, triptych, grid layouts, or multiple images arranged together (horizontally or vertically).
- **NO TEXT/GRAPHICS**: Do NOT include text overlays, subtitles, titles, captions, watermarks, or any visible text/graphics in the scene description.
- **NO SEQUENTIAL LAYOUTS**: Describe only ONE moment in time, not multiple moments shown side-by-side or stacked.
- **PURE CINEMATIC SCENE**: Focus purely on the cinematic scene itself - characters, environment, lighting, camera work, and action.

--- TASK PART 3: LANGUAGE TO ONE SHOT REFERENCE PROMPT ---
Create a STATIC KEYFRAME reference description based on the I2V Prompt above.

**PURPOSE**:
Extract and preserve the VISUAL elements from the I2V Prompt to create a static keyframe reference.
This keyframe will serve as the MASTER REFERENCE for maintaining visual consistency across all shots in {major_scene_id}.

**WHAT TO KEEP (STATIC ELEMENTS)**:
- Environment/setting description (room, location, background)
- Lighting setup (natural/artificial, color temperature, direction)
- Camera geometry (shot size, angle, height, framing)
- Character positions and poses (static, not moving)
- Facial expressions (at the chosen moment)
- Clothing and appearance details
- Composition and framing
- Mood and atmosphere
- Color grading and visual style

**WHAT TO REMOVE (DYNAMIC/ELEMENTS)**:
- ❌ Dialogue/quotes (remove all spoken words in quotes)
- ❌ Action verbs (walks, runs, turns, moves, gestures, etc.)
- ❌ Movement descriptions (across, toward, away, etc.)
- ❌ Time sequence indicators (then, next, suddenly, etc.)
- ❌ Transitional phrases
- ❌ Timestamps

**HOW TO TRANSFORM I2V PROMPT TO STATIC KEYFRAME**:

Example Transformation:
I2V Prompt (dynamic): "Emma walks across the room towards the window, turns her head, and asks 'What do you think?' with a curious expression. The soft daylight illuminates her face as she speaks."

Keyframe (static): "Emma stands in the middle of the room, her body oriented towards a large window on the back wall. She wears a light blue business suit and has a curious, inquiring expression on her face. Her blonde hair catches the soft daylight streaming through the window. The shot is a medium close-up from eye-level, capturing her against a backdrop of floor-to-ceiling windows with a city view. The lighting is natural and diffused, creating gentle shadows on her face."

**INSTRUCTIONS FOR CREATING THE KEYFRAME**:

1. **Choose ONE moment**: Select the most representative/memorable moment from the I2V Prompt
2. **Freeze the action**: Describe that moment as if time has stopped
3. **Remove dialogue**: Delete all quoted speech
4. **Static poses**: Replace active poses with static poses (e.g., "walking to" → "standing at")
5. **Keep visuals**: Preserve all visual descriptions (environment, lighting, clothing, expressions)
6. **Technical details**: Include camera specs (shot size, angle, lens, depth of field)

**OUTPUT FORMAT**:
Write a single, comprehensive paragraph describing the STATIC KEYFRAME.
- Use present tense
- Focus on visual elements
- No action verbs
- No dialogue
- No movement
- Just pure visual description of a frozen moment

This keyframe description will be used to generate a reference image that defines the visual foundation for all shots in this major scene.

--- OUTPUT FORMAT ---
Your response MUST follow this exact JSON format:

{{
  "lighting_setup": "String description...",
  "color_grading": "String description...",
  "composition": "String description...",
  "mood_atmosphere": "String description...",
  "shot_size": "String description...",
  "camera_angle": "String description...",
  "camera_height": "String description...",
  "horizontal_angle": "String description...",
  "focal_length": "String description...",
  "depth_of_field": "String description...",
  "tech_device": "String description...",
  "camera_movement": "String description...",
  "subject_movement": "String description using IDs with dialogue timestamps (e.g., '@character_01 walks: [2.5s] Hello there')",
  "I2V Prompt": "A single, long, deeply detailed cinematic paragraph describing THIS TIME SEGMENT ONLY. Include setting, environment, lighting, atmosphere, character actions with dialogue (NO timestamps in I2V Prompt), interactions, camera movement, visual aesthetics, and spatial dynamics, using ONLY strict character IDs (e.g., @character_01 speaks: their dialogue here).",
  "Language_to_One_Shot_Prompt": "A comprehensive STATIC KEYFRAME reference description extracted from the I2V Prompt. This preserves visual elements (environment, lighting, camera geometry, character poses, facial expressions, clothing, composition) while removing dynamic elements (dialogue, action verbs, movement, time sequences). The keyframe represents ONE frozen moment that defines the visual foundation for maintaining consistency across the major scene. Format: Single paragraph with present tense, no action verbs, no dialogue, pure visual description of a static moment."
}}

**CRITICAL OUTPUT RULES (MUST FOLLOW)**:
1. Output ONLY the JSON object above - nothing else
2. DO NOT include any explanations, commentary, or text outside the JSON
3. Your response must start immediately with '{{' (opening brace)
4. Your response must end with '}}' (closing brace)
5. NO markdown code blocks (do NOT use ```json or ```)
6. NO introductory text like "Here is the analysis:" or "Based on the video:"
7. NO concluding text or summaries
8. Directly output the raw JSON object only

**EXAMPLE OF CORRECT OUTPUT**:
{{"lighting_setup": "Soft natural light...", "color_grading": "Warm tones...", ...}}

**EXAMPLE OF INCORRECT OUTPUT**:
Here is my analysis:
{{"lighting_setup": "..."}}

INCORRECT - Do NOT do this!
"""

        # Call model to analyze current scene
        print(f"    Sending request to Gemini for scene {scene_index}...")
        try:
            response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=[myfile, prompt]
            )

            if not response.text:
                print(f"    ⚠️  No text returned for scene {scene_index}")
                continue

            # Extract and parse JSON
            cleaned_json_str = extract_json_from_response(response.text)

            try:
                scene_data = json.loads(cleaned_json_str)
                # Add scene information
                scene_data["scene_index"] = scene_index
                scene_data["time_range"] = f"{start_time:.2f}s - {end_time:.2f}s"
                scene_data["start_time"] = start_time
                scene_data["end_time"] = end_time
                scene_data["duration"] = duration

                segments.append(scene_data)
                print(f"    ✅ Scene {scene_index} completed")
            except json.JSONDecodeError as e:
                print(f"    ⚠️  Scene {scene_index} invalid JSON: {e}")
                # Save original text
                segments.append({
                    "scene_index": scene_index,
                    "time_range": f"{start_time:.2f}s - {end_time:.2f}s",
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "error": "Invalid JSON",
                    "raw_response": cleaned_json_str
                })

        except Exception as e:
            print(f"    ❌ Error processing scene {scene_index}: {e}")
            segments.append({
                "scene_index": scene_index,
                "time_range": f"{start_time:.2f}s - {end_time:.2f}s",
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "error": str(e)
            })

        # Avoid API rate limiting, wait briefly
        time.sleep(1)

    # Delete cloud video file
    print(f"\n--> Deleting video file from Gemini storage: {myfile.name}")
    try:
        client.files.delete(name=myfile.name)
        print("    ✅ Cloud file deleted successfully")
    except Exception as delete_error:
        print(f"    ⚠️  Failed to delete cloud file: {delete_error}")

    # Build final output
    final_output = {
        "video_file": args.video,
        "video_metadata": {
            "aspect_ratio": video_aspect_ratio
        },
        "total_scenes": len(segments),
        "character_roster": character_roster,  # Global character profiles
        "major_scenes": major_scenes,            # Major scene information
        "scene_wardrobe": scene_wardrobe,        # Scene wardrobe profiles
        "scenes": segments
    }

    # Save to file
    with open(args.output, 'w', encoding='utf-8') as outfile:
        json.dump(final_output, outfile, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅ All done! Processed {len(segments)} scenes")
    print(f"--> Output saved to: {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()