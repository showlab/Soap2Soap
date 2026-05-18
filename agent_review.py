#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Review Agent - Intelligently fix failed generation contexts

Features:
1. Analyze reasons for Gemini generation failures
2. Identify content that may trigger safety policies
3. Intelligently modify descriptions in memory_allocation
4. Save fixed memory allocations

Usage:
    python agent_review.py --shot 4 --error "TypeError: 'NoneType' object is not iterable"
    python agent_review.py --shot 4 --error-file error_log.txt
    python agent_review.py --auto-fix  # Automatically fix all failed shots
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from google import genai
except ImportError:
    print("Error: google-genai library needs to be installed")
    print("Please run: pip install google-genai")
    sys.exit(1)


class ReviewAgent:
    """Review Agent - Intelligently fix failed generation contexts"""

    def __init__(self, memory_file="memory_allocation.json", style="realistic"):
        """
        Initialize

        Args:
            memory_file: Memory allocation file path
            style: Generation style
        """
        self.memory_file = memory_file
        self.style = style

        # Initialize Gemini client
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: Environment variable GENAI_API_KEY not found")

        self.client = genai.Client(api_key=api_key)

        # Backup directory
        self.backup_dir = "memory_backups"
        os.makedirs(self.backup_dir, exist_ok=True)

    def load_memory_allocation(self):
        """Load memory allocation file"""
        if not os.path.exists(self.memory_file):
            raise FileNotFoundError(f"Memory allocation file does not exist: {self.memory_file}")

        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.memory_store = data.get('memory_store', {})
        return self.memory_store

    def save_memory_allocation(self):
        """Save memory allocation file"""
        # Backup first
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f"memory_allocation_{timestamp}.json")

        # Backup current version
        if os.path.exists(self.memory_file):
            import shutil
            shutil.copy2(self.memory_file, backup_file)
            print(f"  💾 Backed up to: {backup_file}")

        # Save new version
        output_data = {
            "metadata": {
                "style": self.style,
                "fixed_at": datetime.now().isoformat(),
                "total_shots": len(self.memory_store)
            },
            "memory_store": self.memory_store
        }

        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved: {self.memory_file}")

    def backup_current_memory(self):
        """Backup current memory file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f"memory_allocation_before_fix_{timestamp}.json")

        if os.path.exists(self.memory_file):
            import shutil
            shutil.copy2(self.memory_file, backup_file)
            print(f"  💾 Backup before fix: {backup_file}")
            return backup_file
        return None

    def analyze_failure_with_gemini(self, shot_id, error_message, memory_package):
        """
        Use Gemini to analyze failure reasons and generate fix suggestions

        Args:
            shot_id: Shot ID
            error_message: Error message
            memory_package: Memory package

        Returns:
            Fixed narrative section
        """
        print(f"\n{'='*70}")
        print(f"🔍 Using Gemini to analyze Shot {shot_id} generation failure")
        print(f"{'='*70}")

        # Build analysis prompt
        prompt = f"""
You are an expert AI content safety analyzer specialized in debugging image generation failures.

**TASK**:
Analyze why the following shot generation FAILED and provide a SAFE, corrected version.

**ERROR INFORMATION**:
Error Type: {error_message}
Common Causes:
- Safety policy violation (violence, gore, inappropriate content)
- Content moderation flag
- Policy violation triggers
- Reference image issues

**ORIGINAL SHOT CONTEXT**:
```json
{json.dumps(memory_package, indent=2, ensure_ascii=False)}
```

**ANALYSIS CHECKLIST**:
1. Identify potentially problematic content in the description:
   - Violent or gory imagery (blood, wounds, graphic violence)
   - Inappropriate or explicit content
   - Harmful activities
   - Disturbing visual descriptions
   - Overly intense dramatic language

2. Identify specific text that may trigger safety filters:
   - Words like: "destroyed", "wreckage", "blood", "gore", "violence", "kill", "death"
   - Descriptions of injuries or wounds
   - Post-apocalyptic or disaster imagery
   - War or battle terminology

3. Identify reference image issues:
   - Reference images that may be inappropriate
   - Character poses that could be misinterpreted
   - Clothing or appearance issues

**FIXING STRATEGY**:
1. Preserve the CORE visual elements (characters, setting, mood, camera)
2. SOFTEN problematic language
3. Replace triggering words with safer alternatives
4. Maintain the artistic intent while ensuring safety compliance
5. Keep technical camera parameters unchanged

**OUTPUT FORMAT**:
Return ONLY a valid JSON object with the corrected narrative section:

```json
{{
  "analysis": {{
    "identified_issues": ["List of specific issues found"],
    "triggering_content": ["Specific words/phrases that caused the failure"],
    "safety_risk_level": "LOW/MEDIUM/HIGH"
  }},
  "corrected_narrative": {{
    "action": "Corrected action description (safer version)",
    "i2v_prompt": "Corrected I2V prompt (safer version)",
    "language_prompt": "Corrected language prompt (safer version)"
  }},
  "changes_made": ["List of specific changes applied"]
}}
```

**CRITICAL RULES**:
1. Keep character identities and visual appearance
2. Keep the scene setting and atmosphere
3. Keep camera angles and technical parameters
4. ONLY modify the problematic descriptive language
5. Make minimal changes necessary to pass safety filters
6. DO NOT change the fundamental scene composition

**EXAMPLE TRANSFORMATION**:

BEFORE (problematic):
"blood-soaked battlefield", "gory wounds", "violent destruction", "brutal death"

AFTER (safe but dramatic):
"battle-scarred terrain", "injuries visible", "devastated landscape", "fallen warriors"

**IMPORTANT**:
- Focus on making content GENERATION-SAFE
- Preserve dramatic tension without graphic details
- Use cinematic language rather than graphic descriptions
- Ensure the corrected version maintains the scene's impact
"""

        try:
            print(f"  📤 Sending analysis request to Gemini...")

            response = self.client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=prompt
            )

            if not response.text:
                print(f"  ⚠️  Gemini did not return a response")
                return None

            # Extract JSON
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Try to extract first JSON object directly
                first_brace = response.text.find('{')
                last_brace = response.text.rfind('}')
                if first_brace != -1 and last_brace != -1:
                    response_text = response.text[first_brace:last_brace + 1]
                else:
                    response_text = response.text

            result = json.loads(response_text)

            # Print analysis results
            analysis = result.get("analysis", {})
            print(f"\n  📋 Analysis Results:")
            print(f"     Issues found: {len(analysis.get('identified_issues', []))}")
            for issue in analysis.get('identified_issues', []):
                print(f"       - {issue}")
            print(f"     Triggering content: {', '.join(analysis.get('triggering_content', []))}")
            print(f"     Risk level: {analysis.get('safety_risk_level', 'UNKNOWN')}")

            changes = result.get('changes_made', [])
            print(f"\n  ✏️  Applied changes ({len(changes)} items):")
            for change in changes[:5]:  # Only show first 5 items
                print(f"       - {change}")

            return result.get("corrected_narrative")

        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse Gemini response: {e}")
            print(f"  📋 Original response (first 500 chars):")
            print(f"     {response.text[:500]}")
            return None
        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            return None

    def fix_shot(self, shot_id, error_message):
        """
        Fix a single shot

        Args:
            shot_id: Shot ID
            error_message: Error message

        Returns:
            True on success, False on failure
        """
        print(f"\n{'='*70}")
        print(f"🔧 Fixing Shot {shot_id}")
        print(f"{'='*70}")
        print(f"Error message: {error_message}")

        # Ensure shot_id is string type
        shot_id_str = str(shot_id)

        # Load memory allocation
        self.load_memory_allocation()

        # Find shot (supports multiple types)
        shot_key = None
        for key in self.memory_store.keys():
            if str(key) == shot_id_str:
                shot_key = key
                break

        if shot_key is None:
            print(f"  ❌ Shot {shot_id} does not exist in memory allocation")
            return False

        memory_package = self.memory_store[shot_key]

        print(f"  ✅ Found Shot {shot_id}")
        print(f"     Scene: {memory_package.get('major_scene', 'N/A')}")
        print(f"     Characters: {', '.join(memory_package.get('characters', []))}")

        # Use Gemini to analyze and generate fix
        corrected_narrative = self.analyze_failure_with_gemini(
            shot_id,
            error_message,
            memory_package
        )

        if not corrected_narrative:
            print(f"  ⚠️  Failed to generate fix, attempting manual fix...")

            # Manual fix: remove common trigger words
            corrected_narrative = self.manual_fix_safety_issues(memory_package.get("narrative", {}))

        # Apply fix
        print(f"\n  ✏️  Applying fix...")
        self.memory_store[shot_key]["narrative"] = corrected_narrative

        # Save fixed memory allocation
        self.save_memory_allocation()

        print(f"\n  ✅ Shot {shot_id} fix completed")
        return True

    def manual_fix_safety_issues(self, narrative):
        """
        Manually fix common safety issues (fallback method)

        Args:
            narrative: Original narrative

        Returns:
            Fixed narrative
        """
        # Problematic word mapping (safer alternatives)
        replacements = {
            # Violence related
            'blood': 'energy',
            'bloody': 'intense',
            'gore': 'dramatic',
            'wound': 'mark',
            'injury': 'impact',
            'violence': 'action',
            'violent': 'intense',
            'kill': 'defeat',
            'death': 'fall',
            'dead': 'fallen',
            'dying': 'fading',
            'slaughter': 'battle',
            'massacre': 'conflict',
            'murder': 'conflict',

            # Disaster related
            'destroyed': 'damaged',
            'destruction': 'devastation',
            'catastrophe': 'event',
            'disaster': 'incident',
            'apocalyptic': 'dramatic',
            'ruined': 'aged',
            'wreckage': 'debris',

            # War related
            'war': 'conflict',
            'battlefield': 'terrain',
            'warzone': 'area',
            'combat': 'action',
            'attack': 'engage',

            # Body parts (too specific)
            'severed': 'damaged',
            'dismembered': 'injured',
            'decapitated': 'fallen',
        }

        def safe_replace(text):
            """Safely replace text"""
            if not isinstance(text, str):
                return text

            result = text
            for problematic, safe in replacements.items():
                # Case-insensitive replacement
                import re
                result = re.sub(rf'\b{problematic}\b', safe, result, flags=re.IGNORECASE)

            return result

        # Fix all fields
        corrected = {}
        for key, value in narrative.items():
            if isinstance(value, str):
                corrected[key] = safe_replace(value)
            elif isinstance(value, dict):
                corrected[key] = {k: safe_replace(v) for k, v in value.items()}
            elif isinstance(value, list):
                corrected[key] = [safe_replace(v) for v in value]
            else:
                corrected[key] = value

        # If no changes made, at least soften prompt
        if corrected == narrative:
            print(f"     ⚠️  No trigger words detected, adding safety notice...")
            if 'language_prompt' in corrected:
                corrected['language_prompt'] = corrected['language_prompt'] + " The scene is stylized and dramatic, suitable for general audiences."
            if 'i2v_prompt' in corrected:
                corrected['i2v_prompt'] = corrected['i2v_prompt'] + " Cinematic and stylized representation."

        return corrected

    def analyze_and_fix_batch(self, failed_shots_info):
        """
        Batch fix failed shots

        Args:
            failed_shots_info: List of failed shots [{"shot_id": 4, "error": "..."}, ...]

        Returns:
            Number of successfully fixed shots
        """
        success_count = 0

        for shot_info in failed_shots_info:
            shot_id = shot_info.get("shot_id")
            error_msg = shot_info.get("error", "Unknown error")

            if self.fix_shot(shot_id, error_msg):
                success_count += 1

        return success_count

    def detect_risky_content(self, memory_package):
        """
        Detect potentially risky content in memory package

        Args:
            memory_package: Memory package

        Returns:
            Risk report dictionary
        """
        risk_keywords = {
            'HIGH_RISK': [
                'blood', 'gore', 'wound', 'injury', 'violence', 'kill', 'death',
                'murder', 'torture', 'dismember', 'decapitat', 'severed', 'impale',
                'mutilat', 'cannibal', 'genocide', 'terrorism', 'suicide'
            ],
            'MEDIUM_RISK': [
                'destroyed', 'destruction', 'catastrophe', 'disaster', 'apocalyptic',
                'warzone', 'battlefield', 'massacre', 'slaughter', 'execution'
            ],
            'LOW_RISK': [
                'attack', 'fight', 'battle', 'combat', 'struggle', 'conflict',
                'explosion', 'collapse', 'ruined', 'damaged'
            ]
        }

        narrative = memory_package.get('narrative', {})
        narrative_text = json.dumps(narrative, ensure_ascii=False).lower()

        detected_risks = {
            'HIGH_RISK': [],
            'MEDIUM_RISK': [],
            'LOW_RISK': []
        }

        for risk_level, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in narrative_text:
                    detected_risks[risk_level].append(keyword)

        # Generate risk report
        total_risks = sum(len(risks) for risks in detected_risks.values())

        if total_risks == 0:
            return {
                'has_risks': False,
                'risk_level': 'NONE',
                'detected_keywords': []
            }

        # Determine overall risk level
        if detected_risks['HIGH_RISK']:
            overall_risk = 'HIGH'
        elif detected_risks['MEDIUM_RISK']:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'

        return {
            'has_risks': True,
            'risk_level': overall_risk,
            'detected_keywords': detected_risks
        }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Review Agent - Intelligently fix failed generation contexts',
        epilog='''
Usage examples:
  python %(prog)s --shot 4 --error "TypeError: NoneType"
  python %(prog)s --shot 4 --error-file error.log
  python %(prog)s --auto-fix --failed-shots failed_shots.json
  python %(prog)s --check-risk --shot 4
        '''
    )

    parser.add_argument('--shot', type=int, help='Shot ID to fix')
    parser.add_argument('--error', help='Error message string')
    parser.add_argument('--error-file', help='Error message file path')
    parser.add_argument('--memory', default='memory_allocation.json', help='Memory allocation file')
    parser.add_argument('--style', default='realistic',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       help='Generation style')

    # Batch fix mode
    parser.add_argument('--auto-fix', action='store_true', help='Auto fix mode')
    parser.add_argument('--failed-shots', help='JSON file of failed shots')

    # Risk detection mode
    parser.add_argument('--check-risk', action='store_true', help='Risk detection mode')

    args = parser.parse_args()

    try:
        # Initialize ReviewAgent
        agent = ReviewAgent(
            memory_file=args.memory,
            style=args.style
        )

        # Risk detection mode
        if args.check_risk:
            if not args.shot:
                print("❌ Error: --check-risk requires --shot parameter")
                return 1

            agent.load_memory_allocation()
            shot_id_str = str(args.shot)

            # Find shot
            shot_key = None
            for key in agent.memory_store.keys():
                if str(key) == shot_id_str:
                    shot_key = key
                    break

            if shot_key is None:
                print(f"❌ Shot {args.shot} does not exist")
                return 1

            memory_package = agent.memory_store[shot_key]
            risk_report = agent.detect_risky_content(memory_package)

            print(f"\n{'='*70}")
            print(f"📊 Shot {args.shot} Risk Assessment Report")
            print(f"{'='*70}")
            print(f"Risk level: {risk_report['risk_level']}")
            print(f"Risk words detected: {sum(len(v) for v in risk_report['detected_keywords'].values())}")

            for level, keywords in risk_report['detected_keywords'].items():
                if keywords:
                    print(f"\n{level}: {', '.join(keywords)}")

            return 0

        # Auto fix mode
        if args.auto_fix:
            if not args.failed_shots:
                print("❌ Error: --auto-fix requires --failed-shots parameter")
                return 1

            with open(args.failed_shots, 'r', encoding='utf-8') as f:
                failed_shots = json.load(f)

            print(f"\n{'='*70}")
            print(f"🔧 Batch Fix Mode")
            print(f"{'='*70}")
            print(f"Shots to fix: {len(failed_shots)}")

            success_count = agent.analyze_and_fix_batch(failed_shots)

            print(f"\n{'='*70}")
            print(f"✅ Batch Fix Completed")
            print(f"{'='*70}")
            print(f"Success: {success_count}/{len(failed_shots)}")

            return 0 if success_count == len(failed_shots) else 1

        # Single shot fix mode
        if args.shot and (args.error or args.error_file):
            # Get error message
            if args.error_file:
                with open(args.error_file, 'r', encoding='utf-8') as f:
                    error_message = f.read().strip()
            else:
                error_message = args.error

            # Execute fix
            success = agent.fix_shot(args.shot, error_message)

            if success:
                print(f"\n{'='*70}")
                print(f"✅ Fix successful!")
                print(f"{'='*70}")
                print(f"💡 Tip: You can re-run the generation command:")
                print(f"   python agent_generation.py {args.memory} --shot {args.shot}")
                print(f"{'='*70}")
                return 0
            else:
                print(f"\n❌ Fix failed")
                return 1

        # No required parameters provided
        print("❌ Error: Please provide fix parameters")
        print("   Single fix: --shot <id> --error <message>")
        print("   Batch fix: --auto-fix --failed-shots <file>")
        print("   Risk check: --check-risk --shot <id>")
        print("\nUse --help for detailed help")
        return 1

    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
