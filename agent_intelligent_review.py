#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Review Agent

Features:
1. Analyze detailed reasons for Gemini generation failures
2. Utilize Gemini 3 to intelligently fix context descriptions
3. Automatically verify the safety of fix solutions
4. Save fix history records

Usage:
    from agent_intelligent_review import IntelligentReviewAgent

    review_agent = IntelligentReviewAgent(memory_file="memory_allocation.json")
    fix_solution = review_agent.diagnose_and_fix(shot_id, error_info)
"""

import os
import sys
import json
import shutil
import re
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


class IntelligentReviewAgent:
    """Intelligent Review Agent - Automatically diagnose and fix generation failures"""

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

        # Fix history records
        self.fix_history_file = "fix_history.json"

    def diagnose_and_fix(self, shot_id, error_info):
        """
        Diagnose and fix a single shot

        Args:
            shot_id: Shot ID
            error_info: Error information dictionary

        Returns:
            Fixed narrative dictionary, returns None on failure
        """
        print(f"\n{'='*70}")
        print(f"🔧 Intelligent Review Agent - Shot {shot_id}")
        print(f"{'='*70}")

        # 1. Read current context
        memory_package = self.load_shot_memory(shot_id)
        if not memory_package:
            print(f"  ❌ Unable to read context for Shot {shot_id}")
            return None

        # 2. Print error information summary
        self._print_error_summary(error_info)

        # 3. Build diagnosis prompt
        diagnosis_prompt = self._build_diagnosis_prompt(
            shot_id,
            error_info,
            memory_package
        )

        # 4. Call Gemini for diagnosis
        print(f"\n  📤 Calling Gemini 3 to analyze failure reason...")
        try:
            response = self.client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=diagnosis_prompt
            )
        except Exception as e:
            print(f"  ❌ Gemini call failed: {e}")
            return None

        # 5. Parse fix solution
        fix_solution = self._parse_fix_response(response.text)

        if not fix_solution:
            print(f"  ❌ Unable to parse fix solution")
            return None

        # 6. Print diagnosis result
        self._print_diagnosis_result(fix_solution)

        # 7. Verify fix solution
        print(f"\n  🔍 Verifying fix solution...")
        verification = self._verify_fix(
            memory_package.get("narrative", {}),
            fix_solution
        )

        if not verification["safe"]:
            print(f"  ❌ Fix solution did not pass verification")
            for warning in verification.get("warnings", []):
                print(f"     ⚠️  {warning}")

            # Try to automatically build complete text (if Gemini only output changes_applied)
            if self._can_auto_build_fix(fix_solution):
                print(f"  🔧 Attempting to automatically build complete fix text based on modification list...")
                self._auto_build_fix_text(memory_package.get("narrative", {}), fix_solution)

                # Re-verify
                verification = self._verify_fix(
                    memory_package.get("narrative", {}),
                    fix_solution
                )

                if verification["safe"]:
                    print(f"  ✅ Automatic build successful!")
                else:
                    print(f"  ❌ Automatic build failed")
                    return None
            else:
                return None

        print(f"  ✅ Fix solution passed verification")

        # 8. Apply fix
        print(f"\n  💾 Applying fix...")
        self._apply_fix(shot_id, fix_solution["fix_strategy"])

        # 9. Record fix history
        self._record_fix_history(shot_id, error_info, fix_solution, verification)

        print(f"\n  ✅ Fix completed!")

        return fix_solution["fix_strategy"]

    def load_shot_memory(self, shot_id):
        """Load memory package for specified shot"""
        if not os.path.exists(self.memory_file):
            return None

        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        memory_store = data.get('memory_store', {})

        # Support both string and number type shot_id
        shot_key = None
        for key in memory_store.keys():
            if str(key) == str(shot_id):
                shot_key = key
                break

        return memory_store.get(shot_key)

    def _print_error_summary(self, error_info):
        """Print error information summary"""
        print(f"\n  📋 Error Information Summary:")
        print(f"     Error Type: {error_info.get('error_type', 'UNKNOWN')}")

        if "response_details" in error_info:
            details = error_info["response_details"]

            # Print prompt_feedback
            if "prompt_feedback" in details:
                feedback = details["prompt_feedback"]
                if feedback.get("block_reason"):
                    print(f"     Block Reason: {feedback['block_reason']}")

            # Print candidates information
            if "candidates" in details:
                for candidate in details["candidates"]:
                    if candidate.get("finish_reason"):
                        print(f"     Finish Reason: {candidate['finish_reason']}")
                    if not candidate.get("has_content"):
                        print(f"     Content: None")
                    elif candidate.get("parts_empty"):
                        print(f"     Content Parts: Empty")

    def _build_diagnosis_prompt(self, shot_id, error_info, memory_package):
        """Build diagnosis prompt"""
        narrative = memory_package.get("narrative", {})
        visual_dna = memory_package.get("visual_dna", {})

        prompt = f"""
You are a professional AI image generation error diagnosis expert. Your task is to analyze the reasons for Gemini image generation failures and provide fix solutions.

## Error Information
```json
{json.dumps(error_info, indent=2, ensure_ascii=False)}
```

## Shot ID
{shot_id}

## Current Narrative
```json
{json.dumps(narrative, indent=2, ensure_ascii=False)}
```

## Visual DNA
```json
{json.dumps(visual_dna, indent=2, ensure_ascii=False)}
```

## Diagnosis Task
Please analyze the following possible reasons:

### 1. Safety Policy Violations
Check if the narrative contains:
- Violence, gore descriptions
- Pornographic or inappropriate content
- Hate speech or discriminatory content
- Self-harm, dangerous behavior descriptions
- Tobacco, alcohol, drug-related content

### 2. Content Policy Violations
Check if it involves:
- Copyright, trademark issues
- Real person portrait rights
- Protected character images
- Commercial brands

### 3. Prompt Issues
Check if prompt length exceeds limit (usually <10000 characters)
Check for format errors
Check for conflicting instructions
Check for overly complex descriptions

### 4. Technical Issues
Check if reference image path descriptions are reasonable
Check if character reference format is correct
Check for descriptions that may confuse the model

## ⚠️ Critical Requirement: Must Output Complete Text
**The action, i2v_prompt, and language_prompt in fix_strategy must be complete fixed text, not abbreviated!**

- ❌ **Incorrect Approach**: Only list modification changes like "replace X with Y"
- ✅ **Correct Approach**: Output complete fixed text (like rewriting the entire narrative)

**Example Explanation**:
If the original action is: "@character_01 brutally killing enemies in a bloody battlefield"
- ❌ Don't just write: changes_applied: ["replace 'bloody battlefield' with 'battlefield'", "replace 'brutally killing' with 'defeat'"]
- ✅ Must write: action: "@character_01 defeating enemies in a battlefield"

## Output Format
Please return diagnosis and fix solution in JSON format:

```json
{{
  "diagnosis": {{
    "primary_reason": "Primary reason",
    "secondary_reasons": ["Secondary reason 1", "Secondary reason 2"],
    "risk_level": "HIGH/MEDIUM/LOW",
    "detailed_analysis": "Detailed analysis"
  }},
  "fix_strategy": {{
    "action": "Complete fixed action text (must include all content, cannot be omitted)",
    "i2v_prompt": "Complete fixed i2v_prompt text (must include all content, cannot be omitted)",
    "language_prompt": "Complete fixed language_prompt text (must include all content, cannot be omitted)",
    "changes_applied": [
      "Modification 1: Describe specifically what was changed",
      "Modification 2: Describe specifically what was changed"
    ],
    "removed_content": ["Removed sensitive words"],
    "safeguard_added": ["Added safety measures"]
  }},
  "confidence": "HIGH/MEDIUM/LOW",
  "recommendation": "Whether regeneration should be done (YES/NO/UNSURE)"
}}
```

## Fix Principles
1. **Maintain Core Visual Elements**: Preserve characters, scenes, atmosphere, camera settings
2. **Soften Sensitive Language**: Replace trigger words with safer expressions
3. **Simplify Complex Descriptions**: Remove redundancy and details that may cause confusion
4. **Maintain Artistic Intent**: Maximize preservation of original creative intent while ensuring safety
5. **Minimize Modifications**: Only modify necessary parts, avoid over-changing original content
6. **Output Complete Text**: action/i2v_prompt/language_prompt must be complete text, not summaries

## Example Transformation

**Original Narrative (Problematic)**:
```json
{{
  "action": "@character_01 brutally killing enemies in a bloody battlefield, blood and flesh flying",
  "i2v_prompt": "Violent battle scene, @character_01 frantically slaughtering",
  "language_prompt": "Image full of blood and violence"
}}
```

**Fixed Narrative (Correct Format)**:
```json
{{
  "action": "@character_01 defeating enemies in an intense battlefield, flames of war flying",
  "i2v_prompt": "Intense battle scene, @character_01 fighting heroically",
  "language_prompt": "Image full of battle tension"
}}
```

**changes_applied field should record**:
```json
[
  "Replace 'bloody battlefield' with 'battlefield' (remove gore)",
  "Replace 'brutally killing' with 'defeating' (soften violence)",
  "Replace 'blood and flesh flying' with 'flames of war flying' (maintain intensity but safer)",
  "Replace 'frantically slaughtering' with 'fighting heroically' (change description angle)",
  "Replace 'full of blood and violence' with 'full of battle tension' (maintain atmosphere but safer)"
]
```

Please begin diagnosis and fixing...
"""
        return prompt

    def _parse_fix_response(self, response_text):
        """Parse Gemini's fix response"""
        # Extract JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        else:
            # Try to directly extract first JSON object
            first_brace = response_text.find('{')
            last_brace = response_text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                response_text = response_text[first_brace:last_brace + 1]

        try:
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            print(f"  ⚠️  Failed to parse JSON: {e}")
            print(f"  📋 Original response (first 500 characters):")
            print(f"     {response_text[:500]}")
            return None

    def _print_diagnosis_result(self, fix_solution):
        """Print diagnosis result"""
        diagnosis = fix_solution.get("diagnosis", {})

        print(f"\n  📊 Diagnosis Result:")
        print(f"     Primary Reason: {diagnosis.get('primary_reason', 'N/A')}")
        print(f"     Risk Level: {diagnosis.get('risk_level', 'N/A')}")

        secondary = diagnosis.get('secondary_reasons', [])
        if secondary:
            print(f"     Secondary Reasons:")
            for reason in secondary:
                print(f"       - {reason}")

        fix_strategy = fix_solution.get("fix_strategy", {})
        changes = fix_strategy.get("changes_applied", [])

        print(f"\n  ✏️  Applied Modifications ({len(changes)} items):")
        for change in changes[:5]:  # Only show first 5 items
            print(f"       - {change}")

        removed = fix_strategy.get("removed_content", [])
        if removed:
            print(f"\n  🗑️  Removed Content:")
            for item in removed:
                print(f"       - {item}")

    def _verify_fix(self, original_narrative, fix_solution):
        """Verify fix solution"""
        warnings = []

        # Get fix_strategy (fix strategy is in nested structure)
        fix_strategy = fix_solution.get("fix_strategy", {})

        # 1. Check if core fields are preserved
        required_fields = ["action", "i2v_prompt", "language_prompt"]
        for field in required_fields:
            if field not in fix_strategy:
                warnings.append(f"Missing required field: {field}")

        if warnings:
            return {"safe": False, "warnings": warnings}

        # 2. Check prompt length
        total_length = (
            len(fix_strategy.get("action", "")) +
            len(fix_strategy.get("i2v_prompt", "")) +
            len(fix_strategy.get("language_prompt", ""))
        )

        if total_length > 12000:
            warnings.append(f"Prompt too long: {total_length} characters (recommend <10000)")

        # 3. Check if obvious sensitive words were removed
        sensitive_keywords = [
            "blood", "gore", "violence", "kill", "death",
            "massacre"
        ]

        combined_text = " ".join([
            fix_strategy.get("action", ""),
            fix_strategy.get("i2v_prompt", ""),
            fix_strategy.get("language_prompt", "")
        ]).lower()

        found_keywords = [kw for kw in sensitive_keywords if kw in combined_text]
        if found_keywords:
            warnings.append(f"Still contains sensitive keywords: {', '.join(found_keywords)}")

        # 4. Check if character references are preserved
        if "@" not in combined_text and original_narrative:
            original_has_refs = "@" in str(original_narrative)
            if original_has_refs:
                warnings.append("Lost character references (@character_XX)")

        return {
            "safe": len(warnings) == 0,
            "warnings": warnings,
            "total_length": total_length
        }

    def _can_auto_build_fix(self, fix_solution):
        """Check if fix text can be automatically built based on modification list"""
        fix_strategy = fix_solution.get("fix_strategy", {})

        # Must have changes_applied field
        if not fix_strategy.get("changes_applied"):
            return False

        # At least one removed content, indicating clear modification intent
        removed = fix_strategy.get("removed_content", [])
        return len(removed) > 0

    def _auto_build_fix_text(self, original_narrative, fix_solution):
        """
        Automatically build complete fix text based on changes_applied

        Args:
            original_narrative: Original narrative dictionary
            fix_solution: Fix solution returned by Gemini (may lack complete text)
        """
        fix_strategy = fix_solution.get("fix_strategy", {})
        changes_applied = fix_strategy.get("changes_applied", [])
        removed_content = fix_strategy.get("removed_content", [])

        # Get text from original narrative
        original_action = original_narrative.get("action", "")
        original_i2v = original_narrative.get("i2v_prompt", "")
        original_language = original_narrative.get("language_prompt", "")

        # Apply modifications: parse and apply changes_applied one by one
        fixed_action = self._apply_changes_to_text(original_action, changes_applied, removed_content)
        fixed_i2v = self._apply_changes_to_text(original_i2v, changes_applied, removed_content)
        fixed_language = self._apply_changes_to_text(original_language, changes_applied, removed_content)

        # Update fix_solution
        fix_strategy["action"] = fixed_action
        fix_strategy["i2v_prompt"] = fixed_i2v
        fix_strategy["language_prompt"] = fixed_language

        print(f"  📝 Automatic build completed:")
        print(f"     - action: {len(fixed_action)} characters")
        print(f"     - i2v_prompt: {len(fixed_i2v)} characters")
        print(f"     - language_prompt: {len(fixed_language)} characters")

    def _apply_changes_to_text(self, original_text, changes_applied, removed_content):
        """
        Apply modification list to original text

        Args:
            original_text: Original text
            changes_applied: Modification list (format: "replace 'X' with 'Y' (reason)")
            removed_content: List of removed content

        Returns:
            Fixed text
        """
        if not original_text:
            return ""

        text = original_text

        # 1. First handle removed content
        for item in removed_content:
            # Remove exact match
            text = text.replace(item, "")
            # Remove lowercase version
            text = text.replace(item.lower(), "")
            # If it's a word, clean surrounding spaces
            text = text.replace(f" {item} ", " ")
            text = text.replace(f" {item}", "")
            text = text.replace(f"{item} ", " ")

        # 2. Handle replacement operations
        for change in changes_applied:
            # Parse format: "replace 'X' with 'Y' (reason)"
            # Use regex to extract content in quotes
            import re
            match = re.search(r"将['\"](.+?)['\"]替换为['\"](.+?)['\"]", change)
            if match:
                old_text = match.group(1)
                new_text = match.group(2)
                text = text.replace(old_text, new_text)

        # 3. Clean extra spaces
        text = re.sub(r'\s+', ' ', text)  # Merge multiple spaces into one
        text = text.strip()  # Remove leading/trailing spaces

        return text

    def _apply_fix(self, shot_id, fix_strategy):
        """Apply fix solution"""
        # 1. Backup original file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = os.path.join(self.backup_dir, f"memory_allocation_before_fix_{timestamp}.json")
        shutil.copy2(self.memory_file, backup_file)
        print(f"  💾 Backed up to: {backup_file}")

        # 2. Read memory allocation
        with open(self.memory_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        memory_store = data.get('memory_store', {})

        # Find shot
        shot_key = None
        for key in memory_store.keys():
            if str(key) == str(shot_id):
                shot_key = key
                break

        if shot_key is None:
            raise ValueError(f"Shot {shot_id} not in memory allocation")

        # 3. Apply fix
        memory_store[shot_key]["narrative"]["action"] = fix_strategy.get("action", "")
        memory_store[shot_key]["narrative"]["i2v_prompt"] = fix_strategy.get("i2v_prompt", "")
        memory_store[shot_key]["narrative"]["language_prompt"] = fix_strategy.get("language_prompt", "")

        # 4. Save
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Updated: {self.memory_file}")

    def _record_fix_history(self, shot_id, error_info, fix_solution, verification):
        """Record fix history"""
        history_entry = {
            "shot_id": shot_id,
            "fixed_at": datetime.now().isoformat(),
            "original_error": error_info.get("error_type", "UNKNOWN"),
            "diagnosis": fix_solution.get("diagnosis", {}),
            "changes": fix_solution.get("fix_strategy", {}).get("changes_applied", []),
            "verification": {
                "safe": verification["safe"],
                "warnings": verification.get("warnings", []),
                "total_length": verification.get("total_length", 0)
            }
        }

        # Read existing history
        history = []
        if os.path.exists(self.fix_history_file):
            with open(self.fix_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)

        # Add new record
        history.append(history_entry)

        # Save
        with open(self.fix_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def main():
    """Main function (for standalone testing)"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Intelligent Review Agent - Diagnose and fix generation failures',
        epilog='''
Usage Examples:
  python %(prog)s --shot 4
  python %(prog)s --shot 4 --memory memory_allocation.json
        '''
    )

    parser.add_argument('--shot', type=int, required=True, help='Shot ID to fix')
    parser.add_argument('--memory', default='memory_allocation.json', help='Memory allocation file')
    parser.add_argument('--style', default='realistic',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       help='Generation style')

    args = parser.parse_args()

    try:
        # Initialize review agent
        review_agent = IntelligentReviewAgent(
            memory_file=args.memory,
            style=args.style
        )

        # Simulate error information (should be obtained from generation_agent in actual use)
        error_info = {
            "error_type": "FINISH_REASON: FinishReason.OTHER",
            "response_details": {
                "candidates": [
                    {
                        "finish_reason": "FinishReason.OTHER",
                        "has_content": False
                    }
                ]
            }
        }

        # Execute fix
        fix_solution = review_agent.diagnose_and_fix(str(args.shot), error_info)

        if fix_solution:
            print(f"\n{'='*70}")
            print(f"✅ Fix successful!")
            print(f"{'='*70}")
            return 0
        else:
            print(f"\n❌ Fix failed")
            return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
