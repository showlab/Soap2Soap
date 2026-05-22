#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Orchestration Agent

Functions:
1. Coordinate all Agent executions
2. Manage generation process
3. Automatic retry mechanism
4. Generate reports

Usage:
    python agent_master.py clip1_script.json              # Full process
    python agent_master.py clip1_script.json --skip-reference  # Skip reference image generation
"""

import os
import sys
import json
import argparse
import time
import logging
import importlib
from pathlib import Path
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DualOutputLogger:
    """Logger that outputs to both terminal and file"""

    def __init__(self, log_file=None):
        """
        Initialize dual output logger

        Args:
            log_file: Log file path (if None, auto-generate)
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"generation_log_{timestamp}.log"

        self.log_file = log_file
        self.terminal = sys.stdout

        # Create log directory
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Open log file
        self.log_handle = open(log_file, 'w', encoding='utf-8')

        # Also redirect stderr
        self.terminal_stderr = sys.stderr
        self.stderr_handle = open(log_file.replace('.log', '_error.log'), 'w', encoding='utf-8')

    def write(self, message):
        """Write to both terminal and file"""
        self.terminal.write(message)
        self.log_handle.write(message)
        self.log_handle.flush()

    def write_error(self, message):
        """Write error message"""
        self.terminal_stderr.write(message)
        self.stderr_handle.write(message)
        self.stderr_handle.flush()

    def flush(self):
        """Flush buffers"""
        self.terminal.flush()
        self.log_handle.flush()

    def close(self):
        """Close log files"""
        self.log_handle.write(f"\n{'='*70}\n")
        self.log_handle.write(f"Log ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_handle.write(f"{'='*70}\n")
        self.log_handle.close()
        self.stderr_handle.close()

    def fileno(self):
        """Return file descriptor"""
        return self.terminal.fileno()

    def isatty(self):
        """Return whether it's a terminal"""
        return self.terminal.isatty()


def setup_logging(script_json, mode, style):
    """
    Setup logging system

    Args:
        script_json: Script file name
        mode: Generation mode
        style: Generation style

    Returns:
        DualOutputLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_name = os.path.splitext(os.path.basename(script_json))[0]
    log_file = f"logs/{script_name}_{mode}_{style}_{timestamp}.log"

    # Create logs directory
    if not os.path.exists("logs"):
        os.makedirs("logs")

    logger = DualOutputLogger(log_file)

    # Write log header
    header = f"""
======================================================================
🎬 Video2Video Auto Generation Process
======================================================================
Script: {script_json}
Mode: {mode}
Style: {style}
Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log file: {log_file}
======================================================================

"""
    logger.log_handle.write(header)
    logger.log_handle.flush()

    # Redirect stdout and stderr
    sys.stdout = logger
    sys.stderr = logger

    return logger

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow library needs to be installed")
    print("Please run: pip install Pillow")
    sys.exit(1)


class MasterAgent:
    """Master Orchestration Agent"""

    def __init__(self, script_json="clip1_script.json", max_retries=3, style="realistic", auto_generate_face=False, auto_yes=False):
        """
        Initialize

        Args:
            script_json: Script JSON file path
            max_retries: Maximum retry count
            style: Generation style ("realistic", "lego", "disney", "anime", "clay", "japanese_anime", "family_guy")
            auto_generate_face: Whether to automatically use AI to generate face images (True=AI generated, False=manually uploaded)
            auto_yes: Skip interactive video generation confirmation prompt
        """
        self.script_json = script_json
        self.max_retries = max_retries
        self.mode = "video"  # Force use video mode
        self.style = style
        self.auto_generate_face = auto_generate_face
        self.auto_yes = auto_yes

        # Configuration file paths
        self.character_mapping_file = "character_mapping.json"
        self.reference_dir = "reference_images"
        self.memory_allocation_file = "memory_allocation.json"

        # Agents
        self.memory_agent = None
        self.generation_agent = None
        self.inspection_agent = None

        # Statistics
        self.stats = {
            "total_shots": 0,
            "success_shots": 0,
            "failed_shots": 0,
            "retry_count": 0,
            "start_time": None,
            "end_time": None
        }

    def check_prerequisites(self):
        """Check prerequisites"""
        print(f"\n{'='*70}")
        print("🔍 Checking prerequisites")
        print(f"{'='*70}")

        issues = []

        # 1. Check script file
        if not os.path.exists(self.script_json):
            issues.append(f"❌ Script file does not exist: {self.script_json}")
        else:
            print(f"✅ Script file: {self.script_json}")

        # 2. Check character pairing configuration (not as fatal error)
        if not os.path.exists(self.character_mapping_file):
            print(f"⚠️  Character pairing file does not exist: {self.character_mapping_file}")
            print(f"   Tip: Will automatically run interactive character pairing process")
        else:
            print(f"✅ Character pairing file: {self.character_mapping_file}")

        # 3. Check character sheet images (main_characters and supporting_characters)
        main_char_sheet = None
        supporting_char_sheet = None

        # Find main character and supporting character sheet files
        for f in os.listdir('.'):
            if f.startswith('main_characters_') and f.endswith('.png'):
                main_char_sheet = f
            elif f.startswith('supporting_characters_') and f.endswith('.png'):
                supporting_char_sheet = f

        if main_char_sheet and supporting_char_sheet:
            # Both files exist
            print(f"✅ Main character sheet: {main_char_sheet}")
            print(f"✅ Supporting character sheet: {supporting_char_sheet}")
        elif main_char_sheet or supporting_char_sheet:
            # Only one file exists
            if main_char_sheet:
                print(f"⚠️  Main character sheet exists: {main_char_sheet}")
            if supporting_char_sheet:
                print(f"⚠️  Supporting character sheet exists: {supporting_char_sheet}")
            print(f"   Tip: Missing some character sheets, recommend re-running create_character_sheet.py")
        else:
            # Neither file exists
            print(f"⚠️  Character sheets do not exist")
            print(f"   Tip: Please run python create_character_sheet.py {self.script_json} first")

        # 4. Check reference image directory
        if not os.path.exists(self.reference_dir):
            issues.append(f"⚠️  Reference image directory does not exist: {self.reference_dir}")
            issues.append(f"   Tip: Please run python agent_reference.py {self.script_json} first")
        else:
            ref_files = [f for f in os.listdir(self.reference_dir) if f.endswith('.png')]
            if len(ref_files) == 0:
                issues.append(f"⚠️  Reference image directory is empty: {self.reference_dir}")
            else:
                print(f"✅ Reference image directory: {self.reference_dir} ({len(ref_files)} files)")

        # 5. Check environment variables
        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            issues.append(f"❌ Environment variable GENAI_API_KEY is not set")
        else:
            print(f"✅ GENAI_API_KEY is set")

        # 6. Check memory allocation file (optional)
        if os.path.exists(self.memory_allocation_file):
            print(f"✅ Memory allocation file: {self.memory_allocation_file} (exists)")
        else:
            print(f"⚠️  Memory allocation file does not exist: {self.memory_allocation_file} (will be auto-created)")

        if issues:
            print(f"\n⚠️  Found {len(issues)} issues:")
            for issue in issues:
                print(f"   {issue}")
            print(f"\nSuggestion: Please resolve the above issues before running this script")
            return False

        print(f"\n✅ All prerequisite checks passed")
        return True

    def load_script_data(self):
        """Load script data"""
        print(f"\n{'='*70}")
        print("📖 Loading script data")
        print(f"{'='*70}")

        with open(self.script_json, 'r', encoding='utf-8') as f:
            script_data = json.load(f)

        # Extract shots
        shots = [s for s in script_data.get("scenes", []) if not s.get("_disabled", False)]
        self.stats["total_shots"] = len(shots)

        print(f"✅ Loaded {len(shots)} shots")

        return shots, script_data

    def generate_single_shot_with_retry(self, shot_data, generation_agent, inspection_agent):
        """
        Generate single shot with inspection, support retry

        Args:
            shot_data: Shot data
            generation_agent: Generation Agent instance
            inspection_agent: Inspection Agent instance

        Returns:
            (success, retry_count)
        """
        shot_id = shot_data.get("scene_index")
        # Ensure shot_id is string type, consistent with keys in memory_allocation.json
        shot_id = str(shot_id) if shot_id is not None else None

        # Get character list in shot (for quality inspection)
        shot_characters = []
        if generation_agent.memory_agent:
            memory_package = generation_agent.memory_agent.get_shot_memory(shot_id)
            if memory_package:
                shot_characters = memory_package.get("characters", [])

        for retry in range(self.max_retries):
            if retry > 0:
                self.stats["retry_count"] += 1
                print(f"\n🔄 Retry {retry + 1}/{self.max_retries}...")

            # ========== New process: First auto-generate keyframe image, then auto-generate video ==========
            # Check if keyframe image already exists (if exists, means it passed quality check, skip image generation)
            image_path = f"shot_{shot_id}.png"
            image_already_exists = os.path.exists(image_path)

            if image_already_exists:
                print(f"✅ Detected existing keyframe image: {image_path}, skipping image generation step")

            # Step 1: Generate keyframe image (only when needed)
            image_success = image_already_exists

            if not image_already_exists:
                print(f"\n📷 Step 1: Generating keyframe image...")
                image_success = generation_agent.generate_image_for_shot(shot_id)

                if not image_success:
                    print(f"❌ Shot {shot_id} keyframe image generation failed")
                    # If image generation fails, trigger intelligent review agent
                    if retry == self.max_retries - 1:
                        print(f"\n{'='*70}")
                        print(f"⚠️  Shot {shot_id} reached maximum retry count")
                        print(f"{'='*70}")

                        # Call intelligent review agent
                        fix_success = self.trigger_intelligent_review_agent(
                            shot_id,
                            generation_agent,
                            inspection_agent,
                            shot_characters
                        )

                        if fix_success:
                            # Fix successful, retry generating image
                            image_success = generation_agent.generate_image_for_shot(shot_id)
                            if not image_success:
                                return False, retry
                        else:
                            # Fix failed
                            return False, retry
                    elif retry < self.max_retries - 1:
                        continue
                    else:
                        return False, retry

                else:
                    print(f"✅ Shot {shot_id} keyframe image generation successful")

            # Step 2: Quality inspection (only for newly generated keyframe images, skip existing ones)
            if image_success and not image_already_exists and shot_characters:
                print(f"\n🔍 Step 2: Keyframe image quality inspection...")
                all_passed = True
                inspection_feedback = []  # Collect inspection feedback

                for char_id in shot_characters:
                    image_path = f"shot_{shot_id}.png"
                    result = inspection_agent.inspect_single_image(image_path, char_id)

                    if not result.get("passed", False):
                        all_passed = False
                        feedback_msg = result.get("message", "No specific reason provided")
                        print(f"❌ Shot {shot_id} quality inspection failed: {char_id}")
                        print(f"   📋 Failure reason: {feedback_msg}")
                        inspection_feedback.append({
                            "character": char_id,
                            "issue": feedback_msg,
                            "passed": False
                        })
                        break
                    else:
                        print(f"✅ Shot {shot_id} quality inspection passed: {char_id}")

                if not all_passed:
                    if retry < self.max_retries - 1:
                        print(f"🔄 Quality inspection failed, preparing to retry...")

                        # Add inspection feedback to memory package for improvement in next generation
                        if inspection_feedback and generation_agent.memory_agent:
                            memory_package = generation_agent.memory_agent.get_shot_memory(shot_id)
                            if memory_package:
                                if "generation_feedback" not in memory_package:
                                    memory_package["generation_feedback"] = []
                                memory_package["generation_feedback"].extend(inspection_feedback)
                                print(f"   💡 Added quality inspection feedback to memory package, next generation will auto-improve")

                        continue
                    else:
                        print(f"⚠️  Shot {shot_id} quality inspection reached maximum retry count, using current result")

            # Step 3: Generate video based on keyframe image
            print(f"\n🎬 Step 3: Generating video based on keyframe image...")
            video_success = generation_agent.generate_video_for_shot(shot_id)

            if not video_success:
                print(f"❌ Shot {shot_id} video generation failed")
                if retry < self.max_retries - 1:
                    print(f"🔄 Video generation failed, only retrying video generation (keyframe image retained)...")
                    # Image file already exists, will skip image generation on next retry
                    continue
                else:
                    print(f"⚠️  Shot {shot_id} video generation reached maximum retry count")
                    # Even if video generation failed, but keyframe image succeeded, mark as partial success
                    return False, retry

            else:
                print(f"✅ Shot {shot_id} video generation successful")

            # If all steps successful, exit retry loop
            return True, retry

        return False, self.max_retries

    def trigger_intelligent_review_agent(self, shot_id, generation_agent, inspection_agent, shot_characters):
        """
        Trigger intelligent review agent for diagnosis and fix

        New features:
        1. If fix fails, restore original context and retry fix (max 3 times)
        2. If fix succeeds but still fails after 3 retries, continue fixing on current basis (max 3 times)

        Args:
            shot_id: Shot ID
            generation_agent: Generation Agent instance
            inspection_agent: Inspection Agent instance
            shot_characters: Character list

        Returns:
            True if fix and retry successful, False if failed
        """
        try:
            # Dynamic import IntelligentReviewAgent
            # Ensure project root is in path
            project_root = os.path.dirname(os.path.abspath(__file__))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            # Import module
            agent_intelligent_review = importlib.import_module('agent_intelligent_review')
            IntelligentReviewAgent = agent_intelligent_review.IntelligentReviewAgent

            print(f"🔧 Triggering intelligent review agent...")

            # Initialize review agent
            review_agent = IntelligentReviewAgent(
                memory_file=self.memory_allocation_file,
                style=self.style
            )

            # Collect error information
            error_info = generation_agent.get_last_error_info(shot_id)
            if not error_info:
                print(f"  ⚠️  Unable to collect error information, skipping intelligent fix")
                return False

            # ============================================================
            # Feature 1: Fix solution generation retry mechanism (max 3 times)
            # ============================================================
            max_fix_attempts = 3
            fix_solution = None

            for fix_attempt in range(max_fix_attempts):
                if fix_attempt > 0:
                    print(f"\n{'='*70}")
                    print(f"🔄 Attempt {fix_attempt + 1}/{max_fix_attempts} to generate fix solution...")
                    print(f"{'='*70}")

                # Execute diagnosis and fix
                print(f"\n📋 Step 1: Intelligently diagnosing failure reason...")
                fix_solution = review_agent.diagnose_and_fix(shot_id, error_info)

                if fix_solution:
                    # Fix successful, break loop
                    break
                else:
                    print(f"  ❌ Unable to generate fix solution")

                    if fix_attempt < max_fix_attempts - 1:
                        print(f"  💡 Preparing to retry fix solution generation...")
                        # Don't restore context, because diagnose_and_fix won't save modifications on failure
                    else:
                        print(f"  ❌ Reached maximum fix attempt count ({max_fix_attempts} times)")
                        return False

            # Reload memory allocation
            generation_agent.memory_agent.load_memory_allocation(self.memory_allocation_file)
            print(f"  ✅ Memory allocation updated")

            # ============================================================
            # Feature 2: Multi-round fix mechanism (max 3 rounds, 3 retries per round)
            # ============================================================
            max_fix_rounds = 3  # Maximum 3 fixes
            retries_per_round = 3  # 3 retries after each fix

            for fix_round in range(max_fix_rounds):
                if fix_round > 0:
                    print(f"\n{'='*70}")
                    print(f"🔧 Round {fix_round + 1}/{max_fix_rounds} of fixes...")
                    print(f"{'='*70}")

                    # Continue fixing on current basis
                    fix_solution = review_agent.diagnose_and_fix(shot_id, error_info)

                    if not fix_solution:
                        print(f"  ❌ Round {fix_round + 1} fix failed")
                        if fix_round < max_fix_rounds - 1:
                            print(f"  💡 Preparing for next round of fixes...")
                            continue
                        else:
                            print(f"  ❌ Reached maximum fix rounds")
                            return False

                    # Reload memory allocation
                    generation_agent.memory_agent.load_memory_allocation(self.memory_allocation_file)
                    print(f"  ✅ Memory allocation updated")

                # ============================================================
                # Phase: Post-fix retry phase (3 retries after each fix)
                # ============================================================
                print(f"\n📋 Step 2: Regenerating with fixed context...")
                print(f"  💡 Additional retry count: {retries_per_round} times")
                if fix_round > 0:
                    print(f"  💡 Currently in round {fix_round + 1} of fixes")

                for extra_retry in range(retries_per_round):
                    print(f"\n{'='*70}")
                    print(f"🔄 [Post-fix] Retry {extra_retry + 1}/{retries_per_round}...")
                    print(f"{'='*70}")

                    # 1. Generate image
                    success = generation_agent.generate_image_for_shot(shot_id)

                    # 2. Handle generation failure
                    if not success:
                        print(f"❌ Shot {shot_id} still failed to generate after fix")
                        print(f"⏭️  Skipping quality inspection (generation failed)")

                        if extra_retry < retries_per_round - 1:
                            print(f"💡 Continuing to next retry...")
                            continue
                        else:
                            print(f"\n❌ Shot {shot_id} reached maximum retry count after fix")
                            # If not last fix round, continue to next round
                            if fix_round < max_fix_rounds - 1:
                                print(f"💡 Preparing for next round of fixes...")
                                break  # Break inner loop, continue next fix round
                            else:
                                print(f"❌ Reached maximum fix rounds, giving up")
                                return False

                    # 3. Generation successful, perform full quality inspection
                    if success and shot_characters:
                        print(f"\n🔍 [Post-fix] Starting quality inspection...")
                        print(f"  📋 Inspection items: Character consistency + Clothing consistency")

                        all_passed = True
                        failed_char = None

                        # Inspect each character
                        for char_id in shot_characters:
                            image_path = f"shot_{shot_id}.png"

                            print(f"\n  🔍 Inspecting character: {char_id}")
                            result = inspection_agent.inspect_single_image(image_path, char_id)

                            if not result.get("passed", False):
                                all_passed = False
                                failed_char = char_id
                                print(f"  ❌ Shot {shot_id} quality inspection failed: {char_id}")

                                # Print detailed failure reason
                                if "face_match" in result:
                                    print(f"     Character consistency: {result.get('face_match', 'N/A')}")
                                if "clothing_match" in result:
                                    print(f"     Clothing consistency: {result.get('clothing_match', 'N/A')}")

                                break  # Stop inspection if any character fails
                            else:
                                print(f"  ✅ Shot {shot_id} quality inspection passed: {char_id}")

                                # Print detailed pass information
                                if "face_match" in result:
                                    print(f"     Character consistency: ✅")
                                if "clothing_match" in result:
                                    print(f"     Clothing consistency: ✅")

                        # 4. Handle quality inspection result
                        if all_passed:
                            # Quality inspection passed, success completed
                            print(f"\n✅ Shot {shot_id} generation successful after fix and passed quality inspection")
                            return True
                        else:
                            # Quality inspection failed
                            print(f"\n🔄 Quality inspection failed, preparing to retry...")

                            if extra_retry < retries_per_round - 1:
                                print(f"💡 Still have {retries_per_round - extra_retry - 1} retry opportunities")
                                continue
                            else:
                                print(f"\n{'='*70}")
                                print(f"⚠️  Shot {shot_id} quality inspection still failed after fix")
                                print(f"{'='*70}")
                                # If not last fix round, continue to next round
                                if fix_round < max_fix_rounds - 1:
                                    print(f"💡 Preparing for next round of fixes...")
                                    break  # Break inner loop, continue next fix round
                                else:
                                    print(f"❌ Reached maximum fix rounds, accepting current result")
                                    return True  # Return True means at least generated image
                    else:
                        # No character information, generation success means pass
                        print(f"\n✅ Shot {shot_id} generation successful after fix (no character information)")
                        return True

            # Theoretically won't reach here
            return False

        except Exception as e:
            print(f"❌ Intelligent review agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_generation_workflow(self, skip_reference=False):
        """Execute complete generation process"""
        self.stats["start_time"] = datetime.now()

        print(f"\n{'='*70}")
        print("🎬 Video2Video Auto Generation Process")
        print(f"{'='*70}")
        print(f"Script: {self.script_json}")
        print(f"Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Maximum retry count: {self.max_retries}")

        # 1. Check prerequisites
        if not self.check_prerequisites():
            return False

        # 1.5 Auto-generate character pairing (if needed)
        if not os.path.exists(self.character_mapping_file):
            print(f"\n{'='*70}")
            print("🎭 Character pairing configuration needed")
            print(f"{'='*70}")
            print(f"Detected missing character pairing file, will automatically run interactive pairing process...")
            print(f"")

            try:
                # Import CharacterSheetGenerator
                from create_character_sheet import CharacterSheetGenerator

                # Initialize Gemini client (for AI-generated face)
                gemini_client = None
                if self.auto_generate_face:
                    try:
                        from google import genai
                        api_key = os.environ.get("GENAI_API_KEY")
                        if api_key:
                            gemini_client = genai.Client(api_key=api_key)
                    except Exception as e:
                        print(f"⚠️  Gemini client initialization failed: {e}")

                generator = CharacterSheetGenerator(
                    script_json=self.script_json,
                    gemini_client=gemini_client,
                    style=self.style,
                    auto_generate_face=self.auto_generate_face
                )

                # Load script data
                result = generator.load_script_data()
                if not result:
                    print(f"❌ Failed to load script data")
                    return False

                data, characters = result

                # Interactive pairing
                print(f"\nStarting interactive character pairing process...")
                generator.interactive_mapping(characters)

                # Save configuration and generate long image
                generator.save_config(self.character_mapping_file)
                # No longer force generate long image, since collage is sufficient
                # success = generator.generate_character_sheet(self.character_sheet_file)
                # if not success:
                #     print(f"❌ Failed to generate character sheet long image")
                #     return False

                print(f"\n{'='*70}")
                print(f"✅ Character pairing completed!")
                print(f"   - Pairing configuration: {self.character_mapping_file}")
# print(f"   - Character sheet long image: {self.character_sheet_file}")  # No longer output long image info
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"❌ Character pairing process failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        # 2. Load script data
        shots, script_data = self.load_script_data()

        # 3. Import Agent modules
        try:
            from agent_memory import MemoryAllocationAgent
            from agent_generation import GenerationAgent
            from agent_inspection import InspectionAgent
        except ImportError as e:
            print(f"❌ Failed to import Agent modules: {e}")
            return False

        # 4. Initialize Memory Allocation Agent
        print(f"\n{'='*70}")
        print("🔧 Initializing Memory Allocation Agent")
        print(f"{'='*70}")

        self.memory_agent = MemoryAllocationAgent(
            script_json=self.script_json,
            character_mapping=self.character_mapping_file,
            reference_dir=self.reference_dir,
            style=self.style
        )

        # Check if need to regenerate memory allocation
        need_reallocate = False

        if not os.path.exists(self.memory_allocation_file):
            print(f"Memory allocation file does not exist, will create new one")
            need_reallocate = True
        else:
            # Check if character_mapping.json is newer than memory_allocation.json
            char_mapping_mtime = os.path.getmtime(self.character_mapping_file)
            memory_allocation_mtime = os.path.getmtime(self.memory_allocation_file)

            if char_mapping_mtime > memory_allocation_mtime:
                print(f"⚠️  Detected character pairing file has been updated, will regenerate memory allocation")
                need_reallocate = True
            else:
                print(f"Loading existing memory allocation: {self.memory_allocation_file}")

        if need_reallocate:
            print(f"Creating new memory allocation...")
            self.memory_agent.load_script_data()
            self.memory_agent.load_character_mappings()
            self.memory_agent.allocate_all_memory()
            self.memory_agent.save_memory_allocation(self.memory_allocation_file)
        else:
            self.memory_agent.load_memory_allocation(self.memory_allocation_file)

        # 5. Initialize Generation and Inspection Agents
        print(f"\n{'='*70}")
        print("🔧 Initializing Generation and Inspection Agents")
        print(f"{'='*70}")

        generation_agent = GenerationAgent(
            script_json=self.script_json,
            character_mapping=self.character_mapping_file,
            reference_dir=self.reference_dir,
            memory_agent=self.memory_agent,
            style=self.style
        )

        generation_agent.load_script_data()
        generation_agent.load_character_mappings()
        # Memory allocation already loaded by master agent, only need to load reference
        # No need to reallocate memory

        inspection_agent = InspectionAgent(
            character_mapping=self.character_mapping_file,
            use_gemini=True
        )

        inspection_agent.load_character_mappings()

        # 6. Pre-generation confirmation (video mode only)
        if self.mode == "video":
            print(f"\n{'='*70}")
            print("⚠️  Important notice: Video generation mode")
            print(f"{'='*70}")
            print(f"You are about to use Gemini Veo 3 to generate video clips.")
            print(f"")
            print(f"📊 Estimated information:")
            print(f"   - Shot count: {len(shots)}")
            print(f"   - Time per shot: 2-5 minutes")
            print(f"   - Total time: {len(shots) * 2}-{len(shots) * 5} minutes")
            print(f"   - API call count: {len(shots)} times")
            print(f"")
            print(f"💰 Cost notice:")
            print(f"   Video generation API call cost is significantly higher than image generation.")
            print(f"   Please confirm you understand and are willing to bear related costs.")
            print(f"")
            print(f"💡 Suggestion:")
            print(f"   If this is the first run, it's recommended to generate keyframes (image) first to verify effects,")
            print(f"   then generate videos after confirming satisfaction.")
            print(f"{'='*70}")

            # User confirmation
            if self.auto_yes:
                print(f"\n✅ Auto-confirmed (--yes flag), starting video generation...")
            else:
                while True:
                    user_input = input("\nContinue to generate video? Please enter [Y]es or [N]o: ").strip().lower()

                    if user_input in ['y', 'yes', '是', '好的']:
                        print(f"\n✅ User confirmed, starting video generation...")
                        break
                    elif user_input in ['n', 'no', '否', '不']:
                        print(f"\n❌ User cancelled, exiting program.")
                        print(f"💡 Tip: You can run the following command to generate keyframes:")
                        print(f"   python agent_master.py {self.script_json} --mode image")
                        return False
                    else:
                        print(f"⚠️  Invalid input, please enter Y or N")

        # 7. Generate all shots
        print(f"\n{'='*70}")
        mode_name = "Video clips" if self.mode == "video" else "Keyframe images"
        print(f"🎨 Starting to generate {len(shots)} shots ({mode_name})")
        print(f"{'='*70}")

        for idx, shot in enumerate(shots, 1):
            shot_id_raw = shot.get("scene_index")
            # Ensure shot_id is string type, consistent with keys in memory_allocation.json
            shot_id = str(shot_id_raw) if shot_id_raw is not None else None

            print(f"\n[{idx}/{len(shots)}] Processing Shot {shot_id}")

            # Temporarily modify scene_index in shot to string
            shot_modified = shot.copy()
            shot_modified["scene_index"] = shot_id

            success, retry_count = self.generate_single_shot_with_retry(
                shot_modified, generation_agent, inspection_agent
            )

            if success:
                self.stats["success_shots"] += 1
                print(f"✅ Shot {shot_id} completed (retried {retry_count} times)")
            else:
                self.stats["failed_shots"] += 1
                print(f"❌ Shot {shot_id} failed (reached maximum retry count)")

            # Avoid requests too fast
            time.sleep(2)

        # 6. Complete video generation
        self.stats["end_time"] = datetime.now()
        self.print_summary()

        # 7. Auto-concatenate videos (in video mode)
        if self.mode == "video":
            final_video_path = self.concatenate_videos()
            if final_video_path:
                print(f"\n{'='*70}")
                print(f"🎉 Final complete video generated: {final_video_path}")
                print(f"{'='*70}")
            else:
                print(f"\n⚠️  Video concatenation failed, please check shot_*_video.mp4 files in the working directory")

        return self.stats["failed_shots"] == 0

    def concatenate_videos(self):
        """
        Concatenate all generated video clips into complete video
        Use ffmpeg to concatenate in shot order

        Returns:
            Returns final video path on success, None on failure
        """
        import subprocess
        import glob
        import re

        print(f"\n{'='*70}")
        print("🎬 Step 5: Video concatenation")
        print(f"{'='*70}")

        # Scan current directory for video clips (legacy code looked in shots/ subdir,
        # but agent_generation saves files to the working directory directly)
        video_dir = "."
        print(f"📁 Scanning for video clips in current directory")

        # Get all video files
        video_files = glob.glob(os.path.join(video_dir, "shot_*_video.mp4"))

        if not video_files:
            print(f"❌ No video files found: {video_dir}/shot_*_video.mp4")
            return None

        # Sort by shot number (use regex to extract number)
        def extract_shot_number(filepath):
            """Extract shot number from filename"""
            match = re.search(r'shot_(\d+)_video\.mp4', os.path.basename(filepath))
            return int(match.group(1)) if match else 0

        video_files.sort(key=extract_shot_number)

        print(f"✅ Found {len(video_files)} video clips:")
        for i, vf in enumerate(video_files, 1):
            print(f"   [{i}] {os.path.basename(vf)}")

        # Generate concatenation list file
        list_file = "concat_list.txt"
        list_file_path = os.path.abspath(list_file)

        with open(list_file, 'w') as f:
            for vf in video_files:
                # Use absolute path to avoid ffmpeg not finding files
                abs_path = os.path.abspath(vf).replace('\\', '/')
                f.write(f"file '{abs_path}'\n")

        print(f"\n📝 Generated concatenation list file: {list_file_path}")

        # Output video filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_video = f"final_output_{timestamp}.mp4"

        print(f"\n🎥 Starting video concatenation...")
        print(f"   Output file: {output_video}")
        print(f"   Using ffmpeg for lossless concatenation (no re-encoding needed)...")
        print(f"   This may take a few seconds...\n")

        # Use ffmpeg to concatenate (concat demuxer, no re-encoding)
        try:
            # Use concat demuxer for fast lossless concatenation
            cmd = [
                "ffmpeg", "-y", "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
                output_video
            ]

            print(f"🔧 Executing command:")
            print(f"   ffmpeg -f concat -safe 0 -i {list_file_path} -c copy {output_video}")
            print(f"")

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Get video information
                if os.path.exists(output_video):
                    file_size = os.path.getsize(output_video) / (1024 * 1024)  # MB
                    print(f"\n{'='*70}")
                    print(f"✅ Video concatenation completed!")
                    print(f"{'='*70}")
                    print(f"🎬 Final video file: {output_video}")
                    print(f"📊 File size: {file_size:.2f} MB")

                    # Get video duration
                    try:
                        probe_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1", output_video]
                        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                        if probe_result.returncode == 0 and probe_result.stdout.strip():
                            print(f"⏱  Video duration: {probe_result.stdout.strip()} seconds")
                    except:
                        pass

                    # Clean temporary files
                    try:
                        os.remove(list_file)
                        print(f"🗑  Cleaned temporary file: {list_file}")
                    except:
                        pass

                    return output_video
                else:
                    print(f"\n❌ Video file not generated: {output_video}")
                    return None
            else:
                print(f"\n❌ Video concatenation failed!")
                print(f"   FFmpeg return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error message:")
                    for line in result.stderr.split('\n')[:10]:
                        if line.strip():
                            print(f"     {line}")
                # Keep list file for debugging
                print(f"   Concatenation list file retained: {list_file_path}")
                return None

        except FileNotFoundError:
            print(f"\n❌ ffmpeg not installed or not in PATH")
            print(f"   Please install ffmpeg: https://ffmpeg.org/download.html")
            return None
        except Exception as e:
            print(f"\n❌ Video concatenation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def print_summary(self):
        """Print execution summary"""
        duration = self.stats["end_time"] - self.stats["start_time"]

        print(f"\n{'='*70}")
        print("📊 Execution Summary")
        print(f"{'='*70}")
        print(f"Start time: {self.stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End time: {self.stats['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        print(f"")
        print(f"Total shots: {self.stats['total_shots']}")
        print(f"✅ Successful: {self.stats['success_shots']}")
        print(f"❌ Failed: {self.stats['failed_shots']}")
        print(f"🔄 Retry count: {self.stats['retry_count']}")

        if self.stats["total_shots"] > 0:
            success_rate = (self.stats["success_shots"] / self.stats["total_shots"]) * 100
            print(f"Success rate: {success_rate:.1f}%")

        print(f"{'='*70}")

        # Save summary to JSON
        summary_file = "generation_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                "script": self.script_json,
                "start_time": self.stats["start_time"].isoformat(),
                "end_time": self.stats["end_time"].isoformat(),
                "duration_seconds": duration.total_seconds(),
                "stats": {
                    "total_shots": self.stats["total_shots"],
                    "success_shots": self.stats["success_shots"],
                    "failed_shots": self.stats["failed_shots"],
                    "retry_count": self.stats["retry_count"]
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"✅ Summary saved to: {summary_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Master Orchestration Agent - Coordinate all agents to execute complete video generation process',
        epilog='''
Usage examples:
  python %(prog)s                                           # Manually upload face, generate complete video
  python %(prog)s --auto-generate-face                      # AI auto-generate face, generate complete video
  python %(prog)s clip2.json                                # Specify other script file
  python %(prog)s clip1.json --auto-generate-face --style disney  # AI generate face + Disney style
  python %(prog)s clip1.json --max-retries 5                # Set maximum retry count

Note: The system will automatically generate video clips and concatenate them into final complete video
        '''
    )

    parser.add_argument(
        'script',
        nargs='?',
        default='clip1_script.json',
        help='Script JSON file path (default: clip1_script.json)'
    )

    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry count (default: 3)')

    parser.add_argument('--style',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       default='realistic',
                       help='Generation style: realistic=realistic(default), lego=Lego, disney=Disney, anime=Japanese anime, clay=clay animation, japanese_anime=Japanese manga, family_guy=Family Guy American cartoon')

    parser.add_argument('--auto-generate-face', action='store_true',
                       help='Enable AI auto-generate face mode (use Gemini to auto-generate face images based on character descriptions, no manual upload needed)')

    parser.add_argument('--yes', '-y', action='store_true',
                       help='Auto-confirm video generation without interactive prompt')

    args = parser.parse_args()

    # ========== Initialize logging system ==========
    logger = setup_logging(args.script, "video", args.style)  # Force use video mode
    print(f"📝 Log file: {logger.log_file}")

    # Display running mode
    print(f"\n{'='*70}")
    print("🎬 Video Generation Mode")
    print(f"{'='*70}")
    print(f"System will automatically generate all video clips and concatenate into final complete video")
    print(f"Style: {args.style}")

    # Display face acquisition mode
    if args.auto_generate_face:
        print("Face mode: 🤖 AI auto-generate (Gemini)")
    else:
        print("Face mode: 📤 Manual upload")
    print(f"{'='*70}")

    try:
        # Initialize Master Agent
        agent = MasterAgent(
            script_json=args.script,
            max_retries=args.max_retries,
            style=args.style,
            auto_generate_face=args.auto_generate_face,
            auto_yes=args.yes
        )

        # Execute complete process
        success = agent.run_generation_workflow()

        # Close logger
        logger.close()
        return 0 if success else 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  User interrupted")
        logger.close()
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        logger.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
