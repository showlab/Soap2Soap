#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyframe Generation/Regeneration Tool

Features:
1. Read all shots to be generated from memory allocation
2. Display status of each shot (generated, not generated, failed)
3. Support interactive selection of one or more shots to generate
4. Can generate ungenerated shots or regenerate existing shots
5. Automatic backup of original images (optional, only for regeneration)
6. Support style selection and quality inspection

Usage:
    python regenerate_shots.py                                    # Use default configuration
    python regenerate_shots.py --script clip1.json                # Specify script
    python regenerate_shots.py --style disney                     # Specify style
    python regenerate_shots.py --backup                           # Enable backup

Interactive commands:
    - Enter numbers (e.g., 1, 2, 3) to select keyframes to generate
    - Multiple selections separated by comma or space (e.g., 1,3,5 or 1 3 5)
    - Support range selection (e.g., 1-5)
    - Enter 'all' to select all (including ungenerated)
    - Enter 'failed' to select only failed and ungenerated
"""

import os
import sys
import json
import shutil
import argparse
import importlib
from pathlib import Path
from datetime import datetime

# Add project root directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agent_memory import MemoryAllocationAgent
    from agent_generation import GenerationAgent
    from agent_inspection import InspectionAgent
except ImportError as e:
    print(f"❌ Failed to import Agent modules: {e}")
    print("Please ensure agent_memory.py, agent_generation.py, and agent_inspection.py exist")
    sys.exit(1)


class RegenerateTool:
    """Keyframe Regeneration Tool"""

    def __init__(self, script_json="clip1_script.json", style="realistic", backup=False, max_retries=3, enable_inspection=True):
        """
        Initialize

        Args:
            script_json: Script JSON file path
            style: Generation style
            backup: Whether to backup original images
            max_retries: Maximum retry count
            enable_inspection: Whether to enable quality inspection
        """
        self.script_json = script_json
        self.style = style
        self.backup = backup
        self.max_retries = max_retries
        self.enable_inspection = enable_inspection

        # Configuration file paths
        self.character_mapping_file = "character_mapping.json"
        self.reference_dir = "reference_images"
        self.memory_allocation_file = "memory_allocation.json"

        # Backup directory
        self.backup_dir = "backup_images"

        # Agents
        self.memory_agent = None
        self.generation_agent = None
        self.inspection_agent = None

        # Generated shots
        self.generated_shots = []

    def load_memory_allocation(self):
        """Load memory allocation"""
        print(f"\n{'='*70}")
        print("🔧 Loading memory allocation")
        print(f"{'='*70}")

        if not os.path.exists(self.memory_allocation_file):
            print(f"❌ Memory allocation file does not exist: {self.memory_allocation_file}")
            print(f"   Please run first: python agent_master.py {self.script_json}")
            return False

        # Initialize Memory Agent
        self.memory_agent = MemoryAllocationAgent(
            script_json=self.script_json,
            character_mapping=self.character_mapping_file,
            reference_dir=self.reference_dir,
            style=self.style
        )

        self.memory_agent.load_memory_allocation(self.memory_allocation_file)
        print(f"✅ Memory allocation loaded")

        return True

    def scan_generated_images(self):
        """Scan generated keyframe images and all shots in memory allocation"""
        print(f"\n{'='*70}")
        print("🔍 Scanning generated keyframes and memory allocation")
        print(f"{'='*70}")

        # Get all shots from memory allocation
        memory_store = self.memory_agent.memory_store
        print(f"✅ Found {len(memory_store)} shots in memory allocation")

        # Scan generated image files
        existing_shots = {}
        for filename in os.listdir('.'):
            if filename.startswith('shot_') and filename.endswith('.png'):
                shot_id = filename.replace('shot_', '').replace('.png', '')
                file_stat = os.stat(filename)
                existing_shots[shot_id] = {
                    "filename": filename,
                    "file_size": file_stat.st_size,
                    "modify_time": datetime.fromtimestamp(file_stat.st_mtime)
                }

        print(f"✅ Found {len(existing_shots)} generated keyframe images")

        # Build complete shot list (based on memory allocation)
        self.generated_shots = []

        for shot_id in memory_store.keys():
            memory_package = memory_store[shot_id]
            time_range = memory_package.get("time_range", "Unknown")

            # Determine status
            if shot_id in existing_shots:
                # Generated
                shot_data = existing_shots[shot_id]

                # Check if file is valid (size is reasonable)
                is_valid = shot_data['file_size'] > 10240  # Greater than 10KB is considered valid
                status = "✅ Generated" if is_valid else "⚠️  Generation failed (file too small)"

                self.generated_shots.append({
                    "shot_id": shot_id,
                    "filename": shot_data['filename'],
                    "file_size": shot_data['file_size'],
                    "modify_time": shot_data['modify_time'],
                    "time_range": time_range,
                    "status": status,
                    "exists": True,
                    "is_valid": is_valid,
                    "selected": False
                })
            else:
                # Not generated
                self.generated_shots.append({
                    "shot_id": shot_id,
                    "filename": f"shot_{shot_id}.png",
                    "file_size": 0,
                    "modify_time": None,
                    "time_range": time_range,
                    "status": "❌ Not generated",
                    "exists": False,
                    "is_valid": False,
                    "selected": False
                })

        # Sort by shot_id
        self.generated_shots.sort(key=lambda x: float(x['shot_id']) if self._is_number(x['shot_id']) else 0)

        # Statistics
        generated_count = sum(1 for s in self.generated_shots if s['exists'] and s['is_valid'])
        not_generated_count = sum(1 for s in self.generated_shots if not s['exists'])
        failed_count = sum(1 for s in self.generated_shots if s['exists'] and not s['is_valid'])

        print(f"\n📊 Statistics:")
        print(f"   - Total: {len(self.generated_shots)} shots")
        print(f"   - ✅ Generated: {generated_count}")
        print(f"   - ❌ Not generated: {not_generated_count}")
        print(f"   - ⚠️  Generation failed: {failed_count}")

        return True

    def _is_number(self, s):
        """Check if string is a number"""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def display_shot_list(self):
        """Display keyframe list for user selection"""
        print(f"\n{'='*70}")
        print("📸 All keyframes list (including generated and ungenerated)")
        print(f"{'='*70}")
        print(f"{'No.':<6} {'Status':<12} {'Shot ID':<10} {'Time Range':<20} {'File Size':<12}")
        print(f"{'-'*70}")

        for idx, shot in enumerate(self.generated_shots, 1):
            shot_id = shot['shot_id']
            status = shot['status']
            time_range = shot['time_range']

            if shot['exists']:
                file_size_mb = shot['file_size'] / (1024 * 1024)
                print(f"{idx:<6} {status:<12} {shot_id:<10} {time_range:<20} {file_size_mb:>6.2f} MB")
            else:
                print(f"{idx:<6} {status:<12} {shot_id:<10} {time_range:<20} {'--':>12}")

        print(f"{'='*70}")
        print(f"💡 Tips:")
        print(f"   - ✅ Generated: File has been normally generated")
        print(f"   - ❌ Not generated: File has not been generated yet (can be generated)")
        print(f"   - ⚠️  Generation failed: File generation failed or file too small (can be regenerated)")
        print(f"{'='*70}")

    def interactive_select_shots(self):
        """Interactive selection of shots to regenerate"""
        print(f"\n💡 Usage instructions:")
        print(f"   - Enter numbers (e.g., 1, 2, 3) to select keyframes to generate")
        print(f"   - Multiple selections separated by comma or space (e.g., 1,3,5 or 1 3 5)")
        print(f"   - Support range selection (e.g., 1-5)")
        print(f"   - Enter 'all' to select all (including ungenerated)")
        print(f"   - Enter 'failed' to select only generation failed and ungenerated")
        print(f"   - Enter 'none' or press Enter to cancel selection")
        print(f"{'='*70}")

        while True:
            user_input = input(f"\nPlease enter the numbers to generate: ").strip()

            if not user_input or user_input.lower() == 'none':
                print(f"❌ Selection cancelled")
                return []

            if user_input.lower() == 'all':
                # Select all
                selected_indices = list(range(len(self.generated_shots)))
            elif user_input.lower() == 'failed':
                # Select only ungenerated and failed
                selected_indices = [
                    i for i, shot in enumerate(self.generated_shots)
                    if not shot['is_valid']
                ]
                if not selected_indices:
                    print(f"✅ No shots need regeneration")
                    return []
                print(f"✅ Automatically selected {len(selected_indices)} ungenerated or failed shots")
            else:
                selected_indices = self._parse_selection(user_input, len(self.generated_shots))

                if selected_indices is None:
                    continue

            # Display selected shots
            selected_shots = [self.generated_shots[i] for i in selected_indices]
            print(f"\n✅ Selected {len(selected_shots)} keyframes:")
            for shot in selected_shots:
                status_info = f"({shot['status']})"
                print(f"   - Shot {shot['shot_id']}: {shot['filename']} {status_info}")

            # Confirm
            confirm = input(f"\nConfirm generation of these keyframes? [Y/n]: ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                return selected_shots
            else:
                print(f"❌ Cancelled, please reselect")

    def _parse_selection(self, input_str, max_index):
        """Parse user input selection"""
        try:
            indices = set()

            # Split input (support comma and space)
            parts = input_str.replace(',', ' ').split()

            for part in parts:
                if '-' in part:
                    # Range selection (e.g., 1-5)
                    start, end = part.split('-')
                    start_idx = int(start.strip()) - 1
                    end_idx = int(end.strip()) - 1
                    indices.update(range(start_idx, end_idx + 1))
                else:
                    # Single selection
                    idx = int(part.strip()) - 1
                    indices.add(idx)

            # Validate index range
            indices_list = sorted(indices)
            for idx in indices_list:
                if idx < 0 or idx >= max_index:
                    print(f"❌ Invalid number: {idx + 1}")
                    return None

            return indices_list

        except ValueError:
            print(f"❌ Invalid input format, please re-enter")
            return None

    def backup_image(self, filename):
        """Backup original image"""
        if not self.backup:
            return True

        try:
            # Create backup directory
            os.makedirs(self.backup_dir, exist_ok=True)

            # Generate backup filename (add timestamp)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"{filename.replace('.png', '')}_{timestamp}.png"
            backup_path = os.path.join(self.backup_dir, backup_filename)

            # Copy file
            shutil.copy2(filename, backup_path)
            print(f"  💾 Backed up: {backup_path}")
            return True

        except Exception as e:
            print(f"  ⚠️  Backup failed: {e}")
            return False

    def regenerate_selected_shots(self, selected_shots):
        """Regenerate selected shots"""
        print(f"\n{'='*70}")
        print(f"🎨 Starting generation of {len(selected_shots)} keyframes")
        print(f"{'='*70}")
        print(f"Style: {self.style.upper()}")
        print(f"Backup: {'Enabled' if self.backup else 'Disabled'}")

        # Initialize Generation Agent
        print(f"\n{'='*70}")
        print("🔧 Initializing Generation Agent")
        print(f"{'='*70}")

        self.generation_agent = GenerationAgent(
            script_json=self.script_json,
            character_mapping=self.character_mapping_file,
            reference_dir=self.reference_dir,
            memory_agent=self.memory_agent,
            style=self.style
        )

        self.generation_agent.load_script_data()
        self.generation_agent.load_character_mappings()

        # Initialize Inspection Agent
        if self.enable_inspection:
            print(f"\n{'='*70}")
            print("🔧 Initializing Inspection Agent")
            print(f"{'='*70}")

            self.inspection_agent = InspectionAgent(
                character_mapping=self.character_mapping_file,
                use_gemini=True
            )

            self.inspection_agent.load_character_mappings()
            print(f"✅ Quality inspection enabled")

        # Statistics
        success_count = 0
        failed_count = 0
        new_generated_count = 0
        regenerated_count = 0

        for idx, shot in enumerate(selected_shots, 1):
            shot_id = shot['shot_id']
            filename = shot['filename']
            is_new = not shot['exists']

            if is_new:
                action = "Generating"
                new_generated_count += 1
            else:
                action = "Regenerating"
                regenerated_count += 1

            print(f"\n[{idx}/{len(selected_shots)}] {action} Shot {shot_id}")

            # Get character list in shot (for quality inspection)
            shot_characters = []
            if self.memory_agent:
                memory_package = self.memory_agent.get_shot_memory(shot_id)
                if memory_package:
                    shot_characters = memory_package.get("characters", [])

            # Retry loop
            shot_success = False
            for retry in range(self.max_retries):
                if retry > 0:
                    print(f"\n🔄 Retry {retry + 1}/{self.max_retries}...")

                # Backup original image (only for regeneration and when backup is enabled)
                if self.backup and not is_new and retry == 0:
                    self.backup_image(filename)

                # Generate
                try:
                    success = self.generation_agent.generate_image_for_shot(shot_id)

                    if not success:
                        print(f"❌ Shot {shot_id} {action} failed")

                        # Last retry failed, trigger intelligent review agent (image-only mode)
                        if retry == self.max_retries - 1:
                            print(f"\n{'='*70}")
                            print(f"⚠️  Shot {shot_id} reached maximum retries")
                            print(f"{'='*70}")

                            # Call intelligent review agent (will complete fix+generate+quality check internally)
                            fix_success = self.trigger_intelligent_review_agent(
                                shot_id,
                                shot_characters
                            )

                            if fix_success:
                                # Intelligent review agent has completed generation and quality check
                                print(f"✅ Shot {shot_id} intelligent review agent completed successfully")
                                shot_success = True
                                success_count += 1
                            else:
                                # Fix failed
                                print(f"❌ Shot {shot_id} intelligent fix failed")
                                failed_count += 1
                            break  # Break retry loop after intelligent review
                        elif retry < self.max_retries - 1:
                            # Still have retry chances, continue retrying
                            continue
                        else:
                            # Really failed
                            failed_count += 1
                            break

                    # Quality inspection (if enabled and has characters)
                    if success and self.enable_inspection and shot_characters:
                        print(f"\n🔍 Starting quality inspection...")

                        # Check each character
                        all_passed = True
                        for char_id in shot_characters:
                            image_path = f"shot_{shot_id}.png"

                            result = self.inspection_agent.inspect_single_image(image_path, char_id)

                            if not result.get("passed", False):
                                all_passed = False
                                print(f"❌ Shot {shot_id} quality inspection failed: {char_id}")
                                break  # Stop checking if any character fails
                            else:
                                print(f"✅ Shot {shot_id} quality inspection passed: {char_id}")

                        if not all_passed:
                            # Quality inspection failed, but image generated, accept current result
                            print(f"\n{'='*70}")
                            print(f"⚠️  Shot {shot_id} quality inspection failed")
                            print(f"{'='*70}")
                            print(f"💡 Image generated, accepting current result")
                            shot_success = True
                            success_count += 1
                            break
                        else:
                            print(f"✅ Quality inspection completed")
                            shot_success = True
                            success_count += 1
                            break
                    elif success:
                        # Generation successful but no quality inspection (not enabled or no characters)
                        if self.enable_inspection and not shot_characters:
                            print(f"⏭️  Skipping quality inspection (no character information)")
                        shot_success = True
                        success_count += 1
                        break

                except Exception as e:
                    print(f"❌ Shot {shot_id} {action} error: {e}")
                    import traceback
                    traceback.print_exc()
                    if retry < self.max_retries - 1:
                        continue
                    else:
                        failed_count += 1
                        break

            if not shot_success:
                failed_count += 1

        # Print summary
        print(f"\n{'='*70}")
        print("📊 Generation Summary")
        print(f"{'='*70}")
        print(f"Total: {len(selected_shots)}")
        print(f"  - New generated: {new_generated_count}")
        print(f"  - Regenerated: {regenerated_count}")
        print(f"✅ Success: {success_count}")
        print(f"❌ Failed: {failed_count}")

        if success_count > 0:
            success_rate = (success_count / len(selected_shots)) * 100
            print(f"Success rate: {success_rate:.1f}%")

        print(f"{'='*70}")

        return failed_count == 0

    def trigger_intelligent_review_agent(self, shot_id, shot_characters):
        """
        Trigger intelligent review agent for diagnosis and fix

        Args:
            shot_id: Shot ID
            shot_characters: Character list

        Returns:
            True if fix and retry successful, False if failed
        """
        try:
            # Dynamically import IntelligentReviewAgent
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
            error_info = self.generation_agent.get_last_error_info(shot_id)
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
                    else:
                        print(f"  ❌ Maximum fix attempts reached ({max_fix_attempts})")
                        return False

            # Reload memory allocation
            self.generation_agent.memory_agent.load_memory_allocation(self.memory_allocation_file)
            print(f"  ✅ Memory allocation updated")

            # ============================================================
            # Feature 2: Multi-round fix mechanism (max 3 rounds, 3 retries per round)
            # ============================================================
            max_fix_rounds = 3  # Maximum 3 fixes
            retries_per_round = 3  # 3 retries after each fix

            for fix_round in range(max_fix_rounds):
                if fix_round > 0:
                    print(f"\n{'='*70}")
                    print(f"🔧 Round {fix_round + 1}/{max_fix_rounds} fix...")
                    print(f"{'='*70}")

                    # Continue fixing on current basis
                    fix_solution = review_agent.diagnose_and_fix(shot_id, error_info)

                    if not fix_solution:
                        print(f"  ❌ Round {fix_round + 1} fix failed")
                        if fix_round < max_fix_rounds - 1:
                            print(f"  💡 Preparing for next round of fix...")
                            continue
                        else:
                            print(f"  ❌ Maximum fix rounds reached")
                            return False

                    # Reload memory allocation
                    self.generation_agent.memory_agent.load_memory_allocation(self.memory_allocation_file)
                    print(f"  ✅ Memory allocation updated")

                # ============================================================
                # Phase: Post-fix retry phase (3 retries after each fix)
                # ============================================================
                print(f"\n📋 Step 2: Regenerating with fixed context...")
                print(f"  💡 Extra retry count: {retries_per_round} times")
                if fix_round > 0:
                    print(f"  💡 Currently in round {fix_round + 1} fix")

                for extra_retry in range(retries_per_round):
                    print(f"\n{'='*70}")
                    print(f"🔄 [Post-fix] Retry {extra_retry + 1}/{retries_per_round}...")
                    print(f"{'='*70}")

                    # 1. Generate image
                    success = self.generation_agent.generate_image_for_shot(shot_id)

                    # 2. Handle generation failure
                    if not success:
                        print(f"❌ Shot {shot_id} still failed after fix")
                        print(f"⏭️  Skipping quality check (generation failed)")

                        if extra_retry < retries_per_round - 1:
                            print(f"💡 Continuing to next retry...")
                            continue
                        else:
                            print(f"\n❌ Shot {shot_id} reached maximum retries after fix")
                            # If not last fix round, continue to next round
                            if fix_round < max_fix_rounds - 1:
                                print(f"💡 Preparing for next round of fix...")
                                break  # Break inner loop, continue next fix round
                            else:
                                print(f"❌ Maximum fix rounds reached, giving up")
                                return False

                    # 3. Generation successful, perform full quality check
                    if success and shot_characters and self.enable_inspection:
                        print(f"\n🔍 [Post-fix] Starting quality check...")
                        print(f"  📋 Check items: Character consistency + Clothing consistency")

                        all_passed = True
                        failed_char = None

                        # Check each character
                        for char_id in shot_characters:
                            image_path = f"shot_{shot_id}.png"

                            print(f"\n  🔍 Checking character: {char_id}")
                            result = self.inspection_agent.inspect_single_image(image_path, char_id)

                            if not result.get("passed", False):
                                all_passed = False
                                failed_char = char_id
                                print(f"  ❌ Shot {shot_id} quality check failed: {char_id}")

                                # Print detailed failure reason
                                if "face_match" in result:
                                    print(f"     Character consistency: {result.get('face_match', 'N/A')}")
                                if "clothing_match" in result:
                                    print(f"     Clothing consistency: {result.get('clothing_match', 'N/A')}")

                                break  # Stop checking if any character fails
                            else:
                                print(f"  ✅ Shot {shot_id} quality check passed: {char_id}")

                                # Print detailed pass information
                                if "face_match" in result:
                                    print(f"     Character consistency: ✅")
                                if "clothing_match" in result:
                                    print(f"     Clothing consistency: ✅")

                        # 4. Handle quality check result
                        if all_passed:
                            # Quality check passed, completed successfully
                            print(f"\n✅ Shot {shot_id} post-fix generation successful and passed quality check")
                            return True
                        else:
                            # Quality check failed, but image generated, accept current result
                            print(f"\n{'='*70}")
                            print(f"⚠️  Shot {shot_id} quality check failed")
                            print(f"{'='*70}")
                            print(f"💡 Image generated, accepting current result (no more fixes)")
                            print(f"✅ Shot {shot_id} post-fix generation successful (quality check failed but accepted)")
                            return True  # Accept as long as generation successful, no more fixes

                    elif success:
                        # No character information or quality check not enabled, pass if generation successful
                        if self.enable_inspection and not shot_characters:
                            print(f"\n✅ Shot {shot_id} post-fix generation successful (no character information)")
                        else:
                            print(f"\n✅ Shot {shot_id} post-fix generation successful (quality check not enabled)")
                        return True

            # Theoretically should not reach here
            return False

        except Exception as e:
            print(f"❌ Intelligent review agent execution error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run(self):
        """Run complete workflow"""
        print(f"\n{'='*70}")
        print("🔄 Keyframe Generation/Regeneration Tool")
        print(f"{'='*70}")
        print(f"Script: {self.script_json}")
        print(f"Style: {self.style.upper()}")
        print(f"Backup: {'Enabled' if self.backup else 'Disabled'}")
        print(f"Quality inspection: {'Enabled' if self.enable_inspection else 'Disabled'}")
        print(f"Maximum retries: {self.max_retries}")

        # 1. Load memory allocation
        if not self.load_memory_allocation():
            return False

        # 2. Scan generated images and memory allocation
        if not self.scan_generated_images():
            return False

        # 3. Display list
        self.display_shot_list()

        # 4. Interactive selection
        selected_shots = self.interactive_select_shots()

        if not selected_shots:
            print(f"\n⚠️  No keyframes selected, exiting")
            return False

        # 5. Generate/Regenerate
        success = self.regenerate_selected_shots(selected_shots)

        if success:
            print(f"\n✅ Generation completed!")
        else:
            print(f"\n⚠️  Some keyframes failed to generate")

        return success


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Keyframe Generation/Regeneration Tool - Support generating incomplete shots or regenerating existing shots',
        epilog='''
Usage examples:
  python %(prog)s                                                # Use default configuration, interactive selection
  python %(prog)s --script clip2.json                            # Specify script file
  python %(prog)s --style disney                                 # Use Disney style for generation
  python %(prog)s --backup                                       # Enable original image backup (only valid for regeneration)
  python %(prog)s --max-retries 5                                # Set maximum retries to 5
  python %(prog)s --no-inspection                                # Disable quality inspection
  python %(prog)s --script clip1.json --style anime --backup     # Combined usage

Interactive commands:
  - Enter numbers to select (e.g., 1,3,5 or 1-5)
  - Enter 'all' to select all shots
  - Enter 'failed' to select only generation failed and ungenerated shots
        '''
    )

    parser.add_argument(
        '--script',
        default='clip1_script.json',
        help='Script JSON file path (default: clip1_script.json)'
    )

    parser.add_argument('--style',
                       choices=['realistic', 'lego', 'disney', 'anime', 'clay', 'japanese_anime', 'family_guy'],
                       default='realistic',
                       help='Generation style: realistic=Realistic(default), lego=Lego, disney=Disney, anime=Anime, clay=Clay, japanese_anime=Japanese anime, family_guy=Family Guy')

    parser.add_argument('--backup',
                       action='store_true',
                       help='Enable original image backup (backup to backup_images directory)')

    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry count (default: 3)')

    parser.add_argument('--no-inspection',
                       action='store_true',
                       help='Disable quality inspection (enabled by default)')

    args = parser.parse_args()

    try:
        # Check if in working directory
        if not os.path.exists(args.script):
            print(f"❌ Script file does not exist: {args.script}")
            return 1

        # Initialize tool
        tool = RegenerateTool(
            script_json=args.script,
            style=args.style,
            backup=args.backup,
            max_retries=args.max_retries,
            enable_inspection=not args.no_inspection
        )

        # Run
        success = tool.run()

        return 0 if success else 1

    except KeyboardInterrupt:
        print(f"\n\n⚠️  User interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
