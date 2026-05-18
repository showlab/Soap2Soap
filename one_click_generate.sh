#!/bin/bash

# ============================================================================
# One-Click Video Generation Script
# Function: One-click generation from raw video to final complete video
# Usage: ./one_click_generate.sh <video_path_or_json> <style> [--auto-generate-face]
# Examples:
#   ./one_click_generate.sh titannictest.mp4 disney              # Start from video, manually upload faces
#   ./one_click_generate.sh titannictest.json disney --auto-generate-face  # AI-generated faces
# ============================================================================

set -e  # Exit immediately on error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored messages
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_step() {
    echo -e "\n${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}📌 $1${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}\n"
}

# Check parameters
if [ $# -lt 2 ]; then
    print_error "Insufficient parameters!"
    echo "Usage: $0 <video_path_or_json> <style> [--auto-generate-face]"
    echo ""
    echo "Parameter descriptions:"
    echo "  video_path_or_json: Video file path or JSON script file path"
    echo "                      - Supports .mp4/.avi/.mov and other video files"
    echo "                      - Supports .json script files (skip video understanding step)"
    echo "  style: Generation style (realistic/lego/disney/anime/clay/japanese_anime/family_guy)"
    echo "  --auto-generate-face: Optional parameter, enable AI automatic face generation mode (no manual upload required)"
    echo ""
    echo "Description:"
    echo "  The system will automatically complete the following steps:"
    echo "  1. Video understanding (if input is a video file)"
    echo "  2. Character puzzle generation"
    echo "  3. Environment and costume reference image generation"
    echo "  4. Video clip generation"
    echo "  5. Automatic stitching into final complete video"
    echo ""
    echo "Examples:"
    echo "  Start from video (manually upload faces):"
    echo "    $0 titannictest.mp4 disney"
    echo ""
    echo "  Start from video (AI auto-generate faces):"
    echo "    $0 titannictest.mp4 disney --auto-generate-face"
    echo ""
    echo "  Start from JSON (skip video understanding):"
    echo "    $0 titannictest.json disney --auto-generate-face"
    exit 1
fi

INPUT_PATH="$1"
STYLE="$2"
shift 2

# Parse optional parameters
AUTO_GENERATE_FACE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --auto-generate-face)
            AUTO_GENERATE_FACE="--auto-generate-face"
            print_info "AI automatic face generation mode enabled"
            shift
            ;;
        *)
            print_error "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Determine if input is video file or JSON file
INPUT_FILENAME=$(basename -- "$INPUT_PATH")
INPUT_EXT="${INPUT_FILENAME##*.}"

if [ "$INPUT_EXT" = "json" ]; then
    # Input is JSON file
    SCRIPT_JSON="$INPUT_FILENAME"
    VIDEO_MODE=false
    print_info "JSON file input detected, will skip video understanding step"
elif [ "$INPUT_EXT" = "mp4" ] || [ "$INPUT_EXT" = "avi" ] || [ "$INPUT_EXT" = "mov" ] || [ "$INPUT_EXT" = "mkv" ]; then
    # Input is video file
    VIDEO_PATH="$INPUT_PATH"
    VIDEO_NAME="${INPUT_FILENAME%.*}"
    SCRIPT_JSON="${VIDEO_NAME}.json"
    VIDEO_MODE=true
    print_info "Video file input detected, will execute complete workflow"
else
    print_error "Unsupported file format: $INPUT_EXT"
    echo "Please input video file (.mp4, .avi, .mov, .mkv) or JSON file (.json)"
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_PATH" ]; then
    print_error "File does not exist: $INPUT_PATH"
    exit 1
fi

print_step "Starting one-click generation workflow"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "                        🎬 Video Generation Process Overview               "
echo "═══════════════════════════════════════════════════════════════════════"
if [ "$VIDEO_MODE" = true ]; then
    echo "  📁 Input video: $VIDEO_PATH"
else
    echo "  📁 Input JSON: $SCRIPT_JSON"
fi
echo "  🎨 Generation style: $STYLE"
echo "  📝 Output script: $SCRIPT_JSON"
if [ -n "$AUTO_GENERATE_FACE" ]; then
    echo "  👤 Face mode: 🤖 AI auto-generate (Gemini)"
else
    echo "  👤 Face mode: 📤 Manual upload"
fi
echo ""
echo "  🔄 Execution steps:"
echo "     1️⃣  Video understanding and script generation (if needed)"
echo "     2️⃣  Character puzzle generation"
echo "     3️⃣  Environment and costume reference image generation"
echo "     4️⃣  Video clip generation + automatic stitching into final video"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# Step 1: Run video understanding script (only in video input mode)
# ============================================================================
if [ "$VIDEO_MODE" = true ]; then
    print_step "Step 1/4: Video Understanding and Script Generation"
    print_info "Running: run_scriptgen.sh $VIDEO_PATH $SCRIPT_JSON"

    if [ -f "run_scriptgen.sh" ]; then
        print_info "Running video understanding script..."
        bash run_scriptgen.sh "$VIDEO_PATH" "$SCRIPT_JSON"

        if [ ! -f "$SCRIPT_JSON" ]; then
            print_error "Video understanding failed, script file not generated: $SCRIPT_JSON"
            exit 1
        fi

        print_success "Video understanding completed!"
        print_info "Intermediate file: $SCRIPT_JSON"
    else
        print_error "run_scriptgen.sh script not found"
        exit 1
    fi
else
    print_step "Step 1/4: Skip Video Understanding"
    print_info "Using existing JSON file: $SCRIPT_JSON"
    print_success "Video understanding step skipped"
fi

# ============================================================================
# Step 2: Generate character puzzle
# ============================================================================
print_step "Step 2/4: Generate Character Puzzle"
print_info "Running: create_character_sheet.py $SCRIPT_JSON --style $STYLE"

if [ -f "create_character_sheet.py" ]; then
    print_info "Generating character puzzle (style: $STYLE)..."
    if [ -n "$AUTO_GENERATE_FACE" ]; then
        python create_character_sheet.py "$SCRIPT_JSON" --style "$STYLE" --auto-generate-face
    else
        python create_character_sheet.py "$SCRIPT_JSON" --style "$STYLE"
    fi

    if [ ! -f "main_characters_1.png" ] && [ ! -f "supporting_characters_1.png" ]; then
        print_warning "No character puzzle files generated, possibly no characters or generation failed"
    else
        print_success "Character puzzle generation completed!"
        print_info "Intermediate files: main_characters_*.png, supporting_characters_*.png"
        print_info "Intermediate files: character_mapping.json, character_position_mapping.json"
    fi
else
    print_warning "create_character_sheet.py not found, skipping character puzzle generation"
fi

# ============================================================================
# Step 3: Generate reference images
# ============================================================================
print_step "Step 3/4: Generate Environment and Costume Reference Images"
print_info "Running: agent_reference.py $SCRIPT_JSON --style $STYLE"

if [ -f "agent_reference.py" ]; then
    print_info "Generating reference images (style: $STYLE)..."
    python agent_reference.py "$SCRIPT_JSON" --style "$STYLE"

    # Check reference_images directory
    if [ -d "reference_images" ]; then
        env_count=$(ls -1 reference_images/*environment.png 2>/dev/null | wc -l)
        clothing_count=$(ls -1 reference_images/*clothing.png 2>/dev/null | wc -l)
        print_success "Reference image generation completed!"
        print_info "Generated $env_count environment reference images, $clothing_count costume reference images"
        print_info "Intermediate file directory: reference_images/"
    else
        print_warning "Reference image directory not generated"
    fi
else
    print_warning "agent_reference.py not found, skipping reference image generation"
fi

# ============================================================================
# Step 4: Generate video clips and automatically stitch into final complete video
# ============================================================================
print_step "Step 4/4: Generate Video Clips and Stitch Complete Video"
print_info "Running: agent_master.py $SCRIPT_JSON --style $STYLE"
print_info ""
print_warning "⚠️  Important note:"
print_warning "   - The system will automatically generate all video clips"
print_warning "   - After generation, it will automatically use ffmpeg to stitch into final complete video"
print_warning "   - Video generation requires a long time, please wait patiently..."
print_info ""

if [ -f "agent_master.py" ]; then
    print_info "Starting video clip generation..."
    print_info "Maximum retry count: 3"
    print_info ""

    if [ -n "$AUTO_GENERATE_FACE" ]; then
        python -B agent_master.py "$SCRIPT_JSON" --max-retries 3 --style "$STYLE" --auto-generate-face
    else
        python -B agent_master.py "$SCRIPT_JSON" --max-retries 3 --style "$STYLE"
    fi

    print_success "Video generation workflow completed!"
    print_info "Intermediate file directory: shots/ (video clips)"
else
    print_error "agent_master.py not found"
    exit 1
fi

# ============================================================================
# Completion
# ============================================================================
print_step "🎉 All Workflows Completed!"

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "                           📁 Output File Summary                         "
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "  📝 Script file: $SCRIPT_JSON"
if [ "$VIDEO_MODE" = true ]; then
    echo "  🎬 Original video: $VIDEO_PATH"
fi
echo ""
echo "  🖼️  Character related:"
echo "     • Character puzzles: main_characters_*.png, supporting_characters_*.png"
echo "     • Character mapping: character_mapping.json"
echo "     • Position mapping: character_position_mapping.json"
echo ""
echo "  🎨 Reference images:"
echo "     • Environment reference: reference_images/*_environment.png"
echo "     • Costume reference: reference_images/*_clothing.png"
echo ""
echo "  🎬 Video files:"
echo "     • Video clips: shots/shot_*_video.mp4"
echo "     • Final video: final_output_*.mp4  ← 🎯 This is your final video!"
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

print_success "All files have been generated!"
echo ""
print_info "💡 Tips:"
echo "   • The final complete video file starts with 'final_output_'"
echo "   • You can directly play or share this final video"
echo ""

if [ "$VIDEO_MODE" = false ]; then
    print_info "To regenerate with a different style, run:"
    echo "   $0 $SCRIPT_JSON <other_style>"
    echo ""
    print_info "Available styles: realistic, lego, disney, anime, clay, japanese_anime, family_guy"
fi
echo ""
