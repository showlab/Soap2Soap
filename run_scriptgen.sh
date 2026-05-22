#!/bin/bash

# =================================================================
# Usage: ./run_scriptgen.sh <VIDEO_PATH> <OUTPUT_JSON_PATH>
# Example: ./run_scriptgen.sh ./video.mov ./my_script.json
# =================================================================

# Resolve python3 executable
PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# 1. Check parameters
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <VIDEO_PATH> <OUTPUT_JSON_PATH>"
    exit 1
fi

VIDEO_PATH=$1
OUTPUT_JSON=$2

# Check API KEY
if [ -z "$GENAI_API_KEY" ]; then
    echo "Error: GENAI_API_KEY environment variable is not set."
    echo "Please run: export GENAI_API_KEY='your_key_here'"
    exit 1
fi

# Get basic file information
DIR=$(dirname "$VIDEO_PATH")
FILENAME=$(basename "$VIDEO_PATH")
NAME_NO_EXT="${FILENAME%.*}"
AUDIO_PATH="$DIR/${NAME_NO_EXT}.wav"
# Use a distinct _transcript suffix so it never collides with the output script JSON
WHISPER_JSON_PATH="$DIR/${NAME_NO_EXT}_transcript.json"

echo "=========================================="
echo "🎬 Processing: $FILENAME"
echo "=========================================="

# --------------------------------------
# Step 1: Extract audio (FFmpeg)
# --------------------------------------
echo "[1/3] Extracting Audio to WAV..."
if [ -f "$AUDIO_PATH" ]; then
    echo "      Audio already exists, skipping ffmpeg."
else
    ffmpeg -y -i "$VIDEO_PATH" -ar 16000 -ac 1 -c:a pcm_s16le "$AUDIO_PATH" -loglevel error
    echo "      Audio extracted: $AUDIO_PATH"
fi

# --------------------------------------
# Step 2: Run Whisper (Speech to Text)
# --------------------------------------
echo "[2/3] Running OpenAI Whisper..."
echo "      Note: Whisper medium model is ~1.4 GB and will be downloaded on first run."
if [ -f "$WHISPER_JSON_PATH" ]; then
     echo "      Transcript already exists, skipping Whisper."
else
    # --task transcribe: Transcribe original language (Chinese), preserve complete semantic information
    # This avoids subject error inference problems during translation
    # Subsequent translation will be handled by Python script with context-aware translation API
    whisper "$AUDIO_PATH" --model medium --task transcribe --language zh --output_format json --output_dir "$DIR" --verbose False
    # Whisper writes <name>.json; rename to <name>_transcript.json to avoid collision
    if [ -f "$DIR/${NAME_NO_EXT}.json" ] && [ ! -f "$WHISPER_JSON_PATH" ]; then
        mv "$DIR/${NAME_NO_EXT}.json" "$WHISPER_JSON_PATH"
    fi
    echo "      Transcript generated: $WHISPER_JSON_PATH"
    echo "      Note: Transcript is in Chinese. Translation will be handled by Python script with context awareness."
fi

# --------------------------------------
# Step 3: Run Gemini (Auto-Extraction + Script)
# --------------------------------------
echo "[3/3] Generating Script with Gemini..."

# Input video + transcribed lines
"$PYTHON" video_to_script2.py \
    --video "$VIDEO_PATH" \
    --transcript "$WHISPER_JSON_PATH" \
    --output "$OUTPUT_JSON"

PYTHON_EXIT=$?
if [ $PYTHON_EXIT -ne 0 ]; then
    echo "Error: video_to_script2.py failed with exit code $PYTHON_EXIT"
    exit $PYTHON_EXIT
fi

echo "=========================================="
echo "✅ Done! Final Script saved to: $OUTPUT_JSON"
echo "=========================================="
