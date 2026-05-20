# Soap2Soap

**Video-to-Video generation powered by Google Gemini.** Transform any video into a fully stylized animated version — Pixar, Disney, LEGO, anime, and more — with consistent characters, environments, and cinematic composition preserved across every shot.

---

## How It Works

```
Input Video
    ↓
Step 0  Whisper audio transcription (dialogue + timestamps)
    ↓
Step 1  SceneDetect cut detection → per-shot clip upload → parallel Gemini analysis
        (character extraction from full video + detailed per-shot schema)
    ↓
Step 2  Character reference images (Imagen 3 / Gemini) + unified Design Sheet
    ↓
Step 3  Prompt compilation + Gemini style rewrite (LEGO, Pixar, Disney, etc.)
    ↓
Step 4  Keyframe generation — Consistency mode: 2×2 grid per scene group → crop
    ↓
Step 4b Keyframe inspection & auto-fix (Pass 1: per-frame; Pass 2: grid consistency)
    ↓
Step 5  Video clips (Veo 3 or static fallback for dev mode)
    ↓
Step 6  ffmpeg merge → final video (1280×720)
```

---

## Setup

### 1. Prerequisites

- **Python 3.8+**
- **ffmpeg** — video processing
  ```bash
  # macOS
  brew install ffmpeg
  # Ubuntu
  sudo apt install ffmpeg
  ```

### 2. Install Python Dependencies

```bash
pip install google-genai Pillow opencv-python scenedetect[opencv] openai-whisper
```

> **macOS (Homebrew Python):** add `--break-system-packages` if needed.

> **First run note:** Whisper downloads the `medium` model (~1.4 GB) automatically.

| Package | Purpose |
|---------|---------|
| `google-genai` | Gemini API (video analysis, image generation) |
| `Pillow` | Image processing |
| `opencv-python` | Video reading |
| `scenedetect[opencv]` | Camera cut detection |
| `openai-whisper` | Speech-to-text transcription |

### 3. Set Gemini API Key

Get your key from [Google AI Studio](https://aistudio.google.com/apikey):

```bash
export GENAI_API_KEY="your_api_key_here"
```

---

## Quick Start

```bash
# Basic usage
python v2/pipeline.py input_video.mp4 --style pixar

# Specify shot count and generation mode
python v2/pipeline.py input_video.mp4 --style disney --shots 20 --mode consistency

# No shot limit (auto-detects all cuts), skip confirmation prompt
python v2/pipeline.py input_video.mp4 --style lego --shots 100 --yes

# Use Veo 3 for real video generation (instead of static fallback)
python v2/pipeline.py input_video.mp4 --style anime --real-video

# Skip Whisper transcription
python v2/pipeline.py input_video.mp4 --style pixar --no-whisper
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--style` | `disney` | Target visual style (see below) |
| `--shots` | `10` | Max shots to generate |
| `--mode` | `consistency` | Keyframe generation mode |
| `--yes` | off | Auto-confirm when >16 shots detected |
| `--real-video` | off | Use Veo 3 instead of static 3s fallback |
| `--no-whisper` | off | Skip audio transcription |
| `--output-dir` | `.` | Output directory |

---

## Styles

| Style | Description |
|-------|-------------|
| `pixar` | Pixar 3D — warm soft lighting, subsurface skin glow, richly detailed environments |
| `disney` | Disney 3D — vibrant colors, expressive characters, polished CG render |
| `anime` | Japanese anime — clean linework, vivid colors, cinematic composition |
| `japanese_anime` | Manga/anime — dynamic poses, expressive faces, bold outlines |
| `clay` | Claymation — visible clay texture, warm handcrafted look |
| `lego` | LEGO — blocky minifigures, bright primary colors, brick-built environments |
| `family_guy` | American cartoon — flat colors, thick outlines, comedic proportions |
| `realistic` | Photorealistic cinematic — 35mm film look |

---

## Generation Modes

| Mode | Description |
|------|-------------|
| `consistency` | **Default.** Groups shots by scene, generates 2×2 grids for visual consistency, then crops each cell as a keyframe. |
| `default` | Each shot generated independently with character refs. |
| `camera_tree` | Groups shots by camera setup (DAG scheduling). Best for complex multi-angle scenes. |

---

## Smart Caching

The pipeline caches intermediate results to avoid redundant API calls across runs:

| Cache | Location | What it stores |
|-------|----------|----------------|
| Analysis JSON | `{input_dir}/{video_name}_analysis.json` | Shot descriptions, character list, scene IDs |
| Shot clips | `{input_dir}/{video_name}_clips/` | Per-shot video segments |
| 720p resize | `{input_dir}/{video_name}_720p.mp4` | Resized video for faster upload |
| Whisper transcript | `./{video_name}.json` | Audio transcription |

On subsequent runs with the same input video, **Step 1 is skipped entirely** if the analysis JSON exists.

---

## Output Files

| File | Description |
|------|-------------|
| `shot_N.png` | Keyframe for shot N |
| `shot_N_video.mp4` | Video clip for shot N (3s static or Veo 3) |
| `final_output_v2_{timestamp}.mp4` | Final merged video (1280×720) |
| `char_character_NN.png` | Character reference image |
| `design_sheet.png` | Unified character design sheet |
| `inspection_grid.png` | All keyframes in a labeled grid (for review) |

---

## Examples

### Titanic → Pixar 3D

```bash
python v2/pipeline.py example/titannic_720p.mp4 --style pixar --shots 100 --yes
```

- 20 shots detected across 3 scenes (ship interior, day deck, night deck)
- 6 characters identified and given reference images
- Fully consistent Pixar-style output

**Output:** `example/titannic_pixar_output.mp4`

### Avengers → LEGO

```bash
python v2/pipeline.py example/avengers_720p.mp4 --style lego --shots 10 --mode consistency
```

**Analysis cache:** `example/avengers_analysis.json` (reuse without re-uploading)

---

## Architecture

### Video Analysis (Step 1)

1. **SceneDetect** finds all camera cuts (frame-accurate)
2. Shots shorter than 1 second are filtered out
3. Full video uploaded once → Gemini extracts character list with consistent `@character_XX` IDs
4. Each shot clip extracted with `ffmpeg -c copy` (fast, no re-encode)
5. Clips uploaded and analyzed **in parallel** (5 workers) — each clip gets a full schema: `scene_id`, `t2i_prompt`, `i2v_prompt`, camera setup, lighting, dialogue, etc.
6. `scene_id` values normalized across calls (fixes zero-padding inconsistencies like `scene_001` vs `scene_01`)

### Prompt Pipeline (Steps 3→4)

- **Step 3:** Raw content description from Step 1 → Gemini rewrites it into target style language (e.g. "LEGO minifigure with printed torso" instead of "man wearing jacket")
- **Step 4:** `@character_XX` tokens replaced with `"Character N from Image M"`, reference images attached to each API call
- **Consistency mode:** Shots in the same scene share a 2×2 grid generation — all 4 cells generated together forcing style/environment coherence, then cropped individually

### Models

| Task | Model |
|------|-------|
| Video analysis, text generation | `gemini-3.1-flash-lite` |
| Keyframe image generation | `gemini-3.1-flash-image-preview` |
| Character reference images | `imagen-3.0-generate-002` → fallback `gemini-3.1-flash-image-preview` |
| Video generation (real mode) | `veo-3.0-generate-preview` |
