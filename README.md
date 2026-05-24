# Soap2Soap

**Video-to-Video generation powered by Google Gemini + Seed Dance 2.0.** Transform any video into a fully stylized animated version — Pixar, Disney, LEGO, anime, clay, and more — with consistent characters, environments, and cinematic composition preserved across every shot.

---

## How It Works

```
Input Video
    ↓
Step 0  Whisper audio transcription (dialogue + timestamps)
    ↓
Step 1  Sliding-window Gemini analysis (~60s chunks, parallel)
        (character extraction + per-shot schema with scene_id, t2i/i2v prompts, dialogue)
    ↓
Step 2  Character reference images (Imagen 3 / Gemini) + unified Design Sheet
    ↓
Step 3  Prompt compilation + Gemini style rewrite (LEGO, Pixar, Disney, etc.)
        Dialogue language unified: --dialogue-lang zh|en|auto
    ↓
Step 4  Keyframe generation — Consistency mode: 2×2 grid per scene group → crop + refine
    ↓
Step 4b Keyframe inspection & auto-fix (skippable with --no-inspect)
    ↓
Step 5  Video clips — Seed Dance 2.0 (default) or Veo 3
    ↓
Step 6  ffmpeg merge → final video (1280×720)
```

---

## Setup

### 1. Prerequisites

- **Python 3.10+**
- **ffmpeg**
  ```bash
  brew install ffmpeg        # macOS
  sudo apt install ffmpeg    # Ubuntu
  ```

### 2. Install Python Dependencies

```bash
pip install google-genai Pillow opencv-python scenedetect[opencv] openai-whisper byteplus-python-sdk-v2 httpx PyJWT
```

> **macOS (Homebrew Python):** add `--break-system-packages` if needed.

| Package | Purpose |
|---------|---------|
| `google-genai` | Gemini API (video analysis, image generation, Veo 3) |
| `Pillow` | Image processing |
| `opencv-python` | Video reading |
| `scenedetect[opencv]` | Camera cut detection |
| `openai-whisper` | Speech-to-text transcription |
| `byteplus-python-sdk-v2` | Seed Dance 2.0 (BytePlus ARK) |
| `httpx` | Async HTTP (video download) |
| `PyJWT` | JWT auth (not needed for Seed Dance, reserved) |

### 3. Set API Keys

**Gemini (required for analysis + keyframes):**
```bash
export GENAI_API_KEY="your_gemini_api_key"
```

**Seed Dance 2.0 (required for real video generation, default):**
```bash
export BYTEPLUS_API_KEY="ark-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
```
Get your key from [BytePlus ARK](https://ark.ap-southeast.bytepluses.com/).

**Veo 3 (optional alternative video model):**
```bash
export GENAI_API_KEY="your_gemini_api_key"   # same key as above
```

**Runware (optional — for GPT Image 2 keyframe generation):**
```bash
export RUNWARE_API_KEY="your_runware_api_key"
```
Get your key from [Runware](https://runware.ai/). Used when `--keyframe-model gpt-image` is set.

---

## Quick Start

```bash
# Seed Dance 2.0 (default), clay style, Chinese dialogue
python v2/pipeline.py my_video.mp4 --style clay --real-video --dialogue-lang zh

# With source-frame layout reference (垫图) — preserves original compositions
python v2/pipeline.py my_video.mp4 --style clay --real-video --source-frame-grid

# GPT Image 2 (Runware) for keyframes
RUNWARE_API_KEY=xxx python v2/pipeline.py my_video.mp4 --style pixar --keyframe-model gpt-image --real-video

# Veo 3 alternative
python v2/pipeline.py my_video.mp4 --style pixar --real-video --video-model veo

# Dev mode (fast, no API cost for video)
python v2/pipeline.py my_video.mp4 --style disney

# Full options
python v2/pipeline.py my_video.mp4 \
  --style clay \
  --shots 100 \
  --mode consistency \
  --real-video \
  --video-model seeddance \
  --dialogue-lang zh \
  --output-dir ./output \
  --yes \
  --no-inspect
```

### All CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--style` | `disney` | Target visual style |
| `--shots` | `10` | Max shots to generate |
| `--mode` | `consistency` | Keyframe generation mode |
| `--keyframe-model` | `gemini` | Image model: `gemini` (default) or `gpt-image` (Runware GPT Image 2) |
| `--source-frame-grid` | off | **垫图**: extract midpoint frame from each source-video shot and compose a 2×2 reference grid to guide keyframe layout. Unlocks concurrent grid generation. |
| `--real-video` | off | Use real video model (Seed Dance / Veo 3) |
| `--video-model` | `seeddance` | Video model: `seeddance` or `veo` |
| `--dialogue-lang` | `auto` | Dialogue language in i2v prompt: `auto`, `zh`, `en` |
| `--yes` | off | Auto-confirm when >16 shots detected |
| `--no-whisper` | off | Skip audio transcription |
| `--no-inspect` | off | Skip Step 4b keyframe inspection |
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

## Video Models

| Model | Flag | Notes |
|-------|------|-------|
| **Seed Dance 2.0** | `--video-model seeddance` | **Default.** BytePlus ARK `seedance-1-5-pro-251215`. High quality, ~7-17MB per clip. Requires `BYTEPLUS_API_KEY`. |
| **Veo 3** | `--video-model veo` | Google `veo-3.0-generate-001`. Requires `GENAI_API_KEY`. Subject to Google IP/safety filters. |

---

## Generation Modes

| Mode | Description |
|------|-------------|
| `consistency` | **Default.** Groups shots by scene, generates 2×2 grids for visual consistency, then crops + refines each cell. |
| `default` | Each shot generated independently with character refs. |
| `camera_tree` | Groups shots by camera setup (DAG scheduling). Best for complex multi-angle scenes. |

---

## Smart Caching

All intermediate results are cached — rerunning with the same input skips completed steps:

| Cache | Location |
|-------|----------|
| Analysis JSON | `{output_dir}/input_720p_analysis.json` |
| Character images | `{output_dir}/char_character_NN.png` |
| Design sheet | `{output_dir}/design_sheet.png` |
| Keyframes | `{output_dir}/shot_N.png` |
| Video clips | `{output_dir}/shot_N_video.mp4` |

---

## Examples

Two complete examples with all intermediate files (analysis JSON, character refs, keyframes, video clips, final output) are included:

### 华强买瓜 — Clay Style

```bash
python v2/pipeline.py example/huaqiang_watermelon/input_720p.mp4 \
  --style clay --shots 100 --mode consistency --yes --real-video \
  --video-model seeddance --dialogue-lang zh \
  --output-dir example/huaqiang_watermelon
```

- 17 shots, 6 characters, all Chinese dialogue preserved
- Final video: `example/huaqiang_watermelon/final_output.mp4`

### Titanic — Pixar Style

```bash
python v2/pipeline.py example/titanic/input_720p.mp4 \
  --style pixar --shots 100 --mode consistency --yes --real-video \
  --video-model seeddance --dialogue-lang en \
  --output-dir example/titanic
```

- 12 shots, 5 characters, English dialogue
- Final video: `example/titanic/final_output.mp4`

---

## Architecture

### Video Analysis (Step 1)

1. Input video resized to 720p (cached)
2. Whisper transcribes audio with timestamps
3. Video split into ~60s chunks, each analyzed by Gemini **in parallel**
4. Each chunk returns ~10 narrative shots (not one per camera cut)
5. Shots merged and scene IDs normalized across chunks
6. Analysis cached as `input_720p_analysis.json` for reuse

### Prompt Pipeline (Step 3)

- Raw scene descriptions → Gemini rewrites into target style language
- Dialogue preserved separately (not rewritten): `--dialogue-lang zh|en|auto`
- `@character_XX` tokens replaced with `"Character N from Image M"` at generation time

### Keyframe Generation (Step 4 — Consistency Mode)

1. Shots grouped by `scene_id` into batches of 4
2. **Reference images per batch** — only the characters appearing in that batch's shots (smart per-grid assignment):
   - Default mode: previous grid(s) from the same scene act as the layout reference (grids run sequentially due to this chain)
   - **Source-frame-grid mode** (`--source-frame-grid`): for each batch, extract the midpoint frame of each shot from the original video and compose them into a 2×2 grid. This becomes the layout reference instead of previous AI grids — and since there's no inter-grid dependency, **all grids run concurrently**.
3. Grid prompts >4000c are auto-compressed by Gemini before generation
4. Gemini generates a 2×2 grid image for the batch
5. Grid cropped into 4 cells → each cell refined with per-character reference images (refinement runs concurrently, up to 10 workers)
6. Result: visually consistent keyframes across all shots in a scene

### I2V Prompt Pipeline (Step 5)

- `@character_XX` tokens replaced with natural-language aliases (e.g. `the man in black zip-up light jacket`) before being sent to the I2V model
- Dialogue speakers use role labels directly from `analysis.json` (e.g. `华强`, `瓜贩1`, `Jack`, `Rose`)
- **Off-screen voiceover** (speaker IDs `旁白`, `背景音`, `街边群众`) gets a special "no visible character speaks" instruction so the I2V model doesn't lip-sync them to on-screen characters
- I2V runs concurrently — up to 17 workers in parallel
