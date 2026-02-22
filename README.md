# Vector-First Cartoon Video Engine

Deterministic cartoon generation from symbolic intent, not pixel diffusion.

```text
DSL -> Rig -> SVG -> Raster -> Video
```

This repository is a production-focused pipeline for controllable 2D cartoon animation, synthetic dataset generation, and motion-model training.

## Why This Stack

Most video generation systems optimize pixels directly. This project keeps a symbolic and geometric intermediate representation.

Benefits:
- Stable character identity (no latent frame drift).
- Fully editable motion parameters.
- Deterministic reproducibility for research and product workflows.
- Clean training data with structured metadata.
- Easy extension with events, props, multi-character logic, and physics.

## Architecture

```text
Script / Node Intention
        ->
Motion DSL
        ->
Pose Timeline (single or multi character)
        ->
Rig Solver
        ->
Renderer
  - stick mode
  - mesh skinning mode
        ->
SVG Frames
        ->
Optional PNG/MP4 export
```

Extended modules:
- `slapstick_events.py`: non-linear motion modifiers (`impact`, `take`, `anticipation`).
- `procedural_props.py`: environment-triggered interactions (`wall`, `trapdoor`, `anvil`).
- `multi_character_orchestrator.py`: synchronized dual/ensemble animation and collision events.
- `physics_hybrid.py`: kinematic timeline + optional fallback/pymunk ragdoll windows.
- `motion_transformer.py`: train/sample Transformer-based pose motion and feed it back into DSL.

## Major Capabilities

- Deterministic keyframe and procedural motion generation.
- Slapstick event system with impulse/reaction modifiers.
- Environment interaction system with prop-derived events.
- Multi-character choreography and duel collisions.
- Vector skinning (`mode mesh`) for modular character assets.
- Motion smear logic for fast-action cartoon aesthetics.
- 2.5D camera and dynamic depth:
  - camera focus/pan/zoom
  - depth-based scale/parallax
  - optional Y-sort
  - impact-triggered camera shake
- Hybrid physics (`off`, `fallback`, `pymunk`, `auto`).
- Batch dataset generation with rich `manifest.jsonl` and per-clip `meta.json`.
- Motion tokenization + Transformer training pipeline.
- Web editor with two preview engines:
  - `server`: FastAPI render roundtrip.
  - `local (Pyodide)`: browser-side Python runtime for near-zero-latency previews.
- PWA-ready web editor assets (`manifest.webmanifest`, `service-worker.js`).

## Repository Layout

```text
.
|-- cartoon_svg_mvp.py                # Renderer core, DSL parser, camera/depth transform
|-- procedural_walk_cycle.py          # Single-character procedural + ai_motion integration
|-- slapstick_events.py               # Non-linear keyframe modifiers
|-- procedural_props.py               # Props + interaction event synthesis
|-- multi_character_orchestrator.py   # Multi-character timelines and interactions
|-- vector_skinning.py                # Bone-based SVG mesh renderer
|-- physics_hybrid.py                 # Hybrid kinematic/physics pass
|-- generate_dataset.py               # Dataset factory (single + multi)
|-- pose_tokenizer.py                 # Pose -> token codebook + corpus export
|-- motion_transformer.py             # Train/sample Transformer for motion tokens
|-- web_editor/
|   |-- server.py                     # FastAPI backend + local-module endpoints
|   |-- requirements.txt
|   `-- static/
|       |-- index.html
|       |-- app.js
|       |-- styles.css
|       |-- manifest.webmanifest
|       `-- service-worker.js
|-- assets/meshes/default/            # Reference mesh asset pack
|-- demo.txt
|-- procedural_demo.txt
`-- multi_character_demo.txt
```

## Requirements

Base:
- Python 3.12+

Optional, depending on features:
- `cairosvg` for PNG rasterization
- `ffmpeg` for MP4 encoding
- `pymunk` for advanced physics mode
- `torch` for `ai_motion` inference/training
- `fastapi`, `uvicorn` for web editor backend

## Installation

```powershell
cd d:\video_generation

python -m pip install --user cairosvg
python -m pip install --user pymunk
python -m pip install --user torch
python -m pip install --user -r web_editor/requirements.txt
```

Check ffmpeg:

```powershell
ffmpeg -version
```

## Quickstart

### 1) Manual keyframes

```powershell
python cartoon_svg_mvp.py --script demo.txt --out out_manual
```

### 2) Procedural single-character scene

```powershell
python procedural_walk_cycle.py --script procedural_demo.txt --out out_proc
```

### 3) Multi-character scene

```powershell
python multi_character_orchestrator.py --script multi_character_demo.txt --out out_multi
```

### 4) Dataset generation

SVG-only:

```powershell
python generate_dataset.py --count 1000 --out dataset_svg --seed 42
```

With camera, mesh, multi, physics and MP4:

```powershell
python generate_dataset.py --count 1000 --out dataset_full --seed 42 --multi-prob 0.5 --mesh-prob 0.6 --mesh-assets "assets/meshes/default/mesh.json" --camera-prob 0.8 --physics-mode auto --physics-prob 0.5 --export-png --export-mp4 --rasterizer auto
```

## DSL Reference

### Global directives

- `canvas <width> <height>`
- `fps <int>`
- `seconds <float>`
- `physics mode=<off|fallback|pymunk|auto> gravity=<f> damping=<0..1> restitution=<0..1> friction=<0..1> impulse_scale=<f> ragdoll_extra=<f> substeps=<int>`
- `camera enabled=<bool> focus=<auto|none|char_id> zoom=<f> pan=<x,y> depth=<bool> depth_min=<f> depth_max=<f> parallax=<f> y_sort=<bool> shake_on_impact=<bool> shake_amp=<f> shake_freq=<f> shake_decay=<f>`

### Character style

Single:

```text
character seed "INK-A" line 7 limb 10 head 28 jitter 1.0 mode stick
```

Mesh mode:

```text
character seed "INK-MESH" line 7 limb 10 head 28 jitter 1.0 mode mesh mesh "assets/meshes/default/mesh.json" tint "#2f2f2f"
```

Multi:

```text
character id=char1 seed "A" mode stick
character id=char2 seed "B" mode mesh mesh "assets/meshes/default/mesh.json"
```

### Motion commands

- Manual keyframes:

```text
pose t=0.0 root=200,330 lh=-70,-120 rh=70,-120 lf=-30,90 rf=30,90
```

- Procedural:
  - `walk from=x,y to=x,y speed=<f> bounce=<f> stride=<f> step_height=<f> cadence=<f>`
  - `wave hand=<left|right> cycles=<f> amplitude=<f> start=<f> duration=<f>`

- AI-driven motion:
  - `ai_motion model="path/to/model.pt" tokenizer_model="path/to/pose_tokenizer_model.json" start=x,y target=x,y steps=<int> temperature=<f> top_k=<int> seed=<int> prompt="..." style="..."`

Multi-character variants:
- `char1: walk ...`
- `char2: chase target=char1 ...`
- `char1: ai_motion ...`
- `char2: wave ...`

### Events and props

Slapstick events:
- `impact t=<f> direction=<x,y> force=<f> duration=<f>`
- `take t=<f> intensity=<f> hold=<int>`
- `anticipation t=<f> action=<name> intensity=<f> duration=<f> direction=<x,y>`

Environment props:
- `wall x=<f> width=<f> height=<f> force=<f> duration=<f>`
- `trapdoor x=<f> width=<f> depth=<f> force=<f> duration=<f> open_time=<f>`
- `anvil x=<f> size=<f> trigger_x=<f> trigger_radius=<f> delay=<f> fall_speed=<f> force=<f> duration=<f>`

Multi-agent collision:
- `duel_collision distance=<f> force=<f> duration=<f> take_intensity=<f> cooldown=<f>`

## 2.5D Camera and Dynamic Depth

The camera system is integrated directly in the render path.

What it does:
- transforms rig and prop geometry into camera space,
- scales by virtual depth,
- optionally reorders by Y for stronger depth illusion,
- applies procedural shake around impact times.

This keeps output in SVG while adding cinematic motion language without full 3D.

## Web Editor (Server + Local Pyodide Engine)

Run backend:

```powershell
python -m pip install --user -r web_editor/requirements.txt
uvicorn web_editor.server:app --host 127.0.0.1 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000`

Preview engines:
- `server`: backend API `/api/preview` renders frame SVG.
- `local`: loads Pyodide in browser and runs renderer modules client-side.

Backend endpoints for local engine bootstrap:
- `GET /py-src/{name}`
- `GET /mesh-assets/{asset_path}`

Notes:
- Local mode still fetches code/assets from your own backend origin.
- First Pyodide load needs network access to the Pyodide CDN package.
- After caching, preview interaction is significantly more responsive.

## Motion AI Pipeline

### Step 1: build tokenized motion corpus

```powershell
python pose_tokenizer.py --dataset dataset_full --out tokenizer_out --codebook-size 256 --iterations 20 --sample-size 50000 --write-corpus
```

Outputs:
- `tokenizer_out/pose_tokenizer_model.json`
- `tokenizer_out/tokens_manifest.jsonl`
- optional `tokenizer_out/token_corpus.txt`

### Step 2: train Transformer

```powershell
python motion_transformer.py train --tokenizer-manifest tokenizer_out/tokens_manifest.jsonl --tokenizer-model tokenizer_out/pose_tokenizer_model.json --out models/motion_model.pt --epochs 12 --batch-size 24
```

### Step 3: generate an `ai_motion` script seed

```powershell
python motion_transformer.py sample --model models/motion_model.pt --tokenizer-model tokenizer_out/pose_tokenizer_model.json --out-script ai_motion_demo.txt --style "drunk_walk" --prompt "wobbly but fast"
```

Then render it with `procedural_walk_cycle.py` or integrate it in multi-character scripts.

## Dataset Output Contract

Each clip folder:

```text
clip_00000/
|-- scene.txt
|-- meta.json
|-- svg/frame_0000.svg
|-- ...
|-- png/frame_0000.png      # optional
`-- clip.mp4                # optional
```

Global manifest:
- `<out>/manifest.jsonl`

Metadata includes:
- scene mode, timing, dimensions,
- character style/motion specs,
- scripted + derived events,
- physics and camera config,
- output paths.

## Preparing a Clean GitHub Repository

Use these rules for upload-ready repositories:
- keep source, config, assets, docs,
- exclude generated outputs (`out*`, `dataset*`, `tokenizer*`, `*.mp4`, rendered frames),
- exclude runtime caches (`__pycache__`, temporary files).

The provided `.gitignore` in `GitHub_Repo/` is configured for this workflow.

## Troubleshooting

`RuntimeError: PyTorch required. Install torch to use motion_transformer.`
- Install torch before using `ai_motion` or training commands.

`no SVG rasterizer available`
- Install `cairosvg` or use another supported rasterizer backend.

`physics mode 'pymunk' requested but pymunk is not installed`
- Install `pymunk` or switch to `--physics-mode fallback` / `auto`.

Web editor does not start:
- verify `fastapi` and `uvicorn` are installed,
- confirm port `8000` is free.

Local Pyodide preview fails:
- switch engine to `server` for fallback,
- check browser console/network for blocked CDN or script fetch errors.

## FAQ

**1) Is this diffusion-based video generation?**
No. It is a geometry-first deterministic pipeline with optional learned motion.

**2) Why is this better for character consistency?**
Identity is defined by rig topology and style parameters, not per-frame pixel sampling.

**3) Do I need MP4 export for training?**
Not necessarily. SVG + metadata is often enough for geometry-aware training.

**4) Is `ai_motion` mandatory?**
No. Procedural and manual DSL motion are first-class paths.

**5) Does `ai_motion` run without torch?**
No. Torch is required for model loading/inference.

**6) Can I use my own SVG character design?**
Yes, through mesh assets in `vector_skinning.py` (`mesh.json` + part SVG files).

**7) Does camera mode require 3D rendering?**
No. It is a 2.5D transform layer on top of 2D vectors.

**8) Can multiple characters share the same world events?**
Yes. Collision and prop/event logic can affect several agents in one scene.

**9) Is the web editor required for generation?**
No. CLI remains the primary and complete workflow.

**10) Can this be shipped as a SaaS foundation?**
Yes. The stack is designed for deterministic rendering, metadata-rich datasets, and scalable generation services.
