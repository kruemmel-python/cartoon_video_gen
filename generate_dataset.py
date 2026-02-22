#!/usr/bin/env python3
"""
generate_dataset.py

Batch dataset generator for the procedural cartoon pipeline.

It produces many short procedural clips:
  random style + random walk/chase + optional wave + optional environment props
  + optional slapstick events
  -> SVG frames + metadata.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from cartoon_svg_mvp import CharacterStyle, ScriptConfig, Vec2, render_frames, render_multi_frames
from multi_character_orchestrator import (
    CharacterProgram,
    ChaseCommand,
    DuelCollisionConfig,
    WalkCommand as MultiWalkCommand,
    WaveCommand as MultiWaveCommand,
    generate_multi_character_scene_detailed,
)
from procedural_walk_cycle import (
    WalkCommand as SingleWalkCommand,
    WaveCommand as SingleWaveCommand,
    generate_procedural_scene_detailed,
    infer_duration,
)
from procedural_props import (
    AnvilProp,
    SceneProp,
    TrapDoorProp,
    WallProp,
    build_prop_items_fn,
    prop_to_meta,
    prop_to_scene_line,
)
from slapstick_events import AnticipationEvent, ImpactEvent, SlapstickEvent, TakeEvent


CLIP_DIR_RE = re.compile(r"^clip_(\d{5})$")
RASTERIZER_CHOICES = ("auto", "cairosvg", "rsvg-convert", "inkscape", "magick", "ffmpeg-svg")
PHYSICS_MODE_CHOICES = ("off", "fallback", "pymunk", "auto")


def fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def vec2_str(v: Vec2) -> str:
    return f"{fmt(v.x)},{fmt(v.y)}"


def discover_mesh_assets(spec: str) -> list[str]:
    """
    Resolve a comma-separated mesh asset spec into concrete `mesh.json` file paths.
    Supports:
    - direct file paths
    - directories containing `mesh.json`
    - glob patterns
    """
    if not spec.strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for token in [p.strip() for p in spec.split(",") if p.strip()]:
        p = Path(token)
        candidates: list[Path] = []
        if any(ch in token for ch in ("*", "?")):
            candidates.extend(sorted(Path().glob(token)))
        elif p.is_dir():
            mesh_file = p / "mesh.json"
            if mesh_file.exists():
                candidates.append(mesh_file)
        elif p.exists():
            candidates.append(p)

        for c in candidates:
            resolved = str(c.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            out.append(resolved)
    return out


def random_mesh_tint(rng: random.Random) -> str:
    palette = [
        "#1b1b1b",
        "#262626",
        "#2f2f2f",
        "#3a3a3a",
        "#404040",
        "#1d2a44",
        "#2b3f66",
        "#4a2f2a",
        "#35553a",
    ]
    return rng.choice(palette)


def choose_clip_physics_mode(
    rng: random.Random,
    *,
    requested_mode: str,
    probability: float,
) -> str:
    mode = requested_mode.lower().strip()
    if mode == "off":
        return "off"
    if rng.random() > clamp01(probability):
        return "off"
    return mode


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def camera_to_meta(cfg: ScriptConfig) -> dict:
    cam = cfg.camera
    return {
        "enabled": cam.enabled,
        "focus": cam.focus,
        "zoom": cam.zoom,
        "pan": asdict(cam.pan),
        "depth_enabled": cam.depth_enabled,
        "depth_min_scale": cam.depth_min_scale,
        "depth_max_scale": cam.depth_max_scale,
        "parallax_strength": cam.parallax_strength,
        "y_sort": cam.y_sort,
        "shake_on_impact": cam.shake_on_impact,
        "shake_amplitude": cam.shake_amplitude,
        "shake_frequency": cam.shake_frequency,
        "shake_decay": cam.shake_decay,
    }


def randomize_camera(
    cfg: ScriptConfig,
    *,
    rng: random.Random,
    probability: float,
    multi: bool,
    character_order: list[str] | None = None,
) -> None:
    if rng.random() > clamp01(probability):
        cfg.camera.enabled = False
        return

    cfg.camera.enabled = True
    cfg.camera.zoom = rng.uniform(0.95, 1.25)
    cfg.camera.pan = Vec2(rng.uniform(-30.0, 30.0), rng.uniform(-12.0, 16.0))
    cfg.camera.depth_enabled = True
    cfg.camera.depth_min_scale = rng.uniform(0.76, 0.9)
    cfg.camera.depth_max_scale = rng.uniform(1.08, 1.28)
    cfg.camera.parallax_strength = rng.uniform(0.10, 0.34)
    cfg.camera.y_sort = True
    cfg.camera.shake_on_impact = True
    cfg.camera.shake_amplitude = rng.uniform(8.0, 18.0)
    cfg.camera.shake_frequency = rng.uniform(10.0, 18.0)
    cfg.camera.shake_decay = rng.uniform(3.8, 6.3)
    if multi and character_order:
        cfg.camera.focus = rng.choice(["auto", "centroid", *character_order])
    else:
        cfg.camera.focus = rng.choice(["", "self", "char1"])


def next_clip_index(out_dir: Path) -> int:
    max_idx = -1
    if not out_dir.exists():
        return 0
    for item in out_dir.iterdir():
        if not item.is_dir():
            continue
        m = CLIP_DIR_RE.match(item.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def run_cmd(cmd: list[str], context: str) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context}: command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise RuntimeError(f"{context}: command failed: {' '.join(cmd)}\n{stderr}") from exc


def ffmpeg_has_svg_decoder(ffmpeg_bin: str) -> bool:
    try:
        p = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-decoders"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return False

    for line in p.stdout.splitlines():
        stripped = line.strip()
        # Decoder list line format usually starts with flags like: "V....D svg ..."
        if not stripped:
            continue
        if " svg " in f" {stripped} " and stripped[0] in {"V", "A", "S"}:
            return True
    return False


def choose_rasterizer(requested: str, ffmpeg_bin: str) -> str:
    available: dict[str, bool] = {
        "cairosvg": importlib.util.find_spec("cairosvg") is not None,
        "rsvg-convert": shutil.which("rsvg-convert") is not None,
        "inkscape": shutil.which("inkscape") is not None,
        "magick": shutil.which("magick") is not None,
        "ffmpeg-svg": ffmpeg_has_svg_decoder(ffmpeg_bin),
    }

    if requested != "auto":
        if available.get(requested, False):
            return requested
        raise RuntimeError(
            f"requested rasterizer '{requested}' is not available. "
            f"Available: {[k for k, v in available.items() if v]}"
        )

    for name in ("cairosvg", "rsvg-convert", "inkscape", "magick", "ffmpeg-svg"):
        if available[name]:
            return name

    raise RuntimeError(
        "no SVG rasterizer available. Install one of:\n"
        "  - pip install cairosvg\n"
        "  - rsvg-convert (librsvg)\n"
        "  - Inkscape\n"
        "  - ImageMagick (magick)\n"
        "  - ffmpeg build with SVG decoder"
    )


def rasterize_svg_to_png(
    *,
    svg_path: Path,
    png_path: Path,
    width: int,
    height: int,
    rasterizer: str,
    ffmpeg_bin: str,
) -> None:
    if rasterizer == "cairosvg":
        import cairosvg  # type: ignore

        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(png_path),
            output_width=width,
            output_height=height,
        )
        return

    if rasterizer == "rsvg-convert":
        run_cmd(
            [
                "rsvg-convert",
                "-w",
                str(width),
                "-h",
                str(height),
                "-o",
                str(png_path),
                str(svg_path),
            ],
            context=f"rasterize {svg_path.name}",
        )
        return

    if rasterizer == "inkscape":
        run_cmd(
            [
                "inkscape",
                str(svg_path),
                "--export-type=png",
                f"--export-filename={png_path}",
                f"--export-width={width}",
                f"--export-height={height}",
            ],
            context=f"rasterize {svg_path.name}",
        )
        return

    if rasterizer == "magick":
        run_cmd(
            [
                "magick",
                str(svg_path),
                "-resize",
                f"{width}x{height}!",
                str(png_path),
            ],
            context=f"rasterize {svg_path.name}",
        )
        return

    if rasterizer == "ffmpeg-svg":
        run_cmd(
            [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(svg_path),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:{height}",
                str(png_path),
            ],
            context=f"rasterize {svg_path.name}",
        )
        return

    raise RuntimeError(f"unknown rasterizer: {rasterizer}")


def rasterize_svg_dir(
    *,
    svg_dir: Path,
    png_dir: Path,
    width: int,
    height: int,
    rasterizer: str,
    ffmpeg_bin: str,
) -> int:
    svg_frames = sorted(svg_dir.glob("frame_*.svg"))
    if not svg_frames:
        raise RuntimeError(f"no SVG frames found in {svg_dir}")

    png_dir.mkdir(parents=True, exist_ok=True)
    for svg in svg_frames:
        png = png_dir / f"{svg.stem}.png"
        rasterize_svg_to_png(
            svg_path=svg,
            png_path=png,
            width=width,
            height=height,
            rasterizer=rasterizer,
            ffmpeg_bin=ffmpeg_bin,
        )
    return len(svg_frames)


def encode_mp4_from_png(
    *,
    png_dir: Path,
    mp4_path: Path,
    fps: int,
    ffmpeg_bin: str,
) -> None:
    pattern = str((png_dir / "frame_%04d.png"))
    run_cmd(
        [
            ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(mp4_path),
        ],
        context=f"encode mp4 {mp4_path.name}",
    )


def random_style(
    rng: random.Random,
    clip_idx: int,
    *,
    mesh_assets: list[str],
    mesh_prob: float,
) -> dict[str, float | str]:
    use_mesh = bool(mesh_assets) and rng.random() < clamp01(mesh_prob)
    mesh_asset = rng.choice(mesh_assets) if use_mesh else ""
    mesh_tint = random_mesh_tint(rng) if use_mesh and rng.random() < 0.7 else ""
    return {
        "seed": f"DATASET-{clip_idx:05d}-{rng.randrange(1_000_000_000)}",
        "line_width": rng.uniform(6.0, 8.5),
        "limb_width": rng.uniform(8.5, 11.5),
        "head_radius": rng.uniform(24.0, 31.0),
        "jitter_amplitude": rng.uniform(0.8, 1.5),
        "smear_speed_threshold": rng.uniform(110.0, 180.0),
        "smear_speed_full": rng.uniform(360.0, 560.0),
        "smear_max_stretch": rng.uniform(0.30, 0.58),
        "smear_max_squeeze": rng.uniform(0.16, 0.34),
        "smear_jitter_boost": rng.uniform(0.65, 1.45),
        "render_mode": "mesh" if use_mesh else "stick",
        "mesh_asset": mesh_asset,
        "mesh_tint": mesh_tint,
    }


def random_style_for_character(
    rng: random.Random,
    clip_idx: int,
    *,
    char_id: str,
    mesh_assets: list[str],
    mesh_prob: float,
) -> dict[str, float | str]:
    base = random_style(
        rng,
        clip_idx,
        mesh_assets=mesh_assets,
        mesh_prob=mesh_prob,
    )
    base["seed"] = f"{base['seed']}-{char_id}"
    return base


def apply_style_spec(style: CharacterStyle, spec: dict[str, float | str]) -> None:
    style.seed = str(spec["seed"])
    style.line_width = float(spec["line_width"])
    style.limb_width = float(spec["limb_width"])
    style.head_radius = float(spec["head_radius"])
    style.jitter_amplitude = float(spec["jitter_amplitude"])
    style.smear_speed_threshold = float(spec.get("smear_speed_threshold", style.smear_speed_threshold))
    style.smear_speed_full = float(spec.get("smear_speed_full", style.smear_speed_full))
    style.smear_max_stretch = float(spec.get("smear_max_stretch", style.smear_max_stretch))
    style.smear_max_squeeze = float(spec.get("smear_max_squeeze", style.smear_max_squeeze))
    style.smear_jitter_boost = float(spec.get("smear_jitter_boost", style.smear_jitter_boost))
    style.render_mode = str(spec.get("render_mode", style.render_mode))
    style.mesh_asset = str(spec.get("mesh_asset", style.mesh_asset))
    style.mesh_tint = str(spec.get("mesh_tint", style.mesh_tint))
    if style.mesh_asset and style.render_mode.lower() != "mesh":
        style.render_mode = "mesh"


def style_to_meta(style: CharacterStyle) -> dict[str, float | str]:
    return {
        "seed": style.seed,
        "line_width": style.line_width,
        "limb_width": style.limb_width,
        "head_radius": style.head_radius,
        "jitter_amplitude": style.jitter_amplitude,
        "smear_speed_threshold": style.smear_speed_threshold,
        "smear_speed_full": style.smear_speed_full,
        "smear_max_stretch": style.smear_max_stretch,
        "smear_max_squeeze": style.smear_max_squeeze,
        "smear_jitter_boost": style.smear_jitter_boost,
        "render_mode": style.render_mode,
        "mesh_asset": style.mesh_asset,
        "mesh_tint": style.mesh_tint,
    }


def random_walk(rng: random.Random, width: int, height: int) -> SingleWalkCommand:
    margin_x = max(90.0, width * 0.12)
    min_distance = max(150.0, width * 0.30)

    x_a = rng.uniform(margin_x, width - margin_x - min_distance)
    x_b = rng.uniform(x_a + min_distance, width - margin_x)
    if rng.random() < 0.5:
        start_x, end_x = x_a, x_b
    else:
        start_x, end_x = x_b, x_a

    root_y = rng.uniform(height * 0.68, height * 0.78)
    end_y = root_y + rng.uniform(-10.0, 10.0)

    return SingleWalkCommand(
        start=Vec2(start_x, root_y),
        end=Vec2(end_x, end_y),
        speed=rng.uniform(0.75, 1.45),
        bounce=rng.uniform(0.12, 0.55),
        stride=rng.uniform(42.0, 72.0),
        step_height=rng.uniform(12.0, 30.0),
        cadence=rng.uniform(1.25, 2.05),
    )


def random_wave(rng: random.Random, probability: float) -> SingleWaveCommand | None:
    if rng.random() > probability:
        return None
    return SingleWaveCommand(
        hand="left" if rng.random() < 0.5 else "right",
        cycles=rng.uniform(1.0, 4.0),
        amplitude=rng.uniform(16.0, 34.0),
        start=rng.uniform(0.15, 1.1),
        duration=0.0 if rng.random() < 0.75 else rng.uniform(0.6, 1.8),
    )


def random_props(
    rng: random.Random,
    *,
    walk: SingleWalkCommand,
    probability: float,
    max_props: int,
) -> list[SceneProp]:
    if probability <= 0.0 or max_props <= 0:
        return []
    if rng.random() > probability:
        return []

    count = rng.randint(1, max_props)
    min_x = min(walk.start.x, walk.end.x)
    max_x = max(walk.start.x, walk.end.x)
    span = max(20.0, max_x - min_x)
    start_pad = span * 0.12
    end_pad = span * 0.12
    lo = min_x + start_pad
    hi = max_x - end_pad
    if hi <= lo:
        lo = min_x
        hi = max_x

    props: list[SceneProp] = []
    for i in range(count):
        x = rng.uniform(lo, hi) if hi > lo else lo
        kind = rng.choices(
            ["wall", "trapdoor", "anvil"],
            weights=[0.45, 0.30, 0.25],
            k=1,
        )[0]

        if kind == "wall":
            props.append(
                WallProp(
                    id=f"wall_gen_{i:03d}",
                    x=x,
                    width=rng.uniform(14.0, 24.0),
                    height=rng.uniform(110.0, 170.0),
                    force=rng.uniform(0.65, 1.15),
                    duration=rng.uniform(0.16, 0.30),
                )
            )
            continue

        if kind == "trapdoor":
            props.append(
                TrapDoorProp(
                    id=f"trapdoor_gen_{i:03d}",
                    x=x,
                    width=rng.uniform(72.0, 112.0),
                    depth=rng.uniform(56.0, 90.0),
                    force=rng.uniform(0.58, 1.0),
                    duration=rng.uniform(0.18, 0.32),
                    open_time=rng.uniform(0.18, 0.34),
                )
            )
            continue

        props.append(
            AnvilProp(
                id=f"anvil_gen_{i:03d}",
                x=x + rng.uniform(-20.0, 20.0),
                size=rng.uniform(26.0, 40.0),
                trigger_x=x,
                trigger_radius=rng.uniform(32.0, 60.0),
                delay=rng.uniform(0.05, 0.16),
                fall_speed=rng.uniform(290.0, 430.0),
                force=rng.uniform(0.85, 1.3),
                duration=rng.uniform(0.14, 0.28),
            )
        )

    props.sort(key=lambda p: getattr(p, "x", 0.0))
    return props


def random_character_ids(count: int) -> list[str]:
    return [f"char{i + 1}" for i in range(count)]


def random_duel_collision(rng: random.Random, *, char_count: int) -> DuelCollisionConfig:
    return DuelCollisionConfig(
        distance=rng.uniform(44.0, 58.0),
        force=rng.uniform(0.75, 1.25),
        duration=rng.uniform(0.14, 0.30),
        take_intensity=rng.uniform(0.5, 1.0),
        cooldown=rng.uniform(0.22, 0.40) + max(0, char_count - 2) * 0.04,
    )


def random_multi_character_programs(
    rng: random.Random,
    *,
    clip_idx: int,
    width: int,
    height: int,
    char_count: int,
    wave_prob: float,
    event_prob: float,
    event_max: int,
    mesh_assets: list[str],
    mesh_prob: float,
) -> tuple[dict[str, CharacterProgram], SingleWalkCommand, DuelCollisionConfig]:
    if char_count < 2:
        raise ValueError("random_multi_character_programs requires at least 2 characters")

    char_ids = random_character_ids(char_count)
    lead_id = char_ids[0]
    lead_walk_single = random_walk(rng, width=width, height=height)
    lead_move_dir = lead_walk_single.end - lead_walk_single.start
    duration_est = infer_duration(lead_walk_single)
    collision = random_duel_collision(rng, char_count=char_count)

    programs: dict[str, CharacterProgram] = {}

    lead_style = CharacterStyle()
    apply_style_spec(
        lead_style,
        random_style_for_character(
            rng,
            clip_idx,
            char_id=lead_id,
            mesh_assets=mesh_assets,
            mesh_prob=mesh_prob,
        ),
    )
    lead_wave_single = random_wave(rng, probability=wave_prob)
    lead_wave = (
        None
        if lead_wave_single is None
        else MultiWaveCommand(
            hand=lead_wave_single.hand,
            cycles=lead_wave_single.cycles,
            amplitude=lead_wave_single.amplitude,
            start=lead_wave_single.start,
            duration=lead_wave_single.duration,
        )
    )
    lead_events = random_slapstick_events(
        rng,
        duration=duration_est,
        probability=min(1.0, event_prob * 0.55),
        max_events=max(0, min(event_max, 2)),
        move_direction=lead_move_dir,
    )
    programs[lead_id] = CharacterProgram(
        char_id=lead_id,
        style=lead_style,
        walk=MultiWalkCommand(
            start=lead_walk_single.start,
            end=lead_walk_single.end,
            speed=lead_walk_single.speed,
            bounce=lead_walk_single.bounce,
            stride=lead_walk_single.stride,
            step_height=lead_walk_single.step_height,
            cadence=lead_walk_single.cadence,
        ),
        chase=None,
        wave=lead_wave,
        events=list(lead_events),
    )

    for idx, char_id in enumerate(char_ids[1:], start=1):
        style = CharacterStyle()
        apply_style_spec(
            style,
            random_style_for_character(
                rng,
                clip_idx,
                char_id=char_id,
                mesh_assets=mesh_assets,
                mesh_prob=mesh_prob,
            ),
        )

        target_id = lead_id if idx == 1 or rng.random() < 0.65 else char_ids[idx - 1]
        base_offset = rng.uniform(collision.distance * 0.58, collision.distance * 0.96)
        stagger = (idx - 1) * rng.uniform(12.0, 24.0)
        offset = -(base_offset + stagger)
        chase = ChaseCommand(
            target=target_id,
            offset=offset,
            aggression=rng.uniform(1.12, 1.72),
            bounce=rng.uniform(0.18, 0.52),
            stride=rng.uniform(44.0, 72.0),
            step_height=rng.uniform(14.0, 30.0),
            cadence=rng.uniform(1.35, 2.25),
        )

        wave_single = random_wave(rng, probability=wave_prob * 0.35)
        wave = (
            None
            if wave_single is None
            else MultiWaveCommand(
                hand=wave_single.hand,
                cycles=wave_single.cycles,
                amplitude=wave_single.amplitude,
                start=wave_single.start,
                duration=wave_single.duration,
            )
        )

        events = random_slapstick_events(
            rng,
            duration=duration_est,
            probability=min(1.0, event_prob * 0.45),
            max_events=max(0, min(event_max, 2)),
            move_direction=lead_move_dir,
        )

        programs[char_id] = CharacterProgram(
            char_id=char_id,
            style=style,
            walk=None,
            chase=chase,
            wave=wave,
            events=list(events),
        )

    return programs, lead_walk_single, collision


def random_slapstick_events(
    rng: random.Random,
    *,
    duration: float,
    probability: float,
    max_events: int,
    move_direction: Vec2,
) -> list[SlapstickEvent]:
    if probability <= 0.0 or max_events <= 0:
        return []
    if duration < 0.35:
        return []
    if rng.random() > probability:
        return []

    count = rng.randint(1, max_events)
    pad = min(0.22, duration * 0.2)
    t_lo = pad
    t_hi = max(t_lo + 1e-3, duration - pad)
    times = sorted(rng.uniform(t_lo, t_hi) for _ in range(count))

    if move_direction.length() <= 1e-6:
        move_dir = Vec2(1.0, 0.0)
    else:
        move_dir = move_direction.normalized()

    events: list[SlapstickEvent] = []
    for t in times:
        kind = rng.choices(
            ["impact", "take", "anticipation"],
            weights=[0.44, 0.27, 0.29],
            k=1,
        )[0]

        if kind == "impact":
            # Usually push against motion direction to simulate hit/shock.
            base = move_dir * -1.0 if rng.random() < 0.7 else move_dir
            direction = Vec2(base.x + rng.uniform(-0.18, 0.18), base.y + rng.uniform(-0.15, 0.15))
            events.append(
                ImpactEvent(
                    t=t,
                    direction=direction,
                    force=rng.uniform(0.55, 1.25),
                    duration=rng.uniform(0.14, 0.32),
                )
            )
            continue

        if kind == "take":
            events.append(
                TakeEvent(
                    t=t,
                    intensity=rng.uniform(0.55, 1.2),
                    hold_frames=rng.randint(1, 3),
                )
            )
            continue

        # anticipation
        action = rng.choice(["move", "run", "sprint", "jump"])
        direction: Vec2 | None = None
        if rng.random() < 0.35:
            direction = Vec2(
                move_dir.x + rng.uniform(-0.1, 0.1),
                move_dir.y + rng.uniform(-0.1, 0.1),
            )
        events.append(
            AnticipationEvent(
                t=t,
                action=action,
                intensity=rng.uniform(0.6, 1.2),
                duration=rng.uniform(0.16, 0.34),
                direction=direction,
            )
        )

    events.sort(key=lambda e: e.t)
    return events


def event_to_scene_line(event: SlapstickEvent) -> str:
    if isinstance(event, ImpactEvent):
        parts = [f"impact t={fmt(event.t)}"]
        if event.direction is not None:
            parts.append(f"direction={vec2_str(event.direction)}")
        parts.append(f"force={fmt(event.force)}")
        parts.append(f"duration={fmt(event.duration)}")
        return " ".join(parts)

    if isinstance(event, TakeEvent):
        return (
            f"take t={fmt(event.t)} "
            f"intensity={fmt(event.intensity)} "
            f"hold={event.hold_frames}"
        )

    if isinstance(event, AnticipationEvent):
        parts = [
            f"anticipation t={fmt(event.t)}",
            f"action={event.action}",
            f"intensity={fmt(event.intensity)}",
            f"duration={fmt(event.duration)}",
        ]
        if event.direction is not None:
            parts.append(f"direction={vec2_str(event.direction)}")
        return " ".join(parts)

    raise RuntimeError(f"unexpected event type: {type(event)!r}")


def event_to_meta(event: SlapstickEvent) -> dict:
    if isinstance(event, ImpactEvent):
        return {
            "type": "impact",
            "t": event.t,
            "force": event.force,
            "duration": event.duration,
            "direction": None if event.direction is None else asdict(event.direction),
        }
    if isinstance(event, TakeEvent):
        return {
            "type": "take",
            "t": event.t,
            "intensity": event.intensity,
            "hold_frames": event.hold_frames,
        }
    if isinstance(event, AnticipationEvent):
        return {
            "type": "anticipation",
            "t": event.t,
            "action": event.action,
            "intensity": event.intensity,
            "duration": event.duration,
            "direction": None if event.direction is None else asdict(event.direction),
        }
    raise RuntimeError(f"unexpected event type: {type(event)!r}")


def collect_impact_times(events: list[SlapstickEvent]) -> list[float]:
    times = [ev.t for ev in events if isinstance(ev, ImpactEvent)]
    times.sort()
    return times


def event_to_meta_with_character(event: SlapstickEvent, *, char_id: str) -> dict:
    row = event_to_meta(event)
    row["character"] = char_id
    return row


def walk_to_meta(walk: SingleWalkCommand) -> dict:
    return {
        "start": asdict(walk.start),
        "end": asdict(walk.end),
        "speed": walk.speed,
        "bounce": walk.bounce,
        "stride": walk.stride,
        "step_height": walk.step_height,
        "cadence": walk.cadence,
    }


def wave_to_meta(wave: SingleWaveCommand | MultiWaveCommand | None) -> dict | None:
    if wave is None:
        return None
    return {
        "hand": wave.hand,
        "cycles": wave.cycles,
        "amplitude": wave.amplitude,
        "start": wave.start,
        "duration": wave.duration,
    }


def build_multi_scene_text(
    cfg: ScriptConfig,
    programs: dict[str, CharacterProgram],
    props: list[SceneProp],
    collision: DuelCollisionConfig,
) -> str:
    lines: list[str] = [
        "# Auto-generated multi-character procedural scene",
        f"canvas {cfg.width} {cfg.height}",
        f"fps {cfg.fps}",
        f"seconds {fmt(cfg.seconds)}",
        (
            f"physics mode={cfg.physics_mode} "
            f"gravity={fmt(cfg.physics_gravity)} "
            f"damping={fmt(cfg.physics_damping)} "
            f"restitution={fmt(cfg.physics_restitution)} "
            f"friction={fmt(cfg.physics_friction)} "
            f"impulse_scale={fmt(cfg.physics_impulse_scale)} "
            f"ragdoll_extra={fmt(cfg.physics_ragdoll_extra)} "
            f"substeps={cfg.physics_substeps}"
        ),
        (
            f"camera enabled={'true' if cfg.camera.enabled else 'false'} "
            f"focus={cfg.camera.focus or 'none'} "
            f"zoom={fmt(cfg.camera.zoom)} "
            f"pan={vec2_str(cfg.camera.pan)} "
            f"depth={'true' if cfg.camera.depth_enabled else 'false'} "
            f"depth_min={fmt(cfg.camera.depth_min_scale)} "
            f"depth_max={fmt(cfg.camera.depth_max_scale)} "
            f"parallax={fmt(cfg.camera.parallax_strength)} "
            f"y_sort={'true' if cfg.camera.y_sort else 'false'} "
            f"shake_on_impact={'true' if cfg.camera.shake_on_impact else 'false'} "
            f"shake_amp={fmt(cfg.camera.shake_amplitude)} "
            f"shake_freq={fmt(cfg.camera.shake_frequency)} "
            f"shake_decay={fmt(cfg.camera.shake_decay)}"
        ),
    ]

    for char_id in programs:
        style = programs[char_id].style
        style_line = (
            f'character id={char_id} seed "{style.seed}" '
            f"line {fmt(style.line_width)} "
            f"limb {fmt(style.limb_width)} "
            f"head {fmt(style.head_radius)} "
            f"jitter {fmt(style.jitter_amplitude)} "
            f"smear_threshold {fmt(style.smear_speed_threshold)} "
            f"smear_full {fmt(style.smear_speed_full)} "
            f"smear_stretch {fmt(style.smear_max_stretch)} "
            f"smear_squeeze {fmt(style.smear_max_squeeze)} "
            f"smear_jitter {fmt(style.smear_jitter_boost)} "
            f"mode {style.render_mode}"
        )
        if style.mesh_asset:
            style_line += f' mesh "{style.mesh_asset}"'
        if style.mesh_tint:
            style_line += f' tint "{style.mesh_tint}"'
        lines.append(style_line)

    for char_id in programs:
        program = programs[char_id]
        if program.walk is not None:
            walk = program.walk
            lines.append(
                (
                    f"{char_id}: walk from={vec2_str(walk.start)} "
                    f"to={vec2_str(walk.end)} "
                    f"speed={fmt(walk.speed)} "
                    f"bounce={fmt(walk.bounce)} "
                    f"stride={fmt(walk.stride)} "
                    f"step_height={fmt(walk.step_height)} "
                    f"cadence={fmt(walk.cadence)}"
                )
            )
        elif program.ai_motion is not None:
            ai = program.ai_motion
            lines.append(
                (
                    f'{char_id}: ai_motion model="{ai.model}" tokenizer_model="{ai.tokenizer_model}" '
                    f"start={vec2_str(ai.start)} target={vec2_str(ai.target)} "
                    f"steps={ai.steps} temperature={fmt(ai.temperature)} top_k={ai.top_k} seed={ai.seed} "
                    f'prompt="{ai.prompt}" style="{ai.style}"'
                )
            )
        elif program.chase is not None:
            chase = program.chase
            lines.append(
                (
                    f"{char_id}: chase target={chase.target} "
                    f"offset={fmt(chase.offset)} "
                    f"aggression={fmt(chase.aggression)} "
                    f"bounce={fmt(chase.bounce)} "
                    f"stride={fmt(chase.stride)} "
                    f"step_height={fmt(chase.step_height)} "
                    f"cadence={fmt(chase.cadence)}"
                )
            )

        if program.wave is not None:
            wave = program.wave
            lines.append(
                f"{char_id}: wave hand={wave.hand} cycles={fmt(wave.cycles)} "
                f"amplitude={fmt(wave.amplitude)} start={fmt(wave.start)} "
                f"duration={fmt(wave.duration)}"
            )
        for event in program.events:
            lines.append(f"{char_id}: {event_to_scene_line(event)}")

    lines.append(
        f"duel_collision distance={fmt(collision.distance)} "
        f"force={fmt(collision.force)} "
        f"duration={fmt(collision.duration)} "
        f"take_intensity={fmt(collision.take_intensity)} "
        f"cooldown={fmt(collision.cooldown)}"
    )

    for prop in props:
        lines.append(prop_to_scene_line(prop))

    return "\n".join(lines) + "\n"


def build_scene_text(
    cfg: ScriptConfig,
    walk: SingleWalkCommand,
    wave: SingleWaveCommand | None,
    props: list[SceneProp],
    events: list[SlapstickEvent],
) -> str:
    lines: list[str] = [
        "# Auto-generated procedural scene",
        f"canvas {cfg.width} {cfg.height}",
        f"fps {cfg.fps}",
        f"seconds {fmt(cfg.seconds)}",
        (
            f"physics mode={cfg.physics_mode} "
            f"gravity={fmt(cfg.physics_gravity)} "
            f"damping={fmt(cfg.physics_damping)} "
            f"restitution={fmt(cfg.physics_restitution)} "
            f"friction={fmt(cfg.physics_friction)} "
            f"impulse_scale={fmt(cfg.physics_impulse_scale)} "
            f"ragdoll_extra={fmt(cfg.physics_ragdoll_extra)} "
            f"substeps={cfg.physics_substeps}"
        ),
        (
            f"camera enabled={'true' if cfg.camera.enabled else 'false'} "
            f"focus={cfg.camera.focus or 'none'} "
            f"zoom={fmt(cfg.camera.zoom)} "
            f"pan={vec2_str(cfg.camera.pan)} "
            f"depth={'true' if cfg.camera.depth_enabled else 'false'} "
            f"depth_min={fmt(cfg.camera.depth_min_scale)} "
            f"depth_max={fmt(cfg.camera.depth_max_scale)} "
            f"parallax={fmt(cfg.camera.parallax_strength)} "
            f"y_sort={'true' if cfg.camera.y_sort else 'false'} "
            f"shake_on_impact={'true' if cfg.camera.shake_on_impact else 'false'} "
            f"shake_amp={fmt(cfg.camera.shake_amplitude)} "
            f"shake_freq={fmt(cfg.camera.shake_frequency)} "
            f"shake_decay={fmt(cfg.camera.shake_decay)}"
        ),
    ]
    style_line = (
        f'character seed "{cfg.style.seed}" '
        f"line {fmt(cfg.style.line_width)} "
        f"limb {fmt(cfg.style.limb_width)} "
        f"head {fmt(cfg.style.head_radius)} "
        f"jitter {fmt(cfg.style.jitter_amplitude)} "
        f"smear_threshold {fmt(cfg.style.smear_speed_threshold)} "
        f"smear_full {fmt(cfg.style.smear_speed_full)} "
        f"smear_stretch {fmt(cfg.style.smear_max_stretch)} "
        f"smear_squeeze {fmt(cfg.style.smear_max_squeeze)} "
        f"smear_jitter {fmt(cfg.style.smear_jitter_boost)} "
        f"mode {cfg.style.render_mode}"
    )
    if cfg.style.mesh_asset:
        style_line += f' mesh "{cfg.style.mesh_asset}"'
    if cfg.style.mesh_tint:
        style_line += f' tint "{cfg.style.mesh_tint}"'
    lines.append(style_line)
    lines.append(
        (
            f"walk from={vec2_str(walk.start)} "
            f"to={vec2_str(walk.end)} "
            f"speed={fmt(walk.speed)} "
            f"bounce={fmt(walk.bounce)} "
            f"stride={fmt(walk.stride)} "
            f"step_height={fmt(walk.step_height)} "
            f"cadence={fmt(walk.cadence)}"
        )
    )
    if wave is not None:
        lines.append(
            f"wave hand={wave.hand} cycles={fmt(wave.cycles)} "
            f"amplitude={fmt(wave.amplitude)} start={fmt(wave.start)} "
            f"duration={fmt(wave.duration)}"
        )
    for prop in props:
        lines.append(prop_to_scene_line(prop))
    for event in events:
        lines.append(event_to_scene_line(event))
    return "\n".join(lines) + "\n"


def render_single_clip(
    *,
    out_dir: Path,
    clip_idx: int,
    rng: random.Random,
    width: int,
    height: int,
    fps: int,
    wave_prob: float,
    prop_prob: float,
    prop_max: int,
    event_prob: float,
    event_max: int,
    mesh_assets: list[str],
    mesh_prob: float,
    physics_mode: str,
    physics_prob: float,
    physics_gravity: float,
    physics_damping: float,
    physics_restitution: float,
    physics_friction: float,
    physics_impulse_scale: float,
    physics_ragdoll_extra: float,
    physics_substeps: int,
    camera_prob: float,
    export_png: bool,
    export_mp4: bool,
    rasterizer: str,
    ffmpeg_bin: str,
    png_dirname: str,
    mp4_name: str,
) -> dict:
    cfg = ScriptConfig(width=width, height=height, fps=fps, seconds=0.0)
    cfg.physics_mode = choose_clip_physics_mode(
        rng,
        requested_mode=physics_mode,
        probability=physics_prob,
    )
    cfg.physics_gravity = physics_gravity
    cfg.physics_damping = physics_damping
    cfg.physics_restitution = physics_restitution
    cfg.physics_friction = physics_friction
    cfg.physics_impulse_scale = physics_impulse_scale
    cfg.physics_ragdoll_extra = physics_ragdoll_extra
    cfg.physics_substeps = max(1, physics_substeps)
    randomize_camera(
        cfg,
        rng=rng,
        probability=camera_prob,
        multi=False,
        character_order=None,
    )

    style = random_style(
        rng,
        clip_idx,
        mesh_assets=mesh_assets,
        mesh_prob=mesh_prob,
    )
    apply_style_spec(cfg.style, style)

    walk = random_walk(rng, width=width, height=height)
    wave = random_wave(rng, probability=wave_prob)
    props = random_props(
        rng,
        walk=walk,
        probability=prop_prob,
        max_props=prop_max,
    )
    duration_est = cfg.seconds if cfg.seconds > 0.0 else infer_duration(walk)
    events = random_slapstick_events(
        rng,
        duration=duration_est,
        probability=event_prob,
        max_events=event_max,
        move_direction=(walk.end - walk.start),
    )

    tl, prop_runtime, prop_derived_events, physics_meta = generate_procedural_scene_detailed(
        cfg,
        walk,
        wave,
        events=events,
        props=props,
    )
    frame_count = int(round(cfg.seconds * cfg.fps))

    clip_name = f"clip_{clip_idx:05d}"
    clip_dir = out_dir / clip_name
    svg_dir = clip_dir / "svg"
    png_dir = clip_dir / png_dirname
    mp4_path = clip_dir / mp4_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    prop_items_fn = build_prop_items_fn(prop_runtime, cfg)
    camera_impact_times = collect_impact_times([*events, *prop_derived_events])
    render_frames(
        cfg=cfg,
        tl=tl,
        out_dir=svg_dir,
        extra_items_fn=prop_items_fn,
        camera_impact_times=camera_impact_times,
    )

    rasterized_frames = 0
    if export_png or export_mp4:
        rasterized_frames = rasterize_svg_dir(
            svg_dir=svg_dir,
            png_dir=png_dir,
            width=cfg.width,
            height=cfg.height,
            rasterizer=rasterizer,
            ffmpeg_bin=ffmpeg_bin,
        )
    if export_mp4:
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"ffmpeg binary not found: {ffmpeg_bin}")
        encode_mp4_from_png(
            png_dir=png_dir,
            mp4_path=mp4_path,
            fps=cfg.fps,
            ffmpeg_bin=ffmpeg_bin,
        )

    scene_text = build_scene_text(cfg, walk, wave, props, events)
    (clip_dir / "scene.txt").write_text(scene_text, encoding="utf-8")

    meta = {
        "clip": clip_name,
        "mode": "single",
        "character_count": 1,
        "frames": frame_count,
        "fps": cfg.fps,
        "seconds": cfg.seconds,
        "width": cfg.width,
        "height": cfg.height,
        "style": style_to_meta(cfg.style),
        "camera": camera_to_meta(cfg),
        "physics": physics_meta,
        "walk": walk_to_meta(walk),
        "wave": wave_to_meta(wave),
        "characters": [
            {
                "id": "char1",
                "style": style_to_meta(cfg.style),
                "physics": physics_meta,
                "motion": {"type": "walk", **walk_to_meta(walk)},
                "wave": wave_to_meta(wave),
                "events_scripted": [event_to_meta(e) for e in events],
                "events_prop_derived": [event_to_meta(e) for e in prop_derived_events],
                "events_collision_derived": [],
            }
        ],
        "collision": None,
        "props": [prop_to_meta(p) for p in props],
        "events_scripted": [event_to_meta(e) for e in events],
        "events_prop_derived": [event_to_meta(e) for e in prop_derived_events],
        "events_collision_derived": [],
        "scene_file": str((clip_dir / "scene.txt").as_posix()),
        "svg_dir": str(svg_dir.as_posix()),
        "png_dir": str(png_dir.as_posix()) if (export_png or export_mp4) else None,
        "mp4_file": str(mp4_path.as_posix()) if export_mp4 else None,
        "rasterized_frames": rasterized_frames if (export_png or export_mp4) else 0,
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def render_multi_clip(
    *,
    out_dir: Path,
    clip_idx: int,
    rng: random.Random,
    width: int,
    height: int,
    fps: int,
    wave_prob: float,
    prop_prob: float,
    prop_max: int,
    event_prob: float,
    event_max: int,
    mesh_assets: list[str],
    mesh_prob: float,
    physics_mode: str,
    physics_prob: float,
    physics_gravity: float,
    physics_damping: float,
    physics_restitution: float,
    physics_friction: float,
    physics_impulse_scale: float,
    physics_ragdoll_extra: float,
    physics_substeps: int,
    camera_prob: float,
    export_png: bool,
    export_mp4: bool,
    rasterizer: str,
    ffmpeg_bin: str,
    png_dirname: str,
    mp4_name: str,
    multi_min_chars: int,
    multi_max_chars: int,
) -> dict:
    cfg = ScriptConfig(width=width, height=height, fps=fps, seconds=0.0)
    cfg.physics_mode = choose_clip_physics_mode(
        rng,
        requested_mode=physics_mode,
        probability=physics_prob,
    )
    cfg.physics_gravity = physics_gravity
    cfg.physics_damping = physics_damping
    cfg.physics_restitution = physics_restitution
    cfg.physics_friction = physics_friction
    cfg.physics_impulse_scale = physics_impulse_scale
    cfg.physics_ragdoll_extra = physics_ragdoll_extra
    cfg.physics_substeps = max(1, physics_substeps)
    char_count = rng.randint(multi_min_chars, multi_max_chars)
    char_order_hint = random_character_ids(char_count)
    randomize_camera(
        cfg,
        rng=rng,
        probability=camera_prob,
        multi=True,
        character_order=char_order_hint,
    )
    programs, lead_walk, collision = random_multi_character_programs(
        rng,
        clip_idx=clip_idx,
        width=width,
        height=height,
        char_count=char_count,
        wave_prob=wave_prob,
        event_prob=event_prob,
        event_max=event_max,
        mesh_assets=mesh_assets,
        mesh_prob=mesh_prob,
    )
    character_order = list(programs.keys())
    if cfg.camera.focus and cfg.camera.focus not in {"auto", "centroid"} and cfg.camera.focus not in character_order:
        cfg.camera.focus = "auto"
    props = random_props(
        rng,
        walk=lead_walk,
        probability=prop_prob,
        max_props=prop_max,
    )

    timelines, prop_runtime, collision_events, prop_derived_events, physics_meta_by_char = (
        generate_multi_character_scene_detailed(
            cfg=cfg,
            programs=programs,
            props=props,
            collision=collision,
        )
    )
    frame_count = int(round(cfg.seconds * cfg.fps))

    clip_name = f"clip_{clip_idx:05d}"
    clip_dir = out_dir / clip_name
    svg_dir = clip_dir / "svg"
    png_dir = clip_dir / png_dirname
    mp4_path = clip_dir / mp4_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    prop_items_fn = build_prop_items_fn(prop_runtime, cfg)
    camera_impact_times_set: set[float] = set()
    for program in programs.values():
        for ev in program.events:
            if isinstance(ev, ImpactEvent):
                camera_impact_times_set.add(ev.t)
    for rows in prop_derived_events.values():
        for ev in rows:
            if isinstance(ev, ImpactEvent):
                camera_impact_times_set.add(ev.t)
    for rows in collision_events.values():
        for ev in rows:
            if isinstance(ev, ImpactEvent):
                camera_impact_times_set.add(ev.t)
    camera_impact_times = sorted(camera_impact_times_set)
    render_multi_frames(
        cfg=cfg,
        timelines=timelines,
        styles={char_id: programs[char_id].style for char_id in character_order},
        out_dir=svg_dir,
        character_order=character_order,
        extra_items_fn=prop_items_fn,
        camera_impact_times=camera_impact_times,
    )

    rasterized_frames = 0
    if export_png or export_mp4:
        rasterized_frames = rasterize_svg_dir(
            svg_dir=svg_dir,
            png_dir=png_dir,
            width=cfg.width,
            height=cfg.height,
            rasterizer=rasterizer,
            ffmpeg_bin=ffmpeg_bin,
        )
    if export_mp4:
        if shutil.which(ffmpeg_bin) is None:
            raise RuntimeError(f"ffmpeg binary not found: {ffmpeg_bin}")
        encode_mp4_from_png(
            png_dir=png_dir,
            mp4_path=mp4_path,
            fps=cfg.fps,
            ffmpeg_bin=ffmpeg_bin,
        )

    scene_text = build_multi_scene_text(cfg, programs, props, collision)
    (clip_dir / "scene.txt").write_text(scene_text, encoding="utf-8")

    events_scripted_rows: list[dict] = []
    events_prop_rows: list[dict] = []
    events_collision_rows: list[dict] = []
    character_rows: list[dict] = []

    for char_id in character_order:
        program = programs[char_id]
        scripted_events = list(program.events)
        prop_events = list(prop_derived_events.get(char_id, []))
        collision_rows = list(collision_events.get(char_id, []))

        for event in scripted_events:
            events_scripted_rows.append(event_to_meta_with_character(event, char_id=char_id))
        for event in prop_events:
            events_prop_rows.append(event_to_meta_with_character(event, char_id=char_id))
        for event in collision_rows:
            events_collision_rows.append(event_to_meta_with_character(event, char_id=char_id))

        if program.walk is not None:
            motion = {
                "type": "walk",
                "start": asdict(program.walk.start),
                "end": asdict(program.walk.end),
                "speed": program.walk.speed,
                "bounce": program.walk.bounce,
                "stride": program.walk.stride,
                "step_height": program.walk.step_height,
                "cadence": program.walk.cadence,
            }
        elif program.ai_motion is not None:
            motion = {
                "type": "ai_motion",
                "model": program.ai_motion.model,
                "tokenizer_model": program.ai_motion.tokenizer_model,
                "start": asdict(program.ai_motion.start),
                "target": asdict(program.ai_motion.target),
                "steps": program.ai_motion.steps,
                "temperature": program.ai_motion.temperature,
                "top_k": program.ai_motion.top_k,
                "seed": program.ai_motion.seed,
                "prompt": program.ai_motion.prompt,
                "style": program.ai_motion.style,
            }
        elif program.chase is not None:
            motion = {
                "type": "chase",
                "target": program.chase.target,
                "offset": program.chase.offset,
                "aggression": program.chase.aggression,
                "bounce": program.chase.bounce,
                "stride": program.chase.stride,
                "step_height": program.chase.step_height,
                "cadence": program.chase.cadence,
            }
        else:
            motion = {"type": "idle"}

        character_rows.append(
            {
                "id": char_id,
                "style": style_to_meta(program.style),
                "physics": physics_meta_by_char.get(char_id),
                "motion": motion,
                "wave": wave_to_meta(program.wave),
                "events_scripted": [event_to_meta(e) for e in scripted_events],
                "events_prop_derived": [event_to_meta(e) for e in prop_events],
                "events_collision_derived": [event_to_meta(e) for e in collision_rows],
            }
        )

    meta = {
        "clip": clip_name,
        "mode": "multi",
        "character_count": len(character_order),
        "frames": frame_count,
        "fps": cfg.fps,
        "seconds": cfg.seconds,
        "width": cfg.width,
        "height": cfg.height,
        "style": None,
        "camera": camera_to_meta(cfg),
        "physics": {
            "mode_requested": cfg.physics_mode,
            "characters": physics_meta_by_char,
        },
        "walk": None,
        "wave": None,
        "characters": character_rows,
        "collision": {
            "distance": collision.distance,
            "force": collision.force,
            "duration": collision.duration,
            "take_intensity": collision.take_intensity,
            "cooldown": collision.cooldown,
        },
        "props": [prop_to_meta(p) for p in props],
        "events_scripted": events_scripted_rows,
        "events_prop_derived": events_prop_rows,
        "events_collision_derived": events_collision_rows,
        "scene_file": str((clip_dir / "scene.txt").as_posix()),
        "svg_dir": str(svg_dir.as_posix()),
        "png_dir": str(png_dir.as_posix()) if (export_png or export_mp4) else None,
        "mp4_file": str(mp4_path.as_posix()) if export_mp4 else None,
        "rasterized_frames": rasterized_frames if (export_png or export_mp4) else 0,
    }
    (clip_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def render_clip(
    *,
    out_dir: Path,
    clip_idx: int,
    rng: random.Random,
    width: int,
    height: int,
    fps: int,
    wave_prob: float,
    prop_prob: float,
    prop_max: int,
    event_prob: float,
    event_max: int,
    mesh_assets: list[str],
    mesh_prob: float,
    physics_mode: str,
    physics_prob: float,
    physics_gravity: float,
    physics_damping: float,
    physics_restitution: float,
    physics_friction: float,
    physics_impulse_scale: float,
    physics_ragdoll_extra: float,
    physics_substeps: int,
    camera_prob: float,
    export_png: bool,
    export_mp4: bool,
    rasterizer: str,
    ffmpeg_bin: str,
    png_dirname: str,
    mp4_name: str,
    multi_prob: float,
    multi_min_chars: int,
    multi_max_chars: int,
) -> dict:
    if rng.random() < multi_prob:
        return render_multi_clip(
            out_dir=out_dir,
            clip_idx=clip_idx,
            rng=rng,
            width=width,
            height=height,
            fps=fps,
            wave_prob=wave_prob,
            prop_prob=prop_prob,
            prop_max=prop_max,
            event_prob=event_prob,
            event_max=event_max,
            mesh_assets=mesh_assets,
            mesh_prob=mesh_prob,
            physics_mode=physics_mode,
            physics_prob=physics_prob,
            physics_gravity=physics_gravity,
            physics_damping=physics_damping,
            physics_restitution=physics_restitution,
            physics_friction=physics_friction,
            physics_impulse_scale=physics_impulse_scale,
            physics_ragdoll_extra=physics_ragdoll_extra,
            physics_substeps=physics_substeps,
            camera_prob=camera_prob,
            export_png=export_png,
            export_mp4=export_mp4,
            rasterizer=rasterizer,
            ffmpeg_bin=ffmpeg_bin,
            png_dirname=png_dirname,
            mp4_name=mp4_name,
            multi_min_chars=multi_min_chars,
            multi_max_chars=multi_max_chars,
        )
    return render_single_clip(
        out_dir=out_dir,
        clip_idx=clip_idx,
        rng=rng,
        width=width,
        height=height,
        fps=fps,
        wave_prob=wave_prob,
        prop_prob=prop_prob,
        prop_max=prop_max,
        event_prob=event_prob,
        event_max=event_max,
        mesh_assets=mesh_assets,
        mesh_prob=mesh_prob,
        physics_mode=physics_mode,
        physics_prob=physics_prob,
        physics_gravity=physics_gravity,
        physics_damping=physics_damping,
        physics_restitution=physics_restitution,
        physics_friction=physics_friction,
        physics_impulse_scale=physics_impulse_scale,
        physics_ragdoll_extra=physics_ragdoll_extra,
        physics_substeps=physics_substeps,
        camera_prob=camera_prob,
        export_png=export_png,
        export_mp4=export_mp4,
        rasterizer=rasterizer,
        ffmpeg_bin=ffmpeg_bin,
        png_dirname=png_dirname,
        mp4_name=mp4_name,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a procedural cartoon dataset (SVG frame sequences).")
    ap.add_argument("--count", type=int, required=True, help="Number of clips to generate.")
    ap.add_argument("--out", type=str, default="dataset", help="Dataset output directory.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--fps", type=int, default=24, help="FPS per clip.")
    ap.add_argument("--width", type=int, default=800, help="Canvas width.")
    ap.add_argument("--height", type=int, default=450, help="Canvas height.")
    ap.add_argument("--wave-prob", type=float, default=0.55, help="Probability for adding a wave action.")
    ap.add_argument("--prop-prob", type=float, default=0.5, help="Probability for adding environment props per clip.")
    ap.add_argument("--prop-max", type=int, default=2, help="Maximum number of props per clip.")
    ap.add_argument("--event-prob", type=float, default=0.45, help="Probability for adding slapstick events per clip.")
    ap.add_argument("--event-max", type=int, default=3, help="Maximum number of events per clip.")
    ap.add_argument(
        "--mesh-prob",
        type=float,
        default=0.0,
        help="Probability to render a character in mesh mode when mesh assets are available.",
    )
    ap.add_argument(
        "--mesh-assets",
        type=str,
        default="assets/meshes/default/mesh.json",
        help="Comma-separated list of mesh assets (file, dir with mesh.json, or glob).",
    )
    ap.add_argument(
        "--physics-mode",
        type=str,
        default="off",
        choices=PHYSICS_MODE_CHOICES,
        help="Hybrid physics mode for clip generation.",
    )
    ap.add_argument(
        "--camera-prob",
        type=float,
        default=0.45,
        help="Probability that a clip enables 2.5D camera/parallax transforms.",
    )
    ap.add_argument(
        "--physics-prob",
        type=float,
        default=0.0,
        help="Probability that a clip enables physics (ignored when --physics-mode=off).",
    )
    ap.add_argument("--physics-gravity", type=float, default=980.0, help="Physics gravity in px/s^2.")
    ap.add_argument("--physics-damping", type=float, default=0.92, help="Physics damping factor in [0,1].")
    ap.add_argument("--physics-restitution", type=float, default=0.22, help="Physics restitution in [0,1].")
    ap.add_argument("--physics-friction", type=float, default=0.85, help="Physics friction in [0,1].")
    ap.add_argument("--physics-impulse-scale", type=float, default=320.0, help="Impulse scale for impact events.")
    ap.add_argument("--physics-ragdoll-extra", type=float, default=0.35, help="Extra ragdoll seconds after impact.")
    ap.add_argument("--physics-substeps", type=int, default=2, help="Physics substeps per rendered frame.")
    ap.add_argument(
        "--multi-prob",
        type=float,
        default=0.0,
        help="Probability that a clip is generated as multi-character scene.",
    )
    ap.add_argument("--multi-min-chars", type=int, default=2, help="Minimum characters for multi-character clips.")
    ap.add_argument("--multi-max-chars", type=int, default=2, help="Maximum characters for multi-character clips.")
    ap.add_argument("--export-png", action="store_true", help="Rasterize SVG frames to PNG per clip.")
    ap.add_argument("--export-mp4", action="store_true", help="Encode MP4 per clip (requires PNG frames).")
    ap.add_argument(
        "--rasterizer",
        type=str,
        default="auto",
        choices=RASTERIZER_CHOICES,
        help="SVG rasterizer backend.",
    )
    ap.add_argument("--png-dirname", type=str, default="png", help="PNG frame directory name inside each clip dir.")
    ap.add_argument("--mp4-name", type=str, default="clip.mp4", help="MP4 file name inside each clip dir.")
    ap.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg binary path/name.")
    ap.add_argument(
        "--start-index",
        type=int,
        default=-1,
        help="First clip index (default: auto-continue after existing clips).",
    )
    args = ap.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.width < 320 or args.height < 180:
        raise ValueError("--width/--height too small")
    wave_prob = min(1.0, max(0.0, args.wave_prob))
    prop_prob = min(1.0, max(0.0, args.prop_prob))
    if args.prop_max < 0:
        raise ValueError("--prop-max must be >= 0")
    event_prob = min(1.0, max(0.0, args.event_prob))
    if args.event_max < 0:
        raise ValueError("--event-max must be >= 0")
    mesh_prob = clamp01(args.mesh_prob)
    mesh_assets = discover_mesh_assets(args.mesh_assets)
    if mesh_prob > 0.0 and not mesh_assets:
        raise ValueError("--mesh-prob > 0 but no valid mesh assets were resolved from --mesh-assets")
    physics_mode = str(args.physics_mode).lower()
    if physics_mode not in PHYSICS_MODE_CHOICES:
        raise ValueError(f"--physics-mode must be one of {PHYSICS_MODE_CHOICES}")
    physics_prob = clamp01(args.physics_prob)
    camera_prob = clamp01(args.camera_prob)
    if args.physics_substeps <= 0:
        raise ValueError("--physics-substeps must be > 0")
    physics_damping = clamp01(args.physics_damping)
    physics_restitution = clamp01(args.physics_restitution)
    physics_friction = clamp01(args.physics_friction)
    if args.physics_impulse_scale < 0.0:
        raise ValueError("--physics-impulse-scale must be >= 0")
    if args.physics_ragdoll_extra < 0.0:
        raise ValueError("--physics-ragdoll-extra must be >= 0")
    multi_prob = min(1.0, max(0.0, args.multi_prob))
    if args.multi_min_chars < 2:
        raise ValueError("--multi-min-chars must be >= 2")
    if args.multi_max_chars < args.multi_min_chars:
        raise ValueError("--multi-max-chars must be >= --multi-min-chars")
    if not args.png_dirname.strip():
        raise ValueError("--png-dirname must not be empty")
    if not args.mp4_name.strip():
        raise ValueError("--mp4-name must not be empty")

    export_png = bool(args.export_png)
    export_mp4 = bool(args.export_mp4)
    do_raster = export_png or export_mp4

    rasterizer = ""
    if do_raster:
        rasterizer = choose_rasterizer(args.rasterizer, ffmpeg_bin=args.ffmpeg_bin)
    if export_mp4 and shutil.which(args.ffmpeg_bin) is None:
        raise RuntimeError(f"ffmpeg binary not found: {args.ffmpeg_bin}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_idx = args.start_index if args.start_index >= 0 else next_clip_index(out_dir)
    rng = random.Random(args.seed)

    manifest_path = out_dir / "manifest.jsonl"
    generated = 0
    total_frames = 0
    generated_multi = 0

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for i in range(args.count):
            clip_idx = start_idx + i
            meta = render_clip(
                out_dir=out_dir,
                clip_idx=clip_idx,
                rng=rng,
                width=args.width,
                height=args.height,
                fps=args.fps,
                wave_prob=wave_prob,
                prop_prob=prop_prob,
                prop_max=args.prop_max,
                event_prob=event_prob,
                event_max=args.event_max,
                mesh_assets=mesh_assets,
                mesh_prob=mesh_prob,
                physics_mode=physics_mode,
                physics_prob=physics_prob,
                physics_gravity=args.physics_gravity,
                physics_damping=physics_damping,
                physics_restitution=physics_restitution,
                physics_friction=physics_friction,
                physics_impulse_scale=args.physics_impulse_scale,
                physics_ragdoll_extra=args.physics_ragdoll_extra,
                physics_substeps=args.physics_substeps,
                camera_prob=camera_prob,
                export_png=export_png,
                export_mp4=export_mp4,
                rasterizer=rasterizer,
                ffmpeg_bin=args.ffmpeg_bin,
                png_dirname=args.png_dirname,
                mp4_name=args.mp4_name,
                multi_prob=multi_prob,
                multi_min_chars=args.multi_min_chars,
                multi_max_chars=args.multi_max_chars,
            )
            manifest.write(json.dumps(meta, ensure_ascii=False) + "\n")

            generated += 1
            total_frames += int(meta["frames"])
            if meta.get("mode") == "multi":
                generated_multi += 1
            if generated % 10 == 0 or generated == args.count:
                print(f"[progress] {generated}/{args.count} clips generated")

    print(f"[ok] dataset generated: {out_dir.resolve()}")
    print(f"[ok] clips={generated} frames={total_frames} manifest={manifest_path.resolve()}")
    print(f"[ok] mode split: single={generated - generated_multi} multi={generated_multi}")
    print(
        f"[ok] style modes: mesh_prob={mesh_prob} mesh_assets={len(mesh_assets)} "
        f"physics_mode={physics_mode} physics_prob={physics_prob} camera_prob={camera_prob}"
    )
    if do_raster:
        print(f"[ok] rasterizer={rasterizer}")
    if export_mp4:
        print(f"[ok] mp4 name per clip: {args.mp4_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
