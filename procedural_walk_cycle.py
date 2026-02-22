#!/usr/bin/env python3
"""
procedural_walk_cycle.py

Procedural walk + optional wave overlay for the SVG-first cartoon pipeline.

This module sits one level above manual keyframes:
Action DSL -> procedural pose synthesis -> Timeline -> existing renderer.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from cartoon_svg_mvp import Pose, ScriptConfig, Timeline, Vec2, render_frames
from physics_hybrid import apply_hybrid_physics
from procedural_props import (
    PropRuntimeState,
    SceneProp,
    build_prop_items_fn,
    derive_prop_events,
    parse_scene_prop_line,
)
from slapstick_events import ImpactEvent, SlapstickEvent, apply_slapstick_events, parse_slapstick_event_line


# -----------------------------
# 1) Command models
# -----------------------------


@dataclass(slots=True)
class WalkCommand:
    start: Vec2
    end: Vec2
    speed: float = 1.0
    bounce: float = 0.3
    stride: float = 54.0
    step_height: float = 22.0
    cadence: float = 1.6


@dataclass(slots=True)
class WaveCommand:
    hand: str = "right"  # "left" or "right"
    cycles: float = 2.0
    amplitude: float = 24.0
    start: float = 0.0
    duration: float = 0.0  # 0 -> auto


@dataclass(slots=True)
class AIMotionCommand:
    model: str
    tokenizer_model: str
    start: Vec2
    target: Vec2
    steps: int = 96
    temperature: float = 0.95
    top_k: int = 24
    seed: int = 0
    prompt: str = ""
    style: str = ""


# -----------------------------
# 2) Parsing (Action DSL)
# -----------------------------


NUMBER = r"-?[0-9]*\.?[0-9]+"
VEC2 = rf"{NUMBER},{NUMBER}"

CANVAS_RE = re.compile(r"^canvas\s+(?P<w>\d+)\s+(?P<h>\d+)\s*$")
FPS_RE = re.compile(r"^fps\s+(?P<fps>\d+)\s*$")
SECONDS_RE = re.compile(r"^seconds\s+(?P<s>[0-9]*\.?[0-9]+)\s*$")
PHYSICS_RE = re.compile(
    r"""^physics
        (?:\s+mode=(?P<mode>off|fallback|pymunk|auto))?
        (?:\s+gravity=(?P<gravity>-?[0-9]*\.?[0-9]+))?
        (?:\s+damping=(?P<damping>[0-9]*\.?[0-9]+))?
        (?:\s+restitution=(?P<restitution>[0-9]*\.?[0-9]+))?
        (?:\s+friction=(?P<friction>[0-9]*\.?[0-9]+))?
        (?:\s+impulse_scale=(?P<impulse>[0-9]*\.?[0-9]+))?
        (?:\s+ragdoll_extra=(?P<rag_extra>[0-9]*\.?[0-9]+))?
        (?:\s+substeps=(?P<substeps>\d+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)
CAMERA_RE = re.compile(
    r"""^camera
        (?:\s+enabled=(?P<enabled>true|false|1|0|on|off|yes|no))?
        (?:\s+focus=(?P<focus>[a-zA-Z_][a-zA-Z0-9_]*|self|auto|centroid|none))?
        (?:\s+zoom=(?P<zoom>[0-9]*\.?[0-9]+))?
        (?:\s+pan=(?P<pan>-?[0-9]*\.?[0-9]+,-?[0-9]*\.?[0-9]+))?
        (?:\s+depth=(?P<depth>true|false|1|0|on|off|yes|no))?
        (?:\s+depth_min=(?P<dmin>[0-9]*\.?[0-9]+))?
        (?:\s+depth_max=(?P<dmax>[0-9]*\.?[0-9]+))?
        (?:\s+parallax=(?P<parallax>[0-9]*\.?[0-9]+))?
        (?:\s+y_sort=(?P<ysort>true|false|1|0|on|off|yes|no))?
        (?:\s+shake_on_impact=(?P<shake>true|false|1|0|on|off|yes|no))?
        (?:\s+shake_amp=(?P<samp>[0-9]*\.?[0-9]+))?
        (?:\s+shake_freq=(?P<sfreq>[0-9]*\.?[0-9]+))?
        (?:\s+shake_decay=(?P<sdecay>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

CHAR_RE = re.compile(
    r"""^character
        (?:\s+seed\s+"(?P<seed>[^"]+)")?
        (?:\s+line\s+(?P<line>[0-9]*\.?[0-9]+))?
        (?:\s+limb\s+(?P<limb>[0-9]*\.?[0-9]+))?
        (?:\s+head\s+(?P<head>[0-9]*\.?[0-9]+))?
        (?:\s+jitter\s+(?P<jit>[0-9]*\.?[0-9]+))?
        (?:\s+smear_threshold\s+(?P<smear_thr>[0-9]*\.?[0-9]+))?
        (?:\s+smear_full\s+(?P<smear_full>[0-9]*\.?[0-9]+))?
        (?:\s+smear_stretch\s+(?P<smear_stretch>[0-9]*\.?[0-9]+))?
        (?:\s+smear_squeeze\s+(?P<smear_squeeze>[0-9]*\.?[0-9]+))?
        (?:\s+smear_jitter\s+(?P<smear_jit>[0-9]*\.?[0-9]+))?
        (?:\s+mode\s+(?P<mode>stick|mesh))?
        (?:\s+mesh\s+"(?P<mesh>[^"]+)")?
        (?:\s+tint\s+"(?P<tint>[^"]+)")?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

WALK_RE = re.compile(
    rf"""^walk\s+
        from=(?P<from>{VEC2})\s+
        to=(?P<to>{VEC2})
        (?:\s+speed=(?P<speed>[0-9]*\.?[0-9]+))?
        (?:\s+bounce=(?P<bounce>[0-9]*\.?[0-9]+))?
        (?:\s+stride=(?P<stride>[0-9]*\.?[0-9]+))?
        (?:\s+step_height=(?P<step_h>[0-9]*\.?[0-9]+))?
        (?:\s+cadence=(?P<cadence>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.VERBOSE,
)

WAVE_RE = re.compile(
    r"""^wave
        (?:\s+hand=(?P<hand>left|right))?
        (?:\s+cycles=(?P<cycles>[0-9]*\.?[0-9]+))?
        (?:\s+amplitude=(?P<amp>[0-9]*\.?[0-9]+))?
        (?:\s+start=(?P<start>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.VERBOSE,
)
AI_MOTION_RE = re.compile(
    rf"""^ai_motion\s+
        model="(?P<model>[^"]+)"\s+
        tokenizer_model="(?P<tok>[^"]+)"\s+
        start=(?P<start>{VEC2})\s+
        target=(?P<target>{VEC2})
        (?:\s+steps=(?P<steps>\d+))?
        (?:\s+temperature=(?P<temp>[0-9]*\.?[0-9]+))?
        (?:\s+top_k=(?P<topk>\d+))?
        (?:\s+seed=(?P<seed>\d+))?
        (?:\s+prompt="(?P<prompt>[^"]*)")?
        (?:\s+style="(?P<style>[^"]*)")?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def parse_vec2(s: str) -> Vec2:
    x_str, y_str = s.split(",")
    return Vec2(float(x_str), float(y_str))


def parse_bool_token(v: str) -> bool:
    s = v.strip().lower()
    if s in {"1", "true", "on", "yes"}:
        return True
    if s in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"invalid bool token: {v}")


def parse_action_script(
    text: str,
) -> tuple[ScriptConfig, WalkCommand | None, WaveCommand | None, list[SlapstickEvent], list[SceneProp], AIMotionCommand | None]:
    cfg = ScriptConfig(width=800, height=450, fps=24, seconds=0.0)
    walk: WalkCommand | None = None
    wave: WaveCommand | None = None
    ai_motion: AIMotionCommand | None = None
    events: list[SlapstickEvent] = []
    props: list[SceneProp] = []

    for ln, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if m := CANVAS_RE.match(line):
            cfg.width = int(m.group("w"))
            cfg.height = int(m.group("h"))
            continue

        if m := FPS_RE.match(line):
            cfg.fps = int(m.group("fps"))
            continue

        if m := SECONDS_RE.match(line):
            cfg.seconds = float(m.group("s"))
            continue

        if m := PHYSICS_RE.match(line):
            if (v := m.group("mode")) is not None:
                cfg.physics_mode = v.lower()
            if (v := m.group("gravity")) is not None:
                cfg.physics_gravity = float(v)
            if (v := m.group("damping")) is not None:
                cfg.physics_damping = float(v)
            if (v := m.group("restitution")) is not None:
                cfg.physics_restitution = float(v)
            if (v := m.group("friction")) is not None:
                cfg.physics_friction = float(v)
            if (v := m.group("impulse")) is not None:
                cfg.physics_impulse_scale = float(v)
            if (v := m.group("rag_extra")) is not None:
                cfg.physics_ragdoll_extra = float(v)
            if (v := m.group("substeps")) is not None:
                cfg.physics_substeps = int(v)
            continue

        if m := CAMERA_RE.match(line):
            cfg.camera.enabled = True
            if (v := m.group("enabled")) is not None:
                cfg.camera.enabled = parse_bool_token(v)
            if (v := m.group("focus")) is not None:
                cfg.camera.focus = "" if v.lower() == "none" else v
            if (v := m.group("zoom")) is not None:
                cfg.camera.zoom = float(v)
            if (v := m.group("pan")) is not None:
                cfg.camera.pan = parse_vec2(v)
            if (v := m.group("depth")) is not None:
                cfg.camera.depth_enabled = parse_bool_token(v)
            if (v := m.group("dmin")) is not None:
                cfg.camera.depth_min_scale = float(v)
            if (v := m.group("dmax")) is not None:
                cfg.camera.depth_max_scale = float(v)
            if (v := m.group("parallax")) is not None:
                cfg.camera.parallax_strength = float(v)
            if (v := m.group("ysort")) is not None:
                cfg.camera.y_sort = parse_bool_token(v)
            if (v := m.group("shake")) is not None:
                cfg.camera.shake_on_impact = parse_bool_token(v)
            if (v := m.group("samp")) is not None:
                cfg.camera.shake_amplitude = float(v)
            if (v := m.group("sfreq")) is not None:
                cfg.camera.shake_frequency = float(v)
            if (v := m.group("sdecay")) is not None:
                cfg.camera.shake_decay = float(v)
            continue

        if m := CHAR_RE.match(line):
            if (seed := m.group("seed")) is not None:
                cfg.style.seed = seed
            if (v := m.group("line")) is not None:
                cfg.style.line_width = float(v)
            if (v := m.group("limb")) is not None:
                cfg.style.limb_width = float(v)
            if (v := m.group("head")) is not None:
                cfg.style.head_radius = float(v)
            if (v := m.group("jit")) is not None:
                cfg.style.jitter_amplitude = float(v)
            if (v := m.group("smear_thr")) is not None:
                cfg.style.smear_speed_threshold = float(v)
            if (v := m.group("smear_full")) is not None:
                cfg.style.smear_speed_full = float(v)
            if (v := m.group("smear_stretch")) is not None:
                cfg.style.smear_max_stretch = float(v)
            if (v := m.group("smear_squeeze")) is not None:
                cfg.style.smear_max_squeeze = float(v)
            if (v := m.group("smear_jit")) is not None:
                cfg.style.smear_jitter_boost = float(v)
            if (v := m.group("mode")) is not None:
                cfg.style.render_mode = v.lower()
            if (v := m.group("mesh")) is not None:
                cfg.style.mesh_asset = v
                cfg.style.render_mode = "mesh"
            if (v := m.group("tint")) is not None:
                cfg.style.mesh_tint = v
            continue

        if m := WALK_RE.match(line):
            walk = WalkCommand(
                start=parse_vec2(m.group("from")),
                end=parse_vec2(m.group("to")),
                speed=float(m.group("speed") or 1.0),
                bounce=float(m.group("bounce") or 0.3),
                stride=float(m.group("stride") or 54.0),
                step_height=float(m.group("step_h") or 22.0),
                cadence=float(m.group("cadence") or 1.6),
            )
            continue

        if m := WAVE_RE.match(line):
            wave = WaveCommand(
                hand=m.group("hand") or "right",
                cycles=float(m.group("cycles") or 2.0),
                amplitude=float(m.group("amp") or 24.0),
                start=float(m.group("start") or 0.0),
                duration=float(m.group("duration") or 0.0),
            )
            continue

        if m := AI_MOTION_RE.match(line):
            ai_motion = AIMotionCommand(
                model=m.group("model"),
                tokenizer_model=m.group("tok"),
                start=parse_vec2(m.group("start")),
                target=parse_vec2(m.group("target")),
                steps=max(2, int(m.group("steps") or 96)),
                temperature=float(m.group("temp") or 0.95),
                top_k=max(1, int(m.group("topk") or 24)),
                seed=int(m.group("seed") or 0),
                prompt=m.group("prompt") or "",
                style=m.group("style") or "",
            )
            continue

        try:
            prop = parse_scene_prop_line(line, index=len(props))
        except ValueError as exc:
            raise ValueError(f"Syntax error in line {ln}: {line}\n{exc}") from exc
        if prop is not None:
            props.append(prop)
            continue

        try:
            event = parse_slapstick_event_line(line)
        except ValueError as exc:
            raise ValueError(f"Syntax error in line {ln}: {line}\n{exc}") from exc
        if event is not None:
            events.append(event)
            continue

        raise ValueError(f"Syntax error in line {ln}: {line}")

    if walk is None and ai_motion is None:
        raise ValueError("Script contains neither 'walk' nor 'ai_motion' command")
    if walk is not None and ai_motion is not None:
        raise ValueError("Script cannot contain both 'walk' and 'ai_motion'")
    if cfg.fps <= 0:
        raise ValueError("fps must be > 0")

    return cfg, walk, wave, events, props, ai_motion


# -----------------------------
# 3) Procedural motion synthesis
# -----------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp2(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(lerp(a.x, b.x, t), lerp(a.y, b.y, t))


def infer_duration(walk: WalkCommand) -> float:
    distance = (walk.end - walk.start).length()
    if distance < 1e-6:
        return 1.0
    pixels_per_second = 130.0 * max(0.1, walk.speed)
    return max(0.6, distance / pixels_per_second)


def auto_wave_duration(wave: WaveCommand, walk: WalkCommand) -> float:
    if wave.duration > 0.0:
        return wave.duration
    # "cycles" at roughly one full arm shake per walk cadence unit.
    return max(0.4, wave.cycles / max(0.1, walk.cadence * walk.speed))


def overlay_wave(
    hand_target: Vec2,
    *,
    t: float,
    total_duration: float,
    wave: WaveCommand,
    walk: WalkCommand,
) -> Vec2:
    wave_duration = auto_wave_duration(wave, walk)
    start = clamp(wave.start, 0.0, total_duration)
    end = min(total_duration, start + wave_duration)
    if end <= start or t < start or t > end:
        return hand_target

    u = (t - start) / (end - start)
    theta = u * wave.cycles * 2.0 * math.pi
    envelope = math.sin(math.pi * u)  # smooth in/out

    dx = wave.amplitude * 0.35 * math.sin(theta)
    dy = -wave.amplitude * 0.75 * envelope + wave.amplitude * 0.2 * math.cos(theta)
    return hand_target + Vec2(dx, dy)


def sample_pose(
    *,
    t: float,
    duration: float,
    walk: WalkCommand,
    wave: WaveCommand | None,
) -> Pose:
    u = clamp(t / max(duration, 1e-6), 0.0, 1.0)
    root_base = lerp2(walk.start, walk.end, u)

    gait_hz = max(0.2, walk.cadence * max(0.1, walk.speed))
    phase = 2.0 * math.pi * gait_hz * t

    left_leg = math.sin(phase)
    right_leg = math.sin(phase + math.pi)

    bounce_px = 26.0 * clamp(walk.bounce, 0.0, 1.5)
    root_bob = -bounce_px * (0.5 - 0.5 * math.cos(phase * 2.0))
    root = Vec2(root_base.x, root_base.y + root_bob)

    # Keep feet near ground while root bobs, then add stride/lift.
    foot_ground_comp = -root_bob
    foot_base_y = 92.0 + foot_ground_comp
    foot_spread = 30.0
    stride = max(8.0, walk.stride)
    lift = max(2.0, walk.step_height)

    l_foot = Vec2(
        -foot_spread + stride * left_leg,
        foot_base_y - lift * max(0.0, left_leg),
    )
    r_foot = Vec2(
        +foot_spread + stride * right_leg,
        foot_base_y - lift * max(0.0, right_leg),
    )

    arm_swing = stride * 0.85
    l_hand = Vec2(
        -72.0 - arm_swing * left_leg,
        -120.0 + 8.0 * math.cos(phase + math.pi),
    )
    r_hand = Vec2(
        +72.0 + arm_swing * left_leg,
        -120.0 + 8.0 * math.cos(phase),
    )

    if wave is not None:
        if wave.hand == "left":
            l_hand = overlay_wave(l_hand, t=t, total_duration=duration, wave=wave, walk=walk)
        else:
            r_hand = overlay_wave(r_hand, t=t, total_duration=duration, wave=wave, walk=walk)

    look_vec = walk.end - walk.start
    look_angle = math.atan2(look_vec.y, look_vec.x) * 0.15 if look_vec.length() > 1e-6 else 0.0
    squash = clamp(walk.bounce * (0.08 + 0.22 * (0.5 - 0.5 * math.cos(phase * 2.0))), 0.0, 1.0)

    return Pose(
        root=root,
        l_hand=l_hand,
        r_hand=r_hand,
        l_foot=l_foot,
        r_foot=r_foot,
        look_angle=look_angle,
        squash=squash,
    )


def timeline_direction_from_keys(tl: Timeline) -> Vec2:
    if len(tl.keyframes) < 2:
        return Vec2(1.0, 0.0)
    a = tl.keyframes[0].pose.root
    b = tl.keyframes[-1].pose.root
    d = b - a
    if d.length() <= 1e-6:
        return Vec2(1.0, 0.0)
    return d.normalized()


def generate_ai_base_timeline(
    cfg: ScriptConfig,
    ai_motion: AIMotionCommand,
) -> Timeline:
    try:
        from motion_transformer import generate_timeline_from_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "ai_motion requires motion_transformer.py + torch. "
            "Install torch or disable ai_motion."
        ) from exc

    tl, inferred_seconds = generate_timeline_from_model(
        model_path=ai_motion.model,
        tokenizer_model_path=ai_motion.tokenizer_model,
        start=ai_motion.start,
        target=ai_motion.target,
        fps=cfg.fps,
        steps=ai_motion.steps,
        temperature=ai_motion.temperature,
        top_k=ai_motion.top_k,
        seed=ai_motion.seed,
        prompt=ai_motion.prompt,
        style=ai_motion.style,
    )
    if cfg.seconds <= 0.0:
        cfg.seconds = inferred_seconds
    return tl


def _generate_procedural_scene_internal(
    cfg: ScriptConfig,
    walk: WalkCommand | None,
    wave: WaveCommand | None,
    *,
    ai_motion: AIMotionCommand | None = None,
    events: list[SlapstickEvent] | None = None,
    props: list[SceneProp] | None = None,
) -> tuple[Timeline, PropRuntimeState | None, list[SlapstickEvent], dict]:
    if ai_motion is not None:
        base_tl = generate_ai_base_timeline(cfg, ai_motion)
        duration = cfg.seconds if cfg.seconds > 0.0 else max((k.t for k in base_tl.keyframes), default=1.0)
        cfg.seconds = duration
        motion_dir = timeline_direction_from_keys(base_tl)
    else:
        if walk is None:
            raise ValueError("walk is required when ai_motion is not set")
        duration = cfg.seconds if cfg.seconds > 0.0 else infer_duration(walk)
        cfg.seconds = duration

        total_keys = int(math.ceil(duration * cfg.fps)) + 1
        base_tl = Timeline()
        for i in range(total_keys):
            t = min(duration, i / cfg.fps)
            base_tl.add(t, sample_pose(t=t, duration=duration, walk=walk, wave=wave))
        motion_dir = (walk.end - walk.start)

    all_events = list(events or [])
    prop_runtime: PropRuntimeState | None = None
    prop_derived_events: list[SlapstickEvent] = []
    if props:
        prop_derived_events, prop_runtime = derive_prop_events(base_tl, cfg=cfg, props=props)
        all_events.extend(prop_derived_events)

    if all_events:
        out_tl = apply_slapstick_events(
            base_tl,
            events=all_events,
            motion_direction=motion_dir,
        )
    else:
        out_tl = base_tl

    physics_meta = {
        "enabled": False,
        "mode_requested": cfg.physics_mode,
        "solver": "off",
        "impact_events": 0,
        "ragdoll_frames": 0,
    }
    if cfg.physics_mode.lower() != "off":
        out_tl, physics_meta = apply_hybrid_physics(
            out_tl,
            cfg=cfg,
            events=all_events,
            ground_y=cfg.height - 60,
        )

    return out_tl, prop_runtime, prop_derived_events, physics_meta


def generate_procedural_scene(
    cfg: ScriptConfig,
    walk: WalkCommand | None,
    wave: WaveCommand | None,
    *,
    ai_motion: AIMotionCommand | None = None,
    events: list[SlapstickEvent] | None = None,
    props: list[SceneProp] | None = None,
) -> tuple[Timeline, PropRuntimeState | None, list[SlapstickEvent]]:
    tl, prop_runtime, prop_derived_events, _ = _generate_procedural_scene_internal(
        cfg,
        walk,
        wave,
        ai_motion=ai_motion,
        events=events,
        props=props,
    )
    return tl, prop_runtime, prop_derived_events


def generate_procedural_scene_detailed(
    cfg: ScriptConfig,
    walk: WalkCommand | None,
    wave: WaveCommand | None,
    *,
    ai_motion: AIMotionCommand | None = None,
    events: list[SlapstickEvent] | None = None,
    props: list[SceneProp] | None = None,
) -> tuple[Timeline, PropRuntimeState | None, list[SlapstickEvent], dict]:
    return _generate_procedural_scene_internal(
        cfg,
        walk,
        wave,
        ai_motion=ai_motion,
        events=events,
        props=props,
    )


def generate_procedural_timeline(
    cfg: ScriptConfig,
    walk: WalkCommand | None,
    wave: WaveCommand | None,
    ai_motion: AIMotionCommand | None = None,
    events: list[SlapstickEvent] | None = None,
    props: list[SceneProp] | None = None,
) -> Timeline:
    tl, _, _ = generate_procedural_scene(
        cfg,
        walk,
        wave,
        ai_motion=ai_motion,
        events=events,
        props=props,
    )
    return tl


def collect_impact_times(events: list[SlapstickEvent] | None) -> list[float]:
    if not events:
        return []
    out = [e.t for e in events if isinstance(e, ImpactEvent)]
    out.sort()
    return out


# -----------------------------
# 4) Demo + CLI
# -----------------------------


DEMO_PROCEDURAL_SCRIPT = """# Procedural walk-cycle demo
canvas 800 450
fps 24
character seed "PROC-WALK-INK" line 7 limb 10 head 28 jitter 1.0
physics mode=off
camera enabled=false

# Optional:
# seconds 3.2

walk from=200,330 to=600,330 speed=1.0 bounce=0.3 stride=56 step_height=22 cadence=1.6
wave hand=right cycles=2 amplitude=26 start=1.2 duration=1.3
wall x=420 width=18 height=140 force=0.9 duration=0.22
trapdoor x=520 width=90 depth=70 force=0.8 duration=0.24 open_time=0.28
anvil x=610 size=34 trigger_x=565 trigger_radius=42 delay=0.08 fall_speed=360 force=1.0 duration=0.2
anticipation t=0.75 action=sprint intensity=0.9 duration=0.25
impact t=2.05 direction=-1,0 force=0.8 duration=0.24
take t=2.45 intensity=0.75 hold=2
"""


def write_demo_script(path: Path) -> None:
    path.write_text(DEMO_PROCEDURAL_SCRIPT, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Procedural walk-cycle generator for cartoon_svg_mvp")
    ap.add_argument("--script", type=str, default="", help="Action DSL file (walk/wave).")
    ap.add_argument("--out", type=str, default="out", help="Output folder for SVG frames.")
    ap.add_argument("--demo", action="store_true", help="Write procedural_demo.txt and exit.")
    ap.add_argument("--fps", type=int, default=0, help="Override script fps.")
    ap.add_argument("--seconds", type=float, default=0.0, help="Override script duration.")
    args = ap.parse_args()

    if args.demo:
        demo_path = Path("procedural_demo.txt")
        write_demo_script(demo_path)
        print(f"[ok] procedural demo script written: {demo_path.resolve()}")
        return 0

    if args.script:
        script_path = Path(args.script)
        text = script_path.read_text(encoding="utf-8")
    else:
        script_path = Path("procedural_demo.txt")
        if not script_path.exists():
            write_demo_script(script_path)
        text = script_path.read_text(encoding="utf-8")

    cfg, walk, wave, events, props, ai_motion = parse_action_script(text)
    if args.fps > 0:
        cfg.fps = args.fps
    if args.seconds > 0.0:
        cfg.seconds = args.seconds

    timeline, prop_runtime, prop_derived_events, physics_meta = generate_procedural_scene_detailed(
        cfg,
        walk,
        wave,
        ai_motion=ai_motion,
        events=events,
        props=props,
    )
    prop_items_fn = build_prop_items_fn(prop_runtime, cfg)
    camera_impact_times = collect_impact_times([*events, *prop_derived_events])
    out_dir = Path(args.out)
    render_frames(
        cfg=cfg,
        tl=timeline,
        out_dir=out_dir,
        extra_items_fn=prop_items_fn,
        camera_impact_times=camera_impact_times,
    )

    total_frames = int(round(cfg.seconds * cfg.fps))
    print(f"[ok] rendered procedural walk cycle: {out_dir.resolve()} ({total_frames} frames)")
    if physics_meta.get("enabled"):
        print(
            "[ok] physics:",
            f"mode={physics_meta.get('solver')}",
            f"impact_events={physics_meta.get('impact_events')}",
            f"ragdoll_frames={physics_meta.get('ragdoll_frames')}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
