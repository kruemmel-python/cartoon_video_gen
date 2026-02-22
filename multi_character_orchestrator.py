#!/usr/bin/env python3
"""
multi_character_orchestrator.py

Two-character (or more) orchestration layer for vector-first slapstick scenes.

Core idea:
- per-character action programs (walk/chase/wave/events)
- shared environment props
- automatic inter-agent collision events (bounce + take)
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from cartoon_svg_mvp import (
    CharacterStyle,
    Pose,
    ScriptConfig,
    Timeline,
    Vec2,
    render_multi_frames,
)
from physics_hybrid import apply_hybrid_physics
from procedural_props import (
    PropRuntimeState,
    SceneProp,
    build_prop_items_fn,
    derive_prop_events,
    parse_scene_prop_line,
)
from slapstick_events import (
    ImpactEvent,
    SlapstickEvent,
    TakeEvent,
    apply_slapstick_events,
    parse_slapstick_event_line,
)


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
    hand: str = "right"
    cycles: float = 2.0
    amplitude: float = 24.0
    start: float = 0.0
    duration: float = 0.0


@dataclass(slots=True)
class ChaseCommand:
    target: str
    offset: float = -95.0
    aggression: float = 1.25
    bounce: float = 0.34
    stride: float = 58.0
    step_height: float = 23.0
    cadence: float = 1.8


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


@dataclass(slots=True)
class DuelCollisionConfig:
    distance: float = 48.0
    force: float = 1.0
    duration: float = 0.22
    take_intensity: float = 0.72
    cooldown: float = 0.32


@dataclass(slots=True)
class CharacterProgram:
    char_id: str
    style: CharacterStyle
    walk: WalkCommand | None = None
    chase: ChaseCommand | None = None
    ai_motion: AIMotionCommand | None = None
    wave: WaveCommand | None = None
    events: list[SlapstickEvent] = field(default_factory=list)


NUMBER = r"-?[0-9]*\.?[0-9]+"
VEC2 = rf"{NUMBER},{NUMBER}"

CANVAS_RE = re.compile(r"^canvas\s+(?P<w>\d+)\s+(?P<h>\d+)\s*$", re.IGNORECASE)
FPS_RE = re.compile(r"^fps\s+(?P<fps>\d+)\s*$", re.IGNORECASE)
SECONDS_RE = re.compile(r"^seconds\s+(?P<s>[0-9]*\.?[0-9]+)\s*$", re.IGNORECASE)
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

CHAR_STYLE_RE = re.compile(
    r"""^character\s+id=(?P<id>[a-zA-Z_][a-zA-Z0-9_]*)
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
    re.IGNORECASE | re.VERBOSE,
)

WAVE_RE = re.compile(
    r"""^wave
        (?:\s+hand=(?P<hand>left|right))?
        (?:\s+cycles=(?P<cycles>[0-9]*\.?[0-9]+))?
        (?:\s+amplitude=(?P<amp>[0-9]*\.?[0-9]+))?
        (?:\s+start=(?P<start>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

CHASE_RE = re.compile(
    r"""^chase\s+
        target=(?P<target>[a-zA-Z_][a-zA-Z0-9_]*)
        (?:\s+offset=(?P<offset>-?[0-9]*\.?[0-9]+))?
        (?:\s+aggression=(?P<aggr>[0-9]*\.?[0-9]+))?
        (?:\s+bounce=(?P<bounce>[0-9]*\.?[0-9]+))?
        (?:\s+stride=(?P<stride>[0-9]*\.?[0-9]+))?
        (?:\s+step_height=(?P<step_h>[0-9]*\.?[0-9]+))?
        (?:\s+cadence=(?P<cadence>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
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

COLLISION_RE = re.compile(
    r"""^(?:duel_collision|collision)
        (?:\s+distance=(?P<distance>[0-9]*\.?[0-9]+))?
        (?:\s+force=(?P<force>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        (?:\s+take_intensity=(?P<take>[0-9]*\.?[0-9]+))?
        (?:\s+cooldown=(?P<cooldown>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

CHAR_PREFIX_CMD_RE = re.compile(
    r"^(?P<char>[a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(?P<cmd>.+)$",
    re.IGNORECASE,
)

CHAR_INLINE_CMD_RE = re.compile(
    r"^(?P<char>[a-zA-Z_][a-zA-Z0-9_]*)\s+(?P<cmd>(?:walk|wave|chase|ai_motion|impact|take|anticipation)\b.*)$",
    re.IGNORECASE,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp2(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(lerp(a.x, b.x, t), lerp(a.y, b.y, t))


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


def ensure_program(programs: dict[str, CharacterProgram], char_id: str) -> CharacterProgram:
    program = programs.get(char_id)
    if program is not None:
        return program
    style = CharacterStyle(seed=f"INK-{char_id.upper()}")
    program = CharacterProgram(char_id=char_id, style=style)
    programs[char_id] = program
    return program


def apply_style_match(style: CharacterStyle, m: re.Match[str]) -> None:
    if (seed := m.group("seed")) is not None:
        style.seed = seed
    if (v := m.group("line")) is not None:
        style.line_width = float(v)
    if (v := m.group("limb")) is not None:
        style.limb_width = float(v)
    if (v := m.group("head")) is not None:
        style.head_radius = float(v)
    if (v := m.group("jit")) is not None:
        style.jitter_amplitude = float(v)
    if (v := m.group("smear_thr")) is not None:
        style.smear_speed_threshold = float(v)
    if (v := m.group("smear_full")) is not None:
        style.smear_speed_full = float(v)
    if (v := m.group("smear_stretch")) is not None:
        style.smear_max_stretch = float(v)
    if (v := m.group("smear_squeeze")) is not None:
        style.smear_max_squeeze = float(v)
    if (v := m.group("smear_jit")) is not None:
        style.smear_jitter_boost = float(v)
    if (v := m.group("mode")) is not None:
        style.render_mode = v.lower()
    if (v := m.group("mesh")) is not None:
        style.mesh_asset = v
        style.render_mode = "mesh"
    if (v := m.group("tint")) is not None:
        style.mesh_tint = v


def parse_character_command(program: CharacterProgram, cmd: str, *, ln: int) -> None:
    if m := WALK_RE.match(cmd):
        if program.chase is not None or program.ai_motion is not None:
            raise ValueError(f"Line {ln}: '{program.char_id}' cannot combine walk with chase/ai_motion")
        program.walk = WalkCommand(
            start=parse_vec2(m.group("from")),
            end=parse_vec2(m.group("to")),
            speed=float(m.group("speed") or 1.0),
            bounce=float(m.group("bounce") or 0.3),
            stride=float(m.group("stride") or 54.0),
            step_height=float(m.group("step_h") or 22.0),
            cadence=float(m.group("cadence") or 1.6),
        )
        return

    if m := CHASE_RE.match(cmd):
        if program.walk is not None or program.ai_motion is not None:
            raise ValueError(f"Line {ln}: '{program.char_id}' cannot combine chase with walk/ai_motion")
        program.chase = ChaseCommand(
            target=m.group("target"),
            offset=float(m.group("offset") or -95.0),
            aggression=float(m.group("aggr") or 1.25),
            bounce=float(m.group("bounce") or 0.34),
            stride=float(m.group("stride") or 58.0),
            step_height=float(m.group("step_h") or 23.0),
            cadence=float(m.group("cadence") or 1.8),
        )
        return

    if m := AI_MOTION_RE.match(cmd):
        if program.walk is not None or program.chase is not None:
            raise ValueError(f"Line {ln}: '{program.char_id}' cannot combine ai_motion with walk/chase")
        program.ai_motion = AIMotionCommand(
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
        return

    if m := WAVE_RE.match(cmd):
        program.wave = WaveCommand(
            hand=m.group("hand") or "right",
            cycles=float(m.group("cycles") or 2.0),
            amplitude=float(m.group("amp") or 24.0),
            start=float(m.group("start") or 0.0),
            duration=float(m.group("duration") or 0.0),
        )
        return

    event = parse_slapstick_event_line(cmd)
    if event is not None:
        program.events.append(event)
        return

    raise ValueError(f"Line {ln}: unsupported character command '{cmd}'")


def parse_multi_script(
    text: str,
) -> tuple[ScriptConfig, dict[str, CharacterProgram], list[SceneProp], DuelCollisionConfig]:
    cfg = ScriptConfig(width=960, height=540, fps=24, seconds=0.0)
    programs: dict[str, CharacterProgram] = {}
    props: list[SceneProp] = []
    collision = DuelCollisionConfig()

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

        if m := CHAR_STYLE_RE.match(line):
            char_id = m.group("id")
            program = ensure_program(programs, char_id)
            apply_style_match(program.style, m)
            continue

        if m := COLLISION_RE.match(line):
            if (v := m.group("distance")) is not None:
                collision.distance = float(v)
            if (v := m.group("force")) is not None:
                collision.force = float(v)
            if (v := m.group("duration")) is not None:
                collision.duration = float(v)
            if (v := m.group("take")) is not None:
                collision.take_intensity = float(v)
            if (v := m.group("cooldown")) is not None:
                collision.cooldown = float(v)
            continue

        try:
            prop = parse_scene_prop_line(line, index=len(props))
        except ValueError as exc:
            raise ValueError(f"Line {ln}: {exc}") from exc
        if prop is not None:
            props.append(prop)
            continue

        prefix_match = CHAR_PREFIX_CMD_RE.match(line)
        if prefix_match is None:
            prefix_match = CHAR_INLINE_CMD_RE.match(line)
        if prefix_match is not None:
            char_id = prefix_match.group("char")
            cmd = prefix_match.group("cmd").strip()
            program = ensure_program(programs, char_id)
            parse_character_command(program, cmd, ln=ln)
            continue

        raise ValueError(f"Syntax error in line {ln}: {line}")

    if cfg.fps <= 0:
        raise ValueError("fps must be > 0")
    if len(programs) < 2:
        raise ValueError("Need at least two characters (use char_id: walk/chase commands)")

    for char_id, program in programs.items():
        motion_count = int(program.walk is not None) + int(program.chase is not None) + int(program.ai_motion is not None)
        if motion_count == 0:
            raise ValueError(f"Character '{char_id}' has no motion command (walk/chase/ai_motion)")
        if motion_count > 1:
            raise ValueError(f"Character '{char_id}' has multiple motion commands")
        if program.chase is not None and program.chase.target == char_id:
            raise ValueError(f"Character '{char_id}' cannot chase itself")

    return cfg, programs, props, collision


def infer_walk_duration(walk: WalkCommand) -> float:
    distance = (walk.end - walk.start).length()
    if distance < 1e-6:
        return 1.0
    pixels_per_second = 130.0 * max(0.1, walk.speed)
    return max(0.6, distance / pixels_per_second)


def auto_wave_duration(wave: WaveCommand, cadence: float, speed: float) -> float:
    if wave.duration > 0.0:
        return wave.duration
    return max(0.4, wave.cycles / max(0.1, cadence * max(0.1, speed)))


def overlay_wave(
    hand_target: Vec2,
    *,
    t: float,
    total_duration: float,
    wave: WaveCommand,
    cadence: float,
    speed: float,
) -> Vec2:
    wave_duration = auto_wave_duration(wave, cadence, speed)
    start = clamp(wave.start, 0.0, total_duration)
    end = min(total_duration, start + wave_duration)
    if end <= start or t < start or t > end:
        return hand_target

    u = (t - start) / (end - start)
    theta = u * wave.cycles * 2.0 * math.pi
    envelope = math.sin(math.pi * u)
    dx = wave.amplitude * 0.35 * math.sin(theta)
    dy = -wave.amplitude * 0.75 * envelope + wave.amplitude * 0.2 * math.cos(theta)
    return hand_target + Vec2(dx, dy)


def pose_from_root_base(
    *,
    root_base: Vec2,
    move_dir: Vec2,
    t: float,
    total_duration: float,
    speed: float,
    bounce: float,
    stride: float,
    step_height: float,
    cadence: float,
    wave: WaveCommand | None,
) -> Pose:
    gait_hz = max(0.2, cadence * max(0.1, speed))
    phase = 2.0 * math.pi * gait_hz * t

    left_leg = math.sin(phase)
    right_leg = math.sin(phase + math.pi)

    bounce_px = 26.0 * clamp(bounce, 0.0, 1.5)
    root_bob = -bounce_px * (0.5 - 0.5 * math.cos(phase * 2.0))
    root = Vec2(root_base.x, root_base.y + root_bob)

    foot_ground_comp = -root_bob
    foot_base_y = 92.0 + foot_ground_comp
    foot_spread = 30.0
    stride_local = max(8.0, stride)
    lift = max(2.0, step_height)

    l_foot = Vec2(
        -foot_spread + stride_local * left_leg,
        foot_base_y - lift * max(0.0, left_leg),
    )
    r_foot = Vec2(
        +foot_spread + stride_local * right_leg,
        foot_base_y - lift * max(0.0, right_leg),
    )

    arm_swing = stride_local * 0.85
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
            l_hand = overlay_wave(
                l_hand,
                t=t,
                total_duration=total_duration,
                wave=wave,
                cadence=cadence,
                speed=speed,
            )
        else:
            r_hand = overlay_wave(
                r_hand,
                t=t,
                total_duration=total_duration,
                wave=wave,
                cadence=cadence,
                speed=speed,
            )

    look_vec = move_dir
    look_angle = math.atan2(look_vec.y, look_vec.x) * 0.16 if look_vec.length() > 1e-6 else 0.0
    squash = clamp(bounce * (0.08 + 0.22 * (0.5 - 0.5 * math.cos(phase * 2.0))), 0.0, 1.0)

    return Pose(
        root=root,
        l_hand=l_hand,
        r_hand=r_hand,
        l_foot=l_foot,
        r_foot=r_foot,
        look_angle=look_angle,
        squash=squash,
    )


def timeline_direction(tl: Timeline) -> Vec2:
    keys = sorted(tl.keyframes, key=lambda k: k.t)
    if len(keys) < 2:
        return Vec2(1.0, 0.0)
    delta = keys[-1].pose.root - keys[0].pose.root
    if delta.length() <= 1e-6:
        return Vec2(1.0, 0.0)
    return delta.normalized()


def resolve_scene_duration(cfg: ScriptConfig, programs: dict[str, CharacterProgram]) -> float:
    if cfg.seconds > 0.0:
        return cfg.seconds
    duration = 0.0
    for program in programs.values():
        if program.walk is not None:
            duration = max(duration, infer_walk_duration(program.walk))
        elif program.ai_motion is not None:
            duration = max(duration, max(0.2, (program.ai_motion.steps - 1) / max(1, cfg.fps)))
    if duration <= 0.0:
        duration = 3.0
    return duration


def generate_walk_timeline(
    *,
    cfg: ScriptConfig,
    walk: WalkCommand,
    wave: WaveCommand | None,
    scene_duration: float,
) -> Timeline:
    motion_duration = infer_walk_duration(walk)
    total_keys = int(math.ceil(scene_duration * cfg.fps)) + 1
    move_dir = walk.end - walk.start
    tl = Timeline()
    for i in range(total_keys):
        t = min(scene_duration, i / cfg.fps)
        motion_t = min(t, motion_duration)
        u = clamp(motion_t / max(motion_duration, 1e-6), 0.0, 1.0)
        root_base = lerp2(walk.start, walk.end, u)
        pose = pose_from_root_base(
            root_base=root_base,
            move_dir=move_dir,
            t=motion_t,
            total_duration=motion_duration,
            speed=walk.speed,
            bounce=walk.bounce,
            stride=walk.stride,
            step_height=walk.step_height,
            cadence=walk.cadence,
            wave=wave,
        )
        tl.add(t, pose)
    return tl


def generate_chase_timeline(
    *,
    cfg: ScriptConfig,
    chase: ChaseCommand,
    wave: WaveCommand | None,
    target_timeline: Timeline,
    scene_duration: float,
) -> Timeline:
    keys = sorted(target_timeline.keyframes, key=lambda k: k.t)
    if not keys:
        raise ValueError("target timeline has no keyframes")

    dt = 1.0 / cfg.fps
    total_keys = int(math.ceil(scene_duration * cfg.fps)) + 1
    target_end_t = keys[-1].t
    fallback_dir = timeline_direction(target_timeline)
    root_base = keys[0].pose.root + fallback_dir * chase.offset

    tl = Timeline()
    for i in range(total_keys):
        t = min(scene_duration, i / cfg.fps)
        target_t = min(target_end_t, t)
        target_root = target_timeline.sample(target_t).root

        t_prev = max(0.0, target_t - dt)
        t_next = min(target_end_t, target_t + dt)
        target_vel = target_timeline.sample(t_next).root - target_timeline.sample(t_prev).root
        move_dir = target_vel.normalized() if target_vel.length() > 1e-6 else fallback_dir

        desired = Vec2(
            target_root.x + move_dir.x * chase.offset,
            target_root.y + move_dir.y * chase.offset * 0.15,
        )
        if i == 0:
            root_base = desired
        else:
            alpha = clamp(1.0 - math.exp(-max(0.1, chase.aggression) * 6.0 * dt), 0.04, 1.0)
            root_base = lerp2(root_base, desired, alpha)

        look_vec = target_root - root_base
        pose = pose_from_root_base(
            root_base=root_base,
            move_dir=look_vec if look_vec.length() > 1e-6 else move_dir,
            t=t,
            total_duration=scene_duration,
            speed=max(0.5, chase.aggression),
            bounce=chase.bounce,
            stride=chase.stride,
            step_height=chase.step_height,
            cadence=chase.cadence,
            wave=wave,
        )
        tl.add(t, pose)
    return tl


def generate_ai_timeline(
    *,
    cfg: ScriptConfig,
    ai_motion: AIMotionCommand,
    scene_duration: float,
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

    if scene_duration <= 0.0:
        scene_duration = inferred_seconds
    # Normalize timeline length to scene_duration.
    total_keys = int(math.ceil(scene_duration * cfg.fps)) + 1
    out = Timeline()
    last_t = tl.keyframes[-1].t if tl.keyframes else 0.0
    for i in range(total_keys):
        t = min(scene_duration, i / cfg.fps)
        sample_t = min(last_t, t)
        out.add(t, tl.sample(sample_t))
    return out


def build_base_timelines(
    *,
    cfg: ScriptConfig,
    programs: dict[str, CharacterProgram],
    scene_duration: float,
) -> dict[str, Timeline]:
    out: dict[str, Timeline] = {}
    active: set[str] = set()

    def build(char_id: str) -> Timeline:
        if char_id in out:
            return out[char_id]
        if char_id in active:
            raise ValueError(f"Chase dependency cycle detected at '{char_id}'")
        if char_id not in programs:
            raise ValueError(f"Unknown character id '{char_id}'")

        active.add(char_id)
        program = programs[char_id]
        if program.walk is not None:
            tl = generate_walk_timeline(
                cfg=cfg,
                walk=program.walk,
                wave=program.wave,
                scene_duration=scene_duration,
            )
        elif program.ai_motion is not None:
            tl = generate_ai_timeline(
                cfg=cfg,
                ai_motion=program.ai_motion,
                scene_duration=scene_duration,
            )
        elif program.chase is not None:
            target_tl = build(program.chase.target)
            tl = generate_chase_timeline(
                cfg=cfg,
                chase=program.chase,
                wave=program.wave,
                target_timeline=target_tl,
                scene_duration=scene_duration,
            )
        else:
            raise ValueError(f"Character '{char_id}' has no motion command")

        active.remove(char_id)
        out[char_id] = tl
        return tl

    for char_id in programs:
        build(char_id)
    return out


def merge_prop_runtimes(
    props: list[SceneProp],
    runtimes: list[PropRuntimeState],
    cfg: ScriptConfig,
) -> PropRuntimeState | None:
    if not props:
        return None

    trigger_times: dict[str, float] = {}
    impact_times: dict[str, float] = {}
    for runtime in runtimes:
        for prop_id, t in runtime.trigger_times.items():
            prev = trigger_times.get(prop_id)
            if prev is None or t < prev:
                trigger_times[prop_id] = t
        for prop_id, t in runtime.impact_times.items():
            prev = impact_times.get(prop_id)
            if prev is None or t < prev:
                impact_times[prop_id] = t

    return PropRuntimeState(
        props=props,
        trigger_times=trigger_times,
        impact_times=impact_times,
        ground_y=cfg.height - 60,
    )


def derive_inter_agent_events(
    *,
    timelines: dict[str, Timeline],
    cfg: ScriptConfig,
    collision: DuelCollisionConfig,
) -> dict[str, list[SlapstickEvent]]:
    events: dict[str, list[SlapstickEvent]] = {char_id: [] for char_id in timelines}
    char_ids = sorted(timelines.keys())
    if len(char_ids) < 2:
        return events
    if collision.distance <= 0.0:
        return events

    total_samples = int(math.ceil(cfg.seconds * cfg.fps)) + 1
    next_allowed: dict[tuple[str, str], float] = {}
    prev_roots: dict[str, Vec2] = {}
    for pair in combinations(char_ids, 2):
        next_allowed[pair] = -1e9

    for i in range(total_samples):
        t = min(cfg.seconds, i / cfg.fps)
        roots = {char_id: timelines[char_id].sample(t).root for char_id in char_ids}

        for a, b in combinations(char_ids, 2):
            if t < next_allowed[(a, b)]:
                continue
            delta = roots[a] - roots[b]
            if delta.length() > collision.distance:
                continue

            dir_ = delta.normalized() if delta.length() > 1e-6 else Vec2(0.0, 0.0)
            if dir_.length() <= 1e-6 and a in prev_roots and b in prev_roots:
                rel_vel = (roots[a] - prev_roots[a]) - (roots[b] - prev_roots[b])
                if rel_vel.length() > 1e-6:
                    dir_ = rel_vel.normalized()
            if dir_.length() <= 1e-6:
                dir_ = Vec2(1.0, 0.0)

            force = max(0.1, collision.force)
            duration = max(0.05, collision.duration)
            events[a].append(ImpactEvent(t=t, direction=dir_, force=force, duration=duration))
            events[b].append(ImpactEvent(t=t, direction=dir_ * -1.0, force=force, duration=duration))
            if collision.take_intensity > 0.0:
                take_t = min(cfg.seconds, t + 0.04)
                intensity = max(0.1, collision.take_intensity)
                events[a].append(TakeEvent(t=take_t, intensity=intensity, hold_frames=1))
                events[b].append(TakeEvent(t=take_t, intensity=intensity, hold_frames=1))

            next_allowed[(a, b)] = t + max(0.05, collision.cooldown)

        prev_roots = roots

    for char_id in events:
        events[char_id].sort(key=lambda e: e.t)
    return events


def _generate_multi_character_scene_internal(
    *,
    cfg: ScriptConfig,
    programs: dict[str, CharacterProgram],
    props: list[SceneProp],
    collision: DuelCollisionConfig,
) -> tuple[
    dict[str, Timeline],
    PropRuntimeState | None,
    dict[str, list[SlapstickEvent]],
    dict[str, list[SlapstickEvent]],
    dict[str, dict],
]:
    scene_duration = resolve_scene_duration(cfg, programs)
    cfg.seconds = scene_duration

    base_timelines = build_base_timelines(cfg=cfg, programs=programs, scene_duration=scene_duration)
    stage1_timelines: dict[str, Timeline] = {}
    runtimes: list[PropRuntimeState] = []
    prop_events_by_char: dict[str, list[SlapstickEvent]] = {}

    for char_id, base_tl in base_timelines.items():
        char_events = list(programs[char_id].events)
        prop_events_local: list[SlapstickEvent] = []
        if props:
            prop_events, runtime = derive_prop_events(base_tl, cfg=cfg, props=props)
            char_events.extend(prop_events)
            prop_events_local.extend(prop_events)
            runtimes.append(runtime)
        prop_events_by_char[char_id] = prop_events_local

        if char_events:
            stage1_timelines[char_id] = apply_slapstick_events(
                base_tl,
                events=char_events,
                motion_direction=timeline_direction(base_tl),
            )
        else:
            stage1_timelines[char_id] = base_tl

    collision_events = derive_inter_agent_events(
        timelines=stage1_timelines,
        cfg=cfg,
        collision=collision,
    )

    final_timelines: dict[str, Timeline] = {}
    for char_id, tl in stage1_timelines.items():
        events = collision_events.get(char_id) or []
        if events:
            final_timelines[char_id] = apply_slapstick_events(
                tl,
                events=events,
                motion_direction=timeline_direction(tl),
            )
        else:
            final_timelines[char_id] = tl

    physics_meta_by_char: dict[str, dict] = {}
    if cfg.physics_mode.lower() != "off":
        physics_timelines: dict[str, Timeline] = {}
        for char_id, tl in final_timelines.items():
            combined_events = [
                *programs[char_id].events,
                *(prop_events_by_char.get(char_id) or []),
                *(collision_events.get(char_id) or []),
            ]
            phys_tl, phys_meta = apply_hybrid_physics(
                tl,
                cfg=cfg,
                events=combined_events,
                ground_y=cfg.height - 60,
            )
            physics_timelines[char_id] = phys_tl
            physics_meta_by_char[char_id] = phys_meta
        final_timelines = physics_timelines
    else:
        for char_id in final_timelines:
            physics_meta_by_char[char_id] = {
                "enabled": False,
                "mode_requested": cfg.physics_mode,
                "solver": "off",
                "impact_events": 0,
                "ragdoll_frames": 0,
            }

    merged_runtime = merge_prop_runtimes(props, runtimes, cfg)
    return final_timelines, merged_runtime, collision_events, prop_events_by_char, physics_meta_by_char


def generate_multi_character_scene(
    *,
    cfg: ScriptConfig,
    programs: dict[str, CharacterProgram],
    props: list[SceneProp],
    collision: DuelCollisionConfig,
) -> tuple[
    dict[str, Timeline],
    PropRuntimeState | None,
    dict[str, list[SlapstickEvent]],
    dict[str, list[SlapstickEvent]],
]:
    timelines, runtime, collision_events, prop_events, _ = _generate_multi_character_scene_internal(
        cfg=cfg,
        programs=programs,
        props=props,
        collision=collision,
    )
    return timelines, runtime, collision_events, prop_events


def generate_multi_character_scene_detailed(
    *,
    cfg: ScriptConfig,
    programs: dict[str, CharacterProgram],
    props: list[SceneProp],
    collision: DuelCollisionConfig,
) -> tuple[
    dict[str, Timeline],
    PropRuntimeState | None,
    dict[str, list[SlapstickEvent]],
    dict[str, list[SlapstickEvent]],
    dict[str, dict],
]:
    return _generate_multi_character_scene_internal(
        cfg=cfg,
        programs=programs,
        props=props,
        collision=collision,
    )


def collect_impact_times_by_char(
    *,
    programs: dict[str, CharacterProgram],
    prop_events_by_char: dict[str, list[SlapstickEvent]],
    collision_events: dict[str, list[SlapstickEvent]],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for char_id, program in programs.items():
        times: list[float] = []
        for ev in program.events:
            if isinstance(ev, ImpactEvent):
                times.append(ev.t)
        for ev in prop_events_by_char.get(char_id, []):
            if isinstance(ev, ImpactEvent):
                times.append(ev.t)
        for ev in collision_events.get(char_id, []):
            if isinstance(ev, ImpactEvent):
                times.append(ev.t)
        times.sort()
        out[char_id] = times
    return out


DEMO_MULTI_SCRIPT = """# Multi-character slapstick demo
canvas 960 540
fps 24
seconds 4.0
physics mode=off
camera enabled=true focus=char1 zoom=1.05 depth=true parallax=0.22 shake_on_impact=true

character id=char1 seed "DUO-ALPHA" line 7 limb 10 head 28 jitter 1.0 mode mesh mesh "assets/meshes/default/mesh.json" tint "#2f2f2f"
character id=char2 seed "DUO-BETA" line 7 limb 10 head 28 jitter 1.05 mode stick

char1: walk from=170,390 to=790,390 speed=1.05 bounce=0.32 stride=58 step_height=23 cadence=1.7
char1: wave hand=right cycles=2 amplitude=26 start=0.8 duration=1.6

char2: chase target=char1 offset=-50 aggression=1.35 bounce=0.38 stride=60 step_height=24 cadence=1.9
char2: anticipation t=1.0 action=sprint intensity=1.0 duration=0.22

duel_collision distance=48 force=1.05 duration=0.2 take_intensity=0.72 cooldown=0.35

wall x=640 width=20 height=160 force=0.95 duration=0.2
trapdoor x=735 width=112 depth=82 force=0.9 duration=0.24 open_time=0.24
anvil x=520 size=36 trigger_x=470 trigger_radius=55 delay=0.08 fall_speed=390 force=1.0 duration=0.2
"""


def write_demo(path: Path) -> None:
    path.write_text(DEMO_MULTI_SCRIPT, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-character orchestrator for slapstick duels")
    ap.add_argument("--script", type=str, default="", help="Path to multi-character DSL script.")
    ap.add_argument("--out", type=str, default="out_multi", help="Output folder for SVG frames.")
    ap.add_argument("--demo", action="store_true", help="Write multi_character_demo.txt and exit.")
    ap.add_argument("--fps", type=int, default=0, help="Override script fps.")
    ap.add_argument("--seconds", type=float, default=0.0, help="Override script duration.")
    args = ap.parse_args()

    if args.demo:
        demo_path = Path("multi_character_demo.txt")
        write_demo(demo_path)
        print(f"[ok] multi-character demo script written: {demo_path.resolve()}")
        return 0

    if args.script:
        text = Path(args.script).read_text(encoding="utf-8")
    else:
        demo_path = Path("multi_character_demo.txt")
        if not demo_path.exists():
            write_demo(demo_path)
        text = demo_path.read_text(encoding="utf-8")

    cfg, programs, props, collision = parse_multi_script(text)
    if args.fps > 0:
        cfg.fps = args.fps
    if args.seconds > 0.0:
        cfg.seconds = args.seconds

    timelines, prop_runtime, collision_events, prop_events_by_char, physics_meta = generate_multi_character_scene_detailed(
        cfg=cfg,
        programs=programs,
        props=props,
        collision=collision,
    )
    styles = {char_id: program.style for char_id, program in programs.items()}
    char_order = list(programs.keys())
    impact_times_by_char = collect_impact_times_by_char(
        programs=programs,
        prop_events_by_char=prop_events_by_char,
        collision_events=collision_events,
    )
    camera_impact_times = sorted({t for rows in impact_times_by_char.values() for t in rows})
    prop_items_fn = build_prop_items_fn(prop_runtime, cfg)
    out_dir = Path(args.out)
    render_multi_frames(
        cfg=cfg,
        timelines=timelines,
        styles=styles,
        out_dir=out_dir,
        character_order=char_order,
        extra_items_fn=prop_items_fn,
        camera_impact_times=camera_impact_times,
    )

    total_frames = int(round(cfg.seconds * cfg.fps))
    total_collision_events = sum(len(v) for v in collision_events.values())
    print(
        f"[ok] rendered multi-character scene: {out_dir.resolve()} "
        f"({len(programs)} characters, {total_frames} frames, {total_collision_events} collision events)"
    )
    if physics_meta:
        enabled = sum(1 for m in physics_meta.values() if m.get("enabled"))
        if enabled > 0:
            solver_counts: dict[str, int] = {}
            for row in physics_meta.values():
                solver = str(row.get("solver", "off"))
                solver_counts[solver] = solver_counts.get(solver, 0) + 1
            solver_info = ", ".join(f"{k}:{v}" for k, v in sorted(solver_counts.items()))
            print(f"[ok] physics enabled for {enabled}/{len(physics_meta)} chars ({solver_info})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
