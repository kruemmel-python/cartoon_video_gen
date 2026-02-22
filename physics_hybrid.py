#!/usr/bin/env python3
"""
physics_hybrid.py

Hybrid kinematic/physics pass for procedural cartoon timelines.

Design goals:
- Keep deterministic DSL/rig pipeline as primary source of motion.
- Activate short physics windows (ragdoll-like) around impact events.
- Work without external deps (fallback solver).
- Use pymunk automatically when available (or required via mode).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from cartoon_svg_mvp import Pose, ScriptConfig, Timeline, Vec2
from slapstick_events import ImpactEvent, SlapstickEvent


PHYSICS_MODE_CHOICES: tuple[str, ...] = ("off", "fallback", "pymunk", "auto")


@dataclass(slots=True)
class PhysicsConfig:
    mode: str = "off"  # off | fallback | pymunk | auto
    gravity: float = 980.0
    damping: float = 0.92
    restitution: float = 0.22
    friction: float = 0.85
    impulse_scale: float = 320.0
    ragdoll_extra: float = 0.35
    substeps: int = 2


def physics_config_from_script(cfg: ScriptConfig) -> PhysicsConfig:
    return PhysicsConfig(
        mode=cfg.physics_mode.lower(),
        gravity=cfg.physics_gravity,
        damping=cfg.physics_damping,
        restitution=cfg.physics_restitution,
        friction=cfg.physics_friction,
        impulse_scale=cfg.physics_impulse_scale,
        ragdoll_extra=cfg.physics_ragdoll_extra,
        substeps=cfg.physics_substeps,
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp2(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(lerp(a.x, b.x, t), lerp(a.y, b.y, t))


def vec_len(v: Vec2) -> float:
    return math.hypot(v.x, v.y)


def vec_norm(v: Vec2) -> Vec2:
    n = vec_len(v)
    if n <= 1e-9:
        return Vec2(1.0, 0.0)
    return Vec2(v.x / n, v.y / n)


def _impact_events(events: Iterable[SlapstickEvent] | None) -> list[ImpactEvent]:
    if not events:
        return []
    out = [e for e in events if isinstance(e, ImpactEvent)]
    out.sort(key=lambda e: e.t)
    return out


class _FallbackBody:
    __slots__ = ("pos", "vel", "cfg", "ground_y")

    def __init__(self, pos: Vec2, cfg: PhysicsConfig, ground_y: float) -> None:
        self.pos = Vec2(pos.x, pos.y)
        self.vel = Vec2(0.0, 0.0)
        self.cfg = cfg
        self.ground_y = ground_y

    def apply_impulse(self, impulse: Vec2) -> None:
        self.vel = self.vel + impulse

    def step(self, dt: float) -> tuple[Vec2, Vec2]:
        substeps = max(1, int(self.cfg.substeps))
        sub_dt = dt / substeps
        for _ in range(substeps):
            self.vel = Vec2(self.vel.x, self.vel.y + self.cfg.gravity * sub_dt)
            # Keep damping stable across fps ranges.
            damp = clamp(self.cfg.damping, 0.0, 1.0)
            self.vel = Vec2(self.vel.x * damp, self.vel.y * damp)
            self.pos = self.pos + self.vel * sub_dt

            if self.pos.y > self.ground_y:
                self.pos = Vec2(self.pos.x, self.ground_y)
                if self.vel.y > 0.0:
                    self.vel = Vec2(
                        self.vel.x * (1.0 - clamp(self.cfg.friction, 0.0, 1.0) * 0.15),
                        -self.vel.y * clamp(self.cfg.restitution, 0.0, 1.0),
                    )

        speed = vec_len(self.vel)
        if speed > 1800.0:
            self.vel = vec_norm(self.vel) * 1800.0
        return self.pos, self.vel


class _PymunkBody:
    __slots__ = ("space", "body", "cfg")

    def __init__(self, pos: Vec2, cfg: PhysicsConfig, ground_y: float) -> None:
        import pymunk  # type: ignore

        self.cfg = cfg
        self.space = pymunk.Space()
        self.space.gravity = (0.0, cfg.gravity)

        body = pymunk.Body(mass=1.0, moment=1200.0)
        body.position = (pos.x, pos.y)
        shape = pymunk.Circle(body, radius=14.0)
        shape.elasticity = clamp(cfg.restitution, 0.0, 1.0)
        shape.friction = clamp(cfg.friction, 0.0, 1.0)
        self.space.add(body, shape)

        ground = pymunk.Segment(
            self.space.static_body,
            (-20000.0, ground_y),
            (20000.0, ground_y),
            2.0,
        )
        ground.elasticity = clamp(cfg.restitution, 0.0, 1.0)
        ground.friction = clamp(cfg.friction, 0.0, 1.0)
        self.space.add(ground)
        self.body = body

    def apply_impulse(self, impulse: Vec2) -> None:
        self.body.apply_impulse_at_local_point((impulse.x, impulse.y))

    def step(self, dt: float) -> tuple[Vec2, Vec2]:
        substeps = max(1, int(self.cfg.substeps))
        sub_dt = dt / substeps
        self.body.velocity = (
            self.body.velocity.x * clamp(self.cfg.damping, 0.0, 1.0),
            self.body.velocity.y * clamp(self.cfg.damping, 0.0, 1.0),
        )
        for _ in range(substeps):
            self.space.step(sub_dt)
        pos = Vec2(float(self.body.position.x), float(self.body.position.y))
        vel = Vec2(float(self.body.velocity.x), float(self.body.velocity.y))
        speed = vec_len(vel)
        if speed > 1800.0:
            vel = vec_norm(vel) * 1800.0
            self.body.velocity = (vel.x, vel.y)
        return pos, vel


def _resolve_solver(mode: str) -> str:
    mode = mode.lower().strip()
    if mode not in PHYSICS_MODE_CHOICES:
        raise ValueError(f"invalid physics mode '{mode}', expected one of: {', '.join(PHYSICS_MODE_CHOICES)}")
    if mode == "off":
        return "off"
    if mode == "fallback":
        return "fallback"
    if mode == "pymunk":
        try:
            import pymunk  # type: ignore  # noqa: F401
        except Exception as exc:
            raise RuntimeError("physics mode 'pymunk' requested but pymunk is not installed") from exc
        return "pymunk"
    # auto
    try:
        import pymunk  # type: ignore  # noqa: F401

        return "pymunk"
    except Exception:
        return "fallback"


def _apply_limb_inertia(base: Pose, root_vel: Vec2, ragdoll_mix: float) -> Pose:
    # Keep baseline articulation but add velocity-dependent drag.
    drag = clamp(vec_len(root_vel) / 520.0, 0.0, 1.0) * ragdoll_mix
    if drag <= 1e-6:
        return base

    hand_drag = Vec2(-root_vel.x * 0.035 * drag, clamp(-root_vel.y * 0.02 * drag, -24.0, 24.0))
    foot_drag = Vec2(-root_vel.x * 0.018 * drag, clamp(-root_vel.y * 0.012 * drag, -14.0, 14.0))
    look_delta = clamp(root_vel.x / 1200.0, -0.15, 0.15) * drag
    squash = clamp(base.squash + clamp(abs(root_vel.y) / 900.0, 0.0, 0.34) * drag, 0.0, 1.0)

    return Pose(
        root=base.root,
        l_hand=base.l_hand + hand_drag,
        r_hand=base.r_hand + hand_drag,
        l_foot=base.l_foot + foot_drag,
        r_foot=base.r_foot + foot_drag,
        look_angle=base.look_angle + look_delta,
        squash=squash,
    )


def apply_hybrid_physics(
    timeline: Timeline,
    *,
    cfg: ScriptConfig,
    events: Iterable[SlapstickEvent] | None = None,
    ground_y: float | None = None,
) -> tuple[Timeline, dict[str, Any]]:
    """
    Apply short physics control windows around impact events.

    Returns:
      (timeline_after_physics, metadata)
    """
    impacts = _impact_events(events)
    mode = physics_config_from_script(cfg)
    solver = _resolve_solver(mode.mode)

    if solver == "off":
        return timeline, {
            "enabled": False,
            "mode_requested": mode.mode,
            "solver": "off",
            "impact_events": len(impacts),
            "ragdoll_frames": 0,
        }

    if not timeline.keyframes:
        return timeline, {
            "enabled": False,
            "mode_requested": mode.mode,
            "solver": solver,
            "impact_events": len(impacts),
            "ragdoll_frames": 0,
        }

    if not impacts:
        return timeline, {
            "enabled": True,
            "mode_requested": mode.mode,
            "solver": solver,
            "impact_events": 0,
            "ragdoll_frames": 0,
        }

    times = [k.t for k in sorted(timeline.keyframes, key=lambda k: k.t)]
    t_end = max(times[-1], cfg.seconds)
    fps = max(1, int(cfg.fps))
    dt = 1.0 / fps
    total = int(math.ceil(t_end * fps)) + 1
    base_poses = [timeline.sample(min(t_end, i * dt)) for i in range(total)]

    floor_y = ground_y if ground_y is not None else (cfg.height - 60)
    body: _FallbackBody | _PymunkBody
    if solver == "pymunk":
        body = _PymunkBody(pos=base_poses[0].root, cfg=mode, ground_y=floor_y)
    else:
        body = _FallbackBody(pos=base_poses[0].root, cfg=mode, ground_y=floor_y)

    impact_idx = 0
    ragdoll_until = -1.0
    out = Timeline()
    ragdoll_frames = 0
    last_root = base_poses[0].root

    for i in range(total):
        t = min(t_end, i * dt)
        base = base_poses[i]

        while impact_idx < len(impacts) and impacts[impact_idx].t <= t + dt * 0.5:
            ev = impacts[impact_idx]
            direction = vec_norm(ev.direction) if ev.direction is not None else Vec2(1.0, 0.0)
            impulse = direction * (max(0.0, ev.force) * max(10.0, mode.impulse_scale))
            body.apply_impulse(impulse)
            ragdoll_until = max(ragdoll_until, ev.t + max(0.05, ev.duration) + max(0.0, mode.ragdoll_extra))
            impact_idx += 1

        sim_pos, sim_vel = body.step(dt)
        active = t <= ragdoll_until
        mix = 0.86 if active else 0.22
        root = lerp2(base.root, sim_pos, mix)
        if active:
            ragdoll_frames += 1

        # Keep feet from tunneling too deep below floor.
        if root.y > floor_y + 2.0:
            root = Vec2(root.x, floor_y + 2.0)

        # Blend by simulated velocity and local root delta to keep momentum visible.
        root_vel = Vec2((root.x - last_root.x) / max(dt, 1e-6), (root.y - last_root.y) / max(dt, 1e-6))
        root_vel = lerp2(root_vel, sim_vel, 0.55 if active else 0.25)
        posed = _apply_limb_inertia(base, root_vel=root_vel, ragdoll_mix=(1.0 if active else 0.45))
        posed = Pose(
            root=root,
            l_hand=posed.l_hand,
            r_hand=posed.r_hand,
            l_foot=posed.l_foot,
            r_foot=posed.r_foot,
            look_angle=posed.look_angle,
            squash=posed.squash,
        )
        out.add(t, posed)
        last_root = root

    return out, {
        "enabled": True,
        "mode_requested": mode.mode,
        "solver": solver,
        "impact_events": len(impacts),
        "ragdoll_frames": ragdoll_frames,
        "seconds": t_end,
    }

