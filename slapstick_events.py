#!/usr/bin/env python3
"""
slapstick_events.py

Non-linear, event-triggered motion modifiers for procedural cartoon timelines.

Supported event types:
- impact: short squash + rebound impulse
- take: quick startled jump with a short hold
- anticipation: opposite-direction preload before an action
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from cartoon_svg_mvp import Pose, Timeline, Vec2


NUMBER = r"-?[0-9]*\.?[0-9]+"
VEC2 = rf"{NUMBER},{NUMBER}"

IMPACT_RE = re.compile(
    rf"""^impact\s+
        t=(?P<t>[0-9]*\.?[0-9]+)
        (?:\s+direction=(?P<direction>{VEC2}))?
        (?:\s+force=(?P<force>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

TAKE_RE = re.compile(
    r"""^take\s+
        t=(?P<t>[0-9]*\.?[0-9]+)
        (?:\s+intensity=(?P<intensity>[0-9]*\.?[0-9]+))?
        (?:\s+hold=(?P<hold>\d+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

ANTICIPATION_RE = re.compile(
    rf"""^anticipation\s+
        t=(?P<t>[0-9]*\.?[0-9]+)
        (?:\s+action=(?P<action>[a-zA-Z_][a-zA-Z0-9_]*))?
        (?:\s+intensity=(?P<intensity>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        (?:\s+direction=(?P<direction>{VEC2}))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(slots=True)
class ImpactEvent:
    t: float
    force: float = 1.0
    direction: Vec2 | None = None
    duration: float = 0.22


@dataclass(slots=True)
class TakeEvent:
    t: float
    intensity: float = 1.0
    hold_frames: int = 2


@dataclass(slots=True)
class AnticipationEvent:
    t: float
    action: str = "move"
    intensity: float = 1.0
    duration: float = 0.24
    direction: Vec2 | None = None


SlapstickEvent = ImpactEvent | TakeEvent | AnticipationEvent


def parse_vec2(s: str) -> Vec2:
    x_str, y_str = s.split(",")
    return Vec2(float(x_str), float(y_str))


def parse_slapstick_event_line(line: str) -> SlapstickEvent | None:
    """
    Parse a DSL line into a slapstick event.

    Returns:
      - Event object if line is a valid event line
      - None if line is not an event command
    Raises:
      - ValueError if line starts with an event keyword but has invalid syntax
    """
    stripped = line.strip()
    if not stripped:
        return None

    keyword = stripped.split()[0].lower()
    if keyword not in {"impact", "take", "anticipation"}:
        return None

    if m := IMPACT_RE.match(stripped):
        direction = parse_vec2(m.group("direction")) if m.group("direction") is not None else None
        return ImpactEvent(
            t=float(m.group("t")),
            direction=direction,
            force=float(m.group("force") or 1.0),
            duration=float(m.group("duration") or 0.22),
        )

    if m := TAKE_RE.match(stripped):
        return TakeEvent(
            t=float(m.group("t")),
            intensity=float(m.group("intensity") or 1.0),
            hold_frames=int(m.group("hold") or 2),
        )

    if m := ANTICIPATION_RE.match(stripped):
        direction = parse_vec2(m.group("direction")) if m.group("direction") is not None else None
        return AnticipationEvent(
            t=float(m.group("t")),
            action=(m.group("action") or "move").lower(),
            intensity=float(m.group("intensity") or 1.0),
            duration=float(m.group("duration") or 0.24),
            direction=direction,
        )

    raise ValueError(f"Invalid slapstick event syntax: {line}")


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def copy_pose(p: Pose) -> Pose:
    return Pose(
        root=Vec2(p.root.x, p.root.y),
        l_hand=Vec2(p.l_hand.x, p.l_hand.y),
        r_hand=Vec2(p.r_hand.x, p.r_hand.y),
        l_foot=Vec2(p.l_foot.x, p.l_foot.y),
        r_foot=Vec2(p.r_foot.x, p.r_foot.y),
        look_angle=p.look_angle,
        squash=p.squash,
    )


def apply_pose_delta(
    pose: Pose,
    *,
    root: Vec2 | None = None,
    l_hand: Vec2 | None = None,
    r_hand: Vec2 | None = None,
    l_foot: Vec2 | None = None,
    r_foot: Vec2 | None = None,
    look_delta: float = 0.0,
    squash_delta: float = 0.0,
) -> Pose:
    return Pose(
        root=pose.root + (root or Vec2(0.0, 0.0)),
        l_hand=pose.l_hand + (l_hand or Vec2(0.0, 0.0)),
        r_hand=pose.r_hand + (r_hand or Vec2(0.0, 0.0)),
        l_foot=pose.l_foot + (l_foot or Vec2(0.0, 0.0)),
        r_foot=pose.r_foot + (r_foot or Vec2(0.0, 0.0)),
        look_angle=pose.look_angle + look_delta,
        squash=clamp(pose.squash + squash_delta, 0.0, 1.0),
    )


def infer_motion_direction(poses: list[Pose], fallback: Vec2) -> Vec2:
    if len(poses) < 2:
        return fallback
    delta = poses[-1].root - poses[0].root
    if delta.length() <= 1e-6:
        return fallback
    return delta.normalized()


def resolve_event_direction(explicit: Vec2 | None, default_dir: Vec2) -> Vec2:
    if explicit is None:
        return default_dir
    if explicit.length() <= 1e-6:
        return default_dir
    return explicit.normalized()


def apply_impact_event(
    poses: list[Pose],
    times: list[float],
    event: ImpactEvent,
    *,
    default_dir: Vec2,
) -> None:
    force = max(0.0, event.force)
    duration = max(0.05, event.duration)
    dir_ = resolve_event_direction(event.direction, default_dir)
    perp = dir_.perp()

    for i, t in enumerate(times):
        u = t - event.t
        if u < 0.0 or u > duration:
            continue
        phase = u / duration
        hit = math.exp(-((phase) / 0.22) ** 2)
        rebound = math.sin(phase * 2.0 * math.pi * 1.7) * math.exp(-phase * 2.5)

        root_delta = dir_ * (14.0 * force * rebound) + perp * (2.0 * force * rebound)
        hand_back = dir_ * (-18.0 * force * hit)
        foot_push = dir_ * (10.0 * force * hit)
        squash = 0.55 * force * hit + 0.16 * force * max(0.0, rebound)
        look = -0.18 * force * rebound

        poses[i] = apply_pose_delta(
            poses[i],
            root=root_delta,
            l_hand=hand_back,
            r_hand=hand_back,
            l_foot=foot_push,
            r_foot=foot_push,
            look_delta=look,
            squash_delta=squash,
        )


def apply_take_event(
    poses: list[Pose],
    times: list[float],
    event: TakeEvent,
) -> None:
    intensity = max(0.0, event.intensity)
    duration = max(0.12, 0.32 * intensity)
    hold_frames = max(0, event.hold_frames)

    for i, t in enumerate(times):
        u = t - event.t
        if u < 0.0 or u > duration:
            continue
        phase = u / duration
        pop = math.exp(-phase * 5.0)
        wobble = math.sin(phase * 2.0 * math.pi * 2.0) * math.exp(-phase * 3.0)

        root_delta = Vec2(0.0, -24.0 * intensity * pop)
        hand_l = Vec2(-4.0 * intensity * wobble, -20.0 * intensity * pop)
        hand_r = Vec2(+4.0 * intensity * wobble, -20.0 * intensity * pop)
        foot_delta = Vec2(0.0, -3.0 * intensity * pop)
        look = 0.14 * intensity * wobble
        squash = 0.32 * intensity * pop

        poses[i] = apply_pose_delta(
            poses[i],
            root=root_delta,
            l_hand=hand_l,
            r_hand=hand_r,
            l_foot=foot_delta,
            r_foot=foot_delta,
            look_delta=look,
            squash_delta=squash,
        )

    if hold_frames <= 0 or not times:
        return

    # Freeze a couple of frames after the take hit.
    nearest_idx = min(range(len(times)), key=lambda idx: abs(times[idx] - event.t))
    frozen_pose = copy_pose(poses[nearest_idx])
    for j in range(1, hold_frames + 1):
        idx = nearest_idx + j
        if idx >= len(poses):
            break
        poses[idx] = copy_pose(frozen_pose)


def action_scale(action: str) -> float:
    match action:
        case "sprint":
            return 1.28
        case "jump":
            return 1.22
        case "run":
            return 1.16
        case _:
            return 1.0


def apply_anticipation_event(
    poses: list[Pose],
    times: list[float],
    event: AnticipationEvent,
    *,
    default_dir: Vec2,
) -> None:
    intensity = max(0.0, event.intensity) * action_scale(event.action)
    duration = max(0.08, event.duration)
    release = max(0.08, duration * 0.7)
    dir_ = resolve_event_direction(event.direction, default_dir)

    start = event.t - duration
    end = event.t + release
    for i, t in enumerate(times):
        if t < start or t > end:
            continue

        if t <= event.t:
            # Pull back before the action.
            u = (t - start) / duration
            pull = math.sin(math.pi * u)
            root_delta = dir_ * (-20.0 * intensity * pull) + Vec2(0.0, +11.0 * intensity * pull)
            hand_l = dir_ * (+13.0 * intensity * pull) + Vec2(0.0, +3.0 * intensity * pull)
            hand_r = dir_ * (+13.0 * intensity * pull) + Vec2(0.0, +3.0 * intensity * pull)
            foot_l = dir_ * (-9.0 * intensity * pull)
            foot_r = dir_ * (-9.0 * intensity * pull)
            squash = 0.30 * intensity * pull
            look = -0.10 * intensity * pull
        else:
            # Release into the action.
            u = (t - event.t) / release
            boost = (1.0 - u) ** 2
            root_delta = dir_ * (+19.0 * intensity * boost) + Vec2(0.0, -10.0 * intensity * boost)
            hand_l = dir_ * (-10.0 * intensity * boost) + Vec2(0.0, -4.0 * intensity * boost)
            hand_r = dir_ * (-10.0 * intensity * boost) + Vec2(0.0, -4.0 * intensity * boost)
            foot_l = dir_ * (+8.0 * intensity * boost)
            foot_r = dir_ * (+8.0 * intensity * boost)
            squash = 0.12 * intensity * boost
            look = +0.08 * intensity * boost

        poses[i] = apply_pose_delta(
            poses[i],
            root=root_delta,
            l_hand=hand_l,
            r_hand=hand_r,
            l_foot=foot_l,
            r_foot=foot_r,
            look_delta=look,
            squash_delta=squash,
        )


def apply_slapstick_events(
    timeline: Timeline,
    *,
    events: list[SlapstickEvent],
    motion_direction: Vec2 | None = None,
) -> Timeline:
    """
    Apply event-based non-linear pose modifiers to a per-frame timeline.
    """
    if not events:
        return timeline
    if not timeline.keyframes:
        return timeline

    ordered = sorted(timeline.keyframes, key=lambda k: k.t)
    times = [k.t for k in ordered]
    poses = [copy_pose(k.pose) for k in ordered]

    fallback = motion_direction or Vec2(1.0, 0.0)
    if fallback.length() <= 1e-6:
        fallback = Vec2(1.0, 0.0)
    default_dir = infer_motion_direction(poses, fallback.normalized())

    for event in sorted(events, key=lambda e: e.t):
        if isinstance(event, ImpactEvent):
            apply_impact_event(poses, times, event, default_dir=default_dir)
        elif isinstance(event, TakeEvent):
            apply_take_event(poses, times, event)
        elif isinstance(event, AnticipationEvent):
            apply_anticipation_event(poses, times, event, default_dir=default_dir)
        else:
            raise ValueError(f"Unsupported event type: {type(event)!r}")

    out = Timeline()
    for t, pose in zip(times, poses):
        out.add(t, pose)
    return out
