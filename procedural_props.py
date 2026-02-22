#!/usr/bin/env python3
"""
procedural_props.py

Environment props + interaction logic for procedural cartoon scenes.

Current scope:
- Wall obstacle (triggers impact on crossing)
- Trapdoor (triggers downward impact on entry)
- Falling anvil (triggers downward impact after proximity trigger)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Callable

from cartoon_svg_mvp import (
    Keyframe,
    ScriptConfig,
    SvgCircle,
    SvgPath,
    SvgStyle,
    Timeline,
    Vec2,
)
from slapstick_events import ImpactEvent, SlapstickEvent, TakeEvent


WALL_RE = re.compile(
    r"""^wall\s+
        x=(?P<x>-?[0-9]*\.?[0-9]+)
        (?:\s+width=(?P<width>[0-9]*\.?[0-9]+))?
        (?:\s+height=(?P<height>[0-9]*\.?[0-9]+))?
        (?:\s+force=(?P<force>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

TRAPDOOR_RE = re.compile(
    r"""^trapdoor\s+
        x=(?P<x>-?[0-9]*\.?[0-9]+)
        (?:\s+width=(?P<width>[0-9]*\.?[0-9]+))?
        (?:\s+depth=(?P<depth>[0-9]*\.?[0-9]+))?
        (?:\s+force=(?P<force>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        (?:\s+open_time=(?P<open_time>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

ANVIL_RE = re.compile(
    r"""^anvil\s+
        x=(?P<x>-?[0-9]*\.?[0-9]+)
        (?:\s+size=(?P<size>[0-9]*\.?[0-9]+))?
        (?:\s+trigger_x=(?P<trigger_x>-?[0-9]*\.?[0-9]+))?
        (?:\s+trigger_radius=(?P<trigger_radius>[0-9]*\.?[0-9]+))?
        (?:\s+delay=(?P<delay>[0-9]*\.?[0-9]+))?
        (?:\s+fall_speed=(?P<fall_speed>[0-9]*\.?[0-9]+))?
        (?:\s+force=(?P<force>[0-9]*\.?[0-9]+))?
        (?:\s+duration=(?P<duration>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(slots=True)
class WallProp:
    id: str
    x: float
    width: float = 18.0
    height: float = 145.0
    force: float = 0.95
    duration: float = 0.22


@dataclass(slots=True)
class TrapDoorProp:
    id: str
    x: float
    width: float = 90.0
    depth: float = 72.0
    force: float = 0.82
    duration: float = 0.26
    open_time: float = 0.28


@dataclass(slots=True)
class AnvilProp:
    id: str
    x: float
    size: float = 34.0
    trigger_x: float = 0.0
    trigger_radius: float = 46.0
    delay: float = 0.11
    fall_speed: float = 360.0
    force: float = 1.08
    duration: float = 0.2


SceneProp = WallProp | TrapDoorProp | AnvilProp


@dataclass(slots=True)
class PropRuntimeState:
    props: list[SceneProp]
    trigger_times: dict[str, float]
    impact_times: dict[str, float]
    ground_y: float


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_prop_id(kind: str, index: int) -> str:
    return f"{kind}_{index:03d}"


def parse_scene_prop_line(line: str, *, index: int) -> SceneProp | None:
    stripped = line.strip()
    if not stripped:
        return None

    keyword = stripped.split()[0].lower()
    if keyword not in {"wall", "trapdoor", "anvil"}:
        return None

    if m := WALL_RE.match(stripped):
        return WallProp(
            id=make_prop_id("wall", index),
            x=float(m.group("x")),
            width=float(m.group("width") or 18.0),
            height=float(m.group("height") or 145.0),
            force=float(m.group("force") or 0.95),
            duration=float(m.group("duration") or 0.22),
        )

    if m := TRAPDOOR_RE.match(stripped):
        return TrapDoorProp(
            id=make_prop_id("trapdoor", index),
            x=float(m.group("x")),
            width=float(m.group("width") or 90.0),
            depth=float(m.group("depth") or 72.0),
            force=float(m.group("force") or 0.82),
            duration=float(m.group("duration") or 0.26),
            open_time=float(m.group("open_time") or 0.28),
        )

    if m := ANVIL_RE.match(stripped):
        x = float(m.group("x"))
        trigger_x = float(m.group("trigger_x") or x)
        return AnvilProp(
            id=make_prop_id("anvil", index),
            x=x,
            size=float(m.group("size") or 34.0),
            trigger_x=trigger_x,
            trigger_radius=float(m.group("trigger_radius") or 46.0),
            delay=float(m.group("delay") or 0.11),
            fall_speed=float(m.group("fall_speed") or 360.0),
            force=float(m.group("force") or 1.08),
            duration=float(m.group("duration") or 0.2),
        )

    raise ValueError(f"Invalid prop syntax: {line}")


def prop_to_scene_line(prop: SceneProp) -> str:
    if isinstance(prop, WallProp):
        return (
            f"wall x={prop.x:.3f} width={prop.width:.3f} height={prop.height:.3f} "
            f"force={prop.force:.3f} duration={prop.duration:.3f}"
        )
    if isinstance(prop, TrapDoorProp):
        return (
            f"trapdoor x={prop.x:.3f} width={prop.width:.3f} depth={prop.depth:.3f} "
            f"force={prop.force:.3f} duration={prop.duration:.3f} open_time={prop.open_time:.3f}"
        )
    if isinstance(prop, AnvilProp):
        return (
            f"anvil x={prop.x:.3f} size={prop.size:.3f} trigger_x={prop.trigger_x:.3f} "
            f"trigger_radius={prop.trigger_radius:.3f} delay={prop.delay:.3f} "
            f"fall_speed={prop.fall_speed:.3f} force={prop.force:.3f} duration={prop.duration:.3f}"
        )
    raise RuntimeError(f"unsupported prop type: {type(prop)!r}")


def prop_to_meta(prop: SceneProp) -> dict:
    if isinstance(prop, WallProp):
        out = asdict(prop)
        out["type"] = "wall"
        return out
    if isinstance(prop, TrapDoorProp):
        out = asdict(prop)
        out["type"] = "trapdoor"
        return out
    if isinstance(prop, AnvilProp):
        out = asdict(prop)
        out["type"] = "anvil"
        return out
    raise RuntimeError(f"unsupported prop type: {type(prop)!r}")


def _cross_time_x(keys: list[Keyframe], x: float) -> float | None:
    for i in range(len(keys) - 1):
        a = keys[i]
        b = keys[i + 1]
        da = a.pose.root.x - x
        db = b.pose.root.x - x
        if abs(da) <= 1e-9:
            return a.t
        if da * db > 0.0:
            continue
        denom = b.pose.root.x - a.pose.root.x
        if abs(denom) <= 1e-9:
            continue
        u = clamp((x - a.pose.root.x) / denom, 0.0, 1.0)
        return a.t + (b.t - a.t) * u
    return None


def _enter_range_time(keys: list[Keyframe], low: float, high: float) -> float | None:
    if low > high:
        low, high = high, low
    for i in range(len(keys) - 1):
        a = keys[i]
        b = keys[i + 1]
        xa = a.pose.root.x
        xb = b.pose.root.x
        inside_a = low <= xa <= high
        inside_b = low <= xb <= high
        if inside_a:
            return a.t
        if not inside_b:
            continue
        if abs(xb - xa) <= 1e-9:
            return b.t
        boundary = low if xb > xa else high
        u = clamp((boundary - xa) / (xb - xa), 0.0, 1.0)
        return a.t + (b.t - a.t) * u
    return None


def _infer_direction(keys: list[Keyframe]) -> Vec2:
    if len(keys) < 2:
        return Vec2(1.0, 0.0)
    d = keys[-1].pose.root - keys[0].pose.root
    if d.length() <= 1e-6:
        return Vec2(1.0, 0.0)
    return d.normalized()


def _anvil_times(prop: AnvilProp, ground_y: float, trigger_t: float) -> tuple[float, float]:
    start_center_y = -prop.size
    hit_center_y = ground_y - prop.size * 0.6
    dist = max(10.0, hit_center_y - start_center_y)
    fall_t = dist / max(40.0, prop.fall_speed)
    drop_t = trigger_t + max(0.0, prop.delay)
    hit_t = drop_t + fall_t
    return drop_t, hit_t


def derive_prop_events(
    timeline: Timeline,
    *,
    cfg: ScriptConfig,
    props: list[SceneProp],
) -> tuple[list[SlapstickEvent], PropRuntimeState]:
    if not timeline.keyframes:
        return [], PropRuntimeState(props=props, trigger_times={}, impact_times={}, ground_y=cfg.height - 60)

    keys = sorted(timeline.keyframes, key=lambda k: k.t)
    direction = _infer_direction(keys)
    direction_x_sign = 1.0 if direction.x >= 0.0 else -1.0
    last_t = keys[-1].t
    ground_y = cfg.height - 60

    events: list[SlapstickEvent] = []
    trigger_times: dict[str, float] = {}
    impact_times: dict[str, float] = {}

    for prop in props:
        if isinstance(prop, WallProp):
            hit_t = _cross_time_x(keys, prop.x)
            if hit_t is None:
                continue
            dir_ = Vec2(-direction_x_sign, 0.0)
            events.append(
                ImpactEvent(
                    t=hit_t,
                    direction=dir_,
                    force=max(0.1, prop.force),
                    duration=max(0.05, prop.duration),
                )
            )
            events.append(TakeEvent(t=min(last_t, hit_t + 0.05), intensity=min(1.4, prop.force * 0.7), hold_frames=1))
            trigger_times[prop.id] = hit_t
            impact_times[prop.id] = hit_t
            continue

        if isinstance(prop, TrapDoorProp):
            low = prop.x - prop.width * 0.5
            high = prop.x + prop.width * 0.5
            enter_t = _enter_range_time(keys, low, high)
            if enter_t is None:
                continue
            impact_t = min(last_t, enter_t + 0.03)
            events.append(
                ImpactEvent(
                    t=impact_t,
                    direction=Vec2(0.0, 1.0),
                    force=max(0.1, prop.force),
                    duration=max(0.06, prop.duration),
                )
            )
            events.append(TakeEvent(t=min(last_t, impact_t + 0.04), intensity=min(1.2, prop.force * 0.75), hold_frames=1))
            trigger_times[prop.id] = enter_t
            impact_times[prop.id] = impact_t
            continue

        if isinstance(prop, AnvilProp):
            trigger_t = _enter_range_time(
                keys,
                prop.trigger_x - prop.trigger_radius,
                prop.trigger_x + prop.trigger_radius,
            )
            if trigger_t is None:
                continue
            drop_t, hit_t = _anvil_times(prop, ground_y, trigger_t)
            if hit_t > last_t + 0.25:
                # Keep animation bounded by timeline horizon.
                continue
            hit_t = min(last_t, hit_t)
            events.append(
                ImpactEvent(
                    t=hit_t,
                    direction=Vec2(0.0, 1.0),
                    force=max(0.1, prop.force),
                    duration=max(0.05, prop.duration),
                )
            )
            events.append(TakeEvent(t=min(last_t, hit_t + 0.05), intensity=min(1.4, prop.force * 0.82), hold_frames=2))
            trigger_times[prop.id] = trigger_t
            impact_times[prop.id] = hit_t
            continue

    events.sort(key=lambda e: e.t)
    runtime = PropRuntimeState(
        props=props,
        trigger_times=trigger_times,
        impact_times=impact_times,
        ground_y=ground_y,
    )
    return events, runtime


def _rect_path(x: float, y_top: float, width: float, height: float) -> str:
    x0 = x - width * 0.5
    x1 = x + width * 0.5
    y0 = y_top
    y1 = y_top + height
    return (
        f"M {x0:.2f} {y0:.2f} "
        f"L {x1:.2f} {y0:.2f} "
        f"L {x1:.2f} {y1:.2f} "
        f"L {x0:.2f} {y1:.2f} Z"
    )


def _draw_wall(prop: WallProp, ground_y: float) -> list[SvgPath | SvgCircle]:
    top = ground_y - prop.height
    style = SvgStyle(stroke="#000", stroke_width=3.0, fill="none")
    crack_style = SvgStyle(stroke="#000", stroke_width=2.0, fill="none")
    items: list[SvgPath | SvgCircle] = [
        SvgPath(d=_rect_path(prop.x, top, prop.width, prop.height), style=style, id=f"{prop.id}_wall")
    ]
    items.append(
        SvgPath(
            d=f"M {prop.x - prop.width * 0.25:.2f} {top + prop.height * 0.35:.2f} "
            f"L {prop.x + prop.width * 0.20:.2f} {top + prop.height * 0.55:.2f}",
            style=crack_style,
            id=f"{prop.id}_crack",
        )
    )
    return items


def _draw_trapdoor(prop: TrapDoorProp, ground_y: float, trigger_t: float | None, t: float) -> list[SvgPath | SvgCircle]:
    style = SvgStyle(stroke="#000", stroke_width=4.0, fill="none")
    x0 = prop.x - prop.width * 0.5
    x1 = prop.x + prop.width * 0.5

    if trigger_t is None or t < trigger_t:
        return [SvgPath(d=f"M {x0:.2f} {ground_y:.2f} L {x1:.2f} {ground_y:.2f}", style=style, id=f"{prop.id}_closed")]

    open_u = clamp((t - trigger_t) / max(0.08, prop.open_time), 0.0, 1.0)
    drop = prop.depth * open_u
    left = SvgPath(
        d=f"M {x0:.2f} {ground_y:.2f} L {prop.x:.2f} {ground_y + drop:.2f}",
        style=style,
        id=f"{prop.id}_left",
    )
    right = SvgPath(
        d=f"M {x1:.2f} {ground_y:.2f} L {prop.x:.2f} {ground_y + drop:.2f}",
        style=style,
        id=f"{prop.id}_right",
    )
    return [left, right]


def _draw_anvil(prop: AnvilProp, ground_y: float, trigger_t: float | None, t: float) -> list[SvgPath | SvgCircle]:
    style = SvgStyle(stroke="#000", stroke_width=3.0, fill="none")
    detail = SvgStyle(stroke="#000", stroke_width=2.0, fill="none")
    start_center_y = -prop.size
    hit_center_y = ground_y - prop.size * 0.6

    if trigger_t is None:
        center_y = start_center_y
    else:
        drop_t, _ = _anvil_times(prop, ground_y, trigger_t)
        if t <= drop_t:
            center_y = start_center_y
        else:
            center_y = min(hit_center_y, start_center_y + prop.fall_speed * (t - drop_t))

    top = center_y - prop.size * 0.5
    body = SvgPath(d=_rect_path(prop.x, top, prop.size, prop.size), style=style, id=f"{prop.id}_body")
    top_line = SvgPath(
        d=f"M {prop.x - prop.size * 0.25:.2f} {top - prop.size * 0.15:.2f} "
        f"L {prop.x + prop.size * 0.25:.2f} {top - prop.size * 0.15:.2f}",
        style=detail,
        id=f"{prop.id}_rim",
    )
    rivet = SvgCircle(
        cx=prop.x,
        cy=top + prop.size * 0.5,
        r=max(2.0, prop.size * 0.08),
        style=SvgStyle(stroke="#000", stroke_width=2.0, fill="#000"),
        id=f"{prop.id}_rivet",
    )
    return [body, top_line, rivet]


def build_prop_items_fn(
    runtime: PropRuntimeState | None,
    cfg: ScriptConfig,
) -> Callable[[int, float], list[SvgPath | SvgCircle]] | None:
    if runtime is None or not runtime.props:
        return None

    ground_y = cfg.height - 60

    def items_fn(_frame_index: int, t: float) -> list[SvgPath | SvgCircle]:
        items: list[SvgPath | SvgCircle] = []
        for prop in runtime.props:
            if isinstance(prop, WallProp):
                items.extend(_draw_wall(prop, ground_y))
            elif isinstance(prop, TrapDoorProp):
                items.extend(_draw_trapdoor(prop, ground_y, runtime.trigger_times.get(prop.id), t))
            elif isinstance(prop, AnvilProp):
                items.extend(_draw_anvil(prop, ground_y, runtime.trigger_times.get(prop.id), t))
            else:
                raise RuntimeError(f"unsupported prop type: {type(prop)!r}")
        return items

    return items_fn
