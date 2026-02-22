#!/usr/bin/env python3
"""
cartoon_svg_mvp.py — SVG/Vektor-first Cartoon-Animation MVP (Python 3.12)

Ziel:
- Strichmännchen / Early-cartoon Look (schwarz/weiß)
- Rig + Rubber-hose Extremitäten
- DSL -> Keyframes -> Frames -> SVG-Sequenz
- Zeitkohärenter Ink-Jitter (kontrolliertes "lebendiges" Linienbild)

Keine externen Dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence
import argparse
import math
import re
import hashlib


# -----------------------------
# 1) Mathe: Vektor & Utility
# -----------------------------

@dataclass(frozen=True, slots=True)
class Vec2:
    """
    Minimaler 2D-Vektor.
    Wir halten es bewusst klein: add/sub/mul, Länge, Normalisieren.
    """
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    def __rmul__(self, k: float) -> "Vec2":
        return self.__mul__(k)

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        """
        Normalisieren mit Schutz gegen Division durch 0.
        """
        n = self.length()
        if n < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / n, self.y / n)

    def perp(self) -> "Vec2":
        """
        Senkrechter Vektor (90°).
        """
        return Vec2(-self.y, self.x)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp2(a: Vec2, b: Vec2, t: float) -> Vec2:
    return Vec2(lerp(a.x, b.x, t), lerp(a.y, b.y, t))


def smoothstep(t: float) -> float:
    """
    Klassisches smoothstep (S-Kurve), gute Default-Easing-Funktion.
    """
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_bool_token(v: str) -> bool:
    s = v.strip().lower()
    if s in {"1", "true", "on", "yes"}:
        return True
    if s in {"0", "false", "off", "no"}:
        return False
    raise ValueError(f"invalid bool token: {v}")


# -----------------------------------------
# 2) Zeitkohärentes "Ink-Jitter" (Noise)
# -----------------------------------------

def stable_hash_to_unit(seed: str) -> float:
    """
    Stabiler Hash -> float in [0, 1).
    Wichtig: unabhängig von Python's hash randomization.
    """
    h = hashlib.sha256(seed.encode("utf-8")).digest()
    # Wir nehmen 8 Bytes -> 64-bit int
    v = int.from_bytes(h[:8], "little", signed=False)
    return (v % (10**12)) / float(10**12)


def jitter_vec(
    *,
    base_seed: str,
    stroke_id: str,
    point_index: int,
    frame_index: int,
    amplitude: float,
) -> Vec2:
    """
    Liefert einen kleinen, zeitkohärenten Versatz (dx, dy).

    Warum "zeitkohärent"?
    - frame_index ist Teil des Seeds -> Bewegung über Zeit
    - stroke_id & point_index fixieren Identität pro Stroke/Kontrollpunkt

    amplitude ist in Pixeln (SVG User Units).
    """
    # Zwei unabhängige Werte für x und y
    sx = f"{base_seed}|{stroke_id}|p{point_index}|f{frame_index}|x"
    sy = f"{base_seed}|{stroke_id}|p{point_index}|f{frame_index}|y"
    ux = stable_hash_to_unit(sx) * 2.0 - 1.0  # [-1, 1]
    uy = stable_hash_to_unit(sy) * 2.0 - 1.0
    return Vec2(ux * amplitude, uy * amplitude)


# -----------------------------------------
# 3) SVG-Primitives: Path / Circle / Group
# -----------------------------------------

@dataclass(slots=True)
class SvgStyle:
    stroke: str = "#000"
    stroke_width: float = 6.0
    fill: str = "none"
    stroke_linecap: str = "round"
    stroke_linejoin: str = "round"


@dataclass(slots=True)
class SvgPath:
    """
    Ein SVG-Pfad als String 'd' plus Style.
    """
    d: str
    style: SvgStyle
    id: str = ""


@dataclass(slots=True)
class SvgCircle:
    cx: float
    cy: float
    r: float
    style: SvgStyle
    id: str = ""


@dataclass(slots=True)
class SvgGroup:
    items: list[SvgPath | SvgCircle | "SvgRaw"] = field(default_factory=list)


@dataclass(slots=True)
class SvgRaw:
    """
    Rohes SVG-Markup (z. B. geskinte externe Mesh-Parts).
    """
    xml: str
    id: str = ""


def svg_doc(
    *,
    width: int,
    height: int,
    items: Sequence[SvgPath | SvgCircle | SvgRaw],
) -> str:
    """
    Baut ein komplettes SVG-Dokument (monochrom, clean).
    """
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" shape-rendering="geometricPrecision">'
    )
    bg = '<rect x="0" y="0" width="100%" height="100%" fill="#fff"/>'

    body = []
    for it in items:
        if isinstance(it, SvgPath):
            st = it.style
            body.append(
                f'<path d="{it.d}" stroke="{st.stroke}" stroke-width="{st.stroke_width}" '
                f'fill="{st.fill}" stroke-linecap="{st.stroke_linecap}" '
                f'stroke-linejoin="{st.stroke_linejoin}"/>'
            )
        elif isinstance(it, SvgCircle):
            st = it.style
            body.append(
                f'<circle cx="{it.cx}" cy="{it.cy}" r="{it.r}" '
                f'stroke="{st.stroke}" stroke-width="{st.stroke_width}" '
                f'fill="{st.fill}" stroke-linecap="{st.stroke_linecap}" '
                f'stroke-linejoin="{st.stroke_linejoin}"/>'
            )
        else:  # SvgRaw
            body.append(it.xml)

    return header + bg + "".join(body) + "</svg>"


# -----------------------------------------
# 4) Rig + Cartoon-Körper (Rubber-hose)
# -----------------------------------------

@dataclass(slots=True)
class Pose:
    """
    Eine Pose ist der Zustand der Figur in einem Frame:
    - root: Basisposition (Hüfte / Körperzentrum)
    - look: Blickrichtung als Winkel (rad) (optional)
    - hand/foot targets relativ zum root (einfaches Control)
    - squash: 0..1 (optional)
    """
    root: Vec2
    l_hand: Vec2
    r_hand: Vec2
    l_foot: Vec2
    r_foot: Vec2
    look_angle: float = 0.0
    squash: float = 0.0


@dataclass(slots=True)
class CharacterStyle:
    """
    Stilparameter für frühen Cartoon-Look.
    """
    line_width: float = 7.0
    head_radius: float = 28.0
    limb_width: float = 10.0  # "hose thickness"
    jitter_amplitude: float = 1.2  # px
    seed: str = "INK-SEED-1"
    # Motion-smear tuning (velocity in px/s)
    smear_speed_threshold: float = 140.0
    smear_speed_full: float = 480.0
    smear_max_stretch: float = 0.45
    smear_max_squeeze: float = 0.28
    smear_jitter_boost: float = 1.0
    # Optional vector-skinning mode
    render_mode: str = "stick"  # stick | mesh
    mesh_asset: str = ""
    mesh_tint: str = ""


@dataclass(slots=True)
class CameraConfig:
    """
    2.5D camera setup for parallax/depth rendering.
    """
    enabled: bool = False
    focus: str = ""  # "", "self", "char1", "char2", ...
    zoom: float = 1.0
    pan: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))
    depth_enabled: bool = True
    depth_min_scale: float = 0.82
    depth_max_scale: float = 1.18
    parallax_strength: float = 0.20
    y_sort: bool = True
    shake_on_impact: bool = True
    shake_amplitude: float = 14.0
    shake_frequency: float = 14.0
    shake_decay: float = 4.8


@dataclass(slots=True)
class CharacterRig:
    """
    Ableitung von Ankerpunkten aus Pose.
    Das ist bewusst simpel, aber stabil.
    """
    root: Vec2
    neck: Vec2
    head_center: Vec2
    l_shoulder: Vec2
    r_shoulder: Vec2
    l_hip: Vec2
    r_hip: Vec2

    l_hand: Vec2
    r_hand: Vec2
    l_foot: Vec2
    r_foot: Vec2


@dataclass(slots=True)
class CharacterRenderState:
    """
    Render-Input für Multi-Character-Frames.
    """
    char_id: str
    pose: Pose
    style: CharacterStyle
    velocity: Vec2 = field(default_factory=lambda: Vec2(0.0, 0.0))


def build_rig(p: Pose) -> CharacterRig:
    """
    Baut ein rig aus einer Pose.

    Idee:
    - root ist Hüftzentrum
    - torso ist ein kleiner vertikaler Offset
    - Schultern und Hüften liegen links/rechts vom Zentrum
    - Hand/Fuß-Ziele kommen direkt aus Pose (relativ zum root)
    """
    torso_up = Vec2(0.0, -70.0)
    neck = p.root + torso_up
    head_center = neck + Vec2(0.0, -35.0)

    shoulder_spread = 34.0
    hip_spread = 26.0

    l_shoulder = neck + Vec2(-shoulder_spread, 0.0)
    r_shoulder = neck + Vec2(+shoulder_spread, 0.0)

    l_hip = p.root + Vec2(-hip_spread, 0.0)
    r_hip = p.root + Vec2(+hip_spread, 0.0)

    return CharacterRig(
        root=p.root,
        neck=neck,
        head_center=head_center,
        l_shoulder=l_shoulder,
        r_shoulder=r_shoulder,
        l_hip=l_hip,
        r_hip=r_hip,
        l_hand=p.root + p.l_hand,
        r_hand=p.root + p.r_hand,
        l_foot=p.root + p.l_foot,
        r_foot=p.root + p.r_foot,
    )


def quadratic_bezier_path(p0: Vec2, p1: Vec2, p2: Vec2) -> str:
    """
    SVG Quadratic-Bezier:
      M p0
      Q p1 p2
    """
    return f"M {p0.x:.2f} {p0.y:.2f} Q {p1.x:.2f} {p1.y:.2f} {p2.x:.2f} {p2.y:.2f}"


def rubber_hose_midpoint(a: Vec2, b: Vec2, bend: float) -> Vec2:
    """
    Erzeugt einen "Rubber-hose" Kontrollpunkt zwischen a und b.

    bend > 0 bedeutet: Ausbuchtung senkrecht zur Verbindungslinie.
    Das ist ein extrem einfacher Trick, wirkt aber sofort cartoonig.
    """
    mid = (a + b) * 0.5
    dir_ = (b - a).normalized()
    n = dir_.perp()
    return mid + n * bend


def apply_jitter_to_points(
    *,
    points: list[Vec2],
    style: CharacterStyle,
    stroke_id: str,
    frame_index: int,
    jitter_amplitude: float | None = None,
) -> list[Vec2]:
    """
    Jittert eine Liste von Punkten zeitkohärent.
    """
    out: list[Vec2] = []
    amp = style.jitter_amplitude if jitter_amplitude is None else jitter_amplitude
    for i, pt in enumerate(points):
        j = jitter_vec(
            base_seed=style.seed,
            stroke_id=stroke_id,
            point_index=i,
            frame_index=frame_index,
            amplitude=amp,
        )
        out.append(pt + j)
    return out


def motion_smear_intensity(speed: float, style: CharacterStyle) -> float:
    """
    Mappt Geschwindigkeit (px/s) auf Smear-Intensität in [0, 1].
    """
    lo = max(0.0, style.smear_speed_threshold)
    hi = max(lo + 1e-6, style.smear_speed_full)
    if speed <= lo:
        return 0.0
    u = (speed - lo) / (hi - lo)
    return smoothstep(max(0.0, min(1.0, u)))


def transform_point_motion_smear(
    *,
    point: Vec2,
    pivot: Vec2,
    dir_: Vec2,
    perp: Vec2,
    stretch: float,
    squeeze: float,
) -> Vec2:
    """
    Anisotrope Skalierung entlang Bewegungsrichtung (dir_) und senkrecht dazu.
    """
    rel = point - pivot
    parallel = rel.x * dir_.x + rel.y * dir_.y
    side = rel.x * perp.x + rel.y * perp.y
    parallel *= stretch
    side *= squeeze
    return pivot + (dir_ * parallel) + (perp * side)


def apply_motion_smear_to_rig(
    *,
    rig: CharacterRig,
    velocity: Vec2,
    style: CharacterStyle,
) -> tuple[CharacterRig, float]:
    speed = velocity.length()
    intensity = motion_smear_intensity(speed, style)
    if intensity <= 1e-6 or speed <= 1e-6:
        return rig, 0.0

    dir_ = velocity.normalized()
    perp = dir_.perp()
    stretch = 1.0 + style.smear_max_stretch * intensity
    squeeze = 1.0 - style.smear_max_squeeze * intensity

    def t(pt: Vec2) -> Vec2:
        return transform_point_motion_smear(
            point=pt,
            pivot=rig.root,
            dir_=dir_,
            perp=perp,
            stretch=stretch,
            squeeze=squeeze,
        )

    smeared = CharacterRig(
        root=rig.root,
        neck=t(rig.neck),
        head_center=t(rig.head_center),
        l_shoulder=t(rig.l_shoulder),
        r_shoulder=t(rig.r_shoulder),
        l_hip=t(rig.l_hip),
        r_hip=t(rig.r_hip),
        l_hand=t(rig.l_hand),
        r_hand=t(rig.r_hand),
        l_foot=t(rig.l_foot),
        r_foot=t(rig.r_foot),
    )
    return smeared, intensity


def render_character_svg(
    *,
    rig: CharacterRig,
    style: CharacterStyle,
    frame_index: int,
    velocity: Vec2 | None = None,
    id_prefix: str = "",
) -> list[SvgPath | SvgCircle | SvgRaw]:
    """
    Rendert die Figur als SVG-Items.

    Wichtig: Wir halten die Stroke-Topologie konstant:
    - Head (circle)
    - Torso (path)
    - Arm L/R (path)
    - Leg L/R (path)
    - Optional: simple "face" (eyes/mouth) als mini-strokes
    """
    items: list[SvgPath | SvgCircle | SvgRaw] = []

    def sid(name: str) -> str:
        if not id_prefix:
            return name
        return f"{id_prefix}{name}"

    v = velocity if velocity is not None else Vec2(0.0, 0.0)
    rig, smear_intensity = apply_motion_smear_to_rig(rig=rig, velocity=v, style=style)
    local_jitter = style.jitter_amplitude * (1.0 + style.smear_jitter_boost * smear_intensity)
    base = SvgStyle(stroke="#000", stroke_width=style.line_width, fill="none")

    # Head (circle) – klassisch, clean.
    head = SvgCircle(
        cx=rig.head_center.x,
        cy=rig.head_center.y,
        r=style.head_radius,
        style=SvgStyle(stroke="#000", stroke_width=style.line_width, fill="none"),
        id=sid("head"),
    )
    items.append(head)

    # Torso – Quadratic curve von neck zu root, leicht gebogen.
    torso_bend = 8.0
    torso_ctrl = rubber_hose_midpoint(rig.neck, rig.root, bend=torso_bend)
    torso_pts = [rig.neck, torso_ctrl, rig.root]
    torso_pts = apply_jitter_to_points(
        points=torso_pts,
        style=style,
        stroke_id=sid("torso"),
        frame_index=frame_index,
        jitter_amplitude=local_jitter,
    )
    items.append(
        SvgPath(
            d=quadratic_bezier_path(torso_pts[0], torso_pts[1], torso_pts[2]),
            style=base,
            id=sid("torso"),
        )
    )

    # Arme: Schulter -> Hand, mehr bend für "rubber hose"
    for side in ("l", "r"):
        match side:
            case "l":
                a0, a2 = rig.l_shoulder, rig.l_hand
                bend = 22.0 * (1.0 + 0.35 * smear_intensity)
                stroke_id = sid("arm_l")
            case "r":
                a0, a2 = rig.r_shoulder, rig.r_hand
                bend = -22.0 * (1.0 + 0.35 * smear_intensity)
                stroke_id = sid("arm_r")
            case _:
                raise ValueError("unreachable")

        a1 = rubber_hose_midpoint(a0, a2, bend=bend)
        arm_pts = [a0, a1, a2]
        arm_pts = apply_jitter_to_points(
            points=arm_pts,
            style=style,
            stroke_id=stroke_id,
            frame_index=frame_index,
            jitter_amplitude=local_jitter,
        )
        arm_style = SvgStyle(
            stroke="#000",
            stroke_width=style.limb_width,
            fill="none",
            stroke_linecap="round",
            stroke_linejoin="round",
        )
        items.append(
            SvgPath(
                d=quadratic_bezier_path(arm_pts[0], arm_pts[1], arm_pts[2]),
                style=arm_style,
                id=stroke_id,
            )
        )

    # Beine: Hip -> Foot
    for side in ("l", "r"):
        match side:
            case "l":
                b0, b2 = rig.l_hip, rig.l_foot
                bend = 16.0 * (1.0 + 0.28 * smear_intensity)
                stroke_id = sid("leg_l")
            case "r":
                b0, b2 = rig.r_hip, rig.r_foot
                bend = -16.0 * (1.0 + 0.28 * smear_intensity)
                stroke_id = sid("leg_r")
            case _:
                raise ValueError("unreachable")

        b1 = rubber_hose_midpoint(b0, b2, bend=bend)
        leg_pts = [b0, b1, b2]
        leg_pts = apply_jitter_to_points(
            points=leg_pts,
            style=style,
            stroke_id=stroke_id,
            frame_index=frame_index,
            jitter_amplitude=local_jitter,
        )
        leg_style = SvgStyle(
            stroke="#000",
            stroke_width=style.limb_width,
            fill="none",
            stroke_linecap="round",
            stroke_linejoin="round",
        )
        items.append(
            SvgPath(
                d=quadratic_bezier_path(leg_pts[0], leg_pts[1], leg_pts[2]),
                style=leg_style,
                id=stroke_id,
            )
        )

    # Mini-Face: zwei Augen + Mund (sehr minimalistisch)
    eye_style = SvgStyle(stroke="#000", stroke_width=style.line_width, fill="#000")
    # Augen als kleine gefüllte Kreise
    items.append(SvgCircle(rig.head_center.x - 10, rig.head_center.y - 5, 3.2, eye_style, id=sid("eye_l")))
    items.append(SvgCircle(rig.head_center.x + 10, rig.head_center.y - 5, 3.2, eye_style, id=sid("eye_r")))

    # Mund als kleine Kurve
    m0 = rig.head_center + Vec2(-10, 10)
    m2 = rig.head_center + Vec2(+10, 10)
    m1 = rubber_hose_midpoint(m0, m2, bend=5.0)
    mouth_pts = apply_jitter_to_points(
        points=[m0, m1, m2],
        style=style,
        stroke_id=sid("mouth"),
        frame_index=frame_index,
        jitter_amplitude=local_jitter,
    )
    items.append(SvgPath(quadratic_bezier_path(mouth_pts[0], mouth_pts[1], mouth_pts[2]),
                         SvgStyle(stroke="#000", stroke_width=4.0, fill="none"),
                         id=sid("mouth")))

    return items


def _render_character_vector_mesh(
    *,
    pose: Pose,
    rig: CharacterRig,
    style: CharacterStyle,
    frame_index: int,
    velocity: Vec2,
    id_prefix: str,
) -> list[SvgPath | SvgCircle | SvgRaw]:
    """
    Optionaler Vektor-Skinning-Pfad (modulare SVG-Meshes auf Bones).
    """
    if not style.mesh_asset:
        return []

    try:
        from vector_skinning import render_skinned_mesh_items
    except Exception:
        return []

    try:
        return render_skinned_mesh_items(
            pose=pose,
            rig=rig,
            style=style,
            frame_index=frame_index,
            velocity=velocity,
            asset_path=style.mesh_asset,
            id_prefix=id_prefix,
        )
    except Exception:
        # Fallback auf Stick-Renderer, falls Asset fehlerhaft ist.
        return []


def render_character_items(
    *,
    pose: Pose,
    style: CharacterStyle,
    frame_index: int,
    velocity: Vec2,
    id_prefix: str = "",
) -> list[SvgPath | SvgCircle | SvgRaw]:
    """
    Auto-Renderer: `stick` (default) oder `mesh` (vector-skinning).
    """
    rig = build_rig(pose)
    if style.render_mode.lower() == "mesh":
        mesh_items = _render_character_vector_mesh(
            pose=pose,
            rig=rig,
            style=style,
            frame_index=frame_index,
            velocity=velocity,
            id_prefix=id_prefix,
        )
        if mesh_items:
            return mesh_items

    return render_character_svg(
        rig=rig,
        style=style,
        frame_index=frame_index,
        velocity=velocity,
        id_prefix=id_prefix,
    )


_PATH_TOKEN_RE = re.compile(r"[A-Za-z]|-?\d+(?:\.\d+)?")


def camera_depth_scale_for_y(*, y: float, cfg: ScriptConfig) -> float:
    cam = cfg.camera
    if not cam.enabled or not cam.depth_enabled:
        return 1.0
    lo = min(cam.depth_min_scale, cam.depth_max_scale)
    hi = max(cam.depth_min_scale, cam.depth_max_scale)
    y_ref_lo = cfg.height * 0.38
    y_ref_hi = cfg.height * 0.88
    u = clamp((y - y_ref_lo) / max(1.0, (y_ref_hi - y_ref_lo)), 0.0, 1.0)
    return lerp(lo, hi, u)


def camera_shake_offset(
    *,
    t: float,
    cfg: ScriptConfig,
    impact_times: Sequence[float] | None,
) -> Vec2:
    cam = cfg.camera
    if not cam.enabled or not cam.shake_on_impact or not impact_times:
        return Vec2(0.0, 0.0)
    ax = 0.0
    ay = 0.0
    freq = max(0.1, cam.shake_frequency)
    decay = max(0.1, cam.shake_decay)
    amp = max(0.0, cam.shake_amplitude)
    for ti in impact_times:
        dt = t - ti
        if dt < 0.0 or dt > 1.5:
            continue
        env = math.exp(-decay * dt)
        phase = dt * freq * 2.0 * math.pi
        ax += math.sin(phase) * amp * env
        ay += math.sin(phase * 1.37 + 0.8) * amp * 0.7 * env
    return Vec2(ax, ay)


def camera_focus_world_anchor(cfg: ScriptConfig) -> Vec2:
    return Vec2(cfg.width * 0.5, cfg.height * 0.72)


def camera_world_to_screen_point(
    *,
    world: Vec2,
    focus_world: Vec2,
    cfg: ScriptConfig,
    t: float,
    impact_times: Sequence[float] | None,
    depth_scale: float,
) -> Vec2:
    cam = cfg.camera
    if not cam.enabled:
        return world

    center = camera_focus_world_anchor(cfg)
    zoom = max(0.05, cam.zoom) * depth_scale
    parallax = (world.y - focus_world.y) * cam.parallax_strength * 0.18
    shake = camera_shake_offset(t=t, cfg=cfg, impact_times=impact_times)
    rel = world - focus_world
    return Vec2(
        center.x + rel.x * zoom + cam.pan.x + parallax + shake.x,
        center.y + rel.y * zoom + cam.pan.y + shake.y,
    )


def camera_transform_pose_style(
    *,
    pose: Pose,
    style: CharacterStyle,
    cfg: ScriptConfig,
    focus_world: Vec2,
    t: float,
    impact_times: Sequence[float] | None,
) -> tuple[Pose, CharacterStyle]:
    cam = cfg.camera
    if not cam.enabled:
        return pose, style

    depth_scale = camera_depth_scale_for_y(y=pose.root.y, cfg=cfg)
    root_screen = camera_world_to_screen_point(
        world=pose.root,
        focus_world=focus_world,
        cfg=cfg,
        t=t,
        impact_times=impact_times,
        depth_scale=depth_scale,
    )
    limbs_scale = max(0.05, depth_scale * max(0.05, cam.zoom))

    out_pose = Pose(
        root=root_screen,
        l_hand=pose.l_hand * limbs_scale,
        r_hand=pose.r_hand * limbs_scale,
        l_foot=pose.l_foot * limbs_scale,
        r_foot=pose.r_foot * limbs_scale,
        look_angle=pose.look_angle,
        squash=pose.squash,
    )
    out_style = replace(style)
    out_style.line_width = max(0.8, style.line_width * limbs_scale)
    out_style.limb_width = max(0.8, style.limb_width * limbs_scale)
    out_style.head_radius = max(2.0, style.head_radius * limbs_scale)
    return out_pose, out_style


def camera_transform_path_d(
    d: str,
    *,
    cfg: ScriptConfig,
    focus_world: Vec2,
    t: float,
    impact_times: Sequence[float] | None,
) -> str:
    cam = cfg.camera
    if not cam.enabled:
        return d

    tokens = _PATH_TOKEN_RE.findall(d)
    if not tokens:
        return d

    out: list[str] = []
    cmd = ""
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isalpha():
            cmd = tok
            out.append(tok)
            i += 1
            continue

        if cmd in {"M", "L", "Q", "C", "S", "T"}:
            if i + 1 >= len(tokens):
                break
            x = float(tokens[i])
            y = float(tokens[i + 1])
            ds = camera_depth_scale_for_y(y=y, cfg=cfg)
            pt = camera_world_to_screen_point(
                world=Vec2(x, y),
                focus_world=focus_world,
                cfg=cfg,
                t=t,
                impact_times=impact_times,
                depth_scale=ds,
            )
            out.append(f"{pt.x:.2f}")
            out.append(f"{pt.y:.2f}")
            i += 2
            continue

        # Unsupported commands are passed through to avoid corrupting paths.
        out.append(tok)
        i += 1

    return " ".join(out)


def camera_transform_svg_items(
    items: Sequence[SvgPath | SvgCircle | SvgRaw],
    *,
    cfg: ScriptConfig,
    focus_world: Vec2,
    t: float,
    impact_times: Sequence[float] | None,
) -> list[SvgPath | SvgCircle | SvgRaw]:
    cam = cfg.camera
    if not cam.enabled:
        return list(items)

    out: list[SvgPath | SvgCircle | SvgRaw] = []
    for it in items:
        if isinstance(it, SvgPath):
            out.append(
                SvgPath(
                    d=camera_transform_path_d(
                        it.d,
                        cfg=cfg,
                        focus_world=focus_world,
                        t=t,
                        impact_times=impact_times,
                    ),
                    style=it.style,
                    id=it.id,
                )
            )
            continue
        if isinstance(it, SvgCircle):
            ds = camera_depth_scale_for_y(y=it.cy, cfg=cfg)
            pt = camera_world_to_screen_point(
                world=Vec2(it.cx, it.cy),
                focus_world=focus_world,
                cfg=cfg,
                t=t,
                impact_times=impact_times,
                depth_scale=ds,
            )
            out.append(
                SvgCircle(
                    cx=pt.x,
                    cy=pt.y,
                    r=max(0.1, it.r * ds * max(0.05, cam.zoom)),
                    style=it.style,
                    id=it.id,
                )
            )
            continue
        out.append(it)
    return out


# -----------------------------------------
# 5) Animation: Keyframes + Actions + DSL
# -----------------------------------------

@dataclass(slots=True)
class Keyframe:
    t: float  # Zeit in Sekunden
    pose: Pose


@dataclass(slots=True)
class Timeline:
    keyframes: list[Keyframe] = field(default_factory=list)

    def add(self, t: float, pose: Pose) -> None:
        self.keyframes.append(Keyframe(t=t, pose=pose))
        self.keyframes.sort(key=lambda k: k.t)

    def sample(self, t: float) -> Pose:
        """
        Interpoliert Pose zwischen Keyframes.
        Bei nur einem Keyframe: konstant.
        """
        if not self.keyframes:
            raise ValueError("Timeline has no keyframes")

        if len(self.keyframes) == 1:
            return self.keyframes[0].pose

        # clamp
        if t <= self.keyframes[0].t:
            return self.keyframes[0].pose
        if t >= self.keyframes[-1].t:
            return self.keyframes[-1].pose

        # finde Segment
        for i in range(len(self.keyframes) - 1):
            a = self.keyframes[i]
            b = self.keyframes[i + 1]
            if a.t <= t <= b.t:
                u = (t - a.t) / (b.t - a.t)
                u = smoothstep(u)

                return Pose(
                    root=lerp2(a.pose.root, b.pose.root, u),
                    l_hand=lerp2(a.pose.l_hand, b.pose.l_hand, u),
                    r_hand=lerp2(a.pose.r_hand, b.pose.r_hand, u),
                    l_foot=lerp2(a.pose.l_foot, b.pose.l_foot, u),
                    r_foot=lerp2(a.pose.r_foot, b.pose.r_foot, u),
                    look_angle=lerp(a.pose.look_angle, b.pose.look_angle, u),
                    squash=lerp(a.pose.squash, b.pose.squash, u),
                )

        # Fallback (sollte nicht passieren)
        return self.keyframes[-1].pose


# ---- DSL: sehr bewusst klein, aber erweiterbar ----
#
# Syntax (Beispiele):
#
#   canvas 800 450
#   character seed "MY-SEED" line 7 limb 10 head 28 jitter 1.2
#             smear_threshold 140 smear_full 480 smear_stretch 0.45
#             smear_squeeze 0.28 smear_jitter 1.0
#   pose t=0   root=400,330 lh=-70,-120 rh=70,-120 lf=-30,90 rf=30,90
#   pose t=0.5 root=420,330 lh=-80,-110 rh=60,-130  lf=-20,90 rf=40,90
#   pose t=1.0 root=440,330 lh=-70,-120 rh=70,-120  lf=-30,90 rf=30,90
#
# Du kannst damit schon Walk/Move/Slapstick als Sequenzen bauen.

@dataclass(slots=True)
class ScriptConfig:
    width: int = 800
    height: int = 450
    fps: int = 12
    seconds: float = 2.0
    style: CharacterStyle = field(default_factory=CharacterStyle)
    # Hybrid physics runtime tuning (used by procedural pipelines)
    physics_mode: str = "off"  # off | fallback | pymunk | auto
    physics_gravity: float = 980.0
    physics_damping: float = 0.92
    physics_restitution: float = 0.22
    physics_friction: float = 0.85
    physics_impulse_scale: float = 320.0
    physics_ragdoll_extra: float = 0.35
    physics_substeps: int = 2
    camera: CameraConfig = field(default_factory=CameraConfig)


POSE_RE = re.compile(
    r"""^pose\s+
        t=(?P<t>[0-9]*\.?[0-9]+)\s+
        root=(?P<root>-?\d+,-?\d+)\s+
        lh=(?P<lh>-?\d+,-?\d+)\s+
        rh=(?P<rh>-?\d+,-?\d+)\s+
        lf=(?P<lf>-?\d+,-?\d+)\s+
        rf=(?P<rf>-?\d+,-?\d+)
        (?:\s+look=(?P<look>-?\d+))?
        (?:\s+squash=(?P<squash>[0-9]*\.?[0-9]+))?
        \s*$""",
    re.VERBOSE,
)

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
    re.VERBOSE,
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
    re.VERBOSE,
)


def parse_vec2(s: str) -> Vec2:
    x_str, y_str = s.split(",")
    return Vec2(float(x_str), float(y_str))


def parse_script(text: str) -> tuple[ScriptConfig, Timeline]:
    """
    Parser für die Mini-DSL.

    Designentscheidung:
    - Wir machen die Grammatik bewusst "hart" (klare Fehlermeldungen),
      weil das später wichtig ist, wenn du daraus ein echtes Tool machst.
    """
    cfg = ScriptConfig()
    tl = Timeline()

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
                cfg.physics_mode = v
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
            seed = m.group("seed")
            if seed is not None:
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
                cfg.style.render_mode = v
            if (v := m.group("mesh")) is not None:
                cfg.style.mesh_asset = v
                cfg.style.render_mode = "mesh"
            if (v := m.group("tint")) is not None:
                cfg.style.mesh_tint = v
            continue

        if m := POSE_RE.match(line):
            t = float(m.group("t"))
            root = parse_vec2(m.group("root"))
            lh = parse_vec2(m.group("lh"))
            rh = parse_vec2(m.group("rh"))
            lf = parse_vec2(m.group("lf"))
            rf = parse_vec2(m.group("rf"))

            look_deg = float(m.group("look")) if m.group("look") is not None else 0.0
            squash = float(m.group("squash")) if m.group("squash") is not None else 0.0

            pose = Pose(
                root=root,
                l_hand=lh,
                r_hand=rh,
                l_foot=lf,
                r_foot=rf,
                look_angle=math.radians(look_deg),
                squash=squash,
            )
            tl.add(t, pose)
            continue

        raise ValueError(f"Syntaxfehler in Zeile {ln}: {line}")

    if not tl.keyframes:
        raise ValueError("Script enthält keine 'pose' Zeilen")

    return cfg, tl


# -----------------------------------------
# 6) Rendering Pipeline: Frames -> SVG files
# -----------------------------------------

def render_character_states(
    *,
    frame_index: int,
    states: Sequence[CharacterRenderState],
    cfg: ScriptConfig | None = None,
    t: float = 0.0,
    focus_world: Vec2 | None = None,
    camera_impact_times: Sequence[float] | None = None,
) -> list[SvgPath | SvgCircle | SvgRaw]:
    """
    Rendert mehrere Characters in einem Frame.
    """
    items: list[SvgPath | SvgCircle | SvgRaw] = []
    ordered_states = list(states)
    if cfg is not None and cfg.camera.enabled and cfg.camera.y_sort:
        ordered_states.sort(key=lambda s: s.pose.root.y)

    focus = focus_world
    if focus is None and cfg is not None:
        focus = camera_focus_world_anchor(cfg)

    for state in ordered_states:
        pose = state.pose
        style = state.style
        if cfg is not None and cfg.camera.enabled and focus is not None:
            pose, style = camera_transform_pose_style(
                pose=state.pose,
                style=state.style,
                cfg=cfg,
                focus_world=focus,
                t=t,
                impact_times=camera_impact_times,
            )
        items.extend(
            render_character_items(
                pose=pose,
                style=style,
                frame_index=frame_index,
                velocity=state.velocity,
                id_prefix=f"{state.char_id}_",
            )
        )
    return items


def render_multi_frames(
    *,
    cfg: ScriptConfig,
    timelines: dict[str, Timeline],
    styles: dict[str, CharacterStyle],
    out_dir: Path,
    character_order: Sequence[str] | None = None,
    extra_items_fn: Callable[[int, float], Sequence[SvgPath | SvgCircle | SvgRaw]] | None = None,
    camera_impact_times: Sequence[float] | None = None,
) -> None:
    """
    Rendert eine Szene mit mehreren Characters (z.B. Slapstick-Duo).
    """
    if not timelines:
        raise ValueError("render_multi_frames requires at least one timeline")

    order = list(character_order) if character_order is not None else sorted(timelines.keys())
    if not order:
        raise ValueError("render_multi_frames received an empty character order")

    for cid in order:
        if cid not in timelines:
            raise ValueError(f"render_multi_frames missing timeline for character '{cid}'")
        if cid not in styles:
            raise ValueError(f"render_multi_frames missing style for character '{cid}'")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(round(cfg.seconds * cfg.fps))
    prev_poses: dict[str, Pose | None] = {cid: None for cid in order}

    for fi in range(total_frames):
        t = fi / cfg.fps
        frame_states: list[CharacterRenderState] = []

        for cid in order:
            pose = timelines[cid].sample(t)
            prev = prev_poses[cid]
            if prev is None:
                velocity = Vec2(0.0, 0.0)
            else:
                dt = 1.0 / cfg.fps
                velocity = (pose.root - prev.root) * (1.0 / max(dt, 1e-6))
            frame_states.append(
                CharacterRenderState(
                    char_id=cid,
                    pose=pose,
                    style=styles[cid],
                    velocity=velocity,
                )
            )
            prev_poses[cid] = pose

        items: list[SvgPath | SvgCircle | SvgRaw] = []
        focus_world = camera_focus_world_anchor(cfg)
        if cfg.camera.enabled:
            focus_id = cfg.camera.focus.strip().lower()
            if focus_id in {"auto", "centroid"}:
                if frame_states:
                    sx = sum(st.pose.root.x for st in frame_states)
                    sy = sum(st.pose.root.y for st in frame_states)
                    focus_world = Vec2(sx / len(frame_states), sy / len(frame_states))
            elif focus_id:
                for st in frame_states:
                    if st.char_id.lower() == focus_id:
                        focus_world = st.pose.root
                        break

        if extra_items_fn is not None:
            extra = list(extra_items_fn(fi, t))
            if cfg.camera.enabled:
                extra = camera_transform_svg_items(
                    extra,
                    cfg=cfg,
                    focus_world=focus_world,
                    t=t,
                    impact_times=camera_impact_times,
                )
            items.extend(extra)

        items.extend(
            render_character_states(
                frame_index=fi,
                states=frame_states,
                cfg=cfg,
                t=t,
                focus_world=focus_world,
                camera_impact_times=camera_impact_times,
            )
        )

        ground_y = cfg.height - 60
        g0 = Vec2(40.0, float(ground_y))
        g1 = Vec2(float(cfg.width - 40), float(ground_y))
        if cfg.camera.enabled:
            g0 = camera_world_to_screen_point(
                world=g0,
                focus_world=focus_world,
                cfg=cfg,
                t=t,
                impact_times=camera_impact_times,
                depth_scale=camera_depth_scale_for_y(y=g0.y, cfg=cfg),
            )
            g1 = camera_world_to_screen_point(
                world=g1,
                focus_world=focus_world,
                cfg=cfg,
                t=t,
                impact_times=camera_impact_times,
                depth_scale=camera_depth_scale_for_y(y=g1.y, cfg=cfg),
            )
        items.append(
            SvgPath(
                d=f"M {g0.x:.2f} {g0.y:.2f} L {g1.x:.2f} {g1.y:.2f}",
                style=SvgStyle(stroke="#000", stroke_width=3.0, fill="none"),
                id="ground",
            )
        )

        svg = svg_doc(width=cfg.width, height=cfg.height, items=items)
        out_path = out_dir / f"frame_{fi:04d}.svg"
        out_path.write_text(svg, encoding="utf-8")


def sample_velocity_from_timeline(
    *,
    tl: Timeline,
    t: float,
    fps: int,
) -> Vec2:
    dt = 1.0 / max(1, fps)
    t0 = max(0.0, t - dt)
    t1 = min(max(0.0, tl.keyframes[-1].t if tl.keyframes else t), t + dt)
    p0 = tl.sample(t0)
    p1 = tl.sample(t1)
    span = max(1e-6, t1 - t0)
    return (p1.root - p0.root) * (1.0 / span)


def render_frame_svg(
    *,
    cfg: ScriptConfig,
    tl: Timeline,
    frame_index: int = 0,
    t: float | None = None,
    extra_items_fn: Callable[[int, float], Sequence[SvgPath | SvgCircle | SvgRaw]] | None = None,
    camera_impact_times: Sequence[float] | None = None,
) -> str:
    """
    Rendert einen einzelnen Frame direkt als SVG-String (ohne Datei-IO).
    """
    if t is None:
        t = frame_index / max(1, cfg.fps)
    t = max(0.0, min(cfg.seconds, t))
    pose = tl.sample(t)
    velocity = sample_velocity_from_timeline(tl=tl, t=t, fps=cfg.fps)
    state = CharacterRenderState(char_id="char1", pose=pose, style=cfg.style, velocity=velocity)
    focus_world = camera_focus_world_anchor(cfg)
    focus_id = cfg.camera.focus.strip().lower()
    if cfg.camera.enabled and focus_id in {"self", "char1"}:
        focus_world = pose.root

    items: list[SvgPath | SvgCircle | SvgRaw] = []
    if extra_items_fn is not None:
        extra = list(extra_items_fn(frame_index, t))
        if cfg.camera.enabled:
            extra = camera_transform_svg_items(
                extra,
                cfg=cfg,
                focus_world=focus_world,
                t=t,
                impact_times=camera_impact_times,
            )
        items.extend(extra)
    items.extend(
        render_character_states(
            frame_index=frame_index,
            states=[state],
            cfg=cfg,
            t=t,
            focus_world=focus_world,
            camera_impact_times=camera_impact_times,
        )
    )
    ground_y = cfg.height - 60
    g0 = Vec2(40.0, float(ground_y))
    g1 = Vec2(float(cfg.width - 40), float(ground_y))
    if cfg.camera.enabled:
        g0 = camera_world_to_screen_point(
            world=g0,
            focus_world=focus_world,
            cfg=cfg,
            t=t,
            impact_times=camera_impact_times,
            depth_scale=camera_depth_scale_for_y(y=g0.y, cfg=cfg),
        )
        g1 = camera_world_to_screen_point(
            world=g1,
            focus_world=focus_world,
            cfg=cfg,
            t=t,
            impact_times=camera_impact_times,
            depth_scale=camera_depth_scale_for_y(y=g1.y, cfg=cfg),
        )
    items.append(
        SvgPath(
            d=f"M {g0.x:.2f} {g0.y:.2f} L {g1.x:.2f} {g1.y:.2f}",
            style=SvgStyle(stroke="#000", stroke_width=3.0, fill="none"),
            id="ground",
        )
    )
    return svg_doc(width=cfg.width, height=cfg.height, items=items)


def render_multi_frame_svg(
    *,
    cfg: ScriptConfig,
    timelines: dict[str, Timeline],
    styles: dict[str, CharacterStyle],
    character_order: Sequence[str] | None = None,
    frame_index: int = 0,
    t: float | None = None,
    extra_items_fn: Callable[[int, float], Sequence[SvgPath | SvgCircle | SvgRaw]] | None = None,
    camera_impact_times: Sequence[float] | None = None,
) -> str:
    """
    Rendert einen einzelnen Multi-Character-Frame direkt als SVG-String.
    """
    if not timelines:
        raise ValueError("render_multi_frame_svg requires at least one timeline")
    order = list(character_order) if character_order is not None else sorted(timelines.keys())
    if not order:
        raise ValueError("render_multi_frame_svg got empty character order")

    if t is None:
        t = frame_index / max(1, cfg.fps)
    t = max(0.0, min(cfg.seconds, t))

    states: list[CharacterRenderState] = []
    for cid in order:
        if cid not in timelines:
            raise ValueError(f"missing timeline for '{cid}'")
        if cid not in styles:
            raise ValueError(f"missing style for '{cid}'")
        pose = timelines[cid].sample(t)
        velocity = sample_velocity_from_timeline(tl=timelines[cid], t=t, fps=cfg.fps)
        states.append(CharacterRenderState(char_id=cid, pose=pose, style=styles[cid], velocity=velocity))

    focus_world = camera_focus_world_anchor(cfg)
    if cfg.camera.enabled:
        focus_id = cfg.camera.focus.strip().lower()
        if focus_id in {"auto", "centroid"}:
            sx = sum(st.pose.root.x for st in states)
            sy = sum(st.pose.root.y for st in states)
            focus_world = Vec2(sx / len(states), sy / len(states))
        elif focus_id:
            for st in states:
                if st.char_id.lower() == focus_id:
                    focus_world = st.pose.root
                    break

    items: list[SvgPath | SvgCircle | SvgRaw] = []
    if extra_items_fn is not None:
        extra = list(extra_items_fn(frame_index, t))
        if cfg.camera.enabled:
            extra = camera_transform_svg_items(
                extra,
                cfg=cfg,
                focus_world=focus_world,
                t=t,
                impact_times=camera_impact_times,
            )
        items.extend(extra)
    items.extend(
        render_character_states(
            frame_index=frame_index,
            states=states,
            cfg=cfg,
            t=t,
            focus_world=focus_world,
            camera_impact_times=camera_impact_times,
        )
    )
    ground_y = cfg.height - 60
    g0 = Vec2(40.0, float(ground_y))
    g1 = Vec2(float(cfg.width - 40), float(ground_y))
    if cfg.camera.enabled:
        g0 = camera_world_to_screen_point(
            world=g0,
            focus_world=focus_world,
            cfg=cfg,
            t=t,
            impact_times=camera_impact_times,
            depth_scale=camera_depth_scale_for_y(y=g0.y, cfg=cfg),
        )
        g1 = camera_world_to_screen_point(
            world=g1,
            focus_world=focus_world,
            cfg=cfg,
            t=t,
            impact_times=camera_impact_times,
            depth_scale=camera_depth_scale_for_y(y=g1.y, cfg=cfg),
        )
    items.append(
        SvgPath(
            d=f"M {g0.x:.2f} {g0.y:.2f} L {g1.x:.2f} {g1.y:.2f}",
            style=SvgStyle(stroke="#000", stroke_width=3.0, fill="none"),
            id="ground",
        )
    )
    return svg_doc(width=cfg.width, height=cfg.height, items=items)


def render_frames(
    *,
    cfg: ScriptConfig,
    tl: Timeline,
    out_dir: Path,
    extra_items_fn: Callable[[int, float], Sequence[SvgPath | SvgCircle | SvgRaw]] | None = None,
    camera_impact_times: Sequence[float] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(round(cfg.seconds * cfg.fps))
    prev_pose: Pose | None = None
    for fi in range(total_frames):
        t = fi / cfg.fps
        pose = tl.sample(t)
        if prev_pose is None:
            velocity = Vec2(0.0, 0.0)
        else:
            dt = 1.0 / cfg.fps
            velocity = (pose.root - prev_pose.root) * (1.0 / max(dt, 1e-6))
        state = CharacterRenderState(char_id="char1", pose=pose, style=cfg.style, velocity=velocity)
        focus_world = camera_focus_world_anchor(cfg)
        if cfg.camera.enabled and cfg.camera.focus.strip().lower() in {"self", "char1"}:
            focus_world = pose.root
        items: list[SvgPath | SvgCircle | SvgRaw] = []
        if extra_items_fn is not None:
            extra = list(extra_items_fn(fi, t))
            if cfg.camera.enabled:
                extra = camera_transform_svg_items(
                    extra,
                    cfg=cfg,
                    focus_world=focus_world,
                    t=t,
                    impact_times=camera_impact_times,
                )
            items.extend(extra)

        items.extend(
            render_character_states(
                frame_index=fi,
                states=[state],
                cfg=cfg,
                t=t,
                focus_world=focus_world,
                camera_impact_times=camera_impact_times,
            )
        )

        # Optional: Bodenlinie für Orientierung (sehr frühe Cartoons hatten oft simple staging lines)
        ground_y = cfg.height - 60
        g0 = Vec2(40.0, float(ground_y))
        g1 = Vec2(float(cfg.width - 40), float(ground_y))
        if cfg.camera.enabled:
            g0 = camera_world_to_screen_point(
                world=g0,
                focus_world=focus_world,
                cfg=cfg,
                t=t,
                impact_times=camera_impact_times,
                depth_scale=camera_depth_scale_for_y(y=g0.y, cfg=cfg),
            )
            g1 = camera_world_to_screen_point(
                world=g1,
                focus_world=focus_world,
                cfg=cfg,
                t=t,
                impact_times=camera_impact_times,
                depth_scale=camera_depth_scale_for_y(y=g1.y, cfg=cfg),
            )
        items.append(
            SvgPath(
                d=f"M {g0.x:.2f} {g0.y:.2f} L {g1.x:.2f} {g1.y:.2f}",
                style=SvgStyle(stroke="#000", stroke_width=3.0, fill="none"),
                id="ground",
            )
        )

        svg = svg_doc(width=cfg.width, height=cfg.height, items=items)
        out_path = out_dir / f"frame_{fi:04d}.svg"
        out_path.write_text(svg, encoding="utf-8")
        prev_pose = pose


# -----------------------------------------
# 7) Demo-Script Generator
# -----------------------------------------

DEMO_SCRIPT = """# Demo: minimaler "walk-ish" move als Keyframes
canvas 800 450
fps 12
seconds 3.0
camera enabled=false
character seed "MYCELIA-INK" line 7 limb 10 head 28 jitter 1.1

# Pose Format:
# pose t=<sec> root=x,y lh=x,y rh=x,y lf=x,y rf=x,y look=<deg> squash=<0..1>

pose t=0.0 root=200,330 lh=-70,-120 rh=70,-120 lf=-30,90 rf=30,90
pose t=0.5 root=260,330 lh=-80,-110 rh=60,-130 lf=-20,90 rf=40,90
pose t=1.0 root=320,330 lh=-70,-120 rh=70,-120 lf=-30,90 rf=30,90
pose t=1.5 root=380,330 lh=-60,-130 rh=80,-110 lf=-40,90 rf=20,90
pose t=2.0 root=440,330 lh=-70,-120 rh=70,-120 lf=-30,90 rf=30,90
pose t=2.5 root=500,330 lh=-80,-110 rh=60,-130 lf=-20,90 rf=40,90
pose t=3.0 root=560,330 lh=-70,-120 rh=70,-120 lf=-30,90 rf=30,90
"""


def write_demo_script(path: Path) -> None:
    path.write_text(DEMO_SCRIPT, encoding="utf-8")


# -----------------------------------------
# 8) CLI
# -----------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="SVG/Vektor-first Cartoon MVP (Strichmännchen Early-Cartoon)")
    ap.add_argument("--script", type=str, default="", help="Pfad zur DSL-Datei. Wenn leer: Demo wird erzeugt und genutzt.")
    ap.add_argument("--out", type=str, default="out", help="Ausgabeordner für SVG-Frames.")
    ap.add_argument("--demo", action="store_true", help="Schreibt eine demo.txt und beendet.")
    # NEW: CLI overrides (optional)
    ap.add_argument("--fps", type=int, default=0, help="Override: Frames per second (0 = Script-Wert nutzen).")
    ap.add_argument("--seconds", type=float, default=0.0, help="Override: Länge in Sekunden (0 = Script-Wert nutzen).")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.demo:
        demo_path = Path("demo.txt")
        write_demo_script(demo_path)
        print(f"[ok] demo script geschrieben: {demo_path.resolve()}")
        return 0

    if args.script:
        script_path = Path(args.script)
        text = script_path.read_text(encoding="utf-8")
    else:
        script_path = Path("demo.txt")
        if not script_path.exists():
            write_demo_script(script_path)
        text = script_path.read_text(encoding="utf-8")

    cfg, tl = parse_script(text)

    # Apply overrides if provided
    if args.fps > 0:
        cfg.fps = args.fps
    if args.seconds > 0.0:
        cfg.seconds = args.seconds

    render_frames(cfg=cfg, tl=tl, out_dir=out_dir)
    print(f"[ok] gerendert: {out_dir.resolve()}  (SVG frames)")
    print("[hint] Video: erst SVG->PNG rasterisieren, dann ffmpeg (wenn du willst, gebe ich dir die One-Liner).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
