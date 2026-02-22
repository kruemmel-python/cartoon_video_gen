#!/usr/bin/env python3
"""
vector_skinning.py

Vector-skinning layer for modular 2D character meshes.

It maps the existing deterministic rig/bones onto external SVG mesh parts via
affine SVG transforms (translate/rotate/scale).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from cartoon_svg_mvp import CharacterRig, CharacterStyle, Pose, SvgRaw, Vec2


SVG_NS = "{http://www.w3.org/2000/svg}"


@dataclass(slots=True)
class BoneTransform:
    start: Vec2
    end: Vec2
    angle_deg: float
    length: float


@dataclass(slots=True)
class MeshPart:
    id: str
    bone: str
    z: int
    svg_file: str | None
    inline_shape: dict | None
    rest_length: float
    translate: Vec2
    rotate_deg: float
    scale: Vec2
    opacity: float
    tint: str


@dataclass(slots=True)
class CharacterMeshAsset:
    path: str
    name: str
    parts: list[MeshPart]
    base_dir: Path


_MESH_CACHE: dict[str, CharacterMeshAsset] = {}
_SVG_INNER_CACHE: dict[str, str] = {}


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _vec_from_seq(v: object, default: Vec2) -> Vec2:
    if not isinstance(v, list | tuple):
        return default
    if len(v) != 2:
        return default
    try:
        return Vec2(float(v[0]), float(v[1]))
    except (TypeError, ValueError):
        return default


def _strip_svg_outer(xml_text: str) -> str:
    """
    Returns inner markup for an SVG file so we can wrap it into transformed groups.
    """
    root = ET.fromstring(xml_text)
    if root.tag == f"{SVG_NS}svg" or root.tag.lower().endswith("svg"):
        return "".join(ET.tostring(ch, encoding="unicode") for ch in list(root))
    return xml_text


def _load_svg_inner(path: Path) -> str:
    key = str(path.resolve())
    cached = _SVG_INNER_CACHE.get(key)
    if cached is not None:
        return cached
    text = path.read_text(encoding="utf-8")
    inner = _strip_svg_outer(text)
    _SVG_INNER_CACHE[key] = inner
    return inner


def _shape_to_svg(shape: dict) -> str:
    typ = str(shape.get("type", "")).lower()
    if typ == "path":
        d = str(shape.get("d", ""))
        fill = str(shape.get("fill", "none"))
        stroke = str(shape.get("stroke", "#000"))
        sw = _fmt(float(shape.get("stroke_width", 3.0)))
        return (
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round"/>'
        )
    if typ == "circle":
        cx = _fmt(float(shape.get("cx", 0.0)))
        cy = _fmt(float(shape.get("cy", 0.0)))
        r = _fmt(float(shape.get("r", 1.0)))
        fill = str(shape.get("fill", "none"))
        stroke = str(shape.get("stroke", "#000"))
        sw = _fmt(float(shape.get("stroke_width", 3.0)))
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    if typ == "ellipse":
        cx = _fmt(float(shape.get("cx", 0.0)))
        cy = _fmt(float(shape.get("cy", 0.0)))
        rx = _fmt(float(shape.get("rx", 1.0)))
        ry = _fmt(float(shape.get("ry", 1.0)))
        fill = str(shape.get("fill", "none"))
        stroke = str(shape.get("stroke", "#000"))
        sw = _fmt(float(shape.get("stroke_width", 3.0)))
        return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    if typ == "rect":
        x = _fmt(float(shape.get("x", 0.0)))
        y = _fmt(float(shape.get("y", 0.0)))
        w = _fmt(float(shape.get("width", 1.0)))
        h = _fmt(float(shape.get("height", 1.0)))
        rx = _fmt(float(shape.get("rx", 0.0)))
        fill = str(shape.get("fill", "none"))
        stroke = str(shape.get("stroke", "#000"))
        sw = _fmt(float(shape.get("stroke_width", 3.0)))
        return (
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
    raise ValueError(f"unsupported mesh inline shape type: {typ}")


def load_character_mesh_asset(asset_path: str) -> CharacterMeshAsset:
    resolved = str(Path(asset_path).resolve())
    cached = _MESH_CACHE.get(resolved)
    if cached is not None:
        return cached

    path = Path(asset_path)
    if not path.exists():
        raise FileNotFoundError(f"mesh asset not found: {asset_path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("mesh asset must be a JSON object")
    name = str(obj.get("name", path.stem))
    parts_raw = obj.get("parts")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise ValueError("mesh asset requires non-empty 'parts' list")

    parts: list[MeshPart] = []
    for i, row in enumerate(parts_raw):
        if not isinstance(row, dict):
            raise ValueError(f"mesh part #{i} must be object")
        part_id = str(row.get("id", f"part_{i:03d}"))
        bone = str(row.get("bone", "root"))
        z = int(row.get("z", i))
        svg_file = row.get("svg_file")
        if svg_file is not None:
            svg_file = str(svg_file)
        inline_shape = row.get("shape")
        if svg_file is None and inline_shape is None:
            raise ValueError(f"mesh part '{part_id}' requires either 'svg_file' or 'shape'")
        parts.append(
            MeshPart(
                id=part_id,
                bone=bone,
                z=z,
                svg_file=svg_file,
                inline_shape=inline_shape if isinstance(inline_shape, dict) else None,
                rest_length=float(row.get("rest_length", 0.0)),
                translate=_vec_from_seq(row.get("translate"), Vec2(0.0, 0.0)),
                rotate_deg=float(row.get("rotate", 0.0)),
                scale=_vec_from_seq(row.get("scale"), Vec2(1.0, 1.0)),
                opacity=float(row.get("opacity", 1.0)),
                tint=str(row.get("tint", "")),
            )
        )

    asset = CharacterMeshAsset(
        path=resolved,
        name=name,
        parts=parts,
        base_dir=path.parent,
    )
    _MESH_CACHE[resolved] = asset
    return asset


def _angle_deg(start: Vec2, end: Vec2) -> float:
    d = end - start
    if d.length() <= 1e-9:
        return 0.0
    return math.degrees(math.atan2(d.y, d.x))


def _squash_point(pt: Vec2, root: Vec2, sx: float, sy: float) -> Vec2:
    rel = pt - root
    return root + Vec2(rel.x * sx, rel.y * sy)


def build_bone_transforms(
    *,
    pose: Pose,
    rig: CharacterRig,
) -> dict[str, BoneTransform]:
    squash = max(0.0, min(1.0, pose.squash))
    sx = 1.0 + 0.22 * squash
    sy = 1.0 - 0.18 * squash

    def sq(v: Vec2) -> Vec2:
        return _squash_point(v, rig.root, sx, sy)

    root = sq(rig.root)
    neck = sq(rig.neck)
    head = sq(rig.head_center)
    ls = sq(rig.l_shoulder)
    rs = sq(rig.r_shoulder)
    lh = sq(rig.l_hand)
    rh = sq(rig.r_hand)
    lhip = sq(rig.l_hip)
    rhip = sq(rig.r_hip)
    lf = sq(rig.l_foot)
    rf = sq(rig.r_foot)

    def bt(a: Vec2, b: Vec2) -> BoneTransform:
        return BoneTransform(start=a, end=b, angle_deg=_angle_deg(a, b), length=(b - a).length())

    face_tip = head + Vec2(math.cos(pose.look_angle) * 22.0, math.sin(pose.look_angle) * 22.0)

    return {
        "root": bt(root, root + Vec2(1.0, 0.0)),
        "torso": bt(root, neck),
        "head": bt(neck, head),
        "face": bt(head, face_tip),
        "arm_l": bt(ls, lh),
        "arm_r": bt(rs, rh),
        "leg_l": bt(lhip, lf),
        "leg_r": bt(rhip, rf),
        "hand_l": bt(lh, lh + Vec2(1.0, 0.0)),
        "hand_r": bt(rh, rh + Vec2(1.0, 0.0)),
        "foot_l": bt(lf, lf + Vec2(1.0, 0.0)),
        "foot_r": bt(rf, rf + Vec2(1.0, 0.0)),
    }


def _part_content(asset: CharacterMeshAsset, part: MeshPart, tint: str) -> str:
    if part.svg_file is not None:
        content = _load_svg_inner(asset.base_dir / part.svg_file)
    elif part.inline_shape is not None:
        content = _shape_to_svg(part.inline_shape)
    else:
        content = ""
    if "{{tint}}" in content:
        content = content.replace("{{tint}}", tint)
    return content


def render_skinned_mesh_items(
    *,
    pose: Pose,
    rig: CharacterRig,
    style: CharacterStyle,
    frame_index: int,
    velocity: Vec2,
    asset_path: str,
    id_prefix: str = "",
) -> list[SvgRaw]:
    """
    Render skinned mesh items for one character frame.
    """
    _ = frame_index
    _ = velocity
    asset = load_character_mesh_asset(asset_path)
    bones = build_bone_transforms(pose=pose, rig=rig)

    out: list[SvgRaw] = []
    tint_style = style.mesh_tint.strip()
    for part in sorted(asset.parts, key=lambda p: p.z):
        bone = bones.get(part.bone, bones.get("root"))
        if bone is None:
            continue
        scale_x = part.scale.x
        if part.rest_length > 1e-6 and bone.length > 1e-6:
            scale_x *= bone.length / part.rest_length
        scale_y = part.scale.y

        tx = bone.start.x
        ty = bone.start.y
        rot = bone.angle_deg + part.rotate_deg
        op = max(0.0, min(1.0, part.opacity))
        tint = part.tint or tint_style or "#111111"
        content = _part_content(asset, part, tint=tint)
        part_id = f"{id_prefix}{part.id}" if id_prefix else part.id
        transform = (
            f"translate({_fmt(tx)} {_fmt(ty)}) "
            f"rotate({_fmt(rot)}) "
            f"translate({_fmt(part.translate.x)} {_fmt(part.translate.y)}) "
            f"scale({_fmt(scale_x)} {_fmt(scale_y)})"
        )
        xml = (
            f'<g id="{part_id}" transform="{transform}" opacity="{_fmt(op)}" '
            f'style="color:{tint}">{content}</g>'
        )
        out.append(SvgRaw(xml=xml, id=part_id))
    return out
