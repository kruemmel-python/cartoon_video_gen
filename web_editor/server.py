#!/usr/bin/env python3
"""
web_editor/server.py

FastAPI backend for live DSL preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "FastAPI is required for web editor. Install: pip install fastapi uvicorn"
    ) from exc

from cartoon_svg_mvp import parse_script, render_frame_svg, render_multi_frame_svg
from multi_character_orchestrator import parse_multi_script, generate_multi_character_scene_detailed
from procedural_props import build_prop_items_fn
from procedural_walk_cycle import parse_action_script, generate_procedural_scene_detailed
from slapstick_events import ImpactEvent


ROOT_DIR = Path(__file__).resolve().parent
STATIC_DIR = ROOT_DIR / "static"
PROJECT_ROOT = ROOT_DIR.parent

PYODIDE_MODULES = {
    "cartoon_svg_mvp.py",
    "procedural_walk_cycle.py",
    "procedural_props.py",
    "slapstick_events.py",
    "multi_character_orchestrator.py",
    "vector_skinning.py",
    "physics_hybrid.py",
    "motion_transformer.py",
}


class PreviewRequest(BaseModel):
    mode: str = Field(default="single")
    dsl: str = Field(min_length=1)
    frame_index: int = Field(default=0, ge=0)
    t: float | None = None


class PreviewResponse(BaseModel):
    svg: str
    mode: str
    frame_index: int
    fps: int
    seconds: float
    width: int
    height: int
    character_count: int
    physics: dict[str, Any] | None = None


app = FastAPI(title="Cartoon DSL Live Editor", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/py-src/{name}")
def py_src(name: str) -> FileResponse:
    if name not in PYODIDE_MODULES:
        raise HTTPException(status_code=404, detail="module not exposed")
    src = PROJECT_ROOT / name
    if not src.exists():
        raise HTTPException(status_code=404, detail="module missing")
    return FileResponse(src)


@app.get("/mesh-assets/{asset_path:path}")
def mesh_assets(asset_path: str) -> FileResponse:
    base = (PROJECT_ROOT / "assets" / "meshes").resolve()
    cleaned = asset_path.replace("\\", "/").lstrip("/")
    if cleaned.startswith("assets/"):
        cleaned = cleaned[len("assets/") :]
    if cleaned.startswith("meshes/"):
        cleaned = cleaned[len("meshes/") :]
    target = (base / cleaned).resolve()
    if not str(target).startswith(str(base)):
        raise HTTPException(status_code=403, detail="asset path denied")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    return FileResponse(target)


def _render_single(req: PreviewRequest) -> PreviewResponse:
    # Try procedural DSL first (walk/wave/props/events). Fallback to keyframe DSL.
    try:
        cfg, walk, wave, events, props, ai_motion = parse_action_script(req.dsl)
        tl, prop_runtime, prop_derived, physics_meta = generate_procedural_scene_detailed(
            cfg,
            walk,
            wave,
            ai_motion=ai_motion,
            events=events,
            props=props,
        )
        camera_impact_times = sorted(
            [ev.t for ev in [*events, *prop_derived] if isinstance(ev, ImpactEvent)]
        )
        prop_items_fn = build_prop_items_fn(prop_runtime, cfg)
        svg = render_frame_svg(
            cfg=cfg,
            tl=tl,
            frame_index=req.frame_index,
            t=req.t,
            extra_items_fn=prop_items_fn,
            camera_impact_times=camera_impact_times,
        )
        return PreviewResponse(
            svg=svg,
            mode="single",
            frame_index=req.frame_index,
            fps=cfg.fps,
            seconds=cfg.seconds,
            width=cfg.width,
            height=cfg.height,
            character_count=1,
            physics=physics_meta,
        )
    except ValueError:
        pass

    try:
        cfg, tl = parse_script(req.dsl)
        svg = render_frame_svg(
            cfg=cfg,
            tl=tl,
            frame_index=req.frame_index,
            t=req.t,
            extra_items_fn=None,
        )
        return PreviewResponse(
            svg=svg,
            mode="single",
            frame_index=req.frame_index,
            fps=cfg.fps,
            seconds=cfg.seconds,
            width=cfg.width,
            height=cfg.height,
            character_count=1,
            physics={"enabled": False, "solver": "off"},
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"single parse/render failed: {exc}") from exc


def _render_multi(req: PreviewRequest) -> PreviewResponse:
    try:
        cfg, programs, props, collision = parse_multi_script(req.dsl)
        timelines, prop_runtime, collision_events, prop_events, physics_meta = generate_multi_character_scene_detailed(
            cfg=cfg,
            programs=programs,
            props=props,
            collision=collision,
        )
        order = list(programs.keys())
        styles = {char_id: programs[char_id].style for char_id in order}
        prop_items_fn = build_prop_items_fn(prop_runtime, cfg)
        camera_impact_times: set[float] = set()
        for program in programs.values():
            for ev in program.events:
                if isinstance(ev, ImpactEvent):
                    camera_impact_times.add(ev.t)
        for rows in prop_events.values():
            for ev in rows:
                if isinstance(ev, ImpactEvent):
                    camera_impact_times.add(ev.t)
        for rows in collision_events.values():
            for ev in rows:
                if isinstance(ev, ImpactEvent):
                    camera_impact_times.add(ev.t)

        svg = render_multi_frame_svg(
            cfg=cfg,
            timelines=timelines,
            styles=styles,
            character_order=order,
            frame_index=req.frame_index,
            t=req.t,
            extra_items_fn=prop_items_fn,
            camera_impact_times=sorted(camera_impact_times),
        )
        return PreviewResponse(
            svg=svg,
            mode="multi",
            frame_index=req.frame_index,
            fps=cfg.fps,
            seconds=cfg.seconds,
            width=cfg.width,
            height=cfg.height,
            character_count=len(order),
            physics={"characters": physics_meta, "mode_requested": cfg.physics_mode},
        )
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"multi parse/render failed: {exc}") from exc


@app.post("/api/preview", response_model=PreviewResponse)
def preview(req: PreviewRequest) -> PreviewResponse:
    mode = req.mode.strip().lower()
    if mode not in {"single", "multi"}:
        raise HTTPException(status_code=400, detail="mode must be 'single' or 'multi'")
    if mode == "single":
        return _render_single(req)
    return _render_multi(req)
