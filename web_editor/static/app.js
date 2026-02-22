(() => {
  const els = {
    previewEngine: document.getElementById("previewEngine"),
    sceneMode: document.getElementById("sceneMode"),
    seconds: document.getElementById("seconds"),
    canvasW: document.getElementById("canvasW"),
    canvasH: document.getElementById("canvasH"),
    fps: document.getElementById("fps"),
    frameIndex: document.getElementById("frameIndex"),
    seed1: document.getElementById("seed1"),
    mode1: document.getElementById("mode1"),
    line1: document.getElementById("line1"),
    limb1: document.getElementById("limb1"),
    head1: document.getElementById("head1"),
    jitter1: document.getElementById("jitter1"),
    mesh1: document.getElementById("mesh1"),
    tint1: document.getElementById("tint1"),
    seed2: document.getElementById("seed2"),
    mode2: document.getElementById("mode2"),
    line2: document.getElementById("line2"),
    limb2: document.getElementById("limb2"),
    head2: document.getElementById("head2"),
    jitter2: document.getElementById("jitter2"),
    mesh2: document.getElementById("mesh2"),
    tint2: document.getElementById("tint2"),
    char2StyleWrap: document.getElementById("char2StyleWrap"),
    physicsMode: document.getElementById("physicsMode"),
    gravity: document.getElementById("gravity"),
    damping: document.getElementById("damping"),
    restitution: document.getElementById("restitution"),
    friction: document.getElementById("friction"),
    substeps: document.getElementById("substeps"),
    cameraEnabled: document.getElementById("cameraEnabled"),
    cameraFocus: document.getElementById("cameraFocus"),
    cameraZoom: document.getElementById("cameraZoom"),
    cameraPan: document.getElementById("cameraPan"),
    cameraDepth: document.getElementById("cameraDepth"),
    cameraParallax: document.getElementById("cameraParallax"),
    cameraDepthMin: document.getElementById("cameraDepthMin"),
    cameraDepthMax: document.getElementById("cameraDepthMax"),
    cameraYSort: document.getElementById("cameraYSort"),
    cameraShake: document.getElementById("cameraShake"),
    cameraShakeAmp: document.getElementById("cameraShakeAmp"),
    cameraShakeFreq: document.getElementById("cameraShakeFreq"),
    cameraShakeDecay: document.getElementById("cameraShakeDecay"),
    nodeType: document.getElementById("nodeType"),
    addNodeBtn: document.getElementById("addNodeBtn"),
    resetBtn: document.getElementById("resetBtn"),
    nodes: document.getElementById("nodes"),
    dsl: document.getElementById("dsl"),
    previewBtn: document.getElementById("previewBtn"),
    autoPreview: document.getElementById("autoPreview"),
    status: document.getElementById("status"),
    previewMeta: document.getElementById("previewMeta"),
    playPauseBtn: document.getElementById("playPauseBtn"),
    stopBtn: document.getElementById("stopBtn"),
    loopPlayback: document.getElementById("loopPlayback"),
    timelineFrame: document.getElementById("timelineFrame"),
    timelineLabel: document.getElementById("timelineLabel"),
    svgViewport: document.getElementById("svgViewport"),
  };

  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js";

  let nextNodeId = 1;
  let nodes = [];

  const perCharacterTypes = new Set([
    "walk",
    "ai_motion",
    "chase",
    "wave",
    "impact",
    "take",
    "anticipation",
  ]);

  const schemas = {
    walk: {
      label: "Walk",
      defaults: { char: "char1", from: "180,390", to: "780,390", speed: 1.05, bounce: 0.32, stride: 58, step_height: 23, cadence: 1.7 },
      fields: [["char", "select", ["char1", "char2"]], ["from", "text"], ["to", "text"], ["speed", "number"], ["bounce", "number"], ["stride", "number"], ["step_height", "number"], ["cadence", "number"]],
    },
    ai_motion: {
      label: "AI Motion",
      defaults: {
        char: "char1",
        model: "motion_model.pt",
        tokenizer_model: "tokenizer_out/pose_tokenizer_model.json",
        start: "200,330",
        target: "620,330",
        steps: 96,
        temperature: 0.95,
        top_k: 24,
        seed: 0,
        prompt: "",
        style: "drunk_walk",
      },
      fields: [
        ["char", "select", ["char1", "char2"]],
        ["model", "text"],
        ["tokenizer_model", "text"],
        ["start", "text"],
        ["target", "text"],
        ["steps", "number"],
        ["temperature", "number"],
        ["top_k", "number"],
        ["seed", "number"],
        ["prompt", "text"],
        ["style", "text"],
      ],
    },
    chase: {
      label: "Chase",
      defaults: { char: "char2", target: "char1", offset: -50, aggression: 1.35, bounce: 0.36, stride: 60, step_height: 24, cadence: 1.9 },
      fields: [["char", "select", ["char1", "char2"]], ["target", "select", ["char1", "char2"]], ["offset", "number"], ["aggression", "number"], ["bounce", "number"], ["stride", "number"], ["step_height", "number"], ["cadence", "number"]],
    },
    wave: {
      label: "Wave",
      defaults: { char: "char1", hand: "right", cycles: 2, amplitude: 26, start: 0.8, duration: 1.2 },
      fields: [["char", "select", ["char1", "char2"]], ["hand", "select", ["left", "right"]], ["cycles", "number"], ["amplitude", "number"], ["start", "number"], ["duration", "number"]],
    },
    impact: {
      label: "Impact",
      defaults: { char: "char1", t: 1.8, direction: "-1,0", force: 0.9, duration: 0.22 },
      fields: [["char", "select", ["char1", "char2"]], ["t", "number"], ["direction", "text"], ["force", "number"], ["duration", "number"]],
    },
    take: {
      label: "Take",
      defaults: { char: "char1", t: 2.1, intensity: 0.8, hold: 2 },
      fields: [["char", "select", ["char1", "char2"]], ["t", "number"], ["intensity", "number"], ["hold", "number"]],
    },
    anticipation: {
      label: "Anticipation",
      defaults: { char: "char2", t: 1.0, action: "sprint", intensity: 1.0, duration: 0.24, direction: "" },
      fields: [["char", "select", ["char1", "char2"]], ["t", "number"], ["action", "text"], ["intensity", "number"], ["duration", "number"], ["direction", "text"]],
    },
    duel_collision: {
      label: "Duel Collision",
      defaults: { distance: 48, force: 1.0, duration: 0.2, take_intensity: 0.72, cooldown: 0.35 },
      fields: [["distance", "number"], ["force", "number"], ["duration", "number"], ["take_intensity", "number"], ["cooldown", "number"]],
    },
    wall: {
      label: "Wall",
      defaults: { x: 640, width: 20, height: 160, force: 0.95, duration: 0.2 },
      fields: [["x", "number"], ["width", "number"], ["height", "number"], ["force", "number"], ["duration", "number"]],
    },
    trapdoor: {
      label: "Trapdoor",
      defaults: { x: 735, width: 110, depth: 82, force: 0.9, duration: 0.24, open_time: 0.24 },
      fields: [["x", "number"], ["width", "number"], ["depth", "number"], ["force", "number"], ["duration", "number"], ["open_time", "number"]],
    },
    anvil: {
      label: "Anvil",
      defaults: { x: 520, size: 36, trigger_x: 470, trigger_radius: 55, delay: 0.08, fall_speed: 390, force: 1.0, duration: 0.2 },
      fields: [["x", "number"], ["size", "number"], ["trigger_x", "number"], ["trigger_radius", "number"], ["delay", "number"], ["fall_speed", "number"], ["force", "number"], ["duration", "number"]],
    },
  };

  class ServerPreviewEngine {
    async render(body) {
      const res = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "preview failed");
      }
      data.engine = "server";
      return data;
    }
  }

  class LocalPyodideEngine {
    constructor() {
      this.ready = false;
      this.initializing = null;
      this.pyodide = null;
    }

    async _loadScript(url) {
      await new Promise((resolve, reject) => {
        const tag = document.createElement("script");
        tag.src = url;
        tag.onload = () => resolve();
        tag.onerror = () => reject(new Error(`failed to load script: ${url}`));
        document.head.appendChild(tag);
      });
    }

    async ensureReady() {
      if (this.ready) {
        return;
      }
      if (this.initializing) {
        await this.initializing;
        return;
      }

      this.initializing = (async () => {
        if (!window.loadPyodide) {
          await this._loadScript(PYODIDE_URL);
        }
        this.pyodide = await window.loadPyodide({
          indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/",
        });

        const py = this.pyodide;
        py.runPython(`import os\nos.chdir('/home/pyodide')`);

        const modules = [
          "cartoon_svg_mvp.py",
          "procedural_walk_cycle.py",
          "procedural_props.py",
          "slapstick_events.py",
          "multi_character_orchestrator.py",
          "vector_skinning.py",
          "physics_hybrid.py",
          "motion_transformer.py",
        ];

        for (const name of modules) {
          const text = await fetch(`/py-src/${name}`).then((r) => {
            if (!r.ok) {
              throw new Error(`missing /py-src/${name}`);
            }
            return r.text();
          });
          py.FS.writeFile(name, text, { encoding: "utf8" });
        }

        const assets = [
          ["assets/meshes/default/mesh.json", "default/mesh.json"],
          ["assets/meshes/default/parts/arm.svg", "default/parts/arm.svg"],
          ["assets/meshes/default/parts/foot.svg", "default/parts/foot.svg"],
          ["assets/meshes/default/parts/hand.svg", "default/parts/hand.svg"],
          ["assets/meshes/default/parts/head.svg", "default/parts/head.svg"],
          ["assets/meshes/default/parts/leg.svg", "default/parts/leg.svg"],
          ["assets/meshes/default/parts/torso.svg", "default/parts/torso.svg"],
        ];

        for (const [fsPath, apiPath] of assets) {
          const blob = await fetch(`/mesh-assets/${apiPath}`).then((r) => {
            if (!r.ok) {
              throw new Error(`missing /mesh-assets/${apiPath}`);
            }
            return r.text();
          });
          const dir = fsPath.substring(0, fsPath.lastIndexOf("/"));
          const parts = dir.split("/");
          let acc = "";
          for (const part of parts) {
            acc = acc ? `${acc}/${part}` : part;
            try {
              py.FS.mkdir(acc);
            } catch (_err) {
              // existing
            }
          }
          py.FS.writeFile(fsPath, blob, { encoding: "utf8" });
        }

        await py.runPythonAsync(`
from cartoon_svg_mvp import parse_script, render_frame_svg, render_multi_frame_svg
from procedural_walk_cycle import parse_action_script, generate_procedural_scene_detailed
from multi_character_orchestrator import parse_multi_script, generate_multi_character_scene_detailed
from procedural_props import build_prop_items_fn
from slapstick_events import ImpactEvent

def _collect_impacts(events):
    return sorted([e.t for e in events if isinstance(e, ImpactEvent)])

def preview_scene(mode, dsl, frame_index=0, t=None):
    mode = (mode or "single").strip().lower()
    if mode == "single":
        try:
            cfg, walk, wave, events, props, ai_motion = parse_action_script(dsl)
            tl, prop_runtime, prop_events, physics_meta = generate_procedural_scene_detailed(
                cfg,
                walk,
                wave,
                ai_motion=ai_motion,
                events=events,
                props=props,
            )
            impacts = _collect_impacts([*events, *prop_events])
            svg = render_frame_svg(
                cfg=cfg,
                tl=tl,
                frame_index=int(frame_index),
                t=t,
                extra_items_fn=build_prop_items_fn(prop_runtime, cfg),
                camera_impact_times=impacts,
            )
            return {
                "svg": svg,
                "mode": "single",
                "frame_index": int(frame_index),
                "fps": cfg.fps,
                "seconds": cfg.seconds,
                "width": cfg.width,
                "height": cfg.height,
                "character_count": 1,
                "physics": physics_meta,
            }
        except Exception:
            cfg, tl = parse_script(dsl)
            svg = render_frame_svg(cfg=cfg, tl=tl, frame_index=int(frame_index), t=t)
            return {
                "svg": svg,
                "mode": "single",
                "frame_index": int(frame_index),
                "fps": cfg.fps,
                "seconds": cfg.seconds,
                "width": cfg.width,
                "height": cfg.height,
                "character_count": 1,
                "physics": {"enabled": False, "solver": "off"},
            }

    cfg, programs, props, collision = parse_multi_script(dsl)
    timelines, prop_runtime, collision_events, prop_events, physics_meta = generate_multi_character_scene_detailed(
        cfg=cfg,
        programs=programs,
        props=props,
        collision=collision,
    )
    order = list(programs.keys())
    styles = {cid: programs[cid].style for cid in order}
    impacts = []
    for program in programs.values():
        impacts.extend(_collect_impacts(program.events))
    for rows in prop_events.values():
        impacts.extend(_collect_impacts(rows))
    for rows in collision_events.values():
        impacts.extend(_collect_impacts(rows))
    impacts = sorted(set(impacts))

    svg = render_multi_frame_svg(
        cfg=cfg,
        timelines=timelines,
        styles=styles,
        character_order=order,
        frame_index=int(frame_index),
        t=t,
        extra_items_fn=build_prop_items_fn(prop_runtime, cfg),
        camera_impact_times=impacts,
    )
    return {
        "svg": svg,
        "mode": "multi",
        "frame_index": int(frame_index),
        "fps": cfg.fps,
        "seconds": cfg.seconds,
        "width": cfg.width,
        "height": cfg.height,
        "character_count": len(order),
        "physics": {"mode_requested": cfg.physics_mode, "characters": physics_meta},
    }
`);

        this.ready = true;
      })();

      await this.initializing;
    }

    async render(body) {
      await this.ensureReady();
      const py = this.pyodide;
      py.globals.set("JS_MODE", body.mode);
      py.globals.set("JS_DSL", body.dsl);
      py.globals.set("JS_FRAME", Number(body.frame_index) || 0);
      py.globals.set("JS_T", body.t == null ? null : Number(body.t));
      await py.runPythonAsync(`JS_RES = preview_scene(JS_MODE, JS_DSL, JS_FRAME, JS_T)`);
      const proxy = py.globals.get("JS_RES");
      const data = proxy.toJs({ dict_converter: Object.fromEntries });
      proxy.destroy();
      py.globals.delete("JS_MODE");
      py.globals.delete("JS_DSL");
      py.globals.delete("JS_FRAME");
      py.globals.delete("JS_T");
      py.globals.delete("JS_RES");
      data.engine = "local";
      return data;
    }
  }

  const serverEngine = new ServerPreviewEngine();
  const localEngine = new LocalPyodideEngine();
  let previewTimer = null;
  let previewInFlight = false;
  let isPlaying = false;
  let playbackTimer = null;
  let playbackToken = 0;
  let lastRenderInfo = {
    fps: Math.max(1, Number(els.fps.value) || 24),
    seconds: Math.max(0, Number(els.seconds.value) || 0),
  };

  function clampFrame(frame) {
    const maxFrame = Math.max(0, Number(els.timelineFrame.max) || 0);
    return Math.max(0, Math.min(maxFrame, Math.floor(Number(frame) || 0)));
  }

  function updateTimelineLabel() {
    const frame = Math.max(0, Number(els.frameIndex.value) || 0);
    const maxFrame = Math.max(0, Number(els.timelineFrame.max) || 0);
    const fps = Math.max(1, Number(lastRenderInfo.fps) || 24);
    const t = frame / fps;
    els.timelineLabel.textContent = `frame ${frame}/${maxFrame} | t=${t.toFixed(2)}s`;
  }

  function syncFrameControls(frame) {
    const clamped = clampFrame(frame);
    els.frameIndex.value = String(clamped);
    els.timelineFrame.value = String(clamped);
    updateTimelineLabel();
  }

  function configureTimeline(fps, seconds) {
    const safeFps = Math.max(1, Number(fps) || 24);
    const safeSeconds = Math.max(0, Number(seconds) || 0);
    lastRenderInfo = { fps: safeFps, seconds: safeSeconds };
    const totalFrames = Math.max(1, Math.floor(safeSeconds * safeFps) + 1);
    els.timelineFrame.max = String(totalFrames - 1);
    syncFrameControls(Number(els.frameIndex.value) || 0);
  }

  function setPlaybackState(playing) {
    isPlaying = playing;
    els.playPauseBtn.textContent = playing ? "Pause" : "Play";
    els.playPauseBtn.setAttribute("aria-pressed", playing ? "true" : "false");
  }

  function pausePlayback() {
    playbackToken += 1;
    if (playbackTimer) {
      clearTimeout(playbackTimer);
      playbackTimer = null;
    }
    setPlaybackState(false);
  }

  async function stopPlayback() {
    pausePlayback();
    syncFrameControls(0);
    await runPreview(0);
  }

  async function startPlayback() {
    if (isPlaying) {
      return;
    }
    setPlaybackState(true);
    playbackToken += 1;
    const token = playbackToken;

    const step = async () => {
      if (!isPlaying || token !== playbackToken) {
        return;
      }
      const frame = clampFrame(Number(els.frameIndex.value) || 0);
      const startedAt = performance.now();
      const ok = await runPreview(frame, { fromPlayback: true });
      if (!ok || !isPlaying || token !== playbackToken) {
        if (!ok) {
          pausePlayback();
        }
        return;
      }

      const maxFrame = Math.max(0, Number(els.timelineFrame.max) || 0);
      let nextFrame = frame + 1;
      if (nextFrame > maxFrame) {
        if (els.loopPlayback.checked) {
          nextFrame = 0;
        } else {
          pausePlayback();
          return;
        }
      }
      syncFrameControls(nextFrame);

      const frameMs = 1000 / Math.max(1, Number(lastRenderInfo.fps) || 24);
      const waitMs = Math.max(0, frameMs - (performance.now() - startedAt));
      playbackTimer = setTimeout(step, waitMs);
    };

    step();
  }

  function fmt(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) {
      return String(v);
    }
    return Number.isInteger(n) ? String(n) : n.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  }

  function mkNode(type, overrides = {}) {
    const schema = schemas[type];
    return { id: nextNodeId++, type, ...schema.defaults, ...overrides };
  }

  function defaultNodesFor(mode) {
    if (mode === "multi") {
      return [
        mkNode("walk", { char: "char1", from: "170,390", to: "790,390" }),
        mkNode("wave", { char: "char1", start: 0.9 }),
        mkNode("chase", { char: "char2", target: "char1", offset: -50 }),
        mkNode("anticipation", { char: "char2", t: 1.0 }),
        mkNode("duel_collision", {}),
        mkNode("wall", {}),
      ];
    }
    return [
      mkNode("walk", { char: "char1", from: "200,330", to: "620,330" }),
      mkNode("wave", { char: "char1", start: 1.1 }),
      mkNode("impact", { char: "char1", t: 2.0 }),
      mkNode("wall", { x: 430, width: 18, height: 140 }),
    ];
  }

  function lineForStyle(charIndex) {
    const seed = charIndex === 1 ? els.seed1.value : els.seed2.value;
    const mode = charIndex === 1 ? els.mode1.value : els.mode2.value;
    const line = charIndex === 1 ? els.line1.value : els.line2.value;
    const limb = charIndex === 1 ? els.limb1.value : els.limb2.value;
    const head = charIndex === 1 ? els.head1.value : els.head2.value;
    const jitter = charIndex === 1 ? els.jitter1.value : els.jitter2.value;
    const mesh = (charIndex === 1 ? els.mesh1.value : els.mesh2.value).trim();
    const tint = (charIndex === 1 ? els.tint1.value : els.tint2.value).trim();

    let out = `seed "${seed}" line ${fmt(line)} limb ${fmt(limb)} head ${fmt(head)} jitter ${fmt(jitter)} ` +
      `smear_threshold 140 smear_full 480 smear_stretch 0.45 smear_squeeze 0.28 smear_jitter 1.0 mode ${mode}`;
    if (mode === "mesh" && mesh) {
      out += ` mesh "${mesh}"`;
    }
    if (mode === "mesh" && tint) {
      out += ` tint "${tint}"`;
    }
    return out;
  }

  function nodeToCommand(node) {
    switch (node.type) {
      case "walk":
        return `walk from=${node.from} to=${node.to} speed=${fmt(node.speed)} bounce=${fmt(node.bounce)} stride=${fmt(node.stride)} step_height=${fmt(node.step_height)} cadence=${fmt(node.cadence)}`;
      case "ai_motion":
        return `ai_motion model="${node.model}" tokenizer_model="${node.tokenizer_model}" start=${node.start} target=${node.target} steps=${Math.max(2, Number(node.steps) || 96)} temperature=${fmt(node.temperature)} top_k=${Math.max(1, Number(node.top_k) || 1)} seed=${Math.max(0, Number(node.seed) || 0)} prompt="${node.prompt || ""}" style="${node.style || ""}"`;
      case "chase":
        return `chase target=${node.target} offset=${fmt(node.offset)} aggression=${fmt(node.aggression)} bounce=${fmt(node.bounce)} stride=${fmt(node.stride)} step_height=${fmt(node.step_height)} cadence=${fmt(node.cadence)}`;
      case "wave":
        return `wave hand=${node.hand} cycles=${fmt(node.cycles)} amplitude=${fmt(node.amplitude)} start=${fmt(node.start)} duration=${fmt(node.duration)}`;
      case "impact":
        return `impact t=${fmt(node.t)} direction=${node.direction || "-1,0"} force=${fmt(node.force)} duration=${fmt(node.duration)}`;
      case "take":
        return `take t=${fmt(node.t)} intensity=${fmt(node.intensity)} hold=${Math.max(0, Number(node.hold) || 0)}`;
      case "anticipation": {
        let s = `anticipation t=${fmt(node.t)} action=${node.action || "move"} intensity=${fmt(node.intensity)} duration=${fmt(node.duration)}`;
        if ((node.direction || "").trim()) {
          s += ` direction=${node.direction.trim()}`;
        }
        return s;
      }
      case "duel_collision":
        return `duel_collision distance=${fmt(node.distance)} force=${fmt(node.force)} duration=${fmt(node.duration)} take_intensity=${fmt(node.take_intensity)} cooldown=${fmt(node.cooldown)}`;
      case "wall":
        return `wall x=${fmt(node.x)} width=${fmt(node.width)} height=${fmt(node.height)} force=${fmt(node.force)} duration=${fmt(node.duration)}`;
      case "trapdoor":
        return `trapdoor x=${fmt(node.x)} width=${fmt(node.width)} depth=${fmt(node.depth)} force=${fmt(node.force)} duration=${fmt(node.duration)} open_time=${fmt(node.open_time)}`;
      case "anvil":
        return `anvil x=${fmt(node.x)} size=${fmt(node.size)} trigger_x=${fmt(node.trigger_x)} trigger_radius=${fmt(node.trigger_radius)} delay=${fmt(node.delay)} fall_speed=${fmt(node.fall_speed)} force=${fmt(node.force)} duration=${fmt(node.duration)}`;
      default:
        return "";
    }
  }

  function buildDsl() {
    const mode = els.sceneMode.value;
    const lines = [
      "# Generated by web editor",
      `canvas ${fmt(els.canvasW.value)} ${fmt(els.canvasH.value)}`,
      `fps ${fmt(els.fps.value)}`,
      `seconds ${fmt(els.seconds.value)}`,
      `physics mode=${els.physicsMode.value} gravity=${fmt(els.gravity.value)} damping=${fmt(els.damping.value)} restitution=${fmt(els.restitution.value)} friction=${fmt(els.friction.value)} substeps=${fmt(els.substeps.value)}`,
      `camera enabled=${els.cameraEnabled.value} focus=${(els.cameraFocus.value || "none").trim()} zoom=${fmt(els.cameraZoom.value)} pan=${(els.cameraPan.value || "0,0").trim()} depth=${els.cameraDepth.value} depth_min=${fmt(els.cameraDepthMin.value)} depth_max=${fmt(els.cameraDepthMax.value)} parallax=${fmt(els.cameraParallax.value)} y_sort=${els.cameraYSort.value} shake_on_impact=${els.cameraShake.value} shake_amp=${fmt(els.cameraShakeAmp.value)} shake_freq=${fmt(els.cameraShakeFreq.value)} shake_decay=${fmt(els.cameraShakeDecay.value)}`,
    ];

    if (mode === "single") {
      lines.push(`character ${lineForStyle(1)}`);
    } else {
      lines.push(`character id=char1 ${lineForStyle(1)}`);
      lines.push(`character id=char2 ${lineForStyle(2)}`);
    }

    for (const node of nodes) {
      if (mode === "single" && (node.type === "chase" || node.type === "duel_collision")) {
        continue;
      }
      const cmd = nodeToCommand(node);
      if (!cmd) {
        continue;
      }
      if (mode === "multi" && perCharacterTypes.has(node.type)) {
        const owner = node.char || "char1";
        lines.push(`${owner}: ${cmd}`);
      } else {
        lines.push(cmd);
      }
    }

    const text = lines.join("\n") + "\n";
    els.dsl.value = text;
    return text;
  }

  function setStatus(message, isError = false) {
    els.status.textContent = message;
    els.status.classList.toggle("error", isError);
  }

  function renderNodes() {
    const mode = els.sceneMode.value;
    els.char2StyleWrap.classList.toggle("hidden", mode !== "multi");

    const frag = document.createDocumentFragment();
    for (const node of nodes) {
      const schema = schemas[node.type];
      const root = document.createElement("div");
      root.className = "node";

      const head = document.createElement("div");
      head.className = "node-head";
      const title = document.createElement("div");
      title.className = "node-title";
      title.textContent = schema.label;
      head.appendChild(title);

      const actions = document.createElement("div");
      actions.className = "node-actions";
      for (const [label, handler] of [
        ["Up", () => {
          const idx = nodes.findIndex((n) => n.id === node.id);
          if (idx > 0) {
            [nodes[idx - 1], nodes[idx]] = [nodes[idx], nodes[idx - 1]];
            onStructureChanged();
          }
        }],
        ["Down", () => {
          const idx = nodes.findIndex((n) => n.id === node.id);
          if (idx >= 0 && idx < nodes.length - 1) {
            [nodes[idx + 1], nodes[idx]] = [nodes[idx], nodes[idx + 1]];
            onStructureChanged();
          }
        }],
        ["Delete", () => {
          nodes = nodes.filter((n) => n.id !== node.id);
          onStructureChanged();
        }],
      ]) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.onclick = handler;
        actions.appendChild(btn);
      }
      head.appendChild(actions);
      root.appendChild(head);

      const grid = document.createElement("div");
      grid.className = "node-grid";

      for (const [field, kind, options] of schema.fields) {
        if (mode === "single" && (field === "char" || field === "target")) {
          continue;
        }

        const label = document.createElement("label");
        label.textContent = field;
        let input;
        if (kind === "select") {
          input = document.createElement("select");
          for (const opt of options || []) {
            const o = document.createElement("option");
            o.value = opt;
            o.textContent = opt;
            input.appendChild(o);
          }
        } else {
          input = document.createElement("input");
          input.type = kind === "number" ? "number" : "text";
          if (kind === "number") {
            input.step = "0.01";
          }
        }

        input.value = node[field] == null ? "" : String(node[field]);
        input.oninput = () => {
          node[field] = input.value;
          onDslChanged();
        };
        label.appendChild(input);
        grid.appendChild(label);
      }

      root.appendChild(grid);
      frag.appendChild(root);
    }

    els.nodes.innerHTML = "";
    els.nodes.appendChild(frag);
  }

  function onDslChanged() {
    if (isPlaying) {
      pausePlayback();
    }
    buildDsl();
    configureTimeline(Number(els.fps.value) || 24, Number(els.seconds.value) || 0);
    if (els.autoPreview.checked) {
      schedulePreview();
    }
  }

  function onStructureChanged() {
    renderNodes();
    onDslChanged();
  }

  function schedulePreview() {
    if (previewTimer) {
      clearTimeout(previewTimer);
    }
    previewTimer = setTimeout(() => {
      previewTimer = null;
      runPreview();
    }, 220);
  }

  async function runPreview(frameOverride = null, options = {}) {
    if (previewInFlight) {
      return false;
    }
    previewInFlight = true;

    const requestedFrame = frameOverride == null
      ? Number(els.frameIndex.value) || 0
      : Number(frameOverride) || 0;
    syncFrameControls(requestedFrame);

    const body = {
      mode: els.sceneMode.value,
      dsl: els.dsl.value,
      frame_index: clampFrame(requestedFrame),
    };

    if (!(options.fromPlayback && isPlaying)) {
      setStatus(`rendering (${els.previewEngine.value})...`);
    }

    try {
      const engine = els.previewEngine.value === "local" ? localEngine : serverEngine;
      const data = await engine.render(body);
      els.svgViewport.innerHTML = data.svg;
      configureTimeline(data.fps, data.seconds);
      syncFrameControls(body.frame_index);

      const physicsText = data.physics
        ? ` | physics=${(data.physics.solver || data.physics.mode_requested || "n/a")}`
        : "";
      els.previewMeta.textContent = `${data.mode} | chars=${data.character_count} | ${data.width}x${data.height} | fps=${data.fps} | seconds=${data.seconds}${physicsText} | engine=${data.engine}`;
      setStatus(options.fromPlayback && isPlaying ? `playing (${data.engine})` : `ok (${data.engine})`);
      return true;
    } catch (err) {
      if (els.previewEngine.value === "local") {
        // fallback path for robustness
        try {
          const data = await serverEngine.render(body);
          els.svgViewport.innerHTML = data.svg;
          configureTimeline(data.fps, data.seconds);
          syncFrameControls(body.frame_index);
          els.previewMeta.textContent = `${data.mode} | chars=${data.character_count} | ${data.width}x${data.height} | fps=${data.fps} | seconds=${data.seconds} | engine=server (fallback)`;
          setStatus(`local failed, fallback server: ${String(err)}`, true);
          return true;
        } catch (err2) {
          els.previewMeta.textContent = "Preview error";
          els.svgViewport.innerHTML = "";
          setStatus(`local+server failed: ${String(err2)}`, true);
          return false;
        }
      }
      els.previewMeta.textContent = "Preview error";
      els.svgViewport.innerHTML = "";
      setStatus(String(err), true);
      return false;
    } finally {
      previewInFlight = false;
    }
  }

  function hookSimpleInputs() {
    const simpleIds = [
      "previewEngine", "sceneMode", "seconds", "canvasW", "canvasH", "fps",
      "seed1", "mode1", "line1", "limb1", "head1", "jitter1", "mesh1", "tint1",
      "seed2", "mode2", "line2", "limb2", "head2", "jitter2", "mesh2", "tint2",
      "physicsMode", "gravity", "damping", "restitution", "friction", "substeps",
      "cameraEnabled", "cameraFocus", "cameraZoom", "cameraPan", "cameraDepth", "cameraParallax",
      "cameraDepthMin", "cameraDepthMax", "cameraYSort", "cameraShake", "cameraShakeAmp", "cameraShakeFreq", "cameraShakeDecay",
    ];

    for (const id of simpleIds) {
      const el = document.getElementById(id);
      el?.addEventListener("input", onDslChanged);
      el?.addEventListener("change", onDslChanged);
    }

    els.sceneMode.addEventListener("change", () => {
      nodes = defaultNodesFor(els.sceneMode.value);
      onStructureChanged();
    });

    els.addNodeBtn.addEventListener("click", () => {
      const type = els.nodeType.value;
      nodes.push(mkNode(type));
      onStructureChanged();
    });

    els.resetBtn.addEventListener("click", () => {
      nodes = defaultNodesFor(els.sceneMode.value);
      onStructureChanged();
    });

    els.frameIndex.addEventListener("input", () => {
      if (isPlaying) {
        pausePlayback();
      }
      syncFrameControls(Number(els.frameIndex.value) || 0);
      if (els.autoPreview.checked) {
        schedulePreview();
      }
    });
    els.frameIndex.addEventListener("change", () => {
      syncFrameControls(Number(els.frameIndex.value) || 0);
      if (els.autoPreview.checked) {
        schedulePreview();
      }
    });

    els.timelineFrame.addEventListener("input", () => {
      if (isPlaying) {
        pausePlayback();
      }
      syncFrameControls(Number(els.timelineFrame.value) || 0);
      if (els.autoPreview.checked) {
        schedulePreview();
      }
    });
    els.timelineFrame.addEventListener("change", () => {
      syncFrameControls(Number(els.timelineFrame.value) || 0);
      if (els.autoPreview.checked) {
        schedulePreview();
      }
    });

    els.previewBtn.addEventListener("click", () => runPreview());
    els.playPauseBtn.addEventListener("click", () => {
      if (isPlaying) {
        pausePlayback();
      } else {
        startPlayback();
      }
    });
    els.stopBtn.addEventListener("click", () => stopPlayback());
  }

  function init() {
    hookSimpleInputs();
    nodes = defaultNodesFor("single");
    renderNodes();
    buildDsl();
    configureTimeline(Number(els.fps.value) || 24, Number(els.seconds.value) || 0);
    setPlaybackState(false);
    runPreview();
  }

  init();
})();
