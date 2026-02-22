#!/usr/bin/env python3
"""
motion_transformer.py

Generative transformer for pose-token sequence synthesis.

Training input:
- tokens manifest from pose_tokenizer.py (tokens_manifest.jsonl)
- tokenizer model (pose_tokenizer_model.json)

Inference output:
- synthesized token sequence
- decoded feature sequence via tokenizer codebook
- reconstructed Timeline of Pose objects
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cartoon_svg_mvp import Pose, Timeline, Vec2


# -----------------------------
# Tokenizer decode utilities
# -----------------------------


def load_tokenizer_model(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"invalid tokenizer model json: {p}")
    required = ["codebook", "normalization", "codebook_size", "feature_dim"]
    for key in required:
        if key not in obj:
            raise ValueError(f"tokenizer model missing field: {key}")
    return obj


def _denorm_feature(norm_vec: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [norm_vec[i] * std[i] + mean[i] for i in range(len(norm_vec))]


def token_to_feature(token: int, tokenizer_model: dict[str, Any]) -> list[float]:
    codebook = tokenizer_model["codebook"]
    n = len(codebook)
    if n == 0:
        raise ValueError("empty tokenizer codebook")
    idx = int(token) % n
    norm = codebook[idx]
    mean = tokenizer_model["normalization"]["mean"]
    std = tokenizer_model["normalization"]["std"]
    return _denorm_feature(norm, mean, std)


def tokens_to_features(tokens: list[int], tokenizer_model: dict[str, Any]) -> list[list[float]]:
    return [token_to_feature(tok, tokenizer_model) for tok in tokens]


# -----------------------------
# Pose reconstruction
# -----------------------------


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _apply_style_features(features: list[list[float]], *, style: str, prompt: str) -> list[list[float]]:
    s = (style or "").strip().lower()
    p = (prompt or "").strip().lower()
    out: list[list[float]] = []

    for i, feat in enumerate(features):
        row = feat[:]
        phase = i * 0.22

        if "sneak" in s or "stealth" in s or "sneak" in p:
            row[0] *= 0.55
            row[1] *= 0.65
            row[3] -= 10.0
            row[5] -= 10.0
            row[11] = clamp(row[11] * 0.7, 0.0, 1.0)

        if "run" in s or "sprint" in s or "run" in p:
            row[0] *= 1.45
            row[6] *= 1.10
            row[8] *= 1.10
            row[11] = clamp(row[11] * 1.12, 0.0, 1.0)

        if "drunk" in s or "wobble" in s or "drunk" in p:
            row[0] *= 0.95
            row[1] += math.sin(phase) * 1.35
            row[2] += math.sin(phase * 1.3) * 6.0
            row[4] -= math.sin(phase * 1.3) * 6.0
            row[10] += math.sin(phase * 0.9) * 0.08

        if "dance" in s or "dance" in p:
            row[2] += math.sin(phase * 2.3) * 16.0
            row[4] -= math.sin(phase * 2.1) * 16.0
            row[3] += math.cos(phase * 1.8) * 9.0
            row[5] += math.cos(phase * 1.7) * 9.0

        out.append(row)

    return out


def features_to_timeline(
    features: list[list[float]],
    *,
    start: Vec2,
    target: Vec2,
    fps: int,
    style: str = "",
    prompt: str = "",
) -> Timeline:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if not features:
        raise ValueError("no features to reconstruct")

    seq = _apply_style_features(features, style=style, prompt=prompt)

    # Integrate root deltas first.
    roots: list[Vec2] = []
    root = start
    for feat in seq:
        root = Vec2(root.x + feat[0], root.y + feat[1])
        roots.append(root)

    # Soft target steering to hit requested endpoint.
    if roots:
        diff = target - roots[-1]
        n = max(1, len(roots) - 1)
        corrected: list[Vec2] = []
        for i, r in enumerate(roots):
            u = i / n
            w = u * u * (3.0 - 2.0 * u)
            corrected.append(Vec2(r.x + diff.x * w, r.y + diff.y * w))
        roots = corrected

    tl = Timeline()
    dt = 1.0 / fps
    for i, feat in enumerate(seq):
        t = i * dt
        root = roots[i]
        pose = Pose(
            root=root,
            l_hand=Vec2(feat[2], feat[3]),
            r_hand=Vec2(feat[4], feat[5]),
            l_foot=Vec2(feat[6], feat[7]),
            r_foot=Vec2(feat[8], feat[9]),
            look_angle=feat[10],
            squash=clamp(feat[11], 0.0, 1.0),
        )
        tl.add(t, pose)
    return tl


# -----------------------------
# Transformer LM
# -----------------------------


@dataclass(slots=True)
class MotionModelConfig:
    vocab_size: int
    bos_id: int
    eos_id: int
    pad_id: int
    d_model: int = 192
    n_head: int = 6
    n_layer: int = 6
    ff_mult: int = 4
    dropout: float = 0.1
    max_len: int = 512


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch required. Install torch to use motion_transformer.") from exc
    return torch, nn, F


def build_model(cfg: MotionModelConfig):
    torch, nn, _ = _require_torch()

    class MotionTransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_head,
                dim_feedforward=cfg.d_model * cfg.ff_mult,
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layer)
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

        def forward(self, idx):
            bsz, tlen = idx.shape
            if tlen > cfg.max_len:
                raise ValueError(f"sequence length {tlen} exceeds max_len {cfg.max_len}")
            pos = torch.arange(tlen, device=idx.device).unsqueeze(0).expand(bsz, tlen)
            x = self.token_emb(idx) + self.pos_emb(pos)
            mask = torch.full((tlen, tlen), float("-inf"), device=idx.device)
            mask = torch.triu(mask, diagonal=1)
            x = self.encoder(x, mask=mask)
            x = self.norm(x)
            return self.head(x)

    return MotionTransformerLM()


def load_token_sequences(manifest_path: Path, *, max_sequences: int = 0, min_len: int = 8) -> list[list[int]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")

    seqs: list[list[int]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            token_file = row.get("token_file")
            if not token_file:
                continue
            token_path = Path(token_file)
            if not token_path.is_absolute():
                token_path = (manifest_path.parent / token_path).resolve()
            if not token_path.exists():
                continue
            tok_obj = json.loads(token_path.read_text(encoding="utf-8"))
            tokens = tok_obj.get("tokens")
            if not isinstance(tokens, list):
                continue
            seq = [int(v) for v in tokens]
            if len(seq) < min_len:
                continue
            seqs.append(seq)
            if max_sequences > 0 and len(seqs) >= max_sequences:
                break
    if not seqs:
        raise ValueError("no valid token sequences found")
    return seqs


def _batched(seqs: list[list[int]], batch_size: int):
    for i in range(0, len(seqs), batch_size):
        yield seqs[i : i + batch_size]


def _make_batch(batch: list[list[int]], *, bos_id: int, eos_id: int, pad_id: int):
    torch, _, _ = _require_torch()
    max_t = max(len(s) + 1 for s in batch)
    x = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    y = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        inp = [bos_id, *seq]
        tgt = [*seq, eos_id]
        x[i, : len(inp)] = torch.tensor(inp, dtype=torch.long)
        y[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
    return x, y


def train_motion_transformer(
    *,
    tokenizer_manifest_path: str,
    tokenizer_model_path: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    d_model: int,
    n_head: int,
    n_layer: int,
    max_len: int,
    seed: int,
    max_sequences: int,
) -> dict[str, Any]:
    torch, _, F = _require_torch()

    rng = random.Random(seed)
    torch.manual_seed(seed)

    tokenizer_model = load_tokenizer_model(tokenizer_model_path)
    codebook_size = int(tokenizer_model["codebook_size"])
    bos_id = codebook_size
    eos_id = codebook_size + 1
    pad_id = codebook_size + 2

    cfg = MotionModelConfig(
        vocab_size=codebook_size + 3,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        max_len=max_len,
    )

    seqs = load_token_sequences(Path(tokenizer_manifest_path), max_sequences=max_sequences)
    rng.shuffle(seqs)

    model = build_model(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    losses: list[float] = []
    model.train()
    for ep in range(epochs):
        rng.shuffle(seqs)
        running = 0.0
        count = 0
        for batch in _batched(seqs, batch_size):
            x, y = _make_batch(batch, bos_id=cfg.bos_id, eos_id=cfg.eos_id, pad_id=cfg.pad_id)
            x = x[:, : cfg.max_len].to(device)
            y = y[:, : cfg.max_len].to(device)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                y.reshape(-1),
                ignore_index=cfg.pad_id,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += float(loss.item())
            count += 1

        ep_loss = running / max(1, count)
        losses.append(ep_loss)
        print(f"[train] epoch={ep+1}/{epochs} loss={ep_loss:.4f}")

    out = {
        "model_config": {
            "vocab_size": cfg.vocab_size,
            "bos_id": cfg.bos_id,
            "eos_id": cfg.eos_id,
            "pad_id": cfg.pad_id,
            "d_model": cfg.d_model,
            "n_head": cfg.n_head,
            "n_layer": cfg.n_layer,
            "ff_mult": cfg.ff_mult,
            "dropout": cfg.dropout,
            "max_len": cfg.max_len,
        },
        "tokenizer_model_path": str(Path(tokenizer_model_path).as_posix()),
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "num_sequences": len(seqs),
        "loss_history": losses,
    }

    checkpoint = {
        "meta": out,
        "state_dict": model.state_dict(),
    }
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_file)
    print(f"[ok] model saved: {out_file.resolve()}")
    return out


def load_motion_model(path: str | Path):
    torch, _, _ = _require_torch()
    ckpt = torch.load(Path(path), map_location="cpu")
    meta = ckpt.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("invalid checkpoint (missing meta)")
    cfg_obj = meta.get("model_config")
    if not isinstance(cfg_obj, dict):
        raise ValueError("invalid checkpoint (missing model_config)")

    cfg = MotionModelConfig(
        vocab_size=int(cfg_obj["vocab_size"]),
        bos_id=int(cfg_obj["bos_id"]),
        eos_id=int(cfg_obj["eos_id"]),
        pad_id=int(cfg_obj["pad_id"]),
        d_model=int(cfg_obj["d_model"]),
        n_head=int(cfg_obj["n_head"]),
        n_layer=int(cfg_obj["n_layer"]),
        ff_mult=int(cfg_obj.get("ff_mult", 4)),
        dropout=float(cfg_obj.get("dropout", 0.1)),
        max_len=int(cfg_obj["max_len"]),
    )
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, meta


def _sample_next(logits, *, temperature: float, top_k: int):
    torch, _, _ = _require_torch()
    temperature = max(1e-3, temperature)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_k > 0:
        top_k = min(top_k, probs.shape[-1])
        vals, idx = torch.topk(probs, k=top_k, dim=-1)
        vals = vals / vals.sum(dim=-1, keepdim=True)
        pick = torch.multinomial(vals, num_samples=1)
        return idx.gather(-1, pick)
    return torch.multinomial(probs, num_samples=1)


def style_sampling_params(style: str, prompt: str, *, temperature: float, top_k: int) -> tuple[float, int]:
    s = (style or "").lower()
    p = (prompt or "").lower()
    temp = temperature
    k = top_k

    if "sneak" in s or "stealth" in s or "sneak" in p:
        temp *= 0.78
        k = max(8, min(k, 18))
    if "drunk" in s or "wobble" in s or "drunk" in p:
        temp *= 1.22
        k = max(k, 30)
    if "run" in s or "sprint" in s:
        temp *= 1.06
        k = max(k, 24)
    if "dance" in s:
        temp *= 1.18
        k = max(k, 36)

    return max(0.3, min(1.8, temp)), max(1, k)


def sample_tokens(
    model,
    cfg: MotionModelConfig,
    *,
    steps: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> list[int]:
    torch, _, _ = _require_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    g = torch.Generator(device=device)
    if seed > 0:
        g.manual_seed(seed)

    tokens = [cfg.bos_id]
    for _ in range(max(1, steps)):
        x = torch.tensor(tokens[-cfg.max_len :], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)[:, -1, :]
            # Keep generated space inside pose codebook ids for decoding.
            logits[:, cfg.bos_id :] = -1e9
            nxt = _sample_next(logits, temperature=temperature, top_k=top_k)
        nxt_id = int(nxt.item())
        tokens.append(nxt_id)

    # strip BOS
    return [tok for tok in tokens[1:] if tok < cfg.bos_id]


def generate_timeline_from_model(
    *,
    model_path: str,
    tokenizer_model_path: str,
    start: Vec2,
    target: Vec2,
    fps: int,
    steps: int,
    temperature: float,
    top_k: int,
    seed: int,
    prompt: str = "",
    style: str = "",
) -> tuple[Timeline, float]:
    model, model_cfg, _meta = load_motion_model(model_path)
    tokenizer_model = load_tokenizer_model(tokenizer_model_path)
    t_adj, k_adj = style_sampling_params(style, prompt, temperature=temperature, top_k=top_k)
    tokens = sample_tokens(
        model,
        model_cfg,
        steps=max(2, steps),
        temperature=t_adj,
        top_k=k_adj,
        seed=seed,
    )
    feats = tokens_to_features(tokens, tokenizer_model)
    tl = features_to_timeline(
        feats,
        start=start,
        target=target,
        fps=fps,
        style=style,
        prompt=prompt,
    )
    seconds = max(0.0, (len(feats) - 1) / max(1, fps))
    return tl, seconds


# -----------------------------
# CLI
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and sample a motion transformer for cartoon pose tokens.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train model from pose_tokenizer outputs.")
    tr.add_argument("--tokenizer-manifest", type=str, required=True, help="Path to tokens_manifest.jsonl")
    tr.add_argument("--tokenizer-model", type=str, required=True, help="Path to pose_tokenizer_model.json")
    tr.add_argument("--out", type=str, required=True, help="Output .pt checkpoint")
    tr.add_argument("--epochs", type=int, default=12)
    tr.add_argument("--batch-size", type=int, default=24)
    tr.add_argument("--lr", type=float, default=3e-4)
    tr.add_argument("--d-model", type=int, default=192)
    tr.add_argument("--n-head", type=int, default=6)
    tr.add_argument("--n-layer", type=int, default=6)
    tr.add_argument("--max-len", type=int, default=512)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--max-sequences", type=int, default=0)

    sm = sub.add_parser("sample", help="Sample tokens and write a scene script using ai_motion DSL.")
    sm.add_argument("--model", type=str, required=True)
    sm.add_argument("--tokenizer-model", type=str, required=True)
    sm.add_argument("--out-script", type=str, default="ai_motion_demo.txt")
    sm.add_argument("--fps", type=int, default=24)
    sm.add_argument("--steps", type=int, default=96)
    sm.add_argument("--start", type=str, default="200,330")
    sm.add_argument("--target", type=str, default="620,330")
    sm.add_argument("--temperature", type=float, default=0.95)
    sm.add_argument("--top-k", type=int, default=24)
    sm.add_argument("--seed", type=int, default=0)
    sm.add_argument("--prompt", type=str, default="")
    sm.add_argument("--style", type=str, default="")

    args = ap.parse_args()

    if args.cmd == "train":
        train_motion_transformer(
            tokenizer_manifest_path=args.tokenizer_manifest,
            tokenizer_model_path=args.tokenizer_model,
            out_path=args.out,
            epochs=max(1, args.epochs),
            batch_size=max(1, args.batch_size),
            lr=max(1e-6, args.lr),
            d_model=max(32, args.d_model),
            n_head=max(1, args.n_head),
            n_layer=max(1, args.n_layer),
            max_len=max(32, args.max_len),
            seed=args.seed,
            max_sequences=max(0, args.max_sequences),
        )
        return 0

    # sample
    sx, sy = [float(v) for v in args.start.split(",")]
    tx, ty = [float(v) for v in args.target.split(",")]
    start = Vec2(sx, sy)
    target = Vec2(tx, ty)

    _tl, seconds = generate_timeline_from_model(
        model_path=args.model,
        tokenizer_model_path=args.tokenizer_model,
        start=start,
        target=target,
        fps=max(1, args.fps),
        steps=max(2, args.steps),
        temperature=args.temperature,
        top_k=max(1, args.top_k),
        seed=max(0, args.seed),
        prompt=args.prompt,
        style=args.style,
    )

    scene = (
        "# AI motion demo\n"
        "canvas 800 450\n"
        f"fps {max(1, args.fps)}\n"
        f"seconds {seconds:.3f}\n"
        "physics mode=off\n"
        "camera enabled=true focus=self zoom=1.05 depth=true parallax=0.18 shake_on_impact=true\n"
        "character seed \"AI-MOTION\" line 7 limb 10 head 28 jitter 1.0 mode stick\n"
        f"ai_motion model=\"{args.model}\" tokenizer_model=\"{args.tokenizer_model}\" "
        f"start={args.start} target={args.target} steps={max(2, args.steps)} "
        f"temperature={args.temperature:.3f} top_k={max(1, args.top_k)} seed={max(0, args.seed)} "
        f"prompt=\"{args.prompt}\" style=\"{args.style}\"\n"
    )
    out_path = Path(args.out_script)
    out_path.write_text(scene, encoding="utf-8")
    print(f"[ok] sample scene script written: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
