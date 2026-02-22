#!/usr/bin/env python3
"""
pose_tokenizer.py

Learns a discrete motion codebook from procedural cartoon poses and exports
token sequences per clip/character for transformer-style training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from cartoon_svg_mvp import Pose
from multi_character_orchestrator import generate_multi_character_scene, parse_multi_script
from procedural_walk_cycle import generate_procedural_timeline, parse_action_script


CLIP_DIR_RE = re.compile(r"^clip_(\d{5})$")


@dataclass(slots=True)
class ClipSequence:
    clip_name: str
    char_id: str
    fps: int
    seconds: float
    features: list[list[float]]


def list_clip_dirs(dataset_dir: Path) -> list[Path]:
    out: list[Path] = []
    for item in sorted(dataset_dir.iterdir()):
        if item.is_dir() and CLIP_DIR_RE.match(item.name):
            out.append(item)
    return out


def pose_to_feature(pose: Pose, prev_pose: Pose | None) -> list[float]:
    """
    Convert a pose to a translation-robust feature vector.

    Design:
    - Use limb targets (already root-relative in this project)
    - Add root velocity from previous pose
    - Add look_angle + squash
    """
    if prev_pose is None:
        root_dx = 0.0
        root_dy = 0.0
    else:
        root_dx = pose.root.x - prev_pose.root.x
        root_dy = pose.root.y - prev_pose.root.y

    return [
        root_dx,
        root_dy,
        pose.l_hand.x,
        pose.l_hand.y,
        pose.r_hand.x,
        pose.r_hand.y,
        pose.l_foot.x,
        pose.l_foot.y,
        pose.r_foot.x,
        pose.r_foot.y,
        pose.look_angle,
        pose.squash,
    ]


def timeline_to_features(timeline) -> list[list[float]]:
    ordered = sorted(timeline.keyframes, key=lambda k: k.t)
    features: list[list[float]] = []
    prev_pose: Pose | None = None
    for key in ordered:
        feat = pose_to_feature(key.pose, prev_pose=prev_pose)
        features.append(feat)
        prev_pose = key.pose
    return features


def clip_features_from_scene(scene_text: str) -> tuple[int, float, dict[str, list[list[float]]]]:
    try:
        cfg, walk, wave, events, props, ai_motion = parse_action_script(scene_text)
        timeline = generate_procedural_timeline(
            cfg,
            walk,
            wave,
            ai_motion=ai_motion,
            events=events,
            props=props,
        )
        return cfg.fps, cfg.seconds, {"char1": timeline_to_features(timeline)}
    except ValueError as single_exc:
        try:
            cfg, programs, props, collision = parse_multi_script(scene_text)
            timelines, _, _, _ = generate_multi_character_scene(
                cfg=cfg,
                programs=programs,
                props=props,
                collision=collision,
            )
            features_by_char: dict[str, list[list[float]]] = {}
            for char_id in programs:
                timeline = timelines.get(char_id)
                if timeline is None:
                    continue
                features_by_char[char_id] = timeline_to_features(timeline)
            if not features_by_char:
                raise ValueError("multi scene generated no character timelines")
            return cfg.fps, cfg.seconds, features_by_char
        except ValueError as multi_exc:
            raise ValueError(
                f"scene parsing failed (single: {single_exc}; multi: {multi_exc})"
            ) from multi_exc


def load_clip_sequences(clip_dir: Path) -> list[ClipSequence]:
    scene_path = clip_dir / "scene.txt"
    if not scene_path.exists():
        raise FileNotFoundError(f"missing scene file: {scene_path}")
    scene_text = scene_path.read_text(encoding="utf-8")
    fps, seconds, features_by_char = clip_features_from_scene(scene_text)
    out: list[ClipSequence] = []
    for char_id, features in features_by_char.items():
        out.append(
            ClipSequence(
                clip_name=clip_dir.name,
                char_id=char_id,
                fps=fps,
                seconds=seconds,
                features=features,
            )
        )
    return out


def stream_all_features(
    dataset_dir: Path,
    *,
    max_clips: int = 0,
) -> tuple[list[ClipSequence], int]:
    clip_dirs = list_clip_dirs(dataset_dir)
    if max_clips > 0:
        clip_dirs = clip_dirs[:max_clips]

    sequences: list[ClipSequence] = []
    total_frames = 0
    skipped_invalid = 0
    skipped_logged = 0
    for clip_dir in clip_dirs:
        try:
            seqs = load_clip_sequences(clip_dir)
        except FileNotFoundError:
            continue
        except ValueError as exc:
            skipped_invalid += 1
            if skipped_logged < 5:
                print(f"[warn] skipping clip with unsupported/invalid scene DSL: {clip_dir.name} ({exc})")
                skipped_logged += 1
            continue
        for seq in seqs:
            sequences.append(seq)
            total_frames += len(seq.features)
    if skipped_invalid > skipped_logged:
        print(f"[warn] additionally skipped {skipped_invalid - skipped_logged} invalid clips")
    return sequences, total_frames


def reservoir_sample(
    sequences: list[ClipSequence],
    *,
    sample_size: int,
    rng: random.Random,
) -> tuple[list[list[float]], int]:
    sample: list[list[float]] = []
    seen = 0

    for seq in sequences:
        for feat in seq.features:
            seen += 1
            if len(sample) < sample_size:
                sample.append(feat)
            else:
                j = rng.randint(1, seen)
                if j <= sample_size:
                    sample[j - 1] = feat
    return sample, seen


def compute_mean_std(vectors: list[list[float]]) -> tuple[list[float], list[float]]:
    if not vectors:
        raise ValueError("no vectors for normalization")
    d = len(vectors[0])
    n = float(len(vectors))

    mean = [0.0] * d
    for vec in vectors:
        for i, v in enumerate(vec):
            mean[i] += v
    for i in range(d):
        mean[i] /= n

    var = [0.0] * d
    for vec in vectors:
        for i, v in enumerate(vec):
            dv = v - mean[i]
            var[i] += dv * dv
    std = [math.sqrt(v / n) for v in var]
    std = [s if s > 1e-8 else 1.0 for s in std]
    return mean, std


def normalize_vectors(
    vectors: list[list[float]],
    *,
    mean: list[float],
    std: list[float],
) -> list[list[float]]:
    out: list[list[float]] = []
    for vec in vectors:
        out.append([(v - m) / s for v, m, s in zip(vec, mean, std)])
    return out


def squared_dist(a: list[float], b: list[float]) -> float:
    s = 0.0
    for av, bv in zip(a, b):
        dv = av - bv
        s += dv * dv
    return s


def nearest_centroid(vec: list[float], centroids: list[list[float]]) -> tuple[int, float]:
    best_i = 0
    best_d = squared_dist(vec, centroids[0])
    for i in range(1, len(centroids)):
        d = squared_dist(vec, centroids[i])
        if d < best_d:
            best_i = i
            best_d = d
    return best_i, best_d


def fit_kmeans(
    vectors: list[list[float]],
    *,
    k: int,
    iterations: int,
    rng: random.Random,
) -> tuple[list[list[float]], list[float]]:
    if not vectors:
        raise ValueError("cannot train kmeans on empty vectors")
    if k <= 1:
        raise ValueError("k must be > 1")
    if k > len(vectors):
        raise ValueError(f"k={k} is larger than sample size={len(vectors)}")

    d = len(vectors[0])
    centroids = [vectors[i][:] for i in rng.sample(range(len(vectors)), k)]
    inertia_history: list[float] = []

    for _ in range(iterations):
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        inertia = 0.0

        for vec in vectors:
            idx, dist = nearest_centroid(vec, centroids)
            inertia += dist
            counts[idx] += 1
            row = sums[idx]
            for j, v in enumerate(vec):
                row[j] += v

        for ci in range(k):
            if counts[ci] == 0:
                centroids[ci] = vectors[rng.randrange(len(vectors))][:]
                continue
            inv = 1.0 / counts[ci]
            centroids[ci] = [v * inv for v in sums[ci]]

        inertia_history.append(inertia / len(vectors))

    return centroids, inertia_history


def tokenize_vectors(vectors: list[list[float]], centroids: list[list[float]]) -> tuple[list[int], float]:
    tokens: list[int] = []
    inertia = 0.0
    for vec in vectors:
        idx, dist = nearest_centroid(vec, centroids)
        tokens.append(idx)
        inertia += dist
    avg = inertia / len(vectors) if vectors else 0.0
    return tokens, avg


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a pose tokenizer and export token sequences per clip.")
    ap.add_argument("--dataset", type=str, required=True, help="Dataset directory with clip_* folders.")
    ap.add_argument("--out", type=str, default="tokenizer", help="Output directory for tokenizer artifacts.")
    ap.add_argument("--codebook-size", type=int, default=256, help="Number of discrete motion tokens.")
    ap.add_argument("--iterations", type=int, default=20, help="K-means iterations.")
    ap.add_argument("--sample-size", type=int, default=50000, help="Max vectors sampled for fitting.")
    ap.add_argument("--max-clips", type=int, default=0, help="Limit number of clips loaded (0 = all).")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument(
        "--write-corpus",
        action="store_true",
        help="Also write token_corpus.txt with one tokenized sequence per line.",
    )
    args = ap.parse_args()

    if args.codebook_size <= 1:
        raise ValueError("--codebook-size must be > 1")
    if args.iterations <= 0:
        raise ValueError("--iterations must be > 0")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be > 0")
    if args.max_clips < 0:
        raise ValueError("--max-clips must be >= 0")

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset dir not found: {dataset_dir}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    sequences, total_frames = stream_all_features(dataset_dir, max_clips=args.max_clips)
    if not sequences:
        raise RuntimeError("no valid clip sequences found (expected clip_*/scene.txt)")

    sample, seen = reservoir_sample(sequences, sample_size=args.sample_size, rng=rng)
    if len(sample) < args.codebook_size:
        raise RuntimeError(
            f"sample too small for codebook: sample={len(sample)} codebook={args.codebook_size}. "
            "Increase data or lower --codebook-size."
        )

    mean, std = compute_mean_std(sample)
    norm_sample = normalize_vectors(sample, mean=mean, std=std)
    centroids, inertia_hist = fit_kmeans(
        norm_sample,
        k=args.codebook_size,
        iterations=args.iterations,
        rng=rng,
    )

    model = {
        "feature_dim": len(mean),
        "codebook_size": args.codebook_size,
        "iterations": args.iterations,
        "seed": args.seed,
        "normalization": {"mean": mean, "std": std},
        "codebook": centroids,
        "fit": {
            "vectors_seen_total": seen,
            "vectors_sampled": len(sample),
            "inertia_history": inertia_hist,
            "final_avg_inertia": inertia_hist[-1] if inertia_hist else None,
        },
    }
    write_json(out_dir / "pose_tokenizer_model.json", model)

    tokens_manifest_path = out_dir / "tokens_manifest.jsonl"
    corpus_path = out_dir / "token_corpus.txt"

    total_tokenized = 0
    unique_clips = {seq.clip_name for seq in sequences}
    with tokens_manifest_path.open("w", encoding="utf-8") as mf:
        corpus_handle = corpus_path.open("w", encoding="utf-8") if args.write_corpus else None
        try:
            for seq in sequences:
                norm_feats = normalize_vectors(seq.features, mean=mean, std=std)
                tokens, avg_inertia = tokenize_vectors(norm_feats, centroids=centroids)

                sequence_id = f"{seq.clip_name}:{seq.char_id}"
                clip_token_obj = {
                    "clip": seq.clip_name,
                    "char_id": seq.char_id,
                    "sequence_id": sequence_id,
                    "fps": seq.fps,
                    "seconds": seq.seconds,
                    "num_tokens": len(tokens),
                    "avg_inertia": avg_inertia,
                    "tokens": tokens,
                }
                token_file = out_dir / "clips" / f"{seq.clip_name}_{seq.char_id}_tokens.json"
                write_json(token_file, clip_token_obj)

                manifest_row = {
                    "clip": seq.clip_name,
                    "char_id": seq.char_id,
                    "sequence_id": sequence_id,
                    "token_file": str(token_file.as_posix()),
                    "fps": seq.fps,
                    "seconds": seq.seconds,
                    "num_tokens": len(tokens),
                    "avg_inertia": avg_inertia,
                }
                mf.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")

                if corpus_handle is not None:
                    # Flat integer corpus line for LM-style sequence modeling.
                    corpus_handle.write(" ".join(str(t) for t in tokens) + "\n")

                total_tokenized += len(tokens)
        finally:
            if corpus_handle is not None:
                corpus_handle.close()

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "clips": len(unique_clips),
        "sequences": len(sequences),
        "frames_total": total_frames,
        "tokens_total": total_tokenized,
        "codebook_size": args.codebook_size,
        "sample_size": len(sample),
        "vectors_seen_total": seen,
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[ok] pose tokenizer trained: {out_dir.resolve()}")
    print(
        f"[ok] clips={len(unique_clips)} sequences={len(sequences)} "
        f"frames={total_frames} tokens={total_tokenized} "
        f"codebook={args.codebook_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
