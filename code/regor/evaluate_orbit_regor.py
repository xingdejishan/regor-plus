from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "source" / "regor"
sys.path.insert(0, str(BASELINE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from stream_acsr_inputs import token_for
from utils.SE3 import integrate_trans


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_metrics(pose: np.ndarray, expected: np.ndarray) -> tuple[float, float, bool]:
    relative = pose[:3, :3].T @ expected[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    rotation = float(np.degrees(np.arccos(cosine)))
    translation = float(np.linalg.norm(pose[:3, 3] - expected[:3, 3]) * 100.0)
    return rotation, translation, rotation < 15.0 and translation < 30.0


def exact_mcnemar(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    rescue = int((~first & second).sum())
    damage = int((first & ~second).sum())
    total = rescue + damage
    if total == 0:
        probability = 1.0
    else:
        tail = sum(math.comb(total, index) for index in range(min(rescue, damage) + 1)) / (2**total)
        probability = min(1.0, 2.0 * tail)
    return {"rescue": rescue, "damage": damage, "net": rescue - damage, "exact_two_sided_p": probability}


def bootstrap_difference(first: np.ndarray, second: np.ndarray, repetitions: int = 10000) -> list[float]:
    rng = np.random.default_rng(20260824)
    difference = second.astype(float) - first.astype(float)
    sampled = rng.choice(difference, size=(repetitions, len(difference)), replace=True).mean(1)
    return [float(np.percentile(sampled, 2.5)), float(np.percentile(sampled, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=BASELINE_ROOT / "3DLoMatch.pkl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    online_dir = args.online_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    complete = json.loads((online_dir / "online_complete.json").read_text(encoding="utf-8"))
    variants = tuple(complete["variants"])
    arms = ("baseline",) + variants
    with (online_dir / "online_rows.csv").open(encoding="utf-8", newline="") as handle:
        online_rows = {row["token"]: row for row in csv.DictReader(handle)}
    if complete["pair_count"] != len(online_rows) or complete["trace_count"] != len(online_rows):
        raise RuntimeError("online output is incomplete")
    if sha256_file(online_dir / "online_rows.csv") != complete["online_rows_sha256"]:
        raise RuntimeError("online ledger hash mismatch")
    with args.metadata.resolve().open("rb") as handle:
        metadata = pickle.load(handle)
    paths = sorted(args.snapshot_dir.resolve().glob("*.pth"), key=lambda path: int(path.stem))
    stop = len(paths) if args.limit is None else min(len(paths), args.start + args.limit)
    paths = paths[args.start:stop]
    success = {arm: [] for arm in arms}
    rotations = {arm: [] for arm in arms}
    translations = {arm: [] for arm in arms}
    by_scene = defaultdict(lambda: {arm: [] for arm in arms})
    telemetry = {variant: [] for variant in variants}
    records = []
    for path in paths:
        index = int(path.stem)
        packet = torch.load(path, map_location="cpu")
        source_count = int(packet["len_src"])
        points = packet["pcd"].detach().cpu().numpy().astype(np.float32, copy=False)
        features = packet["feats"].detach().cpu().numpy().astype(np.float32, copy=False)
        token = token_for(points[:source_count], points[source_count:], features[:source_count], features[source_count:])
        if token not in online_rows:
            raise RuntimeError(f"missing online token for evaluator item {index}")
        row = online_rows[token]
        trace_path = online_dir / "online_traces" / f"{token}.npz"
        if sha256_file(trace_path) != row["trace_sha256"]:
            raise RuntimeError(f"trace hash mismatch for evaluator item {index}")
        expected = integrate_trans(packet["rot"], packet["trans"])
        if torch.is_tensor(expected):
            expected = expected.detach().cpu().numpy()
        scene = metadata["src"][index].split("/")[1]
        record = {"evaluator_index": index, "scene": scene, "token": token}
        with np.load(trace_path, allow_pickle=False) as trace:
            for arm in arms:
                rotation, translation, current_success = pose_metrics(trace[f"{arm}_pose"], expected)
                rotations[arm].append(rotation)
                translations[arm].append(translation)
                success[arm].append(current_success)
                by_scene[scene][arm].append(current_success)
                record[f"{arm}_rre_deg"] = rotation
                record[f"{arm}_rte_cm"] = translation
                record[f"{arm}_success"] = int(current_success)
            for variant in variants:
                current = json.loads(str(trace[f"{variant}_telemetry_json"].item()))
                telemetry[variant].append(current)
        records.append(record)
    if len(records) != len(online_rows):
        raise RuntimeError("evaluator scope does not match sealed online scope")
    arrays = {arm: np.asarray(success[arm], dtype=bool) for arm in arms}
    baseline = arrays["baseline"]
    result = {
        "scope": {"start": args.start, "limit": args.limit, "count": len(records)},
        "integrity": {
            "online_complete_sha256": sha256_file(online_dir / "online_complete.json"),
            "online_rows_sha256": sha256_file(online_dir / "online_rows.csv"),
            "all_trace_hashes_match": True,
            "evaluator_process_separate": True,
        },
        "official": {
            arm: {
                "success": int(arrays[arm].sum()),
                "total": len(arrays[arm]),
                "rr": float(arrays[arm].mean()),
                "rre_mean_deg": float(np.mean(rotations[arm])),
                "rte_mean_cm": float(np.mean(translations[arm])),
            }
            for arm in arms
        },
        "paired_vs_baseline": {
            variant: {
                **exact_mcnemar(baseline, arrays[variant]),
                "paired_bootstrap_95_ci_rr_difference": bootstrap_difference(baseline, arrays[variant]),
            }
            for variant in variants
        },
        "by_scene": {
            scene: {
                arm: {"success": int(sum(values[arm])), "total": len(values[arm]), "rr": float(np.mean(values[arm]))}
                for arm in arms
            }
            for scene, values in sorted(by_scene.items())
        },
        "telemetry": {
            variant: {
                key: float(np.mean([float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]))
                for key in rows[0]
                if rows and all(isinstance(row.get(key), (int, float)) for row in rows)
            }
            for variant, rows in telemetry.items()
        },
    }
    fields = list(records[0]) if records else []
    with (output_dir / "evaluated_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
