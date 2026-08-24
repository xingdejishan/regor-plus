from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT
sys.path.insert(0, str(BASELINE_ROOT))
os.chdir(BASELINE_ROOT)

from benchmark_utils_predator import computeTransformationErr, read_trajectory, read_trajectory_info
from utils.SE3 import integrate_trans


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_pair_indices(path: Path, pair_count: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = [int(value) for value in payload["indices"]]
    if indices != sorted(set(indices)) or not indices:
        raise RuntimeError("analysis pair-index split is invalid")
    if indices[0] < 0 or indices[-1] >= pair_count:
        raise RuntimeError("analysis pair-index split is outside the corpus")
    return indices


def pose_metrics(poses: np.ndarray, references: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(poses).all(axis=(1, 2))
    relative = np.transpose(poses[:, :3, :3], (0, 2, 1)) @ references[:, :3, :3]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1, 1)
    rre = np.degrees(np.arccos(cosine))
    rte = np.linalg.norm(poses[:, :3, 3] - references[:, :3, 3], axis=1) * 100
    passed = finite & (rre < 15) & (rte < 30)
    return rre, rte, passed


def bootstrap_difference(first: np.ndarray, second: np.ndarray, samples: int = 20000) -> tuple[float, float]:
    rng = np.random.default_rng(20260824)
    differences = np.empty(samples, dtype=np.float64)
    count = len(first)
    for start in range(0, samples, 1000):
        stop = min(samples, start + 1000)
        indices = rng.integers(0, count, size=(stop - start, count))
        differences[start:stop] = (second[indices] - first[indices]).mean(1)
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def official_arm(poses: np.ndarray) -> dict[str, object]:
    scenes = sorted(path for path in (BASELINE_ROOT / "benchmarks" / "3DLoMatch").iterdir() if path.is_dir())
    start = 0
    per_scene = {}
    valid_counts = []
    recalls = []
    for scene in scenes:
        pairs, reference_poses = read_trajectory(str(scene / "gt.log"))
        fragment_count, information = read_trajectory_info(str(scene / "gt.info"))
        arm = poses[start : start + len(pairs)]
        start += len(pairs)
        mask = np.full((fragment_count, fragment_count), -1, dtype=np.int64)
        for reference_index, pair in enumerate(pairs):
            if int(pair[1]) - int(pair[0]) > 1:
                mask[int(pair[0]), int(pair[1])] = reference_index
        valid = int(np.sum(mask >= 0))
        good = 0
        invalid = 0
        for index, pair in enumerate(pairs):
            reference_index = int(mask[int(pair[0]), int(pair[1])])
            if reference_index < 0:
                continue
            if not np.isfinite(arm[index]).all():
                invalid += 1
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                error = computeTransformationErr(
                    np.linalg.inv(reference_poses[reference_index]) @ arm[index],
                    information[reference_index],
                )
            if not np.isfinite(error):
                invalid += 1
            elif error <= 0.2**2:
                good += 1
        recall = good / valid if valid else 0.0
        per_scene[scene.name] = {
            "pairs": len(pairs),
            "valid_nonadjacent_pairs": valid,
            "successful_registrations": good,
            "invalid_covariance_errors": invalid,
            "recall": recall,
        }
        valid_counts.append(valid)
        recalls.append(recall)
    if start != len(poses):
        raise RuntimeError("official benchmark pose count mismatch")
    valid_array = np.asarray(valid_counts, dtype=np.float64)
    recall_array = np.asarray(recalls, dtype=np.float64)
    return {
        "scenes": per_scene,
        "mean_scene_recall": float(recall_array.mean()),
        "weighted_recall": float(np.sum(valid_array * recall_array) / valid_array.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--online-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--index-file", type=Path)
    args = parser.parse_args()
    args.online_dir = resolve_project_path(args.online_dir)
    args.output_dir = resolve_project_path(args.output_dir)
    if args.index_file is not None:
        args.index_file = resolve_project_path(args.index_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.online_dir / "final_poses.npz", allow_pickle=False) as archive:
        baseline = np.asarray(archive["baseline"], dtype=np.float64)
        probe = np.asarray(archive["probe"], dtype=np.float64)
    with (BASELINE_ROOT / "3DLoMatch.pkl").open("rb") as handle:
        infos = pickle.load(handle)
    references = np.asarray(
        [integrate_trans(rotation, translation) for rotation, translation in zip(infos["rot"], infos["trans"])],
        dtype=np.float64,
    )
    scenes = np.asarray([str(path).split("/")[1] for path in infos["src"]])
    full_pair_count = len(references)
    if args.index_file is not None:
        indices = load_pair_indices(args.index_file, full_pair_count)
        references = references[indices]
        scenes = scenes[indices]
    else:
        indices = list(range(full_pair_count))
    if len(baseline) != len(references) or len(probe) != len(references):
        raise RuntimeError(
            f"full-set analysis requires {len(references)} poses per arm, got "
            f"{len(baseline)} and {len(probe)}"
        )
    baseline_rre, baseline_rte, baseline_passed = pose_metrics(baseline, references)
    probe_rre, probe_rte, probe_passed = pose_metrics(probe, references)
    rescue = int(np.sum(~baseline_passed & probe_passed))
    damage = int(np.sum(baseline_passed & ~probe_passed))
    discordant = rescue + damage
    p_value = float(binomtest(min(rescue, damage), discordant, 0.5).pvalue) if discordant else 1.0
    one_sided_p = float(binomtest(rescue, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0
    interval = bootstrap_difference(baseline_passed.astype(float), probe_passed.astype(float))
    per_scene = {}
    for scene in dict.fromkeys(scenes.tolist()):
        mask = scenes == scene
        per_scene[scene] = {
            "pairs": int(mask.sum()),
            "baseline_count": int(baseline_passed[mask].sum()),
            "probe_count": int(probe_passed[mask].sum()),
            "baseline_rate": float(baseline_passed[mask].mean()),
            "probe_rate": float(probe_passed[mask].mean()),
            "difference": float(probe_passed[mask].mean() - baseline_passed[mask].mean()),
        }
    row_paths = sorted((args.online_dir / "pair_rows").glob("*.json"))
    expected_names = [f"{index:04d}.json" for index in indices]
    if [path.name for path in row_paths] != expected_names:
        raise RuntimeError("online row files do not exactly match the frozen split")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in row_paths]
    if len(rows) != len(references):
        raise RuntimeError(f"expected {len(references)} online rows, got {len(rows)}")
    if [int(row["pair_index"]) for row in rows] != indices:
        raise RuntimeError("online row pair indices do not exactly match the frozen split")
    full_set = indices == list(range(full_pair_count))
    kill_gate = {
        "applicable": not full_set,
        "rule": "net >= 2 and one-sided McNemar p < 0.25 with zero online errors and all finite poses",
        "advance_to_full": bool(
            not full_set
            and rescue - damage >= 2
            and one_sided_p < 0.25
            and not any(bool(row["error"]) for row in rows)
            and np.isfinite(baseline).all()
            and np.isfinite(probe).all()
        ),
    }
    result = {
        "protocol_review": {
            "track": "strict pairwise",
            "pair_count": len(baseline),
            "online_error_count": sum(bool(row["error"]) for row in rows),
            "finite_baseline_count": int(np.isfinite(baseline).all(axis=(1, 2)).sum()),
            "finite_probe_count": int(np.isfinite(probe).all(axis=(1, 2)).sum()),
            "online_and_analysis_processes_separate": True,
        },
        "threshold_metrics": {
            "rule": "RRE < 15 degrees and RTE < 30 cm",
            "baseline_count": int(baseline_passed.sum()),
            "probe_count": int(probe_passed.sum()),
            "baseline_rate": float(baseline_passed.mean()),
            "probe_rate": float(probe_passed.mean()),
            "rescue": rescue,
            "damage": damage,
            "net": rescue - damage,
            "exact_mcnemar_p": p_value,
            "one_sided_mcnemar_p": one_sided_p,
            "paired_bootstrap_difference_ci95": interval,
            "baseline_median_rre": float(np.nanmedian(baseline_rre)),
            "probe_median_rre": float(np.nanmedian(probe_rre)),
            "baseline_median_rte_cm": float(np.nanmedian(baseline_rte)),
            "probe_median_rte_cm": float(np.nanmedian(probe_rte)),
        },
        "per_scene": per_scene,
        "official": (
            {
                "baseline": official_arm(baseline),
                "probe": official_arm(probe),
                "protocol": "corrected Redwood/3DLoMatch covariance protocol",
            }
            if full_set
            else {"available": False, "reason": "partial label-blind development kill-test"}
        ),
        "kill_gate": kill_gate,
        "telemetry": {
            "mean_rounds": float(np.mean([row.get("probe_rounds", 0) for row in rows])),
            "mean_baseline_runtime_sec": float(np.mean([row.get("baseline_runtime_sec", 0) for row in rows])),
            "mean_probe_runtime_sec": float(np.mean([row.get("probe_runtime_sec", 0) for row in rows])),
        },
    }
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
