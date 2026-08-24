from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "code" / "regor"
DIAGNOSTIC_ROOT = CODE_ROOT / "diagnostics"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(DIAGNOSTIC_ROOT))

from cut_regor.bridge import bridge_domain, choose_mode_anchors, descriptor_correspondences
from cut_regor.config import CutRegorConfig
from cut_regor.frontier import partial_symmetry_frontier, select_queries
from cut_regor.geometry import build_patch_graph, estimate_resolution, pose_metrics
from einfo_online_artifact import load_online_artifact, reconstruct_sampled_arrays


STRATEGIES = ("random", "high_curvature", "high_disagreement", "symmetry_frontier")


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_gt(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as trace:
        return np.asarray(trace["gt_trans"], dtype=np.float64)


def select_modes(
    representative_poses: np.ndarray,
    member_poses: np.ndarray,
    member_offsets: np.ndarray,
    borda_rank_sums: np.ndarray,
    gt: np.ndarray,
    config: CutRegorConfig,
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    rre, rte = pose_metrics(member_poses, gt)
    success = (rre < config.success_rre_deg) & (rte < config.success_rte_cm)
    positive = np.flatnonzero(success)
    representative_rre, representative_rte = pose_metrics(representative_poses, gt)
    representative_success = (
        (representative_rre < config.success_rre_deg) & (representative_rte < config.success_rte_cm)
    )
    if len(positive) == 0:
        return None
    distance = rre / config.success_rre_deg + rte / config.success_rte_cm
    positive_index = int(positive[np.argmin(distance[positive])])
    positive_mode = int(np.searchsorted(member_offsets[1:], positive_index, side="right"))
    negative = np.flatnonzero(~representative_success)
    negative = negative[negative != positive_mode]
    if len(negative) == 0:
        return None
    negative_order = np.lexsort((negative, borda_rank_sums[negative]))
    negative_mode = int(negative[negative_order[0]])
    return positive_index, positive_mode, negative_mode, rre, rte, representative_rre, representative_rte


def evaluate_domains(
    query_indices: torch.Tensor,
    source_centers: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    positive_pose: torch.Tensor,
    negative_pose: torch.Tensor,
    positive_anchors: tuple[torch.Tensor, torch.Tensor] | None,
    negative_anchors: tuple[torch.Tensor, torch.Tensor] | None,
    delta: float,
    config: CutRegorConfig,
    anchor_count: int,
) -> tuple[int, int, int]:
    positive_nonempty = 0
    negative_nonempty = 0
    separated = 0
    if positive_anchors is None or negative_anchors is None:
        return positive_nonempty, negative_nonempty, separated
    positive_source, positive_target = positive_anchors
    negative_source, negative_target = negative_anchors
    for query_index in query_indices.tolist():
        query = source_centers[query_index]
        positive = bridge_domain(
            query,
            source[positive_source],
            target[positive_target],
            target,
            positive_pose,
            delta,
            config,
            anchor_count=anchor_count,
        )
        negative = bridge_domain(
            query,
            source[negative_source],
            target[negative_target],
            target,
            negative_pose,
            delta,
            config,
            anchor_count=anchor_count,
        )
        has_positive = len(positive.candidate_indices) > 0
        has_negative = len(negative.candidate_indices) > 0
        positive_nonempty += int(has_positive)
        negative_nonempty += int(has_negative)
        separated += int(has_positive and not has_negative)
    return positive_nonempty, negative_nonempty, separated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-dir", type=Path, required=True)
    parser.add_argument(
        "--mode-pool-dir",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-24_CutRegorG17ModePool",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-15_Hard247OnlineArtifacts",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-08_失败分流诊断",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    online_dir = project_path(args.online_dir)
    mode_pool_dir = project_path(args.mode_pool_dir)
    manifest = json.loads((mode_pool_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["pair_count"] != 247 or manifest["gt_fields_present_in_output"] is not False:
        raise RuntimeError("complete sanitized G17 mode pool is required")
    config = CutRegorConfig()
    online_dir.mkdir(parents=True, exist_ok=True)
    trace_paths = sorted((mode_pool_dir / "online_traces").glob("*.npz"), key=lambda path: int(path.stem))
    if args.limit is not None:
        trace_paths = trace_paths[: args.limit]
    rows = []
    errors_path = online_dir / "analysis_errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as error_handle:
        for position, trace_path in enumerate(trace_paths, 1):
            pair_index = int(trace_path.stem)
            row: dict[str, object] = {"pair_index": pair_index, "evaluable": 0, "error": ""}
            started = time.perf_counter()
            try:
                with np.load(trace_path, allow_pickle=False) as trace:
                    representative_poses = np.asarray(trace["mode_representative_poses"], dtype=np.float64)
                    member_poses = np.asarray(trace["mode_member_poses"], dtype=np.float64)
                    member_offsets = np.asarray(trace["mode_member_offsets"], dtype=np.int64)
                    borda_rank_sums = np.asarray(trace["mode_borda_rank_sums"], dtype=np.float64)
                gt = load_gt(project_path(args.baseline_dir) / "traces" / f"{pair_index:04d}.npz")
                selected = select_modes(
                    representative_poses,
                    member_poses,
                    member_offsets,
                    borda_rank_sums,
                    gt,
                    config,
                )
                if selected is None:
                    row["reason"] = "missing_positive_or_negative_mode"
                    rows.append(row)
                    continue
                positive_index, positive_mode, negative_mode, rre, rte, representative_rre, representative_rte = selected
                artifact = load_online_artifact(
                    project_path(args.artifact_root) / f"{pair_index:04d}.npz",
                    expected_pair_index=pair_index,
                    expected_pair_seed=config.seed + pair_index,
                    num_node=5000,
                )
                arrays = reconstruct_sampled_arrays(artifact)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                source = torch.from_numpy(arrays["source_points"]).to(device)
                target = torch.from_numpy(arrays["target_points"]).to(device)
                source_features = torch.from_numpy(arrays["source_features"]).to(device)
                target_features = torch.from_numpy(arrays["target_features"]).to(device)
                positive_pose = torch.from_numpy(member_poses[positive_index]).to(device=device, dtype=source.dtype)
                negative_pose = torch.from_numpy(representative_poses[negative_mode]).to(device=device, dtype=source.dtype)
                with torch.no_grad():
                    delta = estimate_resolution(source, target)
                    source_graph = build_patch_graph(source, delta, config)
                    target_graph = build_patch_graph(target, delta, config)
                    frontier = partial_symmetry_frontier(target_graph, positive_pose, negative_pose, delta, config)
                    matches = descriptor_correspondences(source_features, target_features)
                    positive_anchors = choose_mode_anchors(source, target, matches, positive_pose, config)
                    negative_anchors = choose_mode_anchors(source, target, matches, negative_pose, config)
                    for strategy in STRATEGIES:
                        queries = select_queries(
                            source_graph,
                            target_graph,
                            positive_pose,
                            negative_pose,
                            frontier,
                            strategy,
                            config.query_count,
                            config.seed + pair_index,
                        )
                        values = evaluate_domains(
                            queries,
                            source_graph.centers,
                            source,
                            target,
                            positive_pose,
                            negative_pose,
                            positive_anchors,
                            negative_anchors,
                            delta,
                            config,
                            config.anchor_count,
                        )
                        row[f"{strategy}_query_count"] = len(queries)
                        row[f"{strategy}_positive"] = values[0]
                        row[f"{strategy}_negative"] = values[1]
                        row[f"{strategy}_separated"] = values[2]
                    single_queries = select_queries(
                        source_graph,
                        target_graph,
                        positive_pose,
                        negative_pose,
                        frontier,
                        "symmetry_frontier",
                        config.query_count,
                        config.seed + pair_index,
                        single_scale=True,
                    )
                    single_values = evaluate_domains(
                        single_queries,
                        source_graph.centers,
                        source,
                        target,
                        positive_pose,
                        negative_pose,
                        positive_anchors,
                        negative_anchors,
                        delta,
                        config,
                        config.anchor_count,
                    )
                    point_queries = select_queries(
                        source_graph,
                        target_graph,
                        positive_pose,
                        negative_pose,
                        frontier,
                        "symmetry_frontier",
                        config.query_count,
                        config.seed + pair_index,
                    )
                    point_values = evaluate_domains(
                        point_queries,
                        source_graph.centers,
                        source,
                        target,
                        positive_pose,
                        negative_pose,
                        positive_anchors,
                        negative_anchors,
                        delta,
                        config,
                        0,
                    )
                row.update(
                    {
                        "evaluable": 1,
                        "positive_member_index": positive_index,
                        "positive_mode_index": positive_mode,
                        "negative_mode_index": negative_mode,
                        "positive_rre": float(rre[positive_index]),
                        "positive_rte": float(rte[positive_index]),
                        "negative_rre": float(representative_rre[negative_mode]),
                        "negative_rte": float(representative_rte[negative_mode]),
                        "negative_borda_rank_sum": float(borda_rank_sums[negative_mode]),
                        "delta": delta,
                        "symmetry_support": int(frontier.multiscale_support.sum().item()),
                        "multiscale_frontier_count": int(frontier.multiscale_frontier.sum().item()),
                        "single_frontier_count": int(frontier.single_scale_frontier.sum().item()),
                        "single_frontier_query_count": len(single_queries),
                        "single_frontier_positive": single_values[0],
                        "single_frontier_negative": single_values[1],
                        "single_frontier_separated": single_values[2],
                        "point_query_count": len(point_queries),
                        "point_positive": point_values[0],
                        "point_negative": point_values[1],
                        "point_separated": point_values[2],
                        "positive_anchor_valid": int(positive_anchors is not None),
                        "negative_anchor_valid": int(negative_anchors is not None),
                    }
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                error_handle.write(
                    json.dumps(
                        {"pair_index": pair_index, "error": row["error"], "traceback": traceback.format_exc()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_handle.flush()
            row["runtime_sec"] = time.perf_counter() - started
            rows.append(row)
            print(f"[{position}/{len(trace_paths)}] pair={pair_index} evaluable={row['evaluable']} {row['error'] or 'ok'}", flush=True)
    fields = sorted({key for row in rows for key in row})
    with (online_dir / "frontier_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row.get("evaluable") == 1 and not row.get("error")]

    def aggregate(prefix: str) -> dict[str, float | int]:
        queries = sum(int(row.get(f"{prefix}_query_count", 0)) for row in valid)
        positive = sum(int(row.get(f"{prefix}_positive", 0)) for row in valid)
        negative = sum(int(row.get(f"{prefix}_negative", 0)) for row in valid)
        separated = sum(int(row.get(f"{prefix}_separated", 0)) for row in valid)
        return {
            "query_count": queries,
            "positive_nonempty": positive,
            "negative_nonempty": negative,
            "separated": separated,
            "positive_survival_rate": positive / queries if queries else 0.0,
            "negative_survival_rate": negative / queries if queries else 0.0,
            "separation_rate": separated / queries if queries else 0.0,
        }

    metrics = {name: aggregate(name) for name in STRATEGIES}
    metrics["single_frontier"] = aggregate("single_frontier")
    metrics["point"] = aggregate("point")
    frontier_gain = metrics["symmetry_frontier"]["separation_rate"] - metrics["random"]["separation_rate"]
    summary = {
        "scope": "offline Frontier Separability diagnostic; GT selects positive/hard-negative modes only",
        "candidate_input": "sanitized GT-free G17 retained modes and members",
        "input_pair_count": len(rows),
        "evaluable_pair_count": len(valid),
        "unevaluable_pair_count": sum(row.get("evaluable") != 1 and not row.get("error") for row in rows),
        "error_count": sum(bool(row.get("error")) for row in rows),
        "config": config.to_dict(),
        "metrics": metrics,
        "frontier_minus_random_separation": frontier_gain,
        "criteria": {
            "frontier_gain_at_least_0_15": frontier_gain >= 0.15,
            "positive_survival_above_0_90": metrics["symmetry_frontier"]["positive_survival_rate"] > 0.90,
            "multiscale_better_than_single": metrics["symmetry_frontier"]["separation_rate"] > metrics["single_frontier"]["separation_rate"],
            "four_anchor_better_than_point": metrics["symmetry_frontier"]["separation_rate"] > metrics["point"]["separation_rate"],
        },
    }
    summary["continue_to_branch_and_contract"] = all(summary["criteria"].values()) and summary["error_count"] == 0
    (online_dir / "frontier_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
