import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

from redkitchen_fsv_common import (
    REPO_ROOT,
    SCENE,
    cell_to_matrix,
    linear_voxel_keys,
    load_fragment_points,
    nearest_distance_stats,
    read_csv,
    read_json,
    roc_auc_score_binary,
    sample_indices,
    specificity_gate_threshold,
    transform_points,
    write_csv,
    write_json,
)


def dir_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def size_gib(value):
    return round(float(value) / (1024.0 ** 3), 3)


def load_fsv(path):
    data = np.load(path, allow_pickle=True)
    free = data["free_voxel_indices"].astype(np.int64)
    grid_shape = data["grid_shape"].astype(np.int64)
    return {
        "free": free,
        "free_keys": set(linear_voxel_keys(free, grid_shape).tolist()),
        "origin": data["origin"].astype(np.float64),
        "voxel_size": float(data["voxel_size"]),
        "grid_shape": grid_shape,
        "num_depth_frames": int(data["num_depth_frames"]),
        "coordinate_system_note": str(data["coordinate_system_note"]),
    }


def query_fsv_score(src_points, trans, fsv, max_points, seed):
    idx = sample_indices(src_points.shape[0], max_points, seed)
    points = transform_points(src_points[idx], trans)
    vox = np.floor((points - fsv["origin"][None, :]) / fsv["voxel_size"]).astype(np.int64)
    shape = fsv["grid_shape"]
    valid = np.all((vox >= 0) & (vox < shape[None, :]), axis=1)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return float("nan"), 0, 0, int(points.shape[0])
    keys = linear_voxel_keys(vox[valid], shape)
    free_count = int(sum(1 for key in keys.tolist() if key in fsv["free_keys"]))
    return float(free_count / valid_count), free_count, valid_count, int(points.shape[0] - valid_count)


def proxy_nn_score(src_points, tgt_points, trans, max_points, seed):
    idx = sample_indices(src_points.shape[0], max_points, seed)
    warped = transform_points(src_points[idx], trans)
    stats = nearest_distance_stats(warped, tgt_points)
    return float(stats["mean"])


def accepted_metrics(rows, accepted):
    accepted = np.asarray(accepted, dtype=bool)
    total = len(rows)
    successes = np.asarray([int(r["success"]) for r in rows], dtype=np.int32)
    failures = successes == 0
    re_values = np.asarray([float(r["RE"]) for r in rows], dtype=np.float64)
    te_values = np.asarray([float(r["TE"]) for r in rows], dtype=np.float64)
    fsv_values = np.asarray([float(r["fsv_score"]) if r["fsv_score"] != "" else np.nan for r in rows], dtype=np.float64)
    acc_count = int(accepted.sum())
    rej = ~accepted
    failed_total = int(failures.sum())
    intercepted = int((rej & failures).sum())
    return {
        "accepted_pairs": acc_count,
        "coverage": float(acc_count / total) if total else 0.0,
        "accepted_RR": float(successes[accepted].mean() * 100.0) if acc_count else 0.0,
        "accepted_precision": float(successes[accepted].mean() * 100.0) if acc_count else 0.0,
        "rejected_pairs": int(rej.sum()),
        "failed_interception_rate": float(intercepted / failed_total) if failed_total else 0.0,
        "Mean Re on accepted": float(np.nanmean(re_values[accepted])) if acc_count else float("nan"),
        "Mean Te on accepted": float(np.nanmean(te_values[accepted])) if acc_count else float("nan"),
        "mean FSV accepted": float(np.nanmean(fsv_values[accepted])) if acc_count else float("nan"),
        "mean FSV rejected": float(np.nanmean(fsv_values[rej])) if np.any(rej) else float("nan"),
    }


def json_sanitize(value):
    if isinstance(value, dict):
        return {key: json_sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_sanitize(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def compute_auc(rows):
    valid_rows = [r for r in rows if str(r["fsv_valid"]) == "1"]
    labels_success = np.asarray([int(r["success"]) for r in valid_rows], dtype=np.int32)
    fsv = np.asarray([float(r["fsv_score"]) for r in valid_rows], dtype=np.float64)
    labels_failure = 1 - labels_success
    auc_failure_high = roc_auc_score_binary(labels_failure, fsv)
    auc_success_high = roc_auc_score_binary(labels_success, -fsv)
    pairwise = auc_failure_high
    success_scores = fsv[labels_success == 1]
    failure_scores = fsv[labels_success == 0]
    result = {
        "valid_pairs": int(len(valid_rows)),
        "invalid_pairs": int(len(rows) - len(valid_rows)),
        "AUC(success/failure)": auc_success_high,
        "AUC_failure_high": auc_failure_high,
        "pairwise_discrimination_accuracy": pairwise,
        "FSV mean for success": float(np.mean(success_scores)) if success_scores.size else float("nan"),
        "FSV mean for failure": float(np.mean(failure_scores)) if failure_scores.size else float("nan"),
    }
    for specificity in (0.90, 0.95):
        threshold = specificity_gate_threshold(fsv, labels_success, specificity)
        accepted = fsv <= threshold
        failures = labels_success == 0
        successes = labels_success == 1
        result[f"{int(specificity * 100)}% specificity threshold"] = float(threshold)
        result[f"{int(specificity * 100)}% specificity actual"] = float((~accepted[failures]).mean()) if np.any(failures) else float("nan")
        result[f"{int(specificity * 100)}% specificity sensitivity"] = float(accepted[successes].mean()) if np.any(successes) else float("nan")
    return result


def summarize_markdown(args, baseline_metrics, raw_check, precompute_report, fsv_auc, gate_results, proxy_auc, deviations):
    download_log = read_json(args.download_log) if Path(args.download_log).exists() else {}
    raw_root = Path(raw_check.get("raw_scene_root", args.raw_root))
    zip_path = Path(download_log.get("zip_path", Path(args.raw_root) / f"{SCENE}.zip"))
    lines = []
    lines.append("# Redkitchen RGB-D FSV Summary")
    lines.append("")
    lines.append("## Data")
    lines.append(f"- raw RGB-D root: `{raw_root}`")
    lines.append(f"- raw RGB-D directory size GiB: `{size_gib(dir_size(raw_root))}`")
    lines.append(f"- raw RGB-D zip: `{zip_path}`")
    lines.append(f"- raw RGB-D zip size GiB: `{size_gib(dir_size(zip_path))}`")
    lines.append(f"- download status: `{download_log.get('status', 'not_recorded')}`")
    lines.append("")
    lines.append("## Raw Check")
    lines.append(f"- intrinsics exists: `{raw_check.get('intrinsics_exists')}`")
    lines.append(f"- intrinsics path: `{raw_check.get('intrinsics_path', '')}`")
    lines.append(f"- fragments checked: `{raw_check.get('fragment_count')}`")
    lines.append(f"- missing depth frames: `{raw_check.get('missing_depth_count')}`")
    lines.append(f"- missing pose frames: `{raw_check.get('missing_pose_count')}`")
    lines.append("")
    lines.append("## Subset")
    lines.append(f"- redkitchen pair count: `{baseline_metrics.get('num_pairs')}`")
    lines.append(f"- precomputed target fragments: `{len(precompute_report.get('precomputed', []))}`")
    lines.append("")
    lines.append("## Free-Space Parameters")
    params = precompute_report.get("parameters", {})
    for name in [
        "voxel_size",
        "depth_stride",
        "max_depth",
        "surface_truncation",
        "free_space_truncation",
        "min_valid_depth",
        "grid_margin",
        "frame_step",
        "query_source_points",
    ]:
        value = params.get(name, getattr(args, name, "not_used"))
        lines.append(f"- {name}: `{value}`")
    lines.append("")
    lines.append("## Coordinate Sanity")
    sanity = precompute_report.get("sanity", [])
    ok_count = sum(1 for item in sanity if item.get("ok"))
    mean_values = [item["best"]["mean"] for item in sanity if item.get("ok") and "best" in item]
    median_values = [item["best"]["median"] for item in sanity if item.get("ok") and "best" in item]
    lines.append(f"- passed fragments: `{ok_count}/{len(sanity)}`")
    lines.append(f"- mean NN distance avg: `{float(np.mean(mean_values)) if mean_values else float('nan')}`")
    lines.append(f"- median NN distance avg: `{float(np.mean(median_values)) if median_values else float('nan')}`")
    notes = sorted(
        {
            item["best"]["cam_pose_mode"] + "/" + item["best"]["fragment_pose_mode"] + f"/scale={item['best']['depth_scale']}"
            for item in sanity
            if item.get("ok") and "best" in item
        }
    )
    lines.append(f"- coordinate interpretations: `{'; '.join(notes)}`")
    lines.append("")
    lines.append("## Baseline")
    for key in ["Mean Reg Recall", "Mean Re", "Mean Te", "Mean FMR", "Mean model time", "Output precision", "Output recall"]:
        lines.append(f"- {key}: `{baseline_metrics.get(key)}`")
    lines.append("")
    lines.append("## FSV Discrimination")
    for key, value in fsv_auc.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## FSV Gate")
    for label, metrics in gate_results.items():
        lines.append(f"### {label}")
        for key, value in metrics.items():
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Proxy Comparison")
    lines.append("- geometry_nn_proxy is only a comparison baseline, not FSV and not physical self-consistency.")
    lines.append(f"- geometry_nn_proxy AUC_failure_high: `{proxy_auc}`")
    lines.append(f"- FSV better than geometry_nn_proxy: `{bool(fsv_auc.get('AUC_failure_high', float('nan')) > proxy_auc) if not math.isnan(proxy_auc) else 'unknown'}`")
    lines.append("")
    lines.append("## Round2 Reset Prior Readiness")
    ready = (
        raw_check.get("intrinsics_exists")
        and raw_check.get("missing_depth_count") == 0
        and raw_check.get("missing_pose_count") == 0
        and fsv_auc.get("valid_pairs", 0) > 0
        and not math.isnan(float(fsv_auc.get("AUC_failure_high", float("nan"))))
    )
    lines.append(f"- ready_to_connect_to_round2_reset_prior: `{bool(ready)}`")
    lines.append("- current result validates true RGB-D FSV verifier on redkitchen, not full Diagnosis-guided REGOR-S2.")
    lines.append("")
    lines.append("## S2 Parameters Not Used In This Verifier")
    for name in [
        "K_top",
        "tau_fsv",
        "tau_rho",
        "tau_overlap",
        "alpha",
        "lambda_prior",
        "eta_L",
        "gamma_reset",
        "epsilon_log",
        "n_min",
        "n_max",
        "ransac_iters",
        "robust_kernel_c",
        "max_rounds",
        "early_stop_enabled",
        "density_filter_enabled",
    ]:
        lines.append(f"- {name}: `not used; no S2 active regeneration in this experiment`")
    lines.append("")
    lines.append("## Deviations And Repairs")
    if deviations:
        for item in deviations:
            lines.append(f"- {item}")
    else:
        lines.append("- none recorded")
    lines.append("")
    lines.append("## Required Output Files")
    for path in [
        args.manifest,
        args.pairs,
        args.predictions,
        args.baseline_metrics,
        args.fsv_scores,
        args.fsv_auc,
        args.summary,
    ]:
        exists = True if Path(path) == Path(args.summary) else Path(path).exists()
        lines.append(f"- `{path}` exists: `{exists}`")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_manifest.json"))
    parser.add_argument("--pairs", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_pairs.json"))
    parser.add_argument("--fsv-root", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / SCENE))
    parser.add_argument("--fragment-root", default=str(REPO_ROOT / "3dmatch" / "test" / SCENE / "fragments"))
    parser.add_argument("--predictions", default=str(REPO_ROOT / "outputs" / "redkitchen_baseline" / "predictions.csv"))
    parser.add_argument("--baseline-metrics", default=str(REPO_ROOT / "outputs" / "redkitchen_baseline" / "metrics.json"))
    parser.add_argument("--raw-check", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_raw_check.json"))
    parser.add_argument("--precompute-report", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_fsv_precompute_report.json"))
    parser.add_argument("--download-log", default=str(REPO_ROOT / "outputs" / "redkitchen_fsv" / "download_log.json"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "3dmatch_raw" / "test"))
    parser.add_argument("--fsv-scores", default=str(REPO_ROOT / "outputs" / "redkitchen_fsv" / "fsv_scores.csv"))
    parser.add_argument("--fsv-auc", default=str(REPO_ROOT / "outputs" / "redkitchen_fsv" / "fsv_auc.json"))
    parser.add_argument("--summary", default=str(REPO_ROOT / "outputs" / "redkitchen_fsv" / "summary.md"))
    parser.add_argument("--query-source-points", type=int, default=5000)
    args = parser.parse_args()

    manifest = {int(item["fragment_index"]): item for item in read_json(args.manifest)}
    predictions = read_csv(args.predictions)
    baseline_metrics = read_json(args.baseline_metrics)
    raw_check = read_json(args.raw_check)
    precompute_report = read_json(args.precompute_report)
    src_cache = {}
    tgt_cache = {}
    fsv_cache = {}
    rows = []

    for row in predictions:
        src_id = int(row["src_id"])
        tgt_id = int(row["tgt_id"])
        if src_id not in src_cache:
            src_cache[src_id] = load_fragment_points(manifest[src_id]["fragment_ply"])
        if tgt_id not in tgt_cache:
            tgt_cache[tgt_id] = load_fragment_points(manifest[tgt_id]["fragment_ply"])
        if tgt_id not in fsv_cache:
            fsv_path = Path(args.fsv_root) / f"{row['tgt_fragment']}_fsv.npz"
            if not fsv_path.exists():
                raise FileNotFoundError(f"Missing target FSV: {fsv_path}")
            fsv_cache[tgt_id] = load_fsv(fsv_path)
        trans = cell_to_matrix(row["T_est"])
        fsv_score, free_count, valid_count, outside_count = query_fsv_score(
            src_cache[src_id],
            trans,
            fsv_cache[tgt_id],
            args.query_source_points,
            int(row["pair_id"]),
        )
        proxy_nn = proxy_nn_score(src_cache[src_id], tgt_cache[tgt_id], trans, args.query_source_points, int(row["pair_id"]))
        out = dict(row)
        out.update(
            {
                "fsv_score": "" if math.isnan(fsv_score) else fsv_score,
                "fsv_valid": int(not math.isnan(fsv_score)),
                "free_query_count": free_count,
                "valid_query_count": valid_count,
                "outside_query_count": outside_count,
                "geometry_nn_proxy": proxy_nn,
            }
        )
        rows.append(out)

    fields = list(rows[0].keys()) if rows else []
    write_csv(args.fsv_scores, rows, fields)

    fsv_auc = compute_auc(rows)
    valid_rows = [r for r in rows if str(r["fsv_valid"]) == "1"]
    labels = np.asarray([int(r["success"]) for r in valid_rows], dtype=np.int32)
    fsv = np.asarray([float(r["fsv_score"]) for r in valid_rows], dtype=np.float64)
    all_accepted = np.ones(len(rows), dtype=bool)
    gate_results = {"Baseline REGOR": accepted_metrics(rows, all_accepted)}
    for specificity in (0.90, 0.95):
        threshold = specificity_gate_threshold(fsv, labels, specificity)
        accepted = np.zeros(len(rows), dtype=bool)
        valid_lookup = {(int(r["pair_id"]), int(r["src_id"]), int(r["tgt_id"])) for r in valid_rows if float(r["fsv_score"]) <= threshold}
        for i, row in enumerate(rows):
            key = (int(row["pair_id"]), int(row["src_id"]), int(row["tgt_id"]))
            accepted[i] = key in valid_lookup
        gate_results[f"REGOR + FSV gate @ {int(specificity * 100)}% specificity"] = accepted_metrics(rows, accepted)
    proxy_values = np.asarray([float(r["geometry_nn_proxy"]) for r in valid_rows], dtype=np.float64)
    proxy_auc = roc_auc_score_binary(1 - labels, proxy_values) if len(valid_rows) else float("nan")
    fsv_auc["gate_results"] = gate_results
    fsv_auc["geometry_nn_proxy_AUC_failure_high"] = proxy_auc
    write_json(args.fsv_auc, json_sanitize(fsv_auc))

    deviations = []
    download_log = read_json(args.download_log) if Path(args.download_log).exists() else {}
    if download_log.get("mode") == "selective_http_range":
        deviations.append(
            "Full official zip download hit disk-space limits; recovered by HTTP Range extraction of only official depth/pose/intrinsics members."
        )
    if precompute_report.get("parameters", {}).get("frame_step", 1) != 1:
        deviations.append(f"frame_step={precompute_report['parameters'].get('frame_step')} used instead of every raw frame")
    for item in precompute_report.get("sanity", []):
        if item.get("valid_frame_count") is not None and item.get("valid_frame_count") < item.get("total_frame_count", 0):
            deviations.append(
                f"{item['fragment_id']} used {item['valid_frame_count']} / {item['total_frame_count']} raw frames after per-frame coordinate sanity filtering"
            )
    invalid_pairs = int(fsv_auc.get("invalid_pairs", 0))
    if invalid_pairs:
        deviations.append(f"{invalid_pairs} pairs had no transformed source query inside target FSV grid and were rejected by FSV gate")
    deviations.append("geometry_nn_proxy was computed only for comparison; it is not used as FSV")
    summary = summarize_markdown(args, baseline_metrics, raw_check, precompute_report, fsv_auc, gate_results, proxy_auc, deviations)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(summary, encoding="utf-8")
    print(f"fsv_scores={args.fsv_scores}")
    print(f"fsv_auc={args.fsv_auc}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
