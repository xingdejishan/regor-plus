import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return edict(json.load(f))


def min_dist_to_points(query, reference, chunk_size=2048):
    mins = []
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        mins.append(torch.cdist(query[start:end], reference).min(dim=1)[0])
    return torch.cat(mins, dim=0)


def transform_points(points, trans):
    from utils.SE3 import transform

    return transform(points[None], trans)[0]


def pose_errors(trans, gt_trans):
    trans = trans[0]
    gt_trans = gt_trans[0]
    r = trans[:3, :3]
    gt_r = gt_trans[:3, :3]
    trace_value = torch.trace(r.T @ gt_r)
    re = torch.acos(torch.clamp((trace_value - 1.0) / 2.0, min=-1.0, max=1.0)) * 180.0 / np.pi
    te = torch.norm(trans[:3, 3] - gt_trans[:3, 3]) * 100.0
    return float(re.item()), float(te.item())


def pose_success(trans, gt_trans, re_thre, te_thre):
    re, te = pose_errors(trans, gt_trans)
    return int(re < re_thre and te < te_thre), re, te


def gt_overlap_mask(src_points, tgt_points, gt_trans, threshold):
    warped = transform_points(src_points, gt_trans)
    return min_dist_to_points(warped, tgt_points) < threshold


def gt_inlier_mask(src_corr, tgt_corr, gt_trans, threshold):
    from utils.SE3 import transform

    if src_corr.shape[1] == 0:
        return torch.zeros(0, dtype=torch.bool, device=src_corr.device)
    residual = torch.norm(transform(src_corr, gt_trans) - tgt_corr, dim=-1)[0]
    return residual < threshold


def bool_mean(mask):
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def count_novel_r2_inliers(r1_src_corr, r1_tgt_corr, r2_src_corr, r2_tgt_corr, gt_trans, threshold, voxel_size):
    r2_inlier = gt_inlier_mask(r2_src_corr, r2_tgt_corr, gt_trans, threshold)
    if r2_inlier.sum() == 0:
        return 0
    r1_inlier = gt_inlier_mask(r1_src_corr, r1_tgt_corr, gt_trans, threshold)
    r1_src = r1_src_corr[0, r1_inlier]
    r2_src = r2_src_corr[0, r2_inlier]
    if r1_src.shape[0] == 0:
        return int(r2_src.shape[0])
    r1_voxels = torch.unique(torch.floor(r1_src / voxel_size).to(torch.int64), dim=0)
    r2_voxels = torch.floor(r2_src / voxel_size).to(torch.int64)
    r1_keys = (r1_voxels[:, 0] * 73856093) ^ (r1_voxels[:, 1] * 19349663) ^ (r1_voxels[:, 2] * 83492791)
    r2_keys = (r2_voxels[:, 0] * 73856093) ^ (r2_voxels[:, 1] * 19349663) ^ (r2_voxels[:, 2] * 83492791)
    return int((~torch.isin(r2_keys, r1_keys)).sum().item())


def region_counts(indices, guide, eps):
    best = guide["o_best"][indices] > 0
    broad = guide["o_broad"][indices] > 0
    reset = guide["o_reset"][indices] > eps
    broad_only = (~best) & broad
    reset_only = (~best) & (~broad) & reset
    outside = (~best) & (~broad) & (~reset)
    return {
        "best": int(best.sum().item()),
        "local_expand": int(broad_only.sum().item()),
        "reset": int(reset_only.sum().item()),
        "reset_nonexclusive": int(reset.sum().item()),
        "outside": int(outside.sum().item()),
        "count": int(indices.numel()),
        "best_rate": bool_mean(best),
        "local_expand_rate": bool_mean(broad_only),
        "reset_rate": bool_mean(reset_only),
        "reset_nonexclusive_rate": bool_mean(reset),
        "outside_rate": bool_mean(outside),
    }


def dominant_region(counts):
    candidates = {
        "local_expand": counts["local_expand"],
        "reset": counts["reset"],
        "best": counts["best"],
        "outside": counts["outside"],
    }
    name, value = max(candidates.items(), key=lambda item: item[1])
    return name if value > 0 else "empty"


def aggregate_rows(rows):
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ] if rows else []
    return {
        "count": len(rows),
        "means": {
            key: float(np.mean([float(row[key]) for row in rows]))
            for key in numeric_keys
        } if rows else {},
    }


def add_group(groups, key, row):
    groups[key].append(row)


def serialize_r1_topk(topk_trans, src_corr, tgt_corr, gt_trans, config):
    from utils.SE3 import transform

    matrices = topk_trans[0].detach().cpu().tolist()
    scores = []
    for rank in range(topk_trans.shape[1]):
        candidate = topk_trans[:, rank, :, :]
        residual = torch.norm(transform(src_corr, candidate) - tgt_corr, dim=-1)[0]
        model_inlier_count = int((residual < config.active_overlap_threshold).sum().item())
        success, re, te = pose_success(candidate, gt_trans, config.re_thre, config.te_thre)
        scores.append({
            "rank": rank,
            "model_inlier_count": model_inlier_count,
            "gt_success": success,
            "gt_re": re,
            "gt_te": te,
        })
    return json.dumps(matrices, separators=(",", ":")), json.dumps(scores, separators=(",", ":"))


def evaluate(args):
    os.chdir(REPO_ROOT)
    config = load_config(args.config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_Devices

    from Correspondence_regenerate_v2 import Regenerator
    from Tranformation_estimaton import Estimator
    from dataset import ThreeDLoMatchLoader
    from initial_matching_plus import Matcher_plus
    from test_3DLoMatch import (
        active_param,
        build_guided_prior,
        generate_r1_topk_transforms,
        robust_weighted_estimate,
        sample_correspondences,
        sample_guided_seed_correspondences,
        sample_random_uniform_seed_correspondences,
        select_best_r1_transform,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ThreeDLoMatchLoader(
        root=config.data_path,
        descriptor=config.descriptor,
        inlier_threshold=config.inlier_threshold,
        num_node=config.num_node,
        use_mutual=config.use_mutual,
        free_space_root=config.free_space_root,
        rgbd_root=config.rgbd_root,
        overlap_pred_root=config.overlap_pred_root,
        use_overlap_proxy=config.use_overlap_proxy,
        use_rgbd_fsv=config.use_rgbd_fsv,
        fsv_voxel_size=config.active_fsv_voxel_size,
        fsv_depth_scale=config.active_fsv_depth_scale,
        fsv_trunc_margin=config.active_fsv_trunc_margin,
        fsv_stride=config.active_fsv_stride,
        fsv_max_frames=config.active_fsv_max_frames,
    )
    matcher = Matcher_plus(
        inlier_threshold=0.1,
        num_node="all",
        use_mutual=False,
        d_thre=0.1,
        num_iterations=10,
        ratio=0.2,
        nms_radius=0.1,
        max_points=8000,
        k1=60,
        k2=50,
        FS_TCD_thre=0.05,
        relax_match_num=100,
        NS_by_IC=50,
    )
    regenerator = Regenerator()
    estimator = Estimator(num_node=config.num_node)

    num_pairs = min(loader.__len__(), int(config.max_pairs or loader.__len__()))
    if args.max_pairs:
        num_pairs = min(num_pairs, args.max_pairs)

    rows = []
    triggered_rows = []
    candidate_rows = []
    groups = defaultdict(list)

    with torch.no_grad():
        for pair_idx in range(num_pairs):
            data = loader.get_data(pair_idx)
            (
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                gt_trans,
                src_pcd,
                tgt_pcd,
                src_overlap,
                tgt_overlap,
                target_free_space,
                overlap_source,
            ) = data

            (
                _,
                src_keypts_corr_filtered,
                tgt_keypts_corr_filtered,
                src_keypts_corr,
                tgt_keypts_corr,
                _,
                _,
            ) = matcher.estimator(src_keypts, tgt_keypts, src_features, tgt_features, gt_trans)
            seed_src_corr = src_keypts_corr_filtered
            seed_tgt_corr = tgt_keypts_corr_filtered
            if seed_src_corr.shape[1] == 0:
                seed_src_corr = src_keypts_corr
                seed_tgt_corr = tgt_keypts_corr

            r1_seed_src, r1_seed_tgt = sample_correspondences(
                seed_src_corr,
                seed_tgt_corr,
                active_param(config, "active_round1_sampling"),
            )
            r1_src_corr, r1_tgt_corr, r1_trans_initial = regenerator.regenerate(
                r1_seed_src,
                r1_seed_tgt,
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                gt_trans,
                knn_num=active_param(config, "active_round1_knn"),
                sampling_num=active_param(config, "active_round1_sampling"),
            )
            r1_topk_trans = generate_r1_topk_transforms(r1_src_corr, r1_tgt_corr, r1_trans_initial, config)
            t_best, r1_topk_trans, _ = select_best_r1_transform(r1_src_corr, r1_tgt_corr, r1_topk_trans, config)
            r1_only_weights = torch.ones(r1_src_corr.shape[1], dtype=r1_src_corr.dtype, device=r1_src_corr.device)
            t_r1_final, _, _, _, r1_only_final_info = robust_weighted_estimate(
                r1_src_corr,
                r1_tgt_corr,
                t_best,
                config,
                match_weights=r1_only_weights,
            )
            guide, diagnostics = build_guided_prior(
                src_keypts[0],
                tgt_keypts[0],
                src_overlap[0],
                tgt_overlap[0],
                r1_src_corr[0],
                r1_tgt_corr[0],
                t_best,
                config,
                topk_trans=r1_topk_trans,
                target_free_space=target_free_space,
            )
            early_stop = active_param(config, "use_early_stop") and (
                diagnostics["fsv_value"] < active_param(config, "active_tau_fsv")
                and diagnostics["roc_bar"] > active_param(config, "active_tau_rho")
            )
            t_best_success, t_best_re, t_best_te = pose_success(t_best, gt_trans, config.re_thre, config.te_thre)
            r1_topk_trans_json, r1_topk_s2_json = serialize_r1_topk(
                r1_topk_trans,
                r1_src_corr,
                r1_tgt_corr,
                gt_trans,
                config,
            )
            enter_round2 = (
                active_param(config, "use_round2")
                and active_param(config, "active_max_rounds") >= 2
                and active_param(config, "round2_mode") != "none"
                and not early_stop
            )
            pair_meta = {
                "pair_id": pair_idx,
                "pair_index": pair_idx,
                "scene": loader.infos["src"][pair_idx].split("/")[1],
                "src_fragment": loader.infos["src"][pair_idx].split("/")[-1],
                "tgt_fragment": loader.infos["tgt"][pair_idx].split("/")[-1],
                "R1_success": t_best_success,
                "fsv_value": float(diagnostics["fsv_value"]),
                "roc_bar": float(diagnostics["roc_bar"]),
                "enter_round2": int(enter_round2),
                "R2_success": None,
                "r1_topk_trans_json": r1_topk_trans_json,
                "r1_topk_s2_json": r1_topk_s2_json,
            }
            if not enter_round2:
                rows.append(pair_meta)
                if len(rows) % args.progress_interval == 0:
                    print(f"audited pairs {len(rows)}, scanned {pair_idx + 1}/{num_pairs}")
                continue

            round2_mode = active_param(config, "round2_mode")
            regenerate_mode = "local"
            regenerate_guide = {
                "local_radius": active_param(config, "active_round2_local_radius"),
                "local_max_points": active_param(config, "active_round2_local_max_points"),
                "generalized_mutual_k": active_param(config, "active_round2_mutual_k"),
                "max_matches_per_seed": active_param(config, "active_round2_knn"),
            }
            if round2_mode == "original":
                r2_seed_src, r2_seed_tgt = sample_correspondences(
                    r1_src_corr,
                    r1_tgt_corr,
                    active_param(config, "active_round2_sampling"),
                )
                regenerate_mode = "paired_local"
            elif round2_mode == "random_uniform":
                r2_seed_src, r2_seed_tgt = sample_random_uniform_seed_correspondences(
                    src_keypts,
                    tgt_keypts,
                    active_param(config, "active_round2_sampling"),
                )
                regenerate_mode = "paired_local"
            elif round2_mode == "weakness_guided":
                r2_seed_src, r2_seed_tgt = sample_guided_seed_correspondences(
                    src_keypts,
                    tgt_keypts,
                    src_features,
                    tgt_features,
                    r1_src_corr,
                    r1_tgt_corr,
                    guide,
                    active_param(config, "active_round2_sampling"),
                )
                guide["local_radius"] = active_param(config, "active_round2_local_radius")
                guide["local_max_points"] = active_param(config, "active_round2_local_max_points")
                guide["generalized_mutual_k"] = active_param(config, "active_round2_mutual_k")
                guide["max_matches_per_seed"] = active_param(config, "active_round2_knn")
                regenerate_mode = "guided_global"
                regenerate_guide = guide
            else:
                raise ValueError(f"Unsupported round2_mode: {round2_mode}")

            r2_src_corr, r2_tgt_corr, t_r2 = regenerator.regenerate(
                r2_seed_src,
                r2_seed_tgt,
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                gt_trans,
                knn_num=active_param(config, "active_round2_knn"),
                sampling_num=active_param(config, "active_round2_sampling"),
                knn_radius=active_param(config, "active_round2_local_radius"),
                guide=regenerate_guide,
                mode=regenerate_mode,
            )
            for candidate in regenerator.last_r2_candidates:
                candidate_trans = candidate["candidate_trans"]
                if candidate_trans is not None:
                    candidate_trans_for_error = candidate_trans.to(gt_trans.device)
                    candidate_re, candidate_te = pose_errors(candidate_trans_for_error, gt_trans)
                    candidate_pose_json = json.dumps(candidate_trans[0].tolist(), separators=(",", ":"))
                else:
                    candidate_re = None
                    candidate_te = None
                    candidate_pose_json = ""
                candidate_rows.append({
                    "pair_id": pair_idx,
                    "pair_index": pair_idx,
                    "scene": loader.infos["src"][pair_idx].split("/")[1],
                    "src_fragment": loader.infos["src"][pair_idx].split("/")[-1],
                    "tgt_fragment": loader.infos["tgt"][pair_idx].split("/")[-1],
                    "round2_mode": round2_mode,
                    "seed_id": candidate["seed_id"],
                    "seed_src_index": candidate["seed_src_index"],
                    "seed_tgt_index": candidate["seed_tgt_index"],
                    "S2": candidate["s2"],
                    "S2_sum": candidate["s2_sum"],
                    "correspondence_count": candidate["correspondence_count"],
                    "model_inlier_count": candidate["model_inlier_count"],
                    "valid_pose": candidate["valid_pose"],
                    "RE": candidate_re,
                    "TE": candidate_te,
                    "candidate_trans_json": candidate_pose_json,
                })
            if r2_src_corr.shape[1] < 3:
                t_r2 = t_best

            src_final = torch.cat([r1_src_corr, r2_src_corr], dim=1)
            tgt_final = torch.cat([r1_tgt_corr, r2_tgt_corr], dim=1)
            r1_weights = torch.ones(r1_src_corr.shape[1], dtype=r1_src_corr.dtype, device=r1_src_corr.device)
            if regenerator.last_match_weights is not None and regenerator.last_match_weights.shape[0] == r2_src_corr.shape[1]:
                r2_weights = regenerator.last_match_weights.to(device=r1_src_corr.device, dtype=r1_src_corr.dtype)
            else:
                r2_weights = torch.ones(r2_src_corr.shape[1], dtype=r1_src_corr.dtype, device=r1_src_corr.device)
            match_weights = torch.cat([r1_weights, r2_weights], dim=0)
            if active_param(config, "use_ctc") and src_final.shape[1] >= 3:
                t_ctc, _, src_final, tgt_final = estimator.estimator(src_final, tgt_final, src_keypts, tgt_keypts, gt_trans)
                match_weights = torch.ones(src_final.shape[1], dtype=src_final.dtype, device=src_final.device)
            else:
                t_ctc = t_r2
            t_final, _, _, _, final_info = robust_weighted_estimate(
                src_final,
                tgt_final,
                t_ctc,
                config,
                match_weights=match_weights,
            )

            gt_src_overlap = gt_overlap_mask(src_keypts[0], tgt_keypts[0], gt_trans, config.inlier_threshold)
            source_score = (guide["source_prior"] * guide["source_under_support"]).detach()
            topk_count = min(active_param(config, "active_round2_sampling"), source_score.shape[0])
            topk_idx = torch.topk(source_score, k=topk_count, largest=True).indices
            percentile_count = max(1, int(np.ceil(source_score.shape[0] * args.top_percentile / 100.0)))
            percentile_idx = torch.topk(source_score, k=percentile_count, largest=True).indices
            actual_source = r2_src_corr[0]
            r2_seed_gt_overlap = gt_overlap_mask(r2_seed_src[0], tgt_keypts[0], gt_trans, config.inlier_threshold)
            actual_gt_overlap = gt_overlap_mask(actual_source, tgt_keypts[0], gt_trans, config.inlier_threshold)

            selected_indices = topk_idx
            selected_region_counts = region_counts(selected_indices, guide, active_param(config, "active_eps"))
            actual_nearest = torch.argmin(torch.cdist(actual_source, src_keypts[0]), dim=1) if actual_source.shape[0] else torch.empty(0, dtype=torch.long, device=src_keypts.device)
            actual_region_counts = region_counts(actual_nearest, guide, active_param(config, "active_eps")) if actual_nearest.numel() else {
                "best": 0,
                "local_expand": 0,
                "reset": 0,
                "reset_nonexclusive": 0,
                "outside": 0,
                "count": 0,
                "best_rate": 0.0,
                "local_expand_rate": 0.0,
                "reset_rate": 0.0,
                "reset_nonexclusive_rate": 0.0,
                "outside_rate": 0.0,
            }
            generation_region = dominant_region(actual_region_counts)

            r2_inlier = gt_inlier_mask(r2_src_corr, r2_tgt_corr, gt_trans, config.inlier_threshold)
            r2_inlier_count = int(r2_inlier.sum().item())
            r2_corr_count = int(r2_inlier.numel())
            r2_precision = float(r2_inlier.float().mean().item()) if r2_corr_count else 0.0
            novel_r2_inlier_count = count_novel_r2_inliers(
                r1_src_corr,
                r1_tgt_corr,
                r2_src_corr,
                r2_tgt_corr,
                gt_trans,
                config.inlier_threshold,
                active_param(config, "active_voxel_size"),
            )

            t_r1_success, t_r1_re, t_r1_te = pose_success(t_best, gt_trans, config.re_thre, config.te_thre)
            t_r1_only_final_success, t_r1_only_final_re, t_r1_only_final_te = pose_success(
                t_r1_final,
                gt_trans,
                config.re_thre,
                config.te_thre,
            )
            t_r2_success, t_r2_re, t_r2_te = pose_success(t_r2, gt_trans, config.re_thre, config.te_thre)
            t_final_success, t_final_re, t_final_te = pose_success(t_final, gt_trans, config.re_thre, config.te_thre)

            row = {
                **pair_meta,
                "pair_index": pair_idx,
                "round2_mode": round2_mode,
                "generation_region": generation_region,
                "overlap_source": overlap_source,
                "fsv": float(diagnostics["fsv_value"]),
                "roc_bar": float(diagnostics["roc_bar"]),
                "t_best_success": t_best_success,
                "t_best_re": t_best_re,
                "t_best_te": t_best_te,
                "gt_source_overlap_count": int(gt_src_overlap.sum().item()),
                "gt_source_overlap_rate": bool_mean(gt_src_overlap),
                "o_topk_count": int(topk_idx.numel()),
                "o_topk_gt_overlap_rate": bool_mean(gt_src_overlap[topk_idx]),
                "o_top_percentile": float(args.top_percentile),
                "o_top_percentile_count": int(percentile_idx.numel()),
                "o_top_percentile_gt_overlap_rate": bool_mean(gt_src_overlap[percentile_idx]),
                "r2_seed_source_count": int(r2_seed_src.shape[1]),
                "r2_seed_source_gt_overlap_rate": bool_mean(r2_seed_gt_overlap),
                "r2_exploit_quota": int(guide.get("round2_num_exploit", 0)),
                "r2_explore_quota": int(guide.get("round2_num_explore", 0)),
                "r2_exploit_seed_count": int(guide.get("round2_exploit_seed_count", 0)),
                "r2_explore_seed_count": int(guide.get("round2_explore_seed_count", 0)),
                "r2_explore_compatibility_fallback": int(guide.get("round2_explore_compatibility_fallback", False)),
                "r2_candidate_pose_count": len(regenerator.last_r2_candidates),
                "r2_sampled_source_count": int(actual_source.shape[0]),
                "r2_sampled_source_gt_overlap_rate": bool_mean(actual_gt_overlap),
                "r2_corr_count": r2_corr_count,
                "r2_inlier_count": r2_inlier_count,
                "r2_precision": r2_precision,
                "novel_r2_inlier_count": novel_r2_inlier_count,
                "t_r1_success": t_r1_success,
                "t_r1_re": t_r1_re,
                "t_r1_te": t_r1_te,
                "t_r1_only_final_success": t_r1_only_final_success,
                "t_r1_only_final_re": t_r1_only_final_re,
                "t_r1_only_final_te": t_r1_only_final_te,
                "t_r1_only_final_used_count": int(r1_only_final_info["final_used_count"]),
                "t_r2_only_success": t_r2_success,
                "t_r2_only_re": t_r2_re,
                "t_r2_only_te": t_r2_te,
                "t_r1_plus_r2_final_success": t_final_success,
                "t_r1_plus_r2_final_re": t_final_re,
                "t_r1_plus_r2_final_te": t_final_te,
                "final_used_count": int(final_info["final_used_count"]),
                "selected_best_rate": selected_region_counts["best_rate"],
                "selected_local_expand_rate": selected_region_counts["local_expand_rate"],
                "selected_reset_rate": selected_region_counts["reset_rate"],
                "selected_reset_nonexclusive_rate": selected_region_counts["reset_nonexclusive_rate"],
                "selected_outside_rate": selected_region_counts["outside_rate"],
                "actual_best_rate": actual_region_counts["best_rate"],
                "actual_local_expand_rate": actual_region_counts["local_expand_rate"],
                "actual_reset_rate": actual_region_counts["reset_rate"],
                "actual_reset_nonexclusive_rate": actual_region_counts["reset_nonexclusive_rate"],
                "actual_outside_rate": actual_region_counts["outside_rate"],
            }
            row["R2_success"] = t_r2_success
            rows.append(row)
            triggered_rows.append(row)
            add_group(groups, f"t_best_{'success' if t_best_success else 'failure'}", row)
            add_group(groups, f"region_{generation_region}", row)
            add_group(groups, f"final_{'success' if t_final_success else 'failure'}", row)
            if len(rows) % args.progress_interval == 0:
                print(f"audited pairs {len(rows)}, triggered {len(triggered_rows)}, scanned {pair_idx + 1}/{num_pairs}")

    csv_path = output_dir / "round2_generation_audit_pairs.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    candidate_csv_path = output_dir / "r2_candidate_poses.csv"
    candidate_fields = [
        "pair_id", "pair_index", "scene", "src_fragment", "tgt_fragment", "round2_mode",
        "seed_id", "seed_src_index", "seed_tgt_index", "S2", "S2_sum",
        "correspondence_count", "model_inlier_count", "valid_pose", "RE", "TE",
        "candidate_trans_json",
    ]
    with open(candidate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        writer.writerows(candidate_rows)

    aggregate = {
        "num_pairs_scanned": num_pairs,
        "pairs_logged": len(rows),
        "triggered_pairs": len(triggered_rows),
        "top_percentile": args.top_percentile,
        "overall": aggregate_rows(triggered_rows),
        "groups": {key: aggregate_rows(value) for key, value in sorted(groups.items())},
        "split_definitions": {
            "t_best_success": f"RE < {config.re_thre} and TE < {config.te_thre} cm for refined T_best.",
            "generation_region": "Dominant actual R2 source region among best/local_expand/reset/outside; reset uses reset-only unless reset_nonexclusive fields are inspected.",
            "final_success": f"RE < {config.re_thre} and TE < {config.te_thre} cm for robust final pose from C_R1 + C_R2.",
        },
        "config": args.config_path,
        "csv": str(csv_path),
        "r2_candidate_csv": str(candidate_csv_path),
        "r2_candidate_rows": len(candidate_rows),
    }
    with open(output_dir / "round2_generation_audit_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)
    print(json.dumps(aggregate, indent=2))
    print(f"pairs: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json/config_3DLoMatch_FPFH_redkitchen_modified.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/round2_generation_audit"))
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--top-percentile", type=float, default=10.0)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
