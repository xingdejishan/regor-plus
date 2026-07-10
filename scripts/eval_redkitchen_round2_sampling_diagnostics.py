import argparse
import csv
import json
import os
import sys
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


def gt_overlap_mask(src_points, tgt_points, gt_trans, threshold):
    from utils.SE3 import transform

    warped = transform(src_points[None], gt_trans)[0]
    return min_dist_to_points(warped, tgt_points) < threshold


def gt_inlier_mask(src_corr, tgt_corr, gt_trans, threshold):
    from utils.SE3 import transform

    if src_corr.shape[1] == 0:
        return torch.zeros(0, dtype=torch.bool, device=src_corr.device)
    residual = torch.norm(transform(src_corr, gt_trans) - tgt_corr, dim=-1)[0]
    return residual < threshold


def ratio(mask):
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def evaluate(args):
    os.chdir(REPO_ROOT)
    config = load_config(args.config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_Devices

    from Correspondence_regenerate_v2 import Regenerator
    from dataset import ThreeDLoMatchLoader
    from initial_matching_plus import Matcher_plus
    from test_3DLoMatch import (
        active_param,
        build_guided_prior,
        generate_r1_topk_transforms,
        sample_correspondences,
        sample_guided_seed_correspondences,
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
    num_pairs = min(loader.__len__(), int(config.max_pairs or loader.__len__()))
    if args.max_pairs:
        num_pairs = min(num_pairs, args.max_pairs)

    rows = []
    totals = {
        "sample_count": 0,
        "best": 0,
        "broad_only": 0,
        "reset_only": 0,
        "outside": 0,
        "reset_nonexclusive": 0,
        "sample_gt_overlap": 0,
        "r2_corr": 0,
        "r2_inlier": 0,
        "final_success_with_r2_inliers": 0,
    }

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
                pred_trans,
                src_keypts_corr_filtered,
                tgt_keypts_corr_filtered,
                src_keypts_corr,
                tgt_keypts_corr,
                src_desc_corr_final,
                tgt_desc_corr_final,
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
            r1_src_corr, r1_tgt_corr, r1_trans = regenerator.regenerate(
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

            r1_topk_trans = generate_r1_topk_transforms(r1_src_corr, r1_tgt_corr, r1_trans, config)
            r1_trans, r1_topk_trans, r1_best_inliers = select_best_r1_transform(
                r1_src_corr,
                r1_tgt_corr,
                r1_topk_trans,
                config,
            )
            guide, diagnostics = build_guided_prior(
                src_keypts[0],
                tgt_keypts[0],
                src_overlap[0],
                tgt_overlap[0],
                r1_src_corr[0],
                r1_tgt_corr[0],
                r1_trans,
                config,
                topk_trans=r1_topk_trans,
                target_free_space=target_free_space,
            )

            early_stop = active_param(config, "use_early_stop") and (
                diagnostics["fsv_value"] < active_param(config, "active_tau_fsv")
                and diagnostics["roc_bar"] > active_param(config, "active_tau_rho")
            )
            enter_round2 = (
                active_param(config, "use_round2")
                and active_param(config, "active_max_rounds") >= 2
                and active_param(config, "round2_mode") != "none"
                and not early_stop
            )
            if not enter_round2:
                continue

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
            r2_src_corr, r2_tgt_corr, r2_trans = regenerator.regenerate(
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
                guide=guide,
                mode="guided_global",
            )

            selected = torch.cdist(r2_seed_src[0], src_keypts[0]).argmin(dim=1)
            best = guide["o_best"][selected] > 0
            broad = guide["o_broad"][selected] > 0
            reset = guide["o_reset"][selected] > active_param(config, "active_eps")
            broad_only = (~best) & broad
            reset_only = (~best) & (~broad) & reset
            outside = (~best) & (~broad) & (~reset)
            gt_src_overlap = gt_overlap_mask(src_keypts[0], tgt_keypts[0], gt_trans, config.inlier_threshold)[selected]
            r2_inlier = gt_inlier_mask(r2_src_corr, r2_tgt_corr, gt_trans, config.inlier_threshold)
            r2_inlier_count = int(r2_inlier.sum().item())
            r2_corr_count = int(r2_inlier.numel())
            r2_precision = float(r2_inlier.float().mean().item()) if r2_corr_count else 0.0

            totals["sample_count"] += int(selected.numel())
            totals["best"] += int(best.sum().item())
            totals["broad_only"] += int(broad_only.sum().item())
            totals["reset_only"] += int(reset_only.sum().item())
            totals["outside"] += int(outside.sum().item())
            totals["reset_nonexclusive"] += int(reset.sum().item())
            totals["sample_gt_overlap"] += int(gt_src_overlap.sum().item())
            totals["r2_corr"] += r2_corr_count
            totals["r2_inlier"] += r2_inlier_count

            row = {
                "pair_index": pair_idx,
                "fsv": diagnostics["fsv_value"],
                "roc_bar": diagnostics["roc_bar"],
                "sample_count": int(selected.numel()),
                "o_best_sample_ratio": ratio(best),
                "o_broad_only_sample_ratio": ratio(broad_only),
                "o_reset_only_sample_ratio": ratio(reset_only),
                "o_reset_nonexclusive_sample_ratio": ratio(reset),
                "outside_prior_sample_ratio": ratio(outside),
                "r2_sampled_source_gt_overlap_ratio": ratio(gt_src_overlap),
                "r2_corr_count": r2_corr_count,
                "r2_inlier_count": r2_inlier_count,
                "r2_precision": r2_precision,
            }
            rows.append(row)
            if len(rows) % args.progress_interval == 0:
                print(f"trigger diagnostics {len(rows)} pairs, scanned {pair_idx + 1}/{num_pairs}")

    csv_path = output_dir / "round2_sampling_pairs.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    sample_count = max(totals["sample_count"], 1)
    r2_corr_count = max(totals["r2_corr"], 1)
    metrics = {
        "num_pairs": num_pairs,
        "trigger_pairs": len(rows),
        "sample_count_total": totals["sample_count"],
        "aggregate": {
            "o_best_sample_ratio": totals["best"] / sample_count,
            "o_broad_only_sample_ratio": totals["broad_only"] / sample_count,
            "o_reset_only_sample_ratio": totals["reset_only"] / sample_count,
            "o_reset_nonexclusive_sample_ratio": totals["reset_nonexclusive"] / sample_count,
            "outside_prior_sample_ratio": totals["outside"] / sample_count,
            "r2_sampled_source_gt_overlap_ratio": totals["sample_gt_overlap"] / sample_count,
            "r2_correspondence_inlier_count": totals["r2_inlier"],
            "r2_correspondence_count": totals["r2_corr"],
            "r2_precision": totals["r2_inlier"] / r2_corr_count,
        },
        "per_pair_mean": {
            key: float(np.mean([row[key] for row in rows])) if rows else 0.0
            for key in [
                "o_best_sample_ratio",
                "o_broad_only_sample_ratio",
                "o_reset_only_sample_ratio",
                "o_reset_nonexclusive_sample_ratio",
                "outside_prior_sample_ratio",
                "r2_sampled_source_gt_overlap_ratio",
                "r2_precision",
            ]
        },
        "note": "Mutually exclusive prior regions are best, broad-only, reset-only, outside. reset_nonexclusive reports any selected source with o_reset > eps. GT overlap is min_q ||T_gt p - q|| < inlier_threshold.",
        "config": args.config_path,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"pairs: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json/config_3DLoMatch_FPFH_redkitchen_modified.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/redkitchen_round2_sampling_diagnostics"))
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
