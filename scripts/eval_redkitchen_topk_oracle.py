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


def transform_points(points, trans):
    from utils.SE3 import transform

    return transform(points[None], trans)[0]


def pose_error_cm_deg(trans, gt_trans):
    trans = trans[0]
    gt_trans = gt_trans[0]
    r = trans[:3, :3]
    gt_r = gt_trans[:3, :3]
    trace_value = torch.trace(r.T @ gt_r)
    re = torch.acos(torch.clamp((trace_value - 1.0) / 2.0, min=-1.0, max=1.0)) * 180.0 / np.pi
    te = torch.norm(trans[:3, 3] - gt_trans[:3, 3]) * 100.0
    return float(re.item()), float(te.item())


def refine_ranked_topk(src_corr, tgt_corr, topk_trans, threshold):
    from test_3DLoMatch import refine_transform_on_correspondences

    refined = []
    for i in range(topk_trans.shape[1]):
        trans = refine_transform_on_correspondences(topk_trans[:, i, :, :], src_corr, tgt_corr, threshold)
        residual = torch.norm(transform_points(src_corr[0], trans) - tgt_corr[0], dim=1)
        inliers = int((residual < threshold).sum().item())
        refined.append((inliers, i, trans))
    refined.sort(key=lambda item: (-item[0], item[1]))
    return refined


def evaluate(args):
    os.chdir(REPO_ROOT)
    config = load_config(args.config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_Devices

    from Correspondence_regenerate_v2 import Regenerator
    from dataset import ThreeDLoMatchLoader
    from initial_matching_plus import Matcher_plus
    from test_3DLoMatch import active_param, generate_r1_topk_transforms, sample_correspondences

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
        use_rgbd_fsv=False,
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

    k_values = [int(k) for k in args.k_values.split(",")]
    max_k = max(k_values)
    rows = []
    counters = {k: 0 for k in k_values}
    candidate_counts = []

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
            topk_trans = generate_r1_topk_transforms(r1_src_corr, r1_tgt_corr, r1_trans, config)
            ranked = refine_ranked_topk(r1_src_corr, r1_tgt_corr, topk_trans, active_param(config, "active_overlap_threshold"))
            candidate_counts.append(len(ranked))

            row = {
                "pair_index": pair_idx,
                "candidate_count": len(ranked),
                "r1_corr_count": int(r1_src_corr.shape[1]),
                "overlap_source": overlap_source,
            }
            for rank, (internal_inliers, original_index, trans) in enumerate(ranked[:max_k], start=1):
                re, te = pose_error_cm_deg(trans, gt_trans)
                success = int(re < config.re_thre and te < config.te_thre)
                row[f"rank{rank}_source_index"] = int(original_index)
                row[f"rank{rank}_internal_inliers"] = int(internal_inliers)
                row[f"rank{rank}_re"] = re
                row[f"rank{rank}_te"] = te
                row[f"rank{rank}_success"] = success

            for k in k_values:
                rank_limit = min(k, len(ranked))
                oracle = False
                best_rank = ""
                best_re = ""
                best_te = ""
                for rank, (_, _, trans) in enumerate(ranked[:rank_limit], start=1):
                    re, te = pose_error_cm_deg(trans, gt_trans)
                    if re < config.re_thre and te < config.te_thre:
                        oracle = True
                        best_rank = rank
                        best_re = re
                        best_te = te
                        break
                row[f"oracle_success@{k}"] = int(oracle)
                row[f"oracle_best_rank@{k}"] = best_rank
                row[f"oracle_best_re@{k}"] = best_re
                row[f"oracle_best_te@{k}"] = best_te
                counters[k] += int(oracle)

            rows.append(row)
            if (pair_idx + 1) % args.progress_interval == 0:
                print(f"processed {pair_idx + 1}/{num_pairs}")

    csv_path = output_dir / "topk_oracle_pairs.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "num_pairs": num_pairs,
        "k_values": k_values,
        "rr_threshold": {"re_deg": config.re_thre, "te_cm": config.te_thre},
        "candidate_count": {
            "mean": float(np.mean(candidate_counts)),
            "min": int(np.min(candidate_counts)),
            "max": int(np.max(candidate_counts)),
        },
        "topk_oracle_rr": {f"K={k}": float(counters[k] / num_pairs * 100.0) for k in k_values},
        "topk_oracle_success_count": {f"K={k}": int(counters[k]) for k in k_values},
        "note": "Top-K candidates are generated from C_R1, refined on C_R1, ranked by C_R1 inlier count. K=1 is the selected T_best under this ranking.",
        "config": args.config_path,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"pairs: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json/config_3DLoMatch_FPFH_redkitchen_modified.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/redkitchen_topk_oracle"))
    parser.add_argument("--k-values", default="1,3,5,10")
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
