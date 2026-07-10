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

from Correspondence_regenerate_v2 import Regenerator
from dataset import ThreeDLoMatchLoader
from initial_matching_plus import Matcher_plus
from iterative_ray_search import ConstraintBuilder, IterativeRaySearch, RaySelector
from ray_evidence import build_ray_bundle
from ray_guided_regenerator import RayGuidedRegenerator
from test_3DLoMatch import sample_correspondences
from utils.SE3 import transform


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return edict(json.load(f))


def pose_errors(pose, gt):
    relative = pose[0, :3, :3].T @ gt[0, :3, :3]
    re = torch.acos(torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi
    te = torch.linalg.norm(pose[0, :3, 3] - gt[0, :3, 3]) * 100.0
    return float(re.item()), float(te.item())


def success(pose, gt):
    re, te = pose_errors(pose, gt)
    return int(re < 15.0 and te < 30.0), re, te


def fragment_info(loader, index):
    src_path = loader.infos["src"][index]
    tgt_path = loader.infos["tgt"][index]
    scene = src_path.split("/")[1]
    src_id = src_path.split("/")[-1].split("_")[-1].replace(".pth", "")
    tgt_id = tgt_path.split("/")[-1].split("_")[-1].replace(".pth", "")
    return scene, f"cloud_bin_{src_id}", f"cloud_bin_{tgt_id}"


def evaluate(args):
    config = load_config(args.config_path)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_Devices
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
        use_rgbd_fsv=True,
    )
    matcher = Matcher_plus(inlier_threshold=0.1, num_node="all", use_mutual=False, d_thre=0.1, num_iterations=10, ratio=0.2, nms_radius=0.1, max_points=8000, k1=60, k2=50, FS_TCD_thre=0.05, relax_match_num=100, NS_by_IC=50)
    r1_regenerator = Regenerator(inlier_threshold=0.1, num_node="all", use_mutual=False, d_thre=0.1, max_points=8000, k1=60, k2=50)
    local_regenerator = Regenerator(inlier_threshold=0.1, num_node="all", use_mutual=False, d_thre=0.1, max_points=8000, k1=60, k2=50)
    ray_regenerator = RayGuidedRegenerator(top_l=config.ray_top_l, local_regenerator=local_regenerator)
    search = IterativeRaySearch(
        RaySelector(config.ray_active_count),
        ConstraintBuilder(config.ray_surface_mu, config.ray_surface_sigma),
        ray_regenerator,
        config.ray_max_rounds,
        config.ray_candidates_per_round,
        config.ray_pose_nms_threshold,
        config.ray_time_budget,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows, round_rows, candidate_rows = [], [], []
    ray_cache = {}
    max_pairs = min(len(loader), int(config.max_pairs or len(loader)))
    if args.max_pairs:
        max_pairs = min(max_pairs, args.max_pairs)
    with torch.no_grad():
        for pair_id in range(max_pairs):
            data = loader.get_data(pair_id)
            src_points, tgt_points, src_features, tgt_features, gt_trans = data[:5]
            _, filtered_src, filtered_tgt, src_corr, tgt_corr, _, _ = matcher.estimator(src_points, tgt_points, src_features, tgt_features, None)
            seed_src, seed_tgt = sample_correspondences(filtered_src if filtered_src.shape[1] else src_corr, filtered_tgt if filtered_tgt.shape[1] else tgt_corr, 100)
            r1_src, r1_tgt, r1_pose = r1_regenerator.regenerate(seed_src, seed_tgt, src_points, tgt_points, src_features, tgt_features, None, knn_num=100, sampling_num=100)
            scene, src_fragment, tgt_fragment = fragment_info(loader, pair_id)
            for fragment in (src_fragment, tgt_fragment):
                key = (scene, fragment)
                if key not in ray_cache:
                    ray_cache[key] = build_ray_bundle(config.raw_rgbd_root, scene, fragment, config.ray_manifest, config.ray_stride, config.ray_max_frames, config.ray_search_fraction, config.ray_depth_scale, config.ray_min_depth, config.ray_max_depth, "cpu")
            source_rays = ray_cache[(scene, src_fragment)].to(src_points.device)
            target_rays = ray_cache[(scene, tgt_fragment)].to(src_points.device)
            best_pose, archive, logs = search.run(
                r1_pose,
                {"src_points": src_points, "tgt_points": tgt_points, "src_features": src_features, "tgt_features": tgt_features, "source_rays": source_rays, "target_rays": target_rays},
            )
            for log in logs:
                log.update({"pair_id": pair_id, "R1_success": success(r1_pose, gt_trans)[0]})
                round_rows.append(log)
            for candidate in archive.hypotheses:
                gt_success, re, te = success(candidate.pose, gt_trans)
                candidate_rows.append({"pair_id": pair_id, "candidate_id": candidate.candidate_id, "round_id": candidate.round_id, "seed_ids": json.dumps(candidate.seed_ids), "correspondence_count": candidate.correspondence_count, "descriptor_score": candidate.descriptor_score, "predicted_escape_score": candidate.predicted_escape_score, "actual_search_energy": candidate.actual_search_energy, "validation_energy": candidate.validation_energy, "surface_support": candidate.surface_support, "gt_re": re, "gt_te": te, "gt_success": gt_success, "pose": json.dumps(candidate.pose[0].detach().cpu().tolist())})
            oracle = int(any(success(candidate.pose, gt_trans)[0] for candidate in archive.hypotheses))
            selected_success, selected_re, selected_te = success(best_pose, gt_trans)
            pair_rows.append({"pair_id": pair_id, "scene": scene, "src_fragment": src_fragment, "tgt_fragment": tgt_fragment, "R1_success": success(r1_pose, gt_trans)[0], "oracle_success": oracle, "selected_success": selected_success, "selected_re": selected_re, "selected_te": selected_te, "round_count": len(logs), "candidate_count": len(archive.hypotheses)})
            print(f"processed {pair_id + 1}/{max_pairs} pair_id={pair_id} oracle={oracle} selected={selected_success}", flush=True)
    for filename, rows in (("iris_pair_results.csv", pair_rows), ("iris_round_logs.csv", round_rows), ("iris_candidate_logs.csv", candidate_rows)):
        if not rows:
            continue
        with open(output_dir / filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"pairs": len(pair_rows), "oracle_rr": float(np.mean([r["oracle_success"] for r in pair_rows])) if pair_rows else 0.0, "selected_rr": float(np.mean([r["selected_success"] for r in pair_rows])) if pair_rows else 0.0, "candidate_count_mean": float(np.mean([r["candidate_count"] for r in pair_rows])) if pair_rows else 0.0, "config": args.config_path}
    with open(output_dir / "iris_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json/config_iris_reg_redkitchen.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/iris_reg_redkitchen"))
    parser.add_argument("--max-pairs", type=int, default=0)
    evaluate(parser.parse_args())
