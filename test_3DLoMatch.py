import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from Correspondence_regenerate_v2 import Regenerator
from dataset import ThreeDLoMatchLoader
from hypothesis_generation import sample_correspondences
from initial_matching_plus import Matcher_plus
from iterative_ray_config import IterativeRayConfig
from iterative_ray_search import IterativeRaySearch, RaySelector
from ray_constraint_builder import RayConstraintBuilder
from ray_guided_regenerator import PoseHypothesis, RayGuidedRegenerator
from ray_pose_validator import RayPoseValidator


def pose_errors(pose, gt_transform):
    relative = pose[0, :3, :3].transpose(0, 1) @ gt_transform[0, :3, :3]
    rotation = torch.acos(torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi
    translation = torch.linalg.norm(pose[0, :3, 3] - gt_transform[0, :3, 3]) * 100.0
    return float(rotation.item()), float(translation.item())


def audit_hypothesis(hypothesis, gt_transform, rotation_threshold, translation_threshold):
    raw_re, raw_te = pose_errors(hypothesis.pose_raw, gt_transform)
    refined_re, refined_te = pose_errors(hypothesis.pose_local_refined, gt_transform)
    return {
        "raw_re": raw_re,
        "raw_te": raw_te,
        "local_refined_re": refined_re,
        "local_refined_te": refined_te,
        "success": int(refined_re < rotation_threshold and refined_te < translation_threshold),
        "pose_raw": hypothesis.pose_raw[0].detach().cpu().tolist(),
        "pose_local_refined": hypothesis.pose_local_refined[0].detach().cpu().tolist(),
    }


def set_experiment_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def r1_cache_path(cache_dir, pair_index):
    return Path(cache_dir) / f"pair_{pair_index:06d}.pt"


def load_or_run_r1(pair, pair_index, matcher, regenerator, config):
    cache_dir = getattr(config, "r1_cache_dir", "")
    if cache_dir:
        path = r1_cache_path(cache_dir, pair_index)
        if path.exists():
            payload = torch.load(path, map_location=pair.src_keypoints.device, weights_only=True)
            return PoseHypothesis(
                hypothesis_id=-1,
                parent_id=-1,
                round_id=0,
                pose_raw=payload["pose"],
                pose_local_refined=payload["pose"],
                src_corr=payload["src_corr"],
                tgt_corr=payload["tgt_corr"],
                correspondence_scores=torch.ones(payload["src_corr"].shape[1], device=pair.src_keypoints.device, dtype=pair.src_keypoints.dtype),
                seed_ids=torch.empty((0, 2), device=pair.src_keypoints.device, dtype=torch.long),
                generation_mode="r1",
            )
    r1 = run_r1(pair, matcher, regenerator, config)
    if cache_dir:
        path = r1_cache_path(cache_dir, pair_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "pose": r1.pose.detach().cpu(),
            "src_corr": r1.src_corr.detach().cpu(),
            "tgt_corr": r1.tgt_corr.detach().cpu(),
        }, path)
    return r1


def audit_round(archive, audits, round_id, incumbent_id, rotation_threshold, translation_threshold):
    eligible = [(hypothesis, audit) for hypothesis, audit in zip(archive.hypotheses, audits) if hypothesis.round_id <= round_id]
    if not eligible:
        raise ValueError(f"No hypotheses available at round {round_id}.")
    oracle_hypothesis, oracle_audit = min(
        eligible,
        key=lambda item: (
            max(item[1]["local_refined_re"] / rotation_threshold, item[1]["local_refined_te"] / translation_threshold),
            item[1]["local_refined_re"],
            item[1]["local_refined_te"],
        ),
    )
    selected = next((audit for hypothesis, audit in eligible if hypothesis.hypothesis_id == incumbent_id), None)
    if selected is None:
        raise ValueError(f"Round {round_id} incumbent {incumbent_id} is absent from its archive prefix.")
    success_rounds = [hypothesis.round_id for hypothesis, audit in eligible if audit["success"]]
    return {
        "round_oracle_success": int(any(audit["success"] and hypothesis.round_id == round_id for hypothesis, audit in eligible)),
        "cumulative_oracle_success": int(any(audit["success"] for _, audit in eligible)),
        "oracle_best_re": oracle_audit["local_refined_re"],
        "oracle_best_te": oracle_audit["local_refined_te"],
        "selected_re": selected["local_refined_re"],
        "selected_te": selected["local_refined_te"],
        "selected_success": selected["success"],
        "first_success_round": min(success_rounds) if success_rounds else -1,
    }


def build_search(config, local_regenerator):
    ray_config = IterativeRayConfig.from_mapping(config.get("iterative_ray", config)).validate()
    regenerator = RayGuidedRegenerator(
        descriptor_topk=ray_config.descriptor_topk,
        local_corr_max_points=ray_config.local_corr_max_points,
        local_knn_radius=ray_config.local_knn_radius,
        local_mutual_k=ray_config.local_mutual_k,
        escape_lambda=ray_config.escape_lambda,
        history_lambda=ray_config.history_lambda,
        independent_explore_fraction=ray_config.independent_explore_fraction,
        local_regenerator=local_regenerator,
        seed_group_count=ray_config.seed_group_count,
    )
    return ray_config, IterativeRaySearch(
        ray_config,
        RaySelector(ray_config.active_rays_per_round),
        RayConstraintBuilder(
            ray_config.ray_trunc_margin,
            ray_config.ray_surface_sigma,
            ray_config.max_constraints_per_frame,
            ray_config.ray_trunc_margin,
        ),
        regenerator,
        RayPoseValidator(
            ray_config.ray_trunc_margin,
            ray_config.ray_surface_sigma,
            ray_config.validation_surface_weight,
            ray_config.min_valid_ray_count,
        ),
    )


def build_loader(config, ray_config):
    return ThreeDLoMatchLoader(
        root=config.data_path,
        descriptor=config.descriptor,
        inlier_threshold=config.inlier_threshold,
        num_node=config.num_node,
        use_mutual=config.use_mutual,
        rgbd_root=config.rgbd_root,
        overlap_pred_root=getattr(config, "overlap_pred_root", ""),
        use_overlap_proxy=getattr(config, "use_overlap_proxy", False),
        ray_manifest=ray_config.ray_manifest,
        ray_stride=ray_config.ray_stride,
        ray_search_fraction=ray_config.search_frame_fraction,
        ray_min_depth=ray_config.ray_min_depth,
        ray_max_depth=ray_config.ray_max_depth,
        fsv_depth_scale=ray_config.ray_depth_scale,
        fsv_max_frames=ray_config.ray_max_frames,
    )


def run_r1(pair, matcher, regenerator, config):
    output = matcher.estimator(pair.src_keypoints, pair.tgt_keypoints, pair.src_features, pair.tgt_features)
    _, filtered_src, filtered_tgt, src_corr, tgt_corr, _, _ = output
    seed_source = filtered_src if filtered_src.shape[1] >= 3 else src_corr
    seed_target = filtered_tgt if filtered_tgt.shape[1] >= 3 else tgt_corr
    seed_src, seed_tgt = sample_correspondences(seed_source, seed_target, int(config.r1_sampling))
    r1_src, r1_tgt, r1_pose = regenerator.regenerate(
        seed_src, seed_tgt,
        pair.src_keypoints, pair.tgt_keypoints,
        pair.src_features, pair.tgt_features,
        knn_num=int(config.r1_knn), sampling_num=int(config.r1_sampling),
    )
    return PoseHypothesis(
        hypothesis_id=-1,
        parent_id=-1,
        round_id=0,
        pose_raw=r1_pose,
        pose_local_refined=r1_pose,
        src_corr=r1_src,
        tgt_corr=r1_tgt,
        correspondence_scores=torch.ones(r1_src.shape[1], device=r1_src.device, dtype=r1_src.dtype),
        seed_ids=torch.empty((0, 2), device=r1_src.device, dtype=torch.long),
        generation_mode="r1",
    )


def run_experiment(config):
    set_experiment_seed(int(getattr(config, "seed", 51)))
    local_regenerator = Regenerator(
        inlier_threshold=config.inlier_threshold,
        num_node="all",
        use_mutual=config.use_mutual,
        d_thre=config.d_thre,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        nms_radius=config.nms_radius,
        max_points=config.max_points,
        k1=config.k1,
        k2=config.k2,
    )
    r1_regenerator = Regenerator(
        inlier_threshold=config.inlier_threshold,
        num_node="all",
        use_mutual=config.use_mutual,
        d_thre=config.d_thre,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        nms_radius=config.nms_radius,
        max_points=config.max_points,
        k1=config.k1,
        k2=config.k2,
    )
    matcher = Matcher_plus(
        inlier_threshold=config.inlier_threshold,
        num_node="all",
        use_mutual=config.use_mutual,
        d_thre=config.d_thre,
        num_iterations=config.num_iterations,
        ratio=config.ratio,
        nms_radius=config.nms_radius,
        max_points=config.max_points,
        k1=config.k1,
        k2=config.k2,
        FS_TCD_thre=config.FS_TCD_thre,
        relax_match_num=config.relax_match_num,
        NS_by_IC=config.NS_by_IC,
    )
    ray_config, search = build_search(config, local_regenerator)
    loader = build_loader(config, ray_config)
    pair_limit = min(len(loader), int(getattr(config, "max_pairs", 0) or len(loader)))
    if ray_config.method == "shuffled_ray" and pair_limit < 2:
        raise ValueError("shuffled_ray requires at least two pairs to provide a different pair's ray evidence.")
    output_dir = Path(getattr(config, "output_dir", "outputs/iterative_ray"))
    output_dir.mkdir(parents=True, exist_ok=True)
    round_rows, candidate_rows, pair_rows = [], [], []
    logging.info("Iterative ray configuration: %s", json.dumps(ray_config.report(), sort_keys=True))
    with torch.no_grad():
        for index in tqdm(range(pair_limit)):
            load_started = time.perf_counter()
            pair = loader.get_pair(index)
            ray_load_time = time.perf_counter() - load_started
            r1_started = time.perf_counter()
            r1 = load_or_run_r1(pair, index, matcher, r1_regenerator, config)
            r1_time = time.perf_counter() - r1_started
            inference_inputs = pair.inference_inputs()
            if ray_config.method == "shuffled_ray":
                shuffled_pair = loader.get_pair((index + 1) % pair_limit)
                inference_inputs.update({
                    "shuffled_source_rays": shuffled_pair.src_ray_bundle,
                    "shuffled_target_rays": shuffled_pair.tgt_ray_bundle,
                })
            result = search.run(r1.pose, inference_inputs, initial_hypothesis=r1)
            audited = [audit_hypothesis(item, pair.gt_transform, config.re_thre, config.te_thre) for item in result.archive.hypotheses]
            for row, audit in zip(result.candidate_logs, audited):
                candidate_rows.append({**row, **audit})
            selected = audit_hypothesis(result.archive.incumbent, pair.gt_transform, config.re_thre, config.te_thre)
            first_success_round = next((item.round_id for item, audit in zip(result.archive.hypotheses, audited) if audit["success"]), -1)
            r1_audit = audit_hypothesis(r1, pair.gt_transform, config.re_thre, config.te_thre)
            r1_failure = int(not r1_audit["success"])
            repaired_before = False
            for row in result.round_logs:
                audited_round = audit_round(
                    result.archive,
                    audited,
                    row["round_id"],
                    row["incumbent_id"],
                    config.re_thre,
                    config.te_thre,
                )
                repaired_now = bool(r1_failure and audited_round["cumulative_oracle_success"] and not repaired_before)
                repaired_before = repaired_before or bool(r1_failure and audited_round["cumulative_oracle_success"])
                round_rows.append({
                    **row,
                    **audited_round,
                    "r1_failure": r1_failure,
                    "r1_failure_oracle_success": int(r1_failure and audited_round["cumulative_oracle_success"]),
                    "newly_repaired_failure": int(repaired_now),
                })
            pair_rows.append({
                "pair_id": pair.pair_id,
                "r1_success": r1_audit["success"],
                "r1_failure": r1_failure,
                "oracle_success": int(any(item["success"] for item in audited)),
                "selected_success": selected["success"],
                "selected_re": selected["local_refined_re"],
                "selected_te": selected["local_refined_te"],
                "first_success_round": first_success_round,
                "candidate_count": len(result.archive.hypotheses),
                "r1_time": r1_time,
                "ray_load_time": ray_load_time,
                **result.timings,
            })
    for filename, rows in (("pair_results.csv", pair_rows), ("round_logs.csv", round_rows), ("candidate_logs.csv", candidate_rows)):
        if rows:
            with open(output_dir / filename, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "pairs": len(pair_rows),
        "oracle_rr": float(np.mean([row["oracle_success"] for row in pair_rows])) if pair_rows else 0.0,
        "selected_rr": float(np.mean([row["selected_success"] for row in pair_rows])) if pair_rows else 0.0,
        "all_pair_oracle_rr": float(np.mean([row["oracle_success"] for row in pair_rows])) if pair_rows else 0.0,
        "all_pair_selected_rr": float(np.mean([row["selected_success"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_re": float(np.mean([row["selected_re"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_te": float(np.mean([row["selected_te"] for row in pair_rows])) if pair_rows else 0.0,
        "r1_failure_count": int(sum(row["r1_failure"] for row in pair_rows)),
        "r1_failure_oracle_rr": float(np.mean([row["oracle_success"] for row in pair_rows if row["r1_failure"]])) if any(row["r1_failure"] for row in pair_rows) else 0.0,
        "selected_rr_on_repairable": float(np.mean([row["selected_success"] for row in pair_rows if row["r1_failure"] and row["oracle_success"]])) if any(row["r1_failure"] and row["oracle_success"] for row in pair_rows) else 0.0,
        "round_metrics": {
            str(round_id): {
                "pairs": len(rows),
                "all_oracle_rr": float(np.mean([row["cumulative_oracle_success"] for row in rows])),
                "r1_failure_oracle_rr": float(np.mean([row["r1_failure_oracle_success"] for row in rows if row["r1_failure"]])) if any(row["r1_failure"] for row in rows) else 0.0,
                "newly_repaired_failures": int(sum(row["newly_repaired_failure"] for row in rows)),
                "oracle_best_re": float(np.mean([row["oracle_best_re"] for row in rows])),
                "oracle_best_te": float(np.mean([row["oracle_best_te"] for row in rows])),
            }
            for round_id in sorted({row["round_id"] for row in round_rows})
            for rows in [[row for row in round_rows if row["round_id"] == round_id]]
        },
        "config": ray_config.report(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logging.info("%s", json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = edict(json.load(handle))
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_Devices
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    run_experiment(config)
