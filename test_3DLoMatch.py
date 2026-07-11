import argparse
import csv
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from functools import lru_cache

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
from memory_graph_config import MemoryGraphConfig
from memory_guided_registration import MemoryGuidedRegistration
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
        "raw_success": int(raw_re < rotation_threshold and raw_te < translation_threshold),
        "local_refined_success": int(refined_re < rotation_threshold and refined_te < translation_threshold),
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


@lru_cache(maxsize=1)
def source_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def r1_cache_metadata(pair, config):
    settings = {
        "pair_id": pair.pair_id,
        "descriptor": str(config.descriptor),
        "seed": int(getattr(config, "seed", 51)),
        "r1_knn": int(config.r1_knn),
        "r1_sampling": int(config.r1_sampling),
        "inlier_threshold": float(config.inlier_threshold),
        "commit": source_revision(),
    }
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()
    return {**settings, "fingerprint": fingerprint}


def load_or_run_r1(pair, pair_index, matcher, regenerator, config):
    cache_dir = getattr(config, "r1_cache_dir", "")
    if cache_dir:
        path = r1_cache_path(cache_dir, pair_index)
        if path.exists():
            if getattr(config, "r1_cache_rng_warmup", False) and pair_index >= int(getattr(config, "r1_cache_rng_warmup_start_index", 1)):
                run_r1(pair, matcher, regenerator, config)
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
            "metadata": r1_cache_metadata(pair, config),
        }, path)
    return r1


def load_required_r1_cache(pair, pair_index, config):
    cache_dir = getattr(config, "r1_cache_dir", "")
    if not cache_dir:
        raise ValueError("memory_graph requires a non-empty r1_cache_dir with fixed R1 outputs.")
    path = r1_cache_path(cache_dir, pair_index)
    if not path.exists():
        raise FileNotFoundError(f"memory_graph requires fixed R1 cache entry: {path}")
    payload = torch.load(path, map_location=pair.src_keypoints.device, weights_only=True)
    if not {"pose", "src_corr", "tgt_corr", "metadata"}.issubset(payload):
        raise KeyError(f"R1 cache entry {path} must contain pose, src_corr, tgt_corr, and metadata.")
    if payload["metadata"] != r1_cache_metadata(pair, config):
        raise ValueError(f"R1 cache metadata mismatch at {path}; regenerate the fixed cache for this pair/config.")
    pose = payload["pose"].to(device=pair.src_keypoints.device, dtype=pair.src_keypoints.dtype)
    if pose.shape != (1, 4, 4):
        raise ValueError(f"R1 cache pose at {path} must have shape (1, 4, 4), got {tuple(pose.shape)}.")
    return {
        "pose": pose,
        "src_corr": payload["src_corr"].to(device=pair.src_keypoints.device, dtype=pair.src_keypoints.dtype),
        "tgt_corr": payload["tgt_corr"].to(device=pair.tgt_keypoints.device, dtype=pair.tgt_keypoints.dtype),
    }


def audit_archive_prefix(archive, audits, round_id, rotation_threshold, translation_threshold, re_key, te_key, success_key):
    eligible = [(hypothesis, audit) for hypothesis, audit in zip(archive.hypotheses, audits) if hypothesis.round_id <= round_id]
    if not eligible:
        raise ValueError(f"No hypotheses available at round {round_id}.")
    oracle_hypothesis, oracle_audit = min(
        eligible,
        key=lambda item: (
            max(item[1][re_key] / rotation_threshold, item[1][te_key] / translation_threshold),
            item[1][re_key],
            item[1][te_key],
        ),
    )
    success_rounds = [hypothesis.round_id for hypothesis, audit in eligible if audit[success_key]]
    return {
        "round_oracle_success": int(any(audit[success_key] and hypothesis.round_id == round_id for hypothesis, audit in eligible)),
        "cumulative_oracle_success": int(any(audit[success_key] for _, audit in eligible)),
        "oracle_best_re": oracle_audit[re_key],
        "oracle_best_te": oracle_audit[te_key],
        "first_success_round": min(success_rounds) if success_rounds else -1,
    }


def audit_round(raw_archive, raw_audits, post_refinement_archive, post_refinement_audits, round_id, incumbent_id, rotation_threshold, translation_threshold):
    raw = audit_archive_prefix(
        raw_archive, raw_audits, round_id, rotation_threshold, translation_threshold,
        "raw_re", "raw_te", "raw_success",
    )
    post_refinement = audit_archive_prefix(
        post_refinement_archive, post_refinement_audits, round_id, rotation_threshold, translation_threshold,
        "local_refined_re", "local_refined_te", "local_refined_success",
    )
    eligible = [(hypothesis, audit) for hypothesis, audit in zip(post_refinement_archive.hypotheses, post_refinement_audits) if hypothesis.round_id <= round_id]
    selected = next((audit for hypothesis, audit in eligible if hypothesis.hypothesis_id == incumbent_id), None)
    if selected is None:
        raise ValueError(f"Round {round_id} incumbent {incumbent_id} is absent from its archive prefix.")
    if post_refinement["cumulative_oracle_success"] < raw["cumulative_oracle_success"]:
        raise RuntimeError("Post-refinement archive must retain every raw candidate.")
    return {
        "round_raw_oracle_success": raw["round_oracle_success"],
        "cumulative_raw_oracle_success": raw["cumulative_oracle_success"],
        "raw_oracle_best_re": raw["oracle_best_re"],
        "raw_oracle_best_te": raw["oracle_best_te"],
        "first_raw_success_round": raw["first_success_round"],
        "round_post_refinement_oracle_success": post_refinement["round_oracle_success"],
        "cumulative_post_refinement_oracle_success": post_refinement["cumulative_oracle_success"],
        "post_refinement_oracle_best_re": post_refinement["oracle_best_re"],
        "post_refinement_oracle_best_te": post_refinement["oracle_best_te"],
        "first_post_refinement_success_round": post_refinement["first_success_round"],
        "selected_re": selected["local_refined_re"],
        "selected_te": selected["local_refined_te"],
        "selected_success": selected["local_refined_success"],
    }


def build_search(config, local_regenerator):
    ray_config = IterativeRayConfig.from_mapping(config.get("iterative_ray", config)).validate()
    regenerator = RayGuidedRegenerator(
        descriptor_topk=ray_config.descriptor_topk,
        local_corr_max_points=ray_config.local_corr_max_points,
        local_knn_radius=ray_config.local_knn_radius,
        local_mutual_k=ray_config.local_mutual_k,
        refine_corr_radius=ray_config.refine_corr_radius,
        escape_lambda=ray_config.escape_lambda,
        independent_explore_fraction=ray_config.independent_explore_fraction,
        local_regenerator=local_regenerator,
        seed_group_count=ray_config.seed_group_count,
    )
    return ray_config, IterativeRaySearch(
        ray_config,
        RaySelector(ray_config.active_rays_per_round, ray_config.ray_variance_min_observers),
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


def build_loader(config, search_config):
    if isinstance(search_config, MemoryGraphConfig):
        return ThreeDLoMatchLoader(
            root=config.data_path,
            descriptor=config.descriptor,
            inlier_threshold=config.inlier_threshold,
            num_node=config.num_node,
            use_mutual=config.use_mutual,
            overlap_pred_root=getattr(config, "overlap_pred_root", ""),
            use_overlap_proxy=getattr(config, "use_overlap_proxy", False),
            ray_manifest="",
        )
    ray_config = search_config
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


def memory_candidate_fmr(pair, result, threshold):
    if result.candidate_src_indices.numel() == 0:
        return 0.0, 0, 0
    source = pair.src_keypoints[0, result.candidate_src_indices]
    target = pair.tgt_keypoints[0, result.candidate_tgt_indices]
    matrix = pair.gt_transform[0]
    source = source @ matrix[:3, :3].transpose(0, 1) + matrix[:3, 3]
    inlier_count = int((torch.linalg.norm(source - target, dim=1) < threshold).sum().item())
    ratio = inlier_count / result.candidate_src_indices.numel()
    return ratio, int(ratio >= 0.05), inlier_count


def memory_prefix_audit(hypotheses, audits, round_id, rotation_threshold, translation_threshold, re_key, te_key, success_key):
    eligible = [(hypothesis, audit) for hypothesis, audit in zip(hypotheses, audits) if hypothesis.round_id <= round_id]
    if not eligible:
        return None
    oracle_hypothesis, oracle = min(
        eligible,
        key=lambda item: (
            max(item[1][re_key] / rotation_threshold, item[1][te_key] / translation_threshold),
            item[1][re_key],
            item[1][te_key],
        ),
    )
    selected_hypothesis, selected = max(eligible, key=lambda item: item[0].score)
    success_rounds = [hypothesis.round_id for hypothesis, audit in eligible if audit[success_key]]
    return {
        "oracle_hypothesis_id": oracle_hypothesis.hypothesis_id,
        "round_oracle_success": int(any(audit[success_key] and hypothesis.round_id == round_id for hypothesis, audit in eligible)),
        "cumulative_oracle_success": int(any(audit[success_key] for _, audit in eligible)),
        "oracle_best_re": oracle[re_key],
        "oracle_best_te": oracle[te_key],
        "first_success_round": min(success_rounds) if success_rounds else -1,
        "selected_hypothesis_id": selected_hypothesis.hypothesis_id,
        "selected_re": selected["local_refined_re"],
        "selected_te": selected["local_refined_te"],
        "selected_success": selected["local_refined_success"],
    }


def memory_round_audit(result, raw_audits, post_refinement_audits, round_id, rotation_threshold, translation_threshold):
    raw = memory_prefix_audit(
        result.raw_hypotheses,
        raw_audits,
        round_id,
        rotation_threshold,
        translation_threshold,
        "raw_re",
        "raw_te",
        "raw_success",
    )
    post_refinement = memory_prefix_audit(
        result.post_refinement_hypotheses,
        post_refinement_audits,
        round_id,
        rotation_threshold,
        translation_threshold,
        "local_refined_re",
        "local_refined_te",
        "local_refined_success",
    )
    if raw is None or post_refinement is None:
        raise ValueError(f"No memory archive candidates available by round {round_id}.")
    if post_refinement["cumulative_oracle_success"] < raw["cumulative_oracle_success"]:
        raise RuntimeError("Post-refinement memory archive must retain every raw parent.")
    return {
        "round_raw_oracle_success": raw["round_oracle_success"],
        "cumulative_raw_oracle_success": raw["cumulative_oracle_success"],
        "raw_oracle_best_re": raw["oracle_best_re"],
        "raw_oracle_best_te": raw["oracle_best_te"],
        "first_raw_success_round": raw["first_success_round"],
        "round_post_refinement_oracle_success": post_refinement["round_oracle_success"],
        "cumulative_post_refinement_oracle_success": post_refinement["cumulative_oracle_success"],
        "post_refinement_oracle_best_re": post_refinement["oracle_best_re"],
        "post_refinement_oracle_best_te": post_refinement["oracle_best_te"],
        "first_post_refinement_success_round": post_refinement["first_success_round"],
        "selected_hypothesis_id": post_refinement["selected_hypothesis_id"],
        "selected_re": post_refinement["selected_re"],
        "selected_te": post_refinement["selected_te"],
        "selected_success": post_refinement["selected_success"],
    }


def run_memory_graph_experiment(config, memory_config):
    loader = build_loader(config, memory_config)
    registrar = MemoryGuidedRegistration(memory_config)
    pair_limit = min(len(loader), int(getattr(config, "max_pairs", 0) or len(loader)))
    output_dir = Path(getattr(config, "memory_output_dir", "outputs/memory_graph"))
    output_dir.mkdir(parents=True, exist_ok=True)
    round_rows, candidate_rows, pair_rows = [], [], []
    logging.info("Historical correspondence memory configuration: %s", json.dumps(memory_config.report(), sort_keys=True))
    with torch.no_grad():
        for index in tqdm(range(pair_limit)):
            load_started = time.perf_counter()
            pair = loader.get_pair(index)
            load_time = time.perf_counter() - load_started
            r1_cache = load_required_r1_cache(pair, index, config)
            model_started = time.perf_counter()
            result = registrar.run(pair.src_keypoints, pair.tgt_keypoints, pair.src_features, pair.tgt_features, initial_pose=r1_cache["pose"], initial_src_corr=r1_cache["src_corr"], initial_tgt_corr=r1_cache["tgt_corr"])
            model_time = time.perf_counter() - model_started
            if result.r1_hypothesis is None:
                raise RuntimeError("memory_graph required R1 cache but no R1 archive parent was created.")
            raw_audits = [audit_hypothesis(item, pair.gt_transform, config.re_thre, config.te_thre) for item in result.raw_hypotheses]
            post_refinement_audits = [audit_hypothesis(item, pair.gt_transform, config.re_thre, config.te_thre) for item in result.post_refinement_hypotheses]
            audits_by_id = {item.hypothesis_id: audit for item, audit in zip(result.post_refinement_hypotheses, post_refinement_audits)}
            for row in result.candidate_logs:
                audit = audits_by_id.get(row["hypothesis_id"], {})
                candidate_rows.append({"pair_id": pair.pair_id, **row, **audit})
            r1_audit = audit_hypothesis(result.r1_hypothesis, pair.gt_transform, config.re_thre, config.te_thre) if result.r1_hypothesis else None
            r1_failure = int(r1_audit is not None and not r1_audit["raw_success"])
            raw_repaired_before, post_refinement_repaired_before = False, False
            for row in result.round_logs:
                audited_round = memory_round_audit(result, raw_audits, post_refinement_audits, row["round_id"], config.re_thre, config.te_thre)
                raw_repaired_now = bool(r1_failure and audited_round["cumulative_raw_oracle_success"] and not raw_repaired_before)
                post_refinement_repaired_now = bool(r1_failure and audited_round["cumulative_post_refinement_oracle_success"] and not post_refinement_repaired_before)
                raw_repaired_before = raw_repaired_before or bool(r1_failure and audited_round["cumulative_raw_oracle_success"])
                post_refinement_repaired_before = post_refinement_repaired_before or bool(r1_failure and audited_round["cumulative_post_refinement_oracle_success"])
                round_rows.append({
                    "pair_id": pair.pair_id,
                    **row,
                    **audited_round,
                    "r1_failure": r1_failure,
                    "r1_failure_raw_oracle_success": int(r1_failure and audited_round["cumulative_raw_oracle_success"]),
                    "r1_failure_post_refinement_oracle_success": int(r1_failure and audited_round["cumulative_post_refinement_oracle_success"]),
                    "newly_raw_repaired_failure": int(raw_repaired_now),
                    "newly_post_refinement_repaired_failure": int(post_refinement_repaired_now),
                })
            if result.best is None:
                selected_re, selected_te = pose_errors(result.pose, pair.gt_transform)
                selected = {
                    "local_refined_re": selected_re,
                    "local_refined_te": selected_te,
                    "local_refined_success": int(selected_re < config.re_thre and selected_te < config.te_thre),
                }
            else:
                selected = audit_hypothesis(result.best, pair.gt_transform, config.re_thre, config.te_thre)
            fmr_ratio, fmr, fmr_inlier_count = memory_candidate_fmr(pair, result, float(config.inlier_threshold))
            first_raw_success_round = next((item.round_id for item, audit in zip(result.raw_hypotheses, raw_audits) if audit["raw_success"]), -1)
            first_post_refinement_success_round = next((item.round_id for item, audit in zip(result.post_refinement_hypotheses, post_refinement_audits) if audit["local_refined_success"]), -1)
            pair_rows.append({
                "pair_id": pair.pair_id,
                "r1_success": r1_audit["raw_success"] if r1_audit else 0,
                "r1_failure": r1_failure,
                "raw_oracle_success": int(any(audit["raw_success"] for audit in raw_audits)),
                "post_refinement_oracle_success": int(any(audit["local_refined_success"] for audit in post_refinement_audits)),
                "selected_success": selected["local_refined_success"],
                "selected_re": selected["local_refined_re"],
                "selected_te": selected["local_refined_te"],
                "first_raw_success_round": first_raw_success_round,
                "first_post_refinement_success_round": first_post_refinement_success_round,
                "raw_hypothesis_count": len(result.raw_hypotheses),
                "post_refinement_hypothesis_count": len(result.post_refinement_hypotheses),
                "topk_candidate_fmr": fmr,
                "topk_candidate_inlier_ratio": fmr_ratio,
                "topk_candidate_inlier_count": fmr_inlier_count,
                "load_time": load_time,
                "model_time": model_time,
                **result.memory_summary,
                **result.r1_initialization,
            })
    for filename, rows in (("pair_results.csv", pair_rows), ("round_logs.csv", round_rows), ("candidate_logs.csv", candidate_rows)):
        if rows:
            with open(output_dir / filename, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "method": "memory_graph",
        "pairs": len(pair_rows),
        "raw_oracle_rr": float(np.mean([row["raw_oracle_success"] for row in pair_rows])) if pair_rows else 0.0,
        "post_refinement_oracle_rr": float(np.mean([row["post_refinement_oracle_success"] for row in pair_rows])) if pair_rows else 0.0,
        "selected_rr": float(np.mean([row["selected_success"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_re": float(np.mean([row["selected_re"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_te": float(np.mean([row["selected_te"] for row in pair_rows])) if pair_rows else 0.0,
        "topk_candidate_fmr": float(np.mean([row["topk_candidate_fmr"] for row in pair_rows])) if pair_rows else 0.0,
        "topk_candidate_inlier_ratio": float(np.mean([row["topk_candidate_inlier_ratio"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_topk_candidate_inlier_count": float(np.mean([row["topk_candidate_inlier_count"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_model_time": float(np.mean([row["model_time"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_raw_hypothesis_count": float(np.mean([row["raw_hypothesis_count"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_post_refinement_hypothesis_count": float(np.mean([row["post_refinement_hypothesis_count"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_duplicate_basin_rate": float(np.mean([row["duplicate_basin_rate"] for row in round_rows if row["duplicate_basin_rate"] is not None])) if any(row["duplicate_basin_rate"] is not None for row in round_rows) else None,
        "r1_failure_count": int(sum(row["r1_failure"] for row in pair_rows)),
        "r1_failure_raw_oracle_rr": float(np.mean([row["raw_oracle_success"] for row in pair_rows if row["r1_failure"]])) if any(row["r1_failure"] for row in pair_rows) else 0.0,
        "r1_failure_post_refinement_oracle_rr": float(np.mean([row["post_refinement_oracle_success"] for row in pair_rows if row["r1_failure"]])) if any(row["r1_failure"] for row in pair_rows) else 0.0,
        "selected_rr_on_post_refinement_repairable": float(np.mean([row["selected_success"] for row in pair_rows if row["r1_failure"] and row["post_refinement_oracle_success"]])) if any(row["r1_failure"] and row["post_refinement_oracle_success"] for row in pair_rows) else 0.0,
        "round_metrics": {
            str(round_id): {
                "pairs": len(rows),
                "raw_oracle_rr": float(np.mean([row["cumulative_raw_oracle_success"] for row in rows])),
                "post_refinement_oracle_rr": float(np.mean([row["cumulative_post_refinement_oracle_success"] for row in rows])),
                "selected_rr": float(np.mean([row["selected_success"] for row in rows])),
                "r1_failure_raw_oracle_rr": float(np.mean([row["r1_failure_raw_oracle_success"] for row in rows if row["r1_failure"]])) if any(row["r1_failure"] for row in rows) else 0.0,
                "r1_failure_post_refinement_oracle_rr": float(np.mean([row["r1_failure_post_refinement_oracle_success"] for row in rows if row["r1_failure"]])) if any(row["r1_failure"] for row in rows) else 0.0,
                "newly_raw_repaired_failures": int(sum(row["newly_raw_repaired_failure"] for row in rows)),
                "newly_post_refinement_repaired_failures": int(sum(row["newly_post_refinement_repaired_failure"] for row in rows)),
                "raw_oracle_best_re": float(np.mean([row["raw_oracle_best_re"] for row in rows])),
                "raw_oracle_best_te": float(np.mean([row["raw_oracle_best_te"] for row in rows])),
                "post_refinement_oracle_best_re": float(np.mean([row["post_refinement_oracle_best_re"] for row in rows])),
                "post_refinement_oracle_best_te": float(np.mean([row["post_refinement_oracle_best_te"] for row in rows])),
                "duplicate_basin_rate": float(np.mean([row["duplicate_basin_rate"] for row in rows if row["duplicate_basin_rate"] is not None])) if any(row["duplicate_basin_rate"] is not None for row in rows) else None,
            }
            for round_id in sorted({row["round_id"] for row in round_rows})
            for rows in [[row for row in round_rows if row["round_id"] == round_id]]
        },
        "config": memory_config.report(),
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logging.info("%s", json.dumps(summary, ensure_ascii=False))
    return summary


def run_experiment(config):
    set_experiment_seed(int(getattr(config, "seed", 51)))
    if getattr(config, "_selected_method", "") == "memory_graph":
        memory_config = MemoryGraphConfig.from_mapping(config.memory_graph).validate()
        return run_memory_graph_experiment(config, memory_config)
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
            raw_audited = [audit_hypothesis(item, pair.gt_transform, config.re_thre, config.te_thre) for item in result.raw_archive.hypotheses]
            post_refinement_audited = [audit_hypothesis(item, pair.gt_transform, config.re_thre, config.te_thre) for item in result.post_refinement_archive.hypotheses]
            for row, audit in zip(result.raw_candidate_logs, raw_audited):
                candidate_rows.append({**row, **audit})
            for row, audit in zip(result.candidate_logs, post_refinement_audited):
                candidate_rows.append({**row, **audit})
            refinement_audited = [audit_hypothesis(item, pair.gt_transform, config.re_thre, config.te_thre) for item in result.refinement_candidates]
            for row, audit in zip(result.refinement_logs, refinement_audited):
                candidate_rows.append({**row, **audit})
            selected = audit_hypothesis(result.archive.incumbent, pair.gt_transform, config.re_thre, config.te_thre)
            first_raw_success_round = next((item.round_id for item, audit in zip(result.raw_archive.hypotheses, raw_audited) if audit["raw_success"]), -1)
            first_post_refinement_success_round = next((item.round_id for item, audit in zip(result.post_refinement_archive.hypotheses, post_refinement_audited) if audit["local_refined_success"]), -1)
            r1_audit = audit_hypothesis(r1, pair.gt_transform, config.re_thre, config.te_thre)
            r1_failure = int(not r1_audit["local_refined_success"])
            raw_repaired_before, post_refinement_repaired_before = False, False
            for row in result.round_logs:
                correct_raw_until_round = [
                    hypothesis for hypothesis, audit in zip(result.raw_archive.hypotheses, raw_audited)
                    if 0 < hypothesis.round_id <= row["round_id"] and audit["raw_success"]
                ]
                refinement_pool_recall = float(
                    sum(hypothesis.entered_refinement_pool for hypothesis in correct_raw_until_round) / len(correct_raw_until_round)
                ) if correct_raw_until_round else 0.0
                audited_round = audit_round(
                    result.raw_archive,
                    raw_audited,
                    result.post_refinement_archive,
                    post_refinement_audited,
                    row["round_id"],
                    row["incumbent_id"],
                    config.re_thre,
                    config.te_thre,
                )
                raw_repaired_now = bool(r1_failure and audited_round["cumulative_raw_oracle_success"] and not raw_repaired_before)
                post_refinement_repaired_now = bool(r1_failure and audited_round["cumulative_post_refinement_oracle_success"] and not post_refinement_repaired_before)
                raw_repaired_before = raw_repaired_before or bool(r1_failure and audited_round["cumulative_raw_oracle_success"])
                post_refinement_repaired_before = post_refinement_repaired_before or bool(r1_failure and audited_round["cumulative_post_refinement_oracle_success"])
                round_rows.append({
                    **row,
                    **audited_round,
                    "r1_failure": r1_failure,
                    "r1_failure_raw_oracle_success": int(r1_failure and audited_round["cumulative_raw_oracle_success"]),
                    "r1_failure_post_refinement_oracle_success": int(r1_failure and audited_round["cumulative_post_refinement_oracle_success"]),
                    "newly_raw_repaired_failure": int(raw_repaired_now),
                    "newly_post_refinement_repaired_failure": int(post_refinement_repaired_now),
                    "refinement_pool_recall": refinement_pool_recall,
                    "correct_raw_candidate_count": len(correct_raw_until_round),
                })
            correct_raw = [
                hypothesis for hypothesis, audit in zip(result.raw_archive.hypotheses, raw_audited)
                if hypothesis.round_id > 0 and audit["raw_success"]
            ]
            refinement_pool_recall = float(
                sum(hypothesis.entered_refinement_pool for hypothesis in correct_raw) / len(correct_raw)
            ) if correct_raw else 0.0
            pair_rows.append({
                "pair_id": pair.pair_id,
                "r1_success": r1_audit["local_refined_success"],
                "r1_failure": r1_failure,
                "raw_oracle_success": int(any(item["raw_success"] for item in raw_audited)),
                "post_refinement_oracle_success": int(any(item["local_refined_success"] for item in post_refinement_audited)),
                "selected_success": selected["local_refined_success"],
                "selected_re": selected["local_refined_re"],
                "selected_te": selected["local_refined_te"],
                "first_raw_success_round": first_raw_success_round,
                "first_post_refinement_success_round": first_post_refinement_success_round,
                "raw_candidate_count": len(result.raw_archive.hypotheses),
                "post_refinement_candidate_count": len(result.post_refinement_archive.hypotheses),
                "refinement_pool_recall": refinement_pool_recall,
                "correct_raw_candidate_count": len(correct_raw),
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
        "raw_oracle_rr": float(np.mean([row["raw_oracle_success"] for row in pair_rows])) if pair_rows else 0.0,
        "post_refinement_oracle_rr": float(np.mean([row["post_refinement_oracle_success"] for row in pair_rows])) if pair_rows else 0.0,
        "selected_rr": float(np.mean([row["selected_success"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_re": float(np.mean([row["selected_re"] for row in pair_rows])) if pair_rows else 0.0,
        "mean_te": float(np.mean([row["selected_te"] for row in pair_rows])) if pair_rows else 0.0,
        "r1_failure_count": int(sum(row["r1_failure"] for row in pair_rows)),
        "r1_failure_raw_oracle_rr": float(np.mean([row["raw_oracle_success"] for row in pair_rows if row["r1_failure"]])) if any(row["r1_failure"] for row in pair_rows) else 0.0,
        "r1_failure_post_refinement_oracle_rr": float(np.mean([row["post_refinement_oracle_success"] for row in pair_rows if row["r1_failure"]])) if any(row["r1_failure"] for row in pair_rows) else 0.0,
        "selected_rr_on_post_refinement_repairable": float(np.mean([row["selected_success"] for row in pair_rows if row["r1_failure"] and row["post_refinement_oracle_success"]])) if any(row["r1_failure"] and row["post_refinement_oracle_success"] for row in pair_rows) else 0.0,
        "refinement_pool_recall": float(np.mean([row["refinement_pool_recall"] for row in pair_rows if row["correct_raw_candidate_count"]])) if any(row["correct_raw_candidate_count"] for row in pair_rows) else 0.0,
        "round_metrics": {
            str(round_id): {
                "pairs": len(rows),
                "raw_oracle_rr": float(np.mean([row["cumulative_raw_oracle_success"] for row in rows])),
                "post_refinement_oracle_rr": float(np.mean([row["cumulative_post_refinement_oracle_success"] for row in rows])),
                "selected_rr": float(np.mean([row["selected_success"] for row in rows])),
                "r1_failure_raw_oracle_rr": float(np.mean([row["r1_failure_raw_oracle_success"] for row in rows if row["r1_failure"]])) if any(row["r1_failure"] for row in rows) else 0.0,
                "r1_failure_post_refinement_oracle_rr": float(np.mean([row["r1_failure_post_refinement_oracle_success"] for row in rows if row["r1_failure"]])) if any(row["r1_failure"] for row in rows) else 0.0,
                "newly_raw_repaired_failures": int(sum(row["newly_raw_repaired_failure"] for row in rows)),
                "newly_post_refinement_repaired_failures": int(sum(row["newly_post_refinement_repaired_failure"] for row in rows)),
                "refinement_pool_recall": float(np.mean([row["refinement_pool_recall"] for row in rows if row["correct_raw_candidate_count"]])) if any(row["correct_raw_candidate_count"] for row in rows) else 0.0,
                "raw_oracle_best_re": float(np.mean([row["raw_oracle_best_re"] for row in rows])),
                "raw_oracle_best_te": float(np.mean([row["raw_oracle_best_te"] for row in rows])),
                "post_refinement_oracle_best_re": float(np.mean([row["post_refinement_oracle_best_re"] for row in rows])),
                "post_refinement_oracle_best_te": float(np.mean([row["post_refinement_oracle_best_te"] for row in rows])),
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
    parser.add_argument("--method", choices=["iterative_ray", "r1_only", "repeated_regor", "shuffled_ray", "memory_graph"])
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as handle:
        config = edict(json.load(handle))
    if args.method:
        if args.method == "memory_graph":
            config._selected_method = args.method
        else:
            config.iterative_ray["method"] = args.method
    os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_Devices
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    run_experiment(config)
