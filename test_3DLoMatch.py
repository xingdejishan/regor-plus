import json
import sys
sys.path.append('.')
import argparse
import logging
import os
import copy
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
from evaluate_metric import TransformationLoss, ClassificationLoss
from dataset import ThreeDLoMatchLoader
from benchmark_utils import set_seed, icp_refine
from benchmark_utils_predator import *
from utils.timer import Timer
from initial_matching_plus import Matcher_plus
from Correspondence_regenerate_v2 import Regenerator
from Tranformation_estimaton import Estimator
from common import rigid_transform_3d
from free_space import compute_rgbd_fsv
set_seed()
from utils.SE3 import *
from collections import defaultdict


def cfg_get(config, name, default):
    return getattr(config, name, default)


ROUND2_LABELS = {
    'none': 'Round1 only',
    'original': 'Round1 + original Round2',
    'random_uniform': 'Round1 + random/uniform Round2',
    'weakness_guided': 'Round1 + weakness-guided Round2',
}


ACTIVE_REQUIRED_KEYS = (
    'use_ctc',
    'use_early_stop',
    'use_round2',
    'use_o_prior',
    'use_o_reset',
    'use_target_ut',
    'use_adaptive_leverage',
    'use_density_filter',
    'use_robust_final',
    'use_fsv_proxy',
    'use_rgbd_fsv',
    'use_overlap_proxy',
    'use_global_reset_fallback',
    'use_mini_ransac_final',
    'run_round2_ablation_suite',
    'round2_mode',
    'free_space_root',
    'rgbd_root',
    'overlap_pred_root',
    'active_round1_knn',
    'active_round1_sampling',
    'active_round2_knn',
    'active_round2_sampling',
    'active_tau_fsv',
    'active_tau_rho',
    'active_alpha',
    'active_prior_lambda',
    'active_eta_l',
    'active_reset_gamma',
    'active_reset_topk',
    'active_reset_tau_fsv',
    'active_topk_max_corr',
    'active_topk_local_corr',
    'active_overlap_threshold',
    'active_fsv_distance',
    'active_voxel_size',
    'active_broad_radius',
    'active_support_radius',
    'active_final_voxel_size',
    'active_density_min',
    'active_balance_max',
    'active_robust_c',
    'active_ransac_iters',
    'active_max_rounds',
    'active_fsv_voxel_size',
    'active_fsv_depth_scale',
    'active_fsv_trunc_margin',
    'active_fsv_stride',
    'active_fsv_max_frames',
    'active_round2_target_pool',
    'active_feature_chunk_size',
    'active_eps',
)


def active_param(config, name):
    if not hasattr(config, name):
        raise KeyError(f"Missing required active REGOR-S2 config key: {name}")
    return getattr(config, name)


def validate_active_config(config):
    missing = [key for key in ACTIVE_REQUIRED_KEYS if not hasattr(config, key)]
    if missing:
        raise KeyError(f"Missing active REGOR-S2 config keys: {', '.join(missing)}")
    if config.use_rgbd_fsv and config.use_fsv_proxy:
        raise ValueError("use_rgbd_fsv and use_fsv_proxy are mutually exclusive.")
    if not config.use_fsv_proxy and not config.use_rgbd_fsv:
        raise ValueError("Either use_fsv_proxy or use_rgbd_fsv must be enabled for diagnosis-guided early stop/reset.")
    if config.use_rgbd_fsv and not config.free_space_root and not config.rgbd_root:
        raise ValueError("use_rgbd_fsv=True requires free_space_root or rgbd_root.")
    if config.round2_mode not in ROUND2_LABELS:
        raise ValueError("round2_mode must be one of: none, original, random_uniform, weakness_guided")
    if config.round2_mode == 'none' and config.use_round2:
        raise ValueError("round2_mode='none' requires use_round2=false.")
    if config.use_round2 and config.active_max_rounds < 2:
        raise ValueError("use_round2=True requires active_max_rounds >= 2.")
    if config.descriptor != 'predator' and not config.use_overlap_proxy and not config.overlap_pred_root:
        raise ValueError("Non-Predator descriptors require overlap_pred_root or explicit use_overlap_proxy=true.")
    if config.active_max_rounds < 1:
        raise ValueError("active_max_rounds must be >= 1.")


def min_dist_to_points(query, reference, chunk_size=2048):
    mins = []
    for start in range(0, query.shape[0], chunk_size):
        end = min(start + chunk_size, query.shape[0])
        mins.append(torch.cdist(query[start:end], reference).min(dim=1)[0])
    return torch.cat(mins, dim=0)


def transform_points(points, trans):
    return transform(points[None], trans)[0]


def voxel_coverage(points, mask, voxel_size):
    if points.shape[0] == 0:
        return 0.0
    all_voxels = torch.unique(torch.floor(points / voxel_size).to(torch.int64), dim=0)
    if mask.sum() == 0 or all_voxels.shape[0] == 0:
        return 0.0
    selected_voxels = torch.unique(torch.floor(points[mask] / voxel_size).to(torch.int64), dim=0)
    return float(selected_voxels.shape[0] / max(all_voxels.shape[0], 1))


def estimate_fsv_proxy(src_points, tgt_points, trans, threshold):
    warped_src = transform_points(src_points, trans)
    nn_dist = min_dist_to_points(warped_src, tgt_points)
    bbox_min = tgt_points.min(dim=0)[0] - threshold
    bbox_max = tgt_points.max(dim=0)[0] + threshold
    inside = ((warped_src >= bbox_min) & (warped_src <= bbox_max)).all(dim=1)
    if inside.sum() == 0:
        return float(torch.clamp(nn_dist.mean() / (threshold * 4.0 + 1e-6), 0.0, 1.0))
    violation = inside & (nn_dist > threshold)
    return float(violation.float().mean())


def support_density(points, support_points, radius):
    if support_points.shape[0] == 0:
        return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
    density = (torch.cdist(points, support_points) < radius).float().sum(dim=1)
    scale = density[density > 0].median() if torch.any(density > 0) else torch.tensor(1.0, device=points.device)
    return density / (scale + 1e-6)


def noisy_or(a, b):
    return 1.0 - (1.0 - a) * (1.0 - b)


def build_guided_prior(src_points, tgt_points, src_overlap, tgt_overlap, r1_src, r1_tgt, r1_trans, config, topk_trans=None, target_free_space=None):
    threshold = active_param(config, 'active_overlap_threshold')
    broad_radius = active_param(config, 'active_broad_radius')
    support_radius = active_param(config, 'active_support_radius')
    alpha = active_param(config, 'active_alpha')
    gamma = active_param(config, 'active_reset_gamma')
    eps = active_param(config, 'active_eps')

    warped_src = transform_points(src_points, r1_trans)
    nn_dist = min_dist_to_points(warped_src, tgt_points)
    o_best = (nn_dist < threshold).float()

    if torch.any(o_best > 0):
        broad_dist = min_dist_to_points(src_points, src_points[o_best.bool()])
        o_broad = (broad_dist < broad_radius).float()
    else:
        o_broad = torch.zeros_like(o_best)

    if active_param(config, 'use_rgbd_fsv'):
        fsv_value = compute_rgbd_fsv(src_points[None], r1_trans, target_free_space)
        fsv_metric_name = 'rgbd_fsv'
    else:
        fsv_value = estimate_fsv_proxy(src_points, tgt_points, r1_trans, active_param(config, 'active_fsv_distance'))
        fsv_metric_name = 'fsv_proxy'
    v = float(np.exp(-alpha * fsv_value))
    aoc = voxel_coverage(src_points, o_best.bool(), active_param(config, 'active_voxel_size'))
    overlap_pred = float(torch.clamp(src_overlap.mean(), min=eps, max=1.0))
    roc_bar = min(1.0, aoc / (overlap_pred + eps))

    global_matchable = src_overlap.float().flatten()
    global_matchable = global_matchable / (global_matchable.max() + eps)
    global_matchable = global_matchable.clamp(eps, 1.0)
    o_reset = torch.zeros_like(global_matchable)
    if active_param(config, 'use_global_reset_fallback'):
        o_reset = gamma * global_matchable
    if active_param(config, 'use_o_reset') and topk_trans is not None:
        surv_mask = torch.zeros_like(global_matchable)
        topk_num = min(active_param(config, 'active_reset_topk'), topk_trans.shape[1])
        for k in range(topk_num):
            candidate_trans = topk_trans[:, k, :, :]
            if active_param(config, 'use_rgbd_fsv'):
                candidate_fsv_value = compute_rgbd_fsv(src_points[None], candidate_trans, target_free_space)
            else:
                candidate_fsv_value = estimate_fsv_proxy(
                    src_points,
                    tgt_points,
                    candidate_trans,
                    active_param(config, 'active_fsv_distance')
                )
            if candidate_fsv_value < active_param(config, 'active_reset_tau_fsv'):
                candidate_nn = min_dist_to_points(transform_points(src_points, candidate_trans), tgt_points)
                surv_mask = torch.maximum(surv_mask, (candidate_nn < threshold).float())
        if torch.any(surv_mask > 0):
            surv_dist = min_dist_to_points(src_points, src_points[surv_mask.bool()])
            surv_broad = (surv_dist < broad_radius).float()
            fallback = gamma * global_matchable if active_param(config, 'use_global_reset_fallback') else torch.zeros_like(global_matchable)
            o_reset = noisy_or(surv_broad, fallback).clamp(eps, 1.0)

    local_prior = noisy_or(o_best, (1.0 - roc_bar) * o_broad)
    source_prior = (v * local_prior + (1.0 - v) * o_reset).clamp(eps, 1.0)
    if not active_param(config, 'use_o_prior'):
        source_prior = torch.ones_like(source_prior)

    r1_warped = transform_points(r1_src, r1_trans)
    r1_residual = torch.norm(r1_warped - r1_tgt, dim=1)
    inlier_mask = r1_residual < threshold
    src_support = r1_src[inlier_mask]
    tgt_support = r1_tgt[inlier_mask]
    if src_support.shape[0] == 0:
        src_support = r1_src
        tgt_support = r1_tgt

    src_density = support_density(src_points, src_support, support_radius)
    tgt_density = support_density(tgt_points, tgt_support, support_radius)
    source_under = (1.0 / (1.0 + v * src_density)).clamp(eps, 1.0)
    if active_param(config, 'use_target_ut'):
        target_under = (1.0 / (1.0 + v * tgt_density)).clamp(eps, 1.0)
    else:
        target_under = torch.ones_like(tgt_density).clamp(eps, 1.0)

    c1 = src_support.mean(dim=0) if src_support.shape[0] > 0 else src_points.mean(dim=0)
    c_global = src_points.mean(dim=0)
    c_ref = v * c1 + (1.0 - v) * c_global
    r_src = torch.norm(src_points - c_global.view(1, 3), dim=1).quantile(0.95).clamp_min(eps)

    guide = {
        'source_prior': source_prior,
        'source_under_support': source_under,
        'target_under_support': target_under,
        'c_ref': c_ref,
        'r_src': r_src,
        'lambda': active_param(config, 'active_prior_lambda'),
        'eta_l': active_param(config, 'active_eta_l') if active_param(config, 'use_adaptive_leverage') else 0.0,
        'eps': eps,
    }
    diagnostics = {
        'fsv_value': fsv_value,
        'fsv_metric_name': fsv_metric_name,
        'fsv_proxy': fsv_value if fsv_metric_name == 'fsv_proxy' else np.nan,
        'rgbd_fsv': fsv_value if fsv_metric_name == 'rgbd_fsv' else np.nan,
        'aoc': aoc,
        'roc_bar': roc_bar,
        'overlap_pred': overlap_pred,
        'confidence': v,
        'round1_inliers': int(inlier_mask.sum().item()),
    }
    return guide, diagnostics


def generate_r1_topk_transforms(src_corr, tgt_corr, base_trans, config):
    src = src_corr[0]
    tgt = tgt_corr[0]
    num_corr = src.shape[0]
    if num_corr < 3:
        return base_trans[:, None, :, :]

    max_corr = min(num_corr, active_param(config, 'active_topk_max_corr'))
    if num_corr > max_corr:
        idx = torch.randperm(num_corr, device=src.device)[:max_corr]
        src = src[idx]
        tgt = tgt[idx]
        num_corr = max_corr

    d_thre = cfg_get(config, 'd_thre', cfg_get(config, 'inlier_threshold', 0.1))
    src_dist = torch.cdist(src, src)
    tgt_dist = torch.cdist(tgt, tgt)
    compatibility = (torch.abs(src_dist - tgt_dist) < d_thre).float()
    scores = compatibility.sum(dim=1)
    topk = min(active_param(config, 'active_reset_topk'), num_corr)
    seed_idx = torch.topk(scores, k=topk, largest=True).indices

    transforms = [base_trans]
    local_k = min(active_param(config, 'active_topk_local_corr'), num_corr)
    for seed in seed_idx:
        local_idx = torch.topk(compatibility[seed], k=local_k, largest=True).indices
        if local_idx.shape[0] >= 3:
            transforms.append(rigid_transform_3d(src[local_idx][None], tgt[local_idx][None]))

    return torch.cat(transforms, dim=0)[None]


def refine_transform_on_correspondences(trans, src_corr, tgt_corr, threshold, iters=3):
    refined = trans
    for _ in range(iters):
        residual = torch.norm(transform(src_corr, refined) - tgt_corr, dim=-1)
        inlier = residual < threshold
        if inlier.sum() < 3:
            break
        refined = rigid_transform_3d(src_corr[:, inlier[0], :], tgt_corr[:, inlier[0], :])
    return refined


def select_best_r1_transform(src_corr, tgt_corr, topk_trans, config):
    threshold = active_param(config, 'active_overlap_threshold')
    best_trans = topk_trans[:, 0, :, :]
    best_inliers = -1
    refined_transforms = []
    for i in range(topk_trans.shape[1]):
        refined = refine_transform_on_correspondences(topk_trans[:, i, :, :], src_corr, tgt_corr, threshold)
        residual = torch.norm(transform(src_corr, refined) - tgt_corr, dim=-1)
        inliers = int((residual < threshold).sum().item())
        refined_transforms.append(refined)
        if inliers > best_inliers:
            best_inliers = inliers
            best_trans = refined
    return best_trans, torch.stack(refined_transforms, dim=1), best_inliers


def sample_guided_seed_correspondences(src_keypts, tgt_keypts, src_features, tgt_features, guide, sample_num):
    src_points = src_keypts[0]
    tgt_points = tgt_keypts[0]
    src_desc = src_features[0]
    tgt_desc = tgt_features[0]
    eps = guide.get('eps', 1e-4)
    prior_lambda = guide.get('lambda', 0.1)

    src_score = guide['source_prior'] * guide['source_under_support']
    src_score = torch.nan_to_num(src_score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(src_score.sum()) <= 0:
        src_score = torch.ones_like(src_score)
    sample_num = min(sample_num, src_points.shape[0])
    src_idx = torch.topk(src_score, k=sample_num, largest=True).indices

    distance = torch.sqrt(torch.clamp(2 - 2 * (src_desc[src_idx] @ tgt_desc.T), min=1e-6))
    tgt_under = guide['target_under_support'].clamp(eps, 1.0)
    c_ref = guide['c_ref']
    r_src = guide['r_src']
    eta_l = guide.get('eta_l', 0.0)
    lever_d = torch.norm(src_points[src_idx] - c_ref.view(1, 3), dim=-1) / (r_src + eps)
    lever = 1.0 + eta_l * lever_d.clamp(0.0, 1.0)
    source_log_prior = torch.log(guide['source_prior'][src_idx].clamp(eps, 1.0)) + torch.log(guide['source_under_support'][src_idx].clamp(eps, 1.0)) + torch.log(torch.maximum(lever, torch.tensor(eps, device=lever.device, dtype=lever.dtype)))
    target_log_prior = torch.log(tgt_under)
    distance = distance - prior_lambda * (source_log_prior[:, None] + target_log_prior[None, :])
    tgt_idx = torch.argmin(distance, dim=1)

    return src_points[src_idx][None], tgt_points[tgt_idx][None]


def attach_guided_candidate_pools(guide, src_keypts, tgt_keypts, config):
    source_score = guide['source_prior'] * guide['source_under_support']
    source_score = torch.nan_to_num(source_score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(source_score.sum()) <= 0:
        source_score = torch.ones_like(source_score)
    source_k = min(active_param(config, 'active_round2_sampling'), src_keypts.shape[1])
    guide['source_candidate_indices'] = torch.topk(source_score, k=source_k, largest=True).indices

    target_score = guide['target_under_support']
    target_score = torch.nan_to_num(target_score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if float(target_score.sum()) <= 0:
        target_score = torch.ones_like(target_score)
    target_k = min(active_param(config, 'active_round2_target_pool'), tgt_keypts.shape[1])
    guide['target_candidate_indices'] = torch.topk(target_score, k=target_k, largest=True).indices
    guide['chunk_size'] = active_param(config, 'active_feature_chunk_size')
    return guide


def sample_random_uniform_seed_correspondences(src_keypts, tgt_keypts, sample_num):
    sample_num = min(sample_num, src_keypts.shape[1], tgt_keypts.shape[1])
    src_idx = torch.randperm(src_keypts.shape[1], device=src_keypts.device)[:sample_num]
    tgt_idx = torch.randperm(tgt_keypts.shape[1], device=tgt_keypts.device)[:sample_num]
    return src_keypts[:, src_idx, :], tgt_keypts[:, tgt_idx, :]


def sample_correspondences(src_corr, tgt_corr, sample_num):
    if src_corr.shape[1] == 0:
        return src_corr, tgt_corr
    replace = src_corr.shape[1] < sample_num
    sel_ind = np.random.choice(src_corr.shape[1], sample_num, replace=replace)
    return src_corr[:, sel_ind, :], tgt_corr[:, sel_ind, :]


def mini_ransac_transform(src, tgt, weights, threshold, ransac_iters):
    valid = weights > 0
    if valid.sum() < 3 or ransac_iters <= 0:
        return rigid_transform_3d(src[None], tgt[None], weights=weights[None])
    idx_pool = torch.nonzero(valid, as_tuple=False).flatten()
    probs = weights[idx_pool].clamp_min(0)
    probs = probs / probs.sum()
    best_score = -1.0
    best_trans = None
    for _ in range(ransac_iters):
        sample_pos = torch.multinomial(probs, 3, replacement=False if probs.shape[0] >= 3 else True)
        sample_idx = idx_pool[sample_pos]
        trans = rigid_transform_3d(src[sample_idx][None], tgt[sample_idx][None])
        residual = torch.norm(transform_points(src, trans) - tgt, dim=1)
        inlier = residual < threshold
        score = float((weights * inlier.float()).sum().item())
        if score > best_score:
            best_score = score
            best_trans = trans
    if best_trans is None:
        return rigid_transform_3d(src[None], tgt[None], weights=weights[None])
    residual = torch.norm(transform_points(src, best_trans) - tgt, dim=1)
    inlier = residual < threshold
    if inlier.sum() < 3:
        return best_trans
    return rigid_transform_3d(src[inlier][None], tgt[inlier][None], weights=weights[inlier][None])


def robust_weighted_estimate(src_corr, tgt_corr, initial_trans, config, match_weights=None):
    threshold = cfg_get(config, 'inlier_threshold', 0.1)
    voxel_size = active_param(config, 'active_final_voxel_size')
    n_min = active_param(config, 'active_density_min')
    n_max = active_param(config, 'active_balance_max')
    robust_c = active_param(config, 'active_robust_c')

    src = src_corr[0]
    tgt = tgt_corr[0]
    info = {
        'density_drop_ratio': 0.0,
        'density_filter_fallback': 0.0,
        'final_used_count': int(src.shape[0]),
    }
    if src.shape[0] < 3:
        return initial_trans, torch.ones_like(src_corr[:, :, 0]).bool(), src_corr, tgt_corr, info

    voxels = torch.floor(src / voxel_size).to(torch.int64)
    unique_voxels, inverse, counts = torch.unique(voxels, dim=0, return_inverse=True, return_counts=True)
    point_counts = counts[inverse].float()
    density_mask = point_counts >= n_min if active_param(config, 'use_density_filter') else torch.ones_like(point_counts).bool()
    keep_ratio = float(density_mask.float().mean())
    density_filter_fallback = 0.0
    if density_mask.sum() < 3:
        density_mask = torch.ones_like(density_mask).bool()
        keep_ratio = 1.0
        density_filter_fallback = 1.0

    balance = 1.0 / torch.minimum(point_counts, torch.tensor(float(n_max), device=src.device))
    warped = transform_points(src, initial_trans)
    residual = torch.norm(warped - tgt, dim=1)
    robust = (robust_c ** 2) / ((residual ** 2 + robust_c ** 2) ** 2) if active_param(config, 'use_robust_final') else torch.ones_like(residual)
    if match_weights is None:
        match_weights = torch.ones_like(residual)
    weights = match_weights * density_mask.float() * balance * robust
    if torch.sum(weights > 0) < 3:
        weights = torch.ones_like(weights)

    if active_param(config, 'use_mini_ransac_final'):
        pred_trans = mini_ransac_transform(src, tgt, weights, threshold, active_param(config, 'active_ransac_iters'))
    else:
        pred_trans = rigid_transform_3d(src_corr, tgt_corr, weights=weights[None])
    final_residual = torch.norm(transform_points(src, pred_trans) - tgt, dim=1)
    pred_labels = final_residual[None] < threshold
    if pred_labels.sum() < 3:
        pred_labels = torch.ones_like(pred_labels).bool()
    info['density_drop_ratio'] = 1.0 - keep_ratio
    info['density_filter_fallback'] = density_filter_fallback
    info['final_used_count'] = int(pred_labels.sum().item())
    return pred_trans, pred_labels, src_corr[:, pred_labels[0], :], tgt_corr[:, pred_labels[0], :], info


def select_inliers_for_pose(src_corr, tgt_corr, trans, threshold):
    residual = torch.norm(transform(src_corr, trans) - tgt_corr, dim=-1)
    labels = residual < threshold
    if labels.sum() < 3:
        labels = torch.ones_like(labels).bool()
    return labels, src_corr[:, labels[0], :], tgt_corr[:, labels[0], :]


def count_gt_inliers(src_corr, tgt_corr, gt_trans, threshold):
    if src_corr.shape[1] == 0:
        return 0, 0.0
    residual = torch.norm(transform(src_corr, gt_trans) - tgt_corr, dim=-1)
    inlier = residual < threshold
    return int(inlier.sum().item()), float(inlier.float().mean().item())


def novel_inlier_coverage(r1_src, r1_tgt, r2_src, r2_tgt, gt_trans, threshold, voxel_size):
    if r2_src.shape[1] == 0:
        return 0.0
    r1_residual = torch.norm(transform(r1_src, gt_trans) - r1_tgt, dim=-1)[0]
    r2_residual = torch.norm(transform(r2_src, gt_trans) - r2_tgt, dim=-1)[0]
    r1_inlier_src = r1_src[0, r1_residual < threshold]
    r2_inlier_src = r2_src[0, r2_residual < threshold]
    if r2_inlier_src.shape[0] == 0:
        return 0.0
    r1_voxels = torch.unique(torch.floor(r1_inlier_src / voxel_size).to(torch.int64), dim=0) if r1_inlier_src.shape[0] > 0 else torch.empty((0, 3), dtype=torch.int64, device=r2_src.device)
    r2_voxels = torch.unique(torch.floor(r2_inlier_src / voxel_size).to(torch.int64), dim=0)
    if r1_voxels.shape[0] == 0:
        return 1.0
    combined = torch.cat([r1_voxels, r2_voxels], dim=0)
    unique_combined = torch.unique(combined, dim=0)
    novel_count = unique_combined.shape[0] - r1_voxels.shape[0]
    return float(max(novel_count, 0) / max(r2_voxels.shape[0], 1))


def eval_3DLoMatch_scene(loader, matcher, regenerator, estimator, trans_evaluator, cls_evaluator, scene_ind, config):
    num_pair = loader.__len__()
    max_pairs = cfg_get(config, 'max_pairs', 0)
    if max_pairs:
        num_pair = min(num_pair, int(max_pairs))
    final_poses = np.zeros([num_pair, 4, 4])

    stats = np.zeros([num_pair, 30])
    data_timer, model_timer = Timer(), Timer()
    with torch.no_grad():
        error_pair = []
        fall_idx = []
        # for i in bad_list:
        for i in tqdm(range(num_pair)):
            # i = 3
            # print(f"编号：{i}")
            #################################
            # 1. load data
            #################################
            data_timer.tic()
            data = loader.get_data(i)  # 注意该数是从0开始 222 for 1. 11 for 2
            target_free_space = None
            overlap_source = "none"
            if len(data) == 11:
                (
                    src_keypts, tgt_keypts, src_features, tgt_features, gt_trans,
                    src_pcd, tgt_pcd, src_overlap, tgt_overlap, target_free_space, overlap_source
                ) = data
            elif len(data) == 9:
                src_keypts, tgt_keypts, src_features, tgt_features, gt_trans, src_pcd, tgt_pcd, src_overlap, tgt_overlap = data
                overlap_source = "legacy"
            else:
                src_keypts, tgt_keypts, src_features, tgt_features, gt_trans, src_pcd, tgt_pcd = data
                src_overlap = torch.ones(src_keypts.shape[:2], dtype=torch.float32, device=src_keypts.device)
                tgt_overlap = torch.ones(tgt_keypts.shape[:2], dtype=torch.float32, device=tgt_keypts.device)
                overlap_source = "proxy_all_ones"
            # print("编号",i)
            data_time = data_timer.toc()

            #################################
            # 2. match descriptor and compute rigid transformation
            #################################
            model_timer.tic()
            time1, time2, time3 = Timer(), Timer(), Timer()
            time1.tic()
            pred_trans, src_keypts_corr_filtered, tgt_keypts_corr_filtered, src_keypts_corr, tgt_keypts_corr, src_desc_corr_final, tgt_desc_corr_final = matcher.estimator(
                src_keypts, tgt_keypts, src_features, tgt_features, gt_trans)
            time1 = time1.toc()
            time2.tic()
            seed_src_corr = src_keypts_corr_filtered
            seed_tgt_corr = tgt_keypts_corr_filtered
            if seed_src_corr.shape[1] == 0:
                seed_src_corr = src_keypts_corr
                seed_tgt_corr = tgt_keypts_corr

            r1_seed_src, r1_seed_tgt = sample_correspondences(seed_src_corr, seed_tgt_corr, active_param(config, 'active_round1_sampling'))

            r1_src_corr, r1_tgt_corr, r1_trans = regenerator.regenerate(
                r1_seed_src,
                r1_seed_tgt,
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                gt_trans,
                knn_num=active_param(config, 'active_round1_knn'),
                sampling_num=active_param(config, 'active_round1_sampling')
            )

            r1_topk_trans = generate_r1_topk_transforms(r1_src_corr, r1_tgt_corr, r1_trans, config)
            r1_trans, r1_topk_trans, r1_best_inliers = select_best_r1_transform(r1_src_corr, r1_tgt_corr, r1_topk_trans, config)
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
                target_free_space=target_free_space
            )

            early_stop = active_param(config, 'use_early_stop') and (
                diagnostics['fsv_value'] < active_param(config, 'active_tau_fsv')
                and diagnostics['roc_bar'] > active_param(config, 'active_tau_rho')
            )
            round2_mode = active_param(config, 'round2_mode')
            enter_round2 = active_param(config, 'use_round2') and active_param(config, 'active_max_rounds') >= 2 and round2_mode != 'none' and not early_stop

            src_keypts_corr_final = r1_src_corr
            tgt_keypts_corr_final = r1_tgt_corr
            pred_trans = r1_trans
            match_weights = torch.ones(r1_src_corr.shape[1], dtype=r1_src_corr.dtype, device=r1_src_corr.device)
            r2_src_corr = torch.empty((1, 0, 3), dtype=src_keypts.dtype, device=src_keypts.device)
            r2_tgt_corr = torch.empty((1, 0, 3), dtype=tgt_keypts.dtype, device=tgt_keypts.device)

            if enter_round2:
                regenerate_mode = "local"
                regenerate_guide = None
                if round2_mode == 'original':
                    r2_seed_src, r2_seed_tgt = sample_correspondences(
                        seed_src_corr,
                        seed_tgt_corr,
                        active_param(config, 'active_round2_sampling')
                    )
                elif round2_mode == 'random_uniform':
                    r2_seed_src, r2_seed_tgt = sample_random_uniform_seed_correspondences(
                        src_keypts,
                        tgt_keypts,
                        active_param(config, 'active_round2_sampling')
                    )
                elif round2_mode == 'weakness_guided':
                    guide = attach_guided_candidate_pools(guide, src_keypts, tgt_keypts, config)
                    r2_seed_src, r2_seed_tgt = sample_guided_seed_correspondences(
                        src_keypts,
                        tgt_keypts,
                        src_features,
                        tgt_features,
                        guide,
                        active_param(config, 'active_round2_sampling')
                    )
                    regenerate_mode = "guided_global"
                    regenerate_guide = guide
                else:
                    raise ValueError(f"Unsupported round2_mode: {round2_mode}")
                r2_src_corr, r2_tgt_corr, r2_trans = regenerator.regenerate(
                    r2_seed_src,
                    r2_seed_tgt,
                    src_keypts,
                    tgt_keypts,
                    src_features,
                    tgt_features,
                    gt_trans,
                    knn_num=active_param(config, 'active_round2_knn'),
                    sampling_num=active_param(config, 'active_round2_sampling'),
                    guide=regenerate_guide,
                    mode=regenerate_mode
                )
                src_keypts_corr_final = torch.cat([r1_src_corr, r2_src_corr], dim=1)
                tgt_keypts_corr_final = torch.cat([r1_tgt_corr, r2_tgt_corr], dim=1)
                pred_trans = r2_trans
                if regenerator.last_match_weights is not None and regenerator.last_match_weights.shape[0] == r2_src_corr.shape[1]:
                    r2_match_weights = regenerator.last_match_weights.to(device=r1_src_corr.device, dtype=r1_src_corr.dtype)
                else:
                    r2_match_weights = torch.ones(r2_src_corr.shape[1], dtype=r1_src_corr.dtype, device=r1_src_corr.device)
                match_weights = torch.cat([match_weights, r2_match_weights], dim=0)

            time2 = time2.toc()
            time3.tic()
            if not early_stop and active_param(config, 'use_ctc') and src_keypts_corr_final.shape[1] >= 3:
                pred_trans, _, src_keypts_corr_final, tgt_keypts_corr_final = estimator.estimator(
                    src_keypts_corr_final,
                    tgt_keypts_corr_final,
                    src_keypts,
                    tgt_keypts,
                    gt_trans
                )
                match_weights = torch.ones(src_keypts_corr_final.shape[1], dtype=src_keypts_corr_final.dtype, device=src_keypts_corr_final.device)

            if not early_stop:
                pred_trans, pred_labels, src_corr_final, tgt_corr_final, final_info = robust_weighted_estimate(
                    src_keypts_corr_final,
                    tgt_keypts_corr_final,
                    pred_trans,
                    config,
                    match_weights=match_weights
                )
            else:
                pred_labels, src_corr_final, tgt_corr_final = select_inliers_for_pose(
                    src_keypts_corr_final,
                    tgt_keypts_corr_final,
                    pred_trans,
                    config.inlier_threshold
                )
                final_info = {
                    'density_drop_ratio': 0.0,
                    'final_used_count': int(src_corr_final.shape[1]),
                    'density_filter_fallback': 0.0,
                }
            time3 = time3.toc()


            model_time = model_timer.toc()
            #################################
            # 3. generate the ground-truth classification result
            #################################
            frag1_warp = transform(src_keypts_corr, gt_trans)  # 更改
            distance = torch.sum((frag1_warp - tgt_keypts_corr) ** 2, dim=-1) ** 0.5
            gt_labels = (distance < config.inlier_threshold).float()

            #################################
            # 4. evaluate result
            #################################
            recall, Re, Te, rmse = trans_evaluator(pred_trans, gt_trans, src_corr_final, tgt_corr_final)
            class_stats = cls_evaluator(gt_trans, src_corr_final, tgt_corr_final, src_keypts_corr, tgt_keypts_corr)

            # 调参
            class_stats_stage = cls_evaluator(gt_trans, src_keypts_corr_final, tgt_keypts_corr_final,
                                              src_keypts_corr_filtered, tgt_keypts_corr_filtered)
            r1_inlier_count, _ = count_gt_inliers(r1_src_corr, r1_tgt_corr, gt_trans, config.inlier_threshold)
            r2_inlier_count, r2_precision = count_gt_inliers(r2_src_corr, r2_tgt_corr, gt_trans, config.inlier_threshold)
            novel_cov = novel_inlier_coverage(
                r1_src_corr,
                r1_tgt_corr,
                r2_src_corr,
                r2_tgt_corr,
                gt_trans,
                config.inlier_threshold,
                active_param(config, 'active_voxel_size')
            )


            if recall == 0:
                fall_idx.append(i)
            #################################
            # 5. save the result
            #################################
            stats[i, 0] = float(recall / 100.0)  # success
            stats[i, 1] = float(Re)  # Re (deg)
            stats[i, 2] = float(Te)  # Te (cm)
            stats[i, 3] = int(torch.sum(gt_labels))  # input inlier number
            stats[i, 4] = float(torch.mean(gt_labels.float()))  # input inlier ratio
            stats[i, 5] = float(class_stats['output_inlier_number'])  # output inlier number
            stats[i, 6] = float(class_stats['precision'])  # output inlier precision
            stats[i, 7] = float(class_stats['IR_ratio'])  # output inlier recall
            stats[i, 8] = float(class_stats['f1'])  # output inlier f1 score
            stats[i, 9] = model_time
            stats[i, 10] = data_time
            stats[i, 11] = scene_ind
            stats[i, 12] = float(class_stats['feature_match_recall'])  # feature_match_recall
            stats[i, 13] = float(class_stats['feature_match_recall_0.1'])  # feature_match_recall
            stats[i, 14] = float(class_stats['feature_match_recall_0.01'])  # feature_match_recall
            stats[i, 15] = float(class_stats['IR_ratio'])  # IR_ratio
            stats[i, 16] = float(class_stats['INR'])  # INR
            stats[i, 17] = float(class_stats['NR'])  # NR
            stats[i, 18] = float(enter_round2)
            stats[i, 19] = float(class_stats['inlier_nums'])  # inlier_nums
            stats[i, 20] = float(early_stop)
            stats[i, 21] = float(r1_inlier_count)
            stats[i, 22] = float(r2_inlier_count)
            stats[i, 23] = float(r2_precision)
            stats[i, 24] = float(novel_cov)
            stats[i, 25] = float(final_info['final_used_count'])
            stats[i, 26] = float(final_info['density_drop_ratio'])
            stats[i, 27] = float(final_info['density_filter_fallback'])
            stats[i, 28] = float(diagnostics['fsv_value'])
            stats[i, 29] = float(overlap_source in ("proxy_all_ones", "overlap_proxy_all_ones"))

            final_poses[i] = pred_trans[0].detach().cpu().numpy()
        print(fall_idx)

    return stats, final_poses


def eval_3DLoMatch_single(config):
    validate_active_config(config)
    if config.use_overlap_proxy:
        logging.info("Overlap_pred is explicitly running in overlap_proxy mode.")
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
        NS_by_IC=50
    )

    regenerator = Regenerator()
    estimator = Estimator(num_node=config.num_node)
    trans_evaluator = TransformationLoss(re_thre=config.re_thre, te_thre=config.te_thre)
    cls_evaluator = ClassificationLoss(inlier_threshold=config.inlier_threshold)

    allpair_stats, allpair_poses = eval_3DLoMatch_scene(loader, matcher, regenerator, estimator, trans_evaluator, cls_evaluator, 0, config)

    allpair_average = allpair_stats.mean(0)
    allpair_status_ndarray = np.array(allpair_stats, dtype=float)

    if not cfg_get(config, 'max_pairs', 0) and not cfg_get(config, 'skip_benchmark', False):
        benchmark_predator(allpair_poses, gt_folder='benchmarks/3DLoMatch')
    
    # benchmarking using the registration recall defined in DGR 
    allpair_average = allpair_stats.mean(0)
    correct_pair_average = allpair_stats[allpair_stats[:, 0] == 1].mean(0) if np.any(allpair_stats[:, 0] == 1) else np.zeros_like(allpair_average)
    logging.info(f"*" * 40)
    logging.info(f"All {allpair_stats.shape[0]} pairs, Mean Reg Recall={allpair_average[0] * 100:.2f}%, Mean Re={correct_pair_average[1]:.2f}, Mean Te={correct_pair_average[2]:.2f}， Mean FMR={allpair_average[12]* 100:.2f}")
    logging.info(f"\tMean FMR={allpair_average[12]* 100:.2f}, Mean FMR>10%={allpair_average[13]* 100:.2f}, Mean FMR>1%={allpair_average[14]* 100:.2f}")
    logging.info(f"\tInput:  Mean Inlier Num={allpair_average[3]:.2f}(ratio={allpair_average[4] * 100:.2f}%)")
    logging.info(f"\tOutput: Mean Inlier Num={allpair_average[5]:.2f}(precision={allpair_average[6] * 100:.2f}%, recall={allpair_average[7] * 100:.2f}%, f1={allpair_average[8] * 100:.2f}%)")
    logging.info(f"\tcorrs: IR_ratio={allpair_average[15] * 100:.2f}%, INR={allpair_average[16] * 100:.2f}%, inlier_nums={allpair_average[19]:.2f}")
    logging.info(f"\tDiagnosis-guided Round2 trigger rate: {allpair_average[18] * 100:.2f}%")
    logging.info(f"\tEarly stop rate: {allpair_average[20] * 100:.2f}%")
    logging.info(f"\tR1 inlier count: {allpair_average[21]:.2f}, R2 inlier count: {allpair_average[22]:.2f}, R2 precision: {allpair_average[23] * 100:.2f}%")
    logging.info(f"\tNovel inlier coverage: {allpair_average[24] * 100:.2f}%")
    logging.info(f"\tFinal used correspondence count: {allpair_average[25]:.2f}")
    logging.info(f"\tDensity filtering drop ratio: {allpair_average[26] * 100:.2f}%")
    logging.info(f"\tDensity filtering fallback rate: {allpair_average[27] * 100:.2f}%")
    logging.info(f"\tMean {('rgbd_fsv' if config.use_rgbd_fsv else 'fsv_proxy')}: {allpair_average[28]:.4f}")
    logging.info(f"\tOverlap proxy rate: {allpair_average[29] * 100:.2f}%")
    logging.info(f"\tAblation setting: {ROUND2_LABELS[config.round2_mode]}")
    logging.info(
        "\tActive params: "
        f"round2_mode={config.round2_mode}, "
        f"lambda_prior={config.active_prior_lambda}, eta_L={config.active_eta_l}, gamma_reset={config.active_reset_gamma}, "
        f"alpha={config.active_alpha}, tau_fsv={config.active_tau_fsv}, tau_rho={config.active_tau_rho}, "
        f"tau_overlap={config.active_overlap_threshold}, n_min={config.active_density_min}, "
        f"n_max={config.active_balance_max}, K_top={config.active_reset_topk}, "
        f"voxel_size={config.active_voxel_size}, final_voxel_size={config.active_final_voxel_size}, "
        f"ransac_iters={config.active_ransac_iters}, max_rounds={config.active_max_rounds}, "
        f"robust_kernel_c={config.active_robust_c}, epsilon_log={config.active_eps}, "
        f"early_stop_enabled={config.use_early_stop}, density_filter_enabled={config.use_density_filter}, "
        f"use_ctc={config.use_ctc}, use_mini_ransac_final={config.use_mini_ransac_final}, "
        f"use_o_reset={config.use_o_reset}, use_global_reset_fallback={config.use_global_reset_fallback}, "
        f"use_fsv_proxy={config.use_fsv_proxy}, use_rgbd_fsv={config.use_rgbd_fsv}, use_overlap_proxy={config.use_overlap_proxy}"
    )
    logging.info(f"\tMean model time: {allpair_average[9]:.4f}s, Mean data time: {allpair_average[10]:.4f}s")

    # all_stats_npy = np.concatenate([v for k, v in all_stats.items()], axis=0)

    return allpair_stats


def summarize_ablation_stats(label, stats):
    avg = stats.mean(0)
    correct = stats[stats[:, 0] == 1].mean(0) if np.any(stats[:, 0] == 1) else np.zeros_like(avg)
    return (
        f"{label}: Pairs={stats.shape[0]}, RR={avg[0] * 100:.2f}%, "
        f"Re={correct[1]:.2f}, Te={correct[2]:.2f}, FMR={avg[12] * 100:.2f}%, "
        f"Round2={avg[18] * 100:.2f}%, EarlyStop={avg[20] * 100:.2f}%, "
        f"R1={avg[21]:.2f}, R2={avg[22]:.2f}, R2Precision={avg[23] * 100:.2f}%, "
        f"Novel={avg[24] * 100:.2f}%, FinalCorr={avg[25]:.2f}, DensityDrop={avg[26] * 100:.2f}%"
    )


def ablation_config(config, mode):
    cfg = edict(copy.deepcopy(dict(config)))
    cfg.round2_mode = mode
    cfg.use_round2 = mode != 'none'
    return cfg


def eval_3DLoMatch(config):
    if not config.run_round2_ablation_suite:
        return eval_3DLoMatch_single(config)
    results = {}
    for mode in ('none', 'original', 'random_uniform', 'weakness_guided'):
        cfg = ablation_config(config, mode)
        logging.info(f"{'=' * 12} {ROUND2_LABELS[mode]} {'=' * 12}")
        stats = eval_3DLoMatch_single(cfg)
        results[mode] = stats
    logging.info("Round2 ablation suite summary")
    for mode in ('none', 'original', 'random_uniform', 'weakness_guided'):
        logging.info(summarize_ablation_stats(ROUND2_LABELS[mode], results[mode]))
    return results


def benchmark_predator(pred_poses, gt_folder):
    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder,ele) for ele in scenes]

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    n_valids= []

    short_names=['Kitchen','Home 1','Home 2','Hotel 1','Hotel 2','Hotel 3','Study','MIT Lab']
    logging.info(("Scene\t| prec.\t| rec.\t| re\t| te\t| samples\t|"))
    
    start_ind = 0
    for idx,scene in enumerate(scene_names):
        # ground truth info
        gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
        n_valid=0
        for ele in gt_pairs:
            diff=abs(int(ele[0])-int(ele[1]))
            n_valid+=diff>1
        n_valids.append(n_valid)

        n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene,"gt.info"))

        # estimated info
        # est_pairs, est_traj = read_trajectory(os.path.join(est_folder,scenes[idx],'est.log'))
        est_traj = pred_poses[start_ind:start_ind + len(gt_pairs)]
        start_ind = start_ind + len(gt_pairs)

        temp_precision, temp_recall,c_flag = evaluate_registration(n_fragments, est_traj, gt_pairs, gt_pairs, gt_traj, gt_traj_cov)
        
        # Filter out the estimated rotation matrices
        ext_gt_traj = extract_corresponding_trajectors(gt_pairs,gt_pairs, gt_traj)

        re = rotation_error(torch.from_numpy(ext_gt_traj[:,0:3,0:3]), torch.from_numpy(est_traj[:,0:3,0:3])).cpu().numpy()[np.array(c_flag)==0]
        te = translation_error(torch.from_numpy(ext_gt_traj[:,0:3,3:4]), torch.from_numpy(est_traj[:,0:3,3:4])).cpu().numpy()[np.array(c_flag)==0]

        re_per_scene['mean'].append(np.mean(re))
        re_per_scene['median'].append(np.median(re))
        re_per_scene['min'].append(np.min(re))
        re_per_scene['max'].append(np.max(re))
        
        te_per_scene['mean'].append(np.mean(te))
        te_per_scene['median'].append(np.median(te))
        te_per_scene['min'].append(np.min(te))
        te_per_scene['max'].append(np.max(te))


        re_all.extend(re.reshape(-1).tolist())
        te_all.extend(te.reshape(-1).tolist())

        precision.append(temp_precision)
        recall.append(temp_recall)

        logging.info("{}\t| {:.3f}\t| {:.3f}\t| {:.3f}\t| {:.3f}\t| {:3d}|".format(short_names[idx], temp_precision, temp_recall, np.median(re), np.median(te), n_valid))
        # np.save(f'{est_folder}/{scenes[idx]}/flag.npy',c_flag)
    
    weighted_precision = (np.array(n_valids) * np.array(precision)).sum() / np.sum(n_valids)

    logging.info("Mean precision: {:.3f}: +- {:.3f}".format(np.mean(precision),np.std(precision)))
    logging.info("Weighted precision: {:.3f}".format(weighted_precision))

    logging.info("Mean median RRE: {:.3f}: +- {:.3f}".format(np.mean(re_per_scene['median']), np.std(re_per_scene['median'])))
    logging.info("Mean median RTE: {:.3F}: +- {:.3f}".format(np.mean(te_per_scene['median']),np.std(te_per_scene['median'])))
    

if __name__ == '__main__':
    from config import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='', type=str, help='snapshot dir')
    parser.add_argument('--solver', default='SVD', type=str, choices=['SVD', 'RANSAC'])
    parser.add_argument('--use_icp', default=False, type=str2bool)
    parser.add_argument('--save_npy', default=False, type=str2bool)
    args = parser.parse_args()

    config_path = args.config_path
    config = json.load(open(config_path, 'r'))
    config = edict(config)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_Devices
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    round2_suffix = 'round2-suite' if config.run_round2_ablation_suite else (config.round2_mode if cfg_get(config, 'use_round2', False) else 'round1-only')
    fsv_suffix = 'rgbd_fsv' if config.use_rgbd_fsv else 'fsv_proxy'
    overlap_suffix = 'overlap_proxy' if config.use_overlap_proxy else 'overlap_pred'
    log_filename = f'logs/3DLoMatch-{config.descriptor}-{round2_suffix}-{fsv_suffix}-{overlap_suffix}.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_filename,
                        filemode='a',
                        format="")
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # evaluate on the test set
    stats = eval_3DLoMatch(config)
    if args.save_npy:
        if isinstance(stats, dict):
            for mode, value in stats.items():
                save_path = log_filename.replace('.log', f'-{mode}.npy')
                np.save(save_path, value)
                print(f"Save the stats in {save_path}")
        else:
            save_path = log_filename.replace('.log', '.npy')
            np.save(save_path, stats)
            print(f"Save the stats in {save_path}")
