from dataclasses import dataclass, field

import torch

from common import rigid_transform_3d


@dataclass
class PoseHypothesis:
    hypothesis_id: int
    parent_id: int
    round_id: int
    pose_raw: torch.Tensor
    pose_local_refined: torch.Tensor
    src_corr: torch.Tensor
    tgt_corr: torch.Tensor
    correspondence_scores: torch.Tensor
    seed_ids: torch.Tensor
    generation_mode: str
    descriptor_score: float = 0.0
    predicted_escape_score: float = 0.0
    search_ray_score: float = float("inf")
    search_free_violation: float = float("inf")
    validation_ray_score: float = float("inf")
    surface_support: float = 0.0
    valid_observation_count: int = 0
    valid_observation_ratio: float = 0.0
    frame_coverage: float = 0.0
    bidirectional_consistency: float = 0.0
    nearest_history_distance: float = float("inf")
    signature_similarity: float = 0.0
    search_insufficient_evidence: bool = False
    insufficient_evidence: bool = False
    rejected_by_history: bool = False
    ray_signature: object | None = None
    geometry_rmse: float = float("inf")
    refinement_attempted: bool = False
    entered_refinement_pool: bool = False
    refinement_child_id: int = -1
    refinement_accepted: bool | None = None
    refinement_reject_reason: str = ""
    rotation_delta_deg: float = float("nan")
    translation_delta_m: float = float("nan")
    parent_validation_score: float = float("inf")
    child_validation_score: float = float("inf")
    parent_free_violation: float = float("inf")
    child_free_violation: float = float("inf")
    parent_geometry_rmse: float = float("inf")
    child_geometry_rmse: float = float("inf")

    @property
    def pose(self):
        return self.pose_local_refined

    @property
    def candidate_id(self):
        return self.hypothesis_id

    @property
    def parent_candidate_id(self):
        return self.parent_id

    @property
    def correspondence_count(self):
        return int(self.src_corr.shape[1])

    @property
    def has_sufficient_evidence(self):
        return not self.search_insufficient_evidence and not self.insufficient_evidence


@dataclass(frozen=True)
class DescriptorCache:
    topk_indices: torch.Tensor
    topk_scores: torch.Tensor


class RayGuidedRegenerator:
    def __init__(self, descriptor_topk, local_corr_max_points, local_knn_radius, local_mutual_k, escape_lambda, independent_explore_fraction, local_regenerator=None, seed_group_count=None, refine_corr_radius=0.15):
        self.descriptor_topk = int(descriptor_topk)
        self.local_corr_max_points = int(local_corr_max_points)
        self.local_knn_radius = float(local_knn_radius)
        self.local_mutual_k = int(local_mutual_k)
        self.refine_corr_radius = float(refine_corr_radius)
        self.escape_lambda = float(escape_lambda)
        self.independent_explore_fraction = float(independent_explore_fraction)
        self.local_regenerator = local_regenerator
        self.seed_group_count = int(seed_group_count) if seed_group_count is not None else None
        self.descriptor_cache = None

    def prepare_descriptor_cache(self, src_features, tgt_features):
        src = src_features[0] if src_features.ndim == 3 else src_features
        tgt = tgt_features[0] if tgt_features.ndim == 3 else tgt_features
        scores = src @ tgt.transpose(0, 1)
        topk_scores, topk_indices = torch.topk(scores, k=min(self.descriptor_topk, tgt.shape[0]), dim=1)
        self.descriptor_cache = DescriptorCache(topk_indices, topk_scores)
        return self.descriptor_cache

    def _candidate_pairs(self, src_features, tgt_features):
        cache = self.descriptor_cache or self.prepare_descriptor_cache(src_features, tgt_features)
        source = torch.arange(cache.topk_indices.shape[0], device=cache.topk_indices.device)[:, None].expand_as(cache.topk_indices)
        return torch.stack([source.reshape(-1), cache.topk_indices.reshape(-1)], dim=1), cache.topk_scores.reshape(-1)

    @staticmethod
    def _se3_log(transform):
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        cosine = torch.clamp((torch.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
        theta = torch.acos(cosine)
        vee = torch.stack([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
        if theta < 1e-5:
            omega = 0.5 * vee
        else:
            omega = theta * vee / torch.clamp_min(2.0 * torch.sin(theta), 1e-6)
        wx, wy, wz = omega
        zero = torch.zeros((), dtype=transform.dtype, device=transform.device)
        skew = torch.stack([
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero]),
        ])
        identity = torch.eye(3, dtype=transform.dtype, device=transform.device)
        if theta < 1e-5:
            inverse_jacobian = identity - 0.5 * skew + (skew @ skew) / 12.0
        else:
            a = torch.sin(theta) / theta
            b = (1.0 - torch.cos(theta)) / (theta * theta)
            coefficient = (1.0 - a / torch.clamp_min(2.0 * b, 1e-6)) / (theta * theta)
            inverse_jacobian = identity - 0.5 * skew + coefficient * (skew @ skew)
        return torch.cat([omega, inverse_jacobian @ translation])

    def _escape_score(self, candidate_pose, current_pose, constraints):
        if not constraints.constraints:
            return 0.0
        current = current_pose[0] if current_pose.ndim == 3 else current_pose
        candidate = candidate_pose[0] if candidate_pose.ndim == 3 else candidate_pose
        delta = self._se3_log(candidate @ torch.linalg.inv(current))
        residual = torch.relu(constraints.b - constraints.G @ delta)
        return float(-(residual * constraints.weights).mean().item())

    def _select_seed_groups(self, pairs, scores, src_points, tgt_points, group_count, offset=0, used_anchor_pairs=None, used_group_keys=None):
        order = torch.argsort(scores, descending=True)
        if order.numel():
            order = torch.roll(order, shifts=-int(offset) % order.numel())
        groups = []
        used_anchor_pairs = set() if used_anchor_pairs is None else used_anchor_pairs
        used_group_keys = set() if used_group_keys is None else used_group_keys
        for index in order.tolist():
            pair = pairs[index]
            anchor_key = (int(pair[0]), int(pair[1]))
            if anchor_key in used_anchor_pairs:
                continue
            distance_src = torch.linalg.norm(src_points[pairs[:, 0]] - src_points[pair[0]], dim=1)
            distance_tgt = torch.linalg.norm(tgt_points[pairs[:, 1]] - tgt_points[pair[1]], dim=1)
            compatible = torch.where(torch.abs(distance_src - distance_tgt) <= self.local_knn_radius)[0]
            if compatible.numel() < 3:
                continue
            compatible = compatible[torch.argsort(scores[compatible], descending=True)]
            non_anchor = compatible[compatible != index]
            group = torch.cat([
                torch.tensor([index], dtype=torch.long, device=pairs.device),
                non_anchor[:max(0, min(6, compatible.numel()) - 1)],
            ])
            group_key = tuple(sorted((int(pairs[item, 0]), int(pairs[item, 1])) for item in group.tolist()))
            if group_key in used_group_keys:
                continue
            used_anchor_pairs.add(anchor_key)
            used_group_keys.add(group_key)
            groups.append(group)
            if len(groups) >= group_count:
                break
        return groups

    def _local_refine(self, parent, src_points, tgt_points, src_features, tgt_features):
        src = src_points[0] if src_points.ndim == 3 else src_points
        tgt = tgt_points[0] if tgt_points.ndim == 3 else tgt_points
        seed_src = src[parent.seed_ids[:, 0]][None]
        seed_tgt = tgt[parent.seed_ids[:, 1]][None]
        corr_src, corr_tgt = seed_src, seed_tgt
        guide = {
            "local_radius": self.local_knn_radius,
            "local_max_points": self.local_corr_max_points,
            "generalized_mutual_k": self.local_mutual_k,
            "max_matches_per_seed": self.local_corr_max_points,
        }
        if self.local_regenerator is not None:
            try:
                corr_src, corr_tgt, _ = self.local_regenerator.regenerate_seed_group(
                    seed_src, seed_tgt, src_points, tgt_points, src_features, tgt_features, guide,
                )
            except (RuntimeError, ValueError):
                pass
        if corr_src.shape[1] < 3:
            return None
        raw_pose = parent.pose_raw[0] if parent.pose_raw.ndim == 3 else parent.pose_raw
        src_warped = corr_src @ raw_pose[:3, :3].transpose(0, 1) + raw_pose[:3, 3]
        residuals = torch.linalg.norm(src_warped - corr_tgt, dim=-1)
        keep = residuals[0] < self.refine_corr_radius
        if int(keep.sum().item()) < 3:
            return None
        corr_src, corr_tgt = corr_src[:, keep], corr_tgt[:, keep]
        return corr_src, corr_tgt, rigid_transform_3d(corr_src, corr_tgt)

    def _explicit_seed_groups(self, seed_src, seed_tgt, src_points, tgt_points):
        if seed_src is None or seed_tgt is None:
            return []
        source = seed_src[0] if seed_src.ndim == 3 else seed_src
        target = seed_tgt[0] if seed_tgt.ndim == 3 else seed_tgt
        source_ids = torch.cdist(source, src_points).argmin(dim=1)
        target_ids = torch.cdist(target, tgt_points).argmin(dim=1)
        pairs = torch.stack([source_ids, target_ids], dim=1)
        groups = []
        for start in range(0, pairs.shape[0], 6):
            group = pairs[start:start + 6]
            if group.shape[0] >= 3:
                groups.append(group)
        return groups

    def refine_hypothesis(self, parent, src_points, tgt_points, src_features, tgt_features):
        refined = self._local_refine(parent, src_points, tgt_points, src_features, tgt_features)
        if refined is None:
            return None
        corr_src, corr_tgt, refined_pose = refined
        return PoseHypothesis(
            hypothesis_id=-1,
            parent_id=parent.hypothesis_id,
            round_id=parent.round_id,
            pose_raw=parent.pose_raw.detach().clone(),
            pose_local_refined=refined_pose,
            src_corr=corr_src,
            tgt_corr=corr_tgt,
            correspondence_scores=torch.ones(corr_src.shape[1], device=corr_src.device, dtype=corr_src.dtype),
            seed_ids=parent.seed_ids.detach().clone(),
            generation_mode=f"{parent.generation_mode}_refined",
            descriptor_score=parent.descriptor_score,
            predicted_escape_score=parent.predicted_escape_score,
        )

    def generate_hypotheses(self, seed_src, seed_tgt, src_points, tgt_points, src_features, tgt_features, constraints, candidate_count, round_id, parent_id):
        if seed_src is not None and seed_tgt is not None:
            if seed_src.shape[1] != seed_tgt.shape[1] or seed_src.shape[1] < 3:
                raise ValueError("Explicit seed correspondences must be paired and contain at least three entries.")
        src = src_points[0] if src_points.ndim == 3 else src_points
        tgt = tgt_points[0] if tgt_points.ndim == 3 else tgt_points
        pairs, descriptor_scores = self._candidate_pairs(src_features, tgt_features)
        current_pose = getattr(constraints, "current_pose", None)
        if current_pose is None:
            raise ValueError("EscapeConstraints must carry current_pose for correspondence scoring.")
        output_count = min(int(candidate_count), self.seed_group_count) if self.seed_group_count is not None else int(candidate_count)
        explore_count = int(round(output_count * self.independent_explore_fraction))
        guided_count = max(0, output_count - explore_count)
        used_anchor_pairs, used_group_keys = set(), set()
        guided_groups = self._select_seed_groups(
            pairs, descriptor_scores, src, tgt, guided_count,
            round_id * max(1, candidate_count), used_anchor_pairs, used_group_keys,
        ) if guided_count else []
        independent_groups = self._select_seed_groups(
            pairs, descriptor_scores, src, tgt, explore_count,
            round_id * max(1, explore_count), used_anchor_pairs, used_group_keys,
        ) if explore_count else []

        def score_group(group, mode):
            group_pairs = pairs[group]
            grouped_src = src[group_pairs[:, 0]][None]
            grouped_tgt = tgt[group_pairs[:, 1]][None]
            pose_raw = rigid_transform_3d(grouped_src, grouped_tgt)
            descriptor_score = float(descriptor_scores[group].mean().item())
            escape_score = self._escape_score(pose_raw, current_pose, constraints) if mode != "independent" else 0.0
            score = descriptor_score + self.escape_lambda * escape_score
            return {
                "pairs": group_pairs,
                "descriptor_scores": descriptor_scores[group],
                "pose_raw": pose_raw,
                "descriptor_score": descriptor_score,
                "escape_score": escape_score,
                "score": score,
                "mode": mode,
            }

        guided_records = [score_group(group, "ray_guided") for group in guided_groups]
        independent_records = [score_group(group, "independent") for group in independent_groups]
        guided_records.sort(key=lambda record: record["score"], reverse=True)
        independent_records.sort(key=lambda record: record["descriptor_score"], reverse=True)
        selected_records = guided_records[:guided_count] + independent_records[:explore_count]
        explicit_groups = self._explicit_seed_groups(seed_src, seed_tgt, src, tgt)
        if explicit_groups:
            explicit_records = []
            for group_pairs in explicit_groups:
                grouped_src = src[group_pairs[:, 0]][None]
                grouped_tgt = tgt[group_pairs[:, 1]][None]
                pose_raw = rigid_transform_3d(grouped_src, grouped_tgt)
                explicit_records.append({
                    "pairs": group_pairs,
                    "descriptor_scores": torch.zeros(group_pairs.shape[0], device=src.device, dtype=src.dtype),
                    "pose_raw": pose_raw,
                    "descriptor_score": 0.0,
                    "escape_score": self._escape_score(pose_raw, current_pose, constraints),
                    "score": 0.0,
                    "mode": "explicit_seed",
                })
            selected_records = explicit_records + selected_records
        hypotheses = []
        for record in selected_records[:output_count]:
            group_pairs = record["pairs"]
            grouped_src = src[group_pairs[:, 0]][None]
            grouped_tgt = tgt[group_pairs[:, 1]][None]
            hypotheses.append(PoseHypothesis(
                hypothesis_id=-1,
                parent_id=int(parent_id),
                round_id=int(round_id),
                pose_raw=record["pose_raw"],
                pose_local_refined=record["pose_raw"],
                src_corr=grouped_src,
                tgt_corr=grouped_tgt,
                correspondence_scores=record["descriptor_scores"],
                seed_ids=group_pairs.detach().clone(),
                generation_mode=record["mode"],
                descriptor_score=record["descriptor_score"],
                predicted_escape_score=record["escape_score"],
            ))
        return hypotheses
