from dataclasses import dataclass
import math
import time

import torch

from correspondence_memory import CorrespondenceMemory


REJECTION_REASON_KEYS = (
    "support_too_small",
    "duplicate_correspondence",
    "linear_degenerate",
    "planar_degenerate",
    "low_coverage",
    "cross_group_inconsistent",
    "svd_failed",
    "duplicate_pose",
)


@dataclass
class ValidationEvidence:
    candidate_residuals: torch.Tensor
    verified_candidate_inliers: torch.Tensor
    verified_candidate_outliers: torch.Tensor
    source_indices: torch.Tensor
    target_indices: torch.Tensor
    residuals: torch.Tensor
    unique_inlier_ratio: float
    coverage: float
    bidirectional_consistency: float
    free_space_conflict_ratio: float
    free_space_available: bool


@dataclass
class MemoryHypothesis:
    hypothesis_id: int
    parent_id: int
    round_id: int
    stage: str
    pose_raw: torch.Tensor
    pose_local_refined: torch.Tensor
    support_ids: torch.Tensor
    inlier_ids: torch.Tensor
    residuals: torch.Tensor
    validation_score: float
    search_score: float
    inlier_count: int
    mean_inlier_error: float
    coverage: float
    support_signature: torch.Tensor
    basin_key: tuple | None
    unique_inlier_ratio: float = 0.0
    median_residual: float = float("inf")
    bidirectional_consistency: float = 0.0
    free_space_conflict_ratio: float = 0.0
    free_space_available: bool = False
    verified_candidate_outliers: torch.Tensor | None = None
    structure_penalty: float = 0.0
    basin_adjustment: float = 0.0
    refinement_accepted: bool = False
    refinement_reject_reason: str = ""

    @property
    def pose(self):
        return self.pose_local_refined

    @property
    def score(self):
        return self.validation_score


@dataclass
class MemorySearchResult:
    pose: torch.Tensor
    best: MemoryHypothesis | None
    r1_hypothesis: MemoryHypothesis | None
    r1_initialization: dict
    raw_hypotheses: list
    post_refinement_hypotheses: list
    evaluated_hypotheses: list
    round_logs: list
    candidate_logs: list
    memory_summary: dict
    candidate_src_indices: torch.Tensor
    candidate_tgt_indices: torch.Tensor

    @property
    def hypotheses(self):
        return self.post_refinement_hypotheses


class MemoryGuidedRegistration:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def _weighted_rigid(src, tgt, weights):
        weights = weights / torch.clamp_min(weights.sum(), 1e-8)
        src_center = (src * weights[:, None]).sum(dim=0)
        tgt_center = (tgt * weights[:, None]).sum(dim=0)
        src_offset, tgt_offset = src - src_center, tgt - tgt_center
        covariance = (src_offset * weights[:, None]).transpose(0, 1) @ tgt_offset
        left, _, right = torch.linalg.svd(covariance)
        correction = torch.eye(3, dtype=src.dtype, device=src.device)
        correction[-1, -1] = torch.det(right.transpose(0, 1) @ left.transpose(0, 1))
        rotation = right.transpose(0, 1) @ correction @ left.transpose(0, 1)
        translation = tgt_center - rotation @ src_center
        pose = torch.eye(4, dtype=src.dtype, device=src.device)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        return pose[None]

    @staticmethod
    def _transform(points, pose):
        matrix = pose[0] if pose.ndim == 3 else pose
        return points @ matrix[:3, :3].transpose(0, 1) + matrix[:3, 3]

    def _robust_refine(self, memory, pose, use_one_to_one=True):
        all_ids = torch.arange(memory.count, device=memory.device)
        src, tgt = memory.correspondence_points(all_ids)
        src, tgt = src[0], tgt[0]
        current = pose
        fixed_weights = memory.fixed_estimation_weights(all_ids)
        for _ in range(self.config.memory_tls_iters):
            residuals = torch.linalg.norm(self._transform(src, current) - tgt, dim=1)
            truncated = (residuals < self.config.memory_tls_threshold).to(memory.dtype)
            weights = fixed_weights * truncated
            active_ids = torch.where(weights > 0)[0]
            if use_one_to_one and active_ids.numel() > 0:
                unique_ids = memory.one_to_one_filter(active_ids, residuals)
                unique_weights = torch.zeros_like(weights)
                unique_weights[unique_ids] = weights[unique_ids]
                weights = unique_weights
            if int((weights > 0).sum().item()) < self.config.memory_support_min:
                break
            current = self._weighted_rigid(src, tgt, weights)
        return current

    def _chunked_nearest(self, query, reference):
        chunk_size = int(self.config.memory_vdce_validation_chunk_size)
        nearest_indices, nearest_distances = [], []
        for start in range(0, query.shape[0], chunk_size):
            chunk = query[start:start + chunk_size]
            distances = torch.cdist(chunk[None], reference[None])[0]
            distance, indices = distances.min(dim=1)
            nearest_indices.append(indices)
            nearest_distances.append(distance)
        return torch.cat(nearest_indices), torch.cat(nearest_distances)

    def _source_coverage(self, memory, source_indices):
        if source_indices.numel() == 0:
            return 0.0
        points = memory.src_points[source_indices]
        voxels = torch.floor(points / self.config.memory_coverage_voxel_size).long()
        return float(torch.unique(voxels, dim=0).shape[0] / memory.source_voxel_count)

    def _verify(self, memory, pose):
        transformed_source = self._transform(memory.src_points, pose)
        source_to_target, forward_residuals = self._chunked_nearest(transformed_source, memory.tgt_points)
        target_to_source, _ = self._chunked_nearest(memory.tgt_points, transformed_source)
        source_indices = torch.arange(memory.src_points.shape[0], dtype=torch.long, device=memory.device)
        mutual = target_to_source[source_to_target] == source_indices
        forward_inliers = forward_residuals < self.config.memory_inlier_threshold
        verified = mutual & forward_inliers
        verified_source_indices = source_indices[verified]
        verified_target_indices = source_to_target[verified]
        verified_residuals = forward_residuals[verified]
        candidate_source = memory.src_indices
        candidate_target = memory.tgt_indices
        candidate_residuals = torch.linalg.norm(
            self._transform(memory.src_points[candidate_source], pose) - memory.tgt_points[candidate_target],
            dim=1,
        )
        verified_candidate_inliers = memory.candidate_ids_for_pairs(verified_source_indices, verified_target_indices)
        source_has_forward_inlier = forward_inliers[candidate_source]
        predicted_target = source_to_target[candidate_source]
        candidate_verified = verified[candidate_source] & (candidate_target == predicted_target)
        candidate_outlier_mask = source_has_forward_inlier & ~candidate_verified
        candidate_outliers = torch.where(candidate_outlier_mask)[0]
        evidence_cap = int(self.config.memory_vdce_max_evidence_candidates)
        if candidate_outliers.numel() > evidence_cap:
            candidate_outliers = candidate_outliers[torch.topk(candidate_residuals[candidate_outliers], k=evidence_cap).indices]
        if verified_candidate_inliers.numel() > evidence_cap:
            verified_candidate_inliers = verified_candidate_inliers[torch.topk(-candidate_residuals[verified_candidate_inliers], k=evidence_cap).indices]
        forward_count = int(forward_inliers.sum().item())
        return ValidationEvidence(
            candidate_residuals=candidate_residuals,
            verified_candidate_inliers=verified_candidate_inliers,
            verified_candidate_outliers=candidate_outliers,
            source_indices=verified_source_indices,
            target_indices=verified_target_indices,
            residuals=verified_residuals,
            unique_inlier_ratio=float(verified_source_indices.numel() / max(1, memory.src_points.shape[0])),
            coverage=self._source_coverage(memory, verified_source_indices),
            bidirectional_consistency=float(verified_source_indices.numel() / max(1, forward_count)),
            free_space_conflict_ratio=0.0,
            free_space_available=False,
        )

    def _vdce_validation_score(self, evidence):
        unique_count = int(evidence.source_indices.numel())
        if not unique_count:
            return float("-inf"), 0, float("inf"), 0.0, 0.0, 0.0, 0.0
        median_residual = float(evidence.residuals.median().item())
        normalized_residual = min(median_residual / self.config.memory_vdce_normalize_residual_scale, 1.0)
        weights = [
            self.config.memory_vdce_w_r,
            self.config.memory_vdce_w_e,
            self.config.memory_vdce_w_c,
            self.config.memory_vdce_w_b,
        ]
        values = [
            evidence.unique_inlier_ratio,
            -normalized_residual,
            evidence.coverage,
            evidence.bidirectional_consistency,
        ]
        if evidence.free_space_available:
            weights.append(self.config.memory_vdce_w_f)
            values.append(-evidence.free_space_conflict_ratio)
        total_weight = sum(weights)
        score = sum(weight * value for weight, value in zip(weights, values)) / max(total_weight, 1e-8)
        return (
            float(score),
            unique_count,
            median_residual,
            evidence.coverage,
            evidence.unique_inlier_ratio,
            evidence.bidirectional_consistency,
            evidence.free_space_conflict_ratio,
        )

    def _invert_pose(self, pose):
        matrix = pose[0] if pose.ndim == 3 else pose
        inv = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
        inv[:3, :3] = matrix[:3, :3].transpose(0, 1)
        inv[:3, 3] = -inv[:3, :3] @ matrix[:3, 3]
        return inv[None]

    def _memory_update_gate(self, current_eval, best_eval):
        quality_ok = (
            current_eval.unique_inlier_ratio > self.config.memory_vdce_min_inlier_ratio
            and current_eval.coverage > self.config.memory_vdce_min_coverage
            and current_eval.median_residual < self.config.memory_vdce_max_median_residual
        )
        conflict_ok = (
            not current_eval.free_space_available
            or current_eval.free_space_conflict_ratio < self.config.memory_vdce_max_conflict_ratio
        )
        if best_eval is None:
            return quality_ok and conflict_ok
        return (
            current_eval.validation_score - best_eval.validation_score > self.config.memory_vdce_min_score_margin
            and quality_ok
            and conflict_ok
        )

    def _signature_support(self, memory, inliers):
        candidate_ids = torch.where(inliers)[0] if inliers.dtype == torch.bool else torch.as_tensor(inliers, dtype=torch.long, device=memory.device)
        if candidate_ids.numel() <= self.config.memory_support_max:
            return candidate_ids
        fixed_rank = memory.descriptor_score[candidate_ids]
        return candidate_ids[torch.topk(fixed_rank, k=self.config.memory_support_max).indices]

    def _make_hypothesis(
        self,
        memory,
        hypothesis_id,
        parent_id,
        round_id,
        stage,
        pose_raw,
        pose,
        support_ids,
        search_score=0.0,
        structure_penalty=0.0,
    ):
        evidence = self._verify(memory, pose)
        signature_ids = self._signature_support(memory, evidence.verified_candidate_inliers)
        if signature_ids.numel() == 0:
            signature_ids = support_ids
        signature = memory.support_signature(signature_ids)
        vdce_score, unique_inlier_count, median_residual, coverage, unique_inlier_ratio, bidirectional, free_space_conflict_ratio = self._vdce_validation_score(evidence)
        validation_score = vdce_score
        mean_error = float(evidence.residuals.mean().item()) if evidence.residuals.numel() else float("inf")
        basin_key = memory.basin_key(pose) if self.config.memory_use_basin else None
        basin_adjustment = memory.basin_bonus(pose, signature)
        return MemoryHypothesis(
            hypothesis_id=hypothesis_id,
            parent_id=parent_id,
            round_id=round_id,
            stage=stage,
            pose_raw=pose_raw,
            pose_local_refined=pose,
            support_ids=support_ids,
            inlier_ids=evidence.verified_candidate_inliers,
            residuals=evidence.candidate_residuals,
            validation_score=validation_score,
            search_score=float(search_score),
            inlier_count=unique_inlier_count,
            mean_inlier_error=mean_error,
            coverage=coverage,
            unique_inlier_ratio=unique_inlier_ratio,
            median_residual=median_residual,
            bidirectional_consistency=bidirectional,
            free_space_conflict_ratio=free_space_conflict_ratio,
            free_space_available=evidence.free_space_available,
            verified_candidate_outliers=evidence.verified_candidate_outliers,
            support_signature=signature,
            basin_key=basin_key,
            structure_penalty=float(structure_penalty),
            basin_adjustment=float(basin_adjustment),
        )

    def _prosac_seed(self, ranked, round_id, hypothesis_index):
        count = ranked.numel()
        fraction = min(1.0, self.config.memory_prosac_initial_fraction + round_id * self.config.memory_prosac_growth)
        prefix = min(count, max(self.config.memory_support_min, int(math.ceil(fraction * count))))
        offset = (hypothesis_index * 7919 + round_id * 104729) % prefix
        return int(ranked[offset].item())

    @staticmethod
    def _select_best(hypotheses):
        return max(hypotheses, key=lambda item: item.validation_score) if hypotheses else None

    def _strong_stop(self, hypothesis, evaluable_source_count):
        inlier_target = max(
            self.config.memory_strong_stop_min_inliers,
            int(math.ceil(self.config.memory_strong_stop_inlier_fraction * evaluable_source_count)),
        )
        return (
            hypothesis.inlier_count >= inlier_target
            and hypothesis.mean_inlier_error <= self.config.memory_strong_stop_error_ratio * self.config.memory_inlier_threshold
            and hypothesis.coverage >= self.config.memory_strong_stop_coverage
        )

    @staticmethod
    def _pose_distance(first, second):
        first, second = first[0], second[0]
        relative = first[:3, :3].transpose(0, 1) @ second[:3, :3]
        rotation = torch.acos(torch.clamp((torch.trace(relative) - 1.0) * 0.5, -1.0, 1.0)) * 180.0 / math.pi
        translation = torch.linalg.norm(first[:3, 3] - second[:3, 3])
        return float(rotation.item()), float(translation.item())

    def _accept_refined_child(self, parent, child):
        rotation, translation = self._pose_distance(parent.pose, child.pose)
        within_trust_region = (
            rotation <= self.config.memory_refine_trust_rotation_deg
            and translation <= self.config.memory_refine_trust_translation
        )
        score_improved = child.validation_score >= parent.validation_score + self.config.memory_refine_min_score_improvement
        if not within_trust_region:
            return False, "outside_trust_region"
        if not score_improved:
            return False, "insufficient_validation_improvement"
        return True, ""

    @staticmethod
    def _candidate_log(hypothesis, seed_id, support_objective, accepted, reject_reason):
        return {
            "round_id": hypothesis.round_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "parent_id": hypothesis.parent_id,
            "stage": hypothesis.stage,
            "seed_id": seed_id,
            "support_count": int(hypothesis.support_ids.numel()),
            "inlier_count": hypothesis.inlier_count,
            "mean_inlier_error": hypothesis.mean_inlier_error,
            "coverage": hypothesis.coverage,
            "unique_inlier_ratio": hypothesis.unique_inlier_ratio,
            "median_residual": hypothesis.median_residual,
            "bidirectional_consistency": hypothesis.bidirectional_consistency,
            "free_space_conflict_ratio": hypothesis.free_space_conflict_ratio,
            "free_space_available": int(hypothesis.free_space_available),
            "score": hypothesis.score,
            "validation_score": hypothesis.validation_score,
            "search_score": hypothesis.search_score,
            "structure_penalty": hypothesis.structure_penalty,
            "basin_adjustment": hypothesis.basin_adjustment,
            "basin_key": list(hypothesis.basin_key) if hypothesis.basin_key is not None else [],
            "support_objective": support_objective,
            "accepted": int(accepted),
            "reject_reason": reject_reason,
        }

    def run(self, src_points, tgt_points, src_features, tgt_features, initial_pose=None, initial_support_ids=None, initial_src_indices=None, initial_tgt_indices=None):
        if self.config.memory_require_r1_cache and initial_pose is None:
            raise ValueError("memory_graph requires the fixed R1 pose as a permanent round-0 parent.")
        if (initial_src_indices is None) != (initial_tgt_indices is None):
            raise ValueError("Cached R1 source and target correspondence indices must be provided together.")
        memory = CorrespondenceMemory(
            src_points,
            tgt_points,
            src_features,
            tgt_features,
            self.config,
            required_src_indices=initial_src_indices,
            required_tgt_indices=initial_tgt_indices,
        )
        mapping = {"r1_corr_count": 0, "r1_mapped_corr_count": 0, "r1_mapped_count": 0, "r1_mapping_ratio": 0.0}
        if initial_src_indices is not None:
            initial_support_ids, mapping = memory.map_cached_indices(initial_src_indices, initial_tgt_indices)
            if mapping["r1_mapping_ratio"] < self.config.memory_r1_min_mapping_ratio:
                raise RuntimeError(f"R1 direct index mapping ratio {mapping['r1_mapping_ratio']:.3f} is below {self.config.memory_r1_min_mapping_ratio:.3f}.")
        raw_hypotheses, post_refinement_hypotheses, evaluated_hypotheses = [], [], []
        candidate_logs, round_logs = [], []
        next_id = 0
        r1_hypothesis = None
        r1_initialization = {
            "r1_initialized": 0,
            "r1_low_confidence": 0,
            "r1_memory_accepted": 0,
            "r1_support_inlier_ratio": 0.0,
            "r1_support_mean_error": float("inf"),
            "r1_posterior_delta": 0.0,
            "r1_graph_delta": 0.0,
            "r1_candidate_expansion_count": 0,
            **mapping,
        }
        if initial_pose is not None:
            initial_support = initial_support_ids if initial_support_ids is not None else torch.empty(0, dtype=torch.long, device=memory.device)
            r1_hypothesis = self._make_hypothesis(memory, next_id, -1, 0, "r1", initial_pose, initial_pose, initial_support)
            next_id += 1
            raw_hypotheses.append(r1_hypothesis)
            post_refinement_hypotheses.append(r1_hypothesis)
            evaluated_hypotheses.append(r1_hypothesis)
            support_ratio = (
                float(torch.isin(initial_support, r1_hypothesis.inlier_ids).float().mean().item())
                if initial_support.numel() else 0.0
            )
            support_error = (
                float(r1_hypothesis.residuals[initial_support].mean().item())
                if initial_support.numel() else float("inf")
            )
            memory_accepted = self._memory_update_gate(r1_hypothesis, None)
            low_confidence = not memory_accepted
            expansion_count = 0
            if not memory_accepted:
                posterior_delta = 0.0
                graph_delta = 0.0
                memory.update_basin(
                    r1_hypothesis.pose,
                    r1_hypothesis.support_signature,
                    r1_hypothesis.validation_score,
                    improved=False,
                    support_ids=initial_support,
                )
            else:
                posterior_delta = memory.update_pair_posterior(
                    r1_hypothesis.inlier_ids,
                    r1_hypothesis.verified_candidate_outliers,
                )
                graph_delta = memory.update_relation_graph(
                    r1_hypothesis.inlier_ids,
                    r1_hypothesis.verified_candidate_outliers,
                )
                memory.update_basin(
                    r1_hypothesis.pose,
                    r1_hypothesis.support_signature,
                    r1_hypothesis.validation_score,
                    improved=True,
                    support_ids=initial_support,
                )
                new_src, new_tgt, new_scores = memory.pose_guided_candidate_expansion(
                    r1_hypothesis.pose, src_features, tgt_features,
                )
                expansion_count = memory.add_candidates(new_src, new_tgt, new_scores)
            r1_initialization = {
                "r1_initialized": 1,
                "r1_low_confidence": int(low_confidence),
                "r1_memory_accepted": int(memory_accepted),
                "r1_support_inlier_ratio": support_ratio,
                "r1_support_mean_error": support_error,
                "r1_posterior_delta": posterior_delta,
                "r1_graph_delta": graph_delta,
                "r1_candidate_expansion_count": expansion_count,
                **mapping,
            }
            candidate_logs.append(self._candidate_log(
                r1_hypothesis,
                -1,
                0.0,
                bool(r1_initialization["r1_memory_accepted"]),
                "" if r1_initialization["r1_memory_accepted"] else "memory_update_gate",
            ))
        best = self._select_best(post_refinement_hypotheses)
        if r1_hypothesis is not None:
            round_logs.append({
                "round_id": 0,
                "best_hypothesis_id": r1_hypothesis.hypothesis_id,
                "best_score": r1_hypothesis.score,
                "best_validation_score": r1_hypothesis.validation_score,
                "best_search_score": r1_hypothesis.search_score,
                "round_hypothesis_id": r1_hypothesis.hypothesis_id,
                "round_score": r1_hypothesis.score,
                "round_validation_score": r1_hypothesis.validation_score,
                "round_search_score": r1_hypothesis.search_score,
                "raw_candidate_count": 1,
                "raw_generated": 0,
                "accepted_child_count": 0,
                "refinement_attempted": 0,
                "refinement_accepted": 0,
                "sampling_attempts": 0,
                "inlier_count": r1_hypothesis.inlier_count,
                "unique_inlier_ratio": r1_hypothesis.unique_inlier_ratio,
                "median_residual": r1_hypothesis.median_residual,
                "bidirectional_consistency": r1_hypothesis.bidirectional_consistency,
                "free_space_conflict_ratio": r1_hypothesis.free_space_conflict_ratio,
                "free_space_available": int(r1_hypothesis.free_space_available),
                "coverage": r1_hypothesis.coverage,
                "best_coverage": r1_hypothesis.coverage,
                "duplicate_basin_rate": None,
                "novelty": None,
                "posterior_delta": 0.0,
                "graph_delta": 0.0,
                "memory_accepted": r1_initialization["r1_memory_accepted"],
                "candidate_expansion_count": r1_initialization["r1_candidate_expansion_count"],
                "basin_count": 0,
                "stale_rounds": 0,
                "round_runtime_seconds": 0.0,
                **{key: 0 for key in REJECTION_REASON_KEYS},
            })
        stale = 0
        for round_id in range(1, self.config.memory_max_rounds + 1):
            round_started = time.perf_counter()
            best_before_round = best
            ranked = torch.argsort(memory.rank(), descending=True)
            current_round, novel = [], 0
            sampling_attempts, raw_generated = 0, 0
            rejection_counts = {key: 0 for key in REJECTION_REASON_KEYS}
            while raw_generated < self.config.memory_hypotheses_per_round and sampling_attempts < self.config.memory_max_sampling_attempts:
                seed = self._prosac_seed(ranked, round_id - 1, sampling_attempts)
                sampling_attempts += 1
                support = memory.expand_support(seed)
                support_objective = float(memory.support_objective(support).item())
                presearch_basin_penalty = float(memory.presearch_basin_penalty(memory.support_signature(support), support).item())
                diagnostics = memory.support_diagnostics(support)
                constraint_reasons = memory.support_constraint_reasons(diagnostics)
                hard_rejection_reasons = memory.support_hard_rejection_reasons(diagnostics)
                structure_penalty = memory.structure_penalty(diagnostics)
                for key in constraint_reasons:
                    rejection_counts[key] += 1
                attempt_log = {
                    "round_id": round_id,
                    "attempt_index": sampling_attempts,
                    "hypothesis_id": -1,
                    "parent_id": -1,
                    "stage": "attempt",
                    "seed_id": seed,
                    "support_count": int(support.numel()),
                    "support_objective": support_objective,
                    "presearch_basin_penalty": presearch_basin_penalty,
                    "accepted": 0,
                    "raw_generated": 0,
                    "duplicate_pose": 0,
                    "structure_penalty": structure_penalty,
                    "constraint_reasons": "|".join(constraint_reasons),
                    **diagnostics,
                }
                if memory.is_degenerate(support):
                    candidate_logs.append({
                        **attempt_log,
                        "reject_reason": "|".join(hard_rejection_reasons) if hard_rejection_reasons else "unclassified_degenerate",
                    })
                    continue
                src, tgt = memory.correspondence_points(support)
                try:
                    pose_raw = self._weighted_rigid(src[0], tgt[0], memory.fixed_estimation_weights(support))
                    if not bool(torch.isfinite(pose_raw).all()):
                        raise RuntimeError("weighted SVD returned a non-finite pose")
                except RuntimeError:
                    rejection_counts["svd_failed"] += 1
                    candidate_logs.append({
                        **attempt_log,
                        "reject_reason": "svd_failed",
                    })
                    continue
                raw = self._make_hypothesis(
                    memory,
                    next_id,
                    -1,
                    round_id,
                    "raw",
                    pose_raw,
                    pose_raw,
                    support,
                    search_score=support_objective - structure_penalty,
                    structure_penalty=structure_penalty,
                )
                next_id += 1
                raw_hypotheses.append(raw)
                post_refinement_hypotheses.append(raw)
                evaluated_hypotheses.append(raw)
                current_round.append(raw)
                raw_generated += 1
                candidate_logs.append({
                    **attempt_log,
                    "hypothesis_id": raw.hypothesis_id,
                    "accepted": 1,
                    "raw_generated": 1,
                    "reject_reason": "",
                })
                raw_log = self._candidate_log(raw, seed, support_objective, True, "")
                raw_log["presearch_basin_penalty"] = presearch_basin_penalty
                candidate_logs.append(raw_log)
                if self.config.memory_use_basin:
                    _, record = memory.basin_record(raw.pose)
                    novel += int(record is None)
                pose_refined = self._robust_refine(memory, raw.pose)
                child = self._make_hypothesis(
                    memory,
                    next_id,
                    raw.hypothesis_id,
                    round_id,
                    "refinement_child",
                    raw.pose_raw,
                    pose_refined,
                    support,
                    search_score=support_objective - structure_penalty,
                    structure_penalty=structure_penalty,
                )
                next_id += 1
                evaluated_hypotheses.append(child)
                accepted, reject_reason = self._accept_refined_child(raw, child)
                raw.refinement_accepted = accepted
                raw.refinement_reject_reason = reject_reason
                if accepted:
                    child.refinement_accepted = True
                    post_refinement_hypotheses.append(child)
                    current_round.append(child)
                    child_log = self._candidate_log(child, seed, support_objective, True, "")
                    child_log["presearch_basin_penalty"] = presearch_basin_penalty
                    candidate_logs.append(child_log)
                else:
                    child_log = self._candidate_log(child, seed, support_objective, False, reject_reason)
                    child_log["presearch_basin_penalty"] = presearch_basin_penalty
                    candidate_logs.append(child_log)
            if not current_round:
                stale += 1
                round_logs.append({
                    "round_id": round_id,
                    "best_hypothesis_id": best.hypothesis_id,
                    "best_score": best.score,
                    "best_validation_score": best.validation_score,
                    "best_search_score": best.search_score,
                    "round_hypothesis_id": best.hypothesis_id,
                    "round_score": float("-inf"),
                    "round_validation_score": float("-inf"),
                    "round_search_score": float("-inf"),
                    "raw_candidate_count": 0,
                    "raw_generated": 0,
                    "accepted_child_count": 0,
                    "refinement_attempted": 0,
                    "refinement_accepted": 0,
                    "sampling_attempt_count": sampling_attempts,
                    "sampling_attempts": sampling_attempts,
                    "sampling_budget_exhausted": 1,
                    "inlier_count": best.inlier_count,
                    "unique_inlier_ratio": best.unique_inlier_ratio,
                    "median_residual": best.median_residual,
                    "bidirectional_consistency": best.bidirectional_consistency,
                    "free_space_conflict_ratio": best.free_space_conflict_ratio,
                    "free_space_available": int(best.free_space_available),
                    "coverage": best.coverage,
                    "best_coverage": best.coverage,
                    "duplicate_basin_rate": None,
                    "novelty": None,
                    "posterior_delta": 0.0,
                    "graph_delta": 0.0,
                    "basin_count": len(memory.basins),
                    "stale_rounds": stale,
                    "round_runtime_seconds": time.perf_counter() - round_started,
                    **rejection_counts,
                })
                if not self.config.memory_fixed_budget_mode and stale >= self.config.memory_patience:
                    break
                continue
            current = self._select_best(current_round)
            memory_accepted = self._memory_update_gate(current, best_before_round)
            posterior_delta, graph_delta = 0.0, 0.0
            expansion_count = 0
            if memory_accepted:
                best, stale = current, 0
                posterior_delta = memory.update_pair_posterior(
                    current.inlier_ids,
                    current.verified_candidate_outliers,
                )
                graph_delta = memory.update_relation_graph(
                    current.inlier_ids,
                    current.verified_candidate_outliers,
                )
                memory.update_basin(current.pose, current.support_signature, current.validation_score, improved=True, support_ids=current.support_ids)
                new_src, new_tgt, new_scores = memory.pose_guided_candidate_expansion(
                    current.pose, src_features, tgt_features,
                )
                expansion_count = memory.add_candidates(new_src, new_tgt, new_scores)
            else:
                stale += 1
                memory.update_basin(current.pose, current.support_signature, current.validation_score, improved=False, support_ids=current.support_ids)
            memory.compress()
            current_coverage = current.coverage
            best_coverage = best.coverage
            rotation_delta, translation_delta = self._pose_distance(best.pose, best_before_round.pose) if best_before_round is not None else (float("inf"), float("inf"))
            score_delta = float("inf") if best_before_round is None else abs(best.validation_score - best_before_round.validation_score) / max(abs(best_before_round.validation_score), 1e-8)
            novelty = novel / max(1, sum(item.stage == "raw" for item in current_round)) if self.config.memory_use_basin else None
            round_logs.append({
                "round_id": round_id,
                "best_hypothesis_id": best.hypothesis_id,
                "best_score": best.score,
                "best_validation_score": best.validation_score,
                "best_search_score": best.search_score,
                "round_hypothesis_id": current.hypothesis_id,
                "round_score": current.score,
                "round_validation_score": current.validation_score,
                "round_search_score": current.search_score,
                "raw_candidate_count": sum(item.stage == "raw" for item in current_round),
                "raw_generated": raw_generated,
                "accepted_child_count": sum(item.stage == "refinement_child" for item in current_round),
                "refinement_attempted": raw_generated,
                "refinement_accepted": sum(item.stage == "refinement_child" for item in current_round),
                "sampling_attempt_count": sampling_attempts,
                "sampling_attempts": sampling_attempts,
                "sampling_budget_exhausted": int(raw_generated < self.config.memory_hypotheses_per_round),
                "inlier_count": current.inlier_count,
                "unique_inlier_ratio": current.unique_inlier_ratio,
                "median_residual": current.median_residual,
                "bidirectional_consistency": current.bidirectional_consistency,
                "free_space_conflict_ratio": current.free_space_conflict_ratio,
                "free_space_available": int(current.free_space_available),
                "coverage": current_coverage,
                "best_coverage": best_coverage,
                "duplicate_basin_rate": 1.0 - novelty if novelty is not None else None,
                "novelty": novelty,
                "posterior_delta": posterior_delta,
                "graph_delta": graph_delta,
                "memory_accepted": int(memory_accepted),
                "candidate_expansion_count": expansion_count,
                "basin_count": len(memory.basins),
                "stale_rounds": stale,
                "round_runtime_seconds": time.perf_counter() - round_started,
                **rejection_counts,
            })
            if not self.config.memory_fixed_budget_mode and self._strong_stop(best, memory.src_points.shape[0]):
                break
            converged = (
                stale >= self.config.memory_patience
                and score_delta < self.config.memory_score_epsilon
                and rotation_delta < self.config.memory_pose_epsilon_rotation_deg
                and translation_delta < self.config.memory_pose_epsilon_translation_multiplier * self.config.memory_coverage_voxel_size
                and posterior_delta + graph_delta < self.config.memory_delta_threshold
            )
            if self.config.memory_use_basin:
                converged = converged and novelty < self.config.memory_novelty_threshold
            if not self.config.memory_fixed_budget_mode and converged:
                break
        if best is None:
            identity = torch.eye(4, dtype=memory.dtype, device=memory.device)[None]
            return MemorySearchResult(identity, None, r1_hypothesis, r1_initialization, raw_hypotheses, post_refinement_hypotheses, evaluated_hypotheses, round_logs, candidate_logs, memory.summary(), memory.src_indices, memory.tgt_indices)
        return MemorySearchResult(best.pose, best, r1_hypothesis, r1_initialization, raw_hypotheses, post_refinement_hypotheses, evaluated_hypotheses, round_logs, candidate_logs, memory.summary(), memory.src_indices, memory.tgt_indices)
