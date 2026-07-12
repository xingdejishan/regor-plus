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
                unique_ids = memory.one_to_one_filter(active_ids)
                unique_weights = torch.zeros_like(weights)
                unique_weights[unique_ids] = weights[unique_ids]
                weights = unique_weights
            if int((weights > 0).sum().item()) < self.config.memory_support_min:
                break
            current = self._weighted_rigid(src, tgt, weights)
        return current

    def _verify(self, memory, pose):
        """Verify pose using one-to-one filtered correspondences.

        Returns (unique_inliers, residuals, all_inlier_ids).
        """
        src, tgt = memory.correspondence_points(torch.arange(memory.count, device=memory.device))
        residuals = torch.linalg.norm(self._transform(src[0], pose) - tgt[0], dim=1)
        inlier_mask = residuals < self.config.memory_inlier_threshold
        inlier_ids = torch.where(inlier_mask)[0]
        if inlier_ids.numel() == 0:
            unique_inliers = torch.zeros(memory.count, dtype=torch.bool, device=memory.device)
            return unique_inliers, residuals, inlier_ids
        unique_support = memory.one_to_one_filter(inlier_ids)
        unique_inliers = torch.zeros(memory.count, dtype=torch.bool, device=memory.device)
        unique_inliers[unique_support] = True
        return unique_inliers, residuals, inlier_ids

    def _one_to_one_match(self, memory, candidate_ids):
        """Apply one-to-one filter to candidate correspondence IDs."""
        return memory.one_to_one_filter(candidate_ids)

    def _vdce_validation_score(self, memory, unique_inliers, residuals, pose=None):
        """Independent validation score using unique inlier ratio.

        Uses normalized metrics:
        - unique_inlier_ratio: one-to-one inliers / visible source points
        - normalized_median_residual: median residual / normalize scale
        - spatial_coverage: voxel coverage of inlier source points
        - bidirectional_consistency: forward/backward transform agreement
        """
        inlier_ids = torch.where(unique_inliers)[0]
        unique_count = int(inlier_ids.numel())
        if not unique_count:
            return float("-inf"), 0, float("inf"), 0.0, 0.0, 0.0
        visible_sources = int(torch.unique(memory.src_indices[inlier_ids]).numel()) if unique_count else 0
        if visible_sources == 0:
            return float("-inf"), 0, float("inf"), 0.0, 0.0, 0.0
        unique_inlier_ratio = unique_count / max(1, visible_sources)
        median_residual = float(residuals[inlier_ids].median().item())
        normalized_residual = median_residual / self.config.memory_vdce_normalize_residual_scale
        coverage = self._coverage(memory, inlier_ids)
        bidirectional = 0.0
        if pose is not None:
            src_pts = memory.src_points[memory.src_indices[inlier_ids]]
            tgt_pts = memory.tgt_points[memory.tgt_indices[inlier_ids]]
            forward = self._transform(src_pts, pose)
            reverse = self._transform(tgt_pts, self._invert_pose(pose))
            forward_error = torch.linalg.norm(forward - tgt_pts, dim=1).mean()
            reverse_error = torch.linalg.norm(reverse - src_pts, dim=1).mean()
            scale = torch.linalg.norm(src_pts.mean(dim=0) - tgt_pts.mean(dim=0)).clamp_min(1e-8)
            bidirectional = float(torch.exp(-(forward_error + reverse_error) / (2.0 * scale)).item())
        w = self.config
        score = (
            w.memory_vdce_w_r * unique_inlier_ratio
            - w.memory_vdce_w_e * min(normalized_residual, 1.0)
            + w.memory_vdce_w_c * coverage
            + w.memory_vdce_w_b * bidirectional
        )
        return float(score), unique_count, median_residual, coverage, unique_inlier_ratio, bidirectional

    def _invert_pose(self, pose):
        matrix = pose[0] if pose.ndim == 3 else pose
        inv = torch.eye(4, dtype=matrix.dtype, device=matrix.device)
        inv[:3, :3] = matrix[:3, :3].transpose(0, 1)
        inv[:3, 3] = -inv[:3, :3] @ matrix[:3, 3]
        return inv[None]

    def _memory_update_gate(self, current_eval, best_eval):
        """Gate memory updates: only accept if significantly better than global best."""
        if best_eval is None:
            return True
        score_gain = current_eval.validation_score - best_eval.validation_score
        return (
            score_gain > self.config.memory_vdce_min_score_margin
            and current_eval.unique_inlier_ratio > self.config.memory_vdce_min_inlier_ratio
            and current_eval.coverage > self.config.memory_vdce_min_coverage
            and current_eval.median_residual < self.config.memory_vdce_max_median_residual
        )

    def _coverage(self, memory, candidate_ids):
        if candidate_ids.numel() == 0:
            return 0.0
        points = memory.src_points[memory.src_indices[candidate_ids]]
        voxels = torch.floor(points / self.config.memory_coverage_voxel_size).long()
        return float(torch.unique(voxels, dim=0).shape[0] / memory.source_voxel_count)

    def _validation_score(self, memory, inliers, residuals):
        inlier_ids = torch.where(inliers)[0]
        inlier_count = int(inlier_ids.numel())
        if not inlier_count:
            return float("-inf"), 0, float("inf"), 0.0
        mean_error = float(residuals[inliers].mean().item())
        coverage = self._coverage(memory, inlier_ids)
        validation_score = (
            self.config.memory_lambda_inlier * float(inlier_count)
            - self.config.memory_lambda_error * mean_error
            + self.config.memory_lambda_coverage * coverage
        )
        return float(validation_score), inlier_count, mean_error, coverage

    def _signature_support(self, memory, inliers):
        candidate_ids = torch.where(inliers)[0]
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
        unique_inliers, residuals, all_inlier_ids = self._verify(memory, pose)
        signature_ids = support_ids if support_ids.numel() else self._signature_support(memory, unique_inliers)
        signature = memory.support_signature(signature_ids)
        vdce_score, unique_inlier_count, median_residual, coverage, unique_inlier_ratio, bidirectional = self._vdce_validation_score(
            memory, unique_inliers, residuals, pose,
        )
        validation_score = vdce_score
        mean_error = float(residuals[unique_inliers].mean().item()) if unique_inliers.any() else float("inf")
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
            inlier_ids=torch.where(unique_inliers)[0],
            residuals=residuals,
            validation_score=validation_score,
            search_score=float(search_score),
            inlier_count=unique_inlier_count,
            mean_inlier_error=mean_error,
            coverage=coverage,
            unique_inlier_ratio=unique_inlier_ratio,
            median_residual=median_residual,
            bidirectional_consistency=bidirectional,
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

    def _strong_stop(self, hypothesis):
        inlier_target = max(
            self.config.memory_strong_stop_min_inliers,
            int(math.ceil(self.config.memory_strong_stop_inlier_fraction * hypothesis.residuals.numel())),
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
            "r1_support_inlier_ratio": 0.0,
            "r1_support_mean_error": float("inf"),
            "r1_posterior_delta": 0.0,
            "r1_graph_delta": 0.0,
            **mapping,
        }
        if initial_pose is not None:
            initial_support = initial_support_ids if initial_support_ids is not None else torch.empty(0, dtype=torch.long, device=memory.device)
            r1_hypothesis = self._make_hypothesis(memory, next_id, -1, 0, "r1", initial_pose, initial_pose, initial_support)
            next_id += 1
            raw_hypotheses.append(r1_hypothesis)
            post_refinement_hypotheses.append(r1_hypothesis)
            evaluated_hypotheses.append(r1_hypothesis)
            candidate_logs.append(self._candidate_log(r1_hypothesis, -1, 0.0, True, ""))
            if initial_support.numel():
                support_inliers = torch.isin(initial_support, r1_hypothesis.inlier_ids)
                support_ratio = float(support_inliers.float().mean().item())
                support_error = float(r1_hypothesis.residuals[initial_support].mean().item())
                low_confidence = support_ratio < self.config.memory_r1_low_confidence_inlier_ratio or support_error > self.config.memory_r1_low_confidence_error_ratio * self.config.memory_inlier_threshold
                inliers = torch.zeros(
                    memory.count,
                    dtype=torch.bool,
                    device=memory.device,
                )
                inliers[r1_hypothesis.inlier_ids] = True
                if low_confidence:
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
                        initial_support,
                        inliers,
                    )
                    graph_delta = memory.update_relation_graph(
                        initial_support,
                        inliers,
                        r1_hypothesis.residuals,
                    )
                    memory.update_basin(
                        r1_hypothesis.pose,
                        r1_hypothesis.support_signature,
                        r1_hypothesis.validation_score,
                        improved=True,
                        support_ids=initial_support,
                    )
                r1_initialization = {
                    "r1_initialized": 1,
                    "r1_low_confidence": int(low_confidence),
                    "r1_support_inlier_ratio": support_ratio,
                    "r1_support_mean_error": support_error,
                    "r1_posterior_delta": posterior_delta,
                    "r1_graph_delta": graph_delta,
                    **mapping,
                }
        best = self._select_best(post_refinement_hypotheses)
        if r1_hypothesis is not None:
            r1_coverage = self._coverage(memory, r1_hypothesis.inlier_ids)
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
                "inlier_count": int(r1_hypothesis.inlier_ids.numel()),
                "coverage": r1_coverage,
                "best_coverage": r1_coverage,
                "duplicate_basin_rate": None,
                "novelty": None,
                "posterior_delta": 0.0,
                "graph_delta": 0.0,
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
                    search_score=support_objective,
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
                    search_score=support_objective,
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
                    "inlier_count": int(best.inlier_ids.numel()),
                    "coverage": self._coverage(memory, best.inlier_ids),
                    "best_coverage": self._coverage(memory, best.inlier_ids),
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
            improved = best_before_round is None or current.validation_score > best_before_round.validation_score
            if improved:
                best, stale = current, 0
            else:
                stale += 1
            memory_accepted = self._memory_update_gate(current, best_before_round)
            posterior_delta, graph_delta = 0.0, 0.0
            expansion_count = 0
            if memory_accepted and improved:
                inliers = torch.zeros(memory.count, dtype=torch.bool, device=memory.device)
                inliers[current.inlier_ids] = True
                posterior_delta = memory.update_pair_posterior(current.support_ids, inliers)
                graph_delta = memory.update_relation_graph(current.support_ids, inliers, current.residuals)
                memory.update_basin(current.pose, current.support_signature, current.validation_score, improved=True, support_ids=current.support_ids)
                new_src, new_tgt, new_scores = memory.pose_guided_candidate_expansion(
                    current.pose, src_features, tgt_features,
                )
                expansion_count = memory.add_candidates(new_src, new_tgt, new_scores)
            else:
                memory.update_basin(current.pose, current.support_signature, current.validation_score, improved=False, support_ids=current.support_ids)
            memory.compress()
            current_coverage = self._coverage(memory, current.inlier_ids)
            best_coverage = self._coverage(memory, best.inlier_ids)
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
                "inlier_count": int(current.inlier_ids.numel()),
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
            if not self.config.memory_fixed_budget_mode and self._strong_stop(best):
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
