from dataclasses import dataclass
import math

import torch

from correspondence_memory import CorrespondenceMemory


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
    score: float
    evidence_score: float
    support_signature: torch.Tensor
    basin_key: tuple | None
    refinement_accepted: bool = False
    refinement_reject_reason: str = ""

    @property
    def pose(self):
        return self.pose_local_refined


@dataclass
class MemorySearchResult:
    pose: torch.Tensor
    best: MemoryHypothesis | None
    r1_hypothesis: MemoryHypothesis | None
    raw_hypotheses: list
    post_refinement_hypotheses: list
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

    def _robust_refine(self, memory, pose):
        src, tgt = memory.correspondence_points(torch.arange(memory.count, device=memory.device))
        src, tgt = src[0], tgt[0]
        current = pose
        for _ in range(self.config.memory_tls_iters):
            residuals = torch.linalg.norm(self._transform(src, current) - tgt, dim=1)
            truncated = (residuals < self.config.memory_tls_threshold).to(memory.dtype)
            weights = memory.estimation_weights(torch.arange(memory.count, device=memory.device)) * truncated
            if int((weights > 0).sum().item()) < self.config.memory_support_min:
                break
            current = self._weighted_rigid(src, tgt, weights)
        return current

    def _verify(self, memory, pose):
        src, tgt = memory.correspondence_points(torch.arange(memory.count, device=memory.device))
        residuals = torch.linalg.norm(self._transform(src[0], pose) - tgt[0], dim=1)
        return residuals < self.config.memory_inlier_threshold, residuals

    def _coverage(self, memory, candidate_ids):
        if candidate_ids.numel() == 0:
            return 0.0
        points = memory.src_points[memory.src_indices[candidate_ids]]
        voxels = torch.floor(points / self.config.memory_coverage_voxel_size).long()
        return float(torch.unique(voxels, dim=0).shape[0] / memory.source_voxel_count)

    def _pose_score(self, memory, pose, inliers, residuals, signature):
        inlier_ids = torch.where(inliers)[0]
        inlier_weight = float(memory.estimation_weights(inlier_ids).sum().item())
        mean_error = float(residuals[inliers].mean().item()) if inlier_ids.numel() else float("inf")
        coverage = self._coverage(memory, inlier_ids)
        basin_bonus = memory.basin_bonus(pose, signature)
        evidence_score = (
            self.config.memory_lambda_inlier * inlier_weight
            - self.config.memory_lambda_error * mean_error
            + self.config.memory_lambda_coverage * coverage
        )
        return float(evidence_score + basin_bonus), float(evidence_score), mean_error, coverage

    def _signature_support(self, memory, inliers):
        candidate_ids = torch.where(inliers)[0]
        if candidate_ids.numel() <= self.config.memory_support_max:
            return candidate_ids
        rank = memory.rank()[candidate_ids]
        return candidate_ids[torch.topk(rank, k=self.config.memory_support_max).indices]

    def _make_hypothesis(self, memory, hypothesis_id, parent_id, round_id, stage, pose_raw, pose, support_ids):
        inliers, residuals = self._verify(memory, pose)
        signature_ids = support_ids if support_ids.numel() else self._signature_support(memory, inliers)
        signature = memory.support_signature(signature_ids)
        score, evidence_score, _, _ = self._pose_score(memory, pose, inliers, residuals, signature)
        basin_key = memory.basin_key(pose) if self.config.memory_use_basin else None
        return MemoryHypothesis(
            hypothesis_id=hypothesis_id,
            parent_id=parent_id,
            round_id=round_id,
            stage=stage,
            pose_raw=pose_raw,
            pose_local_refined=pose,
            support_ids=support_ids,
            inlier_ids=torch.where(inliers)[0],
            residuals=residuals,
            score=score,
            evidence_score=evidence_score,
            support_signature=signature,
            basin_key=basin_key,
        )

    def _prosac_seed(self, ranked, round_id, hypothesis_index):
        count = ranked.numel()
        fraction = min(1.0, self.config.memory_prosac_initial_fraction + round_id * self.config.memory_prosac_growth)
        prefix = min(count, max(self.config.memory_support_min, int(math.ceil(fraction * count))))
        offset = (hypothesis_index * 7919 + round_id * 104729) % prefix
        return int(ranked[offset].item())

    def _strong_stop(self, hypothesis, coverage):
        inlier_target = max(
            self.config.memory_strong_stop_min_inliers,
            int(math.ceil(self.config.memory_strong_stop_inlier_fraction * hypothesis.residuals.numel())),
        )
        mean_error = float(hypothesis.residuals[hypothesis.inlier_ids].mean().item()) if hypothesis.inlier_ids.numel() else float("inf")
        return (
            hypothesis.inlier_ids.numel() >= inlier_target
            and mean_error <= self.config.memory_strong_stop_error_ratio * self.config.memory_inlier_threshold
            and coverage >= self.config.memory_strong_stop_coverage
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
        score_improved = child.evidence_score >= parent.evidence_score + self.config.memory_refine_min_score_improvement
        if not within_trust_region:
            return False, "outside_trust_region"
        if not score_improved:
            return False, "insufficient_evidence_improvement"
        return True, ""

    @staticmethod
    def _candidate_log(hypothesis, seed_id, support_objective, accepted, reject_reason):
        inlier_count = int(hypothesis.inlier_ids.numel())
        mean_error = float(hypothesis.residuals[hypothesis.inlier_ids].mean().item()) if inlier_count else float("inf")
        return {
            "round_id": hypothesis.round_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "parent_id": hypothesis.parent_id,
            "stage": hypothesis.stage,
            "seed_id": seed_id,
            "support_count": int(hypothesis.support_ids.numel()),
            "inlier_count": inlier_count,
            "mean_inlier_error": mean_error,
            "score": hypothesis.score,
            "evidence_score": hypothesis.evidence_score,
            "basin_key": list(hypothesis.basin_key) if hypothesis.basin_key is not None else [],
            "support_objective": support_objective,
            "accepted": int(accepted),
            "reject_reason": reject_reason,
        }

    def run(self, src_points, tgt_points, src_features, tgt_features, initial_pose=None):
        if self.config.memory_require_r1_cache and initial_pose is None:
            raise ValueError("memory_graph requires the fixed R1 pose as a permanent round-0 parent.")
        memory = CorrespondenceMemory(src_points, tgt_points, src_features, tgt_features, self.config)
        raw_hypotheses, post_refinement_hypotheses, candidate_logs, round_logs = [], [], [], []
        next_id = 0
        r1_hypothesis = None
        if initial_pose is not None:
            initial_support = torch.empty(0, dtype=torch.long, device=memory.device)
            r1_hypothesis = self._make_hypothesis(memory, next_id, -1, 0, "r1", initial_pose, initial_pose, initial_support)
            next_id += 1
            raw_hypotheses.append(r1_hypothesis)
            post_refinement_hypotheses.append(r1_hypothesis)
            candidate_logs.append(self._candidate_log(r1_hypothesis, -1, 0.0, True, ""))
        best = max(post_refinement_hypotheses, key=lambda item: item.score) if post_refinement_hypotheses else None
        if r1_hypothesis is not None:
            r1_coverage = self._coverage(memory, r1_hypothesis.inlier_ids)
            round_logs.append({
                "round_id": 0,
                "best_hypothesis_id": r1_hypothesis.hypothesis_id,
                "best_score": r1_hypothesis.score,
                "round_hypothesis_id": r1_hypothesis.hypothesis_id,
                "round_score": r1_hypothesis.score,
                "raw_candidate_count": 1,
                "accepted_child_count": 0,
                "inlier_count": int(r1_hypothesis.inlier_ids.numel()),
                "coverage": r1_coverage,
                "best_coverage": r1_coverage,
                "duplicate_basin_rate": None,
                "novelty": None,
                "posterior_delta": 0.0,
                "graph_delta": 0.0,
                "basin_count": 0,
                "stale_rounds": 0,
            })
        stale = 0
        for round_id in range(1, self.config.memory_max_rounds + 1):
            best_before_round = best
            ranked = torch.argsort(memory.rank(), descending=True)
            current_round, novel = [], 0
            for hypothesis_index in range(self.config.memory_hypotheses_per_round):
                seed = self._prosac_seed(ranked, round_id - 1, hypothesis_index)
                support = memory.expand_support(seed)
                support_objective = float(memory.support_objective(support).item())
                presearch_basin_penalty = float(memory.presearch_basin_penalty(memory.support_signature(support)).item())
                if memory.is_degenerate(support):
                    candidate_logs.append({
                        "round_id": round_id,
                        "hypothesis_id": -1,
                        "parent_id": -1,
                        "stage": "raw",
                        "seed_id": seed,
                        "support_count": int(support.numel()),
                        "support_objective": support_objective,
                        "presearch_basin_penalty": presearch_basin_penalty,
                        "accepted": 0,
                        "reject_reason": "degenerate_support",
                    })
                    continue
                src, tgt = memory.correspondence_points(support)
                pose_raw = self._weighted_rigid(src[0], tgt[0], memory.estimation_weights(support))
                raw = self._make_hypothesis(memory, next_id, -1, round_id, "raw", pose_raw, pose_raw, support)
                next_id += 1
                raw_hypotheses.append(raw)
                post_refinement_hypotheses.append(raw)
                current_round.append(raw)
                raw_log = self._candidate_log(raw, seed, support_objective, True, "")
                raw_log["presearch_basin_penalty"] = presearch_basin_penalty
                candidate_logs.append(raw_log)
                if self.config.memory_use_basin:
                    _, record = memory.basin_record(raw.pose)
                    novel += int(record is None)
                pose_refined = self._robust_refine(memory, raw.pose)
                child = self._make_hypothesis(memory, next_id, raw.hypothesis_id, round_id, "refinement_child", raw.pose_raw, pose_refined, support)
                next_id += 1
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
                if stale >= self.config.memory_patience:
                    break
                continue
            current = max(current_round, key=lambda item: item.score)
            improved = best_before_round is None or current.score > best_before_round.score
            if improved:
                best, stale = current, 0
            else:
                stale += 1
            inliers = torch.zeros(memory.count, dtype=torch.bool, device=memory.device)
            inliers[current.inlier_ids] = True
            posterior_delta = memory.update_pair_posterior(current.support_ids, inliers)
            graph_delta = memory.update_relation_graph(current.support_ids, inliers, current.residuals)
            memory.update_basin(current.pose, current.support_signature, current.score, improved)
            memory.compress()
            current_coverage = self._coverage(memory, current.inlier_ids)
            best_coverage = self._coverage(memory, best.inlier_ids)
            rotation_delta, translation_delta = self._pose_distance(best.pose, best_before_round.pose) if best_before_round is not None else (float("inf"), float("inf"))
            score_delta = float("inf") if best_before_round is None else abs(best.score - best_before_round.score) / max(abs(best_before_round.score), 1e-8)
            novelty = novel / max(1, sum(item.stage == "raw" for item in current_round)) if self.config.memory_use_basin else None
            round_logs.append({
                "round_id": round_id,
                "best_hypothesis_id": best.hypothesis_id,
                "best_score": best.score,
                "round_hypothesis_id": current.hypothesis_id,
                "round_score": current.score,
                "raw_candidate_count": sum(item.stage == "raw" for item in current_round),
                "accepted_child_count": sum(item.stage == "refinement_child" for item in current_round),
                "inlier_count": int(current.inlier_ids.numel()),
                "coverage": current_coverage,
                "best_coverage": best_coverage,
                "duplicate_basin_rate": 1.0 - novelty if novelty is not None else None,
                "novelty": novelty,
                "posterior_delta": posterior_delta,
                "graph_delta": graph_delta,
                "basin_count": len(memory.basins),
                "stale_rounds": stale,
            })
            if self._strong_stop(best, best_coverage):
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
            if converged:
                break
        if best is None:
            identity = torch.eye(4, dtype=memory.dtype, device=memory.device)[None]
            return MemorySearchResult(identity, None, r1_hypothesis, raw_hypotheses, post_refinement_hypotheses, round_logs, candidate_logs, memory.summary(), memory.src_indices, memory.tgt_indices)
        return MemorySearchResult(best.pose, best, r1_hypothesis, raw_hypotheses, post_refinement_hypotheses, round_logs, candidate_logs, memory.summary(), memory.src_indices, memory.tgt_indices)
