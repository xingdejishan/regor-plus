from dataclasses import dataclass
import math

import torch

from correspondence_memory import CorrespondenceMemory


@dataclass
class MemoryHypothesis:
    hypothesis_id: int
    round_id: int
    pose_raw: torch.Tensor
    pose_local_refined: torch.Tensor
    support_ids: torch.Tensor
    inlier_ids: torch.Tensor
    residuals: torch.Tensor
    score: float
    support_signature: torch.Tensor
    basin_key: tuple
    generation_mode: str = "memory_graph"

    @property
    def pose(self):
        return self.pose_local_refined


@dataclass
class MemorySearchResult:
    pose: torch.Tensor
    best: MemoryHypothesis | None
    hypotheses: list
    round_logs: list
    candidate_logs: list
    memory_summary: dict
    candidate_src_indices: torch.Tensor
    candidate_tgt_indices: torch.Tensor


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
            weights = memory.reliability * truncated
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
        return float(torch.unique(voxels, dim=0).shape[0] / max(1, points.shape[0]))

    def _pose_score(self, memory, pose, inliers, residuals, signature):
        inlier_ids = torch.where(inliers)[0]
        inlier_weight = float(memory.reliability[inliers].sum().item())
        mean_error = float(residuals[inliers].mean().item()) if inlier_ids.numel() else float("inf")
        coverage = self._coverage(memory, inlier_ids)
        basin_bonus = memory.basin_bonus(pose, signature)
        score = (
            self.config.memory_lambda_inlier * inlier_weight
            - self.config.memory_lambda_error * mean_error
            + self.config.memory_lambda_coverage * coverage
            + basin_bonus
        )
        return float(score), mean_error, coverage

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

    def run(self, src_points, tgt_points, src_features, tgt_features):
        memory = CorrespondenceMemory(src_points, tgt_points, src_features, tgt_features, self.config)
        hypotheses, candidate_logs, round_logs = [], [], []
        best, stale = None, 0
        next_id = 0
        for round_id in range(self.config.memory_max_rounds):
            best_before_round = best
            ranked = torch.argsort(memory.rank(), descending=True)
            round_hypotheses = []
            novel = 0
            for hypothesis_index in range(self.config.memory_hypotheses_per_round):
                seed = self._prosac_seed(ranked, round_id, hypothesis_index)
                support = memory.expand_support(seed)
                if memory.is_degenerate(support):
                    candidate_logs.append({
                        "round_id": round_id,
                        "hypothesis_id": -1,
                        "seed_id": seed,
                        "support_count": int(support.numel()),
                        "accepted": 0,
                        "reject_reason": "degenerate_support",
                    })
                    continue
                src, tgt = memory.correspondence_points(support)
                pose_raw = self._weighted_rigid(src[0], tgt[0], memory.reliability[support])
                pose = self._robust_refine(memory, pose_raw)
                inliers, residuals = self._verify(memory, pose)
                signature = memory.support_signature(support)
                score, mean_error, coverage = self._pose_score(memory, pose, inliers, residuals, signature)
                basin_key, basin_record = memory.basin_record(pose)
                novel += int(basin_record is None)
                hypothesis = MemoryHypothesis(
                    hypothesis_id=next_id,
                    round_id=round_id,
                    pose_raw=pose_raw,
                    pose_local_refined=pose,
                    support_ids=support,
                    inlier_ids=torch.where(inliers)[0],
                    residuals=residuals,
                    score=score,
                    support_signature=signature,
                    basin_key=basin_key,
                )
                next_id += 1
                round_hypotheses.append(hypothesis)
                candidate_logs.append({
                    "round_id": round_id,
                    "hypothesis_id": hypothesis.hypothesis_id,
                    "seed_id": seed,
                    "support_count": int(support.numel()),
                    "inlier_count": int(hypothesis.inlier_ids.numel()),
                    "mean_inlier_error": mean_error,
                    "coverage": coverage,
                    "support_objective": float(memory.support_objective(support).item()),
                    "score": score,
                    "basin_key": list(basin_key),
                    "accepted": 1,
                    "reject_reason": "",
                })
            if not round_hypotheses:
                stale += 1
                if stale >= self.config.memory_patience:
                    break
                continue
            current = max(round_hypotheses, key=lambda item: item.score)
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
            hypotheses.extend(round_hypotheses)
            coverage = self._coverage(memory, current.inlier_ids)
            best_coverage = self._coverage(memory, best.inlier_ids)
            rotation_delta, translation_delta = self._pose_distance(best.pose, best_before_round.pose) if best_before_round is not None else (float("inf"), float("inf"))
            score_delta = float("inf") if best_before_round is None else abs(best.score - best_before_round.score) / max(abs(best_before_round.score), 1e-8)
            novelty = novel / max(1, len(round_hypotheses))
            round_logs.append({
                "round_id": round_id,
                "best_hypothesis_id": best.hypothesis_id,
                "best_score": best.score,
                "round_score": current.score,
                "inlier_count": int(current.inlier_ids.numel()),
                "coverage": coverage,
                "best_coverage": best_coverage,
                "duplicate_basin_rate": 1.0 - novelty,
                "novelty": novelty,
                "posterior_delta": posterior_delta,
                "graph_delta": graph_delta,
                "basin_count": len(memory.basins),
                "stale_rounds": stale,
            })
            if self._strong_stop(best, best_coverage):
                break
            if (
                stale >= self.config.memory_patience
                and score_delta < self.config.memory_score_epsilon
                and rotation_delta < self.config.memory_pose_epsilon_rotation_deg
                and translation_delta < self.config.memory_pose_epsilon_translation_multiplier * self.config.memory_coverage_voxel_size
                and novelty < self.config.memory_novelty_threshold
                and posterior_delta + graph_delta < self.config.memory_delta_threshold
            ):
                break
        if best is None:
            identity = torch.eye(4, dtype=memory.dtype, device=memory.device)[None]
            return MemorySearchResult(identity, None, hypotheses, round_logs, candidate_logs, memory.summary(), memory.src_indices, memory.tgt_indices)
        return MemorySearchResult(best.pose, best, hypotheses, round_logs, candidate_logs, memory.summary(), memory.src_indices, memory.tgt_indices)
