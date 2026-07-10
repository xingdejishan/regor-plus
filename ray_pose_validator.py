from dataclasses import dataclass

from ray_evidence import bidirectional_ray_evaluation, evaluate_pose_batch, ray_signature


@dataclass(frozen=True)
class PoseValidation:
    score: float
    free_violation: float
    surface_support: float
    valid_observation_count: int
    valid_observation_ratio: float
    frame_coverage: float
    bidirectional_consistency: float
    insufficient_evidence: bool
    signature: object
    evaluation: object


class RayPoseValidator:
    def __init__(self, surface_mu, surface_sigma, surface_weight, min_valid_observations):
        self.surface_mu = float(surface_mu)
        self.surface_sigma = float(surface_sigma)
        self.surface_weight = float(surface_weight)
        self.min_valid_observations = int(min_valid_observations)

    def evaluate(self, pose, src_points, tgt_points, src_rays, tgt_rays, split_id):
        evidence = bidirectional_ray_evaluation(
            src_points, tgt_points, pose, src_rays, tgt_rays,
            self.surface_mu, self.surface_sigma, split_id,
        )
        target_frames = evidence["target"].per_frame_scores
        source_frames = evidence["source"].per_frame_scores
        frame_scores = [value for value in (target_frames, source_frames) if value.numel()]
        total_frames = sum(value.shape[0] for value in frame_scores)
        covered_frames = sum(int((value[:, 3] > 0).sum().item()) for value in frame_scores)
        insufficient = evidence["valid_observation_count"] < self.min_valid_observations
        score = evidence["free_violation"] + self.surface_weight * (1.0 - evidence["surface_support"])
        if insufficient:
            score = float("inf")
        return PoseValidation(
            score=float(score),
            free_violation=float(evidence["free_violation"]),
            surface_support=float(evidence["surface_support"]),
            valid_observation_count=int(evidence["valid_observation_count"]),
            valid_observation_ratio=float(evidence["valid_observation_count"] / max(1, total_frames)),
            frame_coverage=float(covered_frames / max(1, total_frames)),
            bidirectional_consistency=float(evidence["bidirectional_consistency"]),
            insufficient_evidence=insufficient,
            signature=ray_signature(evidence),
            evaluation=evidence,
        )

    def evaluate_batch(self, poses, src_points, tgt_points, src_rays, tgt_rays, split_id):
        batch = evaluate_pose_batch(
            poses, src_points, tgt_points, src_rays, tgt_rays,
            self.surface_mu, self.surface_sigma, split_id,
        )
        results = []
        for index in range(poses.shape[0]):
            valid_count = int(batch["valid_observation_count"][index].item())
            free = float(batch["free_violation"][index].item())
            surface = float(batch["surface_support"][index].item())
            insufficient = valid_count < self.min_valid_observations
            results.append({
                "score": float("inf") if insufficient else free + self.surface_weight * (1.0 - surface),
                "free_violation": free,
                "surface_support": surface,
                "valid_observation_count": valid_count,
                "insufficient_evidence": insufficient,
            })
        return results
