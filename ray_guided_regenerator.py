from dataclasses import dataclass, field

import torch

from common import rigid_transform_3d
from utils.SE3 import transform


@dataclass
class PoseHypothesis:
    pose: torch.Tensor
    candidate_id: int
    parent_candidate_id: int
    round_id: int
    seed_ids: list = field(default_factory=list)
    correspondence_count: int = 0
    descriptor_score: float = 0.0
    predicted_escape_score: float = 0.0
    actual_search_energy: float = 0.0
    validation_energy: float = 0.0
    surface_support: float = 0.0
    nearest_history_pose_distance: float = float("inf")
    history_signature_similarity: float = 0.0
    gt_re: float = float("nan")
    gt_te: float = float("nan")
    gt_success: int = -1


class RayGuidedRegenerator:
    def __init__(self, top_l=5, min_seed_count=3, local_regenerator=None):
        self.top_l = int(top_l)
        self.min_seed_count = int(min_seed_count)
        self.local_regenerator = local_regenerator

    def _candidate_pairs(self, src_features, tgt_features, active_source_indices):
        src_desc = src_features[0, active_source_indices]
        tgt_desc = tgt_features[0]
        distance = torch.sqrt(torch.clamp(2 - 2 * (src_desc @ tgt_desc.T), min=1e-6))
        topk = distance.topk(min(self.top_l, tgt_desc.shape[0]), dim=1, largest=False).indices
        pairs = []
        for source_local in range(topk.shape[0]):
            for target_index in topk[source_local].tolist():
                pairs.append((int(active_source_indices[source_local]), int(target_index), float(distance[source_local, target_index])))
        pairs.sort(key=lambda item: item[2])
        return pairs

    def _make_pose(self, src_points, tgt_points, pair_group, current_pose):
        if len(pair_group) < self.min_seed_count:
            return None
        src = src_points[torch.tensor([pair[0] for pair in pair_group], device=src_points.device)]
        tgt = tgt_points[torch.tensor([pair[1] for pair in pair_group], device=tgt_points.device)]
        pose = rigid_transform_3d(src[None], tgt[None])
        if self.local_regenerator is not None:
            seed_src = src[None]
            seed_tgt = tgt[None]
            refined_src, refined_tgt, refined_pose = self.local_regenerator.regenerate(
                seed_src,
                seed_tgt,
                src_points[None],
                tgt_points[None],
                self._src_features,
                self._tgt_features,
                None,
                knn_num=20,
                sampling_num=len(pair_group),
                guide={
                    "local_radius": 0.3,
                    "local_max_points": 64,
                    "generalized_mutual_k": 3,
                    "max_matches_per_seed": 20,
                },
                mode="paired_local",
            )
            if refined_src.shape[1] >= 3:
                pose = refined_pose
        return pose

    def generate(self, current_pose, src_points, tgt_points, src_features, tgt_features, escape_constraints, history_memory, candidate_count, round_id=0):
        self._src_features = src_features
        self._tgt_features = tgt_features
        src = src_points[0]
        tgt = tgt_points[0]
        source_count = min(src.shape[0], max(candidate_count * 4, 64))
        source_indices = torch.arange(src.shape[0], device=src.device)
        if source_indices.shape[0] > source_count:
            source_indices = source_indices[:source_count]
        pairs = self._candidate_pairs(src_features, tgt_features, source_indices)
        if not pairs:
            return []
        preferred = escape_constraints.get("preferred_direction", torch.zeros(6, device=src.device))
        hypotheses = []
        used_groups = set()
        for candidate_id in range(int(candidate_count)):
            start = candidate_id % max(1, len(pairs) - self.min_seed_count + 1)
            group = pairs[start:start + max(self.min_seed_count, 6)]
            group_key = tuple((p[0], p[1]) for p in group)
            if len(group) < self.min_seed_count or group_key in used_groups:
                continue
            used_groups.add(group_key)
            pose = self._make_pose(src, tgt, group, current_pose)
            if pose is None:
                continue
            delta_translation = pose[0, :3, 3] - current_pose[0, :3, 3]
            predicted_escape = float(torch.dot(delta_translation, preferred[3:6]).item()) if preferred.numel() == 6 else 0.0
            hypotheses.append(PoseHypothesis(
                pose=pose,
                candidate_id=candidate_id,
                parent_candidate_id=-1,
                round_id=round_id,
                seed_ids=[p[0] for p in group],
                correspondence_count=len(group),
                descriptor_score=float(-sum(p[2] for p in group) / len(group)),
                predicted_escape_score=predicted_escape,
            ))
        return hypotheses
