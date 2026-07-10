from dataclasses import dataclass

import torch

from ray_evidence import RayEvaluation, bidirectional_ray_evaluation, transform_points


@dataclass(frozen=True)
class RayConstraint:
    ray_id: int
    point_id: int
    violation_depth: float
    pose_jacobian: torch.Tensor
    confidence: float
    frame_id: int
    direction: str


@dataclass(frozen=True)
class EscapeConstraints:
    current_pose: torch.Tensor
    constraints: tuple
    G: torch.Tensor
    b: torch.Tensor
    weights: torch.Tensor
    preferred_direction: torch.Tensor
    information_matrix: torch.Tensor
    rank: int
    condition_number: float
    rotation_observability: float
    translation_observability: float
    base_search_energy: float


def _skew(points):
    x, y, z = points.unbind(-1)
    zeros = torch.zeros_like(x)
    return torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)


class RayConstraintBuilder:
    def __init__(self, surface_mu, surface_sigma, max_constraints_per_frame=128, huber_delta=0.05):
        self.surface_mu = float(surface_mu)
        self.surface_sigma = float(surface_sigma)
        self.max_constraints_per_frame = int(max_constraints_per_frame)
        self.huber_delta = float(huber_delta)

    def _select(self, evaluation: RayEvaluation):
        if evaluation.per_ray_residuals.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=evaluation.per_ray_residuals.device)
        positive = torch.where(evaluation.per_ray_residuals > 0)[0]
        if positive.numel() == 0:
            return positive
        selected = []
        frames = evaluation.frame_indices[positive]
        for frame in torch.unique(frames).tolist():
            local = positive[frames == frame]
            scores = evaluation.per_ray_residuals[local]
            order = torch.argsort(scores, descending=True)[:self.max_constraints_per_frame]
            selected.append(local[order])
        return torch.cat(selected) if selected else positive[:0]

    def _target_jacobians(self, pose, src_points, target_rays, evaluation, selected):
        if selected.numel() == 0:
            return torch.empty((0, 6), device=pose.device, dtype=pose.dtype)
        point_ids = evaluation.point_ids[selected]
        frame_ids = evaluation.frame_indices[selected]
        local = transform_points(src_points[point_ids], pose)
        camera = torch.linalg.inv(target_rays.camera_poses[frame_ids])[:, :3, :3]
        fragment_rotation = target_rays.fragment_pose[:3, :3].expand(point_ids.numel(), -1, -1)
        projection = camera @ fragment_rotation
        local_jacobian = torch.cat([-_skew(local), torch.eye(3, device=pose.device, dtype=pose.dtype).expand(point_ids.numel(), -1, -1)], dim=-1)
        return (projection[:, 2:3, :] @ local_jacobian).squeeze(1)

    def _source_jacobians(self, pose, tgt_points, source_rays, evaluation, selected):
        if selected.numel() == 0:
            return torch.empty((0, 6), device=pose.device, dtype=pose.dtype)
        point_ids = evaluation.point_ids[selected]
        frame_ids = evaluation.frame_indices[selected]
        target_points = tgt_points[point_ids]
        camera = torch.linalg.inv(source_rays.camera_poses[frame_ids])[:, :3, :3]
        source_fragment_rotation = source_rays.fragment_pose[:3, :3].expand(point_ids.numel(), -1, -1)
        inverse_rotation = pose[:3, :3].transpose(0, 1).expand(point_ids.numel(), -1, -1)
        local_jacobian = torch.cat([_skew(target_points), -torch.eye(3, device=pose.device, dtype=pose.dtype).expand(point_ids.numel(), -1, -1)], dim=-1)
        return ((camera @ source_fragment_rotation @ inverse_rotation)[:, 2:3, :] @ local_jacobian).squeeze(1)

    def _cap_per_frame(self, evaluation, selected):
        if selected.numel() == 0:
            return selected
        retained = []
        for frame in torch.unique(evaluation.frame_indices[selected]).tolist():
            local = selected[evaluation.frame_indices[selected] == frame]
            local = local[torch.argsort(evaluation.per_ray_residuals[local], descending=True)]
            retained.append(local[:self.max_constraints_per_frame])
        return torch.cat(retained) if retained else selected[:0]

    def build(self, pose, src_points, tgt_points, src_rays, tgt_rays, search_evaluation=None, selected_indices=None):
        pose = pose[0] if pose.ndim == 3 else pose
        src_points = src_points[0] if src_points.ndim == 3 else src_points
        tgt_points = tgt_points[0] if tgt_points.ndim == 3 else tgt_points
        evaluation = search_evaluation or bidirectional_ray_evaluation(
            src_points, tgt_points, pose, src_rays, tgt_rays,
            self.surface_mu, self.surface_sigma, split_id=1,
        )
        target_eval, source_eval = evaluation["target"], evaluation["source"]
        if selected_indices is None:
            target_selected, source_selected = self._select(target_eval), self._select(source_eval)
        else:
            target_selected, source_selected = selected_indices.target_indices, selected_indices.source_indices
        target_selected = self._cap_per_frame(target_eval, target_selected)
        source_selected = self._cap_per_frame(source_eval, source_selected)
        target_G = self._target_jacobians(pose, src_points, tgt_rays, target_eval, target_selected)
        source_G = self._source_jacobians(pose, tgt_points, src_rays, source_eval, source_selected)
        G = torch.cat([target_G, source_G], dim=0)
        b = torch.cat([target_eval.per_ray_residuals[target_selected], source_eval.per_ray_residuals[source_selected]], dim=0)
        if b.numel() == 0:
            empty = torch.empty((0,), device=pose.device, dtype=pose.dtype)
            return EscapeConstraints(pose, (), G, empty, empty, torch.zeros(6, device=pose.device, dtype=pose.dtype), torch.zeros((6, 6), device=pose.device, dtype=pose.dtype), 0, float("inf"), 0.0, 0.0, evaluation["free_violation"])
        weights = torch.clamp(self.huber_delta / torch.clamp_min(b, self.huber_delta), max=1.0)
        frames = torch.cat([target_eval.frame_indices[target_selected], source_eval.frame_indices[source_selected]])
        for frame in torch.unique(frames):
            frame_mask = frames == frame
            weights[frame_mask] /= frame_mask.sum()
        weights *= weights.numel() / torch.clamp_min(weights.sum(), 1e-8)
        information = G.transpose(0, 1) @ (weights[:, None] * G)
        singular = torch.linalg.svdvals(information)
        rank = int(torch.linalg.matrix_rank(information).item())
        positive = singular[singular > 1e-8]
        condition = float((positive.max() / positive.min()).item()) if positive.numel() else float("inf")
        preferred = torch.linalg.pinv(information + 1e-6 * torch.eye(6, device=pose.device, dtype=pose.dtype)) @ (G.transpose(0, 1) @ (weights * b))
        preferred = preferred / torch.clamp_min(torch.linalg.norm(preferred), 1e-8)
        records = []
        for local_index, index in enumerate(target_selected.tolist()):
            records.append(RayConstraint(int(target_eval.ray_ids[index]), int(target_eval.point_ids[index]), float(target_eval.per_ray_residuals[index]), target_G[local_index], float(weights[local_index]), int(target_eval.frame_indices[index]), "source_to_target"))
        offset = target_selected.numel()
        for local_index, index in enumerate(source_selected.tolist()):
            records.append(RayConstraint(int(source_eval.ray_ids[index]), int(source_eval.point_ids[index]), float(source_eval.per_ray_residuals[index]), source_G[local_index], float(weights[offset + local_index]), int(source_eval.frame_indices[index]), "target_to_source"))
        return EscapeConstraints(
            pose, tuple(records), G, b, weights, preferred, information, rank, condition,
            float(torch.trace(information[:3, :3]).item()), float(torch.trace(information[3:, 3:]).item()),
            evaluation["free_violation"],
        )
