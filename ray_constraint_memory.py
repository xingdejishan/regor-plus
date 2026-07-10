from dataclasses import dataclass, field

import torch

from ray_evidence import RayResidualSignature, align_ray_signatures


@dataclass(frozen=True)
class RejectedBasin:
    hypothesis_id: int
    pose: torch.Tensor
    ray_keys: torch.Tensor
    ray_signature: RayResidualSignature
    search_energy: float
    local_information_matrix: torch.Tensor
    rotation_radius: float
    translation_radius: float


@dataclass
class ConstraintMemory:
    basins: list = field(default_factory=list)

    @staticmethod
    def pose_distance(first, second, rotation_scale, translation_scale):
        first = first[0] if first.ndim == 3 else first
        second = second[0] if second.ndim == 3 else second
        relative = first[:3, :3].transpose(0, 1) @ second[:3, :3]
        angle = torch.acos(torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        translation = torch.linalg.norm(first[:3, 3] - second[:3, 3])
        return torch.sqrt((angle / rotation_scale) ** 2 + (translation / translation_scale) ** 2)

    def pose_penalty_batch(self, poses):
        if not self.basins:
            return torch.zeros(poses.shape[0], device=poses.device, dtype=poses.dtype)
        penalties = []
        for pose in poses:
            values = [
                torch.exp(-self.pose_distance(pose, basin.pose, basin.rotation_radius, basin.translation_radius) ** 2)
                for basin in self.basins
            ]
            penalties.append(torch.stack(values).max())
        return torch.stack(penalties)

    def nearest_pose_distance(self, pose):
        if not self.basins:
            return float("inf")
        return min(
            float(self.pose_distance(pose, basin.pose, basin.rotation_radius, basin.translation_radius).item())
            for basin in self.basins
        )

    def repeated_basin(self, pose, signature, search_energy, signature_similarity, energy_tolerance):
        for basin in self.basins:
            distance = self.pose_distance(pose, basin.pose, basin.rotation_radius, basin.translation_radius)
            _, residuals = align_ray_signatures([signature, basin.ray_signature])
            similarity = float(torch.nn.functional.cosine_similarity(
                residuals[0].view(1, -1), residuals[1].view(1, -1), dim=1,
            ).item()) if residuals.shape[1] else 0.0
            unresolved = float(search_energy) >= basin.search_energy - float(energy_tolerance)
            if distance < 1.0 and similarity >= signature_similarity and unresolved:
                return True, float(distance.item()), similarity
        return False, self.nearest_pose_distance(pose), 0.0

    def add(self, basin):
        self.basins.append(basin)

    def __len__(self):
        return len(self.basins)
