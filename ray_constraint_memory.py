from dataclasses import dataclass, field

import torch


@dataclass
class RejectedBasin:
    pose: torch.Tensor
    ray_signature: torch.Tensor
    active_ray_ids: torch.Tensor
    local_G: torch.Tensor
    local_b: torch.Tensor
    radius_rotation: float
    radius_translation: float
    search_energy: float


@dataclass
class ConstraintMemory:
    basins: list = field(default_factory=list)

    def add_rejected(self, candidates):
        for candidate in candidates:
            if isinstance(candidate, RejectedBasin):
                self.basins.append(candidate)

    def pose_distance(self, first, second, rotation_scale=0.08726646, translation_scale=0.10):
        relative = first[:3, :3].T @ second[:3, :3]
        angle = torch.acos(torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        translation = torch.linalg.norm(first[:3, 3] - second[:3, 3])
        return torch.sqrt((angle / rotation_scale) ** 2 + (translation / translation_scale) ** 2)

    def is_repeated(self, pose, signature, search_energy, eta=0.90, delta=0.01):
        for basin in self.basins:
            pose_close = self.pose_distance(pose, basin.pose, basin.radius_rotation, basin.radius_translation) < 1.0
            if basin.ray_signature.numel() and signature.numel():
                signature_similarity = torch.nn.functional.cosine_similarity(
                    signature.float().view(1, -1), basin.ray_signature.float().view(1, -1), dim=1
                ).item()
            else:
                signature_similarity = 0.0
            unresolved = search_energy > basin.search_energy - delta
            if pose_close and signature_similarity > eta and unresolved:
                return True
        return False

    def nearest_pose_distance(self, pose):
        if not self.basins:
            return float("inf")
        return min(self.pose_distance(pose, basin.pose).item() for basin in self.basins)

    def __len__(self):
        return len(self.basins)
