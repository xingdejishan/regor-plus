from dataclasses import dataclass

import torch

from ray_evidence import RayBundle


@dataclass(frozen=True)
class RegistrationPair:
    src_keypoints: torch.Tensor
    tgt_keypoints: torch.Tensor
    src_features: torch.Tensor
    tgt_features: torch.Tensor
    src_overlap: torch.Tensor
    tgt_overlap: torch.Tensor
    src_ray_bundle: RayBundle | None
    tgt_ray_bundle: RayBundle | None
    gt_transform: torch.Tensor | None
    pair_id: str

    def inference_inputs(self):
        return {
            "src_points": self.src_keypoints,
            "tgt_points": self.tgt_keypoints,
            "src_features": self.src_features,
            "tgt_features": self.tgt_features,
            "source_rays": self.src_ray_bundle,
            "target_rays": self.tgt_ray_bundle,
            "pair_id": self.pair_id,
        }
