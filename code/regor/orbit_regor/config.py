from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class OrbitRegorConfig:
    patch_count: int = 192
    patch_knn: int = 64
    scales: tuple[float, float, float] = (4.0, 8.0, 16.0)
    radial_bins: int = 4
    orbit_neighbor_count: int = 3
    orbit_signature_threshold: float = 2.5
    orbit_min_separation: float = 6.0
    orbit_max_size: int = 8
    cross_descriptor_topk: int = 3
    cross_pair_budget: int = 192
    compatibility_tolerance_m: float = 0.10
    clique_min_size: int = 3
    pose_budget: int = 50
    pose_nms_rotation_deg: float = 5.0
    pose_nms_translation_m: float = 0.10
    birth_lift_tolerance: float = 2.0
    coarse_lift_tolerance: float = 4.0
    relation_tolerance: float = 3.0
    lift_rounds: int = 3
    materialized_pair_budget: int = 128
    regor_seed_minimum: int = 100
    regor_knn: tuple[int, int] = (100, 20)
    regor_sampling: tuple[int, int] = (100, 500)

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["scales"] = list(self.scales)
        value["regor_knn"] = list(self.regor_knn)
        value["regor_sampling"] = list(self.regor_sampling)
        return value
