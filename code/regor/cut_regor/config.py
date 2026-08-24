from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CutRegorConfig:
    patch_centers: int = 384
    graph_neighbors: int = 8
    patch_knn: int = 96
    scales: tuple[float, float, float] = (4.0, 8.0, 16.0)
    position_tolerance: float = 2.5
    graph_tolerance: float = 2.5
    normal_tolerance: float = 0.25
    shape_tolerance: float = 0.35
    multiscale_votes: int = 2
    query_count: int = 32
    anchor_count: int = 4
    anchor_residual_m: float = 0.10
    shell_tolerance: float = 2.5
    search_radius: float = 8.0
    max_domain_candidates: int = 8
    success_rre_deg: float = 15.0
    success_rte_cm: float = 30.0
    seed: int = 51

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["scales"] = list(self.scales)
        return value
