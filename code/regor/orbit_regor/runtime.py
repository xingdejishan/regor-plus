from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .atlas import build_patch_atlas, estimate_resolution
from .config import OrbitRegorConfig
from .orbit_cover import build_orbit_cover
from .solver import (
    LiftAtoms,
    birth_poses,
    build_lift_atoms,
    materialized_pairs,
    rigid_transform,
    solve_particles,
)


@dataclass
class OrbitRegorPrepared:
    source_atlas: object
    target_atlas: object
    source_cover: object
    target_cover: object
    singleton_source_cover: object
    singleton_target_cover: object
    atoms: LiftAtoms
    poses: torch.Tensor
    resolution: float


@dataclass
class OrbitRegorResult:
    pose: torch.Tensor
    seed_source: torch.Tensor
    seed_target: torch.Tensor
    telemetry: dict[str, int | float | str]


class OrbitRegorRuntime:
    VARIANTS = ("singleton", "immediate", "no_commuting", "no_coalescence", "no_point_regor", "full")

    def __init__(self, config: OrbitRegorConfig | None = None) -> None:
        self.config = config or OrbitRegorConfig()

    def prepare(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        raw_source: torch.Tensor,
        raw_target: torch.Tensor,
        native_poses: torch.Tensor,
    ) -> OrbitRegorPrepared:
        resolution = estimate_resolution(source, target)
        source_atlas = build_patch_atlas(source, source_features, resolution, self.config)
        target_atlas = build_patch_atlas(target, target_features, resolution, self.config)
        source_cover = build_orbit_cover(source_atlas, resolution, self.config)
        target_cover = build_orbit_cover(target_atlas, resolution, self.config)
        singleton_source_cover = build_orbit_cover(source_atlas, resolution, self.config, singleton_only=True)
        singleton_target_cover = build_orbit_cover(target_atlas, resolution, self.config, singleton_only=True)
        atoms = build_lift_atoms(source_atlas, target_atlas, raw_source, raw_target, self.config)
        poses = birth_poses(atoms, source_atlas, target_atlas, native_poses, self.config)
        return OrbitRegorPrepared(
            source_atlas,
            target_atlas,
            source_cover,
            target_cover,
            singleton_source_cover,
            singleton_target_cover,
            atoms,
            poses,
            resolution,
        )

    def solve(self, prepared: OrbitRegorPrepared, variant: str) -> OrbitRegorResult:
        if variant not in self.VARIANTS:
            raise ValueError(f"unknown OrbitRegor variant: {variant}")
        singleton = variant == "singleton"
        immediate = variant == "immediate"
        commuting = variant != "no_commuting"
        coalescence = variant != "no_coalescence"
        source_cover = prepared.singleton_source_cover if singleton else prepared.source_cover
        target_cover = prepared.singleton_target_cover if singleton else prepared.target_cover
        state = solve_particles(
            prepared.poses,
            prepared.atoms,
            prepared.source_atlas,
            prepared.target_atlas,
            source_cover,
            target_cover,
            prepared.resolution,
            self.config,
            immediate=immediate,
            commuting=commuting,
            coalescence=coalescence,
        )
        source_seed, target_seed = materialized_pairs(
            state,
            prepared.atoms,
            prepared.source_atlas,
            prepared.target_atlas,
            self.config,
        )
        pose = state.pose if len(source_seed) < 3 else rigid_transform(source_seed, target_seed)
        telemetry = {
            "variant": variant,
            "resolution": prepared.resolution,
            "source_orbit_count": len(source_cover.nodes),
            "target_orbit_count": len(target_cover.nodes),
            "source_non_singleton_orbits": sum(not node.singleton for node in source_cover.nodes),
            "target_non_singleton_orbits": sum(not node.singleton for node in target_cover.nodes),
            "lift_atom_count": len(prepared.atoms.source_patch_ids),
            "pose_birth_count": len(prepared.poses),
            "domain_count_before": state.domain_count_before,
            "domain_count_after": state.domain_count_after,
            "birth_domain_count": state.birth_domain_count,
            "regenerated_atom_count": state.regenerated_atom_count,
            "orbit_regeneration_rounds": state.regeneration_rounds,
            "non_singleton_orbits": state.non_singleton_orbits,
            "relation_rank": state.relation_rank,
            "unique_lifts": state.unique_lifts,
            "region_coverage": state.region_coverage,
            "mean_residual": state.residual,
            "coalesced_count": state.coalesced_count,
            "materialized_pair_count": len(source_seed),
        }
        return OrbitRegorResult(pose, source_seed, target_seed, telemetry)

    def ensure_seed_count(self, source: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if len(source) == 0:
            return source, target
        if len(source) >= self.config.regor_seed_minimum:
            return source, target
        repeats = int(np.ceil(self.config.regor_seed_minimum / len(source)))
        return source.repeat((repeats, 1))[: self.config.regor_seed_minimum], target.repeat((repeats, 1))[: self.config.regor_seed_minimum]
