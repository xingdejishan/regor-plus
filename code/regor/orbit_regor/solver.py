from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np
import torch

from .atlas import PatchAtlas
from .config import OrbitRegorConfig
from .orbit_cover import OrbitCover


@dataclass(frozen=True)
class LiftAtoms:
    source_patch_ids: torch.Tensor
    target_patch_ids: torch.Tensor
    descriptor_scores: torch.Tensor


@dataclass
class ParticleState:
    pose: torch.Tensor
    active: torch.Tensor
    objective: tuple[int, int, int, int, float]
    domain_count_before: int
    domain_count_after: int
    non_singleton_orbits: int
    relation_rank: int
    unique_lifts: int
    region_coverage: int
    residual: float
    coalesced_count: int = 1
    birth_domain_count: int = 0
    regenerated_atom_count: int = 0
    regeneration_rounds: int = 0


def transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return points @ pose[:3, :3].T + pose[:3, 3]


def rigid_transform(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pose = torch.eye(4, dtype=source.dtype, device=source.device)
    if len(source) < 3:
        return pose
    source_center = source.mean(0)
    target_center = target.mean(0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
    pose[:3, :3] = rotation
    pose[:3, 3] = target_center - rotation @ source_center
    return pose


def rotation_distance_degrees(first: torch.Tensor, second: torch.Tensor) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = ((torch.trace(relative) - 1) / 2).clamp(-1, 1)
    return float(torch.arccos(cosine).item() * 180 / pi)


def build_lift_atoms(
    source_atlas: PatchAtlas,
    target_atlas: PatchAtlas,
    raw_source: torch.Tensor,
    raw_target: torch.Tensor,
    config: OrbitRegorConfig,
) -> LiftAtoms:
    source_ids = torch.cdist(raw_source, source_atlas.centers).argmin(1)
    target_ids = torch.cdist(raw_target, target_atlas.centers).argmin(1)
    similarity = source_atlas.descriptors @ target_atlas.descriptors.T
    top = torch.topk(similarity, min(config.cross_descriptor_topk, similarity.shape[1]), dim=1)
    descriptor_source = torch.arange(len(source_atlas.centers), device=similarity.device)[:, None].expand_as(top.indices)
    combined = torch.cat(
        [
            torch.stack([source_ids, target_ids], dim=1),
            torch.stack([descriptor_source.reshape(-1), top.indices.reshape(-1)], dim=1),
        ],
        dim=0,
    )
    combined = torch.unique(combined, dim=0)
    scores = similarity[combined[:, 0], combined[:, 1]]
    order = torch.argsort(scores, descending=True)[: config.cross_pair_budget]
    combined = combined[order]
    scores = scores[order]
    return LiftAtoms(combined[:, 0], combined[:, 1], scores)


def compatibility_graph(atoms: LiftAtoms, source: PatchAtlas, target: PatchAtlas, tolerance: float) -> torch.Tensor:
    source_points = source.centers[atoms.source_patch_ids]
    target_points = target.centers[atoms.target_patch_ids]
    matrix = (torch.cdist(source_points, source_points) - torch.cdist(target_points, target_points)).abs() < tolerance
    matrix.fill_diagonal_(False)
    return matrix & matrix.T


def node_guided_cliques(graph: torch.Tensor) -> list[torch.Tensor]:
    matrix = graph.detach().cpu().numpy().astype(bool)
    first = matrix.astype(np.int64)
    second = first * (first @ first)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for root in range(len(matrix)):
        clique = [root]
        candidates = matrix[root].copy()
        while candidates.any():
            ids = np.flatnonzero(candidates)
            scores = second[np.ix_(ids, np.asarray(clique))].sum(1)
            chosen = int(ids[int(np.argmax(scores))])
            clique.append(chosen)
            candidates &= matrix[chosen]
            candidates[clique] = False
        key = tuple(sorted(clique))
        unique.setdefault(key, np.asarray(key, dtype=np.int64))
    return [torch.from_numpy(value).to(graph.device) for value in unique.values()]


def birth_poses(
    atoms: LiftAtoms,
    source: PatchAtlas,
    target: PatchAtlas,
    native_poses: torch.Tensor,
    config: OrbitRegorConfig,
) -> torch.Tensor:
    graph = compatibility_graph(atoms, source, target, config.compatibility_tolerance_m)
    candidates = [pose for pose in native_poses.reshape(-1, 4, 4) if torch.isfinite(pose).all()]
    for clique in node_guided_cliques(graph):
        if len(clique) < config.clique_min_size:
            continue
        candidates.append(
            rigid_transform(
                source.centers[atoms.source_patch_ids[clique]],
                target.centers[atoms.target_patch_ids[clique]],
            )
        )
    kept = []
    for pose in candidates:
        if any(
            rotation_distance_degrees(pose, other) < config.pose_nms_rotation_deg
            and float(torch.linalg.norm(pose[:3, 3] - other[:3, 3])) < config.pose_nms_translation_m
            for other in kept
        ):
            continue
        kept.append(pose)
        if len(kept) >= config.pose_budget:
            break
    if not kept:
        kept = [torch.eye(4, dtype=source.centers.dtype, device=source.centers.device)]
    return torch.stack(kept)


def _group_ids(atoms: LiftAtoms, source_cover: OrbitCover, target_cover: OrbitCover) -> torch.Tensor:
    source_orbit = source_cover.primary_orbit_ids[atoms.source_patch_ids]
    target_orbit = target_cover.primary_orbit_ids[atoms.target_patch_ids]
    return source_orbit * (len(target_cover.nodes) + 1) + target_orbit


def _materialize(active: torch.Tensor, residual: torch.Tensor, atoms: LiftAtoms, budget: int) -> torch.Tensor:
    ids = torch.nonzero(active, as_tuple=False).flatten()
    if len(ids) == 0:
        return ids
    ids = ids[torch.argsort(residual[ids])]
    source_used = set()
    target_used = set()
    selected = []
    for atom_id in ids.tolist():
        source_id = int(atoms.source_patch_ids[atom_id])
        target_id = int(atoms.target_patch_ids[atom_id])
        if source_id in source_used or target_id in target_used:
            continue
        source_used.add(source_id)
        target_used.add(target_id)
        selected.append(atom_id)
        if len(selected) >= budget:
            break
    return torch.tensor(selected, dtype=torch.long, device=active.device)


def evaluate_particle(
    pose: torch.Tensor,
    atoms: LiftAtoms,
    source: PatchAtlas,
    target: PatchAtlas,
    source_cover: OrbitCover,
    target_cover: OrbitCover,
    resolution: float,
    config: OrbitRegorConfig,
    immediate: bool = False,
    commuting: bool = True,
) -> ParticleState:
    source_points = source.centers[atoms.source_patch_ids]
    target_points = target.centers[atoms.target_patch_ids]
    warped = transform_points(source_points, pose)
    residual = torch.linalg.norm(warped - target_points, dim=1)
    eligible = residual <= config.coarse_lift_tolerance * resolution
    active = residual <= config.birth_lift_tolerance * resolution
    if eligible.any() and not active.any():
        active[torch.nonzero(eligible, as_tuple=False).flatten()[residual[eligible].argmin()]] = True
    before = int(eligible.sum())
    birth_count = int(active.sum())
    regenerated = 0
    regeneration_rounds = 0
    groups = _group_ids(atoms, source_cover, target_cover)
    if immediate:
        immediate_active = torch.zeros_like(active)
        for group_id in torch.unique(groups[eligible]):
            ids = torch.nonzero(eligible & (groups == group_id), as_tuple=False).flatten()
            immediate_active[ids[residual[ids].argmin()]] = True
        active = immediate_active
    elif not commuting:
        active = eligible
    elif int(eligible.sum()) > 1:
        relation_error = torch.cdist(warped, warped)
        target_distance = torch.cdist(target_points, target_points)
        scalar_consistency = (relation_error - target_distance).abs() <= config.relation_tolerance * resolution
        vector_error = torch.linalg.norm(
            (warped[:, None] - warped[None, :]) - (target_points[:, None] - target_points[None, :]),
            dim=-1,
        )
        consistency = scalar_consistency & (vector_error <= config.relation_tolerance * resolution)
        different_group = groups[:, None] != groups[None, :]
        for _ in range(config.lift_rounds):
            regeneration_rounds += 1
            current = torch.nonzero(active, as_tuple=False).flatten()
            candidates = torch.nonzero(eligible & ~active, as_tuple=False).flatten()
            if len(current) == 0 or len(candidates) == 0:
                break
            relation = consistency[candidates][:, current] & different_group[candidates][:, current]
            grow = candidates[relation.any(1)]
            if len(grow) == 0:
                break
            active[grow] = True
            regenerated += len(grow)
        current = torch.nonzero(active, as_tuple=False).flatten()
        if len(current) > 1:
            supported = (consistency[current][:, current] & different_group[current][:, current]).any(1)
            keep = torch.zeros_like(active)
            keep[current[supported]] = True
            if keep.any():
                active = keep
    active_groups = torch.unique(groups[active])
    unique_lifts = 0
    non_singleton = 0
    representatives = []
    for group_id in active_groups:
        ids = torch.nonzero(active & (groups == group_id), as_tuple=False).flatten()
        if len(ids) == 1:
            unique_lifts += 1
        representative = ids[residual[ids].argmin()]
        representatives.append(representative)
        source_patch = atoms.source_patch_ids[representative]
        target_patch = atoms.target_patch_ids[representative]
        if int(source_cover.primary_orbit_sizes[source_patch]) > 1 or int(target_cover.primary_orbit_sizes[target_patch]) > 1:
            non_singleton += 1
    if representatives:
        representative_ids = torch.stack(representatives)
        representative_source = source_points[representative_ids]
        centered = representative_source - representative_source.mean(0)
        singular = torch.linalg.svdvals(centered)
        relation_rank = int((singular > resolution).sum())
    else:
        relation_rank = 0
    materialized = _materialize(active, residual, atoms, config.materialized_pair_budget)
    coverage = int(len(torch.unique(atoms.source_patch_ids[materialized])) + len(torch.unique(atoms.target_patch_ids[materialized])))
    mean_residual = float(residual[materialized].mean()) if len(materialized) else float("inf")
    objective = (non_singleton, relation_rank, unique_lifts, coverage, -mean_residual)
    return ParticleState(
        pose=pose,
        active=active,
        objective=objective,
        domain_count_before=before,
        domain_count_after=int(active.sum()),
        non_singleton_orbits=non_singleton,
        relation_rank=relation_rank,
        unique_lifts=unique_lifts,
        region_coverage=coverage,
        residual=mean_residual,
        birth_domain_count=birth_count,
        regenerated_atom_count=regenerated,
        regeneration_rounds=regeneration_rounds,
    )


def coalesce_particles(
    states: list[ParticleState],
    atoms: LiftAtoms,
    source: PatchAtlas,
    target: PatchAtlas,
    source_cover: OrbitCover,
    target_cover: OrbitCover,
    resolution: float,
    config: OrbitRegorConfig,
    immediate: bool,
    commuting: bool,
) -> list[ParticleState]:
    consumed = set()
    output = []
    for index, state in enumerate(states):
        if index in consumed:
            continue
        cluster = [index]
        for other_index in range(index + 1, len(states)):
            if other_index in consumed:
                continue
            other = states[other_index]
            if rotation_distance_degrees(state.pose, other.pose) < config.pose_nms_rotation_deg and float(
                torch.linalg.norm(state.pose[:3, 3] - other.pose[:3, 3])
            ) < config.pose_nms_translation_m:
                cluster.append(other_index)
                consumed.add(other_index)
        active = torch.stack([states[value].active for value in cluster]).any(0)
        residual = torch.linalg.norm(
            transform_points(source.centers[atoms.source_patch_ids], state.pose)
            - target.centers[atoms.target_patch_ids],
            dim=1,
        )
        ids = _materialize(active, residual, atoms, config.materialized_pair_budget)
        pose = state.pose if len(ids) < 3 else rigid_transform(
            source.centers[atoms.source_patch_ids[ids]], target.centers[atoms.target_patch_ids[ids]]
        )
        merged = evaluate_particle(
            pose,
            atoms,
            source,
            target,
            source_cover,
            target_cover,
            resolution,
            config,
            immediate=immediate,
            commuting=commuting,
        )
        merged.coalesced_count = len(cluster)
        output.append(merged)
    return output


def solve_particles(
    poses: torch.Tensor,
    atoms: LiftAtoms,
    source: PatchAtlas,
    target: PatchAtlas,
    source_cover: OrbitCover,
    target_cover: OrbitCover,
    resolution: float,
    config: OrbitRegorConfig,
    immediate: bool = False,
    commuting: bool = True,
    coalescence: bool = True,
) -> ParticleState:
    states = [
        evaluate_particle(
            pose,
            atoms,
            source,
            target,
            source_cover,
            target_cover,
            resolution,
            config,
            immediate=immediate,
            commuting=commuting,
        )
        for pose in poses
    ]
    if coalescence:
        states = coalesce_particles(
            states,
            atoms,
            source,
            target,
            source_cover,
            target_cover,
            resolution,
            config,
            immediate,
            commuting,
        )
    return max(states, key=lambda value: value.objective)


def materialized_pairs(state: ParticleState, atoms: LiftAtoms, source: PatchAtlas, target: PatchAtlas, config: OrbitRegorConfig) -> tuple[torch.Tensor, torch.Tensor]:
    residual = torch.linalg.norm(
        transform_points(source.centers[atoms.source_patch_ids], state.pose)
        - target.centers[atoms.target_patch_ids],
        dim=1,
    )
    ids = _materialize(state.active, residual, atoms, config.materialized_pair_budget)
    return source.centers[atoms.source_patch_ids[ids]], target.centers[atoms.target_patch_ids[ids]]
