from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CutRegorConfig
from .geometry import PatchGraph, invert_pose, transform_points


@dataclass(frozen=True)
class FrontierResult:
    relative_symmetry: torch.Tensor
    mapped_indices: torch.Tensor
    scale_support: torch.Tensor
    multiscale_support: torch.Tensor
    multiscale_frontier: torch.Tensor
    single_scale_frontier: torch.Tensor


def partial_symmetry_frontier(
    target_graph: PatchGraph,
    positive_pose: torch.Tensor,
    negative_pose: torch.Tensor,
    delta: float,
    config: CutRegorConfig,
) -> FrontierResult:
    relative = negative_pose @ invert_pose(positive_pose)
    mapped_centers = transform_points(target_graph.centers, relative)
    mapping_distance = torch.cdist(mapped_centers, target_graph.centers)
    nearest = mapping_distance.argmin(dim=1)
    mapped_offset = mapped_centers - target_graph.centers[nearest]
    plane_error = torch.abs(torch.sum(mapped_offset[None] * target_graph.normals[:, nearest], dim=2))
    rotated_normals = torch.einsum("ij,sbj->sbi", relative[:3, :3], target_graph.normals)
    normal_error = 1.0 - torch.abs(
        torch.sum(rotated_normals * target_graph.normals[:, nearest], dim=2)
    )
    shape_error = torch.abs(target_graph.spectra - target_graph.spectra[:, nearest]).sum(dim=2)
    neighbor = target_graph.neighbors
    mapped_neighbor = nearest[neighbor]
    original_length = torch.linalg.vector_norm(
        target_graph.centers[:, None, :] - target_graph.centers[neighbor], dim=2
    )
    mapped_length = torch.linalg.vector_norm(
        target_graph.centers[nearest][:, None, :] - target_graph.centers[mapped_neighbor], dim=2
    )
    graph_error = torch.median(torch.abs(original_length - mapped_length), dim=1).values
    scale_support = (
        (plane_error <= config.position_tolerance * delta)
        & (normal_error <= config.normal_tolerance)
        & (shape_error <= config.shape_tolerance)
        & (graph_error[None] <= config.graph_tolerance * delta)
    )
    multiscale_support = scale_support.sum(dim=0) >= config.multiscale_votes
    single_support = scale_support[len(config.scales) // 2]
    multiscale_frontier = multiscale_support & torch.any(~multiscale_support[neighbor], dim=1)
    single_frontier = single_support & torch.any(~single_support[neighbor], dim=1)
    return FrontierResult(
        relative_symmetry=relative,
        mapped_indices=nearest,
        scale_support=scale_support,
        multiscale_support=multiscale_support,
        multiscale_frontier=multiscale_frontier,
        single_scale_frontier=single_frontier,
    )


def _diverse_rank(points: torch.Tensor, score: torch.Tensor, count: int) -> torch.Tensor:
    count = min(count, len(points))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    normalized = (score - score.amin()) / (score.amax() - score.amin()).clamp_min(1e-8)
    selected = [int(torch.argmax(normalized).item())]
    closest = torch.linalg.vector_norm(points - points[selected[0]], dim=1)
    for _ in range(1, count):
        novelty = closest / closest.amax().clamp_min(1e-8)
        value = normalized + 0.5 * novelty
        value[torch.tensor(selected, device=points.device)] = -1
        chosen = int(torch.argmax(value).item())
        selected.append(chosen)
        closest = torch.minimum(closest, torch.linalg.vector_norm(points - points[chosen], dim=1))
    return torch.tensor(selected, dtype=torch.long, device=points.device)


def select_queries(
    source_graph: PatchGraph,
    target_graph: PatchGraph,
    positive_pose: torch.Tensor,
    negative_pose: torch.Tensor,
    frontier: FrontierResult,
    strategy: str,
    count: int,
    seed: int,
    single_scale: bool = False,
) -> torch.Tensor:
    source_centers = source_graph.centers
    if strategy == "random":
        generator = torch.Generator(device=source_centers.device)
        generator.manual_seed(seed)
        order = torch.randperm(len(source_centers), generator=generator, device=source_centers.device)
        return order[: min(count, len(order))]
    if strategy == "high_curvature":
        score = source_graph.scattering[len(source_graph.scattering) // 2]
        return _diverse_rank(source_centers, score, count)
    if strategy == "high_disagreement":
        score = torch.linalg.vector_norm(
            transform_points(source_centers, positive_pose) - transform_points(source_centers, negative_pose),
            dim=1,
        )
        return _diverse_rank(source_centers, score, count)
    if strategy != "symmetry_frontier":
        raise ValueError(f"unknown query strategy: {strategy}")
    mask = frontier.single_scale_frontier if single_scale else frontier.multiscale_frontier
    target_indices = torch.where(mask)[0]
    if len(target_indices) == 0:
        return torch.empty(0, dtype=torch.long, device=source_centers.device)
    pulled = transform_points(target_graph.centers[target_indices], invert_pose(positive_pose))
    nearest_source = torch.cdist(pulled, source_centers).argmin(dim=1).unique()
    disagreement = torch.linalg.vector_norm(
        transform_points(source_centers[nearest_source], positive_pose)
        - transform_points(source_centers[nearest_source], negative_pose),
        dim=1,
    )
    local = _diverse_rank(source_centers[nearest_source], disagreement, count)
    return nearest_source[local]
