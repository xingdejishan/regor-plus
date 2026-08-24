from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import CutRegorConfig
from .geometry import transform_points


@dataclass(frozen=True)
class BridgeResult:
    candidate_indices: torch.Tensor
    shell_errors: torch.Tensor


def descriptor_correspondences(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    source = torch.nn.functional.normalize(source_features, dim=1)
    target = torch.nn.functional.normalize(target_features, dim=1)
    matches = []
    for start in range(0, len(source), chunk_size):
        matches.append((source[start : start + chunk_size] @ target.T).argmax(dim=1))
    return torch.cat(matches)


def choose_mode_anchors(
    source: torch.Tensor,
    target: torch.Tensor,
    target_matches: torch.Tensor,
    pose: torch.Tensor,
    config: CutRegorConfig,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    warped = transform_points(source, pose)
    residual = torch.linalg.vector_norm(warped - target[target_matches], dim=1)
    valid = torch.where(residual <= config.anchor_residual_m)[0]
    if len(valid) < config.anchor_count:
        return None
    if len(valid) > 1024:
        order = torch.argsort(residual[valid])[:1024]
        valid = valid[order]
    points = source[valid]
    matched_targets = target_matches[valid]
    first = int(torch.argmin(residual[valid]).item())
    selected = [first]
    distance = torch.linalg.vector_norm(points - points[first], dim=1)
    second = int(torch.argmax(distance).item())
    selected.append(second)
    axis = points[second] - points[first]
    area = torch.linalg.vector_norm(torch.linalg.cross(axis.expand_as(points), points - points[first], dim=1), dim=1)
    area[torch.tensor(selected, device=points.device)] = -1
    third = int(torch.argmax(area).item())
    selected.append(third)
    trial = torch.tensor(selected + [0], dtype=torch.long, device=points.device)[None].repeat(len(points), 1)
    trial[:, 3] = torch.arange(len(points), device=points.device)
    matrices = points[trial[:, 1:]] - points[trial[:, :1]]
    conditioning = torch.linalg.svdvals(matrices)[:, -1]
    conditioning[torch.tensor(selected, device=points.device)] = -1
    duplicate_target = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    for index in selected:
        duplicate_target |= matched_targets == matched_targets[index]
    conditioning[duplicate_target] = -1
    fourth = int(torch.argmax(conditioning).item())
    selected.append(fourth)
    for _ in range(2):
        for slot in range(config.anchor_count):
            trial = torch.tensor(selected, dtype=torch.long, device=points.device)[None].repeat(len(points), 1)
            trial[:, slot] = torch.arange(len(points), device=points.device)
            matrices = points[trial[:, 1:]] - points[trial[:, :1]]
            conditioning = torch.linalg.svdvals(matrices)[:, -1]
            invalid = torch.zeros(len(points), dtype=torch.bool, device=points.device)
            for other_slot, index in enumerate(selected):
                if other_slot == slot:
                    continue
                invalid |= torch.arange(len(points), device=points.device) == index
                invalid |= matched_targets == matched_targets[index]
            conditioning[invalid] = -1
            selected[slot] = int(torch.argmax(conditioning).item())
    source_indices = valid[torch.tensor(selected, device=points.device)]
    target_indices = target_matches[source_indices]
    if len(torch.unique(target_indices)) < config.anchor_count:
        return None
    return source_indices, target_indices


def bridge_domain(
    source_query: torch.Tensor,
    source_anchor_points: torch.Tensor,
    target_anchor_points: torch.Tensor,
    target: torch.Tensor,
    pose: torch.Tensor,
    delta: float,
    config: CutRegorConfig,
    anchor_count: int | None = None,
) -> BridgeResult:
    count = config.anchor_count if anchor_count is None else int(anchor_count)
    if count < 0 or count > len(source_anchor_points):
        raise ValueError("anchor_count is outside the available anchor range")
    predicted = transform_points(source_query[None], pose)[0]
    target_distance = torch.linalg.vector_norm(target - predicted, dim=1)
    candidate_indices = torch.where(target_distance <= config.search_radius * delta)[0]
    if len(candidate_indices) == 0:
        return BridgeResult(candidate_indices, torch.empty(0, device=target.device, dtype=target.dtype))
    if count == 0:
        order = torch.argsort(target_distance[candidate_indices])[: config.max_domain_candidates]
        return BridgeResult(candidate_indices[order], target_distance[candidate_indices[order]])
    source_radii = torch.linalg.vector_norm(source_query[None] - source_anchor_points[:count], dim=1)
    candidate_radii = torch.cdist(target[candidate_indices], target_anchor_points[:count])
    shell_error = torch.abs(candidate_radii - source_radii[None])
    valid = torch.all(shell_error <= config.shell_tolerance * delta, dim=1)
    candidate_indices = candidate_indices[valid]
    shell_error = shell_error[valid]
    if len(candidate_indices) == 0:
        return BridgeResult(candidate_indices, torch.empty(0, device=target.device, dtype=target.dtype))
    aggregate = shell_error.amax(dim=1)
    order = torch.argsort(aggregate)[: config.max_domain_candidates]
    return BridgeResult(candidate_indices[order], aggregate[order])
