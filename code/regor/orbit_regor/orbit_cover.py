from __future__ import annotations

from dataclasses import dataclass

import torch

from .atlas import PatchAtlas
from .config import OrbitRegorConfig


@dataclass(frozen=True)
class OrbitNode:
    orbit_id: int
    patch_ids: torch.Tensor
    singleton: bool


@dataclass(frozen=True)
class OrbitCover:
    nodes: tuple[OrbitNode, ...]
    primary_orbit_ids: torch.Tensor
    primary_orbit_sizes: torch.Tensor


def _components(adjacency: torch.Tensor, maximum_size: int) -> list[list[int]]:
    matrix = adjacency.detach().cpu()
    seen = set()
    output = []
    for root in range(len(matrix)):
        if root in seen:
            continue
        stack = [root]
        component = []
        seen.add(root)
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = torch.nonzero(matrix[current], as_tuple=False).flatten().tolist()
            for neighbor in neighbors:
                if neighbor not in seen and len(component) + len(stack) < maximum_size:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= 2:
            output.append(sorted(component))
    return output


def build_orbit_cover(
    atlas: PatchAtlas,
    resolution: float,
    config: OrbitRegorConfig,
    singleton_only: bool = False,
) -> OrbitCover:
    count = len(atlas.centers)
    singleton_nodes = [
        OrbitNode(index, torch.tensor([index], dtype=torch.long, device=atlas.centers.device), True)
        for index in range(count)
    ]
    primary = torch.arange(count, device=atlas.centers.device, dtype=torch.long)
    sizes = torch.ones(count, device=atlas.centers.device, dtype=torch.long)
    if singleton_only or count < 2:
        return OrbitCover(tuple(singleton_nodes), primary, sizes)
    signature = atlas.signatures
    median = signature.median(0).values
    scale = (signature - median).abs().median(0).values.clamp_min(0.05)
    standardized = (signature - median) / scale
    signature_distance = torch.cdist(standardized, standardized) / standardized.shape[1] ** 0.5
    spatial = torch.cdist(atlas.centers, atlas.centers)
    valid = spatial >= config.orbit_min_separation * resolution
    signature_distance = signature_distance.masked_fill(~valid, float("inf"))
    neighbor_count = min(config.orbit_neighbor_count, max(count - 1, 1))
    nearest = torch.topk(signature_distance, neighbor_count, largest=False, dim=1)
    directed = torch.zeros((count, count), dtype=torch.bool, device=atlas.centers.device)
    row = torch.arange(count, device=atlas.centers.device)[:, None].expand_as(nearest.indices)
    directed[row, nearest.indices] = nearest.values <= config.orbit_signature_threshold
    adjacency = directed & directed.T
    nodes = list(singleton_nodes)
    for component in _components(adjacency, config.orbit_max_size):
        orbit_id = len(nodes)
        patch_ids = torch.tensor(component, dtype=torch.long, device=atlas.centers.device)
        nodes.append(OrbitNode(orbit_id, patch_ids, False))
        for patch_id in component:
            primary[patch_id] = orbit_id
            sizes[patch_id] = len(component)
    return OrbitCover(tuple(nodes), primary, sizes)
