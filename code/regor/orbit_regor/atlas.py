from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import OrbitRegorConfig


@dataclass(frozen=True)
class PatchAtlas:
    centers: torch.Tensor
    center_point_ids: torch.Tensor
    signatures: torch.Tensor
    normals: torch.Tensor
    descriptors: torch.Tensor


def estimate_resolution(source: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    for cloud in (source, target):
        if len(cloud) < 2:
            continue
        stride = max(1, len(cloud) // 2048)
        sampled = cloud[::stride][:2048]
        distance = torch.cdist(sampled, sampled)
        nearest = torch.topk(distance, 2, largest=False, dim=1).values[:, 1]
        values.append(nearest)
    if not values:
        return 0.05
    return float(torch.cat(values).median().clamp_min(1e-3).item())


def farthest_point_indices(points: torch.Tensor, count: int) -> torch.Tensor:
    count = min(count, len(points))
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    center = points.mean(0, keepdim=True)
    first = torch.linalg.norm(points - center, dim=1).argmax()
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    selected[0] = first
    distance = torch.linalg.norm(points - points[first], dim=1)
    for index in range(1, count):
        chosen = distance.argmax()
        selected[index] = chosen
        distance = torch.minimum(distance, torch.linalg.norm(points - points[chosen], dim=1))
    return selected


def _patch_signature(
    center: torch.Tensor,
    points: torch.Tensor,
    distances: torch.Tensor,
    radius: float,
    bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.nonzero(distances <= radius, as_tuple=False).flatten()
    if len(ids) < 6:
        ids = torch.topk(distances, min(16, len(points)), largest=False).indices
    local = points[ids] - center
    covariance = local.T @ local / max(len(local), 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    normalized = eigenvalues / eigenvalues.sum().clamp_min(1e-8)
    local_distance = torch.linalg.norm(local, dim=1) / max(radius, 1e-6)
    quantiles = torch.linspace(0.2, 0.8, bins, device=points.device, dtype=points.dtype)
    radial = torch.quantile(local_distance, quantiles).clamp_max(2.0)
    occupancy = torch.log1p(torch.tensor(float(len(ids)), device=points.device, dtype=points.dtype))
    signature = torch.cat([occupancy[None], normalized, radial])
    normal = eigenvectors[:, 0]
    return signature, normal


def build_patch_atlas(
    points: torch.Tensor,
    features: torch.Tensor,
    resolution: float,
    config: OrbitRegorConfig,
) -> PatchAtlas:
    center_ids = farthest_point_indices(points, config.patch_count)
    centers = points[center_ids]
    center_distance = torch.cdist(centers, points)
    signature_rows = []
    normals = []
    for center_index, center in enumerate(centers):
        scale_rows = []
        normal = None
        for scale in config.scales:
            row, current_normal = _patch_signature(
                center,
                points,
                center_distance[center_index],
                scale * resolution,
                config.radial_bins,
            )
            scale_rows.append(row)
            if normal is None:
                normal = current_normal
        signature_rows.append(torch.cat(scale_rows))
        normals.append(normal)
    descriptors = features[center_ids]
    descriptors = torch.nn.functional.normalize(descriptors, dim=1)
    return PatchAtlas(
        centers=centers,
        center_point_ids=center_ids,
        signatures=torch.stack(signature_rows),
        normals=torch.stack(normals),
        descriptors=descriptors,
    )
