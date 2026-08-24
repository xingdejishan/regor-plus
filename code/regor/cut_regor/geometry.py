from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import CutRegorConfig


@dataclass(frozen=True)
class PatchGraph:
    points: torch.Tensor
    center_indices: torch.Tensor
    centers: torch.Tensor
    neighbors: torch.Tensor
    normals: torch.Tensor
    spectra: torch.Tensor
    scattering: torch.Tensor
    densities: torch.Tensor


def transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return points @ pose[:3, :3].T + pose[:3, 3]


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    inverse = torch.eye(4, dtype=pose.dtype, device=pose.device)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -(pose[:3, :3].T @ pose[:3, 3])
    return inverse


def pose_metrics(poses: np.ndarray, gt_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(poses, dtype=np.float64)
    if values.ndim == 2:
        values = values[None]
    gt = np.asarray(gt_pose, dtype=np.float64)
    relative = np.transpose(values[:, :3, :3], (0, 2, 1)) @ gt[:3, :3]
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rre = np.degrees(np.arccos(cosine))
    rte = np.linalg.norm(values[:, :3, 3] - gt[:3, 3], axis=1) * 100.0
    return rre, rte


def _chunked_nearest(points: torch.Tensor, references: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
    nearest = []
    for start in range(0, len(points), chunk_size):
        nearest.append(torch.cdist(points[start : start + chunk_size], references).amin(dim=1))
    return torch.cat(nearest)


def estimate_resolution(source: torch.Tensor, target: torch.Tensor, sample_count: int = 2048) -> float:
    values = []
    for points in (source, target):
        count = min(sample_count, len(points))
        indices = torch.linspace(0, len(points) - 1, count, device=points.device).round().long().unique()
        sampled = points[indices]
        distances = torch.cdist(sampled, sampled)
        distances.fill_diagonal_(float("inf"))
        values.append(distances.amin(dim=1))
    delta = torch.cat(values).median().item()
    if not np.isfinite(delta) or delta <= 0:
        raise RuntimeError("point-cloud resolution is not positive and finite")
    return float(delta)


def farthest_point_indices(points: torch.Tensor, count: int, seed_index: int = 0) -> torch.Tensor:
    count = min(max(int(count), 1), len(points))
    selected = torch.empty(count, dtype=torch.long, device=points.device)
    selected[0] = int(seed_index) % len(points)
    distance = torch.linalg.vector_norm(points - points[selected[0]], dim=1)
    for index in range(1, count):
        selected[index] = torch.argmax(distance)
        distance = torch.minimum(distance, torch.linalg.vector_norm(points - points[selected[index]], dim=1))
    return selected


def _patch_statistics(
    points: torch.Tensor,
    centers: torch.Tensor,
    neighbor_points: torch.Tensor,
    neighbor_distances: torch.Tensor,
    radius: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = neighbor_distances <= radius
    fallback = mask.sum(dim=1) < 4
    if torch.any(fallback):
        mask[fallback, :4] = True
    weights = mask.to(points.dtype)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (neighbor_points * weights[..., None]).sum(dim=1) / count[:, None]
    centered = neighbor_points - mean[:, None, :]
    covariance = torch.einsum("bki,bkj,bk->bij", centered, centered, weights) / count[:, None, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    scale = eigenvalues.sum(dim=1, keepdim=True).clamp_min(1e-8)
    spectra = eigenvalues / scale
    normals = eigenvectors[:, :, 0]
    scattering = spectra[:, 0]
    density = count / max((4.0 / 3.0) * np.pi * radius**3, 1e-8)
    return normals, spectra, scattering, density


def build_patch_graph(points: torch.Tensor, delta: float, config: CutRegorConfig) -> PatchGraph:
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 8:
        raise ValueError("points must have shape [N,3] with at least eight points")
    center_indices = farthest_point_indices(points, config.patch_centers)
    centers = points[center_indices]
    distances = torch.cdist(centers, points)
    knn = torch.topk(distances, k=min(config.patch_knn, len(points)), largest=False, sorted=True)
    neighbor_points = points[knn.indices]
    normals = []
    spectra = []
    scattering = []
    densities = []
    for multiplier in config.scales:
        values = _patch_statistics(points, centers, neighbor_points, knn.values, multiplier * delta)
        normals.append(values[0])
        spectra.append(values[1])
        scattering.append(values[2])
        densities.append(values[3])
    center_distance = torch.cdist(centers, centers)
    center_distance.fill_diagonal_(float("inf"))
    graph_neighbors = torch.topk(
        center_distance,
        k=min(config.graph_neighbors, len(centers) - 1),
        largest=False,
        sorted=True,
    ).indices
    return PatchGraph(
        points=points,
        center_indices=center_indices,
        centers=centers,
        neighbors=graph_neighbors,
        normals=torch.stack(normals),
        spectra=torch.stack(spectra),
        scattering=torch.stack(scattering),
        densities=torch.stack(densities),
    )
