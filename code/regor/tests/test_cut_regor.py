from __future__ import annotations

import numpy as np
import torch

from cut_regor.bridge import bridge_domain, choose_mode_anchors
from cut_regor.config import CutRegorConfig
from cut_regor.frontier import partial_symmetry_frontier
from cut_regor.geometry import build_patch_graph, estimate_resolution, invert_pose, pose_metrics, transform_points


def rigid_pose(angle: float, translation: tuple[float, float, float]) -> torch.Tensor:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    pose[:3, 3] = torch.tensor(translation)
    return pose


def test_pose_inverse_and_metrics() -> None:
    points = torch.randn(32, 3)
    pose = rigid_pose(0.3, (0.4, -0.2, 0.1))
    restored = transform_points(transform_points(points, pose), invert_pose(pose))
    assert torch.allclose(points, restored, atol=1e-5)
    rre, rte = pose_metrics(pose.numpy(), pose.numpy())
    assert float(rre[0]) < 1e-6
    assert float(rte[0]) < 1e-6


def test_patch_graph_and_frontier_are_finite() -> None:
    generator = torch.Generator().manual_seed(7)
    first = torch.rand((128, 3), generator=generator) * torch.tensor([1.0, 0.3, 0.3])
    second = first + torch.tensor([1.5, 0.0, 0.0])
    target = torch.cat((first, second, torch.rand((64, 3), generator=generator) + torch.tensor([3.0, 0.0, 0.0])))
    config = CutRegorConfig(patch_centers=96, patch_knn=48, query_count=16)
    delta = estimate_resolution(target, target)
    graph = build_patch_graph(target, delta, config)
    positive = torch.eye(4)
    negative = rigid_pose(0.0, (1.5, 0.0, 0.0))
    result = partial_symmetry_frontier(graph, positive, negative, delta, config)
    assert result.scale_support.shape == (3, len(graph.centers))
    assert result.multiscale_frontier.dtype == torch.bool
    assert torch.isfinite(result.relative_symmetry).all()


def test_four_anchor_bridge_keeps_exact_match() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.4, 0.3, 0.2], [0.8, 0.7, 0.4]],
        dtype=torch.float32,
    )
    pose = rigid_pose(0.2, (0.5, -0.1, 0.2))
    target = torch.cat((transform_points(source, pose), torch.tensor([[3.0, 3.0, 3.0]])))
    matches = torch.arange(len(source))
    config = CutRegorConfig(anchor_residual_m=0.01, search_radius=20.0, shell_tolerance=1.0)
    anchors = choose_mode_anchors(source, target, matches, pose, config)
    assert anchors is not None
    source_ids, target_ids = anchors
    matrix = source[source_ids[1:]] - source[source_ids[:1]]
    assert torch.linalg.svdvals(matrix)[-1] > 0.1
    result = bridge_domain(
        source[4],
        source[source_ids],
        target[target_ids],
        target,
        pose,
        0.01,
        config,
    )
    assert 4 in result.candidate_indices.tolist()


def test_wrong_mode_can_be_blocked_by_bridge() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.4, 0.3, 0.2]],
        dtype=torch.float32,
    )
    target = source.clone()
    config = CutRegorConfig(search_radius=20.0, shell_tolerance=0.5)
    correct = bridge_domain(source[4], source[:4], target[:4], target, torch.eye(4), 0.01, config)
    wrong_pose = rigid_pose(0.0, (0.8, 0.0, 0.0))
    wrong_target_anchors = transform_points(source[:4], wrong_pose)
    wrong = bridge_domain(source[4], source[:4], wrong_target_anchors, target, wrong_pose, 0.01, config)
    assert len(correct.candidate_indices) > 0
    assert len(wrong.candidate_indices) == 0
