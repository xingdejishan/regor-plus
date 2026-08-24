from __future__ import annotations

import torch

from probe_regor import ProbeRegorConfig, LineProbeBank, effective_sample_size, transform_points, weighted_rigid_transform


def test_weighted_rigid_transform_recovers_pose() -> None:
    source = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    angle = torch.tensor(0.4)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([0.3, -0.2, 0.1])
    target = source @ rotation.T + translation
    pose = weighted_rigid_transform(source, target, torch.ones(len(source)))
    assert torch.allclose(pose[:3, :3], rotation, atol=1e-5)
    assert torch.allclose(pose[:3, 3], translation, atol=1e-5)


def test_line_probe_prefers_aligned_geometry() -> None:
    grid = torch.linspace(-0.5, 0.5, 8)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    first = torch.stack([x.flatten(), y.flatten(), torch.zeros_like(x).flatten()], dim=1)
    second = torch.stack([x.flatten(), y.flatten(), torch.ones_like(x).flatten() * 0.3], dim=1)
    points = torch.cat([first, second], dim=0)
    config = ProbeRegorConfig(line_count=64, line_candidates=512, point_chunk=64)
    bank = LineProbeBank(config)
    identity = torch.eye(4)
    shifted = torch.eye(4)
    shifted[0, 3] = 0.3
    aligned = bank.observe(points, points, identity)
    misaligned = bank.observe(points, points, shifted)
    scalar_bank = LineProbeBank(
        ProbeRegorConfig(line_count=64, line_candidates=512, point_chunk=64, line_chunk=1)
    )
    scalar = scalar_bank.observe(points, points, identity)
    assert aligned.matched_count > 0
    assert float(aligned.weights.mean()) > float(misaligned.weights.mean())
    assert scalar.matched_count == aligned.matched_count
    assert torch.allclose(scalar.source, aligned.source)
    assert torch.allclose(scalar.target, aligned.target)
    assert torch.allclose(scalar.weights, aligned.weights)


def test_transform_points_and_ess() -> None:
    points = torch.tensor([[1.0, 2.0, 3.0]])
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([0.5, -0.5, 1.0])
    assert torch.allclose(transform_points(points, pose), torch.tensor([[1.5, 1.5, 4.0]]))
    assert effective_sample_size(torch.ones(4)) == 4.0


if __name__ == "__main__":
    test_weighted_rigid_transform_recovers_pose()
    test_line_probe_prefers_aligned_geometry()
    test_transform_points_and_ess()
    print("3 tests passed")
