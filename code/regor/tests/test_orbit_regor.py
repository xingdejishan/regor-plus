from __future__ import annotations

import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orbit_regor.atlas import PatchAtlas
from orbit_regor.config import OrbitRegorConfig
from orbit_regor.orbit_cover import build_orbit_cover
from orbit_regor.solver import LiftAtoms, evaluate_particle, materialized_pairs, rigid_transform


def atlas(points: torch.Tensor, signatures: torch.Tensor) -> PatchAtlas:
    count = len(points)
    descriptors = torch.eye(count, dtype=points.dtype)
    normals = torch.zeros_like(points)
    normals[:, 2] = 1
    return PatchAtlas(points, torch.arange(count), signatures, normals, descriptors)


def test_repeated_nonlocal_patches_form_orbit_and_singletons_remain() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    signatures = torch.tensor([[1.0, 2.0], [1.0, 2.0], [4.0, 7.0]])
    config = OrbitRegorConfig(orbit_min_separation=2.0, orbit_signature_threshold=1.0)
    cover = build_orbit_cover(atlas(points, signatures), 0.1, config)
    assert len(cover.nodes) >= 4
    assert sum(node.singleton for node in cover.nodes) == 3
    assert int(cover.primary_orbit_sizes[0]) == 2
    assert int(cover.primary_orbit_sizes[1]) == 2


def test_delayed_commuting_contraction_never_expands_domain() -> None:
    source_points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    target_points = source_points + torch.tensor([0.2, -0.1, 0.3])
    signatures = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 5.0], [3.0, 5.0]])
    source = atlas(source_points, signatures)
    target = atlas(target_points, signatures)
    config = OrbitRegorConfig(
        orbit_min_separation=2.0,
        orbit_signature_threshold=1.0,
        coarse_lift_tolerance=20.0,
        relation_tolerance=1.5,
    )
    source_cover = build_orbit_cover(source, 0.1, config)
    target_cover = build_orbit_cover(target, 0.1, config)
    atoms = LiftAtoms(
        torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]),
        torch.ones(8),
    )
    pose = rigid_transform(source_points, target_points)
    unconstrained = evaluate_particle(
        pose, atoms, source, target, source_cover, target_cover, 0.1, config, commuting=False
    )
    delayed = evaluate_particle(
        pose, atoms, source, target, source_cover, target_cover, 0.1, config, commuting=True
    )
    assert delayed.domain_count_after <= unconstrained.domain_count_after
    assert delayed.domain_count_after <= delayed.domain_count_before
    assert delayed.birth_domain_count <= delayed.domain_count_before
    assert delayed.regenerated_atom_count >= 0


def test_materialized_lifts_are_partial_bijection() -> None:
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    signatures = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
    source = atlas(points, signatures)
    target = atlas(points.clone(), signatures)
    config = OrbitRegorConfig(coarse_lift_tolerance=20.0)
    source_cover = build_orbit_cover(source, 0.1, config, singleton_only=True)
    target_cover = build_orbit_cover(target, 0.1, config, singleton_only=True)
    atoms = LiftAtoms(
        torch.tensor([0, 0, 1, 1, 2]),
        torch.tensor([0, 1, 0, 1, 2]),
        torch.ones(5),
    )
    state = evaluate_particle(
        torch.eye(4), atoms, source, target, source_cover, target_cover, 0.1, config, commuting=False
    )
    selected_source, selected_target = materialized_pairs(state, atoms, source, target, config)
    assert len(torch.unique(selected_source, dim=0)) == len(selected_source)
    assert len(torch.unique(selected_target, dim=0)) == len(selected_target)


def test_rigid_transform_recovers_known_pose() -> None:
    source = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    angle = torch.tensor(0.4)
    rotation = torch.tensor(
        [[torch.cos(angle), -torch.sin(angle), 0.0], [torch.sin(angle), torch.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([0.3, -0.2, 0.5])
    target = source @ rotation.T + translation
    pose = rigid_transform(source, target)
    assert torch.allclose(pose[:3, :3], rotation, atol=1e-5)
    assert torch.allclose(pose[:3, 3], translation, atol=1e-5)


if __name__ == "__main__":
    test_repeated_nonlocal_patches_form_orbit_and_singletons_remain()
    test_delayed_commuting_contraction_never_expands_domain()
    test_materialized_lifts_are_partial_bijection()
    test_rigid_transform_recovers_known_pose()
