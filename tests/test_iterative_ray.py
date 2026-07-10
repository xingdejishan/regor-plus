import inspect
import unittest

import torch

from Correspondence_regenerate_v2 import Regenerator
from initial_matching_plus import Matcher_plus
from iterative_ray_config import IterativeRayConfig
from iterative_ray_search import HypothesisArchive, IterativeRaySearch, RaySelector
from ray_constraint_builder import EscapeConstraints, RayConstraintBuilder
from ray_constraint_memory import ConstraintMemory, RejectedBasin
from ray_evidence import RayBundle, bidirectional_ray_evaluation, evaluate_pose_batch, evaluate_projected_points
from ray_guided_regenerator import PoseHypothesis, RayGuidedRegenerator
from ray_pose_validator import RayPoseValidator


def bundle(depth=1.0, unknown=False):
    depth_maps = torch.full((2, 3, 3), depth)
    if unknown:
        depth_maps.zero_()
    return RayBundle(
        frame_ids=torch.tensor([0, 1]),
        origins=torch.zeros((2, 3)),
        directions=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        observed_depths=torch.ones(2),
        valid_depth=torch.ones(2, dtype=torch.bool),
        pixels=torch.zeros((2, 2)),
        camera_poses=torch.eye(4).repeat(2, 1, 1),
        intrinsics=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
        confidences=torch.ones(2),
        split_ids=torch.tensor([1, 0]),
        depth_maps=depth_maps,
        fragment_pose=torch.eye(4),
    )


def points():
    value = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    return value[None], value[None].clone(), torch.eye(4)[None], torch.eye(4)[None]


def search(max_rounds=2, time_budget=0.0):
    config = IterativeRayConfig(
        search_max_rounds=max_rounds,
        search_time_budget_seconds=time_budget,
        candidates_per_round=4,
        descriptor_topk=3,
        local_corr_max_points=4,
        local_knn_radius=2.0,
        local_mutual_k=1,
        active_rays_per_round=12,
        min_valid_ray_count=1,
        ray_manifest="test",
    )
    return IterativeRaySearch(
        config,
        RaySelector(config.active_rays_per_round),
        RayConstraintBuilder(config.ray_trunc_margin, config.ray_surface_sigma),
        RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 1.0, 0.2),
        RayPoseValidator(config.ray_trunc_margin, config.ray_surface_sigma, 1.0, 1),
    )


class IterativeRayTests(unittest.TestCase):
    def test_guided_seed_is_consumed(self):
        src, tgt, pose, features = points()
        constraints = EscapeConstraints(pose[0], (), torch.empty((0, 6)), torch.empty(0), torch.empty(0), torch.zeros(6), torch.zeros((6, 6)), 0, float("inf"), 0.0, 0.0, 0.0)
        generator = RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 1.0, 0.2)
        first = generator.generate_hypotheses(src[:, :3], tgt[:, :3], src, tgt, features, features, constraints, ConstraintMemory(), 4, 1, 0)
        second = generator.generate_hypotheses(src[:, 1:], tgt[:, 1:], src, tgt, features, features, constraints, ConstraintMemory(), 4, 1, 0)
        self.assertNotEqual(first[0].seed_ids.tolist(), second[0].seed_ids.tolist())
        self.assertFalse(torch.equal(first[0].src_corr, second[0].src_corr))

    def test_regenerator_returns_multiple_hypotheses(self):
        src, tgt, pose, features = points()
        constraints = EscapeConstraints(pose[0], (), torch.empty((0, 6)), torch.empty(0), torch.empty(0), torch.zeros(6), torch.zeros((6, 6)), 0, float("inf"), 0.0, 0.0, 0.0)
        hypotheses = RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 1.0, 0.2).generate_hypotheses(None, None, src, tgt, features, features, constraints, ConstraintMemory(), 4, 1, 0)
        self.assertGreaterEqual(len(hypotheses), 2)

    def test_pose_nms_removes_duplicates(self):
        src, tgt, pose, features = points()
        make = lambda score: PoseHypothesis(-1, -1, 1, pose, pose.clone(), src[:, :3], tgt[:, :3], torch.ones(3), torch.zeros((3, 2), dtype=torch.long), "test", descriptor_score=score)
        unique, duplicates = search()._nms([make(1.0), make(0.0)])
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(duplicates), 1)

    def test_incumbent_is_never_dropped(self):
        src, tgt, pose, features = points()
        result = search().run(pose, {"src_points": src, "tgt_points": tgt, "src_features": features, "tgt_features": features, "source_rays": bundle(), "target_rays": bundle()})
        self.assertTrue(any(item.generation_mode == "r1" for item in result.archive.hypotheses))
        self.assertIsNotNone(result.archive.incumbent)

    def test_gt_not_in_inference_signature(self):
        self.assertNotIn("gt_trans", inspect.signature(Regenerator.regenerate).parameters)
        self.assertNotIn("gt_trans", inspect.signature(Matcher_plus.estimator).parameters)

    def test_search_validation_rays_are_disjoint(self):
        value = bundle()
        self.assertFalse(torch.isin(value.frame_indices(1), value.frame_indices(0)).any())

    def test_unknown_space_is_not_free_space(self):
        src, _, pose, _ = points()
        result = evaluate_projected_points(src, bundle(unknown=True), split_id=1)
        self.assertEqual(result.valid_observation_count, 0)
        self.assertEqual(result.free_violation, 0.0)

    def test_bidirectional_ray_evaluation(self):
        src, tgt, pose, _ = points()
        result = bidirectional_ray_evaluation(src, tgt, pose, bundle(), bundle(), split_id=1)
        self.assertEqual(result["valid_observation_count"], 8)
        self.assertEqual(result["free_violation"], 0.0)

    def test_batched_ray_scores_match_single_pose_scores(self):
        src, tgt, pose, _ = points()
        shifted = pose.clone()
        shifted[0, 2, 3] = -0.1
        poses = torch.cat([pose, shifted], dim=0)
        batched = evaluate_pose_batch(poses, src, tgt, bundle(), bundle(), split_id=1)
        for index in range(poses.shape[0]):
            single = bidirectional_ray_evaluation(src, tgt, poses[index:index + 1], bundle(), bundle(), split_id=1)
            self.assertAlmostEqual(float(batched["free_violation"][index]), single["free_violation"], places=6)

    def test_history_basin_requires_signature_match(self):
        _, _, pose, _ = points()
        memory = ConstraintMemory()
        memory.add(RejectedBasin(1, pose, torch.tensor([1]), torch.tensor([1.0, 0.0]), 1.0, torch.eye(6), 1.0, 1.0))
        repeated, _, _ = memory.repeated_basin(pose, torch.tensor([0.0, 1.0]), 1.0, 0.9, 0.01)
        self.assertFalse(repeated)

    def test_active_max_rounds_executes_real_loop(self):
        src, tgt, pose, features = points()
        result = search(max_rounds=3).run(pose, {"src_points": src, "tgt_points": tgt, "src_features": features, "tgt_features": features, "source_rays": bundle(), "target_rays": bundle()})
        self.assertEqual(len(result.round_logs), 3)

    def test_equal_seed_reproducibility(self):
        src, tgt, pose, features = points()
        torch.manual_seed(7)
        first = search().run(pose, {"src_points": src, "tgt_points": tgt, "src_features": features, "tgt_features": features, "source_rays": bundle(), "target_rays": bundle()})
        torch.manual_seed(7)
        second = search().run(pose, {"src_points": src, "tgt_points": tgt, "src_features": features, "tgt_features": features, "source_rays": bundle(), "target_rays": bundle()})
        self.assertTrue(torch.allclose(first.pose, second.pose))

    def test_time_budget_stops_search(self):
        src, tgt, pose, features = points()
        result = search(max_rounds=4, time_budget=1e-12).run(pose, {"src_points": src, "tgt_points": tgt, "src_features": features, "tgt_features": features, "source_rays": bundle(), "target_rays": bundle()})
        self.assertEqual(len(result.round_logs), 0)


if __name__ == "__main__":
    unittest.main()
