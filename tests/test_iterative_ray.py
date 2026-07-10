import inspect
import json
import unittest

import torch

from Correspondence_regenerate_v2 import Regenerator
from initial_matching_plus import Matcher_plus
from iterative_ray_config import IterativeRayConfig
from iterative_ray_search import HypothesisArchive, IterativeRaySearch, RaySelector
from ray_constraint_builder import EscapeConstraints, RayConstraintBuilder
from ray_constraint_memory import ConstraintMemory, RejectedBasin
from ray_evidence import RayBundle, RayEvaluation, RayResidualSignature, align_ray_signatures, bidirectional_ray_evaluation, evaluate_pose_batch, evaluate_projected_points, ray_signature
from ray_guided_regenerator import PoseHypothesis, RayGuidedRegenerator
from ray_pose_validator import RayPoseValidator
from test_3DLoMatch import audit_round


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
        RaySelector(config.active_rays_per_round, config.ray_variance_min_observers),
        RayConstraintBuilder(config.ray_trunc_margin, config.ray_surface_sigma),
        RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 0.2),
        RayPoseValidator(config.ray_trunc_margin, config.ray_surface_sigma, 1.0, 1),
    )


class IterativeRayTests(unittest.TestCase):
    def test_guided_seed_is_consumed(self):
        src, tgt, pose, features = points()
        constraints = EscapeConstraints(pose[0], (), torch.empty((0, 6)), torch.empty(0), torch.empty(0), torch.zeros(6), torch.zeros((6, 6)), 0, float("inf"), 0.0, 0.0, 0.0)
        generator = RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 0.2)
        first = generator.generate_hypotheses(src[:, :3], tgt[:, :3], src, tgt, features, features, constraints, 4, 1, 0)
        second = generator.generate_hypotheses(src[:, 1:], tgt[:, 1:], src, tgt, features, features, constraints, 4, 1, 0)
        self.assertNotEqual(first[0].seed_ids.tolist(), second[0].seed_ids.tolist())
        self.assertFalse(torch.equal(first[0].src_corr, second[0].src_corr))

    def test_regenerator_returns_multiple_hypotheses(self):
        src, tgt, pose, features = points()
        constraints = EscapeConstraints(pose[0], (), torch.empty((0, 6)), torch.empty(0), torch.empty(0), torch.zeros(6), torch.zeros((6, 6)), 0, float("inf"), 0.0, 0.0, 0.0)
        hypotheses = RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 0.2).generate_hypotheses(None, None, src, tgt, features, features, constraints, 4, 1, 0)
        self.assertGreaterEqual(len(hypotheses), 2)

    def test_generator_defers_refinement_and_has_no_memory_input(self):
        src, tgt, pose, features = points()
        constraints = EscapeConstraints(pose[0], (), torch.empty((0, 6)), torch.empty(0), torch.empty(0), torch.zeros(6), torch.zeros((6, 6)), 0, float("inf"), 0.0, 0.0, 0.0)
        generator = RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 0.2)
        hypotheses = generator.generate_hypotheses(None, None, src, tgt, features, features, constraints, 4, 1, 0)
        self.assertNotIn("memory", inspect.signature(generator.generate_hypotheses).parameters)
        self.assertTrue(all(torch.allclose(item.pose_raw, item.pose_local_refined) for item in hypotheses))

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
        self.assertLessEqual(result["valid_observation_ratio"], 1.0)

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
        old_signature = RayResidualSignature(torch.tensor([[0, 0, 0], [0, 0, 1]]), torch.tensor([1.0, 0.0]), torch.tensor([0, 1]))
        new_signature = RayResidualSignature(torch.tensor([[1, 0, 0], [1, 0, 1]]), torch.tensor([0.0, 1.0]), torch.tensor([0, 1]))
        memory.add(RejectedBasin(1, pose, old_signature.keys, old_signature, 1.0, torch.eye(6), 1.0, 1.0))
        repeated, _, _ = memory.repeated_basin(pose, new_signature, 1.0, 0.9, 0.01)
        self.assertFalse(repeated)

    def test_history_basin_keeps_physically_improved_nearby_pose(self):
        _, _, pose, _ = points()
        signature = RayResidualSignature(torch.tensor([[0, 0, 0], [0, 0, 1]]), torch.tensor([1.0, 0.5]), torch.tensor([0, 1]))
        memory = ConstraintMemory()
        memory.add(RejectedBasin(1, pose, signature.keys, signature, 1.0, torch.eye(6), 1.0, 1.0))
        repeated, _, _ = memory.repeated_basin(pose, signature, 0.8, 0.9, 0.01)
        self.assertFalse(repeated)

    def test_negative_memory_requires_evidence_and_both_bad_margins(self):
        src, tgt, pose, _ = points()
        incumbent = PoseHypothesis(0, -1, 0, pose, pose, src[:, :3], tgt[:, :3], torch.ones(3), torch.zeros((3, 2), dtype=torch.long), "r1")
        incumbent.validation_ray_score = 0.2
        incumbent.search_free_violation = 0.1
        candidate = PoseHypothesis(1, 0, 1, pose, pose, src[:, :3], tgt[:, :3], torch.ones(3), torch.zeros((3, 2), dtype=torch.long), "ray_guided")
        candidate.validation_ray_score = 0.3
        candidate.search_free_violation = 0.12
        self.assertTrue(search()._is_negative_basin(candidate, incumbent))
        candidate.search_insufficient_evidence = True
        self.assertFalse(search()._is_negative_basin(candidate, incumbent))
        candidate.search_insufficient_evidence = False
        candidate.search_free_violation = 0.105
        self.assertFalse(search()._is_negative_basin(candidate, incumbent))

    def test_constraint_uses_frame_level_camera_pose(self):
        rotation = torch.eye(4)
        rotation[1, 1] = 0.0
        rotation[1, 2] = -1.0
        rotation[2, 1] = 1.0
        rotation[2, 2] = 0.0
        rays = RayBundle(
            frame_ids=torch.tensor([0]),
            origins=torch.zeros((1, 3)),
            directions=torch.tensor([[0.0, 0.0, 1.0]]),
            observed_depths=torch.ones(1),
            valid_depth=torch.ones(1, dtype=torch.bool),
            pixels=torch.zeros((1, 2)),
            camera_poses=torch.eye(4)[None],
            intrinsics=torch.eye(3),
            confidences=torch.ones(1),
            split_ids=torch.tensor([1]),
            depth_maps=torch.ones((1, 3, 3)),
            fragment_pose=torch.eye(4),
            frame_numbers=torch.tensor([0]),
            frame_camera_poses=rotation[None],
            frame_split_ids=torch.tensor([1]),
        )
        evaluation = RayEvaluation(
            0.1,
            0.0,
            1,
            torch.tensor([[0.0, 0.1, 0.0, 1.0]]),
            torch.tensor([0.1]),
            torch.tensor([[0, 0, 0]]),
            torch.tensor([0]),
            torch.tensor([0]),
            1,
        )
        builder = RayConstraintBuilder(0.05, 0.03)
        jacobian = builder._target_jacobians(torch.eye(4), torch.tensor([[0.0, 0.0, 0.9]]), rays, evaluation, torch.tensor([0]))
        self.assertFalse(torch.allclose(jacobian, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])))

    def test_rotation_escape_uses_six_dof_delta(self):
        current = torch.eye(4)
        candidate = torch.eye(4)
        candidate[0, 0] = torch.cos(torch.tensor(0.8))
        candidate[0, 1] = -torch.sin(torch.tensor(0.8))
        candidate[1, 0] = torch.sin(torch.tensor(0.8))
        candidate[1, 1] = torch.cos(torch.tensor(0.8))
        constraints = EscapeConstraints(
            current,
            (object(),),
            torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            torch.tensor([0.5]),
            torch.ones(1),
            torch.zeros(6),
            torch.eye(6),
            1,
            1.0,
            1.0,
            1.0,
            0.5,
        )
        score = RayGuidedRegenerator(3, 4, 2.0, 1, 1.0, 0.2)._escape_score(candidate[None], current, constraints)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_opposite_directions_never_share_a_ray_key(self):
        src, tgt, pose, _ = points()
        evidence = bidirectional_ray_evaluation(src, tgt, pose, bundle(), bundle(), split_id=1)
        target_signature = ray_signature(evidence["target"])
        source_signature = ray_signature(evidence["source"])
        common_keys, _ = align_ray_signatures([target_signature, source_signature])
        self.assertEqual(common_keys.shape[0], 0)

    def test_selector_uses_rays_seen_by_two_candidates_not_all_candidates(self):
        current = RayResidualSignature(torch.tensor([[0, 0, 0], [0, 0, 1]]), torch.tensor([0.1, 0.4]), torch.tensor([0, 1]))
        first = RayResidualSignature(torch.tensor([[0, 0, 0], [0, 0, 2]]), torch.tensor([1.0, 0.2]), torch.tensor([0, 1]))
        second = RayResidualSignature(torch.tensor([[0, 0, 1], [0, 0, 2]]), torch.tensor([0.2, 0.3]), torch.tensor([0, 1]))
        all_common, _ = align_ray_signatures([current, first, second])
        keys, _, presence = align_ray_signatures([current, first, second], min_observers=2, return_presence=True)
        self.assertEqual(all_common.shape[0], 0)
        self.assertEqual(keys.shape[0], 3)
        self.assertTrue(torch.equal(presence.sum(dim=0), torch.tensor([2, 2, 2])))
        order = RaySelector(2)._rank(current, [first, second])
        self.assertEqual(int(order[0]), 0)

    def test_iterative_config_rejects_unknown_and_missing_keys(self):
        values = search().config.report()
        values["obsolete_round2_parameter"] = 1
        with self.assertRaises(KeyError):
            IterativeRayConfig.from_mapping(values)
        values = search().config.report()
        del values["ray_manifest"]
        with self.assertRaises(KeyError):
            IterativeRayConfig.from_mapping(values)
        with open("config_json/config_3DLoMatch_Predator.json", encoding="utf-8") as handle:
            predator = json.load(handle)
        self.assertEqual(IterativeRayConfig.from_mapping(predator["iterative_ray"]).validate().method, "iterative_ray")

    def test_oracle_audit_is_limited_to_current_round(self):
        src, tgt, pose, _ = points()
        first = PoseHypothesis(0, -1, 0, pose, pose, src[:, :3], tgt[:, :3], torch.ones(3), torch.zeros((3, 2), dtype=torch.long), "r1")
        second = PoseHypothesis(1, 0, 1, pose, pose, src[:, :3], tgt[:, :3], torch.ones(3), torch.zeros((3, 2), dtype=torch.long), "ray_guided")
        future = PoseHypothesis(2, 1, 3, pose, pose, src[:, :3], tgt[:, :3], torch.ones(3), torch.zeros((3, 2), dtype=torch.long), "ray_guided")
        archive = HypothesisArchive(hypotheses=[first, second, future])
        audits = [
            {"local_refined_re": 20.0, "local_refined_te": 40.0, "success": 0},
            {"local_refined_re": 18.0, "local_refined_te": 35.0, "success": 0},
            {"local_refined_re": 1.0, "local_refined_te": 1.0, "success": 1},
        ]
        before = audit_round(archive, audits, 1, 1, 15.0, 30.0)
        after = audit_round(archive, audits, 3, 2, 15.0, 30.0)
        self.assertEqual(before["cumulative_oracle_success"], 0)
        self.assertEqual(after["cumulative_oracle_success"], 1)

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
