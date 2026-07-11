import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from correspondence_memory import CorrespondenceMemory
from iterative_ray_config import IterativeRayConfig
from memory_graph_config import MemoryGraphConfig
from memory_guided_registration import MemoryGuidedRegistration, MemoryHypothesis
from test_3DLoMatch import audit_hypothesis, memory_round_audit


def config(**overrides):
    values = {
        "method": "memory_graph",
        "memory_topk": 3,
        "memory_graph_neighbors": 6,
        "memory_sigma_g": 0.05,
        "memory_tau_g": 0.5,
        "memory_support_min": 3,
        "memory_support_max": 6,
        "memory_tau_linear": 0.01,
        "memory_tau_planar": 0.01,
        "memory_min_coverage": 0.001,
        "memory_coverage_voxel_size": 0.05,
        "memory_cross_group_voxel_size": 0.10,
        "memory_min_cross_group_agreement": 0.1,
        "memory_hypotheses_per_round": 6,
        "memory_max_rounds": 3,
        "memory_inlier_threshold": 0.01,
        "memory_tls_threshold": 0.01,
        "memory_tls_iters": 2,
        "memory_prosac_initial_fraction": 0.5,
        "memory_prosac_growth": 0.25,
        "memory_strong_stop_min_inliers": 4,
        "memory_strong_stop_inlier_fraction": 0.1,
        "memory_strong_stop_coverage": 0.2,
    }
    values.update(overrides)
    return MemoryGraphConfig(**values).validate()


def synthetic_pair():
    torch.manual_seed(7)
    source = torch.rand((12, 3))
    angle = torch.tensor(0.35)
    rotation = torch.tensor([
        [torch.cos(angle), -torch.sin(angle), 0.0],
        [torch.sin(angle), torch.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    translation = torch.tensor([0.2, -0.1, 0.05])
    target = source @ rotation.transpose(0, 1) + translation
    features = torch.eye(source.shape[0])
    return source[None], target[None], features[None], features[None], rotation, translation


class CorrespondenceMemoryTests(unittest.TestCase):
    @staticmethod
    def hypothesis(hypothesis_id, round_id, pose, stage="raw", parent_id=-1, score=0.0):
        return MemoryHypothesis(
            hypothesis_id,
            parent_id,
            round_id,
            stage,
            pose,
            pose,
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0),
            score,
            score,
            torch.zeros(5),
            None,
        )

    def test_memory_graph_does_not_require_ray_manifest(self):
        self.assertEqual(config().method, "memory_graph")
        self.assertNotIn("ray_manifest", config().report())

    def test_memory_graph_and_ray_configs_have_separate_strict_schemas(self):
        with open("config_json/config_3DLoMatch_Predator.json", encoding="utf-8") as handle:
            values = json.load(handle)
        self.assertEqual(IterativeRayConfig.from_mapping(values["iterative_ray"]).validate().method, "iterative_ray")
        self.assertEqual(MemoryGraphConfig.from_mapping(values["memory_graph"]).validate().method, "memory_graph")
        invalid = config().report()
        invalid["ray_manifest"] = "not permitted"
        with self.assertRaises(KeyError):
            MemoryGraphConfig.from_mapping(invalid)
        invalid = config().report()
        del invalid["memory_topk"]
        with self.assertRaises(KeyError):
            MemoryGraphConfig.from_mapping(invalid)
        with self.assertRaises(ValueError):
            config(memory_require_r1_cache=False)

    def test_posterior_and_relation_updates_use_current_best_support(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        correct = torch.where(memory.src_indices == memory.tgt_indices)[0]
        wrong = torch.where(memory.src_indices != memory.tgt_indices)[0][0:1]
        support = torch.cat([correct[:3], wrong])
        inliers = torch.zeros(memory.count, dtype=torch.bool)
        inliers[correct] = True
        alpha_before, beta_before = memory.alpha.clone(), memory.beta.clone()
        memory.update_pair_posterior(support, inliers)
        self.assertGreater(float(memory.alpha[correct[0]]), float(alpha_before[correct[0]]))
        self.assertGreater(float(memory.beta[wrong[0]]), float(beta_before[wrong[0]]))
        residuals = torch.ones(memory.count)
        residuals[correct] = 0.0
        memory.update_relation_graph(correct[:6], inliers, residuals)
        self.assertGreater(float(memory.edge_success.sum()), 0.0)

    def test_r1_correspondences_initialize_candidate_memory_before_round_one(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        r1_pose = torch.eye(4)[None]
        r1_pose[0, :3, :3], r1_pose[0, :3, 3] = rotation, translation
        result = MemoryGuidedRegistration(config()).run(
            source,
            target,
            source_features,
            target_features,
            initial_pose=r1_pose,
            initial_src_indices=torch.arange(6),
            initial_tgt_indices=torch.arange(6),
        )
        self.assertEqual(result.r1_initialization["r1_initialized"], 1)
        self.assertEqual(result.r1_initialization["r1_mapped_count"], 6)
        self.assertGreater(result.round_logs[0]["inlier_count"], 0)

    def test_low_confidence_r1_does_not_add_positive_memory(self):
        class CapturingMemory(CorrespondenceMemory):
            instance = None

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                type(self).instance = self
                self.alpha_before = self.alpha.clone()
                self.beta_before = self.beta.clone()
                self.success_before = self.edge_success.clone()
                self.failure_before = self.edge_failure.clone()

            def is_degenerate(self, support):
                return True

        source, target, source_features, target_features, _, _ = synthetic_pair()
        r1_pose = torch.eye(4)[None]
        with patch("memory_guided_registration.CorrespondenceMemory", CapturingMemory):
            result = MemoryGuidedRegistration(config(
                memory_max_sampling_attempts=6,
                memory_r1_low_confidence_inlier_ratio=1.0,
            )).run(
                source,
                target,
                source_features,
                target_features,
                initial_pose=r1_pose,
                initial_src_indices=torch.arange(6),
                initial_tgt_indices=torch.arange(6),
            )
        memory = CapturingMemory.instance
        self.assertTrue(torch.equal(memory.alpha, memory.alpha_before))
        self.assertTrue(torch.equal(memory.beta, memory.beta_before))
        self.assertTrue(torch.equal(memory.edge_success, memory.success_before))
        self.assertTrue(torch.equal(memory.edge_failure, memory.failure_before))
        self.assertEqual(memory.basins[next(iter(memory.basins))].n_nonimproving, 1.0)
        self.assertEqual(result.r1_initialization["r1_low_confidence"], 1)
        self.assertEqual(result.r1_initialization["r1_posterior_delta"], 0.0)
        self.assertEqual(result.r1_initialization["r1_graph_delta"], 0.0)

    def test_disabled_memory_layers_do_not_update_or_affect_estimation_weights(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(
            source,
            target,
            source_features,
            target_features,
            config(memory_use_reliability=False, memory_use_relation_history=False, memory_use_basin=False),
        )
        support = torch.arange(4)
        inliers = torch.zeros(memory.count, dtype=torch.bool)
        alpha_before, beta_before = memory.alpha.clone(), memory.beta.clone()
        self.assertEqual(memory.update_pair_posterior(support, inliers), 0.0)
        self.assertTrue(torch.equal(memory.alpha, alpha_before))
        self.assertTrue(torch.equal(memory.beta, beta_before))
        self.assertTrue(torch.equal(memory.estimation_weights(support), torch.ones_like(support, dtype=memory.dtype)))
        signature = memory.support_signature(support)
        self.assertEqual(memory.update_basin(torch.eye(4)[None], signature, 0.0, improved=False, support_ids=support), (None, None))
        self.assertEqual(len(memory.basins), 0)
        self.assertEqual(float(memory.presearch_basin_penalty(signature, support)), 0.0)

    def test_static_graph_ranking_is_identical_with_history_disabled_before_updates(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        static_only = CorrespondenceMemory(source, target, source_features, target_features, config(memory_use_relation_history=False))
        history_enabled = CorrespondenceMemory(source, target, source_features, target_features, config(memory_use_relation_history=True))
        self.assertTrue(torch.allclose(static_only.graph_quality(), history_enabled.graph_quality()))

    def test_basin_penalty_requires_matching_support_signature(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        support = torch.where(memory.src_indices == memory.tgt_indices)[0][:6]
        signature = memory.support_signature(support)
        objective_before = float(memory.support_objective(support))
        pose = torch.eye(4)[None]
        memory.update_basin(pose, signature, 1.0, improved=False, support_ids=support)
        self.assertLess(memory.basin_bonus(pose, signature), 0.0)
        self.assertGreater(float(memory.presearch_basin_penalty(signature, support)), 0.0)
        self.assertLess(float(memory.support_objective(support)), objective_before)
        self.assertEqual(memory.basin_bonus(pose, torch.zeros_like(signature)), 0.0)

    def test_signature_reports_global_coverage_and_keeps_planar_support_soft(self):
        grid_x, grid_y = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
        source = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1), torch.zeros(16)], dim=1).float() * 0.1
        target = source + torch.tensor([0.1, 0.0, 0.0])
        features = torch.eye(16)
        memory = CorrespondenceMemory(source[None], target[None], features[None], features[None], config())
        support = torch.where(memory.src_indices == memory.tgt_indices)[0][:6]
        signature = memory.support_signature(support)
        self.assertEqual(signature.numel(), 5)
        self.assertLess(float(signature[2]), 0.01)
        self.assertLess(float(signature[3]), 1.0)
        diagnostics = memory.support_diagnostics(support)
        self.assertEqual(diagnostics["planar_degenerate"], 1)
        self.assertFalse(memory.is_degenerate(support))

    def test_memory_search_estimates_full_rigid_pose_and_preserves_raw_seed_pose(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        r1_pose = torch.eye(4)[None]
        r1_pose[0, :3, :3], r1_pose[0, :3, 3] = rotation, translation
        result = MemoryGuidedRegistration(config()).run(source, target, source_features, target_features, initial_pose=r1_pose)
        self.assertIsNotNone(result.best)
        self.assertTrue(result.round_logs)
        self.assertGreaterEqual(result.best.inlier_ids.numel(), 4)
        self.assertTrue(torch.allclose(result.pose[0, :3, :3], rotation, atol=1e-4))
        self.assertTrue(torch.allclose(result.pose[0, :3, 3], translation, atol=1e-4))
        self.assertEqual(result.best.pose_raw.shape, result.best.pose_local_refined.shape)
        with self.assertRaises(ValueError):
            MemoryGuidedRegistration(config()).run(source, target, source_features, target_features)

    def test_raw_parent_survives_rejected_refinement_child(self):
        class RejectingRefiner(MemoryGuidedRegistration):
            def _robust_refine(self, memory, pose):
                rejected = pose.clone()
                rejected[0, 0, 3] += 1.0
                return rejected

        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        r1_pose = torch.eye(4)[None]
        r1_pose[0, :3, :3], r1_pose[0, :3, 3] = rotation, translation
        result = RejectingRefiner(config(
            memory_max_rounds=1,
            memory_tau_linear=0.0,
            memory_tau_planar=0.0,
            memory_min_coverage=0.0,
            memory_min_cross_group_agreement=0.0,
        )).run(source, target, source_features, target_features, initial_pose=r1_pose)
        raw_ids = {item.hypothesis_id for item in result.raw_hypotheses}
        post_ids = {item.hypothesis_id for item in result.post_refinement_hypotheses}
        self.assertTrue(raw_ids.issubset(post_ids))
        self.assertTrue(any(item.stage == "raw" and item.inlier_ids.numel() >= 4 for item in result.raw_hypotheses))
        self.assertFalse(any(item.stage == "refinement_child" for item in result.post_refinement_hypotheses))
        self.assertTrue(any(row["stage"] == "refinement_child" and not row["accepted"] for row in result.candidate_logs))
        self.assertEqual(result.best.hypothesis_id, result.r1_hypothesis.hypothesis_id)

    def test_memory_oracle_prefix_keeps_raw_parent_and_hides_future_success(self):
        identity = torch.eye(4)[None]
        failed_r1_pose = identity.clone()
        failed_r1_pose[0, 0, 3] = 1.0
        r1 = self.hypothesis(0, 0, failed_r1_pose, stage="r1", score=2.0)
        raw = self.hypothesis(1, 1, identity, score=1.0)
        rejected_child = self.hypothesis(2, 1, failed_r1_pose, stage="refinement_child", parent_id=1, score=0.0)
        result = SimpleNamespace(raw_hypotheses=[r1, raw], post_refinement_hypotheses=[r1, raw])
        raw_audits = [audit_hypothesis(item, identity, 15.0, 30.0) for item in result.raw_hypotheses]
        post_audits = [audit_hypothesis(item, identity, 15.0, 30.0) for item in result.post_refinement_hypotheses]
        before = memory_round_audit(result, raw_audits, post_audits, 0, 15.0, 30.0)
        after = memory_round_audit(result, raw_audits, post_audits, 1, 15.0, 30.0)
        self.assertEqual(before["cumulative_raw_oracle_success"], 0)
        self.assertEqual(after["cumulative_raw_oracle_success"], 1)
        self.assertEqual(after["cumulative_post_refinement_oracle_success"], 1)
        self.assertNotIn(rejected_child.hypothesis_id, {item.hypothesis_id for item in result.post_refinement_hypotheses})


if __name__ == "__main__":
    unittest.main()
