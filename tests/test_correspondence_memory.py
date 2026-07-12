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
    def hypothesis(
        hypothesis_id,
        round_id,
        pose,
        stage="raw",
        parent_id=-1,
        score=0.0,
        search_score=0.0,
        structure_penalty=0.0,
        basin_adjustment=0.0,
    ):
        return MemoryHypothesis(
            hypothesis_id=hypothesis_id,
            parent_id=parent_id,
            round_id=round_id,
            stage=stage,
            pose_raw=pose,
            pose_local_refined=pose,
            support_ids=torch.empty(0, dtype=torch.long),
            inlier_ids=torch.empty(0, dtype=torch.long),
            residuals=torch.empty(0),
            validation_score=score,
            search_score=search_score,
            inlier_count=0,
            mean_inlier_error=float("inf"),
            coverage=0.0,
            support_signature=torch.zeros(5),
            basin_key=None,
            structure_penalty=structure_penalty,
            basin_adjustment=basin_adjustment,
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

    def test_posterior_and_relation_updates_use_independent_evidence(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        correct = torch.where(memory.src_indices == memory.tgt_indices)[0]
        wrong = torch.where(memory.src_indices != memory.tgt_indices)[0][0:1]
        alpha_before, beta_before = memory.alpha.clone(), memory.beta.clone()
        memory.update_pair_posterior(correct, wrong)
        self.assertGreater(float(memory.alpha[correct[0]]), float(alpha_before[correct[0]]))
        self.assertGreater(float(memory.beta[wrong[0]]), float(beta_before[wrong[0]]))
        memory.update_relation_graph(correct, wrong)
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

    def test_disabled_memory_layers_do_not_update_and_fixed_estimation_is_unity(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(
            source,
            target,
            source_features,
            target_features,
            config(memory_use_reliability=False, memory_use_relation_history=False, memory_use_basin=False),
        )
        support = torch.arange(4)
        alpha_before, beta_before = memory.alpha.clone(), memory.beta.clone()
        self.assertEqual(memory.update_pair_posterior(support, torch.empty(0, dtype=torch.long)), 0.0)
        self.assertTrue(torch.equal(memory.alpha, alpha_before))
        self.assertTrue(torch.equal(memory.beta, beta_before))
        fixed = memory.fixed_estimation_weights(support)
        self.assertTrue(torch.equal(fixed, torch.ones(support.numel(), dtype=memory.dtype)))
        self.assertTrue(torch.equal(memory.estimation_weights(support), fixed))
        signature = memory.support_signature(support)
        self.assertEqual(memory.update_basin(torch.eye(4)[None], signature, 0.0, improved=False, support_ids=support), (None, None))
        self.assertEqual(len(memory.basins), 0)
        self.assertEqual(float(memory.presearch_basin_penalty(signature, support)), 0.0)

    def test_search_reliability_changes_but_fixed_estimation_weights_do_not(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        support = torch.arange(4)
        search_before = memory.search_reliability(support).clone()
        fixed_before = memory.fixed_estimation_weights(support)
        memory.alpha[support] = 100.0
        memory.beta[support] = 0.1
        search_after = memory.search_reliability(support)
        fixed_after = memory.fixed_estimation_weights(support)
        self.assertFalse(torch.allclose(search_before, search_after))
        self.assertTrue(torch.equal(fixed_before, fixed_after))
        self.assertTrue(torch.equal(fixed_after, torch.ones_like(fixed_after)))

    def test_validation_score_is_history_independent(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        evidence = registrar._verify(memory, pose)
        before = registrar._vdce_validation_score(evidence)
        rank_before = memory.rank().clone()
        memory.alpha.fill_(100.0)
        memory.beta.fill_(0.1)
        memory.edge_success.fill_(50.0)
        memory.edge_failure.zero_()
        support = evidence.verified_candidate_inliers[:config().memory_support_max]
        memory.update_basin(pose, memory.support_signature(support), before[0], improved=False, support_ids=support)
        after = registrar._vdce_validation_score(registrar._verify(memory, pose))
        self.assertFalse(torch.allclose(rank_before, memory.rank()))
        self.assertAlmostEqual(before[0], after[0], places=8)
        self.assertEqual(before[1], after[1])
        self.assertAlmostEqual(before[2], after[2], places=8)
        self.assertAlmostEqual(before[3], after[3], places=8)

    def test_refinement_is_history_independent(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        target = target.clone()
        target[0, :, 0] += torch.linspace(-0.002, 0.002, target.shape[1])
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        refined_before = registrar._robust_refine(memory, pose)
        memory.alpha.copy_(torch.linspace(1.0, 100.0, memory.count))
        memory.beta.copy_(torch.linspace(100.0, 1.0, memory.count))
        refined_after = registrar._robust_refine(memory, pose)
        self.assertTrue(torch.allclose(refined_before, refined_after, atol=1e-6, rtol=0.0))

    def test_structure_penalty_cannot_change_selector(self):
        pose = torch.eye(4)[None]
        first = self.hypothesis(1, 1, pose, score=2.0, search_score=-100.0, structure_penalty=1e6)
        second = self.hypothesis(2, 1, pose, score=1.0, search_score=1e6)
        self.assertIs(MemoryGuidedRegistration._select_best([first, second]), first)
        self.assertEqual(first.score, first.validation_score)

    def test_basin_adjustment_cannot_change_selector(self):
        pose = torch.eye(4)[None]
        first = self.hypothesis(1, 1, pose, score=2.0, basin_adjustment=-1e6)
        second = self.hypothesis(2, 1, pose, score=1.0, basin_adjustment=1e6)
        self.assertIs(MemoryGuidedRegistration._select_best([first, second]), first)

    def test_refinement_acceptance_uses_validation_score_only(self):
        pose = torch.eye(4)[None]
        registrar = MemoryGuidedRegistration(config())
        parent = self.hypothesis(1, 1, pose, score=1.0, search_score=100.0)
        better_validation = self.hypothesis(2, 1, pose, score=2.0, search_score=-100.0)
        worse_validation = self.hypothesis(3, 1, pose, score=0.0, search_score=1e6)
        self.assertTrue(registrar._accept_refined_child(parent, better_validation)[0])
        self.assertFalse(registrar._accept_refined_child(parent, worse_validation)[0])

    def test_same_support_produces_same_raw_pose_with_history_toggles(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        full = CorrespondenceMemory(source, target, source_features, target_features, config())
        no_history = CorrespondenceMemory(
            source,
            target,
            source_features,
            target_features,
            config(memory_use_reliability=False, memory_use_relation_history=False, memory_use_basin=False),
        )
        support = torch.where(full.src_indices == full.tgt_indices)[0][:6]
        full.alpha.fill_(100.0)
        full.beta.fill_(0.1)
        full.edge_success.fill_(25.0)
        src_full, tgt_full = full.correspondence_points(support)
        src_no_history, tgt_no_history = no_history.correspondence_points(support)
        registrar = MemoryGuidedRegistration(config())
        full_pose = registrar._weighted_rigid(src_full[0], tgt_full[0], full.fixed_estimation_weights(support))
        no_history_pose = registrar._weighted_rigid(src_no_history[0], tgt_no_history[0], no_history.fixed_estimation_weights(support))
        self.assertFalse(torch.allclose(full.rank(), no_history.rank()))
        self.assertTrue(torch.allclose(full_pose, no_history_pose, atol=1e-7, rtol=0.0))

    def test_signature_support_is_history_independent(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory_config = config(memory_support_max=3)
        memory = CorrespondenceMemory(source, target, source_features, target_features, memory_config)
        registrar = MemoryGuidedRegistration(memory_config)
        inliers = torch.ones(memory.count, dtype=torch.bool)
        before = registrar._signature_support(memory, inliers)
        memory.alpha.copy_(torch.linspace(1.0, 100.0, memory.count))
        memory.beta.copy_(torch.linspace(100.0, 1.0, memory.count))
        memory.edge_success.fill_(50.0)
        after = registrar._signature_support(memory, inliers)
        expected = torch.topk(memory.descriptor_score, k=memory_config.memory_support_max).indices
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(after, expected))

    def test_candidate_log_exposes_decoupled_scores(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        result = MemoryGuidedRegistration(config(memory_max_rounds=1)).run(
            source,
            target,
            source_features,
            target_features,
            initial_pose=pose,
        )
        scored = [row for row in result.candidate_logs if row["stage"] in {"r1", "raw", "refinement_child"}]
        required = {
            "validation_score",
            "search_score",
            "inlier_count",
            "mean_inlier_error",
            "coverage",
            "basin_adjustment",
            "structure_penalty",
        }
        self.assertTrue(scored)
        self.assertTrue(all(required.issubset(row) for row in scored))
        self.assertTrue(all(row["score"] == row["validation_score"] for row in scored))
        raw_by_id = {row["hypothesis_id"]: row for row in scored if row["stage"] == "raw"}
        children = [row for row in scored if row["stage"] == "refinement_child"]
        self.assertTrue(children)
        self.assertTrue(all(row["search_score"] == raw_by_id[row["parent_id"]]["search_score"] for row in children))

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

    def test_one_to_one_filter_removes_duplicate_sources_and_targets(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        candidates = torch.arange(6)
        filtered = memory.one_to_one_filter(candidates)
        self.assertEqual(int(torch.unique(memory.src_indices[filtered]).numel()), filtered.numel())
        self.assertEqual(int(torch.unique(memory.tgt_indices[filtered]).numel()), filtered.numel())
        self.assertLessEqual(filtered.numel(), candidates.numel())

    def test_one_to_one_filter_orders_by_descriptor_score(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        candidates = torch.arange(6)
        filtered = memory.one_to_one_filter(candidates)
        scores = memory.descriptor_score[filtered]
        self.assertTrue(torch.equal(scores, torch.sort(scores, descending=True).values))

    def test_one_to_one_filter_uses_smallest_residual_for_refinement(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        candidates = torch.where(memory.src_indices == 0)[0]
        residuals = torch.full((memory.count,), 10.0)
        residuals[candidates[0]] = 0.8
        residuals[candidates[1]] = 0.1
        residuals[candidates[2]] = 0.4
        filtered = memory.one_to_one_filter(candidates, residuals)
        self.assertEqual(filtered.tolist(), [int(candidates[1])])

    def test_independent_validation_uses_full_sampled_domain_and_mutual_pairs(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        evidence = registrar._verify(memory, pose)
        self.assertEqual(evidence.source_indices.numel(), source.shape[1])
        self.assertEqual(torch.unique(evidence.target_indices).numel(), evidence.target_indices.numel())
        self.assertEqual(evidence.unique_inlier_ratio, 1.0)
        bad_pose = torch.eye(4)[None]
        bad_pose[0, 0, 3] = 10.0
        bad_evidence = registrar._verify(memory, bad_pose)
        self.assertEqual(bad_evidence.unique_inlier_ratio, 0.0)

    def test_dynamic_candidate_rebuild_preserves_existing_relation_history(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        all_ids = torch.arange(memory.count)
        memory.update_relation_graph(all_ids, torch.empty(0, dtype=torch.long))
        before = {
            (int(row), int(col)): float(value)
            for row, col, value in zip(memory.edge_rows.tolist(), memory.edge_cols.tolist(), memory.edge_success.tolist())
            if value > 0
        }
        existing = set(zip(memory.src_indices.tolist(), memory.tgt_indices.tolist()))
        new_pair = next((src_id, tgt_id) for src_id in range(source.shape[1]) for tgt_id in range(target.shape[1]) if (src_id, tgt_id) not in existing)
        added = memory.add_candidates(
            torch.tensor([new_pair[0]]),
            torch.tensor([new_pair[1]]),
            torch.tensor([0.5]),
        )
        self.assertEqual(added, 1)
        after = {
            (int(row), int(col)): float(value)
            for row, col, value in zip(memory.edge_rows.tolist(), memory.edge_cols.tolist(), memory.edge_success.tolist())
        }
        retained = [edge for edge, value in before.items() if edge in after and after[edge] == value]
        self.assertTrue(retained)
        self.assertIn(memory.count - 1, memory._src_to_candidate_ids[new_pair[0]])
        self.assertEqual(memory.check_graph_symmetry(), 0)

    def test_vdce_validation_score_uses_unique_inlier_ratio(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        evidence = registrar._verify(memory, pose)
        score, count, med_res, coverage, ratio, bidir, _ = registrar._vdce_validation_score(evidence)
        self.assertGreater(score, 0.0)
        self.assertGreater(count, 0)
        self.assertLessEqual(ratio, 1.0)
        self.assertGreater(ratio, 0.0)

    def test_memory_update_gate_blocks_marginally_better_pose(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        best = registrar._make_hypothesis(memory, 0, -1, 0, "r1", pose, pose, torch.arange(6))
        current = registrar._make_hypothesis(memory, 1, 0, 1, "raw", pose, pose, torch.arange(6))
        current.validation_score = best.validation_score + 0.001
        self.assertFalse(registrar._memory_update_gate(current, best))

    def test_memory_update_gate_accepts_significantly_better_pose(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        best = registrar._make_hypothesis(memory, 0, -1, 0, "r1", pose, pose, torch.arange(6))
        current = registrar._make_hypothesis(memory, 1, 0, 1, "raw", pose, pose, torch.arange(6))
        current.validation_score = best.validation_score + 2.0
        current.unique_inlier_ratio = 0.3
        current.coverage = 0.2
        current.median_residual = 0.01
        self.assertTrue(registrar._memory_update_gate(current, best))

    def test_memory_update_gate_blocks_low_quality_pose(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config(
            memory_vdce_min_inlier_ratio=0.5,
            memory_vdce_min_coverage=0.3,
        ))
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        best = registrar._make_hypothesis(memory, 0, -1, 0, "r1", pose, pose, torch.arange(6))
        current = registrar._make_hypothesis(memory, 1, 0, 1, "raw", pose, pose, torch.arange(6))
        current.validation_score = best.validation_score + 2.0
        current.unique_inlier_ratio = 0.01
        current.coverage = 0.01
        self.assertFalse(registrar._memory_update_gate(current, best))

    def test_graph_is_bidirectional(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        failures = memory.check_graph_symmetry()
        self.assertEqual(failures, 0)

    def test_candidate_expansion_adds_new_candidates(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        original_count = memory.count
        new_src, new_tgt, new_scores = memory.pose_guided_candidate_expansion(
            pose, source_features, target_features,
        )
        nearest, mutual, _ = memory._mutual_nearest(memory._transform(memory.src_points, pose), memory.tgt_points, config().memory_vdce_expand_radius)
        self.assertTrue(bool(mutual[new_src].all()))
        self.assertTrue(torch.equal(nearest[new_src], new_tgt))
        added = memory.add_candidates(new_src, new_tgt, new_scores)
        self.assertGreaterEqual(memory.count, original_count)
        self.assertEqual(memory.count, original_count + added)
        self.assertEqual(memory.check_graph_symmetry(), 0)

    def test_validation_score_normalized_to_zero_one_range(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        score, *_ = registrar._vdce_validation_score(registrar._verify(memory, pose))
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, -1.0)

    def test_round_logs_include_memory_accept_and_expansion_fields(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        r1_pose = torch.eye(4)[None]
        r1_pose[0, :3, :3], r1_pose[0, :3, 3] = rotation, translation
        result = MemoryGuidedRegistration(config(memory_max_rounds=1)).run(
            source, target, source_features, target_features,
            initial_pose=r1_pose,
            initial_src_indices=torch.arange(6),
            initial_tgt_indices=torch.arange(6),
        )
        for row in result.round_logs:
            if row["round_id"] > 0:
                self.assertIn("memory_accepted", row)
                self.assertIn("candidate_expansion_count", row)

    def test_robust_refine_uses_one_to_one_matching(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        target = target.clone()
        target[0, :, 0] += torch.linspace(-0.002, 0.002, target.shape[1])
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        registrar = MemoryGuidedRegistration(config())
        pose = torch.eye(4)[None]
        pose[0, :3, :3], pose[0, :3, 3] = rotation, translation
        refined = registrar._robust_refine(memory, pose, use_one_to_one=True)
        self.assertTrue(torch.isfinite(refined).all())
        self.assertEqual(refined.shape, (1, 4, 4))


if __name__ == "__main__":
    unittest.main()
