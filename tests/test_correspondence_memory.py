import json
import unittest

import torch

from correspondence_memory import CorrespondenceMemory
from iterative_ray_config import IterativeRayConfig
from memory_graph_config import MemoryGraphConfig
from memory_guided_registration import MemoryGuidedRegistration


def config(**overrides):
    values = {
        "method": "memory_graph",
        "memory_topk": 3,
        "memory_graph_neighbors": 6,
        "memory_sigma_g": 0.05,
        "memory_tau_g": 0.5,
        "memory_support_min": 3,
        "memory_support_max": 6,
        "memory_min_eigen_entropy": 0.05,
        "memory_min_coverage": 0.2,
        "memory_coverage_voxel_size": 0.05,
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

    def test_basin_penalty_requires_matching_support_signature(self):
        source, target, source_features, target_features, _, _ = synthetic_pair()
        memory = CorrespondenceMemory(source, target, source_features, target_features, config())
        support = torch.where(memory.src_indices == memory.tgt_indices)[0][:6]
        signature = memory.support_signature(support)
        pose = torch.eye(4)[None]
        memory.update_basin(pose, signature, 1.0, improved=False)
        self.assertLess(memory.basin_bonus(pose, signature), 0.0)
        self.assertEqual(memory.basin_bonus(pose, torch.zeros_like(signature)), 0.0)

    def test_memory_search_estimates_full_rigid_pose_and_preserves_raw_seed_pose(self):
        source, target, source_features, target_features, rotation, translation = synthetic_pair()
        result = MemoryGuidedRegistration(config()).run(source, target, source_features, target_features)
        self.assertIsNotNone(result.best)
        self.assertTrue(result.round_logs)
        self.assertGreaterEqual(result.best.inlier_ids.numel(), 4)
        self.assertTrue(torch.allclose(result.pose[0, :3, :3], rotation, atol=1e-4))
        self.assertTrue(torch.allclose(result.pose[0, :3, 3], translation, atol=1e-4))
        self.assertEqual(result.best.pose_raw.shape, result.best.pose_local_refined.shape)


if __name__ == "__main__":
    unittest.main()
