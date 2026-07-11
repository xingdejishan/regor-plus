from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class MemoryGraphConfig:
    method: str = "memory_graph"
    memory_topk: int = 20
    memory_graph_neighbors: int = 32
    memory_sigma_g: float = 0.10
    memory_tau_g: float = 0.60
    memory_alpha0: float = 1.0
    memory_beta0: float = 1.0
    memory_descriptor_prior: float = 2.0
    memory_forgetting: float = 0.95
    memory_eta_positive: float = 1.0
    memory_eta_negative: float = 1.0
    memory_eta_edge_positive: float = 1.0
    memory_eta_edge_negative: float = 1.0
    memory_evidence_cap: float = 100.0
    memory_lambda_descriptor: float = 1.0
    memory_lambda_reliability: float = 1.0
    memory_lambda_graph: float = 1.0
    memory_lambda_edge: float = 1.0
    memory_lambda_degeneracy: float = 1.0
    memory_lambda_edge_success: float = 1.0
    memory_lambda_edge_failure: float = 1.0
    memory_lambda_inlier: float = 1.0
    memory_lambda_error: float = 1.0
    memory_lambda_coverage: float = 1.0
    memory_lambda_basin_nonimproving: float = 1.0
    memory_lambda_basin_positive: float = 1.0
    memory_basin_presearch_penalty: float = 1.0
    memory_basin_presearch_min_support_overlap: float = 0.50
    memory_support_min: int = 3
    memory_support_max: int = 32
    memory_support_trial_count: int = 8
    memory_signature_entropy_temperature: float = 0.10
    memory_min_lambda12_ratio: float = 0.10
    memory_min_lambda13_ratio: float = 0.05
    memory_min_coverage: float = 0.002
    memory_coverage_voxel_size: float = 0.10
    memory_cross_group_voxel_size: float = 0.30
    memory_min_cross_group_agreement: float = 0.50
    memory_hypotheses_per_round: int = 24
    memory_max_rounds: int = 8
    memory_max_sampling_attempts: int = 96
    memory_fixed_budget_mode: bool = True
    memory_inlier_threshold: float = 0.10
    memory_tls_threshold: float = 0.10
    memory_tls_iters: int = 3
    memory_refine_trust_rotation_deg: float = 15.0
    memory_refine_trust_translation: float = 0.30
    memory_refine_min_score_improvement: float = 0.01
    memory_r1_mapping_radius: float = 1e-5
    memory_r1_min_mapping_ratio: float = 0.80
    memory_r1_low_confidence_inlier_ratio: float = 0.50
    memory_r1_low_confidence_error_ratio: float = 1.00
    memory_prosac_initial_fraction: float = 0.20
    memory_prosac_growth: float = 0.15
    memory_basin_rotation_deg: float = 5.0
    memory_basin_translation: float = 0.10
    memory_basin_max: int = 256
    memory_basin_signature_momentum: float = 0.90
    memory_basin_signature_similarity: float = 0.90
    memory_compress_epsilon: float = 0.01
    memory_patience: int = 3
    memory_score_epsilon: float = 0.005
    memory_pose_epsilon_rotation_deg: float = 0.20
    memory_pose_epsilon_translation_multiplier: float = 0.50
    memory_novelty_threshold: float = 0.10
    memory_delta_threshold: float = 0.01
    memory_strong_stop_min_inliers: int = 30
    memory_strong_stop_inlier_fraction: float = 0.02
    memory_strong_stop_error_ratio: float = 0.75
    memory_strong_stop_coverage: float = 0.20
    memory_use_reliability: bool = True
    memory_use_relation_history: bool = True
    memory_use_basin: bool = True
    memory_require_r1_cache: bool = True

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict):
            values = dict(values)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise KeyError(f"Unknown memory_graph config keys: {', '.join(unknown)}")
        missing = sorted(allowed - set(values))
        if missing:
            raise KeyError(f"Missing memory_graph config keys: {', '.join(missing)}")
        return cls(**{key: values[key] for key in allowed})

    def validate(self):
        if self.method != "memory_graph":
            raise ValueError("memory_graph.method must be memory_graph.")
        if self.memory_topk < 3 or self.memory_graph_neighbors < 1 or self.memory_support_min < 3 or self.memory_support_max < self.memory_support_min or self.memory_support_trial_count < 1:
            raise ValueError("memory candidate and support settings are invalid.")
        if self.memory_sigma_g <= 0 or not 0 < self.memory_tau_g <= 1 or self.memory_alpha0 <= 0 or self.memory_beta0 <= 0:
            raise ValueError("memory geometry and posterior priors are invalid.")
        if not 0 < self.memory_forgetting <= 1 or self.memory_evidence_cap <= 0:
            raise ValueError("memory forgetting or evidence cap is invalid.")
        if min(self.memory_eta_positive, self.memory_eta_negative, self.memory_eta_edge_positive, self.memory_eta_edge_negative) < 0:
            raise ValueError("memory update steps must be non-negative.")
        if min(
            self.memory_descriptor_prior,
            self.memory_lambda_descriptor,
            self.memory_lambda_reliability,
            self.memory_lambda_graph,
            self.memory_lambda_edge,
            self.memory_lambda_degeneracy,
            self.memory_lambda_edge_success,
            self.memory_lambda_edge_failure,
            self.memory_lambda_inlier,
            self.memory_lambda_error,
            self.memory_lambda_coverage,
            self.memory_lambda_basin_nonimproving,
            self.memory_lambda_basin_positive,
            self.memory_basin_presearch_penalty,
        ) < 0:
            raise ValueError("memory score weights must be non-negative.")
        if self.memory_signature_entropy_temperature <= 0 or not 0 <= self.memory_min_lambda12_ratio <= 1 or not 0 <= self.memory_min_lambda13_ratio <= 1 or not 0 <= self.memory_min_coverage <= 1 or self.memory_coverage_voxel_size <= 0 or self.memory_cross_group_voxel_size <= 0 or not 0 <= self.memory_min_cross_group_agreement <= 1 or not 0 <= self.memory_basin_presearch_min_support_overlap <= 1:
            raise ValueError("memory structural constraints are invalid.")
        if self.memory_hypotheses_per_round < 1 or self.memory_max_rounds < 1 or self.memory_max_sampling_attempts < self.memory_hypotheses_per_round or self.memory_inlier_threshold <= 0 or self.memory_tls_threshold <= 0 or self.memory_tls_iters < 0:
            raise ValueError("memory search budget is invalid.")
        if self.memory_refine_trust_rotation_deg <= 0 or self.memory_refine_trust_translation <= 0 or self.memory_refine_min_score_improvement < 0:
            raise ValueError("memory refinement settings are invalid.")
        if self.memory_r1_mapping_radius <= 0 or not 0 <= self.memory_r1_min_mapping_ratio <= 1 or not 0 <= self.memory_r1_low_confidence_inlier_ratio <= 1 or self.memory_r1_low_confidence_error_ratio <= 0:
            raise ValueError("memory R1 initialization settings are invalid.")
        if not 0 < self.memory_prosac_initial_fraction <= 1 or self.memory_prosac_growth < 0:
            raise ValueError("memory PROSAC schedule is invalid.")
        if self.memory_basin_rotation_deg <= 0 or self.memory_basin_translation <= 0 or self.memory_basin_max < 1 or not 0 <= self.memory_basin_signature_momentum < 1 or not 0 <= self.memory_basin_signature_similarity <= 1:
            raise ValueError("memory basin configuration is invalid.")
        if self.memory_compress_epsilon < 0 or self.memory_patience < 1 or self.memory_score_epsilon < 0 or self.memory_pose_epsilon_rotation_deg < 0 or self.memory_pose_epsilon_translation_multiplier < 0 or not 0 <= self.memory_novelty_threshold <= 1 or self.memory_delta_threshold < 0:
            raise ValueError("memory stopping configuration is invalid.")
        if self.memory_strong_stop_min_inliers < 3 or not 0 < self.memory_strong_stop_inlier_fraction <= 1 or self.memory_strong_stop_error_ratio <= 0 or not 0 <= self.memory_strong_stop_coverage <= 1:
            raise ValueError("memory strong-stop configuration is invalid.")
        if not self.memory_require_r1_cache:
            raise ValueError("memory_graph requires fixed R1 cache for fair repair evaluation.")
        return self

    def report(self):
        return asdict(self)
