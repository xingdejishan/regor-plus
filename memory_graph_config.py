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
    memory_tau_linear: float = 0.01
    memory_tau_planar: float = 0.05
    memory_min_coverage: float = 0.002
    memory_coverage_voxel_size: float = 0.10
    memory_cross_group_voxel_size: float = 0.30
    memory_min_cross_group_agreement: float = 0.50
    memory_lambda_structure_planar: float = 1.0
    memory_lambda_structure_coverage: float = 100.0
    memory_lambda_structure_cross_group: float = 1.0
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
    # VDCE independent validation weights
    memory_vdce_w_r: float = 0.45
    memory_vdce_w_e: float = 0.20
    memory_vdce_w_c: float = 0.15
    memory_vdce_w_b: float = 0.10
    memory_vdce_w_f: float = 0.10
    memory_vdce_normalize_residual_scale: float = 0.10
    # VDCE memory update gating
    memory_vdce_min_score_margin: float = 0.01
    memory_vdce_min_inlier_ratio: float = 0.05
    memory_vdce_min_coverage: float = 0.01
    memory_vdce_max_median_residual: float = 0.50
    memory_vdce_max_conflict_ratio: float = 0.30
    # VDCE pose-guided candidate expansion
    memory_vdce_expand_radius: float = 0.15
    memory_vdce_expand_max_per_source: int = 3
    memory_vdce_expand_max_total: int = 64
    memory_vdce_expand_descriptor_threshold: float = 0.30

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
        if self.memory_signature_entropy_temperature <= 0 or not 0 <= self.memory_tau_linear <= 1 or not 0 <= self.memory_tau_planar <= 1 or not 0 <= self.memory_min_coverage <= 1 or self.memory_coverage_voxel_size <= 0 or self.memory_cross_group_voxel_size <= 0 or not 0 <= self.memory_min_cross_group_agreement <= 1 or not 0 <= self.memory_basin_presearch_min_support_overlap <= 1:
            raise ValueError("memory structural constraints are invalid.")
        if min(self.memory_lambda_structure_planar, self.memory_lambda_structure_coverage, self.memory_lambda_structure_cross_group) < 0:
            raise ValueError("memory structure penalty weights must be non-negative.")
        if self.memory_hypotheses_per_round < 1 or self.memory_max_rounds < 1 or self.memory_max_sampling_attempts < self.memory_hypotheses_per_round or self.memory_inlier_threshold <= 0 or self.memory_tls_threshold <= 0 or self.memory_tls_iters < 0:
            raise ValueError("memory search budget is invalid.")
        if self.memory_refine_trust_rotation_deg <= 0 or self.memory_refine_trust_translation <= 0 or self.memory_refine_min_score_improvement < 0:
            raise ValueError("memory refinement settings are invalid.")
        if not 0 <= self.memory_r1_min_mapping_ratio <= 1 or not 0 <= self.memory_r1_low_confidence_inlier_ratio <= 1 or self.memory_r1_low_confidence_error_ratio <= 0:
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
        if min(self.memory_vdce_w_r, self.memory_vdce_w_e, self.memory_vdce_w_c, self.memory_vdce_w_b, self.memory_vdce_w_f) < 0:
            raise ValueError("memory VDCE validation weights must be non-negative.")
        if self.memory_vdce_normalize_residual_scale <= 0:
            raise ValueError("memory VDCE residual normalization scale must be positive.")
        if self.memory_vdce_min_score_margin < 0:
            raise ValueError("memory VDCE min score margin must be non-negative.")
        if not 0 <= self.memory_vdce_min_inlier_ratio <= 1:
            raise ValueError("memory VDCE min inlier ratio must be in [0, 1].")
        if not 0 <= self.memory_vdce_min_coverage <= 1:
            raise ValueError("memory VDCE min coverage must be in [0, 1].")
        if self.memory_vdce_max_median_residual <= 0:
            raise ValueError("memory VDCE max median residual must be positive.")
        if not 0 <= self.memory_vdce_max_conflict_ratio <= 1:
            raise ValueError("memory VDCE max conflict ratio must be in [0, 1].")
        if self.memory_vdce_expand_radius <= 0:
            raise ValueError("memory VDCE expand radius must be positive.")
        if self.memory_vdce_expand_max_per_source < 1:
            raise ValueError("memory VDCE expand max per source must be at least 1.")
        if self.memory_vdce_expand_max_total < 1:
            raise ValueError("memory VDCE expand max total must be at least 1.")
        if not 0 < self.memory_vdce_expand_descriptor_threshold <= 1:
            raise ValueError("memory VDCE expand descriptor threshold must be in (0, 1].")
        return self

    def report(self):
        return asdict(self)
