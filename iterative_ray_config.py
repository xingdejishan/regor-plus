from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class IterativeRayConfig:
    method: str = "iterative_ray"
    search_max_rounds: int = 4
    search_time_budget_seconds: float = 0.0
    candidates_per_round: int = 16
    independent_explore_fraction: float = 0.2
    descriptor_topk: int = 16
    seed_group_count: int = 32
    local_corr_max_points: int = 64
    local_knn_radius: float = 0.30
    local_mutual_k: int = 3
    active_rays_per_round: int = 3000
    search_frame_fraction: float = 0.70
    ray_trunc_margin: float = 0.05
    ray_surface_sigma: float = 0.03
    min_valid_ray_count: int = 200
    max_constraints_per_frame: int = 128
    pose_nms_rotation_deg: float = 5.0
    pose_nms_translation: float = 0.10
    pose_nms_threshold: float = 1.0
    history_rotation_radius_deg: float = 10.0
    history_translation_radius: float = 0.20
    history_signature_similarity: float = 0.90
    history_energy_tolerance: float = 0.01
    escape_lambda: float = 1.0
    history_lambda: float = 1.0
    validation_surface_weight: float = 1.0
    validation_min_improvement: float = 0.01
    validation_min_surface_support: float = 0.0
    stagnation_rounds: int = 2
    enable_adaptive_stop: bool = False
    enable_gate: bool = False
    enable_candidate_merge: bool = False
    enable_ctc: bool = False
    ray_depth_scale: float = 1000.0
    ray_min_depth: float = 0.10
    ray_max_depth: float = 8.0
    ray_stride: int = 8
    ray_max_frames: int = 0
    ray_manifest: str = ""

    @classmethod
    def from_mapping(cls, values):
        if not isinstance(values, dict):
            values = dict(values)
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise KeyError(f"Unknown iterative_ray config keys: {', '.join(unknown)}")
        missing = sorted(allowed - set(values))
        if missing:
            raise KeyError(f"Missing iterative_ray config keys: {', '.join(missing)}")
        return cls(**{key: values[key] for key in allowed})

    def validate(self):
        if self.method not in {"r1_only", "iterative_ray", "repeated_regor", "shuffled_ray"}:
            raise ValueError("method must be r1_only, iterative_ray, repeated_regor, or shuffled_ray.")
        if self.search_max_rounds < 0 or self.candidates_per_round < 1:
            raise ValueError("search_max_rounds must be non-negative and candidates_per_round must be positive.")
        if self.method != "r1_only" and self.search_max_rounds < 1:
            raise ValueError("search methods require search_max_rounds >= 1.")
        if self.search_time_budget_seconds < 0:
            raise ValueError("search_time_budget_seconds must be >= 0.")
        if not 0.0 <= self.independent_explore_fraction <= 1.0:
            raise ValueError("independent_explore_fraction must be in [0, 1].")
        if self.descriptor_topk < 1 or self.seed_group_count < 1 or self.local_corr_max_points < 3:
            raise ValueError("descriptor_topk, seed_group_count, and local_corr_max_points are invalid.")
        if self.local_knn_radius <= 0 or self.local_mutual_k < 1:
            raise ValueError("local correspondence settings are invalid.")
        if self.active_rays_per_round < 1 or not 0.0 < self.search_frame_fraction < 1.0:
            raise ValueError("active_rays_per_round or search_frame_fraction is invalid.")
        if self.ray_trunc_margin <= 0 or self.ray_surface_sigma <= 0 or self.min_valid_ray_count < 1:
            raise ValueError("ray evidence settings are invalid.")
        if not self.ray_manifest:
            raise ValueError("ray_manifest must be a non-empty fragment manifest path.")
        if self.max_constraints_per_frame < 1:
            raise ValueError("max_constraints_per_frame must be positive.")
        if self.pose_nms_rotation_deg <= 0 or self.pose_nms_translation <= 0 or self.pose_nms_threshold <= 0:
            raise ValueError("pose NMS settings are invalid.")
        if self.history_rotation_radius_deg <= 0 or self.history_translation_radius <= 0:
            raise ValueError("history basin radii must be positive.")
        if not 0.0 <= self.history_signature_similarity <= 1.0:
            raise ValueError("history_signature_similarity must be in [0, 1].")
        if self.escape_lambda < 0 or self.history_lambda < 0 or self.validation_surface_weight < 0:
            raise ValueError("score weights must be non-negative.")
        if self.validation_min_improvement < 0 or self.stagnation_rounds < 1:
            raise ValueError("validation_min_improvement or stagnation_rounds is invalid.")
        if self.enable_candidate_merge:
            raise ValueError("enable_candidate_merge is not implemented in iterative_ray core mode.")
        if self.enable_ctc:
            raise ValueError("enable_ctc is not implemented in iterative_ray core mode.")
        return self

    def report(self):
        return asdict(self)
