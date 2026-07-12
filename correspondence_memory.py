from collections import OrderedDict
from dataclasses import dataclass
import math

import torch


@dataclass
class BasinRecord:
    n_positive: float
    n_nonimproving: float
    best_score: float
    signature_mean: torch.Tensor
    support_ids: torch.Tensor


class CorrespondenceMemory:
    def __init__(self, src_points, tgt_points, src_features, tgt_features, config, required_src_indices=None, required_tgt_indices=None):
        self.config = config
        self.src_points = self._points(src_points)
        self.tgt_points = self._points(tgt_points)
        self.device = self.src_points.device
        self.dtype = self.src_points.dtype
        self.src_indices, self.tgt_indices, self.descriptor_score, self.topk_descriptor_scores = self._build_candidates(
            src_features,
            tgt_features,
            required_src_indices,
            required_tgt_indices,
        )
        self.count = int(self.src_indices.numel())
        self.topk = int(self.topk_descriptor_scores.shape[1])
        self._src_to_candidate_ids = self._build_candidate_index(self.src_indices)
        self._tgt_to_candidate_ids = self._build_candidate_index(self.tgt_indices)
        self._pair_to_candidate_id = self._build_pair_index(self.src_indices, self.tgt_indices)
        self.alpha = config.memory_alpha0 + config.memory_descriptor_prior * self.descriptor_score
        self.beta = torch.full_like(self.alpha, float(config.memory_beta0))
        self.source_voxel_count = max(1, int(torch.unique(torch.floor(self.src_points / config.memory_coverage_voxel_size).long(), dim=0).shape[0]))
        self.edge_rows, self.edge_cols, self.static_geometry, self.row_ptr = self._build_relation_graph()
        self.edge_success = torch.zeros_like(self.static_geometry)
        self.edge_failure = torch.zeros_like(self.static_geometry)
        self.basins = OrderedDict()
        self._negative_basin_signatures = torch.empty((0, 5), dtype=self.dtype, device=self.device)
        self._negative_basin_weights = torch.empty(0, dtype=self.dtype, device=self.device)
        self._negative_basin_supports = []

    @staticmethod
    def _points(points):
        return points[0] if points.ndim == 3 else points

    @staticmethod
    def _features(features):
        return features[0] if features.ndim == 3 else features

    def _build_candidates(self, src_features, tgt_features, required_src_indices=None, required_tgt_indices=None):
        src = torch.nn.functional.normalize(self._features(src_features), dim=1)
        tgt = torch.nn.functional.normalize(self._features(tgt_features), dim=1)
        scores = src @ tgt.transpose(0, 1)
        topk = min(int(self.config.memory_topk), int(tgt.shape[0]))
        values, indices = torch.topk(scores, k=topk, dim=1)
        source = torch.arange(src.shape[0], device=src.device, dtype=torch.long)[:, None].expand_as(indices)
        flat = values.reshape(-1)
        minimum, maximum = flat.min(), flat.max()
        normalized = (flat - minimum) / torch.clamp_min(maximum - minimum, 1e-8)
        source = source.reshape(-1)
        indices = indices.reshape(-1)
        if (required_src_indices is None) != (required_tgt_indices is None):
            raise ValueError("Required source and target candidate indices must be provided together.")
        if required_src_indices is None:
            return source, indices, normalized, values
        required_source = torch.as_tensor(required_src_indices, device=src.device, dtype=torch.long).reshape(-1)
        required_target = torch.as_tensor(required_tgt_indices, device=tgt.device, dtype=torch.long).reshape(-1)
        if required_source.numel() != required_target.numel():
            raise ValueError("Required source and target candidate index counts must match.")
        if required_source.numel() == 0:
            return source, indices, normalized, values
        if int(required_source.min()) < 0 or int(required_source.max()) >= src.shape[0] or int(required_target.min()) < 0 or int(required_target.max()) >= tgt.shape[0]:
            raise ValueError("Required candidate index is out of range.")
        target_count = int(tgt.shape[0])
        candidate_keys = source * target_count + indices
        required_keys = required_source * target_count + required_target
        sorted_keys = torch.sort(candidate_keys).values
        positions = torch.searchsorted(sorted_keys, required_keys)
        present = torch.zeros_like(positions, dtype=torch.bool)
        in_bounds = positions < sorted_keys.numel()
        present[in_bounds] = sorted_keys[positions[in_bounds]] == required_keys[in_bounds]
        missing = ~present
        if not bool(missing.any()):
            return source, indices, normalized, values
        extra_source = required_source[missing]
        extra_target = required_target[missing]
        extra_score = (scores[extra_source, extra_target] - minimum) / torch.clamp_min(maximum - minimum, 1e-8)
        return (
            torch.cat([source, extra_source]),
            torch.cat([indices, extra_target]),
            torch.cat([normalized, extra_score.clamp(0.0, 1.0)]),
            values,
        )

    @staticmethod
    def _build_candidate_index(indices):
        index = {}
        for candidate_id, point_id in enumerate(indices.tolist()):
            index.setdefault(int(point_id), []).append(candidate_id)
        return index

    @staticmethod
    def _build_pair_index(src_indices, tgt_indices):
        return {
            (int(src_id), int(tgt_id)): candidate_id
            for candidate_id, (src_id, tgt_id) in enumerate(zip(src_indices.tolist(), tgt_indices.tolist()))
        }

    def candidate_ids_for_pairs(self, source_ids, target_ids):
        source_ids = torch.as_tensor(source_ids, dtype=torch.long, device=self.device).reshape(-1)
        target_ids = torch.as_tensor(target_ids, dtype=torch.long, device=self.device).reshape(-1)
        if source_ids.numel() != target_ids.numel():
            raise ValueError("Source and target pair index counts must match.")
        candidate_ids = [
            self._pair_to_candidate_id[(int(source_id), int(target_id))]
            for source_id, target_id in zip(source_ids.tolist(), target_ids.tolist())
            if (int(source_id), int(target_id)) in self._pair_to_candidate_id
        ]
        return torch.tensor(candidate_ids, dtype=torch.long, device=self.device)

    def _build_relation_graph(self):
        source_count = int(self.src_points.shape[0])
        if source_count < 2 or self.count == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_long, empty_long, torch.empty(0, dtype=self.dtype, device=self.device), torch.zeros(self.count + 1, dtype=torch.long, device=self.device)
        max_candidates = max(len(ids) for ids in self._src_to_candidate_ids.values())
        source_neighbors = min(source_count - 1, max(1, math.ceil(self.config.memory_graph_neighbors / max_candidates)))
        source_distance = torch.cdist(self.src_points, self.src_points)
        source_distance.fill_diagonal_(float("inf"))
        _, neighbors = torch.topk(source_distance, k=source_neighbors, dim=1, largest=False)
        candidate_table = torch.full(
            (source_count, max_candidates),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        for source_id, candidate_ids in self._src_to_candidate_ids.items():
            candidate_table[source_id, :len(candidate_ids)] = torch.tensor(candidate_ids, dtype=torch.long, device=self.device)
        neighbor_nodes = candidate_table[neighbors].reshape(source_count, -1)
        candidate_sources = self.src_indices
        local_nodes = neighbor_nodes[candidate_sources]
        valid_nodes = local_nodes >= 0
        safe_local_nodes = local_nodes.clamp_min(0)
        local_targets = self.tgt_indices[safe_local_nodes]
        local_sources = self.src_indices[safe_local_nodes]
        source_gap = torch.linalg.norm(self.src_points[candidate_sources, None] - self.src_points[local_sources], dim=-1)
        target_gap = torch.linalg.norm(self.tgt_points[self.tgt_indices, None] - self.tgt_points[local_targets], dim=-1)
        geometry = torch.exp(-((source_gap - target_gap) ** 2) / (2.0 * self.config.memory_sigma_g ** 2))
        valid = valid_nodes & (safe_local_nodes != torch.arange(self.count, device=self.device)[:, None]) & (self.tgt_indices[:, None] != local_targets) & (geometry > self.config.memory_tau_g)
        geometry = torch.where(valid, geometry, torch.full_like(geometry, float("-inf")))
        take = min(int(self.config.memory_graph_neighbors), int(geometry.shape[1]))
        values, positions = torch.topk(geometry, k=take, dim=1)
        valid = torch.isfinite(values)
        rows = torch.arange(self.count, device=self.device, dtype=torch.long)[:, None].expand_as(positions)
        edge_rows = rows[valid]
        edge_cols = torch.gather(local_nodes, 1, positions)[valid]
        static = values[valid]
        all_rows = torch.cat([edge_rows, edge_cols])
        all_cols = torch.cat([edge_cols, edge_rows])
        all_static = torch.cat([static, static])
        unique_edges, inverse = torch.unique(torch.stack([all_rows, all_cols], dim=1), dim=0, return_inverse=True)
        unique_static = torch.zeros(unique_edges.shape[0], dtype=all_static.dtype, device=self.device)
        unique_static.index_reduce_(0, inverse, all_static, reduce="amax")
        edge_rows = unique_edges[:, 0]; edge_cols = unique_edges[:, 1]
        static = unique_static
        counts = torch.bincount(edge_rows, minlength=self.count)
        row_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), counts.cumsum(0)])
        return edge_rows, edge_cols, static, row_ptr

    def _rebuild_relation_graph_preserving_history(self):
        old_rows = self.edge_rows
        old_cols = self.edge_cols
        old_success = self.edge_success
        old_failure = self.edge_failure
        self.edge_rows, self.edge_cols, self.static_geometry, self.row_ptr = self._build_relation_graph()
        self.edge_success = torch.zeros_like(self.static_geometry)
        self.edge_failure = torch.zeros_like(self.static_geometry)
        if old_rows.numel() == 0 or self.edge_rows.numel() == 0:
            return
        stride = int(self.count)
        old_keys = old_rows.to(torch.long) * stride + old_cols.to(torch.long)
        new_keys = self.edge_rows.to(torch.long) * stride + self.edge_cols.to(torch.long)
        sorted_old_keys, old_order = torch.sort(old_keys)
        positions = torch.searchsorted(sorted_old_keys, new_keys)
        in_bounds = positions < sorted_old_keys.numel()
        valid = torch.zeros_like(in_bounds)
        valid[in_bounds] = sorted_old_keys[positions[in_bounds]] == new_keys[in_bounds]
        if bool(valid.any()):
            old_indices = old_order[positions[valid]]
            self.edge_success[valid] = old_success[old_indices]
            self.edge_failure[valid] = old_failure[old_indices]

    @property
    def posterior_reliability(self):
        return self.alpha / torch.clamp_min(self.alpha + self.beta, 1e-8)

    @property
    def reliability(self):
        return self.posterior_reliability if self.config.memory_use_reliability else torch.ones(self.count, dtype=self.dtype, device=self.device)

    def search_reliability(self, candidate_ids):
        candidate_ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=self.device)
        return self.reliability[candidate_ids]

    def fixed_estimation_weights(self, candidate_ids):
        candidate_ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=self.device)
        return torch.ones(candidate_ids.numel(), dtype=self.dtype, device=self.device)

    def estimation_weights(self, candidate_ids):
        return self.fixed_estimation_weights(candidate_ids)

    def edge_weight(self):
        return self.static_geometry + self.config.memory_lambda_edge_success * torch.log1p(self.edge_success) - self.config.memory_lambda_edge_failure * torch.log1p(self.edge_failure)

    def graph_quality(self):
        quality = torch.zeros(self.count, dtype=self.dtype, device=self.device)
        if self.edge_rows.numel() == 0:
            return quality
        static = torch.zeros_like(quality)
        static.index_add_(0, self.edge_rows, self.static_geometry)
        degree = torch.zeros_like(quality)
        degree.index_add_(0, self.edge_rows, torch.ones_like(self.static_geometry))
        static = static / torch.clamp_min(degree, 1.0)
        if not self.config.memory_use_relation_history:
            return static
        history_weight = self.config.memory_lambda_edge_success * torch.log1p(self.edge_success) - self.config.memory_lambda_edge_failure * torch.log1p(self.edge_failure)
        quality.index_add_(0, self.edge_rows, history_weight)
        return static + quality / torch.clamp_min(degree, 1.0)

    def rank(self):
        reliability = self.reliability
        posterior = torch.log(torch.clamp_min(reliability, 1e-8)) if self.config.memory_use_reliability else torch.zeros_like(reliability)
        graph = self.graph_quality()
        return (
            self.config.memory_lambda_descriptor * self.descriptor_score
            + self.config.memory_lambda_reliability * posterior
            + self.config.memory_lambda_graph * graph
        )

    def support_signature(self, support):
        if support.numel() == 0:
            return torch.zeros(5, dtype=self.dtype, device=self.device)
        points = self.src_points[self.src_indices[support]]
        source_ids = torch.unique(self.src_indices[support])
        descriptor_probability = torch.softmax(self.topk_descriptor_scores[source_ids] / self.config.memory_signature_entropy_temperature, dim=1)
        descriptor_entropy = -torch.sum(descriptor_probability * torch.log(torch.clamp_min(descriptor_probability, 1e-8)), dim=1) / math.log(max(2, self.topk))
        ambiguity = descriptor_entropy.mean()
        centered = points - points.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / max(1, points.shape[0])
        covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        covariance = covariance + torch.eye(3, dtype=self.dtype, device=self.device) * torch.finfo(self.dtype).eps
        try:
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        except RuntimeError:
            eigenvalues = torch.zeros(3, dtype=self.dtype, device=self.device)
        linear_ratio = eigenvalues[1] / torch.clamp_min(eigenvalues[2], 1e-8)
        planar_ratio = eigenvalues[0] / torch.clamp_min(eigenvalues[1], 1e-8)
        voxels = torch.floor(points / self.config.memory_coverage_voxel_size).long()
        coverage = torch.as_tensor(torch.unique(voxels, dim=0).shape[0] / self.source_voxel_count, dtype=self.dtype, device=self.device)
        source_groups = torch.floor(points / self.config.memory_cross_group_voxel_size).long()
        source_distance = torch.cdist(points, points)
        target_points = self.tgt_points[self.tgt_indices[support]]
        target_distance = torch.cdist(target_points, target_points)
        compatibility = torch.exp(-((source_distance - target_distance) ** 2) / (2.0 * self.config.memory_sigma_g ** 2))
        upper = torch.triu(torch.ones_like(compatibility, dtype=torch.bool), diagonal=1)
        cross_group = upper & torch.any(source_groups[:, None] != source_groups[None], dim=-1)
        cross_group_agreement = compatibility[cross_group].mean() if bool(cross_group.any()) else torch.zeros((), dtype=self.dtype, device=self.device)
        return torch.stack([ambiguity, linear_ratio, planar_ratio, coverage, cross_group_agreement])

    def degeneracy_penalty(self, support):
        signature = self.support_signature(support)
        return (
            torch.relu(torch.as_tensor(self.config.memory_tau_linear, dtype=self.dtype, device=self.device) - signature[1])
            + torch.relu(torch.as_tensor(self.config.memory_tau_planar, dtype=self.dtype, device=self.device) - signature[2])
            + torch.relu(torch.as_tensor(self.config.memory_min_coverage, dtype=self.dtype, device=self.device) - signature[3])
            + torch.relu(torch.as_tensor(self.config.memory_min_cross_group_agreement, dtype=self.dtype, device=self.device) - signature[4])
        )

    def is_degenerate(self, support):
        diagnostics = self.support_diagnostics(support)
        return bool(self.support_hard_rejection_reasons(diagnostics))

    def support_diagnostics(self, support):
        support = torch.as_tensor(support, dtype=torch.long, device=self.device).reshape(-1)
        signature = self.support_signature(support)
        source_ids = self.src_indices[support] if support.numel() else torch.empty(0, dtype=torch.long, device=self.device)
        target_ids = self.tgt_indices[support] if support.numel() else torch.empty(0, dtype=torch.long, device=self.device)
        duplicate_correspondence = int(
            torch.unique(source_ids).numel() != source_ids.numel()
            or torch.unique(target_ids).numel() != target_ids.numel()
        )
        lambda2_over_lambda1 = 0.0
        lambda3_over_lambda2 = 0.0
        lambda3_over_lambda1 = 0.0
        eigen_failed = 0
        if support.numel():
            points = self.src_points[source_ids]
            centered = points - points.mean(dim=0, keepdim=True)
            covariance = centered.transpose(0, 1) @ centered / max(1, points.shape[0])
            covariance = torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
            covariance = 0.5 * (covariance + covariance.transpose(0, 1))
            covariance = covariance + torch.eye(3, dtype=self.dtype, device=self.device) * torch.finfo(self.dtype).eps
            try:
                eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
                lambda2_over_lambda1 = float((eigenvalues[1] / torch.clamp_min(eigenvalues[2], 1e-8)).item())
                lambda3_over_lambda2 = float((eigenvalues[0] / torch.clamp_min(eigenvalues[1], 1e-8)).item())
                lambda3_over_lambda1 = float((eigenvalues[0] / torch.clamp_min(eigenvalues[2], 1e-8)).item())
            except RuntimeError:
                eigen_failed = 1
        return {
            "support_size": int(support.numel()),
            "duplicate_correspondence": duplicate_correspondence,
            "lambda2_over_lambda1": lambda2_over_lambda1,
            "lambda3_over_lambda2": lambda3_over_lambda2,
            "lambda3_over_lambda1": lambda3_over_lambda1,
            "coverage": float(signature[3].item()),
            "cross_group_score": float(signature[4].item()),
            "eigen_failed": eigen_failed,
            "linear_degenerate": int(lambda2_over_lambda1 < self.config.memory_tau_linear),
            "planar_degenerate": int(lambda3_over_lambda2 < self.config.memory_tau_planar),
            "low_coverage": int(signature[3] < self.config.memory_min_coverage),
            "cross_group_inconsistent": int(signature[4] < self.config.memory_min_cross_group_agreement),
        }

    @staticmethod
    def support_constraint_reasons(diagnostics):
        reasons = []
        if diagnostics["support_size"] < 3:
            reasons.append("support_too_small")
        if diagnostics["duplicate_correspondence"]:
            reasons.append("duplicate_correspondence")
        if diagnostics["linear_degenerate"]:
            reasons.append("linear_degenerate")
        if diagnostics["planar_degenerate"]:
            reasons.append("planar_degenerate")
        if diagnostics["low_coverage"]:
            reasons.append("low_coverage")
        if diagnostics["cross_group_inconsistent"]:
            reasons.append("cross_group_inconsistent")
        return reasons

    def support_hard_rejection_reasons(self, diagnostics):
        reasons = []
        if diagnostics["support_size"] < self.config.memory_support_min:
            reasons.append("support_too_small")
        if diagnostics["duplicate_correspondence"]:
            reasons.append("duplicate_correspondence")
        if diagnostics["linear_degenerate"]:
            reasons.append("linear_degenerate")
        return reasons

    def structure_penalty(self, diagnostics):
        return (
            self.config.memory_lambda_structure_planar * float(diagnostics["planar_degenerate"])
            + self.config.memory_lambda_structure_coverage * max(0.0, self.config.memory_min_coverage - float(diagnostics["coverage"]))
            + self.config.memory_lambda_structure_cross_group * max(0.0, self.config.memory_min_cross_group_agreement - float(diagnostics["cross_group_score"]))
        )

    def presearch_basin_penalty(self, signature, support):
        if not self.config.memory_use_basin or self._negative_basin_signatures.numel() == 0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        normalized_signature = signature / torch.clamp_min(torch.linalg.norm(signature), 1e-8)
        normalized_history = self._negative_basin_signatures / torch.clamp_min(torch.linalg.norm(self._negative_basin_signatures, dim=1, keepdim=True), 1e-8)
        similarity = normalized_history @ normalized_signature
        repeated = torch.where(similarity >= self.config.memory_basin_signature_similarity)[0]
        penalties = []
        for index in repeated.tolist():
            historical_support = self._negative_basin_supports[index]
            overlap = torch.isin(support, historical_support).sum() / max(1, min(int(support.numel()), int(historical_support.numel())))
            if overlap >= self.config.memory_basin_presearch_min_support_overlap:
                penalties.append(similarity[index] * self._negative_basin_weights[index])
        return torch.stack(penalties).max() if penalties else torch.zeros((), dtype=self.dtype, device=self.device)

    def support_objective(self, support, candidate_score=None):
        candidate_score = self.rank() if candidate_score is None else candidate_score
        node_term = candidate_score[support].sum()
        edge_ids = [torch.arange(int(self.row_ptr[node]), int(self.row_ptr[node + 1]), device=self.device) for node in support.tolist()]
        if edge_ids:
            edge_ids = torch.cat(edge_ids)
            induced = torch.isin(self.edge_cols[edge_ids], support)
            edge_term = self.edge_weight()[edge_ids[induced]].mean() if bool(induced.any()) else torch.zeros((), dtype=self.dtype, device=self.device)
        else:
            edge_term = torch.zeros((), dtype=self.dtype, device=self.device)
        signature = self.support_signature(support)
        return (
            node_term
            + self.config.memory_lambda_edge * edge_term
            - self.config.memory_lambda_degeneracy * self.degeneracy_penalty(support)
            - self.config.memory_basin_presearch_penalty * self.presearch_basin_penalty(signature, support)
        )

    def expand_support(self, seed):
        support = [int(seed)]
        used_source = {int(self.src_indices[seed])}
        used_target = {int(self.tgt_indices[seed])}
        weights = self.edge_weight()
        candidate_score = self.rank()
        current_objective = self.support_objective(torch.tensor(support, dtype=torch.long, device=self.device), candidate_score)
        while len(support) < self.config.memory_support_max:
            frontier = {}
            for node in support:
                begin, end = int(self.row_ptr[node]), int(self.row_ptr[node + 1])
                for edge_id in range(begin, end):
                    neighbor = int(self.edge_cols[edge_id])
                    source_id, target_id = int(self.src_indices[neighbor]), int(self.tgt_indices[neighbor])
                    if source_id in used_source or target_id in used_target:
                        continue
                    frontier[neighbor] = max(frontier.get(neighbor, float("-inf")), float(weights[edge_id].item()))
            if not frontier:
                break
            candidates = torch.tensor(list(frontier), dtype=torch.long, device=self.device)
            incidence = torch.tensor([frontier[int(candidate)] for candidate in candidates.tolist()], dtype=self.dtype, device=self.device)
            preliminary = candidate_score[candidates] + self.config.memory_lambda_edge * incidence
            trial_count = min(int(self.config.memory_support_trial_count), int(candidates.numel()))
            candidates = candidates[torch.topk(preliminary, k=trial_count).indices]
            best_candidate, best_objective = None, current_objective
            for candidate in candidates.tolist():
                trial = torch.tensor([*support, candidate], dtype=torch.long, device=self.device)
                objective = self.support_objective(trial, candidate_score)
                if objective > best_objective:
                    best_candidate, best_objective = candidate, objective
            if best_candidate is None:
                break
            support.append(best_candidate)
            used_source.add(int(self.src_indices[best_candidate]))
            used_target.add(int(self.tgt_indices[best_candidate]))
            current_objective = best_objective
        return torch.tensor(support, dtype=torch.long, device=self.device)

    def correspondence_points(self, candidate_ids):
        return self.src_points[self.src_indices[candidate_ids]][None], self.tgt_points[self.tgt_indices[candidate_ids]][None]

    def map_cached_indices(self, src_corr_indices, tgt_corr_indices):
        source = torch.as_tensor(src_corr_indices, dtype=torch.long, device=self.device).reshape(-1)
        target = torch.as_tensor(tgt_corr_indices, dtype=torch.long, device=self.device).reshape(-1)
        if source.numel() != target.numel():
            raise ValueError("Cached R1 source and target correspondence index counts must match.")
        if source.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device), {
                "r1_corr_count": 0,
                "r1_mapped_corr_count": 0,
                "r1_mapped_count": 0,
                "r1_mapping_ratio": 0.0,
            }
        if int(source.min()) < 0 or int(source.max()) >= self.src_points.shape[0] or int(target.min()) < 0 or int(target.max()) >= self.tgt_points.shape[0]:
            raise ValueError("Cached R1 correspondence index is out of range for the current sampled keypoints.")
        target_count = int(self.tgt_points.shape[0])
        candidate_keys = self.src_indices.to(torch.long) * target_count + self.tgt_indices.to(torch.long)
        cached_keys = source * target_count + target
        sorted_keys, order = torch.sort(candidate_keys)
        positions = torch.searchsorted(sorted_keys, cached_keys)
        in_bounds = positions < sorted_keys.numel()
        valid = torch.zeros_like(in_bounds)
        valid[in_bounds] = sorted_keys[positions[in_bounds]] == cached_keys[in_bounds]
        mapped = order[positions[valid]].unique() if bool(valid.any()) else torch.empty(0, dtype=torch.long, device=self.device)
        count = int(source.numel())
        mapped_corr_count = int(valid.sum().item())
        return mapped, {
            "r1_corr_count": count,
            "r1_mapped_corr_count": mapped_corr_count,
            "r1_mapped_count": int(mapped.numel()),
            "r1_mapping_ratio": float(mapped_corr_count / count),
        }

    def _rotation_vector(self, rotation):
        cosine = torch.clamp((torch.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
        theta = torch.acos(cosine)
        vee = torch.stack([
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ])
        return 0.5 * vee if theta < 1e-6 else theta * vee / torch.clamp_min(2.0 * torch.sin(theta), 1e-8)

    def basin_key(self, pose):
        matrix = pose[0] if pose.ndim == 3 else pose
        rotation_step = math.radians(self.config.memory_basin_rotation_deg)
        rotation = torch.floor(self._rotation_vector(matrix[:3, :3]) / rotation_step).long()
        translation = torch.floor(matrix[:3, 3] / self.config.memory_basin_translation).long()
        return tuple(torch.cat([rotation, translation]).detach().cpu().tolist())

    def basin_record(self, pose):
        key = self.basin_key(pose)
        record = self.basins.get(key)
        if record is not None:
            self.basins.move_to_end(key)
        return key, record

    def basin_bonus(self, pose, signature):
        if not self.config.memory_use_basin:
            return 0.0
        _, record = self.basin_record(pose)
        if record is None:
            return 0.0
        signature = signature.detach().float().cpu()
        similarity = torch.nn.functional.cosine_similarity(signature[None], record.signature_mean[None]).item()
        if similarity < self.config.memory_basin_signature_similarity:
            return 0.0
        return (
            -self.config.memory_lambda_basin_nonimproving * math.log1p(record.n_nonimproving)
            + self.config.memory_lambda_basin_positive * math.log1p(record.n_positive)
        )

    def _refresh_negative_basin_cache(self):
        records = [record for record in self.basins.values() if record.n_nonimproving > record.n_positive]
        if not records:
            self._negative_basin_signatures = torch.empty((0, 5), dtype=self.dtype, device=self.device)
            self._negative_basin_weights = torch.empty(0, dtype=self.dtype, device=self.device)
            self._negative_basin_supports = []
            return
        self._negative_basin_signatures = torch.stack([record.signature_mean for record in records]).to(device=self.device, dtype=self.dtype)
        self._negative_basin_weights = torch.tensor(
            [math.log1p(record.n_nonimproving) - math.log1p(record.n_positive) for record in records],
            dtype=self.dtype,
            device=self.device,
        )
        self._negative_basin_supports = [record.support_ids.to(device=self.device) for record in records]

    def update_pair_posterior(self, verified_inliers, verified_outliers):
        if not self.config.memory_use_reliability:
            return 0.0
        verified_inliers = torch.as_tensor(verified_inliers, dtype=torch.long, device=self.device).unique()
        verified_outliers = torch.as_tensor(verified_outliers, dtype=torch.long, device=self.device).unique()
        previous = self.posterior_reliability.clone()
        self.alpha.mul_(self.config.memory_forgetting)
        self.beta.mul_(self.config.memory_forgetting)
        if verified_inliers.numel():
            self.alpha[verified_inliers] += self.config.memory_eta_positive
        if verified_outliers.numel():
            self.beta[verified_outliers] += self.config.memory_eta_negative
        self.alpha.clamp_(max=self.config.memory_evidence_cap)
        self.beta.clamp_(max=self.config.memory_evidence_cap)
        return float(torch.mean(torch.abs(self.posterior_reliability - previous)).item())

    def update_relation_graph(self, verified_inliers, verified_outliers):
        if self.edge_rows.numel() == 0 or not self.config.memory_use_relation_history:
            return 0.0
        verified_inliers = torch.as_tensor(verified_inliers, dtype=torch.long, device=self.device).unique()
        verified_outliers = torch.as_tensor(verified_outliers, dtype=torch.long, device=self.device).unique()
        self.edge_success.mul_(self.config.memory_forgetting)
        self.edge_failure.mul_(self.config.memory_forgetting)
        active_nodes = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        active_nodes[verified_inliers] = True
        active_nodes[verified_outliers] = True
        inlier_nodes = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        outlier_nodes = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        inlier_nodes[verified_inliers] = True
        outlier_nodes[verified_outliers] = True
        active_edges = active_nodes[self.edge_rows] & active_nodes[self.edge_cols]
        success = active_edges & inlier_nodes[self.edge_rows] & inlier_nodes[self.edge_cols]
        failure = active_edges & (outlier_nodes[self.edge_rows] | outlier_nodes[self.edge_cols])
        self.edge_success[success] += self.config.memory_eta_edge_positive
        self.edge_failure[failure] += self.config.memory_eta_edge_negative
        self.edge_success.clamp_(max=self.config.memory_evidence_cap)
        self.edge_failure.clamp_(max=self.config.memory_evidence_cap)
        return float((success.sum() + failure.sum()).item() / max(1, self.edge_rows.numel()))

    def update_basin(self, pose, signature, score, improved, support_ids):
        if not self.config.memory_use_basin:
            return None, None
        key, record = self.basin_record(pose)
        signature = signature.detach().float().cpu()
        if record is None:
            record = BasinRecord(0.0, 0.0, float("-inf"), signature, support_ids.detach().cpu())
            self.basins[key] = record
        if improved:
            record.n_positive += 1.0
        else:
            record.n_nonimproving += 1.0
        record.best_score = max(record.best_score, float(score))
        record.signature_mean = self.config.memory_basin_signature_momentum * record.signature_mean + (1.0 - self.config.memory_basin_signature_momentum) * signature
        record.support_ids = support_ids.detach().cpu()
        self.basins.move_to_end(key)
        while len(self.basins) > self.config.memory_basin_max:
            self.basins.popitem(last=False)
        self._refresh_negative_basin_cache()
        return key, record

    def one_to_one_filter(self, candidate_ids, residuals=None):
        if candidate_ids.numel() == 0:
            return candidate_ids
        candidate_ids = torch.as_tensor(candidate_ids, dtype=torch.long, device=self.device).reshape(-1)
        if residuals is None:
            sorted_order = torch.argsort(self.descriptor_score[candidate_ids], descending=True)
        else:
            residuals = torch.as_tensor(residuals, dtype=self.dtype, device=self.device)
            values = residuals[candidate_ids] if residuals.numel() == self.count else residuals
            if values.numel() != candidate_ids.numel():
                raise ValueError("Validation residuals must align with candidate_ids or the full candidate pool.")
            sorted_order = torch.argsort(values, descending=False, stable=True)
        sorted_ids = candidate_ids[sorted_order]
        seen_source, seen_target = set(), set()
        kept = []
        for cid in sorted_ids.tolist():
            sid = int(self.src_indices[cid])
            tid = int(self.tgt_indices[cid])
            if sid in seen_source or tid in seen_target:
                continue
            seen_source.add(sid)
            seen_target.add(tid)
            kept.append(cid)
        return torch.tensor(kept, dtype=torch.long, device=self.device)

    def _mutual_nearest(self, src_points_transformed, tgt_points, radius):
        dist = torch.cdist(src_points_transformed[None], tgt_points[None])[0]
        src_to_tgt = dist.argmin(dim=1)
        tgt_to_src = dist.argmin(dim=0)
        mutual = tgt_to_src[src_to_tgt] == torch.arange(src_points_transformed.shape[0], device=self.device)
        within_radius = dist[torch.arange(src_points_transformed.shape[0], device=self.device), src_to_tgt] < radius
        nearest_distance = dist[torch.arange(src_points_transformed.shape[0], device=self.device), src_to_tgt]
        return src_to_tgt, mutual & within_radius, nearest_distance

    def pose_guided_candidate_expansion(self, pose, src_features, tgt_features):
        """Expand candidate pool beyond Top-K using pose-guided radius search.

        Only call after independent validation passes.
        Returns (new_src_ids, new_tgt_ids, new_scores).
        """
        src = torch.nn.functional.normalize(self._features(src_features), dim=1)
        tgt = torch.nn.functional.normalize(self._features(tgt_features), dim=1)
        transformed = self._transform(self.src_points, pose)
        existing_keys = set(zip(self.src_indices.tolist(), self.tgt_indices.tolist()))
        radius = self.config.memory_vdce_expand_radius
        nearest_tgt, mutual, nearest_distance = self._mutual_nearest(transformed, self.tgt_points, radius)
        candidates = []
        desc_threshold = self.config.memory_vdce_expand_descriptor_threshold
        max_per_source = self.config.memory_vdce_expand_max_per_source
        max_total = self.config.memory_vdce_expand_max_total
        descriptor_weight = self.config.memory_vdce_expand_descriptor_weight
        geometry_weight = self.config.memory_vdce_expand_geometry_weight
        total_weight = descriptor_weight + geometry_weight
        for src_id in torch.where(mutual)[0].tolist():
            tgt_id = int(nearest_tgt[src_id])
            key = (src_id, tgt_id)
            if key in existing_keys:
                continue
            descriptor_sim = float(src[src_id] @ tgt[tgt_id])
            if descriptor_sim < desc_threshold:
                continue
            spatial_score = float(torch.exp(-nearest_distance[src_id] ** 2 / (2.0 * radius ** 2)))
            joint_score = (descriptor_weight * descriptor_sim + geometry_weight * spatial_score) / total_weight
            candidates.append((src_id, tgt_id, joint_score))
        if not candidates:
            return torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=self.dtype, device=self.device)
        candidates.sort(key=lambda item: (-item[2], item[0], item[1]))
        selected, deferred, per_source = [], [], {}
        quota = self.config.memory_vdce_expand_voxel_quota
        voxel_counts = {}
        for candidate in candidates:
            src_id = candidate[0]
            if per_source.get(src_id, 0) >= max_per_source:
                continue
            voxel = tuple(torch.floor(self.src_points[src_id] / self.config.memory_coverage_voxel_size).long().tolist())
            if voxel_counts.get(voxel, 0) < quota:
                selected.append(candidate)
                per_source[src_id] = per_source.get(src_id, 0) + 1
                voxel_counts[voxel] = voxel_counts.get(voxel, 0) + 1
            else:
                deferred.append(candidate)
            if len(selected) >= max_total:
                break
        if len(selected) < max_total:
            for candidate in deferred:
                src_id = candidate[0]
                if per_source.get(src_id, 0) >= max_per_source:
                    continue
                selected.append(candidate)
                per_source[src_id] = per_source.get(src_id, 0) + 1
                if len(selected) >= max_total:
                    break
        new_src = torch.tensor([c[0] for c in selected], dtype=torch.long, device=self.device)
        new_tgt = torch.tensor([c[1] for c in selected], dtype=torch.long, device=self.device)
        new_scores = torch.tensor([c[2] for c in selected], dtype=self.dtype, device=self.device)
        return new_src, new_tgt, new_scores

    def add_candidates(self, new_src, new_tgt, new_scores):
        """Add new candidates to the pool and rebuild indices."""
        if new_src.numel() == 0:
            return 0
        existing_keys = set(zip(self.src_indices.tolist(), self.tgt_indices.tolist()))
        kept = []
        for src_id, tgt_id, score in zip(new_src.tolist(), new_tgt.tolist(), new_scores.tolist()):
            key = (int(src_id), int(tgt_id))
            if key not in existing_keys:
                existing_keys.add(key)
                kept.append((key[0], key[1], float(score)))
        if not kept:
            return 0
        new_src = torch.tensor([item[0] for item in kept], dtype=torch.long, device=self.device)
        new_tgt = torch.tensor([item[1] for item in kept], dtype=torch.long, device=self.device)
        new_scores = torch.tensor([item[2] for item in kept], dtype=self.dtype, device=self.device)
        self.src_indices = torch.cat([self.src_indices, new_src])
        self.tgt_indices = torch.cat([self.tgt_indices, new_tgt])
        self.descriptor_score = torch.cat([self.descriptor_score, new_scores])
        self.alpha = torch.cat([self.alpha, self.config.memory_alpha0 + self.config.memory_descriptor_prior * new_scores])
        self.beta = torch.cat([self.beta, torch.full_like(new_scores, float(self.config.memory_beta0))])
        self.count = int(self.src_indices.numel())
        self._src_to_candidate_ids = self._build_candidate_index(self.src_indices)
        self._tgt_to_candidate_ids = self._build_candidate_index(self.tgt_indices)
        self._pair_to_candidate_id = self._build_pair_index(self.src_indices, self.tgt_indices)
        self._rebuild_relation_graph_preserving_history()
        return int(new_src.numel())

    def _transform(self, points, pose):
        matrix = pose[0] if pose.ndim == 3 else pose
        return points @ matrix[:3, :3].transpose(0, 1) + matrix[:3, 3]

    def check_graph_symmetry(self):
        failures = 0
        for node in range(self.count):
            begin, end = int(self.row_ptr[node]), int(self.row_ptr[node + 1])
            for edge_id in range(begin, end):
                neighbor = int(self.edge_cols[edge_id])
                n_begin, n_end = int(self.row_ptr[neighbor]), int(self.row_ptr[neighbor + 1])
                if node not in self.edge_cols[n_begin:n_end].tolist():
                    failures += 1
        return failures

    def compress(self):
        if self.config.memory_use_relation_history and self.edge_success.numel():
            weak = torch.abs(self.edge_success - self.edge_failure) < self.config.memory_compress_epsilon
            self.edge_success[weak] = 0.0
            self.edge_failure[weak] = 0.0

    def summary(self):
        return {
            "candidate_count": self.count,
            "edge_count": int(self.edge_rows.numel()),
            "basin_count": len(self.basins),
            "mean_reliability": float(self.reliability.mean().item()) if self.count else 0.0,
        }
