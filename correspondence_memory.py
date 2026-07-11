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
    def __init__(self, src_points, tgt_points, src_features, tgt_features, config):
        self.config = config
        self.src_points = self._points(src_points)
        self.tgt_points = self._points(tgt_points)
        self.device = self.src_points.device
        self.dtype = self.src_points.dtype
        self.src_indices, self.tgt_indices, self.descriptor_score, self.topk_descriptor_scores = self._build_candidates(src_features, tgt_features)
        self.count = int(self.src_indices.numel())
        self.topk = int(self.topk_descriptor_scores.shape[1])
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

    def _build_candidates(self, src_features, tgt_features):
        src = torch.nn.functional.normalize(self._features(src_features), dim=1)
        tgt = torch.nn.functional.normalize(self._features(tgt_features), dim=1)
        scores = src @ tgt.transpose(0, 1)
        topk = min(int(self.config.memory_topk), int(tgt.shape[0]))
        values, indices = torch.topk(scores, k=topk, dim=1)
        source = torch.arange(src.shape[0], device=src.device, dtype=torch.long)[:, None].expand_as(indices)
        flat = values.reshape(-1)
        minimum, maximum = flat.min(), flat.max()
        normalized = (flat - minimum) / torch.clamp_min(maximum - minimum, 1e-8)
        return source.reshape(-1), indices.reshape(-1), normalized, values

    def _build_relation_graph(self):
        source_count = int(self.src_points.shape[0])
        topk = self.topk
        if source_count < 2 or self.count == 0:
            empty_long = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_long, empty_long, torch.empty(0, dtype=self.dtype, device=self.device), torch.zeros(self.count + 1, dtype=torch.long, device=self.device)
        source_neighbors = min(source_count - 1, max(1, math.ceil(self.config.memory_graph_neighbors / topk)))
        source_distance = torch.cdist(self.src_points, self.src_points)
        source_distance.fill_diagonal_(float("inf"))
        _, neighbors = torch.topk(source_distance, k=source_neighbors, dim=1, largest=False)
        ranks = torch.arange(topk, device=self.device, dtype=torch.long)
        neighbor_nodes = (neighbors[:, :, None] * topk + ranks[None, None, :]).reshape(source_count, -1)
        candidate_sources = self.src_indices
        local_nodes = neighbor_nodes[candidate_sources]
        local_targets = self.tgt_indices[local_nodes]
        local_sources = self.src_indices[local_nodes]
        source_gap = torch.linalg.norm(self.src_points[candidate_sources, None] - self.src_points[local_sources], dim=-1)
        target_gap = torch.linalg.norm(self.tgt_points[self.tgt_indices, None] - self.tgt_points[local_targets], dim=-1)
        geometry = torch.exp(-((source_gap - target_gap) ** 2) / (2.0 * self.config.memory_sigma_g ** 2))
        valid = (self.tgt_indices[:, None] != local_targets) & (geometry > self.config.memory_tau_g)
        geometry = torch.where(valid, geometry, torch.full_like(geometry, float("-inf")))
        take = min(int(self.config.memory_graph_neighbors), int(geometry.shape[1]))
        values, positions = torch.topk(geometry, k=take, dim=1)
        valid = torch.isfinite(values)
        rows = torch.arange(self.count, device=self.device, dtype=torch.long)[:, None].expand_as(positions)
        edge_rows = rows[valid]
        edge_cols = torch.gather(local_nodes, 1, positions)[valid]
        static = values[valid]
        counts = torch.bincount(edge_rows, minlength=self.count)
        row_ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), counts.cumsum(0)])
        return edge_rows, edge_cols, static, row_ptr

    @property
    def posterior_reliability(self):
        return self.alpha / torch.clamp_min(self.alpha + self.beta, 1e-8)

    @property
    def reliability(self):
        return self.posterior_reliability if self.config.memory_use_reliability else torch.ones(self.count, dtype=self.dtype, device=self.device)

    def estimation_weights(self, candidate_ids):
        return self.reliability[candidate_ids]

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
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        lambda12 = eigenvalues[0] / torch.clamp_min(eigenvalues[1], 1e-8)
        lambda13 = eigenvalues[0] / torch.clamp_min(eigenvalues[2], 1e-8)
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
        return torch.stack([ambiguity, lambda12, lambda13, coverage, cross_group_agreement])

    def degeneracy_penalty(self, support):
        signature = self.support_signature(support)
        return (
            torch.relu(torch.as_tensor(self.config.memory_min_lambda12_ratio, dtype=self.dtype, device=self.device) - signature[1])
            + torch.relu(torch.as_tensor(self.config.memory_min_lambda13_ratio, dtype=self.dtype, device=self.device) - signature[2])
            + torch.relu(torch.as_tensor(self.config.memory_min_coverage, dtype=self.dtype, device=self.device) - signature[3])
            + torch.relu(torch.as_tensor(self.config.memory_min_cross_group_agreement, dtype=self.dtype, device=self.device) - signature[4])
        )

    def is_degenerate(self, support):
        if support.numel() < self.config.memory_support_min:
            return True
        signature = self.support_signature(support)
        return bool(
            signature[1] < self.config.memory_min_lambda12_ratio
            or signature[2] < self.config.memory_min_lambda13_ratio
            or signature[3] < self.config.memory_min_coverage
            or signature[4] < self.config.memory_min_cross_group_agreement
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

    def map_cached_correspondences(self, src_corr, tgt_corr, radius):
        source = self._points(src_corr)
        target = self._points(tgt_corr)
        if source.numel() == 0 or target.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device), {"r1_corr_count": 0, "r1_mapped_count": 0, "r1_mapping_ratio": 0.0}
        source_distance, source_ids = torch.cdist(source, self.src_points).min(dim=1)
        target_distance, target_ids = torch.cdist(target, self.tgt_points).min(dim=1)
        valid = (source_distance <= radius) & (target_distance <= radius)
        candidate_ids = []
        for source_id, target_id in zip(source_ids[valid].tolist(), target_ids[valid].tolist()):
            candidates = torch.where((self.src_indices == source_id) & (self.tgt_indices == target_id))[0]
            if candidates.numel():
                candidate_ids.append(int(candidates[0]))
        mapped = torch.tensor(candidate_ids, dtype=torch.long, device=self.device).unique() if candidate_ids else torch.empty(0, dtype=torch.long, device=self.device)
        count = int(source.shape[0])
        return mapped, {"r1_corr_count": count, "r1_mapped_count": int(mapped.numel()), "r1_mapping_ratio": float(mapped.numel() / max(1, count))}

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

    def update_pair_posterior(self, support, inliers):
        if not self.config.memory_use_reliability:
            return 0.0
        previous = self.posterior_reliability.clone()
        self.alpha.mul_(self.config.memory_forgetting)
        self.beta.mul_(self.config.memory_forgetting)
        self.alpha[inliers] += self.config.memory_eta_positive
        support_failures = support[~inliers[support]]
        if support_failures.numel():
            self.beta[support_failures] += self.config.memory_eta_negative
        self.alpha.clamp_(max=self.config.memory_evidence_cap)
        self.beta.clamp_(max=self.config.memory_evidence_cap)
        return float(torch.mean(torch.abs(self.posterior_reliability - previous)).item())

    def update_relation_graph(self, support, inliers, residuals):
        if self.edge_rows.numel() == 0 or not self.config.memory_use_relation_history:
            return 0.0
        self.edge_success.mul_(self.config.memory_forgetting)
        self.edge_failure.mul_(self.config.memory_forgetting)
        active_nodes = torch.zeros(self.count, dtype=torch.bool, device=self.device)
        active_nodes[support] = True
        active_edges = active_nodes[self.edge_rows] & active_nodes[self.edge_cols]
        success = active_edges & inliers[self.edge_rows] & inliers[self.edge_cols]
        failure = active_edges & ((residuals[self.edge_rows] >= self.config.memory_inlier_threshold) | (residuals[self.edge_cols] >= self.config.memory_inlier_threshold))
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
