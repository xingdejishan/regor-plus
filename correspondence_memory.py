from collections import OrderedDict
from dataclasses import dataclass
import math

import torch


@dataclass
class BasinRecord:
    n_positive: float
    n_negative: float
    best_score: float
    signature_mean: torch.Tensor


class CorrespondenceMemory:
    def __init__(self, src_points, tgt_points, src_features, tgt_features, config):
        self.config = config
        self.src_points = self._points(src_points)
        self.tgt_points = self._points(tgt_points)
        self.device = self.src_points.device
        self.dtype = self.src_points.dtype
        self.src_indices, self.tgt_indices, self.descriptor_score = self._build_candidates(src_features, tgt_features)
        self.count = int(self.src_indices.numel())
        self.alpha = config.memory_alpha0 + config.memory_descriptor_prior * self.descriptor_score
        self.beta = torch.full_like(self.alpha, float(config.memory_beta0))
        self.edge_rows, self.edge_cols, self.static_geometry, self.row_ptr = self._build_relation_graph()
        self.edge_success = torch.zeros_like(self.static_geometry)
        self.edge_failure = torch.zeros_like(self.static_geometry)
        self.basins = OrderedDict()

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
        return source.reshape(-1), indices.reshape(-1), normalized

    def _build_relation_graph(self):
        source_count = int(self.src_points.shape[0])
        topk = min(int(self.config.memory_topk), int(self.tgt_points.shape[0]))
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
    def reliability(self):
        return self.alpha / torch.clamp_min(self.alpha + self.beta, 1e-8)

    def edge_weight(self):
        return self.static_geometry + self.config.memory_lambda_edge_success * torch.log1p(self.edge_success) - self.config.memory_lambda_edge_failure * torch.log1p(self.edge_failure)

    def graph_quality(self):
        quality = torch.zeros(self.count, dtype=self.dtype, device=self.device)
        if self.edge_rows.numel() == 0 or not self.config.memory_use_relation_history:
            return quality
        weights = torch.relu(self.edge_weight())
        quality.index_add_(0, self.edge_rows, weights)
        degree = torch.zeros_like(quality)
        degree.index_add_(0, self.edge_rows, torch.ones_like(weights))
        return quality / torch.clamp_min(degree, 1.0)

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
            return torch.zeros(3, dtype=self.dtype, device=self.device)
        points = self.src_points[self.src_indices[support]]
        descriptor_probability = torch.softmax(self.descriptor_score[support], dim=0)
        ambiguity = -torch.sum(descriptor_probability * torch.log(torch.clamp_min(descriptor_probability, 1e-8))) / math.log(max(2, int(support.numel())))
        centered = points - points.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / max(1, points.shape[0])
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
        eigen_probability = eigenvalues / torch.clamp_min(eigenvalues.sum(), 1e-8)
        eigen_entropy = -torch.sum(eigen_probability * torch.log(torch.clamp_min(eigen_probability, 1e-8))) / math.log(3.0)
        voxels = torch.floor(points / self.config.memory_coverage_voxel_size).long()
        coverage = torch.unique(voxels, dim=0).shape[0] / max(1, points.shape[0])
        return torch.stack([ambiguity, eigen_entropy, torch.as_tensor(coverage, dtype=self.dtype, device=self.device)])

    def degeneracy_penalty(self, support):
        signature = self.support_signature(support)
        return torch.relu(torch.as_tensor(self.config.memory_min_eigen_entropy, dtype=self.dtype, device=self.device) - signature[1]) + torch.relu(torch.as_tensor(self.config.memory_min_coverage, dtype=self.dtype, device=self.device) - signature[2])

    def is_degenerate(self, support):
        if support.numel() < self.config.memory_support_min:
            return True
        signature = self.support_signature(support)
        return bool(signature[1] < self.config.memory_min_eigen_entropy or signature[2] < self.config.memory_min_coverage)

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
        return node_term + self.config.memory_lambda_edge * edge_term - self.config.memory_lambda_degeneracy * self.degeneracy_penalty(support)

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
            -self.config.memory_lambda_basin_negative * math.log1p(record.n_negative)
            + self.config.memory_lambda_basin_positive * math.log1p(record.n_positive)
        )

    def update_pair_posterior(self, support, inliers):
        previous = self.reliability.clone()
        self.alpha.mul_(self.config.memory_forgetting)
        self.beta.mul_(self.config.memory_forgetting)
        self.alpha[inliers] += self.config.memory_eta_positive
        support_failures = support[~inliers[support]]
        if support_failures.numel():
            self.beta[support_failures] += self.config.memory_eta_negative
        self.alpha.clamp_(max=self.config.memory_evidence_cap)
        self.beta.clamp_(max=self.config.memory_evidence_cap)
        return float(torch.mean(torch.abs(self.reliability - previous)).item())

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

    def update_basin(self, pose, signature, score, improved):
        key, record = self.basin_record(pose)
        signature = signature.detach().float().cpu()
        if record is None:
            record = BasinRecord(0.0, 0.0, float("-inf"), signature)
            self.basins[key] = record
        if improved:
            record.n_positive += 1.0
        else:
            record.n_negative += 1.0
        record.best_score = max(record.best_score, float(score))
        record.signature_mean = self.config.memory_basin_signature_momentum * record.signature_mean + (1.0 - self.config.memory_basin_signature_momentum) * signature
        self.basins.move_to_end(key)
        while len(self.basins) > self.config.memory_basin_max:
            self.basins.popitem(last=False)
        return key, record

    def compress(self):
        if self.edge_success.numel():
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
