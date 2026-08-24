from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbeRegorConfig:
    line_count: int = 512
    line_candidates: int = 4096
    nu0: float = 0.5
    action_count: int = 100
    local_covariance_k: int = 16
    round_knn: tuple[int, ...] = (100, 50, 20)
    inlier_scale: float = 0.1
    covariance_floor_ratio: float = 0.05
    line_chunk: int = 64
    point_chunk: int = 1024


@dataclass
class LineProbeObservation:
    source: torch.Tensor
    source_warped: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor
    probabilities: torch.Tensor
    evidence: torch.Tensor
    line_count: int
    matched_count: int
    median_gap: float


@dataclass
class ProbeRegorResult:
    pose: torch.Tensor
    source_correspondences: torch.Tensor
    target_correspondences: torch.Tensor
    rounds: int
    line_counts: list[int]
    matched_counts: list[int]
    action_counts: list[int]
    correspondence_counts: list[int]
    r_plus: list[float]
    r_minus: list[float]
    posterior_ess: list[float]


def transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return points @ pose[:3, :3].T + pose[:3, 3]


def skew(points: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(points[:, 0])
    x, y, z = points.unbind(-1)
    return torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(-1, 3, 3)


def weighted_rigid_transform(
    source: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    if len(source) < 3:
        return torch.eye(4, dtype=source.dtype, device=source.device)
    finite = (
        torch.isfinite(source).all(-1)
        & torch.isfinite(target).all(-1)
        & torch.isfinite(weights)
        & (weights > 0)
    )
    source = source[finite]
    target = target[finite]
    weights = weights[finite]
    if len(source) < 3 or float(weights.sum()) <= 0:
        return torch.eye(4, dtype=source.dtype, device=source.device)
    weights = weights / weights.sum()
    source_center = (weights[:, None] * source).sum(0)
    target_center = (weights[:, None] * target).sum(0)
    source_centered = source - source_center
    target_centered = target - target_center
    covariance = (weights[:, None] * source_centered).T @ target_centered
    u, _, vh = torch.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if torch.det(rotation) < 0:
        vh = vh.clone()
        vh[-1] *= -1
        rotation = vh.T @ u.T
    pose = torch.eye(4, dtype=source.dtype, device=source.device)
    pose[:3, :3] = rotation
    pose[:3, 3] = target_center - rotation @ source_center
    return pose


def effective_sample_size(weights: torch.Tensor) -> float:
    if len(weights) == 0 or float(weights.sum()) <= 0:
        return 0.0
    normalized = weights / weights.sum()
    return float(1.0 / torch.sum(normalized.square()).item())


class LineProbeBank:
    def __init__(self, config: ProbeRegorConfig) -> None:
        self.config = config
        self._triplet_cache: dict[
            str, tuple[tuple[int, int, torch.dtype, torch.device], torch.Tensor, torch.Tensor]
        ] = {}

    def _triplets(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = []
        distances = []
        for start in range(0, len(points), self.config.point_chunk):
            block = points[start : start + self.config.point_chunk]
            distance = torch.cdist(block, points)
            values, local_indices = torch.topk(distance, 3, largest=False, dim=1)
            indices.append(local_indices)
            distances.append(values[:, 1:])
        triplets = torch.cat(indices, dim=0)
        mean_2nn = torch.cat(distances, dim=0).mean()
        delta = mean_2nn * (3.0**0.5) / 2.0
        return triplets, delta.clamp_min(torch.finfo(points.dtype).eps)

    def _cached_triplets(
        self, name: str, points: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        signature = (points.data_ptr(), len(points), points.dtype, points.device)
        cached = self._triplet_cache.get(name)
        if cached is None or cached[0] != signature:
            triplets, delta = self._triplets(points)
            self._triplet_cache[name] = (signature, triplets, delta)
            return triplets, delta
        return cached[1], cached[2]

    def _sphere_points(self, values: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        u = values[:, 0] * 2 - 1
        alpha = values[:, 1] * (2 * torch.pi)
        radial = radius * torch.sqrt(torch.clamp(1 - u.square(), min=0))
        return torch.stack([radial * torch.cos(alpha), radial * torch.sin(alpha), radius * u], dim=-1)

    def _box_hit(
        self, origins: torch.Tensor, directions: torch.Tensor, points: torch.Tensor
    ) -> torch.Tensor:
        lower = points.min(0).values
        upper = points.max(0).values
        safe = torch.where(directions.abs() < 1e-8, torch.full_like(directions, 1e-8), directions)
        first = (lower - origins) / safe
        second = (upper - origins) / safe
        near = torch.minimum(first, second).amax(-1)
        far = torch.maximum(first, second).amin(-1)
        return far >= near

    def _lines(
        self, source_warped: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([source_warped, target], dim=0)
        center = (combined.min(0).values + combined.max(0).values) / 2
        radius = torch.linalg.norm(combined - center, dim=-1).max().clamp_min(1e-3) * 1.01
        engine = torch.quasirandom.SobolEngine(4, scramble=False)
        values = engine.draw(self.config.line_candidates).to(combined.device, combined.dtype)
        first = self._sphere_points(values[:, :2], radius) + center
        second = self._sphere_points(values[:, 2:], radius) + center
        directions = torch.nn.functional.normalize(second - first, dim=-1)
        valid = self._box_hit(first, directions, source_warped) & self._box_hit(
            first, directions, target
        )
        selected = torch.nonzero(valid, as_tuple=False).flatten()[: self.config.line_count]
        return first[selected], directions[selected]

    def _cloud_intersections(
        self,
        geometry: torch.Tensor,
        lineage: torch.Tensor,
        triplets: torch.Tensor,
        delta: torch.Tensor,
        origins: torch.Tensor,
        directions: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        empty = geometry.new_empty((0, 3))
        empty_depth = geometry.new_empty((0,))
        output = [(empty, empty, empty_depth) for _ in range(len(origins))]
        for line_start in range(0, len(origins), self.config.line_chunk):
            line_stop = min(len(origins), line_start + self.config.line_chunk)
            block_origins = origins[line_start:line_stop]
            block_directions = directions[line_start:line_stop]
            difference = geometry[None] - block_origins[:, None]
            depth = torch.einsum("lnc,lc->ln", difference, block_directions)
            radial = torch.linalg.norm(
                difference - depth[:, :, None] * block_directions[:, None], dim=-1
            )
            valid = (radial[:, triplets] < delta).all(-1)
            for local_line in range(line_stop - line_start):
                centers = torch.nonzero(valid[local_line], as_tuple=False).flatten()
                if len(centers) == 0:
                    continue
                origin = block_origins[local_line]
                direction = block_directions[local_line]
                local_indices = triplets[centers]
                local_radial = radial[local_line, local_indices].clamp_min(
                    torch.finfo(geometry.dtype).eps
                )
                local_weights = local_radial / local_radial.sum(-1, keepdim=True)
                soft_geometry = (
                    geometry[local_indices] * local_weights[:, :, None]
                ).sum(1)
                soft_lineage = (
                    lineage[local_indices] * local_weights[:, :, None]
                ).sum(1)
                soft_depth = (soft_geometry - origin) @ direction
                order = torch.argsort(soft_depth)
                soft_geometry = soft_geometry[order]
                soft_lineage = soft_lineage[order]
                soft_depth = soft_depth[order]
                groups = [0]
                if len(soft_depth) > 1:
                    boundaries = torch.nonzero(
                        soft_depth[1:] - soft_depth[:-1] > delta, as_tuple=False
                    ).flatten()
                    groups.extend((boundaries + 1).tolist())
                groups.append(len(soft_depth))
                clustered_geometry = []
                clustered_lineage = []
                clustered_depth = []
                for start, stop in zip(groups[:-1], groups[1:]):
                    clustered_geometry.append(soft_geometry[start:stop].mean(0))
                    clustered_lineage.append(soft_lineage[start:stop].mean(0))
                    clustered_depth.append(soft_depth[start:stop].mean())
                output[line_start + local_line] = (
                    torch.stack(clustered_geometry),
                    torch.stack(clustered_lineage),
                    torch.stack(clustered_depth),
                )
        return output

    def observe(
        self, source: torch.Tensor, target: torch.Tensor, pose: torch.Tensor
    ) -> LineProbeObservation:
        source_warped = transform_points(source, pose)
        source_triplets, source_delta = self._cached_triplets("source", source)
        target_triplets, target_delta = self._cached_triplets("target", target)
        origins, directions = self._lines(source_warped, target)
        if len(origins) == 0:
            empty = source.new_empty((0, 3))
            empty_scalar = source.new_empty((0,))
            return LineProbeObservation(
                empty, empty, empty, empty_scalar, empty_scalar, empty_scalar, 0, 0, 0.0
            )
        source_intersections = self._cloud_intersections(
            source_warped,
            source,
            source_triplets,
            source_delta,
            origins,
            directions,
        )
        target_intersections = self._cloud_intersections(
            target,
            target,
            target_triplets,
            target_delta,
            origins,
            directions,
        )
        records = []
        for line_index, (source_line, target_line) in enumerate(
            zip(source_intersections, target_intersections)
        ):
            source_geometry, source_lineage, source_depth = source_line
            target_geometry, target_lineage, target_depth = target_line
            if len(source_depth) == 0 or len(target_depth) == 0:
                continue
            distance = (source_depth[:, None] - target_depth[None, :]).abs()
            forward = torch.argmin(distance, dim=1)
            reverse = torch.argmin(distance, dim=0)
            pairs = {(index, int(value)) for index, value in enumerate(forward.tolist())}
            pairs.update((int(value), index) for index, value in enumerate(reverse.tolist()))
            line_reliability = torch.exp(
                source.new_tensor(-0.5 * abs(len(source_depth) - len(target_depth)))
            )
            divisor = max(len(pairs), 1)
            for source_index, target_index in sorted(pairs):
                records.append(
                    (
                        source_lineage[source_index],
                        source_geometry[source_index],
                        target_lineage[target_index],
                        distance[source_index, target_index],
                        line_reliability / divisor,
                    )
                )
        if not records:
            empty = source.new_empty((0, 3))
            empty_scalar = source.new_empty((0,))
            return LineProbeObservation(
                empty,
                empty,
                empty,
                empty_scalar,
                empty_scalar,
                empty_scalar,
                len(origins),
                0,
                0.0,
            )
        gaps = torch.stack([record[3] for record in records])
        median_gap = gaps.median().clamp_min((source_delta + target_delta) / 2)
        nu = (self.config.nu0 * median_gap).clamp_min(torch.finfo(source.dtype).eps)
        probabilities = torch.stack(
            [torch.exp(-record[3].square() / (2 * nu.square())) for record in records]
        )
        evidence = torch.stack([record[4] for record in records])
        weights = evidence * probabilities
        return LineProbeObservation(
            torch.stack([record[0] for record in records]),
            torch.stack([record[1] for record in records]),
            torch.stack([record[2] for record in records]),
            weights,
            probabilities,
            evidence,
            len(origins),
            len(records),
            float(median_gap.item()),
        )


class ProbeRegor:
    def __init__(self, baseline_regenerator, baseline_estimator, config: ProbeRegorConfig | None = None) -> None:
        self.config = config or ProbeRegorConfig()
        self.regenerator = baseline_regenerator
        self.estimator = baseline_estimator
        self.probes = LineProbeBank(self.config)

    def _posterior_update(
        self,
        source_corr: torch.Tensor,
        target_corr: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        observation: LineProbeObservation,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(source_corr) == 0 or len(observation.source) == 0:
            return alpha, beta
        source_scale = torch.pdist(source_corr).median() if len(source_corr) > 1 else source_corr.new_tensor(1.0)
        target_scale = torch.pdist(target_corr).median() if len(target_corr) > 1 else target_corr.new_tensor(1.0)
        source_scale = source_scale.clamp_min(self.config.inlier_scale)
        target_scale = target_scale.clamp_min(self.config.inlier_scale)
        distance = torch.cdist(observation.source, source_corr) / source_scale
        distance += torch.cdist(observation.target, target_corr) / target_scale
        owners = distance.argmin(1)
        alpha = alpha.clone()
        beta = beta.clone()
        alpha.scatter_add_(0, owners, observation.evidence * observation.probabilities)
        beta.scatter_add_(0, owners, observation.evidence * (1 - observation.probabilities))
        return alpha, beta

    def _actions(
        self,
        source_corr: torch.Tensor,
        pose: torch.Tensor,
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        transformed = transform_points(source_corr, pose)
        jacobian = torch.cat([-skew(transformed), torch.eye(3, device=source_corr.device, dtype=source_corr.dtype).expand(len(source_corr), -1, -1)], dim=2)
        atoms = jacobian.transpose(1, 2) @ jacobian
        information = torch.eye(6, dtype=source_corr.dtype, device=source_corr.device) * 1e-3
        available = torch.ones(len(source_corr), dtype=torch.bool, device=source_corr.device)
        selected = []
        for _ in range(min(self.config.action_count, len(source_corr))):
            _, current_logdet = torch.linalg.slogdet(information)
            candidates = information[None] + posterior[:, None, None] * atoms
            sign, candidate_logdet = torch.linalg.slogdet(candidates)
            marginal = torch.where(sign > 0, candidate_logdet - current_logdet, torch.full_like(candidate_logdet, -torch.inf))
            utility = (2 * posterior - 1) * marginal
            utility = torch.where(available, utility, torch.full_like(utility, -torch.inf))
            index = int(torch.argmax(utility).item())
            if not torch.isfinite(utility[index]) or float(utility[index]) <= 0:
                break
            selected.append(index)
            available[index] = False
            information = information + posterior[index] * atoms[index]
        return torch.as_tensor(selected, dtype=torch.long, device=source_corr.device)

    def _anisotropic_indices(
        self, seeds: torch.Tensor, points: torch.Tensor, count: int
    ) -> torch.Tensor:
        distance = torch.cdist(seeds, points)
        local_count = min(self.config.local_covariance_k, len(points))
        local_indices = torch.topk(distance, local_count, largest=False, dim=1).indices
        neighborhoods = points[local_indices]
        centered = neighborhoods - neighborhoods.mean(1, keepdim=True)
        covariance = centered.transpose(1, 2) @ centered / max(local_count - 1, 1)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        floor = eigenvalues.mean(1, keepdim=True).clamp_min(1e-6) * self.config.covariance_floor_ratio
        inverse = eigenvectors @ torch.diag_embed(1 / torch.maximum(eigenvalues, floor)) @ eigenvectors.transpose(1, 2)
        difference = points[None] - seeds[:, None]
        mahalanobis = torch.einsum("sni,sij,snj->sn", difference, inverse, difference)
        return torch.topk(mahalanobis, min(count, len(points)), largest=False, dim=1).indices[None]

    def _merge(
        self,
        source_corr: torch.Tensor,
        target_corr: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        new_source: torch.Tensor,
        new_target: torch.Tensor,
        parent_posterior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(new_source) == 0:
            return source_corr, target_corr, alpha, beta, source_corr.new_empty((0,))
        owner_distance = torch.cdist(new_source, source_corr) + torch.cdist(new_target, target_corr)
        owners = owner_distance.argmin(1)
        inherited = parent_posterior[owners]
        new_alpha = 1 + inherited
        new_beta = 1 + (1 - inherited)
        all_source = torch.cat([source_corr, new_source], dim=0)
        all_target = torch.cat([target_corr, new_target], dim=0)
        all_alpha = torch.cat([alpha, new_alpha], dim=0)
        all_beta = torch.cat([beta, new_beta], dim=0)
        quantized = torch.round(torch.cat([all_source, all_target], dim=1) / 1e-4).to(torch.int64)
        _, inverse = torch.unique(quantized, dim=0, return_inverse=True)
        posterior = all_alpha / (all_alpha + all_beta)
        group_count = int(inverse.max().item()) + 1
        maximum = posterior.new_full((group_count,), -torch.inf)
        maximum.scatter_reduce_(0, inverse, posterior, reduce="amax", include_self=True)
        positions = torch.arange(len(posterior), device=all_source.device)
        candidates = torch.where(
            posterior == maximum[inverse], positions, torch.full_like(positions, len(posterior))
        )
        keep_tensor = torch.full(
            (group_count,), len(posterior), dtype=torch.long, device=all_source.device
        )
        keep_tensor.scatter_reduce_(0, inverse, candidates, reduce="amin", include_self=True)
        return (
            all_source[keep_tensor],
            all_target[keep_tensor],
            all_alpha[keep_tensor],
            all_beta[keep_tensor],
            inherited,
        )

    def _joint_pose(
        self,
        source_corr: torch.Tensor,
        target_corr: torch.Tensor,
        posterior: torch.Tensor,
        current_pose: torch.Tensor,
        observation: LineProbeObservation,
    ) -> torch.Tensor:
        residual = torch.linalg.norm(transform_points(source_corr, current_pose) - target_corr, dim=-1)
        corr_weights = posterior / (1 + (residual / self.config.inlier_scale).square())
        source = source_corr
        target = target_corr
        weights = corr_weights
        if len(observation.source):
            source = torch.cat([source, observation.source], dim=0)
            target = torch.cat([target, observation.target], dim=0)
            weights = torch.cat([weights, observation.weights], dim=0)
        pose = weighted_rigid_transform(source, target, weights)
        if not torch.isfinite(pose).all():
            return current_pose
        return pose

    def run(
        self,
        source_corr: torch.Tensor,
        target_corr: torch.Tensor,
        source_points: torch.Tensor,
        target_points: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        initial_pose: torch.Tensor,
        placeholder: torch.Tensor,
    ) -> ProbeRegorResult:
        source_corr = source_corr.squeeze(0)
        target_corr = target_corr.squeeze(0)
        source_points_flat = source_points.squeeze(0)
        target_points_flat = target_points.squeeze(0)
        alpha = torch.ones(len(source_corr), dtype=source_corr.dtype, device=source_corr.device)
        beta = torch.ones_like(alpha)
        pose = initial_pose.squeeze(0).clone()
        line_counts = []
        matched_counts = []
        action_counts = []
        correspondence_counts = [len(source_corr)]
        r_plus_values = []
        r_minus_values = []
        posterior_ess_values = []
        completed_rounds = 0
        for knn_count in self.config.round_knn:
            observation = self.probes.observe(source_points_flat, target_points_flat, pose)
            alpha, beta = self._posterior_update(source_corr, target_corr, alpha, beta, observation)
            posterior = alpha / (alpha + beta)
            actions = self._actions(source_corr, pose, posterior)
            line_counts.append(observation.line_count)
            matched_counts.append(observation.matched_count)
            action_counts.append(len(actions))
            posterior_ess_values.append(effective_sample_size(posterior))
            if len(actions) < 3:
                break
            source_indices = self._anisotropic_indices(source_corr[actions], source_points_flat, knn_count)
            target_indices = self._anisotropic_indices(target_corr[actions], target_points_flat, knn_count)
            try:
                new_source, new_target = self.regenerator.local_matching(
                    source_corr[actions][None],
                    target_corr[actions][None],
                    source_indices,
                    target_indices,
                    source_points,
                    target_points,
                    source_features,
                    target_features,
                )
                new_source = new_source.squeeze(0)
                new_target = new_target.squeeze(0)
            except (RuntimeError, IndexError):
                new_source = source_corr.new_empty((0, 3))
                new_target = target_corr.new_empty((0, 3))
            previous_count = len(source_corr)
            selected_correct_mass = posterior[actions].sum().clamp_min(1e-6)
            source_corr, target_corr, alpha, beta, inherited = self._merge(
                source_corr,
                target_corr,
                alpha,
                beta,
                new_source,
                new_target,
                posterior,
            )
            posterior = alpha / (alpha + beta)
            r_plus = float(inherited.sum().item() / selected_correct_mass.item()) if len(inherited) else 0.0
            r_minus = float((1 - inherited).sum().item() / max(previous_count, 1)) if len(inherited) else 0.0
            r_plus_values.append(r_plus)
            r_minus_values.append(r_minus)
            correspondence_counts.append(len(source_corr))
            pose = self._joint_pose(source_corr, target_corr, posterior, pose, observation)
            completed_rounds += 1
            if len(new_source) == 0 or r_plus <= 1.0 or r_minus >= 1.0:
                break
        final_observation = self.probes.observe(source_points_flat, target_points_flat, pose)
        alpha, beta = self._posterior_update(source_corr, target_corr, alpha, beta, final_observation)
        posterior = alpha / (alpha + beta)
        pose = self._joint_pose(source_corr, target_corr, posterior, pose, final_observation)
        refined, final_source, final_target = self.estimator.post_refinement_points(
            pose[None], source_points, target_points, 20
        )
        return ProbeRegorResult(
            refined,
            final_source,
            final_target,
            completed_rounds,
            line_counts,
            matched_counts,
            action_counts,
            correspondence_counts,
            r_plus_values,
            r_minus_values,
            posterior_ess_values,
        )
