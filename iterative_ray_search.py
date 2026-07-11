import math
import time
from dataclasses import dataclass, field, replace

import torch

from ray_constraint_builder import RayConstraintBuilder
from ray_constraint_memory import ConstraintMemory, RejectedBasin
from ray_evidence import align_ray_signatures, bidirectional_ray_evaluation, detach_ray_signature, ray_signature
from ray_guided_regenerator import PoseHypothesis
from ray_pose_validator import RayPoseValidator


@dataclass(frozen=True)
class RaySelection:
    target_indices: torch.Tensor
    source_indices: torch.Tensor
    active_ray_count: int
    new_ray_count: int


@dataclass
class HypothesisArchive:
    hypotheses: list = field(default_factory=list)
    incumbent: PoseHypothesis | None = None
    rejected_this_round: list = field(default_factory=list)
    _next_id: int = 0

    def assign_ids(self, hypotheses):
        for hypothesis in hypotheses:
            if hypothesis.hypothesis_id < 0:
                hypothesis.hypothesis_id = self._next_id
                self._next_id += 1
            else:
                self._next_id = max(self._next_id, hypothesis.hypothesis_id + 1)

    def add(self, hypotheses):
        self.assign_ids(hypotheses)
        self.hypotheses.extend(hypotheses)

    @property
    def best_pose(self):
        return self.incumbent.pose if self.incumbent is not None else None

    @property
    def best_hypothesis(self):
        return self.incumbent


@dataclass(frozen=True)
class SearchResult:
    pose: torch.Tensor
    archive: HypothesisArchive
    raw_archive: HypothesisArchive
    post_refinement_archive: HypothesisArchive
    round_logs: list
    candidate_logs: list
    raw_candidate_logs: list
    refinement_logs: list
    refinement_candidates: list
    timings: dict

    @property
    def refined_archive(self):
        return self.archive


class RaySelector:
    def __init__(self, active_ray_count, variance_min_observers=2):
        self.active_ray_count = int(active_ray_count)
        self.variance_min_observers = int(variance_min_observers)
        self._uses = {}

    def reset(self):
        self._uses.clear()

    @staticmethod
    def _codes(keys, reference):
        combined = torch.cat([keys, reference], dim=0)
        pixel_scale = int(combined[:, 2].max().item()) + 1
        frame_scale = int(combined[:, 1].max().item()) + 1
        return keys[:, 0] * frame_scale * pixel_scale + keys[:, 1] * pixel_scale + keys[:, 2]

    def _rank(self, signature, comparison_signatures):
        if signature.residuals.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=signature.residuals.device)
        use_penalty = torch.tensor(
            [self._uses.get(tuple(key), 0) for key in signature.keys.tolist()],
            device=signature.residuals.device,
            dtype=signature.residuals.dtype,
        )
        disagreement = torch.zeros_like(signature.residuals)
        common_keys, common_residuals, common_presence = align_ray_signatures(
            [signature, *comparison_signatures], min_observers=self.variance_min_observers, return_presence=True,
        )
        current_observed = common_presence[0] if common_presence.shape[0] else torch.empty(0, dtype=torch.bool, device=signature.residuals.device)
        common_keys = common_keys[current_observed]
        common_residuals = common_residuals[:, current_observed]
        common_presence = common_presence[:, current_observed]
        if common_residuals.shape[0] > 1 and common_residuals.shape[1]:
            signature_codes = self._codes(signature.keys, common_keys)
            common_codes = self._codes(common_keys, signature.keys)
            sort_codes, sorter = torch.sort(signature_codes)
            positions = sorter[torch.searchsorted(sort_codes, common_codes)]
            observer_count = common_presence.sum(dim=0).to(common_residuals.dtype)
            mean = (common_residuals * common_presence).sum(dim=0) / observer_count
            variance = ((common_residuals - mean[None]) ** 2 * common_presence).sum(dim=0) / observer_count
            disagreement[positions] = variance.sqrt()
        score = signature.residuals + disagreement + 0.01 / (1.0 + use_penalty)
        return torch.argsort(score, descending=True)

    def _take_diverse(self, signature, order, count):
        if count == 0:
            return order[:0]
        frames = torch.unique(signature.keys[order, 1])
        frame_cap = max(1, math.ceil(count / max(1, frames.numel())))
        selected, per_frame = [], {}
        for index in order.tolist():
            frame = int(signature.keys[index, 1])
            if per_frame.get(frame, 0) >= frame_cap:
                continue
            selected.append(index)
            per_frame[frame] = per_frame.get(frame, 0) + 1
            if len(selected) == count:
                break
        if len(selected) < count:
            selected.extend(index for index in order.tolist() if index not in set(selected))
        return torch.tensor(selected[:count], device=order.device, dtype=torch.long)

    @staticmethod
    def _comparison_signatures(archive, direction):
        incumbent = archive.incumbent
        if incumbent is None:
            return [ray_signature(item.search_evaluation[direction]) for item in archive.hypotheses if hasattr(item, "search_evaluation")]
        return [
            ray_signature(item.search_evaluation[direction])
            for item in archive.hypotheses
            if item is not incumbent
            and hasattr(item, "search_evaluation")
        ]

    def select(self, search_evaluation, archive):
        target_signature = ray_signature(search_evaluation["target"])
        source_signature = ray_signature(search_evaluation["source"])
        target_comparisons = self._comparison_signatures(archive, "target")
        source_comparisons = self._comparison_signatures(archive, "source")
        target_order = self._rank(target_signature, target_comparisons)
        source_order = self._rank(source_signature, source_comparisons)
        target_count = min((self.active_ray_count + 1) // 2, target_order.numel())
        source_count = min(self.active_ray_count - target_count, source_order.numel())
        if source_count < self.active_ray_count // 4 and target_order.numel() > target_count:
            target_count = min(self.active_ray_count - source_count, target_order.numel())
        target_signature_indices = self._take_diverse(target_signature, target_order, target_count)
        source_signature_indices = self._take_diverse(source_signature, source_order, source_count)
        target_indices = target_signature.observation_indices[target_signature_indices]
        source_indices = source_signature.observation_indices[source_signature_indices]
        ray_keys = torch.cat([
            target_signature.keys[target_signature_indices],
            source_signature.keys[source_signature_indices],
        ])
        new_count = sum(tuple(key) not in self._uses for key in ray_keys.tolist())
        for key in ray_keys.tolist():
            canonical_key = tuple(key)
            self._uses[canonical_key] = self._uses.get(canonical_key, 0) + 1
        return RaySelection(target_indices, source_indices, int(ray_keys.shape[0]), int(new_count))


class IterativeRaySearch:
    def __init__(self, config, ray_selector, constraint_builder, regenerator, validator):
        self.config = config.validate()
        self.ray_selector = ray_selector
        self.constraint_builder = constraint_builder
        self.regenerator = regenerator
        self.validator = validator

    @staticmethod
    def _pose_distance(first, second, rotation_scale, translation_scale):
        first = first[0] if first.ndim == 3 else first
        second = second[0] if second.ndim == 3 else second
        relative = first[:3, :3].transpose(0, 1) @ second[:3, :3]
        angle = torch.acos(torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        translation = torch.linalg.norm(first[:3, 3] - second[:3, 3])
        return torch.sqrt((angle / rotation_scale) ** 2 + (translation / translation_scale) ** 2)

    def _score(self, hypothesis, inputs):
        search = self.validator.evaluate(
            hypothesis.pose, inputs["src_points"], inputs["tgt_points"], inputs["source_rays"], inputs["target_rays"], 1,
        )
        validation = self.validator.evaluate(
            hypothesis.pose, inputs["src_points"], inputs["tgt_points"], inputs["source_rays"], inputs["target_rays"], 0,
        )
        hypothesis.search_ray_score = search.score
        hypothesis.search_free_violation = search.free_violation
        hypothesis.search_insufficient_evidence = search.insufficient_evidence
        hypothesis.validation_ray_score = validation.score
        hypothesis.surface_support = validation.surface_support
        hypothesis.valid_observation_count = validation.valid_observation_count
        hypothesis.valid_observation_ratio = validation.valid_observation_ratio
        hypothesis.frame_coverage = validation.frame_coverage
        hypothesis.bidirectional_consistency = validation.bidirectional_consistency
        hypothesis.insufficient_evidence = validation.insufficient_evidence
        hypothesis.ray_signature = search.signature
        hypothesis.search_evaluation = search.evaluation
        hypothesis.geometry_rmse = self._correspondence_rmse(hypothesis)
        return search, validation

    @staticmethod
    def _pose_correspondence_rmse(pose, src_corr, tgt_corr):
        if src_corr.shape[1] == 0 or tgt_corr.shape[1] == 0:
            return float("inf")
        pose = pose[0] if pose.ndim == 3 else pose
        warped = src_corr @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
        residuals = torch.linalg.norm(warped - tgt_corr, dim=-1)
        return float(torch.sqrt(torch.mean(residuals.square())).item())

    @classmethod
    def _correspondence_rmse(cls, hypothesis):
        return cls._pose_correspondence_rmse(hypothesis.pose, hypothesis.src_corr, hypothesis.tgt_corr)

    @staticmethod
    def _candidate_snapshot(hypothesis):
        return PoseHypothesis(
            hypothesis_id=hypothesis.hypothesis_id,
            parent_id=hypothesis.parent_id,
            round_id=hypothesis.round_id,
            pose_raw=hypothesis.pose_raw.detach().clone(),
            pose_local_refined=hypothesis.pose_local_refined.detach().clone(),
            src_corr=hypothesis.src_corr.detach().clone(),
            tgt_corr=hypothesis.tgt_corr.detach().clone(),
            correspondence_scores=hypothesis.correspondence_scores.detach().clone(),
            seed_ids=hypothesis.seed_ids.detach().clone(),
            generation_mode=hypothesis.generation_mode,
            descriptor_score=hypothesis.descriptor_score,
            predicted_escape_score=hypothesis.predicted_escape_score,
            search_ray_score=hypothesis.search_ray_score,
            search_free_violation=hypothesis.search_free_violation,
            validation_ray_score=hypothesis.validation_ray_score,
            surface_support=hypothesis.surface_support,
            valid_observation_count=hypothesis.valid_observation_count,
            valid_observation_ratio=hypothesis.valid_observation_ratio,
            frame_coverage=hypothesis.frame_coverage,
            bidirectional_consistency=hypothesis.bidirectional_consistency,
            nearest_history_distance=hypothesis.nearest_history_distance,
            signature_similarity=hypothesis.signature_similarity,
            search_insufficient_evidence=hypothesis.search_insufficient_evidence,
            insufficient_evidence=hypothesis.insufficient_evidence,
            rejected_by_history=hypothesis.rejected_by_history,
            ray_signature=detach_ray_signature(hypothesis.ray_signature) if hypothesis.ray_signature is not None else None,
            geometry_rmse=hypothesis.geometry_rmse,
            refinement_attempted=hypothesis.refinement_attempted,
            entered_refinement_pool=hypothesis.entered_refinement_pool,
            refinement_child_id=hypothesis.refinement_child_id,
            refinement_accepted=hypothesis.refinement_accepted,
            refinement_reject_reason=hypothesis.refinement_reject_reason,
            rotation_delta_deg=hypothesis.rotation_delta_deg,
            translation_delta_m=hypothesis.translation_delta_m,
            parent_validation_score=hypothesis.parent_validation_score,
            child_validation_score=hypothesis.child_validation_score,
            parent_free_violation=hypothesis.parent_free_violation,
            child_free_violation=hypothesis.child_free_violation,
            parent_geometry_rmse=hypothesis.parent_geometry_rmse,
            child_geometry_rmse=hypothesis.child_geometry_rmse,
        )

    @classmethod
    def _raw_snapshot(cls, hypothesis):
        snapshot = cls._candidate_snapshot(hypothesis)
        snapshot.pose_local_refined = snapshot.pose_raw.detach().clone()
        snapshot.geometry_rmse = cls._correspondence_rmse(snapshot)
        return snapshot

    @staticmethod
    def _copy_refinement_metadata(destination, source):
        for name in (
            "refinement_attempted",
            "entered_refinement_pool",
            "refinement_child_id",
            "refinement_accepted",
            "refinement_reject_reason",
            "rotation_delta_deg",
            "translation_delta_m",
            "parent_validation_score",
            "child_validation_score",
            "parent_free_violation",
            "child_free_violation",
            "parent_geometry_rmse",
            "child_geometry_rmse",
        ):
            setattr(destination, name, getattr(source, name))

    @staticmethod
    def _rotation_delta_deg(relative):
        cosine = torch.clamp((torch.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        return float((torch.acos(cosine) * 180.0 / math.pi).item())

    def _accept_refined_child(self, parent, child):
        parent_pose = parent.pose[0] if parent.pose.ndim == 3 else parent.pose
        child_pose = child.pose[0] if child.pose.ndim == 3 else child.pose
        relative = child_pose @ torch.linalg.inv(parent_pose)
        rotation_delta_deg = self._rotation_delta_deg(relative)
        translation_delta_m = float(torch.linalg.norm(relative[:3, 3]).item())
        within_trust_region = (
            rotation_delta_deg <= self.config.refine_trust_rotation_deg
            and translation_delta_m <= self.config.refine_trust_translation_m
        )
        validation_improved = not child.insufficient_evidence and (
            parent.insufficient_evidence
            or child.validation_ray_score < parent.validation_ray_score - self.config.refine_validation_min_improvement
        )
        free_space_not_worse = child.search_free_violation <= parent.search_free_violation + self.config.refine_free_violation_tolerance
        parent_geometry_rmse = self._pose_correspondence_rmse(parent.pose_raw, child.src_corr, child.tgt_corr)
        geometry_not_worse = child.geometry_rmse <= parent_geometry_rmse + self.config.refine_rmse_tolerance
        accepted = within_trust_region and validation_improved and free_space_not_worse and geometry_not_worse
        diagnostics = {
            "rotation_delta_deg": rotation_delta_deg,
            "translation_delta_m": translation_delta_m,
            "within_trust_region": within_trust_region,
            "validation_improved": validation_improved,
            "free_space_not_worse": free_space_not_worse,
            "geometry_not_worse": geometry_not_worse,
            "parent_geometry_rmse": parent_geometry_rmse,
            "child_geometry_rmse": child.geometry_rmse,
        }
        return accepted, diagnostics

    def _nms(self, hypotheses, score_name="validation_ray_score"):
        unique, duplicates = [], []
        ordered = sorted(hypotheses, key=lambda item: (getattr(item, score_name), -item.descriptor_score))
        rotation_scale = math.radians(self.config.pose_nms_rotation_deg)
        for hypothesis in ordered:
            if any(self._pose_distance(hypothesis.pose, other.pose, rotation_scale, self.config.pose_nms_translation) < self.config.pose_nms_threshold for other in unique):
                duplicates.append(hypothesis)
            else:
                unique.append(hypothesis)
        return unique, duplicates

    def _is_negative_basin(self, hypothesis, incumbent):
        return (
            hypothesis.has_sufficient_evidence
            and hypothesis.validation_ray_score > incumbent.validation_ray_score + self.config.history_bad_margin
            and hypothesis.search_free_violation >= incumbent.search_free_violation + self.config.history_search_energy_margin
        )

    def _candidate_log(self, pair_id, hypothesis):
        return {
            "pair_id": pair_id,
            "round_id": hypothesis.round_id,
            "candidate_id": hypothesis.hypothesis_id,
            "parent_id": hypothesis.parent_id,
            "generation_mode": hypothesis.generation_mode,
            "seed_ids": hypothesis.seed_ids.detach().cpu().tolist(),
            "correspondence_count": hypothesis.correspondence_count,
            "descriptor_score": hypothesis.descriptor_score,
            "escape_score": hypothesis.predicted_escape_score,
            "rejected_by_history": int(hypothesis.rejected_by_history),
            "search_ray_score": hypothesis.search_ray_score,
            "search_free_violation": hypothesis.search_free_violation,
            "validation_ray_score": hypothesis.validation_ray_score,
            "surface_support": hypothesis.surface_support,
            "valid_observation_ratio": hypothesis.valid_observation_ratio,
            "nearest_history_distance": hypothesis.nearest_history_distance,
            "signature_similarity": hypothesis.signature_similarity,
            "search_insufficient_evidence": int(hypothesis.search_insufficient_evidence),
            "validation_insufficient_evidence": int(hypothesis.insufficient_evidence),
            "geometry_rmse": hypothesis.geometry_rmse,
            "refinement_attempted": int(hypothesis.refinement_attempted),
            "entered_refinement_pool": int(hypothesis.entered_refinement_pool),
            "refinement_child_id": hypothesis.refinement_child_id,
            "refinement_accepted": "" if hypothesis.refinement_accepted is None else int(hypothesis.refinement_accepted),
            "refinement_reject_reason": hypothesis.refinement_reject_reason,
            "rotation_delta_deg": hypothesis.rotation_delta_deg,
            "translation_delta_m": hypothesis.translation_delta_m,
            "parent_validation_score": hypothesis.parent_validation_score,
            "child_validation_score": hypothesis.child_validation_score,
            "parent_free_violation": hypothesis.parent_free_violation,
            "child_free_violation": hypothesis.child_free_violation,
            "parent_geometry_rmse": hypothesis.parent_geometry_rmse,
            "child_geometry_rmse": hypothesis.child_geometry_rmse,
            "pose_raw": hypothesis.pose_raw[0].detach().cpu().tolist(),
            "pose_local_refined": hypothesis.pose_local_refined[0].detach().cpu().tolist(),
            "pose_merge_refined": None,
            "pose_system_final": None,
        }

    def run(self, initial_pose, pair_inputs, time_budget_seconds=None, initial_hypothesis=None):
        started = time.perf_counter()
        self.ray_selector.reset()
        budget = self.config.search_time_budget_seconds if time_budget_seconds is None else float(time_budget_seconds)
        archive, raw_archive, post_refinement_archive, memory = HypothesisArchive(), HypothesisArchive(), HypothesisArchive(), ConstraintMemory()
        scoring_inputs = pair_inputs
        if self.config.method == "shuffled_ray":
            if "shuffled_source_rays" not in pair_inputs or "shuffled_target_rays" not in pair_inputs:
                raise ValueError("shuffled_ray requires shuffled_source_rays and shuffled_target_rays from a different pair.")
            scoring_inputs = {**pair_inputs, "source_rays": pair_inputs["shuffled_source_rays"], "target_rays": pair_inputs["shuffled_target_rays"]}
        r1 = initial_hypothesis or PoseHypothesis(
            hypothesis_id=-1,
            parent_id=-1,
            round_id=0,
            pose_raw=initial_pose,
            pose_local_refined=initial_pose,
            src_corr=pair_inputs["src_points"][:, :0],
            tgt_corr=pair_inputs["tgt_points"][:, :0],
            correspondence_scores=torch.empty(0, device=initial_pose.device),
            seed_ids=torch.empty((0, 2), dtype=torch.long, device=initial_pose.device),
            generation_mode="r1",
        )
        self.regenerator.prepare_descriptor_cache(pair_inputs["src_features"], pair_inputs["tgt_features"])
        self._score(r1, scoring_inputs)
        raw_archive.assign_ids([r1])
        raw_r1, post_r1 = self._raw_snapshot(r1), self._raw_snapshot(r1)
        raw_archive.add([raw_r1])
        post_refinement_archive.add([post_r1])
        archive.add([r1])
        archive.incumbent = r1
        round_logs = []
        candidate_logs = [{**self._candidate_log(pair_inputs.get("pair_id", ""), post_r1), "archive_stage": "post_refine"}]
        raw_candidate_logs = [{**self._candidate_log(pair_inputs.get("pair_id", ""), raw_r1), "archive_stage": "raw"}]
        refinement_logs, refinement_candidates = [], []
        next_child_id = 1_000_000
        stagnation = 0
        timings = {"constraint_time": 0.0, "candidate_generation_time": 0.0, "candidate_validation_time": 0.0, "total_time": 0.0}
        if self.config.method == "r1_only":
            timings["total_time"] = time.perf_counter() - started
            return SearchResult(
                archive.best_pose,
                archive,
                raw_archive,
                post_refinement_archive,
                round_logs,
                candidate_logs,
                raw_candidate_logs,
                refinement_logs,
                refinement_candidates,
                timings,
            )
        for round_id in range(1, self.config.search_max_rounds + 1):
            if budget > 0 and time.perf_counter() - started >= budget:
                break
            incumbent = archive.incumbent
            search_evidence = bidirectional_ray_evaluation(
                scoring_inputs["src_points"], scoring_inputs["tgt_points"], incumbent.pose,
                scoring_inputs["source_rays"], scoring_inputs["target_rays"],
                self.config.ray_trunc_margin, self.config.ray_surface_sigma, split_id=1,
            )
            selection = self.ray_selector.select(search_evidence, archive)
            constraint_started = time.perf_counter()
            constraints = self.constraint_builder.build(
                incumbent.pose, scoring_inputs["src_points"], scoring_inputs["tgt_points"],
                scoring_inputs["source_rays"], scoring_inputs["target_rays"], search_evidence, selection,
            )
            if self.config.method == "repeated_regor":
                constraints = replace(
                    constraints,
                    constraints=(),
                    G=constraints.G[:0],
                    b=constraints.b[:0],
                    weights=constraints.weights[:0],
                    preferred_direction=torch.zeros_like(constraints.preferred_direction),
                )
            timings["constraint_time"] += time.perf_counter() - constraint_started
            generation_started = time.perf_counter()
            proposals = self.regenerator.generate_hypotheses(
                None, None, pair_inputs["src_points"], pair_inputs["tgt_points"],
                pair_inputs["src_features"], pair_inputs["tgt_features"], constraints,
                self.config.candidates_per_round * self.config.candidate_pool_multiplier,
                round_id,
                incumbent.hypothesis_id,
            )
            timings["candidate_generation_time"] += time.perf_counter() - generation_started
            validation_started = time.perf_counter()
            raw_archive.assign_ids(proposals)
            for proposal in proposals:
                self._score(proposal, scoring_inputs)
                repeated, distance, similarity = (False, float("inf"), 0.0) if self.config.method == "repeated_regor" else memory.repeated_basin(
                    proposal.pose, proposal.ray_signature, proposal.search_free_violation,
                    self.config.history_signature_similarity, self.config.history_energy_tolerance,
                )
                proposal.nearest_history_distance = distance
                proposal.signature_similarity = similarity
                proposal.rejected_by_history = repeated
            raw_history_rejected_count = sum(item.rejected_by_history for item in proposals)
            raw_snapshots = [self._raw_snapshot(proposal) for proposal in proposals]
            post_raw_snapshots = [self._raw_snapshot(proposal) for proposal in proposals]
            raw_archive.add(raw_snapshots)
            post_refinement_archive.add(post_raw_snapshots)
            raw_snapshots_by_id = {proposal.hypothesis_id: snapshot for proposal, snapshot in zip(proposals, raw_snapshots)}
            post_raw_snapshots_by_id = {proposal.hypothesis_id: snapshot for proposal, snapshot in zip(proposals, post_raw_snapshots)}
            raw_unique, _ = self._nms(proposals, score_name="validation_ray_score")
            admissible_raw = [
                proposal for proposal in raw_unique
                if not proposal.rejected_by_history and not proposal.insufficient_evidence and not math.isinf(proposal.validation_ray_score)
            ]
            refinement_pool = sorted(admissible_raw, key=lambda item: item.validation_ray_score)[:self.config.candidates_per_round]
            refined_children, round_refinement_logs, round_refinement_candidates = [], [], []
            for parent in refinement_pool:
                parent.refinement_attempted = True
                parent.entered_refinement_pool = True
                child = self.regenerator.refine_hypothesis(
                    parent,
                    pair_inputs["src_points"],
                    pair_inputs["tgt_points"],
                    pair_inputs["src_features"],
                    pair_inputs["tgt_features"],
                )
                if child is None:
                    parent.refinement_accepted = False
                    parent.refinement_reject_reason = "insufficient_correspondences"
                    continue
                child.hypothesis_id = next_child_id
                next_child_id += 1
                parent.refinement_child_id = child.hypothesis_id
                self._score(child, scoring_inputs)
                repeated, distance, similarity = (False, float("inf"), 0.0) if self.config.method == "repeated_regor" else memory.repeated_basin(
                    child.pose, child.ray_signature, child.search_free_violation,
                    self.config.history_signature_similarity, self.config.history_energy_tolerance,
                )
                child.nearest_history_distance = distance
                child.signature_similarity = similarity
                child.rejected_by_history = repeated
                accepted, diagnostics = self._accept_refined_child(parent, child)
                parent.rotation_delta_deg = diagnostics["rotation_delta_deg"]
                parent.translation_delta_m = diagnostics["translation_delta_m"]
                parent.parent_validation_score = parent.validation_ray_score
                parent.child_validation_score = child.validation_ray_score
                parent.parent_free_violation = parent.search_free_violation
                parent.child_free_violation = child.search_free_violation
                parent.parent_geometry_rmse = diagnostics["parent_geometry_rmse"]
                parent.child_geometry_rmse = diagnostics["child_geometry_rmse"]
                child.rotation_delta_deg = diagnostics["rotation_delta_deg"]
                child.translation_delta_m = diagnostics["translation_delta_m"]
                child.parent_validation_score = parent.validation_ray_score
                child.child_validation_score = child.validation_ray_score
                child.parent_free_violation = parent.search_free_violation
                child.child_free_violation = child.search_free_violation
                child.parent_geometry_rmse = diagnostics["parent_geometry_rmse"]
                child.child_geometry_rmse = diagnostics["child_geometry_rmse"]
                if child.rejected_by_history:
                    accepted = False
                    reject_reason = "repeated_history"
                elif not diagnostics["within_trust_region"]:
                    reject_reason = "outside_trust_region"
                elif not diagnostics["validation_improved"]:
                    reject_reason = "validation_not_improved"
                elif not diagnostics["free_space_not_worse"]:
                    reject_reason = "free_space_worsened"
                elif not diagnostics["geometry_not_worse"]:
                    reject_reason = "geometry_worsened"
                else:
                    reject_reason = ""
                parent.refinement_accepted = accepted
                parent.refinement_reject_reason = reject_reason
                child.refinement_accepted = accepted
                child.refinement_reject_reason = reject_reason
                round_refinement_candidates.append(child)
                round_refinement_logs.append({
                    **self._candidate_log(pair_inputs.get("pair_id", ""), child),
                    "archive_stage": "refined_child_accepted" if accepted else "refined_child_rejected",
                })
                if accepted:
                    refined_children.append(child)
            for parent in proposals:
                self._copy_refinement_metadata(raw_snapshots_by_id[parent.hypothesis_id], parent)
                self._copy_refinement_metadata(post_raw_snapshots_by_id[parent.hypothesis_id], parent)
            raw_candidate_logs.extend(
                {**self._candidate_log(pair_inputs.get("pair_id", ""), proposal), "archive_stage": "raw"}
                for proposal in raw_snapshots
            )
            accepted_child_snapshots = [self._candidate_snapshot(child) for child in refined_children]
            post_refinement_archive.add(accepted_child_snapshots)
            post_round_candidates = list(admissible_raw) + refined_children
            selection_pool = [incumbent] + post_round_candidates
            best_new = min(selection_pool, key=lambda item: item.validation_ray_score)
            improved = best_new is not incumbent and best_new.validation_ray_score < incumbent.validation_ray_score - self.config.validation_min_improvement and best_new.surface_support >= self.config.validation_min_surface_support
            archive.add(post_round_candidates)
            if improved:
                archive.incumbent = best_new
                stagnation = 0
            else:
                stagnation += 1
            history_candidates = list(admissible_raw) + refined_children
            archive.rejected_this_round = [
                proposal for proposal in history_candidates
                if proposal is not archive.incumbent and self._is_negative_basin(proposal, incumbent)
            ]
            for rejected_hypothesis in archive.rejected_this_round if self.config.method != "repeated_regor" else []:
                if rejected_hypothesis.ray_signature is None:
                    continue
                memory.add(RejectedBasin(
                    hypothesis_id=rejected_hypothesis.hypothesis_id,
                    pose=rejected_hypothesis.pose.detach().clone(),
                    ray_keys=torch.cat([search_evidence["target"].ray_keys, search_evidence["source"].ray_keys]).detach().clone(),
                    ray_signature=detach_ray_signature(rejected_hypothesis.ray_signature),
                    search_energy=float(rejected_hypothesis.search_free_violation),
                    local_information_matrix=constraints.information_matrix.detach().clone(),
                    rotation_radius=math.radians(self.config.history_rotation_radius_deg),
                    translation_radius=self.config.history_translation_radius,
                ))
            timings["candidate_validation_time"] += time.perf_counter() - validation_started
            candidate_logs.extend(
                {**self._candidate_log(pair_inputs.get("pair_id", ""), proposal), "archive_stage": "post_refine"}
                for proposal in post_raw_snapshots + accepted_child_snapshots
            )
            refinement_logs.extend(round_refinement_logs)
            refinement_candidates.extend(round_refinement_candidates)
            round_logs.append({
                "pair_id": pair_inputs.get("pair_id", ""),
                "round_id": round_id,
                "elapsed_time": time.perf_counter() - started,
                "incumbent_id": archive.incumbent.hypothesis_id,
                "incumbent_search_score": archive.incumbent.search_ray_score,
                "incumbent_validation_score": archive.incumbent.validation_ray_score,
                "active_ray_count": selection.active_ray_count,
                "new_ray_count": selection.new_ray_count,
                "conflict_ray_count": int(constraints.b.numel()),
                "valid_observation_count": archive.incumbent.valid_observation_count,
                "constraint_rank": constraints.rank,
                "constraint_condition_number": constraints.condition_number,
                "generated_candidate_count": len(proposals),
                "raw_unique_candidate_count": len(raw_unique),
                "pre_refine_candidate_count": len(refinement_pool),
                "admissible_raw_candidate_count": len(admissible_raw),
                "accepted_refined_child_count": len(refined_children),
                "rejected_refined_child_count": sum(not item.refinement_accepted for item in round_refinement_candidates),
                "post_refinement_candidate_count": len(post_raw_snapshots) + len(accepted_child_snapshots),
                "rejected_by_existing_history_count": raw_history_rejected_count,
                "rejected_by_existing_history_after_refinement_count": sum(item.rejected_by_history for item in round_refinement_candidates),
                "new_negative_basin_count": len(archive.rejected_this_round),
                "independent_candidate_count": sum(item.generation_mode == "independent" for item in admissible_raw),
                "improved": int(improved),
                "history_basin_count": len(memory),
            })
            if not proposals:
                break
            if self.config.enable_adaptive_stop and stagnation >= self.config.stagnation_rounds:
                break
        timings["total_time"] = time.perf_counter() - started
        return SearchResult(
            archive.best_pose,
            archive,
            raw_archive,
            post_refinement_archive,
            round_logs,
            candidate_logs,
            raw_candidate_logs,
            refinement_logs,
            refinement_candidates,
            timings,
        )


ConstraintBuilder = RayConstraintBuilder
