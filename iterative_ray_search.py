import math
import time
from dataclasses import dataclass, field, replace

import torch

from ray_constraint_builder import RayConstraintBuilder
from ray_constraint_memory import ConstraintMemory, RejectedBasin
from ray_evidence import bidirectional_ray_evaluation
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

    def add(self, hypotheses):
        for hypothesis in hypotheses:
            if hypothesis.hypothesis_id < 0:
                hypothesis.hypothesis_id = self._next_id
                self._next_id += 1
            self.hypotheses.append(hypothesis)

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
    round_logs: list
    candidate_logs: list
    timings: dict


class RaySelector:
    def __init__(self, active_ray_count):
        self.active_ray_count = int(active_ray_count)
        self._uses = {}

    def _rank(self, evaluation, comparison_evaluations):
        if evaluation.per_ray_residuals.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=evaluation.per_ray_residuals.device)
        use_penalty = torch.tensor(
            [self._uses.get(int(ray_id), 0) for ray_id in evaluation.ray_ids.tolist()],
            device=evaluation.per_ray_residuals.device,
            dtype=evaluation.per_ray_residuals.dtype,
        )
        comparable = [value.per_ray_residuals for value in comparison_evaluations if value.per_ray_residuals.shape == evaluation.per_ray_residuals.shape]
        disagreement = torch.var(torch.stack(comparable), dim=0, unbiased=False) if len(comparable) > 1 else torch.zeros_like(evaluation.per_ray_residuals)
        score = evaluation.per_ray_residuals + disagreement.sqrt() + 0.01 / (1.0 + use_penalty)
        return torch.argsort(score, descending=True)

    def _take_diverse(self, evaluation, order, count):
        if count == 0:
            return order[:0]
        frames = torch.unique(evaluation.frame_indices[order])
        frame_cap = max(1, math.ceil(count / max(1, frames.numel())))
        selected, per_frame = [], {}
        for index in order.tolist():
            frame = int(evaluation.frame_indices[index])
            if per_frame.get(frame, 0) >= frame_cap:
                continue
            selected.append(index)
            per_frame[frame] = per_frame.get(frame, 0) + 1
            if len(selected) == count:
                break
        if len(selected) < count:
            selected.extend(index for index in order.tolist() if index not in set(selected))
        return torch.tensor(selected[:count], device=order.device, dtype=torch.long)

    def select(self, search_evaluation, archive):
        target_comparisons = [item.search_evaluation["target"] for item in archive.hypotheses if hasattr(item, "search_evaluation")]
        source_comparisons = [item.search_evaluation["source"] for item in archive.hypotheses if hasattr(item, "search_evaluation")]
        target_order = self._rank(search_evaluation["target"], target_comparisons)
        source_order = self._rank(search_evaluation["source"], source_comparisons)
        target_count = min((self.active_ray_count + 1) // 2, target_order.numel())
        source_count = min(self.active_ray_count - target_count, source_order.numel())
        if source_count < self.active_ray_count // 4 and target_order.numel() > target_count:
            target_count = min(self.active_ray_count - source_count, target_order.numel())
        target_indices = self._take_diverse(search_evaluation["target"], target_order, target_count)
        source_indices = self._take_diverse(search_evaluation["source"], source_order, source_count)
        ray_ids = torch.cat([
            search_evaluation["target"].ray_ids[target_indices],
            search_evaluation["source"].ray_ids[source_indices],
        ])
        new_count = sum(int(ray_id) not in self._uses for ray_id in ray_ids.tolist())
        for ray_id in ray_ids.tolist():
            self._uses[int(ray_id)] = self._uses.get(int(ray_id), 0) + 1
        return RaySelection(target_indices, source_indices, int(ray_ids.numel()), int(new_count))


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
        hypothesis.validation_ray_score = validation.score
        hypothesis.surface_support = validation.surface_support
        hypothesis.valid_observation_count = validation.valid_observation_count
        hypothesis.valid_observation_ratio = validation.valid_observation_ratio
        hypothesis.frame_coverage = validation.frame_coverage
        hypothesis.bidirectional_consistency = validation.bidirectional_consistency
        hypothesis.insufficient_evidence = validation.insufficient_evidence
        hypothesis.ray_signature = search.signature
        hypothesis.search_evaluation = search.evaluation
        return search, validation

    def _prescore(self, hypotheses, inputs):
        if not hypotheses:
            return
        poses = torch.cat([item.pose for item in hypotheses], dim=0)
        search_scores = self.validator.evaluate_batch(
            poses, inputs["src_points"], inputs["tgt_points"], inputs["source_rays"], inputs["target_rays"], 1,
        )
        validation_scores = self.validator.evaluate_batch(
            poses, inputs["src_points"], inputs["tgt_points"], inputs["source_rays"], inputs["target_rays"], 0,
        )
        for hypothesis, search, validation in zip(hypotheses, search_scores, validation_scores):
            hypothesis.search_ray_score = search["score"]
            hypothesis.validation_ray_score = validation["score"]
            hypothesis.surface_support = validation["surface_support"]
            hypothesis.valid_observation_count = validation["valid_observation_count"]
            hypothesis.insufficient_evidence = validation["insufficient_evidence"]

    def _nms(self, hypotheses):
        unique, duplicates = [], []
        ordered = sorted(hypotheses, key=lambda item: (item.validation_ray_score, -item.descriptor_score))
        rotation_scale = math.radians(self.config.pose_nms_rotation_deg)
        for hypothesis in ordered:
            if any(self._pose_distance(hypothesis.pose, other.pose, rotation_scale, self.config.pose_nms_translation) < self.config.pose_nms_threshold for other in unique):
                duplicates.append(hypothesis)
            else:
                unique.append(hypothesis)
        return unique, duplicates

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
            "history_penalty": hypothesis.history_penalty,
            "search_ray_score": hypothesis.search_ray_score,
            "validation_ray_score": hypothesis.validation_ray_score,
            "surface_support": hypothesis.surface_support,
            "valid_observation_ratio": hypothesis.valid_observation_ratio,
            "nearest_history_distance": hypothesis.nearest_history_distance,
            "signature_similarity": hypothesis.signature_similarity,
            "insufficient_evidence": int(hypothesis.insufficient_evidence),
            "pose_raw": hypothesis.pose_raw[0].detach().cpu().tolist(),
            "pose_local_refined": hypothesis.pose_local_refined[0].detach().cpu().tolist(),
            "pose_merge_refined": None,
            "pose_system_final": None,
        }

    def run(self, initial_pose, pair_inputs, time_budget_seconds=None, initial_hypothesis=None):
        started = time.perf_counter()
        budget = self.config.search_time_budget_seconds if time_budget_seconds is None else float(time_budget_seconds)
        archive, memory = HypothesisArchive(), ConstraintMemory()
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
        archive.add([r1])
        archive.incumbent = r1
        round_logs, candidate_logs = [], [self._candidate_log(pair_inputs.get("pair_id", ""), r1)]
        stagnation = 0
        timings = {"constraint_time": 0.0, "candidate_generation_time": 0.0, "candidate_validation_time": 0.0, "total_time": 0.0}
        if self.config.method == "r1_only":
            timings["total_time"] = time.perf_counter() - started
            return SearchResult(archive.best_pose, archive, round_logs, candidate_logs, timings)
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
                pair_inputs["src_features"], pair_inputs["tgt_features"], constraints, memory,
                self.config.candidates_per_round, round_id, incumbent.hypothesis_id,
            )
            timings["candidate_generation_time"] += time.perf_counter() - generation_started
            validation_started = time.perf_counter()
            self._prescore(proposals, scoring_inputs)
            unique, duplicates = self._nms(proposals)
            for proposal in unique:
                self._score(proposal, scoring_inputs)
                repeated, distance, similarity = (False, float("inf"), 0.0) if self.config.method == "repeated_regor" else memory.repeated_basin(
                    proposal.pose, proposal.ray_signature, proposal.search_ray_score,
                    self.config.history_signature_similarity, self.config.history_energy_tolerance,
                )
                proposal.nearest_history_distance = distance
                proposal.signature_similarity = similarity
                proposal.history_penalty += float(repeated)
                proposal.rejected_by_history = repeated
            admissible = [proposal for proposal in unique if not proposal.rejected_by_history and not proposal.insufficient_evidence and not math.isinf(proposal.validation_ray_score)]
            best_new = min(admissible, key=lambda item: item.validation_ray_score, default=None)
            improved = best_new is not None and best_new.validation_ray_score < incumbent.validation_ray_score - self.config.validation_min_improvement and best_new.surface_support >= self.config.validation_min_surface_support
            archive.add(unique)
            if improved:
                archive.incumbent = best_new
                stagnation = 0
            else:
                stagnation += 1
            rejected = [proposal for proposal in unique if proposal is not archive.incumbent]
            archive.rejected_this_round = rejected + duplicates
            for rejected_hypothesis in archive.rejected_this_round if self.config.method != "repeated_regor" else []:
                if rejected_hypothesis.ray_signature is None:
                    continue
                memory.add(RejectedBasin(
                    hypothesis_id=rejected_hypothesis.hypothesis_id,
                    pose=rejected_hypothesis.pose.detach().clone(),
                    ray_ids=torch.cat([search_evidence["target"].ray_ids, search_evidence["source"].ray_ids]).detach().clone(),
                    ray_signature=rejected_hypothesis.ray_signature.detach().clone(),
                    search_energy=float(rejected_hypothesis.search_ray_score),
                    local_information_matrix=constraints.information_matrix.detach().clone(),
                    rotation_radius=math.radians(self.config.history_rotation_radius_deg),
                    translation_radius=self.config.history_translation_radius,
                ))
            timings["candidate_validation_time"] += time.perf_counter() - validation_started
            candidate_logs.extend(self._candidate_log(pair_inputs.get("pair_id", ""), proposal) for proposal in unique)
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
                "unique_candidate_count": len(unique),
                "history_rejected_count": len(archive.rejected_this_round),
                "independent_candidate_count": sum(item.generation_mode == "independent" for item in unique),
                "improved": int(improved),
                "history_basin_count": len(memory),
            })
            if not proposals:
                break
            if self.config.enable_adaptive_stop and stagnation >= self.config.stagnation_rounds:
                break
        timings["total_time"] = time.perf_counter() - started
        return SearchResult(archive.best_pose, archive, round_logs, candidate_logs, timings)


ConstraintBuilder = RayConstraintBuilder
