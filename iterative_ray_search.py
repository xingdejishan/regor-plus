import math
import time

import torch

from ray_constraint_memory import ConstraintMemory, RejectedBasin
from ray_evidence import bidirectional_ray_energy, ray_signature
from ray_guided_regenerator import PoseHypothesis


def _perturb_pose(pose, axis, amount):
    result = pose.clone()
    if axis < 3:
        basis = torch.zeros(3, device=pose.device, dtype=pose.dtype)
        basis[axis] = amount
        skew = torch.tensor([
            [0.0, -basis[2], basis[1]],
            [basis[2], 0.0, -basis[0]],
            [-basis[1], basis[0], 0.0],
        ], device=pose.device, dtype=pose.dtype)
        result[:, :3, :3] = (torch.eye(3, device=pose.device, dtype=pose.dtype)[None] + skew) @ pose[:, :3, :3]
    else:
        result[:, axis - 3, 3] += amount
    return result


class HypothesisArchive:
    def __init__(self, r1_pose):
        self.hypotheses = []
        self.best_pose = r1_pose
        self.best_hypothesis = None
        self.rejected_this_round = []

    def add(self, hypotheses):
        self.hypotheses.extend(hypotheses)

    def update_best(self):
        if not self.hypotheses:
            return
        eligible = [h for h in self.hypotheses if math.isfinite(h.validation_energy)]
        if eligible:
            self.best_hypothesis = min(eligible, key=lambda h: h.validation_energy)
            self.best_pose = self.best_hypothesis.pose


class RaySelector:
    def __init__(self, active_ray_count):
        self.active_ray_count = int(active_ray_count)

    def select(self, incumbent, hypotheses, memory, source_rays, target_rays):
        target_ids = torch.where(target_rays.split == 1)[0]
        source_ids = torch.where(source_rays.split == 1)[0]
        target_count = min(self.active_ray_count // 2, target_ids.shape[0])
        source_count = min(self.active_ray_count - target_count, source_ids.shape[0])
        target_selected = target_ids[torch.linspace(0, target_ids.shape[0] - 1, target_count, device=target_ids.device).long()] if target_count else target_ids[:0]
        source_selected = source_ids[torch.linspace(0, source_ids.shape[0] - 1, source_count, device=source_ids.device).long()] if source_count else source_ids[:0]
        return source_selected, target_selected


class ConstraintBuilder:
    def __init__(self, surface_mu=0.05, surface_sigma=0.03):
        self.surface_mu = surface_mu
        self.surface_sigma = surface_sigma

    def build(self, pose, inputs, active_rays):
        source_ids, target_ids = active_rays
        source_rays = inputs["source_rays"]
        target_rays = inputs["target_rays"]
        base = bidirectional_ray_energy(
            inputs["src_points"], inputs["tgt_points"], pose,
            source_rays, target_rays, self.surface_mu, self.surface_sigma, split=1,
        )["free"]
        gradient = []
        step = 0.005
        for axis in range(6):
            perturbed = _perturb_pose(pose, axis, step)
            energy = bidirectional_ray_energy(
                inputs["src_points"], inputs["tgt_points"], perturbed,
                source_rays, target_rays, self.surface_mu, self.surface_sigma, split=1,
            )["free"]
            gradient.append((energy - base) / step)
        gradient = torch.tensor(gradient, device=pose.device, dtype=pose.dtype)
        preferred = -gradient / (torch.linalg.norm(gradient) + 1e-6)
        return {
            "active_source_ray_ids": source_ids,
            "active_target_ray_ids": target_ids,
            "preferred_direction": preferred,
            "base_search_energy": base,
            "G": gradient[None],
            "b": torch.tensor([base], device=pose.device, dtype=pose.dtype),
        }


class IterativeRaySearch:
    def __init__(self, ray_selector, constraint_builder, regenerator, max_rounds=4, candidates_per_round=16, pose_nms_threshold=1.0, time_budget=0.0):
        self.ray_selector = ray_selector
        self.constraint_builder = constraint_builder
        self.regenerator = regenerator
        self.max_rounds = int(max_rounds)
        self.candidates_per_round = int(candidates_per_round)
        self.pose_nms_threshold = float(pose_nms_threshold)
        self.time_budget = float(time_budget)

    def _pose_distance(self, first, second):
        relative = first[0, :3, :3].T @ second[0, :3, :3]
        angle = torch.acos(torch.clamp((torch.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        translation = torch.linalg.norm(first[0, :3, 3] - second[0, :3, 3])
        return torch.sqrt((angle / 0.08726646) ** 2 + (translation / 0.10) ** 2)

    def _nms(self, hypotheses, memory):
        selected = []
        for hypothesis in sorted(hypotheses, key=lambda item: item.descriptor_score, reverse=True):
            if any(self._pose_distance(hypothesis.pose, other.pose) < self.pose_nms_threshold for other in selected):
                continue
            selected.append(hypothesis)
        return selected

    def _score(self, hypotheses, inputs):
        for hypothesis in hypotheses:
            search = bidirectional_ray_energy(
                inputs["src_points"], inputs["tgt_points"], hypothesis.pose,
                inputs["source_rays"], inputs["target_rays"],
                self.constraint_builder.surface_mu, self.constraint_builder.surface_sigma, split=1,
            )
            validation = bidirectional_ray_energy(
                inputs["src_points"], inputs["tgt_points"], hypothesis.pose,
                inputs["source_rays"], inputs["target_rays"],
                self.constraint_builder.surface_mu, self.constraint_builder.surface_sigma, split=0,
            )
            hypothesis.actual_search_energy = search["free"]
            hypothesis.validation_energy = validation["free"] + (1.0 - validation["surface"])
            hypothesis.surface_support = validation["surface"]
            hypothesis.ray_signature = ray_signature(search)

    def run(self, r1_pose, inputs, time_budget=None):
        budget = self.time_budget if time_budget is None else float(time_budget)
        started = time.perf_counter()
        archive = HypothesisArchive(r1_pose)
        memory = ConstraintMemory()
        r1 = PoseHypothesis(r1_pose, -1, -1, 0)
        self._score([r1], inputs)
        archive.add([r1])
        archive.update_best()
        round_logs = []
        for round_id in range(self.max_rounds):
            if budget > 0 and time.perf_counter() - started >= budget:
                break
            incumbent = archive.best_pose
            active_rays = self.ray_selector.select(incumbent, archive.hypotheses, memory, inputs["source_rays"], inputs["target_rays"])
            constraints = self.constraint_builder.build(incumbent, inputs, active_rays)
            proposals = self.regenerator.generate(
                incumbent,
                inputs["src_points"],
                inputs["tgt_points"],
                inputs["src_features"],
                inputs["tgt_features"],
                constraints,
                memory,
                self.candidates_per_round,
                round_id=round_id,
            )
            unique = self._nms(proposals, memory)
            self._score(unique, inputs)
            archive.rejected_this_round = [
                RejectedBasin(
                    pose=h.pose[0].detach(),
                    ray_signature=h.ray_signature.detach(),
                    active_ray_ids=torch.cat(active_rays).detach(),
                    local_G=constraints["G"].detach(),
                    local_b=constraints["b"].detach(),
                    radius_rotation=0.08726646,
                    radius_translation=0.10,
                    search_energy=h.actual_search_energy,
                ) for h in unique if h.validation_energy > min((x.validation_energy for x in unique), default=float("inf"))
            ]
            archive.add(unique)
            previous = archive.best_pose
            archive.update_best()
            memory.add_rejected(archive.rejected_this_round)
            improved = self._pose_distance(previous, archive.best_pose).item() > 1e-6
            round_logs.append({
                "round_id": round_id,
                "elapsed_time": time.perf_counter() - started,
                "active_ray_count": int(sum(x.numel() for x in active_rays)),
                "generated_candidate_count": len(proposals),
                "unique_candidate_count": len(unique),
                "rejected_by_history_count": len(proposals) - len(unique),
                "independent_explore_count": sum(1 for h in unique if h.predicted_escape_score == 0.0),
                "incumbent_search_energy": archive.best_hypothesis.actual_search_energy if archive.best_hypothesis else float("nan"),
                "incumbent_validation_energy": archive.best_hypothesis.validation_energy if archive.best_hypothesis else float("nan"),
                "improved": int(improved),
            })
            if not unique or (round_id > 0 and not improved):
                break
        return archive.best_pose, archive, round_logs
