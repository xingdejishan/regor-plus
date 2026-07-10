import hashlib
import json
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch


SEARCH_SPLIT = 1
VALIDATION_SPLIT = 0


@dataclass(frozen=True)
class RayBundle:
    frame_ids: torch.Tensor
    origins: torch.Tensor
    directions: torch.Tensor
    observed_depths: torch.Tensor
    valid_depth: torch.Tensor
    pixels: torch.Tensor
    camera_poses: torch.Tensor
    intrinsics: torch.Tensor
    confidences: torch.Tensor
    split_ids: torch.Tensor
    depth_maps: torch.Tensor
    fragment_pose: torch.Tensor
    frame_numbers: torch.Tensor | None = None
    frame_camera_poses: torch.Tensor | None = None
    frame_split_ids: torch.Tensor | None = None

    @property
    def observed_depth(self):
        return self.observed_depths

    @property
    def confidence(self):
        return self.confidences

    @property
    def split(self):
        return self.split_ids

    def to(self, device):
        values = [
            self.frame_ids, self.origins, self.directions, self.observed_depths,
            self.valid_depth, self.pixels, self.camera_poses, self.intrinsics,
            self.confidences, self.split_ids, self.depth_maps, self.fragment_pose,
            self.frame_numbers, self.frame_camera_poses, self.frame_split_ids,
        ]
        return RayBundle(*(value.to(device) if value is not None else None for value in values))

    def frame_indices(self, split_id=None):
        poses = self.frame_camera_poses if self.frame_camera_poses is not None else self.camera_poses
        splits = self.frame_split_ids if self.frame_split_ids is not None else self.split_ids
        if split_id is None:
            return torch.arange(poses.shape[0], device=poses.device)
        return torch.where(splits == int(split_id))[0]

    @property
    def projection_poses(self):
        return self.frame_camera_poses if self.frame_camera_poses is not None else self.camera_poses


@dataclass(frozen=True)
class RayEvaluation:
    free_violation: float
    surface_support: float
    valid_observation_count: int
    per_frame_scores: torch.Tensor
    per_ray_residuals: torch.Tensor
    ray_keys: torch.Tensor
    point_ids: torch.Tensor
    frame_indices: torch.Tensor
    evaluated_point_count: int
    bidirectional_consistency: float = 0.0

    @property
    def valid_observation_ratio(self):
        if self.per_frame_scores.numel() == 0 or self.evaluated_point_count == 0:
            return 0.0
        possible = int(self.per_frame_scores.shape[0] * self.evaluated_point_count)
        return float(self.valid_observation_count / max(1, possible))


@dataclass(frozen=True)
class RayResidualSignature:
    keys: torch.Tensor
    residuals: torch.Tensor
    observation_indices: torch.Tensor


def _canonicalize_residuals(keys, residuals):
    if residuals.numel() == 0:
        return RayResidualSignature(keys.reshape(0, 3), residuals, torch.empty(0, device=residuals.device, dtype=torch.long))
    unique_keys, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    values = torch.zeros(unique_keys.shape[0], dtype=residuals.dtype, device=residuals.device)
    values.scatter_reduce_(0, inverse, residuals, reduce="amax", include_self=True)
    positions = torch.arange(residuals.numel(), device=residuals.device)
    winner = residuals == values[inverse]
    representative = torch.full((unique_keys.shape[0],), residuals.numel(), dtype=torch.long, device=residuals.device)
    representative.scatter_reduce_(0, inverse[winner], positions[winner], reduce="amin", include_self=True)
    return RayResidualSignature(unique_keys, values, representative)


def align_ray_signatures(signatures):
    signatures = [signature for signature in signatures if signature.keys.numel()]
    if not signatures:
        device = torch.device("cpu")
        return torch.empty((0, 3), dtype=torch.long, device=device), torch.empty((0, 0), device=device)
    all_keys = torch.cat([signature.keys for signature in signatures], dim=0)
    unique_keys, inverse = torch.unique(all_keys, dim=0, sorted=True, return_inverse=True)
    offsets, present, values = 0, [], []
    for signature in signatures:
        ids = inverse[offsets:offsets + signature.keys.shape[0]]
        offsets += signature.keys.shape[0]
        mask = torch.zeros(unique_keys.shape[0], dtype=torch.bool, device=unique_keys.device)
        mask[ids] = True
        vector = torch.zeros(unique_keys.shape[0], dtype=signature.residuals.dtype, device=unique_keys.device)
        vector[ids] = signature.residuals
        present.append(mask)
        values.append(vector)
    common = torch.stack(present).all(dim=0)
    return unique_keys[common], torch.stack(values, dim=0)[:, common]


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _fragment_entry(manifest, scene, fragment_id):
    normalized = str(fragment_id).replace("cloud_bin_", "")
    for entry in manifest:
        candidate = str(entry["fragment_id"])
        if entry["scene"] == scene and candidate in {str(fragment_id), normalized, f"cloud_bin_{normalized}"}:
            return entry
    raise KeyError(f"Missing {scene}/{fragment_id} in fragment manifest.")


def _read_pose(path):
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose at {path}, got {pose.shape}.")
    return pose


def _stable_search_split(frame_id, search_fraction):
    digest = hashlib.sha256(str(int(frame_id)).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    return SEARCH_SPLIT if value < float(search_fraction) else VALIDATION_SPLIT


def _load_depth(path, depth_scale):
    depth = np.asarray(o3d.io.read_image(path)).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth / float(depth_scale)


def build_ray_bundle(
    rgbd_root,
    scene,
    fragment_id,
    manifest_path,
    stride=8,
    max_frames=0,
    search_fraction=0.7,
    depth_scale=1000.0,
    min_depth=0.1,
    max_depth=8.0,
    device="cpu",
):
    if not manifest_path:
        raise ValueError("RayBundle requires a fragment manifest with sequence and frame ranges.")
    entry = _fragment_entry(_load_manifest(manifest_path), scene, fragment_id)
    frame_ids = list(range(int(entry["frame_start"]), int(entry["frame_end"]) + 1))
    if max_frames > 0:
        frame_ids = frame_ids[:int(max_frames)]
    if not frame_ids:
        raise ValueError(f"No frames selected for {scene}/{fragment_id}.")
    intrinsics_path = os.path.join(rgbd_root, scene, "camera-intrinsics.txt")
    intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)[:3, :3]
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    frame_poses, depth_maps, sampled_frame_ids, origins, directions, observed, valid, pixels, camera_poses, confidences, split_ids = [], [], [], [], [], [], [], [], [], [], []
    sequence = entry["sequence"]
    for frame_id in frame_ids:
        stem = f"frame-{frame_id:06d}"
        root = os.path.join(rgbd_root, scene, sequence)
        depth_path = os.path.join(root, f"{stem}.depth.png")
        pose_path = os.path.join(root, f"{stem}.pose.txt")
        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing RGB-D frame or pose for {scene}/{sequence}/{stem}.")
        depth = _load_depth(depth_path, depth_scale)
        pose = _read_pose(pose_path)
        frame_poses.append(pose)
        depth_maps.append(depth)
        ys, xs = np.mgrid[0:depth.shape[0]:max(1, int(stride)), 0:depth.shape[1]:max(1, int(stride))]
        z = depth[ys, xs].reshape(-1)
        sample_valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
        xs = xs.reshape(-1).astype(np.float32)
        ys = ys.reshape(-1).astype(np.float32)
        z = z.astype(np.float32)
        cam_dirs = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones_like(z)], axis=1)
        cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True) + 1e-8
        directions.append((cam_dirs @ pose[:3, :3].T).astype(np.float32))
        origins.append(np.repeat(pose[:3, 3][None], z.shape[0], axis=0).astype(np.float32))
        observed.append(z)
        valid.append(sample_valid)
        pixels.append(np.stack([xs, ys], axis=1))
        sampled_frame_ids.append(np.full(z.shape[0], frame_id, dtype=np.int64))
        camera_poses.append(np.repeat(pose[None], z.shape[0], axis=0))
        confidences.append(sample_valid.astype(np.float32))
        split_ids.append(np.full(z.shape[0], _stable_search_split(frame_id, search_fraction), dtype=np.int64))
    shapes = {depth.shape for depth in depth_maps}
    if len(shapes) != 1:
        raise ValueError(f"Depth-map shapes differ inside {scene}/{fragment_id}; RayBundle requires a common camera resolution.")
    frame_count = len(frame_ids)
    height, width = depth_maps[0].shape
    frame_ids_tensor = torch.from_numpy(np.concatenate(sampled_frame_ids)).to(device=device, dtype=torch.long)
    frame_pose_tensor = torch.from_numpy(np.stack(frame_poses)).to(device=device, dtype=torch.float32)
    split_per_frame = torch.tensor([_stable_search_split(frame_id, search_fraction) for frame_id in frame_ids], dtype=torch.long, device=device)
    fragment_pose = entry.get("fragment_pose")
    if isinstance(fragment_pose, str):
        fragment_pose = np.asarray(json.loads(fragment_pose), dtype=np.float32)
    if fragment_pose is None:
        fragment_pose = np.eye(4, dtype=np.float32)
    fragment_pose = np.asarray(fragment_pose, dtype=np.float32)
    if fragment_pose.shape != (4, 4):
        raise ValueError(f"Invalid fragment pose for {scene}/{fragment_id}.")
    return RayBundle(
        frame_ids=frame_ids_tensor,
        origins=torch.from_numpy(np.concatenate(origins, axis=0)).to(device=device, dtype=torch.float32),
        directions=torch.from_numpy(np.concatenate(directions, axis=0)).to(device=device, dtype=torch.float32),
        observed_depths=torch.from_numpy(np.concatenate(observed, axis=0)).to(device=device, dtype=torch.float32),
        valid_depth=torch.from_numpy(np.concatenate(valid, axis=0)).to(device=device, dtype=torch.bool),
        pixels=torch.from_numpy(np.concatenate(pixels, axis=0)).to(device=device, dtype=torch.float32),
        camera_poses=torch.from_numpy(np.concatenate(camera_poses, axis=0)).to(device=device, dtype=torch.float32),
        intrinsics=torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32),
        confidences=torch.from_numpy(np.concatenate(confidences, axis=0)).to(device=device, dtype=torch.float32),
        split_ids=split_per_frame,
        depth_maps=torch.from_numpy(np.stack(depth_maps).reshape(frame_count, height, width)).to(device=device, dtype=torch.float32),
        fragment_pose=torch.from_numpy(fragment_pose).to(device=device, dtype=torch.float32),
        frame_numbers=torch.tensor(frame_ids, dtype=torch.long, device=device),
        frame_camera_poses=frame_pose_tensor,
        frame_split_ids=split_per_frame,
    )


def _as_points(points):
    return points[0] if points.ndim == 3 else points


def _as_pose(pose):
    return pose[0] if pose.ndim == 3 else pose


def transform_points(points, pose):
    return points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]


def evaluate_projected_points(points_world, bundle, surface_mu=0.05, surface_sigma=0.03, split_id=None, direction_code=0):
    points_world = _as_points(points_world)
    frame_indices = bundle.frame_indices(split_id)
    device = points_world.device
    empty = torch.empty(0, device=device, dtype=points_world.dtype)
    if frame_indices.numel() == 0 or points_world.numel() == 0:
        return RayEvaluation(0.0, 0.0, 0, torch.empty((0, 4), device=device), empty, torch.empty((0, 3), dtype=torch.long, device=device), empty.long(), empty.long(), int(points_world.shape[0]))
    fx, fy, cx, cy = bundle.intrinsics[0, 0], bundle.intrinsics[1, 1], bundle.intrinsics[0, 2], bundle.intrinsics[1, 2]
    height, width = bundle.depth_maps.shape[-2:]
    per_frame, residuals, ray_keys, point_ids, observation_frames = [], [], [], [], []
    for frame_index in frame_indices.tolist():
        world_to_camera = torch.linalg.inv(bundle.projection_poses[frame_index])
        points_camera = transform_points(points_world, world_to_camera)
        z = points_camera[:, 2]
        projected_u = torch.round(fx * points_camera[:, 0] / torch.clamp_min(z, 1e-8) + cx).long()
        projected_v = torch.round(fy * points_camera[:, 1] / torch.clamp_min(z, 1e-8) + cy).long()
        in_view = (z > 0) & (projected_u >= 0) & (projected_u < width) & (projected_v >= 0) & (projected_v < height)
        if not in_view.any():
            per_frame.append(torch.tensor([float(frame_index), 0.0, 0.0, 0.0], device=device))
            continue
        point_index = torch.where(in_view)[0]
        observed = bundle.depth_maps[frame_index, projected_v[point_index], projected_u[point_index]]
        valid = torch.isfinite(observed) & (observed > 0)
        point_index, observed = point_index[valid], observed[valid]
        if point_index.numel() == 0:
            per_frame.append(torch.tensor([float(frame_index), 0.0, 0.0, 0.0], device=device))
            continue
        predicted_depth = z[point_index]
        informative = predicted_depth <= observed + surface_mu
        point_index, observed, predicted_depth = point_index[informative], observed[informative], predicted_depth[informative]
        if point_index.numel() == 0:
            per_frame.append(torch.tensor([float(frame_index), 0.0, 0.0, 0.0], device=device))
            continue
        violation = torch.relu(observed - surface_mu - predicted_depth)
        support = torch.exp(-torch.abs(predicted_depth - observed) / max(float(surface_sigma), 1e-8))
        per_frame.append(torch.stack([
            torch.tensor(float(frame_index), device=device),
            violation.mean(),
            support.mean(),
            torch.tensor(float(point_index.numel()), device=device),
        ]))
        residuals.append(violation)
        frame_number = int(bundle.frame_numbers[frame_index]) if bundle.frame_numbers is not None else frame_index
        pixel_ids = projected_v[point_index] * width + projected_u[point_index]
        ray_keys.append(torch.stack([
            torch.full_like(pixel_ids, int(direction_code)),
            torch.full_like(pixel_ids, frame_number),
            pixel_ids,
        ], dim=1))
        point_ids.append(point_index)
        observation_frames.append(torch.full_like(point_index, frame_index))
    scores = torch.stack(per_frame) if per_frame else torch.empty((0, 4), device=device)
    count = int(scores[:, 3].sum().item()) if scores.numel() else 0
    observed_frames = scores[:, 3] > 0 if scores.numel() else torch.zeros(0, dtype=torch.bool, device=device)
    free = float(scores[observed_frames, 1].median().item()) if observed_frames.any() else 0.0
    surface = float(scores[observed_frames, 2].median().item()) if observed_frames.any() else 0.0
    return RayEvaluation(
        free_violation=free,
        surface_support=surface,
        valid_observation_count=count,
        per_frame_scores=scores,
        per_ray_residuals=torch.cat(residuals) if residuals else empty,
        ray_keys=torch.cat(ray_keys) if ray_keys else torch.empty((0, 3), dtype=torch.long, device=device),
        point_ids=torch.cat(point_ids) if point_ids else empty.long(),
        frame_indices=torch.cat(observation_frames) if observation_frames else empty.long(),
        evaluated_point_count=int(points_world.shape[0]),
    )


def bidirectional_ray_evaluation(src_points, tgt_points, pose, source_rays, target_rays, surface_mu=0.05, surface_sigma=0.03, split_id=None):
    pose = _as_pose(pose)
    src_in_target = transform_points(_as_points(src_points), pose)
    tgt_in_source = transform_points(_as_points(tgt_points), torch.linalg.inv(pose))
    src_world = transform_points(src_in_target, target_rays.fragment_pose)
    tgt_world = transform_points(tgt_in_source, source_rays.fragment_pose)
    target_eval = evaluate_projected_points(src_world, target_rays, surface_mu, surface_sigma, split_id, direction_code=0)
    source_eval = evaluate_projected_points(tgt_world, source_rays, surface_mu, surface_sigma, split_id, direction_code=1)
    total = target_eval.valid_observation_count + source_eval.valid_observation_count
    if total == 0:
        free, support = 0.0, 0.0
    else:
        free = (target_eval.free_violation * target_eval.valid_observation_count + source_eval.free_violation * source_eval.valid_observation_count) / total
        support = (target_eval.surface_support * target_eval.valid_observation_count + source_eval.surface_support * source_eval.valid_observation_count) / total
    residuals = torch.cat([target_eval.per_ray_residuals, source_eval.per_ray_residuals])
    ray_keys = torch.cat([target_eval.ray_keys, source_eval.ray_keys])
    point_ids = torch.cat([target_eval.point_ids, source_eval.point_ids])
    frame_indices = torch.cat([target_eval.frame_indices, source_eval.frame_indices])
    consistency = 1.0 - abs(target_eval.free_violation - source_eval.free_violation) / (target_eval.free_violation + source_eval.free_violation + 1e-8)
    possible = (
        target_eval.evaluated_point_count * target_eval.per_frame_scores.shape[0]
        + source_eval.evaluated_point_count * source_eval.per_frame_scores.shape[0]
    )
    return {
        "free_violation": float(free),
        "surface_support": float(support),
        "valid_observation_count": int(total),
        "target": target_eval,
        "source": source_eval,
        "per_ray_residuals": residuals,
        "ray_keys": ray_keys,
        "point_ids": point_ids,
        "frame_indices": frame_indices,
        "valid_observation_ratio": float(total / max(1, possible)),
        "bidirectional_consistency": float(consistency),
    }


def _transform_pose_batch(points, poses):
    if points.ndim == 2:
        return torch.einsum("nj,kij->kni", points, poses[:, :3, :3].transpose(1, 2)) + poses[:, None, :3, 3]
    return torch.einsum("knj,kij->kni", points, poses[:, :3, :3].transpose(1, 2)) + poses[:, None, :3, 3]


def _evaluate_direction_batch(points, poses, fragment_pose, bundle, surface_mu, surface_sigma, split_id):
    poses = poses if poses.ndim == 3 else poses[None]
    if points.ndim == 2:
        points = points[None].expand(poses.shape[0], -1, -1)
    point_count = points.shape[1]
    world = _transform_pose_batch(points, poses)
    world = _transform_pose_batch(world.reshape(-1, 3), fragment_pose[None]).reshape(poses.shape[0], point_count, 3)
    frame_indices = bundle.frame_indices(split_id)
    count = torch.zeros(poses.shape[0], device=poses.device, dtype=poses.dtype)
    free_scores, surface_scores, frame_counts = [], [], []
    fx, fy, cx, cy = bundle.intrinsics[0, 0], bundle.intrinsics[1, 1], bundle.intrinsics[0, 2], bundle.intrinsics[1, 2]
    height, width = bundle.depth_maps.shape[-2:]
    for frame_index in frame_indices.tolist():
        world_to_camera = torch.linalg.inv(bundle.projection_poses[frame_index])
        camera = _transform_pose_batch(world.reshape(-1, 3), world_to_camera[None]).reshape_as(world)
        z = camera[..., 2]
        u = torch.round(fx * camera[..., 0] / torch.clamp_min(z, 1e-8) + cx).long()
        v = torch.round(fy * camera[..., 1] / torch.clamp_min(z, 1e-8) + cy).long()
        in_view = (z > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v = u.clamp(0, width - 1), v.clamp(0, height - 1)
        observed = bundle.depth_maps[frame_index].reshape(-1)[(v * width + u).reshape(-1)].reshape_as(z)
        informative = in_view & torch.isfinite(observed) & (observed > 0) & (z <= observed + surface_mu)
        denominator = informative.sum(dim=1).to(poses.dtype)
        violation = torch.relu(observed - surface_mu - z) * informative
        support = torch.exp(-torch.abs(z - observed) / max(float(surface_sigma), 1e-8)) * informative
        free_scores.append(violation.sum(dim=1) / torch.clamp_min(denominator, 1.0))
        surface_scores.append(support.sum(dim=1) / torch.clamp_min(denominator, 1.0))
        frame_counts.append(denominator)
        count += denominator
    if not free_scores:
        zeros = torch.zeros(poses.shape[0], device=poses.device, dtype=poses.dtype)
        return zeros, zeros, zeros
    free_stack, surface_stack = torch.stack(free_scores, dim=1), torch.stack(surface_scores, dim=1)
    per_frame_valid = torch.stack(frame_counts, dim=1) > 0
    free = torch.nanmedian(torch.where(per_frame_valid, free_stack, torch.full_like(free_stack, float("nan"))), dim=1).values.nan_to_num(0.0)
    surface = torch.nanmedian(torch.where(per_frame_valid, surface_stack, torch.full_like(surface_stack, float("nan"))), dim=1).values.nan_to_num(0.0)
    return free, surface, count


def evaluate_pose_batch(poses, src_points, tgt_points, source_rays, target_rays, surface_mu=0.05, surface_sigma=0.03, split_id=None):
    poses = poses if poses.ndim == 3 else poses[None]
    src_in_target = _transform_pose_batch(_as_points(src_points), poses)
    inverse = torch.linalg.inv(poses)
    tgt_in_source = _transform_pose_batch(_as_points(tgt_points), inverse)
    target_free, target_surface, target_count = _evaluate_direction_batch(
        src_in_target, torch.eye(4, device=poses.device, dtype=poses.dtype)[None].expand(poses.shape[0], -1, -1), target_rays.fragment_pose, target_rays, surface_mu, surface_sigma, split_id,
    )
    source_free, source_surface, source_count = _evaluate_direction_batch(
        tgt_in_source, torch.eye(4, device=poses.device, dtype=poses.dtype)[None].expand(poses.shape[0], -1, -1), source_rays.fragment_pose, source_rays, surface_mu, surface_sigma, split_id,
    )
    count = target_count + source_count
    free = (target_free * target_count + source_free * source_count) / torch.clamp_min(count, 1.0)
    surface = (target_surface * target_count + source_surface * source_count) / torch.clamp_min(count, 1.0)
    return {"free_violation": free, "surface_support": surface, "valid_observation_count": count}


def ray_signature(evaluation):
    if isinstance(evaluation, dict):
        return _canonicalize_residuals(evaluation["ray_keys"], evaluation["per_ray_residuals"])
    return _canonicalize_residuals(evaluation.ray_keys, evaluation.per_ray_residuals)


def detach_ray_signature(signature):
    return RayResidualSignature(
        signature.keys.detach().clone(),
        signature.residuals.detach().clone(),
        signature.observation_indices.detach().clone(),
    )
