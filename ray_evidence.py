import json
import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

from utils.SE3 import transform


@dataclass
class RayBundle:
    frame_ids: torch.Tensor
    origins: torch.Tensor
    directions: torch.Tensor
    observed_depth: torch.Tensor
    pixels: torch.Tensor
    camera_poses: torch.Tensor
    confidence: torch.Tensor
    split: torch.Tensor
    intrinsics: torch.Tensor

    def to(self, device):
        return RayBundle(
            self.frame_ids.to(device), self.origins.to(device), self.directions.to(device),
            self.observed_depth.to(device), self.pixels.to(device), self.camera_poses.to(device),
            self.confidence.to(device), self.split.to(device), self.intrinsics.to(device),
        )


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fragment_entry(manifest, scene, fragment_id):
    for entry in manifest:
        if entry["scene"] == scene and entry["fragment_id"] == fragment_id:
            return entry
    raise KeyError(f"Missing {scene}/{fragment_id} in fragment manifest.")


def _read_pose(path):
    pose = np.loadtxt(path).astype(np.float32)
    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose at {path}, got {pose.shape}.")
    return pose


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
    manifest = _load_manifest(manifest_path)
    entry = _fragment_entry(manifest, scene, fragment_id)
    sequence = entry["sequence"]
    start = int(entry["frame_start"])
    end = int(entry["frame_end"])
    frame_ids = list(range(start, end + 1))
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    intrinsics_path = os.path.join(rgbd_root, scene, "camera-intrinsics.txt")
    intrinsics = np.loadtxt(intrinsics_path).astype(np.float32)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    raw_scene = os.path.join(rgbd_root, scene)
    ray_frames, ray_origins, ray_directions, ray_depths, ray_pixels, ray_poses, ray_conf, ray_split = [], [], [], [], [], [], [], []
    for frame_id in frame_ids:
        frame = f"frame-{frame_id:06d}"
        depth_path = os.path.join(raw_scene, sequence, f"{frame}.depth.png")
        pose_path = os.path.join(raw_scene, sequence, f"{frame}.pose.txt")
        if not os.path.exists(depth_path) or not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing RGB-D frame or pose for {scene}/{sequence}/{frame}.")
        depth = np.asarray(o3d.io.read_image(depth_path)).astype(np.float32) / depth_scale
        pose = _read_pose(pose_path)
        ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[ys, xs].reshape(-1)
        valid = (z >= min_depth) & (z <= max_depth)
        if not np.any(valid):
            continue
        xs_flat = xs.reshape(-1)[valid].astype(np.float32)
        ys_flat = ys.reshape(-1)[valid].astype(np.float32)
        z = z[valid].astype(np.float32)
        cam_dirs = np.stack([(xs_flat - cx) / fx, (ys_flat - cy) / fy, np.ones_like(z)], axis=1)
        cam_dirs /= np.linalg.norm(cam_dirs, axis=1, keepdims=True) + 1e-8
        world_dirs = cam_dirs @ pose[:3, :3].T
        origins = np.repeat(pose[:3, 3][None], z.shape[0], axis=0)
        poses = np.repeat(pose[None], z.shape[0], axis=0)
        split_value = int((frame_id % 100) < round(search_fraction * 100.0))
        ray_frames.append(np.full(z.shape[0], frame_id, dtype=np.int64))
        ray_origins.append(origins.astype(np.float32))
        ray_directions.append(world_dirs.astype(np.float32))
        ray_depths.append(z)
        ray_pixels.append(np.stack([xs_flat, ys_flat], axis=1))
        ray_poses.append(poses)
        ray_conf.append(np.ones(z.shape[0], dtype=np.float32))
        ray_split.append(np.full(z.shape[0], split_value, dtype=np.int64))
    if not ray_frames:
        raise ValueError(f"No valid rays for {scene}/{fragment_id}.")
    make = lambda values, dtype=None: torch.from_numpy(np.concatenate(values, axis=0)).to(device=device, dtype=dtype)
    return RayBundle(
        frame_ids=make(ray_frames, torch.long),
        origins=make(ray_origins, torch.float32),
        directions=make(ray_directions, torch.float32),
        observed_depth=make(ray_depths, torch.float32),
        pixels=make(ray_pixels, torch.float32),
        camera_poses=make(ray_poses, torch.float32),
        confidence=make(ray_conf, torch.float32),
        split=make(ray_split, torch.long),
        intrinsics=torch.from_numpy(intrinsics).to(device=device, dtype=torch.float32),
    )


def ray_energy(points, pose, bundle, surface_mu=0.05, surface_sigma=0.03, ray_ids=None):
    if points.ndim == 3:
        points = points[0]
    if pose.ndim == 2:
        pose = pose[None]
    if ray_ids is None:
        ray_ids = torch.arange(bundle.frame_ids.shape[0], device=points.device)
    if ray_ids.numel() == 0:
        return 0.0, 0.0, torch.zeros(0, device=points.device)
    points_world = transform(points[None], pose)[0]
    selected_frames = torch.unique(bundle.frame_ids[ray_ids])
    frame_violations = []
    frame_support = []
    for frame_id in selected_frames.tolist():
        frame_mask = bundle.frame_ids[ray_ids] == frame_id
        ids = ray_ids[frame_mask]
        ray_origin = bundle.origins[ids[0]]
        ray_direction = bundle.directions[ids]
        ray_depth = bundle.observed_depth[ids]
        point_vectors = points_world - ray_origin.view(1, 3)
        point_depth = torch.linalg.norm(point_vectors, dim=1)
        point_direction = point_vectors / (point_depth[:, None] + 1e-8)
        cosine = ray_direction @ point_direction.transpose(0, 1)
        best_cosine, best_point = cosine.max(dim=1)
        predicted_depth = point_depth[best_point]
        valid = best_cosine > 0.999
        violation = torch.relu(ray_depth - surface_mu - predicted_depth) * valid.float()
        support = torch.exp(-torch.abs(predicted_depth - ray_depth) / max(surface_sigma, 1e-6)) * valid.float()
        frame_violations.append((violation * bundle.confidence[ids]).sum() / (bundle.confidence[ids].sum() + 1e-6))
        frame_support.append((support * bundle.confidence[ids]).sum() / (bundle.confidence[ids].sum() + 1e-6))
    violations = torch.stack(frame_violations)
    supports = torch.stack(frame_support)
    return float(violations.median().item()), float(supports.median().item()), violations


def bidirectional_ray_energy(src_points, tgt_points, pose, source_rays, target_rays, surface_mu=0.05, surface_sigma=0.03, split=None):
    source_ids = None if split is None else torch.where(source_rays.split == split)[0]
    target_ids = None if split is None else torch.where(target_rays.split == split)[0]
    target_free, target_surface, target_violations = ray_energy(src_points, pose, target_rays, surface_mu, surface_sigma, target_ids)
    inverse_pose = torch.linalg.inv(pose)
    source_free, source_surface, source_violations = ray_energy(tgt_points, inverse_pose, source_rays, surface_mu, surface_sigma, source_ids)
    return {
        "free": 0.5 * (target_free + source_free),
        "surface": 0.5 * (target_surface + source_surface),
        "target_free": target_free,
        "source_free": source_free,
        "target_violations": target_violations,
        "source_violations": source_violations,
    }


def ray_signature(energy, threshold=0.01):
    values = torch.cat([energy["target_violations"], energy["source_violations"]], dim=0)
    return (values > threshold).to(torch.float32)
