import json
import os
import numpy as np
import torch
import open3d as o3d


class FreeSpaceVolume:
    def __init__(self, free_voxels, origin, voxel_size):
        self.free_voxels = free_voxels
        self.origin = origin
        self.voxel_size = float(voxel_size)

    @classmethod
    def from_npz(cls, path, device):
        data = np.load(path)
        free_voxels = torch.from_numpy(data['free_voxels'].astype(np.int64)).to(device)
        origin = torch.from_numpy(data['origin'].astype(np.float32)).to(device)
        voxel_size = float(data['voxel_size'])
        return cls(free_voxels, origin, voxel_size)

    @classmethod
    def from_numpy(cls, free_points, voxel_size, device):
        if free_points.shape[0] == 0:
            origin = np.zeros(3, dtype=np.float32)
            free_voxels = np.zeros((0, 3), dtype=np.int64)
        else:
            origin = free_points.min(axis=0).astype(np.float32)
            free_voxels = np.unique(np.floor((free_points - origin[None]) / voxel_size).astype(np.int64), axis=0)
        return cls(torch.from_numpy(free_voxels).to(device), torch.from_numpy(origin).to(device), voxel_size)

    def contains(self, points):
        voxel = torch.floor((points - self.origin.view(1, 3)) / self.voxel_size).to(torch.int64)
        if self.free_voxels.numel() == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        voxel_keys = (voxel[:, 0] * 73856093) ^ (voxel[:, 1] * 19349663) ^ (voxel[:, 2] * 83492791)
        free_keys = (self.free_voxels[:, 0] * 73856093) ^ (self.free_voxels[:, 1] * 19349663) ^ (self.free_voxels[:, 2] * 83492791)
        return torch.isin(voxel_keys, free_keys)


def _read_matrix(path):
    mat = np.loadtxt(path).astype(np.float32)
    if mat.shape == (4, 4):
        return mat
    raise ValueError(f"Expected 4x4 matrix in {path}, got {mat.shape}.")


def _find_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _load_intrinsics(scene_root):
    json_path = _find_existing([
        os.path.join(scene_root, 'intrinsics.json'),
        os.path.join(scene_root, 'camera-intrinsics.json'),
    ])
    if json_path:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return float(data['fx']), float(data['fy']), float(data['cx']), float(data['cy'])
    txt_path = _find_existing([
        os.path.join(scene_root, 'camera-intrinsics.txt'),
        os.path.join(scene_root, 'intrinsics.txt'),
        os.path.join(scene_root, 'intrinsic', 'intrinsic_depth.txt'),
    ])
    if not txt_path:
        raise FileNotFoundError(f"Missing camera intrinsics under {scene_root}.")
    mat = np.loadtxt(txt_path).astype(np.float32)
    if mat.shape == (3, 3):
        return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2])
    if mat.shape == (4, 4):
        return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2])
    if mat.size >= 4:
        flat = mat.reshape(-1)
        return float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
    raise ValueError(f"Unsupported intrinsics format in {txt_path}.")


def _load_frame_ids(scene_root, fragment_id):
    explicit = _find_existing([
        os.path.join(scene_root, 'fragments', f'cloud_bin_{fragment_id}_frames.txt'),
        os.path.join(scene_root, f'cloud_bin_{fragment_id}_frames.txt'),
    ])
    if explicit:
        with open(explicit, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    manifest = _find_existing([
        os.path.join(scene_root, 'fragment_ranges.json'),
        os.path.join(scene_root, 'fragments.json'),
    ])
    if not manifest:
        raise FileNotFoundError(f"Missing frame manifest for cloud_bin_{fragment_id} under {scene_root}.")
    with open(manifest, 'r') as f:
        data = json.load(f)
    key = f'cloud_bin_{fragment_id}'
    if key not in data:
        raise KeyError(f"Missing {key} in {manifest}.")
    value = data[key]
    if isinstance(value, dict) and 'frames' in value:
        return [str(v) for v in value['frames']]
    if isinstance(value, dict) and 'start' in value and 'end' in value:
        return [str(v) for v in range(int(value['start']), int(value['end']) + 1)]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ValueError(f"Unsupported frame manifest entry for {key}.")


def _frame_path(scene_root, frame_id, kind):
    names = [
        f'{frame_id}.png',
        f'{int(frame_id):06d}.png' if str(frame_id).isdigit() else f'{frame_id}.png',
        f'{frame_id}.jpg',
        f'{int(frame_id):06d}.jpg' if str(frame_id).isdigit() else f'{frame_id}.jpg',
    ]
    dirs = {
        'depth': ['depth', 'depths', 'sensor_depth'],
        'pose': ['pose', 'poses', 'camera_pose'],
    }[kind]
    for directory in dirs:
        for name in names:
            path = os.path.join(scene_root, directory, name if kind == 'depth' else os.path.splitext(name)[0] + '.txt')
            if os.path.exists(path):
                return path
    return None


def build_free_space_from_rgbd(rgbd_root, scene, fragment_id, voxel_size, depth_scale, trunc_margin, stride, max_frames, device):
    scene_root = os.path.join(rgbd_root, scene)
    if not os.path.isdir(scene_root):
        raise FileNotFoundError(f"Missing RGB-D scene directory: {scene_root}")
    fx, fy, cx, cy = _load_intrinsics(scene_root)
    frame_ids = _load_frame_ids(scene_root, fragment_id)
    if max_frames > 0:
        frame_ids = frame_ids[:max_frames]
    free_points = []
    for frame_id in frame_ids:
        depth_path = _frame_path(scene_root, frame_id, 'depth')
        pose_path = _frame_path(scene_root, frame_id, 'pose')
        if depth_path is None or pose_path is None:
            raise FileNotFoundError(f"Missing depth or pose for frame {frame_id} in {scene_root}.")
        depth = np.asarray(o3d.io.read_image(depth_path)).astype(np.float32) / depth_scale
        pose = _read_matrix(pose_path)
        ys, xs = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
        z = depth[ys, xs].reshape(-1)
        valid = z > trunc_margin
        xs = xs.reshape(-1)[valid].astype(np.float32)
        ys = ys.reshape(-1)[valid].astype(np.float32)
        z = z[valid]
        if z.shape[0] == 0:
            continue
        samples_per_ray = np.maximum(np.floor((z - trunc_margin) / voxel_size).astype(np.int32), 0)
        keep = samples_per_ray > 0
        xs, ys, z, samples_per_ray = xs[keep], ys[keep], z[keep], samples_per_ray[keep]
        for x, y, depth_value, ray_count in zip(xs, ys, z, samples_per_ray):
            ray_depth = np.linspace(voxel_size, max(voxel_size, depth_value - trunc_margin), int(ray_count), dtype=np.float32)
            cam = np.stack([(x - cx) * ray_depth / fx, (y - cy) * ray_depth / fy, ray_depth, np.ones_like(ray_depth)], axis=0)
            world = (pose @ cam)[:3].T
            free_points.append(world)
    if not free_points:
        raise ValueError(f"No free-space samples built for {scene}/cloud_bin_{fragment_id}.")
    return FreeSpaceVolume.from_numpy(np.concatenate(free_points, axis=0), voxel_size, device)


def load_target_free_space(free_space_root, rgbd_root, scene, target_id, device, voxel_size, depth_scale, trunc_margin, stride, max_frames, require=False):
    if free_space_root:
        path = os.path.join(free_space_root, scene, f"cloud_bin_{target_id}_free_space.npz")
        if os.path.exists(path):
            return FreeSpaceVolume.from_npz(path, device)
    if rgbd_root:
        return build_free_space_from_rgbd(rgbd_root, scene, target_id, voxel_size, depth_scale, trunc_margin, stride, max_frames, device)
    if require:
        raise FileNotFoundError("use_rgbd_fsv=True requires free_space_root or rgbd_root.")
    return None


def compute_rgbd_fsv(src_points, trans, target_free_space):
    if target_free_space is None:
        raise FileNotFoundError("Missing target free-space volume for use_rgbd_fsv=True.")
    warped_src = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3], src_points.permute(0, 2, 1)) + trans[:, :3, 3:4]
    warped_src = warped_src.permute(0, 2, 1)[0]
    violation = target_free_space.contains(warped_src)
    return float(violation.float().mean().item())
