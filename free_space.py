import os
import numpy as np
import torch


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

    def contains(self, points):
        voxel = torch.floor((points - self.origin.view(1, 3)) / self.voxel_size).to(torch.int64)
        if self.free_voxels.numel() == 0:
            return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        voxel_keys = voxel[:, 0] * 73856093 ^ voxel[:, 1] * 19349663 ^ voxel[:, 2] * 83492791
        free_keys = self.free_voxels[:, 0] * 73856093 ^ self.free_voxels[:, 1] * 19349663 ^ self.free_voxels[:, 2] * 83492791
        return torch.isin(voxel_keys, free_keys)


def load_target_free_space(free_space_root, scene, target_id, device):
    if not free_space_root:
        return None
    path = os.path.join(free_space_root, scene, f"cloud_bin_{target_id}_free_space.npz")
    if not os.path.exists(path):
        return None
    return FreeSpaceVolume.from_npz(path, device)


def compute_rgbd_fsv(src_points, trans, target_free_space):
    if target_free_space is None:
        raise FileNotFoundError("Missing target free-space volume for use_rgbd_fsv=True.")
    warped_src = torch.einsum('bnm,bmk->bnk', trans[:, :3, :3], src_points.permute(0, 2, 1)) + trans[:, :3, 3:4]
    warped_src = warped_src.permute(0, 2, 1)[0]
    violation = target_free_space.contains(warped_src)
    return float(violation.float().mean().item())
