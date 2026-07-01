import numpy as np
import open3d as o3d


def _to_numpy(points):
    if hasattr(points, "detach"):
        points = points.detach().cpu().numpy()
    points = np.asarray(points)
    if points.ndim == 3:
        points = points[0]
    return points


def make_point_cloud(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_to_numpy(points))
    if color is not None:
        colors = np.tile(np.asarray(color, dtype=float), (len(pcd.points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def draw_registration_corr2(src, tgt, src_corr, tgt_corr, trans=None, threshold=0.1):
    src_pcd = make_point_cloud(src, [1.0, 0.0, 0.0])
    tgt_pcd = make_point_cloud(tgt, [0.0, 0.6, 1.0])
    if trans is not None:
        src_pcd.transform(np.asarray(trans))
    o3d.visualization.draw_geometries([src_pcd, tgt_pcd])
