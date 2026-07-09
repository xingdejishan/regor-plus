import csv
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import open3d as o3d


SCENE = "7-scenes-redkitchen"
REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path, rows, fieldnames):
    ensure_dir(Path(path).parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fragment_id(value):
    text = str(value).replace("\\", "/")
    match = re.search(r"cloud_bin_(\d+)", text)
    if not match:
        raise ValueError(f"Cannot parse fragment id from {value}")
    return int(match.group(1))


def fragment_name(fid):
    return f"cloud_bin_{int(fid)}"


def pair_scene(value):
    parts = str(value).replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "test":
        return parts[1]
    if SCENE in parts:
        return SCENE
    return ""


def matrix_to_cell(mat):
    return json.dumps(np.asarray(mat, dtype=float).reshape(4, 4).tolist(), separators=(",", ":"))


def cell_to_matrix(value):
    text = str(value).strip()
    if text.startswith("["):
        return np.asarray(json.loads(text), dtype=np.float64).reshape(4, 4)
    return np.fromstring(text, sep=" ", dtype=np.float64).reshape(4, 4)


def parse_pose_file(path):
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        raise ValueError(f"Empty fragment pose file: {path}")
    meta = lines[0].strip().split()
    if len(meta) < 4:
        raise ValueError(f"Invalid fragment pose metadata in {path}: {lines[0]}")
    rows = []
    for line in lines[1:]:
        values = np.fromstring(line, sep=" ", dtype=np.float64)
        if values.size:
            rows.append(values)
    mat = np.vstack(rows)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 fragment pose in {path}, got {mat.shape}")
    return {
        "scene": meta[0],
        "sequence": meta[1],
        "frame_start": int(meta[2]),
        "frame_end": int(meta[3]),
        "matrix": mat,
    }


def load_matrix(path):
    mat = np.loadtxt(path, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix in {path}, got {mat.shape}")
    return mat


def invert_transform(mat):
    return np.linalg.inv(np.asarray(mat, dtype=np.float64))


def transform_points(points, trans):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return points.reshape(0, 3)
    homo = np.ones((points.shape[0], 4), dtype=np.float64)
    homo[:, :3] = points
    return (np.asarray(trans, dtype=np.float64) @ homo.T).T[:, :3]


def find_intrinsics(scene_root, recursive=True):
    scene_root = Path(scene_root)
    names = {
        "camera-intrinsics.txt",
        "camera-intrinsic.txt",
        "camera-intrinsics.json",
        "intrinsics.txt",
        "intrinsic_depth.txt",
    }
    direct = [
        scene_root / "camera-intrinsics.txt",
        scene_root / "camera-intrinsic.txt",
        scene_root / "camera-intrinsics.json",
        scene_root / "intrinsics.txt",
        scene_root / "intrinsic" / "intrinsic_depth.txt",
    ]
    for path in direct:
        if path.exists():
            return path
    if recursive and scene_root.exists():
        for path in scene_root.rglob("*"):
            if path.is_file() and path.name in names:
                return path
    return None


def load_intrinsics(scene_root):
    path = find_intrinsics(scene_root)
    if path is None:
        raise FileNotFoundError(f"Missing camera intrinsics under {scene_root}")
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if "camera_matrix" in data:
            mat = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
            return mat, path
        return np.asarray(
            [
                [float(data["fx"]), 0.0, float(data["cx"])],
                [0.0, float(data["fy"]), float(data["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ), path
    mat = np.loadtxt(path, dtype=np.float64)
    flat = mat.reshape(-1)
    if mat.shape in ((3, 3), (4, 4)):
        return mat[:3, :3], path
    if flat.size >= 4:
        return np.asarray(
            [[flat[0], 0.0, flat[2]], [0.0, flat[1], flat[3]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ), path
    raise ValueError(f"Unsupported intrinsics format in {path}")


def has_raw_scene_layout(path):
    path = Path(path)
    if not path.is_dir():
        return False
    seq_dirs = [p for p in path.iterdir() if p.is_dir() and p.name.lower().startswith("seq")]
    return bool(seq_dirs) and find_intrinsics(path, recursive=False) is not None


def locate_raw_scene_root(raw_root, scene=SCENE):
    raw_root = Path(raw_root)
    candidates = [raw_root / scene, raw_root]
    for cand in candidates:
        if has_raw_scene_layout(cand):
            return cand.resolve()
    if raw_root.exists():
        for dirpath, dirnames, _ in os.walk(raw_root):
            path = Path(dirpath)
            if path.name.lower() == scene.lower() and has_raw_scene_layout(path):
                return path.resolve()
            if has_raw_scene_layout(path):
                return path.resolve()
            if len(Path(dirpath).parts) - len(raw_root.parts) > 5:
                dirnames[:] = []
    return (raw_root / scene).resolve()


def find_sequence_dir(scene_root, sequence):
    scene_root = Path(scene_root)
    direct = scene_root / sequence
    if direct.is_dir():
        return direct
    if not scene_root.exists():
        return direct
    target = sequence.lower()
    for child in scene_root.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child
    for child in scene_root.rglob("*"):
        if child.is_dir() and child.name.lower() == target:
            return child
    return direct


def frame_stems(frame_idx):
    frame_idx = int(frame_idx)
    return [f"frame-{frame_idx:06d}", f"{frame_idx:06d}", str(frame_idx)]


def find_frame_file(seq_dir, frame_idx, kind):
    seq_dir = Path(seq_dir)
    if kind == "depth":
        suffixes = [".depth.png", ".png", ".depth.pgm", ".pgm"]
        dirs = [seq_dir, seq_dir / "depth", seq_dir / "depths", seq_dir / "sensor_depth"]
    elif kind == "pose":
        suffixes = [".pose.txt", ".txt"]
        dirs = [seq_dir, seq_dir / "pose", seq_dir / "poses", seq_dir / "camera_pose"]
    else:
        raise ValueError(kind)
    for directory in dirs:
        for stem in frame_stems(frame_idx):
            for suffix in suffixes:
                path = directory / f"{stem}{suffix}"
                if path.exists():
                    return path
    needle = f"{int(frame_idx):06d}"
    if seq_dir.exists():
        for path in seq_dir.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if needle not in lower:
                continue
            if kind == "depth" and ("depth" in lower or path.suffix.lower() in (".png", ".pgm")):
                return path
            if kind == "pose" and ("pose" in lower and path.suffix.lower() == ".txt"):
                return path
    return None


def read_depth(path, depth_scale):
    depth = np.asarray(o3d.io.read_image(str(path))).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth / float(depth_scale)


def load_fragment_points(path):
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"Empty fragment point cloud: {path}")
    return pts


def sample_indices(count, limit, seed):
    count = int(count)
    if limit <= 0 or count <= limit:
        return np.arange(count)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(count, size=int(limit), replace=False))


def depth_to_camera_points(depth, intrinsics, stride, min_depth, max_depth, max_points, seed):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    ys, xs = np.mgrid[0:h:int(stride), 0:w:int(stride)]
    z = depth[ys, xs].reshape(-1)
    xs = xs.reshape(-1).astype(np.float64)
    ys = ys.reshape(-1).astype(np.float64)
    valid = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    z = z[valid].astype(np.float64)
    xs = xs[valid]
    ys = ys[valid]
    idx = sample_indices(z.shape[0], max_points, seed)
    z = z[idx]
    xs = xs[idx]
    ys = ys[idx]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def nearest_distance_stats(query, reference):
    if query.shape[0] == 0 or reference.shape[0] == 0:
        return {"mean": float("inf"), "median": float("inf"), "p95": float("inf"), "count": int(query.shape[0])}
    try:
        from scipy.spatial import cKDTree

        dist, _ = cKDTree(reference).query(query, k=1, workers=-1)
    except Exception:
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(reference)
        kdtree = o3d.geometry.KDTreeFlann(ref_pcd)
        values = []
        for point in query:
            _, _, d2 = kdtree.search_knn_vector_3d(point, 1)
            values.append(math.sqrt(d2[0]) if d2 else float("inf"))
        dist = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(dist)),
        "median": float(np.median(dist)),
        "p95": float(np.percentile(dist, 95)),
        "count": int(query.shape[0]),
    }


def linear_voxel_keys(voxels, grid_shape):
    voxels = np.asarray(voxels, dtype=np.int64)
    shape = np.asarray(grid_shape, dtype=np.int64)
    return voxels[:, 0] + shape[0] * (voxels[:, 1] + shape[1] * voxels[:, 2])


def roc_auc_score_binary(labels, scores):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def specificity_gate_threshold(scores, labels, specificity):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    failures = scores[labels == 0]
    if failures.size == 0:
        return float(np.min(scores) if scores.size else 0.0)
    return float(np.quantile(failures, 1.0 - specificity, method="higher"))
