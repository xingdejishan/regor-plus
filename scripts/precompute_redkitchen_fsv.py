import argparse
import sys
from pathlib import Path

import numpy as np

from redkitchen_fsv_common import (
    REPO_ROOT,
    SCENE,
    cell_to_matrix,
    depth_to_camera_points,
    find_frame_file,
    find_sequence_dir,
    fragment_name,
    invert_transform,
    linear_voxel_keys,
    load_fragment_points,
    load_intrinsics,
    load_matrix,
    locate_raw_scene_root,
    nearest_distance_stats,
    read_depth,
    read_json,
    transform_points,
    write_json,
)


def frame_ids(start, end, max_frames):
    values = list(range(int(start), int(end) + 1))
    if max_frames <= 0 or len(values) <= max_frames:
        return values
    idx = np.linspace(0, len(values) - 1, int(max_frames)).round().astype(int)
    return [values[i] for i in sorted(set(idx.tolist()))]


def voxel_coords_from_linear(linear, shape):
    linear = np.asarray(linear, dtype=np.int64)
    sx, sy, _ = [int(v) for v in shape]
    z = linear // (sx * sy)
    rem = linear - z * sx * sy
    y = rem // sx
    x = rem - y * sx
    return np.stack([x, y, z], axis=1).astype(np.int32)


def transform_note_and_matrix(frame_pose, fragment_pose, cam_mode, fragment_mode):
    if cam_mode == "cam_to_world":
        world_from_cam = frame_pose
    elif cam_mode == "world_to_cam":
        world_from_cam = invert_transform(frame_pose)
    else:
        raise ValueError(cam_mode)

    if fragment_mode == "fragment_to_world":
        fragment_from_world = invert_transform(fragment_pose)
    elif fragment_mode == "world_to_fragment":
        fragment_from_world = fragment_pose
    else:
        raise ValueError(fragment_mode)
    return fragment_from_world @ world_from_cam


def choose_coordinate_system(item, intrinsics, args):
    raw_scene_root = Path(item["raw_scene_root"])
    seq_dir = find_sequence_dir(raw_scene_root, item["sequence"])
    fragment_points = load_fragment_points(item["fragment_ply"])
    fragment_pose = cell_to_matrix(item["fragment_pose"])
    chosen_frames = frame_ids(item["frame_start"], item["frame_end"], args.sanity_frames)
    candidates = []
    depth_scales = [float(v) for v in str(args.depth_scales).split(",")]

    for depth_scale in depth_scales:
        frame_surfaces = []
        frame_poses = []
        for frame_idx in chosen_frames:
            depth_path = find_frame_file(seq_dir, frame_idx, "depth")
            pose_path = find_frame_file(seq_dir, frame_idx, "pose")
            if depth_path is None or pose_path is None:
                continue
            depth = read_depth(depth_path, depth_scale)
            points_cam = depth_to_camera_points(
                depth,
                intrinsics,
                args.sanity_stride,
                args.min_valid_depth,
                args.max_depth,
                args.sanity_points_per_frame,
                frame_idx,
            )
            if points_cam.shape[0] == 0:
                continue
            frame_surfaces.append(points_cam)
            frame_poses.append(load_matrix(pose_path))

        if not frame_surfaces:
            continue

        for cam_mode in ("cam_to_world", "world_to_cam"):
            for fragment_mode in ("fragment_to_world", "world_to_fragment"):
                surfaces = []
                for points_cam, frame_pose in zip(frame_surfaces, frame_poses):
                    frag_from_cam = transform_note_and_matrix(frame_pose, fragment_pose, cam_mode, fragment_mode)
                    surfaces.append(transform_points(points_cam, frag_from_cam))
                query = np.concatenate(surfaces, axis=0)
                if query.shape[0] > args.sanity_total_points:
                    sel = np.random.default_rng(17).choice(query.shape[0], args.sanity_total_points, replace=False)
                    query = query[np.sort(sel)]
                stats = nearest_distance_stats(query, fragment_points)
                candidates.append(
                    {
                        "depth_scale": depth_scale,
                        "cam_pose_mode": cam_mode,
                        "fragment_pose_mode": fragment_mode,
                        "mean": stats["mean"],
                        "median": stats["median"],
                        "p95": stats["p95"],
                        "count": stats["count"],
                        "frames": len(frame_surfaces),
                    }
                )

    if not candidates:
        raise RuntimeError(f"No valid sanity candidates for {item['fragment_id']}")
    best = min(candidates, key=lambda x: (x["mean"], x["median"]))
    ok = best["mean"] <= args.sanity_mean_threshold and best["median"] <= args.sanity_median_threshold
    return best, candidates, ok


def select_valid_frames(item, best, intrinsics, args):
    raw_scene_root = Path(item["raw_scene_root"])
    seq_dir = find_sequence_dir(raw_scene_root, item["sequence"])
    fragment_points = load_fragment_points(item["fragment_ply"])
    fragment_pose = cell_to_matrix(item["fragment_pose"])
    rows = []
    valid_frames = []
    for frame_idx in range(int(item["frame_start"]), int(item["frame_end"]) + 1):
        depth_path = find_frame_file(seq_dir, frame_idx, "depth")
        pose_path = find_frame_file(seq_dir, frame_idx, "pose")
        if depth_path is None or pose_path is None:
            rows.append({"frame": int(frame_idx), "ok": False, "reason": "missing_depth_or_pose"})
            continue
        depth = read_depth(depth_path, best["depth_scale"])
        points_cam = depth_to_camera_points(
            depth,
            intrinsics,
            args.frame_sanity_stride,
            args.min_valid_depth,
            args.max_depth,
            args.frame_sanity_points_per_frame,
            frame_idx,
        )
        if points_cam.shape[0] == 0:
            rows.append({"frame": int(frame_idx), "ok": False, "reason": "no_valid_depth"})
            continue
        frame_pose = load_matrix(pose_path)
        frag_from_cam = transform_note_and_matrix(frame_pose, fragment_pose, best["cam_pose_mode"], best["fragment_pose_mode"])
        stats = nearest_distance_stats(transform_points(points_cam, frag_from_cam), fragment_points)
        ok = stats["mean"] <= args.frame_mean_threshold and stats["median"] <= args.frame_median_threshold
        rows.append(
            {
                "frame": int(frame_idx),
                "ok": bool(ok),
                "mean": stats["mean"],
                "median": stats["median"],
                "p95": stats["p95"],
                "count": stats["count"],
            }
        )
        if ok:
            valid_frames.append(int(frame_idx))
    enough = len(valid_frames) >= args.min_valid_frames and len(valid_frames) / max(len(rows), 1) >= args.min_valid_frame_ratio
    return valid_frames, rows, enough


def project_voxels_for_frame(voxel_points, intrinsics, depth, cam_from_frag, args):
    points_cam = transform_points(voxel_points, cam_from_frag)
    z = points_cam[:, 2]
    valid_z = np.isfinite(z) & (z >= args.min_valid_depth) & (z <= args.max_depth)
    if not np.any(valid_z):
        return np.zeros(voxel_points.shape[0], dtype=bool), np.zeros(voxel_points.shape[0], dtype=bool)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    h, w = depth.shape
    valid_idx = np.where(valid_z)[0]
    pc = points_cam[valid_idx]
    u = np.rint((pc[:, 0] * fx / pc[:, 2]) + cx).astype(np.int64)
    v = np.rint((pc[:, 1] * fy / pc[:, 2]) + cy).astype(np.int64)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    free = np.zeros(voxel_points.shape[0], dtype=bool)
    occupied = np.zeros(voxel_points.shape[0], dtype=bool)
    if not np.any(inside):
        return free, occupied
    idx = valid_idx[inside]
    d = depth[v[inside], u[inside]]
    z_inside = z[idx]
    valid_depth = np.isfinite(d) & (d >= args.min_valid_depth) & (d <= args.max_depth)
    if not np.any(valid_depth):
        return free, occupied
    idx = idx[valid_depth]
    d = d[valid_depth]
    z_inside = z_inside[valid_depth]
    free[idx] = z_inside < (d - args.free_space_truncation)
    occupied[idx] = np.abs(z_inside - d) <= args.surface_truncation
    return free, occupied


def precompute_fragment(item, best, intrinsics, intrinsics_path, args):
    out_dir = Path(args.output_root) / SCENE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item['fragment_id']}_fsv.npz"
    if out_path.exists() and not args.overwrite:
        return {"fragment_id": item["fragment_id"], "path": str(out_path), "status": "exists"}

    fragment_points = load_fragment_points(item["fragment_ply"])
    bbox_min = fragment_points.min(axis=0) - args.grid_margin
    bbox_max = fragment_points.max(axis=0) + args.grid_margin
    grid_shape = np.ceil((bbox_max - bbox_min) / args.voxel_size).astype(np.int64) + 1
    total_voxels = int(np.prod(grid_shape))
    if total_voxels > args.max_grid_voxels:
        raise RuntimeError(f"{item['fragment_id']} grid has {total_voxels} voxels, above max_grid_voxels={args.max_grid_voxels}")

    raw_scene_root = Path(item["raw_scene_root"])
    seq_dir = find_sequence_dir(raw_scene_root, item["sequence"])
    fragment_pose = cell_to_matrix(item["fragment_pose"])
    free = np.zeros(total_voxels, dtype=bool)
    occupied = np.zeros(total_voxels, dtype=bool)
    used_frames = 0

    source_frames = item.get("valid_frame_ids") or frame_ids(item["frame_start"], item["frame_end"], args.frame_stride_max_frames)
    for frame_idx in source_frames:
        if int(frame_idx - item["frame_start"]) % int(args.frame_step) != 0:
            continue
        depth_path = find_frame_file(seq_dir, frame_idx, "depth")
        pose_path = find_frame_file(seq_dir, frame_idx, "pose")
        if depth_path is None or pose_path is None:
            raise FileNotFoundError(f"Missing raw RGB-D frame {item['sequence']}:{frame_idx} for {item['fragment_id']}")
        depth = read_depth(depth_path, best["depth_scale"])
        frame_pose = load_matrix(pose_path)
        frag_from_cam = transform_note_and_matrix(frame_pose, fragment_pose, best["cam_pose_mode"], best["fragment_pose_mode"])
        cam_from_frag = invert_transform(frag_from_cam)
        used_frames += 1
        for start in range(0, total_voxels, args.voxel_chunk):
            end = min(total_voxels, start + args.voxel_chunk)
            linear = np.arange(start, end, dtype=np.int64)
            vox = voxel_coords_from_linear(linear, grid_shape)
            centers = bbox_min[None, :] + (vox.astype(np.float64) + 0.5) * args.voxel_size
            chunk_free, chunk_occ = project_voxels_for_frame(centers, intrinsics, depth, cam_from_frag, args)
            free[start:end] |= chunk_free
            occupied[start:end] |= chunk_occ

    free &= ~occupied
    free_indices = voxel_coords_from_linear(np.where(free)[0], grid_shape)
    occupied_indices = voxel_coords_from_linear(np.where(occupied)[0], grid_shape)
    note = f"depth_scale={best['depth_scale']}; cam_pose={best['cam_pose_mode']}; fragment_pose={best['fragment_pose_mode']}"
    np.savez_compressed(
        out_path,
        free_voxel_indices=free_indices,
        occupied_voxel_indices=occupied_indices,
        free_voxels=free_indices,
        origin=bbox_min.astype(np.float32),
        voxel_size=np.float32(args.voxel_size),
        grid_shape=grid_shape.astype(np.int32),
        scene=np.asarray(SCENE),
        fragment_id=np.asarray(item["fragment_id"]),
        sequence=np.asarray(item["sequence"]),
        frame_start=np.int32(item["frame_start"]),
        frame_end=np.int32(item["frame_end"]),
        used_frame_ids=np.asarray(source_frames, dtype=np.int32),
        depth_scale=np.float32(best["depth_scale"]),
        camera_intrinsics=intrinsics.astype(np.float32),
        intrinsics_path=np.asarray(str(intrinsics_path)),
        num_depth_frames=np.int32(used_frames),
        surface_truncation=np.float32(args.surface_truncation),
        free_space_truncation=np.float32(args.free_space_truncation),
        min_valid_depth=np.float32(args.min_valid_depth),
        max_depth=np.float32(args.max_depth),
        grid_margin=np.float32(args.grid_margin),
        coordinate_system_note=np.asarray(note),
        frame_filter_note=np.asarray(item.get("frame_filter_note", "")),
    )
    return {
        "fragment_id": item["fragment_id"],
        "path": str(out_path),
        "status": "ok",
        "num_depth_frames": int(used_frames),
        "grid_shape": grid_shape.astype(int).tolist(),
        "free_voxels": int(free_indices.shape[0]),
        "occupied_voxels": int(occupied_indices.shape[0]),
        "coordinate_system_note": note,
    }


def target_fragment_ids(pairs):
    return sorted({int(row["tgt_id"]) for row in pairs})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_manifest.json"))
    parser.add_argument("--pairs", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_pairs.json"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "3dmatch_raw" / "test"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space"))
    parser.add_argument("--sanity-output", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_fsv_precompute_report.json"))
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--surface-truncation", type=float, default=0.05)
    parser.add_argument("--free-space-truncation", type=float, default=0.02)
    parser.add_argument("--min-valid-depth", type=float, default=0.2)
    parser.add_argument("--grid-margin", type=float, default=0.25)
    parser.add_argument("--voxel-chunk", type=int, default=200000)
    parser.add_argument("--max-grid-voxels", type=int, default=2000000)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-stride-max-frames", type=int, default=0)
    parser.add_argument("--depth-scales", default="1000,5000")
    parser.add_argument("--sanity-frames", type=int, default=5)
    parser.add_argument("--sanity-stride", type=int, default=12)
    parser.add_argument("--sanity-points-per-frame", type=int, default=4000)
    parser.add_argument("--sanity-total-points", type=int, default=12000)
    parser.add_argument("--sanity-mean-threshold", type=float, default=0.10)
    parser.add_argument("--sanity-median-threshold", type=float, default=0.08)
    parser.add_argument("--frame-sanity-stride", type=int, default=12)
    parser.add_argument("--frame-sanity-points-per-frame", type=int, default=4000)
    parser.add_argument("--frame-mean-threshold", type=float, default=0.10)
    parser.add_argument("--frame-median-threshold", type=float, default=0.08)
    parser.add_argument("--min-valid-frames", type=int, default=10)
    parser.add_argument("--min-valid-frame-ratio", type=float, default=0.2)
    parser.add_argument("--all-fragments", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    pairs = read_json(args.pairs)
    raw_scene_root = locate_raw_scene_root(args.raw_root, SCENE)
    intrinsics, intrinsics_path = load_intrinsics(raw_scene_root)
    for item in manifest:
        item["raw_scene_root"] = str(raw_scene_root)

    wanted = {int(item["fragment_index"]) for item in manifest} if args.all_fragments else set(target_fragment_ids(pairs))
    by_id = {int(item["fragment_index"]): item for item in manifest}
    report = {
        "scene": SCENE,
        "raw_scene_root": str(raw_scene_root),
        "intrinsics_path": str(intrinsics_path),
        "parameters": vars(args),
        "sanity": [],
        "precomputed": [],
        "ok": False,
    }

    failed = []
    for fid in wanted:
        item = by_id[fid]
        best, candidates, ok = choose_coordinate_system(item, intrinsics, args)
        valid_frame_ids, frame_stats, frame_ok = select_valid_frames(item, best, intrinsics, args)
        item["valid_frame_ids"] = valid_frame_ids
        item["frame_filter_note"] = (
            f"used {len(valid_frame_ids)} / {int(item['frame_end']) - int(item['frame_start']) + 1} frames "
            f"with mean<={args.frame_mean_threshold} and median<={args.frame_median_threshold}"
        )
        ok = bool(frame_ok and best["median"] <= args.sanity_median_threshold)
        report["sanity"].append(
            {
                "fragment_id": fragment_name(fid),
                "ok": bool(ok),
                "best": best,
                "candidates": candidates,
                "valid_frame_count": len(valid_frame_ids),
                "total_frame_count": int(item["frame_end"]) - int(item["frame_start"]) + 1,
                "frame_filter_note": item["frame_filter_note"],
                "frame_stats": frame_stats,
            }
        )
        if not ok:
            failed.append(fragment_name(fid))
            continue
        result = precompute_fragment(item, best, intrinsics, intrinsics_path, args)
        report["precomputed"].append(result)
        print(f"{fragment_name(fid)} {result['status']} {result.get('free_voxels', '')} free voxels")

    report["ok"] = len(failed) == 0
    report["failed_sanity"] = failed
    write_json(args.sanity_output, report)
    if failed:
        raise SystemExit(f"Coordinate sanity failed for: {', '.join(failed)}")
    print(f"precomputed={len(report['precomputed'])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
