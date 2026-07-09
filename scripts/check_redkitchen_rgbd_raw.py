import argparse
from pathlib import Path

from redkitchen_fsv_common import (
    REPO_ROOT,
    SCENE,
    find_frame_file,
    find_intrinsics,
    find_sequence_dir,
    locate_raw_scene_root,
    read_json,
    write_json,
)


def check(args):
    manifest_path = Path(args.manifest)
    manifest = read_json(manifest_path)
    raw_scene_root = locate_raw_scene_root(args.raw_root, SCENE)
    intrinsics = find_intrinsics(raw_scene_root)
    records = []
    total_depth = 0
    total_pose = 0
    missing_depth = []
    missing_pose = []
    repaired_root = False

    for item in manifest:
        old_root = Path(item["raw_scene_root"])
        if old_root.resolve() != raw_scene_root.resolve():
            item["raw_scene_root"] = str(raw_scene_root)
            repaired_root = True
        seq_dir = find_sequence_dir(raw_scene_root, item["sequence"])
        seq_exists = seq_dir.is_dir()
        checked = 0
        for frame_idx in range(int(item["frame_start"]), int(item["frame_end"]) + 1):
            checked += 1
            depth_path = find_frame_file(seq_dir, frame_idx, "depth") if seq_exists else None
            pose_path = find_frame_file(seq_dir, frame_idx, "pose") if seq_exists else None
            if depth_path is None:
                missing_depth.append(
                    {
                        "fragment_id": item["fragment_id"],
                        "sequence": item["sequence"],
                        "frame": int(frame_idx),
                    }
                )
            else:
                total_depth += 1
            if pose_path is None:
                missing_pose.append(
                    {
                        "fragment_id": item["fragment_id"],
                        "sequence": item["sequence"],
                        "frame": int(frame_idx),
                    }
                )
            else:
                total_pose += 1
        records.append(
            {
                "fragment_id": item["fragment_id"],
                "sequence": item["sequence"],
                "frame_start": int(item["frame_start"]),
                "frame_end": int(item["frame_end"]),
                "frames_checked": checked,
                "sequence_dir": str(seq_dir),
                "sequence_exists": bool(seq_exists),
            }
        )

    if repaired_root and args.repair_manifest:
        write_json(manifest_path, manifest)

    result = {
        "scene": SCENE,
        "raw_scene_root": str(raw_scene_root),
        "intrinsics_path": str(intrinsics) if intrinsics else "",
        "intrinsics_exists": intrinsics is not None,
        "fragment_count": len(manifest),
        "records": records,
        "total_depth_found": total_depth,
        "total_pose_found": total_pose,
        "missing_depth_count": len(missing_depth),
        "missing_pose_count": len(missing_pose),
        "missing_depth_examples": missing_depth[: int(args.max_examples)],
        "missing_pose_examples": missing_pose[: int(args.max_examples)],
        "repaired_manifest_raw_root": repaired_root and args.repair_manifest,
        "ok": intrinsics is not None and len(missing_depth) == 0 and len(missing_pose) == 0,
    }
    write_json(args.output, result)
    print(f"raw_scene_root={raw_scene_root}")
    print(f"intrinsics={intrinsics}")
    print(f"fragments={len(manifest)}")
    print(f"missing_depth={len(missing_depth)}")
    print(f"missing_pose={len(missing_pose)}")
    if not result["ok"]:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_manifest.json"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "3dmatch_raw" / "test"))
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_raw_check.json"))
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--repair-manifest", action="store_true", default=True)
    args = parser.parse_args()
    check(args)


if __name__ == "__main__":
    main()
