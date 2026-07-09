import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from redkitchen_fsv_common import (
    REPO_ROOT,
    SCENE,
    ensure_dir,
    fragment_id,
    fragment_name,
    matrix_to_cell,
    pair_scene,
    parse_pose_file,
    write_json,
)
from utils.SE3 import integrate_trans


def build_outputs(args):
    pkl_path = Path(args.pkl_path)
    fragment_scene_root = Path(args.fragment_root) / SCENE
    pose_root = fragment_scene_root / "poses"
    ply_root = fragment_scene_root / "fragments"
    raw_scene_root = Path(args.raw_root) / SCENE
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    with open(pkl_path, "rb") as f:
        infos = pickle.load(f)

    pair_rows = []
    fragments = {}
    for index, src in enumerate(infos["src"]):
        if pair_scene(src) != SCENE:
            continue
        tgt = infos["tgt"][index]
        src_id = fragment_id(src)
        tgt_id = fragment_id(tgt)
        gt_transform = integrate_trans(infos["rot"][index], infos["trans"][index])
        pair_rows.append(
            {
                "pair_id": int(index),
                "scene": SCENE,
                "src_fragment": fragment_name(src_id),
                "tgt_fragment": fragment_name(tgt_id),
                "src_id": int(src_id),
                "tgt_id": int(tgt_id),
                "overlap": float(infos["overlap"][index]),
                "gt_transform": matrix_to_cell(gt_transform),
            }
        )
        for fid in (src_id, tgt_id):
            fragments[fid] = None

    manifest = []
    for fid in sorted(fragments):
        pose_file = pose_root / f"{fragment_name(fid)}.txt"
        ply_file = ply_root / f"{fragment_name(fid)}.ply"
        meta = parse_pose_file(pose_file)
        manifest.append(
            {
                "scene": SCENE,
                "fragment_id": fragment_name(fid),
                "fragment_index": int(fid),
                "fragment_ply": str(ply_file.resolve()),
                "fragment_pose_file": str(pose_file.resolve()),
                "sequence": meta["sequence"],
                "frame_start": int(meta["frame_start"]),
                "frame_end": int(meta["frame_end"]),
                "fragment_pose": matrix_to_cell(meta["matrix"]),
                "raw_scene_root": str(raw_scene_root.resolve()),
            }
        )

    manifest_path = output_root / "redkitchen_manifest.json"
    pairs_path = output_root / "redkitchen_pairs.json"
    write_json(manifest_path, manifest)
    write_json(pairs_path, pair_rows)
    print(f"scene={SCENE}")
    print(f"pairs={len(pair_rows)}")
    print(f"fragments={len(manifest)}")
    print(f"manifest={manifest_path}")
    print(f"pairs_json={pairs_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl-path", default=str(REPO_ROOT / "3DLoMatch.pkl"))
    parser.add_argument("--fragment-root", default=str(REPO_ROOT / "3dmatch" / "test"))
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "3dmatch_raw" / "test"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space"))
    args = parser.parse_args()
    build_outputs(args)


if __name__ == "__main__":
    main()
