import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from redkitchen_fsv_common import REPO_ROOT, SCENE, read_json, write_json


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_fpfh_xyz(cache, fpfh_root, fragment_id):
    fragment_id = int(fragment_id)
    if fragment_id not in cache:
        path = fpfh_root / f"cloud_bin_{fragment_id}_fpfh.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        cache[fragment_id] = np.load(path)["xyz"].astype(np.float32)
    return cache[fragment_id]


def transfer_to_fpfh(pred_xyz, query_xyz, values):
    tree = cKDTree(pred_xyz.astype(np.float32))
    distances, indices = tree.query(query_xyz.astype(np.float32), k=1)
    return values[indices].astype(np.float32), distances.astype(np.float32)


class ArrayStats:
    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min_value = None
        self.max_value = None

    def add(self, values):
        values = np.asarray(values)
        if values.size == 0:
            return
        self.count += int(values.size)
        self.total += float(values.sum(dtype=np.float64))
        current_min = float(values.min())
        current_max = float(values.max())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

    def summary(self):
        return {
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "mean": self.total / self.count if self.count else None,
        }


def pair_output_name(src_id, tgt_id):
    return f"cloud_bin_{int(src_id)}_cloud_bin_{int(tgt_id)}_overlap.npz"


def preprocess(args):
    pair_path = Path(args.pairs)
    predator_root = Path(args.predator_root)
    fpfh_root = Path(args.fpfh_root)
    output_dir = Path(args.output_root) / SCENE
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = read_json(pair_path)
    if len(pairs) != args.expected_pairs:
        raise ValueError(f"Expected {args.expected_pairs} redkitchen pairs, got {len(pairs)} from {pair_path}.")

    fpfh_cache = {}
    missing = []
    failed = []
    rows = []
    overlap_stats = ArrayStats()
    saliency_stats = ArrayStats()
    matchability_stats = ArrayStats()
    nn_stats = ArrayStats()
    src_p95_values = []
    tgt_p95_values = []

    for index, row in enumerate(pairs, start=1):
        pair_id = int(row["pair_id"])
        src_id = int(row["src_id"])
        tgt_id = int(row["tgt_id"])
        pth_path = predator_root / f"{pair_id}.pth"
        out_path = output_dir / pair_output_name(src_id, tgt_id)

        if not pth_path.exists():
            missing.append({"pair_id": pair_id, "src_id": src_id, "tgt_id": tgt_id, "path": str(pth_path)})
            continue
        if out_path.exists() and not args.overwrite:
            continue

        try:
            data = torch.load(pth_path, map_location="cpu")
            pred_xyz = as_numpy(data["pcd"]).astype(np.float32)
            overlaps = np.clip(as_numpy(data["overlaps"]).reshape(-1).astype(np.float32), 0.0, 1.0)
            saliency = np.clip(as_numpy(data["saliency"]).reshape(-1).astype(np.float32), 0.0, 1.0)
            len_src = int(data["len_src"])
            if pred_xyz.shape[0] != overlaps.shape[0] or pred_xyz.shape[0] != saliency.shape[0]:
                raise ValueError(
                    f"pcd/overlaps/saliency length mismatch: {pred_xyz.shape[0]}, {overlaps.shape[0]}, {saliency.shape[0]}"
                )
            if len_src <= 0 or len_src >= pred_xyz.shape[0]:
                raise ValueError(f"Invalid len_src={len_src} for total={pred_xyz.shape[0]}")

            src_xyz = load_fpfh_xyz(fpfh_cache, fpfh_root, src_id)
            tgt_xyz = load_fpfh_xyz(fpfh_cache, fpfh_root, tgt_id)

            src_pred_xyz = pred_xyz[:len_src]
            tgt_pred_xyz = pred_xyz[len_src:]
            src_overlap, src_dist = transfer_to_fpfh(src_pred_xyz, src_xyz, overlaps[:len_src])
            tgt_overlap, tgt_dist = transfer_to_fpfh(tgt_pred_xyz, tgt_xyz, overlaps[len_src:])
            src_saliency, _ = transfer_to_fpfh(src_pred_xyz, src_xyz, saliency[:len_src])
            tgt_saliency, _ = transfer_to_fpfh(tgt_pred_xyz, tgt_xyz, saliency[len_src:])

            src_p95 = float(np.percentile(src_dist, 95))
            tgt_p95 = float(np.percentile(tgt_dist, 95))
            src_mean = float(src_dist.mean())
            tgt_mean = float(tgt_dist.mean())
            src_max = float(src_dist.max())
            tgt_max = float(tgt_dist.max())
            if max(src_p95, tgt_p95) > args.nn_fail_threshold or max(src_mean, tgt_mean) > args.nn_warn_threshold:
                raise ValueError(
                    f"NN transfer distance too large: src_mean={src_mean:.6f}, src_p95={src_p95:.6f}, "
                    f"tgt_mean={tgt_mean:.6f}, tgt_p95={tgt_p95:.6f}"
                )

            src_matchability = np.clip(src_overlap * src_saliency, 0.0, 1.0).astype(np.float32)
            tgt_matchability = np.clip(tgt_overlap * tgt_saliency, 0.0, 1.0).astype(np.float32)
            np.savez_compressed(
                out_path,
                src_overlap=src_overlap.astype(np.float32),
                tgt_overlap=tgt_overlap.astype(np.float32),
                src_saliency=src_saliency.astype(np.float32),
                tgt_saliency=tgt_saliency.astype(np.float32),
                src_matchability=src_matchability,
                tgt_matchability=tgt_matchability,
                pair_id=np.int32(pair_id),
                src_id=np.int32(src_id),
                tgt_id=np.int32(tgt_id),
                source=np.array("predator_snapshot"),
                resampling=np.array("nearest_neighbor_predator_to_fpfh"),
                normalization=np.array("clip_0_1"),
            )

            overlap_stats.add(src_overlap)
            overlap_stats.add(tgt_overlap)
            saliency_stats.add(src_saliency)
            saliency_stats.add(tgt_saliency)
            matchability_stats.add(src_matchability)
            matchability_stats.add(tgt_matchability)
            nn_stats.add(src_dist)
            nn_stats.add(tgt_dist)
            src_p95_values.append(src_p95)
            tgt_p95_values.append(tgt_p95)
            rows.append(
                {
                    "pair_id": pair_id,
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "src_count": int(src_xyz.shape[0]),
                    "tgt_count": int(tgt_xyz.shape[0]),
                    "pred_src_count": int(src_pred_xyz.shape[0]),
                    "pred_tgt_count": int(tgt_pred_xyz.shape[0]),
                    "src_nn_mean": src_mean,
                    "tgt_nn_mean": tgt_mean,
                    "src_nn_p95": src_p95,
                    "tgt_nn_p95": tgt_p95,
                    "src_nn_max": src_max,
                    "tgt_nn_max": tgt_max,
                    "src_overlap_mean": float(src_overlap.mean()),
                    "tgt_overlap_mean": float(tgt_overlap.mean()),
                    "output": str(out_path),
                }
            )
        except Exception as exc:
            failed.append({"pair_id": pair_id, "src_id": src_id, "tgt_id": tgt_id, "error": str(exc)})

        if index % args.progress_interval == 0:
            print(f"processed {index}/{len(pairs)} pairs")

    output_files = sorted(output_dir.glob("*_overlap.npz"))
    report = {
        "scene": SCENE,
        "source": "predator_snapshot",
        "proxy": False,
        "pair_dependency": True,
        "resampling": "nearest_neighbor_predator_to_fpfh",
        "input_pairs": str(pair_path),
        "predator_root": str(predator_root),
        "fpfh_root": str(fpfh_root),
        "output_dir": str(output_dir),
        "expected_pairs": args.expected_pairs,
        "processed_pairs": len(rows),
        "output_files": len(output_files),
        "missing_pairs": missing,
        "failed_pairs": failed,
        "params": {
            "normalization": "clip_0_1",
            "allow_proxy": False,
            "nn_warn_threshold": args.nn_warn_threshold,
            "nn_fail_threshold": args.nn_fail_threshold,
        },
        "overlap": overlap_stats.summary(),
        "saliency": saliency_stats.summary(),
        "matchability": matchability_stats.summary(),
        "nn_distance": {
            **nn_stats.summary(),
            "src_p95_mean": float(np.mean(src_p95_values)) if src_p95_values else None,
            "src_p95_max": float(np.max(src_p95_values)) if src_p95_values else None,
            "tgt_p95_mean": float(np.mean(tgt_p95_values)) if tgt_p95_values else None,
            "tgt_p95_max": float(np.max(tgt_p95_values)) if tgt_p95_values else None,
        },
        "pairs": rows,
    }
    write_json(args.report, report)

    if missing or failed or len(output_files) != args.expected_pairs:
        raise RuntimeError(
            f"Overlap_pred preprocessing incomplete: files={len(output_files)}, missing={len(missing)}, failed={len(failed)}. "
            f"See {args.report}."
        )
    print(f"wrote {len(output_files)} overlap_pred files to {output_dir}")
    print(f"report: {args.report}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(REPO_ROOT / "data/3DMatch/free_space/redkitchen_pairs.json"))
    parser.add_argument("--predator-root", default=str(REPO_ROOT / "external/OverlapPredator/snapshot/indoor/3DLoMatch"))
    parser.add_argument("--fpfh-root", default=str(REPO_ROOT / "data/3DMatch/fragments/7-scenes-redkitchen"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "data/3DMatch/overlap_pred"))
    parser.add_argument("--report", default=str(REPO_ROOT / "data/3DMatch/overlap_pred/redkitchen_overlap_pred_report.json"))
    parser.add_argument("--expected-pairs", type=int, default=525)
    parser.add_argument("--nn-warn-threshold", type=float, default=0.05)
    parser.add_argument("--nn-fail-threshold", type=float, default=0.20)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    preprocess(parse_args())
