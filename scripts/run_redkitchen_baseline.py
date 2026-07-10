import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Correspondence_regenerate_v2 import Regenerator
from Tranformation_estimaton import Estimator
from evaluate_metric import ClassificationLoss, TransformationLoss
from initial_matching_plus import Matcher_plus
from redkitchen_fsv_common import REPO_ROOT, SCENE, matrix_to_cell, read_json, write_csv, write_json
from utils.SE3 import integrate_trans, transform


def load_feature_pair(data_root, scene, src_id, tgt_id, descriptor, num_node, device):
    src_data = np.load(Path(data_root) / "fragments" / scene / f"cloud_bin_{src_id}_{descriptor}.npz")
    tgt_data = np.load(Path(data_root) / "fragments" / scene / f"cloud_bin_{tgt_id}_{descriptor}.npz")
    src_keypts = src_data["xyz"].astype(np.float32)
    tgt_keypts = tgt_data["xyz"].astype(np.float32)
    src_features = src_data["feature"].astype(np.float32)
    tgt_features = tgt_data["feature"].astype(np.float32)
    if descriptor == "fpfh":
        src_features = src_features / (np.linalg.norm(src_features, axis=1, keepdims=True) + 1e-6)
        tgt_features = tgt_features / (np.linalg.norm(tgt_features, axis=1, keepdims=True) + 1e-6)

    src_keypts = torch.from_numpy(src_keypts).to(device)
    tgt_keypts = torch.from_numpy(tgt_keypts).to(device)
    src_features = torch.from_numpy(src_features).to(device)
    tgt_features = torch.from_numpy(tgt_features).to(device)

    if num_node != "all":
        limit = int(num_node)
        if src_keypts.shape[0] > limit:
            idx = torch.randperm(src_keypts.shape[0], device=device)[:limit]
            src_keypts = src_keypts[idx]
            src_features = src_features[idx]
        if tgt_keypts.shape[0] > limit:
            idx = torch.randperm(tgt_keypts.shape[0], device=device)[:limit]
            tgt_keypts = tgt_keypts[idx]
            tgt_features = tgt_features[idx]

    return src_keypts[None], tgt_keypts[None], src_features[None], tgt_features[None]


def evaluate_pair(row, infos, args, modules, device):
    matcher, regenerator, estimator, trans_evaluator, cls_evaluator = modules
    pair_id = int(row["pair_id"])
    src_id = int(row["src_id"])
    tgt_id = int(row["tgt_id"])
    gt_trans_np = integrate_trans(infos["rot"][pair_id], infos["trans"][pair_id]).astype(np.float32)
    gt_trans = torch.from_numpy(gt_trans_np).to(device)[None]
    src_keypts, tgt_keypts, src_features, tgt_features = load_feature_pair(
        args.data_path,
        SCENE,
        src_id,
        tgt_id,
        args.descriptor,
        args.num_node,
        device,
    )

    started = time.perf_counter()
    with torch.no_grad():
        pred_trans, src_filtered, tgt_filtered, src_corr, tgt_corr, _, _ = matcher.estimator(
            src_keypts,
            tgt_keypts,
            src_features,
            tgt_features,
        )
        if src_filtered.shape[1] == 0:
            src_filtered = src_corr
            tgt_filtered = tgt_corr

        r_src, r_tgt, pred_trans = regenerator.regenerate(
            src_filtered,
            tgt_filtered,
            src_keypts,
            tgt_keypts,
            src_features,
            tgt_features,
            knn_num=args.round1_knn,
            sampling_num=args.round1_sampling,
        )
        if args.regor_rounds >= 2:
            r_src, r_tgt, pred_trans = regenerator.regenerate(
                r_src,
                r_tgt,
                src_keypts,
                tgt_keypts,
                src_features,
                tgt_features,
                knn_num=args.round2_knn,
                sampling_num=args.round2_sampling,
            )

        pred_trans, pred_labels, src_final, tgt_final = estimator.estimator(
            r_src,
            r_tgt,
            src_keypts,
            tgt_keypts,
            gt_trans,
        )
        model_time = time.perf_counter() - started
        recall, re_value, te_value, _ = trans_evaluator(pred_trans, gt_trans, src_final, tgt_final)
        class_stats = cls_evaluator(gt_trans, src_final, tgt_final, src_corr, tgt_corr)
        input_dist = torch.norm(transform(src_corr, gt_trans) - tgt_corr, dim=-1)
        input_inliers = input_dist < args.inlier_threshold

    return {
        "pair_id": pair_id,
        "scene": SCENE,
        "src_fragment": row["src_fragment"],
        "tgt_fragment": row["tgt_fragment"],
        "src_id": src_id,
        "tgt_id": tgt_id,
        "overlap": float(row["overlap"]),
        "T_est": matrix_to_cell(pred_trans[0].detach().cpu().numpy()),
        "success": int(float(recall) > 0.0),
        "RE": float(re_value.detach().cpu().item() if torch.is_tensor(re_value) else re_value),
        "TE": float(te_value.detach().cpu().item() if torch.is_tensor(te_value) else te_value),
        "FMR": float(class_stats["feature_match_recall"]),
        "output_precision": float(class_stats["precision"]),
        "output_recall": float(class_stats["recall"]),
        "output_f1": float(class_stats["f1"]),
        "model_time": float(model_time),
        "input_inlier_count": int(input_inliers.sum().detach().cpu().item()),
        "input_inlier_ratio": float(input_inliers.float().mean().detach().cpu().item()),
        "output_inlier_count": float(class_stats["output_inlier_number"]),
        "final_corr_count": int(src_final.shape[1]),
    }


def summarize(rows, args):
    success = np.asarray([int(r["success"]) for r in rows], dtype=np.float64)
    re_values = np.asarray([float(r["RE"]) for r in rows], dtype=np.float64)
    te_values = np.asarray([float(r["TE"]) for r in rows], dtype=np.float64)
    fmr = np.asarray([float(r["FMR"]) for r in rows], dtype=np.float64)
    model_time = np.asarray([float(r["model_time"]) for r in rows], dtype=np.float64)
    good = success > 0
    metrics = {
        "num_pairs": int(len(rows)),
        "Mean Reg Recall": float(success.mean() * 100.0) if len(rows) else 0.0,
        "Mean Re": float(re_values[good].mean()) if np.any(good) else 0.0,
        "Mean Te": float(te_values[good].mean()) if np.any(good) else 0.0,
        "Mean FMR": float(fmr.mean() * 100.0) if len(rows) else 0.0,
        "Mean model time": float(model_time.mean()) if len(rows) else 0.0,
        "Output precision": float(np.mean([r["output_precision"] for r in rows])) if rows else 0.0,
        "Output recall": float(np.mean([r["output_recall"] for r in rows])) if rows else 0.0,
        "parameters": {
            "descriptor": args.descriptor,
            "num_node": args.num_node,
            "inlier_threshold": args.inlier_threshold,
            "re_thre": args.re_thre,
            "te_thre": args.te_thre,
            "regor_rounds": args.regor_rounds,
            "round1_knn": args.round1_knn,
            "round1_sampling": args.round1_sampling,
            "round2_knn": args.round2_knn,
            "round2_sampling": args.round2_sampling,
            "matcher_max_points": args.matcher_max_points,
            "matcher_k1": args.matcher_k1,
            "matcher_k2": args.matcher_k2,
            "relax_match_num": args.relax_match_num,
            "NS_by_IC": args.ns_by_ic,
        },
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_pairs.json"))
    parser.add_argument("--pkl-path", default=str(REPO_ROOT / "3DLoMatch.pkl"))
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "3DMatch"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "redkitchen_baseline"))
    parser.add_argument("--descriptor", default="fpfh")
    parser.add_argument("--num-node", default="5000")
    parser.add_argument("--inlier-threshold", type=float, default=0.1)
    parser.add_argument("--re-thre", type=float, default=15.0)
    parser.add_argument("--te-thre", type=float, default=30.0)
    parser.add_argument("--regor-rounds", type=int, default=2)
    parser.add_argument("--round1-knn", type=int, default=100)
    parser.add_argument("--round1-sampling", type=int, default=100)
    parser.add_argument("--round2-knn", type=int, default=20)
    parser.add_argument("--round2-sampling", type=int, default=500)
    parser.add_argument("--matcher-max-points", type=int, default=8000)
    parser.add_argument("--matcher-k1", type=int, default=60)
    parser.add_argument("--matcher-k2", type=int, default=50)
    parser.add_argument("--relax-match-num", type=int, default=100)
    parser.add_argument("--ns-by-ic", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Baseline REGOR code uses CUDA-specific tensors and CUDA is not available")
    device = torch.device("cuda")
    args.num_node = "all" if str(args.num_node).lower() == "all" else int(args.num_node)

    pairs = read_json(args.pairs)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    with open(args.pkl_path, "rb") as f:
        infos = pickle.load(f)

    matcher = Matcher_plus(
        inlier_threshold=args.inlier_threshold,
        num_node="all",
        use_mutual=False,
        d_thre=args.inlier_threshold,
        num_iterations=10,
        ratio=0.2,
        nms_radius=0.1,
        max_points=args.matcher_max_points,
        k1=args.matcher_k1,
        k2=args.matcher_k2,
        FS_TCD_thre=0.05,
        relax_match_num=args.relax_match_num,
        NS_by_IC=args.ns_by_ic,
    )
    modules = (
        matcher,
        Regenerator(),
        Estimator(num_node=args.num_node),
        TransformationLoss(re_thre=args.re_thre, te_thre=args.te_thre),
        ClassificationLoss(inlier_threshold=args.inlier_threshold),
    )

    rows = []
    for offset, row in enumerate(pairs, start=1):
        result = evaluate_pair(row, infos, args, modules, device)
        rows.append(result)
        print(
            f"{offset}/{len(pairs)} pair_id={result['pair_id']} "
            f"success={result['success']} RE={result['RE']:.3f} TE={result['TE']:.3f} time={result['model_time']:.3f}s"
        )
        torch.cuda.empty_cache()

    fields = [
        "pair_id",
        "scene",
        "src_fragment",
        "tgt_fragment",
        "src_id",
        "tgt_id",
        "overlap",
        "T_est",
        "success",
        "RE",
        "TE",
        "FMR",
        "output_precision",
        "output_recall",
        "output_f1",
        "model_time",
        "input_inlier_count",
        "input_inlier_ratio",
        "output_inlier_count",
        "final_corr_count",
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "predictions.csv", rows, fields)
    write_json(output_dir / "metrics.json", summarize(rows, args))
    print(f"predictions={output_dir / 'predictions.csv'}")
    print(f"metrics={output_dir / 'metrics.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
