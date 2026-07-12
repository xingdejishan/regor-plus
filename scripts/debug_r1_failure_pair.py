import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from memory_graph_config import MemoryGraphConfig
from memory_guided_registration import MemoryGuidedRegistration
from test_3DLoMatch import audit_hypothesis, build_loader, load_r1_cache_payload, load_required_r1_cache, pose_errors, set_experiment_seed
from memory_graph_fixed_config import load_fixed_config


VARIANTS = {
    "equal_budget_no_history": {
        "memory_use_reliability": False,
        "memory_use_relation_history": False,
        "memory_use_basin": False,
    },
    "full": {
        "memory_use_reliability": True,
        "memory_use_relation_history": True,
        "memory_use_basin": True,
    },
}


TRACE_FIELDS = (
    "round_id",
    "attempt_index",
    "seed_id",
    "support_size",
    "lambda2_over_lambda1",
    "lambda3_over_lambda1",
    "coverage",
    "cross_group_score",
    "lambda3_over_lambda2",
    "structure_penalty",
    "support_objective",
    "rejection_reasons",
    "raw_generated",
)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def hypothesis_audit_rows(result, pair, re_threshold, te_threshold, variant):
    hypotheses = result.evaluated_hypotheses
    validation_order = sorted(hypotheses, key=lambda item: (-item.validation_score, item.hypothesis_id))
    search_order = sorted(hypotheses, key=lambda item: (-item.search_score, item.hypothesis_id))
    validation_ranks = {item.hypothesis_id: rank for rank, item in enumerate(validation_order, start=1)}
    search_ranks = {item.hypothesis_id: rank for rank, item in enumerate(search_order, start=1)}
    log_by_id = {
        row["hypothesis_id"]: row
        for row in result.candidate_logs
        if row.get("stage") in {"r1", "raw", "refinement_child"}
    }
    rows = []
    for hypothesis in hypotheses:
        audit = audit_hypothesis(hypothesis, pair.gt_transform, re_threshold, te_threshold)
        metadata = log_by_id.get(hypothesis.hypothesis_id, {})
        rows.append({
            "variant": variant,
            "pair_id": pair.pair_id,
            "hypothesis_id": hypothesis.hypothesis_id,
            "parent_id": hypothesis.parent_id,
            "round_id": hypothesis.round_id,
            "stage": hypothesis.stage,
            "support_ids": json.dumps(hypothesis.support_ids.detach().cpu().tolist()),
            "validation_score": hypothesis.validation_score,
            "search_score": hypothesis.search_score,
            "structure_penalty": hypothesis.structure_penalty,
            "basin_adjustment": hypothesis.basin_adjustment,
            "inlier_count": hypothesis.inlier_count,
            "mean_inlier_error": hypothesis.mean_inlier_error,
            "coverage": hypothesis.coverage,
            "validation_rank": validation_ranks[hypothesis.hypothesis_id],
            "search_rank": search_ranks[hypothesis.hypothesis_id],
            "accepted": metadata.get("accepted", 1),
            "reject_reason": metadata.get("reject_reason", ""),
            "gt_success": audit["local_refined_success"],
            "re": audit["local_refined_re"],
            "te": audit["local_refined_te"],
            "raw_success": audit["raw_success"],
            "raw_re": audit["raw_re"],
            "raw_te": audit["raw_te"],
        })
    return rows


def indexed_support_hypotheses(hypotheses):
    grouped = {}
    for hypothesis in sorted(hypotheses, key=lambda item: item.hypothesis_id):
        base = (
            hypothesis.round_id,
            hypothesis.stage,
            tuple(hypothesis.support_ids.detach().cpu().tolist()),
        )
        grouped.setdefault(base, []).append(hypothesis)
    return {
        (*base, occurrence): hypothesis
        for base, items in grouped.items()
        for occurrence, hypothesis in enumerate(items)
    }


def support_pose_consistency(first, second):
    first_index = indexed_support_hypotheses(first)
    second_index = indexed_support_hypotheses(second)
    rows = []
    for key in sorted(set(first_index) & set(second_index), key=str):
        first_hypothesis = first_index[key]
        second_hypothesis = second_index[key]
        raw_delta = float(torch.max(torch.abs(first_hypothesis.pose_raw - second_hypothesis.pose_raw)).item())
        refined_delta = float(torch.max(torch.abs(first_hypothesis.pose - second_hypothesis.pose)).item())
        rows.append({
            "round_id": key[0],
            "stage": key[1],
            "support_ids": json.dumps(list(key[2])),
            "occurrence": key[3],
            "no_history_hypothesis_id": first_hypothesis.hypothesis_id,
            "full_hypothesis_id": second_hypothesis.hypothesis_id,
            "raw_pose_max_abs_delta": raw_delta,
            "refined_pose_max_abs_delta": refined_delta,
            "raw_pose_equal": int(torch.allclose(first_hypothesis.pose_raw, second_hypothesis.pose_raw, atol=1e-6, rtol=0.0)),
            "refined_pose_equal": int(torch.allclose(first_hypothesis.pose, second_hypothesis.pose, atol=1e-6, rtol=0.0)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json" / "config_memory_graph_20_seed51.json"))
    parser.add_argument("--pair-index", type=int, default=1)
    args = parser.parse_args()
    base, protocol, _ = load_fixed_config(args.config_path)
    output_dir = Path(protocol["output_root"]) / f"debug_pair_{args.pair_index:06d}_decoupled"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite debug output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    summaries = []
    selected_summaries = []
    candidate_audits = []
    hypotheses_by_variant = {}
    for name in ("equal_budget_no_history", "full"):
        config = copy.deepcopy(base)
        config.memory_graph.update(VARIANTS[name])
        memory_config = MemoryGraphConfig.from_mapping(config.memory_graph).validate()
        set_experiment_seed(int(config.seed))
        loader = build_loader(config, memory_config)
        payload = load_r1_cache_payload(config, args.pair_index)
        pair = loader.get_pair(
            args.pair_index,
            src_sampled_indices_override=payload["src_sampled_indices"],
            tgt_sampled_indices_override=payload["tgt_sampled_indices"],
        )
        cache = load_required_r1_cache(pair, args.pair_index, config, payload)
        re, te = pose_errors(cache["pose"], pair.gt_transform)
        r1_failure = int(not (re < config.re_thre and te < config.te_thre))
        if not r1_failure:
            raise ValueError(f"pair_index={args.pair_index} pair_id={pair.pair_id} is not an R1 failure.")
        result = MemoryGuidedRegistration(memory_config).run(
            pair.src_keypoints,
            pair.tgt_keypoints,
            pair.src_features,
            pair.tgt_features,
            initial_pose=cache["pose"],
            initial_src_indices=cache["src_corr_indices"],
            initial_tgt_indices=cache["tgt_corr_indices"],
        )
        hypotheses_by_variant[name] = result.evaluated_hypotheses
        variant_audits = hypothesis_audit_rows(result, pair, config.re_thre, config.te_thre, name)
        candidate_audits.extend(variant_audits)
        write_csv(output_dir / f"{name}_candidate_audit.csv", variant_audits)
        attempts = []
        trace_path = output_dir / f"{name}_support_trace.txt"
        with open(trace_path, "w", encoding="utf-8") as trace_handle:
            for row in result.candidate_logs:
                if row["stage"] != "attempt":
                    continue
                trace = {
                    "variant": name,
                    "pair_id": pair.pair_id,
                    **{key: row.get(key) for key in TRACE_FIELDS if key != "rejection_reasons"},
                    "rejection_reasons": row.get("constraint_reasons", ""),
                }
                attempts.append(trace)
                traces.append(trace)
                line = (
                    f"{name} pair_id={pair.pair_id} round={trace['round_id']} attempt={trace['attempt_index']} "
                    f"support_size={trace['support_size']} lambda2/lambda1={trace['lambda2_over_lambda1']:.6f} "
                    f"lambda3/lambda1={trace['lambda3_over_lambda1']:.6f} coverage={trace['coverage']:.6f} "
                    f"cross_group_score={trace['cross_group_score']:.6f} penalty={trace['structure_penalty']:.6f} "
                    f"constraints={trace['rejection_reasons'] or '-'} raw_generated={trace['raw_generated']}"
                )
                print(line)
                trace_handle.write(line + "\n")
        write_csv(output_dir / f"{name}_attempt_logs.csv", attempts)
        for row in result.round_logs:
            if row["round_id"] == 0:
                continue
            summaries.append({
                "variant": name,
                "pair_id": pair.pair_id,
                "r1_failure": r1_failure,
                "r1_re": re,
                "r1_te": te,
                **row,
            })
        selected_audit = audit_hypothesis(result.best, pair.gt_transform, config.re_thre, config.te_thre)
        selected_summaries.append({
            "variant": name,
            "pair_id": pair.pair_id,
            "summary_type": "selected",
            "selected_hypothesis_id": result.best.hypothesis_id,
            "selected_validation_score": result.best.validation_score,
            "selected_search_score": result.best.search_score,
            "selected_success": selected_audit["local_refined_success"],
            "selected_re": selected_audit["local_refined_re"],
            "selected_te": selected_audit["local_refined_te"],
            "raw_oracle_success": int(any(row["raw_success"] for row in variant_audits)),
            "post_refinement_oracle_success": int(any(row["gt_success"] and row["accepted"] for row in variant_audits)),
        })
    consistency_rows = support_pose_consistency(
        hypotheses_by_variant["equal_budget_no_history"],
        hypotheses_by_variant["full"],
    )
    write_csv(output_dir / "round_rejection_counts.csv", summaries)
    write_csv(output_dir / "selected_summary.csv", selected_summaries)
    write_csv(output_dir / "all_attempt_logs.csv", traces)
    write_csv(output_dir / "all_candidate_audits.csv", candidate_audits)
    write_csv(output_dir / "same_support_pose_consistency.csv", consistency_rows)
    with open(output_dir / "context.json", "w", encoding="utf-8") as handle:
        json.dump({
            "pair_index": args.pair_index,
            "pair_id": summaries[0]["pair_id"],
            "seed": int(base.seed),
            "matched_support_pose_count": len(consistency_rows),
            "all_matched_raw_poses_equal": all(row["raw_pose_equal"] for row in consistency_rows),
            "all_matched_refined_poses_equal": all(row["refined_pose_equal"] for row in consistency_rows),
        }, handle, indent=2)


if __name__ == "__main__":
    main()
