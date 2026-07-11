import argparse
import copy
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from memory_graph_config import MemoryGraphConfig
from memory_guided_registration import MemoryGuidedRegistration
from test_3DLoMatch import build_loader, load_r1_cache_payload, load_required_r1_cache, pose_errors, set_experiment_seed
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json" / "config_memory_graph_20_seed51.json"))
    parser.add_argument("--pair-index", type=int, default=1)
    args = parser.parse_args()
    base, protocol, _ = load_fixed_config(args.config_path)
    output_dir = Path(protocol["output_root"]) / f"debug_pair_{args.pair_index:06d}_soft_structure"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite debug output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    traces = []
    summaries = []
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
    write_csv(output_dir / "round_rejection_counts.csv", summaries)
    write_csv(output_dir / "all_attempt_logs.csv", traces)
    with open(output_dir / "context.json", "w", encoding="utf-8") as handle:
        json.dump({"pair_index": args.pair_index, "pair_id": summaries[0]["pair_id"], "seed": int(base.seed)}, handle, indent=2)


if __name__ == "__main__":
    main()
