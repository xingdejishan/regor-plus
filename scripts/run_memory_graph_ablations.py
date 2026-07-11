import argparse
import copy
import json
import sys
from pathlib import Path

from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from test_3DLoMatch import run_experiment


VARIANTS = {
    "equal_budget_no_history": {"memory_use_reliability": False, "memory_use_relation_history": False, "memory_use_basin": False},
    "posterior": {"memory_use_reliability": True, "memory_use_relation_history": False, "memory_use_basin": False},
    "posterior_graph": {"memory_use_reliability": True, "memory_use_relation_history": True, "memory_use_basin": False},
    "posterior_basin": {"memory_use_reliability": True, "memory_use_relation_history": False, "memory_use_basin": True},
    "full": {"memory_use_reliability": True, "memory_use_relation_history": True, "memory_use_basin": True},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json" / "config_3DLoMatch_FPFH_redkitchen_modified.json"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "memory_graph_ablations"))
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--include-repeated-regor", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    with open(args.config_path, encoding="utf-8") as handle:
        base = json.load(handle)
    for name in args.variants:
        config = edict(copy.deepcopy(base))
        config._selected_method = "memory_graph"
        config.memory_graph.update(VARIANTS[name])
        config.memory_output_dir = str(Path(args.output_root) / name)
        run_experiment(config)
    if args.include_repeated_regor:
        config = edict(copy.deepcopy(base))
        config.iterative_ray["method"] = "repeated_regor"
        config.iterative_ray["search_max_rounds"] = config.memory_graph["memory_max_rounds"]
        config.iterative_ray["candidates_per_round"] = config.memory_graph["memory_hypotheses_per_round"]
        config.iterative_ray["candidate_pool_multiplier"] = 1
        config.iterative_ray["enable_adaptive_stop"] = False
        config.iterative_ray["search_time_budget_seconds"] = 0
        config.output_dir = str(Path(args.output_root) / "repeated_regor_ray_auxiliary")
        run_experiment(config)


if __name__ == "__main__":
    main()
