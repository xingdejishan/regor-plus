import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from test_3DLoMatch import run_experiment, set_experiment_seed
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


def output_is_empty(path):
    return not path.exists() or not any(path.iterdir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json" / "config_memory_graph_20_seed51.json"))
    args = parser.parse_args()
    base, protocol, _ = load_fixed_config(args.config_path)
    output_root = Path(protocol["output_root"])
    for name in VARIANTS:
        if not output_is_empty(output_root / name):
            raise FileExistsError(f"Refusing to overwrite formal output directory: {output_root / name}")
    for name in ("equal_budget_no_history", "full"):
        config = copy.deepcopy(base)
        config._selected_method = "memory_graph"
        config.memory_graph.update(VARIANTS[name])
        config.memory_output_dir = str(output_root / name)
        set_experiment_seed(int(config.seed))
        run_experiment(config)
    with open(output_root / "protocol.json", "w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)
    for name in VARIANTS:
        shutil.copyfile(output_root / name / "pair_results.csv", output_root / name / "pair_results_fixed.csv")
        shutil.copyfile(output_root / name / "round_logs.csv", output_root / name / "round_logs_fixed.csv")
        with open(output_root / name / "pair_results.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(base.max_pairs):
            raise RuntimeError(f"{name} wrote {len(rows)} pairs, expected {base.max_pairs}.")


if __name__ == "__main__":
    main()
