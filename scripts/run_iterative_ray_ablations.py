import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict as edict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from test_3DLoMatch import run_experiment


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json" / "config_3DLoMatch_FPFH_redkitchen_modified.json"))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs" / "iterative_ray_ablations"))
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--methods", nargs="+", default=["r1_only", "repeated_regor", "iterative_ray", "shuffled_ray"])
    args = parser.parse_args()
    with open(args.config_path, "r", encoding="utf-8") as handle:
        base = json.load(handle)
    for method in args.methods:
        reset_seed(args.seed)
        config = edict(copy.deepcopy(base))
        config.iterative_ray["method"] = method
        config.output_dir = str(Path(args.output_root) / method)
        run_experiment(config)


if __name__ == "__main__":
    main()
