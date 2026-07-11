import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from memory_graph_config import MemoryGraphConfig
from test_3DLoMatch import build_loader, build_r1_cache_payload, build_r1_modules, r1_cache_path, run_r1, set_experiment_seed
from memory_graph_fixed_config import load_fixed_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", default=str(REPO_ROOT / "config_json" / "config_memory_graph_20_seed51.json"))
    args = parser.parse_args()
    config, protocol, _ = load_fixed_config(args.config_path)
    memory_config = MemoryGraphConfig.from_mapping(config.memory_graph).validate()
    cache_dir = Path(config.r1_cache_dir)
    existing = sorted(cache_dir.glob("pair_*.pt")) if cache_dir.exists() else []
    if existing:
        raise FileExistsError(f"Refusing to regenerate the fixed R1 cache: {cache_dir} already contains {len(existing)} entries.")
    set_experiment_seed(int(config.seed))
    loader = build_loader(config, memory_config)
    matcher, regenerator = build_r1_modules(config)
    manifest = []
    for pair_index in range(int(config.max_pairs)):
        pair = loader.get_pair(pair_index)
        r1 = run_r1(pair, matcher, regenerator, config)
        payload = build_r1_cache_payload(pair, pair_index, r1, config)
        path = r1_cache_path(cache_dir, pair_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        import torch
        torch.save(payload, path)
        manifest.append({
            "pair_index": pair_index,
            "pair_id": pair.pair_id,
            "r1_correspondence_count": int(payload["src_corr_indices"].numel()),
            "r1_config_fingerprint": payload["metadata"]["r1_config_fingerprint"],
        })
        print(f"{pair_index + 1}/{config.max_pairs} pair_id={pair.pair_id} r1_corr={manifest[-1]['r1_correspondence_count']}")
    with open(cache_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"protocol": protocol, "pairs": manifest}, handle, indent=2)


if __name__ == "__main__":
    main()
