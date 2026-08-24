from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-count", type=int, default=1781)
    parser.add_argument("--sample-count", type=int, default=247)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.sample_count < 1 or args.sample_count > args.corpus_count:
        raise ValueError("invalid sample count")
    rng = np.random.default_rng(args.seed)
    indices = sorted(
        int(value)
        for value in rng.choice(args.corpus_count, size=args.sample_count, replace=False)
    )
    payload = {
        "schema_version": 1,
        "selection": "uniform_without_replacement",
        "seed": args.seed,
        "corpus_count": args.corpus_count,
        "sample_count": args.sample_count,
        "uses_scene_metadata": False,
        "uses_outcome_metadata": False,
        "indices": indices,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output)
    print(json.dumps({"output": str(output), "sample_count": len(indices)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
