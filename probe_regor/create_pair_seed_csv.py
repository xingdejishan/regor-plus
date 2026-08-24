from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=1781)
    parser.add_argument("--base-seed", type=int, default=51)
    args = parser.parse_args()
    if args.pair_count < 1:
        raise ValueError("pair-count must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_index", "pair_seed", "error"])
        for pair_index in range(args.pair_count):
            writer.writerow([pair_index, args.base_seed + pair_index, ""])


if __name__ == "__main__":
    main()
