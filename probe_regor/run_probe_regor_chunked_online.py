from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--tail-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=1781)
    parser.add_argument("--chunk-size", type=int, default=250)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    tail_root = args.tail_root.resolve()
    output_dir = args.output_dir.resolve()
    shard_root = output_dir.parent / "online_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().with_name("run_probe_regor_natural1781.py")
    merger = Path(__file__).resolve().with_name("merge_probe_regor_online_shards.py")
    for start in range(0, args.pair_count, args.chunk_size):
        stop = min(args.pair_count, start + args.chunk_size)
        shard = shard_root / f"shard_{start:04d}_{stop - 1:04d}"
        command = [
            sys.executable,
            str(runner),
            "--head-root",
            str(args.head_root.resolve()),
            "--tail-root",
            str(tail_root),
            "--output-dir",
            str(shard),
            "--pair-count",
            str(args.pair_count),
            "--start-index",
            str(start),
            "--limit",
            str(stop - start),
            "--progress-every",
            str(args.progress_every),
        ]
        result = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
    result = subprocess.run(
        [
            sys.executable,
            str(merger),
            "--shard-root",
            str(shard_root),
            "--output-dir",
            str(output_dir),
            "--pair-count",
            str(args.pair_count),
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
