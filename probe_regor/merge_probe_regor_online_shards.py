from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=1781)
    args = parser.parse_args()
    shard_root = args.shard_root.resolve()
    output_dir = args.output_dir.resolve()
    rows_by_index: dict[int, dict[str, object]] = {}
    protocol_records = []
    reference_fields = None
    for shard in sorted(path for path in shard_root.iterdir() if path.is_dir()):
        protocol_path = shard / "method_protocol.json"
        summary_path = shard / "run_summary.json"
        if not protocol_path.is_file() or not summary_path.is_file():
            raise RuntimeError(f"incomplete shard: {shard}")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary["error_count"]) != 0:
            raise RuntimeError(f"shard contains online errors: {shard}")
        fields = {
            key: protocol[key]
            for key in (
                "method",
                "method_config",
                "label_isolation",
                "implementation_hashes",
                "base_seed",
                "split_index",
            )
        }
        if reference_fields is None:
            reference_fields = fields
        elif fields != reference_fields:
            raise RuntimeError(f"shard protocol mismatch: {shard}")
        protocol_records.append(
            {
                "shard": shard.name,
                "protocol_sha256": sha256_file(protocol_path),
                "summary_sha256": sha256_file(summary_path),
            }
        )
        for row_path in sorted((shard / "pair_rows").glob("*.json")):
            row = json.loads(row_path.read_text(encoding="utf-8"))
            pair_index = int(row["pair_index"])
            if pair_index in rows_by_index:
                raise RuntimeError(f"duplicate pair row: {pair_index}")
            rows_by_index[pair_index] = row
    expected = list(range(args.pair_count))
    if sorted(rows_by_index) != expected:
        missing = sorted(set(expected) - set(rows_by_index))
        raise RuntimeError(f"shards do not cover the full pair set: {missing[:20]}")
    rows = [rows_by_index[index] for index in expected]
    errors = [row for row in rows if row.get("error")]
    if errors:
        raise RuntimeError(f"cannot merge {len(errors)} errored rows")
    row_dir = output_dir / "pair_rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        atomic_json(row_dir / f"{int(row['pair_index']):04d}.json", row)
    baseline = np.asarray([row["baseline_pose"] for row in rows], dtype=np.float64).reshape(-1, 4, 4)
    probe = np.asarray([row["probe_pose"] for row in rows], dtype=np.float64).reshape(-1, 4, 4)
    atomic_npz(output_dir / "final_poses.npz", baseline=baseline, probe=probe)
    protocol = {
        "schema_version": 1,
        "pair_count": args.pair_count,
        "target_indices": expected,
        **(reference_fields or {}),
        "shards": protocol_records,
    }
    atomic_json(output_dir / "method_protocol.json", protocol)
    summary = {
        "pair_count": len(rows),
        "error_count": 0,
        "finite_baseline_count": int(np.isfinite(baseline).all(axis=(1, 2)).sum()),
        "finite_probe_count": int(np.isfinite(probe).all(axis=(1, 2)).sum()),
        "pose_file_sha256": sha256_file(output_dir / "final_poses.npz"),
        "shard_count": len(protocol_records),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
