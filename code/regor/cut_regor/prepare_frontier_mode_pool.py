from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_G17_ARRAYS = (
    "selected_mode_first_member_indices",
    "selected_mode_member_offsets",
    "selected_mode_members",
    "selected_mode_collision_counts",
    "selected_mode_q3_masses",
    "selected_mode_borda_rank_sums",
    "generated_pose_sha256",
    "source_support_sha256",
)
ALLOWED_CACHE_ARRAYS = ("generated_trans",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--g17-dir",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-13_Q3BordaRankFusionModeTop50",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-09_Frozen_S0_Refinement_Oracle",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-15_Hard247OnlineArtifacts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-24_CutRegorG17ModePool",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    g17_dir, cache_dir, artifact_root, output_dir = (
        path.resolve() for path in (args.g17_dir, args.cache_dir, args.artifact_root, args.output_dir)
    )
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        if output_dir != REPO_ROOT / "output" / "regor" / "2026-08-24_CutRegorG17ModePool":
            raise RuntimeError("overwrite is restricted to the fixed CutRegor mode-pool directory")
        shutil.rmtree(output_dir)
    summary = json.loads((g17_dir / "summary.json").read_text(encoding="utf-8"))
    if summary["target_count"] != 247 or summary["qa"]["pose_hash_parity_count"] != 247:
        raise RuntimeError("complete reviewed G17 input is required")
    pair_paths = sorted(artifact_root.glob("*.npz"), key=lambda path: int(path.stem))
    pair_indices = [int(path.stem) for path in pair_paths]
    if len(pair_indices) != 247:
        raise RuntimeError("exact hard247 point artifact scope is required")
    output_dir.mkdir(parents=True)
    trace_dir = output_dir / "online_traces"
    trace_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), output_dir / "prepare_frontier_mode_pool.executed.py")
    records = []
    for position, pair_index in enumerate(pair_indices, start=1):
        g17_path = g17_dir / "score_traces" / f"{pair_index:04d}.npz"
        cache_path = cache_dir / "traces" / f"{pair_index:04d}.npz"
        with np.load(g17_path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]).copy() for name in ALLOWED_G17_ARRAYS}
        with np.load(cache_path, allow_pickle=False) as archive:
            cache_values = {name: np.asarray(archive[name]).copy() for name in ALLOWED_CACHE_ARRAYS}
        poses = cache_values["generated_trans"].astype(np.float32, copy=False)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4) or array_hash(poses) != str(values["generated_pose_sha256"]):
            raise RuntimeError(f"pair {pair_index}: frozen pose hash mismatch")
        representatives = values["selected_mode_first_member_indices"].astype(np.int64, copy=False)
        offsets = values["selected_mode_member_offsets"].astype(np.int64, copy=False)
        members = values["selected_mode_members"].astype(np.int64, copy=False)
        if representatives.shape != (50,) or offsets.shape != (51,) or offsets[-1] != len(members):
            raise RuntimeError(f"pair {pair_index}: invalid G17 mode layout")
        if not np.array_equal(members[offsets[:-1]], representatives):
            raise RuntimeError(f"pair {pair_index}: representative/member mismatch")
        if representatives.min() < 0 or members.min() < 0 or representatives.max() >= len(poses) or members.max() >= len(poses):
            raise RuntimeError(f"pair {pair_index}: mode index outside frozen pose pool")
        trace_path = trace_dir / f"{pair_index:04d}.npz"
        np.savez_compressed(
            trace_path,
            pair_index=np.asarray(pair_index, dtype=np.int64),
            mode_representative_poses=poses[representatives],
            mode_member_offsets=offsets,
            mode_member_poses=poses[members],
            mode_collision_counts=values["selected_mode_collision_counts"],
            mode_q3_masses=values["selected_mode_q3_masses"],
            mode_borda_rank_sums=values["selected_mode_borda_rank_sums"],
            generated_pose_sha256=values["generated_pose_sha256"],
            source_support_sha256=values["source_support_sha256"],
            g17_trace_sha256=np.asarray(sha256_file(g17_path)),
            point_artifact_sha256=np.asarray(sha256_file(artifact_root / f"{pair_index:04d}.npz")),
        )
        records.append(
            {
                "pair_index": pair_index,
                "mode_count": len(representatives),
                "member_count": len(members),
                "g17_trace_sha256": sha256_file(g17_path),
                "cache_trace_sha256": sha256_file(cache_path),
                "point_artifact_sha256": sha256_file(artifact_root / f"{pair_index:04d}.npz"),
                "output_trace_sha256": sha256_file(trace_path),
            }
        )
        if position == 1 or position % 25 == 0:
            print(f"[{position}/247] pair={pair_index}", flush=True)
    with (output_dir / "pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(
        output_dir / "manifest.json",
        {
            "status": "sanitized GT-free G17 multi-mode materialization for CutRegor Frontier Test",
            "pair_count": len(records),
            "pair_indices": pair_indices,
            "source_arrays_accessed": {
                "g17": list(ALLOWED_G17_ARRAYS),
                "frozen_cache": list(ALLOWED_CACHE_ARRAYS),
            },
            "forbidden_arrays_not_accessed": [
                "gt_trans",
                "full_pool_any_success",
                "borda_mode_member_oracle_any_success",
                "borda_first_member_any_success",
                "pose_success",
                "rre_deg",
                "rte_cm",
            ],
            "output_arrays": [
                "pair_index",
                "mode_representative_poses",
                "mode_member_offsets",
                "mode_member_poses",
                "mode_collision_counts",
                "mode_q3_masses",
                "mode_borda_rank_sums",
                "generated_pose_sha256",
                "source_support_sha256",
                "g17_trace_sha256",
                "point_artifact_sha256",
            ],
            "gt_fields_present_in_output": False,
            "candidate_origin": "complete frozen current-pair SC2 pool grouped and selected by GT-free G17 Borda mode retention",
            "tree_sha256": hashlib.sha256(
                "".join(record["output_trace_sha256"] for record in records).encode("ascii")
            ).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
