from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = REPO_ROOT
BASELINE_SNAPSHOT = PACKAGE_ROOT / "baseline"
sys.path[:0] = [str(BASELINE_SNAPSHOT), str(PACKAGE_ROOT), str(BASELINE_ROOT)]
os.chdir(BASELINE_ROOT)

from Correspondence_regenerate_v2 import Regenerator
from online_artifact import artifact_filename, load_online_artifact, reconstruct_sampled_arrays, restore_numpy_rng_state, validate_manifest
from initial_matching_plus import Matcher_plus
from probe_regor import ProbeRegor, ProbeRegorConfig
from Tranformation_estimaton import Estimator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_pair_indices(path: Path, pair_count: int) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = [int(value) for value in payload["indices"]]
    if indices != sorted(set(indices)):
        raise RuntimeError("pair-index split must be sorted and unique")
    if not indices or indices[0] < 0 or indices[-1] >= pair_count:
        raise RuntimeError("pair-index split is outside the corpus")
    return indices


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def set_pair_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_matcher() -> Matcher_plus:
    return Matcher_plus(
        inlier_threshold=0.1,
        num_node="all",
        use_mutual=False,
        d_thre=0.1,
        num_iterations=10,
        ratio=0.2,
        nms_radius=0.1,
        max_points=8000,
        k1=60,
        k2=50,
        FS_TCD_thre=0.05,
        relax_match_num=100,
        NS_by_IC=50,
    )


class ShardedOnlineLoader:
    def __init__(self, head_root: Path, tail_root: Path, split_index: int, base_seed: int) -> None:
        self.head_root = head_root
        self.tail_root = tail_root
        self.split_index = split_index
        self.base_seed = base_seed

    def path(self, pair_index: int) -> Path:
        root = self.head_root if pair_index < self.split_index else self.tail_root
        return root / artifact_filename(pair_index)

    def load(self, pair_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
        path = self.path(pair_index)
        artifact = load_online_artifact(
            path,
            expected_pair_index=pair_index,
            expected_pair_seed=self.base_seed + pair_index,
            num_node=5000,
        )
        sampled = reconstruct_sampled_arrays(artifact)
        restore_numpy_rng_state(artifact)
        return (
            torch.from_numpy(sampled["source_points"]).cuda()[None],
            torch.from_numpy(sampled["target_points"]).cuda()[None],
            torch.from_numpy(sampled["source_features"]).cuda()[None],
            torch.from_numpy(sampled["target_features"]).cuda()[None],
            sha256_file(path),
        )


def baseline_pipeline(
    source: torch.Tensor,
    target: torch.Tensor,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    placeholder: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    matcher = create_matcher()
    regenerator = Regenerator()
    estimator = Estimator(num_node=5000)
    pose, source_filtered, target_filtered, _, _, _, _ = matcher.estimator(
        source, target, source_features, target_features, placeholder
    )
    selected = np.random.choice(source_filtered.shape[1], 100)
    source_corr = source_filtered[:, selected]
    target_corr = target_filtered[:, selected]
    counts = [int(source_filtered.shape[1])]
    source_corr, target_corr, _ = regenerator.regenerate(
        source_corr,
        target_corr,
        source,
        target,
        source_features,
        target_features,
        placeholder,
        knn_num=100,
        sampling_num=100,
    )
    counts.append(int(source_corr.shape[1]))
    source_corr, target_corr, _ = regenerator.regenerate(
        source_corr,
        target_corr,
        source,
        target,
        source_features,
        target_features,
        placeholder,
        knn_num=20,
        sampling_num=500,
    )
    counts.append(int(source_corr.shape[1]))
    final_pose, _, _, _ = estimator.estimator(
        source_corr, target_corr, source, target, placeholder
    )
    return final_pose, counts


def probe_pipeline(
    source: torch.Tensor,
    target: torch.Tensor,
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    placeholder: torch.Tensor,
):
    matcher = create_matcher()
    initial_pose, source_filtered, target_filtered, _, _, _, _ = matcher.estimator(
        source, target, source_features, target_features, placeholder
    )
    method = ProbeRegor(
        Regenerator(),
        Estimator(num_node=5000),
        ProbeRegorConfig(),
    )
    return method.run(
        source_filtered,
        target_filtered,
        source,
        target,
        source_features,
        target_features,
        initial_pose,
        placeholder,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--head-root",
        type=Path,
        required=True,
    )
    parser.add_argument("--tail-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--split-index", type=int, default=833)
    parser.add_argument("--pair-count", type=int, default=1781)
    parser.add_argument("--base-seed", type=int, default=51)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--index-file", type=Path)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.head_root = resolve_project_path(args.head_root)
    args.tail_root = resolve_project_path(args.tail_root)
    args.output_dir = resolve_project_path(args.output_dir)
    if args.index_file is not None:
        if args.start_index != 0 or args.limit is not None:
            raise ValueError("--index-file cannot be combined with --start-index or --limit")
        args.index_file = resolve_project_path(args.index_file)
        target_indices = load_pair_indices(args.index_file, args.pair_count)
    else:
        start = max(args.start_index, 0)
        stop = args.pair_count if args.limit is None else min(args.pair_count, start + args.limit)
        target_indices = list(range(start, stop))
    if not target_indices:
        raise ValueError("empty pair range")
    if any(index < args.split_index for index in target_indices):
        validate_manifest(
            args.head_root / "manifest.json",
            expected_indices=list(range(0, args.split_index)),
            expected_base_seed=args.base_seed,
            expected_num_node=5000,
            verify_files=True,
        )
    if any(index >= args.split_index for index in target_indices):
        validate_manifest(
            args.tail_root / "manifest.json",
            expected_indices=list(range(args.split_index, args.pair_count)),
            expected_base_seed=args.base_seed,
            expected_num_node=5000,
            verify_files=True,
        )
    loader = ShardedOnlineLoader(args.head_root, args.tail_root, args.split_index, args.base_seed)
    output_dir = args.output_dir
    row_dir = output_dir / "pair_rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "pair_count": args.pair_count,
        "target_indices": target_indices,
        "split_index": args.split_index,
        "head_root": str(args.head_root),
        "tail_root": str(args.tail_root),
        "base_seed": args.base_seed,
        "index_file": str(args.index_file) if args.index_file is not None else None,
        "index_file_sha256": sha256_file(args.index_file) if args.index_file is not None else None,
        "method": "ProbeRegor",
        "method_config": ProbeRegorConfig().__dict__,
        "label_isolation": {
            "online_artifact_allowlist": True,
            "reference_transform_absent": True,
            "reference_correspondence_absent": True,
            "scene_metadata_absent": True,
            "raw_predator_payload_absent": True,
            "identity_placeholder_for_frozen_calls": True,
        },
        "implementation_hashes": {
            path.name: sha256_file(path)
            for path in [
                Path(__file__).resolve(),
                PACKAGE_ROOT / "probe_regor.py",
                BASELINE_SNAPSHOT / "initial_matching_plus.py",
                BASELINE_SNAPSHOT / "Correspondence_regenerate_v2.py",
                BASELINE_SNAPSHOT / "Tranformation_estimaton.py",
            ]
        },
    }
    atomic_json(output_dir / "method_protocol.json", protocol)
    run_started = time.perf_counter()
    for position, pair_index in enumerate(target_indices, start=1):
        row_path = row_dir / f"{pair_index:04d}.json"
        if row_path.is_file():
            continue
        pair_seed = args.base_seed + pair_index
        row: dict[str, object] = {"pair_index": pair_index, "pair_seed": pair_seed, "error": ""}
        fatal_message = ""
        try:
            set_pair_seed(pair_seed)
            source, target, source_features, target_features, artifact_hash = loader.load(pair_index)
            placeholder = torch.eye(4, dtype=source.dtype, device=source.device)[None]
            baseline_started = time.perf_counter()
            with torch.no_grad():
                baseline_pose, baseline_counts = baseline_pipeline(
                    source, target, source_features, target_features, placeholder
                )
            baseline_runtime = time.perf_counter() - baseline_started
            del source, target, source_features, target_features
            torch.cuda.empty_cache()
            set_pair_seed(pair_seed)
            source, target, source_features, target_features, second_hash = loader.load(pair_index)
            if second_hash != artifact_hash:
                raise RuntimeError("online artifact changed between paired arms")
            placeholder = torch.eye(4, dtype=source.dtype, device=source.device)[None]
            probe_started = time.perf_counter()
            with torch.no_grad():
                probe = probe_pipeline(
                    source, target, source_features, target_features, placeholder
                )
            probe_runtime = time.perf_counter() - probe_started
            baseline_array = baseline_pose.detach().cpu().numpy()[0]
            probe_array = probe.pose.detach().cpu().numpy()[0]
            row.update(
                {
                    "artifact_sha256": artifact_hash,
                    "baseline_pose": baseline_array.reshape(-1).tolist(),
                    "probe_pose": probe_array.reshape(-1).tolist(),
                    "baseline_counts": baseline_counts,
                    "probe_rounds": probe.rounds,
                    "probe_line_counts": probe.line_counts,
                    "probe_matched_counts": probe.matched_counts,
                    "probe_action_counts": probe.action_counts,
                    "probe_correspondence_counts": probe.correspondence_counts,
                    "probe_r_plus": probe.r_plus,
                    "probe_r_minus": probe.r_minus,
                    "probe_posterior_ess": probe.posterior_ess,
                    "baseline_runtime_sec": baseline_runtime,
                    "probe_runtime_sec": probe_runtime,
                    "baseline_finite": bool(np.isfinite(baseline_array).all()),
                    "probe_finite": bool(np.isfinite(probe_array).all()),
                }
            )
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            row["traceback"] = traceback.format_exc()
            row["baseline_pose"] = np.full(16, np.nan).tolist()
            row["probe_pose"] = np.full(16, np.nan).tolist()
            lowered = str(error).lower()
            if any(
                token in lowered
                for token in (
                    "cuda error",
                    "cuda runtime error",
                    "device-side assert",
                    "cublas",
                    "cusolver",
                    "cudnn",
                )
            ):
                fatal_message = str(row["error"])
        atomic_json(row_path, row)
        if fatal_message:
            raise RuntimeError(
                f"fatal CUDA context failure at pair {pair_index}: {fatal_message}"
            )
        if position == 1 or position % max(args.progress_every, 1) == 0:
            print(
                json.dumps(
                    {
                        "position": position,
                        "target": len(target_indices),
                        "pair_index": pair_index,
                        "error": row["error"],
                        "elapsed_sec": time.perf_counter() - run_started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    rows = [json.loads((row_dir / f"{index:04d}.json").read_text(encoding="utf-8")) for index in target_indices]
    baseline_poses = np.asarray([row["baseline_pose"] for row in rows], dtype=np.float64).reshape(-1, 4, 4)
    probe_poses = np.asarray([row["probe_pose"] for row in rows], dtype=np.float64).reshape(-1, 4, 4)
    atomic_npz(output_dir / "final_poses.npz", baseline=baseline_poses, probe=probe_poses)
    summary = {
        "pair_count": len(rows),
        "error_count": sum(bool(row["error"]) for row in rows),
        "finite_baseline_count": int(np.isfinite(baseline_poses).all(axis=(1, 2)).sum()),
        "finite_probe_count": int(np.isfinite(probe_poses).all(axis=(1, 2)).sum()),
        "runtime_sec": time.perf_counter() - run_started,
        "pose_file_sha256": sha256_file(output_dir / "final_poses.npz"),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
