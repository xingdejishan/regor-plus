from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import io
import json
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
BASELINE_ROOT = REPO_ROOT
sys.path.insert(0, str(PACKAGE_ROOT))

from online_artifact import (
    ARTIFACT_KEYS,
    SCHEMA_VERSION,
    TREE_HASH_ALGORITHM,
    artifact_filename,
    artifact_tree_hash,
    load_online_artifact,
    read_regular_file_bytes,
    reconstruct_sampled_arrays,
    sha256_file,
    validate_manifest,
)

RAW_KEYS_USED = ("pcd", "feats", "saliency", "overlaps", "len_src")
RAW_EXPECTED_KEYS = (*RAW_KEYS_USED, "rot", "trans")
FORBIDDEN_OUTPUT_KEYS = ("rot", "trans", "gt", "gt_trans", "overlaps", "saliency")


def read_pair_seeds(path: Path, expected_count: int) -> tuple[dict[int, int], str]:
    data = read_regular_file_bytes(path)
    file_hash = hashlib.sha256(data).hexdigest()
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    header = next(reader)
    positions = {
        field: header.index(field) for field in ("pair_index", "pair_seed", "error")
    }
    rows = list(reader)
    result: dict[int, int] = {}
    for values in rows:
        pair_index = int(values[positions["pair_index"]])
        if pair_index in result:
            raise RuntimeError(f"duplicate pair index {pair_index} in {path}")
        if values[positions["error"]]:
            raise RuntimeError(f"source row {pair_index} contains an error")
        result[pair_index] = int(values[positions["pair_seed"]])
    expected = set(range(expected_count))
    if set(result) != expected:
        raise RuntimeError(
            "source pairs.csv does not contain the complete ordered corpus"
        )
    return result, file_hash


def weighted_indices(
    scores: torch.Tensor,
    num_node: int,
    random_state: np.random.RandomState,
) -> np.ndarray:
    count = int(scores.shape[0])
    if count <= num_node:
        return np.arange(count, dtype=np.int64)
    probabilities = (scores / scores.sum()).numpy().flatten()
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("PREDATOR sampling probabilities are invalid")
    return random_state.choice(
        np.arange(count), size=num_node, replace=False, p=probabilities
    )


def load_raw_online_fields(
    data: bytes, path: Path
) -> dict[str, torch.Tensor | int]:
    try:
        payload = torch.load(
            io.BytesIO(data), map_location="cpu", weights_only=False
        )
    except TypeError:
        payload = torch.load(io.BytesIO(data), map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != set(RAW_EXPECTED_KEYS):
        raise RuntimeError(
            f"raw PREDATOR artifact keys differ from the exact seven-key schema: {path}"
        )
    points = torch.as_tensor(payload["pcd"]).detach().cpu().contiguous()
    features = torch.as_tensor(payload["feats"]).detach().cpu().contiguous()
    saliency = torch.as_tensor(payload["saliency"]).detach().cpu().reshape(-1)
    overlaps = torch.as_tensor(payload["overlaps"]).detach().cpu().reshape(-1)
    source_count = int(payload["len_src"])
    total_count = int(points.shape[0])
    if points.dtype != torch.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"invalid raw pcd in {path}")
    if (
        features.dtype != torch.float32
        or features.ndim != 2
        or len(features) != total_count
    ):
        raise ValueError(f"invalid raw feats in {path}")
    if len(saliency) != total_count or len(overlaps) != total_count:
        raise ValueError(f"invalid raw sampling scores in {path}")
    if not 0 < source_count < total_count:
        raise ValueError(f"invalid raw len_src in {path}")
    for name, value in {
        "pcd": points,
        "feats": features,
        "saliency": saliency,
        "overlaps": overlaps,
    }.items():
        if not torch.is_floating_point(value) or not torch.isfinite(value).all():
            raise ValueError(f"raw {name} must contain finite floating point values")
    return {
        "pcd": points,
        "feats": features,
        "saliency": saliency,
        "overlaps": overlaps,
        "len_src": source_count,
    }


def compact_arrays(
    raw: dict[str, torch.Tensor | int],
    pair_index: int,
    pair_seed: int,
    num_node: int,
    raw_hash: str,
) -> dict[str, np.ndarray]:
    points = raw["pcd"]
    features = raw["feats"]
    saliency = raw["saliency"]
    overlaps = raw["overlaps"]
    source_count = int(raw["len_src"])
    if not all(
        torch.is_tensor(value) for value in (points, features, saliency, overlaps)
    ):
        raise TypeError("raw online fields contain invalid tensors")
    random_state = np.random.RandomState(pair_seed)
    source_indices = weighted_indices(
        overlaps[:source_count] * saliency[:source_count], num_node, random_state
    )
    target_indices = weighted_indices(
        overlaps[source_count:] * saliency[source_count:], num_node, random_state
    )
    source_index_tensor = torch.from_numpy(source_indices)
    target_index_tensor = torch.from_numpy(target_indices)
    sampled_features = torch.cat(
        (
            features[:source_count][source_index_tensor],
            features[source_count:][target_index_tensor],
        ),
        dim=0,
    ).contiguous()
    rng_state = random_state.get_state()
    return {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "pair_index": np.asarray(pair_index, dtype=np.int64),
        "pair_seed": np.asarray(pair_seed, dtype=np.int64),
        "full_pcd": points.numpy(),
        "len_src": np.asarray(source_count, dtype=np.int64),
        "source_indices": source_indices.astype(np.int32, copy=False),
        "target_indices": target_indices.astype(np.int32, copy=False),
        "sampled_feats": sampled_features.numpy(),
        "sampled_len_src": np.asarray(len(source_indices), dtype=np.int64),
        "numpy_rng_state_algorithm": np.asarray(rng_state[0]),
        "numpy_rng_state_keys": np.asarray(rng_state[1], dtype=np.uint32),
        "numpy_rng_state_pos": np.asarray(rng_state[2], dtype=np.int64),
        "numpy_rng_state_has_gauss": np.asarray(rng_state[3], dtype=np.int64),
        "numpy_rng_state_cached_gaussian": np.asarray(rng_state[4], dtype=np.float64),
        "raw_artifact_sha256": np.asarray(raw_hash),
    }


def save_artifact_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    if set(arrays) != set(ARTIFACT_KEYS):
        raise RuntimeError("compact arrays differ from the strict artifact allowlist")
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def verify_compaction(
    artifact_path: Path,
    expected_arrays: dict[str, np.ndarray],
    raw: dict[str, torch.Tensor | int],
    pair_index: int,
    pair_seed: int,
    num_node: int,
) -> dict[str, np.ndarray | int | float | str]:
    with np.load(artifact_path, allow_pickle=False) as archive:
        mismatched_keys = [
            key
            for key in ARTIFACT_KEYS
            if not np.array_equal(
                np.asarray(archive[key]), np.asarray(expected_arrays[key])
            )
        ]
    if mismatched_keys:
        raise RuntimeError(
            f"compact artifact round-trip mismatch for pair {pair_index}: "
            f"{mismatched_keys}"
        )
    compact = load_online_artifact(
        artifact_path,
        expected_pair_index=pair_index,
        expected_pair_seed=pair_seed,
        num_node=num_node,
    )
    reconstructed = reconstruct_sampled_arrays(compact)
    points = raw["pcd"]
    features = raw["feats"]
    source_count = int(raw["len_src"])
    if not torch.is_tensor(points) or not torch.is_tensor(features):
        raise TypeError("raw compaction inputs contain invalid tensors")
    source_indices = torch.from_numpy(reconstructed["source_indices"])
    target_indices = torch.from_numpy(reconstructed["target_indices"])
    expected_source_points = points[:source_count][source_indices].numpy()
    expected_target_points = points[source_count:][target_indices].numpy()
    expected_source_features = features[:source_count][source_indices].numpy()
    expected_target_features = features[source_count:][target_indices].numpy()
    comparisons = (
        (reconstructed["source_points"], expected_source_points),
        (reconstructed["target_points"], expected_target_points),
        (reconstructed["source_features"], expected_source_features),
        (reconstructed["target_features"], expected_target_features),
    )
    if not all(np.array_equal(actual, expected) for actual, expected in comparisons):
        raise RuntimeError(
            f"compact artifact numeric parity failed for pair {pair_index}"
        )
    return compact


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="ascii")
    os.replace(temporary, path)


def acquire_generation_lock(output_dir: Path) -> tuple[Path, str]:
    lock_path = output_dir.parent / f".{output_dir.name}.generation.lock"
    token = secrets.token_hex(16)
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"online artifact generation lock already exists: {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "token": token,
                    "pid": os.getpid(),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "output_dir": str(output_dir),
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return lock_path, token


def release_generation_lock(lock_path: Path, token: str) -> None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if payload.get("token") == token:
            lock_path.unlink()
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return


def remove_internal_directory(path: Path, parent: Path, prefix: str) -> None:
    resolved_parent = parent.resolve()
    resolved_path = path.resolve()
    if (
        resolved_path.parent != resolved_parent
        or not resolved_path.name.startswith(prefix)
        or path.is_symlink()
    ):
        raise RuntimeError(f"refusing to remove unsafe internal directory: {path}")
    if path.is_dir():
        shutil.rmtree(path)


def create_staging_directory(output_dir: Path, token: str) -> Path:
    parent = output_dir.parent
    prefix = f".{output_dir.name}.staging."
    for stale in parent.glob(f"{prefix}*"):
        remove_internal_directory(stale, parent, prefix)
    staging_dir = parent / f"{prefix}{token}"
    staging_dir.mkdir(parents=False, exist_ok=False)
    return staging_dir


def publish_staging_directory(
    staging_dir: Path, output_dir: Path, token: str
) -> None:
    parent = output_dir.parent
    backup_prefix = f".{output_dir.name}.backup."
    for stale in parent.glob(f"{backup_prefix}*"):
        remove_internal_directory(stale, parent, backup_prefix)
    backup_dir = parent / f"{backup_prefix}{token}"
    had_previous = output_dir.exists()
    if had_previous:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise RuntimeError(f"output path is not a regular directory: {output_dir}")
        os.replace(output_dir, backup_dir)
    try:
        os.replace(staging_dir, output_dir)
    except Exception:
        if had_previous and backup_dir.is_dir() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise
    if had_previous:
        remove_internal_directory(backup_dir, parent, backup_prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=BASELINE_ROOT
        / "external"
        / "OverlapPredator"
        / "snapshot"
        / "indoor"
        / "3DLoMatch",
    )
    parser.add_argument(
        "--source-pairs-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-count", type=int, default=1781)
    parser.add_argument("--base-seed", type=int, default=51)
    parser.add_argument("--num-node", type=int, default=5000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.resume:
        raise ValueError(
            "resume is forbidden for sealed online artifacts; restart with --overwrite"
        )
    if not args.overwrite:
        raise ValueError("sealed online artifacts require a fresh --overwrite run")
    if args.expected_count < 1 or args.num_node < 1:
        raise ValueError("expected-count and num-node must be positive")
    raw_data_root = args.raw_data_root.resolve()
    source_pairs_path = args.source_pairs_path.resolve()
    output_dir = args.output_dir.resolve()
    if raw_data_root == output_dir:
        raise ValueError("raw-data-root and output-dir must differ")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    generation_lock_path, generation_lock_token = acquire_generation_lock(output_dir)
    atexit.register(
        release_generation_lock, generation_lock_path, generation_lock_token
    )
    staging_dir = create_staging_directory(output_dir, generation_lock_token)
    staging_prefix = f".{output_dir.name}.staging."
    atexit.register(
        remove_internal_directory,
        staging_dir,
        output_dir.parent,
        staging_prefix,
    )
    pair_seeds, source_pairs_hash = read_pair_seeds(
        source_pairs_path, args.expected_count
    )
    invalid_seeds = [
        index
        for index, pair_seed in pair_seeds.items()
        if pair_seed != args.base_seed + index
    ]
    if invalid_seeds:
        raise RuntimeError(f"pair seed protocol mismatch: {invalid_seeds}")
    start = max(0, args.start_index)
    stop = (
        args.expected_count
        if args.limit is None
        else min(args.expected_count, start + max(args.limit, 0))
    )
    target_indices = list(range(start, stop))
    if not target_indices:
        raise ValueError("empty target set")

    raw_entries = []
    for pair_index in target_indices:
        raw_path = raw_data_root / f"{pair_index}.pth"
        raw_bytes = read_regular_file_bytes(raw_path)
        raw_entries.append(
            {
                "pair_index": pair_index,
                "file": raw_path.name,
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "byte_size": len(raw_bytes),
            }
        )
    raw_entries_by_index = {int(entry["pair_index"]): entry for entry in raw_entries}

    protocol_path = staging_dir / "protocol.json"
    manifest_path = staging_dir / "manifest.json"
    sidecar_path = staging_dir / "manifest.sha256"

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "shared_loader": str((PACKAGE_ROOT / "online_artifact.py").resolve()),
        "shared_loader_sha256": sha256_file(
            PACKAGE_ROOT / "online_artifact.py"
        ),
        "raw_data_root": str(raw_data_root),
        "raw_input_manifest": raw_entries,
        "raw_input_tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "raw_input_tree_sha256": artifact_tree_hash(raw_entries),
        "source_pairs_path": str(source_pairs_path),
        "source_pairs_sha256": source_pairs_hash,
        "source_pairs_fields_read": ["pair_index", "pair_seed", "error"],
        "raw_artifact_keys_used": list(RAW_KEYS_USED),
        "raw_artifact_expected_keys": list(RAW_EXPECTED_KEYS),
        "artifact_keys": list(ARTIFACT_KEYS),
        "forbidden_output_keys": list(FORBIDDEN_OUTPUT_KEYS),
        "pair_indices": target_indices,
        "expected_corpus_count": args.expected_count,
        "base_seed": args.base_seed,
        "num_node": args.num_node,
        "fresh_only": True,
        "overwrite_replaces_only_after_complete": True,
        "generation_lock_path": str(generation_lock_path),
        "generation_lock_atomic_o_excl": True,
        "staging_atomic_publish": True,
        "final_output_dir": str(output_dir),
        "gt_boundary": {
            "stage": "offline_physical_redaction",
            "raw_payload_deserialization_may_include_rot_trans": True,
            "raw_rot_trans_used_for_sampling_or_features": False,
            "rot_trans_or_gt_written_to_online_artifacts": False,
            "online_artifact_keys_are_strict_allowlist": True,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json_atomic(protocol_path, protocol)

    entries: list[dict[str, object]] = []
    run_started = time.perf_counter()
    for position, pair_index in enumerate(target_indices, start=1):
        pair_started = time.perf_counter()
        pair_seed = pair_seeds[pair_index]
        raw_path = raw_data_root / f"{pair_index}.pth"
        output_path = staging_dir / artifact_filename(pair_index)
        raw_hash = str(raw_entries_by_index[pair_index]["sha256"])
        raw_bytes = read_regular_file_bytes(raw_path)
        if hashlib.sha256(raw_bytes).hexdigest() != raw_hash:
            raise RuntimeError(
                f"raw artifact changed before load for pair {pair_index}"
            )
        raw = load_raw_online_fields(raw_bytes, raw_path)
        if sha256_file(raw_path) != raw_hash:
            raise RuntimeError(
                f"raw artifact changed during load for pair {pair_index}"
            )
        arrays = compact_arrays(raw, pair_index, pair_seed, args.num_node, raw_hash)
        save_artifact_atomic(output_path, arrays)
        compact = verify_compaction(
            output_path, arrays, raw, pair_index, pair_seed, args.num_node
        )
        entry = {
            "pair_index": pair_index,
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "raw_artifact_sha256": raw_hash,
            "byte_size": output_path.stat().st_size,
            "full_count": len(np.asarray(compact["full_pcd"])),
            "source_count": int(compact["len_src"]),
            "sampled_source_count": len(np.asarray(compact["source_indices"])),
            "sampled_target_count": len(np.asarray(compact["target_indices"])),
        }
        entries.append(entry)
        if position == 1 or position % max(args.progress_every, 1) == 0:
            print(
                json.dumps(
                    {
                        "pair": pair_index,
                        "position": position,
                        "target": len(target_indices),
                        "artifact_mb": entry["byte_size"] / (1024 * 1024),
                        "runtime_sec": time.perf_counter() - pair_started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    total_bytes = sum(int(entry["byte_size"]) for entry in entries)
    if protocol["script_sha256"] != sha256_file(Path(__file__).resolve()) or protocol[
        "shared_loader_sha256"
    ] != sha256_file(PACKAGE_ROOT / "online_artifact.py"):
        raise RuntimeError("artifact implementation changed during generation")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "shared_loader": str((PACKAGE_ROOT / "online_artifact.py").resolve()),
        "shared_loader_sha256": sha256_file(
            PACKAGE_ROOT / "online_artifact.py"
        ),
        "protocol_sha256": sha256_file(protocol_path),
        "raw_data_root": str(raw_data_root),
        "raw_input_tree_sha256": protocol["raw_input_tree_sha256"],
        "source_pairs_path": str(source_pairs_path),
        "source_pairs_sha256": sha256_file(source_pairs_path),
        "source_pairs_fields_read": ["pair_index", "pair_seed", "error"],
        "raw_artifact_keys_used": list(RAW_KEYS_USED),
        "raw_artifact_expected_keys": list(RAW_EXPECTED_KEYS),
        "artifact_keys": list(ARTIFACT_KEYS),
        "forbidden_output_keys": list(FORBIDDEN_OUTPUT_KEYS),
        "pair_count": len(target_indices),
        "pair_indices": target_indices,
        "strict_complete": target_indices == list(range(args.expected_count)),
        "expected_corpus_count": args.expected_count,
        "base_seed": args.base_seed,
        "num_node": args.num_node,
        "artifacts": entries,
        "total_bytes": total_bytes,
        "tree_hash_algorithm": TREE_HASH_ALGORITHM,
        "tree_sha256": artifact_tree_hash(entries),
        "runtime_sec": time.perf_counter() - run_started,
        "gt_boundary": {
            "stage": "offline_physical_redaction",
            "raw_payload_deserialization_may_include_rot_trans": True,
            "raw_rot_trans_used_for_sampling_or_features": False,
            "rot_trans_or_gt_written_to_online_artifacts": False,
            "online_artifact_keys_are_strict_allowlist": True,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json_atomic(manifest_path, manifest)
    write_text_atomic(sidecar_path, sha256_file(manifest_path) + "\n")
    publish_staging_directory(staging_dir, output_dir, generation_lock_token)
    print(json.dumps({key: manifest[key] for key in ("pair_count", "strict_complete", "total_bytes", "tree_sha256", "runtime_sec")}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
