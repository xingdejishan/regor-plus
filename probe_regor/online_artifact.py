from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
ARTIFACT_SUFFIX = ".npz"
ARTIFACT_KEYS = (
    "schema_version",
    "pair_index",
    "pair_seed",
    "full_pcd",
    "len_src",
    "source_indices",
    "target_indices",
    "sampled_feats",
    "sampled_len_src",
    "numpy_rng_state_algorithm",
    "numpy_rng_state_keys",
    "numpy_rng_state_pos",
    "numpy_rng_state_has_gauss",
    "numpy_rng_state_cached_gaussian",
    "raw_artifact_sha256",
)
TREE_HASH_ALGORITHM = (
    "sha256(concat(utf8(file + '\\0' + sha256 + '\\n'))) sorted by pair_index"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_regular_file_bytes(path: Path) -> bytes:
    metadata = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise RuntimeError(f"sealed input must be a non-symlink regular file: {path}")
    return path.read_bytes()


def artifact_filename(pair_index: int) -> str:
    return f"{pair_index:04d}{ARTIFACT_SUFFIX}"


def artifact_tree_hash(entries: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda value: int(value["pair_index"])):
        digest.update(f"{entry['file']}\0{entry['sha256']}\n".encode())
    return digest.hexdigest()


def _integer_scalar(value: np.ndarray, name: str, path: Path) -> int:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer scalar in {path}")
    return int(array)


def _string_scalar(value: np.ndarray, name: str, path: Path) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "SU":
        raise ValueError(f"{name} must be a string scalar in {path}")
    return str(array)


def load_online_artifact(
    path: Path,
    *,
    expected_pair_index: int | None = None,
    expected_pair_seed: int | None = None,
    num_node: int = 5000,
) -> dict[str, np.ndarray | int | float | str]:
    data = read_regular_file_bytes(path)
    return _load_online_artifact_bytes(
        data,
        path,
        expected_pair_index=expected_pair_index,
        expected_pair_seed=expected_pair_seed,
        num_node=num_node,
    )


def _load_online_artifact_bytes(
    data: bytes,
    path: Path,
    *,
    expected_pair_index: int | None = None,
    expected_pair_seed: int | None = None,
    num_node: int = 5000,
) -> dict[str, np.ndarray | int | float | str]:
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(ARTIFACT_KEYS) or len(archive.files) != len(
            ARTIFACT_KEYS
        ):
            raise RuntimeError(
                f"online artifact keys differ from the strict allowlist in {path}: "
                f"{sorted(archive.files)}"
            )
        schema_version = _integer_scalar(
            archive["schema_version"], "schema_version", path
        )
        pair_index = _integer_scalar(archive["pair_index"], "pair_index", path)
        pair_seed = _integer_scalar(archive["pair_seed"], "pair_seed", path)
        len_src = _integer_scalar(archive["len_src"], "len_src", path)
        sampled_len_src = _integer_scalar(
            archive["sampled_len_src"], "sampled_len_src", path
        )
        rng_algorithm = _string_scalar(
            archive["numpy_rng_state_algorithm"],
            "numpy_rng_state_algorithm",
            path,
        )
        raw_hash = _string_scalar(
            archive["raw_artifact_sha256"], "raw_artifact_sha256", path
        )
        full_pcd = np.asarray(archive["full_pcd"])
        source_indices = np.asarray(archive["source_indices"])
        target_indices = np.asarray(archive["target_indices"])
        sampled_feats = np.asarray(archive["sampled_feats"])
        rng_keys = np.asarray(archive["numpy_rng_state_keys"])
        rng_pos = _integer_scalar(
            archive["numpy_rng_state_pos"], "numpy_rng_state_pos", path
        )
        rng_has_gauss = _integer_scalar(
            archive["numpy_rng_state_has_gauss"],
            "numpy_rng_state_has_gauss",
            path,
        )
        rng_cached_gaussian_array = np.asarray(
            archive["numpy_rng_state_cached_gaussian"]
        )
        if (
            rng_cached_gaussian_array.shape != ()
            or rng_cached_gaussian_array.dtype.kind != "f"
        ):
            raise ValueError(
                f"numpy_rng_state_cached_gaussian must be a float scalar in {path}"
            )
        rng_cached_gaussian = float(rng_cached_gaussian_array)

    if schema_version != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported online artifact schema in {path}")
    if expected_pair_index is not None and pair_index != expected_pair_index:
        raise RuntimeError(f"online artifact pair index mismatch in {path}")
    if expected_pair_seed is not None and pair_seed != expected_pair_seed:
        raise RuntimeError(f"online artifact pair seed mismatch in {path}")
    if num_node < 1:
        raise ValueError("num_node must be positive")
    if full_pcd.dtype != np.float32 or full_pcd.ndim != 2 or full_pcd.shape[1] != 3:
        raise ValueError(f"full_pcd must have float32 shape [N,3] in {path}")
    if not 0 < len_src < len(full_pcd):
        raise ValueError(f"invalid len_src in {path}")
    if not np.all(np.isfinite(full_pcd)):
        raise ValueError(f"full_pcd contains non-finite values in {path}")
    for name, indices, count in (
        ("source_indices", source_indices, len_src),
        ("target_indices", target_indices, len(full_pcd) - len_src),
    ):
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise ValueError(
                f"{name} must be a one-dimensional integer array in {path}"
            )
        if len(indices) != min(count, num_node):
            raise ValueError(f"{name} length violates num_node in {path}")
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= count):
            raise ValueError(f"{name} is out of bounds in {path}")
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{name} contains duplicates in {path}")
        if count <= num_node and not np.array_equal(
            indices, np.arange(count, dtype=indices.dtype)
        ):
            raise ValueError(
                f"{name} must preserve source order without sampling in {path}"
            )
    if sampled_len_src != len(source_indices):
        raise ValueError(f"sampled_len_src mismatch in {path}")
    expected_sample_count = len(source_indices) + len(target_indices)
    if (
        sampled_feats.dtype != np.float32
        or sampled_feats.ndim != 2
        or len(sampled_feats) != expected_sample_count
    ):
        raise ValueError(f"sampled_feats has an invalid shape or dtype in {path}")
    if not np.all(np.isfinite(sampled_feats)):
        raise ValueError(f"sampled_feats contains non-finite values in {path}")
    if rng_algorithm != "MT19937":
        raise ValueError(f"unsupported NumPy RNG algorithm in {path}")
    if rng_keys.dtype != np.uint32 or rng_keys.shape != (624,):
        raise ValueError(f"invalid NumPy RNG keys in {path}")
    if not 0 <= rng_pos <= 624 or rng_has_gauss not in (0, 1):
        raise ValueError(f"invalid NumPy RNG state in {path}")
    if not np.isfinite(rng_cached_gaussian):
        raise ValueError(f"invalid NumPy cached Gaussian in {path}")
    if len(raw_hash) != 64 or any(
        character not in "0123456789abcdef" for character in raw_hash
    ):
        raise ValueError(f"invalid raw artifact SHA-256 in {path}")
    return {
        "schema_version": schema_version,
        "pair_index": pair_index,
        "pair_seed": pair_seed,
        "full_pcd": np.ascontiguousarray(full_pcd),
        "len_src": len_src,
        "source_indices": np.ascontiguousarray(source_indices, dtype=np.int64),
        "target_indices": np.ascontiguousarray(target_indices, dtype=np.int64),
        "sampled_feats": np.ascontiguousarray(sampled_feats),
        "sampled_len_src": sampled_len_src,
        "numpy_rng_state_algorithm": rng_algorithm,
        "numpy_rng_state_keys": np.ascontiguousarray(rng_keys),
        "numpy_rng_state_pos": rng_pos,
        "numpy_rng_state_has_gauss": rng_has_gauss,
        "numpy_rng_state_cached_gaussian": rng_cached_gaussian,
        "raw_artifact_sha256": raw_hash,
    }


def reconstruct_sampled_arrays(
    artifact: dict[str, np.ndarray | int | float | str],
) -> dict[str, np.ndarray]:
    full_pcd = np.asarray(artifact["full_pcd"])
    len_src = int(artifact["len_src"])
    source_indices = np.asarray(artifact["source_indices"], dtype=np.int64)
    target_indices = np.asarray(artifact["target_indices"], dtype=np.int64)
    sampled_feats = np.asarray(artifact["sampled_feats"])
    sampled_len_src = int(artifact["sampled_len_src"])
    return {
        "source_scene": np.ascontiguousarray(full_pcd[:len_src]),
        "target_scene": np.ascontiguousarray(full_pcd[len_src:]),
        "source_points": np.ascontiguousarray(full_pcd[:len_src][source_indices]),
        "target_points": np.ascontiguousarray(full_pcd[len_src:][target_indices]),
        "source_features": np.ascontiguousarray(sampled_feats[:sampled_len_src]),
        "target_features": np.ascontiguousarray(sampled_feats[sampled_len_src:]),
        "source_indices": source_indices,
        "target_indices": target_indices,
    }


def restore_numpy_rng_state(
    artifact: dict[str, np.ndarray | int | float | str],
) -> None:
    np.random.set_state(
        (
            str(artifact["numpy_rng_state_algorithm"]),
            np.asarray(artifact["numpy_rng_state_keys"], dtype=np.uint32),
            int(artifact["numpy_rng_state_pos"]),
            int(artifact["numpy_rng_state_has_gauss"]),
            float(artifact["numpy_rng_state_cached_gaussian"]),
        )
    )


def _json_object_from_bytes(data: bytes, path: Path) -> dict[str, object]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"sealed JSON must contain an object: {path}")
    return value


def _valid_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def validate_manifest(
    manifest_path: Path,
    *,
    expected_indices: list[int] | None = None,
    expected_base_seed: int | None = None,
    expected_num_node: int | None = None,
    verify_files: bool = True,
) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    manifest_bytes = read_regular_file_bytes(manifest_path)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = _json_object_from_bytes(manifest_bytes, manifest_path)
    protocol_path = manifest_path.parent / "protocol.json"
    protocol_bytes = read_regular_file_bytes(protocol_path)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = _json_object_from_bytes(protocol_bytes, protocol_path)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise RuntimeError("online artifact protocol SHA-256 mismatch")
    shared_loader_path = Path(__file__).resolve()
    shared_loader_bytes = read_regular_file_bytes(shared_loader_path)
    shared_loader_hash = hashlib.sha256(shared_loader_bytes).hexdigest()
    if protocol.get("shared_loader_sha256") != shared_loader_hash:
        raise RuntimeError("online artifact shared loader changed after generation")
    generator_path = Path(str(protocol.get("script", "")))
    generator_bytes = read_regular_file_bytes(generator_path)
    generator_hash = hashlib.sha256(generator_bytes).hexdigest()
    if protocol.get("script_sha256") != generator_hash:
        raise RuntimeError("online artifact generator changed after generation")
    sidecar_path = manifest_path.with_suffix(".sha256")
    sidecar_bytes = read_regular_file_bytes(sidecar_path)
    if sidecar_bytes.decode("ascii").strip() != manifest_hash:
        raise RuntimeError("online artifact manifest SHA-256 sidecar mismatch")

    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or protocol.get("schema_version") != SCHEMA_VERSION
    ):
        raise RuntimeError("unsupported online artifact manifest schema")
    if (
        manifest.get("artifact_keys") != list(ARTIFACT_KEYS)
        or protocol.get("artifact_keys") != list(ARTIFACT_KEYS)
    ):
        raise RuntimeError("online artifact manifest allowlist mismatch")
    if manifest.get("tree_hash_algorithm") != TREE_HASH_ALGORITHM:
        raise RuntimeError("online artifact tree hash algorithm mismatch")
    indices = [int(value) for value in manifest.get("pair_indices", [])]
    if indices != sorted(set(indices)) or int(manifest.get("pair_count", -1)) != len(
        indices
    ):
        raise RuntimeError("online artifact manifest indices are invalid")
    cross_seals = {
        "script": protocol.get("script"),
        "script_sha256": protocol.get("script_sha256"),
        "shared_loader": protocol.get("shared_loader"),
        "shared_loader_sha256": protocol.get("shared_loader_sha256"),
        "raw_data_root": protocol.get("raw_data_root"),
        "raw_input_tree_sha256": protocol.get("raw_input_tree_sha256"),
        "source_pairs_path": protocol.get("source_pairs_path"),
        "source_pairs_sha256": protocol.get("source_pairs_sha256"),
        "source_pairs_fields_read": protocol.get("source_pairs_fields_read"),
        "raw_artifact_keys_used": protocol.get("raw_artifact_keys_used"),
        "raw_artifact_expected_keys": protocol.get("raw_artifact_expected_keys"),
        "forbidden_output_keys": protocol.get("forbidden_output_keys"),
        "pair_indices": protocol.get("pair_indices"),
        "expected_corpus_count": protocol.get("expected_corpus_count"),
        "base_seed": protocol.get("base_seed"),
        "num_node": protocol.get("num_node"),
        "gt_boundary": protocol.get("gt_boundary"),
    }
    mismatched_cross_seals = [
        key for key, expected in cross_seals.items() if manifest.get(key) != expected
    ]
    if mismatched_cross_seals:
        raise RuntimeError(
            f"protocol/manifest cross-seal mismatch: {mismatched_cross_seals}"
        )
    if (
        protocol.get("fresh_only") is not True
        or protocol.get("generation_lock_atomic_o_excl") is not True
        or protocol.get("staging_atomic_publish") is not True
    ):
        raise RuntimeError("online artifact generation protocol is not sealed")
    if expected_indices is not None and not set(expected_indices).issubset(indices):
        raise RuntimeError("online artifact manifest does not cover target indices")
    if (
        expected_base_seed is not None
        and int(manifest.get("base_seed", -1)) != expected_base_seed
    ):
        raise RuntimeError("online artifact manifest base seed mismatch")
    if (
        expected_num_node is not None
        and int(manifest.get("num_node", -1)) != expected_num_node
    ):
        raise RuntimeError("online artifact manifest num_node mismatch")

    raw_entries = protocol.get("raw_input_manifest", [])
    if not isinstance(raw_entries, list) or len(raw_entries) != len(indices):
        raise RuntimeError("raw input manifest is incomplete")
    raw_entries_by_index: dict[int, dict[str, object]] = {}
    for raw_value in raw_entries:
        raw_entry = dict(raw_value)
        pair_index = int(raw_entry["pair_index"])
        if (
            pair_index in raw_entries_by_index
            or pair_index not in indices
            or raw_entry.get("file") != f"{pair_index}.pth"
            or not _valid_sha256(raw_entry.get("sha256"))
            or int(raw_entry.get("byte_size", -1)) <= 0
        ):
            raise RuntimeError("raw input manifest contains an invalid entry")
        raw_entries_by_index[pair_index] = raw_entry
    if [int(entry["pair_index"]) for entry in raw_entries] != indices:
        raise RuntimeError("raw input manifest order mismatch")
    if artifact_tree_hash(raw_entries) != protocol.get("raw_input_tree_sha256"):
        raise RuntimeError("raw input manifest tree SHA-256 mismatch")

    entries = manifest.get("artifacts", [])
    if not isinstance(entries, list) or len(entries) != len(indices):
        raise RuntimeError("online artifact manifest entries are incomplete")
    entries_by_index: dict[int, dict[str, object]] = {}
    total_bytes = 0
    for raw_entry in entries:
        entry = dict(raw_entry)
        pair_index = int(entry["pair_index"])
        if pair_index in entries_by_index or pair_index not in indices:
            raise RuntimeError("online artifact manifest contains invalid entries")
        if entry.get("file") != artifact_filename(pair_index):
            raise RuntimeError(
                f"online artifact filename mismatch for pair {pair_index}"
            )
        file_hash = str(entry.get("sha256", ""))
        if not _valid_sha256(file_hash):
            raise RuntimeError(f"invalid online artifact SHA-256 for pair {pair_index}")
        artifact_path = manifest_path.parent / str(entry["file"])
        metadata = os.stat(artifact_path, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode) or artifact_path.is_symlink():
            raise RuntimeError(
                f"online artifact is not a regular non-symlink file: {pair_index}"
            )
        if metadata.st_size != int(entry["byte_size"]):
            raise RuntimeError(f"online artifact size mismatch for pair {pair_index}")
        if entry.get("raw_artifact_sha256") != raw_entries_by_index[pair_index].get(
            "sha256"
        ):
            raise RuntimeError(
                f"online/raw artifact cross-seal mismatch for pair {pair_index}"
            )
        if verify_files:
            artifact_bytes = read_regular_file_bytes(artifact_path)
            actual_hash = hashlib.sha256(artifact_bytes).hexdigest()
            if actual_hash != file_hash:
                raise RuntimeError(
                    f"online artifact hash mismatch for pair {pair_index}"
                )
            artifact = _load_online_artifact_bytes(
                artifact_bytes,
                artifact_path,
                expected_pair_index=pair_index,
                expected_pair_seed=int(manifest["base_seed"]) + pair_index,
                num_node=int(manifest["num_node"]),
            )
            if artifact["raw_artifact_sha256"] != entry["raw_artifact_sha256"]:
                raise RuntimeError(
                    f"online artifact embedded raw hash mismatch for pair {pair_index}"
                )
        total_bytes += int(entry["byte_size"])
        entries_by_index[pair_index] = entry
    ordered_entries = [entries_by_index[index] for index in indices]
    if artifact_tree_hash(ordered_entries) != manifest.get("tree_sha256"):
        raise RuntimeError("online artifact tree SHA-256 mismatch")
    if total_bytes != int(manifest.get("total_bytes", -1)):
        raise RuntimeError("online artifact total byte count mismatch")
    expected_files = {str(entry["file"]) for entry in ordered_entries}
    actual_files = {path.name for path in manifest_path.parent.glob("*.npz")}
    if actual_files != expected_files:
        raise RuntimeError("online artifact directory file set differs from manifest")
    return manifest, entries_by_index
