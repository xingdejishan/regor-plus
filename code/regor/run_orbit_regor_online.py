from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import random
import struct
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code" / "regor"
BASELINE_ROOT = REPO_ROOT / "source" / "regor"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))

from orbit_regor import OrbitRegorConfig, OrbitRegorRuntime
from orbit_regor.pipeline import finalize_orbit_regor, prepare_orbit_regor
from run_probe_regor_natural1781 import baseline_pipeline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_exact(handle, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = handle.read(remaining)
        if not chunk:
            raise EOFError("truncated input stream")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def frames(handle):
    while True:
        size = struct.unpack("<Q", read_exact(handle, 8))[0]
        if size == 0:
            return
        payload = read_exact(handle, size)
        with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
            required = {"token", "source_points", "target_points", "source_features", "target_features"}
            if set(packet.files) != required:
                raise RuntimeError("online frame violates the strict input allowlist")
            yield {key: packet[key] for key in packet.files}


def content_seed(source_points: np.ndarray, target_points: np.ndarray) -> int:
    digest = hashlib.sha256()
    for value in (source_points, target_points):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return int.from_bytes(digest.digest()[:8], "little") % (2**31 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensors(packet: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(packet["source_points"].astype(np.float32, copy=False))[None].cuda(),
        torch.from_numpy(packet["target_points"].astype(np.float32, copy=False))[None].cuda(),
        torch.from_numpy(packet["source_features"].astype(np.float32, copy=False))[None].cuda(),
        torch.from_numpy(packet["target_features"].astype(np.float32, copy=False))[None].cuda(),
    )


def process(packet: dict[str, np.ndarray], variants: tuple[str, ...]) -> dict[str, object]:
    token = str(packet["token"].item())
    seed = content_seed(packet["source_points"], packet["target_points"])
    set_seed(seed)
    source, target, source_features, target_features = tensors(packet)
    placeholder = torch.eye(4, dtype=source.dtype, device=source.device)[None]
    started = time.perf_counter()
    with torch.no_grad():
        baseline_pose, baseline_counts = baseline_pipeline(
            source, target, source_features, target_features, placeholder
        )
    baseline_runtime = time.perf_counter() - started
    set_seed(seed)
    with torch.no_grad():
        runtime, prepared = prepare_orbit_regor(
            source, target, source_features, target_features, placeholder
        )
    output: dict[str, object] = {
        "token": token,
        "seed": seed,
        "baseline_pose": baseline_pose[0].detach().cpu().numpy().astype(np.float64),
        "baseline_counts": np.asarray(baseline_counts, dtype=np.int64),
        "baseline_runtime_sec": baseline_runtime,
    }
    for variant_index, variant in enumerate(variants):
        set_seed(seed + variant_index)
        started = time.perf_counter()
        with torch.no_grad():
            result = finalize_orbit_regor(
                runtime,
                prepared,
                variant,
                source,
                target,
                source_features,
                target_features,
                placeholder,
            )
        pose = result.pose[0].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(pose).all():
            raise RuntimeError(f"{variant} produced a non-finite unique pose")
        output[f"{variant}_pose"] = pose
        output[f"{variant}_telemetry"] = result.telemetry
        output[f"{variant}_runtime_sec"] = time.perf_counter() - started
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=OrbitRegorRuntime.VARIANTS,
        default=list(OrbitRegorRuntime.VARIANTS),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    variants = tuple(dict.fromkeys(args.variants))
    output_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else (REPO_ROOT / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    trace_dir = output_dir / "online_traces"
    trace_dir.mkdir(parents=True)
    implementation = [
        Path(__file__).resolve(),
        CODE_ROOT / "orbit_regor" / "config.py",
        CODE_ROOT / "orbit_regor" / "atlas.py",
        CODE_ROOT / "orbit_regor" / "orbit_cover.py",
        CODE_ROOT / "orbit_regor" / "solver.py",
        CODE_ROOT / "orbit_regor" / "runtime.py",
        CODE_ROOT / "orbit_regor" / "pipeline.py",
        CODE_ROOT / "run_probe_regor_natural1781.py",
        BASELINE_ROOT / "initial_matching_plus.py",
        BASELINE_ROOT / "Correspondence_regenerate_v2.py",
        BASELINE_ROOT / "Tranformation_estimaton.py",
    ]
    write_json(
        output_dir / "protocol.json",
        {
            "status": "strict pairwise PLY-only OrbitRegor online run",
            "input": "framed stream containing only current source/target XYZ and frozen official REGOR features",
            "identity_signals": "pair/scene/file/dataset identity absent; stochastic state derived only from current coordinates",
            "baseline": "official REGOR terminal pipeline",
            "variants": variants,
            "config": OrbitRegorConfig().to_dict(),
            "unique_output": True,
            "fail_closed": True,
            "gt_loaded": False,
            "evaluator_started": False,
            "implementation_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in implementation},
        },
    )
    fields = ["token", "trace_sha256", "runtime_sec", "error"]
    rows = []
    for sequence, packet in enumerate(frames(sys.stdin.buffer), start=1):
        started = time.perf_counter()
        token = str(packet["token"].item())
        row = {"token": token, "trace_sha256": "", "runtime_sec": 0.0, "error": ""}
        try:
            result = process(packet, variants)
            trace_path = trace_dir / f"{token}.npz"
            arrays: dict[str, np.ndarray] = {
                "baseline_pose": result["baseline_pose"],
                "baseline_counts": result["baseline_counts"],
                "baseline_runtime_sec": np.asarray(result["baseline_runtime_sec"]),
            }
            for variant in variants:
                arrays[f"{variant}_pose"] = result[f"{variant}_pose"]
                arrays[f"{variant}_runtime_sec"] = np.asarray(result[f"{variant}_runtime_sec"])
                arrays[f"{variant}_telemetry_json"] = np.asarray(
                    json.dumps(result[f"{variant}_telemetry"], ensure_ascii=False)
                )
            np.savez_compressed(trace_path, **arrays)
            row["trace_sha256"] = sha256_file(trace_path)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            print(
                f"[orbit-online failure {sequence}] token={token[:12]}\n{row['error']}\n{traceback.format_exc()}",
                file=sys.stderr,
                flush=True,
            )
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        row["runtime_sec"] = time.perf_counter() - started
        rows.append(row)
        print(
            f"[orbit-online {sequence}] token={token[:12]} variants={len(variants)} time={row['runtime_sec']:.2f}s",
            file=sys.stderr,
            flush=True,
        )
    with (output_dir / "online_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "online_complete.json",
        {
            "pair_count": len(rows),
            "trace_count": len(list(trace_dir.glob("*.npz"))),
            "error_count": 0,
            "variants": variants,
            "gt_loaded": False,
            "evaluator_started": False,
            "online_rows_sha256": sha256_file(output_dir / "online_rows.csv"),
        },
    )


if __name__ == "__main__":
    main()
