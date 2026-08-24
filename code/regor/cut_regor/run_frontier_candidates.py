from __future__ import annotations

import argparse
import csv
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


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = REPO_ROOT / "code" / "regor"
DIAGNOSTIC_ROOT = CODE_ROOT / "diagnostics"
BASELINE_ROOT = REPO_ROOT / "source" / "regor"
sys.path.insert(0, str(CODE_ROOT))
sys.path.insert(0, str(DIAGNOSTIC_ROOT))
sys.path.insert(0, str(BASELINE_ROOT))
os.chdir(BASELINE_ROOT)

from einfo_online_artifact import load_online_artifact, reconstruct_sampled_arrays
from failure_split_diagnostic import create_matcher


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=REPO_ROOT / "output" / "regor" / "2026-08-15_Hard247OnlineArtifacts",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CutRegor candidate runner requires CUDA")
    artifact_root = project_path(args.artifact_root)
    output_dir = project_path(args.output_dir)
    trace_dir = output_dir / "online_traces"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError("output exists; use --overwrite")
    trace_dir.mkdir(parents=True, exist_ok=True)
    pair_paths = sorted(artifact_root.glob("*.npz"), key=lambda path: int(path.stem))
    if args.limit is not None:
        pair_paths = pair_paths[: args.limit]
    protocol = {
        "track": "strict pairwise diagnostic online stage",
        "method": "CutRegor Frontier Separability candidate generation",
        "online_inputs": "physical no-label current-pair artifact only",
        "output": "refined candidate poses and FS-TCD scores without GT",
        "gt_loaded": False,
        "pair_count": len(pair_paths),
        "seed": args.seed,
        "artifact_root": str(artifact_root),
        "script_sha256": sha256_file(Path(__file__)),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    fields = ["pair_index", "pair_seed", "candidate_count", "selected_index", "artifact_sha256", "trace_sha256", "runtime_sec", "error"]
    errors_path = output_dir / "online_errors.jsonl"
    with errors_path.open("w", encoding="utf-8") as error_handle:
        for position, artifact_path in enumerate(pair_paths, 1):
            pair_index = int(artifact_path.stem)
            pair_seed = args.seed + pair_index
            row = {field: "" for field in fields}
            row.update({"pair_index": pair_index, "pair_seed": pair_seed})
            started = time.perf_counter()
            try:
                artifact = load_online_artifact(
                    artifact_path,
                    expected_pair_index=pair_index,
                    expected_pair_seed=pair_seed,
                    num_node=5000,
                )
                arrays = reconstruct_sampled_arrays(artifact)
                set_seed(pair_seed)
                source = torch.from_numpy(arrays["source_points"]).cuda()[None]
                target = torch.from_numpy(arrays["target_points"]).cuda()[None]
                source_features = torch.from_numpy(arrays["source_features"]).cuda()[None]
                target_features = torch.from_numpy(arrays["target_features"]).cuda()[None]
                matcher = create_matcher()
                with torch.no_grad():
                    matcher.estimator(
                        source,
                        target,
                        source_features,
                        target_features,
                        torch.eye(4, device=source.device, dtype=source.dtype)[None],
                    )
                poses = matcher.trace["refined_topk_trans"].detach().cpu().numpy()[0].astype(np.float32)
                scores = np.asarray(matcher.trace["fstcd_scores"], dtype=np.float64)
                selected = int(matcher.trace["fstcd_selected_index"])
                if poses.shape != (50, 4, 4) or scores.shape != (50,) or not np.isfinite(poses).all() or not np.isfinite(scores).all():
                    raise RuntimeError("candidate trace failed shape or finite checks")
                trace_path = trace_dir / f"{pair_index:04d}.npz"
                np.savez_compressed(
                    trace_path,
                    pair_index=np.asarray(pair_index, dtype=np.int64),
                    pair_seed=np.asarray(pair_seed, dtype=np.int64),
                    refined_poses=poses,
                    fstcd_scores=scores,
                    selected_index=np.asarray(selected, dtype=np.int64),
                    input_artifact_sha256=np.asarray(sha256_file(artifact_path)),
                )
                row.update(
                    {
                        "candidate_count": len(poses),
                        "selected_index": selected,
                        "artifact_sha256": sha256_file(artifact_path),
                        "trace_sha256": sha256_file(trace_path),
                    }
                )
            except Exception as error:
                row["error"] = f"{type(error).__name__}: {error}"
                error_handle.write(
                    json.dumps(
                        {"pair_index": pair_index, "error": row["error"], "traceback": traceback.format_exc()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                error_handle.flush()
            row["runtime_sec"] = time.perf_counter() - started
            rows.append(row)
            print(f"[{position}/{len(pair_paths)}] pair={pair_index} {row['error'] or 'ok'}", flush=True)
    with (output_dir / "online_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    complete = {
        "pair_count": len(rows),
        "valid_count": sum(not bool(row["error"]) for row in rows),
        "error_count": sum(bool(row["error"]) for row in rows),
        "gt_loaded": False,
        "all_finite": all(not row["error"] for row in rows),
        "trace_tree_sha256": hashlib.sha256(
            "".join(str(row["trace_sha256"]) for row in rows if not row["error"]).encode()
        ).hexdigest(),
    }
    (output_dir / "online_complete.json").write_text(json.dumps(complete, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
