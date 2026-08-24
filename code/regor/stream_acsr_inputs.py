from __future__ import annotations

import argparse
import hashlib
import io
import struct
import sys
from pathlib import Path

import numpy as np
import torch


def token_for(source_points: np.ndarray, target_points: np.ndarray, source_features: np.ndarray, target_features: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in (source_points, target_points, source_features, target_features):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_frame(handle, values: dict[str, np.ndarray]) -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    payload = buffer.getvalue()
    handle.write(struct.pack("<Q", len(payload)))
    handle.write(payload)
    handle.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    paths = sorted(args.snapshot_dir.glob("*.pth"), key=lambda path: int(path.stem))
    stop = len(paths) if args.limit is None else min(len(paths), args.start + args.limit)
    output = sys.stdout.buffer
    for path in paths[args.start : stop]:
        packet = torch.load(path, map_location="cpu")
        source_count = int(packet["len_src"])
        points = packet["pcd"].detach().cpu().numpy().astype(np.float32, copy=False)
        features = packet["feats"].detach().cpu().numpy().astype(np.float32, copy=False)
        source_points = points[:source_count]
        target_points = points[source_count:]
        source_features = features[:source_count]
        target_features = features[source_count:]
        token = token_for(source_points, target_points, source_features, target_features)
        write_frame(
            output,
            {
                "token": np.asarray(token),
                "source_points": source_points,
                "target_points": target_points,
                "source_features": source_features,
                "target_features": target_features,
            },
        )
    output.write(struct.pack("<Q", 0))
    output.flush()


if __name__ == "__main__":
    main()
