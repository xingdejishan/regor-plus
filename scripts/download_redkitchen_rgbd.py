import argparse
import struct
import subprocess
import sys
import time
import urllib.request
import zipfile
import zlib
from pathlib import Path

import requests

from redkitchen_fsv_common import REPO_ROOT, SCENE, ensure_dir, has_raw_scene_layout, locate_raw_scene_root, read_json, write_json


URL = "https://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/7-scenes-redkitchen.zip"
SESSION = requests.Session()


def http_range(url, start=None, end=None, suffix=None):
    headers = {}
    if suffix is not None:
        headers["Range"] = f"bytes=-{int(suffix)}"
    elif start is not None and end is not None:
        headers["Range"] = f"bytes={int(start)}-{int(end)}"
    resp = SESSION.get(url, headers=headers, timeout=(30, 120))
    status = resp.status_code
    data = resp.content
    response_headers = dict(resp.headers.items())
    if (suffix is not None or start is not None) and status != 206:
        raise RuntimeError(f"Server did not honor HTTP range request, status={status}")
    resp.close()
    return data, response_headers


def remote_size(url):
    resp = SESSION.head(url, timeout=(30, 120))
    length = resp.headers.get("Content-Length")
    resp.close()
    if length:
        return int(length)
    _, headers = http_range(url, 0, 0)
    content_range = headers.get("Content-Range", "")
    return int(content_range.split("/")[-1])


def parse_zip64_extra(extra, comp_size, uncomp_size, local_offset):
    pos = 0
    values = {}
    while pos + 4 <= len(extra):
        header_id, size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        payload = extra[pos : pos + size]
        pos += size
        if header_id != 0x0001:
            continue
        off = 0
        if uncomp_size == 0xFFFFFFFF and off + 8 <= len(payload):
            values["uncomp_size"] = struct.unpack_from("<Q", payload, off)[0]
            off += 8
        if comp_size == 0xFFFFFFFF and off + 8 <= len(payload):
            values["comp_size"] = struct.unpack_from("<Q", payload, off)[0]
            off += 8
        if local_offset == 0xFFFFFFFF and off + 8 <= len(payload):
            values["local_offset"] = struct.unpack_from("<Q", payload, off)[0]
    return values


def remote_central_directory(url):
    size = remote_size(url)
    tail_size = min(size, 2 * 1024 * 1024)
    tail, _ = http_range(url, suffix=tail_size)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise RuntimeError("Cannot locate ZIP EOCD")
    total_entries = struct.unpack_from("<H", tail, eocd + 10)[0]
    cd_size = struct.unpack_from("<I", tail, eocd + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, eocd + 16)[0]
    if total_entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        if locator < 0:
            raise RuntimeError("ZIP64 locator missing")
        zip64_eocd_offset = struct.unpack_from("<Q", tail, locator + 8)[0]
        record, _ = http_range(url, zip64_eocd_offset, zip64_eocd_offset + 80)
        if record[:4] != b"PK\x06\x06":
            raise RuntimeError("ZIP64 EOCD missing")
        total_entries = struct.unpack_from("<Q", record, 32)[0]
        cd_size = struct.unpack_from("<Q", record, 40)[0]
        cd_offset = struct.unpack_from("<Q", record, 48)[0]
    cd, _ = http_range(url, cd_offset, cd_offset + cd_size - 1)
    entries = []
    pos = 0
    while pos + 46 <= len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            raise RuntimeError(f"Invalid central directory signature at {pos}")
        flag = struct.unpack_from("<H", cd, pos + 8)[0]
        method = struct.unpack_from("<H", cd, pos + 10)[0]
        crc = struct.unpack_from("<I", cd, pos + 16)[0]
        comp_size = struct.unpack_from("<I", cd, pos + 20)[0]
        uncomp_size = struct.unpack_from("<I", cd, pos + 24)[0]
        name_len = struct.unpack_from("<H", cd, pos + 28)[0]
        extra_len = struct.unpack_from("<H", cd, pos + 30)[0]
        comment_len = struct.unpack_from("<H", cd, pos + 32)[0]
        local_offset = struct.unpack_from("<I", cd, pos + 42)[0]
        name_start = pos + 46
        name_bytes = cd[name_start : name_start + name_len]
        encoding = "utf-8" if flag & 0x800 else "cp437"
        name = name_bytes.decode(encoding).replace("\\", "/")
        extra = cd[name_start + name_len : name_start + name_len + extra_len]
        zip64 = parse_zip64_extra(extra, comp_size, uncomp_size, local_offset)
        comp_size = int(zip64.get("comp_size", comp_size))
        uncomp_size = int(zip64.get("uncomp_size", uncomp_size))
        local_offset = int(zip64.get("local_offset", local_offset))
        entries.append(
            {
                "name": name,
                "method": int(method),
                "crc": int(crc),
                "comp_size": comp_size,
                "uncomp_size": uncomp_size,
                "local_offset": local_offset,
            }
        )
        pos = name_start + name_len + extra_len + comment_len
    return size, entries


def local_data_offset(url, entry):
    header, _ = http_range(url, entry["local_offset"], entry["local_offset"] + 30)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local header for {entry['name']}")
    name_len = struct.unpack_from("<H", header, 26)[0]
    extra_len = struct.unpack_from("<H", header, 28)[0]
    return entry["local_offset"] + 30 + name_len + extra_len


def extract_entry(url, entry, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size == entry["uncomp_size"]:
        return "exists"
    data_start = local_data_offset(url, entry)
    comp, _ = http_range(url, data_start, data_start + entry["comp_size"] - 1)
    if entry["method"] == 0:
        data = comp
    elif entry["method"] == 8:
        data = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {entry['method']} for {entry['name']}")
    output_path.write_bytes(data)
    return "written"


def extract_entry_from_span(span, span_start, entry, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size == entry["uncomp_size"]:
        return "exists"
    rel = entry["local_offset"] - span_start
    if span[rel : rel + 4] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local header for {entry['name']}")
    name_len = struct.unpack_from("<H", span, rel + 26)[0]
    extra_len = struct.unpack_from("<H", span, rel + 28)[0]
    data_start = rel + 30 + name_len + extra_len
    data_end = data_start + entry["comp_size"]
    if data_end > len(span):
        raise RuntimeError(f"Span too small for {entry['name']}")
    comp = span[data_start:data_end]
    if entry["method"] == 0:
        data = comp
    elif entry["method"] == 8:
        data = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"Unsupported compression method {entry['method']} for {entry['name']}")
    output_path.write_bytes(data)
    return "written"


def estimated_entry_end(entry):
    return entry["local_offset"] + 30 + len(entry["name"].encode("utf-8")) + 4096 + entry["comp_size"]


def grouped_spans(entries, max_gap):
    entries = sorted(entries, key=lambda item: item["local_offset"])
    groups = []
    current = []
    start = None
    end = None
    for entry in entries:
        entry_start = entry["local_offset"]
        entry_end = estimated_entry_end(entry)
        if not current:
            current = [entry]
            start = entry_start
            end = entry_end
            continue
        if entry_start - end <= max_gap:
            current.append(entry)
            end = max(end, entry_end)
        else:
            groups.append((start, end, current))
            current = [entry]
            start = entry_start
            end = entry_end
    if current:
        groups.append((start, end, current))
    return groups


def extract_entries_batched(url, entries, raw_root, batch_gap_bytes):
    groups = grouped_spans(entries, batch_gap_bytes)
    written = 0
    processed = 0
    for group_index, (start, end, group) in enumerate(groups, start=1):
        span, _ = http_range(url, start, end - 1)
        for entry in group:
            status = extract_entry_from_span(span, start, entry, output_path_for_member(raw_root, entry["name"]))
            processed += 1
            if status == "written":
                written += 1
        print(f"selective_span {group_index}/{len(groups)} processed={processed}/{len(entries)} written={written}", flush=True)
    return written, len(groups), sum(end - start for start, end, _ in groups)


def selected_members(entries, manifest):
    needed_suffixes = set()
    for item in manifest:
        seq = item["sequence"]
        for frame_idx in range(int(item["frame_start"]), int(item["frame_end"]) + 1):
            stem = f"{seq}/frame-{frame_idx:06d}"
            needed_suffixes.add(f"{stem}.depth.png")
            needed_suffixes.add(f"{stem}.pose.txt")
    selected = []
    for entry in entries:
        name = entry["name"]
        if name.endswith("/"):
            continue
        lower = name.lower()
        if lower.endswith("camera-intrinsics.txt") or lower.endswith("intrinsics.txt"):
            selected.append(entry)
            continue
        if any(name.endswith(suffix) for suffix in needed_suffixes):
            selected.append(entry)
    return selected


def output_path_for_member(raw_root, name):
    parts = Path(name).parts
    if SCENE in parts:
        start = parts.index(SCENE)
        rel = Path(*parts[start:])
    else:
        rel = Path(name)
    return Path(raw_root) / rel


def selective_download(args):
    raw_root = Path(args.raw_root)
    ensure_dir(raw_root)
    manifest = read_json(args.manifest)
    log = {
        "url": URL,
        "mode": "selective_http_range",
        "raw_root": str(raw_root.resolve()),
        "attempts": [],
    }
    for attempt in range(1, int(args.retries) + 1):
        started = time.time()
        try:
            size, entries = remote_central_directory(URL)
            selected = selected_members(entries, manifest)
            if not selected:
                raise RuntimeError("No depth/pose/intrinsics members matched in remote zip")
            total_comp = sum(e["comp_size"] for e in selected)
            written, span_count, span_bytes = extract_entries_batched(URL, selected, raw_root, args.batch_gap_bytes)
            scene_root = locate_raw_scene_root(raw_root, SCENE)
            ok = has_raw_scene_layout(scene_root)
            entry = {
                "attempt": attempt,
                "status": "ok" if ok else "missing_raw_layout_after_selective_extract",
                "seconds": round(time.time() - started, 3),
                "remote_zip_size_bytes": int(size),
                "selected_members": len(selected),
                "selected_compressed_bytes": int(total_comp),
                "range_spans": int(span_count),
                "range_span_bytes": int(span_bytes),
                "written_members": written,
                "scene_root": str(scene_root),
            }
            log["attempts"].append(entry)
            log["status"] = entry["status"]
            log["scene_root"] = str(scene_root)
            write_json(args.log_path, log)
            if ok:
                print(f"raw_scene_root={scene_root}")
                return
        except Exception as exc:
            log["attempts"].append({"attempt": attempt, "status": "failed", "error": repr(exc), "seconds": round(time.time() - started, 3)})
            log["status"] = "failed"
            write_json(args.log_path, log)
            if attempt == int(args.retries):
                raise
            time.sleep(2)
    raise RuntimeError("Selective download did not produce a complete raw scene layout")


def download(args):
    if args.selective:
        selective_download(args)
        return
    raw_root = Path(args.raw_root)
    ensure_dir(raw_root)
    scene_root = locate_raw_scene_root(raw_root, SCENE)
    zip_path = Path(args.zip_path) if args.zip_path else raw_root / f"{SCENE}.zip"
    log = {
        "url": URL,
        "raw_root": str(raw_root.resolve()),
        "zip_path": str(zip_path.resolve()),
        "attempts": [],
        "extracted": False,
        "scene_root": str(scene_root),
    }

    if has_raw_scene_layout(scene_root):
        log["status"] = "already_available"
        write_json(args.log_path, log)
        print(f"raw_scene_root={scene_root}")
        return

    for attempt in range(1, int(args.retries) + 1):
        started = time.time()
        cmd = ["curl.exe", "-L", "-C", "-", URL, "-o", str(zip_path)]
        if not args.use_curl_exe:
            cmd = ["curl", "-L", "-C", "-", URL, "-o", str(zip_path)]
        result = subprocess.run(cmd, cwd=str(raw_root))
        entry = {
            "attempt": attempt,
            "command": " ".join(cmd),
            "returncode": int(result.returncode),
            "seconds": round(time.time() - started, 3),
            "zip_exists": zip_path.exists(),
            "zip_size_bytes": zip_path.stat().st_size if zip_path.exists() else 0,
        }
        log["attempts"].append(entry)
        write_json(args.log_path, log)
        if result.returncode == 0 and zip_path.exists() and zip_path.stat().st_size > 0:
            break
        time.sleep(2)

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        log["status"] = "download_failed"
        write_json(args.log_path, log)
        raise RuntimeError(f"Cannot download {URL} after {args.retries} attempts")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(f"Corrupt zip member: {bad}")
            zf.extractall(raw_root)
        log["extracted"] = True
    except Exception as exc:
        log["status"] = "extract_failed"
        log["extract_error"] = repr(exc)
        write_json(args.log_path, log)
        raise

    scene_root = locate_raw_scene_root(raw_root, SCENE)
    log["scene_root"] = str(scene_root)
    log["status"] = "ok" if has_raw_scene_layout(scene_root) else "missing_raw_layout_after_extract"
    write_json(args.log_path, log)
    if log["status"] != "ok":
        raise RuntimeError(f"Downloaded data does not contain a complete raw scene layout under {raw_root}")
    print(f"raw_scene_root={scene_root}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default=str(REPO_ROOT / "3dmatch_raw" / "test"))
    parser.add_argument("--zip-path", default="")
    parser.add_argument("--log-path", default=str(REPO_ROOT / "outputs" / "redkitchen_fsv" / "download_log.json"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "3DMatch" / "free_space" / "redkitchen_manifest.json"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--use-curl-exe", action="store_true", default=True)
    parser.add_argument("--selective", action="store_true")
    parser.add_argument("--batch-gap-bytes", type=int, default=1048576)
    args = parser.parse_args()
    download(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
