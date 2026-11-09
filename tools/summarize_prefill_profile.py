#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract a small communication/copy summary from existing Ascend profiles.

Usage: python tools/summarize_prefill_profile.py /path/to/128-token-profile > summary.json
Reads communication.json and streams trace_view.json with bounded memory.
"""

import argparse
import heapq
import json
import math
import re
from pathlib import Path


def number(value):
    try:
        result = float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def events(path, chunk_size=1024 * 1024):
    """Iterate complete Chrome trace events without loading the full file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8-sig") as stream:
        buffer = ""
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                raise ValueError(f"No traceEvents array found in {path}")
            buffer += chunk
            match = re.search(r'"traceEvents"\s*:\s*\[', buffer)
            if match:
                position = match.end()
                break
            if buffer.lstrip().startswith("["):
                position = buffer.index("[") + 1
                break
            if len(buffer) > 16 * 1024 * 1024:
                raise ValueError(f"Trace header exceeds 16 MiB in {path}")
        while True:
            while position < len(buffer) and buffer[position] in " \r\n\t,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                event, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as error:
                chunk = stream.read(chunk_size)
                if not chunk:
                    raise ValueError(f"Incomplete trace event in {path}") from error
                buffer = buffer[position:] + chunk
                position = 0
                if len(buffer) > 16 * 1024 * 1024:
                    raise ValueError(f"Single trace event exceeds 16 MiB in {path}") from error
                continue
            position = end
            if isinstance(event, dict):
                yield event


def compact(value):
    if isinstance(value, dict):
        return {str(key): compact(item) for key, item in list(value.items())[:24]}
    if isinstance(value, list):
        return [compact(item) for item in value[:12]]
    if isinstance(value, str):
        return value[:240]
    return value


def longest(heap, item, duration, serial, limit):
    entry = (duration, serial, item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif duration > heap[0][0]:
        heapq.heapreplace(heap, entry)


def trace_summary(path, top):
    processes, threads = {}, {}
    counts = {"allreduce": 0, "copy": 0}
    selected = {}
    serial = 0
    for event in events(path):
        name = str(event.get("name", ""))
        pid, tid = str(event.get("pid", "")), str(event.get("tid", ""))
        args = event.get("args") or {}
        if event.get("ph") == "M":
            if name == "process_name":
                processes[pid] = args.get("name")
            elif name == "thread_name":
                threads[(pid, tid)] = args.get("name")
            continue
        normalized = name.lower().replace("_", "").replace(" ", "")
        category = None
        if "allreduce" in normalized:
            category = "allreduce"
        elif "memcpy" in normalized or "hosttodevice" in normalized or "h2d" in normalized:
            category = "copy"
        if category is None or event.get("ph") != "X":
            continue
        duration = number(event.get("dur"))
        counts[category] += 1
        serial += 1
        item = {
            "name": name,
            "category": event.get("cat"),
            "pid": pid,
            "tid": tid,
            "start_us": number(event.get("ts")),
            "duration_ms": duration / 1000,
            "args": compact(args),
        }
        # Keep CPU submission ranges and device tasks independently.
        heap = selected.setdefault((category, pid), [])
        longest(heap, item, duration, serial, top)
    result = {"file": str(path), "counts": counts}
    for category in counts:
        retained = [entry for (kind, _), heap in selected.items() if kind == category for entry in heap]
        rows = [row for _, _, row in sorted(retained, reverse=True)]
        for row in rows:
            row["process"] = processes.get(row["pid"])
            row["thread"] = threads.get((row["pid"], row["tid"]))
        result[category] = rows
    return result


def communication_summary(path, top):
    """Keep HCCL time breakdowns and link totals, excluding packet histograms."""
    with path.open(encoding="utf-8-sig") as stream:
        data = json.load(stream)
    selected = []
    serial = 0

    def visit(value, location):
        nonlocal serial
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, [*location, str(index)])
            return
        if not isinstance(value, dict):
            return
        info = value.get("Communication Time Info")
        if isinstance(info, dict) and "allreduce" in "/".join(location).lower().replace("_", ""):
            duration = max((number(item) for key, item in info.items() if "elap" in key.lower()), default=0)
            links = {
                link: {key: item for key, item in details.items() if "distribution" not in key.lower()}
                for link, details in value.get("Communication Bandwidth Info", {}).items()
                if isinstance(details, dict)
            }
            item = {"operation": "/".join(location), "time": info, "links": links}
            serial += 1
            longest(selected, compact(item), duration, serial, top)
            return
        for key, item in value.items():
            visit(item, [*location, str(key)])

    visit(data, [])
    return {"file": str(path), "allreduce": [row for _, _, row in sorted(selected, reverse=True)]}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("paths", nargs="+", type=Path, help="Profile directories or individual JSON files")
    cli.add_argument("--top", type=int, default=3, help="Longest events of each kind per file")
    args = cli.parse_args(argv)
    if args.top < 1:
        cli.error("--top must be positive")
    files = set()
    for path in args.paths:
        if path.is_file():
            files.add(path.resolve())
        elif path.is_dir():
            for name in ("communication.json", "trace_view.json"):
                files.update(item.resolve() for item in path.rglob(name))
        else:
            cli.error(f"Path does not exist: {path}")
    if not files:
        cli.error("No communication.json or trace_view.json found")
    report = {"communication": [], "traces": []}
    for path in sorted(files):
        if path.name == "communication.json":
            report["communication"].append(communication_summary(path, args.top))
        elif path.name == "trace_view.json":
            report["traces"].append(trace_summary(path, args.top))
        else:
            cli.error(f"Expected communication.json or trace_view.json: {path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
