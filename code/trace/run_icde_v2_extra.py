#!/usr/bin/env python3
"""Run the two ICDE-camera-ready extension blocks.

Blocks:
  sampler-sensitivity  fan-out/depth/sampler-type axes for the central claim
  two-tier             ship-once static tier + bounded mutable LFU hot tier

Every row records the runner version, seed, and evidence level so the results
can be audited alongside the main 945-run matrix.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from run_icde_v2_eval import COLUMNS, DATASETS

SENSITIVITY_CONTROLS = [
    "f=2,d=2,non-uniform", "f=4,d=2,non-uniform", "f=8,d=2,non-uniform",
    "f=2,d=1,non-uniform", "f=2,d=3,non-uniform", "f=2,d=2,uniform",
]
TWO_TIER_CONTROLS = ["beta=0.0", "beta=0.25", "beta=0.5", "beta=0.75", "beta=1.0"]


def configurations(datasets):
    rows = []
    for d in datasets:
        for control in SENSITIVITY_CONTROLS:
            rows.append(dict(block="sampler-sensitivity", dataset=d, model="GCN",
                             sampler="non-uniform", workers="8", control=control))
        for control in TWO_TIER_CONTROLS:
            rows.append(dict(block="two-tier", dataset=d, model="GCN",
                             sampler="non-uniform", workers="8", control=control))
    return rows


def run_one(job):
    manifest = {**job, "run_id": job["run_id"], "horizon_epochs": 100}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "experiments/trace_driven_runner_v2.py",
             manifest_path, "--cache-dir", job["cache_dir"]],
            capture_output=True, text=True, check=True)
        result = json.loads(proc.stdout)
    finally:
        Path(manifest_path).unlink(missing_ok=True)
    row = {key: "" for key in COLUMNS}
    row.update({k: v for k, v in job.items() if k in COLUMNS})
    row.update({k: v for k, v in result.items() if k in COLUMNS})
    row["measurement_utc"] = datetime.now(timezone.utc).isoformat()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="www/icde2027-v2-extra-results.csv")
    parser.add_argument("--cache-dir", default="experiment-cache")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--parallelism", type=int, default=6)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    jobs = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index, config in enumerate(configurations(args.datasets), start=1):
        for seed in args.seeds:
            jobs.append({**config, "seed": seed, "cache_dir": args.cache_dir,
                         "run_id": f"{timestamp}-extra-{index:04d}-{seed}"})
    print(f"{len(jobs)} extension runs", flush=True)
    completed, start = 0, time.time()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        stream.flush()
        with ProcessPoolExecutor(max_workers=args.parallelism) as pool:
            futures = {pool.submit(run_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    writer.writerow(future.result())
                except Exception as exc:
                    row = {key: "" for key in COLUMNS}
                    row.update({k: v for k, v in job.items() if k in COLUMNS})
                    row["measurement_valid"] = False
                    row["notes"] = f"FAILED: {type(exc).__name__}: {exc}"
                    writer.writerow(row)
                completed += 1
                if completed % 50 == 0 or completed == len(jobs):
                    stream.flush()
                    print(f"[{completed}/{len(jobs)}] {time.time() - start:.0f}s", flush=True)
    print(f"done in {time.time() - start:.0f}s -> {output}")


if __name__ == "__main__":
    raise SystemExit(main())
