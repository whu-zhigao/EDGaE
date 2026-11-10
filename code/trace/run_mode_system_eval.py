#!/usr/bin/env python3
"""Run the mode-selection and system-policy blocks for the ICDE v3 revision.

Blocks:
  preprocess      re-measure EDGaE-K16 rows with the new working-set metrics
                  (distinct sampled vertices and bytes) needed for E-mode
                  transition accounting
  system-policy   system-inspired adapters (DistDGL / PaGraph / GNNLab /
                  Ginex / BGL / NeutronStar / DAHA / PlatoD2GL) on the
                  identical trace substrate

The system-policy block reuses the exact sampler, byte accounting, budget
rules, and evaluation traces of the v2 matrix. Adapters differ only in
partition and cache policy. They are policy adapters, not the original
end-system deployments; per-worker online-cache semantics are used (no
central pooling).
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
from trace_driven_runner_v2 import SYSTEM_POLICY_SPECS


def configurations(blocks, datasets):
    rows = []
    for d in datasets:
        if "preprocess" in blocks:
            rows.append(dict(block="preprocess", dataset=d, model="GCN",
                             sampler="non-uniform", workers="8",
                             control="EDGaE-K16"))
        if "system-policy" in blocks:
            for adapter in SYSTEM_POLICY_SPECS:
                rows.append(dict(block="system-policy", dataset=d, model="GCN",
                                 sampler="non-uniform", workers="8",
                                 control=adapter))
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
    parser.add_argument("--output", default="results/icde-v3-mode-system-results.csv")
    parser.add_argument("--cache-dir", default="experiment-cache")
    parser.add_argument("--blocks", nargs="+",
                        choices=["preprocess", "system-policy"],
                        default=["preprocess", "system-policy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--parallelism", type=int, default=6)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    jobs = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index, config in enumerate(configurations(args.blocks, args.datasets), start=1):
        for seed in args.seeds:
            jobs.append({**config, "seed": seed, "cache_dir": args.cache_dir,
                         "run_id": f"{timestamp}-v3-{index:04d}-{seed}"})
    print(f"{len(jobs)} runs ({', '.join(args.blocks)})", flush=True)
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
                if completed % 25 == 0 or completed == len(jobs):
                    stream.flush()
                    print(f"[{completed}/{len(jobs)}] {time.time() - start:.0f}s", flush=True)
    print(f"done in {time.time() - start:.0f}s -> {output}")


if __name__ == "__main__":
    main()
