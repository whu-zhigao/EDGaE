#!/usr/bin/env python3
"""Run the complete v2 trace-driven experiment matrix in parallel."""

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


DATASETS = [
    "web-google", "amazon0601", "wiki-talk", "ogbn-arxiv",
    "amazon0601-full", "web-google-full", "wiki-talk-full",
]

COLUMNS = [
    "run_id", "measurement_utc", "measurement_source", "measurement_valid",
    "block", "dataset", "model", "sampler", "workers", "control", "seed",
    "evidence_level", "environment_type", "hardware_platform", "validity_scope",
    "graph_source", "graph_vertices", "graph_edges", "runner_version",
    "calibration_source", "platform", "notes",
    "C_srv_bytes", "C_edg_bytes", "cache_hit_rate", "cache_coverage",
    "topk_jaccard", "boundary_gap", "warm_misses",
    "replication_factor_vertices", "replication_factor_edges",
    "load_imbalance", "feasibility_failures", "fallbacks",
    "adaptive_k", "assignment_agreement", "replica_jaccard",
    "sampler_fanout", "sampler_depth", "sampler_type", "two_tier_beta",
    "T_partition_s", "T_presample_s", "T_cache_s", "T_pre_total_s",
    "working_set_vertices", "working_set_bytes",
]


def configurations() -> list[dict]:
    rows = []
    for dataset in DATASETS:
        for k in ["1", "2", "4", "8", "16", "full-trace"]:
            rows.append(dict(block="cache-stability", dataset=dataset, model="GCN",
                             sampler="non-uniform", workers="4", control=f"K={k}"))
        for policy in ["random", "even", "PaGraph-style", "EDGaE", "METIS-like"]:
            rows.append(dict(block="partition-policy", dataset=dataset, model="GCN",
                             sampler="non-uniform", workers="8", control=policy))
        for component in ["DGL-E", "+R", "+RC", "+RCS", "+RCSP"]:
            rows.append(dict(block="component-ablation", dataset=dataset, model="GCN",
                             sampler="non-uniform", workers="8", control=component))
        for weights in ["equal-weights", "locality-heavy", "balance-heavy", "memory-heavy"]:
            rows.append(dict(block="score-sensitivity", dataset=dataset, model="GCN",
                             sampler="non-uniform", workers="8", control=weights))
        for policy in ["static-K2", "static-K16", "static-full", "degree", "random", "LRU", "LFU"]:
            rows.append(dict(block="online-cache", dataset=dataset, model="GCN",
                             sampler="non-uniform", workers="8", control=policy))
        for adapter in [
            "Server-C", "DistDGL-like", "PaGraph-like", "GNNLab-like",
            "Ginex-like", "BGL-like", "NeutronStar-like", "DAHA-like",
            "PlatoD2GL-like", "EDGaE",
        ]:
            rows.append(dict(block="system-policy", dataset=dataset, model="GCN",
                             sampler="non-uniform", workers="8", control=adapter))
    return rows


def run_one(job: dict) -> dict:
    manifest = {**job, "run_id": job["run_id"], "horizon_epochs": 100}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(manifest, f)
        manifest_path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "experiments/trace_driven_runner_v2.py",
             manifest_path, "--cache-dir", job["cache_dir"]],
            capture_output=True, text=True, check=True,
        )
        result = json.loads(proc.stdout)
    finally:
        Path(manifest_path).unlink(missing_ok=True)
    row = {key: "" for key in COLUMNS}
    row.update({k: v for k, v in job.items() if k in COLUMNS})
    row.update({k: v for k, v in result.items() if k in COLUMNS})
    row["measurement_utc"] = datetime.now(timezone.utc).isoformat()
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="www/icde2027-v2-trace-results.csv")
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
    index = 0
    for config in configurations():
        if config["dataset"] not in args.datasets:
            continue
        for seed in args.seeds:
            index += 1
            jobs.append({
                **config,
                "seed": seed,
                "run_id": f"{timestamp}-{index:05d}",
                "cache_dir": args.cache_dir,
            })

    print(f"{len(jobs)} runs over {len(args.datasets)} datasets", flush=True)
    completed = 0
    start = time.time()
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        stream.flush()
        with ProcessPoolExecutor(max_workers=args.parallelism) as pool:
            futures = {pool.submit(run_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    row = future.result()
                    writer.writerow(row)
                except Exception as exc:  # keep a complete audit trail
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
