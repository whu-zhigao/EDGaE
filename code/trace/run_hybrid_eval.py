#!/usr/bin/env python3
"""Run the offline and parameterized layers of the hybrid evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List

from emulated_model import DEFAULT_CALIBRATION, derive_emulated_metrics, load_calibration
from schema import COLUMNS
from trace_driven_runner import run_trace_simulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="www2027-hybrid-results-template.csv")
    parser.add_argument("--trace-output", default="www2027-hybrid-results-trace.csv")
    parser.add_argument("--emulated-output", default="www2027-hybrid-results-emulated.csv")
    parser.add_argument("--raw-dir", default="hybrid-raw")
    parser.add_argument("--cache-dir", default="experiment-cache")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--blocks", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--max-runs", type=int)
    return parser.parse_args()


def write_row(path: Path, row: Dict[str, object]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerow(row)


def merged_row(base: Dict[str, str], result: Dict[str, object], run_id: str) -> Dict[str, object]:
    row: Dict[str, object] = dict(base)
    row["run_id"] = run_id
    row["measurement_utc"] = datetime.now(timezone.utc).isoformat()
    for key, value in result.items():
        if key in COLUMNS:
            row[key] = value
    return row


def selected(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if args.blocks and row["block"] not in args.blocks:
        return False
    if args.datasets and row["dataset"] not in args.datasets:
        return False
    return True


def main() -> int:
    args = parse_args()
    template = Path(args.template)
    trace_output = Path(args.trace_output)
    emulated_output = Path(args.emulated_output)
    raw_dir = Path(args.raw_dir)
    cache_dir = Path(args.cache_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    calibration = load_calibration(args.calibration)

    for output in (trace_output, emulated_output):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")

    with template.open(newline="", encoding="utf-8") as f:
        configurations = [row for row in csv.DictReader(f) if selected(row, args)]
    if args.max_runs:
        configurations = configurations[:args.max_runs]

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    completed = 0
    for index, base in enumerate(configurations, start=1):
        for seed in args.seeds:
            run_id = f"{timestamp}-{index:04d}-{base['block']}-{seed}"
            manifest = {
                **base,
                "seed": seed,
                "run_id": run_id,
                "horizon_epochs": int(base.get("horizon_epochs") or 100),
            }
            print(f"[{index}/{len(configurations)} seed={seed}] {run_id}")
            trace = run_trace_simulation(manifest, cache_dir)
            trace["seed"] = seed
            write_row(trace_output, merged_row(base, trace, run_id))

            emulated = derive_emulated_metrics(trace, manifest, calibration)
            emulated["seed"] = seed
            # Preserve trace-level communication/cache/layout values in the
            # parameterized row so one CSV can be analyzed end-to-end.
            for key in [
                "C_srv_bytes", "C_edg_bytes", "effective_B_bytes_s",
                "cache_hit_rate", "cache_coverage", "K", "topk_jaccard",
                "boundary_gap", "replication_factor_vertices",
                "replication_factor_edges", "load_imbalance",
                "feasibility_failures", "fallbacks",
            ]:
                if key in trace:
                    emulated.setdefault(key, trace[key])
            write_row(emulated_output, merged_row(base, emulated, run_id))
            (raw_dir / f"{run_id}.json").write_text(json.dumps({
                "manifest": manifest,
                "trace_result": trace,
                "emulated_result": emulated,
                "calibration": calibration if index == 1 and seed == args.seeds[0] else None,
            }, indent=2, sort_keys=True) + "\n")
            completed += 1

    print(
        f"completed={completed} configurations={len(configurations)}; "
        f"trace_output={trace_output}; emulated_output={emulated_output}; "
        f"publishable_emulation={bool(calibration.get('publishable', False))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
