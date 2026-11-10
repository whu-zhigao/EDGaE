#!/usr/bin/env python3
"""Build a compact matrix for the three-level hybrid evaluation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from schema import COLUMNS, blank_identity


def rows() -> List[dict]:
    out: List[dict] = []
    datasets = ["synthetic-web-small", "synthetic-web-medium"]

    for dataset in datasets:
        for k in ["1", "2", "4", "8", "16", "full-trace"]:
            out.append(blank_identity(
                "cache-stability", dataset, "GCN", "non-uniform", "4", f"K={k}"
            ))
        for partitioner in ["random", "even", "PaGraph-style", "EDGaE", "METIS-like"]:
            out.append(blank_identity(
                "partition-policy", dataset, "GCN", "non-uniform", "8", partitioner
            ))
        for component in ["DGL-E", "+R", "+RC", "+RCS", "+RCSP"]:
            out.append(blank_identity(
                "component-ablation", dataset, "GCN", "non-uniform", "8", component
            ))
        for weights in ["equal-weights", "locality-heavy", "balance-heavy", "memory-heavy"]:
            out.append(blank_identity(
                "score-sensitivity", dataset, "GCN", "non-uniform", "8", weights
            ))

    for control in [
        "10Mbps-5ms", "10Mbps-20ms", "10Mbps-50ms",
        "100Mbps-5ms", "100Mbps-20ms", "100Mbps-50ms",
        "1000Mbps-5ms", "1000Mbps-20ms", "1000Mbps-50ms",
        "100Mbps-20ms-1%loss",
    ]:
        out.append(blank_identity(
            "network-sensitivity", "synthetic-web-medium", "GCN",
            "non-uniform", "8", control,
        ))

    for policy in ["even", "capability-aware"]:
        out.append(blank_identity(
            "cpu-emulation", "synthetic-web-medium", "GCN", "non-uniform", "8",
            f"600/1000/1500MHz-{policy}",
        ))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="www2027-hybrid-results-template.csv")
    args = parser.parse_args()
    matrix = rows()
    with Path(args.output).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(matrix)
    print(f"wrote {len(matrix)} rows to {args.output}")


if __name__ == "__main__":
    main()
