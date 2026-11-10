#!/usr/bin/env python3
"""Build the real-topology trace-driven experiment matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from schema import COLUMNS, blank_identity


def rows(dataset: str) -> List[dict]:
    out: List[dict] = []
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
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="www2027-real-trace-template.csv")
    parser.add_argument("--dataset", default="web-google")
    args = parser.parse_args()
    matrix = rows(args.dataset)
    with Path(args.output).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(matrix)
    print(f"wrote {len(matrix)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
