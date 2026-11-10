#!/usr/bin/env python3
"""Regenerate the long-form extended-evaluation template."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from schema import COLUMNS, blank_identity


def rows() -> List[dict]:
    out: List[dict] = []

    controlled_datasets = ["amazon0601", "web-google", "wiki-talk"]
    models = ["GCN", "GraphSAGE"]
    samplers = ["uniform", "non-uniform"]
    worker_counts = [2, 4, 8]

    # Break-even grid and native-task rows.  ogbn-arxiv uses its official split.
    for dataset in controlled_datasets + ["ogbn-arxiv"]:
        for model in models:
            for sampler in samplers:
                for workers in worker_counts:
                    block = "break-even" if dataset != "ogbn-arxiv" else "native-task"
                    control = "DGL-E-vs-EDGaE"
                    horizon = 100
                    row = blank_identity(block, dataset, model, sampler,
                                         str(workers), control, horizon)
                    if dataset == "ogbn-arxiv":
                        row["notes"] = "Official OGB split; report test accuracy."
                    out.append(row)

    # CPU-frequency emulation: even partition versus capability-aware partition.
    for policy in ["even", "capability-aware"]:
        out.append(blank_identity(
            "cpu-emulation", "wiki-talk", "GCN", "non-uniform", "8",
            f"600/1000/1500MHz-{policy}", 100,
        ))

    # Network emulation with tbf/netem.  RTT is end-to-end; delay is RTT/2.
    for control in [
        "10Mbps-5ms", "10Mbps-20ms", "10Mbps-50ms",
        "100Mbps-5ms", "100Mbps-20ms", "100Mbps-50ms",
        "1000Mbps-5ms", "1000Mbps-20ms", "1000Mbps-50ms",
        "100Mbps-20ms-1%loss",
    ]:
        out.append(blank_identity(
            "network-sensitivity", "wiki-talk", "GCN", "non-uniform", "8",
            control, 100,
        ))

    # Pre-sampling stability, including the full training trace as oracle.
    for dataset in ["web-google", "wiki-talk"]:
        for k in ["1", "2", "4", "8", "16", "full-trace"]:
            out.append(blank_identity(
                "cache-stability", dataset, "GCN", "non-uniform", "4",
                f"K={k}", 100,
            ))

    # Component ablation.
    for component in ["DGL-E", "+R", "+RC", "+RCS", "+RCSP"]:
        for dataset in ["web-google", "wiki-talk"]:
            out.append(blank_identity(
                "component-ablation", dataset, "GCN", "non-uniform", "8",
                component, 100,
            ))

    # Partitioning baselines under an equal memory ceiling.
    for partitioner in ["random", "even", "METIS", "PaGraph-style", "EDGaE"]:
        for dataset in ["web-google", "wiki-talk"]:
            out.append(blank_identity(
                "partition-policy", dataset, "GCN", "non-uniform", "8",
                partitioner, 100,
            ))

    # Score-weight sensitivity on the probability simplex.
    for weights in ["equal", "locality-heavy", "balance-heavy", "memory-heavy"]:
        for dataset in ["web-google", "wiki-talk"]:
            out.append(blank_identity(
                "score-sensitivity", dataset, "GCN", "non-uniform", "8",
                weights, 100,
            ))

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="www2027-extended-results-template.csv")
    args = parser.parse_args()
    path = Path(args.output)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows())
    print(f"Wrote {len(rows())} configuration rows and {len(COLUMNS)} columns to {path}")


if __name__ == "__main__":
    main()
