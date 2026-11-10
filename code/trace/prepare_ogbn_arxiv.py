#!/usr/bin/env python3
"""Materialize the official ogbn-arxiv topology as a zero-based edge list."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path


def read_ids(path: Path) -> list[int]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return [int(row[0]) for row in csv.reader(stream)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/ogbn-arxiv/arxiv")
    parser.add_argument("--output", default="datasets/ogbn-arxiv.edges")
    parser.add_argument("--metadata", default="datasets/ogbn-arxiv-metadata.json")
    args = parser.parse_args()

    root = Path(args.input_dir)
    edge_path = root / "raw" / "edge.csv.gz"
    label_path = root / "raw" / "node-label.csv.gz"
    split_dir = root / "split" / "time"

    edges: set[tuple[int, int]] = set()
    max_node = -1
    with gzip.open(edge_path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if len(row) < 2:
                continue
            source, target = int(row[0]), int(row[1])
            max_node = max(max_node, source, target)
            edges.add((min(source, target), max(source, target)))

    labels = read_ids(label_path)
    splits = {
        name: read_ids(split_dir / f"{name}.csv.gz")
        for name in ["train", "valid", "test"]
    }
    if len(labels) != max_node + 1:
        raise ValueError(
            f"ogbn-arxiv metadata mismatch: {len(labels)} labels but "
            f"max node ID is {max_node}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# official ogbn-arxiv topology, undirected simplification\n")
        stream.write(f"# selected_nodes={len(labels)} selected_edges={len(edges)}\n")
        for source, target in sorted(edges):
            stream.write(f"{source} {target}\n")

    metadata = {
        "dataset": "ogbn-arxiv",
        "source": "OGB official arxiv archive",
        "vertices": len(labels),
        "undirected_edges": len(edges),
        "features": str((root / "raw" / "node-feat.csv.gz").resolve()),
        "labels": str(label_path.resolve()),
        "splits": {name: len(ids) for name, ids in splits.items()},
    }
    Path(args.metadata).write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
