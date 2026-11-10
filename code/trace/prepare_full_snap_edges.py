#!/usr/bin/env python3
"""Prepare full or valid-prefix SNAP topologies for the v2 trace runner.

gzip streams that end before the end-of-stream marker still expose a valid
prefix. This script keeps every edge that can be decoded before truncation,
maps node IDs to a compact zero-based range, removes self-loops and duplicate
undirected edges, and records the exact provenance in a header comment.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path


def read_valid_edges(path: Path) -> tuple[list[tuple[int, int]], bool]:
    """Yield decodable undirected edges from a possibly truncated gzip file."""
    edges: list[tuple[int, int]] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        try:
            for line in stream:
                if line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2:
                    continue
                try:
                    source, target = int(fields[0]), int(fields[1])
                except ValueError:
                    continue
                if source != target:
                    edges.append((min(source, target), max(source, target)))
        except EOFError:
            return edges, True  # stream truncated after a valid prefix
    return edges, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    ids: dict[int, int] = {}
    edges: set[tuple[int, int]] = set()
    raw_edges, truncated = read_valid_edges(args.source)
    for source, target in raw_edges:
        a = ids.setdefault(source, len(ids))
        b = ids.setdefault(target, len(ids))
        if a != b:
            edges.add((min(a, b), max(a, b)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        stream.write(
            f"# SNAP topology, undirected simplification, zero-based compact IDs\n"
            f"# source_file={args.source.name} complete_stream={not truncated}\n"
            f"# distinct_source_nodes={len(ids)} unique_undirected_edges={len(edges)}\n"
        )
        for source, target in sorted(edges):
            stream.write(f"{source} {target}\n")
    print(
        f"{args.output}: vertices={len(ids)} edges={len(edges)} "
        f"complete={not truncated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
