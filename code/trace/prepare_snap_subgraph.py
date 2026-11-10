#!/usr/bin/env python3
"""Create a reproducible real-topology subgraph from a (possibly partial) SNAP gzip."""

from __future__ import annotations

import argparse
import gzip
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple


def unique_edges(path: Path, reservoir_size: int, seed: int) -> Tuple[List[Tuple[int, int]], int]:
    rng = random.Random(seed)
    sampled: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    unique_count = 0
    try:
        with gzip.open(path, "rt", errors="ignore") as stream:
            while True:
                try:
                    line = stream.readline()
                except (EOFError, OSError):
                    # SNAP downloads can be truncated when bandwidth is limited.
                    # A truncated gzip still yields a valid prefix of real edges.
                    break
                if line == "":
                    break
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
                    continue
                source, target = int(fields[0]), int(fields[1])
                if source == target:
                    continue
                edge = (min(source, target), max(source, target))
                if edge in seen:
                    continue
                seen.add(edge)
                unique_count += 1
                if len(sampled) < reservoir_size:
                    sampled.append(edge)
                else:
                    index = rng.randrange(unique_count)
                    if index < len(sampled):
                        sampled[index] = edge
    except EOFError:
        pass
    return sampled, unique_count


def bounded_component(adjacency: Dict[int, Set[int]], maximum_nodes: int) -> Set[int]:
    if not adjacency:
        return set()
    start = min(adjacency)
    selected = {start}
    stack = [start]
    while stack and len(selected) < maximum_nodes:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in selected:
                selected.add(neighbor)
                stack.append(neighbor)
                if len(selected) >= maximum_nodes:
                    break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--reservoir-edges", type=int, default=180_000)
    parser.add_argument("--max-nodes", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    sampled, unique_count = unique_edges(
        Path(args.input), args.reservoir_edges, args.seed
    )
    adjacency: Dict[int, Set[int]] = {}
    for source, target in sampled:
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    selected = bounded_component(adjacency, args.max_nodes)
    selected_edges = set()
    for source in selected:
        for target in adjacency[source]:
            if target in selected:
                selected_edges.add((min(source, target), max(source, target)))

    order = sorted(selected)
    rank = {node: index for index, node in enumerate(order)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write("# real SNAP topology, reservoir-sampled undirected subgraph\n")
        stream.write(
            f"# source_unique_edges={unique_count} reservoir_edges={len(sampled)} "
            f"selected_nodes={len(selected)} selected_edges={len(selected_edges)}\n"
        )
        for source, target in sorted(selected_edges):
            stream.write(f"{rank[source]} {rank[target]}\n")
    print(
        f"source_unique_edges={unique_count} reservoir_edges={len(sampled)} "
        f"selected_nodes={len(selected)} selected_edges={len(selected_edges)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
