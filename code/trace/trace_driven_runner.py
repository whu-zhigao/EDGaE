#!/usr/bin/env python3
"""Trace-driven hybrid runner for placement and cache experiments.

This runner evaluates algorithmic and data-path metrics. It does not execute
DGL/PyTorch and therefore does not emit wall-clock, memory, energy, or accuracy
measurements. Its results are valid only at the trace-driven simulation level.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import random
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple


VERSION = "trace-driven-runner-v1"
DATASETS = {
    "synthetic-web-small": {"vertices": 6000, "attachments": 5},
    "synthetic-web-medium": {"vertices": 12000, "attachments": 6},
}

REAL_DATASET_FILES = {
    "amazon0601": "datasets/amazon0601.edges",
    "web-google": "datasets/web-google.edges",
    "wiki-talk": "datasets/wiki-talk.edges",
    "ogbn-arxiv": "datasets/ogbn-arxiv.edges",
}


class WebGraph:
    def __init__(self, vertices: int, attachments: int, adjacency: List[List[int]]):
        self.vertices = vertices
        self.attachments = attachments
        self.adjacency = adjacency
        self.edge_count = sum(len(neighbors) for neighbors in adjacency) // 2

    def ball(self, source: int, depth: int) -> Set[int]:
        seen = {source}
        frontier = [source]
        for _ in range(depth):
            next_frontier: List[int] = []
            for node in frontier:
                for neighbor in self.adjacency[node]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return seen


def generate_web_graph(name: str, seed: int) -> WebGraph:
    spec = DATASETS[name]
    n = spec["vertices"]
    m = spec["attachments"]
    rng = random.Random(seed)
    adjacency: List[Set[int]] = [set() for _ in range(n)]
    if n <= m:
        raise ValueError("synthetic Web graph requires n > m")
    for target in range(n):
        if target < m:
            for other in range(target):
                adjacency[target].add(other)
                adjacency[other].add(target)
        else:
            repeated: List[int] = []
            for node, neighbors in enumerate(adjacency):
                repeated.extend([node] * len(neighbors))
            choices: Set[int] = set()
            while len(choices) < min(m, target):
                choices.add(repeated[rng.randrange(len(repeated))])
            for other in choices:
                adjacency[target].add(other)
                adjacency[other].add(target)
    return WebGraph(n, m, [sorted(neighbors) for neighbors in adjacency])


def load_graph(dataset: str, seed: int, cache_dir: Path) -> WebGraph:
    external = Path(REAL_DATASET_FILES.get(dataset, ""))
    if external.exists():
        cache_path = cache_dir / f"{dataset}-external-seed{seed}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as f:
                return pickle.load(f)
        adjacency_sets: List[Set[int]] = []
        edge_count = 0
        with external.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                if len(fields) < 2:
                    raise ValueError(f"invalid edge at {external}:{line_number}")
                source, target = int(fields[0]), int(fields[1])
                largest = max(source, target)
                if largest >= len(adjacency_sets):
                    adjacency_sets.extend(
                        set() for _ in range(largest + 1 - len(adjacency_sets))
                    )
                if target not in adjacency_sets[source]:
                    adjacency_sets[source].add(target)
                    adjacency_sets[target].add(source)
                    edge_count += 1
        graph = WebGraph(len(adjacency_sets), 0, [sorted(v) for v in adjacency_sets])
        graph.edge_count = edge_count
        cache_dir.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        return graph
    if dataset not in DATASETS:
        raise ValueError(
            f"dataset {dataset!r} is not available to the offline runner; "
            "add an external edge-list runner for real datasets"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{dataset}-seed{seed}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    graph = generate_web_graph(dataset, seed)
    with path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    return graph


def worker_capabilities(control: str, workers: int) -> List[float]:
    match = re.search(r"(\d+)/(\d+)/(\d+)MHz", control)
    if not match:
        return [1.0] * workers
    frequencies = [int(value) for value in match.groups()]
    maximum = max(frequencies)
    return [frequencies[i % len(frequencies)] / maximum for i in range(workers)]


def target_loads(workers: int, capability: Sequence[float], seeds: int) -> List[int]:
    denominator = sum(capability)
    raw = [seeds * alpha / denominator for alpha in capability]
    targets = [max(1, round(value)) for value in raw]
    while sum(targets) > seeds:
        targets[targets.index(max(targets))] -= 1
    while sum(targets) < seeds:
        targets[targets.index(min(targets))] += 1
    return targets


def sample_receptive_field(
    graph: WebGraph,
    seed: int,
    depth: int,
    fanout: int,
    non_uniform: bool,
    rng: random.Random,
) -> Set[int]:
    result = {seed}
    frontier = [seed]
    for _ in range(depth):
        next_frontier: List[int] = []
        for node in frontier:
            neighbors = graph.adjacency[node]
            if not neighbors:
                continue
            if non_uniform:
                # Degree-weighted neighbour sampling without replacement when
                # possible; repeated draws model sampling with replacement.
                weights = [min(1.0 / max(1, len(graph.adjacency[n])), 1.0)
                           for n in neighbors]
                chosen: Set[int] = set()
                for _ in range(min(fanout, len(neighbors))):
                    value = rng.choices(neighbors, weights=weights, k=1)[0]
                    chosen.add(value)
            else:
                count = min(fanout, len(neighbors))
                chosen = set(rng.sample(neighbors, count))
            next_frontier.extend(chosen)
            result.update(chosen)
        frontier = list(set(next_frontier))
    return result


def access_traces(
    graph: WebGraph,
    training: Sequence[int],
    sampler: str,
    depth: int,
    fanout: int,
    batches: int,
    batch_size: int,
    rng: random.Random,
) -> List[Set[int]]:
    non_uniform = sampler in {"non-uniform", "nonuniform", "degree-weighted"}
    traces: List[Set[int]] = []
    cursor = 0
    for _ in range(batches):
        accessed: Set[int] = set()
        end = min(cursor + batch_size, len(training))
        for seed in training[cursor:end]:
            accessed.update(sample_receptive_field(
                graph, seed, depth, fanout, non_uniform, rng
            ))
        traces.append(accessed)
        cursor = end
        if cursor >= len(training):
            cursor = 0
    return traces


def partition_graph(
    graph: WebGraph,
    training: Sequence[int],
    control: str,
    workers: int,
    depth: int,
    seed: int,
    ball_fn=None,
) -> Tuple[List[Set[int]], List[Set[int]], Dict[str, float]]:
    rng = random.Random(seed * 7919 + 17)
    partitions: List[Set[int]] = [set() for _ in range(workers)]
    replicas: List[Set[int]] = [set() for _ in range(workers)]
    capability = worker_capabilities(control, workers)
    targets = target_loads(workers, capability, len(training))
    capacity = max(256, graph.vertices // 2)
    failures = 0
    fallbacks = 0

    ordered = sorted(training, key=lambda v: len(graph.adjacency[v]), reverse=True)
    for vertex in ordered:
        ball = (ball_fn or graph.ball)(vertex, depth)
        feasible = [i for i in range(workers) if len(replicas[i]) + len(ball - replicas[i]) <= capacity]
        # Eq. (2) is a hard load target, not merely a post-hoc metric. Prefer
        # workers below target; allow overflow only when every worker has reached
        # its integer target (which can occur only at the end of discrete packing).
        below_target = [i for i in feasible if len(partitions[i]) < targets[i]]
        if below_target:
            feasible = below_target
        if not feasible:
            failures += 1
            feasible = list(range(workers))
            # A fallback keeps the seed but stores only the currently available
            # prefix; this is counted and must not be hidden in the metrics.
            fallbacks += 1
            chosen_ball = set(list(ball)[:capacity])
            worker = feasible[len(partitions[0]) % workers]
            partitions[worker].add(vertex)
            replicas[worker].update(chosen_ball)
            continue

        if control in {"random", "DGL-E", "+R", "+RC"}:
            if control == "random":
                worker = feasible[rng.randrange(len(feasible))]
            elif control in {"DGL-E", "+R"}:
                worker = min(feasible, key=lambda i: len(partitions[i]))
            else:
                worker = min(feasible, key=lambda i: len(partitions[i]) / max(capability[i], 1e-9))
        elif control in {"PaGraph-style", "METIS-like"}:
            worker = max(feasible, key=lambda i: len(replicas[i] & ball))
        elif control in {"EDGaE", "+RCS", "+RCSP"}:
            best_score = -math.inf
            worker = feasible[0]
            for i in feasible:
                locality = len(replicas[i] & ball) / len(ball)
                balance = max(0.0, 1.0 - len(partitions[i]) / max(targets[i], 1))
                memory = max(0.0, 1.0 - len(replicas[i]) / capacity)
                if control == "EDGaE":
                    score = 0.45 * locality + 0.30 * balance + 0.25 * memory
                elif control == "locality-heavy":
                    score = 0.70 * locality + 0.15 * balance + 0.15 * memory
                elif control == "balance-heavy":
                    score = 0.20 * locality + 0.60 * balance + 0.20 * memory
                elif control == "memory-heavy":
                    score = 0.20 * locality + 0.20 * balance + 0.60 * memory
                else:
                    score = 1.0 / 3.0 * (locality + balance + memory)
                if score > best_score:
                    best_score = score
                    worker = i
        elif control in {"equal", "even"}:
            worker = min(feasible, key=lambda i: len(partitions[i]))
        else:
            # Score-sensitivity controls.
            locality_values = [len(replicas[i] & ball) / len(ball) for i in feasible]
            balance_values = [max(0.0, 1.0 - len(partitions[i]) / max(targets[i], 1)) for i in feasible]
            memory_values = [max(0.0, 1.0 - len(replicas[i]) / capacity) for i in feasible]
            maximum = 1e-9
            for values in (locality_values, balance_values, memory_values):
                    maximum = max(maximum, max(values) if values else 0.0)
            scores = []
            for locality, balance, memory in zip(locality_values, balance_values, memory_values):
                if control == "locality-heavy":
                    scores.append(0.70 * locality + 0.15 * balance + 0.15 * memory)
                elif control == "balance-heavy":
                    scores.append(0.20 * locality + 0.60 * balance + 0.20 * memory)
                elif control == "memory-heavy":
                    scores.append(0.20 * locality + 0.20 * balance + 0.60 * memory)
                else:
                    scores.append((locality + balance + memory) / 3.0)
            worker = feasible[scores.index(max(scores))]

        partitions[worker].add(vertex)
        replicas[worker].update(ball)

    target = target_loads(workers, capability, len(training))
    imbalance = max(abs(len(partitions[i]) / max(target[i], 1) - 1.0)
                    for i in range(workers))
    return partitions, replicas, {
        "replication_factor_vertices": sum(len(r) for r in replicas) / graph.vertices,
        "replication_factor_edges": sum(
            sum(len(graph.adjacency[v]) for v in replica) / 2
            for replica in replicas
        ) / max(graph.edge_count, 1),
        "load_imbalance": imbalance,
        "feasibility_failures": failures,
        "fallbacks": fallbacks,
    }


def frequency_counts(traces: Sequence[Set[int]], universe: Iterable[int]) -> Dict[int, int]:
    counts = {vertex: 0 for vertex in universe}
    for accessed in traces:
        for vertex in accessed:
            if vertex in counts:
                counts[vertex] += 1
    return counts


def top_cache(counts: Dict[int, int], budget: int) -> Set[int]:
    return set(sorted(counts, key=lambda v: (-counts[v], v))[:budget])


def cache_policy_set(
    graph: WebGraph,
    traces: Sequence[Set[int]],
    universe: Set[int],
    policy: str,
    budget: int,
    seed: int,
) -> Set[int]:
    if policy in {"EDGaE", "sampling-aware", "+RCS", "+RCSP"}:
        counts = frequency_counts(traces, universe)
        return top_cache(counts, budget)
    if policy in {"DGL-Degree", "PaGraph-style", "degree"}:
        return set(sorted(universe, key=lambda v: (-len(graph.adjacency[v]), v))[:budget])
    if policy == "random":
        rng = random.Random(seed)
        values = sorted(universe)
        rng.shuffle(values)
        return set(values[:budget])
    raise ValueError(f"unknown cache policy: {policy}")


def jaccard(a: Set[int], b: Set[int]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def network_control(control: str) -> Tuple[int, float]:
    match = re.match(r"(\d+)(Mbps)-(\d+)ms", control)
    if not match:
        return 1_000_000_000 // 8, 0.005
    return int(match.group(1)) * 1_000_000 // 8, float(match.group(3)) / 1000.0


def network_rtt_and_loss(control: str) -> Tuple[float, float]:
    match = re.match(r"\d+Mbps-(\d+)ms(?:-([0-9.]+)%loss)?", control)
    if not match:
        return 0.005, 0.0
    return float(match.group(1)) / 1000.0, float(match.group(2) or 0.0) / 100.0


def run_trace_simulation(manifest: Dict, cache_dir: Path) -> Dict:
    dataset = manifest["dataset"]
    seed = int(manifest["seed"])
    workers = int(manifest["workers"])
    sampler = manifest["sampler"]
    control = manifest["control"]
    graph = load_graph(dataset, seed, cache_dir)
    rng = random.Random(seed * 104729 + 11)
    training_count = min(graph.vertices, 1800)
    training = rng.sample(range(graph.vertices), training_count)

    reference_batches = 32
    evaluation_batches = 32
    batch_size = 128
    all_traces = access_traces(
        graph, training, sampler, depth=2, fanout=2,
        batches=reference_batches + evaluation_batches, batch_size=batch_size,
        rng=random.Random(seed * 7717 + 3),
    )
    reference = all_traces[:reference_batches]
    evaluation = all_traces[reference_batches:]
    feature_bytes = 100 * 4

    # By default partition seeds evenly for cache/network trace experiments;
    # partition-sensitive blocks execute the actual allocation algorithms.
    if control.startswith("K="):
        worker_of_seed = {vertex: i % workers for i, vertex in enumerate(training)}
        replicas = [set() for _ in range(workers)]
        partition_metrics = {
            "replication_factor_vertices": 0.0,
            "replication_factor_edges": 0.0,
            "load_imbalance": 0.0,
            "feasibility_failures": 0,
            "fallbacks": 0,
        }
        for vertex, worker in worker_of_seed.items():
            replicas[worker].update(graph.ball(vertex, 2))
    else:
        partitioner = control
        partitions, replicas, partition_metrics = partition_graph(
            graph, training, partitioner, workers, 2, seed
        )
        worker_of_seed = {
            vertex: worker
            for worker, vertices in enumerate(partitions) for vertex in vertices
        }

    # Cache budget is deliberately smaller than the graph so policy matters.
    per_worker_budget = max(64, graph.vertices // (workers * 4))
    if control.startswith("K="):
        k_text = control.split("=", 1)[1]
        if k_text == "full-trace":
            pre_traces = reference
        else:
            count = int(k_text)
            pre_traces = reference[:min(count, len(reference))]
    else:
        pre_traces = reference[:2]

    caches: List[Set[int]] = []
    for worker, replica in enumerate(replicas):
        worker_pre = []
        worker_universe = set(replica)
        for accessed in pre_traces:
            local = {vertex for vertex in accessed if worker_of_seed.get(vertex) == worker}
            if local:
                worker_pre.append(local)
        # If a worker has no seed in the small pre-trace, use its replica as a
        # neutral universe and still rank accesses from all reference traces.
        if not worker_pre:
            worker_pre = [set(accessed) & worker_universe for accessed in reference]
        policy = "sampling-aware" if not control.startswith("K=") else "sampling-aware"
        caches.append(cache_policy_set(
            graph, worker_pre, worker_universe, policy,
            per_worker_budget, seed + worker,
        ))

    full_cache: List[Set[int]] = []
    for worker, replica in enumerate(replicas):
        worker_full = []
        for accessed in reference:
            local = {vertex for vertex in accessed if worker_of_seed.get(vertex) == worker}
            if local:
                worker_full.append(local)
        if not worker_full:
            worker_full = [set(accessed) & set(replica) for accessed in reference]
        full_cache.append(cache_policy_set(
            graph, worker_full, set(replica), "sampling-aware",
            per_worker_budget, seed + worker,
        ))

    cache_union = set().union(*caches) if caches else set()
    full_union = set().union(*full_cache) if full_cache else set()
    topk_overlap = jaccard(cache_union, full_union)

    full_counts = frequency_counts(reference, set(range(graph.vertices)))
    ordered = sorted(full_counts, key=lambda v: (-full_counts[v], v))
    selected = set(ordered[:per_worker_budget * workers])
    non_selected = set(ordered[per_worker_budget * workers:per_worker_budget * workers + 1])
    boundary = (
        full_counts[next(iter(selected))] - full_counts[next(iter(non_selected))]
        if selected and non_selected else 0.0
    ) / reference_batches

    server_bytes = 0
    edge_bytes = 0
    accessed_all: Set[int] = set()
    for accessed in evaluation:
        server_bytes += len(accessed) * feature_bytes
        accessed_all.update(accessed)
        edge_bytes += len(accessed - cache_union) * feature_bytes

    coverage = (
        sum(len(accessed & cache_union) for accessed in evaluation)
        / sum(len(accessed) for accessed in evaluation)
    )
    effective_b, rtt = network_control(control) if manifest["block"] == "network-sensitivity" else (
        1_000_000_000 // 8, 0.005
    )
    network_rtt, network_loss = network_rtt_and_loss(control)

    result: Dict = {
        "measurement_source": VERSION,
        "measurement_valid": True,
        "evidence_level": "trace-driven-simulation",
        "environment_type": "offline-simulator",
        "hardware_platform": "none",
        "validity_scope": "algorithmic-placement-cache-communication",
        "graph_source": (
            f"external-edge-list:{REAL_DATASET_FILES[dataset]}"
            if dataset in REAL_DATASET_FILES and Path(REAL_DATASET_FILES[dataset]).exists()
            else f"synthetic-preferential-attachment:{dataset}"
        ),
        "graph_vertices": graph.vertices,
        "graph_edges": graph.edge_count,
        "runner_version": VERSION,
        "calibration_source": "none",
        **partition_metrics,
        "C_srv_bytes": server_bytes,
        "C_edg_bytes": edge_bytes,
        "effective_B_bytes_s": effective_b,
        "network_rtt_s": network_rtt,
        "network_loss_rate": network_loss,
        "cache_hit_rate": 1.0 - edge_bytes / max(server_bytes, 1),
        "cache_coverage": coverage,
        "topk_jaccard": topk_overlap,
        "boundary_gap": boundary,
        "notes": (
            f"reference_batches={reference_batches};evaluation_batches={evaluation_batches};"
            f"per_worker_cache_records={per_worker_budget};rtt_s={rtt}"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--cache-dir", default="experiment-cache")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    real_available = Path(
        REAL_DATASET_FILES.get(manifest.get("dataset", ""), "")
    ).exists()
    if manifest.get("dataset") not in DATASETS and not real_available:
        raise SystemExit(
            "trace_driven_runner.py intentionally refuses to substitute a synthetic graph "
            "for a real dataset. Provide an external edge-list/runner for that dataset."
        )
    print(json.dumps(run_trace_simulation(manifest, Path(args.cache_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
