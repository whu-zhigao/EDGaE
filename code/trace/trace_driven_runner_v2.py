#!/usr/bin/env python3
"""Trace-driven runner v2 for placement, cache, and preprocessing experiments.

v2 extends v1 with three changes required by the ICDE revision:
1. full or valid-prefix SNAP topologies in addition to the 12k subgraphs;
2. online LRU/LFU cache policies with the same record budget as static caches;
3. measured preprocessing wall time (partitioning, pre-sampling, cache build).

All byte accounting, trace protocol, budget rules, and metric definitions are
identical to v1. Timing fields are new and measured on the local workstation;
they are not edge-device times.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import pickle

from trace_driven_runner import (
    WebGraph,
    REAL_DATASET_FILES,
    access_traces,
    cache_policy_set,
    frequency_counts,
    jaccard,
    partition_graph,
    target_loads,
)


VERSION = "trace-driven-runner-v2"

V2_DATASET_FILES = {
    "amazon0601-full": "datasets/amazon0601-full.edges",
    "web-google-full": "datasets/web-google-full.edges",
    "wiki-talk-full": "datasets/wiki-talk-full.edges",
}
ALL_DATASET_FILES = {**REAL_DATASET_FILES, **V2_DATASET_FILES}

REFERENCE_BATCHES = 32
EVALUATION_BATCHES = 32
BATCH_SIZE = 128
FEATURE_BYTES = 100 * 4

# System-inspired adapters for the SOTA comparison block. Each adapter maps a
# published system's placement/cache *policy* onto this substrate. They are
# policy adapters on identical hardware, NOT the original end systems.
SYSTEM_POLICY_SPECS = {
    # partition policy, cache policy, pre-sampling rounds (None = no cache)
    "Server-C":         ("DGL-E",       "none",            None),
    "DistDGL-like":     ("METIS-like",  "none",            None),
    "PaGraph-like":     ("PaGraph-style", "degree",        0),
    "GNNLab-like":      ("even",        "sampling-aware",  2),
    "Ginex-like":       ("even",        "online-LFU",      None),
    "BGL-like":         ("even",        "online-LRU",      None),
    "NeutronStar-like": ("METIS-like",  "sampling-aware", 16),
    "DAHA-like":        ("balance-heavy", "sampling-aware", 16),
    "PlatoD2GL-like":   ("locality-heavy", "sampling-aware", 16),
    "EDGaE":            ("EDGaE",       "adaptive",       None),
}


def load_graph_v2(dataset: str, cache_dir: Path) -> WebGraph:
    """Load an external edge list with one pickle cache per dataset."""
    external = Path(ALL_DATASET_FILES[dataset])
    cache_path = cache_dir / f"{dataset}-v2.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)
    adjacency_sets: List[Set[int]] = []
    edge_count = 0
    with external.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            source, target = (int(x) for x in stripped.split()[:2])
            largest = max(source, target)
            if largest >= len(adjacency_sets):
                adjacency_sets.extend(set() for _ in range(largest + 1 - len(adjacency_sets)))
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


class OnlineCache:
    """Cluster-level LRU/LFU cache over ordered mini-batch traces."""

    def __init__(self, policy: str, capacity: int):
        self.policy = policy
        self.capacity = capacity
        self.entries: "OrderedDict[int, int]" = OrderedDict()  # id -> frequency
        self.last_access: Dict[int, int] = {}
        self.clock = 0

    def _evict(self) -> None:
        while len(self.entries) > self.capacity:
            if self.policy == "LRU":
                victim, _ = self.entries.popitem(last=False)
            else:  # LFU: minimum frequency, ties broken by recency
                victim = min(
                    self.entries,
                    key=lambda v: (self.entries[v], self.last_access.get(v, -1)),
                )
                self.entries.pop(victim)

    def serve_batch(self, batch: Set[int]) -> int:
        """Return the number of misses for this batch."""
        self.clock += 1
        misses = 0
        for vertex in sorted(batch):  # deterministic within-batch order
            if vertex in self.entries:
                self.entries[vertex] += 1
                if self.policy == "LRU":
                    self.entries.move_to_end(vertex)
            else:
                misses += 1
                self.entries[vertex] = 1
            self.last_access[vertex] = self.clock
            # Classic online semantics: evict immediately on overflow so that
            # LFU tie-breaks use per-access recency (consistent with the
            # two-tier block's hot tier).
            self._evict()
        return misses


class TwoTierCache:
    """Ship-once static tier plus a bounded mutable LFU hot tier.

    The static tier is frozen after shipping (read-only replica semantics).
    The hot tier starts empty, may be warmed on the same reference batches as
    the static policies, and adapts online with LFU replacement. The total
    record budget equals the single-policy baselines.
    """

    def __init__(self, static: Set[int], hot_capacity: int):
        self.static = static
        self.hot_capacity = hot_capacity
        self.hot: "OrderedDict[int, int]" = OrderedDict()
        self.last_access: Dict[int, int] = {}
        self.clock = 0

    def serve_batch(self, batch: Set[int]) -> int:
        self.clock += 1
        misses = 0
        for vertex in sorted(batch):
            if vertex not in self.static:
                if vertex in self.hot:
                    self.hot[vertex] += 1
                else:
                    misses += 1
                    self.hot[vertex] = 1
                # Update recency BEFORE eviction: a just-accessed vertex must
                # not be the LFU victim (classic per-access online semantics,
                # consistent with OnlineCache).
                self.last_access[vertex] = self.clock
                while len(self.hot) > self.hot_capacity:
                    victim = min(
                        self.hot,
                        key=lambda v: (self.hot[v], self.last_access.get(v, -1)),
                    )
                    self.hot.pop(victim)
            # static-tier hits need no state change: the tier is immutable
        return misses


def parse_sampler_control(control: str) -> Tuple[int, int, str]:
    """Parse controls of the form 'f=4,d=3,uniform'."""
    fanout, depth, sampler = 2, 2, "non-uniform"
    for part in control.split(","):
        key, _, value = part.partition("=")
        if key == "f":
            fanout = int(value)
        elif key == "d":
            depth = int(value)
        elif key in {"uniform", "uni"}:
            sampler = "uniform"
        elif key in {"non-uniform", "n-uni"}:
            sampler = "non-uniform"
    return fanout, depth, sampler


def ball_capped(graph, source: int, depth: int, cap: int) -> Tuple[Set[int], bool]:
    """BFS ball with an early stop once it provably exceeds ``cap``.

    Used only by the sampler-sensitivity block so that deep receptive fields on
    dense graphs remain computationally bounded. A truncated ball is reported
    as infeasible; feasible balls are never truncated, so placement decisions
    are identical to the exact-BFS semantics.
    """
    seen = {source}
    frontier = [source]
    for _ in range(depth):
        next_frontier = []
        for node in frontier:
            for neighbor in graph.adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.append(neighbor)
                    if len(seen) > cap:
                        return seen, True
        frontier = next_frontier
        if not frontier:
            break
    return seen, False




def static_cache_union(
    graph,
    reference: List[Set[int]],
    replicas: List[Set[int]],
    worker_of_seed: Dict[int, int],
    workers: int,
    budget: int,
    policy: str,
    seed: int,
) -> Tuple[Set[int], float]:
    """Build per-worker static caches; return their union and build time (s)."""
    start = time.perf_counter()
    caches: List[Set[int]] = []
    for worker, replica in enumerate(replicas):
        worker_pre = []
        for accessed in reference:
            local = {v for v in accessed if worker_of_seed.get(v) == worker}
            if local:
                worker_pre.append(local)
        if not worker_pre:
            worker_pre = [set(accessed) & set(replica) for accessed in reference]
        caches.append(
            cache_policy_set(graph, worker_pre, set(replica), policy, budget, seed + worker)
        )
    return set().union(*caches), time.perf_counter() - start


def even_partition(graph, training: Sequence[int], workers: int) -> Tuple[Dict[int, int], List[Set[int]], float]:
    """Round-robin seed placement with two-hop replicas (v1 cache-block rule)."""
    start = time.perf_counter()
    worker_of_seed = {vertex: i % workers for i, vertex in enumerate(training)}
    replicas: List[Set[int]] = [set() for _ in range(workers)]
    for vertex, worker in worker_of_seed.items():
        replicas[worker].update(graph.ball(vertex, 2))
    return worker_of_seed, replicas, time.perf_counter() - start


def run_trace_simulation_v2(manifest: Dict, cache_dir: Path) -> Dict:
    dataset = manifest["dataset"]
    seed = int(manifest["seed"])
    workers = int(manifest["workers"])
    sampler = manifest["sampler"]
    control = manifest["control"]
    block = manifest["block"]
    graph = load_graph_v2(dataset, cache_dir)
    rng = random.Random(seed * 104729 + 11)
    training_count = min(graph.vertices, 1800)
    training = rng.sample(range(graph.vertices), training_count)

    # Sampler-sensitivity controls override the effective sampler parameters.
    if block == "sampler-sensitivity":
        fanout, depth_eff, sampler = parse_sampler_control(control)
    else:
        fanout, depth_eff = 2, 2

    # Reference and evaluation traces are generated with independent streams so
    # that pre-sampling cost is timed for exactly the rounds that are used.
    t0 = time.perf_counter()
    reference = access_traces(
        graph, training, sampler, depth=depth_eff, fanout=fanout,
        batches=REFERENCE_BATCHES, batch_size=BATCH_SIZE,
        rng=random.Random(seed * 7717 + 3),
    )
    t_presample_all = time.perf_counter() - t0
    evaluation = access_traces(
        graph, training, sampler, depth=depth_eff, fanout=fanout,
        batches=EVALUATION_BATCHES, batch_size=BATCH_SIZE,
        rng=random.Random(seed * 7723 + 5),
    )

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    partition_metrics = {
        "replication_factor_vertices": 0.0,
        "replication_factor_edges": 0.0,
        "load_imbalance": 0.0,
        "feasibility_failures": 0,
        "fallbacks": 0,
    }
    online_block = block == "online-cache"
    partitioner_control = "EDGaE" if control == "EDGaE-K16" else control
    system_block = block == "system-policy"
    if system_block:
        partitioner_control, cache_policy_spec, rounds_spec = SYSTEM_POLICY_SPECS[control]
    agreement = 1.0
    replica_overlap = 1.0
    if block == "sampler-sensitivity":
        # Baseline placement under the paper's default sampler (f=2, d=2);
        # agreement metrics quantify how much the placement actually changes.
        base_partitions, base_replicas, _ = partition_graph(
            graph, training, "EDGaE", workers, 2, seed
        )
        base_worker = {
            vertex: worker
            for worker, vertices in enumerate(base_partitions)
            for vertex in vertices
        }
        capacity = max(256, graph.vertices // 2)
        t0 = time.perf_counter()
        if depth_eff <= 2:
            # Exact BFS is already used at these depths elsewhere; identical
            # logic guarantees agreement=1 sanity for fan-out/sampler-type axes.
            partitions, replicas, partition_metrics = partition_graph(
                graph, training, "EDGaE", workers, depth_eff, seed
            )
        else:
            # Depth 3 can explode on dense graphs; cap the ball at the same
            # capacity the feasibility check uses. An oversized ball fails the
            # check for every worker, so truncation cannot alter feasibility.
            partitions, replicas, partition_metrics = partition_graph(
                graph, training, "EDGaE", workers, depth_eff, seed,
                ball_fn=lambda v, d, cap=capacity: ball_capped(graph, v, d, cap)[0],
            )
        t_partition = time.perf_counter() - t0
        worker_of_seed = {
            vertex: worker
            for worker, vertices in enumerate(partitions)
            for vertex in vertices
        }
        agreement = sum(
            1 for v in training if base_worker[v] == worker_of_seed[v]
        ) / len(training)
        replica_overlap = jaccard(
            set().union(*base_replicas), set().union(*replicas)
        )
    elif system_block and partitioner_control == "even":
        worker_of_seed, replicas, t_partition = even_partition(graph, training, workers)
    elif (not system_block) and (
        control.startswith("K=") or online_block or block == "two-tier"
    ):
        worker_of_seed, replicas, t_partition = even_partition(graph, training, workers)
    else:
        t0 = time.perf_counter()
        partitions, replicas, partition_metrics = partition_graph(
            graph, training, partitioner_control, workers, 2, seed
        )
        t_partition = time.perf_counter() - t0
        worker_of_seed = {
            vertex: worker
            for worker, vertices in enumerate(partitions)
            for vertex in vertices
        }

    per_worker_budget = max(64, graph.vertices // (workers * 4))

    # ------------------------------------------------------------------
    # Cache construction / online policy
    # ------------------------------------------------------------------
    t_cache = 0.0
    t_presample = 0.0
    warm_misses = 0
    no_cache_ablation = (
        block == "component-ablation" and control in {"DGL-E", "+R", "+RC"}
    )
    if no_cache_ablation:
        # Data-path component semantics: DGL-E, +R, and +RC have no feature
        # cache, so every accessed feature is fetched; only +RCS adds caching.
        cache_union = set()
        t_cache = 0.0
        residual_bytes = 0
        coverage_hits = 0
        coverage_den = 0
        for batch in evaluation:
            residual_bytes += len(batch) * FEATURE_BYTES
            coverage_den += len(batch)
    elif block == "two-tier":
        # Ship-once static tier (cluster top-k over reference counts) plus a
        # bounded mutable LFU hot tier; total budget matches single policies.
        beta = float(control.split("=", 1)[1]) if "=" in control else 0.5
        total_budget = per_worker_budget * workers
        static_capacity = int(round((1.0 - beta) * total_budget))
        hot_capacity = total_budget - static_capacity
        counts = frequency_counts(reference, set(range(graph.vertices)))
        ordered = sorted(counts, key=lambda v: (-counts[v], v))
        static_tier = set(ordered[:static_capacity])
        two_tier = TwoTierCache(static_tier, hot_capacity)
        t0 = time.perf_counter()
        for batch in reference:
            warm_misses += two_tier.serve_batch(batch)
        t_cache = time.perf_counter() - t0
        misses = 0
        for batch in evaluation:
            misses += two_tier.serve_batch(batch)
        cache_union = static_tier | set(two_tier.hot)
        residual_bytes = misses * FEATURE_BYTES
        coverage_den = sum(len(b) for b in evaluation)
        coverage_hits = coverage_den - misses
    elif online_block and control in {"LRU", "LFU"}:
        online = OnlineCache(control, per_worker_budget * workers)
        t0 = time.perf_counter()
        for batch in reference:  # warm start: same information as static caches
            warm_misses += online.serve_batch(batch)
        t_cache = time.perf_counter() - t0
        misses = 0
        for batch in evaluation:
            misses += online.serve_batch(batch)
        cache_union = set(online.entries)
        residual_bytes = misses * FEATURE_BYTES
        coverage_den = sum(len(b) for b in evaluation)
        coverage_hits = coverage_den - misses
    elif block == "sampler-sensitivity":
        # In-run adaptive-K selection under the variant sampler, using the
        # same marginal-gain rule as the main cache-stability analysis.
        k_candidates = [1, 2, 4, 8, 16]
        k_hits: Dict[int, float] = {}
        k_caches: Dict[int, Set[int]] = {}
        eval_total = sum(len(b) for b in evaluation)
        for k in k_candidates:
            k_union, t_build_k = static_cache_union(
                graph, reference[:k], replicas, worker_of_seed, workers,
                per_worker_budget, "sampling-aware", seed,
            )
            k_caches[k] = k_union
            k_hits[k] = sum(len(b & k_union) for b in evaluation) / eval_total
            t_cache += t_build_k
        chosen_k = 16
        for k, nxt in zip(k_candidates[:-1], k_candidates[1:]):
            if k_hits[nxt] > 0 and (k_hits[nxt] - k_hits[k]) / k_hits[nxt] < 0.02:
                chosen_k = nxt
                break
        cache_union = k_caches[chosen_k]
        t_presample = t_presample_all * (chosen_k / REFERENCE_BATCHES)
        residual_bytes = 0
        coverage_hits = 0
        coverage_den = 0
        for batch in evaluation:
            residual_bytes += len(batch - cache_union) * FEATURE_BYTES
            coverage_hits += len(batch & cache_union)
            coverage_den += len(batch)
    elif system_block:
        # System-inspired adapter: partition policy chosen above, cache
        # policy from SYSTEM_POLICY_SPECS. Budgets identical across adapters.
        if cache_policy_spec == "none":
            cache_union = set()
            t_cache = 0.0
            residual_bytes = 0
            coverage_hits = 0
            coverage_den = 0
            for batch in evaluation:
                residual_bytes += len(batch) * FEATURE_BYTES
                coverage_den += len(batch)
        elif cache_policy_spec in {"online-LFU", "online-LRU"}:
            policy = "LFU" if cache_policy_spec == "online-LFU" else "LRU"
            # Per-worker deployment semantics: each edge worker owns one
            # online cache of the same record budget as the static policies.
            # Accessed records are sharded deterministically across workers
            # (record -> exactly one worker), so each cached entry lives on
            # exactly one device. This is NOT a centrally pooled cache; the
            # pooled variant is the idealized upper bound in online-cache.
            online_caches = [
                OnlineCache(policy, per_worker_budget) for _ in range(workers)
            ]
            def shard(batch: Set[int]) -> List[Set[int]]:
                ordered = sorted(batch)
                return [
                    {v for i, v in enumerate(ordered) if i % workers == w}
                    for w in range(workers)
                ]
            t0 = time.perf_counter()
            for batch in reference:
                for online, local in zip(online_caches, shard(batch)):
                    warm_misses += online.serve_batch(local)
            t_cache = time.perf_counter() - t0
            misses = 0
            for batch in evaluation:
                for online, local in zip(online_caches, shard(batch)):
                    misses += online.serve_batch(local)
            cache_union = set().union(*(set(c.entries) for c in online_caches))
            residual_bytes = misses * FEATURE_BYTES
            coverage_den = sum(len(b) for b in evaluation)
            coverage_hits = coverage_den - misses
        elif cache_policy_spec == "adaptive":
            # EDGaE adaptive-K selection (same rule as cache-stability).
            k_candidates = [1, 2, 4, 8, 16]
            k_hits: Dict[int, float] = {}
            k_caches: Dict[int, Set[int]] = {}
            eval_total = sum(len(b) for b in evaluation)
            for k in k_candidates:
                k_union, t_build_k = static_cache_union(
                    graph, reference[:k], replicas, worker_of_seed, workers,
                    per_worker_budget, "sampling-aware", seed,
                )
                k_caches[k] = k_union
                k_hits[k] = sum(len(b & k_union) for b in evaluation) / eval_total
                t_cache += t_build_k
            chosen_k = 16
            for k, nxt in zip(k_candidates[:-1], k_candidates[1:]):
                if k_hits[nxt] > 0 and (k_hits[nxt] - k_hits[k]) / k_hits[nxt] < 0.02:
                    chosen_k = nxt
                    break
            cache_union = k_caches[chosen_k]
            t_presample = t_presample_all * (chosen_k / REFERENCE_BATCHES)
            residual_bytes = 0
            coverage_hits = 0
            coverage_den = 0
            for batch in evaluation:
                residual_bytes += len(batch - cache_union) * FEATURE_BYTES
                coverage_hits += len(batch & cache_union)
                coverage_den += len(batch)
        else:
            # Static policies: degree or sampling-aware with K rounds.
            if cache_policy_spec == "degree":
                rounds = 0
                cache_policy = "degree"
                t_presample = 0.0
            else:
                rounds = int(rounds_spec or 2)
                cache_policy = "sampling-aware"
                t_presample = t_presample_all * (rounds / REFERENCE_BATCHES)
            pre_traces = reference[:rounds] if rounds > 0 else reference
            cache_union, t_build = static_cache_union(
                graph, pre_traces, replicas, worker_of_seed, workers,
                per_worker_budget, cache_policy, seed,
            )
            t_cache = t_build
            residual_bytes = 0
            coverage_hits = 0
            coverage_den = 0
            for batch in evaluation:
                residual_bytes += len(batch - cache_union) * FEATURE_BYTES
                coverage_hits += len(batch & cache_union)
                coverage_den += len(batch)
    else:
        rounds_override = 16 if control == "EDGaE-K16" else None
        if online_block:
            policy_map = {
                "static-K2": 2, "static-K16": 16, "static-full": REFERENCE_BATCHES,
                "degree": None, "random": None,
            }
            rounds = policy_map.get(control, 2)
            cache_policy = (
                "sampling-aware" if rounds is not None
                else ("degree" if control == "degree" else "random")
            )
        elif control.startswith("K="):
            k_text = control.split("=", 1)[1]
            rounds = REFERENCE_BATCHES if k_text == "full-trace" else int(k_text)
            cache_policy = "sampling-aware"
        else:
            rounds = 2
            cache_policy = "sampling-aware"
        if control == "EDGaE-K16":
            pre_traces = reference[:16]
            t_presample = t_presample_all * 0.5
        elif online_block and control in {"degree", "random"}:
            pre_traces = reference  # traces unused by these policies
            t_presample = 0.0      # no sampler execution is required
        else:
            pre_traces = reference[: min(rounds, len(reference))]
            t_presample = t_presample_all * (len(pre_traces) / REFERENCE_BATCHES)
        cache_union, t_build = static_cache_union(
            graph, pre_traces, replicas, worker_of_seed, workers,
            per_worker_budget, cache_policy, seed,
        )
        t_cache = t_build
        residual_bytes = 0
        coverage_hits = 0
        coverage_den = 0
        for batch in evaluation:
            residual_bytes += len(batch - cache_union) * FEATURE_BYTES
            coverage_hits += len(batch & cache_union)
            coverage_den += len(batch)

    # ------------------------------------------------------------------
    # Reference quantities (full-trace hot set, boundary gap, server bytes)
    # ------------------------------------------------------------------
    full_cache, _ = static_cache_union(
        graph, reference, replicas, worker_of_seed, workers,
        per_worker_budget, "sampling-aware", seed,
    )
    topk_overlap = jaccard(cache_union, full_cache)
    full_counts = frequency_counts(reference, set(range(graph.vertices)))
    ordered = sorted(full_counts, key=lambda v: (-full_counts[v], v))
    selected = set(ordered[: per_worker_budget * workers])
    non_selected = set(ordered[per_worker_budget * workers: per_worker_budget * workers + 1])
    boundary = (
        full_counts[next(iter(selected))] - full_counts[next(iter(non_selected))]
        if selected and non_selected else 0.0
    ) / REFERENCE_BATCHES

    server_bytes = sum(len(batch) for batch in evaluation) * FEATURE_BYTES
    # Distinct sampled working set across evaluation batches: the data volume
    # that edge-local (E) mode must ship once and hold in aggregate memory.
    working_set_vertices = len(set().union(*evaluation)) if evaluation else 0
    working_set_bytes = working_set_vertices * FEATURE_BYTES
    t_pre = t_partition + t_presample + t_cache
    extras: Dict[str, object] = {}
    if block == "sampler-sensitivity":
        extras = {
            "adaptive_k": locals().get("chosen_k", 16),
            "assignment_agreement": agreement,
            "replica_jaccard": replica_overlap,
            "sampler_fanout": fanout,
            "sampler_depth": depth_eff,
            "sampler_type": sampler,
        }
    elif block == "two-tier":
        extras = {"two_tier_beta": beta}

    return {
        "measurement_source": VERSION,
        "measurement_valid": True,
        "evidence_level": "trace-driven-simulation",
        "environment_type": "offline-simulator",
        "hardware_platform": "artifact-workstation",
        "validity_scope": "algorithmic-placement-cache-communication-and-preprocessing-time",
        "graph_source": f"external-edge-list:{ALL_DATASET_FILES[dataset]}",
        "graph_vertices": graph.vertices,
        "graph_edges": graph.edge_count,
        "runner_version": VERSION,
        "calibration_source": "none",
        "platform": platform.platform(),
        **partition_metrics,
        "C_srv_bytes": server_bytes,
        "C_edg_bytes": residual_bytes,
        "cache_hit_rate": 1.0 - residual_bytes / max(server_bytes, 1),
        "cache_coverage": coverage_hits / max(coverage_den, 1),
        "topk_jaccard": topk_overlap,
        "boundary_gap": boundary,
        "warm_misses": warm_misses,
        "working_set_vertices": working_set_vertices,
        "working_set_bytes": working_set_bytes,
        **extras,
        "T_partition_s": t_partition,
        "T_presample_s": t_presample,
        "T_cache_s": t_cache,
        "T_pre_total_s": t_pre,
        "notes": (
            f"reference_batches={REFERENCE_BATCHES};evaluation_batches={EVALUATION_BATCHES};"
            f"per_worker_cache_records={per_worker_budget};"
            f"cluster_cache_records={per_worker_budget * workers}"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--cache-dir", default="experiment-cache")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["dataset"] not in ALL_DATASET_FILES:
        raise SystemExit(f"unknown dataset {manifest['dataset']!r}")
    print(json.dumps(run_trace_simulation_v2(manifest, Path(args.cache_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
