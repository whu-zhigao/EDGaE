#!/usr/bin/env python3
"""EDGaE placement and cache construction ported from the trace runner.

Preserves the paper's semantics: greedy equal-weight scoring, per-worker
vertex capacity max(256, V/2), hard feasibility check with fallback counting,
and per-worker top-hotness caches from K pre-sampling rounds.
"""

from __future__ import annotations

import time
from typing import Dict, List, Set

import numpy as np


def ball(indptr, indices, source: int, depth: int) -> Set[int]:
    seen = {source}
    frontier = [source]
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for nb in indices[indptr[node]:indptr[node + 1]]:
                nb = int(nb)
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
        if not frontier:
            break
    return seen


def place(indptr, indices, degrees, training, workers=8, depth=2, seed=0):
    """Greedy EDGaE placement. Returns partitions, replicas, metrics, time."""
    t0 = time.perf_counter()
    rng = np.random.RandomState(seed * 7919 + 17)
    n = len(indptr) - 1
    capacity = max(256, n // 2)
    targets = [len(training) // workers + (1 if i < len(training) % workers else 0)
               for i in range(workers)]
    partitions: List[Set[int]] = [set() for _ in range(workers)]
    replicas: List[Set[int]] = [set() for _ in range(workers)]
    fallbacks = 0
    ordered = sorted((int(v) for v in training), key=lambda v: -degrees[v])
    for vertex in ordered:
        b = ball(indptr, indices, vertex, depth)
        feasible = [i for i in range(workers)
                    if len(replicas[i]) + len(b - replicas[i]) <= capacity]
        below = [i for i in feasible if len(partitions[i]) < targets[i]]
        if below:
            feasible = below
        if not feasible:
            fallbacks += 1
            worker = len(partitions[0]) % workers
            partitions[worker].add(vertex)
            replicas[worker].update(sorted(b)[:capacity])
            continue
        best, worker = -1.0, feasible[0]
        for i in feasible:
            locality = len(replicas[i] & b) / len(b)
            balance = max(0.0, 1.0 - len(partitions[i]) / max(targets[i], 1))
            memory = max(0.0, 1.0 - len(replicas[i]) / capacity)
            score = (locality + balance + memory) / 3.0
            if score > best:
                best, worker = score, i
        partitions[worker].add(vertex)
        replicas[worker].update(b)
    imbalance = max(abs(len(partitions[i]) / max(targets[i], 1) - 1.0)
                    for i in range(workers))
    metrics = {
        "T_partition_s": time.perf_counter() - t0,
        "fallbacks": fallbacks,
        "load_imbalance": imbalance,
        "replication_factor": sum(len(r) for r in replicas) / n,
    }
    return partitions, replicas, metrics


def build_caches(sampler, partitions, replicas, degrees, n, workers=8,
                 rounds=16, seed=0, presample_seeds=512):
    """Per-worker caches via K pre-sampling rounds on a bounded pilot subset.

    The paper's bounded pilot sweep: hotness is estimated on a deterministic
    per-worker subset (default 512 seeds) so preprocessing stays practical at
    the native 90k-seed scale.
    """
    t0 = time.perf_counter()
    budget = max(64, n // (workers * 4))
    caches: List[Set[int]] = []
    for w in range(workers):
        counts: Dict[int, int] = {int(v): 0 for v in replicas[w]}
        rng = np.random.RandomState(seed * 131 + w)
        own = sorted(partitions[w])
        if len(own) > presample_seeds:
            own = list(rng.choice(own, size=presample_seeds, replace=False))
        for _ in range(rounds):
            for s in own:
                batch = sampler.sample(np.asarray([s], dtype=np.int64))
                for node in np.concatenate(batch.layer_nodes):
                    node = int(node)
                    if node in counts:
                        counts[node] += 1
        caches.append(set(sorted(counts, key=lambda v: (-counts[v], v))[:budget]))
    union_hits = set().union(*caches)
    return caches, {
        "T_presample_cache_s": time.perf_counter() - t0,
        "per_worker_budget": budget,
        "cache_union_size": len(union_hits),
        "cache_records_total": budget * workers,
    }
