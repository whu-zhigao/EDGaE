"""Greedy replicated placement + hotness caches (ported from edgae_gpu).

Semantics preserved: equal-weight scoring, per-worker capacity max(256, V/2),
hard feasibility with fallback counting, per-worker top-hotness caches from
K pre-sampling rounds on a bounded pilot subset.
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
    t0 = time.perf_counter()
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
        pool = below or feasible
        if not pool:
            fallbacks += 1
            worker = len(partitions[0]) % workers
            partitions[worker].add(vertex)
            replicas[worker].update(sorted(b)[:capacity])
            continue
        best, worker = -1.0, pool[0]
        for i in pool:
            loc = len(replicas[i] & b) / len(b)
            bal = max(0.0, 1 - len(partitions[i]) / max(targets[i], 1))
            mem = max(0.0, 1 - len(replicas[i]) / capacity)
            s = (loc + bal + mem) / 3.0
            if s > best:
                best, worker = s, i
        partitions[worker].add(vertex)
        replicas[worker].update(b)
    imb = max(abs(len(partitions[i]) / max(targets[i], 1) - 1.0)
              for i in range(workers))
    return partitions, replicas, {
        "T_partition_s": time.perf_counter() - t0,
        "fallbacks": fallbacks,
        "load_imbalance": imb,
        "replication_factor": sum(len(r) for r in replicas) / n,
    }


def build_caches(sampler, partitions, replicas, n, workers=8,
                 rounds=16, seed=0, presample_seeds=512):
    """Per-worker top-hotness caches from a bounded pilot."""
    from sampling import epoch_rng
    t0 = time.perf_counter()
    budget = max(64, n // (workers * 4))
    caches: List[Set[int]] = []
    for w in range(workers):
        counts: Dict[int, int] = {int(v): 0 for v in replicas[w]}
        rng = np.random.RandomState(seed * 131 + w)
        own = sorted(partitions[w])
        if len(own) > presample_seeds:
            own = list(rng.choice(own, size=presample_seeds, replace=False))
        for s in own:
            for _ in range(rounds):
                r = epoch_rng(int(s), 0, 7)
                b = sampler.sample(np.asarray([s], dtype=np.int64), r)
                for node in np.concatenate(b.layer_nodes):
                    node = int(node)
                    if node in counts:
                        counts[node] += 1
        caches.append(set(sorted(counts, key=lambda v: (-counts[v], v))[:budget]))
    union_hits = set().union(*caches)
    return caches, {
        "T_cache_s": time.perf_counter() - t0,
        "per_worker_budget": budget,
        "cache_union_size": len(union_hits),
        "cache_records_total": budget * workers,
    }
