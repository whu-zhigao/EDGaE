#!/usr/bin/env python3
"""Joint-placement decision-quality study (review Major Concern 2).

Compares the paper's two-stage greedy allocator against placement baselines
and, on small instances, the exact optimum:

  Strategies: two-stage-greedy (paper), locality-only, memory-first,
  cost-aware (marginal amortized cost), random-feasible, local-search
  (best-improvement moves, near-optimal reference), exact-opt (exhaustive
  enumeration, 2 workers and n<=14 seeds).

  Ordering variants: neighborhood-desc (paper), access-desc,
  ship-cost-asc, random.

  Cost model (bytes-only, H=100, matches Eq. (5)):
  total = Ship + H * ResidualPerWindow, where Ship counts 8B vertex+edge
  records and Residual counts evaluation-trace feature misses outside
  per-worker top-hotness caches at the paper budget V/(4N).

Also runs the cache-shift study (Major Concern 5): caches built on reference
traces of seed set A are evaluated on traces of a shifted seed set B with
10/20/50% replacement, reporting hit-rate degradation.

Outputs: results/joint-quality-results.csv, results/cache-shift-results.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_driven_runner import WebGraph, access_traces, frequency_counts
from trace_driven_runner_v2 import ALL_DATASET_FILES, load_graph_v2


H_HORIZON = 100
RECORD_BYTES = 8
FEATURE_BYTES = 100 * 4
BATCHES = 16
BATCH_SIZE = 128


def build_instance(graph, training, seed, n_seeds, depth=2, fanout=2):
    """Per-seed balls, edge counts, and per-seed reference/eval accesses."""
    rng = random.Random(seed * 104729 + 11)
    seeds = list(rng.sample(list(training), n_seeds))
    balls = {v: set(graph.ball(v, depth)) for v in seeds}

    def ball_edges(ball):
        return sum(1 for u in ball
                   for nb in graph.adjacency[u] if nb in ball and nb > u)

    edges = {v: ball_edges(balls[v]) for v in seeds}
    per_seed_ref, per_seed_evl = {}, {}
    for si, v in enumerate(seeds):
        per_seed_ref[v] = set(access_traces(
            graph, [v], "non-uniform", depth=depth, fanout=fanout,
            batches=4, batch_size=32,
            rng=random.Random(seed * 31 + si))[0])
        per_seed_evl[v] = set(access_traces(
            graph, [v], "non-uniform", depth=depth, fanout=fanout,
            batches=4, batch_size=32,
            rng=random.Random(seed * 37 + si))[0])
    return dict(seeds=seeds, balls=balls, edges=edges,
                per_seed_ref=per_seed_ref, per_seed_evl=per_seed_evl)


def cache_budget(graph, workers):
    return max(64, graph.vertices // (workers * 4))


def assignment_cost(inst, assign, workers, budget):
    """Ship bytes + H * residual bytes for a full assignment."""
    replicas = [set() for _ in range(workers)]
    rep_edges = [0] * workers
    for v, w in assign.items():
        new = inst["balls"][v] - replicas[w]
        replicas[w] |= inst["balls"][v]
        rep_edges[w] += int(inst["edges"][v] * len(new) /
                            max(len(inst["balls"][v]), 1))
    ship = sum((len(replicas[w]) + rep_edges[w]) * RECORD_BYTES
               for w in range(workers))
    residual = 0
    for w in range(workers):
        own = [v for v in assign if assign[v] == w]
        if not own:
            continue
        counts = defaultdict(int)
        for v in own:
            for u in inst["per_seed_ref"][v]:
                counts[u] += 1
        ordered = sorted(counts, key=lambda u: (-counts[u], u))
        cache = set(ordered[:budget])
        miss = set()
        for v in own:
            miss |= inst["per_seed_evl"][v] - cache
        residual += len(miss) * FEATURE_BYTES
    return ship + H_HORIZON * residual, ship, residual


def memory_capacity(graph):
    return max(256, graph.vertices // 2)


def place(inst, graph, workers, strategy, order, seed):
    seeds = inst["seeds"]
    if order == "neighborhood-desc":
        ordered = sorted(seeds, key=lambda v: -len(inst["balls"][v]))
    elif order == "access-desc":
        acc = {v: len(inst["per_seed_ref"][v]) for v in seeds}
        ordered = sorted(seeds, key=lambda v: (-acc[v], v))
    elif order == "ship-cost-asc":
        cost = {v: (len(inst["balls"][v]) + inst["edges"][v]) for v in seeds}
        ordered = sorted(seeds, key=lambda v: (cost[v], v))
    else:  # random
        rng_o = random.Random(seed * 13 + 1)
        ordered = list(seeds)
        rng_o.shuffle(ordered)

    cap = memory_capacity(graph)
    targets = [len(ordered) // workers + (1 if i < len(ordered) % workers else 0)
               for i in range(workers)]
    assign, replicas = {}, [set() for _ in range(workers)]
    loads = [0] * workers
    rng = random.Random(seed * 7919 + 17)
    for v in ordered:
        ball = inst["balls"][v]
        feasible = [w for w in range(workers)
                    if len(replicas[w]) + len(ball - replicas[w]) <= cap]
        below = [w for w in feasible if loads[w] < targets[w]] or feasible \
            or list(range(workers))
        if strategy == "locality-only":
            w = max(below, key=lambda w: len(replicas[w] & ball))
        elif strategy == "memory-first":
            w = max(below, key=lambda w: 1 - len(replicas[w]) / cap)
        elif strategy == "random-feasible":
            w = below[rng.randrange(len(below))]
        elif strategy == "cost-aware":
            def marginal(w):
                new = len(ball - replicas[w])
                ship_add = new * RECORD_BYTES
                hit = len(inst["per_seed_evl"][v] & replicas[w]) / \
                    max(len(inst["per_seed_evl"][v]), 1)
                return ship_add + H_HORIZON * (1 - hit) * \
                    len(inst["per_seed_evl"][v]) * FEATURE_BYTES
            w = min(below, key=marginal)
        else:  # two-stage-greedy
            best, w = -1.0, below[0]
            for i in below:
                loc = len(replicas[i] & ball) / len(ball)
                bal = max(0.0, 1 - loads[i] / max(targets[i], 1))
                mem = max(0.0, 1 - len(replicas[i]) / cap)
                s = (loc + bal + mem) / 3
                if s > best:
                    best, w = s, i
        assign[v] = w
        replicas[w] |= ball
        loads[w] += 1
    return assign


def local_search(inst, workers, assign, budget, max_passes=8):
    best = dict(assign)
    best_cost, _, _ = assignment_cost(inst, best, workers, budget)
    improved, passes = True, 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for v in inst["seeds"]:
            cur = best[v]
            for w in range(workers):
                if w == cur:
                    continue
                cand = dict(best)
                cand[v] = w
                c, _, _ = assignment_cost(inst, cand, workers, budget)
                if c < best_cost - 1e-9:
                    best, best_cost = cand, c
                    improved = True
    return best, best_cost


def exact_opt(inst, workers, budget):
    """Exhaustive optimum; 2 workers and n<=14 only."""
    seeds = inst["seeds"]
    n = len(seeds)
    assert workers == 2 and n <= 14, "exact enumeration size exceeded"
    best_cost, best_assign = None, None
    for bits in range(1 << n):
        assign = {seeds[i]: (bits >> i) & 1 for i in range(n)}
        if not (0 in assign.values() and 1 in assign.values()):
            continue
        c, _, _ = assignment_cost(inst, assign, workers, budget)
        if best_cost is None or c < best_cost:
            best_cost, best_assign = c, assign
    return best_assign, best_cost


def cache_shift_study(graph, seed, shift_fractions=(0.1, 0.2, 0.5)):
    rng = random.Random(seed * 104729 + 11)
    training = rng.sample(range(graph.vertices), 1200)
    ref_full = access_traces(graph, training, "non-uniform", depth=2,
                             fanout=2, batches=16, batch_size=128,
                             rng=random.Random(seed * 7717 + 3))
    base_hit, rows = None, []
    for frac in (0.0,) + shift_fractions:
        shifted = list(training)
        k = int(len(training) * frac)
        if k:
            new = rng.sample(range(graph.vertices), k)
            shifted[:k] = new
        evl = access_traces(graph, shifted, "non-uniform", depth=2,
                            fanout=2, batches=16, batch_size=128,
                            rng=random.Random(seed * 7723 + 5))
        counts = frequency_counts(ref_full, set(range(graph.vertices)))
        budget = cache_budget(graph, 4) * 4
        cache = set(sorted(counts, key=lambda u: (-counts[u], u))[:budget])
        total = sum(len(b) for b in evl)
        hit = sum(len(b & cache) for b in evl) / max(total, 1)
        if frac == 0.0:
            base_hit = hit
        rows.append(dict(seed=seed, shift_fraction=frac, hit_rate=hit,
                         hit_drop=base_hit - hit if base_hit else 0.0))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="experiment-cache")
    ap.add_argument("--out", default="results/joint-quality-results.csv")
    ap.add_argument("--shift-out", default="results/cache-shift-results.csv")
    ap.add_argument("--datasets", nargs="+",
                    default=["web-google", "amazon0601", "wiki-talk"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--exact-seeds", type=int, default=5)
    args = ap.parse_args()

    out, shift_out = Path(args.out), Path(args.shift_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    strategies = ["two-stage-greedy", "locality-only", "memory-first",
                  "cost-aware", "random-feasible", "local-search"]
    orders = ["neighborhood-desc", "access-desc", "ship-cost-asc", "random"]
    rows, shift_rows = [], []

    for ds in args.datasets:
        graph = load_graph_v2(ds, Path(args.cache_dir))
        for seed in args.seeds:
            rng = random.Random(seed)
            training = rng.sample(range(graph.vertices), 2000)
            inst = build_instance(graph, training, seed, n_seeds=200)
            budget = cache_budget(graph, 4)
            costs = {}
            for strat in strategies:
                if strat == "local-search":
                    continue
                assign = place(inst, graph, 4, strat, "neighborhood-desc", seed)
                c, ship, resid = assignment_cost(inst, assign, 4, budget)
                costs[strat] = c
                rows.append(dict(dataset=ds, seed=seed, instance="medium-200",
                                 strategy=strat, order="neighborhood-desc",
                                 total_amortized_MB=round(c / 1e6, 4),
                                 ship_MB=round(ship / 1e6, 4),
                                 residual_MB=round(resid / 1e6, 4)))
            start_strat = min(costs, key=costs.get)
            assign0 = place(inst, graph, 4, start_strat, "neighborhood-desc", seed)
            _, c_ls = local_search(inst, 4, assign0, budget)
            costs["local-search"] = c_ls
            rows.append(dict(dataset=ds, seed=seed, instance="medium-200",
                             strategy="local-search",
                             order=f"start:{start_strat}",
                             total_amortized_MB=round(c_ls / 1e6, 4),
                             ship_MB="", residual_MB=""))

            for order in orders:
                if order == "neighborhood-desc":
                    continue
                assign = place(inst, graph, 4, "two-stage-greedy", order, seed)
                c, ship, resid = assignment_cost(inst, assign, 4, budget)
                rows.append(dict(dataset=ds, seed=seed, instance="medium-200",
                                 strategy="two-stage-greedy", order=order,
                                 total_amortized_MB=round(c / 1e6, 4),
                                 ship_MB=round(ship / 1e6, 4),
                                 residual_MB=round(resid / 1e6, 4)))

            inst_s = build_instance(graph, training, seed,
                                    n_seeds=args.exact_seeds)
            budget_s = cache_budget(graph, 2)
            greedy = place(inst_s, graph, 2, "two-stage-greedy",
                           "neighborhood-desc", seed)
            c_g, _, _ = assignment_cost(inst_s, greedy, 2, budget_s)
            _, c_opt = exact_opt(inst_s, 2, budget_s)
            gap = (c_g - c_opt) / c_opt if c_opt else 0.0
            rows.append(dict(dataset=ds, seed=seed,
                             instance=f"exact-{args.exact_seeds}",
                             strategy="two-stage-greedy",
                             order="neighborhood-desc",
                             total_amortized_MB=round(c_g / 1e6, 4),
                             ship_MB="", residual_MB=""))
            rows.append(dict(dataset=ds, seed=seed,
                             instance=f"exact-{args.exact_seeds}",
                             strategy="exact-opt", order="enumeration",
                             total_amortized_MB=round(c_opt / 1e6, 4),
                             ship_MB="", residual_MB=""))
            rows.append(dict(dataset=ds, seed=seed,
                             instance=f"exact-{args.exact_seeds}",
                             strategy="gap-to-opt", order="",
                             total_amortized_MB=round(gap, 6),
                             ship_MB="", residual_MB=""))
            print(f"[{ds} seed {seed}] greedy={c_g/1e6:.4f}MB "
                  f"opt={c_opt/1e6:.4f}MB gap={gap*100:.1f}%", flush=True)
            shift_rows.extend(cache_shift_study(graph, seed))

    fields = ["dataset", "seed", "instance", "strategy", "order",
              "total_amortized_MB", "ship_MB", "residual_MB"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with shift_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "shift_fraction",
                                          "hit_rate", "hit_drop"])
        w.writeheader()
        w.writerows(shift_rows)
    print(f"wrote {out} ({len(rows)} rows), {shift_out} ({len(shift_rows)})")


if __name__ == "__main__":
    main()
