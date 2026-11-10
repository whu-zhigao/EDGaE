#!/usr/bin/env python3
"""Full-scale cost-aware greedy vs score-greedy comparison.

Addresses review §3.1: implement cost-aware greedy that selects the worker
minimizing marginal amortized cost (not heuristic score), then run bounded
local search. Compare four strategies:
  1. score-greedy (current EDGaE, Eq. 7)
  2. cost-aware greedy (marginal cost minimization)
  3. cost-aware greedy + local search
  4. joint optimum (exhaustive on small instances)

Also measures the shared-replication savings from co-assigning overlapping
neighborhoods, directly addressing the interaction-gap analysis.
"""
from __future__ import annotations
import argparse, csv, random, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_driven_runner import WebGraph, access_traces, frequency_counts
from trace_driven_runner_v2 import load_graph_v2

H = 100; RECORD_B = 8; FEAT_B = 400; BATCHES = 16; BATCH_SIZE = 128

def build(graph, training, seed, n, depth=2, fanout=2):
    rng = random.Random(seed * 104729 + 11)
    seeds = rng.sample(list(training), n)
    balls = {v: set(graph.ball(v, depth)) for v in seeds}
    def bedges(ball):
        return sum(1 for u in ball for nb in graph.adjacency[u] if nb in ball and nb > u)
    edges = {v: bedges(balls[v]) for v in seeds}
    psr, pse = {}, {}
    for i, v in enumerate(seeds):
        psr[v] = set(access_traces(graph, [v], "non-uniform", depth=depth, fanout=fanout,
                     batches=4, batch_size=32, rng=random.Random(seed*31+i))[0])
        pse[v] = set(access_traces(graph, [v], "non-uniform", depth=depth, fanout=fanout,
                     batches=4, batch_size=32, rng=random.Random(seed*37+i))[0])
    return dict(seeds=seeds, balls=balls, edges=edges, psr=psr, pse=pse)

def cost(inst, assign, W, budget):
    reps = [set() for _ in range(W)]; redges = [0]*W
    for v, w in assign.items():
        new = inst["balls"][v] - reps[w]
        reps[w] |= inst["balls"][v]
        redges[w] += int(inst["edges"][v] * len(new) / max(len(inst["balls"][v]), 1))
    ship = sum((len(reps[w]) + redges[w]) * RECORD_B for w in range(W))
    resid = 0
    for w in range(W):
        own = [v for v in assign if assign[v] == w]
        if not own: continue
        counts = defaultdict(int)
        for v in own:
            for u in inst["psr"][v]: counts[u] += 1
        cache = set(sorted(counts, key=lambda u: (-counts[u], u))[:budget])
        miss = set()
        for v in own: miss |= inst["pse"][v] - cache
        resid += len(miss) * FEAT_B
    return ship + H * resid, ship, resid

def place_score(inst, graph, W, seed):
    seeds = inst["seeds"]
    ordered = sorted(seeds, key=lambda v: -len(inst["balls"][v]))
    cap = max(256, graph.vertices // 2)
    targets = [len(ordered)//W + (1 if i < len(ordered)%W else 0) for i in range(W)]
    assign, reps, loads = {}, [set() for _ in range(W)], [0]*W
    for v in ordered:
        ball = inst["balls"][v]
        feas = [i for i in range(W) if len(reps[i]) + len(ball - reps[i]) <= cap]
        pool = [i for i in feas if loads[i] < targets[i]] or feas or list(range(W))
        best, w = -1, pool[0]
        for i in pool:
            loc = len(reps[i] & ball) / len(ball)
            bal = max(0, 1 - loads[i] / max(targets[i], 1))
            mem = max(0, 1 - len(reps[i]) / cap)
            s = (loc + bal + mem) / 3
            if s > best: best, w = s, i
        assign[v] = w; reps[w] |= ball; loads[w] += 1
    return assign

def place_cost_aware(inst, graph, W, seed):
    seeds = inst["seeds"]
    ordered = sorted(seeds, key=lambda v: -len(inst["balls"][v]))
    cap = max(256, graph.vertices // 2)
    targets = [len(ordered)//W + (1 if i < len(ordered)%W else 0) for i in range(W)]
    assign, reps, loads = {}, [set() for _ in range(W)], [0]*W
    for v in ordered:
        ball = inst["balls"][v]
        feas = [i for i in range(W) if len(reps[i]) + len(ball - reps[i]) <= cap]
        pool = [i for i in feas if loads[i] < targets[i]] or feas or list(range(W))
        def marg(w):
            new = len(ball - reps[w])
            ship_add = new * RECORD_B
            hit = len(inst["pse"][v] & reps[w]) / max(len(inst["pse"][v]), 1)
            resid_add = H * (1 - hit) * len(inst["pse"][v]) * FEAT_B
            return ship_add + resid_add
        w = min(pool, key=marg)
        assign[v] = w; reps[w] |= ball; loads[w] += 1
    return assign

def local_search(inst, W, assign, budget, max_passes=5):
    best = dict(assign); bc, _, _ = cost(inst, best, W, budget)
    improved, passes = True, 0
    while improved and passes < max_passes:
        improved = False; passes += 1
        for v in inst["seeds"]:
            for w in range(W):
                if w == best[v]: continue
                cand = dict(best); cand[v] = w
                c, _, _ = cost(inst, cand, W, budget)
                if c < bc - 1e-9: best, bc = cand, c; improved = True
    return best, bc

def exact_opt(inst, W, budget):
    seeds = inst["seeds"]; n = len(seeds)
    assert W == 2 and n <= 14
    bc, ba = None, None
    for bits in range(1 << n):
        a = {seeds[i]: (bits >> i) & 1 for i in range(n)}
        if 0 not in a.values() or 1 not in a.values(): continue
        c, _, _ = cost(inst, a, W, budget)
        if bc is None or c < bc: bc, ba = c, a
    return ba, bc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="experiment-cache")
    ap.add_argument("--out", default="results/cost-aware-results.csv")
    ap.add_argument("--datasets", nargs="+", default=["web-google", "amazon0601", "wiki-talk"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-medium", type=int, default=200)
    ap.add_argument("--n-exact", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in args.datasets:
        graph = load_graph_v2(ds, Path(args.cache_dir))
        budget = max(64, graph.vertices // (4 * 4))
        for seed in args.seeds:
            rng = random.Random(seed)
            training = rng.sample(range(graph.vertices), 2000)
            # Medium scale (200 seeds, 4 workers)
            inst = build(graph, training, seed, args.n_medium)
            a1 = place_score(inst, graph, 4, seed)
            c1, s1, r1 = cost(inst, a1, 4, budget)
            a2 = place_cost_aware(inst, graph, 4, seed)
            c2, s2, r2 = cost(inst, a2, 4, budget)
            _, c3 = local_search(inst, 4, a2, budget)
            rows.append(dict(dataset=ds, seed=seed, instance=f"medium-{args.n_medium}",
                            strategy="score-greedy", total=round(c1/1e6,3),
                            ship=round(s1/1e6,3), resid=round(r1/1e6,3)))
            rows.append(dict(dataset=ds, seed=seed, instance=f"medium-{args.n_medium}",
                            strategy="cost-aware", total=round(c2/1e6,3),
                            ship=round(s2/1e6,3), resid=round(r2/1e6,3)))
            rows.append(dict(dataset=ds, seed=seed, instance=f"medium-{args.n_medium}",
                            strategy="cost-aware+LS", total=round(c3/1e6,3), ship="", resid=""))
            # Exact (10 seeds, 2 workers)
            inst_s = build(graph, training, seed, args.n_exact)
            bs = budget // 2
            g = place_score(inst_s, graph, 2, seed)
            cg, _, _ = cost(inst_s, g, 2, bs)
            ca = place_cost_aware(inst_s, graph, 2, seed)
            cca, _, _ = cost(inst_s, ca, 2, bs)
            _, co = exact_opt(inst_s, 2, bs)
            rows.append(dict(dataset=ds, seed=seed, instance=f"exact-{args.n_exact}",
                            strategy="score-greedy", total=round(cg/1e6,4), ship="", resid=""))
            rows.append(dict(dataset=ds, seed=seed, instance=f"exact-{args.n_exact}",
                            strategy="cost-aware", total=round(cca/1e6,4), ship="", resid=""))
            rows.append(dict(dataset=ds, seed=seed, instance=f"exact-{args.n_exact}",
                            strategy="exact-opt", total=round(co/1e6,4), ship="", resid=""))
            gap_g = (cg - co) / co if co else 0
            gap_a = (cca - co) / co if co else 0
            rows.append(dict(dataset=ds, seed=seed, instance=f"exact-{args.n_exact}",
                            strategy="gap-score", total=round(gap_g,6), ship="", resid=""))
            rows.append(dict(dataset=ds, seed=seed, instance=f"exact-{args.n_exact}",
                            strategy="gap-cost-aware", total=round(gap_a,6), ship="", resid=""))
            print(f"[{ds} s{seed}] score={c1/1e6:.2f} cost-aware={c2/1e6:.2f} "
                  f"LS={c3/1e6:.2f} | exact: score={cg/1e6:.4f} "
                  f"cost-aware={cca/1e6:.4f} opt={co/1e6:.4f} "
                  f"gap_s={gap_g*100:.1f}% gap_ca={gap_a*100:.1f}%", flush=True)
    fields = ["dataset", "seed", "instance", "strategy", "total", "ship", "resid"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")

if __name__ == "__main__":
    main()
