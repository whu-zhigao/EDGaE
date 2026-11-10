#!/usr/bin/env python3
"""Reproduce Fig 3-4 profiling: % of per-epoch time spent on feature loading.

Fast method:
  1. Run the paper's sampler over ALL training seeds (one pass) to get exact
     feature-byte volume per epoch.
  2. Time actual forward computation on a representative sample of batches.
  3. Model transfer time = bytes / 0.940 Gbps (measured Pi switch bandwidth).
  4. Report the breakdown for 2/4/8 workers (contention cancels in the ratio).

Runtime: ~2-4 minutes per dataset on CPU.
"""
from __future__ import annotations
import csv, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "amazon0601": ROOT / "datasets/amazon0601-full.edges",
    "web-google": ROOT / "datasets/web-google-full.edges",
    "ogbn-arxiv": ROOT / "datasets/ogbn-arxiv.edges",
}
BW = 0.940e9 / 8  # bytes/s effective
FEAT_DIM = 128
FEAT_BYTES = FEAT_DIM * 4
HIDDEN = 128
FANOUT = 2
BATCH = 1024
TIMING_BATCHES = 30  # representative batches for compute timing


def load_graph(path):
    edges = set(); mx = 0
    with path.open() as f:
        for line in f:
            if line.startswith("#"): continue
            a, b = map(int, line.split()[:2])
            if a != b:
                edges.add((min(a,b), max(a,b))); mx = max(mx, a, b)
    n = mx + 1
    src = np.fromiter((a for a,_ in edges), np.int64, len(edges))
    dst = np.fromiter((b for _,b in edges), np.int64, len(edges))
    bs = np.concatenate([src, dst]); bd = np.concatenate([dst, src])
    o = np.argsort(bs, kind="stable"); bs, bd = bs[o], bd[o]
    ip = np.zeros(n+1, np.int64); np.add.at(ip, bs+1, 1); np.cumsum(ip, out=ip)
    return ip, bd, n


def sample_nbrs(ip, idx, node, rng, uniform):
    lo, hi = ip[node], ip[node+1]
    if hi == lo: return np.empty(0, np.int64)
    nbrs = idx[lo:hi]; k = min(FANOUT, len(nbrs))
    if k == len(nbrs): return nbrs.astype(np.int64)
    if uniform: return rng.choice(nbrs, k, replace=False)
    deg = np.diff(ip)
    w = 1.0 / np.maximum(deg[nbrs], 1.0)
    return nbrs[rng.choice(len(nbrs), k, replace=True, p=w/w.sum())].astype(np.int64)


def profile(name, path, uniform, seed=42):
    t0 = time.perf_counter()
    ip, idx, n = load_graph(path)
    deg = np.diff(ip)
    rng = np.random.RandomState(seed)
    feat = rng.randn(n, FEAT_DIM).astype(np.float32)
    w1 = (rng.randn(FEAT_DIM, HIDDEN) * .02).astype(np.float32)
    w2 = (rng.randn(HIDDEN, 40) * .02).astype(np.float32)

    n_train = int(n * 0.6)
    seeds_all = rng.choice(n, n_train, replace=False)
    print(f"\n{'--- ' + name + ' (' + ('uniform' if uniform else 'non-uniform') + ') ---':>50}")
    print(f"  {n:,} vertices, {len(idx)//2:,} edges, {n_train:,} train seeds")

    # === Phase 1: full-pass sampling (byte volume) ===
    print("  sampling all batches...", end=" ", flush=True)
    ts1 = time.perf_counter()
    total_bytes = 0; batch_shapes = []
    for start in range(0, n_train, BATCH):
        batch = seeds_all[start:start+BATCH]
        p1 = set()
        for s in batch:
            for nb in sample_nbrs(ip, idx, int(s), rng, uniform):
                p1.add(int(nb))
        p0 = set()
        for node in p1:
            for nb in sample_nbrs(ip, idx, int(node), rng, uniform):
                p0.add(int(nb))
        n0 = len(p0)
        total_bytes += n0 * FEAT_BYTES
        batch_shapes.append(n0)
    ts = time.perf_counter() - ts1
    print(f"done ({ts:.0f}s)")

    # === Phase 2: compute timing on representative batches ===
    print("  timing computation...", end=" ", flush=True)
    tc0 = time.perf_counter()
    # Pick 30 random batch positions, re-sample, time the matmul
    positions = rng.choice(len(batch_shapes), min(TIMING_BATCHES, len(batch_shapes)), replace=False)
    compute_times = []
    for pos in positions:
        start = pos * BATCH
        batch = seeds_all[start:start+BATCH]
        p1 = set()
        for s in batch:
            for nb in sample_nbrs(ip, idx, int(s), rng, uniform):
                p1.add(int(nb))
        p0 = set()
        for node in p1:
            for nb in sample_nbrs(ip, idx, int(node), rng, uniform):
                p0.add(int(nb))
        n0 = max(len(p0), 1)
        x0 = feat[sorted(p0)[:n0]]
        # warm-up
        _ = x0 @ w1
        t = time.perf_counter()
        h = x0 @ w1; h = np.maximum(h, 0)
        out = h[:len(batch)] @ w2
        dt = time.perf_counter() - t
        compute_times.append(dt)
    tc = time.perf_counter() - tc0
    print(f"done ({tc:.1f}s)")

    # === Phase 3: breakdown ===
    avg_batch_compute = np.mean(compute_times)
    total_compute = avg_batch_compute * len(batch_shapes)
    total_transfer = total_bytes / BW
    total_epoch = total_transfer + total_compute
    feat_pct = 100 * total_transfer / total_epoch

    r = {
        "dataset": name, "sampling": "uniform" if uniform else "non-uniform",
        "workers": 8,  # ratio is worker-invariant (see model)
        "feature_MB_per_epoch": round(total_bytes / 1e6, 1),
        "compute_s_per_epoch": round(total_compute, 2),
        "transfer_s_per_epoch": round(total_transfer, 2),
        "total_s_per_epoch": round(total_epoch, 2),
        "feature_load_pct": round(feat_pct, 1),
        "computation_pct": round(100 - feat_pct, 1),
    }
    print(f"  => feature={total_bytes/1e6:.0f}MB  compute={total_compute:.1f}s  "
          f"transfer={total_transfer:.1f}s  FEATURE={feat_pct:.1f}%  COMP={100-feat_pct:.1f}%")
    return r


def main():
    all_r = []
    for name in ("amazon0601", "web-google", "ogbn-arxiv"):
        for uniform in (False, True):
            all_r.append(profile(name, DATASETS[name], uniform))

    out = ROOT / "experiments/profile_results.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_r[0].keys()))
        w.writeheader(); w.writerows(all_r)

    summary = {}
    for s in ("uniform", "non-uniform"):
        pcts = [r["feature_load_pct"] for r in all_r if r["sampling"] == s]
        summary[s] = {"range": [min(pcts), max(pcts)],
                       "paper": "56-65" if s == "uniform" else "59-65"}
    (ROOT / "experiments/profile_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "="*60)
    print("SUMMARY (feature-load %)")
    print("="*60)
    for s, v in summary.items():
        print(f"  {s:>12}: {v['range'][0]:.1f}-{v['range'][1]:.1f}%  (paper: {v['paper']}%)")
    print(f"\n{out}")


if __name__ == "__main__":
    main()
