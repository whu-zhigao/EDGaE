#!/usr/bin/env python3
"""Edge-worker process of the emulated fleet.

Resource model (kernel-enforced): CPU affinity to ``cores`` and RLIMIT_AS
memory cap. Data budgets (replica, cache) are enforced by placement.

Modes:
  hybrid          feasible seeds sample locally from the worker sub-CSR and
                  fetch only cache misses from the server (H); fallback seeds
                  request server-assembled batches (S).
  server-centric  every seed runs in S mode (DGL-E analogue).
"""
from __future__ import annotations
import argparse
import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import recv_msg, send_msg, PacedSender
from sampling import NeighborSampler, epoch_rng, merge_parts


class SAGE(torch.nn.Module):
    def __init__(self, in_dim, hidden, classes):
        super().__init__()
        self.w1 = torch.nn.Linear(in_dim, hidden, bias=False)
        self.w2 = torch.nn.Linear(hidden, classes, bias=False)

    def forward(self, x0, e1, e2, n_hidden, n_seeds):
        r1, c1 = e1
        h = torch.zeros(n_hidden, x0.shape[1])
        h.index_add_(0, r1, x0[c1])
        d1 = torch.zeros(n_hidden)
        d1.index_add_(0, r1, torch.ones_like(r1, dtype=torch.float))
        h = F.relu(h / d1.clamp(min=1).unsqueeze(1))
        h = self.w1(h)
        r2, c2 = e2
        out = torch.zeros(n_seeds, h.shape[1])
        out.index_add_(0, r2, h[c2])
        d2 = torch.zeros(n_seeds)
        d2.index_add_(0, r2, torch.ones_like(r2, dtype=torch.float))
        out = out / d2.clamp(min=1).unsqueeze(1)
        return self.w2(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()
    cfg = json.loads(Path(args.manifest).read_text())

    # Kernel-enforced edge-class constraints.
    try:
        os.sched_setaffinity(0, set(cfg["cores"]))
    except (AttributeError, OSError):
        pass  # non-Linux host (local smoke); enforcement logged in env
    limit = int(cfg["as_limit_gb"] * (1 << 30))
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass  # unsupported cap on this host
    torch.set_num_threads(len(cfg["cores"]))

    wid = cfg["worker_id"]
    mode = cfg["mode"]  # "hybrid" | "server-centric"
    run_dir = Path(cfg["run_dir"])
    feats_dim = cfg["feature_dim"]

    # Per-worker placement artifacts (shipped once, before training).
    glob_ids = np.load(run_dir / f"worker{wid}_ids.npy")          # local->global
    indptr = np.load(run_dir / f"worker{wid}_indptr.npy")
    indices = np.load(run_dir / f"worker{wid}_indices.npy")       # local ids
    cache_ids = np.load(run_dir / f"worker{wid}_cache_ids.npy")   # global
    cache_feat = np.load(run_dir / f"worker{wid}_cache_feat.npy")
    cache_pos = {int(g): i for i, g in enumerate(cache_ids)}
    seeds_local = np.load(run_dir / f"worker{wid}_seeds.npy")     # global
    labels_all = np.load(run_dir / "labels.npy")
    lab = torch.from_numpy(labels_all[seeds_local]).long()

    glob_of = glob_ids  # local index -> global id
    local_of = {int(g): i for i, g in enumerate(glob_ids)}
    sampler = NeighborSampler(indptr, indices, fanout=cfg["fanout"])

    feasible_mask = np.load(run_dir / f"worker{wid}_feasible.npy") if mode == "hybrid" \
        else np.zeros(len(seeds_local), dtype=bool)
    feasible_local = np.array([local_of[int(s)] for s, f in
                               zip(seeds_local, feasible_mask) if f], dtype=np.int64)

    sock = socket.create_connection((cfg["server_host"], cfg["server_port"]))
    sender = PacedSender(sock)  # worker sends are small; pacing off
    send_msg(sock, {"op": "hello", "worker": wid,
                    "n_seeds": len(seeds_local)}, sender=sender)

    model = SAGE(feats_dim, cfg["hidden"], cfg["classes"])
    bs = cfg["batch_size"]
    rng_master = np.random.RandomState(cfg["run_seed"] * 31 + wid)

    stats = {"data_s": 0.0, "compute_s": 0.0, "batches": 0,
             "h_seeds": 0, "s_seeds": 0}
    epoch = 0
    while True:
        hdr, body = recv_msg(sock)
        if hdr["op"] == "stop":
            break
        assert hdr["op"] == "params", hdr
        pm = torch.from_numpy(np.frombuffer(body, dtype=np.float32).copy())
        off = 0
        with torch.no_grad():
            for p in model.parameters():
                n = p.numel()
                p.copy_(pm[off:off + n].view_as(p))
                off += n

        order = rng_master.permutation(len(seeds_local))
        model.train()
        gsum = [torch.zeros_like(p) for p in model.parameters()]
        for start in range(0, len(order), bs):
            idx = order[start:start + bs]
            t0 = time.perf_counter()
            local_pos = {int(seeds_local[i]): i for i in idx}
            feas = [i for i in idx if feasible_mask[i]]
            sbatch_ids = [i for i in idx if not feasible_mask[i]]
            # ---- S-mode seeds: server assembles and ships the batch.
            s_batches = []
            if sbatch_ids:
                send_msg(sock, {"op": "sbreq", "epoch": epoch,
                                "seeds": [int(seeds_local[i]) for i in sbatch_ids]},
                         sender=sender)
                while True:
                    h2, b2 = recv_msg(sock)
                    if h2["op"] == "sbresp":
                        break
                l0 = np.frombuffer(b2[:h2["l0_bytes"]], dtype=np.int64).copy()
                o = h2["l0_bytes"]
                e1r = np.frombuffer(b2[o:o + h2["e1_bytes"]], dtype=np.int64).copy()
                o += h2["e1_bytes"]
                e1c = np.frombuffer(b2[o:o + h2["e1_bytes"]], dtype=np.int64).copy()
                o += h2["e1_bytes"]
                e2r = np.frombuffer(b2[o:o + h2["e2_bytes"]], dtype=np.int64).copy()
                o += h2["e2_bytes"]
                e2c = np.frombuffer(b2[o:o + h2["e2_bytes"]], dtype=np.int64).copy()
                o += h2["e2_bytes"]
                seeds_g = np.frombuffer(b2[o:o + h2["sd_bytes"]],
                                        dtype=np.int64).copy()
                o += h2["sd_bytes"]
                x0 = np.frombuffer(b2[o:], dtype=np.float32).copy() \
                    .reshape(-1, feats_dim)
                s_batches.append((l0, e1r, e1c, e2r, e2c, seeds_g, x0))
                stats["s_seeds"] += len(seeds_g)
            # ---- H-mode seeds: sample locally, cache hits local, miss fetch.
            h_feats, h_edges, h_seeds_g = [], [], []
            if feas:
                gpos = [int(seeds_local[i]) for i in feas]
                # Per-seed RNG streams + shared merge => batches identical
                # to server-side S-mode assembly (Proposition 4 parity).
                parts = [sampler.sample(
                    np.asarray([local_of[g]], dtype=np.int64),
                    epoch_rng(g, epoch, wid + 1)) for g in gpos]
                m = merge_parts(parts)
                l0_g = glob_ids[m["l0"]]
                miss = [int(g) for g in l0_g if int(g) not in cache_pos]
                if miss:
                    send_msg(sock, {"op": "freq", "ids": miss}, sender=sender)
                    while True:
                        h3, b3 = recv_msg(sock)
                        if h3["op"] == "fresp":
                            break
                    miss_arr = np.asarray(miss, dtype=np.int64)
                    rows = np.frombuffer(b3, dtype=np.float32).copy() \
                        .reshape(len(miss_arr), feats_dim)
                else:
                    miss_arr = np.empty(0, dtype=np.int64)
                    rows = np.empty((0, feats_dim), dtype=np.float32)
                miss_pos = {int(g): i for i, g in enumerate(miss_arr)}
                x0 = np.empty((len(l0_g), feats_dim), dtype=np.float32)
                for i, g in enumerate(l0_g):
                    j = cache_pos.get(int(g))
                    if j is not None:
                        x0[i] = cache_feat[j]
                    else:
                        x0[i] = rows[miss_pos[int(g)]]
                h_feats.append(x0)
                h_edges.append((m["e1"][0], m["e1"][1],
                                m["e2"][0], m["e2"][1], m["n_hidden"]))
                h_seeds_g.append(glob_ids[m["seeds"]])
                stats["h_seeds"] += len(feas)
            t1 = time.perf_counter()
            stats["data_s"] += t1 - t0

            # ---- Backward on all sub-batches of this mini-batch.
            for p in gsum:
                p.zero_()
            for (l0, e1r, e1c, e2r, e2c, sg, x0) in s_batches:
                xt = torch.from_numpy(x0)
                n_hidden = int(e1r.max()) + 1 if len(e1r) else 1
                logits = model(xt, (torch.from_numpy(e1r), torch.from_numpy(e1c)),
                               (torch.from_numpy(e2r), torch.from_numpy(e2c)),
                               n_hidden, len(sg))
                y = torch.from_numpy(labels_all[sg]).long()
                loss = F.cross_entropy(logits, y, reduction="sum")
                model.zero_grad()
                loss.backward()
                for acc, p in zip(gsum, model.parameters()):
                    acc += p.grad
            for x0, (e1r, e1c, e2r, e2c, n_hidden), sg in zip(
                    h_feats, h_edges, h_seeds_g):
                logits = model(torch.from_numpy(x0),
                               (torch.from_numpy(e1r), torch.from_numpy(e1c)),
                               (torch.from_numpy(e2r), torch.from_numpy(e2c)),
                               n_hidden, len(sg))
                y = torch.from_numpy(labels_all[sg]).long()
                loss = F.cross_entropy(logits, y, reduction="sum")
                model.zero_grad()
                loss.backward()
                for acc, p in zip(gsum, model.parameters()):
                    acc += p.grad
            stats["compute_s"] += time.perf_counter() - t1
            stats["batches"] += 1

        # Ship accumulated gradient (sum over seeds) + seed count.
        gbuf = np.concatenate([g.detach().numpy().ravel() for g in gsum])
        send_msg(sock, {"op": "grads", "epoch": epoch,
                        "n_seeds": len(seeds_local)}, gbuf.tobytes(),
                 sender=sender)
        epoch += 1

    send_msg(sock, {"op": "bye", "worker": wid, **stats}, sender=sender)
    sock.close()


if __name__ == "__main__":
    main()
