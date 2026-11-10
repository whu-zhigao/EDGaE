#!/usr/bin/env python3
"""Training loops: full-batch GCN reference and two SAGE feature paths.

Modes (sampling identical across modes for a given seed, per Prop. 4):
  server : every batch's feature rows move host -> device (DGL-E analogue);
  edgae  : GPU-resident per-worker caches; only misses move host -> device.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from .models import FullGCN, SAGELayers, normalize_adjacency
from .sampling import NeighborSampler


@torch.no_grad()
def _eval_full(model, adj, features, labels, idx, device):
    model.eval()
    pred = model(adj, features).argmax(dim=1)
    return (pred[idx.to(device)] == labels[idx.to(device)]).float().mean().item()


def train_fullbatch_gcn(data, device, epochs=200, lr=1e-2, weight_decay=5e-4,
                        seed=1, log_every=5):
    torch.manual_seed(seed)
    features = data["features"].to(device)
    labels = data["labels"].to(device)
    adj = normalize_adjacency(
        data["indptr"], data["indices"], features.shape[0], device)
    model = FullGCN(features.shape[1], 256, int(data["meta"]["num_classes"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr, va, te = (data["train_idx"].to(device), data["valid_idx"].to(device),
                  data["test_idx"].to(device))
    history, best = [], {"val": 0.0, "test": 0.0, "epoch": 0}
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        opt.zero_grad()
        out = model(adj, features)
        loss = F.cross_entropy(out[tr], labels[tr])
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        if epoch % log_every == 0 or epoch == 1:
            v = _eval_full(model, adj, features, labels, data["valid_idx"], device)
            t = _eval_full(model, adj, features, labels, data["test_idx"], device)
            history.append({"epoch": epoch, "val": v, "test": t, "seconds": dt})
            if v > best["val"]:
                best = {"val": v, "test": t, "epoch": epoch}
            print(f"  [gcn] epoch {epoch:3d} loss {loss.item():.4f} "
                  f"val {v:.4f} test {t:.4f} ({dt*1000:.0f} ms)")
    return {"best": best, "history": history,
            "total_train_s": sum(h["seconds"] for h in history) * log_every}


def _time_to_target(history, target_frac=0.95):
    """First epoch reaching target_frac of final best val acc."""
    if not history:
        return None
    final_val = max(h["val"] for h in history)
    target = final_val * target_frac
    # Approximate cumulative time linearly between logged epochs.
    for i, h in enumerate(history):
        if h["val"] >= target:
            prev_epochs = history[i - 1]["epoch"] if i else 0
            span = h["epoch"] - prev_epochs
            frac = 1.0 if span == 1 else 0.5
            epochs = prev_epochs + span * frac
            per_epoch = h["seconds"]
            return {"epoch": epochs, "train_s": epochs * per_epoch,
                    "target_val": target, "final_val": final_val}
    return None


class FeatureStore:
    """Host feature store with optional GPU-resident cache tiers."""

    def __init__(self, features_cpu, caches, device, feature_dim):
        self.host = features_cpu
        self.device = device
        n = features_cpu.shape[0]
        slot = torch.full((n,), -1, dtype=torch.long)
        tiers = []
        for tier_id, cache in enumerate(caches):
            ids = torch.tensor(sorted(cache), dtype=torch.long)
            tiers.append(features_cpu[ids].to(device))
            slot[ids] = torch.arange(len(ids)) + tier_id * n
        # Fill on CPU first, then move: GPU-side slot lookups must index a
        # GPU-resident table (the smoke GPU run caught the CPU/GPU mismatch).
        self.slot = slot.to(device)
        self.tiers, self.n = tiers, n
        self.bytes_moved = 0

    def gather(self, rows: torch.Tensor) -> torch.Tensor:
        rows = rows.to(self.device)
        slots = self.slot[rows]
        out = torch.empty(len(rows), self.host.shape[1], device=self.device)
        hit_mask = slots >= 0
        for tier_id, tier in enumerate(self.tiers):
            m = hit_mask & (slots >= tier_id * self.n) & (slots < (tier_id + 1) * self.n)
            if m.any():
                out[m] = tier[slots[m] - tier_id * self.n]
        miss = ~hit_mask
        if miss.any():
            miss_rows = rows[miss].cpu()
            out[miss] = self.host[miss_rows].to(self.device, non_blocking=True)
            self.bytes_moved += int(len(miss_rows)) * self.host.shape[1] * 4
        return out


def train_minibatch_sage(data, mode, device, sampler, partitions, caches,
                         epochs=60, lr=1e-2, weight_decay=5e-4, seed=1,
                         batch_size=1024, log_every=5, worker_of_seed=None,
                         feature_dim=128, limit_train=None):
    """Train GraphSAGE under one feature path; sampling is mode-independent."""
    torch.manual_seed(seed)
    labels = data["labels"].to(device)
    model = SAGELayers(feature_dim, 256, int(data["meta"]["num_classes"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    store = FeatureStore(
        data["features"], caches if mode == "edgae" else [], device, feature_dim)
    train_seeds = data["train_idx"].numpy()
    if limit_train:
        train_seeds = train_seeds[:limit_train]
    # Deterministic batch order shared by both modes for parity.
    order_rng = np.random.RandomState(seed * 7717 + 3)
    history, best = [], {"val": 0.0, "test": 0.0, "epoch": 0}
    data_time_s = compute_time_s = 0.0
    for epoch in range(1, epochs + 1):
        order = order_rng.permutation(len(train_seeds))
        for start in range(0, len(order), batch_size):
            t0 = time.perf_counter()
            batch = sampler.sample(train_seeds[order[start:start + batch_size]])
            x0 = store.gather(torch.from_numpy(batch.layer_nodes[0]))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            logits = model(x0, batch.edges[0], batch.edges[1],
                           len(batch.layer_nodes[1]), len(batch.seeds))
            y = labels[torch.from_numpy(batch.seeds).to(device)]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            data_time_s += t1 - t0
            compute_time_s += time.perf_counter() - t1
        if epoch % log_every == 0 or epoch == 1:
            v = _eval_minibatch(model, store, sampler, data["valid_idx"][:4096],
                                labels, device, batch_size)
            t = _eval_minibatch(model, store, sampler, data["test_idx"][:4096],
                                labels, device, batch_size)
            history.append({"epoch": epoch, "val": v, "test": t,
                            "seconds": (data_time_s + compute_time_s) / epoch})
            if v > best["val"]:
                best = {"val": v, "test": t, "epoch": epoch}
            print(f"  [{mode}] epoch {epoch:3d} loss {loss.item():.4f} "
                  f"val {v:.4f} test {t:.4f}")
    total_accessed = store.bytes_moved  # misses only
    return {
        "best": best, "history": history,
        "data_path_s": data_time_s, "compute_s": compute_time_s,
        "miss_bytes": total_accessed,
        "time_to_target": _time_to_target(history),
    }


@torch.no_grad()
def _eval_minibatch(model, store, sampler, idx, labels, device, batch_size):
    model.eval()
    correct = total = 0
    ids = idx.numpy()
    for start in range(0, len(ids), batch_size):
        batch = sampler.sample(ids[start:start + batch_size])
        x0 = store.gather(torch.from_numpy(batch.layer_nodes[0]))
        logits = model(x0, batch.edges[0], batch.edges[1],
                       len(batch.layer_nodes[1]), len(batch.seeds))
        pred = logits.argmax(dim=1).cpu()
        y = labels[torch.from_numpy(batch.seeds).to(device)].cpu()
        correct += int((pred == y).sum()); total += len(y)
    model.train()
    return correct / max(total, 1)
