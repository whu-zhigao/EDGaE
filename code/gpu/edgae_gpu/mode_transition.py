#!/usr/bin/env python3
"""E5: native mode-transition check between server-centric and hybrid paths.

Mechanism
---------
The trace layer predicts break-even horizons H* for hybrid vs. server-centric
execution from byte counts and bandwidth (Table tbl-mode-selection). This
experiment checks the mechanism natively: both feature paths run on the A800,
but host->device feature movement is *paced* to emulate effective edge
bandwidths (10/100/1000 Mbps) in addition to native PCIe.

  S (server-centric): every batch's feature rows move host->device, paced.
  H (hybrid/EDGaE):   GPU-resident cache tiers serve hits locally; only miss
                      rows move host->device, paced.

Only the data path is timed (sampling + feature staging). Compute parity
between the paths is already established by E2/P4; repeating model training
would add no information for the bandwidth comparison.

Evidence discipline
-------------------
Wall-clock here is native GPU timing of a *mechanism* (paced transfer), not an
edge-device deployment. Rows record the pacing target and achieved throughput
so the emulation quality is auditable.
"""

from __future__ import annotations

import time

import numpy as np
import torch


class PacedMover:
    """Move host rows to device in chunks at a target effective bandwidth.

    Chunking keeps the sleep pacing meaningful at GPU scale: a single large
    non_blocking copy would complete at PCIe speed and the sleep would only
    serialize afterwards, misrepresenting a slow shared link. We move
    ~1 MB chunks and sleep between them so achieved throughput is close to
    the target even though each chunk itself crosses at native speed.
    """

    def __init__(self, host, device, target_bytes_per_s=None, chunk_rows=2048):
        self.host = host
        self.device = device
        self.target = target_bytes_per_s  # None => native (no pacing)
        self.chunk_rows = chunk_rows
        self.bytes_moved = 0
        self.pacing_s = 0.0

    def gather(self, rows_np: np.ndarray) -> torch.Tensor:
        rows = torch.from_numpy(np.ascontiguousarray(rows_np))
        n = len(rows)
        out = torch.empty(n, self.host.shape[1], device=self.device)
        if self.target is None:
            out[:] = self.host[rows].to(self.device, non_blocking=True)
            self.bytes_moved += int(n * self.host.shape[1] * self.host.element_size())
            return out
        row_bytes = self.host.shape[1] * self.host.element_size()
        start = time.perf_counter()
        for begin in range(0, n, self.chunk_rows):
            idx = rows[begin:begin + self.chunk_rows]
            chunk = self.host[idx].to(self.device, non_blocking=True)
            out[begin:begin + self.chunk_rows] = chunk
            moved = len(idx) * row_bytes
            self.bytes_moved += moved
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if self.target:
                time.sleep(moved / self.target)
        self.pacing_s += time.perf_counter() - start
        return out


class ModePathStore:
    """Feature store for one mode; hits from GPU tiers, misses paced."""

    def __init__(self, features_cpu, caches, device, target_bytes_per_s):
        self.host = features_cpu
        self.device = device
        self.target = target_bytes_per_s
        n = features_cpu.shape[0]
        slot = torch.full((n,), -1, dtype=torch.long)
        tiers = []
        for tier_id, cache in enumerate(caches):
            ids = torch.tensor(sorted(cache), dtype=torch.long)
            tiers.append(features_cpu[ids].to(device))
            slot[ids] = torch.arange(len(ids)) + tier_id * n
        self.slot = slot.to(device)
        self.tiers, self.n = tiers, n
        self.mover = PacedMover(features_cpu, device, target_bytes_per_s)

    def gather(self, rows_np: np.ndarray) -> torch.Tensor:
        rows = torch.from_numpy(np.ascontiguousarray(rows_np)).to(self.device)
        slots = self.slot[rows]
        out = torch.empty(len(rows), self.host.shape[1], device=self.device)
        hit_mask = slots >= 0
        for tier_id, tier in enumerate(self.tiers):
            m = hit_mask & (slots >= tier_id * self.n) & (slots < (tier_id + 1) * self.n)
            if m.any():
                out[m] = tier[slots[m] - tier_id * self.n]
        miss = (~hit_mask).cpu().numpy()
        if miss.any():
            miss_rows = rows[miss].cpu().numpy()
            out[torch.from_numpy(np.flatnonzero(miss)).to(self.device)] = \
                self.mover.gather(miss_rows)
        return out


def run_mode_transition(data, device, sampler, partitions, caches,
                        bandwidths_mbps, epochs=3, batch_size=1024,
                        limit_train=4096, seed=1):
    """Time the paced data path per (bandwidth, mode).

    Returns a list of result dicts (one per bandwidth x mode).
    """
    train_seeds = data["train_idx"].numpy()[:limit_train]
    order_rng = np.random.RandomState(seed * 7717 + 3)
    order = order_rng.permutation(len(train_seeds))
    batches = [train_seeds[order[s:s + batch_size]]
               for s in range(0, len(order), batch_size)]

    # Pre-sample once so both modes see identical batch node sets.
    sampled = [sampler.sample(b) for b in batches]

    results = []
    for bw in bandwidths_mbps:
        target = None if (bw is None or bw <= 0) else bw * 1e6 / 8.0
        for mode in ("server", "edgae"):
            store = ModePathStore(
                data["features"],
                [] if mode == "server" else caches,
                device, target,
            )
            data_s = 0.0
            for _epoch in range(epochs):
                t_epoch = time.perf_counter()
                for batch in sampled:
                    store.gather(batch.layer_nodes[0])
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                data_s += time.perf_counter() - t_epoch
            per_epoch = data_s / epochs
            bytes_epoch = store.mover.bytes_moved / epochs
            achieved = (store.mover.bytes_moved / (per_epoch * epochs)) if per_epoch > 0 else 0.0
            results.append({
                "bandwidth_mbps": bw if bw is not None else "native",
                "mode": {"server": "S", "edgae": "H"}[mode],
                "epochs": epochs,
                "train_seeds": len(train_seeds),
                "data_path_s_per_epoch": per_epoch,
                "bytes_per_epoch": store.mover.bytes_moved / epochs,
                "achieved_MBps": achieved / 1e6,
                "bytes_total": store.mover.bytes_moved,
            })
            print(f"  [E5] bw={bw or 'native'} mode={mode:6s} "
                  f"{per_epoch:8.3f} s/epoch "
                  f"{bytes_epoch/1e6:7.2f} MB/epoch "
                  f"({achieved/1e6:8.2f} MB/s)")
    return results
