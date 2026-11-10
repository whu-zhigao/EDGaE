"""Deterministic two-layer neighbor sampling (ported from edgae_gpu).

Cross-mode parity: sampling for seed v in epoch e uses the deterministic
RNG epoch_rng(v, e), so server-side assembly (mode S) and worker-local
sampling (mode H) produce identical neighborhoods.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class SampledBatch:
    seeds: np.ndarray
    layer_nodes: List[np.ndarray]
    edges: List[tuple]


class NeighborSampler:
    """Fan-out sampler over CSR; inverse-degree selection by default."""

    def __init__(self, indptr, indices, fanout=2, uniform=False, seed=0):
        self.indptr, self.indices = indptr, indices
        self.fanout, self.uniform = fanout, uniform
        self.rng = np.random.RandomState(seed)

    def _pick(self, node, rng):
        lo, hi = self.indptr[node], self.indptr[node + 1]
        nb = self.indices[lo:hi]
        if len(nb) == 0:
            return nb
        k = min(self.fanout, len(nb))
        if k == len(nb):
            return nb
        if self.uniform:
            return rng.choice(nb, size=k, replace=False)
        deg = np.array([self.indptr[n + 1] - self.indptr[n] for n in nb],
                       dtype=np.float64)
        p = 1.0 / np.maximum(deg, 1.0)
        return rng.choice(nb, size=k, replace=True, p=p / p.sum())

    def sample(self, seeds: np.ndarray, rng=None) -> SampledBatch:
        rng = rng or self.rng
        seeds = np.asarray(seeds, dtype=np.int64)
        pos1, e2r, e2c = {}, [], []
        for sp, s in enumerate(seeds):
            for n in np.unique(self._pick(int(s), rng)):
                e2r.append(sp)
                e2c.append(pos1.setdefault(int(n), len(pos1)))
        layer1 = np.asarray(list(pos1.keys()), dtype=np.int64)
        pos0, e1r, e1c = {}, [], []
        for hp, node in enumerate(layer1):
            for n in np.unique(self._pick(int(node), rng)):
                e1r.append(hp)
                e1c.append(pos0.setdefault(int(n), len(pos0)))
        layer0 = np.asarray(list(pos0.keys()), dtype=np.int64)
        return SampledBatch(
            seeds=seeds,
            layer_nodes=[layer0, layer1, seeds],
            edges=[(np.asarray(e1r, dtype=np.int64),
                    np.asarray(e1c, dtype=np.int64)),
                   (np.asarray(e2r, dtype=np.int64),
                    np.asarray(e2c, dtype=np.int64))],
        )


def epoch_rng(seed: int, epoch: int, stream: int = 0) -> np.random.RandomState:
    """Deterministic per-(seed, epoch, stream) RNG for cross-mode parity."""
    return np.random.RandomState((int(seed) * 1_000_003 + int(epoch) * 9176
                                  + stream) % (2 ** 31 - 1))


def merge_parts(parts):
    """Merge single-seed SampledBatch parts into one batch.

    Identical merge order on server (S mode) and worker (H mode) guarantees
    byte-identical batches for the same (seed, epoch, worker-stream) triple,
    which is the parity required by Proposition 4.
    """
    l0 = np.unique(np.concatenate([p.layer_nodes[0] for p in parts]))
    pos = {int(g): i for i, g in enumerate(l0)}
    e1r, e1c, e2r, e2c = [], [], [], []
    off_h = 0
    seeds = []
    for pi, p in enumerate(parts):
        l0g = p.layer_nodes[0]
        for r, c in zip(p.edges[0][0], p.edges[0][1]):
            e1r.append(int(r) + off_h)
            e1c.append(pos[int(l0g[int(c)])])
        for _r, c in zip(p.edges[1][0], p.edges[1][1]):
            e2r.append(pi)
            e2c.append(off_h + int(c))
        off_h += len(p.layer_nodes[1])
        seeds.append(int(p.seeds[0]))
    return {
        "l0": l0,
        "e1": (np.asarray(e1r, np.int64), np.asarray(e1c, np.int64)),
        "e2": (np.asarray(e2r, np.int64), np.asarray(e2c, np.int64)),
        "seeds": np.asarray(seeds, np.int64),
        "n_hidden": off_h,
    }
