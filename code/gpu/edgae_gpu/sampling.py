#!/usr/bin/env python3
"""Two-layer neighbor sampling with uniform and inverse-degree weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SampledBatch:
    """Nodes and inter-layer edges for one mini-batch.

    layer_nodes[0] = input feature ids, layer_nodes[1] = hidden ids,
    layer_nodes[2] = seed ids. edges[k] = (rows, cols) with rows indexing
    layer k and cols indexing layer k-1; row i aggregates its neighbors.
    """

    seeds: np.ndarray
    layer_nodes: List[np.ndarray]
    edges: List[tuple]


class NeighborSampler:
    """Fan-out sampler over a CSR graph; deterministic given ``seed``."""

    def __init__(self, indptr, indices, degrees, fanout=2, uniform=True,
                 seed=0):
        self.indptr, self.indices, self.degrees = indptr, indices, degrees
        self.fanout = fanout
        self.uniform = uniform
        self.rng = np.random.RandomState(seed)

    def _sample_neighbors(self, node: int) -> np.ndarray:
        lo, hi = self.indptr[node], self.indptr[node + 1]
        if hi == lo:
            return np.empty(0, dtype=np.int64)
        neighbors = self.indices[lo:hi]
        k = min(self.fanout, len(neighbors))
        if k == len(neighbors):
            return neighbors.astype(np.int64)
        if self.uniform:
            return self.rng.choice(neighbors, size=k, replace=False)
        # Inverse-degree weighting (the paper's non-uniform sampler).
        weights = 1.0 / np.maximum(self.degrees[neighbors], 1.0)
        pick = self.rng.choice(
            len(neighbors), size=k, replace=True, p=weights / weights.sum())
        return neighbors[np.unique(pick)].astype(np.int64)

    def sample(self, seeds: np.ndarray) -> SampledBatch:
        seeds = np.asarray(seeds, dtype=np.int64)
        pos1: dict = {}
        e2_rows, e2_cols = [], []
        for s_pos, s in enumerate(seeds):
            for n in self._sample_neighbors(int(s)):
                e2_rows.append(s_pos)
                e2_cols.append(pos1.setdefault(int(n), len(pos1)))
        layer1 = np.asarray(list(pos1.keys()), dtype=np.int64)
        pos0: dict = {}
        e1_rows, e1_cols = [], []
        for h_pos, node in enumerate(layer1):
            for n in self._sample_neighbors(int(node)):
                e1_rows.append(h_pos)
                e1_cols.append(pos0.setdefault(int(n), len(pos0)))
        layer0 = np.asarray(list(pos0.keys()), dtype=np.int64)
        return SampledBatch(
            seeds=seeds,
            layer_nodes=[layer0, layer1, seeds],
            edges=[
                (np.asarray(e1_rows, dtype=np.int64),
                 np.asarray(e1_cols, dtype=np.int64)),
                (np.asarray(e2_rows, dtype=np.int64),
                 np.asarray(e2_cols, dtype=np.int64)),
            ],
        )
