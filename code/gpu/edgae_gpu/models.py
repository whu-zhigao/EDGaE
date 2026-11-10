#!/usr/bin/env python3
"""Models: full-batch GCN (sparse) and two-layer mini-batch GraphSAGE."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FullGCN(nn.Module):
    """Two-layer GCN over a pre-normalized sparse adjacency."""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, adj, features):
        x = torch.sparse.mm(adj, self.w1(features)).relu()
        return torch.sparse.mm(adj, self.w2(x))


class SAGELayers(nn.Module):
    """Mean-aggregation layers applied to sampled bipartite edges."""

    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x0, e1, e2, num_hidden, num_seeds):
        def _tx(x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            return x.long().to(x0.device)
        e1 = (_tx(e1[0]), _tx(e1[1]))
        e2 = (_tx(e2[0]), _tx(e2[1]))
        rows, cols = e1
        h = torch.zeros(num_hidden, x0.shape[1], device=x0.device)
        h.index_add_(0, rows.long(), x0[cols.long()])
        deg = torch.zeros(num_hidden, device=x0.device)
        deg.index_add_(0, rows.long(), torch.ones_like(rows, dtype=torch.float))
        h = F.relu(h / deg.clamp(min=1.0).unsqueeze(1))
        h = self.w1(h)
        rows, cols = e2
        out = torch.zeros(num_seeds, h.shape[1], device=x0.device)
        out.index_add_(0, rows.long(), h[cols.long()])
        deg2 = torch.zeros(num_seeds, device=x0.device)
        deg2.index_add_(0, rows.long(), torch.ones_like(rows, dtype=torch.float))
        out = out / deg2.clamp(min=1.0).unsqueeze(1)
        return self.w2(out)


def normalize_adjacency(indptr, indices, n, device):
    """D^-1/2 (A+I) D^-1/2 as a sparse COO tensor with self-loops."""
    import numpy as np
    counts = np.diff(indptr)
    src_np = np.repeat(np.arange(n, dtype=np.int64), counts)
    dst_np = np.asarray(indices, dtype=np.int64)
    src = torch.from_numpy(src_np)
    dst = torch.from_numpy(dst_np)
    loops = torch.arange(n, dtype=torch.long)
    src = torch.cat([src, loops])
    dst = torch.cat([dst, loops])
    deg = torch.zeros(n)
    deg.index_add_(0, src, torch.ones_like(src, dtype=torch.float))
    inv = deg.pow(-0.5)
    val = inv[src] * inv[dst]
    idx = torch.stack([src, dst])
    return torch.sparse_coo_tensor(
        idx, val, (n, n), device=device).coalesce()
