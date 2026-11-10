#!/usr/bin/env python3
"""OGB-arxiv loader: official files -> torch tensors + undirected CSR."""

from __future__ import annotations

import csv
import gzip
import time
from pathlib import Path

import numpy as np
import torch


DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "ogbn-arxiv"


def _read_gz_lines(path: Path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            if row:
                yield row


def load_ogbn_arxiv(root: Path = DEFAULT_ROOT, cache: bool = True):
    """Load edges, features, labels, and the official time split.

    Symmetrizes the directed citation graph, removes self-loops and duplicate
    arcs, and returns a CSR adjacency plus torch tensors. Parsed arrays are
    cached as .npy so repeated runs start in seconds.
    """
    root = Path(root)
    raw, split = root / "raw", root / "split" / "time"
    npy = root / "_cache"
    if cache and (npy / "features.npy").exists():
        t0 = time.perf_counter()
        features = np.load(npy / "features.npy")
        labels = np.load(npy / "labels.npy")
        indptr = np.load(npy / "indptr.npy")
        indices = np.load(npy / "indices.npy")
        train_idx = np.load(npy / "train_idx.npy")
        valid_idx = np.load(npy / "valid_idx.npy")
        test_idx = np.load(npy / "test_idx.npy")
        unique_edges = int(np.load(npy / "unique_edges.npy"))
        load_s = time.perf_counter() - t0
    else:
        t0 = time.perf_counter()
        feats_rows = [np.asarray(row, dtype=np.float32)
                      for row in _read_gz_lines(raw / "node-feat.csv.gz")]
        features = np.stack(feats_rows)
        labels = np.asarray(
            [int(r[0]) for r in _read_gz_lines(raw / "node-label.csv.gz")],
            dtype=np.int64,
        )
        pairs = set()
        for row in _read_gz_lines(raw / "edge.csv.gz"):
            a, b = int(row[0]), int(row[1])
            if a != b:
                pairs.add((a, b) if a < b else (b, a))
        src = np.fromiter((a for a, _ in pairs), dtype=np.int64, count=len(pairs))
        dst = np.fromiter((b for _, b in pairs), dtype=np.int64, count=len(pairs))
        both_src = np.concatenate([src, dst])
        both_dst = np.concatenate([dst, src])
        order = np.argsort(both_src, kind="stable")
        both_src, both_dst = both_src[order], both_dst[order]
        indptr = np.zeros(features.shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, both_src + 1, 1)
        np.cumsum(indptr, out=indptr)
        indices = both_dst
        train_idx = np.asarray(
            [int(r[0]) for r in _read_gz_lines(split / "train.csv.gz")],
            dtype=np.int64)
        valid_idx = np.asarray(
            [int(r[0]) for r in _read_gz_lines(split / "valid.csv.gz")],
            dtype=np.int64)
        test_idx = np.asarray(
            [int(r[0]) for r in _read_gz_lines(split / "test.csv.gz")],
            dtype=np.int64)
        unique_edges = len(pairs)
        load_s = time.perf_counter() - t0
        if cache:
            npy.mkdir(parents=True, exist_ok=True)
            for name, arr in (
                ("features", features), ("labels", labels),
                ("indptr", indptr), ("indices", indices),
                ("train_idx", train_idx), ("valid_idx", valid_idx),
                ("test_idx", test_idx),
            ):
                np.save(npy / f"{name}.npy", arr)
            np.save(npy / "unique_edges.npy", np.asarray(unique_edges))
    degrees = np.diff(indptr)
    meta = {
        "vertices": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "num_classes": int(labels.max() + 1),
        "unique_undirected_edges": int(unique_edges),
        "train": int(len(train_idx)), "valid": int(len(valid_idx)),
        "test": int(len(test_idx)), "load_seconds": round(load_s, 2),
    }
    return {
        "features": torch.from_numpy(features),
        "labels": torch.from_numpy(labels),
        "indptr": indptr, "indices": indices, "degrees": degrees,
        "train_idx": torch.from_numpy(train_idx),
        "valid_idx": torch.from_numpy(valid_idx),
        "test_idx": torch.from_numpy(test_idx),
        "meta": meta,
    }
