#!/usr/bin/env python3
"""Native PyTorch/DGL runner for ogbn-arxiv accuracy and local DGL runtime.

This runner uses the official OGB files already downloaded under datasets/.
It intentionally does not pretend that a local macOS process is a Raspberry Pi
cluster. Its runtime evidence is for a native, local, logical-worker emulation
only; wall-clock values must not be reported as physical edge measurements.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List


def read_ids(path: Path) -> List[int]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return [int(row[0]) for row in csv.reader(stream)]


def read_labels(path: Path) -> List[int]:
    return read_ids(path)


def read_edges(path: Path) -> List[tuple[int, int]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        return [(int(row[0]), int(row[1])) for row in csv.reader(stream) if len(row) >= 2]


def read_features(path: Path) -> "torch.Tensor":
    import torch
    rows: List[List[float]] = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            rows.append([float(value) for value in row])
    return torch.tensor(rows, dtype=torch.float32)


def accuracy(model, graph, features, labels, mask) -> float:
    import torch
    model.eval()
    with torch.no_grad():
        prediction = model(graph, features).argmax(dim=1)
        return (prediction[mask] == labels[mask]).float().mean().item()


def build_gcn(in_features: int, hidden_dim: int, num_classes: int):
    import torch.nn as nn
    import dgl.nn as dglnn

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = dglnn.GraphConv(in_features, hidden_dim, allow_zero_in_degree=True)
            self.layer2 = dglnn.GraphConv(hidden_dim, num_classes, allow_zero_in_degree=True)

        def forward(self, graph, features):
            hidden = self.layer1(graph, features).relu()
            return self.layer2(graph, hidden)

    return GCN()


def build_sage(in_features: int, hidden_dim: int, num_classes: int):
    import torch.nn as nn
    import dgl.nn as dglnn

    class GraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = dglnn.SAGEConv(in_features, hidden_dim, "mean")
            self.layer2 = dglnn.SAGEConv(hidden_dim, num_classes, "mean")

        def forward(self, blocks, features):
            hidden = self.layer1(blocks[0], features).relu()
            return self.layer2(blocks[1], hidden)

    return GraphSAGE()


def train_full_batch(args, graph, features, labels, train_ids, valid_ids, test_ids):
    import dgl
    import torch

    device = torch.device("cpu")
    graph = dgl.add_self_loop(graph).to(device)
    features = features.to(device)
    labels = labels.to(device)
    num_nodes = graph.num_nodes()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_ids] = True
    valid_mask[valid_ids] = True
    test_mask[test_ids] = True

    model = build_gcn(features.shape[1], args.hidden_dim, int(labels.max().item()) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    epoch_times: List[float] = []
    best_valid = 0.0
    best_test = 0.0
    target_reached: int | None = None

    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        optimizer.zero_grad()
        output = model(graph, features)
        loss = loss_fn(output[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        epoch_times.append(time.perf_counter() - started)
        valid_acc = accuracy(model, graph, features, labels, valid_mask)
        test_acc = accuracy(model, graph, features, labels, test_mask)
        if valid_acc > best_valid:
            best_valid = valid_acc
            best_test = test_acc
        if target_reached is None and test_acc >= args.target_accuracy:
            target_reached = epoch + 1
        print(
            f"epoch={epoch+1} loss={loss.item():.4f} "
            f"valid={valid_acc:.4f} test={test_acc:.4f}",
            flush=True,
        )

    return {
        "native_full_batch_epoch_s": sum(epoch_times) / len(epoch_times),
        "best_valid_accuracy": best_valid,
        "test_accuracy": best_test,
        "target_accuracy": args.target_accuracy,
        "epochs_to_target": target_reached,
    }


def train_minibatch_baseline(args, graph, features, labels, train_ids):
    import dgl
    import dgl.dataloading
    import torch

    device = torch.device("cpu")
    graph = graph.to(device)
    features = features.to(device)
    sampler = dgl.dataloading.NeighborSampler([args.fanout, args.fanout])
    train_tensor = torch.tensor(train_ids, dtype=torch.int64)
    loader = dgl.dataloading.DataLoader(
        graph,
        train_tensor,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        device=device,
        num_workers=0,
    )
    model = build_sage(
        features.shape[1], args.hidden_dim, int(labels.max().item()) + 1
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    timings: List[float] = []

    for epoch in range(args.runtime_epochs):
        model.train()
        started = time.perf_counter()
        losses = []
        for input_nodes, output_nodes, blocks in loader:
            blocks = [block.to(device) for block in blocks]
            batch_features = features[input_nodes]
            batch_labels = labels[output_nodes]
            optimizer.zero_grad()
            prediction = model(blocks, batch_features)
            loss = loss_fn(prediction, batch_labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        timings.append(time.perf_counter() - started)
        print(f"runtime epoch={epoch+1} time={timings[-1]:.4f}s", flush=True)

    return {
        "DGL_E_epoch_s": sum(timings) / len(timings),
        "native_minibatch_epoch_s": sum(timings) / len(timings),
        "runtime_batches_per_epoch": (len(train_ids) + args.batch_size - 1) // args.batch_size,
        "runtime_note": (
            "Native local DGL/PyTorch mini-batch execution with the full feature "
            "matrix resident in the process; no physical network transfer."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/ogbn-arxiv/arxiv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runtime-epochs", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--fanout", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--target-accuracy", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1,
                        help="Logical worker count recorded by this local emulation")
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    args = parser.parse_args()

    try:
        import dgl
        import torch
    except Exception as exc:
        print(json.dumps({
            "measurement_valid": False,
            "error": f"native dependency unavailable: {type(exc).__name__}: {exc}",
        }))
        return 3

    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads() // max(1, args.workers)))

    root = Path(args.data_dir)
    started = time.perf_counter()
    edges = read_edges(root / "raw" / "edge.csv.gz")
    labels = read_labels(root / "raw" / "node-label.csv.gz")
    features = read_features(root / "raw" / "node-feat.csv.gz")
    train_ids = read_ids(root / "split" / "time" / "train.csv.gz")
    valid_ids = read_ids(root / "split" / "time" / "valid.csv.gz")
    test_ids = read_ids(root / "split" / "time" / "test.csv.gz")
    source = [edge[0] for edge in edges]
    target = [edge[1] for edge in edges]
    # ogbn-arxiv's citation topology is directed. Use both directions for the
    # undirected message passing used by the system prototype.
    graph = dgl.graph(
        (torch.tensor(source + target), torch.tensor(target + source)),
        num_nodes=len(labels),
    )
    label_tensor = torch.tensor(labels, dtype=torch.int64)
    load_time = time.perf_counter() - started

    result: Dict[str, object] = {
        "measurement_source": "native-dgl-pytorch-local-emulation-v1",
        "measurement_valid": True,
        "evidence_level": "native-local-emulation",
        "environment_type": "single-host-logical-workers",
        "hardware_platform": "local-cpu",
        "validity_scope": "native-accuracy-and-local-dgl-runtime-only",
        "graph_source": str((root / "raw" / "edge.csv.gz").resolve()),
        "graph_vertices": len(labels),
        "graph_edges": len(edges),
        "feature_dim": int(features.shape[1]),
        "num_classes": int(label_tensor.max().item()) + 1,
        "official_split_sizes": {
            "train": len(train_ids), "valid": len(valid_ids), "test": len(test_ids)
        },
        "T_data_load_s": load_time,
        "torch_version": torch.__version__,
        "dgl_version": dgl.__version__,
        "seed": args.seed,
        "logical_workers": args.workers,
        "measurement_utc": datetime.now(timezone.utc).isoformat(),
        "notes": "Local native execution; not a Raspberry Pi or multi-host measurement.",
    }

    if not args.skip_accuracy:
        result.update(train_full_batch(
            args, graph, features, label_tensor, train_ids, valid_ids, test_ids
        ))
    if not args.skip_runtime:
        result.update(train_minibatch_baseline(
            args, graph, features, label_tensor, train_ids
        ))

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
