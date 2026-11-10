#!/usr/bin/env python3
"""Server process of the emulated fleet.

Authoritative graph/features, paced residual feature service, paced S-mode
batch assembly, synchronous Adam aggregation, and validation evaluation.
One persistent handler thread per worker; gradients flow through per-worker
queues so the main loop blocks cleanly.
"""
from __future__ import annotations
import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from net import recv_msg, send_msg, PacedSender
from sampling import NeighborSampler, epoch_rng, merge_parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()
    cfg = json.loads(Path(args.manifest).read_text())
    try:
        os.sched_setaffinity(0, set(cfg["cores"]))
    except (AttributeError, OSError):
        pass  # non-Linux host (local smoke); enforcement logged in env
    torch.set_num_threads(len(cfg["cores"]))

    run_dir = Path(cfg["run_dir"])
    features = np.load(run_dir / "features.npy")
    labels = np.load(run_dir / "labels.npy")
    indptr = np.load(run_dir / "indptr.npy")
    indices = np.load(run_dir / "indices.npy")
    valid_idx = np.load(run_dir / "valid_idx.npy")
    test_idx = np.load(run_dir / "test_idx.npy")
    feats_dim = features.shape[1]

    sampler = NeighborSampler(indptr, indices, fanout=cfg["fanout"])
    target_bps = cfg["target_mbps"] * 1e6 / 8.0 if cfg.get("target_mbps") else None

    import torch.nn as nn
    import torch.nn.functional as Fn

    class SAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(feats_dim, cfg["hidden"], bias=False)
            self.w2 = nn.Linear(cfg["hidden"], cfg["classes"], bias=False)

        def forward(self, x0, e1, e2, n_hidden, n_seeds):
            r1, c1 = e1
            r2, c2 = e2
            h = torch.zeros(n_hidden, x0.shape[1])
            h.index_add_(0, r1, x0[c1])
            d1 = torch.zeros(n_hidden)
            d1.index_add_(0, r1, torch.ones_like(r1, dtype=torch.float))
            h = Fn.relu(h / d1.clamp(min=1).unsqueeze(1))
            h = self.w1(h)
            out = torch.zeros(n_seeds, h.shape[1])
            out.index_add_(0, r2, h[c2])
            d2 = torch.zeros(n_seeds)
            d2.index_add_(0, r2, torch.ones_like(r2, dtype=torch.float))
            return self.w2(out / d2.clamp(min=1).unsqueeze(1))

    torch.manual_seed(cfg["run_seed"])
    model = SAGE()
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["weight_decay"])

    def flat():
        return np.concatenate(
            [p.detach().numpy().ravel() for p in model.parameters()])

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((cfg["host"], cfg["port"]))
    srv.listen(cfg["workers"])

    socks, senders = {}, {}
    grad_q = {w: queue.Queue() for w in range(cfg["workers"])}
    bye = {}
    bytes_feat = [0] * cfg["workers"]
    bytes_sbatch = [0] * cfg["workers"]

    def assemble(seeds, epoch, wid):
        """Server-side S-mode assembly, per-seed RNG streams (parity with H)."""
        parts = [sampler.sample(np.asarray([s], dtype=np.int64),
                                epoch_rng(int(s), epoch, wid + 1))
                 for s in seeds]
        m = merge_parts(parts)
        return (m["l0"].astype(np.int64),
                m["e1"][0], m["e1"][1], m["e2"][0], m["e2"][1],
                features[m["l0"]])

    def handle(wid: int, sock: socket.socket):
        senders[wid] = PacedSender(sock, target_bps)
        while True:
            try:
                hdr, body = recv_msg(sock)
            except ConnectionError:
                return
            op = hdr["op"]
            if op == "freq":
                ids = np.asarray(hdr["ids"], dtype=np.int64)
                payload = features[ids].tobytes()
                send_msg(sock, {"op": "fresp"}, payload, sender=senders[wid])
                bytes_feat[wid] += len(payload)
            elif op == "sbreq":
                seeds = np.asarray(hdr["seeds"], dtype=np.int64)
                l0, e1r, e1c, e2r, e2c, x0 = assemble(
                    seeds, hdr["epoch"], wid)
                payload = (l0.tobytes() + e1r.tobytes() + e1c.tobytes()
                           + e2r.tobytes() + e2c.tobytes()
                           + seeds.astype(np.int64).tobytes() + x0.tobytes())
                send_msg(sock, {"op": "sbresp",
                                "l0_bytes": len(l0) * 8,
                                "e1_bytes": len(e1r) * 8,
                                "e2_bytes": len(e2r) * 8,
                                "sd_bytes": len(seeds) * 8},
                         payload, sender=senders[wid])
                bytes_sbatch[wid] += len(payload)
            elif op == "grads":
                g = np.frombuffer(body, dtype=np.float32)
                off, gs = 0, []
                for p in model.parameters():
                    n = p.numel()
                    gs.append(torch.from_numpy(
                        g[off:off + n].reshape(p.detach().numpy().shape)
                    ).clone())
                    off += n
                grad_q[wid].put((gs, hdr["n_seeds"]))
            elif op == "bye":
                bye[wid] = hdr
                return

    for _ in range(cfg["workers"]):
        conn, _ = srv.accept()
        hdr, _ = recv_msg(conn)
        wid = hdr["worker"]
        socks[wid] = conn
        threading.Thread(target=handle, args=(wid, conn),
                         daemon=True).start()

    val_rng = np.random.RandomState(cfg["run_seed"] * 7 + 5)
    val_seeds = valid_idx[val_rng.permutation(len(valid_idx))
                          [:cfg["eval_seeds"]]]
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        params = flat().tobytes()
        for wid, sock in socks.items():
            send_msg(sock, {"op": "params"}, params, sender=senders[wid])
        got = {}
        while len(got) < cfg["workers"]:
            for wid in range(cfg["workers"]):
                if wid not in got:
                    try:
                        got[wid] = grad_q[wid].get(timeout=1)
                    except queue.Empty:
                        pass
        opt.zero_grad()
        tot = sum(n for _, n in got.values())
        for gs, n in got.values():
            for p, g in zip(model.parameters(), gs):
                if p.grad is None:
                    p.grad = g * (n / tot)
                else:
                    p.grad += g * (n / tot)
        opt.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for s in range(0, len(val_seeds), 1024):
                chunk = val_seeds[s:s + 1024]
                b = sampler.sample(chunk, epoch_rng(cfg["run_seed"], 0, 999))
                x0 = torch.from_numpy(features[b.layer_nodes[0]])
                logits = model(x0,
                               (torch.from_numpy(b.edges[0][0]),
                                torch.from_numpy(b.edges[0][1])),
                               (torch.from_numpy(b.edges[1][0]),
                                torch.from_numpy(b.edges[1][1])),
                               len(b.layer_nodes[1]), len(chunk))
                correct += int((logits.argmax(1).numpy()
                                == labels[chunk]).sum())
        val_acc = correct / len(val_seeds)
        history.append({"epoch": epoch, "val": val_acc,
                        "t": time.perf_counter() - t0})
        print(f"  [fleet] epoch {epoch:3d} val {val_acc:.4f}", flush=True)

    train_s = time.perf_counter() - t0
    model.eval()
    te = test_idx[:cfg["eval_seeds"]]
    correct = 0
    with torch.no_grad():
        for s in range(0, len(te), 1024):
            chunk = te[s:s + 1024]
            b = sampler.sample(chunk, epoch_rng(cfg["run_seed"], 0, 998))
            x0 = torch.from_numpy(features[b.layer_nodes[0]])
            logits = model(x0,
                           (torch.from_numpy(b.edges[0][0]),
                            torch.from_numpy(b.edges[0][1])),
                           (torch.from_numpy(b.edges[1][0]),
                            torch.from_numpy(b.edges[1][1])),
                           len(b.layer_nodes[1]), len(chunk))
            correct += int((logits.argmax(1).numpy() == labels[chunk]).sum())
    test_acc = correct / len(te)
    final_val = max(h["val"] for h in history)
    target = 0.95 * final_val
    ttt = next((h for h in history if h["val"] >= target), None)

    for wid, sock in socks.items():
        try:
            send_msg(sock, {"op": "stop"}, sender=senders[wid])
        except OSError:
            pass
    time.sleep(2.0)
    result = {
        "best_val": final_val, "test_acc": test_acc,
        "ttt_epoch": ttt["epoch"] if ttt else None,
        "ttt_train_s": ttt["t"] if ttt else None,
        "train_s": train_s,
        "bytes_feature_total": sum(bytes_feat),
        "bytes_sbatch_total": sum(bytes_sbatch),
        "worker_stats": bye, "history": history,
    }
    (run_dir / "server_result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k != "history"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
