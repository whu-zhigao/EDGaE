#!/usr/bin/env python3
"""Orchestrator: build placement, spawn the emulated fleet, collect results.

Real-task protocol on the emulated edge fleet:
  modes  : server-centric (DGL-E analogue, all seeds S) vs hybrid (EDGaE:
           feasible seeds H + fallback seeds S -> natural mixed mode)
  task   : ogbn-arxiv official split, two-layer GraphSAGE, fan-out 2
  metrics: preprocessing-inclusive time-to-95%-val-accuracy, final test
           accuracy, per-mode bytes, placement fallbacks
Constraints: per-worker CPU affinity (2 cores), RLIMIT_AS memory cap,
paced server->worker transfers at the target bandwidth.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_loader import load_ogbn_arxiv
from placement import place, build_caches
from sampling import NeighborSampler


def build_worker_artifacts(run_dir, data, partitions, replicas, caches,
                           feasible, workers):
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "features.npy", data["features"])
    np.save(run_dir / "labels.npy", data["labels"])
    np.save(run_dir / "indptr.npy", data["indptr"])
    np.save(run_dir / "indices.npy", data["indices"])
    np.save(run_dir / "valid_idx.npy", data["valid_idx"])
    np.save(run_dir / "test_idx.npy", data["test_idx"])
    for w in range(workers):
        ids = np.asarray(sorted(replicas[w]), dtype=np.int64)
        local = {int(g): i for i, g in enumerate(ids)}
        indptr_w = np.zeros(len(ids) + 1, dtype=np.int64)
        src, dst = [], []
        for i, g in enumerate(ids):
            for nb in data["indices"][data["indptr"][g]:data["indptr"][g + 1]]:
                j = local.get(int(nb))
                if j is not None:
                    src.append(i)
                    dst.append(j)
        src = np.asarray(src, dtype=np.int64)
        order = np.argsort(src, kind="stable")
        src = src[order]
        dst = np.asarray(dst, dtype=np.int64)[order]
        np.add.at(indptr_w, src + 1, 1)
        np.cumsum(indptr_w, out=indptr_w)
        np.save(run_dir / f"worker{w}_ids.npy", ids)
        np.save(run_dir / f"worker{w}_indptr.npy", indptr_w)
        np.save(run_dir / f"worker{w}_indices.npy", dst)
        cache_ids = np.asarray(sorted(caches[w]), dtype=np.int64)
        np.save(run_dir / f"worker{w}_cache_ids.npy", cache_ids)
        np.save(run_dir / f"worker{w}_cache_feat.npy",
                data["features"][cache_ids])
        seeds = np.asarray(sorted(partitions[w]), dtype=np.int64)
        np.save(run_dir / f"worker{w}_seeds.npy", seeds)
        feas = np.array([int(v) in replicas[w] and
                         _ball_covers(data, int(v), replicas[w]) for v in seeds])
        np.save(run_dir / f"worker{w}_feasible.npy", feas)


def _ball_covers(data, v, replica, depth=2):
    """Check N_depth(v) subset of replica (same semantics as placement)."""
    seen = {v}
    frontier = [v]
    for _ in range(depth):
        nxt = []
        for node in frontier:
            for nb in data["indices"][data["indptr"][node]:data["indptr"][node + 1]]:
                nb = int(nb)
                if nb not in seen:
                    if nb not in replica:
                        return False
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results_fleet")
    ap.add_argument("--modes", nargs="+",
                    default=["server-centric", "hybrid"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--fanout", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--target-mbps", type=float, default=100.0)
    ap.add_argument("--worker-cores", type=int, default=2)
    ap.add_argument("--server-cores", type=int, default=8)
    ap.add_argument("--worker-as-gb", type=float, default=6.0)
    ap.add_argument("--cache-rounds", type=int, default=16)
    ap.add_argument("--eval-seeds", type=int, default=4096)
    ap.add_argument("--train-limit", type=int, default=0,
                    help="cap training seeds (0 = official split)")
    ap.add_argument("--port", type=int, default=7790)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.eval_seeds = 2, 512
        args.train_limit = 2048
        args.seeds = [1]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("== loading ogbn-arxiv ==", flush=True)
    data = load_ogbn_arxiv(Path(args.data_root) if args.data_root else None)
    meta = data["meta"]
    print(json.dumps(meta, indent=1), flush=True)

    train_idx = data["train_idx"]
    if args.train_limit:
        train_idx = train_idx[:args.train_limit]

    rows = []
    for run_seed in args.seeds:
        print(f"\n===== seed {run_seed}: placement =====", flush=True)
        t0 = time.perf_counter()
        partitions, replicas, pm = place(
            data["indptr"], data["indices"], data["degrees"],
            train_idx, workers=args.workers, seed=run_seed)
        sampler = NeighborSampler(data["indptr"], data["indices"],
                                  fanout=args.fanout)
        caches, cm = build_caches(
            sampler, partitions, replicas, meta["vertices"],
            workers=args.workers, rounds=args.cache_rounds, seed=run_seed)
        t_pre = time.perf_counter() - t0
        pm.update(cm)
        pm["T_pre_total_s"] = t_pre
        print(f"placement: fallbacks={pm['fallbacks']} "
              f"repl={pm['replication_factor']:.3f} "
              f"cache_union={pm['cache_union_size']} T_pre={t_pre:.1f}s",
              flush=True)

        for mode in args.modes:
            run_dir = out / f"run_s{run_seed}_{mode}"
            if run_dir.exists():
                shutil.rmtree(run_dir)
            build_worker_artifacts(run_dir, data, partitions, replicas,
                                   caches, None, args.workers)
            # train_idx for this run (same for both modes)
            np.save(run_dir / "train_idx.npy", train_idx)

            manifest = {
                "run_dir": str(run_dir.resolve()),
                "mode": mode, "run_seed": run_seed,
                "workers": args.workers, "fanout": args.fanout,
                "hidden": args.hidden, "classes": meta["num_classes"],
                "feature_dim": meta["feature_dim"],
                "batch_size": args.batch_size, "lr": args.lr,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs, "eval_seeds": args.eval_seeds,
                "target_mbps": args.target_mbps,
                "host": "127.0.0.1", "port": args.port,
            }
            (run_dir / "server_manifest.json").write_text(
                json.dumps(manifest, indent=1))

            procs = []
            server_cfg = dict(manifest)
            server_cfg["cores"] = list(range(args.server_cores))
            (run_dir / "server_cfg.json").write_text(
                json.dumps(server_cfg, indent=1))
            env = dict(os.environ, OMP_NUM_THREADS="1",
                       KMP_DUPLICATE_LIB_OK="TRUE")
            procs.append(subprocess.Popen(
                [sys.executable, str(HERE / "fleet_server.py"),
                 str(run_dir / "server_cfg.json")],
                cwd=str(HERE), env=env,
                stdout=open(run_dir / "server.log", "w"),
                stderr=subprocess.STDOUT))
            time.sleep(3.0)
            for w in range(args.workers):
                wcfg = dict(manifest)
                base = 16 + w * args.worker_cores
                wcfg["cores"] = list(range(base, base + args.worker_cores))
                wcfg["as_limit_gb"] = args.worker_as_gb
                wcfg["worker_id"] = w
                wcfg["server_host"] = "127.0.0.1"
                wcfg["server_port"] = args.port
                p = str(run_dir / f"worker{w}_cfg.json")
                Path(p).write_text(json.dumps(wcfg, indent=1))
                procs.append(subprocess.Popen(
                    [sys.executable, str(HERE / "fleet_worker.py"), p],
                    cwd=str(HERE), env=env,
                    stdout=open(run_dir / f"worker{w}.log", "w"),
                    stderr=subprocess.STDOUT))
            rc = procs[0].wait()
            for p in procs[1:]:
                p.wait(timeout=120)
            result = json.loads((run_dir / "server_result.json").read_text())
            n_h = sum(s.get("h_seeds", 0) for s in
                      (result.get("worker_stats") or {}).values())
            n_s = sum(s.get("s_seeds", 0) for s in
                      (result.get("worker_stats") or {}).values())
            data_s = sum(s.get("data_s", 0.0) for s in
                         (result.get("worker_stats") or {}).values())
            comp_s = sum(s.get("compute_s", 0.0) for s in
                         (result.get("worker_stats") or {}).values())
            row = {
                "experiment": "emulated_fleet", "mode": mode,
                "seed": run_seed, "epochs": args.epochs,
                "best_val": result["best_val"],
                "test_acc": result["test_acc"],
                "ttt_epoch": result["ttt_epoch"],
                "ttt_train_s": result["ttt_train_s"],
                "ttt_total_s": (result["ttt_train_s"] + t_pre)
                if result["ttt_train_s"] is not None else None,
                "T_pre_s": t_pre,
                "bytes_feature": result["bytes_feature_total"],
                "bytes_sbatch": result["bytes_sbatch_total"],
                "worker_data_s": data_s, "worker_compute_s": comp_s,
                "fallbacks": pm["fallbacks"],
                "replication": pm["replication_factor"],
                "h_seed_epochs": n_h, "s_seed_epochs": n_s,
                "target_mbps": args.target_mbps,
            }
            rows.append(row)
            print(json.dumps(row, indent=1), flush=True)
            # next mode reuses the port
            time.sleep(2.0)

    import csv
    fields = list(rows[0].keys())
    with (out / "fleet_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out / 'fleet_results.csv'} ({len(rows)} rows)",
          flush=True)


if __name__ == "__main__":
    main()
