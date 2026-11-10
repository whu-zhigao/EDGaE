#!/usr/bin/env python3
"""Orchestrate EDGaE native GPU experiments (E1-E4) and emit audit-ready CSV.

Quick start on the target image:
    python run_all.py --smoke        # ~2-4 min sanity check
    python run_all.py                # full suite, ~1-2 h on one A100/V100
    python run_all.py --sensitivity  # adds fan-out axis at the native level
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from edgae_gpu.data import load_ogbn_arxiv
from edgae_gpu.placement import build_caches, place
from edgae_gpu.sampling import NeighborSampler
from edgae_gpu.train import train_fullbatch_gcn, train_minibatch_sage


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto", help="auto|cuda|cpu|mps")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--epochs-gcn", type=int, default=200)
    p.add_argument("--epochs-sage", type=int, default=60)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--uniform", action="store_true",
                   help="uniform sampling (default: inverse-degree)")
    p.add_argument("--cache-rounds", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sensitivity", action="store_true",
                   help="add EDGaE runs at fan-out 4 and 8 (E4)")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run to validate the pipeline end to end")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def main():
    args = parse_args()
    device = pick_device(args.device)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    smoke = args.smoke
    epochs_gcn = 4 if smoke else args.epochs_gcn
    epochs_sage = 2 if smoke else args.epochs_sage
    limit_train = 8192 if smoke else None
    seeds = [1] if smoke else args.seeds

    print("== environment ==")
    env = {
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device), "python": platform.python_version(),
        "machine": platform.machine(),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(json.dumps(env, indent=1))
    (out_dir / "env.json").write_text(json.dumps(env, indent=1))

    print("== loading ogbn-arxiv (first run parses ~80 MB gz) ==")
    data = load_ogbn_arxiv(args.data_root) if args.data_root else load_ogbn_arxiv()
    print(json.dumps(data["meta"], indent=1))

    rows = []
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        # ---- E1: full-batch GCN reference --------------------------------
        gcn = train_fullbatch_gcn(data, device, epochs=epochs_gcn, seed=seed)
        rows.append(dict(experiment="E1_fullbatch_gcn", seed=seed, mode="gcn",
                         epochs=epochs_gcn,
                         best_val=gcn["best"]["val"], test_acc=gcn["best"]["test"],
                         T_pre_s=0.0, **_flat_times(gcn)))

        # ---- placement + caches (measured preprocessing) -----------------
        t0 = time.perf_counter()
        place_seeds = data["train_idx"].numpy()[:4096] if smoke \
            else data["train_idx"].numpy()
        partitions, replicas, pmetrics = place(
            data["indptr"], data["indices"], data["degrees"],
            place_seeds, workers=args.workers, seed=seed)
        sampler_once = NeighborSampler(
            data["indptr"], data["indices"], data["degrees"],
            fanout=args.fanout, uniform=args.uniform, seed=seed * 104729 + 11)
        caches, cmetrics = build_caches(
            sampler_once, partitions, replicas, data["degrees"],
            data["meta"]["vertices"], workers=args.workers,
            rounds=(2 if smoke else args.cache_rounds), seed=seed)
        T_pre = time.perf_counter() - t0
        print(f"placement: fallbacks={pmetrics['fallbacks']} "
              f"repl={pmetrics['replication_factor']:.3f} "
              f"imb={pmetrics['load_imbalance']:.3f} T_pre={T_pre:.1f}s")

        # ---- E2/E3: server vs EDGaE feature paths ------------------------
        results = {}
        for mode in ("server", "edgae"):
            sampler = NeighborSampler(
                data["indptr"], data["indices"], data["degrees"],
                fanout=args.fanout, uniform=args.uniform,
                seed=seed * 104729 + 11)
            res = train_minibatch_sage(
                data, mode, device, sampler, partitions, caches,
                epochs=epochs_sage, seed=seed, batch_size=args.batch_size,
                lr=args.lr, limit_train=limit_train,
                feature_dim=data["meta"]["feature_dim"])
            results[mode] = res
            ttt = res["time_to_target"] or {}
            rows.append(dict(
                experiment="E2_E3_sage", seed=seed, mode=mode,
                epochs=epochs_sage, best_val=res["best"]["val"],
                test_acc=res["best"]["test"], T_pre_s=(T_pre if mode == "edgae" else 0.0),
                data_path_s=round(res["data_path_s"], 3),
                compute_s=round(res["compute_s"], 3),
                miss_bytes=res["miss_bytes"],
                ttt_epoch=ttt.get("epoch"),
                ttt_train_s=round(ttt.get("train_s", 0.0), 3) if ttt else None,
                ttt_total_s=round(T_pre + ttt.get("train_s", 0.0), 3) if ttt else None,
                fallbacks=pmetrics["fallbacks"],
                load_imbalance=round(pmetrics["load_imbalance"], 4),
                replication=round(pmetrics["replication_factor"], 4),
                cache_union=cmetrics["cache_union_size"], **_env_suffix(env)))

        # Proposition-4 parity check: identical sampling => identical accuracy
        parity = abs(results["server"]["best"]["val"]
                     - results["edgae"]["best"]["val"])
        print(f"P4 parity |val_server - val_edgae| = {parity:.6f}")
        speedup = (results["server"]["data_path_s"]
                   / max(results["edgae"]["data_path_s"], 1e-9))
        print(f"data-path speedup (server/edgae) = {speedup:.2f}x; "
              f"miss bytes {results['edgae']['miss_bytes']:,}")

        # ---- E4: native fan-out sensitivity -------------------------------
        if args.sensitivity and not smoke:
            for fanout in (4, 8):
                sampler = NeighborSampler(
                    data["indptr"], data["indices"], data["degrees"],
                    fanout=fanout, uniform=args.uniform,
                    seed=seed * 104729 + 11)
                res = train_minibatch_sage(
                    data, "edgae", device, sampler, partitions, caches,
                    epochs=max(10, epochs_sage // 2), seed=seed,
                    batch_size=args.batch_size, lr=args.lr,
                    feature_dim=data["meta"]["feature_dim"])
                rows.append(dict(
                    experiment="E4_fanout_sensitivity", seed=seed,
                    mode=f"edgae_f{fanout}", epochs=max(10, epochs_sage // 2),
                    best_val=res["best"]["val"], test_acc=res["best"]["test"],
                    T_pre_s=T_pre, data_path_s=round(res["data_path_s"], 3),
                    compute_s=round(res["compute_s"], 3),
                    miss_bytes=res["miss_bytes"],
                    fallbacks=pmetrics["fallbacks"],
                    replication=round(pmetrics["replication_factor"], 4), **_env_suffix(env)))

    # ---- write outputs ---------------------------------------------------
    fields = sorted({k for r in rows for k in r})
    with (out_dir / "gpu_native_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    summary = {
        "rows": len(rows), "env": env,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "paper_mapping": {
            "E1": "native-task reference accuracy (Section VII gap)",
            "E2/E3": "feature-path effect + preprocessing-inclusive time-to-target "
                     "(Table VI native column; Proposition 4 parity)",
            "E4": "native fan-out sensitivity (Section VI-E)",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out_dir}/gpu_native_results.csv ({len(rows)} rows)")


def _flat_times(res):
    return {"data_path_s": 0.0, "compute_s": round(res["total_train_s"], 3)}


def _env_suffix(env):
    return {"device": env["device"], "torch": env["torch"]}


if __name__ == "__main__":
    main()
