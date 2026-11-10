#!/usr/bin/env python3
"""Driver for E5: native mode-transition check (paced S vs H data paths).

Runs placement + cache construction once, then times both feature paths at
emulated 10/100/1000 Mbps and native PCIe. Emits results/mode_transition.csv
and a summary with the native break-even horizons.

Usage:
    python run_mode_transition.py [--smoke] [--seeds 1 2 3]
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
from edgae_gpu.mode_transition import run_mode_transition
from edgae_gpu.placement import build_caches, place
from edgae_gpu.sampling import NeighborSampler


BANDWIDTHS = [10, 100, 1000, None]  # Mbps; None = native PCIe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--limit-train", type=int, default=4096)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--fanout", type=int, default=2)
    p.add_argument("--cache-rounds", type=int, default=16)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--bandwidths", nargs="+", type=int, default=None,
                   help="paced bandwidths in Mbps; native PCIe is always "
                        "appended (e.g., --bandwidths 10 50 100 500 1000)")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    limit_train = 512 if args.smoke else args.limit_train
    seeds = [1] if args.smoke else args.seeds
    if args.bandwidths:
        bandwidths = args.bandwidths + [None]
    else:
        bandwidths = [100, None] if args.smoke else BANDWIDTHS

    env = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "python": platform.python_version(),
        "experiment": "E5_mode_transition",
        "measurement_utc": datetime.now(timezone.utc).isoformat(),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(0)
    (out_dir / "env_e5.json").write_text(json.dumps(env, indent=1))
    print(json.dumps(env, indent=1))

    print("== loading ogbn-arxiv ==")
    data = load_ogbn_arxiv(args.data_root) if args.data_root else load_ogbn_arxiv()
    print(json.dumps(data["meta"], indent=1))

    rows = []
    summary_rows = []
    for seed in seeds:
        print(f"\n===== seed {seed} =====")
        place_seeds = data["train_idx"].numpy()[:512] if args.smoke \
            else data["train_idx"].numpy()[:args.limit_train]
        t0 = time.perf_counter()
        partitions, replicas, pmetrics = place(
            data["indptr"], data["indices"], data["degrees"],
            place_seeds, workers=args.workers, seed=seed)
        sampler_once = NeighborSampler(
            data["indptr"], data["indices"], data["degrees"],
            fanout=args.fanout, uniform=False, seed=seed * 104729 + 11)
        caches, cmetrics = build_caches(
            sampler_once, partitions, replicas, data["degrees"],
            data["meta"]["vertices"], workers=args.workers,
            rounds=(2 if args.smoke else args.cache_rounds), seed=seed)
        T_pre = time.perf_counter() - t0
        print(f"placement: fallbacks={pmetrics['fallbacks']} "
              f"repl={pmetrics['replication_factor']:.3f} T_pre={T_pre:.1f}s")

        sampler = NeighborSampler(
            data["indptr"], data["indices"], data["degrees"],
            fanout=args.fanout, uniform=False, seed=seed * 104729 + 11)
        results = run_mode_transition(
            data, device, sampler, partitions, caches,
            bandwidths, epochs=args.epochs,
            limit_train=limit_train, seed=seed)

        # Pair S/H rows per bandwidth and compute the native break-even.
        by_bw = {}
        for r in results:
            by_bw.setdefault(r["bandwidth_mbps"], {})[r["mode"]] = r
        for bw, pair in by_bw.items():
            if "S" not in pair or "H" not in pair:
                continue
            C_S = pair["S"]["bytes_per_epoch"]
            C_H = pair["H"]["bytes_per_epoch"]
            delta = C_S - C_H
            if bw is None or bw == "native":
                # Native PCIe: use achieved S throughput as the fabric rate.
                B = (pair["S"]["bytes_per_epoch"] / pair["S"]["data_path_s_per_epoch"]) * 1e6
            else:
                B = bw * 1e6 / 8.0
            if delta > 0:
                H_star = (T_pre * B) / delta
            else:
                H_star = float("inf")
            summary_rows.append({
                "seed": seed, "bandwidth_mbps": bw,
                "C_S_bytes": C_S, "C_H_bytes": C_H, "delta_bytes": delta,
                "S_s_per_epoch": pair["S"]["data_path_s_per_epoch"],
                "H_s_per_epoch": pair["H"]["data_path_s_per_epoch"],
                "T_pre_s": T_pre, "native_H_star_epochs": H_star,
                "fallbacks": pmetrics["fallbacks"],
                "replication_factor": pmetrics["replication_factor"],
            })
            for r in (pair["S"], pair["H"]):
                rows.append({"experiment": "E5_mode_transition", "seed": seed,
                             "T_pre_s": T_pre, **r})

    csv_path = out_dir / "mode_transition.csv"
    fields = ["experiment", "seed", "bandwidth_mbps", "mode", "epochs",
              "train_seeds", "data_path_s_per_epoch", "bytes_per_epoch",
              "achieved_MBps", "bytes_total", "T_pre_s"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    summary_path = out_dir / "mode_transition_summary.csv"
    sfields = ["seed", "bandwidth_mbps", "C_S_bytes", "C_H_bytes",
               "delta_bytes", "S_s_per_epoch", "H_s_per_epoch", "T_pre_s",
               "native_H_star_epochs", "fallbacks", "replication_factor"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nwrote {csv_path} ({len(rows)} rows) and {summary_path} "
          f"({len(summary_rows)} rows)")


if __name__ == "__main__":
    main()
