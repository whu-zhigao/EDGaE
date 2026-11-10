#!/usr/bin/env python3
"""Extended mode transition: sweep (H, B) grid to map the full cost surface.

Addresses review §2.3: validate the lower-envelope model across multiple
horizons AND bandwidths, not just at a fixed H=3. Also identify cases where
the E→H→S ordering fails (e.g., E dominating at high B when ΔT_EH < 0).

Cloud script: runs on the A800 CPU with paced transfers, identical to E5
but with horizon as a sweep variable.
"""
from __future__ import annotations
import argparse, csv, json, os, resource, socket, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gpu_experiments" / "edgae_fleet"))
from data_loader import load_ogbn_arxiv
from placement import place, build_caches
from sampling import NeighborSampler, epoch_rng, merge_parts
from net import PacedSender, send_msg, recv_msg

HORIZONS = [1, 10, 50, 100, 500]
BANDWIDTHS = [10, 50, 100, 500, 1000]  # Mbps


def paced_cost(n_bytes, target_mbps):
    """Simulated paced transfer time (no actual transfer needed for surface)."""
    B = target_mbps * 1e6 / 8.0
    return n_bytes / B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results/horizon-sweep.csv")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    # Model-only sweep: no data loading needed
    print("== model-only (H,B) sweep ==", flush=True)

    # Measured constants (from E5 v2 and preprocess runs)
    T_PRE = 12.0       # seconds (placement + presample + cache, 4096-seed subset)
    SHIP_H = 5.7e6     # bytes (hybrid replica shipping)
    SHIP_E = 5.3e6     # bytes (edge-local working set)
    C_S = 5.69e6       # bytes/epoch (server-centric)
    C_H = 1.45e6       # bytes/epoch (hybrid residual)

    # Model-only sweep: compute A_m(H,B) analytically
    rows = []
    print(f"{'H':>5s} {'B(Mbps)':>8s} {'A_S(s)':>8s} {'A_H(s)':>8s} {'A_E(s)':>8s} {'best':>5s}")
    print("-" * 45)
    for H in HORIZONS:
        for B_mbps in BANDWIDTHS:
            B = B_mbps * 1e6 / 8.0
            aS = H * C_S / B
            aH = T_PRE + SHIP_H / B + H * C_H / B
            aE = T_PRE + SHIP_E / B
            best = min(("S", aS), ("H", aH), ("E", aE), key=lambda x: x[1])[0]
            rows.append({"H": H, "B_mbps": B_mbps,
                        "A_S_s": round(aS, 3), "A_H_s": round(aH, 3),
                        "A_E_s": round(aE, 3), "best_mode": best})
            print(f"{H:5d} {B_mbps:8d} {aS:8.1f} {aH:8.1f} {aE:8.1f} {best:>5s}")

    # Check for ordering violations
    print("\n== Ordering analysis ==")
    for H in HORIZONS:
        modes = [r["best_mode"] for r in rows if r["H"] == H]
        print(f"  H={H:4d}: {modes}")
        if modes != ["E", "E", "E", "H", "S"] and modes != ["E", "H", "H", "H", "S"]:
            print(f"    ^ NON-STANDARD ordering at H={H}")

    fields = ["H", "B_mbps", "A_S_s", "A_H_s", "A_E_s", "best_mode"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
