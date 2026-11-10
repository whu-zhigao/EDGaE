#!/usr/bin/env python3
"""Mode cost surface with break-even thresholds and E/S optimal regions.

Provenance: constants are the ogbn-arxiv measured values from
results_cloud/results/icde-v3-mode-system-results.csv (preprocess block,
K=16, five-seed means). D_E follows the stable-schedule pricing of
Definition 1 (constant one-time volume = W_E)."""
from __future__ import annotations
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG = Path(__file__).resolve().parent.parent.parent / "paper" / "fig" / "mode-surface.pdf"
FIG.parent.mkdir(parents=True, exist_ok=True)

# ogbn-arxiv measured constants (results_cloud preprocess block, K=16):
# T_pre = 13.11 s, C_S = 7.31 MB, C_H = 3.99 MB, Ship(D_H) = 30.6 MB,
# W_E (D_E, stable-schedule pricing) = 5.32 MB.  D_E is CONSTANT in H
# (one evaluation pass), matching Definition 1 in the paper.
T_PRE = 13.11; SHIP_H = 30.6; W_E = 5.32
C_S = 7.31; C_H = 3.99; H_DEF = 100

def cost_S(B, H): return H * C_S / B
def cost_H(B, H): return T_PRE + SHIP_H / B + H * C_H / B
def cost_E(B, H): return T_PRE + W_E / B

def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ── Panel (a): Cost curves at H=100, with break-even vertical lines ──
    ax = axes[0]
    B = np.linspace(0.5, 120, 300)
    H = H_DEF
    aS = cost_S(B, H); aH = cost_H(B, H); aE = cost_E(B, H)
    ax.plot(B, aS, color="#D55E00", lw=1.8, label="$\\mathcal{A}_S$ (Server)")
    ax.plot(B, aH, color="#0072B2", lw=1.8, label="$\\mathcal{A}_H$ (Hybrid)")
    ax.plot(B, aE, color="#009E73", lw=1.8, ls="--", label="$\\mathcal{A}_E$ (Edge)")

    # Break-even vertical lines
    delta = C_S - C_H
    # A_S = A_H → B_HS = (H(C_S-C_H) - D_H) / T_pre   (Corollary 2(i))
    B_HS = (H * delta - SHIP_H) / T_PRE
    # A_S = A_E → B_SE = (H*C_S - D_E) / T_pre        (Corollary 2(iv))
    # A_E - A_H = (D_E - D_H - H*C_H)/B < 0 always here (D_E < D_H), so E
    # dominates H at every B: two-segment E→S envelope (Corollary 2(ii)).
    B_SE = (H * C_S - W_E) / T_PRE

    for b_val, color, label in [(B_SE, "#009E73", "$B_{SE}$"), (B_HS, "#D55E00", "$B_{HS}$")]:
        if 0 < b_val < 120:
            ax.axvline(b_val, color=color, ls=":", lw=1.0, alpha=0.7, zorder=2)
            ax.text(b_val + 0.5, 1400, label, fontsize=13.5,
                    color=color, va="top", ha="left")

    ax.set_xlim(0, 120); ax.set_ylim(0, 1500)
    ax.set_xlabel("Effective bandwidth (MB/s)", fontsize=14.5)
    ax.set_ylabel("Amortized cost (s)", fontsize=14.5)
    ax.set_title(f"(a) Mode cost at $H{{=}}{H}$", fontsize=14.5, pad=4)
    ax.legend(fontsize=9.5, frameon=True, loc="upper right",
              bbox_to_anchor=(0.99, 0.84), handlelength=1.6,
              handletextpad=0.4, labelspacing=0.3, framealpha=0.95)
    ax.tick_params(labelsize=13.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0); ax.set_axisbelow(True)

    # ── Panel (b): Optimal mode region (E vs S, with H boundary marked) ──
    ax2 = axes[1]
    # grid starts at 1e-6 (not 0) to avoid division by zero while placing the
    # filled regions flush against the xlim/ylim origin (no white edge margin)
    Bs = np.linspace(1e-6, 120, 200); Hs = np.linspace(1e-6, 150, 200)
    BB, HH = np.meshgrid(Bs, Hs)
    cS = cost_S(BB, HH); cH = cost_H(BB, HH); cE = cost_E(BB, HH)
    best_idx = np.argmin(np.stack([cS, cH, cE]), axis=0)  # 0=S, 1=H, 2=E

    COL_S = "#D55E00"; COL_H = "#0072B2"; COL_E = "#009E73"
    ax2.contourf(BB, HH, best_idx, levels=[-0.5, 0.5, 1.5, 2.5],
                 colors=[COL_S, COL_H, COL_E], alpha=0.9, zorder=1)
    ax2.contour(BB, HH, best_idx.astype(float), levels=[0.5, 1.5],
                colors="white", linewidths=1.5, zorder=2)

    ax2.set_xlabel("Bandwidth (MB/s)", fontsize=14.5)
    ax2.set_ylabel("Horizon $H$ (epochs)", fontsize=14.5)
    ax2.set_title("(b) Optimal mode region", fontsize=14.5, pad=4)
    ax2.tick_params(labelsize=13.5)
    ax2.set_xlim(0, 120); ax2.set_ylim(0, 150)

    # Region labels at fixed positions
    ax2.text(100, 25, "$S$", fontsize=16.5, color="white", weight="bold",
             ha="center", va="center", zorder=3)
    ax2.text(30, 110, "$E$", fontsize=16.5, color="white", weight="bold",
             ha="center", va="center", zorder=3)

    # Annotate H's absence


    # Legend
    patches = [
        mpatches.Patch(facecolor=COL_S, alpha=0.9, label="$S$ optimal"),
        mpatches.Patch(facecolor=COL_E, alpha=0.9, label="$E$ optimal"),
    ]
    ax2.legend(handles=patches, fontsize=9.5, frameon=True,
              loc="lower right", bbox_to_anchor=(1.0, 0.30), framealpha=0.9,
              handlelength=1.4, handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIG}")
    for idx, name in [(0,"S"),(1,"H"),(2,"E")]:
        frac = (best_idx == idx).mean() * 100
        print(f"  {name}: {frac:.0f}%")

if __name__ == "__main__":
    main()
