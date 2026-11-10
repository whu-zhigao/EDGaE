#!/usr/bin/env python3
"""Redraw the EDGaE architecture figure in the original paper's visual style.

The original figure (icde-fig-architecture.pdf) was actor-based: a server
pre-training pipeline on the left, edge workers on the right, parameter and
feature servers at the bottom, connected by semantically labelled data-flow
arrows (ship once / gradients / misses only). This script keeps that visual
language and upgrades the content to the governed three-plane system:

  - top strip  : placement governor pipeline (control plane) -> manifest
  - left       : server data plane (authoritative graph, residual service,
                 fallback sampler for mode S)
  - right      : edge workers with S/H/E mode-aware executors
  - bottom     : parameter server; arrows for ship-once, misses-only,
                 gradients, and S-mode batches

Output: paper/fig/architecture-governor.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG_PATH = HERE.parent / "fig" / "architecture-governor.pdf"

BLUE, GREEN, ORANGE, MAGENTA, GRAY = ("#0072B2", "#009E73", "#E69F00",
                                      "#CC79A7", "#4A4A4A")
LBLUE = "#DCEBF7"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})

fig, ax = plt.subplots(figsize=(3.5, 2.75))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")


def box(x, y, w, h, ec, label, fs=5.4, fc="white", lw=1.0, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.25, weight=weight)


def arr(x1, y1, x2, y2, ls="-", color=GRAY, lw=0.9, text=None,
        tx=0.0, ty=0.0, fs=5.0, ha="center"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, lw=lw, ls=ls, color=color,
                                 zorder=1, shrinkA=1.5, shrinkB=1.5))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha=ha,
                fontsize=fs, color=color, zorder=4)


# ======================= Control plane: governor pipeline ==================
ax.text(5.0, 9.78, "Control plane — placement governor (pre-training)",
        ha="center", fontsize=6.2, weight="bold", color=BLUE)
gov = [
    (0.15, "Profile\n$\\alpha_i, q_i$"),
    (2.55, "Cost model\n$\\mathcal{A}_m(H,B)$"),
    (4.95, "Feasibility\n$m_i,\\ \\rho_{\\max}$"),
    (7.35, "Mode select\n$m^\\star(v,i)$"),
]
for x, label in gov:
    box(x, 8.55, 2.3, 0.95, BLUE, label, fs=5.2)
for i in range(3):
    arr(2.45 + 2.4 * i, 9.02, 2.55 + 2.4 * i, 9.02, color=BLUE)

# Manifest bar
box(0.15, 7.35, 9.5, 0.75, BLUE,
    "Placement manifest  $\\mathcal{M}=\\{P_i, R_i, C_i, m(v,i), F_i\\}$  — versioned, auditable",
    fs=5.4, fc=LBLUE, lw=1.2)
arr(8.5, 8.55, 8.5, 8.12, color=BLUE)          # mode select -> manifest
arr(1.4, 7.35, 1.4, 6.9, color=BLUE)           # manifest -> server
arr(7.6, 7.35, 7.6, 6.9, color=BLUE)           # manifest -> workers

# ======================= Server (data plane, left) =========================
ax.text(1.5, 6.72, "Server (data plane)", ha="center", fontsize=6.2,
        weight="bold", color=BLUE)
box(0.15, 5.35, 2.5, 1.05, BLUE,
    "Authoritative graph\ncold features", fs=5.4)
box(0.15, 3.95, 2.5, 1.05, BLUE,
    "Residual feature\nservice", fs=5.4)
box(0.15, 2.55, 2.5, 1.05, BLUE,
    "Fallback sampler\n(mode $S$)", fs=5.4)
arr(1.4, 5.35, 1.4, 5.0, color=BLUE)

# ======================= Edge workers (right) ==============================
ax.text(7.55, 6.72, "Edge workers (execution plane)", ha="center",
        fontsize=6.2, weight="bold", color=GREEN)
worker_y = (5.35, 3.95, 2.55)
for i, y in enumerate(worker_y):
    # Worker box with header line (name + mode chip) and data-line below.
    ax.add_patch(FancyBboxPatch((5.45, y), 4.2, 1.05,
                                boxstyle="round,pad=0.08", fc="white",
                                ec=GREEN, lw=1.0, zorder=2))
    ax.text(6.45, y + 0.68, f"Worker {i + 1}", ha="center", va="center",
            fontsize=5.4, zorder=3)
    chip = "$S/H/E$" if i == 0 else "$H$"
    box(8.55, y + 0.52, 1.0, 0.40, ORANGE, chip, fs=4.6, lw=0.8)
    ax.text(7.55, y + 0.28, "$R_i$ (read-only) · $C_i$ · local sampler",
            ha="center", va="center", fontsize=4.6, zorder=3)

# Ship-once: server graph -> each worker
for y in worker_y:
    arr(2.65, 5.85, 5.45, y + 0.52, color=ORANGE)
ax.text(4.05, 6.32, "ship once", fontsize=5.0, color=ORANGE, ha="center")

# Misses only: worker 1 -> residual service (dotted, reverse direction)
arr(5.45, 5.60, 2.65, 4.55, ls=":", color=BLUE)
ax.text(4.05, 5.05, "misses only", fontsize=5.0, color=BLUE, ha="center")

# Mode-S batches: fallback sampler -> worker 3
arr(2.65, 3.05, 5.45, 3.05, ls="--", color=BLUE)
ax.text(4.05, 3.22, "$S$ batches", fontsize=5.0, color=BLUE, ha="center")

# ======================= Parameter server (bottom) =========================
box(3.4, 0.55, 3.2, 0.95, MAGENTA, "Parameter server", fs=5.8)
for y in worker_y:
    arr(7.55, y, 6.35, 1.5, ls="--", color=MAGENTA, lw=0.8)
ax.text(7.35, 1.28, "gradients", fontsize=5.0, color=MAGENTA, ha="center")

fig.tight_layout(pad=0.15)
fig.savefig(FIG_PATH, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"wrote {FIG_PATH}")
