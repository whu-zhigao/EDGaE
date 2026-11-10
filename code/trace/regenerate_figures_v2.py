#!/usr/bin/env python3
"""Regenerate all five paper figures in one unified publication style.

Style system (applied to every figure):
  - Okabe-Ito colorblind-safe palette, consistent per topology/policy;
  - open axes (no top/right spines), subtle y-grid;
  - legends placed OUTSIDE the data area (top strip or bottom strip);
  - error bars without end caps (thin, neutral gray);
  - panel titles as left-aligned (a)/(b) labels.

Also exports every plotted number to ICDE/figsource/figure_data_master.xlsx.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "www/icde2027-v2-summary.json").read_text())
FIG = ROOT / "ICDE/fig"
FIG.mkdir(exist_ok=True)

# ---------- unified design system ----------
TOPO = ["web-google", "amazon0601", "wiki-talk", "ogbn-arxiv",
        "amazon0601-full", "web-google-full", "wiki-talk-full"]
SHORT = {"web-google": "web-G 12k", "amazon0601": "amazon 12k",
         "wiki-talk": "wiki 12k", "ogbn-arxiv": "ogbn-arxiv",
         "amazon0601-full": "amazon full", "web-google-full": "web-G full",
         "wiki-talk-full": "wiki full"}
C = {"web-google": "#0072B2", "amazon0601": "#E69F00", "wiki-talk": "#009E73",
     "ogbn-arxiv": "#CC79A7", "amazon0601-full": "#D55E00",
     "web-google-full": "#56B4E9", "wiki-talk-full": "#6B6B6B"}
POL = ["random", "even", "METIS-like", "PaGraph-style", "EDGaE"]
POLL = {"random": "Random", "even": "Even", "METIS-like": "METIS-like",
        "PaGraph-style": "PaGraph-style", "EDGaE": "EDGaE"}
PC = {"random": "#B8B8B8", "even": "#56B4E9", "METIS-like": "#009E73",
      "PaGraph-style": "#E69F00", "EDGaE": "#D55E00"}
GRID = dict(color="#E6E6E6", lw=0.6, zorder=0)

plt.rcParams.update({
    "font.size": 8, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "figure.dpi": 200})

def m(b, d, c, f):
    return S[f"{b}|{d}|{c}"][f]

def style(ax):
    ax.grid(axis="y", **GRID)
    ax.set_axisbelow(True)

# ================= Fig 2: cache stability (two panels, legend below) =======
K = [1, 2, 4, 8, 16]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.85))
for d in TOPO:
    for ax, f in ((axes[0], "cache_hit_rate"), (axes[1], "topk_jaccard")):
        xs, ys, lo, hi = [], [], [], []
        for k in K + ["full"]:
            c = "K=full-trace" if k == "full" else f"K={k}"
            e = m("cache-stability", d, c, f)
            xs.append(17 if k == "full" else k)
            ys.append(e["mean"]); lo.append(e["ci95_low"]); hi.append(e["ci95_high"])
        ax.plot(xs, ys, "-o", color=C[d], lw=1.4, ms=3.2,
                markeredgecolor="white", markeredgewidth=0.5, label=SHORT[d])
        ax.fill_between(xs, lo, hi, color=C[d], alpha=0.13, lw=0)
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels([1, 2, 4, 8, 16])
    ax.set_xlim(0.85, 19.5); style(ax); ax.set_xlabel("$K$")
axes[0].set_ylabel("Cache hit rate")
axes[1].set_ylabel("Top-$k$ Jaccard")
axes[0].set_title("(a) Effectiveness vs. $K$", loc="left")
axes[1].set_title("(b) Hot-set recovery vs. $K$", loc="left")
fig.legend(handles=[plt.Line2D([], [], color=C[d], marker="o", ms=3.5, lw=1.4,
                               label=SHORT[d]) for d in TOPO],
           ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.16),
           columnspacing=1.2, handletextpad=0.4)
fig.tight_layout()
fig.savefig(FIG / "icde-v2-cache-stability.pdf", bbox_inches="tight")
plt.close(fig)

# ============ Fig 3: partition policies (4 panels, legend top strip) ========
fig, axes = plt.subplots(2, 2, figsize=(7.0, 2.85))
def bars(ax, datasets, field, ylab, fmt_mb=False):
    w = 0.16
    for j, p in enumerate(POL):
        xs, ys, el, eh = [], [], [], []
        for i, d in enumerate(datasets):
            e = m("partition-policy", d, p, field)
            xs.append(i + (j - 2) * w)
            v = e["mean"] / 1e6 if fmt_mb else e["mean"]
            ys.append(v)
            el.append(v - (e["ci95_low"] / 1e6 if fmt_mb else e["ci95_low"]))
            eh.append((e["ci95_high"] / 1e6 if fmt_mb else e["ci95_high"]) - v)
        ax.bar(xs, ys, width=w * 0.92, color=PC[p], label=POLL[p],
               yerr=[el, eh], capsize=0,
               error_kw={"ecolor": "#5A5A5A", "elinewidth": 0.8,
                         "alpha": 0.85})
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([SHORT[d] for d in datasets], fontsize=6.8)
    ax.set_ylabel(ylab, fontsize=7.5); style(ax)
sub12, subF = TOPO[:3] + [TOPO[3]], TOPO[4:]
bars(axes[0][0], sub12, "C_edg_bytes", "Residual MB", True)
axes[0][0].set_title("(a) Residual bytes, 12k / arxiv", loc="left")
bars(axes[0][1], subF, "C_edg_bytes", "Residual MB", True)
axes[0][1].set_title("(b) Residual bytes, full graphs", loc="left")
bars(axes[1][0], sub12, "replication_factor_vertices", "Vertex replication")
axes[1][0].set_title("(c) Replication, 12k / arxiv", loc="left")
bars(axes[1][1], subF, "replication_factor_vertices", "Vertex replication")
axes[1][1].set_title("(d) Replication, full graphs", loc="left")
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PC[p], label=POLL[p])
                    for p in POL],
           ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.045),
           columnspacing=1.6, handletextpad=0.5)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "icde-v2-partition.pdf", bbox_inches="tight")
plt.close(fig)

# ============ Fig 4: sampler sensitivity (legend below) =====================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.8))
CONF_F = ["f=2,d=2,non-uniform", "f=4,d=2,non-uniform", "f=8,d=2,non-uniform"]
CONF_D = ["f=2,d=1,non-uniform", "f=2,d=2,non-uniform", "f=2,d=3,non-uniform"]
for d in TOPO:
    for ax, confs, xs in ((axes[0], CONF_F, [2, 4, 8]), (axes[1], CONF_D, [1, 2, 3])):
        ys, lo, hi = [], [], []
        for c in confs:
            e = m("sampler-sensitivity", d, c, "C_edg_bytes")
            ys.append(e["mean"] / 1e6); lo.append(e["ci95_low"] / 1e6)
            hi.append(e["ci95_high"] / 1e6)
        ax.plot(xs, ys, "-s", color=C[d], lw=1.4, ms=3.2,
                markeredgecolor="white", markeredgewidth=0.5, label=SHORT[d])
        ax.fill_between(xs, lo, hi, color=C[d], alpha=0.13, lw=0)
axes[0].set_yscale("log"); axes[1].set_yscale("log")
axes[0].set_xticks([2, 4, 8]); axes[1].set_xticks([1, 2, 3])
axes[0].set_xlabel("fan-out $f$"); axes[1].set_xlabel("depth $L$")
axes[0].set_ylabel("Residual MB"); style(axes[0]); style(axes[1])
axes[0].set_title("(a) Fan-out axis ($L{=}2$)", loc="left")
axes[1].set_title("(b) Depth axis ($f{=}2$)", loc="left")
fig.legend(handles=[plt.Line2D([], [], color=C[d], marker="s", ms=3.5, lw=1.4,
                               label=SHORT[d]) for d in TOPO],
           ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.17),
           columnspacing=1.2, handletextpad=0.4)
fig.tight_layout()
fig.savefig(FIG / "icde-v2-sensitivity.pdf", bbox_inches="tight")
plt.close(fig)
print("Figs 2-4 regenerated")

# ============ Fig 1: architecture (unified palette) =========================
fig, ax = plt.subplots(figsize=(3.45, 1.75))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
BLUE, GREEN, ORANGE, MAGENTA, GRAY = ("#0072B2", "#009E73", "#E69F00",
                                      "#CC79A7", "#4A4A4A")

def box(x, y, w, h, ec, label, fs=6.5, fc="white", lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=fs, zorder=3, linespacing=1.25)

def arr(x1, y1, x2, y2, ls="-", color=GRAY, lw=1.0, text=None, dy=0.25):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8, lw=lw, ls=ls, color=color,
                                 zorder=1, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + dy, text, ha="center",
                fontsize=5.8, color=color)

ax.text(1.4, 9.55, "Server (pre-training)", ha="center", fontsize=7.0,
        weight="bold", color=BLUE)
box(0.2, 8.0, 2.4, 1.1, BLUE, "Profile workers\n$\\alpha_i$, $q_i$")
box(0.2, 6.5, 2.4, 1.1, BLUE, "Greedy partition\n$\\{P_i, R_i\\}$")
box(0.2, 5.0, 2.4, 1.1, BLUE, "Pilot sweep\nhotness, $K^\\star$")
box(0.2, 3.5, 2.4, 1.1, BLUE, "Build caches $\\{C_i\\}$")
for y in (8.0, 6.5, 5.0):
    arr(1.4, y, 1.4, y + 1.1)

ax.text(7.2, 9.55, "Edge workers (training)", ha="center", fontsize=7.0,
        weight="bold", color=GREEN)
for i, y in enumerate((6.9, 5.0, 3.1)):
    box(5.5, y, 3.4, 1.35, GREEN,
        f"Worker {i+1}\nlocal sampler | cache | trainer", fs=5.8)
    box(6.6, y + 0.78, 1.2, 0.5, ORANGE, "cache", fs=5.6, lw=0.9)
arr(2.6, 4.05, 5.5, 7.0, text="ship once", dy=0.3)
arr(2.6, 4.05, 5.5, 5.1)
arr(2.6, 4.05, 5.5, 3.2)
box(0.2, 0.7, 2.4, 1.1, MAGENTA, "Parameter\nserver")
for y in (6.9, 5.0, 3.1):
    arr(5.5, y + 0.65, 2.6, 1.25, ls="--", color=MAGENTA, lw=0.9)
ax.text(3.9, 1.0, "gradients", fontsize=5.8, color=MAGENTA)
box(8.35, 0.7, 1.5, 1.1, BLUE, "Feature\nserver")
for y in (6.9, 5.0, 3.1):
    arr(8.9, 1.8, 8.9, y, ls=":", color=BLUE, lw=0.9)
ax.text(9.0, 2.35, "misses\nonly", fontsize=5.6, color=BLUE, ha="center")
fig.tight_layout(pad=0.2)
fig.savefig(FIG / "icde-fig-architecture.pdf", bbox_inches="tight")
plt.close(fig)

# ============ Fig 5: physical relative hit rate (EDGaE = 1) =================
# Derived solely from the ratios stated in Section VII of the paper:
# EDGaE vs DGL-Degree: 1.37x (wiki-talk), 1.04x (web-google)
# EDGaE vs DGL-Random: 1.72x (wiki-talk), 1.99x (web-google)
RATIO = {  # dataset -> policy -> hit / EDGaE_hit
    "web-google": {"EDGaE": 1.00, "DGL-Degree": 1/1.04, "DGL-Random": 1/1.99},
    "wiki-talk":  {"EDGaE": 1.00, "DGL-Degree": 1/1.37, "DGL-Random": 1/1.72},
}
fig, ax = plt.subplots(figsize=(3.45, 1.75))
pol = ["DGL-Random", "DGL-Degree", "EDGaE"]
colr = {"EDGaE": "#D55E00", "DGL-Degree": "#0072B2", "DGL-Random": "#9A9A9A"}
w = 0.24
for j, p in enumerate(pol):
    xs = [i + (j - 1) * w for i in range(2)]
    ys = [RATIO[d][p] for d in ("web-google", "wiki-talk")]
    ax.bar(xs, ys, width=w * 0.9, color=colr[p], label=p)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", fontsize=6.2)
ax.set_xticks(range(2)); ax.set_xticklabels(["web-google", "wiki-talk"])
ax.set_ylabel("Hit rate (EDGaE = 1)")
ax.set_ylim(0, 1.16); style(ax)
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02),
          fontsize=6.6, columnspacing=0.9, handletextpad=0.4)
fig.tight_layout()
fig.savefig(FIG / "icde-fig-cache-hit-relative.pdf", bbox_inches="tight")
plt.close(fig)
print("Figs 1 & 5 regenerated")

# ============ Excel export: every plotted number ============================
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

wb = Workbook(); wb.remove(wb.active)
HDR = Font(bold=True, color="FFFFFF")
FILL = PatternFill("solid", fgColor="0072B2")

def sheet(name, header, rows):
    ws = wb.create_sheet(name)
    ws.append(header)
    for c in ws[1]: c.font = HDR; c.fill = FILL
    for r in rows: ws.append(r)
    ws.freeze_panes = "A2"
    for col, w in zip("ABCDEFGHIJ", (18, 14, 14, 14, 14, 14, 14, 14, 14, 14)):
        ws.column_dimensions[col].width = w

# Fig 2 data
rows = []
for d in TOPO:
    for k in K + ["full"]:
        c = "K=full-trace" if k == "full" else f"K={k}"
        h = m("cache-stability", d, c, "cache_hit_rate")
        j = m("cache-stability", d, c, "topk_jaccard")
        rows.append([SHORT[d], k, h["mean"], h["ci95_low"], h["ci95_high"],
                     j["mean"], j["ci95_low"], j["ci95_high"]])
sheet("Fig2_cache_stability",
      ["topology", "K", "hit_mean", "hit_ci_lo", "hit_ci_hi",
       "jaccard_mean", "jaccard_ci_lo", "jaccard_ci_hi"], rows)

# Fig 3 data
rows = []
for d in TOPO:
    for p in POL:
        b = m("partition-policy", d, p, "C_edg_bytes")
        r = m("partition-policy", d, p, "replication_factor_vertices")
        rows.append([SHORT[d], POLL[p], b["mean"]/1e6, b["ci95_low"]/1e6,
                     b["ci95_high"]/1e6, r["mean"], r["ci95_low"],
                     r["ci95_high"]])
sheet("Fig3_partition",
      ["topology", "policy", "bytes_MB_mean", "bytes_ci_lo", "bytes_ci_hi",
       "repl_mean", "repl_ci_lo", "repl_ci_hi"], rows)

# Fig 4 data
rows = []
for d in TOPO:
    for tag, confs in (("fanout", CONF_F), ("depth", CONF_D)):
        for c in confs:
            e = m("sampler-sensitivity", d, c, "C_edg_bytes")
            a = m("sampler-sensitivity", d, c, "assignment_agreement")
            rows.append([SHORT[d], tag, c.split(",")[0],
                         e["mean"]/1e6, e["ci95_low"]/1e6, e["ci95_high"]/1e6,
                         a["mean"]])
sheet("Fig4_sensitivity",
      ["topology", "axis", "config", "bytes_MB_mean", "ci_lo", "ci_hi",
       "placement_agreement"], rows)

# Fig 5 data (ratios stated in Section VII; EDGaE normalized to 1)
rows = [[d, p, RATIO[d][p]] for d in RATIO for p in RATIO[d]]
sheet("Fig5_relative_hit", ["dataset", "policy", "hit_vs_EDGaE"], rows)

# Palette reference
rows = [[SHORT[d], C[d]] for d in TOPO] + [[POLL[p], PC[p]] for p in POL]
sheet("palette", ["series", "hex_color"], rows)

out = ROOT / "ICDE/figsource/figure_data_master.xlsx"
wb.save(out)
print("Excel written:", out)
