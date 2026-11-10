#!/usr/bin/env python3
"""Regenerate the partition-policy figure as a single-column bar chart.

Reads the canonical v2 trace CSV (partition-policy block, five seeds),
plots residual bytes and vertex replication for five policies over the seven
topologies, and exports every plotted number (mean + bootstrap CI) to Excel.

Styling (colors, layout, fonts) matches the ICDE_v2 figure exactly; only the
data source has moved to the Nov-7 cloud run.

Outputs
-------
paper/fig/trace-partition.pdf          single-column 2x2 bar figure
paper/figsource/partition-data.xlsx    plotted data workbook
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill


HERE = Path(__file__).resolve().parent
CSV_PATH = (HERE.parent.parent / "exp" / "experiments" / "results_cloud"
            / "results" / "icde2027-v2-trace-results.csv")
FIG_PATH = HERE.parent / "fig" / "trace-partition.pdf"
XLSX_PATH = HERE / "partition-data.xlsx"

TOPO12 = ["web-google", "amazon0601", "wiki-talk", "ogbn-arxiv"]
TOPOFULL = ["amazon0601-full", "web-google-full", "wiki-talk-full"]
SHORT = {"web-google": "wg", "amazon0601": "am",
         "wiki-talk": "wt", "ogbn-arxiv": "arxiv",
         "amazon0601-full": "am", "web-google-full": "wg",
         "wiki-talk-full": "wt"}
POL = ["random", "even", "greedy-overlap", "EDGaE"]
POLL = {"random": "Random", "even": "Even",
        "greedy-overlap": "Greedy-overlap", "EDGaE": "EDGaE"}
PC = {"random": "#B8B8B8", "even": "#56B4E9", "greedy-overlap": "#009E73",
      "EDGaE": "#D55E00"}


def bootstrap_ci(values, samples=3000, seed=12345):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(samples))
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def load_means():
    groups = defaultdict(list)
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if (r.get("measurement_valid") == "True"
                    and r.get("block") == "partition-policy"):
                groups[(r["dataset"], r["control"])].append(r)
    # METIS-like and PaGraph-style share one branch in the runner and are
    # byte-identical on every (dataset, seed) cell.  Verify, then merge the
    # two series into a single "greedy-overlap" baseline.
    for ds in set(d for (d, _) in groups):
        met = sorted(r["C_edg_bytes"] for r in groups[(ds, "METIS-like")])
        pag = sorted(r["C_edg_bytes"] for r in groups[(ds, "PaGraph-style")])
        if met != pag:
            raise SystemExit(f"METIS-like != PaGraph-style on {ds}; "
                             "do not merge without re-implementing them")
        groups[(ds, "greedy-overlap")] = groups[(ds, "METIS-like")]
    out = {}
    for (ds, pol), rows in groups.items():
        entry = {}
        for field, scale in (("C_edg_bytes", 1e6),
                             ("replication_factor_vertices", 1.0)):
            vals = [float(r[field]) for r in rows]
            mean = sum(vals) / len(vals)
            lo, hi = bootstrap_ci(vals)
            entry[field] = (mean / scale, lo / scale, hi / scale)
        out[(ds, pol)] = entry
    return out


def draw_panel(ax, means, datasets, field, ylabel):
    w = 0.185
    for j, pol in enumerate(POL):
        xs, ys, el, eh = [], [], [], []
        for i, ds in enumerate(datasets):
            mean, lo, hi = means[(ds, pol)][field]
            xs.append(i + (j - 1.5) * w)
            ys.append(mean)
            el.append(mean - lo)
            eh.append(hi - mean)
        ax.bar(xs, ys, width=w * 0.9, color=PC[pol], label=POLL[pol],
               zorder=3)
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([SHORT[d] for d in datasets], fontsize=6.5)
    ax.set_ylabel(ylabel, fontsize=7.0)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def build_figure(means):
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.1))
    draw_panel(axes[0][0], means, TOPO12, "C_edg_bytes", "Residual MB")
    axes[0][0].set_title("(a) Residual bytes (12k graphs)",
                         loc="left", fontsize=7.0)
    draw_panel(axes[0][1], means, TOPOFULL, "C_edg_bytes", "Residual MB")
    axes[0][1].set_title("(b) Residual bytes (full graphs)",
                         loc="left", fontsize=7.0)
    draw_panel(axes[1][0], means, TOPO12,
               "replication_factor_vertices", "Vertex replication")
    axes[1][0].set_title("(c) Replication (12k graphs)",
                         loc="left", fontsize=7.0)
    draw_panel(axes[1][1], means, TOPOFULL,
               "replication_factor_vertices", "Vertex replication")
    axes[1][1].set_title("(d) Replication (full graphs)",
                         loc="left", fontsize=7.0)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PC[p], label=POLL[p])
               for p in POL]
    fig.legend(handles=handles, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.008), fontsize=6.5,
               columnspacing=0.8, handletextpad=0.3, handlelength=1.0,
               frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(FIG_PATH, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIG_PATH}")


def build_workbook(means):
    wb = Workbook()
    ws = wb.active
    ws.title = "Partition figure data"
    fill = PatternFill("solid", fgColor="D9E2F3")
    bold = Font(bold=True)
    ws.append(["Fig. partition-policy data: 5 policies x 7 topologies, "
               "mean and bootstrap 95% CI over five seeds"])
    ws["A1"].font = Font(bold=True, size=12)
    ws.append([])
    ws.append(["Topology", "Policy", "Residual MB (mean)",
               "Residual MB (CI low)", "Residual MB (CI high)",
               "Vertex replication (mean)", "Replication (CI low)",
               "Replication (CI high)"])
    for c in range(1, 9):
        cell = ws.cell(row=3, column=c)
        cell.font = bold
        cell.fill = fill
    for ds in TOPO12 + TOPOFULL:
        for pol in POL:
            m1, lo1, hi1 = means[(ds, pol)]["C_edg_bytes"]
            m2, lo2, hi2 = means[(ds, pol)]["replication_factor_vertices"]
            ws.append([ds, POLL[pol], round(m1, 3), round(lo1, 3),
                       round(hi1, 3), round(m2, 3), round(lo2, 3),
                       round(hi2, 3)])
    for col, width in zip("ABCDEFGH", [16, 14, 18, 18, 18, 20, 18, 18]):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A4"
    wb.save(XLSX_PATH)
    print(f"wrote {XLSX_PATH}")


def main():
    means = load_means()
    build_figure(means)
    build_workbook(means)


if __name__ == "__main__":
    main()
