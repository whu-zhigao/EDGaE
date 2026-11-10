#!/usr/bin/env python3
"""Regenerate Fig. 1 (profiling breakdown) and its Excel raw-data workbook.

Inputs
------
exp/experiments/profile_results_calibrated.csv : 18 rows, the calibrated
    profiling study (3 datasets x 2 samplers x 3 worker counts). The
    ``feature_pct`` / ``compute_pct`` columns are exactly the stacked-bar
    values plotted in the paper figure.

Outputs
-------
paper/fig/fig3-styled.pdf       : single-column figure, dataset names on the
                                   x-axis under each bar group
paper/figsource/fig1-data.xlsx  : raw data workbook
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent.parent / "exp" / "experiments" / "profile_results_calibrated.csv"
FIG_PATH = HERE.parent / "fig" / "fig3-styled.pdf"
XLSX_PATH = HERE / "fig1-data.xlsx"

DATASETS = ["amazon0601", "web-google", "ogbn-arxiv"]
WORKERS = [2, 4, 8]
SAMPLERS = [("non-uniform", "(a) Non-uniform sampling"),
            ("uniform", "(b) Uniform sampling")]

C_FEATURE = "#4C72B0"
C_COMPUTE = "#C4C4C4"
EDGE = "#333333"


def load_rows():
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def index_rows(rows):
    return {(r["dataset"], r["sampling"], int(r["workers"])): r for r in rows}


def build_figure(idx):
    # Two stacked panels (single-column layout): each panel gives the full
    # column width to three dataset groups, keeping dataset names horizontal
    # and legible at IEEE print size.
    fig, axes = plt.subplots(
        2, 1, figsize=(3.5, 2.8), sharex=True,
        gridspec_kw={"hspace": 0.42},
    )
    bar_w = 0.24
    for ax, (sampler, title) in zip(axes, SAMPLERS):
        for gi, ds in enumerate(DATASETS):
            for wi, w in enumerate(WORKERS):
                r = idx[(ds, sampler, w)]
                feat = float(r["feature_pct"])
                comp = float(r["compute_pct"])
                x = gi + (wi - 1) * bar_w
                ax.bar(x, feat, width=bar_w * 0.9, color=C_FEATURE,
                       edgecolor=EDGE, linewidth=0.4, zorder=3)
                ax.bar(x, comp, bottom=feat, width=bar_w * 0.9,
                       color=C_COMPUTE, edgecolor=EDGE, linewidth=0.4, zorder=3)
                ax.text(x, feat / 2, f"{feat:.0f}", ha="center", va="center",
                        fontsize=6.8, color="white", rotation=90, zorder=4)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_ylabel("Epoch time (%)", fontsize=7.0)
        ax.set_title(title, fontsize=7.5, pad=4)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels(DATASETS if ax is axes[1] else ["", "", ""], fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, zorder=0)
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=C_FEATURE, ec=EDGE, lw=0.4),
        plt.Rectangle((0, 0), 1, 1, fc=C_COMPUTE, ec=EDGE, lw=0.4),
    ]
    axes[0].legend(
        handles, ["Feature loading", "Computation"],
        loc="upper left", bbox_to_anchor=(0.0, 1.38), ncol=2,
        fontsize=6.5, frameon=False, handlelength=1.2, handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.text(
        0.92, 0.955,
        "bars: 2/4/8 workers",
        ha="right", va="top", fontsize=6.0,
    )
    fig.savefig(FIG_PATH, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {FIG_PATH}")


HEADER = ["Dataset", "Workers", "Feature loading (%)", "Computation (%)",
          "Feature MB/epoch", "Transfer s/epoch", "Pi compute per worker (s)",
          "Calibration factor"]


def _sheet_rows(idx, sampler):
    out = []
    for ds in DATASETS:
        for w in WORKERS:
            r = idx[(ds, sampler, w)]
            out.append([ds, w, float(r["feature_pct"]),
                        float(r["compute_pct"]), float(r["feature_MB"]),
                        float(r["transfer_s"]),
                        float(r["pi_compute_per_worker_s"]),
                        float(r["calibration_factor"])])
    return out


def build_workbook(rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "Fig1 plotted values"
    fill = PatternFill("solid", fgColor="D9E2F3")
    bold = Font(bold=True)

    ws.append(["Fig. 1 raw data: feature-loading / computation share of per-epoch time (%)"])
    ws["A1"].font = Font(bold=True, size=12)
    ws.append([])
    ws.append(["Panel (a): non-uniform sampling"])
    ws["A3"].font = bold
    ws.append(HEADER)
    for c in range(1, len(HEADER) + 1):
        cell = ws.cell(row=4, column=c)
        cell.font = bold
        cell.fill = fill
    for row in _sheet_rows(index_rows(rows), "non-uniform"):
        ws.append(row)
    ws.append([])
    b_start = ws.max_row + 1
    ws.append(["Panel (b): uniform sampling"])
    ws.cell(row=b_start, column=1).font = bold
    ws.append(HEADER)
    hdr2 = ws.max_row
    for c in range(1, len(HEADER) + 1):
        cell = ws.cell(row=hdr2, column=c)
        cell.font = bold
        cell.fill = fill
    for row in _sheet_rows(index_rows(rows), "uniform"):
        ws.append(row)
    for col, width in zip("ABCDEFGH", [14, 9, 18, 16, 16, 15, 21, 14]):
        ws.column_dimensions[col].width = width
    ws.freeze_panes = "A5"

    ws2 = wb.create_sheet("Fig1 matrix view")
    ws2.append(["Matrix view matching the figure: rows = dataset x workers, columns = sampler"])
    ws2["A1"].font = Font(bold=True, size=12)
    ws2.append([])
    hdr = ["Dataset", "Workers", "Feature load % (non-uniform)",
           "Computation % (non-uniform)", "Feature load % (uniform)",
           "Computation % (uniform)"]
    ws2.append(hdr)
    for c in range(1, len(hdr) + 1):
        ws2.cell(row=3, column=c).font = bold
        ws2.cell(row=3, column=c).fill = fill
    idx = index_rows(rows)
    for ds in DATASETS:
        for w in WORKERS:
            a, b = idx[(ds, "non-uniform", w)], idx[(ds, "uniform", w)]
            ws2.append([ds, w, float(a["feature_pct"]), float(a["compute_pct"]),
                        float(b["feature_pct"]), float(b["compute_pct"])])
    for col, width in zip("ABCDEF", [14, 9, 24, 24, 22, 22]):
        ws2.column_dimensions[col].width = width
    ws2.freeze_panes = "A4"

    ws3 = wb.create_sheet("Provenance")
    ws3.append(["Item", "Value"])
    ws3.append(["Source CSV", "exp/experiments/profile_results_calibrated.csv"])
    ws3.append(["Rows", "18"])
    ws3.append(["Design", "3 datasets x 2 samplers x 3 worker counts"])
    ws3.append(["Plotted quantity", "feature_pct (bottom segment), compute_pct (top segment)"])
    ws3.append(["Figure", "paper/fig/fig3-styled.pdf"])
    ws3.append(["Note",
                "feature_pct = transfer_s / (transfer_s + pi_compute_per_worker_s) "
                "after calibration; computation is the complementary share."])
    for c in range(1, 3):
        ws3.cell(row=1, column=c).font = bold
    for col, width in zip("AB", [20, 95]):
        ws3.column_dimensions[col].width = width
    wb.save(XLSX_PATH)
    print(f"wrote {XLSX_PATH}")


def main():
    rows = load_rows()
    assert len(rows) == 18, f"expected 18 rows, got {len(rows)}"
    build_figure(index_rows(rows))
    build_workbook(rows)


if __name__ == "__main__":
    main()
