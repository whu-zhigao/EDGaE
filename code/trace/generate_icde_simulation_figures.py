#!/usr/bin/env python3
"""Generate IEEE-column figures for the real-topology trace-driven study."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WWW_DIR = ROOT / "www"
FIG_DIR = ROOT / "ICDE" / "fig"

FILES = {
    "web-google": WWW_DIR / "www2027-real-trace-web-google-summary-v3.csv",
    "amazon0601": WWW_DIR / "www2027-real-trace-amazon-summary-v2.csv",
    "wiki-talk": WWW_DIR / "www2027-real-trace-wiki-summary-v2.csv",
    "ogbn-arxiv": WWW_DIR / "www2027-real-trace-arxiv-summary-v2.csv",
}

# Colors follow the pastel, black-edged style used by the paper's existing plots.
DATASET_COLORS = {
    "web-google": "#6195ce",
    "amazon0601": "#82c47c",
    "wiki-talk": "#ea7e7e",
    "ogbn-arxiv": "#e2b83d",
}

POLICY_COLORS = {
    "even": "#dad6d7",
    "random": "#b0d4ec",
    "METIS-like": "#ffe699",
    "PaGraph-style": "#ea7e7e",
    "EDGaE": "#82c47c",
}

K_ORDER = ["K=1", "K=2", "K=4", "K=8", "K=16", "K=full-trace"]
K_LABELS = [1, 2, 4, 8, 16, r"$\infty$"]
POLICY_ORDER = ["even", "random", "METIS-like", "PaGraph-style", "EDGaE"]


def style_canvas() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.edgecolor": "#4a4a4a",
            "axes.labelcolor": "#1f1f1f",
            "xtick.color": "#4a4a4a",
            "ytick.color": "#4a4a4a",
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.55,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
        }
    )


def load_frames() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in FILES.items()}


def cache_stability_figure(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), constrained_layout=True)
    panels = [
        ("cache_hit_rate_mean", "cache_hit_rate_ci95_low", "cache_hit_rate_ci95_high", "Cache hit rate"),
        ("topk_jaccard_mean", "topk_jaccard_ci95_low", "topk_jaccard_ci95_high", "Top-$k$ Jaccard"),
    ]

    for ax, (mean_col, low_col, high_col, ylabel) in zip(axes, panels):
        for dataset, frame in frames.items():
            rows = frame[frame["block"] == "cache-stability"].set_index("control").loc[K_ORDER]
            x = np.arange(len(K_ORDER))
            mean = rows[mean_col].to_numpy()
            low = rows[low_col].to_numpy()
            high = rows[high_col].to_numpy()
            ax.fill_between(x, low, high, color=DATASET_COLORS[dataset], alpha=0.16, linewidth=0)
            ax.plot(
                x,
                mean,
                marker="o",
                markersize=3.2,
                linewidth=1.25,
                markeredgecolor="white",
                markeredgewidth=0.35,
                color=DATASET_COLORS[dataset],
                label=dataset,
            )
        ax.set_xticks(np.arange(len(K_ORDER)), K_LABELS)
        ax.set_xlabel("Pre-sampling rounds $K$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.28, len(K_ORDER) - 0.72)
        ax.set_ylim(0, 1.03)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("(a) Cache effectiveness", pad=4)
    axes[1].set_title("(b) Hot-set stability", pad=4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
        handlelength=1.7,
        columnspacing=1.1,
    )
    fig.savefig(FIG_DIR / "icde-simulation-cache-stability.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def partition_figure(frames: dict[str, pd.DataFrame]) -> None:
    datasets = list(frames)
    policies = POLICY_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.25), constrained_layout=True)

    metrics = [
        ("C_edg_bytes_mean", "Residual feature bytes (MB)", 1 / (1024**2), "(a) Communication volume"),
        ("replication_factor_vertices_mean", "Vertex replication factor", 1.0, "(b) Replication overhead"),
    ]

    group_width = 0.82
    bar_width = group_width / len(policies)
    centers = np.arange(len(datasets))
    for ax, (column, ylabel, scale, title) in zip(axes, metrics):
        for j, policy in enumerate(policies):
            values = []
            for dataset in datasets:
                frame = frames[dataset]
                row = frame[(frame["block"] == "partition-policy") & (frame["control"] == policy)]
                values.append(float(row.iloc[0][column]) * scale)
            offsets = (j - (len(policies) - 1) / 2) * bar_width
            ax.bar(
                centers + offsets,
                values,
                width=bar_width * 0.88,
                color=POLICY_COLORS[policy],
                edgecolor="#303030",
                linewidth=0.55,
                label=policy,
            )
        ax.set_xticks(centers, datasets, rotation=18, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=4)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylim(0, 7.0)
    axes[1].set_ylim(0, 6.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=5,
        frameon=False,
        handlelength=1.25,
        columnspacing=0.75,
    )
    fig.savefig(FIG_DIR / "icde-simulation-partition-feasibility.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _line_panel(
    ax: plt.Axes,
    frames: dict[str, pd.DataFrame],
    block: str,
    controls: list[str],
    labels: list[str],
    column: str,
    ylabel: str,
    scale: float = 1.0,
) -> None:
    x = np.arange(len(controls))
    for dataset, frame in frames.items():
        rows = frame[frame["block"] == block].set_index("control").loc[controls]
        values = rows[column].to_numpy() * scale
        ax.plot(
            x,
            values,
            marker="o",
            markersize=3.1,
            linewidth=1.2,
            markeredgecolor="white",
            markeredgewidth=0.3,
            color=DATASET_COLORS[dataset],
            label=dataset,
        )
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_xlabel("Configuration")
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)


def ablation_sensitivity_figure(frames: dict[str, pd.DataFrame]) -> None:
    component_controls = ["DGL-E", "+R", "+RC", "+RCS", "+RCSP"]
    component_labels = ["DGL-E", "+R", "+RC", "+RCS", "+RCSP"]
    score_controls = ["equal-weights", "locality-heavy", "balance-heavy", "memory-heavy"]
    score_labels = ["Equal", "Locality", "Balance", "Memory"]

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 4.35), constrained_layout=True)
    _line_panel(
        axes[0, 0],
        frames,
        "component-ablation",
        component_controls,
        component_labels,
        "C_edg_bytes_mean",
        "Residual bytes (MB)",
        1 / (1024**2),
    )
    _line_panel(
        axes[0, 1],
        frames,
        "component-ablation",
        component_controls,
        component_labels,
        "cache_hit_rate_mean",
        "Cache hit rate",
    )
    _line_panel(
        axes[1, 0],
        frames,
        "score-sensitivity",
        score_controls,
        score_labels,
        "C_edg_bytes_mean",
        "Residual bytes (MB)",
        1 / (1024**2),
    )
    _line_panel(
        axes[1, 1],
        frames,
        "score-sensitivity",
        score_controls,
        score_labels,
        "replication_factor_vertices_mean",
        "Vertex replication",
    )

    axes[0, 0].set_title("(a) Component bytes", pad=3)
    axes[0, 1].set_title("(b) Component hit rate", pad=3)
    axes[1, 0].set_title("(c) Score-weight bytes", pad=3)
    axes[1, 1].set_title("(d) Score-weight replication", pad=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=4,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
    )
    fig.savefig(FIG_DIR / "icde-simulation-ablation-sensitivity.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    style_canvas()
    FIG_DIR.mkdir(exist_ok=True)
    frames = load_frames()
    cache_stability_figure(frames)
    partition_figure(frames)
    ablation_sensitivity_figure(frames)


if __name__ == "__main__":
    main()
