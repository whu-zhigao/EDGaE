#!/usr/bin/env python3
"""Mode-transition accounting and system-inspired SOTA comparison tables.

This script derives the placement-mode (server-centric / hybrid / edge-local)
decision surface from the *measured* v2 trace results, and emits two LaTeX
 tables:

  1. tab-mode-selection.tex  -- S/H/E amortized transition accounting
  2. tab-sota-systems.tex    -- system-inspired adapter byte-path comparison

Evidence discipline
-------------------
All numbers are derived from trace-driven simulation rows (exact sampler and
byte accounting on real topologies). They are NOT end-to-end system runtimes,
and the SOTA columns are policy adapters inspired by published systems, not
the original end-system deployments. The generated captions state this.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


DATASET_LABELS = {
    "web-google": "web-google 12k",
    "amazon0601": "amazon0601 12k",
    "wiki-talk": "wiki-talk 12k",
    "ogbn-arxiv": "ogbn-arxiv",
    "amazon0601-full": "amazon0601 full",
    "web-google-full": "web-google full",
    "wiki-talk-full": "wiki-talk full",
}
DATASETS = list(DATASET_LABELS)

# Effective bandwidths (bytes/s): 10 Mbps, 100 Mbps, 1 Gbps.
BANDWIDTHS = [("10", 1.25e6), ("100", 1.25e7), ("1000", 1.25e8)]

# Shipping accounting matches the paper: 8 B per replicated vertex record and
# 8 B per replicated unique edge record (Table preprocessing protocol).
RECORD_BYTES = 8
FEATURE_BYTES = 100 * 4

# Aggregate edge-memory envelope for E-mode feasibility (default: the paper's
# physical testbed, 8 x 8 GB Raspberry Pi 4 workers = 64 GB).
DEFAULT_EDGE_MEMORY_GB = 64.0


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values):
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else math.nan


def load_rows(path: Path):
    with path.open() as f:
        return [r for r in csv.DictReader(f) if r.get("measurement_valid") == "True"]


def shipping_bytes_mb(row):
    """One-time replica shipping volume in MB (vertex + edge records)."""
    rf_v = fnum(row.get("replication_factor_vertices"))
    rf_e = fnum(row.get("replication_factor_edges"))
    vertices = fnum(row.get("graph_vertices"))
    edges = fnum(row.get("graph_edges"))
    return (rf_v * vertices + rf_e * edges) * RECORD_BYTES / 1e6


def aggregate_preprocess(rows):
    """Mean per-dataset preprocess metrics (EDGaE-K16 rows)."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("block") == "preprocess" and r.get("control") == "EDGaE-K16":
            groups[r["dataset"]].append(r)
    summary = {}
    for ds, items in groups.items():
        summary[ds] = {
            "C_srv_MB": mean([fnum(r["C_srv_bytes"]) for r in items]) / 1e6,
            "C_edg_MB": mean([fnum(r["C_edg_bytes"]) for r in items]) / 1e6,
            "T_pre_s": mean([fnum(r["T_pre_total_s"]) for r in items]),
            "Ship_MB": mean([shipping_bytes_mb(r) for r in items]),
            "rf_v": mean([fnum(r["replication_factor_vertices"]) for r in items]),
            "rf_e": mean([fnum(r["replication_factor_edges"]) for r in items]),
            "vertices": mean([fnum(r["graph_vertices"]) for r in items]),
            "edges": mean([fnum(r["graph_edges"]) for r in items]),
            "fallbacks": mean([fnum(r["fallbacks"]) for r in items]),
            "working_set_MB": mean([
                fnum(r.get("working_set_bytes", math.nan)) for r in items
            ]) / 1e6,
        }
    return summary


# ---------------------------------------------------------------------------
# Mode-transition accounting (Theorem 1 v2: S/H/E cooperative modes)
# ---------------------------------------------------------------------------

def mode_costs(entry, bandwidth, edge_memory_gb):
    """Amortized per-window cost of each mode at ``bandwidth`` (bytes/s).

    S (server-centric):  no shipping, per-epoch C_S residual traffic.
    H (hybrid, EDGaE):   one-time T_pre + replica shipping, residual C_H.
    E (edge-local):      ship the sampled working set once, C_E = 0,
                         feasible only if the working set fits aggregate
                         edge memory. E is the locality mode, not the
                         throughput mode: its pricing assumes the sampled
                         batch schedule is stable across the horizon.
    """
    B = bandwidth
    C_S = entry["C_srv_MB"]
    C_H = entry["C_edg_MB"]
    ship_H = entry["Ship_MB"]
    ship_E = entry.get("working_set_MB", entry["Ship_MB"]) + entry["Ship_MB"] * 0.1
    T_pre = entry["T_pre_s"]
    e_feasible = ship_E <= edge_memory_gb * 1e3
    return {
        "S": lambda H: H * C_S / (B / 1e6),
        "H": lambda H: T_pre + ship_H / (B / 1e6) + H * C_H / (B / 1e6),
        "E": (lambda H: T_pre + ship_E / (B / 1e6)) if e_feasible
             else (lambda H: math.inf),
    }


def efficiency_mode(entry, bandwidth, horizon=100):
    """Efficiency argmin over the deployable throughput modes {S, H}."""
    costs = mode_costs(entry, bandwidth, DEFAULT_EDGE_MEMORY_GB)
    return "S" if costs["S"](horizon) <= costs["H"](horizon) else "H"

def select_modes(entry, edge_memory_gb):
    """Return per-bandwidth argmin mode and break-even horizons."""
    out = {}
    for label, bw in BANDWIDTHS:
        costs = mode_costs(entry, bw, edge_memory_gb)
        H = 100  # deployment horizon proxy (100 epochs)
        chosen = min(("S", "H", "E"), key=lambda m: costs[m](H))
        # Break-even H vs S: H*  = (T_pre + Ship/B) * B / (C_S - C_H)
        C_S, C_H = entry["C_srv_MB"], entry["C_edg_MB"]
        delta = C_S - C_H
        if delta > 0:
            H_star = (entry["T_pre_s"] * (bw / 1e6) + entry["Ship_MB"]) / delta
        else:
            H_star = math.inf
        out[label] = {"mode": chosen, "H_star": H_star, "delta_MB": delta}
    return out


def gen_mode_table(summary, outdir, edge_memory_gb):
    lines = [
        "% AUTO-GENERATED by experiments/analyze_mode_and_sota.py",
        "% Derived from measured trace-driven preprocess rows (EDGaE-K16).",
        "\\begin{tabular}{@{}lrrrrrrrlll@{}}",
        "\\toprule",
        "Topology & $C_S$ & $C_H$ & Ship & $W_E$ & $H^\\star_{10}$ & $H^\\star_{100}$ & $H^\\star_{1000}$ & Eff$_{10}$ & Eff$_{100}$ & E-fit\\\\",
        "\\midrule",
    ]
    for ds in DATASETS:
        if ds not in summary:
            continue
        e = summary[ds]
        modes = select_modes(e, edge_memory_gb)
        costs = mode_costs(e, 1.25e7, edge_memory_gb)
        e_fit = costs["E"](1) != math.inf
        hs = []
        effs = []
        for label, bw in BANDWIDTHS:
            v = modes[label]["H_star"]
            hs.append("$\\infty$" if not math.isfinite(v) else str(math.ceil(v)))
            effs.append(efficiency_mode(e, bw, horizon=100))
        row = " & ".join([
            "\\texttt{" + DATASET_LABELS[ds] + "}",
            f"{e['C_srv_MB']:.2f}",
            f"{e['C_edg_MB']:.2f}",
            f"{e['Ship_MB']:.1f}",
            f"{e['working_set_MB']:.2f}",
            hs[0], hs[1], hs[2],
            effs[0], effs[1],
            "yes" if e_fit else "no",
        ])
        lines.append(row + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (outdir / "tab-mode-selection.tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# System-inspired SOTA adapters (policy-level comparison, same trace protocol)
# ---------------------------------------------------------------------------

# Each adapter maps to the measured partition/cache policy in the v2 runner.
# Venue labels document the origin system; the caption states that these are
# system-inspired adapters on the identical substrate, not original systems.
SOTA_ADAPTERS = {
    "Server-C":       {"label": "Server-centric", "venue": "--",             "block": "component-ablation", "control": "DGL-E"},
    "DistDGL-like":   {"label": "DistDGL-like",   "venue": "2020",           "block": "partition-policy",   "control": "METIS-like"},
    "PaGraph-like":   {"label": "PaGraph-like",   "venue": "EuroSys'20",     "block": "partition-policy",   "control": "PaGraph-style"},
    "GNNLab-like":    {"label": "GNNLab-like",    "venue": "EuroSys'22",     "block": "online-cache",      "control": "static-K2"},
    "Ginex-like":     {"label": "Ginex-like",     "venue": "PVLDB'22",       "block": "online-cache",      "control": "LFU"},
    "BGL-like":       {"label": "BGL-like",       "venue": "OSDI'22",        "block": "online-cache",      "control": "LRU"},
    "NeutronStar-like": {"label": "NeutronStar-like", "venue": "SIGMOD'22",  "block": "partition-policy",  "control": "METIS-like"},
    "DAHA-like":      {"label": "DAHA-like",      "venue": "PVLDB'24",       "block": "score-sensitivity", "control": "balance-heavy"},
    "PlatoD2GL-like": {"label": "PlatoD2GL-like", "venue": "ICDE'24",        "block": "score-sensitivity", "control": "locality-heavy"},
    "EDGaE":          {"label": "EDGaE (ours)",   "venue": "--",             "block": "partition-policy",  "control": "EDGaE"},
}


def gen_sota_table(trace_rows, outdir):
    """Per-dataset residual bytes for each system-policy adapter.

    Reads the system-policy block (run_mode_system_eval.py) and emits one
    row per adapter: the four complete official graphs plus the mean over
    all seven topologies.
    """
    agg = defaultdict(list)
    for r in trace_rows:
        if r.get("measurement_valid") != "True" or r.get("block") != "system-policy":
            continue
        agg[(r["dataset"], r["control"])].append(fnum(r["C_edg_bytes"]))
    means = {k: mean(v) / 1e6 for k, v in agg.items()}

    show_datasets = [
        "ogbn-arxiv", "amazon0601-full", "web-google-full", "wiki-talk-full",
    ]
    short = {
        "ogbn-arxiv": "arxiv", "amazon0601-full": "amazon-f",
        "web-google-full": "web-f", "wiki-talk-full": "wiki-f",
    }

    header = [
        "% AUTO-GENERATED by experiments/analyze_mode_and_sota.py",
        "% System-inspired policy adapters measured on the identical trace",
        "% substrate. NOT original end-system deployments.",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Adapter & Venue & " + " & ".join(short[d] for d in show_datasets)
        + " & Mean$_{7}$\\\\",
        "\\midrule",
    ]
    body = []
    for name, spec in SOTA_ADAPTERS.items():
        per_ds = [means[(ds, name)] for ds in DATASETS if (ds, name) in means]
        shown = [means[(ds, name)] for ds in show_datasets if (ds, name) in means]
        if not per_ds:
            continue
        avg = sum(per_ds) / len(per_ds)
        body.append(
            f"{spec['label']} & {spec['venue']} & "
            + " & ".join(f"{v:.2f}" for v in shown)
            + f" & {avg:.2f}\\\\"
        )
    lines = header + body + ["\\bottomrule", "\\end{tabular}"]
    (outdir / "tab-sota-systems.tex").write_text("\n".join(lines) + "\n")


def gen_mixed_mode(summary, outdir):
    """Per-dataset mixed-mode bytes: fallback seeds served by S, rest by H.

    Uses the measured fallback counts (natural mixed deployment) and the
    measured C_S/C_H per-window bytes. Only topologies with fallbacks > 0
    produce a genuinely mixed row.
    """
    lines = [
        "% AUTO-GENERATED by experiments/analyze_mode_and_sota.py",
        "% Mixed-mode accounting from measured fallback counts.",
    ]
    rows = []
    for ds in DATASETS:
        if ds not in summary:
            continue
        e = summary[ds]
        fb_ratio = e.get("fallbacks", 0.0) / 1800.0
        C_mix = fb_ratio * e["C_srv_MB"] + (1 - fb_ratio) * e["C_edg_MB"]
        rows.append((DATASET_LABELS[ds], fb_ratio, e["C_srv_MB"], e["C_edg_MB"], C_mix))
    for label, fr, cs, ch, cm in rows:
        lines.append(
            f"% {label}: fallback={fr:.3f} C_S={cs:.2f} C_H={ch:.2f} "
            f"C_mix={cm:.2f}"
        )
    (outdir / "mixed-mode-accounting.txt").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Sensitivity of the mode decision to H/B misestimation.
# ---------------------------------------------------------------------------

def decision(entry, bandwidth, horizon, edge_memory_gb):
    costs = mode_costs(entry, bandwidth, edge_memory_gb)
    feas = [m for m in ("S", "H") if costs[m](horizon) < math.inf]
    return min(feas, key=lambda m: costs[m](horizon)) if feas else "S"


def gen_sensitivity(summary, outdir, edge_memory_gb):
    """Flip rate of the S/H efficiency decision under H and B misestimation.

    For each topology, the nominal decision uses H=100 epochs and the nominal
    bandwidth. We perturb H in {0.5x, 2x} and B in {0.5x, 2x} independently and
    report the fraction of (dataset x bandwidth) cells whose decision changes.
    """
    results = []
    for ds in DATASETS:
        if ds not in summary:
            continue
        e = summary[ds]
        for label, bw in BANDWIDTHS:
            base = decision(e, bw, 100, edge_memory_gb)
            flips = []
            for hmul in (0.5, 2.0):
                for bmul in (0.5, 2.0):
                    alt = decision(e, bw * bmul, int(100 * hmul), edge_memory_gb)
                    flips.append(1 if alt != base else 0)
            results.append({
                "dataset": ds, "bandwidth": label, "base": base,
                "flip_rate": sum(flips) / len(flips),
            })
    # Aggregate: mean flip rate per bandwidth label.
    agg = defaultdict(list)
    for r in results:
        agg[r["bandwidth"]].append(r["flip_rate"])
    out_lines = [
        "% AUTO-GENERATED by experiments/analyze_mode_and_sota.py",
        "% S/H decision flip rate under H in {0.5x,2x} and B in {0.5x,2x}.",
    ]
    for label in ("10", "100", "1000"):
        fr = agg.get(label, [])
        out_lines.append(
            f"% {label} Mbps: mean flip rate "
            f"{(sum(fr)/len(fr)) if fr else 0:.3f}"
        )
    (outdir / "mode-sensitivity.txt").write_text("\n".join(out_lines) + "\n")
    return {label: (sum(v) / len(v) if v else 0.0)
            for label, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--outdir", default="../paper")
    ap.add_argument("--trace-csv", default="results/icde2027-v2-trace-results.csv")
    ap.add_argument("--preprocess-csv", default="results/icde2027-v2-preprocess-results.csv")
    ap.add_argument("--edge-memory-gb", type=float, default=DEFAULT_EDGE_MEMORY_GB)
    args = ap.parse_args()

    results = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pre_rows = load_rows(Path(args.preprocess_csv))
    summary = aggregate_preprocess(pre_rows)
    gen_mode_table(summary, outdir, args.edge_memory_gb)

    trace_rows = load_rows(Path(args.trace_csv))
    gen_sota_table(trace_rows, outdir)
    gen_mixed_mode(summary, outdir)
    flip = gen_sensitivity(summary, outdir, args.edge_memory_gb)
    regret = gen_regret(summary, outdir, args.edge_memory_gb)
    gen_horizon_table(summary, outdir)
    nsplit, ntotal = gen_heterogeneous_bw(summary, outdir)
    print(f"wrote {outdir/'tab-mode-selection.tex'} and {outdir/'tab-sota-systems.tex'}")
    print(f"heterogeneous B_i: {nsplit}/{ntotal} cells split modes")
    print(f"decision flip rates: {flip}")
    print(f"regret: {regret}")




# ---------------------------------------------------------------------------
# Regret of the mode decision under H/B misestimation (review MC3).
# ---------------------------------------------------------------------------

def _mode_cost(entry, mode, bandwidth, horizon, edge_memory_gb):
    """Bytes-based amortized cost of a mode at (B, H)."""
    B = bandwidth / 1e6
    H = horizon
    C_S, C_H = entry["C_srv_MB"], entry["C_edg_MB"]
    ship_H = entry["Ship_MB"]
    ship_E = entry.get("working_set_MB", entry["Ship_MB"])
    T_pre = entry["T_pre_s"]
    e_ok = ship_E <= edge_memory_gb * 1e3
    if mode == "S":
        return H * C_S / B
    if mode == "H":
        return T_pre + ship_H / B + H * C_H / B
    if mode == "E":
        return T_pre + ship_E / B if e_ok else math.inf
    raise ValueError(mode)


def gen_regret(summary, outdir, edge_memory_gb):
    """Mean worst-case regret of the chosen mode vs the oracle mode.

    The governor plans at (H0, B0) but the deployment runs at (H, B).
    Regret = (A_chosen(H,B) - A_oracle(H,B)) / A_oracle(H,B).
    """
    results = []
    for ds in DATASETS:
        if ds not in summary:
            continue
        e = summary[ds]
        for label, bw0 in BANDWIDTHS:
            for hmul, bmul in ((0.5, 0.5), (2.0, 2.0), (0.5, 2.0), (2.0, 0.5)):
                true_bw, true_h = bw0 * bmul, 100 * hmul
                chosen = min(("S", "H", "E"),
                             key=lambda m: _mode_cost(e, m, bw0, 100,
                                                      edge_memory_gb))
                a_chosen = _mode_cost(e, chosen, true_bw, true_h,
                                      edge_memory_gb)
                oracle = min(("S", "H", "E"),
                             key=lambda m: _mode_cost(e, m, true_bw, true_h,
                                                      edge_memory_gb))
                a_oracle = _mode_cost(e, oracle, true_bw, true_h,
                                      edge_memory_gb)
                regret = ((a_chosen - a_oracle) / a_oracle
                          if a_oracle > 0 else 0.0)
                results.append({"dataset": ds, "bw0": label,
                                "hmul": hmul, "bmul": bmul,
                                "chosen": chosen, "oracle": oracle,
                                "regret": regret})
    agg = defaultdict(list)
    for r in results:
        agg[r["bw0"]].append(r["regret"])
    lines = ["% Regret of the governor's mode choice vs oracle under "
             "H/B misestimation (worst of four perturbations per cell)."]
    out = {}
    for label in ("10", "100", "1000"):
        vals = agg.get(label, [])
        worst = max(vals) if vals else 0.0
        mean = sum(vals) / len(vals) if vals else 0.0
        frac_nonzero = sum(1 for v in vals if v > 1e-9) / max(len(vals), 1)
        out[label] = (mean, worst, frac_nonzero)
        lines.append(f"% {label} Mbps: mean={mean:.4f} worst={worst:.4f} "
                     f"fraction_cells_with_regret={frac_nonzero:.3f}")
    (outdir / "mode-regret.txt").write_text("\n".join(lines) + "\n")
    return out


# ---------------------------------------------------------------------------
# Unified horizon table: total amortized transfer time per mode (MC4.5).
# ---------------------------------------------------------------------------

HORIZONS = [1, 10, 50, 100, 500]


def gen_horizon_table(summary, outdir):
    """Total bytes-based amortized time (T_pre + shipping + H*C_m/B).

    Compute time is identical across modes on the same fabric and is
    therefore excluded; the table isolates the data-path terms that the
    governor prices. Times in seconds at 10/100/1000 Mbps.
    """
    lines = [
        "% AUTO-GENERATED: total data-path amortized time per mode.",
        "% time_m(H) = T_pre,m + D_m/B + H * C_m/B   (seconds)",
    ]
    rows = []
    for ds in DATASETS:
        if ds not in summary:
            continue
        e = summary[ds]
        for bw in (10, 100, 1000):
            B = bw * 1e6 / 8.0 / 1e6  # MB/s
            for hz in HORIZONS:
                tS = hz * e["C_srv_MB"] / B
                tH = e["T_pre_s"] + e["Ship_MB"] / B + hz * e["C_edg_MB"] / B
                tE = e["T_pre_s"] + e.get("working_set_MB", 0) / B
                best = min(("S", "H", "E"), key=lambda m: {"S": tS, "H": tH,
                                                           "E": tE}[m])
                rows.append((DATASET_LABELS[ds], bw, hz, tS, tH, tE, best))
    # Emit a compact LaTeX table: per (dataset, bw), the crossing horizons.
    lines.append("% Crossing analysis: first horizon where each mode wins.")
    for ds in DATASETS:
        if ds not in summary:
            continue
        seq = {}
        for (label, bw, hz, tS, tH, tE, best) in rows:
            if label == DATASET_LABELS.get(ds):
                seq.setdefault(bw, {})[hz] = best
        for bw, m in seq.items():
            order = [m.get(h, "?") for h in HORIZONS]
            lines.append(f"% {DATASET_LABELS[ds]} @{bw}Mbps: "
                         + " ".join(f"H{h}={b}" for h, b in zip(HORIZONS, order)))
    (outdir / "horizon-table.txt").write_text("\n".join(lines) + "\n")
    return rows




# ---------------------------------------------------------------------------
# Heterogeneous per-worker bandwidth (trace-level B_i analysis).
# ---------------------------------------------------------------------------

def gen_heterogeneous_bw(summary, outdir):
    """Mode decision under asymmetric per-worker bandwidth.

    Scenario: one straggler worker at 10 Mbps while the others sit at
    100/500/1000 Mbps. Under scalar-B accounting the fleet decision uses the
    harmonic-mean bandwidth; under per-worker B_i each worker may pick its own
    efficiency mode. We report, per topology, whether the straggler's mode
    differs from the fleet majority.
    """
    rows = []
    for ds in DATASETS:
        if ds not in summary:
            continue
        e = summary[ds]
        C_S, C_H = e["C_srv_MB"], e["C_edg_MB"]
        T_pre, ship = e["T_pre_s"], e["Ship_MB"]
        for fast in (100, 500, 1000):
            B_slow, B_fast = 10 * 1e6 / 8e6, fast * 1e6 / 8e6  # MB/s
            # Slow worker: S vs H at B_slow, H = 100
            aS = 100 * C_S / B_slow
            aH = T_pre + ship / B_slow + 100 * C_H / B_slow
            slow_mode = "S" if aS <= aH else "H"
            # Fast workers
            bS = 100 * C_S / B_fast
            bH = T_pre + ship / B_fast + 100 * C_H / B_fast
            fast_mode = "S" if bS <= bH else "H"
            rows.append({
                "dataset": DATASET_LABELS[ds], "fast_mbps": fast,
                "slow_mode": slow_mode, "fast_mode": fast_mode,
                "split": slow_mode != fast_mode,
            })
    n_split = sum(1 for r in rows if r["split"])
    lines = [
        "% Heterogeneous B_i: straggler at 10 Mbps vs fleet at 100/500/1000.",
        f"% cells where straggler mode differs from fleet mode: {n_split}/{len(rows)}",
    ] + [f"% {r['dataset']} @fast={r['fast_mbps']}: slow={r['slow_mode']} "
         f"fast={r['fast_mode']}" + ("  <-- split" if r["split"] else "")
         for r in rows]
    (outdir / "heterogeneous-bw.txt").write_text("\n".join(lines) + "\n")
    return n_split, len(rows)


if __name__ == "__main__":
    main()
