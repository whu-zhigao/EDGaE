#!/usr/bin/env python3
"""Validate, derive, summarize, and export Appendix-B measurements."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import math
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

from schema import COLUMNS, REQUIRED_FOR_MEASUREMENT


GROUP_KEYS = [
    "block", "dataset", "model", "sampler", "workers", "control",
]

TRACE_REQUIRED = [
    "C_srv_bytes",
    "C_edg_bytes",
    "cache_hit_rate",
    "cache_coverage",
    "replication_factor_vertices",
    "replication_factor_edges",
    "load_imbalance",
]

EMULATED_REQUIRED = [
    "T_pre_total_s",
    "DGL_E_epoch_s",
    "EDGaE_epoch_s",
    "B_star_bytes_s",
    "peak_rss_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default="www2027-extended-results-measured.csv")
    parser.add_argument("--derived-output", default="www2027-extended-results-derived.csv")
    parser.add_argument("--summary-output", default="www2027-extended-results-summary.csv")
    parser.add_argument("--latex-output", default="www2027-appendix-b-tables.tex")
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def derive(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    timing = {k: as_float(row.get(k)) for k in [
        "T_partition_s", "T_presample_s", "T_cache_s", "T_ship_s",
        "T_comp_dgl_s", "T_comp_edgae_s", "T_sync_dgl_s", "T_sync_edgae_s",
        "DGL_E_epoch_s", "EDGaE_epoch_s", "C_srv_bytes", "C_edg_bytes",
        "effective_B_bytes_s", "horizon_epochs",
    ]}

    pre_parts = [timing[k] for k in [
        "T_partition_s", "T_presample_s", "T_cache_s", "T_ship_s"
    ]]
    if all(v is not None for v in pre_parts):
        out["T_pre_total_s"] = sum(pre_parts)

    if timing["C_srv_bytes"] is not None and timing["C_edg_bytes"] is not None:
        out["Delta_C_bytes"] = timing["C_srv_bytes"] - timing["C_edg_bytes"]

    delta_t_parts = []
    if timing["T_comp_edgae_s"] is not None and timing["T_comp_dgl_s"] is not None:
        delta_t_parts.append(timing["T_comp_edgae_s"] - timing["T_comp_dgl_s"])
    if timing["T_sync_edgae_s"] is not None and timing["T_sync_dgl_s"] is not None:
        delta_t_parts.append(timing["T_sync_edgae_s"] - timing["T_sync_dgl_s"])
    if delta_t_parts:
        out["Delta_T_s"] = sum(delta_t_parts)

    H = timing["horizon_epochs"] or 100.0
    delta_c = as_float(out.get("Delta_C_bytes"))
    delta_t = as_float(out.get("Delta_T_s"))
    t_pre = as_float(out.get("T_pre_total_s"))
    if delta_c is not None and delta_t is not None and t_pre is not None:
        denominator = t_pre + H * delta_t
        if denominator > 0:
            out["B_star_bytes_s"] = H * delta_c / denominator

    B = timing["effective_B_bytes_s"]
    if B is not None and t_pre is not None and delta_c is not None and delta_t is not None:
        denominator = delta_c - B * delta_t
        if denominator > 0:
            out["break_even_epochs"] = math.ceil(t_pre * B / denominator)

    dgl_epoch = timing["DGL_E_epoch_s"]
    edg_epoch = timing["EDGaE_epoch_s"]
    if dgl_epoch is not None and edg_epoch is not None and edg_epoch > 0:
        out["epoch_speedup"] = dgl_epoch / edg_epoch
    return out


def bootstrap_ci(values: List[float], samples: int, seed: int) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    stats = []
    for _ in range(samples):
        stats.append(mean(rng.choice(values) for _ in values))
    stats.sort()
    lo_index = int(0.025 * (len(stats) - 1))
    hi_index = int(0.975 * (len(stats) - 1))
    return stats[lo_index], stats[hi_index]


def numeric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    excluded = set(GROUP_KEYS + ["seed", "run_id", "measurement_utc", "notes"])
    return [c for c in COLUMNS if c not in excluded and any(as_float(r.get(c)) is not None for r in rows)]


def summarize(rows: List[Dict[str, Any]], samples: int, seed: int) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in GROUP_KEYS)].append(row)
    columns = numeric_columns(rows)
    summaries: List[Dict[str, Any]] = []
    base_seed = seed
    for key, group in sorted(groups.items()):
        summary: Dict[str, Any] = dict(zip(GROUP_KEYS, key))
        summary["n_seeds"] = len({r.get("seed") for r in group})
        summary["n_runs"] = len(group)
        for column in columns:
            values = [v for v in (as_float(r.get(column)) for r in group) if v is not None]
            if not values:
                continue
            summary[f"{column}_mean"] = mean(values)
            if len(values) > 1:
                summary[f"{column}_std"] = stdev(values)
            lo, hi = bootstrap_ci(values, samples, base_seed)
            summary[f"{column}_ci95_low"] = lo
            summary[f"{column}_ci95_high"] = hi
            base_seed += 1
        summaries.append(summary)
    return summaries


def status_report(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("block", "")].append(row)
    out = []
    for block, group in sorted(groups.items()):
        complete = 0
        for row in group:
            evidence = row.get("evidence_level", "")
            if evidence == "trace-driven-simulation":
                required = TRACE_REQUIRED
            elif evidence == "parameterized-emulation":
                required = EMULATED_REQUIRED
            else:
                required = REQUIRED_FOR_MEASUREMENT
            missing = [k for k in required if as_float(row.get(k)) is None]
            if not missing and as_bool(row.get("measurement_valid")):
                complete += 1
        out.append({
            "block": block,
            "rows": len(group),
            "valid_complete_rows": complete,
            "completion_rate": complete / len(group) if group else 0.0,
        })
    return out


def latex_escape(value: Any) -> str:
    text = str(value)
    for char in ["&", "%", "$", "#", "_", "{", "}"]:
        text = text.replace(char, "\\" + char)
    return text


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "--"
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}"


def make_tables(rows: List[Dict[str, Any]], summaries: List[Dict[str, Any]]) -> str:
    invalid = [r for r in rows if not as_bool(r.get("measurement_valid"))]
    warning = ""
    if invalid:
        warning = (
            "\\textbf{Data-quality warning:} This file contains "
            f"{len(invalid)} row(s) not marked as valid measurements; do not "
            "include it in a manuscript.\n\n"
        )

    parts = [
        "% AUTO-GENERATED BY experiments/analyze_extended_results.py\n",
        "% Do not edit by hand; regenerate from raw measurements.\n",
        warning,
    ]

    parts.append("\\section{Extended Evaluation Results}\n")
    parts.append("\\label{app:extended-results}\n\n")

    # Completion status.
    parts.append("\\begin{table}[t]\n")
    parts.append("\\caption{Extended-evaluation completion status.}\n")
    parts.append("\\label{tbl:extended-status}\n")
    parts.append("\\centering\n")
    parts.append("\\begin{tabular}{lrrr}\n")
    parts.append("\\toprule\nBlock & Rows & Complete & Rate\\\\\n\\midrule\n")
    for status in status_report(rows):
        parts.append(
            f"{latex_escape(status['block'])} & {status['rows']} & "
            f"{status['valid_complete_rows']} & {status['completion_rate']:.1f}\\\\\n"
        )
    parts.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    # Focused summaries by block.
    focus = {
        "break-even": [
            ("Tpre(s)", "T_pre_total_s"), ("B*(B/s)", "B_star_bytes_s"),
            ("H*", "break_even_epochs"), ("Speedup", "epoch_speedup"),
        ],
        "native-task": [
            ("Test acc.", "test_accuracy"), ("Total time(s)", "total_time_to_target_s"),
            ("Peak RSS(MB)", "peak_rss_mb"),
        ],
        "cpu-emulation": [
            ("Epoch(s)", "EDGaE_epoch_s"), ("Wait(s)", "gradient_wait_s"),
            ("Imbalance", "load_imbalance"),
        ],
        "network-sensitivity": [
            ("B(B/s)", "effective_B_bytes_s"), ("DGL-E(s)", "DGL_E_epoch_s"),
            ("EDGaE(s)", "EDGaE_epoch_s"), ("H*", "break_even_epochs"),
        ],
        "cache-stability": [
            ("Hit", "cache_hit_rate"), ("Jaccard", "topk_jaccard"),
            ("Gap", "boundary_gap"), ("Presample(s)", "T_presample_s"),
        ],
        "component-ablation": [
            ("Epoch(s)", "EDGaE_epoch_s"), ("Traffic(B)", "C_edg_bytes"),
            ("Peak RSS(MB)", "peak_rss_mb"),
        ],
        "partition-policy": [
            ("Vert.rep.", "replication_factor_vertices"),
            ("Edge rep.", "replication_factor_edges"),
            ("Imbalance", "load_imbalance"), ("Epoch(s)", "EDGaE_epoch_s"),
        ],
        "score-sensitivity": [
            ("Vert.rep.", "replication_factor_vertices"),
            ("Imbalance", "load_imbalance"), ("Epoch(s)", "EDGaE_epoch_s"),
        ],
    }
    for block, columns in focus.items():
        block_rows = [s for s in summaries if s["block"] == block]
        if not block_rows:
            continue
        parts.append(f"\\paragraph{{{latex_escape(block)}}}\n")
        header = " & ".join(latex_escape(label) for label, _ in columns)
        parts.append("\\noindent\\small\n")
        parts.append("\\begin{tabular}{l" + "r" * len(columns) + "}\n")
        parts.append("\\toprule\nSetting & " + header + "\\\\\n\\midrule\n")
        for summary in block_rows:
            setting = "/".join(str(summary.get(k, "")) for k in [
                "dataset", "model", "sampler", "workers", "control"
            ])
            vals = " & ".join(fmt(summary.get(key + "_mean")) for _, key in columns)
            parts.append(f"{latex_escape(setting)} & {vals}\\\\\n")
        parts.append("\\bottomrule\n\\end{tabular}\n\n")
    return "".join(parts)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    all_columns = COLUMNS + [
        c for c in rows[0] if c not in COLUMNS
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input not found: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not args.include_invalid:
        rows = [r for r in rows if as_bool(r.get("measurement_valid"))]
    if not rows:
        raise SystemExit("no valid measurement rows")

    derived = [derive(r) for r in rows]
    write_csv(Path(args.derived_output), derived)
    summaries = summarize(derived, args.bootstrap_samples, args.seed)
    write_csv(Path(args.summary_output), summaries)
    Path(args.latex_output).write_text(make_tables(derived, summaries))

    statuses = status_report(derived)
    print(f"input_rows={len(rows)} derived_rows={len(derived)} groups={len(summaries)}")
    for status in statuses:
        print(
            f"  {status['block']}: {status['valid_complete_rows']}/"
            f"{status['rows']} complete"
        )
    if args.require_complete and any(s["valid_complete_rows"] != s["rows"] for s in statuses):
        print("ERROR: required fields are missing", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
