#!/usr/bin/env python3
"""Audit the manuscript against its data artifacts before every submission.

Checks:
  1. stale language patterns must not appear in any section source;
  2. headline numeric ranges must equal values recomputed from
     www/icde2027-v2-summary.json at full precision;
  3. generated tables must be newer than the newest results CSV;
  4. the compile log has no undefined references and the body fits 12 pages.

Exit code 0 means the manuscript is consistent with its artifacts.
"""

from __future__ import annotations

import csv as csvmod
import json
import math
import re
import statistics as st
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "ICDE"
WWW = ROOT / "www"
SECTIONS = sorted(TEX.glob("sec*.tex")) + [TEX / "IEEEtran.tex"]
SUMMARY = WWW / "icde2027-v2-summary.json"
LOG = TEX / "IEEEtran.log"
TABLES = list(TEX.glob("tab-*.tex"))
RESULT_CSVS = sorted(WWW.glob("icde2027-v2-*-results.csv"))

STALE_PATTERNS = [
    r"this revision", r"previous version", r"new in this", r"now shows",
    r"revised evaluation", r"five feasible", r"neighbour", r"favours",
    r"has so far", r"Belady-style", r"empirical setting", r"truthful data",
    r"Rabbit\-Pi", r"previously conflated",
    r"identified for the previous", r"89\\% fallbacks", r"17\\.4--53\\.4",
    r"650k", r"0\\.512--0\\.753", r"the retained",
]

DS = ["web-google", "amazon0601", "wiki-talk", "ogbn-arxiv",
      "amazon0601-full", "web-google-full", "wiki-talk-full"]
FEASIBLE = ["web-google", "amazon0601", "ogbn-arxiv",
            "amazon0601-full", "web-google-full", "wiki-talk-full"]


def check_language(failures):
    for path in SECTIONS:
        text = path.read_text()
        for pattern in STALE_PATTERNS:
            for hit in re.finditer(pattern, text, flags=re.IGNORECASE):
                failures.append(
                    f"stale language {hit.group(0)!r} in {path.name}")


def mean(S, block, dataset, control, field):
    return S[f"{block}|{dataset}|{control}"][field]["mean"]


def check_numbers(S, failures):
    checks = []
    red = [(mean(S, "partition-policy", d, "even", "C_edg_bytes")
            - mean(S, "partition-policy", d, "EDGaE", "C_edg_bytes"))
           / mean(S, "partition-policy", d, "even", "C_edg_bytes") * 100
           for d in FEASIBLE]
    checks.append(("11.8--46.1 byte-reduction range",
                   (round(min(red), 1), round(max(red), 1)) == (11.8, 46.1)))
    ratios = [mean(S, "partition-policy", d, "even",
                   "replication_factor_vertices")
              / mean(S, "partition-policy", d, "EDGaE",
                     "replication_factor_vertices") for d in FEASIBLE]
    checks.append(("up-to-3.5x replication reduction",
                   round(max(ratios), 1) == 3.5))
    j2 = [mean(S, "cache-stability", d, "K=2", "topk_jaccard") for d in DS]
    checks.append(("K=2 Jaccard 0.285--0.991",
                   (round(min(j2), 3), round(max(j2), 3)) == (0.285, 0.991)))
    h16 = [mean(S, "cache-stability", d, "K=16", "cache_hit_rate") for d in DS]
    h2 = [mean(S, "cache-stability", d, "K=2", "cache_hit_rate") for d in DS]
    checks.append(("K=2 hits 0.172--0.343",
                   (round(min(h2), 3), round(max(h2), 3)) == (0.172, 0.343)))
    checks.append(("K=16 hits 0.320--0.518",
                   (round(min(h16), 3), round(max(h16), 3)) == (0.320, 0.518)))
    checks.append(("K=16 equals oracle on all seven",
                   all(mean(S, "cache-stability", d, "K=16", "cache_hit_rate")
                       == mean(S, "cache-stability", d, "K=full-trace",
                               "cache_hit_rate") for d in DS)))
    gstatic = [mean(S, "two-tier", d, "beta=0.0", "cache_hit_rate") for d in DS]
    lfu = [mean(S, "online-cache", d, "LFU", "cache_hit_rate") for d in DS]
    checks.append(("centralized static beats LFU on 7/7",
                   all(g > l for g, l in zip(gstatic, lfu))))
    checks.append(("centralized-static range 0.512--0.737",
                   (round(min(gstatic), 3), round(max(gstatic), 3))
                   == (0.512, 0.737)))
    lfu_wins = sum(1 for d in DS
                   if mean(S, "online-cache", d, "LFU", "cache_hit_rate")
                   > mean(S, "online-cache", d, "static-K16",
                          "cache_hit_rate"))
    checks.append(("LFU beats per-worker K16 on 6/7", lfu_wins == 6))
    fb3 = {d: mean(S, "sampler-sensitivity", d, "f=2,d=3,non-uniform",
                   "fallbacks") / 18 for d in DS}
    checks.append(("depth-3 fallbacks 92/89/22 percent",
                   (round(fb3["wiki-talk-full"]), round(fb3["ogbn-arxiv"]),
                    round(fb3["wiki-talk"])) == (92, 89, 22)))
    agreement = [mean(S, "sampler-sensitivity", d, c, "assignment_agreement")
                 for d in DS for c in ("f=4,d=2,non-uniform",
                                       "f=8,d=2,non-uniform",
                                       "f=2,d=2,uniform")]
    checks.append(("sampler-invariance agreement exactly 1.00 (21/21)",
                   all(a == 1.0 for a in agreement)))
    growth = [mean(S, "sampler-sensitivity", d, "f=8,d=2,non-uniform",
                   "C_edg_bytes")
              / mean(S, "sampler-sensitivity", d, "f=2,d=2,non-uniform",
                     "C_edg_bytes") for d in DS]
    checks.append(("f8/f2 byte growth 2.0--4.9x",
                   (round(min(growth), 1), round(max(growth), 1))
                   == (2.0, 4.9)))

    # Later files override earlier ones per (dataset, seed), mirroring the
    # analyzer's dedup so complete-graph rows replace prefix-era rows.
    keyed = {}
    for name in ("icde2027-v2-preprocess-results.csv",
                 "icde2027-v2-web-google-complete-results.csv",
                 "icde2027-v2-wiki-talk-complete-results.csv"):
        path = WWW / name
        if not path.exists():
            failures.append(f"missing canonical CSV {name}")
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for r in csvmod.DictReader(f):
                if r["block"] == "preprocess":
                    keyed[(r["dataset"], r["seed"])] = r
    rows = defaultdict(list)
    for (dataset, _), r in keyed.items():
        rows[dataset].append(r)
    h10, h100, h1000, ships = [], [], [], []
    for d in DS:
        rs = rows[d]
        tt = st.mean(float(r["T_pre_total_s"]) for r in rs)
        dc = st.mean(float(r["C_srv_bytes"]) - float(r["C_edg_bytes"])
                     for r in rs)
        V = int(mean(S, "partition-policy", d, "EDGaE", "graph_vertices"))
        E = int(mean(S, "partition-policy", d, "EDGaE", "graph_edges"))
        rv = mean(S, "preprocess", d, "EDGaE-K16",
                  "replication_factor_vertices")
        re_ = mean(S, "preprocess", d, "EDGaE-K16",
                   "replication_factor_edges")
        ship = rv * V * 8 + re_ * E * 8
        ships.append(ship / 1e6)
        hs = [math.ceil((tt * b + ship) / dc)
              for b in (1.25e6, 1.25e7, 1.25e8)]
        h10.append(hs[0]); h100.append(hs[1]); h1000.append(hs[2])
    checks.append(("ship range 0.2--215.6 MB",
                   (round(min(ships), 1), round(max(ships), 1))
                   == (0.2, 215.6)))
    checks.append(("dense wiki H* = 67/114/586",
                   (h10[-1], h100[-1], h1000[-1]) == (67, 114, 586)))
    checks.append(("web-google full H* = 7/12/54",
                   (h10[5], h100[5], h1000[5]) == (7, 12, 54)))

    for name, ok in checks:
        if not ok:
            failures.append(f"numeric claim mismatch: {name}")
    return checks


def check_freshness(failures):
    if not RESULT_CSVS:
        failures.append("no result CSVs found under www/")
        return
    newest_csv = max(p.stat().st_mtime for p in RESULT_CSVS)
    for table in TABLES:
        if table.stat().st_mtime < newest_csv:
            failures.append(
                f"stale generated table {table.name}: older than result "
                f"CSVs; rerun experiments/analyze_icde_v2.py")




GPU_CSV = ROOT / "gpu_experiments" / "results" / "gpu_native_results.csv"
GPU_TABLE = TEX / "tab-gpu-native.tex"


def check_gpu(failures):
    """Native A800 claims: parity, accuracy, byte reduction, fallbacks."""
    if not GPU_CSV.exists():
        failures.append("missing gpu_experiments/results/gpu_native_results.csv")
        return
    import csv as csvmod
    rows = list(csvmod.DictReader(GPU_CSV.open(newline="", encoding="utf-8")))
    gcn = [r for r in rows if r["experiment"] == "E1_fullbatch_gcn"]
    srv = [r for r in rows if r["experiment"] == "E2_E3_sage"
           and r["mode"] == "server"]
    edg = [r for r in rows if r["experiment"] == "E2_E3_sage"
           and r["mode"] == "edgae"]
    checks = [
        ("GPU: 3 seeds per mode", len(gcn) == len(srv) == len(edg) == 3),
        ("GPU: GCN test 68.0 +/- 0.2",
         abs(100 * statistics.mean(float(r["test_acc"]) for r in gcn) - 68.0) < 0.05),
        ("GPU: Proposition-4 parity exactly 0",
         all(abs(float(s["best_val"]) - float(e["best_val"])) == 0.0
             for s, e in zip(srv, edg))),
        ("GPU: miss bytes reduced 24.3%",
         abs((statistics.mean(float(r["miss_bytes"]) for r in srv)
              - statistics.mean(float(r["miss_bytes"]) for r in edg))
             / statistics.mean(float(r["miss_bytes"]) for r in srv) - 0.243)
         < 0.005),
        ("GPU: native fallbacks 61818/90941 (68%)",
         all(r["fallbacks"] == "61818" for r in edg)),
        ("GPU table newer than results CSV",
         GPU_TABLE.exists() and GPU_TABLE.stat().st_mtime >= GPU_CSV.stat().st_mtime),
    ]
    for name, ok in checks:
        if not ok:
            failures.append(f"GPU claim mismatch: {name}")
    return checks

def check_log(failures):
    if not LOG.exists():
        failures.append("missing compile log; build the paper first")
        return
    text = LOG.read_text(errors="ignore")
    if re.search(r"Reference .* undefined|Citation .* undefined", text):
        failures.append("compile log reports undefined references")
    m = re.search(r"Output written on .*? \((\d+) pages", text)
    if not m:
        failures.append("cannot determine page count from log")
    else:
        pages = int(m.group(1))
        if pages > 14:
            failures.append(f"document is {pages} pages; body exceeds 12")


def main():
    failures = []
    if not SUMMARY.exists():
        sys.exit(f"missing {SUMMARY}")
    S = json.loads(SUMMARY.read_text())
    check_language(failures)
    checks = check_numbers(S, failures)
    check_freshness(failures)
    gpu_checks = check_gpu(failures)
    check_log(failures)
    print(f"consistency audit: {len(checks) + len(gpu_checks or [])} numeric checks, "
          f"{len(failures)} failures")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for f in failures:
        print(f"  [FAIL] {f}")
    if failures:
        sys.exit(1)
    print("AUDIT PASS")


if __name__ == "__main__":
    main()
