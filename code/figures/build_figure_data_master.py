#!/usr/bin/env python3
"""Build figsource/figure_data_master.xlsx — chart-ready data for every paper figure.

Layout:
  chart_*  : wide tables (categories x series), three stacked blocks per metric:
             [Mean] / [CI95 low] / [CI95 high] -> select a block, Insert chart.
  charts_preview : native editable Excel charts referencing the chart_* blocks.
  src_*    : long-format source data (topology, config, mean, CI lo, CI hi).
  README   : sheet <-> figure mapping and usage notes.

Ground truth: ../www/icde2027-v2-summary.json (same source as the figure PDFs).
Regenerate:  python3 figsource/build_figure_data_master.py
"""
from pathlib import Path
import json
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import LineChart, BarChart, Reference
from openpyxl.chart.marker import Marker
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
S = json.loads((HERE.parent.parent / "www/icde2027-v2-summary.json").read_text())

TOPO = ["web-google", "amazon0601", "wiki-talk", "ogbn-arxiv",
        "amazon0601-full", "web-google-full", "wiki-talk-full"]
LABEL = {"web-google": "web-google 12k", "amazon0601": "amazon0601 12k",
         "wiki-talk": "wiki-talk 12k", "ogbn-arxiv": "ogbn-arxiv",
         "amazon0601-full": "amazon0601 full", "web-google-full": "web-google full",
         "wiki-talk-full": "wiki-talk full"}
POLICIES = ["random", "even", "METIS-like", "PaGraph-style", "EDGaE"]
POLICY_LABEL = {"random": "Random", "even": "Even", "METIS-like": "METIS-like",
                "PaGraph-style": "PaGraph-style", "EDGaE": "EDGaE"}
ONLINE_POLICIES = ["static-full", "LFU", "static-K16", "LRU", "degree", "static-K2", "random"]
ONLINE_LABEL = {"random": "Random", "degree": "Degree", "static-K2": "Static K=2",
                "LRU": "Online LRU", "LFU": "Online LFU", "static-K16": "Static K=16",
                "static-full": "Full-trace oracle"}
KS = [1, 2, 4, 8, 16, "full-trace"]
CONF_F = ["f=2,d=2,non-uniform", "f=4,d=2,non-uniform", "f=8,d=2,non-uniform"]
CONF_D = ["f=2,d=1,non-uniform", "f=2,d=2,non-uniform", "f=2,d=3,non-uniform"]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
MB = 1e6

TITLE_FONT = Font(bold=True, size=11, color="1F4E79")
HDR_FONT = Font(bold=True, color="FFFFFF")
HDR_FILL = PatternFill("solid", fgColor="1F4E79")
MEAN_FILL = PatternFill("solid", fgColor="DDEBF7")


def m(block, dataset, control, field):
    return S[f"{block}|{dataset}|{control}"][field]


wb = Workbook()
wb.remove(wb.active)


# ---------------------------------------------------------------- helpers --
def wide_block(ws, row, title, cat_header, cats, col_keys, col_labels, value_fn, fmt):
    """Write one wide block; return (header_row, first_data_row, last_data_row, last_col)."""
    ws.cell(row, 1, title).font = TITLE_FONT
    hr = row + 1
    ws.cell(hr, 1, cat_header).font = HDR_FONT
    ws.cell(hr, 1).fill = HDR_FILL
    for j, cl in enumerate(col_labels, 2):
        c = ws.cell(hr, j, cl)
        c.font = HDR_FONT; c.fill = HDR_FILL
        c.alignment = Alignment(horizontal="center", text_rotation=0, wrap_text=True)
    for i, cat in enumerate(cats):
        ws.cell(hr + 1 + i, 1, cat)
        for j, ck in enumerate(col_keys, 2):
            c = ws.cell(hr + 1 + i, j, value_fn(cat, ck))
            c.number_format = fmt
            if title.endswith("Mean"):
                c.fill = MEAN_FILL
    return hr, hr + 1, hr + len(cats), 1 + len(col_keys)


def line_chart(title, x_title, y_title, ws, hdr, r0, r1, c1, logy=False):
    ch = LineChart()
    ch.title = title; ch.style = 2
    ch.x_axis.title = x_title; ch.y_axis.title = y_title
    ch.height = 9; ch.width = 20
    data = Reference(ws, min_col=2, min_row=hdr, max_col=c1, max_row=r1)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=r0, max_row=r1))
    for s in ch.series:
        s.marker = Marker(symbol="circle", size=5)
        s.smooth = False
    if logy:
        ch.y_axis.scaling.logBase = 10
    return ch


def bar_chart(title, x_title, y_title, ws, hdr, r0, r1, c1):
    ch = BarChart(); ch.type = "col"; ch.grouping = "clustered"
    ch.title = title; ch.style = 2
    ch.x_axis.title = x_title; ch.y_axis.title = y_title
    ch.height = 9; ch.width = 20
    data = Reference(ws, min_col=2, min_row=hdr, max_col=c1, max_row=r1)
    ch.add_data(data, titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=r0, max_row=r1))
    ch.gapWidth = 60
    return ch


def src_sheet(name, header, rows, note):
    ws = wb.create_sheet(name)
    ws.cell(1, 1, note).font = Font(italic=True, color="555555")
    for j, h in enumerate(header, 1):
        c = ws.cell(3, j, h); c.font = HDR_FONT; c.fill = HDR_FILL
    for i, row in enumerate(rows, 4):
        for j, v in enumerate(row, 1):
            ws.cell(i, j, v)
    ws.freeze_panes = "A4"
    for j in range(1, len(header) + 1):
        ws.column_dimensions[get_column_letter(j)].width = 12
    return ws


# ------------------------------------------------- chart: Fig 2 stability --
ws = wb.create_sheet("chart_fig2_cache_stability")
cats = ["1", "2", "4", "8", "16", "full-trace"]
blocks, r = [], 1
for metric, field, mname in (("hit", "cache_hit_rate", "Cache hit rate"),
                             ("jaccard", "topk_jaccard", "Top-k Jaccard")):
    for tag, key in (("Mean", "mean"), ("CI95 low", "ci95_low"), ("CI95 high", "ci95_high")):
        blocks.append((metric, tag, wide_block(
            ws, r, f"{mname} vs K — {tag}", "K", cats, TOPO, [LABEL[d] for d in TOPO],
            lambda k, d, f=field, kk=key: m("cache-stability", d, f"K={k}", f)[kk], "0.000")))
        r += len(cats) + 3
for j in range(1, 9):
    ws.column_dimensions[get_column_letter(j)].width = 13
fig2a = line_chart("Fig 2(a)  Cache hit rate vs K", "K", "Cache hit rate",
                   ws, *blocks[0][2])
fig2b = line_chart("Fig 2(b)  Top-k Jaccard vs K", "K", "Top-k Jaccard",
                   ws, *blocks[3][2])

# ------------------------------------------------ chart: Fig 3 partition ---
ws = wb.create_sheet("chart_fig3_partition")
cats = [LABEL[d] for d in TOPO]
blocks, r = [], 1
for metric, field, mname, fmt in (("bytes", "C_edg_bytes", "Residual feature bytes (MB)", "0.00"),
                                  ("repl", "replication_factor_vertices", "Vertex replication factor", "0.000")):
    for tag, key in (("Mean", "mean"), ("CI95 low", "ci95_low"), ("CI95 high", "ci95_high")):
        blocks.append((metric, tag, wide_block(
            ws, r, f"{mname} by policy — {tag}", "Topology", cats, POLICIES,
            [POLICY_LABEL[p] for p in POLICIES],
            lambda cat, p, f=field, kk=key, dd=dict(zip(cats, TOPO)): m("partition-policy", dd[cat], p, f)[kk] / (MB if f == "C_edg_bytes" else 1),
            fmt)))
        r += len(cats) + 3
for j in range(1, 7):
    ws.column_dimensions[get_column_letter(j)].width = 14
fig3a = bar_chart("Fig 3(a,b)  Residual feature bytes by policy", "Topology", "Residual bytes (MB)",
                  ws, *blocks[0][2])
fig3b = bar_chart("Fig 3(c,d)  Vertex replication by policy", "Topology", "Vertex replication factor",
                  ws, *blocks[3][2])

# --------------------------------------------- chart: Fig 4 sensitivity ----
ws = wb.create_sheet("chart_fig4_sensitivity")
blocks, r = [], 1
for axis, confs, xs in (("fan-out", CONF_F, ["2", "4", "8"]), ("depth", CONF_D, ["1", "2", "3"])):
    for tag, key in (("Mean", "mean"), ("CI95 low", "ci95_low"), ("CI95 high", "ci95_high")):
        blocks.append((axis, tag, wide_block(
            ws, r, f"Residual bytes vs {axis} — {tag}", axis, xs, TOPO, [LABEL[d] for d in TOPO],
            lambda x, d, cc=confs, xx=xs, kk=key: S[f"sampler-sensitivity|{d}|{cc[xx.index(x)]}"]["C_edg_bytes"][kk] / MB,
            "0.00")))
        r += len(xs) + 3
for j in range(1, 9):
    ws.column_dimensions[get_column_letter(j)].width = 13
fig4a = line_chart("Fig 4(a)  Residual bytes vs fan-out (L=2)", "fan-out f", "Residual bytes (MB, log)",
                   ws, *blocks[0][2], logy=True)
fig4b = line_chart("Fig 4(b)  Residual bytes vs depth (f=2)", "depth L", "Residual bytes (MB, log)",
                   ws, *blocks[3][2], logy=True)

# ---------------------------------------------- chart: online cache --------
ws = wb.create_sheet("chart_online_cache")
cats = [LABEL[d] for d in TOPO]
blocks, r = [], 1
for tag, key in (("Mean", "mean"), ("CI95 low", "ci95_low"), ("CI95 high", "ci95_high")):
    blocks.append(wide_block(
        ws, r, f"Online-cache regimes, hit rate — {tag}", "Topology", cats,
        ONLINE_POLICIES, [ONLINE_LABEL[p] for p in ONLINE_POLICIES],
        lambda cat, p, kk=key, dd=dict(zip(cats, TOPO)): m("online-cache", dd[cat], p, "cache_hit_rate")[kk],
        "0.000"))
    r += len(cats) + 3
for j in range(1, 9):
    ws.column_dimensions[get_column_letter(j)].width = 14
chart_online = bar_chart("Online-cache regimes: hit rate", "Topology", "Cache hit rate",
                         ws, *blocks[0])

# -------------------------------------------------- chart: two-tier --------
ws = wb.create_sheet("chart_two_tier")
cats = ["0", "0.25", "0.5", "0.75", "1"]
blocks, r = [], 1
for tag, key in (("Mean", "mean"), ("CI95 low", "ci95_low"), ("CI95 high", "ci95_high")):
    blocks.append(wide_block(
        ws, r, f"Two-tier hit rate vs beta — {tag}", "beta (mutable fraction)", cats,
        TOPO, [LABEL[d] for d in TOPO],
        lambda b, d, kk=key: S[f"two-tier|{d}|beta={float(b)}"]["cache_hit_rate"][kk], "0.000"))
    r += len(cats) + 3
for j in range(1, 9):
    ws.column_dimensions[get_column_letter(j)].width = 13
chart_2t = line_chart("Two-tier cache: hit rate vs beta", "beta (mutable fraction)", "Cache hit rate",
                      ws, *blocks[0])

# --------------------------------------------- chart: Fig 10 physical ------
ws = wb.create_sheet("chart_fig10_physical_hit")
RATIO = {"web-google": {"EDGaE": 1.00, "DGL-Degree": 1 / 1.04, "DGL-Random": 1 / 1.99},
         "wiki-talk":  {"EDGaE": 1.00, "DGL-Degree": 1 / 1.37, "DGL-Random": 1 / 1.72}}
cats = ["web-google", "wiki-talk"]
pols = ["EDGaE", "DGL-Degree", "DGL-Random"]
blk = wide_block(ws, 1, "Physical-study hit rate (EDGaE = 1) — Mean", "Dataset", cats,
                 pols, pols, lambda d, p: RATIO[d][p], "0.00")
for j in range(1, 5):
    ws.column_dimensions[get_column_letter(j)].width = 13
chart_phys = bar_chart("Fig 10  Physical study: hit rate (EDGaE = 1)", "Dataset", "Hit rate (EDGaE = 1)",
                       ws, *blk)

# ------------------------------------------------------ charts preview -----
prev = wb.create_sheet("charts_preview")
prev.cell(1, 1, "预置原生图表（可直接改样式/复制到论文流程）：数据引用自各 chart_* 工作表，修改数据自动更新。").font = Font(bold=True)
for i, ch in enumerate([fig2a, fig2b, fig3a, fig3b, fig4a, fig4b, chart_online, chart_2t, chart_phys]):
    col = "ABCDEFGHIJKLMNOPQR"[(i % 2) * 9]
    anchor = f"{col}{3 + (i // 2) * 19}"
    prev.add_chart(ch, anchor)

# ---------------------------------------------------------- src sheets -----
rows = []
for d in TOPO:
    for k in KS:
        h, j = m("cache-stability", d, f"K={k}", "cache_hit_rate"), m("cache-stability", d, f"K={k}", "topk_jaccard")
        rows.append([d, LABEL[d], k, h["mean"], h["ci95_low"], h["ci95_high"],
                     j["mean"], j["ci95_low"], j["ci95_high"]])
src_sheet("src_cache_stability", ["topology", "topology_label", "K", "hit_mean", "hit_ci_lo",
          "hit_ci_hi", "jaccard_mean", "jaccard_ci_lo", "jaccard_ci_hi"], rows,
          "Fig icde-v2-cache-stability.pdf — hit rate + top-k Jaccard vs K; 5 seeds, bootstrap 95% CI; K=full-trace = oracle.")

rows = []
for d in TOPO:
    for p in POLICIES:
        b, v = m("partition-policy", d, p, "C_edg_bytes"), m("partition-policy", d, p, "replication_factor_vertices")
        rows.append([d, LABEL[d], p, POLICY_LABEL[p], b["mean"] / MB, b["ci95_low"] / MB, b["ci95_high"] / MB,
                     v["mean"], v["ci95_low"], v["ci95_high"]])
src_sheet("src_partition", ["topology", "topology_label", "policy", "policy_label",
          "residual_MB_mean", "residual_MB_ci_lo", "residual_MB_ci_hi",
          "replication_mean", "replication_ci_lo", "replication_ci_hi"], rows,
          "Fig icde-v2-partition.pdf — residual bytes + vertex replication by policy; MB = bytes/1e6; 5 seeds, bootstrap 95% CI.")

rows = []
for d in TOPO:
    for axis, confs, xs in (("fan-out", CONF_F, [2, 4, 8]), ("depth", CONF_D, [1, 2, 3])):
        for c, x in zip(confs, xs):
            e, ag = S[f"sampler-sensitivity|{d}|{c}"]["C_edg_bytes"], S[f"sampler-sensitivity|{d}|{c}"].get("assignment_agreement", {})
            rows.append([d, LABEL[d], axis, c, x, e["mean"] / MB, e["ci95_low"] / MB, e["ci95_high"] / MB, ag.get("mean")])
src_sheet("src_sensitivity", ["topology", "topology_label", "axis", "config", "f_or_L",
          "residual_MB_mean", "residual_MB_ci_lo", "residual_MB_ci_hi", "placement_agreement"], rows,
          "Fig icde-v2-sensitivity.pdf — residual bytes vs fan-out (L=2) and depth (f=2), non-uniform sampling; 5 seeds, bootstrap 95% CI.")

rows = []
for d in TOPO:
    for p in ONLINE_POLICIES:
        e = S[f"online-cache|{d}|{p}"]["cache_hit_rate"]
        rows.append([d, LABEL[d], p, ONLINE_LABEL[p], e["mean"], e["ci95_low"], e["ci95_high"]])
src_sheet("src_online_cache", ["topology", "topology_label", "policy", "policy_label",
          "hit_mean", "hit_ci_lo", "hit_ci_hi"], rows,
          "Fig icde-v2-online-cache.pdf — 7 cache regimes at equal total record budget; 5 seeds, bootstrap 95% CI.")

rows = []
for d in TOPO:
    for b in BETAS:
        e = S[f"two-tier|{d}|beta={float(b)}"]["cache_hit_rate"]
        rows.append([d, LABEL[d], b, e["mean"], e["ci95_low"], e["ci95_high"]])
src_sheet("src_two_tier", ["topology", "topology_label", "beta", "hit_mean", "hit_ci_lo", "hit_ci_hi"], rows,
          "Fig icde-v2-two-tier.pdf — hit rate vs mutable fraction beta (0=static, 1=online LFU); 5 seeds, bootstrap 95% CI.")

rows = [[d, p, RATIO[d][p]] for d in ("web-google", "wiki-talk") for p in pols]
src_sheet("src_physical_relative_hit", ["dataset", "policy", "hit_vs_EDGaE"], rows,
          "Fig fig10.pdf (Section VII) — derived from paper-stated ratios; absolute per-policy values are not in the artifact. Coverage: web-google 49.53%, wiki-talk 42.05%.")

# ----------------------------------------------------------- palette -------
ws = wb.create_sheet("palette")
for j, h in enumerate(["series", "hex_color"], 1):
    c = ws.cell(1, j, h); c.font = HDR_FONT; c.fill = HDR_FILL
for i, (s, col) in enumerate([("web-google 12k", "#0072B2"), ("amazon0601 12k", "#E69F00"),
        ("wiki-talk 12k", "#009E73"), ("ogbn-arxiv", "#CC79A7"), ("amazon0601 full", "#D55E00"),
        ("web-google full", "#56B4E9"), ("wiki-talk full", "#6B6B6B"), ("Random", "#B8B8B8"),
        ("Even", "#56B4E9"), ("METIS-like", "#009E73"), ("PaGraph-style", "#E69F00"),
        ("EDGaE", "#D55E00")], 2):
    ws.cell(i, 1, s); ws.cell(i, 2, col)
ws.column_dimensions["A"].width = 20; ws.column_dimensions["B"].width = 12

# ------------------------------------------------------------- README ------
info = [
    ["figure_data_master.xlsx — 论文全部数据图的作图数据（Excel 快速作图版）"],
    ["数据唯一真值：www/icde2027-v2-summary.json；重生成：python3 figsource/build_figure_data_master.py"],
    [],
    ["快速作图方法：打开对应 chart_* 工作表 → 选中浅蓝底色的 Mean 块（含系列名表头）→ 插入 → 折线图/柱状图。"],
    ["误差带：同表下方有对应 CI95 low / CI95 high 块，可添加为面积图或用作自定义误差线。"],
    ["charts_preview 已预置 9 张原生可编辑图表（引用 chart_* 数据，改数自动更新），可直接调整样式。"],
    [],
    ["工作表", "对应图", "内容"],
    ["chart_fig2_cache_stability", "icde-v2-cache-stability.pdf", "(a) 命中率 vs K；(b) Jaccard vs K（7 拓扑）"],
    ["chart_fig3_partition", "icde-v2-partition.pdf", "残余字节 MB / 顶点复制因子 × 5 策略（7 拓扑）"],
    ["chart_fig4_sensitivity", "icde-v2-sensitivity.pdf", "残余字节 vs fan-out / depth（对数轴）"],
    ["chart_online_cache", "icde-v2-online-cache.pdf", "7 种缓存策略命中率（论文当前未引用，备用）"],
    ["chart_two_tier", "icde-v2-two-tier.pdf", "命中率 vs β（论文当前未引用，备用）"],
    ["chart_fig10_physical_hit", "fig10.pdf", "§VII 物理实验相对命中率（EDGaE=1）"],
    ["src_*", "—", "长表源数据（每行=拓扑×配置，含均值与 CI）"],
    ["palette", "—", "Okabe-Ito 色板参考"],
    [],
    ["统计口径：5 种子均值；CI = bootstrap 95%；MB = bytes/1e6；K=full-trace 为 oracle。"],
]
ws = wb.create_sheet("README", 0)
for i, row in enumerate(info, 1):
    for j, v in enumerate(row, 1):
        ws.cell(i, j, v)
ws["A1"].font = Font(bold=True, size=12)
for j in range(1, 4):
    c = ws.cell(8, j); c.font = HDR_FONT; c.fill = HDR_FILL
ws.column_dimensions["A"].width = 30
ws.column_dimensions["B"].width = 34
ws.column_dimensions["C"].width = 52

out = HERE / "figure_data_master.xlsx"
wb.save(out)
print("written:", out)
for s in wb.worksheets:
    print(f"  {s.title}: {s.max_row} rows x {s.max_column} cols")
