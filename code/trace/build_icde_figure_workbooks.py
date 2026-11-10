#!/usr/bin/env python3
"""Build editable Excel data/chart sources for the ICDE simulation figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.chart.marker import Marker
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


ROOT = Path(__file__).resolve().parents[1]
WWW_DIR = ROOT / "www"
OUTPUT_DIR = ROOT / "ICDE" / "figsource"

SUMMARY_FILES = {
    "web-google": WWW_DIR / "www2027-real-trace-web-google-summary-v3.csv",
    "amazon0601": WWW_DIR / "www2027-real-trace-amazon-summary-v2.csv",
    "wiki-talk": WWW_DIR / "www2027-real-trace-wiki-summary-v2.csv",
    "ogbn-arxiv": WWW_DIR / "www2027-real-trace-arxiv-summary-v2.csv",
}

DATASET_COLORS = {
    "web-google": "6195CE",
    "amazon0601": "82C47C",
    "wiki-talk": "EA7E7E",
    "ogbn-arxiv": "E2B83D",
}

POLICY_COLORS = {
    "even": "DAD6D7",
    "random": "B0D4EC",
    "METIS-like": "FFE699",
    "PaGraph-style": "EA7E7E",
    "EDGaE": "82C47C",
}

K_CONTROLS = ["K=1", "K=2", "K=4", "K=8", "K=16", "K=full-trace"]
K_LABELS = ["1", "2", "4", "8", "16", "∞"]
POLICIES = ["even", "random", "METIS-like", "PaGraph-style", "EDGaE"]
COMPONENT_CONTROLS = ["DGL-E", "+R", "+RC", "+RCS", "+RCSP"]
SCORE_CONTROLS = ["equal-weights", "locality-heavy", "balance-heavy", "memory-heavy"]
SCORE_LABELS = ["Equal", "Locality", "Balance", "Memory"]

TITLE_FILL = PatternFill("solid", fgColor="1F4E78")
SECTION_FILL = PatternFill("solid", fgColor="D9EAF7")
HEADER_FILL = PatternFill("solid", fgColor="D9EAF7")
SUBHEADER_FILL = PatternFill("solid", fgColor="E2F0D9")
NOTE_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN_GRAY = Side(style="thin", color="B7C9D6")
BOX = Border(left=THIN_GRAY, right=THIN_GRAY, top=THIN_GRAY, bottom=THIN_GRAY)


def load_rows() -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = {}
    for dataset, path in SUMMARY_FILES.items():
        with path.open(newline="", encoding="utf-8") as stream:
            result[dataset] = list(csv.DictReader(stream))
    return result


def find_row(rows: Iterable[dict[str, str]], block: str, control: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("block") == block and row.get("control") == control]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {block}/{control}, found {len(matches)}")
    return matches[0]


def number(value: str, default: float | None = None) -> float | None:
    if value == "":
        return default
    return float(value)


def setup_workbook(name: str, purpose: str, source_paths: list[Path]) -> Workbook:
    workbook = Workbook()
    workbook.calculation.fullCalcOnLoad = True
    workbook.properties.title = name
    workbook.properties.subject = "ICDE 2027 simulation figure source data"
    workbook.properties.creator = "EDGaE authors"
    workbook.properties.description = purpose

    readme = workbook.active
    readme.title = "README"
    readme.sheet_properties.tabColor = "1F4E78"
    readme.sheet_view.showGridLines = False
    readme["A1"] = name
    readme["A1"].font = Font(size=16, bold=True, color="FFFFFF")
    readme["A1"].fill = TITLE_FILL
    readme.merge_cells("A1:F1")
    readme["A1"].alignment = Alignment(horizontal="left", vertical="center")
    readme.row_dimensions[1].height = 28

    items = [
        ("Purpose", purpose),
        ("Evidence level", "Trace-driven simulation on real graph topologies"),
        ("Seeds", "Five independent sampler seeds per configuration"),
        ("SourceData sheet", "Auditable raw means, dispersion, and confidence intervals used by the figure"),
        ("ChartData sheet", "Formula-backed compact series tables referenced by native Excel charts"),
        ("Charts sheet", "Editable native Excel charts corresponding to the manuscript figure"),
        ("Generated", "2026-09-27"),
    ]
    for row, (label, value) in enumerate(items, start=3):
        readme.cell(row, 1, label).font = Font(bold=True)
        readme.cell(row, 2, value)
        readme.merge_cells(start_row=row, start_column=2, end_row=row, end_column=6)

    readme["A11"] = "Source files"
    readme["A11"].font = Font(bold=True, size=12)
    for row, path in enumerate(source_paths, start=12):
        readme.cell(row, 1, str(path.relative_to(ROOT)))

    readme["A18"] = "Notes"
    readme["A18"].font = Font(bold=True, size=12)
    notes = [
        "Confidence intervals are stored in SourceData even where the manuscript chart displays only means.",
        "ChartData cells use INDEX/MATCH formulas against SourceData; edit SourceData to update charts.",
        "Excel recalculates formulas and refreshes charts when the workbook is opened.",
    ]
    for row, note in enumerate(notes, start=19):
        readme.cell(row, 1, note)
        readme.merge_cells(start_row=row, start_column=1, end_row=row, end_column=6)
        readme.cell(row, 1).alignment = Alignment(wrap_text=True, vertical="top")
        readme.cell(row, 1).fill = NOTE_FILL
        readme.row_dimensions[row].height = 30

    for row in range(3, 10):
        readme.cell(row, 1).fill = SUBHEADER_FILL
    for col, width in zip("ABCDEF", [24, 28, 24, 24, 24, 24]):
        readme.column_dimensions[col].width = width

    return workbook


def style_table_header(ws, row: int, min_col: int, max_col: int) -> None:
    for col in range(min_col, max_col + 1):
        cell = ws.cell(row, col)
        cell.fill = HEADER_FILL
        cell.font = Font(bold=True, color="17365D")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BOX
    ws.row_dimensions[row].height = 32


def style_data_block(ws, min_row: int, max_row: int, min_col: int, max_col: int) -> None:
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            cell = ws.cell(row, col)
            cell.border = BOX
            cell.alignment = Alignment(vertical="center")


def add_title(ws, cell: str, text: str, span: int = 6) -> None:
    ws[cell] = text
    ws[cell].font = Font(size=13, bold=True, color="17365D")
    row = ws[cell].row
    col = ws[cell].column
    ws.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + span - 1)
    ws[cell].fill = SECTION_FILL
    ws[cell].alignment = Alignment(vertical="center")


def format_range(ws, min_row: int, max_row: int, col: int, fmt: str) -> None:
    for row in range(min_row, max_row + 1):
        ws.cell(row, col).number_format = fmt


def style_line_series(chart: LineChart, colors: dict[str, str]) -> None:
    for series, color in zip(chart.series, colors.values()):
        series.graphicalProperties.line = LineProperties(solidFill=color, w=19050)
        series.marker = Marker(symbol="circle", size=5)
        series.marker.graphicalProperties.solidFill = color
        series.marker.graphicalProperties.line = LineProperties(solidFill="FFFFFF", w=6350)
        series.smooth = False


def configure_line_chart(
    chart: LineChart,
    title: str,
    x_title: str,
    y_title: str,
    y_format: str,
    min_scale: float | None = None,
    max_scale: float | None = None,
) -> None:
    chart.title = title
    chart.style = 2
    chart.x_axis.title = x_title
    chart.y_axis.title = y_title
    chart.y_axis.numFmt = y_format
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.legend.position = "b"
    chart.height = 7.4
    chart.width = 12.8
    if min_scale is not None:
        chart.y_axis.scaling.min = min_scale
    if max_scale is not None:
        chart.y_axis.scaling.max = max_scale


def configure_bar_chart(chart: BarChart, title: str, x_title: str, y_title: str, y_format: str) -> None:
    chart.type = "col"
    chart.grouping = "clustered"
    chart.title = title
    chart.style = 2
    chart.x_axis.title = x_title
    chart.y_axis.title = y_title
    chart.y_axis.numFmt = y_format
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.legend.position = "b"
    chart.height = 7.4
    chart.width = 12.8
    chart.gapWidth = 55
    chart.overlap = -8


def style_bar_series(chart: BarChart, colors: dict[str, str]) -> None:
    for series, color in zip(chart.series, colors.values()):
        series.graphicalProperties.solidFill = color
        series.graphicalProperties.line = LineProperties(solidFill="303030", w=9525)


def build_cache_workbook(rows_by_dataset: dict[str, list[dict[str, str]]]) -> Path:
    name = "icde-simulation-cache-stability.xlsx"
    workbook = setup_workbook(
        "ICDE Simulation Cache Stability Source",
        "Editable data and charts for Fig. 3: cache hit rate and top-k hot-set stability versus pre-sampling rounds K.",
        list(SUMMARY_FILES.values()),
    )
    source = workbook.create_sheet("SourceData")
    chart_data = workbook.create_sheet("ChartData")
    charts = workbook.create_sheet("Charts")

    headers = [
        "Key", "Dataset", "K control", "K", "Hit rate mean", "Hit rate CI low",
        "Hit rate CI high", "Hit rate std", "Top-k Jaccard mean", "Jaccard CI low",
        "Jaccard CI high", "Jaccard std", "Residual feature bytes", "Residual MB",
        "Server feature bytes", "Byte saving",
    ]
    source["A1"] = "Source data: cache stability"
    source["A1"].font = Font(size=13, bold=True)
    source.append(headers)
    source._current_row = 3
    row_index = 3
    dataset_order = list(SUMMARY_FILES)
    for dataset in dataset_order:
        for control, label in zip(K_CONTROLS, K_LABELS):
            row = find_row(rows_by_dataset[dataset], "cache-stability", control)
            source.cell(row_index, 1, f'=B{row_index}&"|"&C{row_index}')
            source.cell(row_index, 2, dataset)
            source.cell(row_index, 3, control)
            source.cell(row_index, 4, label)
            source.cell(row_index, 5, number(row["cache_hit_rate_mean"]))
            source.cell(row_index, 6, number(row["cache_hit_rate_ci95_low"]))
            source.cell(row_index, 7, number(row["cache_hit_rate_ci95_high"]))
            source.cell(row_index, 8, number(row["cache_hit_rate_std"]))
            source.cell(row_index, 9, number(row["topk_jaccard_mean"]))
            source.cell(row_index, 10, number(row["topk_jaccard_ci95_low"]))
            source.cell(row_index, 11, number(row["topk_jaccard_ci95_high"]))
            source.cell(row_index, 12, number(row["topk_jaccard_std"]))
            source.cell(row_index, 13, number(row["C_edg_bytes_mean"]))
            source.cell(row_index, 14, f"=M{row_index}/1048576")
            source.cell(row_index, 15, number(row["C_srv_bytes_mean"]))
            source.cell(row_index, 16, f"=1-M{row_index}/O{row_index}")
            row_index += 1

    last_source_row = row_index - 1
    style_table_header(source, 2, 1, len(headers))
    style_data_block(source, 3, last_source_row, 1, len(headers))
    source.auto_filter.ref = f"A2:P{last_source_row}"
    source.freeze_panes = "C3"
    for col, width in zip(range(1, len(headers) + 1), [20, 16, 14, 8, 13, 13, 13, 12, 14, 13, 13, 12, 20, 13, 20, 12]):
        source.column_dimensions[get_column_letter(col)].width = width
    for col in range(5, 13):
        format_range(source, 3, last_source_row, col, "0.000")
    format_range(source, 3, last_source_row, 13, "#,##0")
    format_range(source, 3, last_source_row, 14, "0.00")
    format_range(source, 3, last_source_row, 15, "#,##0")
    format_range(source, 3, last_source_row, 16, "0.0%")
    source.sheet_properties.tabColor = "D9EAF7"

    chart_data["A1"] = "Formula-backed chart series"
    chart_data["A1"].font = Font(size=13, bold=True)
    chart_data["A3"] = "Cache hit rate"
    chart_data["A4"] = "K"
    for col, dataset in enumerate(dataset_order, start=2):
        chart_data.cell(4, col, dataset)
    chart_data.cell(4, 7, "Control helper")
    for row, label in enumerate(K_LABELS, start=5):
        chart_data.cell(row, 1, label)
        chart_data.cell(row, 7, f'=IF(A{row}="∞","K=full-trace","K="&TEXT(A{row},"0"))')
        for col in range(2, 6):
            letter = get_column_letter(col)
            chart_data.cell(
                row,
                col,
                f'=INDEX(SourceData!$E$3:$E${last_source_row},MATCH({letter}$4&"|"&$G{row},SourceData!$A$3:$A${last_source_row},0))',
            )

    chart_data["A13"] = "Top-k Jaccard overlap"
    for col, dataset in enumerate(dataset_order, start=2):
        chart_data.cell(14, col, dataset)
    chart_data.cell(14, 1, "K")
    chart_data.cell(14, 7, "Control helper")
    for row, label in enumerate(K_LABELS, start=15):
        chart_data.cell(row, 1, label)
        chart_data.cell(row, 7, f'=IF(A{row}="∞","K=full-trace","K="&TEXT(A{row},"0"))')
        for col in range(2, 6):
            letter = get_column_letter(col)
            chart_data.cell(
                row,
                col,
                f'=INDEX(SourceData!$I$3:$I${last_source_row},MATCH({letter}$14&"|"&$G{row},SourceData!$A$3:$A${last_source_row},0))',
            )

    style_table_header(chart_data, 4, 1, 7)
    style_table_header(chart_data, 14, 1, 7)
    style_data_block(chart_data, 5, 10, 1, 7)
    style_data_block(chart_data, 15, 20, 1, 7)
    chart_data["A3"].font = Font(bold=True, size=12, color="17365D")
    chart_data["A13"].font = Font(bold=True, size=12, color="17365D")
    chart_data["H3"] = "Helper cells are not chart series."
    chart_data["H3"].font = Font(italic=True, color="666666")
    chart_data.column_dimensions["A"].width = 14
    for col in range(2, 7):
        chart_data.column_dimensions[get_column_letter(col)].width = 16
    chart_data.column_dimensions["G"].width = 18
    chart_data.sheet_properties.tabColor = "E2F0D9"
    chart_data.freeze_panes = "B5"

    hit_chart = LineChart()
    hit_data = Reference(chart_data, min_col=2, max_col=5, min_row=4, max_row=10)
    hit_cats = Reference(chart_data, min_col=1, min_row=5, max_row=10)
    hit_chart.add_data(hit_data, titles_from_data=True)
    hit_chart.set_categories(hit_cats)
    configure_line_chart(hit_chart, "Cache hit rate by pre-sampling rounds", "Pre-sampling rounds K", "Cache hit rate", "0%", 0, 1)
    style_line_series(hit_chart, DATASET_COLORS)
    charts.add_chart(hit_chart, "A1")

    jaccard_chart = LineChart()
    jaccard_data = Reference(chart_data, min_col=2, max_col=5, min_row=14, max_row=20)
    jaccard_cats = Reference(chart_data, min_col=1, min_row=15, max_row=20)
    jaccard_chart.add_data(jaccard_data, titles_from_data=True)
    jaccard_chart.set_categories(jaccard_cats)
    configure_line_chart(jaccard_chart, "Top-k Jaccard by pre-sampling rounds", "Pre-sampling rounds K", "Top-k Jaccard", "0%", 0, 1)
    style_line_series(jaccard_chart, DATASET_COLORS)
    charts.add_chart(jaccard_chart, "J1")

    charts.sheet_view.showGridLines = False
    charts["A19"] = "Source tables: SourceData and ChartData. Confidence intervals remain available in SourceData."
    charts["A19"].font = Font(italic=True)
    charts.sheet_properties.tabColor = "FFE699"
    charts.column_dimensions["A"].width = 18

    workbook.active = workbook.index(charts)
    output = OUTPUT_DIR / name
    workbook.save(output)
    return output


def build_partition_workbook(rows_by_dataset: dict[str, list[dict[str, str]]]) -> Path:
    name = "icde-simulation-partition-feasibility.xlsx"
    workbook = setup_workbook(
        "ICDE Simulation Partition Feasibility Source",
        "Editable data and charts for Fig. 4: residual feature bytes and vertex replication under equal memory budgets.",
        list(SUMMARY_FILES.values()),
    )
    source = workbook.create_sheet("SourceData")
    chart_data = workbook.create_sheet("ChartData")
    charts = workbook.create_sheet("Charts")
    headers = [
        "Key", "Dataset", "Policy", "Vertex replication mean", "Vertex replication std",
        "Vertex CI low", "Vertex CI high", "Edge replication mean", "Edge replication std",
        "Edge CI low", "Edge CI high", "Load imbalance mean", "Load imbalance std",
        "Residual feature bytes", "Residual MB", "Cache hit rate mean", "Hit rate std",
        "Hit rate CI low", "Hit rate CI high",
    ]
    source["A1"] = "Source data: partition-policy comparison"
    source.append(headers)
    row_index = 3
    dataset_order = list(SUMMARY_FILES)
    for dataset in dataset_order:
        for policy in POLICIES:
            row = find_row(rows_by_dataset[dataset], "partition-policy", policy)
            source.cell(row_index, 1, f'=B{row_index}&"|"&C{row_index}')
            source.cell(row_index, 2, dataset)
            source.cell(row_index, 3, policy)
            for target, keys in [
                (4, ["replication_factor_vertices_mean", "replication_factor_vertices_std", "replication_factor_vertices_ci95_low", "replication_factor_vertices_ci95_high"]),
                (8, ["replication_factor_edges_mean", "replication_factor_edges_std", "replication_factor_edges_ci95_low", "replication_factor_edges_ci95_high"]),
                (12, ["load_imbalance_mean", "load_imbalance_std", "load_imbalance_ci95_low", "load_imbalance_ci95_high"]),
            ]:
                for offset, key in enumerate(keys):
                    source.cell(row_index, target + offset, number(row[key]))
            source.cell(row_index, 14, number(row["C_edg_bytes_mean"]))
            source.cell(row_index, 15, f"=N{row_index}/1048576")
            source.cell(row_index, 16, number(row["cache_hit_rate_mean"]))
            source.cell(row_index, 17, number(row["cache_hit_rate_std"]))
            source.cell(row_index, 18, number(row["cache_hit_rate_ci95_low"]))
            source.cell(row_index, 19, number(row["cache_hit_rate_ci95_high"]))
            row_index += 1

    last_source_row = row_index - 1
    style_table_header(source, 2, 1, len(headers))
    style_data_block(source, 3, last_source_row, 1, len(headers))
    source.auto_filter.ref = f"A2:S{last_source_row}"
    source.freeze_panes = "D3"
    widths = [20, 16, 18] + [15] * 16
    for col, width in enumerate(widths, start=1):
        source.column_dimensions[get_column_letter(col)].width = width
    for col in list(range(4, 14)) + list(range(16, 20)):
        format_range(source, 3, last_source_row, col, "0.000")
    format_range(source, 3, last_source_row, 14, "#,##0")
    format_range(source, 3, last_source_row, 15, "0.00")
    source.sheet_properties.tabColor = "D9EAF7"

    chart_data["A1"] = "Formula-backed chart series"
    chart_data["A3"] = "Residual feature bytes (MB)"
    chart_data.cell(4, 1, "Dataset")
    for col, policy in enumerate(POLICIES, start=2):
        chart_data.cell(4, col, policy)
    for row, dataset in enumerate(dataset_order, start=5):
        chart_data.cell(row, 1, dataset)
        for col in range(2, 7):
            letter = get_column_letter(col)
            chart_data.cell(
                row,
                col,
                f'=INDEX(SourceData!$O$3:$O${last_source_row},MATCH($A{row}&"|"&{letter}$4,SourceData!$A$3:$A${last_source_row},0))',
            )

    chart_data["A13"] = "Vertex replication factor"
    chart_data.cell(14, 1, "Dataset")
    for col, policy in enumerate(POLICIES, start=2):
        chart_data.cell(14, col, policy)
    for row, dataset in enumerate(dataset_order, start=15):
        chart_data.cell(row, 1, dataset)
        for col in range(2, 7):
            letter = get_column_letter(col)
            chart_data.cell(
                row,
                col,
                f'=INDEX(SourceData!$D$3:$D${last_source_row},MATCH($A{row}&"|"&{letter}$14,SourceData!$A$3:$A${last_source_row},0))',
            )

    style_table_header(chart_data, 4, 1, 6)
    style_table_header(chart_data, 14, 1, 6)
    style_data_block(chart_data, 5, 8, 1, 6)
    style_data_block(chart_data, 15, 18, 1, 6)
    chart_data["A3"].font = Font(bold=True, size=12, color="17365D")
    chart_data["A13"].font = Font(bold=True, size=12, color="17365D")
    chart_data.column_dimensions["A"].width = 18
    for col in range(2, 7):
        chart_data.column_dimensions[get_column_letter(col)].width = 18
    chart_data.sheet_properties.tabColor = "E2F0D9"

    bytes_chart = BarChart()
    bytes_data = Reference(chart_data, min_col=2, max_col=6, min_row=4, max_row=8)
    bytes_cats = Reference(chart_data, min_col=1, min_row=5, max_row=8)
    bytes_chart.add_data(bytes_data, titles_from_data=True)
    bytes_chart.set_categories(bytes_cats)
    configure_bar_chart(bytes_chart, "Residual feature bytes by partition policy (MB)", "Topology", "Residual bytes (MB)", "0.0")
    bytes_chart.y_axis.scaling.min = 0
    bytes_chart.y_axis.scaling.max = 7
    style_bar_series(bytes_chart, POLICY_COLORS)
    charts.add_chart(bytes_chart, "A1")

    replication_chart = BarChart()
    replication_data = Reference(chart_data, min_col=2, max_col=6, min_row=14, max_row=18)
    replication_cats = Reference(chart_data, min_col=1, min_row=15, max_row=18)
    replication_chart.add_data(replication_data, titles_from_data=True)
    replication_chart.set_categories(replication_cats)
    configure_bar_chart(replication_chart, "Vertex replication by partition policy", "Topology", "Vertex replication factor", "0.0")
    replication_chart.y_axis.scaling.min = 0
    replication_chart.y_axis.scaling.max = 7
    style_bar_series(replication_chart, POLICY_COLORS)
    charts.add_chart(replication_chart, "J1")

    charts.sheet_view.showGridLines = False
    charts["A19"] = "wiki-talk is retained as the negative feasibility case."
    charts["A19"].font = Font(italic=True)
    charts.sheet_properties.tabColor = "FFE699"
    charts.column_dimensions["A"].width = 18

    workbook.active = workbook.index(charts)
    output = OUTPUT_DIR / name
    workbook.save(output)
    return output


def build_ablation_workbook(rows_by_dataset: dict[str, list[dict[str, str]]]) -> Path:
    name = "icde-simulation-ablation-sensitivity.xlsx"
    workbook = setup_workbook(
        "ICDE Simulation Ablation and Sensitivity Source",
        "Editable data and charts for Fig. 5: component ablation and score-weight sensitivity.",
        list(SUMMARY_FILES.values()),
    )
    source = workbook.create_sheet("SourceData")
    chart_data = workbook.create_sheet("ChartData")
    charts = workbook.create_sheet("Charts")
    headers = [
        "Key", "Block", "Dataset", "Configuration control", "Configuration label",
        "Residual feature bytes", "Residual MB", "Residual bytes std", "Residual bytes CI low",
        "Residual bytes CI high", "Cache hit rate mean", "Hit rate std", "Hit rate CI low",
        "Hit rate CI high", "Vertex replication mean", "Vertex replication std",
        "Vertex CI low", "Vertex CI high", "Edge replication mean", "Load imbalance mean",
    ]
    source["A1"] = "Source data: component ablation and score sensitivity"
    source.append(headers)
    row_index = 3
    dataset_order = list(SUMMARY_FILES)
    configs = [
        ("component-ablation", COMPONENT_CONTROLS, COMPONENT_CONTROLS),
        ("score-sensitivity", SCORE_CONTROLS, SCORE_LABELS),
    ]
    for block, controls, labels in configs:
        for dataset in dataset_order:
            for control, label in zip(controls, labels):
                row = find_row(rows_by_dataset[dataset], block, control)
                source.cell(row_index, 1, f'=B{row_index}&"|"&C{row_index}&"|"&D{row_index}')
                source.cell(row_index, 2, block)
                source.cell(row_index, 3, dataset)
                source.cell(row_index, 4, control)
                source.cell(row_index, 5, label)
                source.cell(row_index, 6, number(row["C_edg_bytes_mean"]))
                source.cell(row_index, 7, f"=F{row_index}/1048576")
                source.cell(row_index, 8, number(row["C_edg_bytes_std"]))
                source.cell(row_index, 9, number(row["C_edg_bytes_ci95_low"]))
                source.cell(row_index, 10, number(row["C_edg_bytes_ci95_high"]))
                source.cell(row_index, 11, number(row["cache_hit_rate_mean"]))
                source.cell(row_index, 12, number(row["cache_hit_rate_std"]))
                source.cell(row_index, 13, number(row["cache_hit_rate_ci95_low"]))
                source.cell(row_index, 14, number(row["cache_hit_rate_ci95_high"]))
                source.cell(row_index, 15, number(row["replication_factor_vertices_mean"]))
                source.cell(row_index, 16, number(row["replication_factor_vertices_std"]))
                source.cell(row_index, 17, number(row["replication_factor_vertices_ci95_low"]))
                source.cell(row_index, 18, number(row["replication_factor_vertices_ci95_high"]))
                source.cell(row_index, 19, number(row["replication_factor_edges_mean"]))
                source.cell(row_index, 20, number(row["load_imbalance_mean"]))
                row_index += 1
    last_source_row = row_index - 1
    style_table_header(source, 2, 1, len(headers))
    style_data_block(source, 3, last_source_row, 1, len(headers))
    source.auto_filter.ref = f"A2:T{last_source_row}"
    source.freeze_panes = "E3"
    widths = [28, 22, 16, 20, 18] + [16] * 15
    for col, width in enumerate(widths, start=1):
        source.column_dimensions[get_column_letter(col)].width = width
    format_range(source, 3, last_source_row, 6, "#,##0")
    format_range(source, 3, last_source_row, 7, "0.00")
    for col in [8, 9, 10, 19, 20]:
        format_range(source, 3, last_source_row, col, "#,##0")
    for col in [11, 12, 13, 14, 15, 16, 17, 18]:
        format_range(source, 3, last_source_row, col, "0.000")
    source.sheet_properties.tabColor = "D9EAF7"

    chart_data["A1"] = "Formula-backed chart series"
    blocks = [
        (3, 4, "Component residual feature bytes (MB)", COMPONENT_CONTROLS, COMPONENT_CONTROLS, "G", "component-ablation"),
        (13, 14, "Component cache hit rate", COMPONENT_CONTROLS, COMPONENT_CONTROLS, "K", "component-ablation"),
        (23, 24, "Score-weight residual feature bytes (MB)", SCORE_CONTROLS, SCORE_LABELS, "G", "score-sensitivity"),
        (33, 34, "Score-weight vertex replication", SCORE_CONTROLS, SCORE_LABELS, "O", "score-sensitivity"),
    ]
    for title_row, header_row, title, controls, labels, source_col, block in blocks:
        chart_data.cell(title_row, 1, title).font = Font(bold=True, size=12, color="17365D")
        chart_data.cell(header_row, 1, "Configuration")
        for col, dataset in enumerate(dataset_order, start=2):
            chart_data.cell(header_row, col, dataset)
        if block == "score-sensitivity":
            chart_data.cell(header_row, 7, "Control helper")
        first_data_row = header_row + 1
        for row, (control, label) in enumerate(zip(controls, labels), start=first_data_row):
            chart_data.cell(row, 1, label)
            if block == "score-sensitivity":
                chart_data.cell(
                    row,
                    7,
                    f'=IF(A{row}="Equal","equal-weights",IF(A{row}="Locality","locality-heavy",IF(A{row}="Balance","balance-heavy","memory-heavy")))',
                )
            for col in range(2, 6):
                letter = get_column_letter(col)
                if block == "component-ablation":
                    key_expr = f'"component-ablation|"&{letter}${header_row}&"|"&$A{row}'
                else:
                    key_expr = f'"score-sensitivity|"&{letter}${header_row}&"|"&$G{row}'
                chart_data.cell(
                    row,
                    col,
                    f'=INDEX(SourceData!${source_col}$3:${source_col}${last_source_row},MATCH({key_expr},SourceData!$A$3:$A${last_source_row},0))',
                )
        style_table_header(chart_data, header_row, 1, 7 if block == "score-sensitivity" else 6)
        style_data_block(chart_data, first_data_row, first_data_row + len(controls) - 1, 1, 7 if block == "score-sensitivity" else 6)

    chart_data.column_dimensions["A"].width = 18
    for col in range(2, 8):
        chart_data.column_dimensions[get_column_letter(col)].width = 18
    chart_data.sheet_properties.tabColor = "E2F0D9"

    chart_specs = [
        (4, 9, "Component ablation: residual feature bytes (MB)", "Component", "Residual bytes (MB)", "0.0", "A1", "line"),
        (14, 19, "Component ablation: cache hit rate", "Component", "Cache hit rate", "0%", "J1", "line"),
        (24, 28, "Score weights: residual feature bytes (MB)", "Score-weight setting", "Residual bytes (MB)", "0.0", "A20", "line"),
        (34, 38, "Score weights: vertex replication", "Score-weight setting", "Vertex replication factor", "0.0", "J20", "line"),
    ]
    for header_row, last_row, title, x_title, y_title, y_format, anchor, kind in chart_specs:
        chart = LineChart()
        data = Reference(chart_data, min_col=2, max_col=5, min_row=header_row, max_row=last_row)
        cats = Reference(chart_data, min_col=1, min_row=header_row + 1, max_row=last_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        min_scale = 0 if y_format == "0%" else None
        max_scale = 1 if y_format == "0%" else None
        configure_line_chart(chart, title, x_title, y_title, y_format, min_scale, max_scale)
        style_line_series(chart, DATASET_COLORS)
        charts.add_chart(chart, anchor)

    charts.sheet_view.showGridLines = False
    charts["A38"] = "+RCS and +RCSP have identical data-path rows because pipelining changes overlap, not bytes."
    charts["A38"].font = Font(italic=True)
    charts.sheet_properties.tabColor = "FFE699"
    charts.column_dimensions["A"].width = 18

    workbook.active = workbook.index(charts)
    output = OUTPUT_DIR / name
    workbook.save(output)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    outputs = [
        build_cache_workbook(rows),
        build_partition_workbook(rows),
        build_ablation_workbook(rows),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
