# ICDE simulation figure sources

Each workbook contains four sheets:

- `README`: provenance, evidence level, and usage notes.
- `SourceData`: the complete data used by the figure, including means,
  dispersion, and confidence intervals where available.
- `ChartData`: formula-backed compact series tables used as chart inputs.
- `Charts`: native, editable Excel charts.

| Workbook | Corresponding manuscript figure |
|---|---|
| `icde-simulation-cache-stability.xlsx` | Fig. 3, cache stability |
| `icde-simulation-partition-feasibility.xlsx` | Fig. 4, partition feasibility |
| `icde-simulation-ablation-sensitivity.xlsx` | Fig. 5, component ablation and score sensitivity |

The source CSV files are stored under `../../www/`. Regenerate all three
workbooks with:

```bash
make -C ../../experiments icde-figure-workbooks
```

## Master workbook

`figure_data_master.xlsx` is the **chart-ready** master workbook for every
paper figure. Structure:

- `chart_*` sheets — wide tables (categories x series), one block per metric:
  blue-shaded **Mean** block (select it and insert a chart directly), plus
  matching **CI95 low / CI95 high** blocks for error bands or custom error bars.
- `charts_preview` — nine pre-built native Excel charts (line / clustered bar /
  log-scale line) referencing the `chart_*` blocks; restyle freely, they update
  when the data changes.
- `src_*` sheets — long-format source data (one row per topology x config,
  with mean and CI).
- `palette` — Okabe-Ito color reference.

Covers: cache stability (Fig 2), partition (Fig 3), sampler sensitivity
(Fig 4), online-cache regimes, two-tier sweeps, and the Section VII
physical-study relative-hit ratios (Fig 5). Regenerate with:

```bash
python3 figsource/build_figure_data_master.py
```

Ground truth: `../../www/icde2027-v2-summary.json` (single source; the same
JSON feeds `../../experiments/analyze_icde_v2.py`, which generates the
figure PDFs).

## Fig. 2 architecture — editable PowerPoint

`EDGaEarch.pptx` is the fully editable PowerPoint version of the
architecture figure (Fig. 2 in the v3 manuscript). Every element is a
native shape (rounded rectangles, connectors, device glyphs, text boxes)
with descriptive shape names, so it can be restyled directly in
PowerPoint/Keynote. Regenerate with:

```bash
python3 figsource/build_fig2_architecture_pptx.py
```

The manuscript references `../fig/EDGaEarch.pdf` — a lossless Flate
wrapper of the master PNG `../fig/EDGaEarch.png` (pixel-identical
artwork, PDF container). Rebuild the wrapper after replacing the master
PNG with:

```bash
python3 figsource/convert_arch_png_to_pdf.py
```

`preview_pptx.py` renders an approximate PNG preview of the pptx (geometry,
fills, dashes, text) via matplotlib for quick visual checks when no Office
renderer is available:

```bash
python3 figsource/preview_pptx.py   # writes .pptx-preview/pptx-preview.png
```
