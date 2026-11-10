# EDGaE Artifact

Code and data for **"EDGaE: A Placement-Governed Server–Edge System for
Memory-Constrained GNN Training"** (ICDE 2027 submission).

This artifact contains the trace-driven experiment suite, the A800 native
and emulated-fleet studies, the figure/table generation pipeline, all
result files backing the numbers in the paper, and the LaTeX source.
Datasets and raw per-worker arrays are excluded (see *Not included*).

## Layout

```
artifact/
├── README.md                 ← this file
├── code/
│   ├── trace/                ← trace-driven experiment suite (Secs. IV–V of the paper)
│   │   ├── trace_driven_runner_v2.py     core governed-matrix runner
│   │   ├── run_icde_v2_eval.py           26-config core matrix
│   │   ├── run_icde_v2_extra_eval* / run_mode_system_eval.py
│   │   ├── run_joint_quality_eval.py     Sec. VI-D decision-quality block
│   │   ├── run_cost_aware_greedy.py      allocator baselines
│   │   ├── run_horizon_sweep.py          Fig. 3(b) / Sec. VI-F sweep
│   │   ├── analyze_mode_and_sota.py      generates Table II / IV from CSVs
│   │   ├── generate_real_trace_tables.py, generate_multi_real_trace_tables.py
│   │   ├── plot_mode_surface.py          Fig. 3
│   │   ├── profile_feature_loading.py    Fig. 1 profiling pass
│   │   ├── profile_memory_ledger.py      Sec. VI-D memory ledger
│   │   ├── prepare_snap_subgraph.py      12k reservoir construction
│   │   ├── prepare_full_snap_edges.py / prepare_ogbn_arxiv.py
│   │   ├── parallel_download.sh          dataset download
│   │   ├── audit_paper_consistency.py    paper-number vs CSV checker
│   │   └── verify_against_reference.py   deterministic re-run verification
│   ├── gpu/                   ← A800 studies (Sec. VII)
│   │   ├── edgae_gpu/        single-GPU native study + paced mode transition
│   │   ├── edgae_fleet/      emulated edge fleet (server/worker/placement/sampling)
│   │   ├── run_all.py / run_mode_transition.py / make_gpu_table.py
│   │   └── requirements.txt
│   └── figures/               ← paper figure sources (matplotlib + workbooks)
│       ├── build_fig1_profiling.py       Fig. 1  (input: fig1-data.xlsx)
│       ├── build_fig_partition.py        Fig. 4  (input: partition-data.xlsx)
│       ├── build_fig2_architecture_pptx.py  Fig. 2 (editable PowerPoint)
│       └── *.xlsx                         per-cell 95% bootstrap CI workbooks
├── data/
│   ├── trace-results/
│   │   ├── cloud-run/        ★ canonical results (Nov-2026 cloud re-run; see
│   │   │                       README_CLOUD_RUN.md for the verification report)
│   │   │   ├── results/*.csv             12 result CSVs + env.txt
│   │   │   ├── results/tables_v2/        paper Table II/IV + Sec. VI-F tables
│   │   │   └── timeline.txt              run timeline
│   │   └── local/             local-run copies + memory-ledger JSONs
│   └── gpu-results/
│       ├── results_full/      A800 native study (gpu_native_results.csv,
│       │                      summary.json, env.json — accuracy, miss rates,
│       │                      TTT, fallback accounting)
│       ├── results_e5*/       paced mode-transition runs (10–1000 Mbps)
│       └── results_fleet/     emulated fleet: 6 runs × {server_result.json,
│                              server_manifest.json (the placement manifests),
│                              worker cfgs/logs} + fleet_results.csv
├── env/
│   ├── trace-run-env.txt      recorded cloud-run environment
│   └── gpu-requirements.txt
└── paper/                     LaTeX source + compiled figures
```

## Recorded environments

| Layer | Recorded environment |
|---|---|
| Trace cloud run (`data/trace-results/cloud-run/results/env.txt`) | Python 3.12.3, 112-core Intel Xeon Gold 6348, 32-way parallelism |
| A800 / fleet runs (`data/gpu-results/results_full/env.json`) | PyTorch 2.7.0+cu128, CUDA 12.8, Python 3.12.3, NVIDIA A800 80GB PCIe |
| GPU suite deps | `env/gpu-requirements.txt` (numpy ≥ 1.24; torch preinstalled) |

The Raspberry Pi fleet runs (Table V, Sec. VII-A–C) predate this artifact and
their raw logs are **not** included; only per-configuration means appear in the
paper. All other numbers in the paper trace to files in `data/`.

## Paper → artifact mapping

| Paper item | Source |
|---|---|
| Fig. 1 (time split) | `code/figures/build_fig1_profiling.py`, `code/trace/profile_feature_loading.py`, `code/trace/profile_results_calibrated.csv`, `code/figures/fig1-data.xlsx` |
| Fig. 2 (architecture) | `code/figures/build_fig2_architecture_pptx.py` (→ `EDGaEarch-v2.pptx`) |
| Fig. 3 (mode surface) | `code/trace/plot_mode_surface.py`, constants from `data/trace-results/cloud-run/results/icde2027-v2-preprocess-results.csv` |
| Fig. 4 (partition) | `code/figures/build_fig_partition.py`, `partition-data.xlsx`, CIs in `icde-simulation-partition-feasibility.xlsx` |
| Table I (topologies) | `code/trace/prepare_snap_subgraph.py`, `prepare_full_snap_edges.py`, `prepare_ogbn_arxiv.py` |
| Table II (residual bytes) | `data/trace-results/cloud-run/results/tables_v2/tab-sota-systems.tex` ← `icde-v3-mode-system-results.csv` via `analyze_mode_and_sota.py` |
| Table III (decision quality) | `run_joint_quality_eval.py` → `joint-quality-results.csv` |
| Table IV (mode transitions) | `tables_v2/tab-mode-selection.tex` ← `icde2027-v2-preprocess-results.csv` + `icde-v3-mode-system-results.csv` |
| Table V (Pi runtime) | not in artifact (see above) |
| Sec. VI cache/regime numbers | `cache-shift-results.csv`, `icde2027-v2-trace-results.csv`, `icde2027-v2-extra-results.csv` |
| Sec. VI-F sweep/regret/heterogeneous | `tables_v2/{horizon-table,mode-regret,mode-sensitivity,heterogeneous-bw}.txt` ← `run_horizon_sweep.py` |
| Sec. VI-D memory ledger | `data/trace-results/local/memory-ledger{,-linux}.json` |
| Sec. VII A800 native | `data/gpu-results/results_full/{gpu_native_results.csv,summary.json}` |
| Sec. VII paced break-even | `data/gpu-results/results_e5*/mode_transition*.csv` |
| Sec. VII emulated fleet | `data/gpu-results/results_fleet/` (esp. `fleet_results.csv`, per-run `server_result.json`, `server_manifest.json`) |

`code/trace/audit_paper_consistency.py` re-checks headline paper numbers
against the CSVs; `verify_against_reference.py` reproduces the deterministic
verification in `cloud-run/README_CLOUD_RUN.md` (6/9 result files match the
frozen reference byte-exactly; the 3 differing files and their causes are
documented there).

## Reproducing

1. **Datasets** (not redistributed; download then prepare):
   ```bash
   bash code/trace/parallel_download.sh     # SNAP web-Google, wiki-Talk,
                                            # amazon0601 + OGB ogbn-arxiv
   python3 code/trace/prepare_snap_subgraph.py    # 12k reservoirs
   python3 code/trace/prepare_full_snap_edges.py  # complete graphs
   python3 code/trace/prepare_ogbn_arxiv.py
   ```
2. **Trace matrix** (Secs. V–VI): `run_icde_v2_eval.py`,
   `run_icde_v2_extra.py`, `run_mode_system_eval.py`,
   `run_joint_quality_eval.py`, `run_cost_aware_greedy.py`,
   `run_horizon_sweep.py`; then `analyze_mode_and_sota.py` and
   `generate_real_trace_tables.py` to regenerate Tables II/IV.
3. **A800 studies** (Sec. VII): `code/gpu/run_all.py` (native),
   `run_mode_transition.py` (paced), `edgae_fleet/run_fleet.py` (fleet).
4. **Figures**: the `build_fig*.py` scripts in `code/figures/` read the
   committed workbooks and emit the PDFs used in `paper/fig/`.

Approximate trace-matrix cost: ~2,320 runs in ~55 min on 112 cores
(see `cloud-run/timeline.txt`).

## Not included

- **Datasets** (~311 MB): obtain from SNAP and OGB (step 1 above).
- **Fleet worker arrays** (~1.5 GB of `.npy` per-run tensors): the derived
  per-run results, manifests, configs, and logs are included; raw arrays are
  regenerable from `run_fleet.py`.
- **Experiment caches / framework caches** (`.pkl`, `_cache/`).

## Notes

- `code/trace/README.md` documents an earlier (pre-ICDE) iteration of the
  harness ("www2027"); the ICDE runner entry points are the `run_icde_v2_*`
  / `run_mode_system_eval.py` family, and `results_cloud` is the canonical
  result set for this paper.
- License: to be added upon public release.
