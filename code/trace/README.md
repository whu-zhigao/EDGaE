# EDGaE extended-evaluation harness

This directory implements the measurement pipeline required by Appendix B of
`www2027-acmart.tex`. It does **not** contain the EDGaE system implementation,
the Raspberry Pi cluster, the datasets, or the power-meter interface. Therefore
it cannot by itself produce publishable measurements. It provides the complete
runner contract, system controls, raw-data schema, metric collection, derivation
rules, confidence intervals, and LaTeX table generation.

## Design goals

The experiment matrix covers seven questions:

1. **Break-even accounting** — measure preprocessing cost, saved bytes, effective
   bandwidth, \(B^\star\), and the epoch at which EDGaE becomes faster.
2. **Native node classification** — run `ogbn-arxiv` with its official split and
   report test accuracy and preprocessing-inclusive time-to-target.
3. **CPU-frequency emulation** — compare even and capability-aware partitions
   under 600/1000/1500 MHz workers.
4. **Network sensitivity** — measure 10/100/1000 Mbps, 5/20/50 ms RTT, and loss.
5. **Cache stability** — vary \(K\) and compare hit rate/top-\(k\) overlap with a
   full training trace.
6. **Component and partition ablation** — isolate replicated partitioning,
   capability awareness, cache, and pipeline.
7. **Memory and energy accounting** — capture RSS, PSS, page cache, SD reads,
   page faults, and separated worker/server/switch energy.

## Files

- `schema.py`: canonical CSV columns and required fields.
- `build_template.py`: regenerates the experiment matrix.
- `run_extended_eval.py`: orchestrates real or simulated runs.
- `metric_sampler.py`: samples the local process tree and optional power meter.
- `system_control.py`: optional local `cpufreq` and `tc netem/tbf` controls.
- `measure_bandwidth.py`: measures effective throughput using `iperf3`.
- `analyze_extended_results.py`: derives \(T_{\mathrm{pre}}\), \(\Delta C\),
  \(\Delta T\), \(B^\star\), break-even epoch, summaries, CIs, and LaTeX.
- `simulated_runner.py`: deterministic smoke-test runner only.
- `trace_driven_runner.py`: offline placement/cache/communication simulation.
- `build_hybrid_matrix.py`: compact synthetic hybrid-evaluation matrix.
- `emulated_model.py`: parameterized timing/energy model.
- `run_hybrid_eval.py`: runs trace simulation and parameterized emulation.

## Runner contract

A real run must provide `--runner-command`. The command receives a JSON manifest
path as its final argument (or wherever `{manifest}` appears) and must print a
JSON object as the final JSON object on stdout.

Example:

```bash
python3 experiments/run_extended_eval.py \
  --runner-command "python3 /path/to/edgae_cluster_runner.py" \
  --output www2027-extended-results-measured.csv \
  --raw-dir experiment-raw \
  --blocks break-even native-task cpu-emulation network-sensitivity \
           cache-stability component-ablation partition-policy score-sensitivity \
  --seeds 1 2 3 4 5
```

The runner is responsible for:

- loading the requested dataset and official split;
- applying EDGaE/DGL-E or the requested ablation;
- executing all nodes in the cluster;
- aggregating node metrics;
- returning the fields in `schema.py`;
- setting `measurement_valid` to `true` only for a complete real measurement.

At minimum, a real runner should emit:

```json
{
  "measurement_source": "cluster-name",
  "measurement_valid": true,
  "T_partition_s": 0.0,
  "T_presample_s": 0.0,
  "T_cache_s": 0.0,
  "T_ship_s": 0.0,
  "T_comp_dgl_s": 0.0,
  "T_comp_edgae_s": 0.0,
  "T_sync_dgl_s": 0.0,
  "T_sync_edgae_s": 0.0,
  "DGL_E_epoch_s": 0.0,
  "EDGaE_epoch_s": 0.0,
  "C_srv_bytes": 0,
  "C_edg_bytes": 0,
  "effective_B_bytes_s": 0,
  "total_time_to_target_s": 0.0,
  "test_accuracy": 0.0,
  "cache_hit_rate": 0.0,
  "cache_coverage": 0.0,
  "topk_jaccard": 0.0,
  "boundary_gap": 0.0,
  "replication_factor_vertices": 0.0,
  "replication_factor_edges": 0.0,
  "load_imbalance": 0.0,
  "feasibility_failures": 0,
  "fallbacks": 0,
  "peak_rss_mb": 0.0,
  "peak_pss_mb": 0.0,
  "worker_energy_j": 0.0,
  "server_energy_j": 0.0,
  "switch_energy_j": 0.0,
  "energy_to_target_j": 0.0,
  "gradient_wait_s": 0.0
}
```

Missing values should be `null`, not inferred or fabricated.

## Cluster controls

The orchestrator can apply local network controls only when explicitly enabled:

```bash
--network-interface eth0 --allow-system-control
```

For a real cluster, apply the same `tc` policy to the edge uplink or invoke the
runner on each node with the manifest's `control` field. CPU frequency caps are
likewise cluster-specific; a local cap is available for debugging through
`--local-frequency-mhz --allow-system-control`.

Measure effective bandwidth separately:

```bash
iperf3 -s                    # on the server
python3 experiments/measure_bandwidth.py --server SERVER_IP --duration 10
```

## Hybrid trace/model runs

The Raspberry Pi cluster is unavailable, so the hybrid runner uses explicit
evidence levels. It will not label parameterized timing estimates as
publishable measurements by default.

```bash
python3 experiments/build_hybrid_matrix.py www2027-hybrid-results-template.csv

python3 experiments/run_hybrid_eval.py \
  --template www2027-hybrid-results-template.csv \
  --trace-output www2027-hybrid-results-trace.csv \
  --emulated-output www2027-hybrid-results-emulated.csv \
  --raw-dir hybrid-raw \
  --cache-dir experiment-cache \
  --calibration experiments/calibrations/illustrative.json
```

See `HYBRID_EVALUATION.md` for the completed pilot runs and the exact validity
scope of each evidence level.

## Real SNAP topology run

The current real-topology experiment uses a reproducible subgraph of SNAP
`web-Google`. Prepare the compact edge list, run five seeds, and generate the
manuscript tables with:

```bash
python3 experiments/prepare_snap_subgraph.py \
  datasets/web-Google.txt.gz datasets/web-google.edges \
  --reservoir-edges 180000 --max-nodes 12000 --seed 2027

python3 experiments/build_real_matrix.py \
  www2027-real-trace-template-v2.csv --dataset web-google

python3 experiments/run_hybrid_eval.py \
  --template www2027-real-trace-template-v2.csv \
  --trace-output www2027-real-trace-results-v2.csv \
  --emulated-output www2027-real-model-results-v2.csv \
  --raw-dir hybrid-raw-real-web-google-v2 \
  --cache-dir experiment-cache \
  --seeds 1 2 3 4 5

python3 experiments/analyze_extended_results.py \
  www2027-real-trace-results-v2.csv \
  --derived-output www2027-real-trace-derived-v2.csv \
  --summary-output www2027-real-trace-summary-v2.csv \
  --latex-output www2027-real-trace-tables-raw.tex \
  --bootstrap-samples 5000

python3 experiments/generate_real_trace_tables.py \
  www2027-real-trace-summary-v2.csv www2027-real-trace-tables.tex
```

`www2027-real-trace-tables.tex` is included by `www2027_appendix.tex`.

## Analyze real results

```bash
python3 experiments/analyze_extended_results.py \
  www2027-extended-results-measured.csv \
  --require-complete
```

Outputs:

- `www2027-extended-results-derived.csv`
- `www2027-extended-results-summary.csv`
- `www2027-appendix-b-tables.tex`

Only rows with `measurement_valid=true` are analyzed by default. The script
fails under `--require-complete` if required fields are absent.

## Smoke test

The simulated runner tests only the plumbing:

```bash
python3 experiments/run_extended_eval.py \
  --simulate \
  --blocks cache-stability \
  --datasets web-google \
  --controls K=1 \
  --seeds 1 \
  --output /tmp/edgae-smoke.csv \
  --raw-dir /tmp/edgae-smoke-raw

python3 experiments/analyze_extended_results.py \
  /tmp/edgae-smoke.csv --include-invalid
```

All simulated rows have `measurement_valid=false` and
`measurement_source=SIMULATED_SMOKE_TEST_ONLY`. Never copy them into the paper.

## ICDE simulation figures

Regenerate the three trace-driven figures used by `IEEEtran.tex`:

```bash
make -C experiments icde-figures
```

The script reads the four real-trace summary CSV files from `www/` and writes:

- `ICDE/fig/icde-simulation-cache-stability.pdf`
- `ICDE/fig/icde-simulation-partition-feasibility.pdf`
- `ICDE/fig/icde-simulation-ablation-sensitivity.pdf`

Use the following target to create editable Excel source workbooks with the
underlying data and native Excel charts:

```bash
make -C experiments icde-figure-workbooks
```

The workbooks are written to `ICDE/figsource/`.

## v2 extension blocks (ICDE camera-ready revision)

`trace_driven_runner_v2.py` extends v1 with full/prefix topologies, measured
preprocessing time, online caches, sampler sensitivity, and two-tier caches.

Canonical result files and how to regenerate them:

| File | Runs | Regenerate with |
|---|---|---|
| `www/icde2027-v2-trace-results.csv` | 945 | `run_icde_v2_eval.py` |
| `www/icde2027-v2-preprocess-results.csv` | 35 | EDGaE+K=16 driver (see analyzer inputs) |
| `www/icde2027-v2-ablation-results.csv` | 175 | no-cache component semantics |
| `www/icde2027-v2-sensitivity-results.csv` | 210 | sampler axes, fixed ball-injection allocator |
| `www/icde2027-v2-two-tier-results.csv` | 175 | static+LFU tiers, per-access eviction |
| `www/icde2027-v2-online-results.csv` | 70 | unified LRU/LFU semantics |
| `www/icde2027-v2-web-google-complete-results.csv` | 195 | all blocks, complete official web-Google (875k) |
| `www/icde2027-v2-wiki-talk-complete-results.csv` | 195 | all blocks, complete official wiki-Talk (2.4M) |

Before every submission, run the consistency auditor (CI-ready, exit code 0
on success). It re-checks every headline numeric claim against the summary
JSON at full precision, scans the LaTeX sources for stale language patterns,
verifies that generated tables are newer than the result CSVs, and parses the
compile log for undefined references and page-count violations:

```bash
python3 experiments/audit_paper_consistency.py
```

`analyze_icde_v2.py` consumes all of the above (preferring corrected reruns),
writes `www/icde2027-v2-summary.json` with bootstrap 95% CIs, regenerates
every table under `ICDE/tab-*.tex`, and every figure under
`ICDE/fig/icde-v2-*.pdf`. The paper contains no hand-typed result number.

## GPU native-task package

`gpu_experiments/` (see its own README) contains the pure-PyTorch native
ogbn-arxiv experiments for CUDA 13 / Torch 2.7 (no DGL): full-batch GCN
reference accuracy, server-vs-EDGaE feature paths, preprocessing-inclusive
time-to-target, Proposition-4 parity, and fan-out sensitivity. The upload
artifact is `edgae_gpu_experiments.tar.gz` (81.8 MB, sha256
7aa2a7c42d18b6238e3ad90f0c29bc313137a2fd99f19a8dc73b7449d8e2e895);
smoke-verified end to end, including exact server/EDGaE accuracy parity.
Return the cloud `results/` directory here for integration into the paper.

---

## ICDE v3 revision additions (mode selection + system-policy)

### New scripts

- `analyze_mode_and_sota.py`: derives the S/H/E placement-mode decision surface
  from measured preprocess rows and emits `tab-mode-selection.tex` plus
  `tab-sota-systems.tex` into `../paper/`.
- `run_mode_system_eval.py`: runs the two new blocks:
  - `preprocess` — re-measures EDGaE-K16 rows with the new **working-set**
    metrics (`working_set_vertices`, `working_set_bytes`) needed to price
    edge-local (E) mode.
  - `system-policy` — ten system-inspired adapters on the identical substrate
    (sampler, byte accounting, per-worker budgets): Server-C, DistDGL-like,
    PaGraph-like, GNNLab-like, Ginex-like, BGL-like, NeutronStar-like,
    DAHA-like (PVLDB'24), PlatoD2GL-like (ICDE'24), and EDGaE.
- `trace_driven_runner_v2.py`: extended with the `system-policy` block
  (`SYSTEM_POLICY_SPECS`), per-worker sharded online caches, and
  working-set emission.

### Reproduce locally (trace-driven, artifact workstation)

```bash
cd ICDE_v2/exp
python3 experiments/run_mode_system_eval.py \
  --output experiments/results/icde-v3-mode-system-results.csv \
  --cache-dir experiment-cache --parallelism 6

cd experiments
python3 analyze_mode_and_sota.py \
  --outdir ../../paper \
  --preprocess-csv results/icde-v3-mode-system-results.csv \
  --trace-csv results/icde-v3-mode-system-results.csv
```

Runtime: ~5.5 minutes for 385 runs (7 datasets x 5 seeds x 11 configs) on
Apple silicon with warm dataset pickles.

### Cloud / physical run (to be executed on the server)

The trace-driven evidence is complete locally. The remaining physical runs for
the ICDE v3 revision are:

1. **Pi cluster system-policy ports** — run the ten adapters as policy ports on
   the eight-worker Raspberry Pi testbed (same substrate contract as
   PaGraph-E/GNNLab-E/BGL-E in the current paper) to add steady-state
   per-epoch time for each adapter, not only byte accounting.
2. **A800 native mode-transition check** — measure native S vs. H timing at the
   break-even bandwidth points predicted by Table `tbl:mode-selection`, i.e.
   confirm that dense graphs at high bandwidth prefer S natively.

Commands are issued via `run_extended_eval.py --runner-command ...` with the
cluster runner; see the runner contract above. All rows must record
`evidence_level`, `hardware_platform`, and `measurement_valid` per the schema;
system-policy rows must state that they are policy adapters, not the original
end systems.

### Evidence discipline (unchanged)

- Trace-driven rows never claim runtime/energy/accuracy.
- Adapter rows are policy adapters on identical software, not the original
  systems' native deployments.
- E-mode pricing assumes a stable batch schedule across the horizon; drifting
  stochastic samplers trigger governor re-plans (documented in the paper).

---

## Review-response experiments (joint quality, regret, cache shift)

### New scripts

- `run_joint_quality_eval.py`: decision-quality study for the placement
  allocator (review Major Concern 2).
  - Strategies: two-stage-greedy (paper), locality-only, memory-first,
    cost-aware (marginal amortized cost), random-feasible, best-improvement
    local search.
  - Exact optimum: exhaustive enumeration over all 2^n assignments on
    10-seed / 2-worker instances (no MILP solver needed).
  - Ordering sensitivity: neighborhood-desc, access-desc, ship-cost-asc,
    random.
  - Cache-shift study: caches built on seed set A, evaluated on seed sets
    with 10/20/50% replacement.
- `analyze_mode_and_sota.py` (extended): cost-regret analysis under 2x
  horizon/bandwidth misestimation and the unified horizon-crossing table
  (`mode-regret.txt`, `horizon-table.txt`).

### Reproduce

```bash
cd ICDE_v2/exp
python3 experiments/run_joint_quality_eval.py \
  --datasets web-google amazon0601 wiki-talk --seeds 1 2 3 --exact-seeds 10

cd experiments
python3 analyze_mode_and_sota.py \
  --outdir ../../paper \
  --preprocess-csv results/icde-v3-mode-system-results.csv \
  --trace-csv results/icde-v3-mode-system-results.csv
```

### Key findings (as reported in the paper)

- Greedy attains the exact optimum on all 6 sparse-graph instances
  (web-google, amazon0601; gap 0.0%) and is within 2.0-6.6% (mean 3.7%)
  on dense wiki-talk.
- Medium scale (200 seeds, 4 workers): greedy is the best single-pass
  heuristic on both sparse graphs; cost-aware placement wins by 4.6% on
  dense wiki-talk, and local search recovers a further 4.7%.
- Ordering sensitivity: <=1.9%.
- Regret under 2x H/B misestimation: 0 at 10 Mbps; 0.4% mean / 11.9%
  worst at 100 Mbps; 10.1% mean at 1 Gbps (concentrated at break-even
  boundaries).
- Cache shift: hit rate drops 4.5 / 9.0 / 21.9 pp under 10 / 20 / 50%
  seed replacement.
