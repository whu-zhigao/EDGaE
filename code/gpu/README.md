# EDGaE GPU Experiment Package

Native-task experiments for the ICDE paper, packaged for
`NVIDIA/cuda-samples/CUDA13.0-Torch2.7.0-Python3.11:v2`.

**Pure PyTorch (no DGL dependency)** — DGL has no wheels for
Torch 2.7/CUDA 13, so GCN and GraphSAGE are implemented directly.

## Quick start (cloud GPU node)

```bash
tar xzf edgae_gpu_experiments.tar.gz && cd gpu_experiments
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

python run_all.py --smoke        # 2-4 min sanity (also runs on CPU)
python run_all.py                # full suite: E1-E3, ~30-60 min on 1 GPU
python run_all.py --sensitivity  # adds E4 fan-out axis
```

No installs are needed beyond the image's own `torch` + `numpy`
(first run parses the bundled official OGB-arxiv files, ~5 min CPU-side,
then caches `.npy` for instant restarts).

## What each experiment measures (and where it lands in the paper)

| ID | Experiment | Paper gap it closes |
|----|------------|---------------------|
| E1 | Full-batch 2-layer GCN, official split, 200 epochs | Native-task reference accuracy (§VII: "not retained") |
| E2 | Mini-batch GraphSAGE (fan-out 2, depth 2, inverse-degree sampling), `server` vs `edgae` feature paths | Feature-path effect: every batch's features moved host→device (DGL-E analogue) vs GPU-resident per-worker caches with miss-only transfer |
| E3 | Preprocessing-inclusive time-to-target | Table VI native column; T_pre = measured placement + bounded pilot-sweep cache construction |
| P4 | Parity check | **Proposition 4 validated natively**: both modes share identical seeded batch streams, so accuracies must match bit-exactly |
| E4 | Fan-out 4/8 EDGaE runs | §VI-E sampler sensitivity at native level |

## Honest scope labels

- Single-GPU, logical 8-worker study. Wall-clock is native GPU timing, not
  Raspberry Pi; `env.json` records device/torch for evidence labeling.
- `edgae` mode isolates the **feature-path mechanism** (residency + miss
  bytes); topology locality is already covered by the trace layer.
- Placement uses the paper's exact greedy semantics (capacity V/2, equal
  weights, fallback counting). At the native 90,941-seed scale the fallback
  count is itself a feasibility datum — report it, do not hide it.
- Cache hotness uses the paper's bounded pilot sweep (512 seeds/worker,
  K=16 rounds) — documented deviation from exhaustive presampling.

## Outputs (copy the whole `results/` directory back to the repo)

- `gpu_native_results.csv` — one row per run: test/val accuracy, T_pre,
  data-path vs compute seconds, miss bytes, time-to-target epochs/seconds,
  fallbacks, replication, cache union size, device, torch version.
- `env.json`, `summary.json` — environment provenance and paper mapping.

## Verified smoke evidence (local CPU, torch 2.5.1)

```placement fallbacks=0 repl=1.834 imb=0.000 T_pre=4.0s
[server] epoch 1 loss 3.0429 val 0.0940
[edgae]  epoch 1 loss 3.0429 val 0.0940
P4 parity |val_server - val_edgae| = 0.000000
data-path speedup = 1.01x (CPU; GPU shows the real residency effect)
miss bytes 21,931,008
```

Expected full-run reference points (2-layer, 128-dim features, lr 1e-2):
GCN test accuracy on arxiv typically lands near 0.55-0.57; mini-batch SAGE
with fan-out 2 near 0.50-0.53. Values outside these bands indicate a
pipeline problem — check `env.json` and the loss curves first.

## Returned results (2026-10-10)

Cloud run artifacts live in `results/` (15 rows, A800). Regenerate the paper table with `python make_gpu_table.py`.
