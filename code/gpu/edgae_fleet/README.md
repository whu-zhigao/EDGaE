# Emulated Edge Fleet — real-task experiment (Plan A)

Replaces the retired Raspberry Pi fleet with a **resource-constrained
emulated edge cluster** on one multi-core Linux host (the A800 server).
Kernel-enforced constraints reproduce the edge-class environment; the task,
model, sampler, and protocol are real.

## Emulation model (all constraints disclosed in the paper)

| Edge property | Enforcement |
|---|---|
| 8 CPU-only workers | 8 worker processes, 2 pinned cores each (`sched_setaffinity`) |
| Per-worker memory budget | `RLIMIT_AS` hard cap (default 6 GB) + placement data budgets |
| Shared slow link | paced server→worker transfers at target Mbps (auditable counters; same pacing machinery validated in E5) |
| 1 server | server process on 8 pinned cores, authoritative graph/features |

## Real-task protocol

- Dataset: **ogbn-arxiv official split** (90,941 train / 29,799 val / 48,603 test)
- Model: 2-layer GraphSAGE (hidden 256), fan-out 2, Adam lr 1e-2, wd 5e-4
- Modes:
  - `server-centric` — DGL-E analogue: every seed's batch is assembled by the
    server and shipped to its worker (S mode for all seeds)
  - `hybrid` — EDGaE: feasible seeds sample locally from per-worker sub-CSRs
    with ship-once caches and paced miss-only fetches (H); fallback seeds run
    S mode — **natural per-vertex mixed-mode deployment on a real task**
- Parity: both modes use the same deterministic per-(seed, epoch, worker)
  RNG streams and the same `merge_parts` assembly, so S and H see identical
  sampled distributions for the same seed (Proposition 4 scope)
- Metrics: preprocessing-inclusive **time-to-95%-val-accuracy**, final test
  accuracy, per-mode bytes (feature vs S-batch), placement fallbacks,
  H/S seed-epoch counts, achieved pacing throughput

## Files

- `sampling.py` — sampler + `epoch_rng` parity streams + `merge_parts`
- `placement.py` — greedy placement + hotness caches (ported, paper semantics)
- `net.py` — length-prefixed RPC + paced sender
- `fleet_server.py` — authoritative store, paced services, Adam, evaluation
- `fleet_worker.py` — constrained worker: local sampling, cache, grads
- `run_fleet.py` — orchestrator (artifacts, spawn, CSV)
- `data_loader.py` — ogbn-arxiv loader (copied from edgae_gpu)

## Run

```bash
# Smoke (2 epochs, 2048 seeds, 1 run seed) — always run first on the host
python3 run_fleet.py --smoke --data-root ../data/ogbn-arxiv --out results_fleet_smoke

# Full protocol: 2 modes x 3 seeds x 60 epochs @ 100 Mbps
python3 run_fleet.py --data-root ../data/ogbn-arxiv \
  --out results_fleet --target-mbps 100

# Optional axes (time permitting)
python3 run_fleet.py ... --target-mbps 10     # slow-link regime
python3 run_fleet.py ... --modes hybrid       # mixed-mode deep dive
```

Runtime estimate on the A800 CPU: smoke ~2 min; full ~3–5 h
(server-centric dominates: ~327 MB/epoch paced at 100 Mbps).

## Output

`results_fleet/fleet_results.csv` — one row per (mode, seed):
test_acc, best_val, ttt_epoch, ttt_train_s, **ttt_total_s (T_pre included)**,
bytes_feature, bytes_sbatch, fallbacks, replication, h/s_seed_epochs.

Per-run artifacts: `run_s{seed}_{mode}/` with server/worker logs and
`server_result.json` (full history).

## Evidence label

`emulated-edge`: real task, real CPU execution, kernel-enforced memory/CPU
bounds, transport-layer pacing. Not a physical device study; constraints and
achieved throughput are logged for audit.
