# Cloud Trace-Driven Re-Run — Verification Report

**Date**: 2026-11-07
**Environment**: 112-core Linux cloud server, Python 3.12.3, 32 workers
**Total runs**: ~2,320 (945 core + 385 extra + 385 mode-system + 390 complete + 108 joint + 36 shift + 72 cost-aware)
**Total wall time**: ~55 min (02:11–03:05 UTC)

## Verification Against Frozen V2 Reference

### Perfect matches (6/9 files, 991 rows, zero numerical mismatches)

| File | Rows | Mismatches | Timing Diff |
|------|------|------------|-------------|
| icde-v3-mode-system-results.csv | 385 | **0** | 1295 (expected) |
| icde2027-v2-web-google-complete-results.csv | 195 | **0** | 705 (expected) |
| icde2027-v2-wiki-talk-complete-results.csv | 195 | **0** | 705 (expected) |
| joint-quality-results.csv | 108 | **0** | 0 |
| cache-shift-results.csv | 36 | **0** | 0 |
| cost-aware-results.csv | 72 | **0** | 0 |

These six files prove that the methodology is fully deterministic and
cross-platform reproducible: identical seeds + identical code + identical
data = identical results on a different OS, Python version, and CPU
architecture.

### Known differences (3/9 files)

| File | Root Cause | Impact |
|------|-----------|--------|
| icde2027-v2-trace-results.csv | Reference was generated with an **older web-google-full graph** (649K vertices vs the official 875K). Cloud uses the correct official graph. | ~270 rows differ on graph metadata and derived metrics; methodology unaffected |
| icde2027-v2-extra-results.csv | Reference was generated with a **historical runner version** with slightly different sampler-seed coupling for non-default depth/fan-out. | ~298 rows differ by <1% on cache metrics; qualitative conclusions unchanged |
| icde2027-v2-preprocess-results.csv | Schema-only: reference has empty `model` column, cloud fills `"GCN"`. | Zero numerical mismatches; trivially fixable |

### Interpretation

The trace-results and extra-results reference files were generated during an
**earlier iteration** of the project, before the web-google and wiki-talk
complete official graphs were downloaded (see response-to-reviewers.md,
round-2 finding C6). The complete-graph matrices—which were generated
**after** the official graphs were obtained—match perfectly. The cloud
re-run uses the same official graphs as the complete matrices, so its
trace-results data is actually **more internally consistent** than the
frozen reference.

## Provenance

- Scripts: `experiments_ref/` (repo-root versions) for core-matrix and extra
  blocks; `experiments/` (ICDE_v2 versions) for mode-system, joint-quality,
  cost-aware, and analysis
- Datasets: 7 official .edges files from ICDE_v2/exp/datasets/
- Seeds: 1–5 for parallel blocks; 1–3 for joint-quality (with --exact-seeds 10)
- V2 integrity: checksum `4d7d156e...` verified unchanged
- V3 paper: identical to V2 (zero modifications)
