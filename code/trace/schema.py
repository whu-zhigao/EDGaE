"""Canonical schema for the EDGaE extended evaluation.

The CSV is intentionally long-form: one row is one measured configuration and
seed.  A row may compare DGL-E and EDGaE because the runner executes both
systems under the same manifest.
"""

from __future__ import annotations

from typing import List


IDENTITY_COLUMNS = [
    "run_id",
    "measurement_utc",
    "measurement_source",
    "measurement_valid",
    "block",
    "dataset",
    "model",
    "sampler",
    "workers",
    "control",
    "seed",
    "horizon_epochs",
    "notes",
    "evidence_level",
    "environment_type",
    "hardware_platform",
    "validity_scope",
    "graph_source",
    "graph_vertices",
    "graph_edges",
    "runner_version",
    "calibration_source",
]

TIMING_COLUMNS = [
    "T_partition_s",
    "T_presample_s",
    "T_cache_s",
    "T_ship_s",
    "T_pre_total_s",
    "T_comp_dgl_s",
    "T_comp_edgae_s",
    "T_sync_dgl_s",
    "T_sync_edgae_s",
    "DGL_E_epoch_s",
    "EDGaE_epoch_s",
]

COMMUNICATION_COLUMNS = [
    "C_srv_bytes",
    "C_edg_bytes",
    "Delta_C_bytes",
    "Delta_T_s",
    "effective_B_bytes_s",
    "network_rtt_s",
    "network_loss_rate",
    "B_star_bytes_s",
    "break_even_epochs",
]

ACCURACY_COLUMNS = [
    "total_time_to_target_s",
    "target_accuracy",
    "test_accuracy",
    "accuracy_ci95",
]

CACHE_COLUMNS = [
    "cache_hit_rate",
    "cache_coverage",
    "K",
    "topk_jaccard",
    "boundary_gap",
]

PARTITION_COLUMNS = [
    "replication_factor_vertices",
    "replication_factor_edges",
    "load_imbalance",
    "feasibility_failures",
    "fallbacks",
]

MEMORY_ENERGY_COLUMNS = [
    "peak_rss_mb",
    "peak_pss_mb",
    "page_cache_mb",
    "sdcard_read_mb",
    "page_faults",
    "worker_energy_j",
    "server_energy_j",
    "switch_energy_j",
    "idle_power_w",
    "energy_to_target_j",
    "gradient_wait_s",
]

COLUMNS = (
    IDENTITY_COLUMNS
    + TIMING_COLUMNS
    + COMMUNICATION_COLUMNS
    + ACCURACY_COLUMNS
    + CACHE_COLUMNS
    + PARTITION_COLUMNS
    + MEMORY_ENERGY_COLUMNS
)


REQUIRED_FOR_MEASUREMENT: List[str] = [
    "T_partition_s",
    "T_presample_s",
    "T_cache_s",
    "T_ship_s",
    "DGL_E_epoch_s",
    "EDGaE_epoch_s",
    "C_srv_bytes",
    "C_edg_bytes",
    "effective_B_bytes_s",
    "peak_rss_mb",
]


def blank_identity(block: str, dataset: str, model: str, sampler: str,
                   workers: str, control: str, horizon: int = 100) -> dict:
    return {
        "run_id": "",
        "measurement_utc": "",
        "measurement_source": "",
        "measurement_valid": "",
        "block": block,
        "dataset": dataset,
        "model": model,
        "sampler": sampler,
        "workers": workers,
        "control": control,
        "seed": "",
        "horizon_epochs": horizon,
        "notes": "",
    }
