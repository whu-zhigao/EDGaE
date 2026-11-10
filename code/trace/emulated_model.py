"""Parameterized system model for the emulated/calibrated hybrid layer."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Dict


DEFAULT_CALIBRATION = {
    "partition_edge_rate_edges_per_s": 1_000_000,
    "partition_score_rate_vertices_per_s": 250_000,
    "sample_rate_features_per_s": 120_000,
    "feature_extract_rate_bytes_per_s": 200_000_000,
    "cache_build_rate_bytes_per_s": 150_000_000,
    "ship_rate_bytes_per_s": 80_000_000,
    "compute_s_per_accessed_feature": 0.00010,
    "sync_fixed_s": 0.010,
    "worker_power_w": 6.0,
    "server_power_w": 30.0,
    "switch_power_w": 5.0,
    "calibration_source": "illustrative-not-publishable",
    "publishable": False,
}


def load_calibration(path: Path | None) -> Dict:
    if path is None or not path.exists():
        return DEFAULT_CALIBRATION
    calibration = json.loads(path.read_text())
    return {**DEFAULT_CALIBRATION, **calibration}


def derive_emulated_metrics(
    trace: Dict[str, object], manifest: Dict[str, object], calibration: Dict
) -> Dict[str, object]:
    workers = int(manifest["workers"])
    feature_bytes = 100 * 4
    replication_vertices = float(trace.get("replication_factor_vertices") or 0.0)
    replication_edges = float(trace.get("replication_factor_edges") or 0.0)

    vertices = int(trace.get("graph_vertices") or (
        6000 if manifest["dataset"] == "synthetic-web-small" else 12000
    ))
    edges = int(trace.get("graph_edges") or vertices * (
        5 if manifest["dataset"] == "synthetic-web-small" else 6
    ))
    replicated_vertices = replication_vertices * vertices
    replicated_edges = replication_edges * edges

    t_partition = (
        replicated_edges / calibration["partition_edge_rate_edges_per_s"]
        + replicated_vertices / calibration["partition_score_rate_vertices_per_s"]
    )
    t_presample = (32 * 128 * 4) / calibration["sample_rate_features_per_s"]
    per_worker_cache_records = max(64, vertices // (workers * 4))
    cache_bytes = per_worker_cache_records * workers * feature_bytes
    t_cache = cache_bytes / calibration["cache_build_rate_bytes_per_s"]
    t_ship = (replicated_vertices * feature_bytes + cache_bytes) / calibration["ship_rate_bytes_per_s"]

    c_srv = float(trace.get("C_srv_bytes") or 0.0)
    c_edg = float(trace.get("C_edg_bytes") or 0.0)
    effective_b = float(trace.get("effective_B_bytes_s") or 1_000_000_000)
    network_rtt = float(trace.get("network_rtt_s") or 0.0)
    network_loss = float(trace.get("network_loss_rate") or 0.0)
    # The trace represents 32 mini-batches; each batch incurs at least one
    # feature round trip. Loss is modeled as a bounded retransmission factor.
    retransmission_factor = 1.0 + min(network_loss * 20.0, 2.0)
    network_latency = 32 * network_rtt * retransmission_factor
    dgl_feature_time = (
        c_srv / min(effective_b, calibration["feature_extract_rate_bytes_per_s"])
        + network_latency
    )
    edg_feature_time = (
        c_edg / min(effective_b, calibration["feature_extract_rate_bytes_per_s"])
        + network_latency * min(1.0, c_edg / max(c_srv, 1.0))
    )
    accessed_features = c_srv / feature_bytes
    edg_accessed_features = c_edg / feature_bytes

    capability_multiplier = 1.0
    if manifest["block"] == "cpu-emulation" and str(manifest["control"]).endswith("even"):
        # Even placement wastes time waiting for the 600-MHz class.
        capability_multiplier = 1.8
    elif manifest["block"] == "cpu-emulation":
        capability_multiplier = 1.25

    dgl_compute = accessed_features * calibration["compute_s_per_accessed_feature"] * capability_multiplier
    edg_compute = (
        edg_accessed_features * calibration["compute_s_per_accessed_feature"]
        + accessed_features * 0.25 * calibration["compute_s_per_accessed_feature"]
    ) * capability_multiplier
    dgl_sync = calibration["sync_fixed_s"] * capability_multiplier
    edg_sync = calibration["sync_fixed_s"]

    dgl_epoch = dgl_feature_time + dgl_compute + dgl_sync
    edg_epoch = edg_feature_time + edg_compute + edg_sync
    if str(manifest["control"]) == "+RCSP":
        # A bounded one-batch prefetch can hide at most the shorter of feature
        # fetch and compute; it cannot hide synchronization.
        edg_epoch -= min(edg_feature_time, edg_compute) * 0.75

    delta_c = c_srv - c_edg
    delta_t = (edg_compute - dgl_compute) + (edg_sync - dgl_sync)
    t_pre = t_partition + t_presample + t_cache + t_ship
    horizon = int(manifest.get("horizon_epochs") or 100)
    denominator = t_pre + horizon * delta_t
    b_star = horizon * delta_c / denominator if denominator > 0 else None
    break_even = None
    if delta_c - effective_b * delta_t > 0:
        break_even = math.ceil(t_pre * effective_b / (delta_c - effective_b * delta_t))

    total_time_to_target = t_pre + edg_epoch * 20
    peak_memory_mb = (replicated_vertices * feature_bytes + cache_bytes) / (1024.0 * 1024.0)
    worker_energy = calibration["worker_power_w"] * edg_epoch * workers
    server_energy = calibration["server_power_w"] * edg_epoch
    switch_energy = calibration["switch_power_w"] * edg_epoch
    energy_to_target = (
        calibration["worker_power_w"] * workers
        + calibration["server_power_w"]
        + calibration["switch_power_w"]
    ) * total_time_to_target

    return {
        "measurement_source": "parameterized-emulated-edge-v1",
        "measurement_valid": bool(calibration.get("publishable", False)),
        "evidence_level": "parameterized-emulation",
        "environment_type": "cpu-memory-network-model",
        "hardware_platform": "calibrated-model",
        "validity_scope": "relative-timing-and-energy-trends-only",
        "graph_source": trace.get("graph_source"),
        "runner_version": "parameterized-emulated-edge-v1",
        "calibration_source": calibration["calibration_source"],
        "T_partition_s": t_partition,
        "T_presample_s": t_presample,
        "T_cache_s": t_cache,
        "T_ship_s": t_ship,
        "T_pre_total_s": t_pre,
        "T_comp_dgl_s": dgl_compute,
        "T_comp_edgae_s": edg_compute,
        "T_sync_dgl_s": dgl_sync,
        "T_sync_edgae_s": edg_sync,
        "DGL_E_epoch_s": dgl_epoch,
        "EDGaE_epoch_s": edg_epoch,
        "Delta_C_bytes": delta_c,
        "Delta_T_s": delta_t,
        "B_star_bytes_s": b_star,
        "break_even_epochs": break_even,
        "total_time_to_target_s": total_time_to_target,
        "target_accuracy": None,
        "test_accuracy": None,
        "peak_rss_mb": peak_memory_mb,
        "peak_pss_mb": peak_memory_mb,
        "worker_energy_j": worker_energy,
        "server_energy_j": server_energy,
        "switch_energy_j": switch_energy,
        "idle_power_w": calibration["worker_power_w"] * workers
                         + calibration["server_power_w"]
                         + calibration["switch_power_w"],
        "energy_to_target_j": energy_to_target,
        "gradient_wait_s": max(0.0, edg_sync),
        "notes": (
            "Parameterized model output; not a native DGL/PyTorch measurement "
            "and not a Raspberry Pi measurement."
        ),
    }
