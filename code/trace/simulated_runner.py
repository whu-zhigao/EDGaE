#!/usr/bin/env python3
"""Deterministic smoke-test runner.

IMPORTANT: this runner does not measure EDGaE.  It exists only to test the
orchestrator, CSV schema, metric sampler, and analysis formulas end-to-end.
Never copy its values into the manuscript.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
import os


DATASET_SCALE = {
    "amazon0601": 1.0,
    "web-google": 1.3,
    "wiki-talk": 1.8,
    "ogbn-arxiv": 0.8,
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: simulated_runner.py MANIFEST.json", file=sys.stderr)
        return 2
    if os.environ.get("EDGAE_ALLOW_SIMULATION") != "1":
        print(
            "Refusing to simulate without EDGAE_ALLOW_SIMULATION=1; "
            "simulated values are not manuscript measurements.",
            file=sys.stderr,
        )
        return 3

    manifest = json.loads(Path(sys.argv[1]).read_text())
    key = "|".join(str(manifest.get(k, "")) for k in [
        "block", "dataset", "model", "sampler", "workers", "control", "seed",
    ])
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))
    scale = DATASET_SCALE.get(manifest.get("dataset", ""), 1.0)
    workers = int(manifest.get("workers", 8))

    dgl_epoch = (150 + rng.random() * 350) * scale
    speedup = 2.2 + rng.random() * 15
    edgae_epoch = dgl_epoch / speedup
    t_partition = 2 + rng.random() * 18
    t_presample = 1 + rng.random() * 8
    t_cache = 1 + rng.random() * 10
    t_ship = 2 + rng.random() * 20
    c_srv = int((50 + rng.random() * 300) * 1024 * 1024)
    c_edg = int(c_srv * (0.12 + rng.random() * 0.35))
    t_comp_dgl = dgl_epoch * (0.35 + rng.random() * 0.15)
    t_comp_edg = t_comp_dgl * (0.8 + rng.random() * 0.35)
    t_sync_dgl = dgl_epoch * (0.05 + rng.random() * 0.08)
    t_sync_edg = t_sync_dgl * (0.6 + rng.random() * 0.6)

    native = manifest.get("dataset") == "ogbn-arxiv"
    if native:
        dgl_acc = 0.51 + rng.random() * 0.07
        edgae_acc = dgl_acc + rng.normalvariate(0, 0.002)
    else:
        dgl_acc = 0.75 + rng.random() * 0.20
        edgae_acc = dgl_acc + rng.normalvariate(0, 0.002)

    control = str(manifest.get("control", ""))
    cache_hit = 0.45 + rng.random() * 0.40
    if manifest.get("block") == "cache-stability":
        if control == "K=full-trace":
            cache_hit = 0.86 + rng.random() * 0.06
            topk_jaccard = 1.0
            boundary_gap = 0.0
        else:
            k = int(control.split("=", 1)[1])
            quality = 1.0 / (1.0 + math.exp(-(k - 2.5) / 1.5))
            cache_hit = 0.52 + 0.38 * quality + rng.random() * 0.02
            topk_jaccard = max(0.0, min(1.0, 0.60 + 0.38 * quality + rng.random() * 0.02))
            boundary_gap = 1e-5 + rng.random() * 1e-4
    else:
        topk_jaccard = None
        boundary_gap = None

    result = {
        "measurement_source": "SIMULATED_SMOKE_TEST_ONLY",
        "measurement_valid": False,
        "T_partition_s": t_partition,
        "T_presample_s": t_presample,
        "T_cache_s": t_cache,
        "T_ship_s": t_ship,
        "T_comp_dgl_s": t_comp_dgl,
        "T_comp_edgae_s": t_comp_edg,
        "T_sync_dgl_s": t_sync_dgl,
        "T_sync_edgae_s": t_sync_edg,
        "DGL_E_epoch_s": dgl_epoch,
        "EDGaE_epoch_s": edgae_epoch,
        "C_srv_bytes": c_srv,
        "C_edg_bytes": c_edg,
        "effective_B_bytes_s": int(900 + rng.random() * 40) * 1000 * 1000 // 8,
        "total_time_to_target_s": t_partition + t_presample + t_cache + t_ship + edgae_epoch * 20,
        "target_accuracy": 0.90,
        "test_accuracy": edgae_acc if native else None,
        "cache_hit_rate": cache_hit,
        "cache_coverage": 0.35 + rng.random() * 0.25,
        "topk_jaccard": topk_jaccard,
        "boundary_gap": boundary_gap,
        "replication_factor_vertices": 1.1 + rng.random() * 1.4,
        "replication_factor_edges": 1.0 + rng.random() * 0.9,
        "load_imilibrium_typo_should_not_exist": None,
        "load_imbalance": rng.random() * 0.18,
        "feasibility_failures": rng.randint(0, 2),
        "fallbacks": rng.randint(0, 5),
        "peak_rss_mb": 900 + rng.random() * 1200,
        "peak_pss_mb": 850 + rng.random() * 1100,
        "worker_energy_j": 400 + rng.random() * 900,
        "server_energy_j": 100 + rng.random() * 300,
        "switch_energy_j": 5 + rng.random() * 20,
        "idle_power_w": 20 + rng.random() * 10,
        "energy_to_target_j": 1000 + rng.random() * 5000,
        "gradient_wait_s": rng.random() * 8,
        "runner_accuracy_dgl": dgl_acc,
        "runner_accuracy_edgae": edgae_acc,
        "workers_seen": workers,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
