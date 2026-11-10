#!/usr/bin/env python3
"""Run Appendix-B experiments through an executable runner contract."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any, Dict, Iterable, List, Optional

from metric_sampler import run_sampled_command
from schema import COLUMNS
from system_control import CPUFrequencyControl, NetworkEmulation


BLOCKS = [
    "break-even",
    "native-task",
    "cpu-emulation",
    "network-sensitivity",
    "cache-stability",
    "component-ablation",
    "partition-policy",
    "score-sensitivity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="www2027-extended-results-template.csv")
    parser.add_argument("--output", default="www2027-extended-results-measured.csv")
    parser.add_argument("--raw-dir", default="experiment-raw")
    parser.add_argument("--runner-command", help="Executable followed by fixed arguments")
    parser.add_argument("--simulate", action="store_true",
                        help="Smoke-test the pipeline with fake, invalid data")
    parser.add_argument("--blocks", nargs="+", default=BLOCKS, choices=BLOCKS)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--samplers", nargs="+")
    parser.add_argument("--worker-counts", nargs="+", type=int)
    parser.add_argument("--controls", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--power-command", help="Command whose last numeric output is watts")
    parser.add_argument("--power-scope", choices=["worker", "server", "none"], default="none")
    parser.add_argument("--network-interface")
    parser.add_argument("--local-frequency-mhz", type=int,
                        help="Local cap for single-node CPU-emulation debugging")
    parser.add_argument("--allow-system-control", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def selected(row: Dict[str, str], args: argparse.Namespace) -> bool:
    if row["block"] not in args.blocks:
        return False
    if args.datasets and row["dataset"] not in args.datasets:
        return False
    if args.models and row["model"] not in args.models:
        return False
    if args.samplers and row["sampler"] not in args.samplers:
        return False
    if args.worker_counts and int(row["workers"]) not in args.worker_counts:
        return False
    if args.controls and row["control"] not in args.controls:
        return False
    return True


def command_for(args: argparse.Namespace, manifest_path: Path) -> List[str]:
    if args.simulate:
        return [sys.executable, str(Path(__file__).with_name("simulated_runner.py")),
                str(manifest_path)]
    if not args.runner_command:
        raise SystemExit(
            "--runner-command is required for real measurements. It must accept a "
            "JSON manifest path and print a JSON result object."
        )
    command = shlex.split(args.runner_command)
    if "{manifest}" in command:
        return [str(manifest_path) if token == "{manifest}" else token for token in command]
    return command + [str(manifest_path)]


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def result_row(base: Dict[str, str], runner: Dict[str, Any], local: Dict[str, Any],
               run_id: str, source: str) -> Dict[str, Any]:
    row: Dict[str, Any] = dict(base)
    row["run_id"] = run_id
    row["measurement_utc"] = datetime.now(timezone.utc).isoformat()
    row["measurement_source"] = str(runner.get("measurement_source", source))
    measurement_valid = runner.get("measurement_valid", True)
    if local.get("returncode") != 0 or local.get("error"):
        measurement_valid = False
    row["measurement_valid"] = bool(measurement_valid)

    for key in COLUMNS:
        if key in runner:
            row[key] = csv_value(runner[key])

    # Local process-tree observations fill fields omitted by the runner.
    mappings = {
        "peak_rss_mb": local.get("peak_hwm_mb") or local.get("peak_rss_mb"),
        "peak_pss_mb": local.get("peak_pss_mb"),
        "page_cache_mb": local.get("page_cache_mb"),
        "sdcard_read_mb": local.get("sdcard_read_mb"),
        "page_faults": local.get("page_faults"),
    }
    for key, value in mappings.items():
        if value is not None and not row.get(key):
            row[key] = value

    if args_power_scope == "worker" and local.get("sampled_energy_j") is not None:
        row.setdefault("worker_energy_j", local["sampled_energy_j"])
    elif args_power_scope == "server" and local.get("sampled_energy_j") is not None:
        row.setdefault("server_energy_j", local["sampled_energy_j"])

    if local.get("error"):
        row["notes"] = (row.get("notes", "") + " " + str(local["error"])).strip()
    return row


# Populated in main to avoid passing an extra global through run_sampled_command.
args_power_scope = "none"


def write_raw(raw_dir: Path, run_id: str, payload: Dict[str, Any]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{run_id}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def append_row(path: Path, row: Dict[str, Any]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    global args_power_scope
    args = parse_args()
    args_power_scope = args.power_scope

    template = Path(args.template)
    output = Path(args.output)
    raw_dir = Path(args.raw_dir)
    if not template.exists():
        raise SystemExit(f"template not found: {template}")
    if not args.append and output.exists():
        raise SystemExit(f"output exists (use --append to resume): {output}")

    env = os.environ.copy()
    if args.simulate:
        env["EDGAE_ALLOW_SIMULATION"] = "1"
        os.environ["EDGAE_ALLOW_SIMULATION"] = "1"

    with template.open(newline="", encoding="utf-8") as f:
        configurations = [row for row in csv.DictReader(f) if selected(row, args)]

    executed = 0
    failed = 0
    source = "simulated-smoke-test" if args.simulate else "runner"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    for config_index, base in enumerate(configurations, start=1):
        for seed in args.seeds:
            run_index = executed + 1
            run_id = f"{timestamp}-{run_index:05d}-{base['block']}-{seed}"
            manifest = {
                **base,
                "seed": seed,
                "run_id": run_id,
                "epochs_after_warmup": 10,
                "warmup_epochs": 1,
                "accuracy_seeds": 5,
                "power_scope": args.power_scope,
            }
            manifest_path = raw_dir / f"{run_id}-manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            command = command_for(args, manifest_path)

            cpu_control = CPUFrequencyControl(
                args.local_frequency_mhz or 1500,
                allow=args.allow_system_control,
            ) if base["block"] == "cpu-emulation" and args.local_frequency_mhz else None
            network_control = NetworkEmulation(
                args.network_interface or "eth0", base["control"],
                allow=args.allow_system_control,
            ) if base["block"] == "network-sensitivity" and args.network_interface else None

            print(f"[{config_index}/{len(configurations)} seed={seed}] {run_id}")
            try:
                context_cpu = cpu_control.__enter__() if cpu_control else None
                context_network = network_control.__enter__() if network_control else None
                try:
                    runner, local, stdout, stderr = run_sampled_command(
                        command,
                        interval=args.sample_interval,
                        timeout=args.timeout,
                        power_command=args.power_command,
                    )
                finally:
                    if network_control:
                        network_control.__exit__(None, None, None)
                    if cpu_control:
                        cpu_control.__exit__(None, None, None)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                failed += 1
                runner, local = {}, {"returncode": -1, "error": str(exc)}
                stdout, stderr = "", str(exc)

            row = result_row(base, runner, local, run_id, source)
            row["seed"] = seed
            append_row(output, row)
            write_raw(raw_dir, run_id, {
                "manifest": manifest,
                "command": command,
                "runner_json": runner,
                "local_metrics": local,
                "stdout": stdout,
                "stderr": stderr,
            })
            executed += 1
            if local.get("returncode") not in (0, None):
                failed += 1

    print(f"executed={executed} failed={failed} output={output}")
    return 1 if failed and not args.continue_on_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
