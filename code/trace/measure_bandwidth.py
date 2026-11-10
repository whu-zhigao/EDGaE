#!/usr/bin/env python3
"""Measure effective TCP throughput with iperf3."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="iperf3 server address")
    parser.add_argument("--port", type=int, default=5201)
    parser.add_argument("--duration", type=positive_float, default=10.0)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    iperf = shutil.which("iperf3")
    if not iperf:
        print("iperf3 is required", file=sys.stderr)
        return 2
    command = [
        iperf, "-c", args.server, "-p", str(args.port),
        "-t", str(args.duration), "-P", str(args.parallel), "-J",
    ]
    if args.reverse:
        command.append("-R")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(result.stdout or result.stderr, file=sys.stderr)
        return result.returncode or 3
    if result.returncode != 0:
        print(json.dumps(payload, indent=2), file=sys.stderr)
        return result.returncode

    end = payload.get("end", {})
    if args.reverse and "sum_received" in end:
        bits_per_second = end["sum_received"].get("bits_per_second")
    else:
        bits_per_second = end.get("sum_sent", {}).get("bits_per_second")
    if bits_per_second is None:
        print("iperf3 JSON did not contain bits_per_second", file=sys.stderr)
        return 4
    print(json.dumps({
        "effective_B_bytes_s": bits_per_second / 8.0,
        "bits_per_second": bits_per_second,
        "parallel_streams": args.parallel,
        "reverse": args.reverse,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
