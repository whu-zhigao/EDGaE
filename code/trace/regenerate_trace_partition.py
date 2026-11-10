#!/usr/bin/env python3
"""Regenerate the 4-panel partition-policy comparison figure from cloud data.

Thin wrapper: the canonical builder lives in paper/figsource/build_fig_partition.py
(ICDE_v2 styling, reads results_cloud/results/icde2027-v2-trace-results.csv and
also exports partition-data.xlsx). Kept so historical invocations keep working.
"""
import runpy
import sys
from pathlib import Path

BUILDER = (Path(__file__).resolve().parents[2] / "paper" / "figsource"
           / "build_fig_partition.py")

if __name__ == "__main__":
    runpy.run_path(str(BUILDER), run_name="__main__")
    sys.exit(0)
