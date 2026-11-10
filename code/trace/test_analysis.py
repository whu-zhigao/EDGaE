#!/usr/bin/env python3
"""Unit tests for derived Appendix-B quantities."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_extended_results import bootstrap_ci, derive, fmt


class DerivationTests(unittest.TestCase):
    def test_break_even_derivation(self):
        row = {
            "T_partition_s": 10,
            "T_presample_s": 5,
            "T_cache_s": 5,
            "T_ship_s": 20,
            "T_comp_dgl_s": 20,
            "T_comp_edgae_s": 22,
            "T_sync_dgl_s": 5,
            "T_sync_edgae_s": 4,
            "C_srv_bytes": 1000,
            "C_edg_bytes": 400,
            "effective_B_bytes_s": 10,
            "horizon_epochs": 100,
        }
        out = derive(row)
        self.assertAlmostEqual(out["T_pre_total_s"], 40.0)
        self.assertAlmostEqual(out["Delta_C_bytes"], 600.0)
        self.assertAlmostEqual(out["Delta_T_s"], 1.0)
        self.assertAlmostEqual(out["B_star_bytes_s"], 60000.0 / 140.0)
        self.assertEqual(out["break_even_epochs"], 1)

    def test_large_break_even_when_saved_bytes_are_small(self):
        row = {
            "T_partition_s": 100,
            "T_presample_s": 0,
            "T_cache_s": 0,
            "T_ship_s": 0,
            "C_srv_bytes": 101,
            "C_edg_bytes": 100,
            "effective_B_bytes_s": 10,
            "T_comp_dgl_s": 0,
            "T_comp_edgae_s": 0,
            "T_sync_dgl_s": 0,
            "T_sync_edgae_s": 0,
            "horizon_epochs": 100,
        }
        out = derive(row)
        self.assertEqual(out["break_even_epochs"], 1000)

    def test_bootstrap_ci_order(self):
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        lo, hi = bootstrap_ci(values, 500, 7)
        self.assertLessEqual(lo, hi)

    def test_formatter(self):
        self.assertEqual(fmt(None), "--")
        self.assertEqual(fmt(1234.0), "1,234")
        self.assertEqual(fmt(0.123456), "0.123")


if __name__ == "__main__":
    unittest.main()
