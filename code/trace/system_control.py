"""Optional local CPU-frequency and network-emulation controls."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


def _run(command: List[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    return result.stdout


class CPUFrequencyControl:
    """Set a local maximum CPU frequency; mainly for single-node debugging."""

    def __init__(self, frequency_mhz: int, allow: bool = False):
        self.frequency_mhz = frequency_mhz
        self.allow = allow
        self.saved: List[tuple[Path, str]] = []

    @staticmethod
    def _cpu_paths() -> List[Path]:
        base = Path("/sys/devices/system/cpu")
        out: List[Path] = []
        if not base.exists():
            return out
        for cpu in sorted(base.glob("cpu[0-9]*")):
            path = cpu / "cpufreq" / "scaling_max_freq"
            if path.exists():
                out.append(path)
        return out

    def __enter__(self):
        paths = self._cpu_paths()
        if not paths:
            if self.allow:
                raise RuntimeError("no cpufreq scaling_max_freq files found")
            return self
        if not self.allow:
            raise RuntimeError("refusing to change CPU frequency without --allow-system-control")
        target_hz = str(int(self.frequency_mhz * 1000))
        for path in paths:
            old = path.read_text().strip()
            self.saved.append((path, old))
            path.write_text(target_hz)
        return self

    def __exit__(self, exc_type, exc, tb):
        for path, old in self.saved:
            try:
                path.write_text(old)
            except OSError:
                pass
        return False


class NetworkEmulation:
    """Apply tbf rate and netem delay/loss to one local interface."""

    def __init__(self, interface: str, control: str, allow: bool = False):
        self.interface = interface
        self.control = control
        self.allow = allow
        self.applied = False

    @staticmethod
    def _parse(control: str):
        match = re.match(r"(\d+)(Mbps)-(\d+)ms(?:-(\d+(?:\.\d+)?)%loss)?", control)
        if not match:
            raise ValueError(f"unsupported network control: {control}")
        rate_mbps = int(match.group(1))
        rtt_ms = float(match.group(3))
        loss = float(match.group(4) or 0.0)
        return rate_mbps, rtt_ms / 2.0, loss

    def __enter__(self):
        if not self.allow:
            return self
        if not shutil.which("tc"):
            raise RuntimeError("tc is required for network emulation")
        rate_mbps, delay_ms, loss = self._parse(self.control)
        rate = f"{rate_mbps}mbit"
        delay = f"{delay_ms:.3f}ms"
        loss_text = f"{loss:.3f}%"
        # Root netem first, then a TBF child.  This is a local debugging setup;
        # cluster experiments should apply the same policy on the edge uplink.
        _run(["tc", "qdisc", "replace", "dev", self.interface,
              "root", "handle", "1:", "netem", "delay", delay, "loss", loss_text])
        _run(["tc", "qdisc", "add", "dev", self.interface,
              "parent", "1:", "handle", "2:", "tbf", "rate", rate,
              "burst", "256kb", "latency", "400ms"])
        self.applied = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.applied:
            try:
                _run(["tc", "qdisc", "del", "dev", self.interface, "root"])
                _run(["tc", "qdisc", "add", "dev", self.interface,
                      "root", "fq_codel"])
            except RuntimeError:
                # Do not hide an experiment exception because restoration failed.
                pass
        return False
