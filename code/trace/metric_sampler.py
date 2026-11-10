"""Run an experiment command and sample local process-tree metrics."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _children(pid: int) -> Iterable[int]:
    text = _read_text(Path(f"/proc/{pid}/task/{pid}/children"))
    for token in text.split():
        try:
            yield int(token)
        except ValueError:
            pass


def _process_tree(root_pid: int) -> List[int]:
    seen: set[int] = set()
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen or not Path(f"/proc/{pid}").exists():
            continue
        seen.add(pid)
        stack.extend(_children(pid))
    return sorted(seen)


def _status_value(pid: int, key: str) -> float:
    for line in _read_text(Path(f"/proc/{pid}/status")).splitlines():
        if line.startswith(key + ":"):
            match = re.search(r"([0-9]+)", line.split(":", 1)[1])
            return float(match.group(1)) if match else 0.0
    return 0.0


def _pss_kb(pid: int) -> float:
    for line in _read_text(Path(f"/proc/{pid}/smaps_rollup")).splitlines():
        if line.startswith("Pss:"):
            match = re.search(r"([0-9]+)", line.split(":", 1)[1])
            return float(match.group(1)) if match else 0.0
    return 0.0


def _faults(pid: int) -> Tuple[int, int]:
    text = _read_text(Path(f"/proc/{pid}/stat"))
    if ")" not in text:
        return 0, 0
    tail = text.rsplit(")", 1)[1].split()
    if len(tail) < 10:
        return 0, 0
    try:
        minor = int(tail[7])
        major = int(tail[9])
        return minor, major
    except ValueError:
        return 0, 0


def _read_bytes(pid: int) -> float:
    for line in _read_text(Path(f"/proc/{pid}/io")).splitlines():
        if line.startswith("read_bytes:"):
            try:
                return float(line.split(":", 1)[1].strip())
            except ValueError:
                return 0.0
    return 0.0


def _page_cache_mb() -> float:
    cached = 0.0
    reclaimable = 0.0
    for line in _read_text(Path("/proc/meminfo")).splitlines():
        if line.startswith("Cached:"):
            match = re.search(r"([0-9]+)", line.split(":", 1)[1])
            cached = float(match.group(1)) if match else 0.0
        elif line.startswith("SReclaimable:"):
            match = re.search(r"([0-9]+)", line.split(":", 1)[1])
            reclaimable = float(match.group(1)) if match else 0.0
    return (cached + reclaimable) / 1024.0


def _sample_process_tree(root_pid: int) -> Dict[str, float]:
    rss = hwm = pss = read_bytes = 0.0
    faults = 0
    for pid in _process_tree(root_pid):
        rss += _status_value(pid, "VmRSS")
        hwm += _status_value(pid, "VmHWM")
        pss += _pss_kb(pid)
        read_bytes += _read_bytes(pid)
        minor, major = _faults(pid)
        faults += minor + major
    return {
        "rss_kb": rss,
        "hwm_kb": hwm,
        "pss_kb": pss,
        "read_bytes": read_bytes,
        "faults": faults,
    }


def _parse_power(text: str) -> Optional[float]:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


def _sample_power(power_command: Optional[str]) -> Optional[float]:
    if not power_command:
        return None
    try:
        result = subprocess.run(
            power_command,
            shell=True,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        return _parse_power(result.stdout or result.stderr)
    except (OSError, subprocess.SubprocessError):
        return None


def _extract_json_objects(text: str) -> List[Any]:
    decoder = json.JSONDecoder()
    objects: List[Any] = []
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
            objects.append(obj)
            index = start + end
        except json.JSONDecodeError:
            index = start + 1
    return objects


def run_sampled_command(
    command: List[str],
    interval: float = 0.5,
    timeout: Optional[float] = None,
    power_command: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], str, str]:
    """Execute a command and return runner JSON plus local measurements."""
    started = time.time()
    stdout_fd, stdout_name = tempfile.mkstemp(prefix="edgae-stdout-", suffix=".json")
    os.close(stdout_fd)
    stdout_path = Path(stdout_name)
    try:
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file:
            proc = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
                text=True,
            )
            max_rss = max_hwm = max_pss = max_page_cache = 0.0
            start_read_bytes = 0.0
            last_read_bytes = 0.0
            start_faults = 0
            last_faults = 0
            power_samples: List[Tuple[float, float]] = []
            last_error: Optional[str] = None

            while True:
                if proc.poll() is not None:
                    break
                elapsed = time.time() - started
                if timeout is not None and elapsed > timeout:
                    proc.kill()
                    last_error = f"timeout after {timeout:.1f}s"
                    break
                sample = _sample_process_tree(proc.pid)
                max_rss = max(max_rss, sample["rss_kb"])
                max_hwm = max(max_hwm, sample["hwm_kb"])
                max_pss = max(max_pss, sample["pss_kb"])
                max_page_cache = max(max_page_cache, _page_cache_mb())
                if start_read_bytes == 0.0:
                    start_read_bytes = sample["read_bytes"]
                last_read_bytes = sample["read_bytes"]
                if start_faults == 0:
                    start_faults = sample["faults"]
                last_faults = sample["faults"]
                watts = _sample_power(power_command)
                if watts is not None:
                    power_samples.append((time.time(), watts))
                time.sleep(max(0.05, interval))

            stderr = proc.stderr.read() if proc.stderr is not None else ""
            returncode = proc.wait()

        stdout = stdout_path.read_text(errors="replace")
        objects = _extract_json_objects(stdout)
        runner_json = objects[-1] if objects and isinstance(objects[-1], dict) else {}

        energy_j = None
        if len(power_samples) >= 2:
            energy_j = 0.0
            for (t0, p0), (t1, p1) in zip(power_samples, power_samples[1:]):
                energy_j += max(0.0, (t1 - t0) * (p0 + p1) / 2.0)

        local_metrics: Dict[str, Any] = {
            "returncode": returncode,
            "wall_time_s": time.time() - started,
            "peak_rss_mb": max_rss / 1024.0 if max_rss else None,
            "peak_hwm_mb": max_hwm / 1024.0 if max_hwm else None,
            "peak_pss_mb": max_pss / 1024.0 if max_pss else None,
            "page_cache_mb": max_page_cache if max_page_cache else None,
            "sdcard_read_mb": (last_read_bytes - start_read_bytes) / (1024.0 * 1024.0),
            "page_faults": max(0, last_faults - start_faults),
            "sampled_energy_j": energy_j,
            "power_samples": [p for _, p in power_samples],
            "metric_scope": "local-process-tree",
            "error": last_error,
        }
        if returncode != 0:
            local_metrics["error"] = last_error or f"runner exited with code {returncode}"
        return runner_json, local_metrics, stdout, stderr
    finally:
        try:
            stdout_path.unlink()
        except OSError:
            pass
