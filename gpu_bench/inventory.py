"""Best-effort inventory; unavailable values stay null, never zero."""
import csv
import io
import math
import platform
import os
from pathlib import Path
import re
import socket
import subprocess

from .common import utc_now

GPU_FIELDS = {
    "uuid": "uuid", "name": "name", "pci_bus_id": "pci.bus_id",
    "driver_version": "driver_version", "memory_total_mib": "memory.total",
    "memory_used_mib": "memory.used", "power_draw_w": "power.draw",
    "power_limit_w": "power.limit", "temperature_c": "temperature.gpu",
    "utilization_percent": "utilization.gpu",
}
TEXT_FIELDS = {"uuid", "name", "pci_bus_id", "driver_version"}


def command(argv):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
        if result.returncode:
            return None, f"{argv[0]} exited {result.returncode}"
        return result.stdout.strip(), None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{argv[0]}: {type(exc).__name__}"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (TypeError, ValueError):
        return None


def gpu_snapshot(uuids=None):
    output, error = command(["nvidia-smi", "--query-gpu=" + ",".join(GPU_FIELDS.values()),
                             "--format=csv,noheader,nounits"])
    if error:
        return [], [error]
    records = []
    warnings = []
    for row in csv.reader(io.StringIO(output or ""), skipinitialspace=True):
        if len(row) != len(GPU_FIELDS):
            warnings.append("Unexpected nvidia-smi column count")
            continue
        record = {key: (value.strip() if key in TEXT_FIELDS else number(value.strip()))
                  for key, value in zip(GPU_FIELDS, row)}
        if not uuids or record["uuid"] in uuids:
            records.append(record)
    if uuids:
        missing = set(uuids) - {r["uuid"] for r in records}
        warnings.extend(f"GPU unavailable: {uuid}" for uuid in sorted(missing))
    return records, warnings


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def inventory(machine_id):
    gpus, warnings = gpu_snapshot()
    smi, _ = command(["nvidia-smi"])
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi or "")
    cpu = platform.processor() or None
    cpuinfo = read_text("/proc/cpuinfo")
    if cpuinfo:
        match = re.search(r"(?:model name|Hardware)\s*:\s*(.+)", cpuinfo)
        cpu = match.group(1) if match else cpu
    memory = read_text("/proc/meminfo")
    match = re.search(r"MemTotal:\s*(\d+) kB", memory or "")
    docker, _ = command(["docker", "--version"])
    toolkit, _ = command(["nvidia-ctk", "--version"])
    nvcc, _ = command(["nvcc", "--version"])
    return {
        "schema_version": "1.0", "record_type": "inventory", "captured_at": utc_now(),
        "machine_id": machine_id, "hostname": socket.gethostname(),
        "scope": "container" if Path("/.dockerenv").exists() else "host",
        "system": {"os": platform.system(), "release": platform.release(),
                   "architecture": platform.machine(), "cpu_model": cpu,
                   "logical_cpus": os.cpu_count(),
                   "memory_total_bytes": int(match.group(1)) * 1024 if match else None,
                   "os_release": read_text("/etc/os-release"), "python": platform.python_version()},
        "software": {"docker": docker, "nvidia_container_toolkit": toolkit,
                     "cuda_toolkit_nvcc": nvcc,
                     "driver_cuda_max_version": cuda.group(1) if cuda else None},
        "gpus": gpus, "warnings": warnings,
    }
