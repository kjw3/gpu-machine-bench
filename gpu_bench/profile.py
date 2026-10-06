"""Public hardware profiles built from an explicit allowlist, never raw inventory."""
import csv
import io
import json
import os
import platform
import re
from pathlib import Path

from .inventory import command, number, read_text


def version(text):
    match = re.search(r"\b\d+\.\d+(?:\.\d+)?\b", text or "")
    return match.group() if match else None


def gpu_column(field):
    raw, error = command(["nvidia-smi", "--query-gpu=" + field,
                          "--format=csv,noheader,nounits"])
    if error or not raw:
        return None
    return [row[0].strip() for row in csv.reader(io.StringIO(raw)) if len(row) == 1]


def cpu_groups(raw):
    """Preserve repeated model sections on heterogeneous CPUs."""
    groups = []
    fields = {"Core(s) per socket": "cores_per_socket", "Socket(s)": "sockets",
              "Thread(s) per core": "threads_per_core"}
    try:
        for row in json.loads(raw or "{}").get("lscpu", []):
            key = row["field"].rstrip(":")
            if key == "Model name":
                groups.append({"model": row["data"], **{k: None for k in fields.values()}})
            elif groups and key in fields:
                groups[-1][fields[key]] = number(row["data"])
    except (ValueError, TypeError, KeyError):
        return []
    return groups


def public_profile(profile_id):
    # Neutral labels cannot accidentally contain a hostname, user or asset name.
    if not re.fullmatch(r"machine-[0-9]{3}", profile_id):
        raise ValueError("Profile ID must be a neutral label such as machine-001")
    cpu_raw, _ = command(["env", "LC_ALL=C", "lscpu", "--json"])
    mem = re.search(r"MemTotal:\s*(\d+) kB", read_text("/proc/meminfo") or "")
    names = gpu_column("name")
    fields = {
        "memory_total_mib": "memory.total", "power_limit_w": "power.limit",
        "max_sm_clock_mhz": "clocks.max.sm", "max_memory_clock_mhz": "clocks.max.memory",
        "pcie_max_generation": "pcie.link.gen.max", "pcie_max_width": "pcie.link.width.max",
        "compute_capability": "compute_cap", "driver_version": "driver_version",
    }
    columns = {key: gpu_column(field) for key, field in fields.items()} if names else {}
    gpus = []
    for i, name in enumerate(names or []):
        gpu = {"device": f"gpu-{i + 1}", "model": name,
               "memory_architecture": "unified" if name == "NVIDIA GB10" else "unspecified"}
        for key, values in columns.items():
            value = values[i] if values and len(values) == len(names) else None
            gpu[key] = version(value) if key in ("driver_version", "compute_capability") else number(value)
        if name.startswith("NVIDIA GeForce") and gpu["memory_total_mib"] is not None:
            gpu["memory_architecture"] = "dedicated"
        gpus.append(gpu)
    smi, _ = command(["nvidia-smi"])
    cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi or "")
    docker, _ = command(["docker", "--version"])
    toolkit, _ = command(["nvidia-ctk", "--version"])
    return {
        "schema_version": "1.0", "record_type": "public_compute_profile",
        "profile_id": profile_id,
        "collection_scope": "container" if Path("/.dockerenv").exists() else "host",
        "system": {"os_family": platform.system(), "architecture": platform.machine(),
                   "cpu_groups": cpu_groups(cpu_raw), "logical_cpus": os.cpu_count(),
                   "memory_total_gib": round(int(mem.group(1)) / 1024**2, 1) if mem else None},
        "software": {"docker_version": version(docker), "nvidia_container_toolkit_version": version(toolkit),
                     "driver_cuda_max_version": cuda.group(1) if cuda else None},
        "gpus": gpus,
        "capabilities": {"nvidia_gpu_count": len(gpus), "cuda_devices_detected": bool(gpus),
                         "multi_gpu": len(gpus) > 1,
                         "shared_system_gpu_memory": any(g["memory_architecture"] == "unified" for g in gpus)},
        "limitations": ["Hardware discovery does not verify framework compatibility or model fit.",
                        "CUDA version is driver compatibility, not installed toolkit version.",
                        "GPU memory is per device; multiple devices are not a shared VRAM pool.",
                        "Null values indicate unavailable or unsupported counters.",
                        "Hardware combinations may be recognizable despite removal of direct identifiers."],
    }
