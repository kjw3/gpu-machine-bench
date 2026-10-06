"""Run lifecycle, provenance, and aggregate metrics independent of adapters."""
import copy
import math
import statistics
import time
import uuid

from . import __version__
from .common import fingerprint, utc_now, write_json
from .inventory import inventory
from .telemetry import Sampler
from .workloads import WORKLOADS


def validate_config(config):
    required = {"workload", "name", "model", "runtime", "prompt", "output_tokens",
                "seed", "warmup_runs", "repetitions", "timeout_s", "sample_interval_s"}
    if set(config) != required:
        raise ValueError(f"Config keys must be exactly: {', '.join(sorted(required))}")
    if config["workload"] not in WORKLOADS:
        raise ValueError("Unsupported workload")
    for key in ("name", "prompt"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    for key, minimum in (("output_tokens", 1), ("warmup_runs", 0), ("repetitions", 1), ("seed", 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for key in ("timeout_s", "sample_interval_s"):
        if type(config[key]) not in (float, int) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    model = config["model"]
    if not isinstance(model, dict) or set(model) != {"name", "sha256", "quantization"}:
        raise ValueError("model requires name, sha256, quantization")
    import re
    if not isinstance(model["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", model["sha256"]):
        raise ValueError("model.sha256 must be the actual lowercase GGUF SHA-256")
    if any(not isinstance(model[k], str) or not model[k].strip() for k in ("name", "quantization")):
        raise ValueError("Model name and quantization must be nonempty strings")
    runtime = config["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {"image", "server_args"}:
        raise ValueError("runtime requires image and server_args")
    if not isinstance(runtime["image"], str) or not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", runtime["image"]):
        raise ValueError("runtime.image must be pinned by sha256 digest")
    if (not isinstance(runtime["server_args"], list) or not runtime["server_args"] or
            any(not isinstance(x, str) for x in runtime["server_args"])):
        raise ValueError("runtime.server_args must be a nonempty array of strings")


def summarize(trials):
    good = [t for t in trials if t["status"] == "ok"]
    result = {"successful_trials": len(good), "failed_or_invalid_trials": len(trials) - len(good),
              "metrics": {}}
    for key in ("ttft_s", "latency_s", "prompt_tok_s", "generation_tok_s"):
        values = sorted(t["metrics"][key] for t in good if t["metrics"].get(key) is not None)
        result["metrics"][key] = {
            "count": len(values), "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "p95": values[math.ceil(.95 * len(values)) - 1] if values else None,
        }
    return result


def run(config, endpoint, machine_id, output, host_inventory=None, gpu_uuids=None, token=None):
    validate_config(config)
    if not isinstance(machine_id, str) or not machine_id.strip():
        raise ValueError("machine_id must be nonempty")
    gpu_uuids = gpu_uuids or []
    config = copy.deepcopy(config)
    host_inventory = host_inventory or inventory(machine_id)
    if host_inventory.get("machine_id") != machine_id or host_inventory.get("record_type") != "inventory":
        raise ValueError("Host inventory must be an inventory record for this machine_id")
    record = {
        "schema_version": "1.0", "record_type": "benchmark", "run_id": str(uuid.uuid4()),
        "started_at": utc_now(), "finished_at": None, "status": "running",
        "harness_version": __version__, "machine_id": machine_id,
        "workload": config["workload"], "config": config,
        "comparison_key": fingerprint(config), "prompt_sha256": fingerprint(config["prompt"]),
        "provenance": {"model_and_runtime": "operator_declared", "endpoint": endpoint,
                       "gpu_uuids": gpu_uuids, "concurrency": 1},
        "inventory": host_inventory, "warmups": [], "trials": [], "summary": {},
        "warnings": [] if gpu_uuids else ["GPU telemetry disabled; no local GPU UUIDs selected"],
    }
    write_json(output, record)
    adapter = WORKLOADS[config["workload"]]

    def trial(index, measured):
        started = time.perf_counter()
        sampler = Sampler(gpu_uuids, config["sample_interval_s"]) if measured and gpu_uuids else None
        try:
            if sampler:
                with sampler:
                    result = adapter(config, endpoint, token)
            else:
                result = adapter(config, endpoint, token)
        except Exception as exc:
            # Exception text can include URLs or response content; keep credentials out of results.
            result = {"status": "error", "error_type": type(exc).__name__, "metrics": {}}
        result.update({"index": index, "wall_s": time.perf_counter() - started,
                       "telemetry": sampler.result() if sampler else None})
        return result

    try:
        for i in range(config["warmup_runs"]):
            record["warmups"].append(trial(i, False))
            write_json(output, record)
        if any(t["status"] != "ok" for t in record["warmups"]):
            record["warnings"].append("Warmup failed or was invalid; measured trials skipped")
        else:
            for i in range(config["repetitions"]):
                record["trials"].append(trial(i, True))
                write_json(output, record)
        record["summary"] = summarize(record["trials"])
        record["status"] = "ok" if (len(record["trials"]) == config["repetitions"] and
                                    all(t["status"] == "ok" for t in record["trials"])) else "failed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["summary"] = summarize(record["trials"])
    finally:
        record["finished_at"] = utc_now()
        write_json(output, record)
    return record
