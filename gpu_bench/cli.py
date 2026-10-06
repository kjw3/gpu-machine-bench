import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

from .common import file_sha256, write_json
from .inventory import inventory
from .runner import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="Decent GPU machine benchmarks")
    commands = parser.add_subparsers(dest="command", required=True)
    inv = commands.add_parser("inventory", help="Collect local host/GPU metadata")
    inv.add_argument("--machine-id", required=True)
    inv.add_argument("--output", type=Path, required=True)
    digest = commands.add_parser("hash-model", help="Hash the exact model artifact")
    digest.add_argument("path", type=Path)
    bench = commands.add_parser("run", help="Run a reproducible workload against llama-server")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument("--endpoint", default="http://127.0.0.1:8080")
    bench.add_argument("--machine-id", required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--host-inventory", type=Path)
    bench.add_argument("--gpu", action="append", default=[], help="Local GPU UUID to sample (repeatable)")
    bench.add_argument("--telemetry-local", action="store_true",
                       help="Confirm the runner and inference server share the selected GPUs")
    bench.add_argument("--api-key-env", default="GPU_BENCH_API_KEY")
    args = parser.parse_args(argv)
    try:
        if args.command == "hash-model":
            print(file_sha256(args.path))
        elif args.command == "inventory":
            write_json(args.output, inventory(args.machine_id))
            print(args.output)
        else:
            parsed = urlsplit(args.endpoint)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                    or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/")):
                raise ValueError("Endpoint must be an HTTP(S) origin without credentials, path, or query")
            if args.gpu and not args.telemetry_local:
                raise ValueError("--gpu requires --telemetry-local; remote GPU metrics cannot be sampled")
            if args.output.exists():
                raise ValueError("Output already exists; choose a new run filename")
            config = json.loads(args.config.read_text())
            host = json.loads(args.host_inventory.read_text()) if args.host_inventory else None
            result = run(config, args.endpoint, args.machine_id, args.output, host, args.gpu,
                         os.environ.get(args.api_key_env))
            print(f"{result['status']}: {args.output}")
            return 0 if result["status"] == "ok" else 1
    except (ValueError, OSError, TypeError, KeyError) as exc:
        print(f"gpu-bench: {exc}", file=sys.stderr)
        return 2
    return 0
