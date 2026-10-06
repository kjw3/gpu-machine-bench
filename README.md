# Decent GPU Machine Bench

A dependency-free Python 3.11+ harness for repeatable, single-request LLM measurements and machine inventory. Results are versioned JSON, with raw trials and sampled GPU telemetry retained for future reports. This initial release supports **llama.cpp's native `/completion` streaming endpoint**. It does not yet implement embeddings, reranking, images, or speech.

Designed for NVIDIA GPU hosts, including discrete GeForce GPUs and ARM64 systems with unified memory. Treat each GPU as a separate target, not an aggregate VRAM pool. The Python client works on x86_64 and ARM64; inference image compatibility must be verified separately on each architecture.

## Quick start: inventory

From the repository on the machine being measured:

```sh
python3 -m gpu_bench inventory --machine-id example-node --output results/example-node-inventory.json
python3 -m unittest discover -s tests -v
```

Inventory works without NVIDIA tools and records warnings when they are absent. Run it on the **host** to capture host OS, Docker and NVIDIA Container Toolkit versions. A container's OS and CPU/memory view are explicitly labeled as container scope. Linux RAM is recorded from `/proc/meminfo`; unsupported OS memory fields remain null. `driver_cuda_max_version` is the driver's advertised CUDA compatibility, **not** the installed CUDA toolkit or runtime version. `cuda_toolkit_nvcc` is separate and may be null.

## Prepare a GPU host

Install a suitable NVIDIA driver and Docker, then follow [NVIDIA's Container Toolkit installation instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Configure the Docker runtime and restart Docker as documented there. Verify `nvidia-smi` on the host and inside a GPU-enabled container before benchmarking. This repository does not change drivers or host services.

Use Linux for the supplied GPU Compose setup. Docker Desktop on macOS can test the runner, but cannot provide NVIDIA GPU execution. On Spark, use an ARM64 inference build explicitly supporting GB10; do not assume an x86 CUDA image works. Unsupported power or device-memory readings remain null, and Spark's unified memory must be interpreted separately from discrete VRAM.

## Start a controlled inference server

Choose one **identical GGUF artifact that fits the 8 GB cards**, such as a suitably small quantized model, for the baseline. Pin its exact bytes and use the same prompt, context, output count and runtime settings everywhere. This project intentionally does not download a model or select a floating runtime release for you.

1. Place the GGUF at `/absolute/models/baseline.gguf`.
2. Choose and test a llama.cpp server image with CUDA support for the target GPU. Resolve its digest with `docker image inspect` after pulling it. Record the platform-specific runtime/build if architecture requires different images.
3. Set the image digest, model folder and GPU UUID:

```sh
export LLAMA_IMAGE='ghcr.io/ggml-org/llama.cpp@sha256:REPLACE_WITH_REAL_DIGEST'
export MODEL_DIR='/absolute/models'
export GPU_UUID='GPU-REPLACE_WITH_UUID_FROM_NVIDIA_SMI'
docker compose up -d inference
docker compose logs inference
curl --fail http://127.0.0.1:8080/health
```

The server binds only to localhost on the host. Wait for a successful health response. Confirm logs show the expected model, all intended layers on the selected GPU, context 4096 and a single slot. Some builds may need an explicit `--perf` to enable timings; if required, add it to both Compose and the recorded `server_args`. Consult the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) for the pinned build.

Copy `configs/llm-baseline.json` to `configs/local-baseline.json`. Fill in the exact model filename, quantization, model hash, runtime image digest and complete server arguments. The shipped template intentionally fails validation until real identities are provided:

```sh
python3 -m gpu_bench hash-model /absolute/models/baseline.gguf
```

Model/runtime provenance is **operator-declared**: the harness does not prove that a remote endpoint loaded those bytes or applied those arguments. Preserve server logs and image inspection output alongside results when publishing comparisons. Run one GPU at a time on multi-GPU hosts; record CPU offloading as a different configuration. Keep other workloads idle and power limits stable.

## Execute the benchmark

On the inference host:

```sh
python3 -m gpu_bench run \
  --config configs/local-baseline.json \
  --endpoint http://127.0.0.1:8080 \
  --machine-id example-node \
  --host-inventory results/example-node-inventory.json \
  --gpu "$GPU_UUID" --telemetry-local \
  --output results/example-node-baseline-run1.json
```

Or containerize the runner on that Linux host:

```sh
docker build -t gpu-machine-bench:local .
docker run --rm --network host --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$GPU_UUID" \
  -e NVIDIA_DRIVER_CAPABILITIES=utility \
  --user "$(id -u):$(id -g)" \
  -v "$PWD/configs:/configs:ro" -v "$PWD/results:/results" \
  gpu-machine-bench:local run \
  --config /configs/local-baseline.json \
  --machine-id example-node --host-inventory /results/example-node-inventory.json \
  --gpu "$GPU_UUID" --telemetry-local \
  --output /results/example-node-baseline-run2.json
```

The runner needs only the NVIDIA **utility** capability for telemetry; computation occurs in the inference container. It needs no Docker socket or privileged mode. The runner base image can be pinned with `docker build --build-arg PYTHON_IMAGE=python@sha256:...`; the default tag is convenient for development, not a frozen build.

For a remote server, omit `--gpu` and `--telemetry-local`. Supply inventory collected on the inference host. Otherwise the embedded inventory describes the **client**, not the server. Remote runs include network latency and have no server GPU telemetry. Never compare their TTFT directly with local runs. If authentication is enabled, set `GPU_BENCH_API_KEY`; secrets are not included in results. Keep endpoints private using the lab network or Tailscale.

Each run creates one file; the CLI refuses to overwrite an existing result. Exit status is 0 for success, 1 for a failed/interrupted run, and 2 for setup/configuration errors. Completed trials are checkpointed atomically. A forcibly killed process can leave `status: running`; exclude it from reports. Ctrl-C saves an interrupted result.

Stop the test service when finished:

```sh
docker compose down
```

## Measurement contract

- Concurrency is one. One warmup precedes five measured repetitions by default. Warmups are saved but excluded from summaries; a failed warmup prevents measurement.
- Greedy sampling, fixed seed, disabled prompt cache and `ignore_eos` request a fixed output length. Truncation, cache reuse, missing timing metrics, and a short output invalidate a trial. A fixed text prompt has a fixed token count only when the tokenizer/model is unchanged.
- `prompt_tok_s` and `generation_tok_s` come from **server timings**. SSE chunks are never counted as tokens. TTFT is client time from request start to first **nonempty text chunk**, a practical proxy for first token; empty/special tokens and buffering can delay it. Latency ends at the final stop event.
- GPU samples cover each measured request plus sampling overhead. Memory is total device used memory, not a model allocation measurement; power is board power, not whole-system power. Peaks are sampled peaks and can miss spikes. Mean power is an arithmetic mean of available samples, not an energy measurement. Sampling uses `nvidia-smi` and can perturb short runs. Raw timestamps and per-device readings are saved.
- Missing tools, unsupported counters, or unavailable UUIDs produce warnings and null/empty telemetry. Timing success does not certify telemetry completeness. Inspect `telemetry.warnings` and `sample_count` before reporting hardware metrics.
- Summary statistics include only valid trials; p95 uses nearest rank. Five repeats are a starting point, not a statistically stable tail-latency estimate. A partially failed run retains successful-trial statistics but has failed overall status.
- `comparison_key` hashes the complete configuration, including model bytes and runtime settings. Equal keys indicate the same declared experiment, not proof of identical host conditions. Different platform image digests intentionally produce different keys; group cross-architecture comparisons explicitly. Maximum-useful-model tests should use a separate configuration and not join the baseline leaderboard.

## Results and extension points

`schemas/result.schema.json` describes the inventory and benchmark envelopes. Metric names encode their units. JSON contains the full configuration, inventory, provenance, warmups, trials, backend timings, telemetry and summary. Hostnames/GPU UUIDs are included; inspect artifacts before publishing them.

Add workloads under `gpu_bench/workloads/` and register their adapter in `WORKLOADS`. An adapter accepts configuration, endpoint and optional token, and returns status, metrics and backend evidence. The runner owns repeats, persistence, inventory and telemetry. Before adding embedding, reranking, image or Whisper adapters, extend the config validator and summary selection with workload-specific parameters and units (documents/s, images/s, audio real-time factor, quality fixtures). Do not reuse LLM token metrics for other workloads. Add deterministic fixtures, failure tests and a schema version change for incompatible result changes.

## Validation status

Tests use a real localhost HTTP/SSE fixture plus mocked NVIDIA output. They cover timing interpretation, configuration, incomplete streams, server errors, invalid trials, warmup exclusion, persistence, telemetry and interruption. They require permission to bind localhost but no GPU or model downloads. CI also builds and smoke-tests the CPU runner image. Actual GPU performance, Blackwell compatibility and Spark counters must be validated on the lab hardware before publishing benchmark numbers.

Install `python3 -m pip install '.[test]'` in a virtual environment to include JSON Schema validation tests. Without that optional dependency the schema test is skipped; the runner itself remains dependency-free. `timeout_s` bounds socket inactivity and is checked between complete stream events; it is not a hard process-kill deadline for a server trickling partial SSE events.


## Public repository hygiene

Commit source code, schemas, synthetic test fixtures, and generic workload configurations only. Keep machine inventories, benchmark results, comparison reports, endpoint addresses, GPU UUIDs, service configuration, and credentials outside the repository or in ignored `private/`, `results/`, or `configs/local/` directories. Local configuration files named `configs/local*.json` are also ignored.

The pinned `configs/baseline-qwen3-4b.json` contains public model and container identifiers only. Place the matching GGUF artifact at `/models/baseline.gguf` in the inference container and verify its SHA-256 before running.

Ignore rules prevent accidental addition of new files; they do not remove files already tracked by Git and can be bypassed with force-add. Review the staged diff and file list before every public push. The Docker build context includes only the runner source and Dockerfile.
