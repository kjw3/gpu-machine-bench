# Public compute profiles

Generate a profile directly on a host with:

```sh
python3 -m gpu_bench profile --id machine-001 --output machines/machine-001.json
```

Use neutral labels; retain the mapping to actual machines privately. Profiles are constructed from an explicit field allowlist rather than by deleting identifiers from raw inventories. No hostnames, IPs, UUIDs, serial numbers, PCI bus addresses, timestamps, usernames, paths, endpoints, processes, service names, or raw command output are exported. Hardware models and software versions are retained intentionally; their combination can still be recognizable.

`schemas/profile.schema.json` disallows extra fields at every object level. Values describe observed hardware capabilities, not guaranteed AI workload support. Clock rates and PCIe fields are reported maxima, not active link rates or sustained performance. Power limits are configured limits, not measured workload consumption. Unified memory is shared with the CPU and OS; unsupported GPU memory counters remain null and system memory is not added to GPU memory.

These files are hardware profiles, not benchmark results. Do not add raw inventory or result files here. Regenerate when hardware or software changes, review the diff, then publish deliberately. The profiler requires Python 3.11+; NVIDIA and Linux tools are optional, with missing fields recorded as null. Run on the host for host CPU/RAM metadata; container collection is labeled explicitly.

## Collected profiles

| Profile | GPU models | Logical CPUs | OS-visible RAM (GiB) |
|---|---|---:|---:|
| [machine-001](machine-001.json) | NVIDIA GeForce RTX 5090 | 24 | 29.9 |
| [machine-002](machine-002.json) | NVIDIA GeForce RTX 3080, NVIDIA GeForce RTX 3070 Ti, NVIDIA GeForce RTX 3070 | 8 | 23.5 |
| [machine-003](machine-003.json) | NVIDIA GeForce RTX 3060 Ti | 2 | 7.7 |
| [machine-004](machine-004.json) | NVIDIA GB10 | 20 | 121.7 |

CPU groups preserve heterogeneous core types as reported by `lscpu`; per-group socket fields must not be summed as physical socket counts. RAM is OS-visible capacity rather than advertised installed memory.
