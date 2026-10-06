"""Local GPU sampling. Does not infer remote server telemetry."""
import threading
import time

from .inventory import gpu_snapshot


class Sampler:
    def __init__(self, uuids, interval):
        self.uuids = uuids
        self.interval = interval
        self.samples = []
        self.warnings = set()
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def _run(self):
        while not self.stop.is_set():
            try:
                gpus, warnings = gpu_snapshot(self.uuids)
                self.warnings.update(warnings)
                self.samples.append({"elapsed_s": time.perf_counter() - self.started, "gpus": gpus})
            except Exception as exc:
                self.warnings.add(f"Sampler failed: {type(exc).__name__}")
                return
            self.stop.wait(self.interval)

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()

    def result(self):
        summary = {}
        for uuid in self.uuids:
            rows = [gpu for sample in self.samples for gpu in sample["gpus"] if gpu["uuid"] == uuid]
            def values(key):
                return [r[key] for r in rows if r.get(key) is not None]
            power = values("power_draw_w")
            memory = values("memory_used_mib")
            summary[uuid] = {
                "sample_count": len(rows),
                "peak_memory_used_mib": max(memory) if memory else None,
                "mean_sampled_power_w": sum(power) / len(power) if power else None,
                "peak_power_w": max(power) if power else None,
            }
        return {"scope": "local_selected_devices", "interval_s": self.interval,
                "samples": self.samples, "summary": summary, "warnings": sorted(self.warnings)}
