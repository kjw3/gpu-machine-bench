import copy
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from gpu_bench.cli import main
from gpu_bench.common import file_sha256, write_json
from gpu_bench.inventory import gpu_snapshot, inventory, number
from gpu_bench.runner import run, summarize, validate_config
from gpu_bench.telemetry import Sampler
from gpu_bench.workloads.llama_cpp import events, run_trial

try:
    import jsonschema
except ImportError:
    jsonschema = None


def config():
    value = json.loads(Path("configs/llm-baseline.json").read_text())
    value["model"]["sha256"] = "a" * 64
    value["runtime"]["image"] = "test/runtime@sha256:" + "b" * 64
    value["output_tokens"] = 4
    value["warmup_runs"] = 1
    value["repetitions"] = 2
    value["timeout_s"] = 2
    return value


class FakeServer(BaseHTTPRequestHandler):
    mode = "ok"
    requests = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append((self.path, payload))
        if self.mode == "http_error":
            self.send_error(503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if self.mode == "malformed":
            self.wfile.write(b"data: not json\n\n")
            return
        self.wfile.write(b': ping\n\ndata: {"content":""}\n\n')
        self.wfile.flush()
        time.sleep(.015)
        self.wfile.write(b'data: {"content":"hello"}\n\n')
        self.wfile.flush()
        if self.mode == "incomplete":
            return
        time.sleep(.01)
        value = {"stop": True, "content": "", "truncated": self.mode == "truncated",
                 "timings": {"prompt_n": 20, "prompt_ms": 10, "prompt_per_second": 2000,
                             "predicted_n": 4, "predicted_ms": 20, "predicted_per_second": 200,
                             "cache_n": 1 if self.mode == "cached" else 0}}
        if self.mode == "short":
            value["timings"]["predicted_n"] = 2
        if self.mode == "missing_timings":
            value.pop("timings")
        self.wfile.write(("data: " + json.dumps(value) + "\n\n").encode())


class StreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.endpoint = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        FakeServer.mode = "ok"
        FakeServer.requests.clear()

    def test_stream_metrics_and_request(self):
        result = run_trial(config(), self.endpoint)
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["metrics"]["ttft_s"], .01)
        self.assertLess(result["metrics"]["ttft_s"], result["metrics"]["latency_s"])
        self.assertEqual(result["metrics"]["generation_tok_s"], 200)
        self.assertEqual(result["backend"]["content_chunks"], 1)
        path, request = FakeServer.requests[0]
        self.assertEqual(path, "/completion")
        self.assertFalse(request["cache_prompt"])
        self.assertTrue(request["ignore_eos"])

    def test_invalid_trials(self):
        for mode in ("truncated", "short", "missing_timings", "cached"):
            with self.subTest(mode=mode):
                FakeServer.mode = mode
                self.assertEqual(run_trial(config(), self.endpoint)["status"], "invalid")

    def test_incomplete_and_malformed(self):
        for mode in ("incomplete", "malformed"):
            FakeServer.mode = mode
            with self.assertRaises(ValueError):
                run_trial(config(), self.endpoint)

    def test_runner_warmups_and_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = run(config(), self.endpoint, "test", path, host_inventory={
                "record_type": "inventory", "machine_id": "test"})
            self.assertEqual(result["status"], "ok")
            self.assertEqual(len(FakeServer.requests), 3)
            self.assertEqual(result["summary"]["successful_trials"], 2)
            self.assertEqual(result, json.loads(path.read_text()))

    @unittest.skipUnless(jsonschema, "Install .[test] for JSON Schema validation")
    def test_result_schema(self):
        with patch("gpu_bench.inventory.command", return_value=(None, "unavailable")):
            host = inventory("test")
        schema = json.loads(Path("schemas/result.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(host, schema)
        with tempfile.TemporaryDirectory() as directory:
            result = run(config(), self.endpoint, "test", Path(directory) / "r.json", host)
        jsonschema.validate(result, schema)
        invalid = copy.deepcopy(result)
        invalid["trials"][0]["metrics"]["ttft_s"] = "not-a-number"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)

    def test_warmup_error_skips_trials(self):
        FakeServer.mode = "http_error"
        with tempfile.TemporaryDirectory() as directory:
            result = run(config(), self.endpoint, "test", Path(directory) / "r.json",
                         host_inventory={"record_type": "inventory", "machine_id": "test"})
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["trials"], [])
            self.assertEqual(result["warmups"][0]["error_type"], "HTTPError")

    def test_measured_failures_are_persisted(self):
        FakeServer.mode = "http_error"
        c = config()
        c["warmup_runs"] = 0
        with tempfile.TemporaryDirectory() as directory:
            result = run(c, self.endpoint, "test", Path(directory) / "r.json",
                         host_inventory={"record_type": "inventory", "machine_id": "test"})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["summary"]["failed_or_invalid_trials"], 2)
        self.assertEqual(result["summary"]["successful_trials"], 0)


class UnitTests(unittest.TestCase):
    def test_multiline_sse_crlf_and_comments(self):
        stream = io.BytesIO(b': comment\r\nevent: message\r\ndata: {"a":\r\ndata: 1}\r\n\r\ndata: [DONE]\n\n')
        self.assertEqual(list(events(stream)), ['{"a":\n1}', '[DONE]'])

    def test_invalid_configuration(self):
        for key, value in (("repetitions", 0), ("warmup_runs", -1), ("seed", True),
                           ("timeout_s", float("nan")), ("sample_interval_s", 0),
                           ("prompt", ""), ("workload", "unknown")):
            bad = config()
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(bad)
        bad = config()
        bad["runtime"]["image"] = "runtime:latest"
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_template_requires_artifact_identity(self):
        with self.assertRaises(ValueError):
            validate_config(json.loads(Path("configs/llm-baseline.json").read_text()))

    def test_gpu_csv_and_unsupported_metrics(self):
        text = 'GPU-1, "NVIDIA, Example", 0000:01:00.0, 570.0, 8192, 123, [N/A], 200, 40, 1\n'
        with patch("gpu_bench.inventory.command", return_value=(text, None)):
            gpus, warnings = gpu_snapshot(["GPU-1"])
        self.assertEqual(gpus[0]["name"], "NVIDIA, Example")
        self.assertIsNone(gpus[0]["power_draw_w"])
        self.assertEqual(gpus[0]["memory_total_mib"], 8192)
        self.assertFalse(warnings)

    def test_absent_gpu_is_not_zero(self):
        with patch("gpu_bench.inventory.command", return_value=(None, "not installed")):
            gpus, warnings = gpu_snapshot()
            value = inventory("cpu-only")
        self.assertEqual(gpus, [])
        self.assertTrue(warnings)
        self.assertIsNone(value["software"]["driver_cuda_max_version"])
        for value in ("N/A", "nan", "inf", -1):
            self.assertIsNone(number(value))

    def test_missing_uuid_warns(self):
        with patch("gpu_bench.inventory.command", return_value=("", None)):
            self.assertIn("GPU unavailable: GPU-x", gpu_snapshot(["GPU-x"])[1])

    def test_device_selection(self):
        data = 'GPU-1, One, bus1, 570, 8192, 10, 20, 100, 40, 0\nGPU-2, Two, bus2, 570, 8192, 30, 40, 100, 50, 0'
        with patch("gpu_bench.inventory.command", return_value=(data, None)):
            rows, warnings = gpu_snapshot(["GPU-2"])
        self.assertEqual([r["uuid"] for r in rows], ["GPU-2"])
        self.assertEqual(warnings, [])

    def test_sampler_stops_after_failure(self):
        with patch("gpu_bench.telemetry.gpu_snapshot", return_value=([], ["unavailable"])):
            sampler = Sampler(["GPU-x"], .01)
            with self.assertRaises(RuntimeError):
                with sampler:
                    time.sleep(.02)
                    raise RuntimeError("request failed")
        self.assertFalse(sampler.thread.is_alive())
        self.assertIn("unavailable", sampler.result()["warnings"])

    def test_mismatched_inventory_rejected(self):
        with self.assertRaises(ValueError):
            run(config(), "http://localhost", "target", "unused.json",
                host_inventory={"record_type": "inventory", "machine_id": "other"})

    def test_sampler_summary(self):
        sampler = Sampler(["GPU-x"], .1)
        sampler.samples = [{"elapsed_s": 0, "gpus": [
            {"uuid": "GPU-x", "memory_used_mib": 30, "power_draw_w": None}]},
            {"elapsed_s": 1, "gpus": [
            {"uuid": "GPU-x", "memory_used_mib": 50, "power_draw_w": 10}]}]
        summary = sampler.result()["summary"]["GPU-x"]
        self.assertEqual(summary["peak_memory_used_mib"], 50)
        self.assertEqual(summary["mean_sampled_power_w"], 10)

    def test_failed_trials_excluded(self):
        summary = summarize([{"status": "error", "metrics": {}},
                             {"status": "ok", "metrics": {"ttft_s": 1, "latency_s": 2,
                              "prompt_tok_s": 3, "generation_tok_s": 4}}])
        self.assertEqual(summary["successful_trials"], 1)
        self.assertEqual(summary["metrics"]["generation_tok_s"]["median"], 4)
        self.assertIsNone(summarize([])["metrics"]["ttft_s"]["median"])

    def test_atomic_write_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            write_json(path, {"a": 1})
            old_hash = file_sha256(path)
            with self.assertRaises(ValueError):
                write_json(path, {"a": float("nan")})
            self.assertEqual(file_sha256(path), old_hash)
            self.assertEqual(len(list(path.parent.iterdir())), 1)

    def test_interrupt_preserves_result(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("gpu_bench.runner.WORKLOADS", {"llm.llama_cpp": lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())}):
                path = Path(directory) / "r.json"
                result = run(config(), "http://localhost", "test", path,
                             host_inventory={"machine_id": "test", "record_type": "inventory"})
            self.assertEqual(result["status"], "interrupted")
            self.assertEqual(json.loads(path.read_text())["status"], "interrupted")

    def test_endpoint_secrets_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            code = main(["run", "--config", "unused", "--machine-id", "test", "--output",
                         directory + "/r.json", "--endpoint", "http://user:secret@localhost"])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
