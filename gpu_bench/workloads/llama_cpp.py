"""Native llama.cpp SSE adapter; server timings are not derived from chunks."""
import json
import time
import urllib.request
import urllib.error

from ..inventory import number


def events(response):
    lines = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if lines:
                yield "\n".join(lines)
                lines = []
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip(" "))
    if lines:
        yield "\n".join(lines)


def run_trial(config, endpoint, token=None):
    payload = {"prompt": config["prompt"], "n_predict": config["output_tokens"],
               "temperature": 0, "seed": config["seed"], "stream": True,
               "cache_prompt": False, "ignore_eos": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(endpoint.rstrip("/") + "/completion",
                                     data=json.dumps(payload).encode(), headers=headers)
    started = time.perf_counter()
    first = None
    final = None
    chunks = 0
    try:
        response = urllib.request.urlopen(request, timeout=config["timeout_s"])
    except urllib.error.HTTPError as exc:
        exc.close()
        raise
    with response:
        for event in events(response):
            if time.perf_counter() - started > config["timeout_s"]:
                raise TimeoutError("Trial exceeded configured deadline")
            if event == "[DONE]":
                break
            value = json.loads(event)
            if "error" in value:
                raise ValueError("Backend returned an error event")
            if value.get("content"):
                chunks += 1
                if first is None:
                    first = time.perf_counter()
            if value.get("stop") is True:
                final = value
                break
    elapsed = time.perf_counter() - started
    if final is None:
        raise ValueError("Incomplete stream: no final stop event")
    timings = final.get("timings") or {}
    metrics = {
        "ttft_s": first - started if first is not None else None,
        "latency_s": elapsed,
        "prompt_tokens": number(timings.get("prompt_n")),
        "output_tokens": number(timings.get("predicted_n")),
        "prompt_tok_s": number(timings.get("prompt_per_second")),
        "generation_tok_s": number(timings.get("predicted_per_second")),
        "server_prompt_ms": number(timings.get("prompt_ms")),
        "server_generation_ms": number(timings.get("predicted_ms")),
    }
    reasons = []
    if first is None:
        reasons.append("No nonempty output observed")
    if any(metrics[k] is None or metrics[k] <= 0 for k in
           ("prompt_tokens", "prompt_tok_s", "generation_tok_s")):
        reasons.append("Missing or invalid server token/timing metrics")
    if metrics["output_tokens"] != config["output_tokens"]:
        reasons.append("Output token count differs from requested count")
    if final.get("truncated"):
        reasons.append("Context was truncated")
    if number(timings.get("cache_n")) not in (None, 0):
        reasons.append("Server reused prompt cache")
    return {"status": "ok" if not reasons else "invalid", "metrics": metrics,
            "invalid_reasons": reasons,
            "backend": {"timings": timings, "truncated": final.get("truncated"),
                        "stop_type": final.get("stop_type"), "content_chunks": chunks}}
