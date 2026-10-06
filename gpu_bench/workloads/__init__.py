"""Adapters return per-trial metrics, validity, and backend evidence."""
from .llama_cpp import run_trial

WORKLOADS = {"llm.llama_cpp": run_trial}
