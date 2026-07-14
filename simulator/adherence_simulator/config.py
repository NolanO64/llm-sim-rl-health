"""Configuration shared by the dataset validation runner and LLM backends."""

from dataclasses import dataclass


@dataclass
class SimulatorConfig:
    backend: str = "nebula"
    model: str = "SURF.Qwen3.5 122B A10B NVFP4"
    temperature: float = 0.6
    top_p: float | None = None
    ensemble_size: int = 1
    thinking_budget: int = 0
    parallel: bool = True
    no_reasoning: bool = False
    max_retries: int = 3
    retry_delay: float = 1.0
    outage_retry_seconds: float = 1800.0
    outage_retry_enabled: bool = True
