from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    model: str = "gpt-4o"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self):
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY")


@dataclass
class Config:
    actor: LLMConfig = field(default_factory=LLMConfig)
    critic: LLMConfig = field(default_factory=lambda: LLMConfig(temperature=0.2))

    memory_dir: Path = Path("memory")
    max_drills_per_addressable: int = 5
    max_episodes_per_bucket: int = 10
    exec_timeout_seconds: int = 10

    benchmark: str = "humaneval"
    benchmark_subset: str | None = None
    task_limit: int | None = None

    seed: int = 42
