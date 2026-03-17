"""Benchmark loaders for HumanEval and MBPP."""

from __future__ import annotations

import logging
import re
from typing import Iterator

from .models import BenchmarkTask

log = logging.getLogger(__name__)


def load_benchmark(name: str, limit: int | None = None) -> list[BenchmarkTask]:
    loaders = {
        "humaneval": _load_humaneval,
        "mbpp": _load_mbpp,
    }
    loader = loaders.get(name.lower())
    if loader is None:
        raise ValueError(f"Unknown benchmark: {name!r}. Choose from {list(loaders)}")
    tasks = list(loader())
    if limit is not None:
        tasks = tasks[:limit]
    log.info("Loaded %d tasks from %s", len(tasks), name)
    return tasks


def _load_humaneval() -> Iterator[BenchmarkTask]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    for row in ds:
        test_code = row.get("test", "")
        entry_point = row.get("entry_point", "")
        if entry_point and f"check({entry_point})" not in test_code:
            test_code += f"\n\ncheck({entry_point})"

        yield BenchmarkTask(
            task_id=row["task_id"],
            prompt=row["prompt"],
            entry_point=entry_point,
            test_code=test_code,
            canonical_solution=row.get("canonical_solution", ""),
        )


def _load_mbpp() -> Iterator[BenchmarkTask]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True)
    for row in ds:
        tests = row.get("test_list", [])
        test_code = "\n".join(tests)

        prompt_text = row.get("prompt", row.get("text", ""))
        code = row.get("code", "")
        entry_point = _extract_entry_point(code)

        yield BenchmarkTask(
            task_id=f"mbpp/{row.get('task_id', '')}",
            prompt=prompt_text,
            entry_point=entry_point,
            test_code=test_code,
            canonical_solution=code,
        )


def _extract_entry_point(code: str) -> str:
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else ""
