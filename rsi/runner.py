"""Main self-improvement loop: Actor → Evaluator → Critic → Memory.

Each task gets a single attempt. Memory accumulates across tasks so that
patterns learned from Task N can help on Task N+1, N+2, etc.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .actor import generate_solution
from .config import Config
from .critic import run_critic
from .evaluator import evaluate_code
from .memory import MemoryStore
from .models import BenchmarkTask, Episode, EvalResult, TestStatus

log = logging.getLogger(__name__)


@dataclass
class TaskResult:
    task_id: str
    solved: bool = False
    code: str = ""
    status: str = ""
    elapsed_s: float = 0.0
    buckets_available: int = 0


def run_task(
    task: BenchmarkTask,
    config: Config,
    memory: MemoryStore,
    *,
    use_memory: bool = True,
) -> TaskResult:
    """Run a single attempt on *task*, then update memory if enabled."""
    t0 = time.perf_counter()
    index_text = memory.load_index_text() if use_memory else "(memory disabled)"
    buckets_available = len(memory.list_bucket_ids()) if use_memory else 0

    log.info("[%s] solving (buckets available: %d)", task.task_id, buckets_available)

    code = generate_solution(
        config.actor,
        task.prompt,
        index_text,
        memory=memory if use_memory else None,
    )

    eval_result = evaluate_code(
        task.task_id,
        code,
        task.test_code,
        task.entry_point,
        timeout=config.exec_timeout_seconds,
    )

    solved = eval_result.status == TestStatus.PASS
    log.info("[%s] %s", task.task_id, "PASS" if solved else eval_result.status.value)

    if use_memory:
        critic_out = run_critic(
            config.critic, task.prompt, code, eval_result, index_text
        )
        if critic_out is not None:
            episode = Episode(
                task_id=task.task_id,
                code=code,
                diagnosis=critic_out.diagnosis,
                error_trace=_build_error_trace(eval_result),
                outcome="success" if solved else "failure",
            )
            memory.apply_critic_output(
                critic_out,
                episode,
                max_episodes=config.max_episodes_per_bucket,
                max_drills=config.max_drills_per_addressable,
            )

    return TaskResult(
        task_id=task.task_id,
        solved=solved,
        code=code,
        status=eval_result.status.value,
        elapsed_s=time.perf_counter() - t0,
        buckets_available=buckets_available,
    )


def _build_error_trace(result: EvalResult) -> str:
    """Assemble the full error trace from evaluator output."""
    if result.status == TestStatus.PASS:
        return ""
    parts: list[str] = []
    parts.append(f"Status: {result.status.value}")
    if result.stderr:
        parts.append(result.stderr)
    for ft in result.failed_tests:
        if ft.input:
            parts.append(f"Input: {ft.input}")
        if ft.expected:
            parts.append(f"Expected: {ft.expected}")
        if ft.actual:
            parts.append(f"Actual: {ft.actual}")
        if ft.traceback:
            parts.append(ft.traceback)
    return "\n".join(parts)
