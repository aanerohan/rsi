"""Experiment harness: baseline (no memory) vs memory-enabled, single attempt per task."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .benchmarks import load_benchmark
from .config import Config
from .memory import MemoryStore
from .runner import TaskResult, run_task

log = logging.getLogger(__name__)
console = Console()


@dataclass
class RunMetrics:
    label: str
    total_tasks: int = 0
    solved: int = 0
    pass_rate: float = 0.0
    total_buckets_created: int = 0
    elapsed_s: float = 0.0
    per_task: list[dict] = field(default_factory=list)


def run_experiment(config: Config, output_dir: Path | None = None) -> dict:
    """Run baseline + memory-enabled and compare."""
    tasks = load_benchmark(config.benchmark, limit=config.task_limit)
    if not tasks:
        raise ValueError("No tasks loaded")

    output_dir = output_dir or Path("results") / f"run_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold]Baseline run (no memory)")
    baseline = _run_suite(tasks, config, use_memory=False, label="baseline")

    console.rule("[bold]Memory-enabled run")
    memory_run = _run_suite(tasks, config, use_memory=True, label="memory")

    summary = {
        "config": {
            "benchmark": config.benchmark,
            "task_limit": config.task_limit,
            "actor_model": config.actor.model,
            "critic_model": config.critic.model,
        },
        "baseline": asdict(baseline),
        "memory": asdict(memory_run),
        "delta_pass_rate": memory_run.pass_rate - baseline.pass_rate,
    }

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Results written to %s", results_path)

    _print_comparison(baseline, memory_run)
    return summary


def _run_suite(
    tasks: list,
    config: Config,
    *,
    use_memory: bool,
    label: str,
) -> RunMetrics:
    memory = MemoryStore(config.memory_dir)
    if use_memory:
        memory.reset()

    metrics = RunMetrics(label=label, total_tasks=len(tasks))
    t0 = time.perf_counter()

    for i, task in enumerate(tasks):
        n_buckets = len(memory.list_bucket_ids()) if use_memory else 0
        console.print(
            f"  [{i+1}/{len(tasks)}] {task.task_id}  (buckets: {n_buckets})",
            style="dim",
        )
        result: TaskResult = run_task(task, config, memory, use_memory=use_memory)
        metrics.per_task.append(
            {
                "task_id": result.task_id,
                "solved": result.solved,
                "status": result.status,
                "buckets_available": result.buckets_available,
                "elapsed_s": round(result.elapsed_s, 2),
            }
        )
        if result.solved:
            metrics.solved += 1

    metrics.elapsed_s = time.perf_counter() - t0
    metrics.pass_rate = metrics.solved / max(len(tasks), 1)
    if use_memory:
        metrics.total_buckets_created = len(memory.list_bucket_ids())

    console.print(
        f"  {label}: solved {metrics.solved}/{len(tasks)} "
        f"(pass_rate={metrics.pass_rate:.1%})",
        style="bold",
    )
    return metrics


def _print_comparison(baseline: RunMetrics, memory: RunMetrics) -> None:
    table = Table(title="Experiment Results")
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Delta", justify="right")

    delta_rate = memory.pass_rate - baseline.pass_rate

    rows = [
        ("Pass rate", f"{baseline.pass_rate:.1%}", f"{memory.pass_rate:.1%}",
         f"{delta_rate:+.1%}"),
        ("Solved", str(baseline.solved), str(memory.solved),
         f"{memory.solved - baseline.solved:+d}"),
        ("Buckets created", "—", str(memory.total_buckets_created), ""),
        ("Time (s)", f"{baseline.elapsed_s:.0f}", f"{memory.elapsed_s:.0f}", ""),
    ]

    # Show pass rate in early vs late halves to see if memory helps over time
    half = max(len(baseline.per_task) // 2, 1)
    if half > 1:
        mem_early = sum(1 for t in memory.per_task[:half] if t["solved"]) / half
        mem_late = sum(1 for t in memory.per_task[half:] if t["solved"]) / max(len(memory.per_task) - half, 1)
        base_early = sum(1 for t in baseline.per_task[:half] if t["solved"]) / half
        base_late = sum(1 for t in baseline.per_task[half:] if t["solved"]) / max(len(baseline.per_task) - half, 1)
        rows.append(("Early-half pass rate", f"{base_early:.1%}", f"{mem_early:.1%}", f"{mem_early - base_early:+.1%}"))
        rows.append(("Late-half pass rate", f"{base_late:.1%}", f"{mem_late:.1%}", f"{mem_late - base_late:+.1%}"))

    for row in rows:
        table.add_row(*row)

    console.print()
    console.print(table)
