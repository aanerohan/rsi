#!/usr/bin/env python3
"""CLI entry point for the RSI (Runtime Self-Improvement) system."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rsi.config import Config, LLMConfig
from rsi.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test-time self-improvement loop via critique-maintained memory"
    )
    parser.add_argument(
        "--benchmark",
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="Benchmark to evaluate on (default: humaneval)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of tasks to run (default: all)",
    )
    parser.add_argument(
        "--actor-model", default="gpt-4o",
        help="Model name for the Actor LLM",
    )
    parser.add_argument(
        "--critic-model", default="gpt-4o",
        help="Model name for the Critic LLM",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--memory-dir", default="memory",
        help="Directory for memory storage (default: memory/)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for experiment results",
    )
    parser.add_argument(
        "--timeout", type=int, default=10,
        help="Code execution timeout in seconds (default: 10)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config(
        actor=LLMConfig(
            model=args.actor_model,
            base_url=args.base_url,
            api_key=args.api_key,
        ),
        critic=LLMConfig(
            model=args.critic_model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=0.2,
        ),
        memory_dir=Path(args.memory_dir),
        benchmark=args.benchmark,
        task_limit=args.limit,
        exec_timeout_seconds=args.timeout,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        results = run_experiment(config, output_dir=output_dir)
        delta = results.get("delta_pass_rate", 0)
        if delta > 0:
            print(f"\nMemory improved pass rate by {delta:+.1%}")
        elif delta < 0:
            print(f"\nMemory decreased pass rate by {delta:+.1%}")
        else:
            print("\nNo change in pass rate")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
