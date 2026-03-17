"""Deterministic code evaluator: executes generated code against test cases."""

from __future__ import annotations

import multiprocessing
import time
import traceback
from typing import Any

from .models import EvalResult, FailedTest, TestStatus


def _run_in_process(code: str, result_queue: multiprocessing.Queue) -> None:  # type: ignore[type-arg]
    """Target for the child process. Executes *code* and sends results back."""
    import io
    import sys

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        exec_globals: dict[str, Any] = {}
        exec(code, exec_globals)  # noqa: S102
        result_queue.put(
            {
                "ok": True,
                "stdout": sys.stdout.getvalue(),
                "stderr": sys.stderr.getvalue(),
            }
        )
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "error": traceback.format_exc(),
                "stdout": sys.stdout.getvalue(),
                "stderr": sys.stderr.getvalue(),
            }
        )
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def evaluate_code(
    task_id: str,
    generated_code: str,
    test_code: str,
    entry_point: str = "",
    timeout: int = 10,
) -> EvalResult:
    """Execute *generated_code* + *test_code* in a subprocess with timeout.

    Returns a structured ``EvalResult``.
    """
    full_code = generated_code + "\n\n" + test_code
    t0 = time.perf_counter()

    queue: multiprocessing.Queue[dict[str, Any]] = multiprocessing.Queue()
    proc = multiprocessing.Process(target=_run_in_process, args=(full_code, queue))
    proc.start()
    proc.join(timeout=timeout)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
        return EvalResult(
            task_id=task_id,
            status=TestStatus.TIMEOUT,
            runtime_ms=elapsed_ms,
            stderr=f"Execution timed out after {timeout}s",
        )

    if queue.empty():
        return EvalResult(
            task_id=task_id,
            status=TestStatus.ERROR,
            runtime_ms=elapsed_ms,
            stderr="Process exited without producing results",
        )

    result = queue.get_nowait()
    if result["ok"]:
        return EvalResult(
            task_id=task_id,
            status=TestStatus.PASS,
            runtime_ms=elapsed_ms,
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
        )

    error_text = result.get("error", "")
    failed_tests = _parse_assertion_errors(error_text)
    is_assertion = "AssertionError" in error_text or "assert" in error_text.lower()
    status = TestStatus.FAIL if is_assertion else TestStatus.ERROR

    return EvalResult(
        task_id=task_id,
        status=status,
        failed_tests=failed_tests,
        runtime_ms=elapsed_ms,
        stdout=result.get("stdout", ""),
        stderr=error_text,
    )


def _parse_assertion_errors(tb: str) -> list[FailedTest]:
    """Best-effort extraction of assertion failures from a traceback string."""
    if not tb:
        return []
    is_assertion = "AssertionError" in tb or "assert" in tb.lower()
    return [
        FailedTest(
            traceback=tb[-2000:],
            assertion="AssertionError" if is_assertion else "RuntimeError",
        )
    ]
