"""Critic/Guide: analyzes attempts, produces addressable-fix abstractions."""

from __future__ import annotations

import json
import logging
import re

from .config import LLMConfig
from .llm import chat
from .models import CriticOutput, EvalResult, MergeDecision, SyntheticDrill, TestStatus
from .prompts import CRITIC_SUCCESS_TEMPLATE, CRITIC_SYSTEM, CRITIC_TASK_TEMPLATE

log = logging.getLogger(__name__)


def run_critic(
    cfg: LLMConfig,
    task_prompt: str,
    actor_code: str,
    eval_result: EvalResult,
    index_text: str,
) -> CriticOutput | None:
    """Invoke the Critic LLM and parse its structured JSON output."""
    if eval_result.status == TestStatus.PASS:
        user_msg = CRITIC_SUCCESS_TEMPLATE.format(
            task_prompt=task_prompt,
            actor_code=actor_code,
            runtime_ms=eval_result.runtime_ms,
            index_text=index_text,
        )
    else:
        eval_details = _format_eval_details(eval_result)
        user_msg = CRITIC_TASK_TEMPLATE.format(
            task_prompt=task_prompt,
            actor_code=actor_code,
            eval_status=eval_result.status.value,
            eval_details=eval_details,
            index_text=index_text,
        )

    raw = chat(cfg, CRITIC_SYSTEM, user_msg)
    return _parse_critic_output(raw)


def _format_eval_details(result: EvalResult) -> str:
    parts: list[str] = []
    if result.stderr:
        parts.append(f"Stderr:\n{result.stderr[:2000]}")
    for i, ft in enumerate(result.failed_tests):
        parts.append(f"Failed test {i+1}:")
        if ft.input:
            parts.append(f"  Input: {ft.input}")
        if ft.expected:
            parts.append(f"  Expected: {ft.expected}")
        if ft.actual:
            parts.append(f"  Actual: {ft.actual}")
        if ft.traceback:
            parts.append(f"  Traceback:\n{ft.traceback[:1000]}")
    return "\n".join(parts) if parts else "(no details)"


def _parse_critic_output(raw: str) -> CriticOutput | None:
    """Extract JSON from LLM response, tolerating markdown fences."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            log.warning("Critic output contained no valid JSON:\n%s", raw[:500])
            return None
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            log.warning("Failed to parse Critic JSON:\n%s", raw[:500])
            return None

    if data.get("addressable_fix") == "NO_PATTERN":
        return None

    drills = []
    for d in data.get("synthetic_drills", []):
        if isinstance(d, dict) and "prompt" in d:
            drills.append(
                SyntheticDrill(
                    prompt=d["prompt"],
                    expected_behavior=d.get("expected_behavior", ""),
                    test_code=d.get("test_code", ""),
                    rationale=d.get("rationale", ""),
                )
            )

    decision_str = data.get("decision", "create").lower()
    decision = MergeDecision.MERGE if decision_str == "merge" else MergeDecision.CREATE

    return CriticOutput(
        diagnosis=data.get("diagnosis", ""),
        addressable_fix=data.get("addressable_fix", ""),
        trigger_signals=data.get("trigger_signals", []),
        playbook=data.get("playbook", []),
        trace_summary=data.get("trace_summary", ""),
        when_to_open=data.get("when_to_open", ""),
        summary_paragraph=data.get("summary_paragraph", ""),
        decision=decision,
        merge_target_bucket_id=data.get("merge_target_bucket_id"),
        merge_confidence=float(data.get("merge_confidence", 0.0)),
        merge_rationale=data.get("merge_rationale", ""),
        synthetic_drills=drills,
        is_success_pattern=bool(data.get("is_success_pattern", False)),
    )
