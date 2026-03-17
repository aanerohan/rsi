"""Actor LLM: generates solutions with selective bucket access via tool use."""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict

from .config import LLMConfig
from .llm import chat, chat_with_tools
from .memory import MemoryStore
from .prompts import ACTOR_SYSTEM, ACTOR_TASK_TEMPLATE

log = logging.getLogger(__name__)

READ_BUCKET_TOOL = {
    "type": "function",
    "function": {
        "name": "read_bucket",
        "description": (
            "Fetch the full content of an addressable-fix bucket by its ID. "
            "Returns the detailed playbook (step-by-step fix instructions), "
            "synthetic drill examples, and canonical failure episodes. "
            "Use this when the bucket's summary paragraph in the index "
            "suggests it is relevant to the current task."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bucket_id": {
                    "type": "string",
                    "description": "The bucket ID from the index (e.g. '178f57542cba')",
                },
            },
            "required": ["bucket_id"],
        },
    },
}


def generate_solution(
    cfg: LLMConfig,
    task_prompt: str,
    index_text: str,
    memory: MemoryStore | None = None,
) -> str:
    """Generate a solution, optionally using tool calls to read buckets.

    If *memory* is provided and the index has entries, the Actor can call
    ``read_bucket(bucket_id)`` to fetch full bucket content before writing
    code. Otherwise falls back to a simple single-turn call.
    """
    user_msg = ACTOR_TASK_TEMPLATE.format(
        task_prompt=task_prompt,
        index_text=index_text,
    )

    has_buckets = memory is not None and "(no addressable fixes" not in index_text
    if not has_buckets:
        raw = chat(cfg, ACTOR_SYSTEM, user_msg)
        return _clean_code(raw)

    def dispatcher(name: str, args: Dict[str, Any]) -> str:
        if name != "read_bucket":
            return f"Unknown tool: {name}"
        bid = args.get("bucket_id", "")
        log.info("Actor requested bucket: %s", bid)
        ctx = memory.fetch_bucket_context(bid)  # type: ignore[union-attr]
        if ctx is None:
            return f"Bucket '{bid}' not found."
        return ctx

    raw = chat_with_tools(
        cfg,
        ACTOR_SYSTEM,
        user_msg,
        tools=[READ_BUCKET_TOOL],
        tool_dispatcher=dispatcher,
    )
    return _clean_code(raw)


def _clean_code(raw: str) -> str:
    """Strip markdown fences and trailing commentary from LLM output."""
    text = raw.strip()
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    return text
