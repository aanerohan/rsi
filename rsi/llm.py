"""Thin wrapper around the OpenAI chat-completions API with tool-use support."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import LLMConfig


def _client(cfg: LLMConfig) -> OpenAI:
    return OpenAI(
        api_key=cfg.api_key or "no-key",
        base_url=cfg.base_url,
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def chat(
    cfg: LLMConfig,
    system: str,
    user: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Simple single-turn chat completion (no tools)."""
    client = _client(cfg)
    resp = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature if temperature is not None else cfg.temperature,
        max_tokens=max_tokens or cfg.max_tokens,
    )
    return resp.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
def chat_with_tools(
    cfg: LLMConfig,
    system: str,
    user: str,
    tools: List[Dict[str, Any]],
    tool_dispatcher: Callable[[str, Dict[str, Any]], str],
    max_tool_rounds: int = 5,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Multi-turn chat that handles tool calls.

    The model may call tools; ``tool_dispatcher(name, args) -> result``
    resolves them. Loops until the model produces a final text response
    or ``max_tool_rounds`` is exhausted.
    """
    import json

    client = _client(cfg)
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    temp = temperature if temperature is not None else cfg.temperature
    tokens = max_tokens or cfg.max_tokens

    for _ in range(max_tool_rounds):
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temp,
            max_tokens=tokens,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return msg.content or ""

        messages.append(msg.model_dump())

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = tool_dispatcher(tc.function.name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    final = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=temp,
        max_tokens=tokens,
    )
    return final.choices[0].message.content or ""
