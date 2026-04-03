"""SingleAgentAdapter -- wraps s02-style agent modules for eval execution."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from evals.adapters.base import AgentAdapter, AgentResult
from evals.harness.interceptor import ToolInterceptor


class SingleAgentAdapter(AgentAdapter):
    """Adapter for single-agent modules (s02_tool_use style).

    Patches module globals (WORKDIR, client, TOOL_HANDLERS, SYSTEM)
    then calls agent_loop(messages).
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.module: Any = None

    def prepare(
        self,
        workdir: Path,
        system_prompt: Optional[str],
        client: Any,
    ) -> None:
        # Force reimport to avoid state leaking between scenarios
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]

        self.module = importlib.import_module(self.module_name)
        self.module.WORKDIR = workdir
        self.module.client = client

        if system_prompt is not None:
            self.module.SYSTEM = system_prompt

        # Patch safe_path to use the new workdir
        # Resolve workdir to handle macOS /tmp -> /private/tmp symlink
        resolved_workdir = workdir.resolve()

        def patched_safe_path(p: str) -> Path:
            path = (resolved_workdir / p).resolve()
            if not path.is_relative_to(resolved_workdir):
                raise ValueError(f"Path escapes workspace: {p}")
            return path
        self.module.safe_path = patched_safe_path
        # Also patch WORKDIR to the resolved path so subprocess cwd matches
        self.module.WORKDIR = resolved_workdir

    def get_base_handlers(self) -> dict[str, Callable]:
        return dict(self.module.TOOL_HANDLERS)

    def install_handlers(self, wrapped_handlers: dict[str, Callable]) -> None:
        self.module.TOOL_HANDLERS = wrapped_handlers

    def run(self, user_prompt: str, max_turns: int) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        try:
            self._run_with_turn_limit(messages, max_turns)
        except _TurnLimitReached:
            pass
        except Exception as e:
            return AgentResult(messages=messages, error=str(e))

        final_text = self._extract_final_text(messages)
        return AgentResult(messages=messages, final_text=final_text)

    def _run_with_turn_limit(self, messages: list, max_turns: int) -> None:
        """Run agent_loop but enforce a turn limit to prevent runaway agents."""
        original_create = self.module.client.messages.create
        turn_count = 0

        def counted_create(**kwargs: Any) -> Any:
            nonlocal turn_count
            turn_count += 1
            if turn_count > max_turns:
                raise _TurnLimitReached()
            return original_create(**kwargs)

        self.module.client.messages.create = counted_create
        try:
            self.module.agent_loop(messages)
        finally:
            self.module.client.messages.create = original_create

    @staticmethod
    def _extract_final_text(messages: list[dict]) -> str:
        """Extract the last assistant text from the message history."""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    texts = []
                    for block in content:
                        if hasattr(block, "text") and block.text:
                            texts.append(block.text)
                        elif isinstance(block, dict) and block.get("text"):
                            texts.append(block["text"])
                    if texts:
                        return "\n".join(texts)
        return ""


class _TurnLimitReached(Exception):
    pass
