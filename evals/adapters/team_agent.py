"""TeamAgentAdapter -- wraps s09_agent_teams for multi-agent eval execution."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from evals.adapters.base import AgentAdapter, AgentResult
from evals.harness.interceptor import ToolInterceptor


class TeamAgentAdapter(AgentAdapter):
    """Adapter for multi-agent team modules (s09_agent_teams style).

    Patches module globals (WORKDIR, client, MODEL, BUS, TEAM, TOOL_HANDLERS,
    safe_path) then calls agent_loop(messages). Intercepts tool calls from
    both the lead agent and all spawned teammates.
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.module: Any = None
        self._teammate_messages: list[dict] = []
        self._message_lock = threading.Lock()
        self._bus: Any = None
        self._team: Any = None

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
        self.module.MODEL = os.environ.get("MODEL_ID", "claude-sonnet-4-20250514")

        if system_prompt is not None:
            self.module.SYSTEM = system_prompt

        # Patch safe_path to use the sandbox workdir
        # Resolve workdir to handle macOS /tmp -> /private/tmp symlink
        resolved_workdir = workdir.resolve()
        self.module.WORKDIR = resolved_workdir

        def patched_safe_path(p: str) -> Path:
            path = (resolved_workdir / p).resolve()
            if not path.is_relative_to(resolved_workdir):
                raise ValueError(f"Path escapes workspace: {p}")
            return path
        self.module._safe_path = patched_safe_path

        # Rebuild BUS and TEAM to use sandbox paths
        team_dir = resolved_workdir / ".team"
        inbox_dir = team_dir / "inbox"
        team_dir.mkdir(parents=True, exist_ok=True)
        inbox_dir.mkdir(parents=True, exist_ok=True)

        self.module.TEAM_DIR = team_dir
        self.module.INBOX_DIR = inbox_dir

        bus = self.module.MessageBus(inbox_dir)
        team = self.module.TeammateManager(team_dir)

        self.module.BUS = bus
        self.module.TEAM = team
        self._bus = bus
        self._team = team

        # Re-bind the module TOOL_HANDLERS to use new TEAM/BUS instances.
        # At this point WORKDIR is already sandbox-patched, so the base tool
        # lambdas (bash, read_file, etc.) will resolve paths via the patched
        # module globals.
        self.module.TOOL_HANDLERS = {
            "bash":            lambda **kw: self.module._run_bash(kw["command"]),
            "read_file":       lambda **kw: self.module._run_read(kw["path"], kw.get("limit")),
            "write_file":      lambda **kw: self.module._run_write(kw["path"], kw["content"]),
            "edit_file":       lambda **kw: self.module._run_edit(kw["path"], kw["old_text"], kw["new_text"]),
            "spawn_teammate":  lambda **kw: team.spawn(kw["name"], kw["role"], kw["prompt"]),
            "list_teammates":  lambda **kw: team.list_all(),
            "send_message":    lambda **kw: bus.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
            "read_inbox":      lambda **kw: json.dumps(bus.read_inbox("lead"), indent=2),
            "broadcast":       lambda **kw: bus.broadcast("lead", kw["content"], team.member_names()),
        }

    def get_base_handlers(self) -> dict[str, Callable]:
        return dict(self.module.TOOL_HANDLERS)

    def install_handlers(self, wrapped_handlers: dict[str, Callable]) -> None:
        self.module.TOOL_HANDLERS = wrapped_handlers
        self._wrapped_handlers = wrapped_handlers

        bus = self._bus
        team = self._team

        # Intercept message bus sends for MESSAGE_CONTAINS detection
        original_bus_send = bus.__class__.send

        adapter_self = self

        def intercepted_send(bus_self, sender: str, to: str, content: str,
                             msg_type: str = "message", extra: dict = None) -> str:
            with adapter_self._message_lock:
                adapter_self._teammate_messages.append({
                    "from": sender, "to": to, "content": content,
                    "msg_type": msg_type, "timestamp": time.time(),
                })
            return original_bus_send(bus_self, sender, to, content, msg_type, extra)

        import types
        bus.send = types.MethodType(intercepted_send, bus)

        # Patch TeammateManager._exec to route through interceptor wrappers
        def intercepted_exec(sender: str, tool_name: str, args: dict) -> str:
            handler = self._wrapped_handlers.get(tool_name)
            if handler:
                try:
                    return handler(**args)
                except Exception as e:
                    return f"Error: {e}"
            # Messaging tools not in lead's TOOL_HANDLERS
            if tool_name == "send_message":
                return bus.send(sender, args["to"], args["content"],
                                args.get("msg_type", "message"))
            if tool_name == "read_inbox":
                return json.dumps(bus.read_inbox(sender), indent=2)
            return f"Unknown tool: {tool_name}"

        team._exec = intercepted_exec

    def run(self, user_prompt: str, max_turns: int) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]

        try:
            self._run_with_turn_limit(messages, max_turns)
        except _TurnLimitReached:
            pass
        except Exception as e:
            return AgentResult(messages=messages, error=str(e))

        # Wait for all teammate threads to finish (with timeout)
        self._wait_for_teammates(timeout=60)

        final_text = self._extract_final_text(messages)
        return AgentResult(messages=messages, final_text=final_text)

    def _run_with_turn_limit(self, messages: list, max_turns: int) -> None:
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

    def _wait_for_teammates(self, timeout: float = 60) -> None:
        deadline = time.time() + timeout
        for name, thread in self._team.threads.items():
            remaining = deadline - time.time()
            if remaining > 0:
                thread.join(timeout=remaining)

    @staticmethod
    def _extract_final_text(messages: list[dict]) -> str:
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

    @property
    def teammate_messages(self) -> list[dict]:
        with self._message_lock:
            return list(self._teammate_messages)


class _TurnLimitReached(Exception):
    pass
