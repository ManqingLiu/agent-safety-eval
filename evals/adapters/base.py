"""AgentAdapter ABC -- uniform interface for running any agent in eval context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from evals.harness.interceptor import ToolInterceptor


@dataclass
class AgentResult:
    """Structured result from an agent run."""
    messages: list[dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    error: Optional[str] = None
    turn_count: int = 0


class AgentAdapter(ABC):
    """Uniform interface for running any agent module in eval context.

    Lifecycle (called by ScenarioRunner):
        1. prepare(workdir, system_prompt, client)  -- import module, patch globals
        2. get_base_handlers()                       -- return raw tool handlers for interceptor
        3. install_handlers(wrapped)                 -- install interceptor-wrapped handlers
        4. run(user_prompt, max_turns)               -- execute agent loop
    """

    @abstractmethod
    def prepare(
        self,
        workdir: Path,
        system_prompt: Optional[str],
        client: Any,
    ) -> None:
        """Import the agent module and patch globals (WORKDIR, client, etc.).

        After this call, tool handlers will reference the sandbox workdir.
        """

    @abstractmethod
    def get_base_handlers(self) -> dict[str, Callable]:
        """Return the module's raw TOOL_HANDLERS (after prepare patched WORKDIR)."""

    @abstractmethod
    def install_handlers(self, wrapped_handlers: dict[str, Callable]) -> None:
        """Install interceptor-wrapped handlers into the module."""

    @abstractmethod
    def run(self, user_prompt: str, max_turns: int) -> AgentResult:
        """Execute the agent loop and return structured result."""

    # Keep old configure() as a convenience that calls prepare+install in one shot.
    def configure(
        self,
        workdir: Path,
        system_prompt: Optional[str],
        tool_interceptor: ToolInterceptor,
        client: Any,
    ) -> None:
        """Legacy one-shot configure (prepare + wrap + install)."""
        self.prepare(workdir, system_prompt, client)
        self.install_handlers(tool_interceptor.wrap())
