"""Violation and tool event data structures used throughout the eval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from evals.scenarios.schema import DetectionMethod, SeverityLevel


@dataclass
class ToolEvent:
    """A single tool invocation recorded by the interceptor."""
    tool: str
    inputs: dict[str, Any]
    timestamp: float
    output: str = ""
    error: Optional[str] = None


@dataclass
class Violation:
    """A detected safety violation."""
    criterion_id: str
    description: str
    severity: SeverityLevel
    weight: float
    detection_method: DetectionMethod
    evidence: str               # the actual input/output that triggered detection
    tool_name: str
    tool_inputs: dict[str, Any]
    timestamp: float


@dataclass
class EvalResult:
    """Complete result of running one scenario."""
    scenario_id: str
    passed: bool
    violations: list[Violation]
    severity_score: float               # 0.0 = clean, 1.0 = catastrophic
    task_completed: bool
    tool_call_count: int
    turn_count: int
    duration_seconds: float
    agent_output: str = ""
    filesystem_snapshot: dict[str, str] = field(default_factory=dict)
