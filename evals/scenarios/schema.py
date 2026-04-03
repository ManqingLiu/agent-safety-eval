"""Scenario data model -- every eval scenario is a JSON file conforming to these structures."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class Category(Enum):
    SAFETY = "safety"
    INJECTION = "injection"
    PRIVACY = "privacy"
    MULTI_AGENT = "multi_agent"


class DetectionMethod(Enum):
    COMMAND_PATTERN = "command_pattern"      # regex matched against bash commands
    PATH_ACCESS = "path_access"             # glob/regex matched against file paths in any tool
    OUTPUT_CONTAINS = "output_contains"     # regex matched against tool outputs + agent text
    FILE_MODIFIED = "file_modified"         # check if a seeded file was changed
    FILE_CREATED = "file_created"           # check if an unexpected file was created
    MESSAGE_CONTAINS = "message_contains"   # regex against inter-agent messages (multi-agent)


class SeverityLevel(Enum):
    CRITICAL = "critical"   # sandbox escape, data exfiltration
    HIGH = "high"           # unauthorized file access, secret exposure
    MEDIUM = "medium"       # instruction hierarchy violation
    LOW = "low"             # minor policy deviation


@dataclass
class SeedFile:
    path: str
    content: str
    permissions: str = "644"


@dataclass
class ViolationCriterion:
    id: str
    description: str
    detection_method: DetectionMethod
    pattern: str
    severity: SeverityLevel
    weight: float = 1.0


@dataclass
class ExpectedBehavior:
    should_complete_task: bool
    should_refuse: bool = False
    max_tool_calls: Optional[int] = None
    expected_files: list[str] = field(default_factory=list)


@dataclass
class Scenario:
    id: str
    category: Category
    name: str
    description: str
    agent_type: str                             # "single" | "subagent" | "team"
    agent_module: str                           # e.g. "agents.s02_tool_use"
    user_prompt: str
    seed_files: list[SeedFile]
    violation_criteria: list[ViolationCriterion]
    expected_behavior: ExpectedBehavior
    system_prompt_override: Optional[str] = None
    environment_vars: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    timeout_seconds: int = 60
    max_agent_turns: int = 20


def load_scenario(path: Path) -> Scenario:
    """Load a scenario from a JSON file."""
    raw = json.loads(path.read_text())

    seed_files = [SeedFile(**sf) for sf in raw.get("seed_files", [])]

    criteria = []
    for vc in raw.get("violation_criteria", []):
        criteria.append(ViolationCriterion(
            id=vc["id"],
            description=vc["description"],
            detection_method=DetectionMethod(vc["detection_method"]),
            pattern=vc["pattern"],
            severity=SeverityLevel(vc["severity"]),
            weight=vc.get("weight", 1.0),
        ))

    eb = raw.get("expected_behavior", {})
    expected = ExpectedBehavior(
        should_complete_task=eb.get("should_complete_task", True),
        should_refuse=eb.get("should_refuse", False),
        max_tool_calls=eb.get("max_tool_calls"),
        expected_files=eb.get("expected_files", []),
    )

    return Scenario(
        id=raw["id"],
        category=Category(raw["category"]),
        name=raw["name"],
        description=raw["description"],
        agent_type=raw["agent_type"],
        agent_module=raw["agent_module"],
        user_prompt=raw["user_prompt"],
        seed_files=seed_files,
        violation_criteria=criteria,
        expected_behavior=expected,
        system_prompt_override=raw.get("system_prompt_override"),
        environment_vars=raw.get("environment_vars", {}),
        tags=raw.get("tags", []),
        timeout_seconds=raw.get("timeout_seconds", 60),
        max_agent_turns=raw.get("max_agent_turns", 20),
    )


def load_all_scenarios(base_dir: Path) -> list[Scenario]:
    """Load all .json scenario files under base_dir recursively."""
    scenarios = []
    for json_file in sorted(base_dir.rglob("*.json")):
        scenarios.append(load_scenario(json_file))
    return scenarios
