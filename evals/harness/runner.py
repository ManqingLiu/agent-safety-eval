"""ScenarioRunner -- orchestrates sandbox, adapter, execution, and scoring."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from evals.adapters.base import AgentAdapter
from evals.adapters.single_agent import SingleAgentAdapter
from evals.adapters.team_agent import TeamAgentAdapter
from evals.harness.interceptor import ToolInterceptor
from evals.harness.mock_client import MockAnthropicClient
from evals.harness.recorder import RecordingClient
from evals.harness.sandbox import SandboxedWorkspace
from evals.scenarios.schema import Scenario
from evals.scoring.scorer import ScenarioScorer
from evals.scoring.violations import EvalResult


def create_adapter(scenario: Scenario) -> AgentAdapter:
    """Factory: create the right adapter for the scenario's agent_type."""
    if scenario.agent_type in ("single", "subagent"):
        return SingleAgentAdapter(scenario.agent_module)
    elif scenario.agent_type == "team":
        return TeamAgentAdapter(scenario.agent_module)
    else:
        raise ValueError(f"Unknown agent_type: {scenario.agent_type}")


class ScenarioRunner:
    """Runs a single scenario end-to-end: sandbox -> agent -> score."""

    def __init__(
        self,
        client: Optional[Any] = None,
        record_dir: Optional[Path] = None,
        trace_dir: Optional[Path] = None,
    ):
        self.client = client
        self.record_dir = record_dir
        self.trace_dir = trace_dir

    def run(self, scenario: Scenario, recording_path: Optional[Path] = None) -> EvalResult:
        """Execute a scenario and return scored results.

        Lifecycle:
            1. Determine client (live or mock)
            2. Create sandbox with seeded files
            3. adapter.prepare()  -- import module, patch WORKDIR to sandbox
            4. Build interceptor from adapter.get_base_handlers()
            5. adapter.install_handlers(interceptor.wrap())
            6. adapter.run()
            7. Post-run checks + scoring
        """
        # Determine client
        recorder = None
        if recording_path:
            client = MockAnthropicClient.from_recording(recording_path)
        elif self.client:
            client = self.client
            if self.record_dir:
                recorder = RecordingClient(client)
                client = recorder
        else:
            raise ValueError("No client: provide recording_path for mock mode or client for live mode")

        # Create adapter
        adapter = create_adapter(scenario)

        # Run in sandbox
        with SandboxedWorkspace(scenario) as sandbox:
            # Step 1: prepare -- import module, patch WORKDIR to sandbox root
            adapter.prepare(
                workdir=sandbox.root,
                system_prompt=scenario.system_prompt_override,
                client=client,
            )

            # Step 2: build interceptor from handlers that NOW reference sandbox
            interceptor = ToolInterceptor(
                original_handlers=adapter.get_base_handlers(),
                violation_criteria=scenario.violation_criteria,
            )

            # Step 3: install wrapped handlers
            adapter.install_handlers(interceptor.wrap())

            # Step 4: run
            start = time.time()
            result = adapter.run(scenario.user_prompt, scenario.max_agent_turns)
            duration = time.time() - start

            # Post-run checks
            final_fs = sandbox.snapshot_filesystem()
            interceptor.check_agent_output(result.final_text)
            interceptor.check_filesystem(sandbox.original_files, final_fs)

            # Save recording if in record mode
            if recorder and self.record_dir:
                filename = scenario.id.replace(".", "_") + ".json"
                recorder.save(self.record_dir / filename)

            # Save detailed trace for human review
            if self.trace_dir:
                self._save_trace(
                    scenario, interceptor, result, final_fs,
                    sandbox.original_files, duration,
                )

            # Score
            scorer = ScenarioScorer(scenario)
            return scorer.score(
                tool_log=interceptor.log,
                violations=interceptor.violations,
                agent_output=result.final_text,
                final_filesystem=final_fs,
                duration_seconds=duration,
            )

    def _save_trace(
        self,
        scenario: Scenario,
        interceptor: ToolInterceptor,
        result: Any,
        final_fs: dict[str, str],
        original_fs: dict[str, str],
        duration: float,
    ) -> None:
        """Save a detailed trace JSON for human review of agent behavior."""
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        filename = scenario.id.replace(".", "_") + ".json"

        # Build tool log with full inputs and outputs
        tool_log = []
        for event in interceptor.log:
            tool_log.append({
                "tool": event.tool,
                "inputs": event.inputs,
                "output": event.output,
                "error": event.error,
            })

        # Build violations list
        violations = []
        for v in interceptor.violations:
            violations.append({
                "criterion_id": v.criterion_id,
                "severity": v.severity.value,
                "description": v.description,
                "evidence": v.evidence,
                "tool_name": v.tool_name,
                "tool_inputs": v.tool_inputs,
            })

        # Identify files created or modified in the sandbox
        files_created = {p: c for p, c in final_fs.items() if p not in original_fs}
        files_modified = {
            p: {"before": original_fs[p], "after": final_fs[p]}
            for p in final_fs
            if p in original_fs and original_fs[p] != final_fs[p]
        }

        trace = {
            "scenario_id": scenario.id,
            "scenario_description": scenario.description,
            "user_prompt": scenario.user_prompt,
            "system_prompt_override": scenario.system_prompt_override,
            "duration_seconds": round(duration, 2),
            "agent_output": result.final_text,
            "tool_log": tool_log,
            "violations": violations,
            "files_seeded": {sf.path: sf.content for sf in scenario.seed_files},
            "files_created": files_created,
            "files_modified": files_modified,
            "final_filesystem": final_fs,
        }

        (self.trace_dir / filename).write_text(
            json.dumps(trace, indent=2, default=str)
        )
