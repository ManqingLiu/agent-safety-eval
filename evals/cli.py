"""CLI entry point for the safety eval harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.scenarios.schema import Category, load_all_scenarios, load_scenario
from evals.scoring.report import aggregate_results, to_json, to_markdown

SCENARIO_DIR = Path(__file__).parent / "scenarios"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="safety-eval",
        description="Safety evaluation harness for tool-using code agents",
    )
    parser.add_argument("--category", choices=[c.value for c in Category],
                        help="Run scenarios from a specific category")
    parser.add_argument("--scenario", help="Run a specific scenario by ID")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--live", action="store_true",
                        help="Use live LLM API calls (default: mock)")
    parser.add_argument("--model", default=None,
                        help="Model ID for live mode (uses MODEL_ID env var if not set)")
    parser.add_argument("--recording-dir", type=Path, default=None,
                        help="Directory containing recorded LLM responses for mock mode")
    parser.add_argument("--record-dir", type=Path, default=None,
                        help="Save live LLM responses to this directory for later replay")
    parser.add_argument("--trace-dir", type=Path, default=None,
                        help="Save detailed per-scenario traces (tool log, agent output, filesystem)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON results to file")
    parser.add_argument("--markdown", action="store_true",
                        help="Output markdown report to stdout")
    parser.add_argument("--list", action="store_true", dest="list_scenarios",
                        help="List available scenarios without running")

    args = parser.parse_args(argv)

    if args.list_scenarios:
        _list_scenarios()
        return

    if not (args.all or args.category or args.scenario):
        parser.print_help()
        sys.exit(1)

    # Collect scenarios to run
    scenarios = _collect_scenarios(args)
    if not scenarios:
        print("No scenarios found.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(scenarios)} scenario(s)...\n")

    # Build runner
    from evals.harness.runner import ScenarioRunner
    client = _build_client(args) if args.live else None
    runner = ScenarioRunner(client=client, record_dir=args.record_dir, trace_dir=args.trace_dir)

    results = []
    for scenario in scenarios:
        print(f"  [{scenario.category.value}] {scenario.id} ... ", end="", flush=True)
        try:
            recording = _find_recording(args.recording_dir, scenario.id) if not args.live else None
            result = runner.run(scenario, recording_path=recording)
            results.append(result)
            status = "PASS" if result.passed else f"FAIL (severity={result.severity_score:.2f})"
            print(status)
            if result.violations:
                for v in result.violations:
                    print(f"    [{v.severity.value}] {v.description}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Report
    model = args.model or "unknown"
    report = aggregate_results(results, model=model)

    if args.output:
        args.output.write_text(to_json(report))
        print(f"\nResults written to {args.output}")

    if args.markdown:
        print("\n" + to_markdown(report))

    # Summary
    print(f"\n{'='*50}")
    print(f"Overall: {report['total_passed']}/{report['total_scenarios']} passed "
          f"({report['overall_safety_score']:.1%})")
    for cat, data in report["categories"].items():
        print(f"  {cat}: {data['passed']}/{data['total']} passed, avg_severity={data['avg_severity']:.3f}")


def _collect_scenarios(args: argparse.Namespace) -> list:
    if args.scenario:
        # Find specific scenario by ID
        for json_file in SCENARIO_DIR.rglob("*.json"):
            s = load_scenario(json_file)
            if s.id == args.scenario:
                return [s]
        print(f"Scenario not found: {args.scenario}", file=sys.stderr)
        return []

    all_scenarios = load_all_scenarios(SCENARIO_DIR)

    if args.category:
        return [s for s in all_scenarios if s.category.value == args.category]

    return all_scenarios  # --all


def _build_client(args: argparse.Namespace):
    import os
    from anthropic import Anthropic
    if args.model:
        os.environ["MODEL_ID"] = args.model
    return Anthropic()


def _find_recording(recording_dir: Path | None, scenario_id: str) -> Path | None:
    if not recording_dir:
        return None
    # Try scenario_id.json (dots replaced with underscores)
    filename = scenario_id.replace(".", "_") + ".json"
    path = recording_dir / filename
    return path if path.exists() else None


def _list_scenarios() -> None:
    scenarios = load_all_scenarios(SCENARIO_DIR)
    if not scenarios:
        print("No scenarios found.")
        return
    current_cat = ""
    for s in sorted(scenarios, key=lambda x: x.id):
        cat = s.category.value
        if cat != current_cat:
            print(f"\n[{cat}]")
            current_cat = cat
        tags = ", ".join(s.tags) if s.tags else ""
        print(f"  {s.id:<45} {s.name}")
        if tags:
            print(f"    tags: {tags}")


if __name__ == "__main__":
    main()
