"""Report generation -- aggregate EvalResults into leaderboard JSON and markdown."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from evals.scoring.violations import EvalResult


def aggregate_results(results: list[EvalResult], model: str = "unknown") -> dict[str, Any]:
    """Aggregate a list of EvalResults into a leaderboard-format dict."""
    by_category: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        cat = r.scenario_id.split(".")[0]
        by_category[cat].append(r)

    categories = {}
    total_passed = 0
    total_scenarios = len(results)

    for cat, cat_results in sorted(by_category.items()):
        passed = sum(1 for r in cat_results if r.passed)
        failed = len(cat_results) - passed
        avg_severity = (
            sum(r.severity_score for r in cat_results) / len(cat_results)
            if cat_results else 0.0
        )
        total_passed += passed
        categories[cat] = {
            "passed": passed,
            "failed": failed,
            "total": len(cat_results),
            "avg_severity": round(avg_severity, 3),
        }

    overall = total_passed / total_scenarios if total_scenarios > 0 else 0.0

    return {
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_safety_score": round(overall, 3),
        "total_scenarios": total_scenarios,
        "total_passed": total_passed,
        "categories": categories,
        "details": [_result_to_dict(r) for r in results],
    }


def to_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, default=str)


def to_markdown(report: dict[str, Any]) -> str:
    """Render a leaderboard report as markdown."""
    lines = [
        f"# Safety Eval Report",
        f"",
        f"**Model**: {report['model']}  ",
        f"**Date**: {report['timestamp']}  ",
        f"**Overall Safety Score**: {report['overall_safety_score']:.1%} "
        f"({report['total_passed']}/{report['total_scenarios']} passed)",
        f"",
        f"## Results by Category",
        f"",
        f"| Category | Passed | Failed | Avg Severity |",
        f"|----------|--------|--------|-------------|",
    ]
    for cat, data in report["categories"].items():
        lines.append(
            f"| {cat} | {data['passed']}/{data['total']} | {data['failed']} | {data['avg_severity']:.3f} |"
        )

    lines += ["", "## Scenario Details", ""]
    for detail in report["details"]:
        status = "PASS" if detail["passed"] else "FAIL"
        lines.append(f"### [{status}] {detail['scenario_id']}")
        lines.append(f"- Severity: {detail['severity_score']:.3f}")
        lines.append(f"- Task completed: {detail['task_completed']}")
        lines.append(f"- Tool calls: {detail['tool_call_count']}")
        lines.append(f"- Duration: {detail['duration_seconds']:.1f}s")
        if detail["violations"]:
            lines.append(f"- Violations:")
            for v in detail["violations"]:
                lines.append(f"  - **[{v['severity']}]** {v['description']}")
                lines.append(f"    Evidence: `{v['evidence'][:100]}`")
        lines.append("")

    return "\n".join(lines)


def _result_to_dict(r: EvalResult) -> dict[str, Any]:
    return {
        "scenario_id": r.scenario_id,
        "passed": r.passed,
        "severity_score": r.severity_score,
        "task_completed": r.task_completed,
        "tool_call_count": r.tool_call_count,
        "turn_count": r.turn_count,
        "duration_seconds": round(r.duration_seconds, 2),
        "violations": [
            {
                "criterion_id": v.criterion_id,
                "severity": v.severity.value,
                "description": v.description,
                "evidence": v.evidence[:200],
                "tool_name": v.tool_name,
            }
            for v in r.violations
        ],
    }
