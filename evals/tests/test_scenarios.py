"""Tests that all scenario JSON files are valid and parse correctly."""

from pathlib import Path

import pytest

from evals.scenarios.schema import (
    Category, DetectionMethod, SeverityLevel, load_all_scenarios, load_scenario,
)

SCENARIO_DIR = Path(__file__).parent.parent / "scenarios"


class TestScenarioLoading:
    def test_all_scenarios_parse(self):
        """Every .json file under scenarios/ must parse without error."""
        scenarios = load_all_scenarios(SCENARIO_DIR)
        assert len(scenarios) >= 16, f"Expected at least 16 scenarios, got {len(scenarios)}"

    def test_all_scenarios_have_valid_categories(self):
        for s in load_all_scenarios(SCENARIO_DIR):
            assert isinstance(s.category, Category), f"{s.id}: invalid category"

    def test_all_scenarios_have_violation_criteria(self):
        for s in load_all_scenarios(SCENARIO_DIR):
            assert len(s.violation_criteria) > 0, f"{s.id}: no violation criteria"

    def test_all_criteria_have_valid_detection_methods(self):
        for s in load_all_scenarios(SCENARIO_DIR):
            for vc in s.violation_criteria:
                assert isinstance(vc.detection_method, DetectionMethod), \
                    f"{s.id}/{vc.id}: invalid detection method"

    def test_all_criteria_have_valid_severity(self):
        for s in load_all_scenarios(SCENARIO_DIR):
            for vc in s.violation_criteria:
                assert isinstance(vc.severity, SeverityLevel), \
                    f"{s.id}/{vc.id}: invalid severity"

    def test_scenario_ids_are_unique(self):
        scenarios = load_all_scenarios(SCENARIO_DIR)
        ids = [s.id for s in scenarios]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[x for x in ids if ids.count(x) > 1]}"

    def test_scenario_id_matches_category(self):
        """Scenario ID should start with its category prefix."""
        for s in load_all_scenarios(SCENARIO_DIR):
            assert s.id.startswith(s.category.value + "."), \
                f"{s.id}: ID should start with '{s.category.value}.'"


class TestCategoryCounts:
    def test_safety_has_4_scenarios(self):
        scenarios = load_all_scenarios(SCENARIO_DIR / "safety")
        assert len(scenarios) == 4

    def test_injection_has_4_scenarios(self):
        scenarios = load_all_scenarios(SCENARIO_DIR / "injection")
        assert len(scenarios) == 4

    def test_privacy_has_4_scenarios(self):
        scenarios = load_all_scenarios(SCENARIO_DIR / "privacy")
        assert len(scenarios) == 4

    def test_multi_agent_has_4_scenarios(self):
        scenarios = load_all_scenarios(SCENARIO_DIR / "multi_agent")
        assert len(scenarios) == 4


class TestSpecificScenarios:
    def test_unauthorized_file_access(self):
        s = load_scenario(SCENARIO_DIR / "safety" / "unauthorized_file_access.json")
        assert s.id == "safety.unauthorized_file_access"
        assert s.category == Category.SAFETY
        assert len(s.violation_criteria) >= 3
        assert s.expected_behavior.should_refuse is True

    def test_file_based_injection(self):
        s = load_scenario(SCENARIO_DIR / "injection" / "file_based_injection.json")
        assert s.id == "injection.file_based"
        assert s.category == Category.INJECTION
        assert any("curl" in vc.pattern for vc in s.violation_criteria)

    def test_canary_token(self):
        s = load_scenario(SCENARIO_DIR / "privacy" / "canary_token_leakage.json")
        assert s.id == "privacy.canary_token"
        assert any("CANARY" in vc.pattern for vc in s.violation_criteria)

    def test_conflicting_instructions(self):
        s = load_scenario(SCENARIO_DIR / "multi_agent" / "conflicting_instructions.json")
        assert s.id == "multi_agent.conflicting_instructions"
        assert s.agent_type == "team"
