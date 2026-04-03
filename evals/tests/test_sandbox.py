"""Tests for SandboxedWorkspace -- isolated filesystem for scenarios."""

import os

import pytest

from evals.harness.sandbox import SandboxedWorkspace
from evals.scenarios.schema import (
    Category, ExpectedBehavior, Scenario, SeedFile, SeverityLevel, ViolationCriterion,
    DetectionMethod,
)


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="test.sandbox_scenario",
        category=Category.SAFETY,
        name="Test Scenario",
        description="A test scenario",
        agent_type="single",
        agent_module="agents.s02_tool_use",
        user_prompt="do something",
        seed_files=[],
        violation_criteria=[],
        expected_behavior=ExpectedBehavior(should_complete_task=True),
    )
    defaults.update(overrides)
    return Scenario(**defaults)


class TestSandboxCreation:
    def test_creates_temp_directory(self):
        scenario = _make_scenario()
        with SandboxedWorkspace(scenario) as sandbox:
            assert sandbox.root.exists()
            assert sandbox.root.is_dir()

    def test_cleanup_removes_directory(self):
        scenario = _make_scenario()
        with SandboxedWorkspace(scenario) as sandbox:
            root = sandbox.root
        assert not root.exists()

    def test_seeds_files(self):
        scenario = _make_scenario(seed_files=[
            SeedFile(path="README.md", content="# Hello"),
            SeedFile(path="src/main.py", content="print('hi')"),
        ])
        with SandboxedWorkspace(scenario) as sandbox:
            assert (sandbox.root / "README.md").read_text() == "# Hello"
            assert (sandbox.root / "src/main.py").read_text() == "print('hi')"

    def test_creates_nested_directories(self):
        scenario = _make_scenario(seed_files=[
            SeedFile(path="a/b/c/deep.txt", content="deep"),
        ])
        with SandboxedWorkspace(scenario) as sandbox:
            assert (sandbox.root / "a/b/c/deep.txt").read_text() == "deep"

    def test_tracks_original_files(self):
        scenario = _make_scenario(seed_files=[
            SeedFile(path="data.txt", content="original"),
        ])
        with SandboxedWorkspace(scenario) as sandbox:
            assert sandbox.original_files == {"data.txt": "original"}


class TestEnvironmentVars:
    def test_sets_and_restores_env_vars(self):
        scenario = _make_scenario(environment_vars={"TEST_EVAL_VAR": "hello"})

        assert "TEST_EVAL_VAR" not in os.environ

        with SandboxedWorkspace(scenario) as sandbox:
            assert os.environ["TEST_EVAL_VAR"] == "hello"

        assert "TEST_EVAL_VAR" not in os.environ

    def test_preserves_existing_env_vars(self):
        os.environ["TEST_EXISTING"] = "original"
        scenario = _make_scenario(environment_vars={"TEST_EXISTING": "overridden"})

        with SandboxedWorkspace(scenario) as sandbox:
            assert os.environ["TEST_EXISTING"] == "overridden"

        assert os.environ["TEST_EXISTING"] == "original"
        del os.environ["TEST_EXISTING"]


class TestFilesystemSnapshot:
    def test_snapshot_captures_all_files(self):
        scenario = _make_scenario(seed_files=[
            SeedFile(path="a.txt", content="aaa"),
            SeedFile(path="dir/b.txt", content="bbb"),
        ])
        with SandboxedWorkspace(scenario) as sandbox:
            snapshot = sandbox.snapshot_filesystem()
            assert snapshot["a.txt"] == "aaa"
            assert snapshot["dir/b.txt"] == "bbb"

    def test_snapshot_captures_new_files(self):
        scenario = _make_scenario(seed_files=[
            SeedFile(path="original.txt", content="exists"),
        ])
        with SandboxedWorkspace(scenario) as sandbox:
            # Simulate agent creating a new file
            (sandbox.root / "new_file.txt").write_text("agent created this")
            snapshot = sandbox.snapshot_filesystem()
            assert "new_file.txt" in snapshot
            assert snapshot["new_file.txt"] == "agent created this"

    def test_snapshot_detects_modifications(self):
        scenario = _make_scenario(seed_files=[
            SeedFile(path="data.txt", content="original"),
        ])
        with SandboxedWorkspace(scenario) as sandbox:
            (sandbox.root / "data.txt").write_text("modified")
            snapshot = sandbox.snapshot_filesystem()
            assert snapshot["data.txt"] == "modified"
            assert sandbox.original_files["data.txt"] == "original"
