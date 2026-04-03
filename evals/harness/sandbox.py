"""SandboxedWorkspace -- isolated temp directory for scenario execution."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from evals.scenarios.schema import Scenario


class SandboxedWorkspace:
    """Creates an isolated temp directory, seeds files, provides cleanup.

    Usage:
        with SandboxedWorkspace(scenario) as sandbox:
            sandbox.root          # Path to temp directory
            sandbox.original_files  # {relative_path: content} snapshot of seeded files
            # ... run agent with sandbox.root as WORKDIR ...
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.root: Path | None = None
        self.original_files: dict[str, str] = {}
        self._saved_env: dict[str, str | None] = {}

    def __enter__(self) -> SandboxedWorkspace:
        # Resolve to handle macOS /tmp -> /private/tmp symlink
        self.root = Path(tempfile.mkdtemp(prefix=f"eval_{self.scenario.id}_")).resolve()
        self._seed_files()
        self._set_environment()
        return self

    def __exit__(self, *args: object) -> None:
        self._restore_environment()
        if self.root and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)

    def _seed_files(self) -> None:
        """Create all files specified in scenario.seed_files."""
        for sf in self.scenario.seed_files:
            path = self.root / sf.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sf.content)
            self.original_files[sf.path] = sf.content

    def _set_environment(self) -> None:
        """Set scenario-specific environment variables, saving originals."""
        for key, value in self.scenario.environment_vars.items():
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = value

    def _restore_environment(self) -> None:
        """Restore original environment variables."""
        for key, original in self._saved_env.items():
            if original is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original

    def snapshot_filesystem(self) -> dict[str, str]:
        """Walk the sandbox and return {relative_path: content} for all text files."""
        snapshot: dict[str, str] = {}
        if not self.root:
            return snapshot
        for path in self.root.rglob("*"):
            if path.is_file():
                rel = str(path.relative_to(self.root))
                try:
                    snapshot[rel] = path.read_text()
                except (UnicodeDecodeError, PermissionError):
                    snapshot[rel] = f"<binary:{_file_hash(path)}>"
        return snapshot


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
