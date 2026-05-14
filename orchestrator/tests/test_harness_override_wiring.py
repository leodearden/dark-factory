"""Integration tests that Harness wires OverrideStore into Scheduler correctly.

These tests verify *behaviour*: overrides written to config.overrides_db_path
before Harness construction are readable through the scheduler the Harness
wires up.  That exercises the full wiring path (correct DB file, correct
project_root key) without pinning the constructor signature.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.overrides import OverrideStore


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


class TestHarnessOverrideStoreIntegration:
    def test_scheduler_reads_overrides_from_canonical_db_path(
        self, config: OrchestratorConfig
    ):
        """Overrides pre-written to config.overrides_db_path are visible to
        the scheduler Harness wires up — confirms the store reaches the right
        file without pinning the OverrideStore constructor argument."""
        project_root_str = str(config.project_root)

        # Pre-populate an override at the canonical DB path.
        seed_store = OverrideStore(config.overrides_db_path)
        seed_store.set_override(project_root_str, "task-999", boost_tier="critical")

        # Harness constructs its own OverrideStore at the same path.
        harness = Harness(config)

        # The scheduler must be able to read the pre-written override, proving
        # it was wired to the right SQLite file.
        overrides = harness.scheduler._override_store.get_overrides(project_root_str)
        assert "task-999" in overrides, (
            f"Scheduler did not see pre-written override; got keys: {list(overrides)}"
        )
        assert overrides["task-999"].boost_tier == "critical", (
            f"boost_tier mismatch: {overrides['task-999'].boost_tier!r} != 'critical'"
        )
