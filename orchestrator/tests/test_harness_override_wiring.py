"""Tests that Harness wires OverrideStore into Scheduler at construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.overrides import OverrideStore


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


class TestHarnessOverrideStoreWiring:
    def test_harness_wires_override_store_into_scheduler(
        self, config: OrchestratorConfig
    ):
        harness = Harness(config)

        # (a) must be set
        assert harness.scheduler._override_store is not None, (
            "Harness must wire OverrideStore into Scheduler at construction"
        )
        # (b) must be the right type
        assert isinstance(harness.scheduler._override_store, OverrideStore), (
            f"Expected OverrideStore, got {type(harness.scheduler._override_store)}"
        )
        # (c) must point at the canonical config path
        assert harness.scheduler._override_store.db_path == config.overrides_db_path, (
            f"db_path mismatch: {harness.scheduler._override_store.db_path!r} "
            f"!= {config.overrides_db_path!r}"
        )
