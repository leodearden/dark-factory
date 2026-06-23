"""Verify Harness wires ParkEvictionRequestStore into Scheduler correctly.

Mirrors test_harness_override_wiring.py: construct the Harness for a
temp-project config and assert that harness.scheduler._park_eviction_store
is a ParkEvictionRequestStore instance at config.park_eviction_requests_db_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.park_eviction_requests import ParkEvictionRequestStore


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


class TestHarnessParkEvictionStoreWiring:
    def test_scheduler_has_park_eviction_store_at_canonical_db_path(
        self, config: OrchestratorConfig
    ):
        """Harness must wire a ParkEvictionRequestStore into Scheduler at
        config.park_eviction_requests_db_path — mirroring the override_store
        wiring so the drain actually runs each tick in production.
        """
        harness = Harness(config)

        store = harness.scheduler._park_eviction_store
        assert isinstance(store, ParkEvictionRequestStore), (
            f'Expected ParkEvictionRequestStore, got {type(store)!r}'
        )
        assert store.db_path == config.park_eviction_requests_db_path, (
            f'db_path mismatch: store has {store.db_path!r}, '
            f'config expects {config.park_eviction_requests_db_path!r}'
        )
