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


class TestMockConfigParkEvictionNoLeak:
    """Regression coverage for task 2045 (same failure mode as tasks 1313/1339):
    Harness(mock_orch_config) must not stringify an unpatched MagicMock
    park_eviction_requests_db_path into a stray on-disk sqlite file.
    """

    def test_harness_from_mock_config_leaks_no_magicmock_park_eviction_file(
        self, mock_orch_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # Isolate any relative-path side-effect into the auto-cleaned tmp dir
        # instead of wherever pytest was invoked from (normally orchestrator/).
        monkeypatch.chdir(tmp_path)

        harness = Harness(mock_orch_config)

        # Guard against the assertion below passing vacuously: confirm the
        # store/connect path was actually exercised, not skipped by a future
        # refactor that stops wiring ParkEvictionRequestStore into Harness.
        assert isinstance(
            harness.scheduler._park_eviction_store, ParkEvictionRequestStore
        ), (
            f'Expected ParkEvictionRequestStore, got '
            f'{type(harness.scheduler._park_eviction_store)!r}'
        )

        strays = sorted(
            str(p) for p in tmp_path.glob('<MagicMock*park_eviction_requests_db_path*')
        )
        assert strays == [], (
            f'mock_orch_config leaked stray MagicMock db file(s): {strays}'
        )
