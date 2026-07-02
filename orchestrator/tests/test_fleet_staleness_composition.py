"""Fleet-staleness δ (task 2027): scenarios 9-10 — coordinator composition
against the COMMITTED orchestrator/config.yaml, and burst-coalescing under
that same committed config.

Context — what α/β/γ already cover vs. what δ adds
----------------------------------------------------
- ``service_restart.py``'s coordinator decision logic (arm/debounce/fire) is
  unit-tested directly.
- ``orchestrator/tests/test_harness_service_restart.py::TestOrchestratorCoordinatorEndToEnd``
  already exercises ``_build_orchestrator_restart_coordinator`` end-to-end
  (note_merge -> drain-gate -> systemd-run argv) but against the harness
  fixture's MOCK config values: ``scripts/restart-orchestrator.sh`` and
  watch_prefixes=``['orchestrator/src/']`` (a single-unit placeholder, not the
  fleet script the PRD ships).
- ``tests/scripts/test_orchestrator_restart_config_drift.py`` (γ) guards the
  COMMITTED ``orchestrator/config.yaml`` bytes and their pydantic round-trip,
  but never drives that config through the harness builder or a fire.

δ closes the remaining gap: that ``_build_orchestrator_restart_coordinator``
plumbs the COMMITTED fleet values (``scripts/restart-all-orchestrators.sh`` +
the 5 committed watch prefixes) end-to-end through to both the coordinator's
fields and the systemd-run executor argv (scenario 9, I3), and that the
existing debounce/pending burst-coalescing behaviour holds under that same
committed config (scenario 10, I2).

Scenarios 9-10 load the committed config via ``OrchestratorConfig()``
(``ORCH_CONFIG_PATH``, mirroring the γ drift test) and graft its
``orchestrator_restart_*`` fields onto a harness built from the shared
``mock_orch_config`` fixture (``project_root=tmp_path``) rather than
constructing ``Harness(real_config)`` — a full real-config harness would
resolve ``project_root``/``overrides_db_path`` under the real repo and write
to ``data/`` (test-hygiene risk). Grafting the model-parsed committed values
ties the deployment config to the code path (a field rename/typo would show
up as a reverted-to-default value) while keeping the test hermetic.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.service_restart import StaleServiceRestartCoordinator

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DF_CONFIG_PATH = REPO_ROOT / 'orchestrator' / 'config.yaml'

# The committed fleet-restart watch prefixes (orchestrator/config.yaml,
# task γ) — asserted verbatim below so a drift in the committed YAML fails
# this test as well as the γ drift test.
_COMMITTED_WATCH_PREFIXES = [
    'orchestrator/src/',
    'escalation/src/',
    'orchestrator/pyproject.toml',
    'orchestrator/uv.lock',
    'escalation/pyproject.toml',
]


def _load_committed_orchestrator_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Load the COMMITTED orchestrator/config.yaml through the real pydantic model.

    Mirrors ``test_orchestrator_restart_config_drift.py``'s round-trip pattern
    (γ): this drives the ACTUAL parsed ``orchestrator_restart_*`` values, not
    the raw YAML, so a field rename/typo in config.py would surface here as a
    reverted-to-default value (``OrchestratorConfig`` uses ``extra='ignore'``)
    exactly as it would in the γ drift test.

    Note: the orchestrator/tests autouse ``_isolate_orch_config`` fixture
    deletes ``ORCH_CONFIG_PATH`` and pins ``ORCH_PROJECT_ROOT`` to ``tmp_path``
    for every test in this package; setting ``ORCH_CONFIG_PATH`` here (via the
    same function-scoped ``monkeypatch``) runs after that autouse setup and
    wins, so this always loads the committed file. The resulting
    ``project_root`` on the returned object is still pinned to ``tmp_path`` by
    that same autouse fixture, which is irrelevant here — only the
    ``orchestrator_restart_*`` fields are read off this object and grafted
    onto the (already-hermetic) harness config below.
    """
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(DF_CONFIG_PATH))
    return OrchestratorConfig()


def _graft_committed_restart_config(harness: Harness, committed: OrchestratorConfig) -> None:
    """Copy the committed config's orchestrator_restart_* fields onto harness.config."""
    harness.config.orchestrator_restart_on_merge_enabled = (
        committed.orchestrator_restart_on_merge_enabled
    )
    harness.config.orchestrator_restart_script = committed.orchestrator_restart_script
    harness.config.orchestrator_restart_watch_prefixes = (
        committed.orchestrator_restart_watch_prefixes
    )
    harness.config.orchestrator_restart_debounce_secs = (
        committed.orchestrator_restart_debounce_secs
    )
    harness.config.orchestrator_restart_on_active_secs = (
        committed.orchestrator_restart_on_active_secs
    )


# ---------------------------------------------------------------------------
# Fixture (mirrors orchestrator/tests/test_harness_service_restart.py::harness)
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Minimal harness with mocked heavy deps, configured for the orchestrator restart.

    Only the fields ``_build_orchestrator_restart_coordinator`` reads are
    given non-MagicMock placeholder values here (mirroring the fixture
    defaults in test_harness_service_restart.py); each test grafts the
    COMMITTED config's ``orchestrator_restart_*`` values on top before calling
    the builder or arming a merge.
    """
    mock_orch_config.orchestrator_restart_on_merge_enabled = False
    mock_orch_config.orchestrator_restart_debounce_secs = 300.0
    mock_orch_config.orchestrator_restart_watch_prefixes = ['orchestrator/src/']
    mock_orch_config.orchestrator_restart_script = 'scripts/restart-orchestrator.sh'
    mock_orch_config.orchestrator_restart_on_active_secs = 10

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
    return h


# ---------------------------------------------------------------------------
# Scenario 9 (I3): coordinator composition against the COMMITTED fleet config
# ---------------------------------------------------------------------------


class TestOrchestratorCoordinatorCommittedConfigComposition:
    """_build_orchestrator_restart_coordinator() plumbs the COMMITTED fleet
    config (scripts/restart-all-orchestrators.sh + 5 watch prefixes) through
    to the coordinator's fields, with the I3 merge-drain gate unchanged."""

    def test_coordinator_fields_match_committed_fleet_config(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Coordinator fields reflect the committed fleet script/prefixes, not
        the single-unit placeholder (restart-orchestrator.sh) used elsewhere."""
        committed = _load_committed_orchestrator_config(monkeypatch)
        _graft_committed_restart_config(harness, committed)

        coord = harness._build_orchestrator_restart_coordinator()

        assert isinstance(coord, StaleServiceRestartCoordinator)
        assert coord.enabled is True
        assert coord._script_path == 'scripts/restart-all-orchestrators.sh'
        assert coord._watch_prefixes == _COMMITTED_WATCH_PREFIXES
        assert coord._require_idle is True
        # I3: the merge-drain precondition is unchanged by the fleet config.
        assert coord._restart_precondition == harness._merge_pipeline_idle

    @pytest.mark.asyncio
    async def test_fires_systemd_run_with_fleet_script_via_committed_config(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: note_merge arms the real (list-index-2) coordinator on a
        watched-path diff; once the merge pipeline is drained, the systemd-run
        argv targets the FLEET script (restart-all-orchestrators.sh) — not
        restart-orchestrator.sh — with the committed on_active_secs.
        """
        committed = _load_committed_orchestrator_config(monkeypatch)
        _graft_committed_restart_config(harness, committed)
        # Deterministic fire — no need to wait the committed 300s debounce.
        harness.config.orchestrator_restart_debounce_secs = 0.0

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        # Drained pipeline: no in-flight/verifying merge, empty queue.
        mock_smw.return_value.snapshot.return_value = {'depth': 0}
        assert harness._merge_queue.empty()

        harness.git_ops.get_merge_diff_files = AsyncMock(
            return_value=(['orchestrator/src/orchestrator/git_ops.py'], None)
        )
        orch_coord = harness._service_restart_coordinators[2]
        assert orch_coord._script_path == 'scripts/restart-all-orchestrators.sh'
        expected_script = str(
            Path(harness.config.project_root) / 'scripts/restart-all-orchestrators.sh'
        )

        await harness._note_merge_all('task-1', 'base', 'head')
        assert orch_coord.is_pending is True

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b'', None))
        fake_proc.returncode = 0
        with patch(
            'orchestrator.service_restart.asyncio.create_subprocess_exec',
            new_callable=AsyncMock,
            return_value=fake_proc,
        ) as mock_exec:
            fired = await harness._maybe_restart_stale_service(agents_idle=True)

        assert fired is True
        mock_exec.assert_awaited_once()
        pos_args = mock_exec.call_args.args
        assert pos_args[0] == 'systemd-run'
        assert '--on-active=10' in pos_args
        assert '--unit=orch-selfrestart-on-merge-0.service' in pos_args
        assert expected_script in pos_args
        assert orch_coord.is_pending is False


# ---------------------------------------------------------------------------
# Scenario 10 (I2): burst coalescing under the COMMITTED fleet config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBurstCoalescingUnderCommittedConfig:
    """A burst of two core merges inside the debounce window coalesces into
    exactly one restart-all fire — re-asserted under the COMMITTED fleet
    config (not the mock config's single-unit script)."""

    async def test_burst_of_two_merges_fires_exactly_once(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        committed = _load_committed_orchestrator_config(monkeypatch)
        _graft_committed_restart_config(harness, committed)
        harness.config.orchestrator_restart_debounce_secs = 0.0

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker') as mock_smw, \
             patch('asyncio.create_task'), \
             patch('orchestrator.merge_queue.check_merge_liveness_margin'):
            await harness._start_merge_worker()

        # Drained pipeline throughout: no in-flight/verifying merge, empty queue.
        mock_smw.return_value.snapshot.return_value = {'depth': 0}
        assert harness._merge_queue.empty()

        orch_coord = harness._service_restart_coordinators[2]
        assert orch_coord._script_path == 'scripts/restart-all-orchestrators.sh'

        harness.git_ops.get_merge_diff_files = AsyncMock(
            return_value=(['orchestrator/src/orchestrator/git_ops.py'], None)
        )

        # Burst: two core merges land inside the debounce window before any
        # fire is attempted.
        await harness._note_merge_all('task-1', 'b1', 'h1')
        assert orch_coord.is_pending is True
        await harness._note_merge_all('task-2', 'b2', 'h2')
        assert orch_coord.is_pending is True  # still a single pending burst

        fake_proc = MagicMock()
        fake_proc.communicate = AsyncMock(return_value=(b'', None))
        fake_proc.returncode = 0
        with patch(
            'orchestrator.service_restart.asyncio.create_subprocess_exec',
            new_callable=AsyncMock,
            return_value=fake_proc,
        ) as mock_exec:
            fired = await harness._maybe_restart_stale_service(agents_idle=True)

        assert fired is True
        mock_exec.assert_awaited_once()
        assert orch_coord.is_pending is False
