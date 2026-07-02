"""Tests for Harness.reload_config().

Task 2006 (config-reload beta) — plans/config-hot-reload-prd.md. Exercises the
harness-level orchestration around task alpha's pure config.py engine
(RELOADABLE_FIELDS / diff_config / apply_reload): thread-off load (I1
fail-closed), same-turn atomic apply (I4), config_path injection, and the
config_reload audit event (I7).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.config import (
    ConfigRequiredError,
    ModuleConfig,
    OrchestratorConfig,
    diff_config,
)
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

# ---------------------------------------------------------------------------
# Helpers replicated (not imported) from test_harness_park_stop.py:64/87 —
# the established pattern for these harness test helpers.
# ---------------------------------------------------------------------------


def _make_harness_with_mocks(tmp_path: Path) -> tuple[Harness, MagicMock, EventStore]:
    """Build a Harness with a real Scheduler, a MagicMock RunStore, and a real EventStore.

    The Harness is constructed via __new__ to avoid the full startup sequence
    (which requires a running MCP server), then the relevant attributes are
    populated directly — matching the pattern in test_harness_terminal_status_watcher.py.
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)

    # Inject a mock RunStore so we can spy on persistence calls.
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-test-0001'

    # Inject a real EventStore pointing at a tmp DB so we can query events.
    db_path = tmp_path / 'events.db'
    event_store = EventStore(db_path, 'run-test-0001')
    harness.event_store = event_store

    return harness, mock_run_store, event_store


def _query_events(event_store: EventStore, event_type_str: str) -> list[dict]:
    """Query all rows matching event_type_str from the EventStore's DB."""
    conn = sqlite3.connect(str(event_store.db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT * FROM events WHERE event_type = ? ORDER BY id',
        (event_type_str,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _other_model(current: str) -> str:
    """Return a model name distinct from *current*, for building a `fresh` config.

    NOTE: a live OrchestratorConfig's field values are NOT the bare pydantic
    class-level ``Field(default=...)`` values — ``settings_customise_sources``
    layers the package-bundled ``defaults.yaml`` under the project's
    ``config.yaml`` (see ``YamlSettingsSource``), so e.g. ``models.implementer``
    resolves to ``'sonnet'`` in this repo rather than the class default
    ``'opus'``. Deriving the "changed to" value from the actual live value
    (instead of hardcoding literal 'opus'/'sonnet' strings) keeps these tests
    correct regardless of that layering.
    """
    return 'opus' if current != 'opus' else 'sonnet'


class TestHarnessReloadConfigSuccess:
    """Success-path behavior for Harness.reload_config() (task 2006 step 1)."""

    @pytest.mark.asyncio
    async def test_reload_config_applies_allowlisted_change(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """One allowlisted leaf differs: applied, live mutated, audited, no warning.

        Also locks in PRD decision 7: fresh's discovered ``_module_configs``
        (populated by the real load_config's filesystem walk) is never copied
        onto the live config — it's a PrivateAttr outside model_fields, so
        alpha's ``_iter_leaves``/``apply_reload`` never visits it.
        """
        harness, _, event_store = _make_harness_with_mocks(tmp_path)

        live_implementer = harness.config.models.implementer
        new_implementer = _other_model(live_implementer)

        fresh = OrchestratorConfig(project_root=tmp_path)
        fresh.models.implementer = new_implementer
        fresh._module_configs = {'foo': ModuleConfig(prefix='foo')}

        # Independent snapshot equal to harness.config's pre-call state, used
        # to precompute the expected diff the same way apply_reload does
        # internally — avoids hardcoding leaf counts/values in the assertion.
        live_snapshot = OrchestratorConfig(project_root=tmp_path)
        expected_diff = diff_config(live_snapshot, fresh)
        assert expected_diff.applied_candidates == {
            'models.implementer': {'old': live_implementer, 'new': new_implementer}
        }
        assert expected_diff.restart_required == {}

        before_module_configs = harness.config._module_configs

        config_path = tmp_path / 'orchestrator.yaml'
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))

        with (
            patch('orchestrator.harness.load_config', return_value=fresh),
            caplog.at_level(logging.WARNING, logger='orchestrator.harness'),
        ):
            report = await harness.reload_config()

        expected_report = {
            'reloaded': True,
            'config_path': str(config_path),
            'applied': dict(expected_diff.applied_candidates),
            'restart_required': {},
            'unchanged': expected_diff.unchanged,
            'error': None,
        }
        assert report == expected_report

        assert harness.config.models.implementer == new_implementer
        assert harness.config._module_configs == before_module_configs, (
            "fresh's discovered _module_configs must NOT be copied onto live "
            '(PRD decision 7)'
        )

        rows = _query_events(event_store, 'config_reload')
        assert len(rows) == 1, f'Expected exactly one config_reload event row; got {rows!r}'
        assert json.loads(rows[0]['data']) == report

        harness_warnings = [
            r for r in caplog.records
            if r.name == 'orchestrator.harness' and r.levelno >= logging.WARNING
        ]
        assert not harness_warnings, (
            'Expected no WARNING on a clean success; got '
            + repr([r.getMessage() for r in harness_warnings])
        )


class TestHarnessReloadConfigLoadFailure:
    """I1 fail-closed behavior when load_config() raises (task 2006 step 3)."""

    @pytest.mark.asyncio
    async def test_reload_config_load_raises_is_fail_closed(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """A corrupted/invalid config file: reloaded=false, live untouched, audited, warns."""
        harness, _, event_store = _make_harness_with_mocks(tmp_path)

        config_path = tmp_path / 'orchestrator.yaml'
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))

        before = harness.config.model_dump_json()

        with (
            patch(
                'orchestrator.harness.load_config',
                side_effect=ConfigRequiredError('boom'),
            ),
            caplog.at_level(logging.WARNING, logger='orchestrator.harness'),
        ):
            report = await harness.reload_config()

        assert report['reloaded'] is False
        assert report['config_path'] == str(config_path)
        assert report['applied'] == {}
        assert report['restart_required'] == {}
        assert report['unchanged'] == 0
        assert report['error'] and 'boom' in report['error']

        assert harness.config.model_dump_json() == before, (
            'live config must be byte-identical after a failed load (I1)'
        )

        rows = _query_events(event_store, 'config_reload')
        assert len(rows) == 1, f'Expected exactly one config_reload event row; got {rows!r}'
        assert json.loads(rows[0]['data']) == report

        harness_warnings = [
            r for r in caplog.records
            if r.name == 'orchestrator.harness' and r.levelno >= logging.WARNING
        ]
        assert harness_warnings, 'Expected at least one WARNING on a failed reload'
