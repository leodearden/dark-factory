"""`orchestrator verify-cache status`: per-role modes, registry counts and the kill switch (task 5652)."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from click.testing import CliRunner

from orchestrator.cli import main
from orchestrator.verify_cache import keys, registry
from orchestrator.verify_evidence import UnitTier

_MODULE_KEY = keys.UnitKey.of('module:alpha:test', UnitTier.MODULE, 's', (('spec', 's'), ('tree:alpha', 'a' * 40)))
_TREE_KEY = keys.UnitKey.of('tree', UnitTier.TREE, 'p', (('spec', 'p'), ('tree', 'b' * 40)))


def _config(project: Path, **extra: object) -> Path:
    """A config at the project root, which IS ``tmp_path``.

    conftest's autouse ``_isolate_orch_config`` pins ``ORCH_PROJECT_ROOT`` to
    ``tmp_path`` and env outranks the YAML, so a nested root would be silently
    overridden (test_flake_report.py::TestFlakeLedgerCommand states the same).
    """
    cfg = project / 'dark-factory-orchestrator.yaml'
    cfg.write_text(yaml.dump({'project_root': str(project), **extra}))
    return cfg


def _status(cfg: Path):
    return CliRunner().invoke(main, ['verify-cache', 'status', '--config', str(cfg)])


def _store(project: Path) -> registry.Registry:
    return registry.Registry(registry.registry_db_path(project))


class TestVerifyCacheStatus:
    def test_a_project_without_a_registry_reports_a_cold_start_and_creates_nothing(self, tmp_path):
        result = _status(_config(tmp_path))

        assert result.exit_code == 0, result.output
        for shown in ('task=off', 'merge=off', 'tree=0', 'module=0', 'test=0', 'absent', 'policy: enabled'):
            assert shown in result.output, result.output
        assert str(registry.registry_db_path(tmp_path)) in result.output, result.output
        assert not registry.registry_db_path(tmp_path).exists(), result.output
        assert not (tmp_path / 'data').exists(), result.output

    def test_each_role_shows_its_own_mode(self, tmp_path):
        result = _status(_config(tmp_path, verify_cache={'task_mode': 'log'}))

        assert result.exit_code == 0, result.output
        assert 'task=log' in result.output, result.output
        assert 'merge=off' in result.output, result.output

    def test_recorded_greens_are_counted_by_tier(self, tmp_path):
        store = _store(tmp_path)
        for key in (_MODULE_KEY, _TREE_KEY):
            asyncio.run(store.record_green(key, evidence_event_id=7, run_id='r1', sha='abc'))

        result = _status(_config(tmp_path))

        assert result.exit_code == 0, result.output
        for shown in ('tree=1', 'module=1', 'test=0', 'present'):
            assert shown in result.output, result.output

    def test_a_tripped_kill_switch_shows_until_when_and_why(self, tmp_path):
        until = (datetime.now(UTC) + timedelta(days=1)).replace(microsecond=0)
        state = registry.PolicyState(disabled_until=until, reason='two leaks', streak=2, leaks=('l1', 'l2'))
        asyncio.run(_store(tmp_path).write_policy_state(state))

        result = _status(_config(tmp_path))

        assert result.exit_code == 0, result.output
        for shown in ('DISABLED', until.isoformat(), 'two leaks', 'l1', 'l2'):
            assert shown in result.output, result.output

    def test_a_garbage_registry_is_unreadable_and_claims_no_counts(self, tmp_path):
        db_path = registry.registry_db_path(tmp_path)
        db_path.parent.mkdir(parents=True)
        db_path.write_bytes(b'this is not a sqlite database' * 64)

        result = _status(_config(tmp_path))

        assert result.exit_code == 0, result.output
        assert 'UNREADABLE' in result.output, result.output
        assert 'not a database' in result.output, result.output
        assert 'tree=0' not in result.output, result.output
        assert 'policy:' not in result.output, result.output

    def test_a_policy_row_of_the_wrong_type_is_unreadable_not_a_traceback(self, tmp_path):
        asyncio.run(_store(tmp_path).policy_state())
        conn = sqlite3.connect(registry.registry_db_path(tmp_path))
        try:
            with conn:
                conn.execute("INSERT INTO policy_state (id, streak, leaks) VALUES (1, 0, '5')")
        finally:
            conn.close()

        result = _status(_config(tmp_path))

        assert result.exit_code == 0, result.output
        assert 'UNREADABLE' in result.output, result.output
        assert 'policy:' not in result.output, result.output

    def test_no_config_and_no_env_exits_one_with_an_error(self, monkeypatch):
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)

        result = CliRunner().invoke(main, ['verify-cache', 'status'])

        assert result.exit_code == 1, result.output
        assert 'Error:' in result.output, result.output
        assert '--config is required' in result.output, result.output
