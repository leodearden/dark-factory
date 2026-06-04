"""Tests for orchestrator.b3_gate module.

Covers: state load/save, record_launch idempotency, charge rolling-24h cap,
concurrency serialisation, check_proposal mechanical classification, freshness
P1/P2 against real git fixtures, _read_latest_proposal, _resolve_cap, CLI
main() wiring, and the two-way boundary test against the real _build_entry
producer from dry_run_unblock.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Shared helpers (adapted from test_dry_run_unblock.py)
# ---------------------------------------------------------------------------

def _init_git_repo(path: Path) -> str:
    """Init a minimal git repo at *path* (on 'main' branch), return HEAD sha."""
    p = str(path)
    subprocess.run(['git', 'init', '-b', 'main', p], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.name', 'Test User'],
                   check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'config', 'user.email', 'test@example.com'],
                   check=True, capture_output=True)
    (path / 'README.md').write_text('init')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'initial commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _diverge_feature_branch(path: Path) -> str:
    """Create a feature branch with one extra commit so HEAD != main.

    Returns the new HEAD sha (feature commit).
    """
    p = str(path)
    subprocess.run(['git', '-C', p, 'checkout', '-b', 'feature'],
                   check=True, capture_output=True)
    (path / 'extra.txt').write_text('feature work')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'feature commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _make_agent_result(*, success=True, cost_usd=0.50, structured_output=None,
                       subtype='', output='', duration_ms=1000):
    """Create a minimal MagicMock AgentResult stand-in for _build_entry tests."""
    r = MagicMock()
    r.success = success
    r.cost_usd = cost_usd
    r.structured_output = structured_output
    r.subtype = subtype
    r.output = output
    r.duration_ms = duration_ms
    return r


def _seed_tasks_db(project_root: Path, task_id: int, proposals: list) -> None:
    """Seed a minimal tasks.db at <project_root>/.taskmaster/tasks/tasks.db.

    Creates the table with columns: tag, id, status, priority, metadata, updated_at.
    Inserts one row with tag='master', the given task_id, and the proposals list
    serialised into metadata JSON as {'dry_run_proposals': proposals}.
    """
    db_dir = project_root / '.taskmaster' / 'tasks'
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / 'tasks.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag TEXT NOT NULL,
            id INTEGER NOT NULL,
            status TEXT,
            priority TEXT,
            metadata TEXT,
            updated_at TEXT,
            PRIMARY KEY (tag, id)
        )
    """)
    metadata = json.dumps({'dry_run_proposals': proposals})
    conn.execute(
        "INSERT OR REPLACE INTO tasks (tag, id, status, priority, metadata, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ('master', task_id, 'blocked', 'medium', metadata, '2026-06-04T00:00:00+00:00'),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# step-1: state load/save round-trip + atomicity
# ---------------------------------------------------------------------------

class TestStateLoadSave:
    def test_load_missing_returns_empty(self, tmp_path):
        from orchestrator.b3_gate import _load_state
        path = tmp_path / 'b3-state.json'
        state = _load_state(path)
        assert state == {'launches': [], 'charges': []}

    def test_save_creates_parents_and_roundtrips(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _save_state
        path = tmp_path / 'deep' / 'nested' / 'b3-state.json'
        state = {'launches': [{'task_id': '1', 'head_sha': 'abc'}], 'charges': []}
        _save_state(path, state)
        assert path.exists()
        loaded = _load_state(path)
        assert loaded == state

    def test_save_uses_atomic_replace(self, tmp_path, monkeypatch):
        """Verify os.replace is called (atomic swap), not a direct write."""
        from orchestrator import b3_gate
        replaced = []
        real_replace = os.replace

        def mock_replace(src, dst):
            replaced.append((src, dst))
            real_replace(src, dst)

        monkeypatch.setattr(os, 'replace', mock_replace)
        path = tmp_path / 'b3-state.json'
        b3_gate._save_state(path, {'launches': [], 'charges': []})
        assert len(replaced) == 1, 'expected exactly one os.replace call'
        src, dst = replaced[0]
        assert dst == str(path) or Path(dst) == path
        assert src != dst  # tmp != final

    def test_no_leftover_tmp(self, tmp_path):
        """No leftover .tmp file after a successful save."""
        from orchestrator.b3_gate import _save_state
        path = tmp_path / 'b3-state.json'
        _save_state(path, {'launches': [], 'charges': []})
        tmp_files = list(tmp_path.glob('*.tmp'))
        assert tmp_files == [], f'leftover tmp files: {tmp_files}'


# ---------------------------------------------------------------------------
# step-3: record_launch durable + idempotent
# ---------------------------------------------------------------------------

class TestRecordLaunch:
    _NOW = datetime(2026, 6, 4, 10, 0, 0, tzinfo=UTC)

    def test_first_call_records_and_persists(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _state_path, record_launch
        sp = _state_path(tmp_path)
        result = record_launch(
            '42', 'aabbcc', '2026-06-04T09:00:00+00:00',
            state_path=sp, now=self._NOW,
        )
        assert result == {'recorded': True, 'already_attempted': False}
        # Durability: reload from disk
        state = _load_state(sp)
        assert len(state['launches']) == 1
        entry = state['launches'][0]
        assert entry['task_id'] == '42'
        assert entry['head_sha'] == 'aabbcc'
        assert entry['investigated_at'] == '2026-06-04T09:00:00+00:00'
        assert 'recorded_at' in entry

    def test_second_call_same_key_is_idempotent(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _state_path, record_launch
        sp = _state_path(tmp_path)
        record_launch('42', 'aabbcc', '2026-06-04T09:00:00+00:00',
                      state_path=sp, now=self._NOW)
        # Second call with same (task_id, head_sha, investigated_at)
        result2 = record_launch('42', 'aabbcc', '2026-06-04T09:00:00+00:00',
                                state_path=sp, now=self._NOW)
        assert result2 == {'recorded': False, 'already_attempted': True}
        state = _load_state(sp)
        assert len(state['launches']) == 1, 'spent key must not be re-added'

    def test_different_investigated_at_rearms(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _state_path, record_launch
        sp = _state_path(tmp_path)
        record_launch('42', 'aabbcc', '2026-06-04T09:00:00+00:00',
                      state_path=sp, now=self._NOW)
        # New investigated_at == fresh investigation -> re-armed
        result = record_launch('42', 'aabbcc', '2026-06-04T09:30:00+00:00',
                               state_path=sp, now=self._NOW)
        assert result == {'recorded': True, 'already_attempted': False}
        state = _load_state(sp)
        assert len(state['launches']) == 2

    def test_different_head_sha_rearms(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _state_path, record_launch
        sp = _state_path(tmp_path)
        record_launch('42', 'aabbcc', '2026-06-04T09:00:00+00:00',
                      state_path=sp, now=self._NOW)
        result = record_launch('42', 'deadbeef', '2026-06-04T09:00:00+00:00',
                               state_path=sp, now=self._NOW)
        assert result == {'recorded': True, 'already_attempted': False}
        state = _load_state(sp)
        assert len(state['launches']) == 2

    def test_durability_across_restart(self, tmp_path):
        """Simulate restart: re-import and reload from disk confirms persistence."""
        from orchestrator.b3_gate import _load_state, _state_path, record_launch
        sp = _state_path(tmp_path)
        record_launch('7', 'sha1', 'ts1', state_path=sp, now=self._NOW)
        # Simulate restart: load fresh state from disk (no in-memory cache)
        state_after = _load_state(sp)
        assert len(state_after['launches']) == 1
        # Now call again — should see the spent key
        result = record_launch('7', 'sha1', 'ts1', state_path=sp, now=self._NOW)
        assert result['already_attempted'] is True
