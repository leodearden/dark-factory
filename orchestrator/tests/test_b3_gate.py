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


# ---------------------------------------------------------------------------
# step-5: charge rolling-24h cap
# ---------------------------------------------------------------------------

class TestCharge:
    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_two_charges_succeed_third_refused(self, tmp_path):
        from orchestrator.b3_gate import _state_path, charge
        sp = _state_path(tmp_path)
        r1 = charge('42', state_path=sp, cap=2, now=self._NOW)
        assert r1['charged'] is True
        assert r1['remaining'] == 1
        r2 = charge('42', state_path=sp, cap=2, now=self._NOW)
        assert r2['charged'] is True
        assert r2['remaining'] == 0
        r3 = charge('42', state_path=sp, cap=2, now=self._NOW)
        assert r3['charged'] is False
        assert r3['remaining'] == 0

    def test_old_charge_outside_window_not_counted(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _save_state, _state_path, charge
        sp = _state_path(tmp_path)
        # Seed an old charge older than 24h
        old_ts = (self._NOW - timedelta(hours=25)).isoformat()
        state = {'launches': [], 'charges': [{'task_id': '42', 'charged_at': old_ts}]}
        _save_state(sp, state)
        # With cap=1, the old charge must NOT count -> remaining=1 -> charge succeeds
        r = charge('42', state_path=sp, cap=1, now=self._NOW)
        assert r['charged'] is True
        assert r['remaining'] == 0

    def test_in_window_charge_counted(self, tmp_path):
        from orchestrator.b3_gate import _load_state, _save_state, _state_path, charge
        sp = _state_path(tmp_path)
        # Seed an in-window charge (12h ago)
        recent_ts = (self._NOW - timedelta(hours=12)).isoformat()
        state = {'launches': [], 'charges': [{'task_id': '42', 'charged_at': recent_ts}]}
        _save_state(sp, state)
        # With cap=1, in-window charge exhausts cap
        r = charge('42', state_path=sp, cap=1, now=self._NOW)
        assert r['charged'] is False
        assert r['remaining'] == 0

    def test_durability_in_window_persists_across_reload(self, tmp_path):
        """Charges survive disk reload — count resets only when window expires."""
        from orchestrator.b3_gate import _cap_remaining, _load_state, _state_path, charge
        sp = _state_path(tmp_path)
        charge('42', state_path=sp, cap=3, now=self._NOW)
        charge('42', state_path=sp, cap=3, now=self._NOW)
        # Reload state from disk
        state = _load_state(sp)
        rem = _cap_remaining(state, 3, self._NOW)
        assert rem == 1


# ---------------------------------------------------------------------------
# step-7: concurrency — two concurrent charge calls must serialize
# ---------------------------------------------------------------------------

class TestChargeConcurrency:
    """Two threads racing on cap=1 must not both succeed."""

    def test_only_one_charge_wins(self, tmp_path):
        import threading
        from orchestrator.b3_gate import _load_state, _state_path, charge

        sp = _state_path(tmp_path)
        # Ensure state file exists before threads start (avoids mkdir race)
        from orchestrator.b3_gate import _save_state
        _save_state(sp, {'launches': [], 'charges': []})

        now = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)
        results = []
        barrier = threading.Barrier(2)

        def worker():
            barrier.wait()  # Both threads release simultaneously
            r = charge('42', state_path=sp, cap=1, now=now)
            results.append(r)

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert len(results) == 2
        charged = [r for r in results if r['charged']]
        refused = [r for r in results if not r['charged']]
        assert len(charged) == 1, f'expected exactly 1 charged, got {results}'
        assert len(refused) == 1, f'expected exactly 1 refused, got {results}'

        # Persisted state must have exactly one in-window charge
        state = _load_state(sp)
        assert len(state['charges']) == 1, (
            f'expected exactly 1 persisted charge, got {state["charges"]}'
        )


# ---------------------------------------------------------------------------
# step-9: check_proposal mechanical classification (fake git runner)
# ---------------------------------------------------------------------------

def _fake_git_fresh(args, cwd):
    """Fake git runner: HEAD == recorded sha, diff is empty (fresh)."""
    if 'rev-parse' in args:
        return (0, 'aaabbbccc')
    return (0, '')  # diff empty


def _fake_git_never_called(args, cwd):
    raise AssertionError(f'git should not have been called; args={args}')


_LOW_RISK_ENTRY = {
    'proposal_text': 'Fix it',
    'risk_label': 'low',
    'files_referenced': ['foo.py'],
    'block_reason': 'test',
    'investigated_at': '2026-06-04T09:00:00+00:00',
    'head_sha': 'aaabbbccc',
    'main_sha': 'dddeeefff',
}


class TestCheckProposalMechanical:
    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_none_entry_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        result = check_proposal(
            None, worktree='/tmp', category=None,
            run_git=_fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT
        assert 'proposal' in result['reason'].lower()

    def test_medium_risk_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = {**_LOW_RISK_ENTRY, 'risk_label': 'medium'}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=_fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_human_review_risk_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = {**_LOW_RISK_ENTRY, 'risk_label': 'human-review-required'}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=_fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_status_key_present_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = {**_LOW_RISK_ENTRY, 'status': 'investigation_failed'}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=_fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_invalid_category_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        result = check_proposal(
            _LOW_RISK_ENTRY, worktree='/tmp', category='infra_issue',
            run_git=_fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT

    def test_valid_categories_not_aborted_on_category_check(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        for cat in ('task_failure', 'review_issues'):
            # Should NOT abort on category check (may abort/drift/fresh for other reasons)
            result = check_proposal(
                _LOW_RISK_ENTRY, worktree='/tmp', category=cat,
                run_git=_fake_git_fresh, now=self._NOW,
            )
            # Category check passes -> proceeds to freshness check -> fresh
            assert result['verdict'] != ABORT or 'category' not in result['reason'].lower(), \
                f'Unexpected abort on valid category={cat}: {result}'

    def test_no_category_skips_category_check(self):
        from orchestrator.b3_gate import check_proposal
        # category=None -> skip category check entirely -> proceeds to freshness
        result = check_proposal(
            _LOW_RISK_ENTRY, worktree='/tmp', category=None,
            run_git=_fake_git_fresh, now=self._NOW,
        )
        # Should not abort on category; may be fresh or drift depending on git
        assert 'verdict' in result

    def test_missing_head_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = {**_LOW_RISK_ENTRY}
        del entry['head_sha']
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=_fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
        assert 'sha anchor' in result['reason'].lower()

    def test_missing_main_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = {**_LOW_RISK_ENTRY}
        del entry['main_sha']
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=_fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
        assert 'sha anchor' in result['reason'].lower()

    def test_none_head_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = {**_LOW_RISK_ENTRY, 'head_sha': None}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=_fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
