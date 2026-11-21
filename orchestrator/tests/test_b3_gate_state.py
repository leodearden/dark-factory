"""Tests for orchestrator.b3_gate's state file and the two writers over it.

Covers: b3-state.json load/save round-trip and atomicity (delegated to
shared.safe_io.atomic_write_text), fail-closed on a corrupt state file,
record_launch idempotency, and the rolling-24h charge cap with its
concurrency serialisation.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

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
        from orchestrator.b3_gate import _save_state
        replaced = []
        real_replace = os.replace

        def mock_replace(src, dst):
            replaced.append((src, dst))
            real_replace(src, dst)

        monkeypatch.setattr(os, 'replace', mock_replace)
        path = tmp_path / 'b3-state.json'
        _save_state(path, {'launches': [], 'charges': []})
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
        from orchestrator.b3_gate import _save_state, _state_path, charge
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
        from orchestrator.b3_gate import _save_state, _state_path, charge
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
# task-1808 step-5: corrupt b3-state.json → cap-bearing paths fail CLOSED
# ---------------------------------------------------------------------------

class TestCorruptStateFailsClosed:
    """Corrupt b3-state.json → cap-bearing paths must fail CLOSED (cap exhausted).

    All three cap-bearing callers (charge, run_check, record_launch) must
    treat a corrupt state as fully exhausted and must NOT overwrite the file
    (preserves the corrupt bytes for forensics; prevents silent cap-reset).
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def _make_corrupt_state(self, tmp_path):
        """Write a corrupt JSON state file and return its Path."""
        from orchestrator.b3_gate import _state_path
        sp = _state_path(tmp_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text('not valid json {{{{', encoding='utf-8')
        return sp

    def test_charge_on_corrupt_returns_exhausted(self, tmp_path):
        """charge() on corrupt state → charged=False, remaining=0 (NOT a full-cap reset)."""
        from orchestrator.b3_gate import charge
        sp = self._make_corrupt_state(tmp_path)
        result = charge('t', state_path=sp, cap=6, now=self._NOW)
        assert result['charged'] is False, (
            f'Expected charged=False for corrupt state; got {result}'
        )
        assert result['remaining'] == 0, (
            f'Expected remaining=0 for corrupt state; got {result}'
        )

    def test_charge_on_corrupt_does_not_overwrite(self, tmp_path):
        """charge() on corrupt state must NOT overwrite the file."""
        from orchestrator.b3_gate import charge
        sp = self._make_corrupt_state(tmp_path)
        original = sp.read_text(encoding='utf-8')
        charge('t', state_path=sp, cap=6, now=self._NOW)
        assert sp.read_text(encoding='utf-8') == original, (
            'charge() must not overwrite a corrupt state file'
        )

    def test_run_check_cap_remaining_zero_on_corrupt(self, tmp_path):
        """run_check with a corrupt state file reports cap_remaining==0 and already_attempted==False."""
        import argparse
        import contextlib
        import io as _io
        import json as _json

        from orchestrator.b3_gate import run_check
        self._make_corrupt_state(tmp_path)

        args = argparse.Namespace(
            task_id='42',
            worktree=str(tmp_path),
            project_root=str(tmp_path),
            config=None,
            now=self._NOW.isoformat(),
            tag='master',
            category=None,
            verb='check',
        )
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_check(args)
        result = _json.loads(buf.getvalue())
        assert result['cap_remaining'] == 0, (
            f'Expected cap_remaining=0 for corrupt state; got {result}'
        )
        # Both state-derived fields must reflect the fail-closed contract.
        # already_attempted=False is the *permissive* value, but it is correct
        # here because we cannot read prior records — consumers must gate on
        # cap_remaining (or charge()), not already_attempted.
        assert result['already_attempted'] is False, (
            f'Expected already_attempted=False for corrupt state; got {result}'
        )

    def test_record_launch_on_corrupt_fails_closed(self, tmp_path):
        """record_launch() on corrupt state → recorded=False without overwriting."""
        from orchestrator.b3_gate import record_launch
        sp = self._make_corrupt_state(tmp_path)
        original = sp.read_text(encoding='utf-8')
        result = record_launch(
            '42', 'deadbeef', '2026-06-04T09:00:00+00:00',
            state_path=sp, now=self._NOW,
        )
        assert result['recorded'] is False, (
            f'Expected recorded=False for corrupt state; got {result}'
        )
        assert sp.read_text(encoding='utf-8') == original, (
            'record_launch() must not overwrite a corrupt state file'
        )

    def test_missing_state_still_yields_full_cap(self, tmp_path):
        """MISSING state → fresh (benign), charge() succeeds at full cap."""
        from orchestrator.b3_gate import _state_path, charge
        sp = _state_path(tmp_path)
        # File does not exist → benign fresh state
        result = charge('t', state_path=sp, cap=6, now=self._NOW)
        assert result['charged'] is True, (
            f'Expected charged=True for missing (fresh) state; got {result}'
        )
        assert result['remaining'] == 5, (
            f'Expected remaining=5 for fresh state with cap=6; got {result}'
        )


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


class TestDelegatesToSharedAtomicWriter:
    """``b3_gate._save_state`` delegates to ``shared.safe_io.atomic_write_text``.

    Task 3223 consolidated the repo's tmp+rename writers into ``shared.safe_io``,
    which also gives this site a unique-per-writer temp name in place of the old
    fixed ``<dest>.json.tmp`` (two concurrent writers used to share it).
    ``mode`` must stay at the umask default: this file is read by other
    processes (the dashboard, the gamma/epsilon watchers, scripts/drain_check.py),
    so narrowing it to 0o600 is the specific silent regression this task avoids.
    """

    @staticmethod
    def _recorder(monkeypatch):
        import shared.safe_io as _safe_io

        calls = []
        real = _safe_io.atomic_write_text

        def recorder(path, text, **kwargs):
            calls.append((path, text, kwargs))
            return real(path, text, **kwargs)

        monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
        return calls

    @staticmethod
    def _assert_common(kwargs):
        assert kwargs.get('mkdir') is True, 'this site created its parent dir'
        assert kwargs.get('encoding') == 'utf-8'
        assert not kwargs.get('fsync'), 'this site never fsynced'
        assert kwargs.get('mode') is None, (
            'umask default, NOT 0o600 — this file is read by other processes'
        )

    def test_delegates_with_preserved_semantics(self, tmp_path: Path, monkeypatch) -> None:
        from orchestrator.b3_gate import _save_state

        calls = self._recorder(monkeypatch)
        _save_state(tmp_path / 'nested' / 'b3.json', {'launches': [], 'charges': []})

        assert len(calls) == 1, f'expected exactly one delegated call, got {calls}'
        self._assert_common(calls[0][2])

    def test_on_disk_mode_matches_write_text_reference(self, tmp_path: Path) -> None:
        from orchestrator.b3_gate import _save_state

        reference = tmp_path / 'reference.json'
        reference.write_text('ref', encoding='utf-8')
        path = tmp_path / 'b3.json'
        _save_state(path, {'launches': [], 'charges': []})
        assert path.stat().st_mode & 0o777 == reference.stat().st_mode & 0o777

    def test_oserror_still_propagates(self, tmp_path: Path, monkeypatch) -> None:
        """This site propagates — it has no fail-open boundary."""
        import shared.safe_io as _safe_io

        from orchestrator.b3_gate import _save_state

        def boom(*_a, **_kw):
            raise OSError('disk full')

        monkeypatch.setattr(_safe_io, 'atomic_write_text', boom)
        with pytest.raises(OSError, match='disk full'):
            _save_state(tmp_path / 'b3.json', {})
