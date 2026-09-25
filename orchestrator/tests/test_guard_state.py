"""Tests for the ``orchestrator.guard_state`` durable-guard primitive (task 5352).

Covers:
  step-1: the ``collections.abc`` contract both facades must satisfy, driven
          purely in-memory (``path=None``) — this is the interface the nine
          existing call sites already speak, so the wiring steps can swap the
          facades in without editing a single one of them.
  step-3: durability against a real file — round-trip across a simulated
          restart, write-through on every mutation, fail-open loads, and the
          load-merge-write that keeps two owners of one file from erasing
          each other.
  step-5: the TTL that bounds it — expiry hides AND prunes, the boundary is
          exact, and a re-asserted set member neither refreshes its clock nor
          rewrites its file.
  step-15: a batch of N inserts costs ONE write, not N+1 — every flush
          re-serialises the whole map, so the per-element form made a cold
          start's first tick O(N^2).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, MutableMapping, MutableSet
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import shared.safe_io as _safe_io

from orchestrator.guard_state import PersistentMap, PersistentSet

_TTL = timedelta(days=1)

# A round, arbitrary instant.  Every TTL assertion is relative to it, so no
# test here depends on the wall clock.
_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _frozen_clock() -> tuple[list[datetime], Callable[[], datetime]]:
    """A clock the test advances by hand.

    ``guard_state`` never reads the wall clock internally (the
    ``chronic_flake.FilingLedger`` convention), which is what makes expiry
    exactly assertable instead of sleep-dependent.
    """
    frozen = [_T0]
    return frozen, lambda: frozen[0]


def _spy_on_writes(monkeypatch) -> list[tuple]:
    """Record every ``atomic_write_text`` call, then perform it for real.

    Mirrors the recorder in ``test_b3_gate.py`` /
    ``test_landed_outbox.py``: patching the shared module attribute is what
    makes the delegation itself observable, so a hand-rolled tmp+rename would
    fail these tests as well as the repo's AST fence.
    """
    calls: list[tuple] = []
    real = _safe_io.atomic_write_text

    def recorder(path, text, **kwargs):
        calls.append((path, text, kwargs))
        return real(path, text, **kwargs)

    monkeypatch.setattr(_safe_io, 'atomic_write_text', recorder)
    return calls


# ---------------------------------------------------------------------------
# step-1 — PersistentSet speaks MutableSet
# ---------------------------------------------------------------------------

class TestPersistentSetSemantics:
    """The four set-shaped guards use only ``add`` / ``discard`` / ``in`` /
    ``len`` / ``sorted`` / ``update`` / ``==``; pin exactly those."""

    def test_is_a_mutable_set(self):
        assert isinstance(PersistentSet(None, ttl=_TTL), MutableSet)

    def test_add_then_membership(self):
        s = PersistentSet(None, ttl=_TTL)
        assert 'esc-1' not in s
        s.add('esc-1')
        assert 'esc-1' in s

    def test_discard_removes(self):
        s = PersistentSet(None, ttl=_TTL)
        s.add('esc-1')
        s.discard('esc-1')
        assert 'esc-1' not in s

    def test_discard_of_absent_key_is_a_noop(self):
        """``_maybe_promote_blocker``'s done-clear arm discards unconditionally."""
        s = PersistentSet(None, ttl=_TTL)
        s.discard('never-added')
        assert len(s) == 0

    def test_len_counts_entries(self):
        s = PersistentSet(None, ttl=_TTL)
        assert len(s) == 0
        s.add('a')
        s.add('b')
        assert len(s) == 2

    def test_add_is_idempotent_for_membership_and_length(self):
        s = PersistentSet(None, ttl=_TTL)
        s.add('a')
        s.add('a')
        assert len(s) == 1

    def test_sorted_iterates_keys(self):
        """``_watch_for_escalation`` builds its ``--exclude-id`` argv this way."""
        s = PersistentSet(None, ttl=_TTL)
        s.update(['esc-2', 'esc-1'])
        assert sorted(s) == ['esc-1', 'esc-2']

    def test_update_inserts_every_element(self):
        """``_mark_coalesce_derailed`` marks a whole train in one call."""
        s = PersistentSet(None, ttl=_TTL)
        s.update(['a', 'b'])
        assert 'a' in s
        assert 'b' in s

    def test_equals_plain_set_in_both_directions(self):
        """``test_merge_queue_coalesce.py`` asserts ``== {'m1','m2'}`` against
        the real attribute, so the facade must compare equal to a plain set.

        Satisfied by inheriting ``collections.abc.Set.__eq__`` — no ``__eq__``
        is written on the facade.
        """
        s = PersistentSet(None, ttl=_TTL)
        s.update(['m1', 'm2'])
        plain = {'m1', 'm2'}
        assert s == plain
        # Bound to a local rather than written inline so the reflected
        # direction — the one that exercises `set.__eq__` returning
        # NotImplemented and Python retrying ours — is not a lint-flagged
        # Yoda condition.
        assert plain == s

    def test_empty_equals_plain_empty_set(self):
        assert PersistentSet(None, ttl=_TTL) == set()


# ---------------------------------------------------------------------------
# step-1 — PersistentMap speaks MutableMapping
# ---------------------------------------------------------------------------

class TestPersistentMapSemantics:
    """The five map-shaped guards use only ``[k]`` / ``[k] = v`` / ``[k] += 1``
    / ``get`` / ``pop`` / ``in`` / ``len`` / ``==``; pin exactly those."""

    def test_is_a_mutable_mapping(self):
        assert isinstance(PersistentMap(None, ttl=_TTL), MutableMapping)

    def test_setitem_then_getitem(self):
        m = PersistentMap(None, ttl=_TTL)
        m['k'] = 1
        assert m['k'] == 1

    def test_stores_str_values_too(self):
        """``open_fix_tasks[fp] = task_id`` stores a task id, not a count."""
        m = PersistentMap(None, ttl=_TTL)
        m['fp'] = 'fix-1'
        assert m['fp'] == 'fix-1'

    def test_get_returns_default_for_missing(self):
        """``self._retry_counts.get(esc_id, 0)`` is the steward's ladder read."""
        m = PersistentMap(None, ttl=_TTL)
        assert m.get('missing', 0) == 0

    def test_augmented_increment(self):
        """``self._red_advance_counts[fp] += 1`` is the offline lane's idiom."""
        m = PersistentMap(None, ttl=_TTL)
        m['k'] = 1
        m['k'] += 1
        assert m['k'] == 2

    def test_pop_removes_and_returns_then_defaults(self):
        """The resolve-path cleanup pops with a default and must not raise."""
        m = PersistentMap(None, ttl=_TTL)
        m['k'] = 7
        assert m.pop('k', None) == 7
        assert 'k' not in m
        assert m.pop('k', None) is None

    def test_membership_and_len(self):
        m = PersistentMap(None, ttl=_TTL)
        assert len(m) == 0
        m['k'] = 1
        assert 'k' in m
        assert len(m) == 1
        del m['k']
        assert 'k' not in m
        assert len(m) == 0

    def test_iterates_keys(self):
        m = PersistentMap(None, ttl=_TTL)
        m['b'] = 1
        m['a'] = 2
        assert sorted(m) == ['a', 'b']

    def test_equals_plain_dict_in_both_directions(self):
        """``assert worker.open_fix_tasks == {}`` in test_offline_lane.py
        depends on this; ``Mapping.__eq__`` supplies it."""
        m = PersistentMap(None, ttl=_TTL)
        empty: dict[str, str] = {}
        assert m == empty
        assert empty == m
        m['k'] = 'v'
        filled = {'k': 'v'}
        assert m == filled
        assert filled == m


# ---------------------------------------------------------------------------
# step-3 — the state actually survives a restart
# ---------------------------------------------------------------------------

class TestGuardStateDurability:
    """A second instance on the same path is how a fleet redeploy looks from
    inside the guard: same file, fresh process memory."""

    def test_set_round_trips_across_a_restart(self, tmp_path: Path):
        """The whole task, as one assertion."""
        path = tmp_path / 'capped.json'
        PersistentSet(path, ttl=_TTL).add('esc-1')
        assert 'esc-1' in PersistentSet(path, ttl=_TTL)

    def test_map_round_trips_int_and_str_values(self, tmp_path: Path):
        path = tmp_path / 'counts.json'
        first = PersistentMap(path, ttl=_TTL)
        first['esc-1'] = 3
        first['fp-1'] = 'fix-42'

        second = PersistentMap(path, ttl=_TTL)
        assert second['esc-1'] == 3
        assert second['fp-1'] == 'fix-42'

    def test_write_is_through_not_at_exit(self, tmp_path: Path):
        """No ``flush``/``close`` call is made between the mutation and the
        re-read — a guard that only persisted at exit would lose everything
        to the SIGKILL half of a redeploy."""
        path = tmp_path / 'capped.json'
        PersistentSet(path, ttl=_TTL).add('esc-1')
        assert path.exists(), 'the single add must already be on disk'

    def test_deletion_is_durable(self, tmp_path: Path):
        """A consumed guard must not resurrect on the next start."""
        set_path = tmp_path / 'capped.json'
        first_set = PersistentSet(set_path, ttl=_TTL)
        first_set.add('esc-1')
        first_set.discard('esc-1')
        assert 'esc-1' not in PersistentSet(set_path, ttl=_TTL)

        map_path = tmp_path / 'counts.json'
        first_map = PersistentMap(map_path, ttl=_TTL)
        first_map['esc-1'] = 1
        del first_map['esc-1']
        assert 'esc-1' not in PersistentMap(map_path, ttl=_TTL)

    def test_absent_file_is_an_empty_store(self, tmp_path: Path):
        assert len(PersistentSet(tmp_path / 'never-written.json', ttl=_TTL)) == 0

    def test_corrupt_file_fails_open(self, tmp_path: Path):
        """Fail-open, per ``load_json_or_warn(..., on_corrupt='warn')``: a
        guard whose state file got truncated must still run."""
        path = tmp_path / 'capped.json'
        path.write_text('{not json', encoding='utf-8')

        guard = PersistentSet(path, ttl=_TTL)
        assert len(guard) == 0
        guard.add('esc-2')
        assert 'esc-2' in PersistentSet(path, ttl=_TTL)

    def test_schema_drifted_rows_are_dropped_individually(self, tmp_path: Path):
        """One bad row must not void the whole file — otherwise fail-open
        degrades to fail-empty for every other key in it
        (the ``landed_outbox._load_raw`` precedent)."""
        path = tmp_path / 'counts.json'
        path.write_text(json.dumps({
            'a': {'value': 1, 'updated_at': '2026-01-01T00:00:00+00:00'},
            'b': 'not-a-record',
            'c': {'value': 1, 'updated_at': 'nonsense'},
        }), encoding='utf-8')

        guard = PersistentMap(path, ttl=timedelta(days=36500))
        assert sorted(guard) == ['a']
        assert guard['a'] == 1

    def test_unwritable_path_degrades_to_memory(self, tmp_path: Path, monkeypatch, caplog):
        """A disk failure must never crash a steward loop, a merge worker or
        a scheduler tick; it degrades to today's in-memory behaviour."""
        def boom(*_a, **_kw):
            raise OSError('disk full')

        monkeypatch.setattr(_safe_io, 'atomic_write_text', boom)

        guard = PersistentSet(tmp_path / 'capped.json', ttl=_TTL)
        with caplog.at_level(logging.WARNING):
            guard.add('esc-1')

        assert 'esc-1' in guard, 'in-memory state stays authoritative'
        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_two_owners_of_one_file_do_not_erase_each_other(self, tmp_path: Path):
        """A replacement owner is constructed while its predecessor is still
        alive — a redeployed ``TaskSteward`` for the same task overlaps the one
        it replaces — so a blind overwrite would silently re-arm the very guard
        this task makes durable.  Keys are globally unique (escalation ids,
        task ids), so merge-by-key is sound."""
        path = tmp_path / 'capped.json'
        owner_a = PersistentSet(path, ttl=_TTL)
        owner_b = PersistentSet(path, ttl=_TTL)

        owner_a.add('esc-a')
        owner_b.add('esc-b')

        witness = PersistentSet(path, ttl=_TTL)
        assert sorted(witness) == ['esc-a', 'esc-b']

        owner_a.discard('esc-a')
        after = PersistentSet(path, ttl=_TTL)
        assert sorted(after) == ['esc-b'], "a's removal must not take b's key"

    def test_delegates_to_the_shared_atomic_writer(self, tmp_path: Path, monkeypatch):
        """``tests/scripts/test_atomic_write_regrowth.py`` is an AST fence that
        fails the build on a new hand-rolled ``os.replace``/``os.rename``
        writer; this pins the delegation at the unit level too."""
        calls = _spy_on_writes(monkeypatch)

        PersistentSet(tmp_path / 'nested' / 'capped.json', ttl=_TTL).add('esc-1')

        assert len(calls) == 1, f'expected exactly one delegated write, got {calls}'
        kwargs = calls[0][2]
        assert kwargs.get('encoding') == 'utf-8'
        assert kwargs.get('mkdir') is True, 'the guards/ directory is created on demand'


# ---------------------------------------------------------------------------
# step-5 — the TTL that bounds the state
# ---------------------------------------------------------------------------

class TestGuardStateTtl:
    """A guard must not outlive its subject, and its file must not grow
    without bound.  Both follow from one TTL applied on read AND on write."""

    _WEEK = timedelta(days=7)

    def test_an_expired_set_member_is_indistinguishable_from_an_absent_one(
        self, tmp_path: Path,
    ):
        frozen, clock = _frozen_clock()
        guard = PersistentSet(tmp_path / 'capped.json', ttl=self._WEEK, now=clock)
        guard.add('k')

        frozen[0] = _T0 + self._WEEK + timedelta(seconds=1)

        assert 'k' not in guard
        assert len(guard) == 0
        assert list(guard) == []

    def test_an_expired_map_entry_is_indistinguishable_from_an_absent_one(
        self, tmp_path: Path,
    ):
        frozen, clock = _frozen_clock()
        guard = PersistentMap(tmp_path / 'counts.json', ttl=self._WEEK, now=clock)
        guard['k'] = 3

        frozen[0] = _T0 + self._WEEK + timedelta(seconds=1)

        assert 'k' not in guard
        assert guard.get('k', 0) == 0
        assert len(guard) == 0
        with pytest.raises(KeyError):
            guard['k']

    def test_the_boundary_is_exact(self, tmp_path: Path):
        """One second short of the TTL the guard is still in force.

        Pinned so a later refactor cannot quietly halve the window.
        """
        frozen, clock = _frozen_clock()
        guard = PersistentSet(tmp_path / 'capped.json', ttl=self._WEEK, now=clock)
        guard.add('k')

        frozen[0] = _T0 + self._WEEK - timedelta(seconds=1)

        assert 'k' in guard

    def test_expiry_is_pruned_from_disk_not_merely_hidden(self, tmp_path: Path):
        """Hiding alone would leave the file growing forever; the prune on
        flush is what bounds it."""
        path = tmp_path / 'capped.json'
        frozen, clock = _frozen_clock()
        guard = PersistentSet(path, ttl=self._WEEK, now=clock)
        guard.add('stale')

        frozen[0] = _T0 + self._WEEK + timedelta(seconds=1)
        guard.add('fresh')

        on_disk = json.loads(path.read_text(encoding='utf-8'))
        assert sorted(on_disk) == ['fresh']

    def test_expiry_is_pruned_on_load_too(self, tmp_path: Path):
        """A restart after a long quiet period must not re-import dead rows,
        even with no write to trigger a prune."""
        path = tmp_path / 'capped.json'
        frozen, clock = _frozen_clock()
        PersistentSet(path, ttl=self._WEEK, now=clock).add('stale')

        frozen[0] = _T0 + self._WEEK + timedelta(seconds=1)

        assert len(PersistentSet(path, ttl=self._WEEK, now=clock)) == 0

    def test_re_adding_a_live_member_neither_refreshes_nor_writes(
        self, tmp_path: Path, monkeypatch,
    ):
        """The scheduler re-asserts every non-pending task id on every ~15s
        tick.  A refreshing ``add`` would make the TTL measure "last seen"
        instead of "first observed" — so a task that stays terminal would
        never expire — and would rewrite the file ~5,760 times a day.
        """
        path = tmp_path / 'was_non_pending.json'
        frozen, clock = _frozen_clock()
        guard = PersistentSet(path, ttl=self._WEEK, now=clock)
        guard.add('k')

        calls = _spy_on_writes(monkeypatch)
        frozen[0] = _T0 + timedelta(days=1)
        guard.add('k')

        assert calls == [], 'a re-assertion of a live member must be silent'
        stored = json.loads(path.read_text(encoding='utf-8'))['k']['updated_at']
        assert datetime.fromisoformat(stored) == _T0, 'first observed, not last seen'

    def test_re_setting_a_map_entry_always_refreshes(self, tmp_path: Path):
        """The opposite rule, deliberately: a counter still being bumped has a
        subject that is demonstrably still alive, so its TTL tracks the bump.
        """
        path = tmp_path / 'counts.json'
        frozen, clock = _frozen_clock()
        guard = PersistentMap(path, ttl=self._WEEK, now=clock)
        guard['k'] = 1

        frozen[0] = _T0 + timedelta(days=1)
        guard['k'] += 1

        stored = json.loads(path.read_text(encoding='utf-8'))['k']['updated_at']
        assert datetime.fromisoformat(stored) == _T0 + timedelta(days=1)

    def test_the_file_is_unlinked_once_nothing_is_left(self, tmp_path: Path):
        """``data/orchestrator/guards/`` must not accumulate dead files, so an
        empty merged map releases the file rather than writing ``{}``."""
        path = tmp_path / 'capped.json'
        frozen, clock = _frozen_clock()
        guard = PersistentSet(path, ttl=self._WEEK, now=clock)
        guard.add('k')
        assert path.exists()

        frozen[0] = _T0 + self._WEEK + timedelta(seconds=1)
        guard.discard('k')
        assert not path.exists()

        guard.add('fresh')
        assert path.exists(), 'and a later mark recreates it'


# ---------------------------------------------------------------------------
# step-15 — a batch of N inserts is ONE write, not N+1
# ---------------------------------------------------------------------------

class TestGuardStateBatchedWrites:
    """Both batch entry points coalesce their writes into one.

    ``_GuardStore.flush`` re-reads, re-merges and re-serialises the WHOLE map,
    so a per-element flush costs a batch of N some N+1 writes and O(N^2)
    parse/serialise work.  Measured in this worktree before the fix: N=373
    performed 374 writes in 3.17s and N=1500 performed 1501 in 19.22s.  N=373
    is not hypothetical — it is today's count of active non-pending tasks, so
    it is what the first ``acquire_next`` tick after every fleet redeploy
    pays, inline on the event loop.

    ``|=`` is the form that matters: all three production batch call sites use
    it, not ``update`` —
    ``orchestrator/src/orchestrator/scheduler.py::Scheduler._update_age_anchors``,
    ``orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_stale_sweep``
    and
    ``orchestrator/src/orchestrator/merge_queue.py::SpeculativeMergeWorker._mark_coalesce_derailed``.
    """

    _N = 50

    @staticmethod
    def _keys(start: int, stop: int) -> list[str]:
        return [f'task-{i}' for i in range(start, stop)]

    def test_update_of_new_keys_performs_exactly_one_write(
        self, tmp_path: Path, monkeypatch,
    ):
        guard = PersistentSet(tmp_path / 'derailed.json', ttl=_TTL)

        calls = _spy_on_writes(monkeypatch)
        guard.update(self._keys(0, self._N))

        assert len(calls) == 1, f'expected one write for {self._N} keys, got {len(calls)}'

    def test_ior_performs_exactly_one_write(self, tmp_path: Path, monkeypatch):
        """The production form.  ``MutableSet.__ior__`` loops calling ``add``,
        so the override has to keep delegating to the batched path."""
        guard = PersistentSet(tmp_path / 'derailed.json', ttl=_TTL)

        calls = _spy_on_writes(monkeypatch)
        guard |= set(self._keys(0, self._N))

        assert len(calls) == 1, f'expected one write for {self._N} keys, got {len(calls)}'

    def test_a_wholly_redundant_batch_performs_no_write(
        self, tmp_path: Path, monkeypatch,
    ):
        """The steady-state property, at batch granularity: the scheduler
        re-asserts its whole observation set on every ~15s tick."""
        guard = PersistentSet(tmp_path / 'was_non_pending.json', ttl=_TTL)
        guard.update(self._keys(0, self._N))

        calls = _spy_on_writes(monkeypatch)
        guard |= set(self._keys(0, self._N))

        assert calls == [], 'a re-assertion of an unchanged batch must be silent'

    def test_a_mixed_batch_performs_one_write_and_keeps_every_key(
        self, tmp_path: Path, monkeypatch,
    ):
        """One new id among a tick's worth of already-known ones is the normal
        case, so it must cost one write rather than one per new id."""
        guard = PersistentSet(tmp_path / 'was_non_pending.json', ttl=_TTL)
        guard.update(self._keys(0, self._N))

        calls = _spy_on_writes(monkeypatch)
        guard |= set(self._keys(0, self._N + 5))

        assert len(calls) == 1, f'expected one write for a mixed batch, got {len(calls)}'
        assert sorted(guard) == sorted(self._keys(0, self._N + 5))

    def test_the_single_write_is_still_durable(self, tmp_path: Path, monkeypatch):
        """The assertion that stops the fix being "skip the writes" rather
        than "coalesce them"."""
        path = tmp_path / 'derailed.json'
        guard = PersistentSet(path, ttl=_TTL)

        calls = _spy_on_writes(monkeypatch)
        guard |= set(self._keys(0, self._N))

        assert len(calls) == 1
        assert sorted(PersistentSet(path, ttl=_TTL)) == sorted(self._keys(0, self._N))

    def test_an_aborted_batch_self_heals(self, tmp_path: Path):
        """An insert that never got its flush is not lost.

        ``flush`` writes the whole merged map rather than a delta, so ANY
        later write persists it.  The trailing flush in a batch therefore
        closes a crash-before-next-write window, not a correctness hole.
        """
        path = tmp_path / 'derailed.json'
        guard = PersistentSet(path, ttl=_TTL)

        def _explodes():
            yield 'task-a'
            yield 'task-b'
            raise RuntimeError('source of truth went away mid-batch')

        with pytest.raises(RuntimeError):
            guard.update(_explodes())

        assert sorted(guard) == ['task-a', 'task-b'], 'the partial batch is in memory'

        guard.add('task-c')

        assert sorted(PersistentSet(path, ttl=_TTL)) == ['task-a', 'task-b', 'task-c']
