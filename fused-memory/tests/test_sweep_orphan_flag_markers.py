"""Tests for scripts/sweep_orphan_flag_markers.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'sweep_orphan_flag_markers.py'


def _load_module() -> types.ModuleType:
    """Load sweep_orphan_flag_markers.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'sweep_orphan_flag_markers'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# Helpers
# ===========================================================================

def _member(
    id: str,
    kind: str | None = 'stage1_flag_marker',
    task_id: str | None = '1970',
) -> dict:
    """Build a scroll-shaped member dict.

    Defaults to the production shape written by the Mem0 mirror write in
    ``flag_dedup.dedup_flags`` (task 2227; formerly
    ``_write_and_confirm_marker``): both ``kind`` and ``task_id`` present.
    Pass ``kind=None`` / ``task_id=None`` to omit either key and isolate
    the corresponding orphan dimension.
    """
    metadata: dict = {'source': 'stage1_flag_marker'}
    if kind is not None:
        metadata['kind'] = kind
    if task_id is not None:
        metadata['task_id'] = task_id
    return {'id': id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def _orphan(id: str) -> dict:
    """Member that has source and task_id but no kind — the kind-orphan shape.

    Carries a task_id so it isolates the kind dimension only (does not also
    trip the taskless predicate).
    """
    return _member(id, kind=None)


def _wrong_kind(id: str) -> dict:
    """Member with source, task_id, and a mismatched kind value.

    Carries a task_id so it isolates the kind dimension only.
    """
    return _member(id, kind='something_else')


def _taskless(id: str) -> dict:
    """Member that has source+kind but NO task_id key — the 1e2b9417 shape.

    This is the orphan class task 2108 adds: a stage1_flag_marker with a
    valid kind that is nonetheless dead weight because it lacks a task_id
    (see find_taskless_markers).
    """
    return _member(id, kind='stage1_flag_marker', task_id=None)


def _bothmissing(id: str) -> dict:
    """Member lacking BOTH kind and task_id — the union-dedup edge case.

    Must be caught by both find_orphan_markers and find_taskless_markers,
    but deleted exactly once when run() unions the two by id.
    """
    return _member(id, kind=None, task_id=None)


# ===========================================================================
# Tests: find_orphan_markers
# ===========================================================================

class TestFindOrphanMarkers:
    """Tests for the pure function find_orphan_markers(members)."""

    def test_all_valid_returns_empty(self):
        """Members where all have kind=='stage1_flag_marker' produce an empty result."""
        members = [_member('m1'), _member('m2'), _member('m3')]
        result = _mod.find_orphan_markers(members)
        assert result == [], f'Expected [], got: {result!r}'

    def test_missing_kind_is_orphan(self):
        """A member whose metadata lacks the kind key is an orphan."""
        members = [_orphan('o1')]
        result = _mod.find_orphan_markers(members)
        assert len(result) == 1
        assert result[0]['id'] == 'o1', f'Expected id=o1, got: {result!r}'

    def test_wrong_kind_is_orphan(self):
        """A member whose metadata.kind != 'stage1_flag_marker' is an orphan."""
        members = [_wrong_kind('o2')]
        result = _mod.find_orphan_markers(members)
        assert len(result) == 1
        assert result[0]['id'] == 'o2', f'Expected id=o2, got: {result!r}'

    def test_mixed_returns_only_orphans(self):
        """Only members lacking kind=='stage1_flag_marker' are returned."""
        members = [
            _member('good1'),
            _orphan('bad1'),
            _member('good2'),
            _wrong_kind('bad2'),
            _member('good3'),
        ]
        result = _mod.find_orphan_markers(members)
        ids = [m['id'] for m in result]
        assert sorted(ids) == ['bad1', 'bad2'], f'Expected bad1 and bad2, got: {ids!r}'

    def test_empty_input_returns_empty(self):
        """Empty input list returns empty list."""
        result = _mod.find_orphan_markers([])
        assert result == []

    def test_preserves_member_identity(self):
        """Returned orphan dicts are the same objects (not copies)."""
        orphan = _orphan('o99')
        members = [_member('g1'), orphan]
        result = _mod.find_orphan_markers(members)
        assert len(result) == 1
        assert result[0] is orphan, 'Expected same object identity'

    def test_member_with_no_metadata_is_orphan(self):
        """A member with missing metadata dict entirely is treated as an orphan."""
        members = [{'id': 'no-meta', 'created_at': '2026-01-01', 'metadata': None}]
        result = _mod.find_orphan_markers(members)
        assert len(result) == 1
        assert result[0]['id'] == 'no-meta'


# ===========================================================================
# Tests: find_taskless_markers
# ===========================================================================

class TestFindTasklessMarkers:
    """Tests for the pure function find_taskless_markers(members) (task 2108).

    A marker is "taskless" when its metadata lacks a usable task_id — this
    is the shape of orphan marker 1e2b9417, which carried source+kind but no
    task_id and was therefore invisible to find_orphan_markers.
    """

    def test_missing_task_id_key_is_taskless(self):
        """A member whose metadata lacks the task_id key entirely is taskless."""
        member = _taskless('t1')
        result = _mod.find_taskless_markers([member])
        assert len(result) == 1
        assert result[0]['id'] == 't1', f'Expected id=t1, got: {result!r}'

    def test_task_id_none_is_taskless(self):
        """A member whose metadata.task_id is explicitly None is taskless."""
        member = {
            'id': 't2',
            'created_at': '2026-01-01T00:00:00Z',
            'metadata': {
                'source': 'stage1_flag_marker',
                'kind': 'stage1_flag_marker',
                'task_id': None,
            },
        }
        result = _mod.find_taskless_markers([member])
        assert len(result) == 1
        assert result[0]['id'] == 't2', f'Expected id=t2, got: {result!r}'

    def test_task_id_empty_string_is_taskless(self):
        """A member whose metadata.task_id is '' is taskless."""
        member = {
            'id': 't3',
            'created_at': '2026-01-01T00:00:00Z',
            'metadata': {
                'source': 'stage1_flag_marker',
                'kind': 'stage1_flag_marker',
                'task_id': '',
            },
        }
        result = _mod.find_taskless_markers([member])
        assert len(result) == 1
        assert result[0]['id'] == 't3', f'Expected id=t3, got: {result!r}'

    def test_real_task_id_is_not_taskless_even_if_kind_missing(self):
        """A member with a real task_id is NOT taskless, even lacking kind."""
        member = _orphan('o1')  # kind=None, task_id='1970' (default from _member)
        result = _mod.find_taskless_markers([member])
        assert result == [], f'Expected [], got: {result!r}'

    def test_fp_hash_task_id_is_not_taskless(self):
        """A member with an fp:-hash task_id carries a valid (non-taskless) task_id."""
        member = _member('f1', task_id='fp:9a8b7c6d5e4f')
        result = _mod.find_taskless_markers([member])
        assert result == [], f'Expected [], got: {result!r}'

    def test_member_with_no_metadata_is_taskless(self):
        """A member with missing metadata dict entirely is treated as taskless."""
        members = [{'id': 'no-meta', 'created_at': '2026-01-01', 'metadata': None}]
        result = _mod.find_taskless_markers(members)
        assert len(result) == 1
        assert result[0]['id'] == 'no-meta'

    def test_empty_input_returns_empty(self):
        """Empty input list returns empty list."""
        result = _mod.find_taskless_markers([])
        assert result == []

    def test_preserves_identity_and_order(self):
        """Returned dicts are the same objects, in input order."""
        valid1 = _member('v1')
        taskless_a = _taskless('ta')
        valid2 = _member('v2')
        taskless_b = _taskless('tb')
        members = [valid1, taskless_a, valid2, taskless_b]
        result = _mod.find_taskless_markers(members)
        assert result == [taskless_a, taskless_b], f'Expected [ta, tb], got: {result!r}'
        assert result[0] is taskless_a, 'Expected same object identity'
        assert result[1] is taskless_b, 'Expected same object identity'


# ===========================================================================
# Tests: classify_marker_task_id
# ===========================================================================

class TestClassifyMarkerTaskId:
    """Tests for the pure function classify_marker_task_id(tid) (task 2596).

    Buckets a marker's ``task_id`` into one of four shapes so the sweep can
    report per-bucket counts and route the age/terminal predicates. Splits
    on ',' and validates each sub-id individually — AMENDMENT 1: a naive
    whole-string ``isdigit()`` would mis-bucket a comma-joined tid like the
    live record a07972e7's ``'1944,2408'``.
    """

    @pytest.mark.parametrize('tid', ['2408', '2315'])
    def test_numeric(self, tid):
        """A single all-digit task_id string is classified 'numeric'."""
        assert _mod.classify_marker_task_id(tid) == 'numeric', (
            f'Expected numeric for {tid!r}'
        )

    def test_fp_hash(self):
        """A canonical 'fp:' + 32-hex-char task_id is classified 'fp_hash'."""
        tid = 'fp:' + 'a' * 32
        assert _mod.classify_marker_task_id(tid) == 'fp_hash', (
            f'Expected fp_hash for {tid!r}'
        )

    @pytest.mark.parametrize('tid', ['1944,2408', '2405, 540'])
    def test_comma_joined(self, tid):
        """A comma-joined multi-id string (with or without spaces around
        components) is classified 'comma_joined'."""
        assert _mod.classify_marker_task_id(tid) == 'comma_joined', (
            f'Expected comma_joined for {tid!r}'
        )

    @pytest.mark.parametrize('tid', [None, '', 'garbage', 'fp:bad', '12,x'])
    def test_null_or_invalid(self, tid):
        """None, empty string, non-numeric garbage, a malformed 'fp:'
        variant, and a comma-joined string with a non-digit component are
        all classified 'null_or_invalid'."""
        assert _mod.classify_marker_task_id(tid) == 'null_or_invalid', (
            f'Expected null_or_invalid for {tid!r}'
        )

    def test_splits_on_comma_never_naive_isdigit_on_whole_string(self):
        """Pin that comma-joined classification splits on ',' and validates
        each sub-id, rather than calling ``.isdigit()`` on the whole string
        (which is False for any comma-joined value and would otherwise risk
        mis-bucketing it). Live record a07972e7 carries task_id='1944,2408'.
        """
        assert '1944,2408'.isdigit() is False, 'sanity: whole-string isdigit is False'
        assert _mod.classify_marker_task_id('1944,2408') == 'comma_joined'


# ===========================================================================
# Tests: find_stale_markers
# ===========================================================================

class TestFindStaleMarkers:
    """Tests for the pure function find_stale_markers(members, now, max_age_days=14)
    (task 2596 — restores the age-drain semantics of the retired
    _sweep_stale_flag_markers, task 1944).
    """

    _NOW = datetime(2026, 7, 14, tzinfo=UTC)

    @staticmethod
    def _dated(id: str, created_at: str | None) -> dict:
        """Build a minimal member dict with an explicit (or missing) created_at."""
        member: dict = {'id': id, 'metadata': {'source': 'stage1_flag_marker'}}
        if created_at is not None:
            member['created_at'] = created_at
        return member

    def test_old_marker_is_stale(self):
        """A marker created well before the max_age_days cutoff is returned."""
        old = self._dated('old1', '2026-01-01T00:00:00+00:00')
        result = _mod.find_stale_markers([old], self._NOW, max_age_days=14)
        assert result == [old], f'Expected [old1], got: {result!r}'

    def test_fresh_marker_is_kept(self):
        """A marker created within the max_age_days window is NOT returned."""
        fresh = self._dated('fresh1', '2026-07-10T00:00:00+00:00')
        result = _mod.find_stale_markers([fresh], self._NOW, max_age_days=14)
        assert result == [], f'Expected [], got: {result!r}'

    def test_missing_created_at_is_kept_fail_safe(self):
        """A marker with no created_at key is KEPT (never returned) — fail-safe."""
        missing = self._dated('missing1', None)
        result = _mod.find_stale_markers([missing], self._NOW, max_age_days=14)
        assert result == [], f'Expected [] (kept), got: {result!r}'

    def test_none_created_at_is_kept_fail_safe(self):
        """A marker with created_at explicitly None is KEPT — fail-safe."""
        member = {'id': 'none1', 'created_at': None, 'metadata': {}}
        result = _mod.find_stale_markers([member], self._NOW, max_age_days=14)
        assert result == [], f'Expected [] (kept), got: {result!r}'

    def test_unparseable_created_at_is_kept_fail_safe(self):
        """A marker with an unparseable created_at string is KEPT — fail-safe."""
        member = {'id': 'bad1', 'created_at': 'not-a-date', 'metadata': {}}
        result = _mod.find_stale_markers([member], self._NOW, max_age_days=14)
        assert result == [], f'Expected [] (kept), got: {result!r}'

    def test_max_age_days_zero_drains_all_dated_members(self):
        """max_age_days=0 sets the cutoff to `now` itself, draining every
        dated member strictly older than `now` — including the marker that
        was fresh (kept) under the default 14-day window. The missing-
        created_at member is still kept regardless.
        """
        old = self._dated('old1', '2026-01-01T00:00:00+00:00')
        fresh = self._dated('fresh1', '2026-07-10T00:00:00+00:00')
        missing = self._dated('missing1', None)
        members = [old, fresh, missing]
        result = _mod.find_stale_markers(members, self._NOW, max_age_days=0)
        assert result == [old, fresh], f'Expected [old1, fresh1], got: {result!r}'

    def test_preserves_order_and_identity(self):
        """Returned dicts are the same objects, in input order."""
        old_a = self._dated('old_a', '2026-01-01T00:00:00+00:00')
        fresh = self._dated('fresh1', '2026-07-10T00:00:00+00:00')
        old_b = self._dated('old_b', '2026-02-01T00:00:00+00:00')
        members = [old_a, fresh, old_b]
        result = _mod.find_stale_markers(members, self._NOW, max_age_days=14)
        assert result == [old_a, old_b], f'Expected [old_a, old_b], got: {result!r}'
        assert result[0] is old_a, 'Expected same object identity'
        assert result[1] is old_b, 'Expected same object identity'

    def test_empty_input_returns_empty(self):
        """Empty input list returns empty list."""
        result = _mod.find_stale_markers([], self._NOW, max_age_days=14)
        assert result == []


# ===========================================================================
# Tests: find_undated_markers
# ===========================================================================

class TestFindUndatedMarkers:
    """Tests for the pure function find_undated_markers(members) (task 2596
    amendment, reviewer_comprehensive #1/#2): the diagnostic mirror of
    find_stale_markers' fail-safe KEEP conditions — surfaces exactly the
    members no --max-age-days value (including 0) can ever drain.
    """

    @staticmethod
    def _dated(id: str, created_at: str | None) -> dict:
        """Build a minimal member dict with an explicit (or missing) created_at."""
        member: dict = {'id': id, 'metadata': {'source': 'stage1_flag_marker'}}
        if created_at is not None:
            member['created_at'] = created_at
        return member

    def test_missing_created_at_key_is_undated(self):
        """A marker with no created_at key at all is undated."""
        missing = self._dated('missing1', None)
        result = _mod.find_undated_markers([missing])
        assert result == [missing], f'Expected [missing1], got: {result!r}'

    def test_none_created_at_is_undated(self):
        """A marker with created_at explicitly None is undated."""
        member = {'id': 'none1', 'created_at': None, 'metadata': {}}
        result = _mod.find_undated_markers([member])
        assert result == [member], f'Expected [none1], got: {result!r}'

    def test_unparseable_created_at_is_undated(self):
        """A marker with an unparseable created_at string is undated."""
        member = {'id': 'bad1', 'created_at': 'not-a-date', 'metadata': {}}
        result = _mod.find_undated_markers([member])
        assert result == [member], f'Expected [bad1], got: {result!r}'

    def test_valid_dated_marker_is_not_undated(self):
        """A marker with a well-formed, parseable created_at is not undated,
        regardless of how old or fresh it is."""
        old = self._dated('old1', '2026-01-01T00:00:00+00:00')
        fresh = self._dated('fresh1', '2026-07-10T00:00:00+00:00')
        result = _mod.find_undated_markers([old, fresh])
        assert result == [], f'Expected [], got: {result!r}'

    def test_mixed_returns_only_undated(self):
        """Only members with a missing/None/unparseable created_at are returned."""
        dated = self._dated('dated1', '2026-01-01T00:00:00+00:00')
        missing = self._dated('missing1', None)
        bad = {'id': 'bad1', 'created_at': 'garbage', 'metadata': {}}
        members = [dated, missing, bad]
        result = _mod.find_undated_markers(members)
        ids = [m['id'] for m in result]
        assert sorted(ids) == ['bad1', 'missing1'], f'Expected bad1/missing1, got: {ids!r}'

    def test_empty_input_returns_empty(self):
        """Empty input list returns empty list."""
        result = _mod.find_undated_markers([])
        assert result == []

    def test_preserves_order_and_identity(self):
        """Returned dicts are the same objects, in input order."""
        missing_a = self._dated('missing_a', None)
        dated = self._dated('dated1', '2026-01-01T00:00:00+00:00')
        missing_b = self._dated('missing_b', None)
        members = [missing_a, dated, missing_b]
        result = _mod.find_undated_markers(members)
        assert result == [missing_a, missing_b], f'Expected [missing_a, missing_b], got: {result!r}'
        assert result[0] is missing_a, 'Expected same object identity'
        assert result[1] is missing_b, 'Expected same object identity'


# ===========================================================================
# Tests: find_terminal_task_markers
# ===========================================================================

class TestFindTerminalTaskMarkers:
    """Tests for the pure function
    find_terminal_task_markers(members, terminal_task_ids) (task 2596 —
    restores the terminal-drain semantics of the retired
    _sweep_terminal_task_flag_markers, task 2103/2150).
    """

    _TERMINAL = {'2440', '1944', '12', '15'}

    def test_numeric_terminal_task_id_is_returned(self):
        """A numeric task_id present in terminal_task_ids is returned."""
        member = _member('m1', task_id='2440')
        result = _mod.find_terminal_task_markers([member], self._TERMINAL)
        assert result == [member], f'Expected [m1], got: {result!r}'

    def test_numeric_pending_task_id_is_kept(self):
        """A numeric task_id NOT in terminal_task_ids is kept (not returned)."""
        member = _member('m2', task_id='2408')
        result = _mod.find_terminal_task_markers([member], self._TERMINAL)
        assert result == [], f'Expected [], got: {result!r}'

    def test_comma_joined_requires_all_components_terminal(self):
        """A comma-joined task_id with one non-terminal component is kept."""
        member = _member('m3', task_id='1944,2408')  # 1944 done, 2408 pending
        result = _mod.find_terminal_task_markers([member], self._TERMINAL)
        assert result == [], f'Expected [] (2408 not terminal), got: {result!r}'

    def test_comma_joined_all_terminal_is_returned(self):
        """A comma-joined task_id whose every component is terminal is returned."""
        member = _member('m4', task_id='12,15')
        result = _mod.find_terminal_task_markers([member], self._TERMINAL)
        assert result == [member], f'Expected [m4], got: {result!r}'

    def test_fp_hash_task_id_is_never_returned(self):
        """An fp:-hash task_id is never matched, regardless of terminal_task_ids."""
        member = _member('m5', task_id='fp:' + 'a' * 32)
        result = _mod.find_terminal_task_markers([member], self._TERMINAL)
        assert result == [], f'Expected [], got: {result!r}'

    def test_null_or_invalid_task_id_is_never_returned(self):
        """A missing/None task_id is never matched."""
        member = _taskless('m6')
        result = _mod.find_terminal_task_markers([member], self._TERMINAL)
        assert result == [], f'Expected [], got: {result!r}'

    def test_empty_terminal_set_returns_empty(self):
        """An empty terminal_task_ids set matches nothing, even a would-be
        terminal id."""
        member = _member('m7', task_id='2440')
        result = _mod.find_terminal_task_markers([member], set())
        assert result == [], f'Expected [], got: {result!r}'

    def test_preserves_order_and_identity(self):
        """Returned dicts are the same objects, in input order."""
        term_a = _member('ta', task_id='2440')
        kept = _member('kp', task_id='2408')
        term_b = _member('tb', task_id='12,15')
        members = [term_a, kept, term_b]
        result = _mod.find_terminal_task_markers(members, self._TERMINAL)
        assert result == [term_a, term_b], f'Expected [ta, tb], got: {result!r}'
        assert result[0] is term_a, 'Expected same object identity'
        assert result[1] is term_b, 'Expected same object identity'

    def test_empty_input_returns_empty(self):
        """Empty input list returns empty list."""
        result = _mod.find_terminal_task_markers([], self._TERMINAL)
        assert result == []


# ===========================================================================
# Tests: delete_orphan_markers
# ===========================================================================

class TestDeleteOrphanMarkers:
    """Tests for async delete_orphan_markers(memory_service, project_id, orphans, ...)."""

    @pytest.mark.asyncio
    async def test_calls_delete_once_per_orphan(self):
        """delete_memory is called exactly once per orphan with correct kwargs."""
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(return_value=None)

        orphans = [_orphan('id-a'), _orphan('id-b')]
        result = await _mod.delete_orphan_markers(
            memory_service, 'dark_factory', orphans,
        )

        assert memory_service.delete_memory.call_count == 2
        calls = memory_service.delete_memory.call_args_list
        called_ids = {c.kwargs.get('memory_id') for c in calls}
        assert called_ids == {'id-a', 'id-b'}
        # All calls must use store='mem0' and _source='sweep_orphan_flag_markers'
        for c in calls:
            assert c.kwargs.get('store') == 'mem0', f'Expected store=mem0: {c}'
            assert c.kwargs.get('project_id') == 'dark_factory', f'Expected project_id: {c}'
            assert c.kwargs.get('_source') == 'sweep_orphan_flag_markers', (
                f'Expected _source=sweep_orphan_flag_markers: {c}'
            )
        assert result['deleted'] == 2

    @pytest.mark.asyncio
    async def test_best_effort_one_failure_does_not_abort_batch(self):
        """A single delete_memory failure does not abort remaining deletes or raise."""
        memory_service = AsyncMock()
        # id-a raises; id-b succeeds
        memory_service.delete_memory = AsyncMock(
            side_effect=[RuntimeError('Qdrant error'), None]
        )

        orphans = [_orphan('id-a'), _orphan('id-b')]
        # Must not raise
        result = await _mod.delete_orphan_markers(
            memory_service, 'dark_factory', orphans,
        )

        # Both were attempted
        assert memory_service.delete_memory.call_count == 2
        # Only the success is counted; the failure is in failed list
        assert result['deleted'] == 1
        assert 'id-a' in result['failed']

    @pytest.mark.asyncio
    async def test_empty_orphans_returns_zero_deleted(self):
        """No orphans → no delete calls, deleted=0."""
        memory_service = AsyncMock()
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await _mod.delete_orphan_markers(
            memory_service, 'dark_factory', [],
        )

        memory_service.delete_memory.assert_not_called()
        assert result['deleted'] == 0
        assert result['failed'] == []


# ===========================================================================
# Tests: run()
# ===========================================================================

class TestRun:
    """Tests for async run(args, memory_service, *, now=None, terminal_task_ids=None)."""

    # Matches _member()'s hardcoded default created_at ('2026-01-01T00:00:00Z').
    # Pre-existing tests below inject this as `now` so the new age-based stale
    # sweep (task 2596) never fires for their fixtures regardless of the real
    # wall-clock date the suite happens to run on — cutoff (now - 14 days)
    # falls well before every fixture's created_at.
    _NEUTRAL_NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def _args(
        self,
        apply: bool = False,
        project_id: str = 'dark_factory',
        max_age_days: int = 14,
    ):
        """Build a minimal args namespace."""
        import types as _types
        return _types.SimpleNamespace(
            apply=apply, project_id=project_id, max_age_days=max_age_days,
        )

    @pytest.mark.asyncio
    async def test_dry_run_does_not_call_delete(self):
        """Dry-run (args.apply=False): delete_memory is NOT called.
        Report includes before counts, identified orphan ids/count, and dry_run=True.
        """
        memory_service = AsyncMock()
        # count_memories_by_metadata: total=3, total+kind=1  (so 2 orphans by count)
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[3, 1, 0])
        # get_memories_by_metadata returns 1 valid + 2 orphans
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[
            _member('v1'),
            _orphan('o1'),
            _orphan('o2'),
        ])
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
        )

        # delete must NOT be called in dry-run mode
        memory_service.delete_memory.assert_not_called()
        assert report['dry_run'] is True
        assert set(report['orphan_ids']) == {'o1', 'o2'}
        assert report['orphan_count'] == 2
        assert 'before' in report

    @pytest.mark.asyncio
    async def test_apply_calls_delete_and_recounts(self):
        """Apply (args.apply=True): delete_memory called once per orphan;
        report includes before AND after counts, dry_run=False.
        """
        memory_service = AsyncMock()
        # before: total=3, kind=1 → 2 orphans by count
        # after: total=1, kind=1 → 0 orphans
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[3, 1, 0, 1, 1]
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[
            _member('v1'),
            _orphan('o1'),
            _orphan('o2'),
        ])
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=True), memory_service, now=self._NEUTRAL_NOW,
        )

        assert memory_service.delete_memory.call_count == 2
        assert report['dry_run'] is False
        assert 'after' in report
        assert report.get('deleted') == 2

    @pytest.mark.asyncio
    async def test_apply_partial_failure_populates_failed_list(self):
        """Apply with one delete failure: report['failed'] is populated,
        report['deleted'] reflects only successes, and after-counts are re-fetched
        (reflecting residual orphans that were not deleted).
        """
        memory_service = AsyncMock()
        # before: total=3, kind=1 → 2 orphans by count; after: total=2, kind=1 → 1 residual
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[3, 1, 0, 2, 1]
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[
            _member('v1'),
            _orphan('o1'),   # this delete will fail
            _orphan('o2'),   # this delete succeeds
        ])
        # id-o1's delete raises; id-o2's delete succeeds
        memory_service.delete_memory = AsyncMock(
            side_effect=[RuntimeError('Qdrant timeout'), None]
        )

        report = await _mod.run(
            self._args(apply=True), memory_service, now=self._NEUTRAL_NOW,
        )

        assert report['dry_run'] is False
        # Both orphans were found
        assert set(report['orphan_ids']) == {'o1', 'o2'}
        assert report['orphan_count'] == 2
        # Only one deletion succeeded
        assert report['deleted'] == 1
        # The failed id is reported
        assert 'o1' in report['failed']
        # After-counts are present (second count pass was done)
        assert 'after' in report
        assert report['after']['total_source'] == 2
        assert report['after']['total_with_kind'] == 1

    @pytest.mark.asyncio
    async def test_enumeration_uses_get_memories_by_metadata_not_search(self):
        """get_memories_by_metadata is called with filters={'source':'stage1_flag_marker'};
        semantic search (memory_service.search) is never called.
        """
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(return_value=0)
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.search = AsyncMock()

        await _mod.run(
            self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
        )

        memory_service.get_memories_by_metadata.assert_called_once()
        call_kwargs = memory_service.get_memories_by_metadata.call_args.kwargs
        assert call_kwargs.get('filters', {}).get('source') == 'stage1_flag_marker', (
            f'Expected source filter, got: {call_kwargs!r}'
        )
        memory_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_sweeps_union_of_kind_orphans_and_taskless_markers(self):
        """run() sweeps the id-deduplicated union of find_orphan_markers and
        find_taskless_markers (task 2108).

        members: a fully valid marker (v1), a taskless marker with a valid
        kind (t1 — the 1e2b9417 shape), a kind-orphan that carries a task_id
        (o1), and a marker missing BOTH kind and task_id (b1). b1 is caught
        by both predicates but must be deleted exactly once.
        """
        members = [
            _member('v1'),
            _taskless('t1'),
            _orphan('o1'),
            _bothmissing('b1'),
        ]

        # --- Dry run: verify enumeration/report shape ---
        dry_service = AsyncMock()
        dry_service.count_memories_by_metadata = AsyncMock(side_effect=[4, 2, 0])
        dry_service.get_memories_by_metadata = AsyncMock(return_value=members)
        dry_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False), dry_service, now=self._NEUTRAL_NOW,
        )

        dry_service.delete_memory.assert_not_called()
        assert set(report['orphan_ids']) == {'t1', 'o1', 'b1'}, (
            f"Expected {{'t1', 'o1', 'b1'}}, got: {report['orphan_ids']!r}"
        )
        assert report['orphan_count'] == 3, f"Expected 3, got: {report['orphan_count']!r}"
        assert report['taskless_orphan_count'] == 2, (
            f"Expected 2 (t1 and b1), got: {report['taskless_orphan_count']!r}"
        )

        # --- Apply: verify delete fan-out is deduped by id ---
        apply_service = AsyncMock()
        apply_service.count_memories_by_metadata = AsyncMock(side_effect=[4, 2, 0, 1, 1])
        apply_service.get_memories_by_metadata = AsyncMock(return_value=members)
        apply_service.delete_memory = AsyncMock(return_value=None)

        apply_report = await _mod.run(
            self._args(apply=True), apply_service, now=self._NEUTRAL_NOW,
        )

        assert apply_service.delete_memory.call_count == 3, (
            'Expected exactly 3 deletes (b1 deduped, not double-deleted): '
            f'{apply_service.delete_memory.call_args_list!r}'
        )
        deleted_ids = {
            c.kwargs.get('memory_id') for c in apply_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'t1', 'o1', 'b1'}, f'Expected no v1, got: {deleted_ids!r}'
        assert apply_report['taskless_orphan_count'] == 2

    @pytest.mark.asyncio
    async def test_run_sweeps_stale_and_terminal_members_too(self):
        """run() additionally sweeps the age-stale and terminal-task-referenced
        predicates (task 2596), unioned with the existing kind-orphan/taskless
        predicates and deduped by id. A member matched by BOTH the stale and
        terminal predicates (overlap1) is deleted exactly once. Report gains
        per-bucket counts (classify_marker_task_id) over the final union.
        """
        now = datetime(2026, 7, 14, tzinfo=UTC)

        valid_fresh = _member('v1', task_id='1970')
        valid_fresh['created_at'] = '2026-07-10T00:00:00Z'
        orphan = _orphan('o1')
        orphan['created_at'] = '2026-07-10T00:00:00Z'
        stale_only = _member('stale1', task_id='9001')
        stale_only['created_at'] = '2026-01-01T00:00:00Z'
        terminal_only = _member('term1', task_id='2440')
        terminal_only['created_at'] = '2026-07-10T00:00:00Z'
        overlap = _member('overlap1', task_id='2440')
        overlap['created_at'] = '2026-01-01T00:00:00Z'

        members = [valid_fresh, orphan, stale_only, terminal_only, overlap]
        expected_bucket_counts = {
            'numeric': 4, 'fp_hash': 0, 'comma_joined': 0, 'null_or_invalid': 0,
        }

        # --- Dry run ---
        dry_service = AsyncMock()
        dry_service.count_memories_by_metadata = AsyncMock(side_effect=[5, 3, 0])
        dry_service.get_memories_by_metadata = AsyncMock(return_value=members)
        dry_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False, max_age_days=14),
            dry_service,
            now=now,
            terminal_task_ids={'2440'},
        )

        dry_service.delete_memory.assert_not_called()
        assert set(report['orphan_ids']) == {'o1', 'stale1', 'term1', 'overlap1'}, (
            f'Expected o1/stale1/term1/overlap1 swept, got: {report["orphan_ids"]!r}'
        )
        assert report['orphan_count'] == 4, f"Expected 4, got: {report['orphan_count']!r}"
        assert report['bucket_counts'] == expected_bucket_counts, (
            f"Expected {expected_bucket_counts!r}, got: {report['bucket_counts']!r}"
        )

        # --- Apply: overlap1 (stale AND terminal) deleted exactly once ---
        apply_service = AsyncMock()
        apply_service.count_memories_by_metadata = AsyncMock(side_effect=[5, 3, 0, 1, 1])
        apply_service.get_memories_by_metadata = AsyncMock(return_value=members)
        apply_service.delete_memory = AsyncMock(return_value=None)

        apply_report = await _mod.run(
            self._args(apply=True, max_age_days=14),
            apply_service,
            now=now,
            terminal_task_ids={'2440'},
        )

        assert apply_service.delete_memory.call_count == 4, (
            'Expected exactly 4 deletes (overlap1 deduped, not double-deleted): '
            f'{apply_service.delete_memory.call_args_list!r}'
        )
        deleted_ids = {
            c.kwargs.get('memory_id') for c in apply_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'o1', 'stale1', 'term1', 'overlap1'}, (
            f'Expected no v1, got: {deleted_ids!r}'
        )
        assert apply_report['bucket_counts'] == expected_bucket_counts
        assert 'after' in apply_report
        assert apply_report['after']['total_source'] == 1

    @pytest.mark.asyncio
    async def test_undated_members_are_kept_and_reported_with_warning(self, caplog):
        """Members with a missing/unparseable created_at are never swept (no
        predicate matches them) but are counted in undated_kept_count and
        trigger a WARNING log so operators can see why the backlog floors
        above zero (task 2596 amendment, reviewer_comprehensive #1/#2)."""
        dated_stale = _member('stale1', task_id='9001')
        dated_stale['created_at'] = '2026-01-01T00:00:00Z'
        undated_missing = {
            'id': 'undated1',
            'metadata': {'source': 'stage1_flag_marker', 'kind': 'stage1_flag_marker',
                          'task_id': '9002'},
        }
        undated_bad = {
            'id': 'undated2',
            'created_at': 'not-a-date',
            'metadata': {'source': 'stage1_flag_marker', 'kind': 'stage1_flag_marker',
                          'task_id': '9003'},
        }
        members = [dated_stale, undated_missing, undated_bad]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[3, 3, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            report = await _mod.run(
                self._args(apply=False, max_age_days=14),
                memory_service,
                now=datetime(2026, 7, 14, tzinfo=UTC),
            )

        # Undated members are never in the delete set (no predicate matches them).
        assert set(report['orphan_ids']) == {'stale1'}
        assert report['undated_kept_count'] == 2
        assert any(
            'missing/unparseable created_at' in record.message
            and '2 of 3' in record.message  # count of undated / total enumerated
            and record.levelno == logging.WARNING
            for record in caplog.records
        ), f'Expected a WARNING mentioning the undated count, got: {[r.message for r in caplog.records]}'

    @pytest.mark.asyncio
    async def test_no_undated_members_yields_zero_count_and_no_warning(self, caplog):
        """All-dated members: undated_kept_count is 0 and no WARNING is logged."""
        valid_fresh = _member('v1', task_id='1970')
        valid_fresh['created_at'] = '2026-07-10T00:00:00Z'
        members = [valid_fresh]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 1, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            report = await _mod.run(
                self._args(apply=False),
                memory_service,
                now=self._NEUTRAL_NOW,
            )

        assert report['undated_kept_count'] == 0
        assert not any(
            'missing/unparseable created_at' in record.message for record in caplog.records
        )

    # -----------------------------------------------------------------
    # cross_check census block (task 3897)
    #
    # run() issues a THIRD count_memories_by_metadata call — the
    # flag_for_stage2 census probe — between the two `before` counts and
    # the scroll. Mock ordering for these tests is therefore:
    #     [before_source, before_kind, flag_for_stage2_probe, (after_source,
    #      after_kind on --apply)]
    # -----------------------------------------------------------------

    _BLIND_SPOT_MSG = 'flag_for_stage2'

    def _blind_spot_warnings(self, caplog) -> list:
        """Return the WARNING records that are the blind-spot warning."""
        return [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and r.name == 'sweep_orphan_flag_markers'
            and self._BLIND_SPOT_MSG in r.message
        ]

    @pytest.mark.asyncio
    async def test_cross_check_reports_blind_spot_on_live_dark_factory_shape(self, caplog):
        """The live 2026-08-09 dark_factory shape: the source enumeration
        matches 0 while 61 flag_for_stage2 records exist -> blind_spot True.

        This is the exact false all-clear task 3897 exists to make loud:
        before.total_source == 0 makes backlog_verdict hold unconditionally,
        so without this block the run reads as a clean bill of health.
        """
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[0, 0, 61])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            report = await _mod.run(
                self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
            )

        assert report['cross_check'] == {
            'source_total': 0,
            'flag_for_stage2_total': 61,
            'blind_spot': True,
            'probe_failed': False,
        }

        # The probe must use the boolean-typed adjacent filter and the same
        # project as the sweep — a probe against a different project would
        # compare two unrelated populations.
        probe_call = memory_service.count_memories_by_metadata.call_args_list[2]
        assert probe_call.kwargs['filters'] == {'flag_for_stage2': True}
        assert probe_call.kwargs['project_id'] == 'dark_factory'

        warnings = self._blind_spot_warnings(caplog)
        assert warnings, (
            'Expected a blind-spot WARNING, got: '
            f'{[r.message for r in caplog.records]}'
        )
        # The warning must name BOTH counts, or an operator cannot tell how
        # large the unseen population is.
        text = warnings[0].getMessage()
        assert '0' in text and '61' in text

    @pytest.mark.asyncio
    async def test_cross_check_no_blind_spot_when_enumeration_sees_its_pool(self, caplog):
        """A non-zero source enumeration is not a blind spot, even when an
        adjacent flag_for_stage2 population also exists.

        Both pools being non-empty is the healthy steady state; warning here
        would make the signal noise.
        """
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[3, 3, 61])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[
            _member('v1'), _member('v2'), _member('v3'),
        ])
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            report = await _mod.run(
                self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
            )

        assert report['cross_check'] == {
            'source_total': 3,
            'flag_for_stage2_total': 61,
            'blind_spot': False,
            'probe_failed': False,
        }
        assert not self._blind_spot_warnings(caplog)

    @pytest.mark.asyncio
    async def test_cross_check_true_no_op_is_not_a_blind_spot(self, caplog):
        """Nothing enumerated AND nothing adjacent is a genuine no-op.

        This is the case that must stay silent, or every clean run warns and
        operators learn to ignore the warning.
        """
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[0, 0, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            report = await _mod.run(
                self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
            )

        assert report['cross_check'] == {
            'source_total': 0,
            'flag_for_stage2_total': 0,
            'blind_spot': False,
            'probe_failed': False,
        }
        assert not self._blind_spot_warnings(caplog)

    # -----------------------------------------------------------------
    # cross_check probe is fail-safe (task 3897)
    #
    # The census probe is a DIAGNOSTIC. It must never abort the sweep, never
    # alter the delete set, and never assert a blind spot it did not
    # actually observe. Mirrors the posture of
    # task_knowledge_sync._warn_on_flag_for_stage2_type_drift.
    # -----------------------------------------------------------------

    def _members_for_failsafe(self) -> list[dict]:
        """Two orphans + one valid member — a delete set the probe must not perturb."""
        return [_member('v1'), _orphan('o1'), _orphan('o2')]

    @pytest.mark.asyncio
    async def test_probe_failure_does_not_raise_and_marks_probe_failed(self, caplog):
        """A raising census probe degrades to 'unknown', never a crash.

        blind_spot must be False, not True: an unobserved population is not
        an observed blind spot, and a diagnostic that invents findings when
        it fails is worse than one that stays silent.
        """
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[3, 1, RuntimeError('qdrant down')]
        )
        memory_service.get_memories_by_metadata = AsyncMock(
            return_value=self._members_for_failsafe()
        )
        memory_service.delete_memory = AsyncMock(return_value=None)

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            report = await _mod.run(
                self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
            )

        assert report['cross_check'] == {
            'source_total': 3,
            'flag_for_stage2_total': None,
            'blind_spot': False,
            'probe_failed': True,
        }

        # A failed probe must carry a stack trace, or a genuine wiring bug is
        # indistinguishable in the journal from a transient backend blip.
        failed_probe_records = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and r.exc_info is not None
        ]
        assert failed_probe_records, (
            'Expected a WARNING with exc_info for the failed probe, got: '
            f'{[(r.message, r.exc_info) for r in caplog.records]}'
        )
        assert 'RuntimeError' in caplog.text and 'qdrant down' in caplog.text

    @pytest.mark.asyncio
    async def test_probe_failure_leaves_delete_set_identical(self):
        """The delete set is byte-identical with and without a probe failure.

        This is the contract that makes the probe safe to add to a script
        whose --apply path performs irreversible deletes.
        """
        async def _run(probe_outcome):
            memory_service = AsyncMock()
            memory_service.count_memories_by_metadata = AsyncMock(
                side_effect=[3, 1, probe_outcome]
            )
            memory_service.get_memories_by_metadata = AsyncMock(
                return_value=self._members_for_failsafe()
            )
            memory_service.delete_memory = AsyncMock(return_value=None)
            return await _mod.run(
                self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
            )

        healthy = await _run(61)
        broken = await _run(RuntimeError('qdrant down'))

        assert healthy['orphan_ids'] == broken['orphan_ids'] == ['o1', 'o2']
        assert healthy['orphan_count'] == broken['orphan_count'] == 2
        # Everything except the cross_check block itself is untouched.
        assert (
            {k: v for k, v in healthy.items() if k != 'cross_check'}
            == {k: v for k, v in broken.items() if k != 'cross_check'}
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad_return', [None, object(), '61', 3.5])
    async def test_non_int_probe_return_is_treated_as_a_failed_probe(self, bad_return):
        """An unexpected return shape degrades to 'unknown' rather than
        crashing on the `> 0` comparison inside enumeration_blind_spot.

        Note '61' (a str) and 3.5 (a float) are included deliberately: both
        would either raise or silently misbehave under a naive comparison.
        """
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[0, 0, bad_return]
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
        )

        assert report['cross_check'] == {
            'source_total': 0,
            'flag_for_stage2_total': None,
            'blind_spot': False,
            'probe_failed': True,
        }

    @pytest.mark.asyncio
    async def test_bool_probe_return_is_treated_as_a_failed_probe(self):
        """`True` is an int subclass in Python, so an isinstance(x, int)
        guard alone would let a boolean through and report
        flag_for_stage2_total: True — a nonsense census. Guard against it
        explicitly."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[0, 0, True]
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False), memory_service, now=self._NEUTRAL_NOW,
        )

        assert report['cross_check']['probe_failed'] is True
        assert report['cross_check']['flag_for_stage2_total'] is None
        assert report['cross_check']['blind_spot'] is False


# ===========================================================================
# Tests: targeted correction (--delete-ids)
# ===========================================================================

class TestTargetedCorrection:
    """Tests for the targeted --delete-ids correction pass in run() (task 2596).

    A targeted correction force-deletes specific marker UUIDs regardless of
    age/terminal/orphan status — the deterministic lever for correcting known-
    mistagged records that the other predicates cannot catch: the mistagged
    eb92453f-shaped record (numeric task_id '2408', which is PENDING so the
    terminal predicate can't catch it) and the composite a07972e7-shaped
    record (task_id='1944,2408', kept because not ALL components are
    terminal). Delete (not delete+recreate) is the honest correction — see
    design_decisions.

    count_memories_by_metadata mock ordering (task 3897): the THIRD call is
    run()'s flag_for_stage2 census probe, sitting between the two 'before'
    counts and any 'after' counts. These fixtures pass 0 for it (no adjacent
    population), which keeps blind_spot False and their semantics unchanged.
    """

    _NOW = datetime(2026, 7, 14, tzinfo=UTC)
    # 1944 and 2440 are done (terminal); 2408 is pending (not terminal) —
    # matches the task's verified get_statuses premises.
    _TERMINAL = {'1944', '2440'}

    def _args(
        self,
        apply: bool = False,
        project_id: str = 'dark_factory',
        max_age_days: int = 14,
        delete_ids: list[str] | None = None,
    ):
        """Build a minimal args namespace, with delete_ids support."""
        import types as _types
        return _types.SimpleNamespace(
            apply=apply, project_id=project_id, max_age_days=max_age_days,
            delete_ids=delete_ids,
        )

    @staticmethod
    def _fresh(id: str, task_id: str) -> dict:
        """A valid-kind, fresh (not stale), non-taskless member — isolates
        the targeted-correction path from every other predicate."""
        member = _member(id, task_id=task_id)
        member['created_at'] = '2026-07-10T00:00:00Z'
        return member

    def test_mistagged_and_composite_records_not_otherwise_swept(self):
        """Sanity: neither shape trips any of the pre-existing predicates —
        this is what makes them require the targeted correction lever."""
        mistagged = self._fresh('eb92453f', task_id='2408')
        composite = self._fresh('a07972e7', task_id='1944,2408')
        members = [mistagged, composite]

        assert _mod.find_orphan_markers(members) == []
        assert _mod.find_taskless_markers(members) == []
        assert _mod.find_stale_markers(members, self._NOW, max_age_days=14) == []
        assert _mod.find_terminal_task_markers(members, self._TERMINAL) == []

    @pytest.mark.asyncio
    async def test_delete_ids_force_includes_mistagged_and_composite_records(self):
        """run() with args.delete_ids naming both records sweeps them even
        though no other predicate would, and reports them under
        targeted_correction_ids."""
        mistagged = self._fresh('eb92453f', task_id='2408')
        composite = self._fresh('a07972e7', task_id='1944,2408')
        members = [mistagged, composite]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[2, 2, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False, delete_ids=['eb92453f', 'a07972e7']),
            memory_service,
            now=self._NOW,
            terminal_task_ids=self._TERMINAL,
        )

        assert set(report['orphan_ids']) == {'eb92453f', 'a07972e7'}, (
            f"Expected both targeted ids swept, got: {report['orphan_ids']!r}"
        )
        assert report['orphan_count'] == 2
        assert set(report['targeted_correction_ids']) == {'eb92453f', 'a07972e7'}

    @pytest.mark.asyncio
    async def test_delete_ids_actually_deletes_on_apply(self):
        """With args.apply=True, both targeted records are actually deleted
        via delete_memory."""
        mistagged = self._fresh('eb92453f', task_id='2408')
        composite = self._fresh('a07972e7', task_id='1944,2408')
        members = [mistagged, composite]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[2, 2, 0, 0, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=True, delete_ids=['eb92453f', 'a07972e7']),
            memory_service,
            now=self._NOW,
            terminal_task_ids=self._TERMINAL,
        )

        assert memory_service.delete_memory.call_count == 2
        deleted_ids = {
            c.kwargs.get('memory_id') for c in memory_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'eb92453f', 'a07972e7'}
        assert report['after']['total_source'] == 0

    @pytest.mark.asyncio
    async def test_delete_ids_entry_not_present_is_ignored(self):
        """A delete_ids UUID that doesn't match any enumerated member is
        silently ignored — no crash, no phantom entry in the report."""
        valid = self._fresh('v1', task_id='1970')
        members = [valid]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 1, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False, delete_ids=['does-not-exist']),
            memory_service,
            now=self._NOW,
        )

        assert report['orphan_ids'] == []
        assert report['orphan_count'] == 0
        assert report['targeted_correction_ids'] == []

    @pytest.mark.asyncio
    async def test_delete_ids_deduped_with_other_predicates(self):
        """A delete_ids entry that's also caught by another predicate (e.g.
        kind-orphan) is deleted exactly once, and is still reported under
        targeted_correction_ids (found-intersection semantics, not
        exclusive-to-targeted)."""
        orphan = _orphan('o1')
        orphan['created_at'] = '2026-07-10T00:00:00Z'
        members = [orphan]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 0, 0, 0, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=True, delete_ids=['o1']),
            memory_service,
            now=self._NOW,
        )

        assert memory_service.delete_memory.call_count == 1
        assert report['targeted_correction_ids'] == ['o1']

    @pytest.mark.asyncio
    async def test_no_delete_ids_yields_empty_targeted_correction_ids(self):
        """When args.delete_ids is None (default/absent), targeted_correction_ids
        is [] and behavior is unchanged from before this feature existed."""
        valid = self._fresh('v1', task_id='1970')
        members = [valid]

        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 1, 0])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False),  # delete_ids defaults to None
            memory_service,
            now=self._NOW,
        )

        assert report['targeted_correction_ids'] == []


# ===========================================================================
# Tests: the flag_for_stage2 pool is CENSUSED, never deleted (task 3897)
# ===========================================================================

class TestFlagForStage2IsNeverDeleted:
    """The cross-check must never widen the delete set. Guard against a
    future well-meaning edit that turns the census into a deletion.

    Why this boundary is load-bearing, from live measurement (2026-08-09):

    1. 23 of the 61 live flag_for_stage2 records carry NO usable task_id, so
       this script's existing find_taskless_markers predicate would delete
       all 23 on the very next nightly --apply run. Those are LIVE Stage-1 ->
       Stage-2 relay markers, not dead weight. The nightly timer runs
       --apply --terminal-drain, which would additionally reap markers citing
       already-done tasks.
    2. This script's delete_orphan_markers has NEITHER the
       is_protected_mirror_record guard NOR the record_mem0_deletion_tombstones
       write that the shared in-cycle _sweep_stale_mem0_pool applies.
       flag_for_stage2 is an LLM-supplied key any writer can stamp on any
       record — mem0_tombstone.py's module docstring names this exact filter
       as its motivating over-breadth case.
    3. The pool is already drained correctly, on a rolling 14-day window, by
       the in-cycle _sweep_stale_mem0_flag_for_stage2_markers (task 2966).

    So the probe is count-only by design, and this class is what stops that
    from silently regressing.
    """

    _NEUTRAL_NOW = datetime(2026, 1, 1, tzinfo=UTC)

    def _args(self):
        import types as _types
        return _types.SimpleNamespace(
            apply=True, project_id='dark_factory', max_age_days=14,
        )

    @pytest.mark.asyncio
    async def test_census_never_widens_the_delete_set(self):
        """--apply + --terminal-drain + a 61-record adjacent population:
        only scrolled, source-filtered members are ever deleted."""
        # The scroll returns the source-filtered population only. 'keep1' is
        # dated, non-terminal and has a task_id, so no predicate catches it.
        members = [
            _member('keep1', task_id='1970'),
            _member('term1', task_id='2440'),   # terminal -> deleted
            _taskless('taskless1'),             # no task_id -> deleted
        ]
        # Ids that live in the adjacent flag_for_stage2 pool. They are never
        # scrolled, so they can only reach the delete set via a regression.
        forbidden_ids = {'ffs2-a', 'ffs2-b', 'ffs2-c'}

        memory_service = AsyncMock()
        # [before_source, before_kind, flag_for_stage2 probe, after_source, after_kind]
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[3, 3, 61, 1, 1]
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(),
            memory_service,
            now=self._NEUTRAL_NOW,
            terminal_task_ids={'2440'},
        )

        # The blind spot IS reported — the census did its job.
        assert report['cross_check']['flag_for_stage2_total'] == 61
        assert report['cross_check']['blind_spot'] is False  # source_total is 3

        # (a) Every delete targets an id drawn from the scrolled members.
        scrolled_ids = {m['id'] for m in members}
        deleted_ids = {
            call.kwargs['memory_id']
            for call in memory_service.delete_memory.call_args_list
        }
        assert deleted_ids == {'term1', 'taskless1'}
        assert deleted_ids <= scrolled_ids

        # (b) Nothing from the adjacent pool is touched, by id or by filter.
        assert not (deleted_ids & forbidden_ids)
        for call in memory_service.delete_memory.call_args_list:
            assert 'filters' not in call.kwargs, (
                'delete_memory must be called per-id, never with a bulk '
                f'filter: {call.kwargs}'
            )
            assert 'flag_for_stage2' not in repr(call.kwargs)

        # (c) The enumeration was NOT widened: exactly one scroll, on source.
        memory_service.get_memories_by_metadata.assert_called_once()
        scroll_kwargs = memory_service.get_memories_by_metadata.call_args.kwargs
        assert scroll_kwargs['filters'] == {'source': _mod.MARKER_SOURCE}
        assert 'flag_for_stage2' not in scroll_kwargs['filters']

    @pytest.mark.asyncio
    async def test_probe_only_ever_counts_never_scrolls(self):
        """The adjacent filter appears in count calls only — never in a
        get_memories_by_metadata scroll, which is the only way records could
        become delete-set candidates."""
        memory_service = AsyncMock()
        memory_service.count_memories_by_metadata = AsyncMock(
            side_effect=[0, 0, 61, 0, 0]
        )
        memory_service.get_memories_by_metadata = AsyncMock(return_value=[])
        memory_service.delete_memory = AsyncMock(return_value=None)

        await _mod.run(
            self._args(), memory_service, now=self._NEUTRAL_NOW,
        )

        scroll_filters = [
            call.kwargs.get('filters')
            for call in memory_service.get_memories_by_metadata.call_args_list
        ]
        assert all(
            'flag_for_stage2' not in (f or {}) for f in scroll_filters
        ), f'Adjacent filter leaked into a scroll: {scroll_filters}'

        count_filters = [
            call.kwargs.get('filters')
            for call in memory_service.count_memories_by_metadata.call_args_list
        ]
        assert {'flag_for_stage2': True} in count_filters
        memory_service.delete_memory.assert_not_called()


# ===========================================================================
# Tests: enumeration_blind_spot (task 3897)
# ===========================================================================

class TestEnumerationBlindSpot:
    """Tests for the pure predicate enumeration_blind_spot(enumerated, adjacent).

    Task 3897: this script enumerates on {'source': 'stage1_flag_marker'},
    which measures 0 records in BOTH dark_factory and reify as of
    2026-08-09, while the adjacent {'flag_for_stage2': True} relay pool
    measures 61 and 80 respectively. The consequence is a structural false
    all-clear: `before.total_source` is always 0, so backlog_verdict(0, N)
    holds unconditionally and forever, and the nightly sweep prints
    `orphan_count: 0` every night against a pool it cannot see.

    This predicate is what makes that divergence nameable. It distinguishes
    "swept nothing because there was nothing" (a true no-op) from "swept
    nothing because the enumeration filter cannot see the population" (a
    blind spot).
    """

    @pytest.mark.parametrize('enumerated,adjacent,expected', [
        # The live dark_factory shape as measured 2026-08-09: the source
        # filter sees nothing while 61 flag_for_stage2 records exist.
        (0, 61, True),
        # The live reify shape, same defect, different magnitude.
        (0, 80, True),
        # Genuinely nothing anywhere — a true no-op, NOT a blind spot. This
        # is the case that must not warn, or the warning becomes noise on
        # every clean run.
        (0, 0, False),
        # The sweep can see its own pool. An adjacent population merely
        # existing is not a blind spot — the two pools are distinct
        # populations and both being non-empty is the healthy steady state.
        (3, 61, False),
        (3, 0, False),
    ])
    def test_blind_spot_predicate(self, enumerated, adjacent, expected):
        """Blind spot iff the enumeration saw nothing AND an adjacent
        population is non-empty."""
        assert _mod.enumeration_blind_spot(enumerated, adjacent) is expected

    def test_flag_for_stage2_filters_constant_shape(self):
        """The census filter mirrors task_knowledge_sync._FLAG_FOR_STAGE2_ENUM_FILTERS."""
        assert _mod.FLAG_FOR_STAGE2_FILTERS == {'flag_for_stage2': True}

    def test_flag_for_stage2_filter_value_is_boolean_not_string(self):
        """The boolean True is load-bearing: Qdrant payload filters are
        type-sensitive, and the string variant {'flag_for_stage2': 'true'}
        measures 0 against live dark_factory (independently re-confirmed
        2026-08-09). A silent str/bool drift here would reintroduce exactly
        the zero-matching blind spot this cross-check exists to detect —
        `== {'flag_for_stage2': True}` alone would NOT catch it, since
        `True == 1` and Python's dict equality would also accept 1.
        """
        value = _mod.FLAG_FOR_STAGE2_FILTERS['flag_for_stage2']
        assert isinstance(value, bool)
        assert value is True


# ===========================================================================
# Tests: backlog_verdict
# ===========================================================================

class TestBacklogVerdict:
    """Tests for the pure function backlog_verdict(after_total_source, max_backlog)
    (task 2596). Mirrors scripts/check_merge_flakiness.sh's exit-code-only
    predicate contract (0=holds, 1=violated) so the sweep is directly usable
    as a task_kind='deterministic' before_done.script predicate
    (--apply --check --max-backlog N; the orchestrator reads the exit code
    only).
    """

    @pytest.mark.parametrize('after_total_source,max_backlog', [
        (0, 0),
        (5, 10),
        (10, 10),
    ])
    def test_at_or_under_max_backlog_returns_0(self, after_total_source, max_backlog):
        """Residual backlog at or below the ceiling holds → 0."""
        assert _mod.backlog_verdict(after_total_source, max_backlog) == 0

    @pytest.mark.parametrize('after_total_source,max_backlog', [
        (11, 10),
        (1, 0),
        (100, 43),
    ])
    def test_over_max_backlog_returns_1(self, after_total_source, max_backlog):
        """Residual backlog above the ceiling is violated → 1."""
        assert _mod.backlog_verdict(after_total_source, max_backlog) == 1


# ===========================================================================
# Tests: _resolve_check_exit_code (task 2596 amendment, reviewer_comprehensive #1)
# ===========================================================================

class TestResolveCheckExitCode:
    """Tests for _resolve_check_exit_code(report, max_backlog) — the report
    -> exit-code wiring main() delegates to for --check.

    Extracted from main() (task 2596 amendment, reviewer_comprehensive #1)
    so this resolution — pick after.total_source when an 'after' key is
    present (an --apply run), else fall back to before.total_source (a
    dry-run/--check-only invocation) — is unit-testable without any live
    I/O. Previously this branch was reachable only by running main() itself,
    so a regression (e.g. a KeyError from a reshaped report, or resolving
    against the wrong count on a dry-run) had no test coverage even though
    the sweep is meant to drive a deterministic before_done predicate, where
    a wrong exit code silently mis-gates a deployment.
    """

    def test_dry_run_report_holds_uses_before_total_source(self):
        """A report with no 'after' key (dry-run) resolves against
        before.total_source; at-or-under max_backlog holds -> 0."""
        report = {'before': {'total_source': 5, 'total_with_kind': 5}}
        assert _mod._resolve_check_exit_code(report, max_backlog=5) == 0

    def test_dry_run_report_violated_uses_before_total_source(self):
        """A report with no 'after' key (dry-run) resolves against
        before.total_source; over max_backlog is violated -> 1."""
        report = {'before': {'total_source': 6, 'total_with_kind': 6}}
        assert _mod._resolve_check_exit_code(report, max_backlog=5) == 1

    def test_apply_report_holds_uses_after_not_before_total_source(self):
        """A report WITH an 'after' key (--apply) must resolve against
        after.total_source, not before.total_source — before is still over
        budget here, but the post-delete after count holds."""
        report = {
            'before': {'total_source': 50, 'total_with_kind': 50},
            'after': {'total_source': 0, 'total_with_kind': 0},
        }
        assert _mod._resolve_check_exit_code(report, max_backlog=0) == 0

    def test_apply_report_violated_uses_after_not_before_total_source(self):
        """A report WITH an 'after' key (--apply) must resolve against
        after.total_source even when it is still violated post-delete."""
        report = {
            'before': {'total_source': 50, 'total_with_kind': 50},
            'after': {'total_source': 3, 'total_with_kind': 3},
        }
        assert _mod._resolve_check_exit_code(report, max_backlog=0) == 1


# ===========================================================================
# Tests: _build_parser (task 2596 CLI surface)
# ===========================================================================

class TestBuildParser:
    """Tests for _build_parser() — argparse factored out of main() so the
    new CLI surface (--max-age-days, --delete-ids, --check, --max-backlog,
    --terminal-drain) is testable without any live I/O (mirrors
    check_merge_flakiness.sh's exit-code-only contract; see main()'s
    docstring for how these flags are consumed).
    """

    def test_defaults(self):
        """With no CLI args, every new flag has a sane, backwards-compatible
        default and every pre-existing flag is unaffected."""
        parser = _mod._build_parser()
        args = parser.parse_args([])
        assert args.apply is False
        assert args.project_id == 'dark_factory'
        assert args.limit == 1000
        assert args.max_age_days == 14
        assert args.delete_ids == []
        assert args.check is False
        assert args.max_backlog == 0
        assert args.terminal_drain is False

    def test_max_age_days_override(self):
        """--max-age-days accepts an int override."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--max-age-days', '30'])
        assert args.max_age_days == 30

    def test_max_age_days_zero_is_accepted(self):
        """0 is the documented, explicit 'drain everything' lever and must
        be accepted (task 2596 amendment, reviewer_comprehensive #2)."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--max-age-days', '0'])
        assert args.max_age_days == 0

    def test_max_age_days_negative_is_rejected_at_parse_time(self):
        """A negative --max-age-days sets find_stale_markers's cutoff to a
        FUTURE instant, which would silently drain every dated member (not
        just stale ones) — the same effect as 0 but reached by what reads
        as a typo. This must be rejected at parse time, not accepted and
        silently misbehave (task 2596 amendment, reviewer_comprehensive #2)."""
        parser = _mod._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--max-age-days', '-1'])

    def test_delete_ids_comma_split(self):
        """--delete-ids splits a comma-joined string into a list of ids."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--delete-ids', 'eb92453f,a07972e7'])
        assert args.delete_ids == ['eb92453f', 'a07972e7']

    def test_delete_ids_strips_whitespace_around_components(self):
        """--delete-ids tolerates whitespace around comma-separated ids."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--delete-ids', 'eb92453f, a07972e7 '])
        assert args.delete_ids == ['eb92453f', 'a07972e7']

    def test_check_and_max_backlog(self):
        """--check is a boolean flag; --max-backlog accepts an int."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--check', '--max-backlog', '5'])
        assert args.check is True
        assert args.max_backlog == 5

    def test_max_backlog_zero_is_accepted(self):
        """0 is the documented default ceiling and must be accepted."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--max-backlog', '0'])
        assert args.max_backlog == 0

    def test_max_backlog_negative_is_rejected_at_parse_time(self):
        """A negative --max-backlog makes backlog_verdict() report a
        violation for ANY residual count (even 0), forever, with no
        explanation — the same silent-misbehavior-on-typo class already
        rejected for --max-age-days. This must be rejected at parse time,
        not accepted and silently misbehave (task 2596 amendment,
        reviewer_comprehensive #1)."""
        parser = _mod._build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(['--max-backlog', '-1'])

    def test_terminal_drain_flag(self):
        """--terminal-drain is a boolean flag, defaulting to False."""
        parser = _mod._build_parser()
        args = parser.parse_args(['--terminal-drain'])
        assert args.terminal_drain is True


# ===========================================================================
# Tests: _resolve_terminal_task_ids (task 2596 amendment, reviewer_comprehensive #3)
# ===========================================================================

class TestResolveTerminalTaskIds:
    """Tests for the async _resolve_terminal_task_ids() fail-safe resolver.

    A genuine wiring failure (wrong attr, backend import error) must still
    fail safe to an empty set — --terminal-drain degrades to an age-only
    sweep rather than crashing — but the WARNING must now carry exc_info
    so a real mis-wiring is distinguishable in logs from the unconfigured-
    taskmaster no-op, which returns early and never logs at all.
    """

    @pytest.mark.asyncio
    async def test_unconfigured_taskmaster_returns_empty_set_without_warning(
        self, monkeypatch, caplog
    ):
        """config.taskmaster is None (no taskmaster configured): returns
        set() with no WARNING logged — the clean, expected no-op path."""
        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            lambda: types.SimpleNamespace(taskmaster=None),
        )

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            result = await _mod._resolve_terminal_task_ids()

        assert result == set()
        assert not any(
            record.name == 'sweep_orphan_flag_markers' for record in caplog.records
        ), f'Expected no WARNING logs, got: {[r.message for r in caplog.records]}'

    @pytest.mark.asyncio
    async def test_backend_wiring_failure_fails_safe_with_exc_info(
        self, monkeypatch, caplog
    ):
        """A stubbed backend that raises on construction (simulating a real
        mis-wiring, e.g. a bad attribute or import error) must not propagate
        — _resolve_terminal_task_ids returns set() — but the WARNING it logs
        must carry exc_info=True so the failure is distinguishable from the
        unconfigured-taskmaster no-op above."""

        class _BoomBackend:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError('boom: backend wiring is broken')

        monkeypatch.setattr(
            'fused_memory.config.schema.FusedMemoryConfig',
            lambda: types.SimpleNamespace(taskmaster=object()),
        )
        monkeypatch.setattr(
            'fused_memory.backends.sqlite_task_backend.SqliteTaskBackend',
            _BoomBackend,
        )

        with caplog.at_level(logging.WARNING, logger='sweep_orphan_flag_markers'):
            result = await _mod._resolve_terminal_task_ids()

        assert result == set()
        matching = [
            record for record in caplog.records
            if record.name == 'sweep_orphan_flag_markers' and record.levelno == logging.WARNING
        ]
        assert matching, f'Expected a WARNING log, got: {[r.message for r in caplog.records]}'
        assert any(record.exc_info for record in matching), (
            f'Expected exc_info attached to the WARNING so the failure is '
            f'distinguishable from the unconfigured no-op, got: '
            f'{[(r.message, r.exc_info) for r in matching]}'
        )
        assert 'RuntimeError' in caplog.text and 'boom' in caplog.text, (
            f'Expected the traceback text in caplog output, got: {caplog.text!r}'
        )
