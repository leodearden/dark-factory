"""Tests for scripts/sweep_orphan_flag_markers.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import importlib.util
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
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[3, 1])
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
            side_effect=[3, 1, 1, 1]
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
            side_effect=[3, 1, 2, 1]
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
        dry_service.count_memories_by_metadata = AsyncMock(side_effect=[4, 2])
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
        apply_service.count_memories_by_metadata = AsyncMock(side_effect=[4, 2, 1, 1])
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
        dry_service.count_memories_by_metadata = AsyncMock(side_effect=[5, 3])
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
        apply_service.count_memories_by_metadata = AsyncMock(side_effect=[5, 3, 1, 1])
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
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[2, 2])
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
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[2, 2, 0, 0])
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
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 1])
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
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 0, 0, 0])
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
        memory_service.count_memories_by_metadata = AsyncMock(side_effect=[1, 1])
        memory_service.get_memories_by_metadata = AsyncMock(return_value=members)
        memory_service.delete_memory = AsyncMock(return_value=None)

        report = await _mod.run(
            self._args(apply=False),  # delete_ids defaults to None
            memory_service,
            now=self._NOW,
        )

        assert report['targeted_correction_ids'] == []
