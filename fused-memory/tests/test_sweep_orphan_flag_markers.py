"""Tests for scripts/sweep_orphan_flag_markers.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
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

    Defaults to the production shape written by
    ``flag_dedup._write_and_confirm_marker``: both ``kind`` and ``task_id``
    present. Pass ``kind=None`` / ``task_id=None`` to omit either key and
    isolate the corresponding orphan dimension.
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
    """Tests for async run(args, memory_service)."""

    def _args(self, apply: bool = False, project_id: str = 'dark_factory'):
        """Build a minimal args namespace."""
        import types as _types
        return _types.SimpleNamespace(apply=apply, project_id=project_id)

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

        report = await _mod.run(self._args(apply=False), memory_service)

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

        report = await _mod.run(self._args(apply=True), memory_service)

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

        report = await _mod.run(self._args(apply=True), memory_service)

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

        await _mod.run(self._args(apply=False), memory_service)

        memory_service.get_memories_by_metadata.assert_called_once()
        call_kwargs = memory_service.get_memories_by_metadata.call_args.kwargs
        assert call_kwargs.get('filters', {}).get('source') == 'stage1_flag_marker', (
            f'Expected source filter, got: {call_kwargs!r}'
        )
        memory_service.search.assert_not_called()
