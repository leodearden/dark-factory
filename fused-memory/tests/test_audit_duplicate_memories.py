"""Tests for audit_duplicate_memories.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_duplicate_memories.py'


def _load_module() -> types.ModuleType:
    """Load audit_duplicate_memories.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'audit_duplicate_memories'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
find_near_duplicate_memory_groups = _mod.find_near_duplicate_memory_groups
pick_survivor = _mod.pick_survivor
build_sweep_plan = _mod.build_sweep_plan
fetch_procedural_memories = _mod.fetch_procedural_memories
apply_deletions = _mod.apply_deletions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _memory(
    id: str,
    content: str,
    created_at: str | None = None,
    category: str = 'procedural_knowledge',
    metadata: dict | None = None,
) -> dict:
    return {
        'id': id,
        'content': content,
        'created_at': created_at,
        'category': category,
        'metadata': metadata or {},
    }


# Real-world fixture basis: the worktree-local-venv-vs-shared-checkout-venv
# gotcha, rewritten as a near-duplicate procedural_knowledge memory >=13 times
# by different task-worker agents (Stage-1 finding 2cf1b99f). The three
# variants below differ by only a few words each (SequenceMatcher ratio
# ~0.92-0.96 pairwise, verified with difflib before authoring this fixture)
# -- clearly above the 0.75 threshold used below. The distractor is on an
# unrelated topic (ratio ~0.17-0.18 against each variant) -- clearly below
# threshold. No assertion in this module pins those exact ratio values.
_VENV_GOTCHA_A = (
    "Each git worktree has its own local .venv; running uv sync inside a "
    "worktree creates a worktree-local venv that is separate from the shared "
    "checkout's venv at the main project root."
)
_VENV_GOTCHA_B = (
    "Each git worktree has its own local .venv directory; running uv sync "
    "inside a worktree creates a worktree-local venv that is separate from "
    "the shared checkout's venv in the main project root."
)
_VENV_GOTCHA_C = (
    "Each git worktree keeps its own local .venv; running uv sync inside a "
    "worktree creates a worktree-local venv separate from the shared "
    "checkout's venv at the main project root."
)
_DISTRACTOR = (
    "Reconciliation Stage 1 consolidates near-duplicate Mem0 "
    "procedural_knowledge memories during its nightly sweep cycle."
)

_THRESHOLD = 0.75


# ===========================================================================
# Step-1: find_near_duplicate_memory_groups
# ===========================================================================

class TestFindNearDuplicateMemoryGroupsClustering:
    """Near-identical rewrites of the same gotcha cluster; unrelated content does not."""

    def test_three_near_identical_rewrites_cluster_together(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
        ]
        result = find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD)
        assert len(result) == 1
        ids = {m['id'] for m in result[0]}
        assert ids == {'m1', 'm2', 'm3'}

    def test_unrelated_distractor_not_included(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m4', _DISTRACTOR, created_at='2026-07-13T02:00:00+00:00'),
        ]
        result = find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD)
        assert len(result) == 1
        all_ids = {m['id'] for g in result for m in g}
        assert 'm4' not in all_ids
        assert all_ids == {'m1', 'm2'}


class TestFindNearDuplicateMemoryGroupsDeterminism:
    """Deterministic ordering (groups by min id, members by id) + non-mutation."""

    def test_groups_sorted_by_min_id_members_sorted_by_id(self):
        # Deliberately out-of-order ids within the group.
        memories = [
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
        ]
        result = find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD)
        assert len(result) == 1
        assert [m['id'] for m in result[0]] == ['m1', 'm2', 'm3']

    def test_input_list_not_mutated(self):
        memories = [
            _memory('m3', _VENV_GOTCHA_C),
            _memory('m1', _VENV_GOTCHA_A),
            _memory('m2', _VENV_GOTCHA_B),
        ]
        original_order = [m['id'] for m in memories]
        find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD)
        assert [m['id'] for m in memories] == original_order

    def test_result_is_deterministic_regardless_of_input_order(self):
        import random  # noqa: PLC0415

        memories = [
            _memory('m1', _VENV_GOTCHA_A),
            _memory('m2', _VENV_GOTCHA_B),
            _memory('m3', _VENV_GOTCHA_C),
            _memory('m4', _DISTRACTOR),
        ]
        result_a = find_near_duplicate_memory_groups(list(memories), threshold=_THRESHOLD)
        shuffled = list(memories)
        random.shuffle(shuffled)
        result_b = find_near_duplicate_memory_groups(shuffled, threshold=_THRESHOLD)
        ids_a = sorted(sorted(m['id'] for m in g) for g in result_a)
        ids_b = sorted(sorted(m['id'] for m in g) for g in result_b)
        assert ids_a == ids_b


class TestFindNearDuplicateMemoryGroupsEmptyContent:
    """Empty/blank content never clusters (safe degradation), even though
    ``SequenceMatcher(None, '', '').ratio()`` returns 1.0."""

    def test_two_empty_content_records_do_not_cluster(self):
        memories = [
            _memory('m1', '', created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', '', created_at='2026-07-13T00:00:00+00:00'),
        ]
        assert find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD) == []

    def test_empty_content_does_not_join_real_cluster(self):
        # An unextractable-content record must not be swept up with genuine
        # near-duplicates.
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m3', '', created_at='2026-07-13T01:00:00+00:00'),
        ]
        result = find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD)
        assert len(result) == 1
        assert {m['id'] for m in result[0]} == {'m1', 'm2'}

    def test_blank_whitespace_content_treated_as_empty(self):
        memories = [
            _memory('m1', '   ', created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', '\t\n', created_at='2026-07-13T00:00:00+00:00'),
        ]
        assert find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD) == []


class TestFindNearDuplicateMemoryGroupsEdgeCases:
    """Degenerate inputs: empty / single-element / all-dissimilar lists produce no groups."""

    def test_empty_list_returns_empty(self):
        assert find_near_duplicate_memory_groups([], threshold=_THRESHOLD) == []

    def test_single_memory_returns_empty(self):
        memories = [_memory('m1', _VENV_GOTCHA_A)]
        assert find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD) == []

    def test_no_pair_above_threshold_returns_empty(self):
        memories = [_memory('m1', _VENV_GOTCHA_A), _memory('m4', _DISTRACTOR)]
        assert find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD) == []


# ===========================================================================
# Step-3: pick_survivor
# ===========================================================================

class TestPickSurvivorCanonicalFlag:
    """A metadata.canonical-flagged member always wins, regardless of age."""

    def test_canonical_wins_over_older_non_canonical(self):
        group = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-01T00:00:00+00:00'),
            _memory(
                'm2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00',
                metadata={'canonical': True},
            ),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'
        assert [m['id'] for m in losers] == ['m1']


class TestPickSurvivorOldestByCreatedAt:
    """With no canonical flag, the oldest by created_at wins."""

    def test_oldest_created_at_wins(self):
        group = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'
        assert {m['id'] for m in losers} == {'m1', 'm3'}


class TestPickSurvivorTieBreaks:
    """Ties (equal or absent created_at) are broken by the lowest id."""

    def test_equal_created_at_lowest_id_wins(self):
        group = [
            _memory('m9', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-12T00:00:00+00:00'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'
        assert [m['id'] for m in losers] == ['m9']

    def test_absent_created_at_both_none_lowest_id_wins(self):
        group = [
            _memory('m9', _VENV_GOTCHA_A, created_at=None),
            _memory('m2', _VENV_GOTCHA_B, created_at=None),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'


class TestPickSurvivorUnparseableCreatedAt:
    """None/unparseable created_at values sort last -- never picked as oldest
    unless every member in the group lacks a usable timestamp."""

    def test_unparseable_created_at_loses_to_parseable(self):
        group = [
            _memory('m1', _VENV_GOTCHA_A, created_at='not-a-date'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'

    def test_none_created_at_loses_to_parseable(self):
        group = [
            _memory('m1', _VENV_GOTCHA_A, created_at=None),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'

    def test_all_unparseable_falls_back_to_lowest_id(self):
        group = [
            _memory('m9', _VENV_GOTCHA_A, created_at='garbage'),
            _memory('m2', _VENV_GOTCHA_B, created_at=None),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor['id'] == 'm2'


class TestPickSurvivorEdgeCases:
    """Degenerate input: a group of < 2 memories is invalid."""

    def test_single_memory_raises_value_error(self):
        with pytest.raises(ValueError):
            pick_survivor([_memory('m1', _VENV_GOTCHA_A)])

    def test_empty_group_raises_value_error(self):
        with pytest.raises(ValueError):
            pick_survivor([])

    def test_losers_are_all_non_survivor_members(self):
        group = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-14T00:00:00+00:00'),
        ]
        survivor, losers = pick_survivor(group)
        assert survivor not in losers
        assert len(losers) == len(group) - 1
        assert {m['id'] for m in losers} | {survivor['id']} == {m['id'] for m in group}


# ===========================================================================
# Step-5: build_sweep_plan
# ===========================================================================

class TestBuildSweepPlanCategoryFiltering:
    """Only category=='procedural_knowledge' memories are considered."""

    def test_non_procedural_memories_excluded_from_clustering(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory(
                'm2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00',
                category='observations_and_summaries',
            ),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []


class TestBuildSweepPlanClusterReport:
    """A near-dup cluster reports its survivor and losers under delete_candidates."""

    def test_survivor_and_losers_reported(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 1
        group_report = plan['near_duplicate_groups'][0]
        assert group_report['survivor_id'] == 'm2'
        assert set(group_report['member_ids']) == {'m1', 'm2', 'm3'}
        assert set(plan['delete_candidates']) == {'m1', 'm3'}
        assert 'm2' not in plan['delete_candidates']


class TestBuildSweepPlanShape:
    """The plan always has the documented keys, correctly populated."""

    def test_plan_has_required_keys(self):
        plan = build_sweep_plan([], threshold=_THRESHOLD)
        required = {'clusters_total', 'near_duplicate_groups', 'delete_candidates'}
        assert required <= set(plan.keys())

    def test_group_report_has_survivor_content_and_members(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-12T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        group_report = plan['near_duplicate_groups'][0]
        assert group_report['survivor_id'] == 'm2'
        assert group_report['survivor_content'] == _VENV_GOTCHA_B
        assert set(group_report['member_ids']) == {'m1', 'm2'}


class TestBuildSweepPlanEmptyAndNoDuplicates:
    """No duplicates / empty input produces an empty plan."""

    def test_empty_input_produces_empty_plan(self):
        plan = build_sweep_plan([], threshold=_THRESHOLD)
        assert plan['clusters_total'] == 0
        assert plan['near_duplicate_groups'] == []
        assert plan['delete_candidates'] == []

    def test_no_duplicates_produces_empty_plan(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A),
            _memory('m4', _DISTRACTOR),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []


class TestBuildSweepPlanJsonSerializable:
    """The plan is always JSON-serializable (backs the CLI's dry-run report)."""

    def test_plan_is_json_serializable(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m4', _DISTRACTOR),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        serialized = json.dumps(plan, default=str)
        assert isinstance(serialized, str)


# ===========================================================================
# Step-7: fetch_procedural_memories
# ===========================================================================

def _raw_record(id: str, created_at: str | None, metadata: dict) -> dict:
    """Build a scroll_by_metadata()-shaped raw record (mirrors Mem0Backend's
    {'id', 'created_at', 'metadata'} return shape, where 'metadata' is the
    full Qdrant payload dict)."""
    return {'id': id, 'created_at': created_at, 'metadata': metadata}


@pytest.mark.asyncio
class TestFetchProceduralMemoriesCallShape:
    """scroll_by_metadata is called with the category filter and scan_limit."""

    async def test_called_with_category_filter_and_scan_limit(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=[])

        await fetch_procedural_memories(memory, 'dark_factory', scan_limit=1234)

        memory.mem0.scroll_by_metadata.assert_awaited_once()
        call = memory.mem0.scroll_by_metadata.call_args
        assert call.args[0].project_id == 'dark_factory'
        assert call.args[1] == {'category': 'procedural_knowledge'}
        assert call.kwargs.get('limit') == 1234


@pytest.mark.asyncio
class TestFetchProceduralMemoriesNormalization:
    """Raw records are normalized to {'id','content','created_at','metadata'}."""

    async def test_content_extracted_from_memory_key(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        raw = [_raw_record(
            'm1', '2026-07-12T00:00:00+00:00',
            {'memory': 'gotcha text', 'category': 'procedural_knowledge'},
        )]
        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=raw)

        result = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)

        assert result == [{
            'id': 'm1',
            'content': 'gotcha text',
            'category': 'procedural_knowledge',
            'created_at': '2026-07-12T00:00:00+00:00',
            'metadata': {'memory': 'gotcha text', 'category': 'procedural_knowledge'},
        }]

    async def test_content_extracted_from_data_key(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        raw = [_raw_record('m2', None, {'data': 'gotcha text v2'})]
        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=raw)

        result = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)

        assert result[0]['content'] == 'gotcha text v2'
        assert result[0]['id'] == 'm2'
        assert result[0]['created_at'] is None


@pytest.mark.asyncio
class TestFetchProceduralMemoriesSafeDegradation:
    """A record with no extractable content normalizes to content=''."""

    async def test_no_extractable_content_degrades_to_empty_string(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        raw = [_raw_record('m3', '2026-07-12T00:00:00+00:00', {'unrelated_field': 'x'})]
        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=raw)

        result = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)

        assert result[0]['content'] == ''


@pytest.mark.asyncio
class TestFetchProceduralMemoriesEmptyScan:
    """An empty/[] scan returns []."""

    async def test_empty_scan_returns_empty_list(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=[])

        result = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)

        assert result == []


# ===========================================================================
# Integration: fetch_procedural_memories output -> build_sweep_plan
# ===========================================================================

@pytest.mark.asyncio
class TestFetchToSweepPlanIntegration:
    """The real fetch normalization shape must flow through build_sweep_plan
    and actually cluster — guards against a category-key mismatch making the
    sweep a silent no-op in production."""

    async def test_fetched_records_cluster_and_report(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        # scroll_by_metadata-shaped raw records: category lives inside the
        # payload ('metadata'), NOT at the top level — the shape the real
        # Mem0 fetch pipeline produces.
        raw = [
            _raw_record(
                'm1', '2026-07-12T00:00:00+00:00',
                {'memory': _VENV_GOTCHA_A, 'category': 'procedural_knowledge'},
            ),
            _raw_record(
                'm2', '2026-07-13T00:00:00+00:00',
                {'memory': _VENV_GOTCHA_B, 'category': 'procedural_knowledge'},
            ),
            _raw_record(
                'm3', '2026-07-13T01:00:00+00:00',
                {'memory': _VENV_GOTCHA_C, 'category': 'procedural_knowledge'},
            ),
        ]
        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=raw)

        records = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)
        plan = build_sweep_plan(records, threshold=_THRESHOLD)

        assert plan['clusters_total'] == 1
        group_report = plan['near_duplicate_groups'][0]
        assert group_report['survivor_id'] == 'm1'  # oldest created_at
        assert set(group_report['member_ids']) == {'m1', 'm2', 'm3'}
        assert set(plan['delete_candidates']) == {'m2', 'm3'}

    async def test_fetched_unextractable_content_not_deleted(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        # Two records whose content cannot be extracted degrade to '' and must
        # not cluster with each other (or be marked for deletion).
        raw = [
            _raw_record(
                'm1', '2026-07-12T00:00:00+00:00',
                {'unrelated': 'x', 'category': 'procedural_knowledge'},
            ),
            _raw_record(
                'm2', '2026-07-13T00:00:00+00:00',
                {'unrelated': 'y', 'category': 'procedural_knowledge'},
            ),
        ]
        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=raw)

        records = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)
        plan = build_sweep_plan(records, threshold=_THRESHOLD)

        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []


# ===========================================================================
# Step-9: apply_deletions
# ===========================================================================

@pytest.mark.asyncio
class TestApplyDeletionsDryRun:
    """dry_run=True performs no delete_memory calls and returns zero counts."""

    async def test_dry_run_no_calls_zero_counts(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        plan = {'delete_candidates': ['m1', 'm2']}

        result = await apply_deletions(memory, 'dark_factory', plan, dry_run=True)

        memory.delete_memory.assert_not_awaited()
        assert result == {'deleted': 0, 'delete_errors': 0}


@pytest.mark.asyncio
class TestApplyDeletionsApply:
    """dry_run=False deletes each candidate once with the expected kwargs."""

    async def test_delete_memory_called_once_per_candidate(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        plan = {'delete_candidates': ['m1', 'm2', 'm3']}

        result = await apply_deletions(memory, 'dark_factory', plan, dry_run=False)

        assert memory.delete_memory.await_count == 3
        assert result == {'deleted': 3, 'delete_errors': 0}

    async def test_delete_memory_called_with_expected_kwargs(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.delete_memory = AsyncMock(return_value={'status': 'deleted'})
        plan = {'delete_candidates': ['m1']}

        await apply_deletions(memory, 'dark_factory', plan, dry_run=False)

        memory.delete_memory.assert_awaited_once_with(
            'm1', store='mem0', project_id='dark_factory',
        )


@pytest.mark.asyncio
class TestApplyDeletionsPartialFailure:
    """A raising delete_memory call does not abort the remaining deletes."""

    async def test_partial_failure_counted_others_proceed(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        async def _side_effect(memory_id, **kwargs):
            if memory_id == 'bad':
                raise RuntimeError('Qdrant error')
            return {'status': 'deleted'}

        memory = MagicMock()
        memory.delete_memory = AsyncMock(side_effect=_side_effect)
        plan = {'delete_candidates': ['good1', 'bad', 'good2']}

        result = await apply_deletions(memory, 'dark_factory', plan, dry_run=False)

        assert memory.delete_memory.await_count == 3
        assert result == {'deleted': 2, 'delete_errors': 1}
