"""Tests for audit_duplicate_memories.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import importlib.util
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
