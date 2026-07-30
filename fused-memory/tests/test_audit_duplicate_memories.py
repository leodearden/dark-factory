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
cluster_memories_by_pairs = _mod.cluster_memories_by_pairs
ann_pairs_from_neighbors = _mod.ann_pairs_from_neighbors
_ANN_DISCLOSURE_KEYS = _mod._ANN_DISCLOSURE_KEYS
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


class TestFindNearDuplicateMemoryGroupsProductionThreshold:
    """Every other test in this module pins threshold=_THRESHOLD (0.75), but
    the production default (function signature + --threshold CLI arg) is
    0.85. Guard the value the sweep actually runs with, so a real-world
    duplicate rewrite drifting below 0.85 would be caught here first instead
    of silently clustering nothing in production."""

    def test_venv_gotcha_variants_cluster_at_production_default_threshold(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
        ]
        result = find_near_duplicate_memory_groups(memories, threshold=0.85)
        assert len(result) == 1
        assert {m['id'] for m in result[0]} == {'m1', 'm2', 'm3'}

    def test_build_sweep_plan_clusters_at_production_default_threshold(self):
        # build_sweep_plan(memories) with no explicit threshold arg exercises
        # the exact default the CLI and callers get when they don't override it.
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
        ]
        plan = build_sweep_plan(memories)
        assert plan['clusters_total'] == 1
        assert set(plan['near_duplicate_groups'][0]['member_ids']) == {'m1', 'm2', 'm3'}


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
# cluster_memories_by_pairs — the shared transitive-closure core (task 3210)
# ===========================================================================

class TestClusterMemoriesByPairs:
    """The union-find closure, exercised directly on explicit index pairs.

    Task 3210 adds a SECOND candidate generator (ANN neighbours from Qdrant)
    alongside the existing O(n^2) lexical difflib scan.  Rather than let each
    generator carry its own clustering, the transitive-closure core is
    extracted here so both feed the SAME closure — one implementation, two
    producers.  Two closures that must agree is exactly the failure mode this
    avoids.

    These tests pin the closure independently of ANY similarity metric: they
    hand it index pairs directly, so a change to lexical thresholds or ANN
    cutoffs can never mask a clustering regression.
    """

    def test_transitive_closure_across_pairs(self):
        """(0,1) and (1,2) must collapse into ONE group of three, not two pairs."""
        memories = [_memory(f'm{i}', f'content {i}') for i in range(3)]
        result = cluster_memories_by_pairs(memories, [(0, 1), (1, 2)])
        assert len(result) == 1
        assert [m['id'] for m in result[0]] == ['m0', 'm1', 'm2']

    def test_disjoint_pairs_stay_separate_groups(self):
        memories = [_memory(f'm{i}', f'content {i}') for i in range(4)]
        result = cluster_memories_by_pairs(memories, [(0, 1), (2, 3)])
        assert len(result) == 2
        assert [[m['id'] for m in g] for g in result] == [['m0', 'm1'], ['m2', 'm3']]

    def test_singletons_are_dropped(self):
        """A memory in no pair is not a cluster and must not appear in the output."""
        memories = [_memory(f'm{i}', f'content {i}') for i in range(3)]
        result = cluster_memories_by_pairs(memories, [(0, 1)])
        assert len(result) == 1
        assert [m['id'] for m in result[0]] == ['m0', 'm1']
        assert 'm2' not in {m['id'] for g in result for m in g}

    def test_deterministic_ordering_groups_by_min_id_members_by_id(self):
        """Members sorted by str(id); groups sorted by their minimum str(id).

        Same contract as _sort_groups_deterministically, asserted through the
        public function so the ordering cannot silently change.
        """
        memories = [
            _memory('m9', 'nine'), _memory('m3', 'three'),
            _memory('m1', 'one'), _memory('m5', 'five'),
        ]
        # Pair the out-of-order indices: (m9,m3) and (m1,m5).
        result = cluster_memories_by_pairs(memories, [(0, 1), (2, 3)])
        assert [[m['id'] for m in g] for g in result] == [['m1', 'm5'], ['m3', 'm9']]

    def test_duplicate_and_mirrored_pairs_are_idempotent(self):
        """Repeating a pair, or supplying both (i,j) and (j,i), changes nothing."""
        memories = [_memory(f'm{i}', f'content {i}') for i in range(3)]
        once = cluster_memories_by_pairs(memories, [(0, 1), (1, 2)])
        twice = cluster_memories_by_pairs(memories, [(0, 1), (1, 0), (1, 2), (0, 1), (2, 1)])
        assert [[m['id'] for m in g] for g in twice] == [[m['id'] for m in g] for g in once]

    def test_input_list_not_mutated(self):
        memories = [_memory(f'm{i}', f'content {i}') for i in range(3)]
        snapshot = [dict(m) for m in memories]
        order_before = [id(m) for m in memories]
        cluster_memories_by_pairs(memories, [(0, 1), (1, 2)])
        assert memories == snapshot
        assert [id(m) for m in memories] == order_before, 'input list order was mutated'

    def test_empty_memories_returns_empty(self):
        assert cluster_memories_by_pairs([], []) == []

    def test_empty_pairs_returns_empty(self):
        """No pairs means no clusters — every record is a singleton."""
        memories = [_memory(f'm{i}', f'content {i}') for i in range(3)]
        assert cluster_memories_by_pairs(memories, []) == []

    def test_accepts_any_iterable_of_pairs(self):
        """Pairs arrive from a generator in the ANN path, not only from a list."""
        memories = [_memory(f'm{i}', f'content {i}') for i in range(3)]
        result = cluster_memories_by_pairs(memories, ((i, i + 1) for i in range(2)))
        assert len(result) == 1
        assert [m['id'] for m in result[0]] == ['m0', 'm1', 'm2']

    def test_returned_members_are_the_original_dict_objects(self):
        """Groups carry the caller's dicts through by identity, not copies.

        build_sweep_plan hands these straight to pick_survivor, which reads
        metadata.canonical and created_at off them — a copy would still work
        but would quietly double memory on a 5000-record scan.
        """
        memories = [_memory(f'm{i}', f'content {i}') for i in range(2)]
        result = cluster_memories_by_pairs(memories, [(0, 1)])
        assert result[0][0] is memories[0]
        assert result[0][1] is memories[1]


class TestFindNearDuplicateMemoryGroupsBehaviourPreserved:
    """The lexical path must be byte-identical after delegating to the shared core.

    Task 3210 refactors find_near_duplicate_memory_groups to yield index pairs
    into cluster_memories_by_pairs instead of running its own inline union-find.
    That is a pure refactor: the clustering it produces on the real venv-gotcha
    fixtures must not move.  (The pre-existing clustering/determinism/empty-content
    classes above are the primary guard and stay untouched; this class states the
    equivalence explicitly so the intent survives a future reading.)
    """

    def test_lexical_path_matches_manual_closure_over_the_same_pairs(self):
        """Running the closure manually over the lexical pairs reproduces the result."""
        import difflib

        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-12T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-13T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-07-13T01:00:00+00:00'),
            _memory('m4', _DISTRACTOR, created_at='2026-07-13T02:00:00+00:00'),
        ]
        # Independently recompute which pairs clear the threshold, using the
        # same normalisation the production path applies.
        normalized = [(m.get('content') or '').strip().lower() for m in memories]
        expected_pairs = [
            (i, j)
            for i in range(len(memories))
            for j in range(i + 1, len(memories))
            if normalized[i] and normalized[j]
            and difflib.SequenceMatcher(None, normalized[i], normalized[j]).ratio() >= _THRESHOLD
        ]
        expected = cluster_memories_by_pairs(memories, expected_pairs)
        actual = find_near_duplicate_memory_groups(memories, threshold=_THRESHOLD)
        assert [[m['id'] for m in g] for g in actual] == [[m['id'] for m in g] for g in expected]
        # And concretely: the three rewrites cluster, the distractor does not.
        assert [[m['id'] for m in g] for g in actual] == [['m1', 'm2', 'm3']]

    def test_empty_content_guard_survives_the_extraction(self):
        """Two blank-content records must still not cluster.

        SequenceMatcher(None, '', '').ratio() is 1.0, so without the guard the
        lexical path would union two records whose content merely failed to
        extract and mark one for deletion.  The guard lives in the pair
        GENERATOR, so moving the closure out must not lose it.
        """
        memories = [_memory('m1', ''), _memory('m2', '   '), _memory('m3', _VENV_GOTCHA_A)]
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

    async def test_data_key_wins_over_memory_key_when_both_present(self):
        # 'data' is the canonical scroll_by_metadata()/Qdrant payload key
        # (mirrors prune_recon_cycle_summaries.py, which reads only
        # metadata.get('data')); 'memory' is a search-result-layer key
        # (MemoryResult.content) that can appear stale on a scroll payload.
        # When both are present on the same payload, 'data' must win.
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        raw = [_raw_record(
            'm5', '2026-07-12T00:00:00+00:00',
            {
                'data': 'canonical scroll-payload text',
                'memory': 'stale search-layer text',
                'category': 'procedural_knowledge',
            },
        )]
        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = AsyncMock(return_value=raw)

        result = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)

        assert result[0]['content'] == 'canonical scroll-payload text'


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


# ===========================================================================
# ann_pairs_from_neighbors — ANN candidate generation + cap disclosure (3210)
# ===========================================================================

def _hit(memory_id: str, score: float) -> dict:
    """A single Qdrant neighbour hit, as fetch_ann_neighbors hands it over."""
    return {'id': memory_id, 'score': score}


class TestAnnPairsFromNeighbors:
    """Turn per-record ANN neighbour lists into index pairs, counting every loss.

    The ANN path exists to catch the paraphrase class the lexical difflib scan
    structurally cannot: same meaning, almost no shared wording. But an ANN
    scan loses recall in ways a full O(n^2) scan does not — a top_k cap, a
    score cutoff, a record that has no stored vector to query with. Every one
    of those losses is COUNTED and returned, so a cap firing is a metric an
    operator can alarm on rather than invisible recall loss.
    """

    def _memories(self, n: int, *, vectors: bool = True) -> list[dict]:
        out = []
        for i in range(n):
            m = _memory(f'm{i}', f'content {i}')
            m['vector'] = [0.1 * i, 0.2] if vectors else None
            out.append(m)
        return out

    def test_hits_at_or_above_threshold_become_pairs(self):
        memories = self._memories(3)
        neighbors = {'m0': [_hit('m1', 0.95)], 'm1': [], 'm2': []}
        pairs, _disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == [(0, 1)]

    def test_score_exactly_at_threshold_is_kept(self):
        """>= threshold, not > threshold — the cutoff is inclusive."""
        memories = self._memories(2)
        neighbors = {'m0': [_hit('m1', 0.9)], 'm1': []}
        pairs, disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == [(0, 1)]
        assert disclosure['below_threshold_dropped'] == 0

    def test_self_hit_is_dropped_and_not_counted_as_a_loss(self):
        """A record is always its own nearest neighbour; that is not a duplicate."""
        memories = self._memories(2)
        neighbors = {'m0': [_hit('m0', 1.0), _hit('m1', 0.95)], 'm1': []}
        pairs, disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == [(0, 1)], 'self-hit must not produce a (0,0) pair'
        assert disclosure['unknown_neighbor_dropped'] == 0, (
            'a self-hit is expected, not an unknown neighbour'
        )

    def test_mirrored_hits_dedupe_to_one_pair(self):
        """m0->m1 and m1->m0 are the same candidate pair, emitted once."""
        memories = self._memories(2)
        neighbors = {'m0': [_hit('m1', 0.95)], 'm1': [_hit('m0', 0.95)]}
        pairs, _disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == [(0, 1)]

    def test_pairs_are_normalised_low_index_first(self):
        """Ordering is canonical so the closure and any dedupe see one form."""
        memories = self._memories(2)
        neighbors = {'m1': [_hit('m0', 0.95)]}
        pairs, _disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == [(0, 1)], f'expected (0,1) normalised form, got {pairs}'

    def test_below_threshold_hits_are_dropped_and_counted(self):
        memories = self._memories(3)
        neighbors = {'m0': [_hit('m1', 0.5), _hit('m2', 0.2)], 'm1': [], 'm2': []}
        pairs, disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == []
        assert disclosure['below_threshold_dropped'] == 2

    def test_unknown_neighbor_id_is_dropped_and_counted(self):
        """A neighbour outside the scanned set cannot be clustered — count it.

        Happens when the ANN query returns a record the category filter
        excluded, or one written between the scroll and the query.
        """
        memories = self._memories(2)
        neighbors = {'m0': [_hit('not-in-scan', 0.99), _hit('m1', 0.95)], 'm1': []}
        pairs, disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert pairs == [(0, 1)]
        assert disclosure['unknown_neighbor_dropped'] == 1

    def test_top_k_saturation_is_counted(self):
        """A neighbour list filled to top_k may have dropped further matches."""
        memories = self._memories(4)
        neighbors = {
            'm0': [_hit('m1', 0.95), _hit('m2', 0.94)],  # exactly top_k=2 -> saturated
            'm1': [_hit('m0', 0.95)],                    # under cap -> not saturated
            'm2': [],
            'm3': [],
        }
        _pairs, disclosure = ann_pairs_from_neighbors(
            memories, neighbors, threshold=0.9, top_k=2,
        )
        assert disclosure['top_k_saturated'] == 1

    def test_top_k_saturation_counted_even_when_hits_are_below_threshold(self):
        """Saturation is a property of the QUERY cap, not of what survived it."""
        memories = self._memories(3)
        neighbors = {'m0': [_hit('m1', 0.1), _hit('m2', 0.1)], 'm1': [], 'm2': []}
        _pairs, disclosure = ann_pairs_from_neighbors(
            memories, neighbors, threshold=0.9, top_k=2,
        )
        assert disclosure['top_k_saturated'] == 1
        assert disclosure['below_threshold_dropped'] == 2

    def test_missing_vector_record_is_counted_and_contributes_no_pairs(self):
        """No stored vector means the record was never queried — disclose it."""
        memories = self._memories(3)
        memories[2]['vector'] = None
        neighbors = {'m0': [_hit('m1', 0.95)], 'm1': [], 'm2': []}
        pairs, disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert disclosure['missing_vector'] == 1
        assert all(2 not in p for p in pairs), (
            f'a vector-less record must contribute no pairs from its own query, got {pairs}'
        )

    def test_absent_vector_key_counts_as_missing_vector(self):
        """A record scrolled without with_vectors=True has no 'vector' key at all."""
        memories = [_memory('m0', 'a'), _memory('m1', 'b')]  # no 'vector' key
        pairs, disclosure = ann_pairs_from_neighbors(memories, {}, threshold=0.9)
        assert disclosure['missing_vector'] == 2
        assert pairs == []

    def test_disclosure_keys_always_present_and_integer_even_when_zero(self):
        """INV-2/INV-4: a cap that never fired still reports 0, never an absent key.

        An absent key is indistinguishable from 'not measured' downstream; the
        metrics layer emits one metric per counter and must never silently skip
        one because this run happened to be clean.
        """
        memories = self._memories(2)
        neighbors = {'m0': [_hit('m1', 0.95)], 'm1': []}
        _pairs, disclosure = ann_pairs_from_neighbors(
            memories, neighbors, threshold=0.9, top_k=10,
        )
        assert set(disclosure) == set(_ANN_DISCLOSURE_KEYS), (
            f'disclosure keys {sorted(disclosure)} != {sorted(_ANN_DISCLOSURE_KEYS)}'
        )
        for key, value in disclosure.items():
            assert isinstance(value, int) and not isinstance(value, bool), (
                f'disclosure[{key!r}] must be an int, got {value!r}'
            )
        assert all(v == 0 for v in disclosure.values()), (
            f'clean run must report all-zero counters, got {disclosure}'
        )

    def test_empty_inputs_produce_no_pairs_and_zeroed_disclosure(self):
        pairs, disclosure = ann_pairs_from_neighbors([], {}, threshold=0.9)
        assert pairs == []
        assert set(disclosure) == set(_ANN_DISCLOSURE_KEYS)
        assert all(v == 0 for v in disclosure.values())

    def test_transitive_closure_via_the_shared_core(self):
        """ANN pairs feed cluster_memories_by_pairs — the SAME closure the lexical path uses."""
        memories = self._memories(3)
        neighbors = {'m0': [_hit('m1', 0.95)], 'm1': [_hit('m2', 0.93)], 'm2': []}
        pairs, _disclosure = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        groups = cluster_memories_by_pairs(memories, pairs)
        assert len(groups) == 1
        assert [m['id'] for m in groups[0]] == ['m0', 'm1', 'm2']

    def test_pairs_are_deterministic_and_sorted(self):
        """Two runs over the same input emit the same pair list in the same order."""
        memories = self._memories(4)
        neighbors = {
            'm3': [_hit('m1', 0.95)],
            'm0': [_hit('m2', 0.95)],
            'm1': [_hit('m3', 0.95)],
            'm2': [],
        }
        first, _ = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        second, _ = ann_pairs_from_neighbors(memories, neighbors, threshold=0.9)
        assert first == second
        assert first == sorted(first), f'pairs must be emitted in sorted order, got {first}'

    def test_input_memories_not_mutated(self):
        memories = self._memories(2)
        snapshot = [dict(m) for m in memories]
        ann_pairs_from_neighbors(memories, {'m0': [_hit('m1', 0.95)]}, threshold=0.9)
        assert memories == snapshot
