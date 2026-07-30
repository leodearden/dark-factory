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
partition_topic_carveout = _mod.partition_topic_carveout
fetch_procedural_memories = _mod.fetch_procedural_memories
fetch_memories = _mod.fetch_memories
_ALL_CATEGORIES = _mod._ALL_CATEGORIES
apply_deletions = _mod.apply_deletions
resolve_ann_threshold = _mod.resolve_ann_threshold
fetch_ann_neighbors = _mod.fetch_ann_neighbors
compute_cluster_metrics = _mod.compute_cluster_metrics
build_metric_series = _mod.build_metric_series
split_plan_by_category = _mod.split_plan_by_category
_EVAL_ID = _mod._EVAL_ID
_GLOBAL_SCOPE = _mod._GLOBAL_SCOPE
emit_metrics_artifact = _mod.emit_metrics_artifact
details_artifact_path = _mod.details_artifact_path
_build_parser = _mod._build_parser
_run = _mod._run


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


# ===========================================================================
# The user-observable signal: the ANN path catches a paraphrase the lexical
# path structurally cannot (task 3210)
# ===========================================================================
#
# PROVENANCE OF THE CONSTANTS BELOW — all MEASURED at authoring time
# (2026-07-30), never chosen. No new labeling was done: the pair comes from
# task 3130's curator-ground-truth dataset,
# tests/fixtures/write_triage_calibration.jsonl (104 records / 20 clusters),
# and both members carry the curator label 'duplicate' in the same cluster
# e0a41fcd-0745-426b-bf71-45466e6d34e0.
#
# Measurement method (throwaway script, not committed): reused
# scripts/calibrate_write_triage.py's load_fixture / build_pair_sets /
# build_embed_fn / cosine_similarity to score all 301 true-duplicate pairs,
# then selected the qualifying pair with the LOWEST lexical ratio — i.e. the
# hardest paraphrase, the class this program exists to detect.
#
#   embedder:    text-embedding-3-small, no `dimensions` override (1536)
#                — the same embedder and metric space as the committed
#                calibration report, so the cosine is comparable to t_high.
#   cosine:      calibrate_write_triage.cosine_similarity
#   lexical:     stdlib difflib.SequenceMatcher(None, a, b).ratio()
#
# Cross-check against the committed report
# (calibration/write_triage_calibration_report.json), same fixture and
# embedder: true_dup n measured 301 vs report 301 (exact); true_dup max
# cosine measured 0.9657190014923018 vs report 0.9657105894445585 (agree to
# ~8.4e-6, the endpoint's run-to-run drift). The report records
# per_band.true_dup.deterministic = 7 pairs at/above t_high; this measurement
# found 6 — see escalation esc-3210-3. Immaterial here: the selected pair
# clears t_high by 0.0185, roughly 2200x that drift.
#
# GOTCHA: SequenceMatcher.ratio() is ORDER-SENSITIVE — ratio(b,a) is
# 0.09662716499544212, a different number. The ids below are in the sorted
# order build_pair_sets itself produces, and the recompute assertion uses
# that same order. Once the order is fixed the value is bit-reproducible.
_FIXTURE_PAIR_IDS = (
    '243b6dec-f0ce-4123-bb09-16d834b7e9c8',
    'c315352b-6d4e-467d-9a3f-360bc2d53229',
)
# Cosine between the two stored embeddings. Fed to the ANN path through an
# injected neighbour provider, so this test makes NO network call.
_MEASURED_COSINE = 0.9053783099307539
# Raw difflib ratio on the fixture contents, argument order as above. Pins
# the fixture text against drift: it is recomputed live and compared.
_MEASURED_LEXICAL_RATIO = 0.10209662716499544
# The ratio the detector's lexical path actually sees, after its
# (content or '').strip().lower() normalisation. Both numbers are far below
# the 0.85 lexical threshold — reported as measured, never asserted.
_MEASURED_LEXICAL_RATIO_NORMALIZED = 0.09480401093892434

_FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'write_triage_calibration.jsonl'


def _load_fixture_pair() -> list[dict]:
    """Load the two selected fixture records, in _FIXTURE_PAIR_IDS order."""
    by_id: dict[str, dict] = {}
    with _FIXTURE_PATH.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            by_id[str(record['memory_id'])] = record
    missing = [i for i in _FIXTURE_PAIR_IDS if i not in by_id]
    assert not missing, f'fixture no longer contains {missing} — re-run the pre-2 measurement'
    return [
        _memory(
            id=str(by_id[i]['memory_id']),
            content=by_id[i]['content'],
            created_at=by_id[i].get('created_at'),
            category='procedural_knowledge',
        )
        for i in _FIXTURE_PAIR_IDS
    ]


def _calibrated_ann_threshold() -> float:
    """Read the ANN cutoff from the committed calibration OUTPUT.

    Deliberately NOT a literal in this test. config.yaml's
    write_triage.t_high is a derived order statistic of a measured
    distribution (provenance: calibration/write_triage_calibration_report.json),
    and it is the single home for that number — the detector reads the same
    one. Re-deriving or hardcoding it here would let the two drift apart.
    """
    import yaml  # noqa: PLC0415

    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    data = yaml.safe_load(config_path.read_text())
    t_high = data['write_triage']['t_high']
    assert isinstance(t_high, float), (
        f'config.write_triage.t_high must be a calibrated float, got {t_high!r}'
    )
    return t_high


class TestAnnPathCatchesParaphraseLexicalPathMisses:
    """The signal this whole task exists to produce, on curator-labeled data.

    Two records the curator labeled as duplicates of each other, stating the
    identical reify-debug-MCP fact in almost entirely disjoint wording. The
    lexical difflib scan cannot see them as related at any usable threshold;
    the ANN path clusters them comfortably. Both verdicts are REPORTED, so an
    operator can audit the disagreement rather than trust one detector.
    """

    def test_live_recomputed_lexical_ratio_matches_the_measurement(self):
        """Pins the fixture text against drift — deterministic stdlib, no network.

        If someone edits either fixture record, this fails loudly instead of
        letting the recorded cosine silently describe different text.
        """
        import difflib  # noqa: PLC0415

        a, b = _load_fixture_pair()
        live_raw = difflib.SequenceMatcher(None, a['content'], b['content']).ratio()
        assert live_raw == pytest.approx(_MEASURED_LEXICAL_RATIO, abs=1e-9), (
            f'fixture content drifted: recomputed {live_raw!r}, '
            f'recorded {_MEASURED_LEXICAL_RATIO!r}'
        )
        live_norm = difflib.SequenceMatcher(
            None, a['content'].strip().lower(), b['content'].strip().lower(),
        ).ratio()
        assert live_norm == pytest.approx(_MEASURED_LEXICAL_RATIO_NORMALIZED, abs=1e-9)

    def test_measured_cosine_clears_the_calibrated_cutoff(self):
        """The premise the ANN assertion rests on, checked against the live config.

        A recalibration that moved t_high above this pair's measured cosine
        would invalidate the fixture choice — fail here, with instructions,
        rather than somewhere confusing downstream.
        """
        cutoff = _calibrated_ann_threshold()
        assert cutoff <= _MEASURED_COSINE, (
            f'the recorded pair no longer clears the calibrated cutoff '
            f'(cosine {_MEASURED_COSINE!r} < t_high {cutoff!r}) — re-run the '
            f'pre-2 measurement and select a new pair'
        )

    def test_ann_path_clusters_the_pair_that_lexical_cannot(self):
        """THE signal: ANN clusters it, lexical does not, and both are reported."""
        memories = _load_fixture_pair()
        cutoff = _calibrated_ann_threshold()

        # Injected neighbour provider — the measured cosine, no network call.
        neighbors = {
            memories[0]['id']: [{'id': memories[1]['id'], 'score': _MEASURED_COSINE}],
            memories[1]['id']: [{'id': memories[0]['id'], 'score': _MEASURED_COSINE}],
        }
        for m in memories:
            m['vector'] = [0.0, 1.0]  # presence only; scores come from `neighbors`

        ann_pairs, ann_disclosure = ann_pairs_from_neighbors(
            memories, neighbors, threshold=cutoff, top_k=10,
        )
        assert ann_pairs == [(0, 1)], 'ANN must propose the pair at the calibrated cutoff'

        plan = build_sweep_plan(
            memories,
            ann_pairs=ann_pairs,
            ann_scores={(0, 1): _MEASURED_COSINE},
            ann_disclosure=ann_disclosure,
            ann_threshold=cutoff,
        )

        # (1) The pair is clustered, and it got there via the ANN path.
        assert plan['clusters_total'] == 1, (
            f'expected the ANN path to cluster the pair, got {plan["clusters_total"]} cluster(s)'
        )
        group = plan['near_duplicate_groups'][0]
        assert set(group['member_ids']) == set(_FIXTURE_PAIR_IDS)
        assert group['found_by'] == ['ann'], (
            f'expected ann-only attribution, got {group["found_by"]}'
        )

        # (2) The lexical verdict is REPORTED alongside — as measured, not
        # asserted to be negative. It happens to be negative here, and the
        # report says so rather than staying silent about the disagreement.
        assert 'lexical_clustered' in group
        assert group['lexical_clustered'] is (group['found_by'] != ['ann'])
        verdicts = plan['path_verdicts']
        assert verdicts['ann_only_clusters'] == 1
        assert verdicts['lexical_only_clusters'] == 0
        assert verdicts['both_paths_clusters'] == 0

        # (3) BOTH measured numbers are in the operator-visible output.
        assert group['ann_max_score'] == pytest.approx(_MEASURED_COSINE, abs=1e-12)
        assert group['lexical_max_ratio'] == pytest.approx(
            _MEASURED_LEXICAL_RATIO_NORMALIZED, abs=1e-9,
        ), 'the lexical score must be reported even when it lost, so the disagreement is auditable'
        assert plan['ann_threshold'] == cutoff
        assert plan['threshold'] == 0.85

        # (4) The whole report stays JSON-serialisable (stdout contract).
        json.dumps(plan, default=str)

    def test_lexical_path_alone_does_not_cluster_the_pair(self):
        """Measured, not assumed: with no ANN pairs the plan is empty.

        This is the recall gap the ANN path closes. Stated as its own test so
        the signal test above cannot pass for the wrong reason.
        """
        memories = _load_fixture_pair()
        plan = build_sweep_plan(memories)
        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []


# ===========================================================================
# All three Mem0 categories, both filter sites (task 3210)
# ===========================================================================

_PK = 'procedural_knowledge'
_PN = 'preferences_and_norms'
_OS = 'observations_and_summaries'


def _scroll_mock(by_category: dict[str, list[dict]]):
    """AsyncMock scroll_by_metadata dispatching on the filter's category."""
    from unittest.mock import AsyncMock  # noqa: PLC0415

    async def _scroll(scope, filters, limit=1000, **kwargs):
        return list(by_category.get(filters.get('category'), []))

    return AsyncMock(side_effect=_scroll)


@pytest.mark.asyncio
class TestFetchMemoriesAllCategories:
    """One scroll per category — scroll_by_metadata is AND-equality only.

    There is no OR filter available, so a single widened call cannot express
    "any of these three categories". Three scrolls is the shape the primitive
    actually supports, not a missed optimisation.
    """

    async def test_one_scroll_per_category_with_vectors_forwarded(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = _scroll_mock({})

        await fetch_memories(
            memory, 'dark_factory', categories=(_PK, _PN, _OS),
            scan_limit=99, with_vectors=True,
        )

        assert memory.mem0.scroll_by_metadata.await_count == 3
        seen = []
        for call in memory.mem0.scroll_by_metadata.await_args_list:
            assert call.args[0].project_id == 'dark_factory'
            seen.append(call.args[1]['category'])
            assert call.kwargs.get('limit') == 99
            assert call.kwargs.get('with_vectors') is True
        assert seen == [_PK, _PN, _OS]

    async def test_default_categories_are_the_three_mem0_categories(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = _scroll_mock({})

        await fetch_memories(memory, 'dark_factory')

        requested = [
            c.args[1]['category'] for c in memory.mem0.scroll_by_metadata.await_args_list
        ]
        assert set(requested) == {_PK, _PN, _OS}
        assert set(_ALL_CATEGORIES) == {_PK, _PN, _OS}

    async def test_merges_records_and_lifts_category_and_vector(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = _scroll_mock({
            _PK: [{'id': 'a', 'created_at': None,
                   'metadata': {'data': 'pk text', 'category': _PK}, 'vector': [0.1]}],
            _PN: [{'id': 'b', 'created_at': None,
                   'metadata': {'data': 'pn text', 'category': _PN}, 'vector': [0.2]}],
            _OS: [{'id': 'c', 'created_at': None,
                   'metadata': {'data': 'os text', 'category': _OS}, 'vector': None}],
        })

        records, _stats = await fetch_memories(
            memory, 'dark_factory', with_vectors=True,
        )

        by_id = {r['id']: r for r in records}
        assert set(by_id) == {'a', 'b', 'c'}
        assert by_id['a']['category'] == _PK
        assert by_id['b']['category'] == _PN
        assert by_id['c']['category'] == _OS
        assert by_id['a']['content'] == 'pk text'
        assert by_id['a']['vector'] == [0.1]
        assert by_id['c']['vector'] is None, 'a vector-less record survives, degraded not dropped'

    async def test_per_category_truncation_is_counted_separately(self):
        """A cap firing on ONE category must not be reported as a clean scan."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = _scroll_mock({
            _PK: [{'id': f'p{i}', 'created_at': None,
                   'metadata': {'data': f't{i}', 'category': _PK}} for i in range(3)],
            _PN: [{'id': 'q', 'created_at': None, 'metadata': {'data': 'x', 'category': _PN}}],
        })

        _records, stats = await fetch_memories(memory, 'dark_factory', scan_limit=3)

        assert stats[_PK]['scanned'] == 3
        assert stats[_PK]['truncated'] == 1, 'len(records) >= scan_limit must count as truncated'
        assert stats[_PN]['scanned'] == 1
        assert stats[_PN]['truncated'] == 0
        assert stats[_OS]['scanned'] == 0
        assert stats[_OS]['truncated'] == 0

    async def test_fetch_procedural_memories_alias_still_single_category(self):
        """The 3136-scheduled path keeps working byte-for-byte."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = _scroll_mock({
            _PK: [{'id': 'a', 'created_at': None,
                   'metadata': {'data': 'pk text', 'category': _PK}}],
            _PN: [{'id': 'b', 'created_at': None,
                   'metadata': {'data': 'pn text', 'category': _PN}}],
        })

        result = await fetch_procedural_memories(memory, 'dark_factory', scan_limit=100)

        assert memory.mem0.scroll_by_metadata.await_count == 1
        assert [r['id'] for r in result] == ['a']
        assert isinstance(result, list), 'alias returns a bare list, not a (records, stats) tuple'
        assert 'vector' not in result[0], 'alias defaults to payload-only'


class TestBuildSweepPlanAllCategories:
    """The Python-side filter widens to all three categories, but never merges them."""

    def test_preferences_and_observations_are_no_longer_dropped(self):
        memories = [
            _memory('p1', _VENV_GOTCHA_A, category=_PN),
            _memory('p2', _VENV_GOTCHA_B, category=_PN),
            _memory('o1', _VENV_GOTCHA_A, category=_OS),
            _memory('o2', _VENV_GOTCHA_B, category=_OS),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 2, (
            'preferences_and_norms and observations_and_summaries must each cluster'
        )
        assert len(plan['delete_candidates']) == 2

    def test_other_categories_still_excluded(self):
        memories = [
            _memory('e1', _VENV_GOTCHA_A, category='entities_and_relations'),
            _memory('e2', _VENV_GOTCHA_B, category='entities_and_relations'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []

    def test_cross_category_near_duplicates_never_union(self):
        """Identical text in two categories is two records, not one duplicate.

        A preference and a procedure that happen to read alike are different
        kinds of knowledge; deleting one because the other exists would be a
        cross-store data loss, not a deduplication.
        """
        memories = [
            _memory('a', _VENV_GOTCHA_A, category=_PK),
            _memory('b', _VENV_GOTCHA_A, category=_PN),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []

    def test_each_category_clusters_independently_in_one_run(self):
        memories = [
            _memory('k1', _VENV_GOTCHA_A, category=_PK),
            _memory('k2', _VENV_GOTCHA_B, category=_PK),
            _memory('k3', _VENV_GOTCHA_C, category=_PK),
            _memory('n1', _VENV_GOTCHA_A, category=_PN),
            _memory('n2', _VENV_GOTCHA_B, category=_PN),
            _memory('d1', _DISTRACTOR, category=_OS),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 2
        members = [set(g['member_ids']) for g in plan['near_duplicate_groups']]
        assert {'k1', 'k2', 'k3'} in members
        assert {'n1', 'n2'} in members

    def test_explicit_categories_argument_narrows_the_sweep(self):
        memories = [
            _memory('k1', _VENV_GOTCHA_A, category=_PK),
            _memory('k2', _VENV_GOTCHA_B, category=_PK),
            _memory('n1', _VENV_GOTCHA_A, category=_PN),
            _memory('n2', _VENV_GOTCHA_B, category=_PN),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD, categories=(_PK,))
        assert plan['clusters_total'] == 1
        assert set(plan['near_duplicate_groups'][0]['member_ids']) == {'k1', 'k2'}

    def test_ann_pairs_spanning_categories_are_dropped(self):
        """Even if ANN proposes a cross-category pair, it must not cluster."""
        memories = [
            _memory('a', 'alpha text', category=_PK),
            _memory('b', 'beta text', category=_PN),
        ]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 1)], ann_threshold=0.9,
        )
        assert plan['clusters_total'] == 0
        assert plan['delete_candidates'] == []


# ===========================================================================
# Option-C topic carve-out (task 3136's rule, honored as input filtering)
# ===========================================================================

def _topic_memory(id: str, content: str, topic: str | None, category: str = _PK) -> dict:
    metadata: dict = {} if topic is None else {'topic': topic}
    return _memory(id, content, created_at='2026-07-12T00:00:00+00:00',
                   category=category, metadata=metadata)


class TestPartitionTopicCarveout:
    """The pure partition: clusters sharing one non-empty topic are carved out."""

    def test_shared_non_empty_topic_is_carved_out(self):
        group = [_topic_memory('a', 'x', 'venv'), _topic_memory('b', 'y', 'venv')]
        actionable, carved = partition_topic_carveout([group])
        assert actionable == []
        assert carved == [group]

    def test_differing_topics_stay_actionable(self):
        group = [_topic_memory('a', 'x', 'venv'), _topic_memory('b', 'y', 'rustfmt')]
        actionable, carved = partition_topic_carveout([group])
        assert actionable == [group]
        assert carved == []

    def test_partially_missing_topic_stays_actionable(self):
        """Only a FULLY shared non-empty topic carves out."""
        group = [_topic_memory('a', 'x', 'venv'), _topic_memory('b', 'y', None)]
        actionable, carved = partition_topic_carveout([group])
        assert actionable == [group]
        assert carved == []

    def test_shared_empty_topic_stays_actionable(self):
        """An empty-string topic is absent, not a shared topic."""
        group = [_topic_memory('a', 'x', ''), _topic_memory('b', 'y', '')]
        actionable, carved = partition_topic_carveout([group])
        assert actionable == [group]
        assert carved == []

    def test_no_topic_at_all_stays_actionable(self):
        """Today's corpus: metadata.topic has zero footprint, so nothing carves out."""
        group = [_topic_memory('a', 'x', None), _topic_memory('b', 'y', None)]
        actionable, carved = partition_topic_carveout([group])
        assert actionable == [group]
        assert carved == []

    def test_mixed_groups_partition_independently(self):
        shared = [_topic_memory('a', 'x', 'venv'), _topic_memory('b', 'y', 'venv')]
        mixed = [_topic_memory('c', 'x', 'venv'), _topic_memory('d', 'y', 'rustfmt')]
        actionable, carved = partition_topic_carveout([shared, mixed])
        assert actionable == [mixed]
        assert carved == [shared]

    def test_empty_input(self):
        assert partition_topic_carveout([]) == ([], [])


class TestBuildSweepPlanTopicCarveout:
    """A carved-out cluster never reaches pick_survivor, so it is never deleted."""

    def test_carved_out_cluster_yields_no_delete_candidates(self):
        memories = [
            _topic_memory('m1', _VENV_GOTCHA_A, 'venv'),
            _topic_memory('m2', _VENV_GOTCHA_B, 'venv'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['delete_candidates'] == []
        assert plan['near_duplicate_groups'] == []
        assert plan['topic_carveout_clusters'] == 1
        assert len(plan['topic_carveout_groups']) == 1
        carved = plan['topic_carveout_groups'][0]
        assert set(carved['member_ids']) == {'m1', 'm2'}
        assert carved['topic'] == 'venv'

    def test_differing_topics_still_yield_delete_candidates(self):
        memories = [
            _topic_memory('m1', _VENV_GOTCHA_A, 'venv'),
            _topic_memory('m2', _VENV_GOTCHA_B, 'rustfmt'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 1
        assert len(plan['delete_candidates']) == 1
        assert plan['topic_carveout_clusters'] == 0

    def test_partially_missing_topic_still_yields_delete_candidates(self):
        memories = [
            _topic_memory('m1', _VENV_GOTCHA_A, 'venv'),
            _topic_memory('m2', _VENV_GOTCHA_B, None),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert len(plan['delete_candidates']) == 1
        assert plan['topic_carveout_clusters'] == 0

    def test_carveout_applies_to_all_three_categories(self):
        memories = []
        for cat in (_PK, _PN, _OS):
            memories += [
                _topic_memory(f'{cat}-1', _VENV_GOTCHA_A, 'venv', category=cat),
                _topic_memory(f'{cat}-2', _VENV_GOTCHA_B, 'venv', category=cat),
            ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['delete_candidates'] == []
        assert plan['topic_carveout_clusters'] == 3

    def test_carved_out_cluster_is_excluded_from_clusters_total(self):
        """clusters_total counts ACTIONED clusters; carve-outs are reported separately."""
        memories = [
            _topic_memory('m1', _VENV_GOTCHA_A, 'venv'),
            _topic_memory('m2', _VENV_GOTCHA_B, 'venv'),
            _topic_memory('n1', _DISTRACTOR, 'other'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        assert plan['clusters_total'] == 0
        assert plan['topic_carveout_clusters'] == 1

    def test_plan_stays_json_serializable_with_carveouts(self):
        memories = [
            _topic_memory('m1', _VENV_GOTCHA_A, 'venv'),
            _topic_memory('m2', _VENV_GOTCHA_B, 'venv'),
        ]
        json.dumps(build_sweep_plan(memories, threshold=_THRESHOLD), default=str)


# ===========================================================================
# Calibrated ANN threshold resolution + the Qdrant query shape (task 3210)
# ===========================================================================

def _config_double(t_high):
    """MemoryService double whose config carries a write_triage.t_high."""
    from unittest.mock import MagicMock  # noqa: PLC0415

    service = MagicMock()
    service.config.write_triage.t_high = t_high
    return service


class TestResolveAnnThreshold:
    """The cutoff is READ from the committed calibration output, never invented.

    config.write_triage.t_high is a derived order statistic of a measured
    similarity distribution (provenance: the calibration report). This script
    must not carry its own copy or its own fallback — an uncalibrated
    stand-in would be exactly the invented threshold G6 forbids.
    """

    def test_reads_t_high_from_config(self):
        """Point the config at a distinctive value and get that value back.

        Asserting a DISTINCT value (not the real 0.8868...) is the point: it
        proves the number came from config rather than from a literal that
        happens to match.
        """
        resolved, disclosure = resolve_ann_threshold(_config_double(0.4242424242))
        assert resolved == 0.4242424242
        assert disclosure['ann_disabled_uncalibrated'] == 0

    def test_explicit_override_wins(self):
        resolved, _ = resolve_ann_threshold(_config_double(0.4242424242), override=0.77)
        assert resolved == 0.77

    def test_override_wins_even_when_uncalibrated(self):
        """An operator override rescues an uncalibrated deployment."""
        resolved, disclosure = resolve_ann_threshold(_config_double(None), override=0.77)
        assert resolved == 0.77
        assert disclosure['ann_disabled_uncalibrated'] == 0

    def test_uncalibrated_disables_ann_and_counts_it(self, caplog):
        """No t_high and no override: DISABLE the path, log ERROR, count it.

        Never substitute a plausible-looking number — a wrong cutoff would
        silently mis-cluster the corpus, and under --apply that means wrong
        deletions. Disabled-and-counted is the loud degradation.
        """
        import logging  # noqa: PLC0415

        with caplog.at_level(logging.ERROR, logger='audit_duplicate_memories'):
            resolved, disclosure = resolve_ann_threshold(_config_double(None))

        assert resolved is None, 'must not fabricate a threshold'
        assert disclosure['ann_disabled_uncalibrated'] == 1
        assert any(r.levelno >= logging.ERROR for r in caplog.records), (
            'an uncalibrated ANN path must be loud, not silent'
        )

    def test_non_numeric_t_high_is_treated_as_uncalibrated(self):
        """A Mock/str/bool leaf must not be accepted as a threshold."""
        for bad in ('0.9', True, object()):
            resolved, disclosure = resolve_ann_threshold(_config_double(bad))
            assert resolved is None, f'{bad!r} must not resolve as a threshold'
            assert disclosure['ann_disabled_uncalibrated'] == 1

    def test_missing_config_hops_are_survived(self):
        """A config without write_triage at all degrades, it does not raise."""
        resolved, disclosure = resolve_ann_threshold(object())
        assert resolved is None
        assert disclosure['ann_disabled_uncalibrated'] == 1


@pytest.mark.asyncio
class TestFetchAnnNeighbors:
    """query_points once per record, using the record's OWN stored vector."""

    def _memory_service(self, query_points):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        client = MagicMock()
        client.query_points = query_points
        service = MagicMock()
        service.config.mem0.collection_prefix = 'mem0'
        service.mem0._get_async_qdrant = AsyncMock(return_value=client)
        return service, client

    def _points(self, hits):
        from unittest.mock import MagicMock  # noqa: PLC0415

        result = MagicMock()
        result.points = [
            MagicMock(id=i, score=s) for i, s in hits
        ]
        return result

    async def test_queries_once_per_record_with_its_own_vector(self):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._points([]))
        service, _client = self._memory_service(query_points)
        records = [
            {'id': 'a', 'vector': [0.1, 0.2]},
            {'id': 'b', 'vector': [0.3, 0.4]},
        ]

        await fetch_ann_neighbors(
            service, 'dark_factory', records, top_k=5, score_threshold=0.88,
        )

        assert query_points.await_count == 2
        first = query_points.await_args_list[0].kwargs
        assert first['collection_name'] == 'mem0_dark_factory'
        assert first['query'] == [0.1, 0.2], (
            "the query vector must be the record's OWN stored vector — "
            'no embedding API call at detector runtime'
        )
        assert first['limit'] == 6, 'limit is top_k+1 to absorb the self-hit'
        assert first['score_threshold'] == 0.88
        assert first['with_payload'] is False

    async def test_self_hit_is_dropped(self):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._points([('a', 1.0), ('b', 0.93)]))
        service, _client = self._memory_service(query_points)
        records = [{'id': 'a', 'vector': [0.1]}]

        neighbors, _disclosure = await fetch_ann_neighbors(
            service, 'dark_factory', records, top_k=5, score_threshold=0.88,
        )

        assert neighbors['a'] == [{'id': 'b', 'score': 0.93}]

    async def test_vectorless_record_is_not_queried(self):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._points([]))
        service, _client = self._memory_service(query_points)
        records = [{'id': 'a', 'vector': None}, {'id': 'b', 'vector': [0.1]}]

        await fetch_ann_neighbors(
            service, 'dark_factory', records, top_k=5, score_threshold=0.88,
        )

        assert query_points.await_count == 1, 'a record with no vector cannot be a query point'

    async def test_per_record_failure_is_counted_and_does_not_abort(self):
        """One bad record must not lose the whole sweep."""
        from unittest.mock import AsyncMock  # noqa: PLC0415

        async def _side_effect(**kwargs):
            if kwargs['query'] == [0.1]:
                raise RuntimeError('Qdrant blew up')
            return self._points([('c', 0.95)])

        query_points = AsyncMock(side_effect=_side_effect)
        service, _client = self._memory_service(query_points)
        records = [{'id': 'a', 'vector': [0.1]}, {'id': 'b', 'vector': [0.2]}]

        neighbors, disclosure = await fetch_ann_neighbors(
            service, 'dark_factory', records, top_k=5, score_threshold=0.88,
        )

        assert disclosure['ann_query_errors'] == 1
        assert 'a' not in neighbors
        assert neighbors['b'] == [{'id': 'c', 'score': 0.95}], 'the sweep continued'

    async def test_empty_records_makes_no_queries(self):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._points([]))
        service, _client = self._memory_service(query_points)

        neighbors, disclosure = await fetch_ann_neighbors(
            service, 'dark_factory', [], top_k=5, score_threshold=0.88,
        )

        assert query_points.await_count == 0
        assert neighbors == {}
        assert disclosure['ann_query_errors'] == 0


# ===========================================================================
# Cluster metrics + the M1 metric series (task 3210)
# ===========================================================================

def _dated(id: str, content: str, created_at: str | None, *,
           category: str = _PK, metadata: dict | None = None) -> dict:
    return _memory(id, content, created_at=created_at, category=category,
                   metadata=metadata or {})


def _plan_slice(groups: list[dict] | None = None,
                carveouts: list[dict] | None = None) -> dict:
    """A per-category plan slice, as produced by split_plan_by_category."""
    return {
        'near_duplicate_groups': groups or [],
        'topic_carveout_groups': carveouts or [],
    }


def _by_id(metrics) -> dict:
    return {m.metric_id: m for m in metrics}


class TestSplitPlanByCategory:
    """The plan carries `category` per group; the split just reads it.

    build_sweep_plan already clusters per category and tags every emitted
    group, so deriving the per-category slices is a projection of data the
    plan states — not a second, re-derived categorisation that could drift
    from the one the plan actually used.
    """

    def test_groups_and_carveouts_route_to_their_category(self):
        plan = {
            'near_duplicate_groups': [
                {'member_ids': ['a', 'b'], 'category': _PK},
                {'member_ids': ['c', 'd'], 'category': _PN},
            ],
            'topic_carveout_groups': [
                {'member_ids': ['e', 'f'], 'category': _PN, 'topic': 'venv'},
            ],
        }
        by_cat = split_plan_by_category(plan, (_PK, _PN, _OS))

        assert [g['member_ids'] for g in by_cat[_PK]['near_duplicate_groups']] == [['a', 'b']]
        assert by_cat[_PK]['topic_carveout_groups'] == []
        assert [g['member_ids'] for g in by_cat[_PN]['near_duplicate_groups']] == [['c', 'd']]
        assert [g['member_ids'] for g in by_cat[_PN]['topic_carveout_groups']] == [['e', 'f']]

    def test_requested_category_with_nothing_still_gets_a_slice(self):
        """An empty slice, not a missing key — every requested category reports."""
        by_cat = split_plan_by_category({'near_duplicate_groups': []}, (_PK, _OS))
        assert set(by_cat) == {_PK, _OS}
        assert by_cat[_OS] == {'near_duplicate_groups': [], 'topic_carveout_groups': []}


class TestComputeClusterMetricsPerCategory:
    """Per category: cluster count, size distribution, clustered-member share."""

    def _fixture(self):
        records = {
            _PK: [
                _dated('m1', _VENV_GOTCHA_A, '2026-01-01T00:00:00+00:00'),
                _dated('m2', _VENV_GOTCHA_B, '2026-01-02T00:00:00+00:00'),
                _dated('m3', _VENV_GOTCHA_C, '2026-01-03T00:00:00+00:00'),
                _dated('m4', _DISTRACTOR, '2026-01-04T00:00:00+00:00'),
            ],
        }
        plans = {
            _PK: _plan_slice([{'member_ids': ['m1', 'm2', 'm3'], 'category': _PK}]),
        }
        return records, plans

    def test_cluster_count_is_a_directional_count(self):
        records, plans = self._fixture()
        metrics, _details = compute_cluster_metrics(records, plans, {})
        metric = _by_id(metrics)[f'near_duplicate_clusters.{_PK}']

        assert metric.kind == 'count'
        assert metric.value == 1
        assert metric.n == 4, 'n is the exposure: the category corpus size'
        assert metric.direction == 'higher_is_worse'

    def test_size_max_and_mean_are_scalars_with_a_histogram_in_details(self):
        records = {
            _PK: [_dated(f'm{i}', f'body {i}', '2026-01-01T00:00:00+00:00') for i in range(7)],
        }
        plans = {
            _PK: _plan_slice([
                {'member_ids': ['m0', 'm1', 'm2'], 'category': _PK},
                {'member_ids': ['m3', 'm4'], 'category': _PK},
            ]),
        }
        metrics, details = compute_cluster_metrics(records, plans, {})
        by_id = _by_id(metrics)

        assert by_id[f'cluster_size_max.{_PK}'].kind == 'scalar'
        assert by_id[f'cluster_size_max.{_PK}'].value == 3
        assert by_id[f'cluster_size_max.{_PK}'].n == 2, 'n is the number of clusters'
        assert by_id[f'cluster_size_mean.{_PK}'].kind == 'scalar'
        assert by_id[f'cluster_size_mean.{_PK}'].value == pytest.approx(2.5)
        # A max/mean pair cannot show a bimodal distribution; the histogram can.
        assert details['categories'][_PK]['cluster_size_histogram'] == {'2': 1, '3': 1}

    def test_clustered_member_share_is_a_proportion_over_the_corpus(self):
        records, plans = self._fixture()
        metrics, _details = compute_cluster_metrics(records, plans, {})
        metric = _by_id(metrics)[f'clustered_member_share.{_PK}']

        assert metric.kind == 'proportion'
        assert metric.denominator == 4, 'denominator is the category corpus size'
        assert metric.n == 4
        assert metric.value == pytest.approx(0.75)
        assert metric.direction == 'higher_is_worse'
        # M1 cross-field rule: successes must be a whole number of records.
        assert metric.value * metric.denominator == pytest.approx(3.0)

    def test_topic_carveout_count_is_a_scalar_never_alarmed(self):
        """More carve-outs is an EXPECTED Option-C shape, not a regression.

        Emitting it as a `count` would attach M2's Poisson tail test and alarm
        the moment topics start landing (3195/3201) — i.e. alarm on the
        feature working. Scalar reports it without arming a rule.
        """
        records = {_PK: [_dated('a', 'x', None), _dated('b', 'y', None)]}
        plans = {_PK: _plan_slice(carveouts=[
            {'member_ids': ['a', 'b'], 'category': _PK, 'topic': 'venv'},
        ])}
        metric = _by_id(compute_cluster_metrics(records, plans, {})[0])[
            f'topic_carveout_clusters.{_PK}'
        ]

        assert metric.kind == 'scalar'
        assert metric.direction is None
        assert metric.value == 1

    def test_each_category_reports_independently(self):
        records = {
            _PK: [_dated('a', 'x', None), _dated('b', 'y', None)],
            _PN: [_dated('c', 'x', None, category=_PN)],
            _OS: [],
        }
        plans = {
            _PK: _plan_slice([{'member_ids': ['a', 'b'], 'category': _PK}]),
            _PN: _plan_slice(),
            _OS: _plan_slice(),
        }
        by_id = _by_id(compute_cluster_metrics(records, plans, {})[0])

        assert by_id[f'near_duplicate_clusters.{_PK}'].value == 1
        assert by_id[f'near_duplicate_clusters.{_PN}'].value == 0
        assert by_id[f'near_duplicate_clusters.{_OS}'].value == 0


class TestComputeClusterMetricsTopicAccretion:
    """Per-topic accretion: members per day across that topic's created_at span."""

    def test_rate_is_members_over_the_span_in_days(self):
        records = {
            _PK: [
                _dated('a', 'x', '2026-01-01T00:00:00+00:00', metadata={'topic': 'venv'}),
                _dated('b', 'y', '2026-01-03T00:00:00+00:00', metadata={'topic': 'venv'}),
                _dated('c', 'z', '2026-01-05T00:00:00+00:00', metadata={'topic': 'venv'}),
            ],
        }
        metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})
        table = details['categories'][_PK]['topic_accretion']

        assert [row['topic'] for row in table] == ['venv']
        assert table[0]['members'] == 3
        assert table[0]['span_days'] == pytest.approx(4.0)
        assert table[0]['rate_per_day'] == pytest.approx(0.75)

        metric = _by_id(metrics)[f'topic_accretion_rate_mean.{_PK}']
        assert metric.kind == 'scalar'
        assert metric.value == pytest.approx(0.75)
        assert metric.n == 1, 'n is the number of topics with a measurable rate'

    def test_zero_span_topic_has_no_defined_rate(self):
        """A rate needs elapsed time. One instant is not a rate — it is null.

        Dividing by a fudged epsilon would manufacture an enormous rate out of
        a single-record topic, so the row reports rate_per_day=None and is
        excluded from the mean's n.
        """
        records = {
            _PK: [_dated('a', 'x', '2026-01-01T00:00:00+00:00', metadata={'topic': 'venv'})],
        }
        metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})

        assert details['categories'][_PK]['topic_accretion'][0]['rate_per_day'] is None
        metric = _by_id(metrics)[f'topic_accretion_rate_mean.{_PK}']
        assert metric.n == 0
        assert metric.value == 0.0

    def test_no_topics_still_emits_a_well_formed_metric(self):
        """Today's corpus has zero topics — n=0 is a reported state, not an error."""
        records = {_PK: [_dated('a', 'x', '2026-01-01T00:00:00+00:00')]}
        metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})

        assert details['categories'][_PK]['topic_accretion'] == []
        metric = _by_id(metrics)[f'topic_accretion_rate_mean.{_PK}']
        assert metric.n == 0
        assert metric.value == 0.0

    def test_unparseable_created_at_never_crashes_a_rate_and_is_counted(self):
        records = {
            _PK: [
                _dated('a', 'x', 'not-a-timestamp', metadata={'topic': 'venv'}),
                _dated('b', 'y', None, metadata={'topic': 'venv'}),
                _dated('c', 'z', '2026-01-01T00:00:00+00:00', metadata={'topic': 'venv'}),
                _dated('d', 'w', '2026-01-03T00:00:00+00:00', metadata={'topic': 'venv'}),
            ],
        }
        metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})
        row = details['categories'][_PK]['topic_accretion'][0]

        assert row['span_days'] == pytest.approx(2.0), 'span uses only parseable stamps'
        assert row['members'] == 2, 'only records the span can actually place'
        assert row['rate_per_day'] == pytest.approx(1.0)

        counted = _by_id(metrics)[f'unparseable_created_at.{_PK}']
        assert counted.kind == 'count'
        assert counted.value == 2
        assert counted.direction == 'higher_is_worse'


class TestComputeClusterMetricsConsolidationNetDelta:
    """Net delta: what accreted after a consolidation minus what it absorbed."""

    def test_net_delta_is_accretion_after_minus_absorbed(self):
        records = {
            _PK: [
                _dated('old', 'x', '2026-01-01T00:00:00+00:00', metadata={'topic': 'venv'}),
                _dated('canon', 'merged', '2026-01-03T00:00:00+00:00',
                       metadata={'topic': 'venv', 'canonical': True,
                                 'supersedes': ['gone-1', 'gone-2']}),
                _dated('after', 'z', '2026-01-05T00:00:00+00:00', metadata={'topic': 'venv'}),
            ],
        }
        metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})
        events = details['categories'][_PK]['consolidation_events']

        assert len(events) == 1
        assert events[0]['memory_id'] == 'canon'
        assert events[0]['absorbed'] == 2
        assert events[0]['accreted_after'] == 1, 'only records created AFTER the event'
        assert events[0]['net_delta'] == -1

        metric = _by_id(metrics)[f'consolidation_net_delta.{_PK}']
        assert metric.kind == 'scalar', 'a signed delta has no count/proportion rule'
        assert metric.value == -1
        assert metric.n == 1, 'n is the number of consolidation events'

    def test_supersedes_alone_is_a_consolidation_event(self):
        records = {
            _PK: [
                _dated('s', 'x', '2026-01-01T00:00:00+00:00',
                       metadata={'topic': 'venv', 'supersedes': 'gone-1'}),
                _dated('after', 'y', '2026-01-02T00:00:00+00:00', metadata={'topic': 'venv'}),
            ],
        }
        _metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})
        events = details['categories'][_PK]['consolidation_events']

        assert [e['memory_id'] for e in events] == ['s']
        assert events[0]['absorbed'] == 1, 'a scalar supersedes absorbed one predecessor'
        assert events[0]['net_delta'] == 0

    def test_canonical_without_supersedes_absorbed_nothing_recorded(self):
        """`canonical: true` names no predecessors, so absorbed is 0, not a guess."""
        records = {
            _PK: [
                _dated('c', 'x', '2026-01-01T00:00:00+00:00',
                       metadata={'topic': 'venv', 'canonical': True}),
            ],
        }
        _metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})
        assert details['categories'][_PK]['consolidation_events'][0]['absorbed'] == 0

    def test_untopiced_event_uses_its_cluster_as_the_peer_set(self):
        records = {
            _PK: [
                _dated('c', 'x', '2026-01-01T00:00:00+00:00', metadata={'canonical': True}),
                _dated('sib', 'y', '2026-01-02T00:00:00+00:00'),
                _dated('far', 'z', '2026-01-03T00:00:00+00:00'),
            ],
        }
        plans = {_PK: _plan_slice([{'member_ids': ['c', 'sib'], 'category': _PK}])}
        _metrics, details = compute_cluster_metrics(records, plans, {})
        event = details['categories'][_PK]['consolidation_events'][0]

        assert event['accreted_after'] == 1, "'far' is outside the event's cluster"

    def test_no_consolidation_events_still_emits_a_well_formed_metric(self):
        records = {_PK: [_dated('a', 'x', '2026-01-01T00:00:00+00:00')]}
        metrics, details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})

        assert details['categories'][_PK]['consolidation_events'] == []
        metric = _by_id(metrics)[f'consolidation_net_delta.{_PK}']
        assert metric.n == 0
        assert metric.value == 0.0


class TestComputeClusterMetricsDisclosures:
    """Every cap and every loss is its own emitted metric. No cap is silent."""

    def test_global_ann_disclosures_each_become_a_metric(self):
        disclosures = {
            'top_k_saturated': 3,
            'below_threshold_dropped': 41,
            'unknown_neighbor_dropped': 2,
            'missing_vector': 1,
            'ann_disabled_uncalibrated': 0,
            'ann_query_errors': 0,
        }
        records = {_PK: [_dated('a', 'x', None), _dated('b', 'y', None)]}
        metrics, _details = compute_cluster_metrics(
            records, {_PK: _plan_slice()}, disclosures,
        )
        by_id = _by_id(metrics)

        for key, expected in disclosures.items():
            metric = by_id[f'{key}.{_GLOBAL_SCOPE}']
            assert metric.kind == 'count', key
            assert metric.value == expected, key
            assert metric.direction == 'higher_is_worse', key
            assert metric.n == 2, f'{key}: n is the total corpus exposure'

    def test_zero_valued_disclosures_are_still_emitted(self):
        """A clean run must publish the zero.

        An absent metric is indistinguishable from "not measured" — the M2
        evaluator would read a suppressed zero as a gap in the series rather
        than as evidence the cap did not fire.
        """
        records = {_PK: [_dated('a', 'x', None)]}
        metrics, _details = compute_cluster_metrics(
            records, {_PK: _plan_slice()}, dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        by_id = _by_id(metrics)

        for key in _ANN_DISCLOSURE_KEYS:
            assert by_id[f'{key}.{_GLOBAL_SCOPE}'].value == 0

    def test_per_category_disclosure_emits_one_metric_per_category(self):
        """scan_truncated is per category: a cap on one is not a clean whole."""
        records = {
            _PK: [_dated('a', 'x', None)],
            _PN: [_dated('b', 'y', None, category=_PN), _dated('c', 'z', None, category=_PN)],
        }
        metrics, _details = compute_cluster_metrics(
            records,
            {_PK: _plan_slice(), _PN: _plan_slice()},
            {'scan_truncated': {_PK: 1, _PN: 0}},
        )
        by_id = _by_id(metrics)

        assert by_id[f'scan_truncated.{_PK}'].value == 1
        assert by_id[f'scan_truncated.{_PK}'].n == 1
        assert by_id[f'scan_truncated.{_PN}'].value == 0
        assert by_id[f'scan_truncated.{_PN}'].n == 2
        assert f'scan_truncated.{_GLOBAL_SCOPE}' not in by_id


class TestComputeClusterMetricsEmptyCorpus:
    """An empty corpus reports; it does not raise and it does not lie."""

    def test_empty_category_emits_well_formed_zero_metrics(self):
        metrics, details = compute_cluster_metrics(
            {_PK: []}, {_PK: _plan_slice()}, {},
        )
        by_id = _by_id(metrics)

        assert by_id[f'near_duplicate_clusters.{_PK}'].value == 0
        assert by_id[f'near_duplicate_clusters.{_PK}'].n == 0
        assert by_id[f'cluster_size_max.{_PK}'].value == 0.0
        assert by_id[f'cluster_size_mean.{_PK}'].value == 0.0
        assert details['categories'][_PK]['corpus_size'] == 0

    def test_share_over_an_empty_corpus_is_disclosed_not_faked(self):
        """0/0 is undefined, and M1 forbids a zero-trial proportion.

        Emitting value=0.0 would assert "0% of this corpus is duplicated",
        which is a claim about a corpus that was not measured. The metric is
        withheld and the withholding is named in details — loud, not silent.
        """
        metrics, details = compute_cluster_metrics({_PK: []}, {_PK: _plan_slice()}, {})

        assert f'clustered_member_share.{_PK}' not in _by_id(metrics)
        assert f'clustered_member_share.{_PK}' in (
            details['categories'][_PK]['undefined_metrics']
        )

    def test_no_categories_at_all_yields_an_empty_but_valid_result(self):
        metrics, details = compute_cluster_metrics({}, {}, {})
        assert metrics == []
        assert details['categories'] == {}


class TestComputeClusterMetricsDetailsPath:
    """Metrics whose full shape lives in the companion file point at it."""

    def test_details_path_is_stamped_onto_the_distribution_metrics(self):
        records = {_PK: [_dated('a', 'x', None), _dated('b', 'y', None)]}
        plans = {_PK: _plan_slice([{'member_ids': ['a', 'b'], 'category': _PK}])}
        metrics, _details = compute_cluster_metrics(
            records, plans, {}, details_path='details-20260730T000000Z.json',
        )
        by_id = _by_id(metrics)

        for name in ('cluster_size_max', 'cluster_size_mean',
                     'topic_accretion_rate_mean', 'consolidation_net_delta'):
            assert by_id[f'{name}.{_PK}'].details_path == 'details-20260730T000000Z.json', name

    def test_details_path_defaults_to_absent(self):
        records = {_PK: [_dated('a', 'x', None)]}
        metrics, _details = compute_cluster_metrics(records, {_PK: _plan_slice()}, {})
        assert _by_id(metrics)[f'cluster_size_max.{_PK}'].details_path is None


class TestBuildMetricSeries:
    """The assembled series is the M1 artifact — validated before it is written."""

    def _metrics(self):
        records = {_PK: [_dated('a', 'x', None), _dated('b', 'y', None)]}
        plans = {_PK: _plan_slice([{'member_ids': ['a', 'b'], 'category': _PK}])}
        return compute_cluster_metrics(records, plans, {})[0]

    def test_series_shape_and_corpus_counts(self):
        from shared.memory_eval_metrics import validate_metric_series  # noqa: PLC0415

        series = build_metric_series(
            'dark_factory', self._metrics(),
            corpus_counts={_PK: 2, _PN: 0, _OS: 5},
            run_stamp='20260730T000000Z',
        )

        assert series.schema_version == 1
        assert series.eval_id == _EVAL_ID == 'e6-corpus-health'
        assert series.run_stamp == '20260730T000000Z'
        assert series.corpus.project_id == 'dark_factory'
        assert series.corpus.counts == {_PK: 2, _PN: 0, _OS: 5}
        validate_metric_series(series)

    def test_metric_ids_are_unique_and_category_suffixed(self):
        series = build_metric_series(
            'dark_factory', self._metrics(), corpus_counts={_PK: 2},
            run_stamp='20260730T000000Z',
        )
        ids = [m.metric_id for m in series.metrics]

        assert len(ids) == len(set(ids))
        assert all(i.rsplit('.', 1)[-1] in (*_ALL_CATEGORIES, _GLOBAL_SCOPE) for i in ids)

    def test_run_stamp_defaults_to_the_shared_helper(self, monkeypatch):
        from shared.memory_eval_metrics import RUN_STAMP_ENV_VAR  # noqa: PLC0415

        monkeypatch.setenv(RUN_STAMP_ENV_VAR, '20260101T010101Z')
        series = build_metric_series('dark_factory', self._metrics(), corpus_counts={_PK: 2})
        assert series.run_stamp == '20260101T010101Z'

    def test_malformed_metric_raises_metric_schema_error_at_build_time(self):
        """M1: rejected in the runner that computed it, not by the reader."""
        from shared.memory_eval_metrics import MetricSchemaError  # noqa: PLC0415

        malformed = {
            'metric_id': f'clustered_member_share.{_PK}',
            'kind': 'proportion',
            'value': 0.5,
            'n': 0,
            'denominator': 0,  # a proportion over zero trials
            'direction': 'higher_is_worse',
        }
        with pytest.raises(MetricSchemaError):
            build_metric_series('dark_factory', [malformed], corpus_counts={_PK: 0})

    def test_duplicate_metric_id_raises_metric_schema_error(self):
        """A duplicate would make one of the two invisible to the baseline join."""
        from shared.memory_eval_metrics import MetricSchemaError  # noqa: PLC0415

        one = {'metric_id': 'dup.all', 'kind': 'scalar', 'value': 1.0, 'n': 1}
        with pytest.raises(MetricSchemaError):
            build_metric_series('dark_factory', [one, dict(one)], corpus_counts={})

    def test_missing_direction_on_a_count_raises(self):
        """The exact test is two-sided; without a direction an improvement alarms."""
        from shared.memory_eval_metrics import MetricSchemaError  # noqa: PLC0415

        with pytest.raises(MetricSchemaError):
            build_metric_series(
                'dark_factory',
                [{'metric_id': 'x.all', 'kind': 'count', 'value': 3.0, 'n': 10}],
                corpus_counts={},
            )

    def test_computed_metrics_round_trip_through_the_schema(self):
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            parse_metric_series,
            serialize_metric_series,
        )

        series = build_metric_series(
            'dark_factory', self._metrics(), corpus_counts={_PK: 2},
            run_stamp='20260730T000000Z',
        )
        reparsed = parse_metric_series(json.loads(serialize_metric_series(series)))
        assert reparsed == series


# ===========================================================================
# M1 artifact emission + CLI back-compat (task 3210)
# ===========================================================================

_STAMP = '20260730T112233Z'


def _series(project_id: str = 'dark_factory', stamp: str = _STAMP):
    records = {_PK: [_dated('a', 'x', None), _dated('b', 'y', None)]}
    plans = {_PK: _plan_slice([{'member_ids': ['a', 'b'], 'category': _PK}])}
    metrics, _details = compute_cluster_metrics(records, plans, {})
    return build_metric_series(
        project_id, metrics, corpus_counts={_PK: 2}, run_stamp=stamp,
    )


class TestEmitMetricsArtifact:
    """The producer side of the M1 boundary: three files, one stamp."""

    def test_writes_metrics_report_and_details(self, tmp_path):
        metrics_path, report_path, det_path = emit_metrics_artifact(
            _series(), {'categories': {}}, tmp_path, _STAMP,
        )

        assert metrics_path == tmp_path / _EVAL_ID / f'metrics-{_STAMP}.json'
        assert report_path == tmp_path / _EVAL_ID / f'report-{_STAMP}.txt'
        assert det_path == tmp_path / _EVAL_ID / f'details-{_STAMP}.json'
        assert metrics_path.is_file()
        assert report_path.is_file()
        assert det_path.is_file()

    def test_written_metrics_round_trip_through_the_shared_loader(self, tmp_path):
        """The artifact is read by a process that never imports this script."""
        from shared.memory_eval_metrics import load_metric_series  # noqa: PLC0415

        series = _series()
        metrics_path, _report, _details = emit_metrics_artifact(
            series, {'categories': {}}, tmp_path, _STAMP,
        )
        assert load_metric_series(metrics_path) == series

    def test_details_payload_round_trips_as_json(self, tmp_path):
        details = {'categories': {_PK: {'cluster_size_histogram': {'2': 1}}}}
        _m, _r, det_path = emit_metrics_artifact(_series(), details, tmp_path, _STAMP)
        assert json.loads(det_path.read_text()) == details

    def test_details_path_helper_matches_what_is_written(self, tmp_path):
        """The name stamped onto a Metric.details_path must be the file's name."""
        _m, _r, det_path = emit_metrics_artifact(
            _series(), {'categories': {}}, tmp_path, _STAMP,
        )
        assert det_path.name == details_artifact_path(_STAMP)

    def test_run_stamp_env_var_pins_the_filenames(self, tmp_path, monkeypatch):
        from shared.memory_eval_metrics import RUN_STAMP_ENV_VAR, run_stamp  # noqa: PLC0415

        monkeypatch.setenv(RUN_STAMP_ENV_VAR, '20260101T000000Z')
        stamp = run_stamp()
        metrics_path, _r, det_path = emit_metrics_artifact(
            _series(stamp=stamp), {'categories': {}}, tmp_path, stamp,
        )

        assert metrics_path.name == 'metrics-20260101T000000Z.json'
        assert det_path.name == 'details-20260101T000000Z.json'

    def test_malformed_metric_leaves_no_directory_and_no_partial_artifact(self, tmp_path):
        """Validate BEFORE mkdir: a bad run must not leave a half-artifact.

        A stray directory containing only details-<STAMP>.json would look to
        the limits evaluator like a run whose metrics file went missing —
        strictly worse than the run never having written anything.
        """
        from shared.memory_eval_metrics import MetricSchemaError  # noqa: PLC0415

        malformed = {
            'schema_version': 1,
            'eval_id': _EVAL_ID,
            'run_stamp': _STAMP,
            'corpus': {'project_id': 'dark_factory', 'counts': {}},
            'metrics': [{'metric_id': 'x.all', 'kind': 'count', 'value': -1.0, 'n': 1,
                         'direction': 'higher_is_worse'}],
        }
        with pytest.raises(MetricSchemaError):
            emit_metrics_artifact(malformed, {'categories': {}}, tmp_path, _STAMP)

        assert not (tmp_path / _EVAL_ID).exists(), 'no directory may survive a rejected series'
        assert list(tmp_path.iterdir()) == []


class TestCliBackCompat:
    """Task 3136 schedules this script; today's invocation must keep working."""

    def test_bare_project_id_still_parses_with_every_new_flag_defaulted(self):
        args = _build_parser().parse_args(['--project-id', 'dark_factory'])

        assert args.project_id == 'dark_factory'
        assert args.apply is False, 'dry run is still the default'
        assert args.threshold == 0.85
        assert args.scan_limit == 5000
        assert args.config is None
        # Every flag added by this task is optional and defaulted.
        assert list(args.categories) == list(_ALL_CATEGORIES)
        assert args.ann_threshold is None, 'defaults to the CALIBRATED value, not a literal'
        assert isinstance(args.ann_top_k, int) and args.ann_top_k > 0
        assert args.eval_id == _EVAL_ID
        assert args.no_metrics is False
        assert args.metrics_root is not None

    def test_apply_flag_still_parses(self):
        args = _build_parser().parse_args(['--project-id', 'dark_factory', '--apply'])
        assert args.apply is True

    def test_metrics_root_defaults_under_the_scripts_package_not_the_cwd(self):
        """A scheduled run's cwd is not guaranteed; the artifact root must be."""
        args = _build_parser().parse_args(['--project-id', 'dark_factory'])
        assert Path(args.metrics_root).is_absolute()
        assert Path(args.metrics_root).parts[-2:] == ('data', 'memory-evals')

    def test_new_flags_accept_overrides(self):
        args = _build_parser().parse_args([
            '--project-id', 'p', '--categories', _PK, _OS,
            '--ann-top-k', '3', '--ann-threshold', '0.91',
            '--metrics-root', '/tmp/x', '--eval-id', 'other', '--no-metrics',
        ])
        assert list(args.categories) == [_PK, _OS]
        assert args.ann_top_k == 3
        assert args.ann_threshold == 0.91
        assert args.metrics_root == '/tmp/x'
        assert args.eval_id == 'other'
        assert args.no_metrics is True


# --- _run wiring doubles -----------------------------------------------------

def _fake_config(t_high: float | None = 0.9):
    return types.SimpleNamespace(
        write_triage=types.SimpleNamespace(t_high=t_high),
        mem0=types.SimpleNamespace(collection_prefix='mem0'),
    )


class _FakeQdrant:
    def __init__(self):
        self.calls = []

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(points=[])


class _FakeMem0:
    def __init__(self, raw_by_category: dict[str, list[dict]], qdrant: _FakeQdrant):
        self._raw = raw_by_category
        self._qdrant = qdrant
        self.scroll_calls = []

    async def scroll_by_metadata(self, scope, filters, limit=1000, *, with_vectors=False):
        self.scroll_calls.append({'filters': filters, 'limit': limit,
                                  'with_vectors': with_vectors})
        return list(self._raw.get(filters.get('category'), []))

    async def _get_async_qdrant(self):
        return self._qdrant


class _FakeMemoryService:
    """Stand-in for MemoryService with the surface _run actually touches."""

    instances: list = []
    raw_by_category: dict[str, list[dict]] = {}

    def __init__(self, config):
        self.config = config
        self.mem0 = _FakeMem0(type(self).raw_by_category, _FakeQdrant())
        self.deleted: list = []
        self.closed = False
        type(self).instances.append(self)

    async def initialize(self):
        return None

    async def close(self):
        self.closed = True

    async def delete_memory(self, memory_id, store=None, project_id=None):
        self.deleted.append(memory_id)


def _install_run_doubles(monkeypatch, raw_by_category, *, t_high: float | None = 0.9):
    import fused_memory.config.schema as schema_mod  # noqa: PLC0415
    import fused_memory.services.memory_service as service_mod  # noqa: PLC0415

    _FakeMemoryService.instances = []
    _FakeMemoryService.raw_by_category = raw_by_category
    monkeypatch.setattr(schema_mod, 'FusedMemoryConfig', lambda: _fake_config(t_high))
    monkeypatch.setattr(service_mod, 'MemoryService', _FakeMemoryService)
    return _FakeMemoryService


def _raw(id: str, content: str, created_at: str = '2026-01-01T00:00:00+00:00',
         category: str = _PK, extra: dict | None = None) -> dict:
    return {
        'id': id,
        'created_at': created_at,
        'metadata': {'data': content, 'category': category, **(extra or {})},
        'vector': [0.1, 0.2],
    }


@pytest.mark.asyncio
class TestRunApplyGuards:
    """--apply never fires against a scan that cannot be trusted."""

    async def _invoke(self, monkeypatch, raw, argv, tmp_path):
        _install_run_doubles(monkeypatch, raw)
        args = _build_parser().parse_args(
            [*argv, '--metrics-root', str(tmp_path)],
        )
        return await _run(args), _FakeMemoryService.instances[-1]

    async def test_dry_run_is_the_default_and_deletes_nothing(self, monkeypatch, tmp_path):
        raw = {_PK: [_raw('m1', _VENV_GOTCHA_A), _raw('m2', _VENV_GOTCHA_B)]}
        rc, service = await self._invoke(
            monkeypatch, raw, ['--project-id', 'p', '--threshold', '0.75'], tmp_path,
        )
        assert rc == 0
        assert service.deleted == []

    async def test_apply_refuses_on_an_empty_scan(self, monkeypatch, tmp_path):
        rc, service = await self._invoke(
            monkeypatch, {}, ['--project-id', 'p', '--apply'], tmp_path,
        )
        assert rc == 1, 'an empty scan must never reach deletions'
        assert service.deleted == []

    async def test_apply_refuses_on_a_truncated_category_scan(self, monkeypatch, tmp_path):
        """Truncation on ONE category still refuses the whole apply.

        A truncated scan means the survivor may be sitting outside the window,
        so 'delete the losers' can delete the last copy.
        """
        raw = {_PK: [_raw('m1', _VENV_GOTCHA_A), _raw('m2', _VENV_GOTCHA_B)]}
        rc, service = await self._invoke(
            monkeypatch, raw,
            ['--project-id', 'p', '--apply', '--scan-limit', '2', '--threshold', '0.75'],
            tmp_path,
        )
        assert rc == 1
        assert service.deleted == []

    async def test_apply_deletes_the_losers_of_a_real_cluster(self, monkeypatch, tmp_path):
        raw = {_PK: [
            _raw('m1', _VENV_GOTCHA_A, '2026-01-01T00:00:00+00:00'),
            _raw('m2', _VENV_GOTCHA_B, '2026-01-02T00:00:00+00:00'),
        ]}
        rc, service = await self._invoke(
            monkeypatch, raw, ['--project-id', 'p', '--apply', '--threshold', '0.75'],
            tmp_path,
        )
        assert rc == 0
        assert service.deleted == ['m2'], 'the oldest survives'

    async def test_topic_carved_out_cluster_is_never_deleted_under_apply(
        self, monkeypatch, tmp_path,
    ):
        """3136's Option-C carve-out survives all the way to the delete call."""
        raw = {_PK: [
            _raw('m1', _VENV_GOTCHA_A, '2026-01-01T00:00:00+00:00',
                 extra={'topic': 'venv'}),
            _raw('m2', _VENV_GOTCHA_B, '2026-01-02T00:00:00+00:00',
                 extra={'topic': 'venv'}),
        ]}
        rc, service = await self._invoke(
            monkeypatch, raw, ['--project-id', 'p', '--apply', '--threshold', '0.75'],
            tmp_path,
        )
        assert rc == 1, 'nothing actionable remains, so the empty-plan guard holds'
        assert service.deleted == []


@pytest.mark.asyncio
class TestRunWiring:
    """The wiring _run performs, asserted where a silent regression would hide."""

    async def _parse(self, tmp_path, *argv):
        return _build_parser().parse_args(
            ['--project-id', 'p', '--metrics-root', str(tmp_path), *argv],
        )

    async def test_all_three_categories_are_scrolled_with_vectors(
        self, monkeypatch, tmp_path,
    ):
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]})
        await _run(await self._parse(tmp_path))
        calls = _FakeMemoryService.instances[-1].mem0.scroll_calls

        assert [c['filters']['category'] for c in calls] == list(_ALL_CATEGORIES)
        assert all(c['with_vectors'] is True for c in calls), (
            'ANN candidate generation needs the stored vectors'
        )

    async def test_threshold_tunes_the_lexical_path_only(self, monkeypatch, tmp_path):
        """--threshold and the ANN cutoff are separate knobs with separate homes."""
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]}, t_high=0.9)
        captured = {}
        real = _mod.build_sweep_plan

        def _spy(memories, threshold=0.85, **kwargs):
            captured.update({'threshold': threshold, **kwargs})
            return real(memories, threshold, **kwargs)

        monkeypatch.setattr(_mod, 'build_sweep_plan', _spy)
        await _run(await self._parse(tmp_path, '--threshold', '0.99'))

        assert captured['threshold'] == 0.99
        assert captured['ann_threshold'] == 0.9, 'the ANN cutoff came from the calibration'

    async def test_ann_threshold_override_reaches_the_query(self, monkeypatch, tmp_path):
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]}, t_high=0.9)
        await _run(await self._parse(tmp_path, '--ann-threshold', '0.77'))
        qdrant = _FakeMemoryService.instances[-1].mem0._qdrant

        assert qdrant.calls, 'a record with a vector must be queried'
        assert qdrant.calls[0]['score_threshold'] == 0.77

    async def test_uncalibrated_config_disables_ann_but_still_sweeps(
        self, monkeypatch, tmp_path, capsys,
    ):
        """No cutoff: skip the ANN queries, count it, and keep the lexical path."""
        _install_run_doubles(
            monkeypatch,
            {_PK: [_raw('m1', _VENV_GOTCHA_A), _raw('m2', _VENV_GOTCHA_B)]},
            t_high=None,
        )
        rc = await _run(await self._parse(tmp_path, '--threshold', '0.75'))
        plan = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert _FakeMemoryService.instances[-1].mem0._qdrant.calls == [], (
            'no cutoff means no ANN query — never a guessed one'
        )
        assert plan['ann_threshold'] is None
        assert plan['ann_disclosure']['ann_disabled_uncalibrated'] == 1
        assert plan['clusters_total'] == 1, 'the lexical path still ran'

    async def test_plan_json_is_printed_to_stdout(self, monkeypatch, tmp_path, capsys):
        """3136 consumes stdout; the report contract is unchanged."""
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]})
        await _run(await self._parse(tmp_path))
        plan = json.loads(capsys.readouterr().out)

        assert {'clusters_total', 'near_duplicate_groups', 'delete_candidates'} <= set(plan)

    async def test_metrics_artifact_is_emitted_under_metrics_root(
        self, monkeypatch, tmp_path,
    ):
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            RUN_STAMP_ENV_VAR,
            load_metric_series,
        )

        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _STAMP)
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]})
        await _run(await self._parse(tmp_path))

        series = load_metric_series(tmp_path / _EVAL_ID / f'metrics-{_STAMP}.json')
        assert series.corpus.project_id == 'p'
        assert series.corpus.counts == {_PK: 1, _PN: 0, _OS: 0}
        assert (tmp_path / _EVAL_ID / f'details-{_STAMP}.json').is_file()

        ids = {m.metric_id for m in series.metrics}
        assert f'near_duplicate_clusters.{_PK}' in ids
        for key in (*_ANN_DISCLOSURE_KEYS, 'ann_query_errors', 'ann_disabled_uncalibrated'):
            assert f'{key}.{_GLOBAL_SCOPE}' in ids, f'{key} must be disclosed'
        assert f'scan_truncated.{_PK}' in ids

    async def test_no_metrics_writes_nothing(self, monkeypatch, tmp_path):
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]})
        await _run(await self._parse(tmp_path, '--no-metrics'))
        assert list(tmp_path.iterdir()) == []

    async def test_memory_service_is_always_closed(self, monkeypatch, tmp_path):
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', 'x')]})
        await _run(await self._parse(tmp_path))
        assert _FakeMemoryService.instances[-1].closed is True
