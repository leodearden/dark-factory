"""Tests for audit_duplicate_memories.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — mirrors the pattern in test_audit_duplicate_tasks.py.
"""
from __future__ import annotations

import importlib.util
import json
import types
from datetime import datetime
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
ann_scores_for_pairs = _mod.ann_scores_for_pairs
_ANN_DISCLOSURE_KEYS = _mod._ANN_DISCLOSURE_KEYS
find_near_duplicate_memory_groups = _mod.find_near_duplicate_memory_groups
pick_survivor = _mod.pick_survivor
build_sweep_plan = _mod.build_sweep_plan
restrict_delete_candidates_for_apply = _mod.restrict_delete_candidates_for_apply
_PLAN_DISCLOSURE_KEYS = _mod._PLAN_DISCLOSURE_KEYS
_MAX_LEXICAL_RATIO_MEMBERS = _mod._MAX_LEXICAL_RATIO_MEMBERS
partition_topic_carveout = _mod.partition_topic_carveout
fetch_procedural_memories = _mod.fetch_procedural_memories
fetch_memories = _mod.fetch_memories
_ALL_CATEGORIES = _mod._ALL_CATEGORIES
extract_liveness_snapshot_fact = _mod.extract_liveness_snapshot_fact
liveness_snapshot_subject_task_ids = _mod.liveness_snapshot_subject_task_ids
find_liveness_snapshot_recurrences = _mod.find_liveness_snapshot_recurrences
_LIVENESS_DISCLOSURE_KEYS = _mod._LIVENESS_DISCLOSURE_KEYS
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
# ANN score back-mapping survives neighbour-list ASYMMETRY (task 3210)
# ===========================================================================

class TestAnnScoresForPairs:
    """Recover each pair's cosine from EITHER endpoint's neighbour list.

    ``ann_pairs_from_neighbors`` emits canonical ``(low, high)`` pairs
    discovered from either side, so "the left endpoint's list names the
    right" is a FALSE invariant. Two states this detector explicitly expects
    and counts break it:

      - ``top_k_saturated`` — the right lists the left while the left's cap
        is filled by nearer points.
      - ``missing_vector`` — a vector-less record is never a query point
        (``fetch_ann_neighbors`` skips it) yet is still reachable as another
        record's hit.

    Scanning one side silently mis-reports precisely the cases the
    disclosure counters exist to make legible, and defaulting the miss to
    0.0 compounds it into a fabricated fact: 0.0 is not None, so it passes
    ``build_sweep_plan``'s guard and publishes a score BELOW the cutoff that
    produced the cluster — in the stdout report task 3136 consumes, and the
    figure an operator uses to sanity-check an irreversible ``--apply``.

    Pure: no I/O, following ``TestAnnPairsFromNeighbors``'s style. The "real
    score" is step-7's ``_MEASURED_COSINE`` — a measured number from the
    curator-labeled fixture, not an invented one.
    """

    # Mutually dissimilar contents, so the LEXICAL path never clusters these
    # and every cluster under test is unambiguously ANN-attributed.
    _CONTENTS = (
        'quantum harmonic oscillator eigenstates',
        'baking sourdough bread at home',
        'tidal patterns along rocky coastlines',
        'jazz improvisation over modal changes',
    )

    def _memories(self, n: int) -> list[dict]:
        out = []
        for i in range(n):
            m = _memory(f'm{i}', self._CONTENTS[i])
            m['vector'] = [0.1 * i, 0.2]
            out.append(m)
        return out

    def test_score_recovered_when_only_the_right_endpoint_listed_the_pair(self):
        """top_k saturation: the left's slots are filled by NEARER points.

        m0's two neighbour slots go to m2/m3, so m1 never appears in m0's
        list even though the pair cleared the cutoff. This is the counted
        ``top_k_saturated`` state, not a malfunction — the score is still on
        record, from m1's side.
        """
        memories = self._memories(4)
        neighbors = {
            'm0': [_hit('m2', 0.99), _hit('m3', 0.98)],  # cap full, m1 crowded out
            'm1': [_hit('m0', _MEASURED_COSINE)],
            'm2': [],
            'm3': [],
        }
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert scores[(0, 1)] == pytest.approx(_MEASURED_COSINE, abs=1e-12), (
            f'the real cosine must be recovered from the RIGHT endpoint; '
            f'got {scores.get((0, 1))!r} (0.0 means only the left was scanned)'
        )

    def test_score_recovered_when_the_left_endpoint_has_no_vector(self):
        """A vector-less record is never a query point, yet is still a hit.

        ``fetch_ann_neighbors`` skips it, so it is absent from *neighbors*
        entirely and counted as ``missing_vector`` — but m1's query found it,
        which is exactly how the pair came to exist.
        """
        memories = self._memories(2)
        memories[0]['vector'] = None
        neighbors = {'m1': [_hit('m0', _MEASURED_COSINE)]}  # no 'm0' key at all
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert scores[(0, 1)] == pytest.approx(_MEASURED_COSINE, abs=1e-12), (
            f'a left endpoint absent from the mapping must not zero the score; '
            f'got {scores.get((0, 1))!r}'
        )

    def test_max_is_returned_when_both_directions_scored_the_pair(self):
        memories = self._memories(2)
        neighbors = {
            'm0': [_hit('m1', _MEASURED_COSINE - 0.01)],
            'm1': [_hit('m0', _MEASURED_COSINE)],
        }
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert scores[(0, 1)] == pytest.approx(_MEASURED_COSINE, abs=1e-12)

    def test_pair_neither_endpoint_scored_is_absent_not_zero(self):
        """Absence, never a 0.0 default — that is the load-bearing half.

        ``build_sweep_plan`` already degrades an unscored pair to
        ``ann_max_score: null``; omitting the key reuses that path instead of
        introducing a second sentinel that lies about the measurement.
        """
        memories = self._memories(3)
        neighbors = {'m0': [_hit('m2', 0.99)], 'm1': [], 'm2': []}
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert (0, 1) not in scores, (
            f'an unrecoverable score must be ABSENT so the plan reports null; '
            f'got {scores.get((0, 1))!r}'
        )

    def test_self_hit_never_supplies_a_pair_score(self):
        """A record is its own nearest neighbour; that resolves to itself, not the partner."""
        memories = self._memories(2)
        neighbors = {'m0': [_hit('m0', 1.0)], 'm1': [_hit('m1', 1.0)]}
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert (0, 1) not in scores, f'self-hits must not score the pair, got {scores!r}'

    def test_hit_without_a_score_is_not_treated_as_zero(self):
        """A scoreless hit is an unrecorded measurement, not a measurement of 0.0."""
        memories = self._memories(2)
        neighbors = {'m0': [{'id': 'm1'}], 'm1': []}
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert (0, 1) not in scores, f'expected the key to be absent, got {scores!r}'

    def test_unknown_neighbour_ids_are_ignored_without_raising(self):
        memories = self._memories(2)
        neighbors = {'m0': [_hit('not-in-scan', 0.99), _hit('m1', _MEASURED_COSINE)]}
        scores = ann_scores_for_pairs(memories, neighbors, [(0, 1)])
        assert scores[(0, 1)] == pytest.approx(_MEASURED_COSINE, abs=1e-12)

    def test_empty_pairs_produce_an_empty_mapping(self):
        memories = self._memories(2)
        assert ann_scores_for_pairs(memories, {'m0': [_hit('m1', 0.99)]}, []) == {}

    def test_input_memories_not_mutated(self):
        memories = self._memories(2)
        snapshot = [dict(m) for m in memories]
        ann_scores_for_pairs(memories, {'m1': [_hit('m0', 0.99)]}, [(0, 1)])
        assert memories == snapshot

    # -- end to end through build_sweep_plan -------------------------------

    def test_plan_reports_the_real_cosine_for_a_right_side_only_pair(self):
        """The regression, end to end: never publish a score below the cutoff.

        Runs the real generator over an asymmetric mapping (m0's query
        returned nothing), then back-maps and reports.
        """
        memories = self._memories(2)
        cutoff = _calibrated_ann_threshold()
        neighbors = {'m1': [_hit('m0', _MEASURED_COSINE)]}

        pairs, _disclosure = ann_pairs_from_neighbors(
            memories, neighbors, threshold=cutoff, top_k=10,
        )
        assert pairs == [(0, 1)], 'precondition: the right endpoint alone proposes the pair'

        plan = build_sweep_plan(
            memories,
            ann_pairs=pairs,
            ann_scores=ann_scores_for_pairs(memories, neighbors, pairs),
            ann_threshold=cutoff,
        )
        group = plan['near_duplicate_groups'][0]
        assert group['found_by'] == ['ann']
        assert group['ann_max_score'] == pytest.approx(_MEASURED_COSINE, abs=1e-12)
        assert group['ann_max_score'] >= cutoff, (
            f'reported score {group["ann_max_score"]!r} is below the cutoff '
            f'{cutoff!r} that produced the cluster — a self-contradicting number'
        )

    def test_plan_reports_null_when_no_endpoint_scored_the_pair(self):
        """A caller supplying ann_pairs without scores gets null, not 0.0."""
        memories = self._memories(2)
        plan = build_sweep_plan(
            memories,
            ann_pairs=[(0, 1)],
            ann_scores=ann_scores_for_pairs(memories, {}, [(0, 1)]),
            ann_threshold=0.88,
        )
        assert plan['clusters_total'] == 1
        assert plan['near_duplicate_groups'][0]['ann_max_score'] is None


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


class TestBuildSweepPlanRemapDisclosure:
    """A dropped ANN pair is COUNTED — the remap is not a silent loss path.

    Every other way the ANN path can lose a candidate is a metric; a pair
    naming two real records that the per-category remap refuses must be too,
    or a candidate that consumed a top_k slot vanishes with no accounting.
    """

    def test_cross_category_pair_is_dropped_and_counted(self):
        memories = [
            _memory('a', 'alpha text', category=_PK),
            _memory('b', 'beta text', category=_PN),
        ]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 1)],
            ann_disclosure=dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        assert plan['clusters_total'] == 0
        assert plan['ann_disclosure']['cross_category_dropped'] == 1
        assert plan['ann_disclosure']['unswept_category_dropped'] == 0

    def test_unswept_category_endpoint_is_counted_separately(self):
        """A pair reaching outside the swept set is a different loss."""
        memories = [
            _memory('a', 'alpha text', category=_PK),
            _memory('b', 'beta text', category='entities_and_relations'),
        ]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 1)],
            ann_disclosure=dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        assert plan['ann_disclosure']['unswept_category_dropped'] == 1
        assert plan['ann_disclosure']['cross_category_dropped'] == 0

    def test_a_dropped_pair_is_counted_once_not_once_per_category(self):
        """The remap loop visits every category; the count must not multiply."""
        memories = [
            _memory('a', 'alpha text', category=_PK),
            _memory('b', 'beta text', category=_PN),
        ]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 1)],
            categories=_ALL_CATEGORIES,
            ann_disclosure=dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        assert plan['ann_disclosure']['cross_category_dropped'] == 1

    def test_mirrored_duplicates_of_one_pair_count_once(self):
        memories = [
            _memory('a', 'alpha text', category=_PK),
            _memory('b', 'beta text', category=_PN),
        ]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 1), (1, 0), (0, 1)],
            ann_disclosure=dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        assert plan['ann_disclosure']['cross_category_dropped'] == 1

    def test_same_category_pair_is_not_counted_as_a_drop(self):
        memories = [
            _memory('a', _VENV_GOTCHA_A, category=_PK),
            _memory('b', _DISTRACTOR, category=_PK),
        ]
        plan = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1)],
            ann_disclosure=dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        assert plan['clusters_total'] == 1, 'the ANN pair clustered it'
        assert plan['ann_disclosure']['cross_category_dropped'] == 0
        assert plan['ann_disclosure']['unswept_category_dropped'] == 0

    def test_out_of_range_index_is_counted_not_raised(self):
        """A caller bug degrades to a counted drop, never an IndexError."""
        memories = [_memory('a', 'alpha text', category=_PK)]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 99)],
            ann_disclosure=dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0),
        )
        assert plan['ann_disclosure']['unswept_category_dropped'] == 1

    def test_the_callers_disclosure_dict_is_not_mutated(self):
        caller = dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0)
        memories = [
            _memory('a', 'alpha text', category=_PK),
            _memory('b', 'beta text', category=_PN),
        ]
        build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(0, 1)],
            ann_disclosure=caller,
        )
        assert caller == dict.fromkeys(_ANN_DISCLOSURE_KEYS, 0)

    def test_no_ann_input_at_all_keeps_the_disclosure_absent(self):
        """A caller that never used the ANN path still reports None."""
        plan = build_sweep_plan([_memory('a', 'x')], threshold=_THRESHOLD)
        assert plan['ann_disclosure'] is None


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
# The --apply gate: only chain-free evidence may DELETE
# ===========================================================================

def _grp(member_ids: list[str], losers: list[str], **flags) -> dict:
    """A near_duplicate_groups entry, only the keys the gate reads."""
    return {
        'member_ids': member_ids,
        'delete_candidate_ids': losers,
        'category': _PK,
        'lexical_spanning': False,
        'ann_clique': False,
        **flags,
    }


class TestRestrictDeleteCandidatesForApply:
    """The pure gate: what --apply may act on, and what it must withhold."""

    def test_a_lexically_spanning_cluster_is_actionable(self):
        groups = [_grp(['a', 'b'], ['b'], lexical_spanning=True)]
        actionable, withheld = restrict_delete_candidates_for_apply(groups)
        assert actionable == ['b']
        assert withheld == []

    def test_an_ann_clique_is_actionable(self):
        """Every member pair cleared the cutoff — no member is in by chaining."""
        groups = [_grp(['a', 'b'], ['b'], ann_clique=True)]
        actionable, withheld = restrict_delete_candidates_for_apply(groups)
        assert actionable == ['b']
        assert withheld == []

    def test_an_ann_chain_is_withheld_with_its_ids_and_a_reason(self):
        groups = [_grp(['a', 'b', 'c'], ['b', 'c'])]
        actionable, withheld = restrict_delete_candidates_for_apply(groups)

        assert actionable == [], 'membership rests on transitivity — never delete'
        assert len(withheld) == 1
        assert withheld[0]['withheld_ids'] == ['b', 'c']
        assert withheld[0]['member_ids'] == ['a', 'b', 'c']
        assert withheld[0]['reason'], 'the withholding must say why'

    def test_a_single_lexical_pair_does_not_unlock_an_ann_merged_cluster(self):
        """The exact case a naive `lexical_clustered` check would wave through.

        ANN merged two halves; one half happens to contain a lexical pair. The
        MERGE still rests entirely on ANN, so the cluster is not actionable.
        """
        groups = [_grp(['a', 'b', 'c', 'd'], ['b', 'c', 'd'],
                       lexical_clustered=True)]
        actionable, withheld = restrict_delete_candidates_for_apply(groups)
        assert actionable == []
        assert len(withheld) == 1

    def test_groups_with_no_losers_are_neither_actioned_nor_withheld(self):
        actionable, withheld = restrict_delete_candidates_for_apply(
            [_grp(['a'], [])],
        )
        assert actionable == []
        assert withheld == []

    def test_mixed_groups_partition_independently(self):
        groups = [
            _grp(['a', 'b'], ['b'], lexical_spanning=True),
            _grp(['c', 'd', 'e'], ['d', 'e']),
        ]
        actionable, withheld = restrict_delete_candidates_for_apply(groups)
        assert actionable == ['b']
        assert [g['member_ids'] for g in withheld] == [['c', 'd', 'e']]

    def test_empty_input(self):
        assert restrict_delete_candidates_for_apply([]) == ([], [])


class TestBuildSweepPlanApplyGateFlags:
    """build_sweep_plan measures the evidence the gate then judges."""

    def test_a_lexical_cluster_is_spanning_and_fully_actionable(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-01-01T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-01-02T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-01-03T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(memories, threshold=_THRESHOLD)
        group = plan['near_duplicate_groups'][0]

        assert group['lexical_spanning'] is True
        assert plan['apply_delete_candidates'] == plan['delete_candidates']
        assert plan['apply_withheld_clusters'] == 0

    def test_a_two_member_ann_only_cluster_is_a_clique(self):
        memories = [
            _memory('a', 'alpha one', created_at='2026-01-01T00:00:00+00:00'),
            _memory('b', 'beta two', created_at='2026-01-02T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1)], ann_threshold=0.9,
        )
        group = plan['near_duplicate_groups'][0]

        assert group['found_by'] == ['ann']
        assert group['lexical_spanning'] is False
        assert group['ann_clique'] is True
        assert plan['apply_delete_candidates'] == ['b']

    def test_a_chained_ann_cluster_is_not_a_clique_and_is_withheld(self):
        """A~B and B~C above the cutoff; A~C never was. Report yes, delete no."""
        memories = [
            _memory('a', 'alpha one', created_at='2026-01-01T00:00:00+00:00'),
            _memory('b', 'beta two', created_at='2026-01-02T00:00:00+00:00'),
            _memory('c', 'gamma three', created_at='2026-01-03T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1), (1, 2)], ann_threshold=0.9,
        )
        group = plan['near_duplicate_groups'][0]

        assert set(group['member_ids']) == {'a', 'b', 'c'}
        assert group['ann_clique'] is False
        assert plan['delete_candidates'] == ['b', 'c'], 'still reported'
        assert plan['apply_delete_candidates'] == [], 'but never deleted'
        assert plan['apply_withheld_clusters'] == 1

    def test_a_fully_connected_ann_triangle_is_a_clique(self):
        memories = [
            _memory('a', 'alpha one', created_at='2026-01-01T00:00:00+00:00'),
            _memory('b', 'beta two', created_at='2026-01-02T00:00:00+00:00'),
            _memory('c', 'gamma three', created_at='2026-01-03T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1), (1, 2), (0, 2)],
            ann_threshold=0.9,
        )
        assert plan['near_duplicate_groups'][0]['ann_clique'] is True
        assert plan['apply_delete_candidates'] == ['b', 'c']

    def test_an_ann_merge_of_two_lexical_pairs_is_withheld(self):
        """The compound risk: ANN welds two real lexical clusters into one."""
        memories = [
            _memory('a1', _VENV_GOTCHA_A, created_at='2026-01-01T00:00:00+00:00'),
            _memory('a2', _VENV_GOTCHA_B, created_at='2026-01-02T00:00:00+00:00'),
            _memory('b1', _DISTRACTOR, created_at='2026-01-03T00:00:00+00:00'),
            _memory('b2', _DISTRACTOR + ' Rerun nightly.',
                    created_at='2026-01-04T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(
            memories, threshold=_THRESHOLD, ann_pairs=[(1, 2)], ann_threshold=0.9,
        )
        group = plan['near_duplicate_groups'][0]

        assert set(group['member_ids']) == {'a1', 'a2', 'b1', 'b2'}
        assert group['lexical_clustered'] is True, 'lexical pairs exist inside it'
        assert group['lexical_spanning'] is False, 'but they do not span the merge'
        assert group['ann_clique'] is False
        assert plan['apply_delete_candidates'] == []
        assert plan['apply_withheld_clusters'] == 1

    def test_delete_candidate_ids_are_reported_per_group(self):
        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-01-01T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-01-02T00:00:00+00:00'),
        ]
        group = build_sweep_plan(memories, threshold=_THRESHOLD)['near_duplicate_groups'][0]
        assert group['delete_candidate_ids'] == ['m2']

    def test_plan_stays_json_serializable_with_the_gate_keys(self):
        memories = [
            _memory('a', 'alpha one', created_at='2026-01-01T00:00:00+00:00'),
            _memory('b', 'beta two', created_at='2026-01-02T00:00:00+00:00'),
            _memory('c', 'gamma three', created_at='2026-01-03T00:00:00+00:00'),
        ]
        plan = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1), (1, 2)], ann_threshold=0.9,
        )
        json.dumps(plan, default=str)


class TestLexicalMaxRatioReuseAndBound:
    """lexical_max_ratio is exact when it can be, and says so when it is not."""

    def test_reported_ratio_matches_an_exhaustive_recomputation(self):
        """The prefilter + seeded maximum must not change the answer."""
        import difflib  # noqa: PLC0415

        memories = [
            _memory('m1', _VENV_GOTCHA_A, created_at='2026-01-01T00:00:00+00:00'),
            _memory('m2', _VENV_GOTCHA_B, created_at='2026-01-02T00:00:00+00:00'),
            _memory('m3', _VENV_GOTCHA_C, created_at='2026-01-03T00:00:00+00:00'),
        ]
        group = build_sweep_plan(memories, threshold=_THRESHOLD)['near_duplicate_groups'][0]

        contents = [(m['content'] or '').strip().lower() for m in memories]
        expected = max(
            difflib.SequenceMatcher(None, contents[a], contents[b]).ratio()
            for a in range(len(contents))
            for b in range(a + 1, len(contents))
        )
        assert group['lexical_max_ratio'] == pytest.approx(expected)
        assert group['lexical_max_ratio_sampled'] is False

    def test_an_ann_only_cluster_still_reports_the_losing_paths_ratio(self):
        memories = [
            _memory('a', 'alpha one', created_at='2026-01-01T00:00:00+00:00'),
            _memory('b', 'beta two', created_at='2026-01-02T00:00:00+00:00'),
        ]
        group = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1)], ann_threshold=0.9,
        )['near_duplicate_groups'][0]

        assert group['lexical_max_ratio'] is not None, (
            'the disagreement between the two detectors is the point'
        )
        assert group['lexical_max_ratio'] < 0.99

    def test_an_oversized_cluster_bounds_the_scan_and_discloses_it(self):
        """A chained ANN cluster must not make a report-only field dominate."""
        size = _MAX_LEXICAL_RATIO_MEMBERS + 5
        memories = [
            _memory(f'm{i:03d}', f'record number {i} about venv setup',
                    created_at=f'2026-01-01T00:00:{i:02d}+00:00')
            for i in range(size)
        ]
        # A chain, not a clique: consecutive ANN pairs only.
        pairs = [(i, i + 1) for i in range(size - 1)]
        group = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=pairs, ann_threshold=0.9,
        )['near_duplicate_groups'][0]

        assert len(group['member_ids']) == size
        assert group['lexical_max_ratio_sampled'] is True, (
            'a bounded maximum must never be read as an exhaustive one'
        )
        assert group['lexical_max_ratio'] is not None

    def test_a_cluster_of_empty_content_reports_no_ratio(self):
        memories = [
            _memory('a', '', created_at='2026-01-01T00:00:00+00:00'),
            _memory('b', '', created_at='2026-01-02T00:00:00+00:00'),
        ]
        group = build_sweep_plan(
            memories, threshold=0.99, ann_pairs=[(0, 1)], ann_threshold=0.9,
        )['near_duplicate_groups'][0]
        assert group['lexical_max_ratio'] is None


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


@pytest.mark.asyncio
class TestFetchAnnNeighborsCategoryFilter:
    """The record's category is pushed DOWN, so top_k is not spent off-category.

    The collection holds every category. Unfiltered, a record's nearest
    ``top_k`` can be filled entirely by records this sweep will discard, and
    the same-category candidates below them are cut off — recall the ANN path
    exists to add. Filtering also makes a cross-category pair structurally
    impossible instead of merely dropped downstream.
    """

    def _service(self, query_points):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        client = MagicMock()
        client.query_points = query_points
        service = MagicMock()
        service.config.mem0.collection_prefix = 'mem0'
        service.mem0._get_async_qdrant = AsyncMock(return_value=client)
        return service

    def _empty(self):
        return types.SimpleNamespace(points=[])

    async def test_the_records_category_becomes_a_payload_equality_filter(self):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        from qdrant_client.http import models as qmodels  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._empty())
        records = [{'id': 'a', 'vector': [0.1], 'category': _PK}]

        await fetch_ann_neighbors(
            self._service(query_points), 'dark_factory', records,
            top_k=5, score_threshold=0.88,
        )

        assert query_points.await_args_list[0].kwargs['query_filter'] == qmodels.Filter(
            must=[qmodels.FieldCondition(
                key='category', match=qmodels.MatchValue(value=_PK),
            )],
        ), 'same Filter/FieldCondition/MatchValue shape scroll_by_metadata builds'

    async def test_each_record_is_filtered_to_its_own_category(self):
        from unittest.mock import AsyncMock  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._empty())
        records = [
            {'id': 'a', 'vector': [0.1], 'category': _PK},
            {'id': 'b', 'vector': [0.2], 'category': _OS},
        ]

        await fetch_ann_neighbors(
            self._service(query_points), 'dark_factory', records,
            top_k=5, score_threshold=0.88,
        )

        values = [
            call.kwargs['query_filter'].must[0].match.value
            for call in query_points.await_args_list
        ]
        assert values == [_PK, _OS]

    async def test_a_categoryless_record_is_queried_unfiltered(self):
        """It cannot be clustered at all, so the query shape is unchanged."""
        from unittest.mock import AsyncMock  # noqa: PLC0415

        query_points = AsyncMock(return_value=self._empty())
        records = [{'id': 'a', 'vector': [0.1]}]

        await fetch_ann_neighbors(
            self._service(query_points), 'dark_factory', records,
            top_k=5, score_threshold=0.88,
        )

        assert query_points.await_args_list[0].kwargs['query_filter'] is None


@pytest.mark.asyncio
class TestFetchAnnNeighborsConcurrency:
    """Queries fan out under a bound — never one blocking round-trip at a time."""

    def _service(self, query_points):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        client = MagicMock()
        client.query_points = query_points
        service = MagicMock()
        service.config.mem0.collection_prefix = 'mem0'
        service.mem0._get_async_qdrant = AsyncMock(return_value=client)
        return service

    async def test_in_flight_queries_are_concurrent_and_capped(self):
        import asyncio  # noqa: PLC0415
        from unittest.mock import AsyncMock  # noqa: PLC0415

        in_flight = 0
        peak = 0

        async def _side_effect(**_kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)  # yield, so overlap is observable
            in_flight -= 1
            return types.SimpleNamespace(points=[])

        query_points = AsyncMock(side_effect=_side_effect)
        records = [{'id': f'r{i}', 'vector': [float(i)]} for i in range(10)]

        await fetch_ann_neighbors(
            self._service(query_points), 'dark_factory', records,
            top_k=5, score_threshold=0.88, concurrency=3,
        )

        assert query_points.await_count == 10, 'every record is still queried'
        assert peak > 1, 'sequential issue would serialise 15k round-trips'
        assert peak <= 3, 'the bound protects the Qdrant the live path shares'

    async def test_results_are_keyed_in_input_order_despite_the_fan_out(self):
        """Two identical runs must serialise identically."""
        import asyncio  # noqa: PLC0415
        from unittest.mock import AsyncMock  # noqa: PLC0415

        async def _side_effect(**kwargs):
            # Later records finish FIRST, so insertion order cannot be
            # completion order.
            await asyncio.sleep(0.01 * (3 - kwargs['query'][0]))
            return types.SimpleNamespace(points=[])

        query_points = AsyncMock(side_effect=_side_effect)
        records = [{'id': f'r{i}', 'vector': [float(i)]} for i in range(3)]

        neighbors, _disclosure = await fetch_ann_neighbors(
            self._service(query_points), 'dark_factory', records,
            top_k=5, score_threshold=0.88,
        )

        assert list(neighbors) == ['r0', 'r1', 'r2']

    async def test_one_failure_is_counted_without_losing_the_rest(self):
        """Per-record error accounting survives the fan-out."""
        from unittest.mock import AsyncMock  # noqa: PLC0415

        async def _side_effect(**kwargs):
            if kwargs['query'] == [1.0]:
                raise RuntimeError('Qdrant blew up')
            return types.SimpleNamespace(
                points=[types.SimpleNamespace(id='z', score=0.95)],
            )

        query_points = AsyncMock(side_effect=_side_effect)
        records = [{'id': f'r{i}', 'vector': [float(i)]} for i in range(3)]

        neighbors, disclosure = await fetch_ann_neighbors(
            self._service(query_points), 'dark_factory', records,
            top_k=5, score_threshold=0.88, concurrency=3,
        )

        assert disclosure['ann_query_errors'] == 1
        assert 'r1' not in neighbors
        assert set(neighbors) == {'r0', 'r2'}


# ===========================================================================
# Cluster metrics + the M1 metric series (task 3210)
# ===========================================================================

def _dated(id: str, content: str, created_at: str | None, *,
           category: str = _PK, metadata: dict | None = None) -> dict:
    return _memory(id, content, created_at=created_at, category=category,
                   metadata=metadata or {})


def _plan_slice(groups: list[dict] | None = None,
                carveouts: list[dict] | None = None,
                liveness: list[dict] | None = None) -> dict:
    """A per-category plan slice, as produced by split_plan_by_category."""
    return {
        'near_duplicate_groups': groups or [],
        'topic_carveout_groups': carveouts or [],
        'liveness_snapshot_recurrence_groups': liveness or [],
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
        assert by_cat[_OS] == _plan_slice()


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

    def test_no_temp_file_survives_a_successful_write(self, tmp_path):
        """All three files go through the same temp-in-dir + os.replace writer."""
        emit_metrics_artifact(_series(), {'categories': {}}, tmp_path, _STAMP)
        written = sorted(p.name for p in (tmp_path / _EVAL_ID).iterdir())
        assert written == [
            f'details-{_STAMP}.json', f'metrics-{_STAMP}.json', f'report-{_STAMP}.txt',
        ]

    def test_a_details_write_failure_is_loud_and_leaves_nothing_behind(
        self, tmp_path, caplog,
    ):
        """Every metric names the details file; a half-written one is unreadable.

        A plain write_text could leave a truncated JSON document that a reader
        of cluster_size_max would follow and fail to parse. The atomic writer
        leaves the details file absent instead — diagnosable — and the failure
        is logged and re-raised rather than leaving a dangling details_path
        silently behind a successful-looking run.
        """
        import logging  # noqa: PLC0415

        artifact_dir = tmp_path / _EVAL_ID
        artifact_dir.mkdir(parents=True)
        # os.replace onto a directory raises IsADirectoryError (an OSError).
        (artifact_dir / details_artifact_path(_STAMP)).mkdir()

        with caplog.at_level(logging.ERROR), pytest.raises(OSError):
            emit_metrics_artifact(_series(), {'categories': {}}, tmp_path, _STAMP)

        assert 'FAILED to write its companion' in caplog.text
        assert [p.name for p in artifact_dir.iterdir() if p.suffix == '.tmp'] == [], (
            'a failed atomic write unlinks its temp file'
        )


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


class TestCategoriesFlagRejectsUnsupportedNames:
    """An unscrollable category must fail loudly, not report a clean zero sweep.

    Only the three Mem0-backed categories live in Qdrant. A Graphiti-backed or
    misspelled name scrolls nothing, so without a parse-time guard the run
    reports a spotless corpus it never actually looked at -- a silent no-op in
    a module whose whole ethos is that lost recall must be a counted metric.
    """

    @pytest.mark.parametrize('bad', [
        'decisions_and_rationale',  # Graphiti-backed, plausible, and wrong
        'entities_and_relations',
        'temporal_facts',
        'procedural-knowledge',     # hyphen typo
        'nonsense',
    ])
    def test_unsupported_category_exits_at_parse_time(self, bad, capsys):
        with pytest.raises(SystemExit) as exc:
            _build_parser().parse_args(['--project-id', 'p', '--categories', bad])

        assert exc.value.code == 2
        message = capsys.readouterr().err
        assert bad in message, 'the rejection must name the offending value'
        for supported in _ALL_CATEGORIES:
            assert supported in message, (
                f'the rejection must name the supported category {supported!r} '
                'so the operator can fix the invocation without reading source'
            )

    def test_a_valid_subset_is_still_accepted(self):
        """The guard rejects unknowns without narrowing legitimate use."""
        args = _build_parser().parse_args(
            ['--project-id', 'p', '--categories', _OS],
        )
        assert list(args.categories) == [_OS]


@pytest.mark.asyncio
class TestRepeatedCategoryDoesNotDoubleIngest:
    """A repeated --categories value must not make a record its own duplicate.

    Ingesting one category twice appends a second copy of every record under
    the SAME id. The copies are byte-identical, so they cluster at ratio 1.0
    and pick_survivor returns one copy as survivor and the other -- carrying
    that same id -- as a loser. Under --apply the id lands in delete_candidates
    and the only copy of the record is destroyed, breaking the documented
    "a survivor is always retained per cluster" carve-out. It also doubles
    every metric denominator.
    """

    async def test_fetch_memories_skips_an_already_scanned_category(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        memory = MagicMock()
        memory.mem0 = MagicMock()
        memory.mem0.scroll_by_metadata = _scroll_mock({
            _PK: [_raw('m1', _VENV_GOTCHA_A, '2026-01-01T00:00:00+00:00')],
        })

        records, scan_stats = await fetch_memories(
            memory, 'p', categories=(_PK, _PK, _PK),
        )

        assert memory.mem0.scroll_by_metadata.await_count == 1, (
            'a category already in scan_stats must not be re-scrolled'
        )
        assert [r['id'] for r in records] == ['m1'], (
            'the corpus must hold one copy per id, not one per repetition'
        )
        assert scan_stats[_PK]['scanned'] == 1, 'the denominator must not double'

    async def test_run_dedupes_before_the_scroll(self, monkeypatch, tmp_path):
        _install_run_doubles(monkeypatch, {_PK: [_raw('m1', _VENV_GOTCHA_A)]})
        args = _build_parser().parse_args([
            '--project-id', 'p', '--metrics-root', str(tmp_path),
            '--categories', _PK, _PN, _PK,
        ])
        await _run(args)

        calls = _FakeMemoryService.instances[-1].mem0.scroll_calls
        assert [c['filters']['category'] for c in calls] == [_PK, _PN], (
            'the repeat is dropped and the caller-supplied order is preserved'
        )

    async def test_a_repeated_category_never_makes_a_record_its_own_loser(
        self, monkeypatch, tmp_path,
    ):
        """The end-to-end consequence: --apply must not delete the last copy."""
        raw = {_PK: [_raw('m1', _VENV_GOTCHA_A, '2026-01-01T00:00:00+00:00')]}
        _install_run_doubles(monkeypatch, raw)
        args = _build_parser().parse_args([
            '--project-id', 'p', '--metrics-root', str(tmp_path), '--apply',
            '--categories', _PK, _PK,
        ])
        rc = await _run(args)
        service = _FakeMemoryService.instances[-1]

        assert service.deleted == [], (
            'a lone record duplicated only by a repeated flag is not a duplicate'
        )
        assert rc == 1, 'no actionable candidates, so the empty-plan guard refuses'


# --- _run wiring doubles -----------------------------------------------------

def _fake_config(t_high: float | None = 0.9):
    return types.SimpleNamespace(
        write_triage=types.SimpleNamespace(t_high=t_high),
        mem0=types.SimpleNamespace(collection_prefix='mem0'),
    )


class _FakeQdrant:
    def __init__(self, hits_by_vector: dict[tuple, list[tuple]] | None = None):
        self.calls = []
        self._hits = hits_by_vector or {}

    async def query_points(self, **kwargs):
        self.calls.append(kwargs)
        hits = self._hits.get(tuple(kwargs.get('query') or ()), [])
        return types.SimpleNamespace(
            points=[types.SimpleNamespace(id=i, score=s) for i, s in hits],
        )


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
    ann_hits: dict[tuple, list[tuple]] = {}

    def __init__(self, config):
        self.config = config
        self.mem0 = _FakeMem0(
            type(self).raw_by_category, _FakeQdrant(type(self).ann_hits),
        )
        self.deleted: list = []
        self.closed = False
        type(self).instances.append(self)

    async def initialize(self):
        return None

    async def close(self):
        self.closed = True

    async def delete_memory(self, memory_id, store=None, project_id=None):
        self.deleted.append(memory_id)


def _install_run_doubles(monkeypatch, raw_by_category, *, t_high: float | None = 0.9,
                         ann_hits: dict[tuple, list[tuple]] | None = None):
    import fused_memory.config.schema as schema_mod  # noqa: PLC0415
    import fused_memory.services.memory_service as service_mod  # noqa: PLC0415

    _FakeMemoryService.instances = []
    _FakeMemoryService.raw_by_category = raw_by_category
    _FakeMemoryService.ann_hits = ann_hits or {}
    monkeypatch.setattr(schema_mod, 'FusedMemoryConfig', lambda: _fake_config(t_high))
    monkeypatch.setattr(service_mod, 'MemoryService', _FakeMemoryService)
    return _FakeMemoryService


def _raw(id: str, content: str, created_at: str = '2026-01-01T00:00:00+00:00',
         category: str = _PK, extra: dict | None = None,
         vector: list[float] | None = None) -> dict:
    return {
        'id': id,
        'created_at': created_at,
        'metadata': {'data': content, 'category': category, **(extra or {})},
        'vector': [0.1, 0.2] if vector is None else vector,
    }


@pytest.mark.asyncio
class TestRunApplyGuards:
    """--apply never fires against a scan that cannot be trusted."""

    async def _invoke(self, monkeypatch, raw, argv, tmp_path, ann_hits=None):
        _install_run_doubles(monkeypatch, raw, ann_hits=ann_hits)
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
class TestRunApplyAnnOnlyClusters:
    """End-to-end: what --apply does with deletions the ANN path alone produced.

    The riskiest behaviour this detector gained — records the lexical path
    would never have touched now reach an irreversible delete. These pin which
    ids actually reach delete_memory.
    """

    _OLD = '2026-01-01T00:00:00+00:00'
    _MID = '2026-01-02T00:00:00+00:00'
    _NEW = '2026-01-03T00:00:00+00:00'

    async def _invoke(self, monkeypatch, raw, ann_hits, tmp_path, *argv):
        _install_run_doubles(monkeypatch, raw, ann_hits=ann_hits)
        args = _build_parser().parse_args([
            '--project-id', 'p', '--metrics-root', str(tmp_path), '--apply',
            '--threshold', '0.99', *argv,
        ])
        return await _run(args), _FakeMemoryService.instances[-1]

    async def test_an_ann_only_clique_deletes_its_losers_and_keeps_the_survivor(
        self, monkeypatch, tmp_path, capsys,
    ):
        """Paraphrase, not rewrite: lexical cannot cluster it, ANN can."""
        raw = {_PK: [
            _raw('m1', 'alpha one', self._OLD, vector=[1.0, 0.0]),
            _raw('m2', 'beta two', self._MID, vector=[0.0, 1.0]),
        ]}
        ann_hits = {
            (1.0, 0.0): [('m2', 0.95)],
            (0.0, 1.0): [('m1', 0.95)],
        }
        rc, service = await self._invoke(monkeypatch, raw, ann_hits, tmp_path)
        plan = json.loads(capsys.readouterr().out)

        assert plan['near_duplicate_groups'][0]['found_by'] == ['ann'], (
            'the lexical threshold of 0.99 cannot have clustered these'
        )
        assert rc == 0
        assert service.deleted == ['m2'], 'the oldest survives; only the loser goes'

    async def test_an_ann_chained_cluster_is_reported_but_never_deleted(
        self, monkeypatch, tmp_path, capsys,
    ):
        """m1~m2 and m2~m3 cleared the cutoff; m1~m3 never did.

        m3 is in the cluster purely by transitivity, so 'delete the losers'
        would irreversibly delete a record no measurement ever tied to the
        survivor.
        """
        raw = {_PK: [
            _raw('m1', 'alpha one', self._OLD, vector=[1.0, 0.0, 0.0]),
            _raw('m2', 'beta two', self._MID, vector=[0.0, 1.0, 0.0]),
            _raw('m3', 'gamma three', self._NEW, vector=[0.0, 0.0, 1.0]),
        ]}
        ann_hits = {
            (1.0, 0.0, 0.0): [('m2', 0.95)],
            (0.0, 1.0, 0.0): [('m1', 0.95), ('m3', 0.95)],
            (0.0, 0.0, 1.0): [('m2', 0.95)],
        }
        rc, service = await self._invoke(monkeypatch, raw, ann_hits, tmp_path)
        plan = json.loads(capsys.readouterr().out)

        assert set(plan['near_duplicate_groups'][0]['member_ids']) == {'m1', 'm2', 'm3'}
        assert plan['delete_candidates'] == ['m2', 'm3'], 'still reported'
        assert plan['apply_withheld_clusters'] == 1
        assert service.deleted == [], 'the chain never reaches delete_memory'
        assert rc == 1, 'nothing applicable remains, so the empty-plan guard holds'

    async def test_a_fully_connected_ann_cluster_is_deleted_down_to_the_survivor(
        self, monkeypatch, tmp_path,
    ):
        """Every member pair independently cleared the cutoff — no transitivity."""
        raw = {_PK: [
            _raw('m1', 'alpha one', self._OLD, vector=[1.0, 0.0, 0.0]),
            _raw('m2', 'beta two', self._MID, vector=[0.0, 1.0, 0.0]),
            _raw('m3', 'gamma three', self._NEW, vector=[0.0, 0.0, 1.0]),
        ]}
        ann_hits = {
            (1.0, 0.0, 0.0): [('m2', 0.95), ('m3', 0.94)],
            (0.0, 1.0, 0.0): [('m1', 0.95), ('m3', 0.93)],
            (0.0, 0.0, 1.0): [('m1', 0.94), ('m2', 0.93)],
        }
        rc, service = await self._invoke(monkeypatch, raw, ann_hits, tmp_path)

        assert rc == 0
        assert service.deleted == ['m2', 'm3']

    async def test_an_ann_merge_of_two_lexical_clusters_is_withheld(
        self, monkeypatch, tmp_path, capsys,
    ):
        """The compound risk: one ANN pair welds two real clusters into one."""
        raw = {_PK: [
            _raw('a1', _VENV_GOTCHA_A, self._OLD, vector=[1.0, 0.0]),
            _raw('a2', _VENV_GOTCHA_B, self._MID, vector=[0.9, 0.1]),
            _raw('b1', _DISTRACTOR, self._NEW, vector=[0.0, 1.0]),
            _raw('b2', _DISTRACTOR + ' Rerun nightly.', '2026-01-04T00:00:00+00:00',
                 vector=[0.1, 0.9]),
        ]}
        # a2 <-> b1: the single cross-cluster ANN pair.
        ann_hits = {(0.9, 0.1): [('b1', 0.95)], (0.0, 1.0): [('a2', 0.95)]}
        rc, service = await self._invoke(
            monkeypatch, raw, ann_hits, tmp_path, '--threshold', '0.75',
        )
        plan = json.loads(capsys.readouterr().out)

        assert len(plan['near_duplicate_groups']) == 1, 'ANN merged the two halves'
        assert plan['apply_withheld_clusters'] == 1
        assert service.deleted == [], (
            'a lexical pair inside the merge does not corroborate the merge'
        )
        assert rc == 1

    async def test_a_lexical_cluster_still_applies_when_ann_agrees(
        self, monkeypatch, tmp_path,
    ):
        """The pre-existing path is unchanged by the gate."""
        raw = {_PK: [
            _raw('m1', _VENV_GOTCHA_A, self._OLD, vector=[1.0, 0.0]),
            _raw('m2', _VENV_GOTCHA_B, self._MID, vector=[0.9, 0.1]),
        ]}
        ann_hits = {(1.0, 0.0): [('m2', 0.95)], (0.9, 0.1): [('m1', 0.95)]}
        rc, service = await self._invoke(
            monkeypatch, raw, ann_hits, tmp_path, '--threshold', '0.75',
        )

        assert rc == 0
        assert service.deleted == ['m2']


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
        for key in (*_ANN_DISCLOSURE_KEYS, *_PLAN_DISCLOSURE_KEYS,
                    'ann_query_errors', 'ann_disabled_uncalibrated'):
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


# ---------------------------------------------------------------------------
# Liveness-snapshot recurrence fixtures (task 3098)
# ---------------------------------------------------------------------------
#
# Real-world fixture basis: four solar_challenge observations_and_summaries
# memories written by agent_id `recon-stage-task_knowledge_sync`, surfaced by
# Stage-1 finding 724e7be4 (run e149def2). Provenance by memory id:
#
#   6b245659  metadata.task_id="94"  kind=stranded_task_investigation
#   08aa0017  metadata.task_id="96"  kind=live_workflow_deferral_note
#   68dd5f93  metadata.task_id="96"  kind=investigation_note
#   1eef7df7  metadata.task_id="99", related_task_ids=["94","96"]
#                                    kind=task_lifecycle_reverification
#
# Contents below are the real ones, abbreviated in the tail but VERBATIM
# through the status-bearing clause. `metadata.kind` differs across all four,
# so keying on it is not viable; the repeated CORE FACT is.
#
# Why a third detector class is needed: the pairwise difflib.SequenceMatcher
# ratios between the three liveness snapshots are 0.1173 / 0.2328 / 0.1770 on
# the full records (measured and recorded in task 3098's plan) and
# 0.1648 / 0.2882 / 0.2687 on the abbreviated fixture text below (measured
# before authoring this fixture) -- against a production LEXICAL threshold of
# 0.85. The unique investigative payload is several times the length of the
# repeated core status fact and swamps every similarity metric, so the lexical
# path structurally cannot reach these records. Following the discipline of
# the _VENV_GOTCHA_* block above, no assertion pins those ratio values.
_LIVENESS_SNAPSHOT_94_JUL24 = (
    'Point-in-time liveness check performed 2026-07-24 during reconciliation '
    'run 4b262e10-3624-44e5-b420-4dbedfd2bb8b (Stage 2, task_knowledge_sync) '
    'on task 94 ("Point orchestrator config references at canonical '
    'dark-factory-orchestrator.yaml and drop the transitional symlink"): '
    'get_task reports status="in-progress" but claimant_run_id=null and '
    'heartbeat_at=null, and this cycle\'s Live-Workflow Signals section does '
    'NOT list task/94 (no registered worktree signal, no recent-commit '
    'signal, no active-orchestrator signal) — even though a worktree '
    'directory with a populated .venv exists on disk at .worktrees/94/. '
    'Direct file reads this cycle confirm the task\'s SCOPE item 1 has NOT '
    'landed on the main checkout: CLAUDE.md:95 still reads "orchestrator.yaml" '
    'and .envrc:5-6 still reference the legacy '
    '/home/leo/src/solar-challenge/orchestrator.yaml path, both of which task '
    '94 is supposed to retarget to dark-factory-orchestrator.yaml.'
)

_LIVENESS_SNAPSHOT_96_JUL24 = (
    'Point-in-time liveness check performed 2026-07-24 during reconciliation '
    'run dcd89ed9-8df5-404a-8e33-caa296a0a73c (Stage 2, task_knowledge_sync) '
    'on task 96 ("Fix stale briefing.yaml POSTURE header + deferred-task '
    'tally — third recurrence"): get_task at that timestamp reported '
    'status="in-progress" (claimant_run_id=null, heartbeat_at=null), and the '
    'reconciliation payload\'s Live-Workflow Signals section at that same '
    'timestamp listed task/96 as having a registered git worktree AND an '
    'active orchestrator process holding the project lock. Stage 1 had '
    'separately confirmed via direct file read that '
    '/home/leo/src/solar-challenge/review/briefing.yaml already contained the '
    'corrected POSTURE header + deferred-task tally content matching task '
    "96's acceptance criteria."
)

# The "genuinely unique investigative content" the task says to preserve: it
# reports task 96 as in-progress in PROSE, with no <field>=<value> assignment
# and no point-in-time framing at all. It must never be grouped.
_INVESTIGATION_NOTE_96 = (
    'Task 96 ("Fix stale briefing.yaml POSTURE header + deferred-task tally — '
    'third recurrence") remains in-progress as of run f867d648 (2026-07-24). '
    'Direct read of /home/leo/src/solar-challenge/review/briefing.yaml on the '
    'MAIN checkout (non-worktree path) shows the intended fix already present '
    '(POSTURE header re-dated 2026-07-22, third-recurrence hedge language). '
    'However, /home/leo/src/solar-challenge/.git/logs/HEAD shows main HEAD has '
    'NOT advanced past commit af4911565ed76a81fa1a02641aebdd1653ce20b0 — no '
    'later commit appears in the reflog. Conclusion: do NOT close task 96 via '
    'set_task_status/done_provenance without a verified commit SHA — there is '
    'none to cite.'
)

_LIVENESS_REVERIFY_94_96_JUL26 = (
    'Point-in-time liveness check performed 2026-07-26 during reconciliation '
    'run e3eaa351-5173-41a9-b187-801fb05400bf (Stage 2, task_knowledge_sync) '
    'on task 94 ("Point orchestrator config references at canonical '
    'dark-factory-orchestrator.yaml and drop the transitional symlink"): '
    'get_task reports status="in-progress" but claimant_run_id=null and '
    'heartbeat_at=null, and this cycle\'s Live-Workflow Signals section does '
    'NOT list task/94 — matching the same stranded pattern first recorded in '
    'prior investigation 6b245659 (run 4b262e10). A fresh grep of .envrc at '
    'this timestamp still found lines 5-6 referencing the legacy top-level '
    '"orchestrator.yaml" path.\n\n'
    'Separately, point-in-time check on task 96 ("Fix stale briefing.yaml '
    'POSTURE header + deferred-task tally"): get_task reports '
    'status="in-progress" but claimant_run_id=null and heartbeat_at=null, '
    'matching the same stranded pattern. A fresh read of review/briefing.yaml '
    'at this timestamp found the POSTURE header already carrying task 96\'s '
    'intended fix text as a working-tree state only — no corresponding commit '
    'was observed advancing main HEAD. New supplementary detail for whoever '
    "resolves task 99's commit-vs-discard decision: the diff's own embedded "
    'snapshot numbers were already one cycle old at authoring time.'
)

# The core fact all three snapshots re-assert, in the canonical rendering:
# distinct `field=value` pairs, values lowercased and unquoted, sorted by
# field name, joined on '|'.
_CORE_FACT_STRANDED = 'claimant_run_id=null|heartbeat_at=null|status=in-progress'

# The real created_at values, kept so first_seen/last_seen/span_days are
# exercised against real elapsed time rather than round numbers.
_TS_94_JUL24 = '2026-07-24T13:14:03.534125+00:00'
_TS_96_JUL24 = '2026-07-24T09:23:28.354341+00:00'
_TS_NOTE_96 = '2026-07-24T17:26:17.120293+00:00'
_TS_REVERIFY = '2026-07-26T05:52:25.057487+00:00'


class TestExtractLivenessSnapshotFact:
    """A repeated core FACT, keyed exactly -- not a similarity score.

    Two snapshots taken days apart share one clause verbatim and differ
    everywhere else, so the equivalence relation here is "asserts the same
    live fields", not "reads alike". Exact string equality on a canonical key
    means there is no threshold to tune and no false cluster to delete.
    """

    def test_all_three_real_snapshots_yield_the_same_key(self):
        keys = [
            extract_liveness_snapshot_fact(
                _memory('6b245659', _LIVENESS_SNAPSHOT_94_JUL24, category=_OS),
            ),
            extract_liveness_snapshot_fact(
                _memory('08aa0017', _LIVENESS_SNAPSHOT_96_JUL24, category=_OS),
            ),
            extract_liveness_snapshot_fact(
                _memory('1eef7df7', _LIVENESS_REVERIFY_94_96_JUL26, category=_OS),
            ),
        ]

        assert all(k is not None for k in keys), 'each is a recognised snapshot'
        assert len(set(keys)) == 1, 'the repeated core fact is one key, not three'
        assert keys[0] == _CORE_FACT_STRANDED

    def test_prose_only_investigation_note_is_not_a_snapshot(self):
        """No `<field>=<value>` assignment -> genuinely unique content, preserved."""
        assert extract_liveness_snapshot_fact(
            _memory('68dd5f93', _INVESTIGATION_NOTE_96, category=_OS),
        ) is None

    def test_a_different_status_is_a_different_key(self):
        """Distinct facts must never collapse into one group."""
        done = _LIVENESS_SNAPSHOT_94_JUL24.replace(
            'status="in-progress"', 'status="done"',
        )
        key = extract_liveness_snapshot_fact(_memory('x', done, category=_OS))

        assert key == 'claimant_run_id=null|heartbeat_at=null|status=done'
        assert key != _CORE_FACT_STRANDED

    def test_live_fields_without_point_in_time_framing_are_not_a_snapshot(self):
        """Incidental prose carrying `status=` is not a liveness observation.

        Gate 2628 BLOCKS the bare current-fact framing in the gated
        categories, so every liveness observation written since it landed
        necessarily carries the point-in-time form. Requiring it costs no
        recall on the swept corpus.
        """
        assert extract_liveness_snapshot_fact(_memory(
            'x',
            'The dispatcher rejected the claim because status=in-progress and '
            'claimant_run_id=null are inconsistent for a queued task.',
            category=_OS,
        )) is None

    def test_paraphrase_without_an_assignment_is_not_a_snapshot(self):
        """A LIVE_TASK_STATUS_RE paraphrase alone yields no readable fields."""
        assert extract_liveness_snapshot_fact(_memory(
            'x',
            'Point-in-time liveness check performed 2026-07-24 on task 94: '
            'the task is actively driven by an orchestrator process holding '
            'the project lock, so Stage 2 did not write.',
            category=_OS,
        )) is None


class TestLivenessSnapshotSubjectTaskIds:
    """Which task(s) a snapshot is ABOUT -- a union, not `metadata.task_id`.

    The task description phrases the pattern as "sharing the same task_id",
    but the live corpus disproves the strict reading: 1eef7df7 is filed under
    task_id='99' while being a re-verification of tasks 94 AND 96. Under a
    strict grouping every bucket in the motivating corpus is a singleton and
    the detector emits nothing on the exact pattern it was built for.
    """

    def test_metadata_task_id_alone(self):
        assert liveness_snapshot_subject_task_ids(
            _memory('m', 'no task refs here', metadata={'task_id': '94'}),
        ) == {'94'}

    def test_int_task_id_is_coerced_to_str(self):
        """The project-wide convention `_normalize_task_id_metadata` enforces."""
        assert liveness_snapshot_subject_task_ids(
            _memory('m', 'no task refs here', metadata={'task_id': 94}),
        ) == {'94'}

    def test_related_task_ids_and_content_refs_union_in(self):
        """The real 1eef7df7 shape: filed under 99, about 94 and 96."""
        subjects = liveness_snapshot_subject_task_ids(_memory(
            '1eef7df7', _LIVENESS_REVERIFY_94_96_JUL26, category=_OS,
            metadata={'task_id': '99', 'related_task_ids': ['94', '96']},
        ))

        assert {'94', '96', '99'} <= subjects

    def test_content_ref_alone_still_groups_a_snapshot(self):
        """Absent/blank metadata must not make a snapshot ungroupable."""
        assert liveness_snapshot_subject_task_ids(
            _memory('6b245659', _LIVENESS_SNAPSHOT_94_JUL24, category=_OS),
        ) == {'94'}

    def test_no_metadata_id_and_no_content_ref_is_empty(self):
        assert liveness_snapshot_subject_task_ids(
            _memory('m', 'a note that names no task at all'),
        ) == set()

    @pytest.mark.parametrize('related', ['77', None, 12, {'nested': 'dict'}])
    def test_non_list_related_task_ids_degrades_without_raising(self, related):
        subjects = liveness_snapshot_subject_task_ids(
            _memory('m', 'no task refs here',
                    metadata={'task_id': '94', 'related_task_ids': related}),
        )

        assert '94' in subjects, 'the metadata task id survives regardless'
        assert all(isinstance(s, str) for s in subjects)

    def test_the_record_is_not_mutated(self):
        record = _memory(
            '1eef7df7', _LIVENESS_REVERIFY_94_96_JUL26, category=_OS,
            metadata={'task_id': '99', 'related_task_ids': ['94', '96']},
        )
        before = json.dumps(record, sort_keys=True, default=str)

        liveness_snapshot_subject_task_ids(record)

        assert json.dumps(record, sort_keys=True, default=str) == before


def _liveness_corpus() -> list[dict]:
    """The four real solar_challenge records, in their real shapes."""
    return [
        _dated('6b245659', _LIVENESS_SNAPSHOT_94_JUL24, _TS_94_JUL24,
               category=_OS, metadata={'task_id': '94',
                                       'kind': 'stranded_task_investigation'}),
        _dated('08aa0017', _LIVENESS_SNAPSHOT_96_JUL24, _TS_96_JUL24,
               category=_OS, metadata={'task_id': '96',
                                       'kind': 'live_workflow_deferral_note'}),
        _dated('68dd5f93', _INVESTIGATION_NOTE_96, _TS_NOTE_96,
               category=_OS, metadata={'task_id': '96',
                                       'kind': 'investigation_note'}),
        _dated('1eef7df7', _LIVENESS_REVERIFY_94_96_JUL26, _TS_REVERIFY,
               category=_OS, metadata={'task_id': '99',
                                       'related_task_ids': ['94', '96'],
                                       'kind': 'task_lifecycle_reverification'}),
    ]


def _span_days(earlier: str, later: str) -> float:
    return (datetime.fromisoformat(later).timestamp()
            - datetime.fromisoformat(earlier).timestamp()) / 86400.0


class TestFindLivenessSnapshotRecurrences:
    """The two groups the real corpus contains -- and nothing else.

    `metadata.kind` differs across all four records, so the grouping key is
    (category, subject_task_id, core_fact). Subject 99 is a singleton and
    68dd5f93 is not a snapshot at all: both fall out, which is the
    "preserve genuinely unique investigative content" requirement.
    """

    def test_exactly_the_two_real_groups(self):
        groups, _ = find_liveness_snapshot_recurrences(_liveness_corpus())

        assert [(g['subject_task_id'], g['member_ids']) for g in groups] == [
            ('94', ['1eef7df7', '6b245659']),
            ('96', ['08aa0017', '1eef7df7']),
        ]
        assert all(g['snapshot_count'] == 2 for g in groups)
        assert all(g['category'] == _OS for g in groups)
        assert all(g['core_fact'] == _CORE_FACT_STRANDED for g in groups)

    def test_singleton_subject_and_non_snapshot_are_absent(self):
        groups, _ = find_liveness_snapshot_recurrences(_liveness_corpus())

        assert '99' not in {g['subject_task_id'] for g in groups}, (
            'one snapshot about task 99 is a singleton, not a recurrence'
        )
        assert all('68dd5f93' not in g['member_ids'] for g in groups), (
            'the prose-only investigation note carries no repeated core fact'
        )

    def test_first_seen_last_seen_and_span_days(self):
        groups, _ = find_liveness_snapshot_recurrences(_liveness_corpus())
        by_subject = {g['subject_task_id']: g for g in groups}

        assert by_subject['94']['first_seen'] == _TS_94_JUL24
        assert by_subject['94']['last_seen'] == _TS_REVERIFY
        assert by_subject['94']['span_days'] == _span_days(_TS_94_JUL24, _TS_REVERIFY)
        assert by_subject['96']['first_seen'] == _TS_96_JUL24
        assert by_subject['96']['last_seen'] == _TS_REVERIFY
        assert by_subject['96']['span_days'] == _span_days(_TS_96_JUL24, _TS_REVERIFY)

    def test_no_parseable_timestamp_reports_none_not_a_fabricated_span(self):
        undated = [
            _dated('a', _LIVENESS_SNAPSHOT_94_JUL24, None,
                   category=_OS, metadata={'task_id': '94'}),
            _dated('b', _LIVENESS_REVERIFY_94_96_JUL26, 'not-a-timestamp',
                   category=_OS, metadata={'task_id': '94'}),
        ]
        groups, _ = find_liveness_snapshot_recurrences(undated)

        assert len(groups) == 1
        assert groups[0]['first_seen'] is None
        assert groups[0]['last_seen'] is None
        assert groups[0]['span_days'] is None

    def test_input_order_does_not_change_the_result(self):
        corpus = _liveness_corpus()
        shuffled = [corpus[3], corpus[1], corpus[0], corpus[2]]

        assert (find_liveness_snapshot_recurrences(shuffled)[0]
                == find_liveness_snapshot_recurrences(corpus)[0])

    def test_input_is_not_mutated(self):
        corpus = _liveness_corpus()
        before = json.dumps(corpus, sort_keys=True, default=str)

        find_liveness_snapshot_recurrences(corpus)

        assert json.dumps(corpus, sort_keys=True, default=str) == before

    def test_grouping_never_unions_across_categories(self):
        """A preference and an observation reporting the same live fields are
        different kinds of knowledge -- the same invariant build_sweep_plan
        enforces by clustering per category."""
        corpus = _liveness_corpus()
        corpus[0]['category'] = _PN  # 6b245659, the other half of subject 94

        groups, _ = find_liveness_snapshot_recurrences(corpus)

        assert [g['subject_task_id'] for g in groups] == ['96'], (
            'subject 94 is now one snapshot per category -- two singletons'
        )

    def test_unswept_categories_are_ignored(self):
        corpus = _liveness_corpus()

        groups, _ = find_liveness_snapshot_recurrences(corpus, categories=(_PK,))

        assert groups == []


# A recognised snapshot that names no task anywhere -- no metadata.task_id, no
# related_task_ids, and no TASK_REF_RE match in the content ("get_task" is not
# a task reference). It cannot be grouped, so it is a genuine recall loss.
_LIVENESS_SNAPSHOT_UNTASKED = (
    'Point-in-time liveness check performed 2026-07-24 during reconciliation '
    'run 4b262e10-3624-44e5-b420-4dbedfd2bb8b (Stage 2, task_knowledge_sync): '
    'get_task reports status="in-progress" but claimant_run_id=null and '
    'heartbeat_at=null for the item under investigation.'
)

# Point-in-time framing plus a LIVE_TASK_STATUS_RE PARAPHRASE, with no
# <field>=<value> assignment anywhere: a snapshot whose fields could not be
# read, which is a different loss from "not a snapshot at all".
_LIVENESS_SNAPSHOT_UNFIELDED = (
    'Point-in-time liveness check performed 2026-07-24 on task 94: the task '
    'is actively driven by an orchestrator process holding the project lock, '
    'so Stage 2 did not write.'
)

# Point-in-time framing and NOTHING live-status-shaped: never a snapshot, so
# neither counter may fire on it.
_POINT_IN_TIME_NON_STATUS = (
    'Point-in-time liveness check performed 2026-07-24 on the config sweep: '
    'nothing of note was observed this cycle.'
)


class TestFindLivenessSnapshotRecurrencesDisclosure:
    """Every way this path can lose a candidate is a counted, alarmable int.

    An absent key downstream is indistinguishable from "not measured", so the
    counters are always all present, always ints, always zero-filled on a
    clean run -- the same contract _ANN_DISCLOSURE_KEYS carries.
    """

    def test_clean_run_zero_fills_every_key(self):
        _, disclosure = find_liveness_snapshot_recurrences(_liveness_corpus())

        assert set(disclosure) == set(_LIVENESS_DISCLOSURE_KEYS)
        assert all(type(v) is int for v in disclosure.values()), 'ints, not bools'
        assert set(disclosure.values()) == {0}

    def test_untasked_snapshot_is_counted_not_silently_dropped(self):
        _, disclosure = find_liveness_snapshot_recurrences([
            _dated('u', _LIVENESS_SNAPSHOT_UNTASKED, _TS_94_JUL24, category=_OS),
        ])

        assert disclosure['liveness_snapshot_untasked'] == 1
        assert disclosure['liveness_snapshot_unfielded'] == 0, (
            'its fields read fine — it is only the subject that is missing'
        )

    def test_unfielded_snapshot_is_counted(self):
        _, disclosure = find_liveness_snapshot_recurrences([
            _dated('p', _LIVENESS_SNAPSHOT_UNFIELDED, _TS_94_JUL24,
                   category=_OS, metadata={'task_id': '94'}),
        ])

        assert disclosure['liveness_snapshot_unfielded'] == 1
        assert disclosure['liveness_snapshot_untasked'] == 0, (
            'it names task 94 — it is only the fields that could not be read'
        )

    def test_point_in_time_prose_with_no_live_status_counts_as_neither(self):
        """"Not a snapshot at all" is not a loss — nothing was ever in hand."""
        _, disclosure = find_liveness_snapshot_recurrences([
            _dated('n', _POINT_IN_TIME_NON_STATUS, _TS_94_JUL24,
                   category=_OS, metadata={'task_id': '94'}),
        ])

        assert set(disclosure.values()) == {0}

    def test_unswept_category_counts_neither(self):
        """It was never in scope, so it is not a recall loss."""
        _, disclosure = find_liveness_snapshot_recurrences(
            [
                _dated('u', _LIVENESS_SNAPSHOT_UNTASKED, _TS_94_JUL24, category=_OS),
                _dated('p', _LIVENESS_SNAPSHOT_UNFIELDED, _TS_94_JUL24, category=_OS),
            ],
            categories=(_PK,),
        )

        assert set(disclosure.values()) == {0}


# The plan keys that existed before task 3098. Pinned so an additive change
# stays additive: no pre-existing key may be dropped or renamed.
_PRE_3098_PLAN_KEYS = frozenset({
    'clusters_total', 'near_duplicate_groups', 'delete_candidates',
    'threshold', 'ann_threshold', 'ann_disclosure',
    'topic_carveout_clusters', 'topic_carveout_groups',
    'apply_delete_candidates', 'apply_withheld_clusters',
    'apply_withheld_groups', 'path_verdicts',
})

_LIVENESS_PLAN_KEYS = frozenset({
    'liveness_snapshot_recurrence_groups',
    'liveness_snapshot_recurrence_clusters',
    'liveness_snapshot_disclosure',
})


class TestBuildSweepPlanLivenessRecurrences:
    """The third class reaches the plan -- and never reaches a delete."""

    def test_the_two_real_groups_are_reported(self):
        plan = build_sweep_plan(_liveness_corpus())

        assert plan['liveness_snapshot_recurrence_clusters'] == 2
        assert [(g['subject_task_id'], g['member_ids'])
                for g in plan['liveness_snapshot_recurrence_groups']] == [
            ('94', ['1eef7df7', '6b245659']),
            ('96', ['08aa0017', '1eef7df7']),
        ]
        assert plan['liveness_snapshot_disclosure'] == dict.fromkeys(
            _LIVENESS_DISCLOSURE_KEYS, 0,
        )

    def test_no_liveness_member_is_ever_a_delete_candidate(self):
        """Task 3098 authorizes DETECTION only.

        Disposition of these specific solar_challenge entries is a separate
        human-gated decision on solar_challenge task 99. If these groups fed
        `delete_candidates`, a routine --apply run would delete them and
        silently override that gate.
        """
        plan = build_sweep_plan(_liveness_corpus())

        for member_id in ('6b245659', '08aa0017', '1eef7df7', '68dd5f93'):
            assert member_id not in plan['delete_candidates']
            assert member_id not in plan['apply_delete_candidates']
        assert plan['delete_candidates'] == []
        assert plan['apply_delete_candidates'] == []

    def test_the_apply_gate_is_never_handed_a_liveness_group(self, monkeypatch):
        """Report-only by CONSTRUCTION, not by a downstream filter.

        A filter can be dropped by a later refactor; never handing the gate
        these groups in the first place cannot be.
        """
        seen: list[list[dict]] = []
        real = _mod.restrict_delete_candidates_for_apply

        def _spy(groups):
            seen.append(groups)
            return real(groups)

        monkeypatch.setattr(_mod, 'restrict_delete_candidates_for_apply', _spy)
        build_sweep_plan(_liveness_corpus())

        assert seen, 'the apply gate still runs'
        assert all(
            'core_fact' not in g and 'subject_task_id' not in g
            for handed in seen for g in handed
        )

    def test_pre_existing_keys_are_untouched_and_new_keys_are_never_absent(self):
        """A corpus with no liveness snapshots reports empty, not missing."""
        plan = build_sweep_plan(
            [
                _memory('m1', _VENV_GOTCHA_A, created_at='2026-07-13T00:00:00+00:00'),
                _memory('m2', _VENV_GOTCHA_B, created_at='2026-07-12T00:00:00+00:00'),
                _memory('m4', _DISTRACTOR),
            ],
            threshold=_THRESHOLD,
        )

        assert set(plan) == _PRE_3098_PLAN_KEYS | _LIVENESS_PLAN_KEYS
        assert plan['clusters_total'] == 1, 'the lexical path is unaffected'
        assert plan['delete_candidates'] == (
            plan['near_duplicate_groups'][0]['delete_candidate_ids']
        )
        assert plan['liveness_snapshot_recurrence_groups'] == []
        assert plan['liveness_snapshot_recurrence_clusters'] == 0
        assert plan['liveness_snapshot_disclosure'] == dict.fromkeys(
            _LIVENESS_DISCLOSURE_KEYS, 0,
        )

    def test_plan_is_still_json_serializable(self):
        """stdout is task 3136's report contract."""
        assert isinstance(
            json.dumps(build_sweep_plan(_liveness_corpus()), default=str), str,
        )


class TestLivenessRecurrenceMetrics:
    """The recurrence count is an ARMED, directional metric -- not a scalar.

    `topic_carveout_clusters` is a scalar because more carve-outs is the
    EXPECTED Option-C shape, so alarming on it would alarm on the feature
    working. Indefinite accretion of the same status fact is the opposite:
    it IS the pathology this detector exists to surface.
    """

    def _fixture(self):
        plan = build_sweep_plan(_liveness_corpus())
        records = {_OS: _liveness_corpus(), _PK: [_dated('x', 'unrelated', None)]}
        return records, split_plan_by_category(plan, (_OS, _PK))

    def test_split_plan_projects_the_groups_into_their_category(self):
        plan = build_sweep_plan(_liveness_corpus())

        by_cat = split_plan_by_category(plan, (_OS, _PK, _PN))

        assert [g['subject_task_id']
                for g in by_cat[_OS]['liveness_snapshot_recurrence_groups']] == ['94', '96']
        assert by_cat[_PK]['liveness_snapshot_recurrence_groups'] == []
        assert by_cat[_PN]['liveness_snapshot_recurrence_groups'] == [], (
            'an absent key downstream would read as "not measured"'
        )

    def test_metric_is_an_armed_directional_count_per_category(self):
        records, plans = self._fixture()

        metrics, _details = compute_cluster_metrics(records, plans, {})
        by_id = _by_id(metrics)

        metric = by_id[f'liveness_snapshot_recurrences.{_OS}']
        assert metric.kind == 'count'
        assert metric.direction == 'higher_is_worse'
        assert metric.value == 2.0
        assert metric.n == 4, 'n is the exposure: the category corpus size'

    def test_a_category_with_none_still_reports_zero(self):
        records, plans = self._fixture()

        metrics, _details = compute_cluster_metrics(records, plans, {})

        assert _by_id(metrics)[f'liveness_snapshot_recurrences.{_PK}'].value == 0.0

    def test_details_carry_the_per_group_table(self):
        records, plans = self._fixture()

        _metrics, details = compute_cluster_metrics(records, plans, {})
        table = details['categories'][_OS]['liveness_snapshot_recurrences']

        assert [row['subject_task_id'] for row in table] == ['94', '96']
        assert all(set(row) == {'subject_task_id', 'core_fact', 'member_ids',
                                'snapshot_count', 'span_days'} for row in table)
        assert table[0]['member_ids'] == ['1eef7df7', '6b245659']
        assert table[0]['snapshot_count'] == 2
        assert table[0]['core_fact'] == _CORE_FACT_STRANDED
        assert table[0]['span_days'] == _span_days(_TS_94_JUL24, _TS_REVERIFY)
        assert details['categories'][_PK]['liveness_snapshot_recurrences'] == []

    def test_series_still_validates_and_metrics_stay_sorted(self):
        from shared.memory_eval_metrics import validate_metric_series  # noqa: PLC0415

        records, plans = self._fixture()
        metrics, _details = compute_cluster_metrics(records, plans, {})

        ids = [m.metric_id for m in metrics]
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids)), 'no duplicate metric_id'
        validate_metric_series(build_metric_series(
            'dark_factory', metrics,
            corpus_counts={c: len(v) for c, v in records.items()},
            run_stamp='20260731T000000Z',
        ))


def _liveness_raw_corpus() -> dict[str, list[dict]]:
    """The real four records in scroll_by_metadata's raw payload shape."""
    return {_OS: [
        _raw('6b245659', _LIVENESS_SNAPSHOT_94_JUL24, _TS_94_JUL24, _OS,
             {'task_id': '94', 'kind': 'stranded_task_investigation'}),
        _raw('08aa0017', _LIVENESS_SNAPSHOT_96_JUL24, _TS_96_JUL24, _OS,
             {'task_id': '96', 'kind': 'live_workflow_deferral_note'}),
        _raw('68dd5f93', _INVESTIGATION_NOTE_96, _TS_NOTE_96, _OS,
             {'task_id': '96', 'kind': 'investigation_note'}),
        _raw('1eef7df7', _LIVENESS_REVERIFY_94_96_JUL26, _TS_REVERIFY, _OS,
             {'task_id': '99', 'related_task_ids': ['94', '96'],
              'kind': 'task_lifecycle_reverification'}),
    ]}


@pytest.mark.asyncio
class TestRunLivenessRecurrenceWiring:
    """End to end on the real corpus: the report and the metrics both carry it."""

    async def _parse(self, tmp_path, *argv):
        return _build_parser().parse_args(
            ['--project-id', 'p', '--metrics-root', str(tmp_path), *argv],
        )

    async def test_stdout_report_carries_the_recurrences(
        self, monkeypatch, tmp_path, capsys,
    ):
        _install_run_doubles(monkeypatch, _liveness_raw_corpus())

        rc = await _run(await self._parse(tmp_path))
        plan = json.loads(capsys.readouterr().out)

        assert rc == 0
        assert plan['liveness_snapshot_recurrence_clusters'] == 2
        assert [g['subject_task_id']
                for g in plan['liveness_snapshot_recurrence_groups']] == ['94', '96']
        assert plan['liveness_snapshot_disclosure'] == dict.fromkeys(
            _LIVENESS_DISCLOSURE_KEYS, 0,
        )
        assert plan['delete_candidates'] == [], 'detection only — never a delete'

    async def test_metric_series_carries_the_count_and_both_disclosures(
        self, monkeypatch, tmp_path,
    ):
        from shared.memory_eval_metrics import (  # noqa: PLC0415
            RUN_STAMP_ENV_VAR,
            load_metric_series,
        )

        monkeypatch.setenv(RUN_STAMP_ENV_VAR, _STAMP)
        _install_run_doubles(monkeypatch, _liveness_raw_corpus())
        await _run(await self._parse(tmp_path))

        series = load_metric_series(tmp_path / _EVAL_ID / f'metrics-{_STAMP}.json')
        by_id = _by_id(series.metrics)

        assert by_id[f'liveness_snapshot_recurrences.{_OS}'].value == 2.0
        # Run-global scope: these are properties of ONE pass over the whole
        # scanned set, not of a category, so a per-category split would be a
        # fabricated attribution.
        for key in _LIVENESS_DISCLOSURE_KEYS:
            assert f'{key}.{_GLOBAL_SCOPE}' in by_id, f'{key} must be disclosed'
        for key in (*_ANN_DISCLOSURE_KEYS, *_PLAN_DISCLOSURE_KEYS):
            assert f'{key}.{_GLOBAL_SCOPE}' in by_id, 'the ANN disclosures survive'

    async def test_no_metrics_still_writes_no_artifact(self, monkeypatch, tmp_path):
        _install_run_doubles(monkeypatch, _liveness_raw_corpus())
        await _run(await self._parse(tmp_path, '--no-metrics'))

        assert list(tmp_path.iterdir()) == []


class TestLivenessDetectorAddsNoCliSurface:
    """The detector is O(n), always runs and never deletes -- so no knob."""

    def test_scheduled_invocation_contract_is_byte_for_byte_unchanged(self):
        """Task 3136 schedules `--project-id X [--apply]`; no new flag was added."""
        dry = _build_parser().parse_args(['--project-id', 'X'])
        applied = _build_parser().parse_args(['--project-id', 'X', '--apply'])

        assert (dry.project_id, dry.apply) == ('X', False)
        assert (applied.project_id, applied.apply) == ('X', True)
