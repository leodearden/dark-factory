"""The detector's retention rule and report-feed fields (task 3136).

Two contracts of ``audit_duplicate_memories.build_sweep_plan`` that the weekly
duplicate-cluster report (``scripts/duplicate_cluster_report.py``) relies on:

* RETENTION. A cluster member carrying a non-empty ``metadata.parent_id``
  (``'child_record'``) or a non-empty ``metadata.topic`` (``'topic_stamped'``)
  is never a delete candidate, whatever the rest of its cluster looks like. It
  is listed in the group's ``retained_member_ids`` instead, and the plan counts
  it in ``retained_carveout``. Every delete list derives from
  ``delete_candidate_ids``, so ``--apply`` can never act on one.
* REPORT FEED. Per-member facts, plan-level child counts and the echoed scan
  coverage, the inputs a consecutive-run regrowth diff needs.

Every record here is synthetic. The texts are near-verbatim rewrites of one
sentence (pairwise ``SequenceMatcher`` ratio 0.93-0.98), so the lexical path
alone clusters them at the production threshold and no ANN double is needed.
"""
from __future__ import annotations

from pathlib import Path

from _fm_helpers import load_script_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_duplicate_memories.py'
_mod = load_script_module(SCRIPT_PATH, mod_name='audit_duplicate_memories')
build_sweep_plan = _mod.build_sweep_plan

_PK = 'procedural_knowledge'
_PN = 'preferences_and_norms'
_THRESHOLD = 0.85

_SENTENCE = (
    'The weekly duplicate sweep must never delete a topic-stamped retained '
    'peer because consolidation keeps those records on purpose{}.'
)
_REWRITES = tuple(
    _SENTENCE.format(suffix)
    for suffix in ('', ' always', ' deliberately', ' by design', ' intentionally')
)


def _record(
    record_id: str,
    rewrite: int,
    created_at: str,
    *,
    category: str = _PK,
    **metadata: object,
) -> dict:
    return {
        'id': record_id,
        'content': _REWRITES[rewrite],
        'created_at': created_at,
        'category': category,
        'metadata': dict(metadata),
    }


def _only_group(plan: dict) -> dict:
    (group,) = plan['near_duplicate_groups']
    return group


def _retained(group: dict) -> list[dict]:
    return group['retained_member_ids']


class TestStampedLosersAreRetainedNotDeleted:
    def test_a_topic_stamped_peer_beside_a_canonical_is_retained(self) -> None:
        plan = build_sweep_plan([
            _record('a', 0, '2026-03-01T00:00:00+00:00', topic='venv', canonical=True),
            _record('b', 1, '2026-02-01T00:00:00+00:00', topic='venv'),
            _record('u', 2, '2026-01-01T00:00:00+00:00'),
        ], threshold=_THRESHOLD)

        group = _only_group(plan)
        assert group['survivor_id'] == 'a'
        assert group['delete_candidate_ids'] == ['u']
        assert _retained(group) == [{'id': 'b', 'reason': 'topic_stamped'}]
        assert 'b' not in plan['delete_candidates']
        assert 'b' not in plan['apply_delete_candidates']
        assert plan['apply_delete_candidates'] == ['u']

    def test_a_stamped_peer_newer_than_an_unstamped_survivor_is_retained(self) -> None:
        plan = build_sweep_plan([
            _record('u', 0, '2026-01-01T00:00:00+00:00'),
            _record('p', 1, '2026-02-01T00:00:00+00:00', topic='venv'),
            _record('v', 2, '2026-03-01T00:00:00+00:00'),
        ], threshold=_THRESHOLD)

        group = _only_group(plan)
        assert group['survivor_id'] == 'u'
        assert group['delete_candidate_ids'] == ['v']
        assert _retained(group) == [{'id': 'p', 'reason': 'topic_stamped'}]
        assert 'p' not in plan['delete_candidates']
        assert 'p' not in plan['apply_delete_candidates']

    def test_two_canonicals_of_different_topics_are_both_kept(self) -> None:
        plan = build_sweep_plan([
            _record('c1', 0, '2026-01-01T00:00:00+00:00', topic='venv', canonical=True),
            _record('c2', 1, '2026-02-01T00:00:00+00:00', topic='rustfmt', canonical=True),
        ], threshold=_THRESHOLD)

        group = _only_group(plan)
        assert group['delete_candidate_ids'] == []
        assert plan['delete_candidates'] == []
        assert plan['apply_delete_candidates'] == []
        loser = ({'c1', 'c2'} - {group['survivor_id']}).pop()
        assert _retained(group) == [{'id': loser, 'reason': 'topic_stamped'}]


class TestChildRecordsAreRetainedNotDeleted:
    def test_sighting_and_amendment_children_are_never_delete_candidates(self) -> None:
        plan = build_sweep_plan([
            _record('p', 0, '2026-01-01T00:00:00+00:00'),
            _record('s', 1, '2026-02-01T00:00:00+00:00', parent_id='p', kind='sighting'),
            _record('m', 2, '2026-03-01T00:00:00+00:00', parent_id='p', kind='amendment'),
        ], threshold=_THRESHOLD)

        group = _only_group(plan)
        assert group['survivor_id'] == 'p'
        assert group['member_ids'] == ['m', 'p', 's']
        assert group['delete_candidate_ids'] == []
        assert _retained(group) == [
            {'id': 'm', 'reason': 'child_record'},
            {'id': 's', 'reason': 'child_record'},
        ]
        assert plan['delete_candidates'] == []
        assert plan['apply_delete_candidates'] == []
        assert plan['apply_withheld_groups'] == []

    def test_a_child_that_is_also_stamped_is_reported_once_as_a_child(self) -> None:
        plan = build_sweep_plan([
            _record('x', 0, '2026-01-01T00:00:00+00:00'),
            _record('y', 1, '2026-02-01T00:00:00+00:00', parent_id='x', topic='venv'),
        ], threshold=_THRESHOLD)

        assert _retained(_only_group(plan)) == [{'id': 'y', 'reason': 'child_record'}]
        assert plan['retained_carveout'] == {'topic_stamped': 0, 'child_record': 1}


class TestRetainedCarveoutCounts:
    def test_counts_are_summed_across_categories(self) -> None:
        plan = build_sweep_plan([
            _record('x', 0, '2026-01-01T00:00:00+00:00'),
            _record('t1', 1, '2026-02-01T00:00:00+00:00', topic='venv'),
            _record('t2', 2, '2026-03-01T00:00:00+00:00', topic='venv'),
            _record('p', 0, '2026-01-01T00:00:00+00:00', category=_PN),
            _record('s', 1, '2026-02-01T00:00:00+00:00', category=_PN,
                    parent_id='p', kind='sighting'),
        ], threshold=_THRESHOLD)

        assert plan['clusters_total'] == 2
        assert plan['retained_carveout'] == {'topic_stamped': 2, 'child_record': 1}

    def test_both_reasons_are_present_as_zero_when_nothing_is_retained(self) -> None:
        plan = build_sweep_plan([
            _record('x', 0, '2026-01-01T00:00:00+00:00'),
            _record('y', 1, '2026-02-01T00:00:00+00:00'),
        ], threshold=_THRESHOLD)

        assert _only_group(plan)['delete_candidate_ids'] == ['y']
        assert _retained(_only_group(plan)) == []
        assert plan['retained_carveout'] == {'topic_stamped': 0, 'child_record': 0}

    def test_an_empty_scan_still_carries_both_reasons(self) -> None:
        plan = build_sweep_plan([], threshold=_THRESHOLD)
        assert plan['retained_carveout'] == {'topic_stamped': 0, 'child_record': 0}
