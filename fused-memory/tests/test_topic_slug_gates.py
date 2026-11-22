"""Tests for ``fused_memory/maintenance/topic_slug_gates.py``, the gate census
and gate pairing behind ``scripts/normalize_topic_slugs.py`` (task 4878).

A consolidation gate is a block on TASK metadata, so a slug's memories and
its gate must move as a pair.  The census reads the gates through an
injected MCP client, so no test here opens a socket or loads the script.
"""
from __future__ import annotations

import dataclasses
from unittest.mock import call

import pytest
from _topic_slug_migration_doubles import by_reason, census_client, gate_task, task_listing

from fused_memory.maintenance.topic_slug_gates import (
    GateCensus,
    census_consolidation_gates,
    pair_gate_blocks,
)
from fused_memory.maintenance.topic_slug_plan import Rename
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY


class TestPairGateBlocks:
    """``pair_gate_blocks(renames, gate_tasks) -> (groups, skips)``.

    Pure.  It decides WHICH renames travel with WHICH gate task, before
    anything is written, so the pairing is reproducible from the plan alone.
    """

    def test_a_gate_on_a_renamed_slug_is_paired_with_it(self):
        renames = [Rename('dark_factory', 'm1', 'mem0_tombstone_coverage',
                               'mem0-tombstone-coverage')]
        gates = [gate_task('4220', 'mem0_tombstone_coverage')]

        groups, skips = pair_gate_blocks(renames, gates)

        assert len(groups) == 1
        assert groups[0].old_topic == 'mem0_tombstone_coverage'
        assert groups[0].new_topic == 'mem0-tombstone-coverage'
        assert groups[0].gate_task_id == '4220'
        assert skips == []

    def test_the_patch_moves_the_topic_and_touches_no_sibling_key(self):
        provenance = {'report_run': 'r1', 'observed_members': ['a'],
                      'detector': 'd', 'authoritative': False}
        kept = [{'id': 'x', 'why': 'peer'}]
        gates = [gate_task('4220', 'gate_topic', provenance=provenance,
                            considered_and_kept=kept)]
        renames = [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')]

        groups, _skips = pair_gate_blocks(renames, gates)

        assert groups[0].gate_block == {
            'topic': 'gate-topic',
            'provenance': provenance,
            'considered_and_kept': kept,
        }

    def test_the_patch_does_not_mutate_the_source_task(self):
        gates = [gate_task('4220', 'gate_topic', provenance={'a': 1})]
        renames = [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')]

        pair_gate_blocks(renames, gates)

        assert gates[0]['metadata'][GATE_METADATA_KEY]['topic'] == 'gate_topic'

    def test_every_rename_on_the_slug_joins_the_group(self):
        renames = [
            Rename('dark_factory', f'm{i}', 'gate_topic', 'gate-topic')
            for i in range(4)
        ]
        groups, _skips = pair_gate_blocks(
            renames, [gate_task('4220', 'gate_topic')])

        assert len(groups) == 1
        assert [r.memory_id for r in groups[0].renames] == ['m0', 'm1', 'm2', 'm3']

    def test_a_rename_with_no_gate_still_forms_a_group(self):
        """Most topics are not gates; those renames must proceed normally."""
        renames = [Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic')]

        groups, skips = pair_gate_blocks(renames, [])

        assert len(groups) == 1
        assert groups[0].gate_task_id is None
        assert groups[0].gate_block is None
        assert skips == []

    def test_a_gate_is_matched_within_its_own_project(self):
        renames = [Rename('reify', 'm1', 'gate_topic', 'gate-topic')]
        gates = [gate_task('4220', 'gate_topic', project_id='dark_factory')]

        groups, skips = pair_gate_blocks(renames, gates)

        assert groups[0].gate_task_id is None
        assert by_reason(skips, 'orphan_gate_topic')

    def test_a_gate_whose_slug_matches_no_record_is_an_orphan_not_a_silence(self):
        gates = [gate_task('4774', 'ghost_topic')]

        groups, skips = pair_gate_blocks([], gates)

        assert groups == []
        orphans = by_reason(skips, 'orphan_gate_topic')
        assert len(orphans) == 1
        assert orphans[0]['gate_task_id'] == '4774'
        assert orphans[0]['gate_topic'] == 'ghost_topic'
        assert orphans[0]['project_id'] == 'dark_factory'

    def test_a_gate_already_on_a_conforming_slug_is_not_an_orphan(self):
        """Nothing to migrate is not the same fact as a dangling gate."""
        groups, skips = pair_gate_blocks([], [gate_task('4220', 'good-topic')])

        assert groups == []
        assert skips == []

    def test_a_gate_task_without_a_block_is_ignored(self):
        task = {'id': '9', 'project_id': 'dark_factory', 'metadata': {}}

        groups, skips = pair_gate_blocks([], [task])

        assert (groups, skips) == ([], [])

    def test_groups_are_sorted_for_a_clean_diff(self):
        renames = [
            Rename('reify', 'm9', 'z_topic', 'z-topic'),
            Rename('dark_factory', 'm2', 'b_topic', 'b-topic'),
            Rename('dark_factory', 'm1', 'a_topic', 'a-topic'),
        ]
        groups, _skips = pair_gate_blocks(renames, [])

        assert [(g.project_id, g.new_topic) for g in groups] == [
            ('dark_factory', 'a-topic'),
            ('dark_factory', 'b-topic'),
            ('reify', 'z-topic'),
        ]

    def test_the_group_is_frozen(self):
        groups, _skips = pair_gate_blocks(
            [Rename('dark_factory', 'm1', 'a_topic', 'a-topic')], [])
        assert dataclasses.is_dataclass(groups[0])
        with pytest.raises(dataclasses.FrozenInstanceError):
            groups[0].new_topic = 'other'  # type: ignore[misc]


class TestAmbiguousGate:
    """A legacy slug carrying MORE THAN ONE consolidation gate is refused.

    A ``GateGroup`` moves exactly one gate.  Moving one of several would strand
    the rest on the vacated slug — the uncloseable-gate trap again — so the
    slug's renames are withheld and both stores stay on the legacy value, the
    posture ``slug_collision`` already takes.
    """

    @staticmethod
    def _gates_on(slug: str, *task_ids: str) -> list[dict]:
        return [gate_task(task_id, slug) for task_id in task_ids]

    def test_a_doubly_gated_slug_forms_no_group_and_is_refused_once(self):
        renames = [
            Rename('dark_factory', 'm2', 'shared_gate', 'shared-gate'),
            Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate'),
        ]

        groups, skips = pair_gate_blocks(
            renames, self._gates_on('shared_gate', '7718', '7011'))

        assert groups == []
        (entry,) = by_reason(skips, 'gate_ambiguous')
        assert entry['project_id'] == 'dark_factory'
        assert entry['old_topic'] == 'shared_gate'
        assert entry['new_topic'] == 'shared-gate'
        assert entry['gate_task_ids'] == ['7011', '7718']
        assert entry['memory_ids'] == ['m1', 'm2']

    def test_neither_gate_is_reported_as_an_orphan(self):
        """They are matched to live records; "matched to no record" would be false."""
        _groups, skips = pair_gate_blocks(
            [Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')],
            self._gates_on('shared_gate', '7011', '7718'))

        assert by_reason(skips, 'orphan_gate_topic') == []

    def test_the_refusal_is_per_slug(self):
        renames = [
            Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate'),
            Rename('dark_factory', 'm2', 'plain_topic', 'plain-topic'),
            Rename('dark_factory', 'm3', 'single_gate', 'single-gate'),
        ]
        single = gate_task('4220', 'single_gate')

        groups, _skips = pair_gate_blocks(
            renames, [*self._gates_on('shared_gate', '7011', '7718'), single])

        assert [(g.old_topic, g.gate_task_id) for g in groups] == [
            ('plain_topic', None),
            ('single_gate', '4220'),
        ]
        unaffected, _ = pair_gate_blocks(
            [r for r in renames if r.old_topic != 'shared_gate'], [single])
        assert groups == unaffected

    def test_the_verdict_does_not_depend_on_gate_order(self):
        renames = [Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')]
        gates = self._gates_on('shared_gate', '7011', '7718')

        for ordered in (gates, list(reversed(gates))):
            groups, skips = pair_gate_blocks(renames, ordered)

            assert groups == []
            (entry,) = by_reason(skips, 'gate_ambiguous')
            assert entry['gate_task_ids'] == ['7011', '7718']

    def test_gates_in_different_projects_are_not_ambiguous(self):
        """Gates are keyed per project, as the closure scroll that reads them is."""
        renames = [Rename('dark_factory', 'm1', 'shared_gate', 'shared-gate')]
        gates = [
            gate_task('4220', 'shared_gate', project_id='dark_factory'),
            gate_task('6852', 'shared_gate', project_id='reify'),
        ]

        groups, skips = pair_gate_blocks(renames, gates)

        assert [g.gate_task_id for g in groups] == ['4220']
        assert by_reason(skips, 'gate_ambiguous') == []


def _task_row(task_id: object, metadata: object) -> dict:
    """One task as ``get_tasks`` lists it, carrying fields the census must drop."""
    return {'id': task_id, 'title': f'task {task_id}', 'status': 'pending',
            'dependencies': [], 'metadata': metadata}


_GATE_META = {GATE_METADATA_KEY: {'topic': 'gate_slug'}}


class TestConsolidationGateCensus:
    """``census_consolidation_gates(client, roots) -> GateCensus``.

    The input the lockstep never had on the live path.  Its contract is to
    state what it could NOT see: a root that cannot be read becomes a
    structured failure, never a silent gap that reads as "no gates here".
    """

    @pytest.mark.asyncio
    async def test_one_unpaged_get_tasks_per_distinct_root(self):
        client = census_client({
            '/df': task_listing('dark_factory', '/df'),
            '/rf': task_listing('reify', '/rf'),
        })

        await census_consolidation_gates(client, ['/df', '/rf', '/df'])

        assert client.call_tool.await_args_list == [
            call('get_tasks', {'project_root': '/df'}),
            call('get_tasks', {'project_root': '/rf'}),
        ]

    @pytest.mark.asyncio
    async def test_only_tasks_carrying_a_gate_block_are_kept(self):
        client = census_client({'/df': task_listing(
            'dark_factory', '/df',
            _task_row(1, _GATE_META),
            _task_row(2, {'task_kind': 'deterministic'}),
            _task_row(3, 'not-a-dict'),
            _task_row(4, {GATE_METADATA_KEY: 'not-a-dict'}),
            _task_row(5, None),
        )})

        census = await census_consolidation_gates(client, ['/df'])

        assert [t['id'] for t in census.tasks] == ['1']

    @pytest.mark.asyncio
    async def test_a_kept_task_carries_the_server_stamp_and_nothing_else(self):
        """``project_id``/``project_root`` come off the RESPONSE, never the caller.

        The server normalizes a worktree root to its main checkout and derives
        the project id from that — the derivation the closure scroll runs
        under — so recomputing either here would be a second copy free to
        diverge.
        """
        block = {'topic': 'gate_slug', 'provenance': {'report_run': 'r1'}}
        client = census_client({'/main/.worktrees/7': task_listing(
            'dark_factory', '/main',
            _task_row(4220, {GATE_METADATA_KEY: block, 'task_kind': 'deterministic'}),
        )})

        census = await census_consolidation_gates(client, ['/main/.worktrees/7'])

        assert list(census.tasks) == [{
            'id': '4220',
            'project_id': 'dark_factory',
            'project_root': '/main',
            'metadata': {GATE_METADATA_KEY: block},
        }]

    @pytest.mark.asyncio
    async def test_tasks_are_sorted_by_project_then_id(self):
        client = census_client({
            '/rf': task_listing('reify', '/rf', _task_row(20, _GATE_META), _task_row(10, _GATE_META)),
            '/df': task_listing('dark_factory', '/df', _task_row(30, _GATE_META)),
        })

        census = await census_consolidation_gates(client, ['/rf', '/df'])

        assert [(t['project_id'], t['id']) for t in census.tasks] == [
            ('dark_factory', '30'), ('reify', '10'), ('reify', '20'),
        ]

    @pytest.mark.asyncio
    async def test_covered_projects_are_the_projects_that_answered(self):
        client = census_client({
            '/df': task_listing('dark_factory', '/df'),
            '/rf': task_listing('reify', '/rf'),
        })

        census = await census_consolidation_gates(client, ['/df', '/rf'])

        assert census.covered_projects == {'dark_factory', 'reify'}
        assert list(census.failures) == []

    @pytest.mark.parametrize('answer', [
        pytest.param({'error': 'no such project root', 'error_type': 'ValidationError'},
                     id='error-envelope'),
        pytest.param({'tasks': {'1': {}}, 'project_id': 'ghost', 'project_root': '/bad'},
                     id='tasks-not-a-list'),
        pytest.param({'tasks': [], 'project_root': '/bad'}, id='no-project-id'),
        pytest.param({'tasks': [], 'project_id': 'ghost'}, id='no-project-root'),
        pytest.param(RuntimeError('get_tasks failed: timed out'), id='raises'),
    ])
    @pytest.mark.asyncio
    async def test_a_root_that_cannot_be_read_is_a_failure_not_a_gap(self, answer):
        client = census_client({
            '/bad': answer,
            '/df': task_listing('dark_factory', '/df', _task_row(1, _GATE_META)),
        })

        census = await census_consolidation_gates(client, ['/bad', '/df'])

        (failure,) = census.failures
        assert failure['reason'] == 'gate_census_incomplete'
        assert failure['project_root'] == '/bad'
        assert failure['error']
        assert census.covered_projects == {'dark_factory'}
        assert [t['id'] for t in census.tasks] == ['1']

    @pytest.mark.parametrize('answer, cause', [
        pytest.param({'error': 'no such project root', 'error_type': 'ValidationError'},
                     'no such project root', id='error-envelope'),
        pytest.param(RuntimeError('get_tasks failed: timed out'), 'timed out', id='raises'),
    ])
    @pytest.mark.asyncio
    async def test_the_failure_carries_the_cause_it_was_given(self, answer, cause):
        census = await census_consolidation_gates(
            census_client({'/bad': answer}), ['/bad'])

        (failure,) = census.failures
        assert cause in failure['error']

    @pytest.mark.asyncio
    async def test_one_project_answering_from_two_task_stores_is_ambiguous(self):
        client = census_client({
            '/a': task_listing('reify', '/a'),
            '/b': task_listing('reify', '/b'),
            '/df': task_listing('dark_factory', '/df'),
        })

        census = await census_consolidation_gates(client, ['/a', '/b', '/df'])

        assert census.covered_projects == {'dark_factory'}
        (failure,) = census.failures
        assert failure['reason'] == 'gate_census_incomplete'
        assert failure['project_id'] == 'reify'
        assert failure['project_root'] == '/b'
        assert '/a' in failure['error']

    @pytest.mark.asyncio
    async def test_two_roots_normalized_to_one_store_are_deduplicated(self):
        """A worktree path and its main checkout are one task store, read twice."""
        listing = task_listing('dark_factory', '/main', _task_row(1, _GATE_META))
        client = census_client({'/main/.worktrees/7': listing, '/main': listing})

        census = await census_consolidation_gates(
            client, ['/main/.worktrees/7', '/main'])

        assert list(census.failures) == []
        assert census.covered_projects == {'dark_factory'}
        assert [t['id'] for t in census.tasks] == ['1']

    @pytest.mark.asyncio
    async def test_the_census_records_the_roots_it_attempted(self):
        client = census_client({
            '/df': task_listing('dark_factory', '/df'),
            '/bad': RuntimeError('down'),
        })

        census = await census_consolidation_gates(client, ['/df', '/bad', '/df'])

        assert list(census.roots) == ['/df', '/bad']

    def test_the_census_is_a_frozen_value(self):
        census = GateCensus(
            roots=('/df',), tasks=(), covered_projects=frozenset({'dark_factory'}),
            failures=())

        assert dataclasses.is_dataclass(census)
        with pytest.raises(dataclasses.FrozenInstanceError):
            census.tasks = ()  # type: ignore[misc]
