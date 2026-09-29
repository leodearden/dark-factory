"""Post-write check of Graphiti dependency-direction extraction (task 3770).

The extraction names real, adjacent task ids but can state the direction wrong:
it flattens parallel tasks into a sequence and inverts transitive chains. The
decision core is pure, so most of this suite runs against a plain dict; the
write-path tests at the end drive the MemoryService seam the durable queue
dispatches into.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import MockAddEpisodeResult, MockEdge, install_identity_mocks

from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.dependency_direction_check import (
    REVERSED,
    SIBLING_SEQUENTIAL,
    UNSUPPORTED,
    DependencyAssertion,
    DependencyIndex,
    build_dependency_index,
    check_dependency_direction,
    classify_dependency_assertion,
    extract_dependency_assertions,
    extract_dependency_facts,
)
from fused_memory.services.memory_service import MemoryService, ReconcileStats

_MS_LOGGER = 'fused_memory.services.memory_service'

# The dark_factory dependency graph as of the bad extraction (2026-08-05). Keep
# it FROZEN: the live graph has drifted (task 5020 now depends on both 3730 and
# 3733), and only in this shape do 3730/3733 share dependencies but no
# dependent, which is what exercises the shared-dependency sibling arm.
#   closure(3578) = {3256, 3619, 3727, 3618}
#   closure(3730) = {3578, 3727, 3728, 3256, 3619, 3618}
#   closure(3733) = {3578, 3728, 3256, 3727, 3619, 3618}
#   closure(3619) = {3256, 3618}
#   closure(3618) = {}
LIVE_SHAPE_EDGES: dict[int, list[int]] = {
    3256: [],
    3618: [],
    3619: [3256, 3618],
    3727: [3256],
    3728: [3256, 3727],
    3578: [3256, 3619, 3727],
    3730: [3578, 3727, 3728],
    3733: [3578, 3728],
}


class _DictEdge(dict):
    """A dict-shaped edge, to prove the pure core needs no graphiti_core."""


class _AttrEdge:
    def __init__(self, uuid: str, fact: str) -> None:
        self.uuid = uuid
        self.fact = fact


# ── extract_dependency_assertions — the per-fact parser ────────────────────


class TestExtractDependencyAssertions:
    def test_forward_phrase_binds_dependent_then_dependency(self):
        assertions = extract_dependency_assertions(
            'Task 3727 waits behind task 3619'
        )
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]
        assert assertions[0].phrase.lower() == 'waits behind'

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 3727 needs task 3619',
            'Task 3727 depends on task 3619',
            'Task 3727 is blocked by task 3619',
            'Task 3727 is blocked on task 3619',
            'Task 3727 is waiting on task 3619',
        ],
    )
    def test_other_forward_phrasings_bind_the_same_direction(self, fact):
        assertions = extract_dependency_assertions(fact)
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 3619 blocks task 3727',
            'Task 3619 gates task 3727',
            'Task 3619 unblocks task 3727',
            'Task 3619 is a dependency of task 3727',
        ],
    )
    def test_inverse_phrase_yields_the_swapped_pair(self, fact):
        """Direction comes from the PHRASE, never from word order."""
        assertions = extract_dependency_assertions(fact)
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]

    def test_multi_clause_chain_yields_both_pairs(self):
        assertions = extract_dependency_assertions(
            'task 3727 waits behind task 3619, task 3730 needs task 3733'
        )
        assert [(a.dependent, a.dependency) for a in assertions] == [
            (3727, 3619),
            (3730, 3733),
        ]

    @pytest.mark.parametrize(
        'fact',
        [
            # No direction-bearing phrase at all.
            'Task 3727 and task 3619 were both filed today',
            # Only one task reference.
            'Task 3727 depends on the merge queue draining',
            # A dependency phrase but no task references.
            'The parser depends on the extraction stage running first',
            # The phrase and the two refs sit in DIFFERENT clauses.
            'Task 3727 waits behind. Task 3619 was filed',
            'Task 3727 waits behind; task 3619 landed',
            # Past tense is history, not a claim about the current graph.
            'Task 3727 was blocked by task 3619',
            'Task 3727 was waiting on task 3619',
            'Task 3727 and task 3730 were blocked by task 3619',
        ],
    )
    def test_out_of_scope_returns_empty(self, fact):
        assert extract_dependency_assertions(fact) == []

    @pytest.mark.parametrize(
        'fact',
        [
            'Task 3727 needs task 3619',
            'task #3727 needs task #3619',
            'task/3727 needs task/3619',
            '#3727 needs #3619',
        ],
    )
    def test_accepts_the_inherited_task_ref_spellings(self, fact):
        assertions = extract_dependency_assertions(fact)
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]

    def test_assertion_record_is_frozen(self):
        assertion = extract_dependency_assertions(
            'Task 3727 waits behind task 3619'
        )[0]
        assert isinstance(assertion, DependencyAssertion)
        with pytest.raises(dataclasses.FrozenInstanceError):
            assertion.dependent = 1  # type: ignore[misc]

    @pytest.mark.parametrize('fact', [None, '', '   ', 12345, object()])
    def test_never_raises_on_a_non_string_or_empty_fact(self, fact):
        assert extract_dependency_assertions(fact) == []


# ── extract_dependency_facts — the episode-level scope gate ────────────────


class TestExtractDependencyFacts:
    def test_keeps_only_edges_whose_fact_makes_an_assertion(self):
        facts = extract_dependency_facts(
            [
                _AttrEdge('e1', 'Task 3727 waits behind task 3619'),
                _AttrEdge('e2', 'Task 3727 and task 3619 were both filed today'),
            ]
        )
        assert [(f.edge_uuid, f.fact) for f in facts] == [
            ('e1', 'Task 3727 waits behind task 3619')
        ]
        assert [(a.dependent, a.dependency) for a in facts[0].assertions] == [
            (3727, 3619)
        ]

    def test_dict_shaped_edges_are_accepted_too(self):
        facts = extract_dependency_facts(
            [_DictEdge(uuid='e1', fact='Task 3727 needs task 3619')]
        )
        assert [f.edge_uuid for f in facts] == ['e1']

    @pytest.mark.parametrize(
        'edge',
        [
            _AttrEdge('', 'Task 3727 waits behind task 3619'),   # no uuid
            _AttrEdge('edge-x', ''),                             # no fact
            _AttrEdge('edge-x', 'Tasks were filed today'),       # unparseable
            _DictEdge(),                                         # neither
            None,
        ],
    )
    def test_unusable_edges_are_skipped_never_raised_on(self, edge):
        assert extract_dependency_facts([edge]) == []


# ── build_dependency_index — direct, reverse and TRANSITIVE reachability ───


class TestBuildDependencyIndex:
    @pytest.fixture
    def index(self):
        return build_dependency_index(LIVE_SHAPE_EDGES)

    def test_index_is_frozen(self, index):
        assert isinstance(index, DependencyIndex)
        with pytest.raises(dataclasses.FrozenInstanceError):
            index.direct = {}  # type: ignore[misc]

    def test_direct_mirrors_the_input_edges(self, index):
        assert index.direct[3578] == frozenset({3256, 3619, 3727})
        assert index.direct[3730] == frozenset({3578, 3727, 3728})
        assert index.direct[3618] == frozenset()

    def test_closure_is_transitive(self, index):
        # 3618 is reachable from 3578 via 3619, though direct[3578] excludes it.
        assert 3618 not in index.direct[3578]
        assert index.closure_of(3578) == frozenset({3256, 3619, 3727, 3618})
        assert index.closure_of(3730) == frozenset(
            {3578, 3727, 3728, 3256, 3619, 3618}
        )
        assert index.closure_of(3733) == frozenset(
            {3578, 3728, 3256, 3727, 3619, 3618}
        )
        assert index.closure_of(3619) == frozenset({3256, 3618})
        assert index.closure_of(3618) == frozenset()

    def test_reverse_adjacency(self, index):
        assert index.dependents[3578] == frozenset({3730, 3733})
        assert index.dependents[3256] == frozenset({3619, 3727, 3728, 3578})
        assert index.dependents[3733] == frozenset()

    def test_cycle_terminates(self):
        index = build_dependency_index({1: [2], 2: [1]})
        assert index.closure_of(1) == frozenset({1, 2})
        assert index.closure_of(2) == frozenset({1, 2})

    def test_a_chain_deeper_than_the_recursion_limit_is_walked(self):
        depth = 5000
        index = build_dependency_index({i: [i + 1] for i in range(depth)})
        assert len(index.closure_of(0)) == depth
        assert (
            classify_dependency_assertion(
                DependencyAssertion(depth, 0, 'needs'), index
            )
            == REVERSED
        )

    def test_id_present_only_as_a_dependency_value_is_still_a_known_node(self):
        index = build_dependency_index({10: [11]})
        assert index.direct[11] == frozenset()
        assert index.dependents[11] == frozenset({10})
        assert index.closure_of(11) == frozenset()

    def test_caller_input_is_not_mutated(self):
        edges = {1: [2], 2: []}
        before = {k: list(v) for k, v in edges.items()}
        build_dependency_index(edges)
        assert edges == before


# ── classify_dependency_assertion — the decision core, and its ORDERING ────


def _assertion(dependent: int, dependency: int) -> DependencyAssertion:
    return DependencyAssertion(dependent, dependency, 'waits behind')


class TestClassifyDependencyAssertion:
    @pytest.fixture
    def index(self):
        return build_dependency_index(LIVE_SHAPE_EDGES)

    def test_a_supported_direct_edge_is_not_flagged(self, index):
        assert classify_dependency_assertion(_assertion(3578, 3619), index) is None

    def test_b_supported_transitively_is_not_flagged(self, index):
        assert 3618 not in index.direct[3730]
        assert classify_dependency_assertion(_assertion(3730, 3618), index) is None

    def test_c_inverted_transitive_chain_is_reversed(self, index):
        assert 3618 not in index.direct[3578]
        assert (
            classify_dependency_assertion(_assertion(3618, 3578), index) == REVERSED
        )

    def test_d_shared_dependent_arm_is_sibling_sequential(self, index):
        """3619 and 3727 are both depended on by 3578."""
        assert (
            classify_dependency_assertion(_assertion(3727, 3619), index)
            == SIBLING_SEQUENTIAL
        )

    def test_e_shared_dependency_arm_is_sibling_sequential(self, index):
        """3730 and 3733 share the dependencies 3728 and 3578 but no dependent."""
        assert not (index.dependents[3730] & index.dependents[3733])
        assert (
            classify_dependency_assertion(_assertion(3730, 3733), index)
            == SIBLING_SEQUENTIAL
        )

    def test_f_closure_precedence_beats_the_sibling_arm(self, index):
        """3730 and 3728 share the dependency 3727, yet 3730 -> 3728 is real."""
        assert index.direct[3730] & index.direct[3728] == frozenset({3727})
        assert classify_dependency_assertion(_assertion(3730, 3728), index) is None

    @pytest.mark.parametrize(
        'pair', [(99999, 3619), (3727, 99999), (99999, 88888)]
    )
    def test_g_an_unknown_id_is_never_flagged(self, index, pair):
        assert classify_dependency_assertion(_assertion(*pair), index) is None

    def test_h_no_path_and_not_siblings_is_unsupported(self):
        index = build_dependency_index({1: [2], 3: [4]})
        assert (
            classify_dependency_assertion(_assertion(1, 3), index) == UNSUPPORTED
        )

    def test_i_a_self_referential_assertion_is_never_flagged(self, index):
        assert classify_dependency_assertion(_assertion(3727, 3727), index) is None


# ── THE LEAF SIGNAL ───────────────────────────────────────────────────────
#
# The three bad facts are verbatim from Graphiti episode
# f3d18584-4041-4397-9faa-d4a14c01f71d.

BAD_FACTS = {
    'edge-bad-1': 'Task 3727 waits behind task 3619',
    'edge-bad-2': 'Task 3730 needs task 3733',
    'edge-bad-3': 'Task 3618 waits behind task 3578',
}

CORRECT_FACTS = {
    # A direct edge.
    'edge-ok-1': 'Task 3578 waits behind task 3619',
    # A direct edge whose pair also shares the dependency 3727.
    'edge-ok-2': 'Task 3730 needs task 3728',
    # True only transitively (3730 -> 3578 -> 3619 -> 3618).
    'edge-ok-3': 'Task 3730 waits behind task 3618',
}


def _check(facts: dict[str, str], edges: dict[int, list[int]] = LIVE_SHAPE_EDGES):
    parsed = extract_dependency_facts(
        [_AttrEdge(uuid, fact) for uuid, fact in facts.items()]
    )
    return check_dependency_direction(parsed, build_dependency_index(edges))


class TestKnownBadEpisodeFacts:
    def test_three_known_bad_facts_flagged_and_correct_facts_are_not(self):
        findings = _check({**BAD_FACTS, **CORRECT_FACTS})

        assert {f.edge_uuid: f.classification for f in findings} == {
            'edge-bad-1': SIBLING_SEQUENTIAL,
            'edge-bad-2': SIBLING_SEQUENTIAL,
            'edge-bad-3': REVERSED,
        }
        assert all(f.contradicts_ground_truth for f in findings)

    def test_findings_carry_verbatim_evidence(self):
        findings = {f.edge_uuid: f for f in _check(BAD_FACTS)}

        inverted = findings['edge-bad-3']
        assert inverted.fact == BAD_FACTS['edge-bad-3']
        assert (inverted.dependent, inverted.dependency) == (3618, 3578)
        assert inverted.ground_truth['dependent_direct_dependencies'] == frozenset()
        assert inverted.ground_truth['dependency_direct_dependencies'] == frozenset(
            {3256, 3619, 3727}
        )

    def test_to_dict_round_trips_and_is_json_serialisable(self):
        for finding in _check(BAD_FACTS):
            record = finding.to_dict()
            assert record['edge_uuid'] == finding.edge_uuid
            assert record['fact'] == finding.fact
            assert record['dependent'] == finding.dependent
            assert record['dependency'] == finding.dependency
            assert record['classification'] == finding.classification
            assert record['contradicts_ground_truth'] is True
            json.dumps(record)

    def test_finding_is_frozen(self):
        finding = _check(BAD_FACTS)[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.classification = REVERSED  # type: ignore[misc]


# ── True facts Taskmaster does not model are reported, never contradicted ──

#: Two orderings the dependency graph does not carry: merge-queue order between
#: unrelated tasks, and a dependency stated before it is wired.
UNMODELLED_EDGES: dict[int, list[int]] = {
    **LIVE_SHAPE_EDGES,
    5100: [4000],
    5101: [4001],
    5102: [4002],
}
UNMODELLED_FACTS = {
    'edge-merge-queue': 'Task 5101 waits behind task 5100',
    'edge-planned': 'Task 5102 depends on task 3619',
}


class TestTrueFactsTaskmasterDoesNotModel:
    def test_they_are_unsupported_not_contradicted(self):
        findings = _check(UNMODELLED_FACTS, UNMODELLED_EDGES)

        assert {f.edge_uuid: f.classification for f in findings} == {
            'edge-merge-queue': UNSUPPORTED,
            'edge-planned': UNSUPPORTED,
        }
        assert not any(f.contradicts_ground_truth for f in findings)
        assert all(
            f.to_dict()['contradicts_ground_truth'] is False for f in findings
        )


# ── The write path — the only place any I/O happens ───────────────────────

GROUP = 'dark_factory'


def _service(
    mock_config, *, ground_truth: dict[int, list[int]] = LIVE_SHAPE_EDGES
) -> MemoryService:
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    install_identity_mocks(svc.graphiti)
    svc.graphiti.update_edge = AsyncMock(return_value={})
    svc.taskmaster = AsyncMock()
    _tm(svc).get_dependency_edges = AsyncMock(return_value=ground_truth)
    svc.set_known_projects({GROUP: '/srv/dark-factory'})
    return svc


def _tm(svc: MemoryService) -> AsyncMock:
    return cast(AsyncMock, svc.taskmaster)


def _update_edge(svc: MemoryService) -> AsyncMock:
    return cast(AsyncMock, svc.graphiti.update_edge)


def _result(facts: dict[str, str], *, field: str = 'edges') -> MockAddEpisodeResult:
    edges = [MockEdge(fact=fact, uuid=uuid) for uuid, fact in facts.items()]
    if field == 'entity_edges':
        return MockAddEpisodeResult(entity_edges=edges)
    return MockAddEpisodeResult(edges=edges)


async def _write(
    svc: MemoryService,
    facts: dict[str, str],
    *,
    group_id: str = GROUP,
    field: str = 'edges',
) -> object:
    """Run one queued add_episode write whose extraction yields *facts*."""
    result = _result(facts, field=field)
    svc.graphiti.add_episode = AsyncMock(return_value=result)
    return await svc._execute_graphiti_write(
        'add_episode',
        {
            'name': 'ep',
            'content': 'episode body',
            'source': 'text',
            'group_id': group_id,
            'source_description': '',
        },
    )


async def _reconcile(
    svc: MemoryService, facts: dict[str, str], *, group_id: str = GROUP
) -> ReconcileStats:
    return await svc._reconcile_episode_identity(_result(facts), group_id=group_id)


def _retired(svc: MemoryService) -> set[str]:
    return {c.args[0] for c in _update_edge(svc).await_args_list}


def _warnings(caplog) -> str:
    return '\n'.join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )


class TestWritePath:
    @pytest.mark.asyncio
    async def test_an_out_of_scope_episode_never_reads_taskmaster(self, mock_config):
        svc = _service(mock_config)

        await _write(svc, {'e1': 'Task 3727 and task 3619 were both filed today'})

        _tm(svc).get_dependency_edges.assert_not_awaited()
        _update_edge(svc).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_historical_claim_is_out_of_scope(self, mock_config):
        svc = _service(mock_config)

        await _write(svc, {'e1': 'Task 3727 was blocked by task 3619'})

        _tm(svc).get_dependency_edges.assert_not_awaited()
        _update_edge(svc).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_contradicted_edges_are_retired_never_rewritten(self, mock_config):
        svc = _service(mock_config)

        await _write(svc, {**BAD_FACTS, **CORRECT_FACTS})

        assert _retired(svc) == set(BAD_FACTS)
        for call in _update_edge(svc).await_args_list:
            # invalid_at only: no positional or keyword fact.
            assert len(call.args) == 1
            assert call.kwargs.keys() == {'group_id', 'invalid_at'}
            assert call.kwargs['group_id'] == GROUP
            assert isinstance(call.kwargs['invalid_at'], datetime)

    @pytest.mark.asyncio
    async def test_the_project_root_comes_from_known_projects(self, mock_config):
        svc = _service(mock_config)

        await _write(svc, BAD_FACTS)

        _tm(svc).get_dependency_edges.assert_awaited_once_with('/srv/dark-factory')

    @pytest.mark.asyncio
    async def test_each_contradiction_is_logged_at_warning(self, mock_config, caplog):
        svc = _service(mock_config)

        with caplog.at_level(logging.WARNING, logger=_MS_LOGGER):
            await _write(svc, BAD_FACTS)

        blob = _warnings(caplog)
        for uuid, fact in BAD_FACTS.items():
            assert uuid in blob
            assert fact in blob
        assert REVERSED in blob and SIBLING_SEQUENTIAL in blob

    @pytest.mark.asyncio
    async def test_true_facts_taskmaster_does_not_model_stay_valid(
        self, mock_config, caplog
    ):
        svc = _service(mock_config, ground_truth=UNMODELLED_EDGES)

        with caplog.at_level(logging.WARNING, logger=_MS_LOGGER):
            await _write(svc, UNMODELLED_FACTS)

        _update_edge(svc).assert_not_awaited()
        blob = _warnings(caplog)
        for uuid in UNMODELLED_FACTS:
            assert uuid in blob
        assert UNSUPPORTED in blob

    @pytest.mark.asyncio
    async def test_an_unregistered_group_is_refused_never_given_a_fallback_root(
        self, mock_config, caplog
    ):
        # A non-empty configured root, so a fallback would have somewhere to go.
        mock_config.taskmaster = TaskmasterConfig(project_root='/srv/other-project')
        svc = _service(mock_config)
        svc.set_known_projects({'some_other_project': '/srv/elsewhere'})

        with caplog.at_level(logging.WARNING, logger=_MS_LOGGER):
            await _write(svc, BAD_FACTS)

        _tm(svc).get_dependency_edges.assert_not_awaited()
        _update_edge(svc).assert_not_awaited()
        assert GROUP in _warnings(caplog)

    @pytest.mark.asyncio
    async def test_no_taskmaster_leaves_every_edge_alone(self, mock_config):
        svc = _service(mock_config)
        svc.taskmaster = None

        await _write(svc, BAD_FACTS)

        _update_edge(svc).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_ground_truth_read_leaves_every_edge_alone(
        self, mock_config
    ):
        svc = _service(mock_config)
        _tm(svc).get_dependency_edges = AsyncMock(side_effect=RuntimeError('db'))

        await _write(svc, BAD_FACTS)

        _update_edge(svc).assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['edges', 'entity_edges'])
    async def test_both_result_edge_fields_are_honoured(self, mock_config, field):
        svc = _service(mock_config)

        await _write(svc, BAD_FACTS, field=field)

        assert _retired(svc) == set(BAD_FACTS)

    @pytest.mark.asyncio
    async def test_a_raising_check_never_fails_the_committed_write(
        self, mock_config
    ):
        svc = _service(mock_config, ground_truth={3727: 5})  # type: ignore[dict-item]

        result = await _write(svc, BAD_FACTS)

        assert isinstance(result, MockAddEpisodeResult)
        _update_edge(svc).assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt])
    async def test_lifecycle_exceptions_propagate(self, mock_config, exc):
        svc = _service(mock_config)
        _tm(svc).get_dependency_edges = AsyncMock(side_effect=exc)
        with pytest.raises(exc):
            await _write(svc, BAD_FACTS)

        svc = _service(mock_config)
        svc.graphiti.update_edge = AsyncMock(side_effect=exc)
        with pytest.raises(exc):
            await _write(svc, BAD_FACTS)


class TestReconcileStats:
    @pytest.mark.asyncio
    async def test_findings_are_folded_into_reconcile_stats(self, mock_config):
        svc = _service(mock_config)

        stats = await _reconcile(svc, {**BAD_FACTS, **CORRECT_FACTS})

        assert stats.errors == []
        assert stats.dependency_direction_flagged == 3
        assert {
            r['edge_uuid']: (r['fact'], r['contradicts_ground_truth'])
            for r in stats.dependency_direction_findings
        } == {uuid: (fact, True) for uuid, fact in BAD_FACTS.items()}

    @pytest.mark.asyncio
    async def test_unsupported_findings_are_recorded_as_not_contradicting(
        self, mock_config
    ):
        svc = _service(mock_config, ground_truth=UNMODELLED_EDGES)

        stats = await _reconcile(svc, UNMODELLED_FACTS)

        assert {
            r['edge_uuid']: (r['classification'], r['contradicts_ground_truth'])
            for r in stats.dependency_direction_findings
        } == {uuid: (UNSUPPORTED, False) for uuid in UNMODELLED_FACTS}

    @pytest.mark.asyncio
    async def test_a_contradiction_is_recorded_only_once_its_edge_is_retired(
        self, mock_config
    ):
        svc = _service(mock_config)
        svc.graphiti.update_edge = AsyncMock(side_effect=[RuntimeError('boom'), {}, {}])

        stats = await _reconcile(svc, BAD_FACTS)

        assert _update_edge(svc).await_count == 3
        failed_uuid = _update_edge(svc).await_args_list[0].args[0]
        assert stats.dependency_direction_flagged == 2
        assert failed_uuid not in {
            r['edge_uuid'] for r in stats.dependency_direction_findings
        }

    @pytest.mark.asyncio
    async def test_a_raising_check_is_recorded_as_an_error_with_no_findings(
        self, mock_config
    ):
        svc = _service(mock_config, ground_truth={3727: 5})  # type: ignore[dict-item]

        stats = await _reconcile(svc, BAD_FACTS)

        assert stats.errors == ['_check_dependency_direction']
        assert stats.dependency_direction_findings == []

    @pytest.mark.asyncio
    async def test_an_out_of_scope_episode_records_nothing(self, mock_config):
        svc = _service(mock_config)

        stats = await _reconcile(
            svc, {'e1': 'Task 3727 and task 3619 were both filed today'}
        )

        assert stats.dependency_direction_findings == []
        assert stats.dependency_direction_flagged == 0

    @pytest.mark.asyncio
    async def test_findings_never_leak_between_concurrently_reconciling_groups(
        self, mock_config
    ):
        # The identity lock is per group_id, so two groups reconcile
        # concurrently on one service. Group A is held inside its ground-truth
        # read while group B runs a whole reconcile.
        svc = _service(mock_config)
        svc.set_known_projects({'proj_a': '/srv/a', 'proj_b': '/srv/b'})
        a_reading = asyncio.Event()
        release_a = asyncio.Event()

        async def _edges(project_root: str) -> dict[int, list[int]]:
            if project_root == '/srv/a':
                a_reading.set()
                await release_a.wait()
            return LIVE_SHAPE_EDGES

        _tm(svc).get_dependency_edges = AsyncMock(side_effect=_edges)
        facts_a = {f'a-{k}': v for k, v in BAD_FACTS.items()}
        facts_b = {f'b-{k}': v for k, v in BAD_FACTS.items()}

        task_a = asyncio.create_task(_reconcile(svc, facts_a, group_id='proj_a'))
        await asyncio.wait_for(a_reading.wait(), 5)
        stats_b = await _reconcile(svc, facts_b, group_id='proj_b')
        release_a.set()
        stats_a = await asyncio.wait_for(task_a, 5)

        assert {r['edge_uuid'] for r in stats_a.dependency_direction_findings} == set(
            facts_a
        )
        assert {r['edge_uuid'] for r in stats_b.dependency_direction_findings} == set(
            facts_b
        )
