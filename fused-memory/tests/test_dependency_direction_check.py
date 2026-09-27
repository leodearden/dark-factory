"""Post-write sanity check for Graphiti dependency-direction extraction (task 3770).

Graphiti's extraction LLM does not hallucinate task numbers here — every task
it names is real, and adjacent to the ones it is genuinely about. What it gets
wrong is the DIRECTION: it FLATTENS sibling/parallel relations into sequential
ones ("A waits behind B" for two tasks that are merely both prerequisites of C)
and INVERTS transitive chains ("A waits behind B" where B in fact reaches A).
Because the numbers are plausible and adjacent, a planning read accepts the
resulting fact without friction — which makes this strictly more dangerous than
random hallucination, and is why the check exists.

The decision core is pure: the ground-truth graph is passed IN as data, so
every classification below is exercised against a plain dict with no Taskmaster,
no Graphiti and no I/O.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _fm_helpers import MockAddEpisodeResult, MockEdge, install_identity_mocks

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
)
from fused_memory.services.memory_service import (
    MemoryService,
    ReferentRepairStats,
    ReferentStats,
)

#: The logger the sub-pass emits its structured WARNING through.
_MS_LOGGER = 'fused_memory.services.memory_service'

# ── The frozen ground-truth fixture ────────────────────────────────────────
#
# PROVENANCE. Read READ-ONLY from `/home/leo/src/dark-factory/.taskmaster/
# tasks/tasks.db` (table `dependencies(tag, task_id, depends_on)`, tag
# 'master') during task-3770 planning. These are measured values, NOT invented
# ones. Note the DB path: `.taskmaster/tasks.db` is a 0-byte DECOY and is not
# the live database — the live one is `.taskmaster/tasks/tasks.db`.
#
# WHY IT IS FROZEN, and deliberately NOT refreshed from live. The constant
# reproduces the graph AS OF the bad extraction (2026-08-05). The LIVE graph has
# since DRIFTED, in two ways that both matter:
#
#   1. Task 3578 has gained the dependencies {3983, 4005}.
#   2. Task 5020 now depends on BOTH 3730 AND 3733 — so today those two share a
#      DEPENDENT, which they did NOT at extraction time.
#
# Drift (2) is the load-bearing one. In this frozen shape 3730/3733 share only
# DEPENDENCIES, which is the only reason the shared-dependency arm of the
# sibling predicate gets exercised at all. Refreshing against live would let
# that arm go untested, and the predicate could silently narrow back to the
# escalation's literal "two tasks that share a dependent" one-liner with no test
# failing. Independently, a live read would make the leaf signal
# non-reproducible: this suite must fail before the change and pass after, and a
# fixture that mutates under operator activity guarantees neither.
#
# A future reader who diffs this constant against live data will find the
# discrepancy already explained here rather than "correcting" it.
#
# DERIVED CLOSURES (recorded so expectations can be re-verified without
# re-deriving them by hand):
#   closure(3578) = {3256, 3619, 3727, 3618}
#   closure(3730) = {3578, 3727, 3728, 3256, 3619, 3618}
#   closure(3733) = {3578, 3728, 3256, 3727, 3619, 3618}
#   closure(3619) = {3256, 3618}
#   closure(3618) = {}  (a leaf: 3618 depends on nothing)
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


# ── extract_dependency_assertions — the scope gate plus parser ─────────────


class TestExtractDependencyAssertions:
    """The SCOPE GATE. Only compact multi-task dependency shorthand is in scope;
    everything else must return `[]` without the caller ever touching Taskmaster.
    """

    def test_forward_phrase_binds_dependent_then_dependency(self):
        assertions = extract_dependency_assertions(
            'Task 3727 waits behind task 3619'
        )
        assert [(a.dependent, a.dependency) for a in assertions] == [(3727, 3619)]
        assert 'waits behind' in assertions[0].phrase.lower()

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
            # The phrase and the two refs sit in DIFFERENT clauses: no
            # cross-clause binding.
            'Task 3727 waits behind. Task 3619 was filed',
            'Task 3727 waits behind; task 3619 landed',
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
        """The gate must not be narrower than TASK_REF_RE's canonical vocabulary."""
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


# ── build_dependency_index — direct, reverse and TRANSITIVE closure ────────


class TestBuildDependencyIndex:
    """The index is where refinement #1 lives: the closure is TRANSITIVE, so an
    inverted chain is detectable even when neither id appears in the other's
    direct edge list.
    """

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
        # 3618 is reachable from 3578 via 3619, though direct[3578] EXCLUDES
        # it — the whole point of computing a closure rather than checking
        # direct edges.
        assert 3618 not in index.direct[3578]
        assert index.closure[3578] == frozenset({3256, 3619, 3727, 3618})
        assert index.closure[3730] == frozenset(
            {3578, 3727, 3728, 3256, 3619, 3618}
        )
        assert index.closure[3733] == frozenset(
            {3578, 3728, 3256, 3727, 3619, 3618}
        )
        assert index.closure[3619] == frozenset({3256, 3618})
        assert index.closure[3618] == frozenset()

    def test_reverse_adjacency(self, index):
        assert index.dependents[3578] == frozenset({3730, 3733})
        assert index.dependents[3256] == frozenset({3619, 3727, 3728, 3578})
        # Frozen-fixture value: 5020 is deliberately absent, see the drift
        # note on LIVE_SHAPE_EDGES.
        assert index.dependents[3733] == frozenset()

    def test_diamond_is_deduplicated_not_double_counted(self):
        # 3578 reaches 3256 directly AND via 3619 — a set, not a multiset.
        index = build_dependency_index(LIVE_SHAPE_EDGES)
        assert sorted(index.closure[3578]) == [3256, 3618, 3619, 3727]

    def test_cycle_terminates(self):
        """A malformed cyclic graph must terminate, not blow the stack."""
        index = build_dependency_index({1: [2], 2: [1]})
        assert index.closure[1] == frozenset({1, 2})
        assert index.closure[2] == frozenset({1, 2})

    def test_id_present_only_as_a_dependency_value_is_still_a_known_node(self):
        index = build_dependency_index({10: [11]})
        assert 11 in index.closure
        assert index.closure[11] == frozenset()
        assert index.dependents[11] == frozenset({10})
        assert index.direct[11] == frozenset()

    def test_caller_input_is_not_mutated(self):
        edges = {1: [2], 2: []}
        before = {k: list(v) for k, v in edges.items()}
        build_dependency_index(edges)
        assert edges == before


# ── classify_dependency_assertion — the decision core, and its ORDERING ────


def _assertion(dependent: int, dependency: int) -> DependencyAssertion:
    return DependencyAssertion(dependent, dependency, 'waits behind')


class TestClassifyDependencyAssertion:
    """Five rules evaluated in a STRICT order. The order is behaviour, not an
    implementation detail: case (f) below fails outright if the sibling check is
    ever hoisted above the closure check.
    """

    @pytest.fixture
    def index(self):
        return build_dependency_index(LIVE_SHAPE_EDGES)

    def test_a_supported_direct_edge_is_not_flagged(self, index):
        assert classify_dependency_assertion(_assertion(3578, 3619), index) is None

    def test_b_supported_transitively_is_not_flagged(self, index):
        """3618 is reachable from 3730 only transitively — a direct-edge-only
        check would false-positive here."""
        assert 3618 not in index.direct[3730]
        assert classify_dependency_assertion(_assertion(3730, 3618), index) is None

    def test_c_inverted_transitive_chain_is_reversed(self, index):
        """Invisible against direct[3578]; detectable only via closure[3578]."""
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
        """3730 and 3733 both depend on 3728 and 3578, and in the FROZEN
        fixture share NO dependent — a predicate limited to the escalation's
        literal "share a dependent" would MISS this one."""
        assert not (index.dependents[3730] & index.dependents[3733])
        assert (
            classify_dependency_assertion(_assertion(3730, 3733), index)
            == SIBLING_SEQUENTIAL
        )

    def test_f_closure_precedence_beats_the_sibling_arm(self, index):
        """3730 and 3728 share the dependency 3727, so arm (e) matches — yet
        3730 -> 3728 is a REAL direct edge. Flagging it would be exactly the
        false-positive class the task forbids, so the closure check MUST be
        evaluated strictly before the sibling check."""
        assert index.direct[3730] & index.direct[3728] == frozenset({3727})
        assert classify_dependency_assertion(_assertion(3730, 3728), index) is None

    @pytest.mark.parametrize(
        'pair', [(99999, 3619), (3727, 99999), (99999, 88888)]
    )
    def test_g_an_unknown_id_is_never_flagged(self, index, pair):
        """Fail-safe under-selection: unknown means unknown, not wrong."""
        assert classify_dependency_assertion(_assertion(*pair), index) is None

    def test_h_no_path_and_not_siblings_is_unsupported(self):
        index = build_dependency_index({1: [2], 3: [4]})
        assert (
            classify_dependency_assertion(_assertion(1, 3), index) == UNSUPPORTED
        )

    def test_i_a_self_referential_assertion_is_never_flagged(self, index):
        assert classify_dependency_assertion(_assertion(3727, 3727), index) is None

    def test_vocabulary_has_one_normative_site(self):
        assert (REVERSED, SIBLING_SEQUENTIAL, UNSUPPORTED) == (
            'reversed',
            'sibling_sequential',
            'unsupported',
        )


# ── THE LEAF SIGNAL ───────────────────────────────────────────────────────
#
# The three VERBATIM facts below were extracted into Graphiti episode
# f3d18584-4041-4397-9faa-d4a14c01f71d. Every task number in them is real and
# adjacent; only the DIRECTION is wrong. Against the frozen ground truth all
# three must be flagged and none of the three genuinely-correct facts from the
# same neighbourhood may be.

BAD_FACTS = {
    'edge-bad-1': 'Task 3727 waits behind task 3619',
    'edge-bad-2': 'Task 3730 needs task 3733',
    'edge-bad-3': 'Task 3618 waits behind task 3578',
}

CORRECT_FACTS = {
    # A direct edge: 3619 IS in direct[3578].
    'edge-ok-1': 'Task 3578 waits behind task 3619',
    # A direct edge whose pair ALSO shares the dependency 3727 — the
    # sibling-arm false-positive trap.
    'edge-ok-2': 'Task 3730 needs task 3728',
    # True only TRANSITIVELY (3730 -> 3578 -> 3619 -> 3618) — the
    # direct-edge-only false-positive trap.
    'edge-ok-3': 'Task 3730 waits behind task 3618',
}


class _DictEdge(dict):
    """A dict-shaped edge, to prove the pure core needs no graphiti_core."""


class _AttrEdge:
    def __init__(self, uuid: str, fact: str) -> None:
        self.uuid = uuid
        self.fact = fact


class TestKnownBadEpisodeFacts:
    """Fails before the change, passes after: the three real bad facts are
    flagged with the right classifications and the three correct ones are not.
    """

    @pytest.fixture
    def index(self):
        return build_dependency_index(LIVE_SHAPE_EDGES)

    @pytest.fixture
    def edges(self):
        return [
            _AttrEdge(uuid, fact)
            for uuid, fact in {**BAD_FACTS, **CORRECT_FACTS}.items()
        ]

    def test_three_known_bad_facts_flagged_and_correct_facts_are_not(
        self, edges, index
    ):
        findings = check_dependency_direction(edges, index)

        assert len(findings) == 3
        assert {f.edge_uuid: f.classification for f in findings} == {
            'edge-bad-1': SIBLING_SEQUENTIAL,
            'edge-bad-2': SIBLING_SEQUENTIAL,
            'edge-bad-3': REVERSED,
        }
        # The no-false-positive companion assertion.
        flagged = {f.edge_uuid for f in findings}
        assert flagged.isdisjoint(CORRECT_FACTS)

    def test_findings_carry_verbatim_evidence(self, edges, index):
        findings = {f.edge_uuid: f for f in check_dependency_direction(edges, index)}

        inverted = findings['edge-bad-3']
        # The extraction's OWN wording, never a corrected one.
        assert inverted.fact == BAD_FACTS['edge-bad-3']
        assert (inverted.dependent, inverted.dependency) == (3618, 3578)
        assert inverted.classification == REVERSED
        assert inverted.ground_truth

    def test_to_dict_round_trips_and_is_json_serialisable(self, edges, index):
        findings = check_dependency_direction(edges, index)
        for finding in findings:
            record = finding.to_dict()
            assert record['edge_uuid'] == finding.edge_uuid
            assert record['fact'] == finding.fact
            assert record['dependent'] == finding.dependent
            assert record['dependency'] == finding.dependency
            assert record['classification'] == finding.classification
            # Sets rendered as sorted lists so this cannot raise.
            json.dumps(record)

    def test_dict_shaped_edges_are_accepted_too(self, index):
        edges = [
            _DictEdge(uuid=uuid, fact=fact) for uuid, fact in BAD_FACTS.items()
        ]
        findings = check_dependency_direction(edges, index)
        assert {f.edge_uuid for f in findings} == set(BAD_FACTS)

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
    def test_unusable_edges_are_skipped_never_raised_on(self, edge, index):
        assert check_dependency_direction([edge], index) == []

    def test_finding_is_frozen(self, edges, index):
        finding = check_dependency_direction(edges, index)[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            finding.classification = REVERSED  # type: ignore[misc]


# ── The MemoryService adapter — the only place any I/O happens ─────────────

GROUP = 'dark_factory'


def _service(mock_config):
    """MemoryService with fully-mocked backends.

    `install_identity_mocks` is required, not decorative: the post-write
    sub-passes run inside `_execute_graphiti_write`'s
    `async with self.graphiti._identity_lock_for(...)`, which a bare MagicMock
    cannot satisfy.
    """
    svc = MemoryService(mock_config)
    svc.graphiti = MagicMock()
    svc.graphiti.add_episode = AsyncMock(return_value=None)
    svc.graphiti._require_client = MagicMock()
    install_identity_mocks(svc.graphiti)
    svc.update_edge = AsyncMock(return_value={})
    svc.taskmaster = AsyncMock()
    svc.taskmaster.get_dependency_edges = AsyncMock(return_value=LIVE_SHAPE_EDGES)
    svc.set_known_projects({GROUP: '/srv/dark-factory'})
    return svc


def _tm(svc: MemoryService) -> AsyncMock:
    """The mocked Taskmaster backend, typed as the mock the test installed.

    `MemoryService.taskmaster` is declared `TaskBackendProtocol | None`, so a
    bare `svc.taskmaster.get_dependency_edges` reads as possibly-None AND as
    the protocol's own method -- neither of which carries the await-assertion
    surface these tests are written against.
    """
    return cast(AsyncMock, svc.taskmaster)


def _ue(svc: MemoryService) -> AsyncMock:
    """The mocked `update_edge`, typed as the mock the test installed.

    Same reason as `_tm`: the declared attribute is a real bound method.
    """
    return cast(AsyncMock, svc.update_edge)


def _result(facts: dict[str, str], *, field: str = 'edges'):
    edges = [MockEdge(fact=fact, uuid=uuid) for uuid, fact in facts.items()]
    # Dispatched explicitly rather than through `**{field: edges}`: unpacking a
    # `dict[str, list[MockEdge]]` makes the checker test that one value type
    # against EVERY field, `nodes: list[MockNode]` included.
    if field == 'entity_edges':
        return MockAddEpisodeResult(entity_edges=edges)
    return MockAddEpisodeResult(edges=edges)


class TestMemoryServiceSubPass:
    """The thin adapter: gate, resolve, read, classify, flag. It never repairs."""

    @pytest.mark.asyncio
    async def test_a_out_of_scope_short_circuits_before_any_taskmaster_read(
        self, mock_config
    ):
        svc = _service(mock_config)
        result = _result({'e1': 'Task 3727 and task 3619 were both filed today'})

        assert await svc._check_dependency_direction(result, group_id=GROUP) == []

        # The restriction to compact dependency shorthand is what keeps a
        # blanket check off every write. Pinned so it cannot silently erode.
        _tm(svc).get_dependency_edges.assert_not_awaited()
        _ue(svc).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_b_flagged_edges_are_invalidated_never_rewritten(self, mock_config):
        svc = _service(mock_config)
        result = _result({**BAD_FACTS, **CORRECT_FACTS})

        records = await svc._check_dependency_direction(result, group_id=GROUP)

        assert {r['edge_uuid'] for r in records} == set(BAD_FACTS)
        for record in records:
            assert record['fact'] == BAD_FACTS[record['edge_uuid']]
            assert isinstance(record['dependent'], int)
            assert isinstance(record['dependency'], int)
            assert record['classification'] in (REVERSED, SIBLING_SEQUENTIAL)
            assert 'ground_truth' in record

        calls = _ue(svc).await_args_list
        assert {c.args[0] for c in calls} == set(BAD_FACTS)
        for call in calls:
            assert call.kwargs['invalid_at'] is not None
            # Flag, never silently correct.
            assert 'fact' not in call.kwargs

    @pytest.mark.asyncio
    async def test_c_correct_facts_are_never_invalidated(self, mock_config):
        svc = _service(mock_config)
        result = _result({**BAD_FACTS, **CORRECT_FACTS})
        await svc._check_dependency_direction(result, group_id=GROUP)
        touched = {c.args[0] for c in _ue(svc).await_args_list}
        assert touched.isdisjoint(CORRECT_FACTS)

    @pytest.mark.asyncio
    async def test_d_each_mismatch_is_logged_at_warning_with_the_finding(
        self, mock_config, caplog
    ):
        svc = _service(mock_config)
        with caplog.at_level(logging.WARNING, logger=_MS_LOGGER):
            await svc._check_dependency_direction(_result(BAD_FACTS), group_id=GROUP)

        blob = '\n'.join(
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        )
        for uuid, fact in BAD_FACTS.items():
            assert uuid in blob
            assert fact in blob
        assert REVERSED in blob and SIBLING_SEQUENTIAL in blob

    @pytest.mark.asyncio
    async def test_e_project_root_resolution_and_its_fallback(self, mock_config):
        svc = _service(mock_config)
        result = _result(BAD_FACTS)

        await svc._check_dependency_direction(result, group_id=GROUP)
        assert _tm(svc).get_dependency_edges.await_args.args[0] == (
            '/srv/dark-factory'
        )

        # Unknown group -> the _memory_metadata_project_root() fallback.
        _tm(svc).get_dependency_edges.reset_mock()
        svc.set_known_projects({})
        svc._memory_metadata_project_root = MagicMock(return_value='/fallback')
        await svc._check_dependency_direction(result, group_id=GROUP)
        assert _tm(svc).get_dependency_edges.await_args.args[0] == '/fallback'

        # Unresolvable -> 0, and no taskmaster call at all.
        _tm(svc).get_dependency_edges.reset_mock()
        svc._memory_metadata_project_root = MagicMock(return_value='')
        assert await svc._check_dependency_direction(result, group_id=GROUP) == []
        _tm(svc).get_dependency_edges.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_f_no_taskmaster_returns_no_records_without_raising(self, mock_config):
        svc = _service(mock_config)
        svc.taskmaster = None
        assert await svc._check_dependency_direction(
            _result(BAD_FACTS), group_id=GROUP
        ) == []

    @pytest.mark.asyncio
    async def test_g_ground_truth_read_failure_returns_no_records(self, mock_config):
        svc = _service(mock_config)
        _tm(svc).get_dependency_edges = AsyncMock(side_effect=RuntimeError('db'))
        assert await svc._check_dependency_direction(
            _result(BAD_FACTS), group_id=GROUP
        ) == []
        _ue(svc).assert_not_awaited()

    @pytest.mark.asyncio
    async def test_g_per_edge_failure_is_best_effort(self, mock_config):
        svc = _service(mock_config)
        svc.update_edge = AsyncMock(
            side_effect=[RuntimeError('boom'), {}, {}]
        )
        # The remaining flagged edges are still attempted.
        records = await svc._check_dependency_direction(
            _result(BAD_FACTS), group_id=GROUP
        )
        assert svc.update_edge.await_count == 3

        # A record means the edge was really invalidated, never that
        # invalidation was merely attempted.
        failed_uuid = svc.update_edge.await_args_list[0].args[0]
        assert len(records) == 2
        assert failed_uuid not in {r['edge_uuid'] for r in records}

    @pytest.mark.asyncio
    @pytest.mark.parametrize('exc', [asyncio.CancelledError, KeyboardInterrupt])
    async def test_g_lifecycle_exceptions_propagate(self, mock_config, exc):
        svc = _service(mock_config)
        _tm(svc).get_dependency_edges = AsyncMock(side_effect=exc)
        with pytest.raises(exc):
            await svc._check_dependency_direction(_result(BAD_FACTS), group_id=GROUP)

        svc = _service(mock_config)
        svc.update_edge = AsyncMock(side_effect=exc)
        with pytest.raises(exc):
            await svc._check_dependency_direction(_result(BAD_FACTS), group_id=GROUP)

    @pytest.mark.asyncio
    @pytest.mark.parametrize('field', ['edges', 'entity_edges'])
    async def test_h_both_result_edge_fields_are_honoured(self, mock_config, field):
        svc = _service(mock_config)
        result = _result(BAD_FACTS, field=field)
        records = await svc._check_dependency_direction(result, group_id=GROUP)
        assert len(records) == 3


class TestReconcileEpisodeIdentityWiring:
    """The check is the NINTH `_run_pass` — appended AFTER zeta/eta, whose
    documented load-bearing "runs last" ordering must stay undisturbed.
    """

    @pytest.mark.asyncio
    async def test_findings_are_folded_into_reconcile_stats(self, mock_config):
        svc = _service(mock_config)
        stats = await svc._reconcile_episode_identity(
            _result(BAD_FACTS), group_id=GROUP
        )

        assert stats.dependency_direction_flagged == 3
        assert len(stats.dependency_direction_findings) == 3
        for record in stats.dependency_direction_findings:
            assert record['edge_uuid'] in BAD_FACTS
            assert record['fact'] == BAD_FACTS[record['edge_uuid']]
            assert isinstance(record['dependent'], int)
            assert isinstance(record['dependency'], int)
            assert record['classification'] in (REVERSED, SIBLING_SEQUENTIAL)

    @pytest.mark.asyncio
    async def test_the_eight_pre_existing_stats_fields_are_undisturbed(
        self, mock_config
    ):
        svc = _service(mock_config)
        stats = await svc._reconcile_episode_identity(
            _result(BAD_FACTS), group_id=GROUP
        )

        assert stats.errors == []
        for name in (
            'edges_deduped',
            'dependency_edges_restored',
            'sibling_edges_restored',
            'stale_ttl_edges_invalidated',
            'nodes_resolved',
            'task_names_normalized',
        ):
            assert isinstance(getattr(stats, name), int)
        # zeta/eta still return their own dataclasses: the new pass did not
        # disturb their ordering contract or eta's data dependency on zeta.
        assert isinstance(stats.referent_stats, ReferentStats)
        assert isinstance(stats.repair_stats, ReferentRepairStats)

    @pytest.mark.asyncio
    async def test_a_raising_check_never_fails_the_committed_write(
        self, mock_config, monkeypatch
    ):
        svc = _service(mock_config)

        async def _boom(*a, **kw):
            raise RuntimeError('checker bug')

        monkeypatch.setattr(svc, '_check_dependency_direction', _boom)
        stats = await svc._reconcile_episode_identity(
            _result(BAD_FACTS), group_id=GROUP
        )

        assert '_check_dependency_direction' in stats.errors
        assert stats.dependency_direction_flagged == 0
        # A swallowed failure must never leave partial findings that read as a
        # clean result.
        assert stats.dependency_direction_findings == []

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates(self, mock_config, monkeypatch):
        svc = _service(mock_config)

        async def _cancel(*a, **kw):
            raise asyncio.CancelledError

        monkeypatch.setattr(svc, '_check_dependency_direction', _cancel)
        with pytest.raises(asyncio.CancelledError):
            await svc._reconcile_episode_identity(_result(BAD_FACTS), group_id=GROUP)

    @pytest.mark.asyncio
    async def test_findings_never_leak_between_concurrently_reconciling_groups(
        self, mock_config
    ):
        # The identity lock in `_execute_graphiti_write` is per-group_id, so
        # writes for DIFFERENT groups reconcile concurrently on one service.
        # Group A is held inside its ground-truth read while group B runs a
        # whole reconcile; neither may see the other's records.
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
        result_a = _result({f'a-{k}': v for k, v in BAD_FACTS.items()})
        result_b = _result({f'b-{k}': v for k, v in BAD_FACTS.items()})

        task_a = asyncio.create_task(
            svc._reconcile_episode_identity(result_a, group_id='proj_a')
        )
        await asyncio.wait_for(a_reading.wait(), 5)
        stats_b = await svc._reconcile_episode_identity(result_b, group_id='proj_b')
        release_a.set()
        stats_a = await asyncio.wait_for(task_a, 5)

        assert {r['edge_uuid'] for r in stats_a.dependency_direction_findings} == {
            f'a-{k}' for k in BAD_FACTS
        }
        assert {r['edge_uuid'] for r in stats_b.dependency_direction_findings} == {
            f'b-{k}' for k in BAD_FACTS
        }
        assert stats_a.dependency_direction_flagged == 3
        assert stats_b.dependency_direction_flagged == 3

    @pytest.mark.asyncio
    async def test_an_out_of_scope_episode_leaves_both_fields_at_defaults(
        self, mock_config
    ):
        svc = _service(mock_config)
        stats = await svc._reconcile_episode_identity(
            _result({'e1': 'Task 3727 and task 3619 were both filed today'}),
            group_id=GROUP,
        )
        assert stats.dependency_direction_flagged == 0
        assert stats.dependency_direction_findings == []
