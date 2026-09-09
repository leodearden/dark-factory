"""C3 boundary policy contract — the declared matrix, then the behaviour.

This file opens with assertions on the module's DATA, before any middleware
runs. That is deliberate and is what INV-1 asks for: the policy is a declared,
total matrix over outcome x tool class, so it can be checked as a table rather
than inferred by driving every path and hoping the cases were exhaustive.

## Why the behavioural half drives a Client

Middleware sits at the SERVER REQUEST layer, and both in-process idioms this
repo established elsewhere BYPASS it entirely. MEASURED, and recorded in
``shared/src/shared/mcp_markup_middleware.py``'s own test module:

* fused-memory's ``await server._tool_manager.call_tool(name, args)``
* orchestrator/escalation's ``await server.get_tool(n)`` then ``tool.fn(...)``
  / ``await tool.run({...})``

A test written either way would pass while exercising nothing — the guard
would never run. Only ``async with Client(mcp)`` runs registered middleware, so
it is the only harness that can exercise this contract.

## Why toy servers, and a fake resolver rather than a mock

``shared`` is the base layer: its tests may not import ``fused_memory``, which
is exactly why the resolver is an injected PORT rather than a dependency. The
toy tools below mirror the real victim signatures (``add_memory``,
``update_task``, ``submit_task``, ``delete_memory``) and RECORD the arguments
they actually received. That is what makes "the tool received the expanded
value" and "the tool never ran" DIRECTLY assertable, rather than inferred from
what the guard says on ``meta`` — meta reports intent, the recorder reports
what actually landed, and only the second one is the contract.

No test here asserts on docstring or comment prose.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping
from typing import Any, get_args

import pytest
from fastmcp import Client, FastMCP

from shared import uuid_prefix_guard as guard

# --- the resolution vocabulary is typed and closed ----------------------


def test_candidate_and_resolution_are_namedtuples() -> None:
    for declared in (guard.Candidate, guard.Resolution):
        assert issubclass(declared, tuple)
        assert hasattr(declared, '_fields')
        assert hasattr(declared, '_replace')


def test_candidate_fields() -> None:
    assert guard.Candidate._fields == ('namespace', 'id', 'preview')


def test_resolution_fields() -> None:
    assert guard.Resolution._fields == ('outcome', 'candidates')


def test_namespace_constants_are_derived_from_the_literal_type() -> None:
    """One source for the type and the constant, so they cannot drift apart."""
    assert get_args(guard.Namespace) == guard.NAMESPACES
    assert set(guard.NAMESPACES) == {'mem0', 'graphiti_node', 'graphiti_edge'}


def test_resolution_outcome_constants_are_derived_from_the_literal_type() -> None:
    assert get_args(guard.ResolutionOutcome) == guard.RESOLUTION_OUTCOMES
    assert set(guard.RESOLUTION_OUTCOMES) == {'unique', 'ambiguous', 'none'}


def test_resolver_unavailable_is_an_exception_carrying_the_failed_store() -> None:
    """INV-11: an outage must name which store failed, never fail soft silently."""
    assert issubclass(guard.ResolverUnavailable, Exception)
    error = guard.ResolverUnavailable('mem0')
    assert error.store == 'mem0'
    assert 'mem0' in str(error)


# --- the fact and tool-class vocabularies -------------------------------


def test_fact_outcome_vocabulary_is_exactly_the_four_declared_values() -> None:
    assert get_args(guard.FactOutcome) == guard.FACT_OUTCOMES
    assert set(guard.FACT_OUTCOMES) == {
        'expanded',
        'rejected',
        'forwarded_ambiguous',
        'resolver_unavailable',
    }


def test_tool_classes_are_exactly_the_three_declared_ones() -> None:
    assert {c.value for c in guard.ToolClass} == {'default', 'forward_on_ambiguity', 'exempt'}


def test_prefix_actions_are_exactly_the_five_declared_ones() -> None:
    assert {a.value for a in guard.PrefixAction} == {
        'substitute_and_forward',
        'reject_with_candidates',
        'forward_with_candidates',
        'forward_unchanged',
        'inert',
    }


# --- the matrix is TOTAL ------------------------------------------------

NON_EXEMPT_CLASSES = (guard.ToolClass.DEFAULT, guard.ToolClass.FORWARD_ON_AMBIGUITY)


@pytest.mark.parametrize(
    ('outcome', 'tool_class'),
    list(itertools.product(guard.RESOLUTION_OUTCOMES, NON_EXEMPT_CLASSES)),
)
def test_every_cell_is_declared(
    outcome: guard.ResolutionOutcome, tool_class: guard.ToolClass
) -> None:
    """No cell is reachable by falling off the end of a lookup (INV-1).

    Parametrized over the PRODUCT of the two vocabularies rather than over a
    hand-listed set of pairs, so adding an outcome or a tool class without
    declaring its cells fails here instead of at a call site in production.
    """
    assert (outcome, tool_class) in guard.POLICY_MATRIX
    assert isinstance(guard.POLICY_MATRIX[(outcome, tool_class)], guard.PrefixAction)


def test_matrix_declares_no_cell_beyond_the_product() -> None:
    expected = set(itertools.product(guard.RESOLUTION_OUTCOMES, NON_EXEMPT_CLASSES))
    assert set(guard.POLICY_MATRIX) == expected


def test_matrix_transcribes_the_prd_table() -> None:
    """PRD §4-C3's table, cell by cell."""
    m = guard.POLICY_MATRIX
    assert m[('unique', guard.ToolClass.DEFAULT)] is guard.PrefixAction.SUBSTITUTE_AND_FORWARD
    assert m[('unique', guard.ToolClass.FORWARD_ON_AMBIGUITY)] is guard.PrefixAction.SUBSTITUTE_AND_FORWARD
    assert m[('ambiguous', guard.ToolClass.DEFAULT)] is guard.PrefixAction.REJECT_WITH_CANDIDATES
    assert m[('ambiguous', guard.ToolClass.FORWARD_ON_AMBIGUITY)] is guard.PrefixAction.FORWARD_WITH_CANDIDATES
    assert m[('none', guard.ToolClass.DEFAULT)] is guard.PrefixAction.INERT
    assert m[('none', guard.ToolClass.FORWARD_ON_AMBIGUITY)] is guard.PrefixAction.INERT


def test_matrix_is_immutable() -> None:
    """The matrix IS the policy, so it must not be edited at runtime."""
    with pytest.raises(TypeError):
        guard.POLICY_MATRIX[('unique', guard.ToolClass.DEFAULT)] = guard.PrefixAction.INERT  # type: ignore[index]


def test_exempt_is_not_a_key_in_the_matrix() -> None:
    """An exemption is a declaration that this is not a repair site.

    It short-circuits ahead of resolution, so it has no outcome and therefore
    no row — a cell for it would imply the resolver had been consulted.
    """
    for outcome in guard.RESOLUTION_OUTCOMES:
        assert (outcome, guard.ToolClass.EXEMPT) not in guard.POLICY_MATRIX


# --- what is storm-counted, declared as data ----------------------------


def test_storm_counted_outcomes_are_exactly_the_two_fail_soft_ones() -> None:
    """INV-4, and the reason `expanded` is excluded is a measurement.

    `expanded` is the DESIGNED SUCCESS PATH. At ~10% of 124 writes/day it
    would fire the 3/3600 thresholds continuously and be ignored, which is how
    a storm escape stops being an escape.
    """
    assert frozenset(
        {'forwarded_ambiguous', 'resolver_unavailable'}
    ) == guard.STORM_COUNTED_OUTCOMES


def test_expanded_is_not_storm_counted() -> None:
    assert 'expanded' not in guard.STORM_COUNTED_OUTCOMES


def test_rejected_is_not_storm_counted() -> None:
    """A rejection is not fail-soft: the caller is told, so it needs no escape."""
    assert 'rejected' not in guard.STORM_COUNTED_OUTCOMES


def test_storm_counted_outcomes_are_all_real_fact_outcomes() -> None:
    assert set(guard.FACT_OUTCOMES) >= guard.STORM_COUNTED_OUTCOMES


# ---------------------------------------------------------------------------
# The harness.
# ---------------------------------------------------------------------------


class _Recorder:
    """Records the arguments each toy tool actually received.

    ``calls`` being EMPTY is the direct assertion that a rejected call wrote
    nothing. ``calls[0]`` carrying the expanded value is the direct assertion
    that a substitution LANDED rather than merely being reported on ``meta``.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, tool: str, **kwargs: Any) -> dict[str, Any]:
        entry = {'tool': tool, **kwargs}
        self.calls.append(entry)
        return entry

    @property
    def args(self) -> dict[str, Any]:
        """The single recorded call — asserts there was exactly one."""
        assert len(self.calls) == 1, f'expected exactly one call, got {self.calls!r}'
        return self.calls[0]


#: What a resolver returns for a prefix no live id starts with.
NO_MATCH = guard.Resolution('none', ())


class _FakeResolver:
    """The injected C2 port, answering from a DECLARED table.

    A table, not a mock. Every answer it can give is one a real resolver could
    give — ``Resolution`` is a NamedTuple with a closed outcome vocabulary — so
    a test can never accidentally assert against a shape C2 forbids. A bare
    ``MagicMock`` would answer anything at all, which is also why
    ``scripts/check_bare_magicmock_config.py`` gates ``shared/tests``.

    ``calls`` records ``(project, prefix)`` in order, which is what makes C1's
    "distinct tokens are resolved once each per call" directly assertable
    instead of inferred from the number of facts.
    """

    def __init__(self, answers: Mapping[str, guard.Resolution]) -> None:
        self.answers = dict(answers)
        self.calls: list[tuple[str | None, str]] = []

    async def __call__(self, project: str | None, prefix: str) -> guard.Resolution:
        self.calls.append((project, prefix))
        return self.answers.get(prefix, NO_MATCH)

    @property
    def prefixes(self) -> list[str]:
        return [prefix for _, prefix in self.calls]


#: The project every toy call in this file attributes itself to. A constant
#: rather than read off the arguments, because ``project_for`` is its own
#: contract with its own erratum and is pinned separately.
PROJECT = 'reify'


class Harness:
    """A toy FastMCP server plus the guard under test, driven by a Client."""

    def __init__(
        self,
        mcp: FastMCP,
        recorder: _Recorder,
        resolver: _FakeResolver,
        facts: list[Any],
        escalations: list[Any],
    ) -> None:
        self.mcp = mcp
        self.recorder = recorder
        self.resolver = resolver
        self.facts = facts
        self.escalations = escalations

    async def call(self, tool: str, arguments: dict[str, Any]):
        async with Client(self.mcp) as client:
            return await client.call_tool(tool, arguments)


def build_harness(
    *,
    answers: Mapping[str, guard.Resolution] | None = None,
    project_for: Callable[[Mapping[str, Any]], str | None] | None = None,
    **guard_kwargs: Any,
) -> Harness:
    """Build a server whose tools mirror the real victim signatures.

    ``update_task`` and ``submit_task`` declare ``project_root`` while
    ``add_memory`` declares ``project_id``, because that divergence is real (it
    is what erratum 7 turns on) and a harness that smoothed it over would make
    the ``project_for`` contract look easier than it is.
    """
    mcp = FastMCP('uuid-prefix-guard-harness')
    rec = _Recorder()
    facts: list[Any] = []
    escalations: list[Any] = []

    @mcp.tool
    def add_memory(
        content: str,
        category: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
        # dict OR JSON string, mirroring the real surface — which is what the
        # override helper's shape tolerance exists for.
        metadata: dict | str | None = None,
    ) -> str:
        rec.record(
            'add_memory',
            content=content,
            category=category,
            project_id=project_id,
            agent_id=agent_id,
            metadata=metadata,
        )
        return 'mem_1'

    @mcp.tool
    def update_task(
        task_id: str,
        project_root: str,
        description: str | None = None,
        details: str | None = None,
        metadata: dict | str | None = None,
    ) -> str:
        """D3's forward-on-ambiguity tool: losing the write is worse than the defect."""
        rec.record(
            'update_task',
            task_id=task_id,
            project_root=project_root,
            description=description,
            details=details,
            metadata=metadata,
        )
        return 'task_1'

    @mcp.tool
    def submit_task(
        title: str,
        description: str,
        project_root: str,
        priority: str = 'medium',
        metadata: dict | str | None = None,
    ) -> str:
        """Carries the B8 nested path: ``metadata.cluster_memory_ids[i]``."""
        rec.record(
            'submit_task',
            title=title,
            description=description,
            project_root=project_root,
            priority=priority,
            metadata=metadata,
        )
        return 'tkt_1'

    @mcp.tool
    def delete_memory(memory_id: str, store: str = 'mem0', project_id: str | None = None) -> str:
        """B9's exemption: an id-addressed destructive tool stays refuse-only."""
        rec.record('delete_memory', memory_id=memory_id, store=store, project_id=project_id)
        return 'deleted'

    resolver = _FakeResolver(answers or {})
    guard_kwargs.setdefault('fact_sink', facts.append)
    guard_kwargs.setdefault('escalation_sink', escalations.append)
    mcp.add_middleware(
        guard.UuidPrefixGuardMiddleware(
            resolver,
            project_for if project_for is not None else (lambda arguments: PROJECT),
            **guard_kwargs,
        )
    )
    return Harness(mcp, rec, resolver, facts, escalations)


def repair_of(result: Any) -> dict[str, Any] | None:
    """The result's ``uuid_prefix_repair`` block, or ``None`` when there is none.

    ONE accessor for both directions. ``meta`` is Optional on the wire type — a
    tool result legitimately carries none — so an absent block and an absent
    meta are the same answer here, and a test asserting inertness never has to
    tell them apart.
    """
    meta = result.meta or {}
    return meta.get('uuid_prefix_repair')


async def resolved_prefixes(arguments: dict[str, Any]) -> list[str]:
    """The prefixes the resolver was asked about, in order, for one add_memory call.

    Document order with duplicates collapsed is the observable form of "each
    distinct token, once" — so both halves of that invariant are one assertion.
    """
    h = build_harness()
    await h.call('add_memory', dict(arguments))
    return h.resolver.prefixes


# ---------------------------------------------------------------------------
# The clean path: no prefix-shaped token anywhere.
# ---------------------------------------------------------------------------

CLEAN_CALL = {
    'content': 'the reconciler ran twice and both passes agreed',
    'project_id': PROJECT,
    'agent_id': 'claude-interactive',
}


class TestCleanCallPassesThrough:
    async def test_the_arguments_reach_the_tool_verbatim(self) -> None:
        h = build_harness()
        await h.call('add_memory', dict(CLEAN_CALL))
        assert h.recorder.args == {
            'tool': 'add_memory',
            'content': CLEAN_CALL['content'],
            'category': None,
            'project_id': PROJECT,
            'agent_id': 'claude-interactive',
            'metadata': None,
        }

    async def test_no_repair_block_rides_on_meta(self) -> None:
        h = build_harness()
        result = await h.call('add_memory', dict(CLEAN_CALL))
        assert repair_of(result) is None

    async def test_no_fact_is_emitted(self) -> None:
        h = build_harness()
        await h.call('add_memory', dict(CLEAN_CALL))
        assert h.facts == []

    async def test_the_resolver_is_never_consulted(self) -> None:
        """The fast path costs one scan and NO awaited round trip (INV-8).

        This guard sits on every tool call on the server, and the overwhelming
        majority carry no prefix at all. A resolver round-trip on those would
        put a store read on the loop thread for every call in the factory.
        """
        h = build_harness()
        await h.call('add_memory', dict(CLEAN_CALL))
        assert h.resolver.calls == []


# ---------------------------------------------------------------------------
# B5 — a token that resolves to nothing is INERT.
# ---------------------------------------------------------------------------
#
# The strongest form of "inert": not merely unmodified, but UNRECORDED. 33,076
# of the corpus's hex-8 occurrences are this case, so a fact per miss would be
# 75% of the stream and would teach every reader to skip it (INV-2's reason for
# emitting only unique/ambiguous/unavailable).

NONE_CALL = {
    'content': (
        'the recon-3f2a9c1e pass over the 20260904 snapshot left a deadbeef '
        'marker, and deadbeef is not an id'
    ),
    'project_id': PROJECT,
}


class TestB5ZeroMatchIsInert:
    async def test_the_body_reaches_the_tool_byte_identical(self) -> None:
        h = build_harness()
        await h.call('add_memory', dict(NONE_CALL))
        assert h.recorder.args['content'] == NONE_CALL['content']

    async def test_no_repair_block_rides_on_meta(self) -> None:
        h = build_harness()
        result = await h.call('add_memory', dict(NONE_CALL))
        assert repair_of(result) is None

    async def test_no_fact_is_emitted_not_even_one_recording_the_miss(self) -> None:
        h = build_harness()
        await h.call('add_memory', dict(NONE_CALL))
        assert h.facts == []

    async def test_the_glue_rule_keeps_the_composite_handle_out_of_the_resolver(self) -> None:
        """``recon-3f2a9c1e`` is not a token, so it is never even looked up.

        Detection is ``shared.uuid_prefix``'s job and the guard does not
        second-guess it — but the consequence is visible HERE, as a store read
        that never happens.
        """
        assert '3f2a9c1e' not in await resolved_prefixes(NONE_CALL)

    async def test_each_distinct_token_is_resolved_exactly_once(self) -> None:
        """C1's stated invariant, asserted where it costs something.

        ``deadbeef`` occurs twice in one string. Resolving per OCCURRENCE
        rather than per distinct token would double the store reads on exactly
        the calls that cite one id repeatedly, and each of those reads is a
        full collection walk.
        """
        assert await resolved_prefixes(NONE_CALL) == ['20260904', 'deadbeef']
