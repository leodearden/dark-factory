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
import json
from collections.abc import Callable, Mapping
from typing import Any, get_args

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

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


# ---------------------------------------------------------------------------
# Resolution builders and a hand-advanced clock.
# ---------------------------------------------------------------------------


def unique(
    full_id: str, namespace: guard.Namespace = 'mem0', preview: str = ''
) -> guard.Resolution:
    """A `unique` resolution — exactly one candidate, as C2 requires."""
    return guard.Resolution('unique', (guard.Candidate(namespace, full_id, preview),))


class _Clock:
    """A hand-advanced clock, so a storm window is a decision and not a race."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


#: Full ids, each with its own prefix as a literal prefix — the monotonicity
#: `substitute` enforces, so a wrong expansion stays visible and reversible.
BFF = 'bff81530'
BFF_FULL = 'bff81530-1a2b-4c3d-8e9f-0123456789ab'
F1C = 'f1c4a651'
F1C_FULL = 'f1c4a651-2b3c-4d5e-8f90-1234567890ab'
B7B = 'b7b0f63b'
B7B_FULL = 'b7b0f63b-3c4d-4e5f-8a01-234567890abc'
BEC = '8bec9cd6'
BEC_FULL = '8bec9cd6-4d5e-4f60-8b12-34567890abcd'

#: A dead full uuid, which corpus convention treats as provenance. It sits
#: beside an expandable prefix in the B8 list so "the guard touched exactly one
#: entry" is asserted rather than assumed.
SIBLING_FULL = '48433882-ee71-480d-aff7-c91aa4640ff5'

AGENT = 'claude-interactive'


def detection_fact(**overrides: Any) -> dict[str, Any]:
    """One complete ``uuid_prefix_detected`` record, defaulted to the B1 expansion.

    Spelled as a COMPLETE record with overrides rather than as a subset check,
    so a key the guard silently stopped emitting fails a test instead of
    passing one — and one builder for every arm, so a key that drifted between
    arms could not pass either.
    """
    fact = {
        'fact': 'uuid_prefix_detected',
        'tool': 'add_memory',
        'field': 'content',
        'token': BFF,
        'outcome': 'expanded',
        'candidate_ids': [BFF_FULL],
        'namespace': 'mem0',
        'agent_id': AGENT,
        'project': PROJECT,
    }
    fact.update(overrides)
    return fact


# ---------------------------------------------------------------------------
# B1 — unique expansion on a memory write.
# ---------------------------------------------------------------------------


class TestB1UniqueExpansion:
    CALL = {
        'content': f'the {BFF} record already answers this',
        'project_id': PROJECT,
        'agent_id': AGENT,
    }

    def harness(self) -> Harness:
        return build_harness(answers={BFF: unique(BFF_FULL, preview='the reconciler ran twice')})

    async def test_the_tool_received_the_expanded_value(self) -> None:
        """Read off the RECORDER, not off meta.

        ``meta`` reports what the guard says it did; the recorder reports what
        the tool was actually handed. A guard that reported a substitution it
        never applied would pass every meta assertion in this file.
        """
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert h.recorder.args['content'] == f'the {BFF_FULL} record already answers this'

    async def test_the_other_arguments_are_untouched(self) -> None:
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert h.recorder.args['project_id'] == PROJECT
        assert h.recorder.args['agent_id'] == AGENT
        assert h.recorder.args['metadata'] is None

    async def test_the_substitution_is_reported_on_meta(self) -> None:
        h = self.harness()
        result = await h.call('add_memory', dict(self.CALL))
        assert repair_of(result) == {
            'substitutions': [
                {
                    'field': 'content',
                    'path': ['content'],
                    'from': BFF,
                    'to': BFF_FULL,
                    'namespace': 'mem0',
                }
            ]
        }

    async def test_exactly_one_fact_with_outcome_expanded(self) -> None:
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert h.facts == [detection_fact()]

    async def test_fastmcps_own_meta_is_preserved_not_replaced(self) -> None:
        """``call_next``'s result already carries FastMCP's signalling.

        Discarding it to deliver our own would be a poor trade, so the report
        is FOLDED into whatever meta came back.
        """
        h = self.harness()
        result = await h.call('add_memory', dict(self.CALL))
        assert (result.meta or {}).get('fastmcp') == {'wrap_result': True}


# ---------------------------------------------------------------------------
# B7 — a slash-separated pair: the glue rule keeps both, and both expand.
# ---------------------------------------------------------------------------


class TestB7SlashSeparatedPair:
    CALL = {
        'content': f'both {F1C}/{B7B} are cited',
        'project_id': PROJECT,
        'agent_id': AGENT,
    }

    def harness(self) -> Harness:
        return build_harness(answers={F1C: unique(F1C_FULL), B7B: unique(B7B_FULL)})

    async def test_both_tokens_reach_the_tool_expanded(self) -> None:
        """The reverse-document-order fold, observed rather than argued.

        Applying the earlier span first would shift every later offset by the
        length of the expansion, so the second replacement would land inside
        the first one's new text. Both landing span-exact is the direct
        evidence that did not happen.
        """
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert h.recorder.args['content'] == f'both {F1C_FULL}/{B7B_FULL} are cited'

    async def test_two_substitutions_are_reported_in_document_order(self) -> None:
        """Applied in reverse, REPORTED in reading order — the consumer's order."""
        h = self.harness()
        result = await h.call('add_memory', dict(self.CALL))
        repair = repair_of(result)
        assert repair is not None
        assert [(s['from'], s['to']) for s in repair['substitutions']] == [
            (F1C, F1C_FULL),
            (B7B, B7B_FULL),
        ]

    async def test_one_fact_per_token(self) -> None:
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert [(f['token'], f['candidate_ids']) for f in h.facts] == [
            (F1C, [F1C_FULL]),
            (B7B, [B7B_FULL]),
        ]


# ---------------------------------------------------------------------------
# B8 — the nested path, which is the shape the 4643 incident actually took.
# ---------------------------------------------------------------------------


class TestB8NestedPath:
    CALL = {
        'title': 'follow up on the cluster',
        'description': 'nothing citable here',
        'project_root': '/home/leo/src/reify',
        'metadata': {'cluster_memory_ids': [BEC, SIBLING_FULL]},
    }

    def harness(self) -> Harness:
        return build_harness(answers={BEC: unique(BEC_FULL)})

    async def test_the_tool_received_the_expanded_list_entry(self) -> None:
        h = self.harness()
        await h.call('submit_task', dict(self.CALL))
        assert h.recorder.args['metadata']['cluster_memory_ids'][0] == BEC_FULL

    async def test_the_sibling_full_uuid_is_untouched(self) -> None:
        """A full uuid is never a token, so nothing may happen to one."""
        h = self.harness()
        await h.call('submit_task', dict(self.CALL))
        assert h.recorder.args['metadata']['cluster_memory_ids'][1] == SIBLING_FULL

    async def test_the_substitution_carries_the_structured_path(self) -> None:
        """A list, not ``'metadata.cluster_memory_ids[0]'`` (heuristic 12).

        An encoded path would need an ad-hoc parser at every consumer and
        would be ambiguous the moment a key contains a dot or a bracket.
        """
        h = self.harness()
        result = await h.call('submit_task', dict(self.CALL))
        repair = repair_of(result)
        assert repair is not None
        assert repair['substitutions'] == [
            {
                'field': 'metadata',
                'path': ['metadata', 'cluster_memory_ids', 0],
                'from': BEC,
                'to': BEC_FULL,
                'namespace': 'mem0',
            }
        ]

    async def test_the_fact_names_the_top_level_argument(self) -> None:
        """``field`` stays the FLAT vocabulary the markup fact stream uses.

        Both streams stay queryable the same way; ``path`` is what carries the
        depth.
        """
        h = self.harness()
        await h.call('submit_task', dict(self.CALL))
        assert h.facts == [
            detection_fact(
                tool='submit_task',
                field='metadata',
                token=BEC,
                candidate_ids=[BEC_FULL],
                agent_id=None,
            )
        ]


# ---------------------------------------------------------------------------
# The `unique` row is the SAME cell for both tool classes.
# ---------------------------------------------------------------------------


class TestUniqueIsIdenticalOnTheForwardOnAmbiguityClass:
    """A tool's class changes what happens on AMBIGUITY, and nothing else.

    Driven rather than inferred from the matrix, because the matrix says the
    two cells are equal and this says the body honours that.
    """

    CALL = {
        'task_id': '4643',
        'project_root': '/home/leo/src/reify',
        'description': f'supersedes {BFF}',
    }

    def harness(self) -> Harness:
        return build_harness(
            answers={BFF: unique(BFF_FULL)},
            forward_on_ambiguity_tools=frozenset({'update_task'}),
        )

    async def test_the_tool_received_the_expanded_value(self) -> None:
        h = self.harness()
        await h.call('update_task', dict(self.CALL))
        assert h.recorder.args['description'] == f'supersedes {BFF_FULL}'

    async def test_the_fact_is_an_expansion(self) -> None:
        h = self.harness()
        await h.call('update_task', dict(self.CALL))
        assert [(f['tool'], f['outcome']) for f in h.facts] == [('update_task', 'expanded')]


# ---------------------------------------------------------------------------
# One id cited twice: resolved once, edited twice, reported twice.
# ---------------------------------------------------------------------------


class TestARepeatedCitationIsOneResolutionAndTwoEdits:
    """The two vocabularies are per-DIFFERENT-things, and that is deliberate.

    A resolution is per distinct token — the expensive part, a store walk. A
    substitution and its fact are per OCCURRENCE, because they describe an
    edit at a site, which is what an operator querying by ``field`` needs.
    """

    CALL = {
        'content': f'{BFF} is the record; see {BFF} again',
        'project_id': PROJECT,
        'agent_id': AGENT,
    }

    def harness(self) -> Harness:
        return build_harness(answers={BFF: unique(BFF_FULL)})

    async def test_the_store_is_walked_once(self) -> None:
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert h.resolver.prefixes == [BFF]

    async def test_both_occurrences_are_expanded(self) -> None:
        h = self.harness()
        await h.call('add_memory', dict(self.CALL))
        assert h.recorder.args['content'] == f'{BFF_FULL} is the record; see {BFF_FULL} again'

    async def test_two_substitutions_and_two_facts(self) -> None:
        h = self.harness()
        result = await h.call('add_memory', dict(self.CALL))
        repair = repair_of(result)
        assert repair is not None
        assert len(repair['substitutions']) == 2
        assert h.facts == [detection_fact(), detection_fact()]


# ---------------------------------------------------------------------------
# INV-4 — `expanded` is the designed success path and is NOT storm-counted.
# ---------------------------------------------------------------------------


class TestExpandedIsNeverStormCounted:
    """Ten expansions inside one window, through ONE middleware instance.

    At ~10% of 124 writes/day a counted success path would fire the 3/3600
    thresholds continuously and be ignored, which is how a storm escape stops
    being an escape. The clock is injected and never advanced, so all ten
    events sit inside one window by construction rather than by being fast.
    """

    async def _ten_calls(self) -> Harness:
        h = build_harness(answers={BFF: unique(BFF_FULL)}, time_provider=_Clock())
        for index in range(10):
            await h.call(
                'add_memory',
                {'content': f'{BFF} cited in call {index}', 'project_id': PROJECT},
            )
        return h

    async def test_all_ten_calls_landed(self) -> None:
        h = await self._ten_calls()
        assert len(h.recorder.calls) == 10

    async def test_no_storm_summary_ever_appears(self) -> None:
        h = build_harness(answers={BFF: unique(BFF_FULL)}, time_provider=_Clock())
        for index in range(10):
            result = await h.call(
                'add_memory',
                {'content': f'{BFF} cited in call {index}', 'project_id': PROJECT},
            )
            repair = repair_of(result)
            assert repair is not None
            assert 'storm' not in repair

    async def test_the_escalation_sink_received_nothing(self) -> None:
        h = await self._ten_calls()
        assert h.escalations == []


# ---------------------------------------------------------------------------
# B2 / B4 — ambiguity on a DEFAULT-class tool is a rejection.
# ---------------------------------------------------------------------------
#
# B4 is the cross-namespace case, so the two candidates here sit in DIFFERENT
# namespaces: one Mem0 id and one Graphiti node uuid sharing a prefix is
# `ambiguous`, and it is the shape a single-namespace resolver would have
# missed entirely.

AMB = '208f1bdf'
AMB_MEM0 = '208f1bdf-5e6f-4071-8c23-4567890abcde'
AMB_NODE = '208f1bdf-6f70-4182-8d34-567890abcdef'

AMBIGUOUS = guard.Resolution(
    'ambiguous',
    (
        guard.Candidate('mem0', AMB_MEM0, 'the reconciler ran twice and both passes'),
        guard.Candidate('graphiti_node', AMB_NODE, 'ReconciliationRun'),
    ),
)


def rejection_payload(excinfo: Any) -> dict[str, Any]:
    """The refusal's JSON payload, which must survive the boundary intact."""
    return json.loads(str(excinfo.value))


class TestB2AmbiguityIsRejectedOnADefaultTool:
    CALL = {
        'content': f'see {AMB} for the earlier pass',
        'category': 'observations_and_summaries',
        'project_id': PROJECT,
        'agent_id': AGENT,
    }

    def harness(self) -> Harness:
        return build_harness(answers={AMB: AMBIGUOUS})

    async def _reject(self, harness: Harness) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_memory', dict(self.CALL))
        return rejection_payload(excinfo)

    async def test_the_tool_never_ran(self) -> None:
        """What makes "nothing written" TRUE rather than merely intended."""
        h = self.harness()
        await self._reject(h)
        assert h.recorder.calls == []

    async def test_the_payload_carries_exactly_the_declared_keys(self) -> None:
        h = self.harness()
        payload = await self._reject(h)
        assert set(payload) == {
            'error',
            'error_type',
            'outcome',
            'tool',
            'field',
            'token',
            'candidates',
            'original_call',
            'hint',
        }

    async def test_the_payload_names_the_outcome_and_the_site(self) -> None:
        h = self.harness()
        payload = await self._reject(h)
        assert payload['error_type'] == 'ambiguous_uuid_prefix'
        assert payload['outcome'] == 'rejected'
        assert payload['tool'] == 'add_memory'
        assert payload['field'] == 'content'
        assert payload['token'] == AMB

    async def test_both_candidates_arrive_with_namespace_and_preview(self) -> None:
        """A caller told only "ambiguous" cannot choose; this is what lets it."""
        h = self.harness()
        payload = await self._reject(h)
        assert payload['candidates'] == [
            {
                'namespace': 'mem0',
                'id': AMB_MEM0,
                'preview': 'the reconciler ran twice and both passes',
            },
            {'namespace': 'graphiti_node', 'id': AMB_NODE, 'preview': 'ReconciliationRun'},
        ]

    async def test_original_call_is_the_complete_submitted_argument_map(self) -> None:
        """The 3936 gap, answered locally: a resubmit is mechanical.

        The COMPLETE map, not just the offending field — a caller reassembling
        the rest by hand is how the other arguments get lost for good.
        """
        h = self.harness()
        payload = await self._reject(h)
        assert payload['original_call'] == dict(self.CALL)

    async def test_there_is_no_repaired_call(self) -> None:
        """The choice is the author's, and the guard must not appear to have made it.

        This is the one place this guard's refusal deliberately DIVERGES from
        the markup guard's, which does carry a `repaired_call`: a mis-closed
        tag has one correct repair, and an ambiguous citation has two correct
        answers and no way to tell them apart.
        """
        h = self.harness()
        payload = await self._reject(h)
        assert 'repaired_call' not in payload

    async def test_the_hint_tells_the_caller_to_pick_one_and_resubmit(self) -> None:
        """The one action available to the caller, named.

        A hint that offered to choose would be a promise the guard cannot
        keep — "exactly one candidate, or nothing is changed" is the whole
        design.
        """
        h = self.harness()
        payload = await self._reject(h)
        assert 'resubmit' in payload['hint'].lower()

    async def test_exactly_one_fact_with_outcome_rejected(self) -> None:
        h = self.harness()
        await self._reject(h)
        assert h.facts == [
            detection_fact(
                token=AMB,
                outcome='rejected',
                candidate_ids=[AMB_MEM0, AMB_NODE],
                # Two namespaces, so there is no single one — present and null
                # rather than absent, so a consumer never has to tell "no
                # single namespace" from "that arm forgot the key".
                namespace=None,
            )
        ]


class TestAMixedCallIsRejectedWhole:
    """One unique token and one ambiguous token in the same call.

    The two-phase shape is what makes this structural: resolution completes
    before ANY substitution is applied, so the rejection leaves the argument
    map untouched rather than half-expanded. Interleaved, the unique token
    would already have been written into the arguments by the time the
    ambiguous one was resolved — and a rejected call would have mutated the
    caller's data on its way out.
    """

    CALL = {
        'content': f'{BFF} supersedes {AMB}',
        'project_id': PROJECT,
        'agent_id': AGENT,
    }

    def harness(self) -> Harness:
        return build_harness(answers={BFF: unique(BFF_FULL), AMB: AMBIGUOUS})

    async def _reject(self, harness: Harness) -> dict[str, Any]:
        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_memory', dict(self.CALL))
        return rejection_payload(excinfo)

    async def test_the_tool_never_ran(self) -> None:
        h = self.harness()
        await self._reject(h)
        assert h.recorder.calls == []

    async def test_no_partial_substitution_reaches_the_payload(self) -> None:
        """`original_call` is the map the CALLER sent, not a half-repaired one."""
        h = self.harness()
        payload = await self._reject(h)
        assert payload['original_call']['content'] == f'{BFF} supersedes {AMB}'

    async def test_the_refusal_names_the_ambiguous_token(self) -> None:
        h = self.harness()
        payload = await self._reject(h)
        assert payload['token'] == AMB


# ---------------------------------------------------------------------------
# B3 / D3 — ambiguity on a declared forward-on-ambiguity tool.
# ---------------------------------------------------------------------------
#
# Losing the write is worse than the defect: an `update_task` bounced for an
# ambiguous citation costs the whole update, and the citation is a detail of
# it. So the prefix travels UNCHANGED and the ambiguity is reported instead.


def project_of(arguments: Mapping[str, Any]) -> str | None:
    """A test-local ``project_for``: whichever identity argument the tool declares.

    Deliberately trivial, and deliberately not the shipped one — that has its
    own contract, its own erratum and its own step. All this has to do is make
    two projects distinguishable so the storm counter's keying is testable.
    """
    for key in ('project_id', 'project_root'):
        value = arguments.get(key)
        if isinstance(value, str):
            return value
    return None


def ambiguity_harness(**guard_kwargs: Any) -> Harness:
    return build_harness(
        answers={AMB: AMBIGUOUS},
        forward_on_ambiguity_tools=frozenset({'update_task'}),
        project_for=project_of,
        **guard_kwargs,
    )


def update_call(index: int = 0, project_root: str = PROJECT) -> dict[str, Any]:
    return {
        'task_id': str(4643 + index),
        'project_root': project_root,
        'description': f'supersedes {AMB}',
    }


class TestB3AmbiguityForwardsOnTheDeclaredClass:
    async def test_the_tool_received_the_prefix_unchanged(self) -> None:
        h = ambiguity_harness()
        await h.call('update_task', update_call())
        assert h.recorder.args['description'] == f'supersedes {AMB}'

    async def test_the_ambiguity_is_reported_on_meta(self) -> None:
        h = ambiguity_harness()
        result = await h.call('update_task', update_call())
        assert repair_of(result) == {
            'ambiguous': [
                {
                    'field': 'description',
                    'path': ['description'],
                    'token': AMB,
                    'candidates': [
                        {
                            'namespace': 'mem0',
                            'id': AMB_MEM0,
                            'preview': 'the reconciler ran twice and both passes',
                        },
                        {
                            'namespace': 'graphiti_node',
                            'id': AMB_NODE,
                            'preview': 'ReconciliationRun',
                        },
                    ],
                }
            ]
        }

    async def test_no_substitutions_key_claims_an_expansion_that_did_not_happen(self) -> None:
        """Omitted, not empty. A key that is always present teaches a reader to
        skip it, and this one distinguishes "changed nothing" from "changed
        something and did not say what"."""
        h = ambiguity_harness()
        result = await h.call('update_task', update_call())
        repair = repair_of(result)
        assert repair is not None
        assert 'substitutions' not in repair

    async def test_one_fact_with_outcome_forwarded_ambiguous(self) -> None:
        h = ambiguity_harness()
        await h.call('update_task', update_call())
        assert h.facts == [
            detection_fact(
                tool='update_task',
                field='description',
                token=AMB,
                outcome='forwarded_ambiguous',
                candidate_ids=[AMB_MEM0, AMB_NODE],
                namespace=None,
                agent_id=None,
            )
        ]

    async def test_a_single_forward_carries_no_storm_key(self) -> None:
        h = ambiguity_harness()
        result = await h.call('update_task', update_call())
        repair = repair_of(result)
        assert repair is not None
        assert 'storm' not in repair


# ---------------------------------------------------------------------------
# INV-4 — the storm escape on the fail-soft outcomes.
# ---------------------------------------------------------------------------
#
# A forwarded ambiguity is INVISIBLE to its caller: the call succeeded. So a
# burst of them is the one signal an operator has that the boundary is
# absorbing a defect at scale, and it must not be poolable into an alarm that
# names an outcome or a project that never burst.


async def drive_forwards(h: Harness, count: int, project_root: str = PROJECT) -> list[Any]:
    return [await h.call('update_task', update_call(i, project_root)) for i in range(count)]


class TestTheStormEscape:
    async def test_two_forwards_do_not_fire(self) -> None:
        h = ambiguity_harness(time_provider=_Clock())
        results = await drive_forwards(h, 2)
        assert all('storm' not in (repair_of(r) or {}) for r in results)
        assert h.escalations == []

    async def test_the_third_fires_and_the_summary_rides_on_meta(self) -> None:
        """The forwarding tier is the one whose callers cannot learn of the
        burst any other way — their calls all SUCCEEDED."""
        h = ambiguity_harness(time_provider=_Clock())
        results = await drive_forwards(h, 3)
        repair = repair_of(results[-1])
        assert repair is not None
        assert repair['storm'] == {
            'count': 3,
            'threshold': 3,
            'window_seconds': 3600.0,
            'outcome': 'forwarded_ambiguous',
            'project': PROJECT,
        }

    async def test_the_defaults_are_a_tripwire_not_a_rate_limit(self) -> None:
        """3 in 3600s, read off the fired summary rather than off an attribute.

        Three ambiguous citations inside one agent session is not a rate to be
        limited, it is a signal that a whole batch of writes is citing ids that
        do not identify anything.
        """
        h = ambiguity_harness(time_provider=_Clock())
        results = await drive_forwards(h, 3)
        repair = repair_of(results[-1]) or {}
        assert (repair['storm']['threshold'], repair['storm']['window_seconds']) == (3, 3600.0)

    async def test_the_escalation_sink_received_exactly_one_record(self) -> None:
        h = ambiguity_harness(time_provider=_Clock())
        await drive_forwards(h, 3)
        assert h.escalations == [
            {
                'error_type': 'uuid_prefix_boundary_storm',
                'count': 3,
                'threshold': 3,
                'window_seconds': 3600.0,
                'outcome': 'forwarded_ambiguous',
                'project': PROJECT,
            }
        ]

    async def test_every_call_still_landed(self) -> None:
        """A storm changes what an operator is told, never what the caller gets."""
        h = ambiguity_harness(time_provider=_Clock())
        await drive_forwards(h, 3)
        assert len(h.recorder.calls) == 3
        assert all(call['description'] == f'supersedes {AMB}' for call in h.recorder.calls)

    async def test_two_projects_do_not_pool_into_a_premature_fire(self) -> None:
        """Keyed per (project, outcome). One counter whose window spans every
        event regardless of label would fire on the fourth event here and name
        a project that saw only two."""
        h = ambiguity_harness(time_provider=_Clock())
        results = await drive_forwards(h, 2, 'alpha') + await drive_forwards(h, 2, 'beta')
        assert all('storm' not in (repair_of(r) or {}) for r in results)
        assert h.escalations == []

    async def test_a_rejection_never_advances_the_forwarded_counter(self) -> None:
        """`rejected` is not in STORM_COUNTED_OUTCOMES, and not merely because
        its threshold is high: it is never handed to a counter at all. A
        rejection is not fail-soft — the caller is told — so it needs no
        escape."""
        h = ambiguity_harness(time_provider=_Clock())
        for _ in range(2):
            with pytest.raises(ToolError):
                await h.call('add_memory', {'content': f'see {AMB}', 'project_id': PROJECT})
        results = await drive_forwards(h, 2)
        assert all('storm' not in (repair_of(r) or {}) for r in results)
        assert h.escalations == []
