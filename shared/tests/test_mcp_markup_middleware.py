"""Tests for ``shared.mcp_markup_middleware`` — the FastMCP boundary guard.

PRD ``plans/toolcall-markup-containment-prd.md`` task beta, contract C2. The
middleware is a pure boundary layer over task 3688's ``detect``/``repair``: it
contributes POLICY, structured FACTS and the storm escape, and no parsing of
its own.

## Why these tests drive a Client, and not the repo's two in-process idioms

Middleware sits at the SERVER REQUEST layer. Both established in-process
patterns in this repo bypass it entirely:

* fused-memory's ``await server._tool_manager.call_tool(name, args)``
* orchestrator/escalation's ``await server.get_tool(n)`` then ``tool.fn(...)``
  / ``await tool.run({...})``

A test written either way would pass while exercising nothing — the guard
would never run. Measured: ``async with Client(mcp)`` DOES run registered
middleware, so it is the only harness that can exercise this contract. There
is no prior use of the in-memory Client transport in this repo; this is a new
pattern, adopted because it is the only one that works.

## Why toy servers, and not the four real ones

``shared`` is the base layer — its tests may not import ``orchestrator``,
``escalation`` or ``fused_memory``. The toy tools below mirror the real victim
signatures (submit_task, escalate_info, add_memory) and RECORD the arguments
they actually received, which makes both "the tool never ran" and "the tool
received the repaired value" directly assertable rather than inferred.

## Substrate facts, measured in this worktree against fastmcp 3.2.2

* ``FastMCP.get_tool`` is a COROUTINE and must be awaited.
* ``tool.parameters`` is a full JSON Schema dict; the parameter NAMES are
  ``tool.parameters['properties'].keys()``.
* Mutating ``context.message.arguments`` in place REACHES the tool.
* Raising ``ToolError`` from ``on_call_tool`` prevents the tool body from
  running, and a ``json.dumps`` payload round-trips to the caller byte-intact.
* A middleware-authored ``ToolResult`` whose ``structured_content`` is reshaped
  FAILS the tool's output schema ("'result' is a required property"), because a
  tool with a return annotation carries one. So the reject path must raise and
  the forward path must pass ``structured_content`` through untouched.
* ``ToolResult.meta`` survives to the client, and ``call_next``'s result
  already carries ``{'fastmcp': {'wrap_result': True}}`` — which is why the
  forward path FOLDS meta rather than replacing it.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal in this file is spelled with the ``\\x3c`` escape for
``<``, exactly as ``shared/src/shared/toolcall_markup.py`` requires. Writing
``<`` verbatim here would force any agent editing this file to emit that
literal inside its own tool-call envelope, reproducing the very defect these
tests pin — its Write argument would terminate early, truncating this file and
silently dropping the sibling arguments of that same call. ``\\x3c`` is
byte-identical at runtime and never appears verbatim in the file text.
"""
from __future__ import annotations

import enum
import json
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import detect

# ---------------------------------------------------------------------------
# Envelope-literal builders (same spelling as tests/test_toolcall_markup.py).
# ---------------------------------------------------------------------------


def _closer(name: str) -> str:
    """The name-echoing closing tag the model drifts into."""
    return '\x3c/' + name + '>'


def _opener(name: str) -> str:
    """The name-echoing opening tag."""
    return '\x3c' + name + '>'


def _canonical_opener(name: str) -> str:
    """The canonical dialect's opening tag."""
    return '\x3cparameter name="' + name + '">'


INVOKE_CLOSER = '\x3c/invoke>'


# ---------------------------------------------------------------------------
# The harness.
# ---------------------------------------------------------------------------


class _Recorder:
    """Records the arguments each toy tool actually received.

    ``calls`` being EMPTY is the direct assertion that the tool body never ran,
    which is what "REJECT_WITH_REPAIR writes nothing" means. ``calls[0]`` being
    the repaired argument map is the direct assertion that a FORWARD_REPAIR
    recovery actually LANDED rather than merely being reported.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record(self, tool: str, **kwargs: Any) -> dict[str, Any]:
        entry = {'tool': tool, **kwargs}
        self.calls.append(entry)
        return entry

    @property
    def args(self) -> dict[str, Any]:
        """The single recorded call's arguments — asserts there was exactly one."""
        assert len(self.calls) == 1, f'expected exactly one call, got {self.calls!r}'
        return self.calls[0]


class Harness:
    """A toy FastMCP server plus the guard under test, driven by a Client."""

    def __init__(self, mcp: FastMCP, recorder: _Recorder, facts: list, escalations: list):
        self.mcp = mcp
        self.recorder = recorder
        self.facts = facts
        self.escalations = escalations

    async def call(self, tool: str, arguments: dict[str, Any]):
        async with Client(self.mcp) as client:
            return await client.call_tool(tool, arguments)


def build_harness(
    policy: RepairPolicy,
    *,
    exempt_tools: frozenset[str] = frozenset(),
    **guard_kwargs: Any,
) -> Harness:
    """Build a server whose tools mirror the real victim signatures."""
    mcp = FastMCP('markup-guard-harness')
    rec = _Recorder()
    facts: list[Any] = []
    escalations: list[Any] = []

    @mcp.tool
    def submit_task(
        title: str,
        description: str,
        priority: str = 'medium',
        agent_id: str | None = None,
        # dict OR JSON string, mirroring the real submit_task/update_task —
        # which is what the override helper's shape tolerance exists for.
        metadata: dict | str | None = None,
    ) -> str:
        rec.record(
            'submit_task',
            title=title,
            description=description,
            priority=priority,
            agent_id=agent_id,
            metadata=metadata,
        )
        return 'tkt_1'

    @mcp.tool
    def escalate_info(
        summary: str,
        detail: str = '',
        suggested_action: str = '',
        project_root: str | None = None,
    ) -> str:
        rec.record(
            'escalate_info',
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            project_root=project_root,
        )
        return 'esc_1'

    @mcp.tool
    def add_memory(
        content: str,
        category: str | None = None,
        project_id: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        rec.record(
            'add_memory',
            content=content,
            category=category,
            project_id=project_id,
            agent_id=agent_id,
        )
        return 'mem_1'

    @mcp.tool
    def scan_memory_content(needle: str) -> str:
        """The exemption case: its whole job is to be handed literal substrings."""
        rec.record('scan_memory_content', needle=needle)
        return 'scanned'

    guard_kwargs.setdefault('fact_sink', facts.append)
    guard_kwargs.setdefault('escalation_sink', escalations.append)
    mcp.add_middleware(
        MarkupGuardMiddleware(policy, exempt_tools=exempt_tools, **guard_kwargs)
    )
    return Harness(mcp, rec, facts, escalations)


BOTH_POLICIES = pytest.mark.parametrize(
    'policy',
    [RepairPolicy.REJECT_WITH_REPAIR, RepairPolicy.FORWARD_REPAIR],
    ids=lambda p: p.name,
)


# ---------------------------------------------------------------------------
# INV-1: the policy is DECLARED, not inferred.
# ---------------------------------------------------------------------------


class TestRepairPolicyIsDeclared:
    def test_is_an_enum(self):
        assert issubclass(RepairPolicy, enum.Enum)

    def test_has_exactly_the_two_declared_tiers(self):
        """Exactly two, and no third — a tier nobody declared cannot exist.

        INV-1: the policy is a registration-time declaration, so the set of
        things a registration site can declare is closed and machine-checkable.
        """
        assert {m.name for m in RepairPolicy} == {'REJECT_WITH_REPAIR', 'FORWARD_REPAIR'}


# ---------------------------------------------------------------------------
# The fast path: a clean call is untouched.
# ---------------------------------------------------------------------------


class TestCleanCallPassesThrough:
    @BOTH_POLICIES
    async def test_arguments_reach_the_tool_verbatim(self, policy):
        h = build_harness(policy)

        await h.call(
            'submit_task',
            {'title': 'Fix the thing', 'description': 'A wholly ordinary description.'},
        )

        assert h.recorder.args == {
            'tool': 'submit_task',
            'title': 'Fix the thing',
            'description': 'A wholly ordinary description.',
            'priority': 'medium',
            'agent_id': None,
            'metadata': None,
        }

    @BOTH_POLICIES
    async def test_the_result_is_unchanged(self, policy):
        h = build_harness(policy)

        result = await h.call('submit_task', {'title': 't', 'description': 'd'})

        assert result.data == 'tkt_1'

    @BOTH_POLICIES
    async def test_no_fact_is_emitted(self, policy):
        h = build_harness(policy)

        await h.call('submit_task', {'title': 't', 'description': 'd'})

        assert h.facts == [], 'a clean call must not enter the fact stream'
        assert h.escalations == []

    @BOTH_POLICIES
    async def test_a_closing_tag_that_is_not_an_envelope_literal_is_untouched(self, policy):
        """The guard keys on the envelope enumeration, not on angle brackets.

        Ordinary prose containing markup — an HTML snippet, a diff — must not
        be mistaken for a leaked envelope, or the guard becomes a tax on every
        caller that quotes code.
        """
        prose = 'Use \x3cdiv class="x">hello\x3c/div> in the template.'
        h = build_harness(policy)

        await h.call('submit_task', {'title': 't', 'description': prose})

        assert h.recorder.args['description'] == prose
        assert h.facts == []


# ---------------------------------------------------------------------------
# B7 — a declared exemption.
# ---------------------------------------------------------------------------


class TestB7ExemptTool:
    """A tool whose whole job is to be handed envelope literals.

    ``scan_memory_content`` exists to search the corpus for exactly these
    substrings; guarding it would make the retroactive-sweep tool unable to
    look for the thing it was built to find.
    """

    @BOTH_POLICIES
    async def test_an_exempt_tool_receives_envelope_literals_verbatim(self, policy):
        needle = _closer('content') + INVOKE_CLOSER
        h = build_harness(policy, exempt_tools=frozenset({'scan_memory_content'}))

        await h.call('scan_memory_content', {'needle': needle})

        assert h.recorder.args['needle'] == needle, (
            'an exempt tool must receive its argument byte-for-byte, including '
            'the envelope literals it exists to search for'
        )

    @BOTH_POLICIES
    async def test_an_exempt_tool_emits_no_fact(self, policy):
        h = build_harness(policy, exempt_tools=frozenset({'scan_memory_content'}))

        await h.call('scan_memory_content', {'needle': _closer('content')})

        assert h.facts == [], (
            'an exemption is a declaration that this is not a leak, so it must '
            'not pollute the fact stream with a non-event'
        )
        assert h.escalations == []

    @BOTH_POLICIES
    async def test_the_exemption_is_scoped_to_the_named_tool(self, policy):
        """Exempting one tool must not disarm the guard for its siblings."""
        h = build_harness(policy, exempt_tools=frozenset({'scan_memory_content'}))

        with pytest.raises(ToolError):
            await h.call(
                'add_memory',
                {'content': 'x' + _closer('content') + INVOKE_CLOSER + 'junk'},
            )

    @BOTH_POLICIES
    async def test_no_exemptions_by_default(self, policy):
        """The default is guarded. A registration site opts a tool OUT explicitly."""
        h = build_harness(policy)

        with pytest.raises(ToolError):
            await h.call(
                'scan_memory_content',
                {'needle': 'x' + _closer('content') + INVOKE_CLOSER + 'junk'},
            )


# ---------------------------------------------------------------------------
# B6 — the deliberate-quoting override.
# ---------------------------------------------------------------------------


class TestB6DeliberateQuotingOverride:
    """``metadata={'allow_mcp_markup': True}`` — an author quoting the markup.

    Live, not hypothetical: the decompose session that filed these very tasks
    had to set it to quote the literals in its own task text.
    """

    @BOTH_POLICIES
    async def test_the_call_proceeds_with_the_markup_intact(self, policy):
        quoted = 'The leak looks like ' + _closer('content') + INVOKE_CLOSER
        h = build_harness(policy)

        await h.call(
            'submit_task',
            {
                'title': 't',
                'description': quoted,
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert h.recorder.args['description'] == quoted, (
            'the whole point of the override is that the quoted markup survives'
        )

    @BOTH_POLICIES
    async def test_the_override_key_is_stripped_before_dispatch(self, policy):
        """The flag is a write-time control, never payload the tool should see."""
        h = build_harness(policy)

        await h.call(
            'submit_task',
            {
                'title': 't',
                'description': 'quoting ' + _closer('content'),
                'metadata': {'allow_mcp_markup': True, 'keep': 'this'},
            },
        )

        assert h.recorder.args['metadata'] == {'keep': 'this'}

    @BOTH_POLICIES
    async def test_no_fact_is_emitted(self, policy):
        h = build_harness(policy)

        await h.call(
            'submit_task',
            {
                'title': 't',
                'description': 'quoting ' + _closer('content'),
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert h.facts == [], (
            'a declared, deliberate quote is not a detection — recording it as '
            'one would make the fact stream measure author intent, not leaks'
        )
        assert h.escalations == []

    @BOTH_POLICIES
    async def test_the_override_is_fail_closed(self, policy):
        """Only a literal boolean True. A truthy value is not a declaration."""
        h = build_harness(policy)

        with pytest.raises(ToolError):
            await h.call(
                'submit_task',
                {
                    'title': 't',
                    'description': 'x' + _closer('content') + INVOKE_CLOSER + 'junk',
                    'metadata': {'allow_mcp_markup': 'yes'},
                },
            )

    @BOTH_POLICIES
    async def test_the_override_is_accepted_as_a_json_string(self, policy):
        """submit_task/update_task accept metadata as an object OR a JSON string."""
        quoted = 'quoting ' + _closer('content')
        h = build_harness(policy)

        await h.call(
            'submit_task',
            {
                'title': 't',
                'description': quoted,
                'metadata': json.dumps({'allow_mcp_markup': True, 'keep': 'this'}),
            },
        )

        assert h.recorder.args['description'] == quoted
        assert json.loads(h.recorder.args['metadata']) == {'keep': 'this'}
        assert h.facts == []


# ---------------------------------------------------------------------------
# B1, B2 — REJECT_WITH_REPAIR.
# ---------------------------------------------------------------------------


def _reject_payload(excinfo) -> dict[str, Any]:
    """The rejection payload, parsed off the ToolError.

    Measured: a ``json.dumps`` payload raised from ``on_call_tool``
    round-trips to the caller BYTE-INTACT, which is what lets ``repaired_call``
    survive the boundary at all.
    """
    return json.loads(str(excinfo.value))


class TestB1PartialDrift:
    """The description absorbed the next parameter's opener, unterminated.

    ``…\\x3c/description>`` + newline + ``\\x3cparameter name="priority">low`` —
    the harness mis-closed one parameter and the caller's ``priority`` was
    swallowed by the value of ``description``.
    """

    DESCRIPTION = (
        'Do the thing.'
        + _closer('description')
        + '\n'
        + _canonical_opener('priority')
        + 'low'
    )

    async def _reject(self):
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)
        with pytest.raises(ToolError) as excinfo:
            await h.call('submit_task', {'title': 'A task', 'description': self.DESCRIPTION})
        return h, _reject_payload(excinfo)

    async def test_the_tool_never_ran(self):
        h, _ = await self._reject()

        assert h.recorder.calls == [], (
            'REJECT_WITH_REPAIR must write NOTHING — the tool body never runs, '
            'which is why the middleware raises rather than short-circuiting '
            'with a middleware-authored ToolResult'
        )

    async def test_the_error_is_machine_readable(self):
        _, payload = await self._reject()

        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['tool'] == 'submit_task'
        assert payload['field'] == 'description'

    async def test_the_repaired_description_is_the_prefix_only(self):
        _, payload = await self._reject()

        assert payload['repaired_call']['description'] == 'Do the thing.'

    async def test_no_repaired_value_still_trips_the_detector(self):
        """The load-bearing one. A repaired_call carrying residue would be
        rejected all over again on retry — and would have silently dropped the
        arguments hiding in that residue."""
        _, payload = await self._reject()

        for name, value in payload['repaired_call'].items():
            if isinstance(value, str):
                assert detect(value) is None, f'{name} still carries envelope markup'

    async def test_the_swallowed_parameter_is_recovered(self):
        _, payload = await self._reject()

        assert payload['repaired_call']['priority'] == 'low'
        assert payload['recovered_params'] == ['priority']

    async def test_repaired_call_is_the_COMPLETE_argument_map(self):
        """So the retry is mechanical: resubmit repaired_call verbatim.

        A payload carrying only the repaired field would make the caller
        reassemble the rest by hand — and an agent reassembling by hand is
        exactly how the swallowed arguments get lost for good.
        """
        _, payload = await self._reject()

        assert payload['repaired_call'] == {
            'title': 'A task',
            'description': 'Do the thing.',
            'priority': 'low',
        }

    async def test_the_payload_names_the_outcome_and_how_to_act(self):
        """The rejected caller must be able to act without reading the PRD."""
        _, payload = await self._reject()

        assert payload['outcome'] == 'rejected'
        assert payload['hint'], 'a rejection with no remediation is a dead end'

    async def test_the_diagnostic_names_the_pattern_and_the_misclose(self):
        _, payload = await self._reject()

        assert payload['matched_pattern'] == _closer('description')
        assert payload['misclose'] == _closer('description')


class TestB2TotalDrift:
    """PRD section 2.1's first specimen shape — the parser fell back to
    ``\\x3c/invoke>`` and the description absorbed THREE parameters, in the
    blended dialect (a name-echoing opener, plus a stray ``"`` on the metadata
    tags) that task 3688's tolerance exists to support.
    """

    DESCRIPTION = (
        'Fix it.'
        + _closer('description')
        + '\n' + _opener('priority') + 'medium' + _closer('priority')
        + '\n' + _opener('agent_id') + 'claude-x' + _closer('agent_id')
        + '\n' + '\x3cmetadata">' + '{"source": "probe"}' + '\x3c/metadata">'
        + '\n' + INVOKE_CLOSER
    )

    async def _reject(self):
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)
        with pytest.raises(ToolError) as excinfo:
            await h.call('submit_task', {'title': 'A task', 'description': self.DESCRIPTION})
        return h, _reject_payload(excinfo)

    async def test_all_three_swallowed_parameters_are_recovered(self):
        _, payload = await self._reject()

        assert payload['repaired_call'] == {
            'title': 'A task',
            'description': 'Fix it.',
            'priority': 'medium',
            'agent_id': 'claude-x',
            'metadata': '{"source": "probe"}',
        }

    async def test_recovered_params_names_all_three(self):
        _, payload = await self._reject()

        assert sorted(payload['recovered_params']) == ['agent_id', 'metadata', 'priority']

    async def test_the_tool_never_ran(self):
        h, _ = await self._reject()

        assert h.recorder.calls == []

    async def test_the_invoke_closer_is_not_left_in_any_value(self):
        _, payload = await self._reject()

        for name, value in payload['repaired_call'].items():
            if isinstance(value, str):
                assert detect(value) is None, f'{name} still carries envelope markup'
