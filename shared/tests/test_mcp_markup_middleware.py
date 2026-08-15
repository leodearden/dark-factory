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

import contextlib
import enum
import json
from pathlib import Path
from typing import Any

import pytest
import toolcall_markup_corpus_extract as extract
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import MARKUP_OVERRIDE_KEY, detect

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
# Real specimens, drawn from the committed corpus.
# ---------------------------------------------------------------------------
#
# The unrepairable rows are driven by ACTUAL leaked payloads rather than
# hand-authored ones. A refusal is the one outcome an invented specimen can
# trivially fake — anything sufficiently mangled refuses — so the assertion
# that this guard never guesses is only worth something against input the
# harness really produced.

CORPUS_PATH = Path(__file__).resolve().parent / 'fixtures' / 'toolcall_markup_corpus.jsonl'


def specimen(tool_use_id: str) -> dict[str, Any]:
    """The committed corpus record with this ``tool_use_id``.

    Keyed by id rather than by index or by "the shortest one", so a corpus
    refresh that adds records cannot silently repoint a test at a different
    payload.
    """
    for record in extract.load_corpus(CORPUS_PATH):
        if record.get('tool_use_id') == tool_use_id:
            return record
    raise AssertionError(f'specimen {tool_use_id!r} is missing from {CORPUS_PATH}')


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


#: What the fake escalation sink hands back. The real emitter returns an
#: escalation id (see ``markup_tripwire.emit_markup_storm_escalation``), and the
#: refusal payload has to be able to name it — a caller told only "unrepairable"
#: cannot point an operator at the record holding its data.
ESCALATION_ID = 'esc-markup-residue-1'


class _EscalationSink:
    """Records residue escalations and returns an id, like the real emitter.

    A bare ``list.append`` would return ``None``, which would let a middleware
    that never propagated the sink's return value pass the payload assertions
    by accident.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __call__(self, record: dict[str, Any]) -> str:
        self.records.append(record)
        return ESCALATION_ID


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
        # The real add_memory takes BOTH, which is what makes the identity
        # precedence a real question rather than a hypothetical one.
        project_root: str | None = None,
        agent_id: str | None = None,
    ) -> str:
        rec.record(
            'add_memory',
            content=content,
            category=category,
            project_id=project_id,
            project_root=project_root,
            agent_id=agent_id,
        )
        return 'mem_1'

    @mcp.tool
    def add_memory_strict(
        content: str,
        project_id: str,
        category: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """``add_memory`` with ``project_id`` REQUIRED, as the real one declares it.

        PRD boundary row B14. The real tool
        (``fused-memory/src/fused_memory/server/tools.py:2845``) declares
        ``content: str, project_id: str`` — BOTH required — while the legacy toy
        above declares ``project_id: str | None = None``. That divergence is
        precisely why the required-parameter shape had never been exercised:
        every recovery target asserted anywhere in this file (B1 ``priority``,
        B2 ``priority``/``agent_id``/``metadata``, B3 ``suggested_action``, B9
        ``agent_id``) is a DEFAULTED parameter.

        This is ADDITIVE rather than a correction to the legacy toy: roughly
        fourteen existing call sites (B4, the INV-2 rows, the storm rows) call
        ``add_memory`` with only ``{'content': ...}``, so tightening
        ``project_id`` there would break every one of them for no gain.

        The optional tail is kept so this stays an honest mirror rather than a
        two-parameter reduction that would make any recovery trivially
        unambiguous. ``metadata`` and ``dual_write`` are deliberately left off —
        ``metadata`` would collide with the B6 override path, which this row is
        not about — and ``ctx`` is FastMCP-injected.
        """
        rec.record(
            'add_memory_strict',
            content=content,
            project_id=project_id,
            category=category,
            agent_id=agent_id,
            session_id=session_id,
        )
        return 'mem_strict_1'

    @mcp.tool
    def add_reuse_item(what: str, how: str, where: str) -> str:
        """PRD boundary row B5's named specimen is an ``add_reuse_item.how``."""
        rec.record('add_reuse_item', what=what, how=how, where=where)
        return 'reuse_1'

    @mcp.tool
    def scan_memory_content(needle: str) -> str:
        """The exemption case: its whole job is to be handed literal substrings."""
        rec.record('scan_memory_content', needle=needle)
        return 'scanned'

    guard_kwargs.setdefault('fact_sink', facts.append)
    guard_kwargs.setdefault('escalation_sink', _EscalationSink(escalations))
    mcp.add_middleware(
        MarkupGuardMiddleware(policy, exempt_tools=exempt_tools, **guard_kwargs)
    )
    return Harness(mcp, rec, facts, escalations)


def meta_of(result: Any) -> dict[str, Any]:
    """The forwarded result's ``meta``, asserting there is one at all.

    ``meta`` is Optional on the wire type — a tool result legitimately carries
    none — so every read below goes through here rather than subscripting it
    directly. That buys two things at once: the type checker stops reporting
    ``reportOptionalSubscript`` on five call sites, and a middleware that
    dropped meta wholesale fails with the sentence below instead of an
    unattributable ``TypeError: 'NoneType' object is not subscriptable``.
    """
    meta = result.meta
    assert meta is not None, 'the forwarded result carried no meta at all'
    return meta


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
    async def test_a_metadata_taking_tool_receives_the_flag_INTACT(self, policy):
        """ONE override, ONE lifecycle — the second guard has to be able to see it.

        fused-memory's write-time tripwire keys the SAME hatch off the metadata
        that reaches the tool body (``tools.py``'s ``_markup_gate``: ``if
        markup_override_requested(metadata): return None``), and strips it there
        (2703/2933/6565/7118). Consuming the flag at this boundary would let a
        caller quoting envelope literals in ``submit_task.description`` past
        THIS guard and straight into a rejection from the NEXT one — bounced
        with a hint telling it to set a flag it had already set. So on a tool
        that declares ``metadata``, the map travels UNCHANGED and the tool body
        owns the rest of the lifecycle.
        """
        h = build_harness(policy)

        await h.call(
            'submit_task',
            {
                'title': 't',
                'description': 'quoting ' + _closer('content'),
                'metadata': {'allow_mcp_markup': True, 'keep': 'this'},
            },
        )

        assert h.recorder.args['metadata'] == {'allow_mcp_markup': True, 'keep': 'this'}

    @BOTH_POLICIES
    async def test_a_tool_with_no_metadata_parameter_has_the_key_dropped(self, policy):
        """The other half of the same decision — nothing to hand the flag to.

        ``escalate_info`` declares no ``metadata``, so there is no second guard
        to inform and forwarding the parameter at all would be
        ``ToolError: Unexpected keyword argument``.
        """
        h = build_harness(policy)

        await h.call(
            'escalate_info',
            {
                'summary': 's',
                'detail': 'quoting ' + _closer('content'),
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert 'metadata' not in h.recorder.args

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
        # submit_task declares ``metadata``, so the map travels unchanged — the
        # JSON-string shape included, since that is the shape the caller chose
        # and the tool body's own ``markup_override_requested`` parses it.
        assert json.loads(h.recorder.args['metadata']) == {
            'allow_mcp_markup': True, 'keep': 'this',
        }
        assert h.facts == []

    # -- the hatch has to work on the tools that have no metadata ----------

    @BOTH_POLICIES
    async def test_the_override_works_on_a_tool_that_takes_no_metadata(self, policy):
        """MEASURED FAILURE MODE: most guarded tools declare no ``metadata``.

        ``escalate_info``, ``escalate_blocker``, ``add_reuse_item`` and most of
        the escalation/orchestrator surface take none. Stripping the flag but
        leaving ``metadata={}`` in the argument map forwards a parameter the
        tool does not have, and fastmcp bounces it with ``ToolError: Unexpected
        keyword argument`` — so an agent quoting envelope markup in an
        escalation summary (a very likely topic, given DF 3083) would do exactly
        what the hint told it to and land in a second, more confusing error.
        The key is therefore DROPPED when stripping empties it.
        """
        quoted = 'The leak looks like ' + _closer('detail') + INVOKE_CLOSER
        h = build_harness(policy)

        await h.call(
            'escalate_info',
            {
                'summary': 's',
                'detail': quoted,
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert h.recorder.args['detail'] == quoted
        assert h.facts == []

    @BOTH_POLICIES
    async def test_the_json_string_form_also_drops_to_nothing(self, policy):
        """``json.dumps({})`` is ``'{}'``, which fails the same way ``{}`` does."""
        quoted = 'quoting ' + _closer('detail')
        h = build_harness(policy)

        await h.call(
            'escalate_info',
            {
                'summary': 's',
                'detail': quoted,
                'metadata': json.dumps({'allow_mcp_markup': True}),
            },
        )

        assert h.recorder.args['detail'] == quoted

    @BOTH_POLICIES
    async def test_metadata_the_caller_also_sent_is_NOT_dropped(self, policy):
        """Only the emptied case is dropped — the rest is the caller's payload.

        On a tool with no ``metadata`` parameter, a residue the caller sent
        alongside the flag is the caller's own bug; leaving it in place surfaces
        that as an unknown-parameter error, where dropping it would silently
        discard data.
        """
        h = build_harness(policy)

        with pytest.raises(ToolError):
            await h.call(
                'escalate_info',
                {
                    'summary': 's',
                    'detail': 'quoting ' + _closer('content'),
                    'metadata': {'allow_mcp_markup': True, 'keep': 'this'},
                },
            )

        assert h.recorder.calls == []


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
        assert MARKUP_OVERRIDE_KEY in payload['hint'], (
            'a rejection with no remediation is a dead end'
        )

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


# ---------------------------------------------------------------------------
# B3, B4 — FORWARD_REPAIR.
# ---------------------------------------------------------------------------


class TestB3StrandRiskTierForwards:
    """The tier where bouncing costs more than proceeding.

    An escalation that reports a strand risk is exactly the call you cannot
    afford to lose to a rejection the caller may never retry — so this tier
    repairs in place and lets it through, warning that the arguments were
    altered. INV-6/INV-7: the recovered parameter must actually LAND, not
    merely be reported.
    """

    DETAIL = (
        'Strand risk.'
        + _closer('detail')
        + '\n' + _opener('suggested_action') + 'do the thing' + _closer('suggested_action')
        + INVOKE_CLOSER
    )

    async def _forward(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)
        result = await h.call(
            'escalate_info', {'summary': 'It stranded', 'detail': self.DETAIL}
        )
        return h, result

    async def test_the_tool_ran_exactly_once(self):
        h, _ = await self._forward()

        assert len(h.recorder.calls) == 1

    async def test_the_tool_received_the_repaired_detail(self):
        h, _ = await self._forward()

        assert h.recorder.args['detail'] == 'Strand risk.'
        assert detect(h.recorder.args['detail']) is None

    async def test_the_recovered_parameter_actually_lands(self):
        """Not merely reported — the whole point of this tier.

        A forward that dropped the recovered parameter would deliver a call
        the caller never made, silently missing the argument it did supply.
        """
        h, _ = await self._forward()

        assert h.recorder.args['suggested_action'] == 'do the thing'

    async def test_the_untouched_arguments_are_untouched(self):
        h, _ = await self._forward()

        assert h.recorder.args['summary'] == 'It stranded'

    async def test_structured_content_is_byte_identical_to_the_unguarded_tool(self):
        """The middleware must NOT reshape it.

        A tool with a return annotation carries an output schema, and a
        middleware-authored structured_content that does not satisfy it fails
        validation outright — so the forward path passes it through untouched
        and carries its warning somewhere else entirely.
        """
        _, guarded = await self._forward()
        bare = build_harness(RepairPolicy.FORWARD_REPAIR)
        unguarded = await bare.call(
            'escalate_info', {'summary': 'It stranded', 'detail': 'clean'}
        )

        assert guarded.structured_content == unguarded.structured_content
        assert guarded.data == 'esc_1'

    async def test_meta_carries_the_repair_warning(self):
        """Attached via meta, verified to survive to the client untouched.

        Never by appending a content block, which would corrupt any caller
        that indexes content[0].
        """
        _, result = await self._forward()

        warning = meta_of(result)['markup_repair']
        assert warning['field'] == 'detail'
        assert warning['recovered_params'] == ['suggested_action']
        assert warning['outcome'] == 'repaired'

    async def test_the_warning_names_the_pattern_and_the_misclose(self):
        _, result = await self._forward()

        warning = meta_of(result)['markup_repair']
        assert warning['matched_pattern'] == INVOKE_CLOSER
        assert warning['misclose'] == _closer('detail')

    async def test_fastmcps_own_meta_is_preserved_not_replaced(self):
        """call_next's result already carries {'fastmcp': {'wrap_result': True}}.

        Replacing meta wholesale would discard FastMCP's own signalling, so
        the warning is FOLDED in.
        """
        _, result = await self._forward()

        assert meta_of(result).get('fastmcp') == {'wrap_result': True}


class TestB4LastParameterNothingDropped:
    """The absorbing parameter was the LAST one, so there is nothing to recover.

    An empty recovery is a SUCCESS — the value is cleanly separable and no
    caller argument was lost — not a refusal. Treating it as one would bounce
    the most benign specimen in the corpus.
    """

    CONTENT = 'A memory.' + _closer('content') + INVOKE_CLOSER

    async def test_it_forwards_with_the_content_trimmed(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call('add_memory', {'content': self.CONTENT})

        assert h.recorder.args['content'] == 'A memory.'
        assert detect(h.recorder.args['content']) is None

    async def test_the_empty_recovery_is_reported_as_empty_not_omitted(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        result = await h.call('add_memory', {'content': self.CONTENT})

        warning = meta_of(result)['markup_repair']
        assert warning['recovered_params'] == []
        assert warning['outcome'] == 'repaired'

    async def test_the_warning_still_appears(self):
        """The caller's argument WAS altered, so it must still be told."""
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        result = await h.call('add_memory', {'content': self.CONTENT})

        assert 'markup_repair' in meta_of(result)

    async def test_the_same_input_is_REJECTED_under_the_other_tier(self):
        """The TIER decides, not the shape of the repair.

        Same specimen, same repair, opposite outcome — which is what makes the
        policy a declaration rather than something inferred from the damage.
        """
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError) as excinfo:
            await h.call('add_memory', {'content': self.CONTENT})

        payload = _reject_payload(excinfo)
        assert payload['outcome'] == 'rejected'
        assert payload['repaired_call'] == {'content': 'A memory.'}
        assert h.recorder.calls == []


# ---------------------------------------------------------------------------
# B5, B8, B9 — UNREPAIRABLE. One refusal, shared by both tiers.
# ---------------------------------------------------------------------------


class TestB5UnrepairableIsNeverGuessed:
    """PRD boundary row B5, on its own named specimen.

    ``repair()`` returning ``None`` means the value's boundary is a GUESS. The
    two tiers disagree about what to do with a repair; they do not disagree
    about this. FORWARD_REPAIR forwards a REPAIR, not a guess — a tier that
    forwarded unparsed residue would deliver a call the caller never made and
    permanently drop whatever arguments hid inside it, which is the exact
    silent fail-soft this guard exists to end.

    The residue escalation is what makes the refusal non-destructive: the
    payload the caller sent survives in a record an operator can act on even if
    the caller never retries (INV-7).
    """

    #: A real ``mcp__plan-tools__add_reuse_item.how`` from the corpus, named by
    #: PRD section 12's B5 row. Its tail carries an ``\\x3c/invoke>`` in the
    #: MIDDLE, so no candidate parses with zero leftover.
    SPECIMEN_ID = 'toolu_01A4bMDaYzQZnLYqacRnYJbV'

    @staticmethod
    def _value() -> str:
        record = specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)
        assert record['expected_outcome'] == 'unrepairable', (
            'this row is only meaningful while the corpus still classifies the '
            'specimen as unrepairable'
        )
        return record['value']

    async def _refuse(self, policy):
        h = build_harness(policy)
        value = self._value()
        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'add_reuse_item',
                {'what': 'the sampler', 'how': value, 'where': 'shared/'},
            )
        return h, value, excinfo

    @BOTH_POLICIES
    async def test_both_tiers_refuse(self, policy):
        await self._refuse(policy)

    @BOTH_POLICIES
    async def test_nothing_is_written_under_either_tier(self, policy):
        """Including under FORWARD_REPAIR, whose whole tier otherwise forwards."""
        h, _, _ = await self._refuse(policy)

        assert h.recorder.calls == [], (
            'an unrepairable value must not reach the tool under ANY tier — a '
            'partial write is worse than a refusal, because it looks like a '
            'successful call'
        )

    @BOTH_POLICIES
    async def test_exactly_one_escalation_is_filed(self, policy):
        h, _, _ = await self._refuse(policy)

        assert len(h.escalations) == 1

    @BOTH_POLICIES
    async def test_the_escalation_carries_the_FULL_raw_payload_verbatim(self, policy):
        """Untruncated, and byte-for-byte.

        ``build_markup_block``'s ``content_excerpt`` is capped at 200 chars
        because it is a DIAGNOSTIC sitting beside a payload the caller still
        holds. This record is the only surviving copy of data the caller may
        never resend, so truncating it would discard the thing the escalation
        exists to preserve.
        """
        h, value, _ = await self._refuse(policy)

        assert h.escalations[0]['raw_value'] == value
        assert len(h.escalations[0]['raw_value']) == len(value)

    @BOTH_POLICIES
    async def test_the_escalation_names_its_owner_and_its_bound(self, policy):
        """INV-7: a hold names a machine-readable owner and carries a bound.

        This one is a queue-backed handoff, so the bound is the standing L2 age
        surfacing rather than a deadline of its own.
        """
        h, _, _ = await self._refuse(policy)

        record = h.escalations[0]
        assert record['owner'], 'a hold with no named owner is an unbounded hold'
        assert record['level'] == 2
        assert record['category']

    @BOTH_POLICIES
    async def test_the_escalation_locates_the_leak(self, policy):
        h, _, _ = await self._refuse(policy)

        record = h.escalations[0]
        assert record['tool'] == 'add_reuse_item'
        assert record['field'] == 'how'
        assert record['matched_pattern'] == INVOKE_CLOSER

    @BOTH_POLICIES
    async def test_the_refusal_names_the_escalation_and_offers_no_repair(self, policy):
        """The caller must be able to point at the record holding its data.

        And must NOT be handed a ``repaired_call``: there is no repair. A
        payload carrying one would invite a retry that re-sends a guess.
        """
        _, _, excinfo = await self._refuse(policy)

        payload = _reject_payload(excinfo)
        assert payload['outcome'] == 'unrepairable'
        assert payload['escalation_id'] == ESCALATION_ID
        assert 'repaired_call' not in payload
        assert MARKUP_OVERRIDE_KEY in payload['hint']

    @BOTH_POLICIES
    async def test_a_sink_that_raises_does_not_change_the_outcome(self, policy):
        """The refusal is already decided; escalation is purely additive.

        Same reasoning as ``emit_markup_storm_escalation``'s never-raises
        contract — a queue I/O failure must cost the operator a heads-up, not
        turn a clean refusal into an opaque middleware crash.
        """
        def boom(record):
            raise RuntimeError('queue unavailable')

        h = build_harness(policy, escalation_sink=boom)

        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'add_reuse_item',
                {'what': 'w', 'how': self._value(), 'where': 'shared/'},
            )

        assert _reject_payload(excinfo)['outcome'] == 'unrepairable'
        assert h.recorder.calls == []


class TestB8RecoveredNameOutsideTheToolSchema:
    """A tail that parses CLEANLY but names a parameter the tool does not have.

    The recovery is well-formed markup and wrong anyway: accepting it would
    invent an argument for a tool that cannot take one, so the honest answer is
    that the value's boundary is unknown. The check runs against the tool's
    LIVE schema, which is why the guard reads it per call.
    """

    #: A real ``mcp__fused-memory__add_memory.content`` whose tail recovers
    #: ``session_id`` — not a parameter of add_memory, here or in production.
    SPECIMEN_ID = 'toolu_01MoQr2NFzcWNf9mhYfFcaQp'

    @staticmethod
    def _record() -> dict[str, Any]:
        record = specimen(TestB8RecoveredNameOutsideTheToolSchema.SPECIMEN_ID)
        assert 'session_id' not in record['schema_params'], (
            'the whole row rests on the recovered name being outside the schema'
        )
        return record

    async def _refuse(self, policy):
        h = build_harness(policy)
        record = self._record()
        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'add_memory',
                {
                    'content': record['value'],
                    'category': 'observations_and_summaries',
                    'project_id': 'dark_factory',
                    'agent_id': 'claude-caller',
                },
            )
        return h, record, excinfo

    @BOTH_POLICIES
    async def test_it_is_unrepairable_under_both_tiers(self, policy):
        _, _, excinfo = await self._refuse(policy)

        assert _reject_payload(excinfo)['outcome'] == 'unrepairable'

    @BOTH_POLICIES
    async def test_it_is_never_forwarded(self, policy):
        h, _, _ = await self._refuse(policy)

        assert h.recorder.calls == []

    @BOTH_POLICIES
    async def test_the_out_of_schema_name_is_never_invented_as_an_argument(self, policy):
        """Not in the escalation, and — since the tool never ran — nowhere else."""
        h, _, _ = await self._refuse(policy)

        assert 'session_id' not in h.escalations[0]

    @BOTH_POLICIES
    async def test_the_payload_survives_in_the_escalation(self, policy):
        h, record, _ = await self._refuse(policy)

        assert h.escalations[0]['raw_value'] == record['value']


class TestB9RecoveredNameCollidesWithASuppliedArgument:
    """A tail recovering ``agent_id`` when the caller already supplied one.

    Two values claim one parameter and nothing in the payload says which is
    authoritative. Overwriting the caller's own argument with a value scraped
    out of a mangled tail is precisely the fabrication this guard forbids, so
    the collision is unrepairable.
    """

    DESCRIPTION = (
        'Do it.'
        + _closer('description')
        + '\n' + _opener('agent_id') + 'claude-from-the-tail' + _closer('agent_id')
        + INVOKE_CLOSER
    )

    async def _refuse(self, policy):
        h = build_harness(policy)
        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'submit_task',
                {
                    'title': 'A task',
                    'description': self.DESCRIPTION,
                    'agent_id': 'claude-the-caller',
                },
            )
        return h, excinfo

    @BOTH_POLICIES
    async def test_the_collision_is_unrepairable(self, policy):
        _, excinfo = await self._refuse(policy)

        assert _reject_payload(excinfo)['outcome'] == 'unrepairable'

    @BOTH_POLICIES
    async def test_the_tool_never_ran(self, policy):
        h, _ = await self._refuse(policy)

        assert h.recorder.calls == []

    @BOTH_POLICIES
    async def test_the_callers_own_agent_id_is_never_overwritten(self, policy):
        """The record must attribute the leak to the caller that made it.

        Attributing it to the tail's value would point an operator at an agent
        that never made the call.
        """
        h, _ = await self._refuse(policy)

        assert h.escalations[0]['agent_id'] == 'claude-the-caller'

    @BOTH_POLICIES
    async def test_the_callers_value_survives_intact_in_the_raw_payload(self, policy):
        h, _ = await self._refuse(policy)

        assert h.escalations[0]['raw_value'] == self.DESCRIPTION

    @BOTH_POLICIES
    async def test_the_SAME_tail_repairs_when_the_caller_supplied_nothing(self, policy):
        """The control. Only the collision makes this unrepairable.

        Same value, same schema, same tier — drop the caller's own ``agent_id``
        and the identical tail becomes a clean recovery. That is what proves
        the refusal is the collision rule firing, and not the guard simply
        failing to parse this shape.
        """
        h = build_harness(policy)

        if policy is RepairPolicy.REJECT_WITH_REPAIR:
            with pytest.raises(ToolError) as excinfo:
                await h.call(
                    'submit_task', {'title': 'A task', 'description': self.DESCRIPTION}
                )
            payload = _reject_payload(excinfo)
            assert payload['outcome'] == 'rejected'
            assert payload['repaired_call']['agent_id'] == 'claude-from-the-tail'
        else:
            await h.call(
                'submit_task', {'title': 'A task', 'description': self.DESCRIPTION}
            )
            assert h.recorder.args['agent_id'] == 'claude-from-the-tail'
        assert h.escalations == [], 'a repairable value is not residue'


# ---------------------------------------------------------------------------
# INV-2 — structured facts. No consumer log-scrapes.
# ---------------------------------------------------------------------------


#: The eight keys PRD section 6 contracts for ``markup_detected``, plus the key
#: naming the fact itself. Asserted as an EXACT set, which catches drift in both
#: directions: a missing key sends a consumer back to log-scraping, and an extra
#: value-bearing key would turn the fact stream into a second copy of the
#: caller's payload.
FACT_KEYS = {
    'fact',
    'tool',
    'param',
    'pattern',
    'misclose',
    'outcome',
    'recovered_params',
    'agent_id',
    'project',
}


class TestINV2StructuredFacts:
    """Every outcome emits one ``markup_detected``, complete enough to consume.

    PRD section 2.2's whole diagnosis failure was that the write-time guard's
    account of a leak was ambiguous — it enumerated one closing tag, so a
    mis-closed ``description`` could not report its own tag and the guard blamed
    whatever followed. A fact stream that made an operator reconstruct that from
    log lines would rebuild the same ambiguity one layer up.
    """

    # -- one fact per outcome, and the vocabulary is closed ----------------

    async def test_a_forwarded_repair_emits_repaired(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'escalate_info',
            {'summary': 'It stranded', 'detail': TestB3StrandRiskTierForwards.DETAIL},
        )

        assert [f['outcome'] for f in h.facts] == ['repaired']

    async def test_a_rejected_repair_emits_rejected(self):
        """SAME repair, other tier. The tier names the outcome, not the damage."""
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call(
                'escalate_info',
                {'summary': 'It stranded', 'detail': TestB3StrandRiskTierForwards.DETAIL},
            )

        assert [f['outcome'] for f in h.facts] == ['rejected']

    @BOTH_POLICIES
    async def test_an_unrepairable_value_emits_unrepairable_under_both_tiers(self, policy):
        h = build_harness(policy)

        with pytest.raises(ToolError):
            await h.call(
                'add_reuse_item',
                {
                    'what': 'w',
                    'how': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'where': 'shared/',
                },
            )

        assert [f['outcome'] for f in h.facts] == ['unrepairable']

    async def test_the_outcome_vocabulary_is_exactly_these_three(self):
        """A closed vocabulary, exported so INV-2's facts and INV-4's storm key
        read it from ONE place and cannot drift apart."""
        from shared.mcp_markup_middleware import OUTCOMES

        assert set(OUTCOMES) == {'repaired', 'rejected', 'unrepairable'}
        assert len(OUTCOMES) == 3

    async def test_every_emitted_outcome_is_in_the_declared_vocabulary(self):
        """The vocabulary is not merely declared — it is what the code emits."""
        from shared.mcp_markup_middleware import OUTCOMES

        seen = set()
        for policy in RepairPolicy:
            forward = build_harness(policy)
            # One tier forwards this and the other rejects it; the point here
            # is which outcome each EMITS, so the raise is beside the point.
            with contextlib.suppress(ToolError):
                await forward.call(
                    'escalate_info',
                    {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL},
                )
            refuse = build_harness(policy)
            with pytest.raises(ToolError):
                await refuse.call(
                    'add_reuse_item',
                    {
                        'what': 'w',
                        'how': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                        'where': 'shared/',
                    },
                )
            seen |= {f['outcome'] for f in forward.facts + refuse.facts}

        assert seen == set(OUTCOMES), 'every declared outcome is reachable, and only those'

    # -- shape completeness on all three paths -----------------------------

    async def test_the_repaired_fact_is_shape_complete(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'escalate_info',
            {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL},
        )

        assert set(h.facts[0]) == FACT_KEYS

    async def test_the_rejected_fact_is_shape_complete(self):
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call('submit_task', {'title': 'A task', 'description': TestB1PartialDrift.DESCRIPTION})

        assert set(h.facts[0]) == FACT_KEYS

    async def test_the_unrepairable_fact_is_shape_complete(self):
        """Complete even where the answer is ``None``.

        There is no accepted mis-close on this path — that IS why it is
        unrepairable — so ``misclose`` is present and null rather than absent. A
        consumer must never have to distinguish "no mis-close" from "this
        emitter forgot the key".
        """
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        with pytest.raises(ToolError):
            await h.call(
                'add_reuse_item',
                {
                    'what': 'w',
                    'how': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'where': 'shared/',
                },
            )

        assert set(h.facts[0]) == FACT_KEYS
        assert h.facts[0]['misclose'] is None
        assert h.facts[0]['recovered_params'] == []

    async def test_the_fact_names_itself(self):
        """So a multiplexed sink does not have to infer the record's type."""
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'escalate_info',
            {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL},
        )

        assert h.facts[0]['fact'] == 'markup_detected'

    async def test_the_fact_locates_the_leak_by_tool_and_param(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'escalate_info',
            {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL},
        )

        assert h.facts[0]['tool'] == 'escalate_info'
        assert h.facts[0]['param'] == 'detail'

    # -- recovered_params: NAMES, never values -----------------------------

    async def test_recovered_params_carries_names_only(self):
        """The fact stream must not become a second copy of the caller's payload.

        Names are what a consumer needs to see WHICH parameters a leak is
        eating; the values are the caller's data, and this stream is not the
        place it survives — that is the residue escalation's job, on the one
        path where the data would otherwise be lost.
        """
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call(
                'submit_task',
                {'title': 'A task', 'description': TestB1PartialDrift.DESCRIPTION},
            )

        assert h.facts[0]['recovered_params'] == ['priority']
        assert 'low' not in json.dumps(h.facts[0]), 'the recovered VALUE leaked into the fact'

    async def test_no_recovered_value_and_no_payload_rides_along(self):
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call(
                'submit_task',
                {'title': 'A task', 'description': TestB2TotalDrift.DESCRIPTION},
            )

        assert sorted(h.facts[0]['recovered_params']) == ['agent_id', 'metadata', 'priority']
        serialized = json.dumps(h.facts[0])
        assert '{"source": "probe"}' not in serialized, 'a recovered value leaked'
        assert 'Fix it.' not in serialized, 'the absorbing value leaked'

    async def test_an_agent_id_the_leak_SWALLOWED_still_attributes_the_leak(self):
        """The one recovered value the fact may carry — because it is not a
        value, it is the attribution.

        PRD section 2.1's first specimen has the leak swallowing ``agent_id``
        itself, so resolving identity from the pre-repair arguments would blank
        the attribution on the single most common corruption shape — losing the
        one thing "who is leaking?" exists to answer.

        Safe because the repair VALIDATED: repair() only returns a recovery
        whose names are disjoint from the supplied ones (B9), so this can never
        overwrite an ``agent_id`` the caller actually sent. And it is coherent
        with what the same rejection already does — ``repaired_call`` hands this
        exact value back and tells the caller to resend it, so treating it as
        untrustworthy for attribution alone would be incoherent.
        """
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'submit_task',
                {'title': 'A task', 'description': TestB2TotalDrift.DESCRIPTION},
            )

        assert h.facts[0]['agent_id'] == 'claude-x'
        assert _reject_payload(excinfo)['repaired_call']['agent_id'] == 'claude-x'

    async def test_a_supplied_agent_id_always_wins_over_the_tail(self):
        """The other side of that decision, and the one that must not slip.

        B9 makes a colliding tail unrepairable outright, so the recovered value
        never reaches identity resolution at all — but this pins the OUTCOME
        rather than the mechanism, so a future refactor cannot quietly start
        preferring the tail.
        """
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call(
                'submit_task',
                {
                    'title': 'A task',
                    'description': TestB9RecoveredNameCollidesWithASuppliedArgument.DESCRIPTION,
                    'agent_id': 'claude-the-caller',
                },
            )

        assert h.facts[0]['agent_id'] == 'claude-the-caller'

    async def test_recovered_params_is_empty_for_the_last_parameter_case(self):
        """B4 recovers nothing, and an empty list is the honest report of that.

        Omitting the key would make "nothing was swallowed" indistinguishable
        from "this path forgot to say".
        """
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call('add_memory', {'content': TestB4LastParameterNothingDropped.CONTENT})

        assert h.facts[0]['recovered_params'] == []
        assert h.facts[0]['outcome'] == 'repaired'

    # -- pattern vs misclose: the diagnostic 2.2 could not produce ---------

    async def test_pattern_and_misclose_are_different_questions(self):
        """``pattern`` is the literal detect() matched; ``misclose`` is the tag
        that actually drifted — verbatim, INCLUDING the dialect blend's stray
        quote, which is not an envelope literal at all.

        PRD section 2.2: the old guard enumerated one closing tag, so a
        mis-closed ``description`` could not report its own tag and the guard
        blamed whatever happened to follow it. Collapsing these two into one
        field would rebuild exactly that ambiguity.
        """
        blended = (
            'Do the thing.'
            + '\x3c/description">'
            + '\n' + _canonical_opener('priority') + 'low'
        )
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call('submit_task', {'title': 'A task', 'description': blended})

        assert h.facts[0]['misclose'] == '\x3c/description">'
        assert h.facts[0]['pattern'] == '\x3cparameter name='
        assert h.facts[0]['pattern'] != h.facts[0]['misclose']

    async def test_the_unrepairable_fact_still_names_the_matched_pattern(self):
        """A refusal that could not say WHAT it saw would be unactionable."""
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)

        with pytest.raises(ToolError):
            await h.call(
                'add_reuse_item',
                {
                    'what': 'w',
                    'how': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'where': 'shared/',
                },
            )

        assert h.facts[0]['pattern'] == INVOKE_CLOSER

    # -- identity: read from the call's own arguments -----------------------

    async def test_agent_id_comes_from_the_calls_own_argument(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'add_memory',
            {
                'content': TestB4LastParameterNothingDropped.CONTENT,
                'agent_id': 'claude-task-3689-implementer',
            },
        )

        assert h.facts[0]['agent_id'] == 'claude-task-3689-implementer'

    async def test_agent_id_is_none_when_the_caller_supplied_none(self):
        """None, not a placeholder. An unattributed leak is a real state, and
        inventing an attribution would point an operator at an innocent agent."""
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call('add_memory', {'content': TestB4LastParameterNothingDropped.CONTENT})

        assert h.facts[0]['agent_id'] is None

    async def test_project_root_is_the_project(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'escalate_info',
            {
                'summary': 's',
                'detail': TestB3StrandRiskTierForwards.DETAIL,
                'project_root': '/home/leo/src/dark-factory',
            },
        )

        assert h.facts[0]['project'] == '/home/leo/src/dark-factory'

    async def test_project_id_is_used_when_there_is_no_project_root(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'add_memory',
            {
                'content': TestB4LastParameterNothingDropped.CONTENT,
                'project_id': 'dark_factory',
            },
        )

        assert h.facts[0]['project'] == 'dark_factory'

    async def test_project_root_WINS_when_both_are_supplied(self):
        """A declared precedence, not an accident of dict order.

        project_root is the label MarkupStormCounter already keys on and the one
        an escalation queue is addressed by, so a burst attributed here lands in
        the same queue attribution the write-time tripwire would have chosen.
        """
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call(
            'add_memory',
            {
                'content': TestB4LastParameterNothingDropped.CONTENT,
                'project_id': 'dark_factory',
                'project_root': '/home/leo/src/dark-factory',
            },
        )

        assert h.facts[0]['project'] == '/home/leo/src/dark-factory'

    async def test_project_is_none_when_neither_is_resolvable(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)

        await h.call('add_memory', {'content': TestB4LastParameterNothingDropped.CONTENT})

        assert h.facts[0]['project'] is None

    # -- the sink is additive, on every path -------------------------------

    @BOTH_POLICIES
    async def test_a_raising_fact_sink_does_not_change_a_repair(self, policy):
        """The outcome is already decided by the time the fact is emitted.

        A sink outage must cost an operator visibility, not turn a working
        guard into an outage of its own.
        """
        def boom(fact):
            raise RuntimeError('fact stream unavailable')

        h = build_harness(policy, fact_sink=boom)
        payload = {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL}

        if policy is RepairPolicy.FORWARD_REPAIR:
            await h.call('escalate_info', payload)
            assert h.recorder.args['suggested_action'] == 'do the thing'
        else:
            with pytest.raises(ToolError) as excinfo:
                await h.call('escalate_info', payload)
            assert _reject_payload(excinfo)['outcome'] == 'rejected'
            assert h.recorder.calls == []

    @BOTH_POLICIES
    async def test_a_raising_fact_sink_does_not_change_a_refusal(self, policy):
        def boom(fact):
            raise RuntimeError('fact stream unavailable')

        h = build_harness(policy, fact_sink=boom)

        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'add_reuse_item',
                {
                    'what': 'w',
                    'how': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'where': 'shared/',
                },
            )

        assert _reject_payload(excinfo)['outcome'] == 'unrepairable'
        assert h.recorder.calls == []
        assert len(h.escalations) == 1, (
            'a failed fact must not suppress the residue escalation — they are '
            'independent channels and the payload only survives in one of them'
        )

    # -- an ASYNC emitter is awaited, not dropped --------------------------

    @BOTH_POLICIES
    async def test_an_async_fact_sink_is_awaited(self, policy):
        """The registration site (task 3690) may wire an ``async def`` emitter.

        This repo's queue/escalation machinery is largely async, so that is a
        legitimate thing to pass. A sink CALLED but not AWAITED records nothing
        while handing back a coroutine that looks like a result, and the only
        trace is a bare ``coroutine was never awaited`` RuntimeWarning — the
        silent fail-soft this module exists to end, committed by the module.
        """
        recorded: list[dict[str, Any]] = []

        async def sink(fact):
            recorded.append(fact)

        h = build_harness(policy, fact_sink=sink)
        payload = {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL}

        if policy is RepairPolicy.FORWARD_REPAIR:
            await h.call('escalate_info', payload)
        else:
            with pytest.raises(ToolError):
                await h.call('escalate_info', payload)

        assert [fact['outcome'] for fact in recorded] == [
            'repaired' if policy is RepairPolicy.FORWARD_REPAIR else 'rejected'
        ]

    @BOTH_POLICIES
    async def test_an_async_escalation_sink_is_awaited_and_its_id_reaches_the_caller(
        self, policy
    ):
        """The path where dropping the coroutine would DESTROY the payload.

        The residue escalation is the only surviving copy of an unrepairable
        call's data. An unawaited coroutine queues nothing, and because it is not
        a ``str`` the refusal reports ``escalation_id: null`` — so the caller is
        told its data is preserved somewhere that does not exist.
        """
        recorded: list[dict[str, Any]] = []

        async def sink(record):
            recorded.append(record)
            return 'esc-async-1'

        h = build_harness(policy, escalation_sink=sink)
        raw = specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value']

        with pytest.raises(ToolError) as excinfo:
            await h.call('add_reuse_item', {'what': 'w', 'how': raw, 'where': 'shared/'})

        assert [record['raw_value'] for record in recorded] == [raw]
        assert _reject_payload(excinfo)['escalation_id'] == 'esc-async-1'

    @BOTH_POLICIES
    async def test_an_async_sink_that_raises_still_does_not_change_the_outcome(self, policy):
        """The never-raises contract has to survive the await, not just the call."""
        async def boom(record):
            raise RuntimeError('queue unavailable')

        h = build_harness(policy, fact_sink=boom, escalation_sink=boom)

        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'add_reuse_item',
                {
                    'what': 'w',
                    'how': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'where': 'shared/',
                },
            )

        payload = _reject_payload(excinfo)
        assert payload['outcome'] == 'unrepairable'
        assert payload['escalation_id'] is None, (
            'a failed sink reports no id — the caller is better told nothing '
            'than pointed at a record that was never queued'
        )
        assert h.recorder.calls == []

    # -- the two non-events ------------------------------------------------

    @BOTH_POLICIES
    async def test_an_exemption_and_an_override_stay_out_of_the_fact_stream(self, policy):
        """B7 and B6 re-asserted against the fact contract itself.

        Both are DECLARATIONS that the markup is not a leak. The stream measures
        leaks, so recording them would make it measure author intent instead —
        and the 0.26% corruption rate it exists to track would be unreadable.
        """
        h = build_harness(policy, exempt_tools=frozenset({'scan_memory_content'}))

        await h.call('scan_memory_content', {'needle': _closer('content') + INVOKE_CLOSER})
        await h.call(
            'submit_task',
            {
                'title': 't',
                'description': 'quoting ' + _closer('content') + INVOKE_CLOSER,
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert h.facts == []
        assert h.escalations == []


# ---------------------------------------------------------------------------
# B10 — the storm escape (INV-4).
# ---------------------------------------------------------------------------


class _Clock:
    """A hand-advanced clock, so a 3600s window is exercised without sleeping."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestB10StormEscape:
    """Repair is a FAIL-SOFT path, so it carries the rate/streak escalation.

    One corrupted call is routine — the measured rate is 0.26%. A BURST means
    the upstream serialization leak is running RIGHT NOW, which is the
    condition worth an operator's attention rather than a log line.

    The generalisation this task owns is the key. The write-time counter keys
    bursts by project alone, which was enough when the only outcome was
    "rejected". With two tiers there are three outcomes, and a burst of
    REPAIRS is exactly as urgent as a burst of rejections while being, by
    construction, invisible to every caller — the calls all succeeded.
    """

    STORM_TOOL = 'escalate_info'

    def _harness(self, policy, clock, **kwargs):
        return build_harness(
            policy, time_provider=clock, storm_threshold=3, storm_window_seconds=3600.0, **kwargs
        )

    async def _repair(self, h, project='/srv/alpha'):
        """One FORWARD_REPAIR repair attributed to *project*."""
        await h.call(
            self.STORM_TOOL,
            {
                'summary': 's',
                'detail': TestB3StrandRiskTierForwards.DETAIL,
                'project_root': project,
            },
        )

    async def _reject(self, h, project='/srv/alpha'):
        """One REJECT_WITH_REPAIR rejection attributed to *project*."""
        with pytest.raises(ToolError) as excinfo:
            await h.call(
                self.STORM_TOOL,
                {
                    'summary': 's',
                    'detail': TestB3StrandRiskTierForwards.DETAIL,
                    'project_root': project,
                },
            )
        return excinfo

    async def _unrepairable(self, h, project='/srv/alpha'):
        with pytest.raises(ToolError):
            await h.call(
                'add_memory',
                {
                    'content': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'category': 'observations_and_summaries',
                    'project_id': 'dark_factory',
                    'project_root': project,
                    'agent_id': 'claude-caller',
                },
            )

    @staticmethod
    def _storms(h) -> list[dict[str, Any]]:
        """The escalations that are STORMS, not residue records."""
        return [e for e in h.escalations if e.get('error_type') == 'mcp_markup_storm']

    # -- a burst of REPAIRS escalates --------------------------------------

    async def test_three_repairs_in_the_window_fire_exactly_one_storm(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(3):
            await self._repair(h)
            clock.advance(60)

        assert len(self._storms(h)) == 1

    async def test_the_storm_names_the_outcome_that_burst(self):
        """Not merely "markup rejections".

        A burst of repairs is invisible to every caller — the calls all
        succeeded — so a record that could not say WHICH outcome burst would
        send an operator looking for failures that never happened.
        """
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(3):
            await self._repair(h)
            clock.advance(60)

        assert self._storms(h)[0]['outcome'] == 'repaired'

    async def test_the_storm_carries_the_count_threshold_window_and_projects(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(3):
            await self._repair(h)
            clock.advance(60)

        storm = self._storms(h)[0]
        assert storm['count'] == 3
        assert storm['threshold'] == 3
        assert storm['window_seconds'] == 3600.0
        assert storm['project'] == '/srv/alpha'

    # -- the rate limit, and the window ------------------------------------

    async def test_a_fourth_event_inside_the_window_does_not_re_fire(self):
        """Without the rate limit a runaway would file one escalation per call."""
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(4):
            await self._repair(h)
            clock.advance(60)

        assert len(self._storms(h)) == 1

    async def test_a_fresh_burst_after_the_window_fires_again(self):
        """The leak coming back is news again — the alarm must not latch off."""
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(3):
            await self._repair(h)
            clock.advance(60)
        clock.advance(3600 * 2)
        for _ in range(3):
            await self._repair(h)
            clock.advance(60)

        assert len(self._storms(h)) == 2

    # -- the key is (project, outcome) -------------------------------------

    async def test_two_outcomes_on_one_project_do_NOT_pool(self):
        """The generalisation this task owns.

        2 repairs plus 2 rejections is four corrupted calls, but neither KEY
        crossed the threshold. Pooling them would fire an alarm naming an
        outcome that never burst, and an operator chasing a three-repair burst
        that does not exist learns to ignore the alarm.
        """
        clock = _Clock()
        forward = self._harness(RepairPolicy.FORWARD_REPAIR, clock)
        reject = self._harness(RepairPolicy.REJECT_WITH_REPAIR, clock)

        for _ in range(2):
            await self._repair(forward)
            clock.advance(60)
            await self._reject(reject)
            clock.advance(60)

        assert self._storms(forward) == []
        assert self._storms(reject) == []

    async def test_three_rejections_fire_a_burst_naming_rejected(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.REJECT_WITH_REPAIR, clock)

        for _ in range(3):
            await self._reject(h)
            clock.advance(60)

        assert [s['outcome'] for s in self._storms(h)] == ['rejected']

    async def test_two_projects_do_not_pool(self):
        """One server serves every project.

        A window that mixed them would file the escalation into whichever
        project happened to cross the threshold, while the actually-leaking
        project got nothing — the failure MarkupStormCounter's own docstring
        records.
        """
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for project in ('/srv/alpha', '/srv/beta', '/srv/alpha'):
            await self._repair(h, project=project)
            clock.advance(60)

        assert self._storms(h) == []

    async def test_an_unrepairable_burst_keys_independently(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(3):
            await self._unrepairable(h)
            clock.advance(60)

        assert [s['outcome'] for s in self._storms(h)] == ['unrepairable']

    async def test_a_residue_escalation_is_filed_for_every_unrepairable_call(self):
        """The storm's rate limit must not suppress the residue records.

        They answer different questions — "is a leak running?" versus "where is
        this caller's data?" — and rate-limiting the second would discard
        payloads.
        """
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(3):
            await self._unrepairable(h)
            clock.advance(60)

        residue = [e for e in h.escalations if e.get('error_type') == 'mcp_markup_unrepairable']
        assert len(residue) == 3

    # -- the caller-facing half --------------------------------------------

    async def test_the_storm_summary_is_folded_into_the_rejection(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.REJECT_WITH_REPAIR, clock)

        for _ in range(2):
            await self._reject(h)
            clock.advance(60)
        excinfo = await self._reject(h)

        storm = _reject_payload(excinfo)['storm']
        assert storm['count'] == 3
        assert storm['threshold'] == 3

    async def test_the_storm_summary_is_folded_into_the_FORWARDED_result(self):
        """The one tier whose callers cannot learn of the burst any other way.

        A FORWARD_REPAIR caller is not bounced — its call SUCCEEDED — so the
        ``meta['markup_repair']`` block is the only channel it has. Leaving the
        burst out of exactly the tier that is invisible, while both refusal
        payloads carry it, would make the storm escape blind on the path
        ``_record_storm``'s own docstring argues is just as urgent.
        """
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(2):
            await self._repair(h)
            clock.advance(60)
        result = await h.call(
            self.STORM_TOOL,
            {
                'summary': 's',
                'detail': TestB3StrandRiskTierForwards.DETAIL,
                'project_root': '/srv/alpha',
            },
        )

        storm = meta_of(result)['markup_repair']['storm']
        assert storm['count'] == 3
        assert storm['threshold'] == 3
        assert storm['outcome'] == 'repaired'

    async def test_the_forwarded_storm_key_is_OMITTED_when_no_burst_fired(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        result = await h.call(
            self.STORM_TOOL,
            {'summary': 's', 'detail': TestB3StrandRiskTierForwards.DETAIL},
        )

        assert 'storm' not in meta_of(result)['markup_repair']

    async def test_the_storm_key_is_OMITTED_when_no_burst_fired(self):
        """Presence is the signal — markup_tripwire's convention.

        A key set to null would make "no burst" and "a burst reported as
        nothing" the same wire shape.
        """
        clock = _Clock()
        h = self._harness(RepairPolicy.REJECT_WITH_REPAIR, clock)

        excinfo = await self._reject(h)

        assert 'storm' not in _reject_payload(excinfo)

    async def test_the_unrepairable_refusal_also_carries_a_fired_storm(self):
        clock = _Clock()
        h = self._harness(RepairPolicy.REJECT_WITH_REPAIR, clock)

        for _ in range(2):
            await self._unrepairable(h)
            clock.advance(60)
        with pytest.raises(ToolError) as excinfo:
            await h.call(
                'add_memory',
                {
                    'content': specimen(TestB5UnrepairableIsNeverGuessed.SPECIMEN_ID)['value'],
                    'project_root': '/srv/alpha',
                },
            )

        assert _reject_payload(excinfo)['storm']['count'] == 3

    # -- exemptions and overrides are not events ---------------------------

    @BOTH_POLICIES
    async def test_exemptions_and_overrides_never_count_toward_a_burst(self, policy):
        clock = _Clock()
        h = self._harness(
            policy, clock, exempt_tools=frozenset({'scan_memory_content'})
        )

        for _ in range(6):
            await h.call('scan_memory_content', {'needle': _closer('content')})
            await h.call(
                'submit_task',
                {
                    'title': 't',
                    'description': 'quoting ' + _closer('content'),
                    'metadata': {'allow_mcp_markup': True},
                },
            )
            clock.advance(60)

        assert h.escalations == []

    async def test_the_counter_is_per_instance(self):
        """No burst state bleeds between servers, or between tests."""
        clock = _Clock()
        first = self._harness(RepairPolicy.FORWARD_REPAIR, clock)
        second = self._harness(RepairPolicy.FORWARD_REPAIR, clock)

        for _ in range(2):
            await self._repair(first)
            clock.advance(60)
        for _ in range(2):
            await self._repair(second)
            clock.advance(60)

        assert self._storms(first) == []
        assert self._storms(second) == []


# ---------------------------------------------------------------------------
# B14 — a REQUIRED absorbed parameter is recoverable.
# ---------------------------------------------------------------------------


class TestB14RequiredParameterAbsorption:
    """``on_call_tool`` runs BEFORE pydantic argument validation, so a REQUIRED
    absorbed parameter is still recoverable — PRD boundary row B14, the one leak
    shape the write-time tripwire cannot see at all.

    Measured against fastmcp 3.2.2 rather than inferred from framework docs:
    ``server.py:1112-1132`` runs ``_run_middleware`` FIRST and its ``call_next``
    re-reads the arguments back off the context at ``server.py:1127`` (which is
    why an in-place mutation lands); validation is
    ``type_adapter.validate_python(arguments)`` down at
    ``tools/function_tool.py:249/253/273/276``, which is where
    ``Missing required argument`` comes from. Middleware strictly precedes it.

    Why this row had to exist. Every recovery target asserted anywhere else in
    this file is a DEFAULTED parameter — B1 ``priority``, B2
    ``priority``/``agent_id``/``metadata``, B3 ``suggested_action``, B9
    ``agent_id`` — because the legacy toy ``add_memory`` declares
    ``project_id: str | None = None`` while the real one
    (``fused_memory/server/tools.py:2845``) declares ``project_id: str``. The
    toy under-modelled the real signature at exactly the point in question, so
    the loudest observed leak shape was structurally inexpressible here.
    ``add_memory_strict`` closes that gap; the negative control below proves the
    parameter really is required, so none of the positive rows can pass
    vacuously.
    """

    #: ``content`` absorbed the REQUIRED ``project_id``, which is then absent
    #: from the argument map entirely — the shape observed first-hand on
    #: 2026-08-11 (task 3083, burst8), where the caller got a pydantic
    #: missing-required-field error and the write-time tripwire stayed silent.
    CONTENT = (
        'A fact worth keeping.'
        + _closer('content')
        + '\n'
        + _opener('project_id')
        + 'dark_factory'
        + _closer('project_id')
        + INVOKE_CLOSER
    )

    #: What ``content`` should be left holding once the absorbed tail is peeled.
    CLEAN = 'A fact worth keeping.'

    # -- the premise: project_id really is required ------------------------

    @BOTH_POLICIES
    async def test_negative_control_the_parameter_is_genuinely_required(self, policy):
        """A CLEAN value with ``project_id`` simply omitted must fail.

        Without this, a toy whose parameter was not really required would make
        every positive assertion below pass vacuously — which is precisely the
        failure mode that let this shape go uncovered across 504 corpus records.
        """
        assert detect(self.CLEAN) is None, (
            'the control value must carry no markup at all, or it would prove '
            'something about the repair path rather than about required-ness'
        )
        h = build_harness(policy)

        with pytest.raises(ToolError) as excinfo:
            await h.call('add_memory_strict', {'content': self.CLEAN})

        text = str(excinfo.value)
        assert 'project_id' in text, text
        assert 'missing' in text.lower() and 'required' in text.lower(), text
        assert h.facts == [], 'a clean value must not enter the fact stream'
        assert h.recorder.calls == []

    # -- FORWARD_REPAIR: the ordering proof --------------------------------

    async def _forward(self):
        h = build_harness(RepairPolicy.FORWARD_REPAIR)
        result = await h.call('add_memory_strict', {'content': self.CONTENT})
        return h, result

    async def test_the_tool_runs_bound_to_a_required_argument_never_on_the_wire(self):
        """The strongest form of the ordering proof.

        ``project_id`` was never supplied by the caller — it existed only inside
        ``content``'s leaked tail. The tool body nonetheless executed bound to
        it, which is only possible if the guard ran before validation.
        """
        h, _ = await self._forward()

        assert len(h.recorder.calls) == 1
        assert h.recorder.args['project_id'] == 'dark_factory'

    async def test_the_repaired_content_is_the_prefix_only(self):
        h, _ = await self._forward()

        assert h.recorder.args['content'] == self.CLEAN
        assert detect(h.recorder.args['content']) is None

    async def test_meta_names_the_recovered_required_parameter(self):
        _, result = await self._forward()

        warning = meta_of(result)['markup_repair']
        assert warning['field'] == 'content'
        assert warning['recovered_params'] == ['project_id']
        assert warning['outcome'] == 'repaired'

    # -- REJECT_WITH_REPAIR ------------------------------------------------

    async def _reject(self):
        h = build_harness(RepairPolicy.REJECT_WITH_REPAIR)
        with pytest.raises(ToolError) as excinfo:
            await h.call('add_memory_strict', {'content': self.CONTENT})
        return h, _reject_payload(excinfo)

    async def test_the_caller_is_bounced_with_a_markup_error_not_a_pydantic_one(self):
        """The distinction B14 exists to pin.

        A guard that ran AFTER validation could only ever produce the pydantic
        missing-required-field error the caller already gets today — useless,
        because it names the absent parameter and not the leak that ate it.
        """
        _, payload = await self._reject()

        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['tool'] == 'add_memory_strict'
        assert payload['field'] == 'content'

    async def test_the_repaired_call_carries_the_recovered_required_argument(self):
        _, payload = await self._reject()

        assert payload['repaired_call']['project_id'] == 'dark_factory'
        assert payload['repaired_call']['content'] == self.CLEAN
        assert payload['recovered_params'] == ['project_id']

    async def test_nothing_is_written(self):
        h, _ = await self._reject()

        assert h.recorder.calls == []

    # -- the fact, under both tiers ----------------------------------------

    @BOTH_POLICIES
    async def test_the_fact_names_the_recovered_required_parameter(self, policy):
        """Tier-independent: the recovery is a property of the ordering."""
        h = build_harness(policy)

        with contextlib.suppress(ToolError):
            await h.call('add_memory_strict', {'content': self.CONTENT})

        assert len(h.facts) == 1
        assert h.facts[0]['fact'] == 'markup_detected'
        assert h.facts[0]['param'] == 'content'
        assert h.facts[0]['recovered_params'] == ['project_id']
