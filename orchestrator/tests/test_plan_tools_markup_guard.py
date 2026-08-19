"""The envelope-markup boundary guard REGISTERED on the plan-tools server.

Task 4457 (PRD ``plans/toolcall-markup-containment-prd.md``, leaf gamma-2).
The subject under test is the REGISTRATION itself — that
``plan_tools.create_server`` puts :class:`shared.mcp_markup_middleware.
MarkupGuardMiddleware` in front of every plan-tools tool under the declared
policy, that a rejection writes nothing, and that the guard COMPOSES with task
3692's read-time repair path instead of superseding it.

Detection, repair and policy are owned by ``shared.toolcall_markup`` and
``shared.mcp_markup_middleware`` and are pinned by THEIR tests. Nothing here
re-derives them; the assertions below are about this server.

## Why these tests drive a Client over the REAL server

Middleware sits at the SERVER REQUEST layer, and all three established
in-process idioms in this repo bypass it entirely: ``tool.fn(...)``,
``await tool.run({...})`` and ``server._tool_manager.call_tool(...)``. A test
written any of those ways would pass while running none of the guard. Only
``async with Client(server)`` traverses the middleware chain (measured for task
3689 in ``shared/tests/test_mcp_markup_middleware.py``).

That same fact is why registering the guard cannot break the existing
plan-tools suites — ``test_plan_tools_markup_repair.py`` calls the standalone
``_add_design_decision``-style helpers directly, and the ``create_server``
suites drive ``tool.fn`` / ``tool.run``.

Unlike shared's toy harness, this module drives the REAL
``create_server(artifacts)``: the thing under test is the registration, and a
toy server would prove nothing about it.

## Async marker

Every async test carries an explicit ``@pytest.mark.asyncio``. orchestrator
does NOT set ``asyncio_mode = auto`` (shared does — do not copy that half of
the idiom from its middleware tests).

## Sentinel-literal hazard — every specimen is BUILT, never written verbatim

This module describes MCP tool-call envelope markup, so it is exactly the file
that must not contain any of it literally. The rationale is the one recorded at
``shared/src/shared/toolcall_markup.py`` lines 52-62: an agent editing a file
that holds a raw envelope literal has to emit that literal INSIDE its own
tool-call argument, which reproduces the very over-consumption defect under
test — the Write/Edit argument terminates early, truncating this file and
silently dropping that call's sibling arguments.

So every specimen is assembled from :func:`_close` / :func:`_open_param`, which
build their angle bracket from ``chr(60)``, and
:func:`_assert_no_raw_sentinels` enforces that on this module's OWN BYTES at
import — checked against ``shared.toolcall_markup.ENVELOPE_LITERALS``, the
single owner of the literal set (INV-5), plus the two structural prefixes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import ENVELOPE_LITERALS, MARKUP_OVERRIDE_KEY, detect

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

# ---------------------------------------------------------------------------
# Sentinel BUILDERS — the only way markup enters this module.
# ---------------------------------------------------------------------------

#: The opening angle bracket, spelled so it never appears verbatim in the file.
_LT = chr(60)


def _close(name: str) -> str:
    """Build the closing tag for *name* (the mis-close shape the harness emits)."""
    return _LT + '/' + name + '>'


def _open_param(name: str) -> str:
    """Build the canonical opening tag for parameter *name*."""
    return _LT + 'parameter name="' + name + '">'


#: The bare invoke closer — the terminator that trails a last-parameter leak.
_INVOKE_CLOSER = _close('invoke')


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal.

    Checked against ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single
    owner of the literal set, INV-5) plus the two structural prefixes every
    built specimen uses, so a builder output spelled out by hand is caught even
    when it is not itself one of the enumerated literals.
    """
    source = Path(__file__).read_text(encoding='utf-8')
    forbidden = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')
    for sequence in forbidden:
        if sequence in source:
            raise AssertionError(
                f'{Path(__file__).name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from _close()/_open_param() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See the module docstring.'
            )


_assert_no_raw_sentinels()


# ---------------------------------------------------------------------------
# Specimens — the measured plan-tools leak shapes.
# ---------------------------------------------------------------------------

_DECISION_PROSE = (
    'Register the boundary guard on plan-tools rather than relying on the '
    'read-time repair alone, because the read-time path cannot see inbound '
    'arguments at all.'
)
_RATIONALE_PROSE = (
    'The two layers guard different populations: arguments being sent now, '
    'versus damage already stored in plan.json.'
)
_TITLE_PROSE = 'Register the markup guard on plan-tools'
_ANALYSIS_PROSE = 'The registration site is create_server, which owns the declaration.'

#: ABSORBED SIBLING on ``add_design_decision.decision`` — the measured
#: plan-tools specimen (45 corrupted calls, the largest single victim on this
#: server). The parser mis-closed ``decision`` and swallowed the whole
#: ``rationale`` parameter into it; the final opener is UNTERMINATED because its
#: closer was consumed as the terminator.
ABSORBED_RATIONALE = (
    _DECISION_PROSE + _close('decision') + '\n' + _open_param('rationale') + _RATIONALE_PROSE
)

#: The same shape on ``create_plan.title``, absorbing ``analysis``. This is the
#: tool ``_create_plan``'s own comment delegates to the write-time middleware.
ABSORBED_ANALYSIS = (
    _TITLE_PROSE + _close('title') + '\n' + _open_param('analysis') + _ANALYSIS_PROSE
)

#: TRAILING RESIDUE on a STORED ``design_decisions[].rationale`` — the dominant
#: live shape, and the population task 3692's read-time path owns. Nothing was
#: absorbed: the parameter was last in the call, so only the mis-close and the
#: invoke closer trail it.
STORED_TRAILING_RATIONALE = (
    _RATIONALE_PROSE + _close('rationale') + '\n' + _INVOKE_CLOSER + '\n'
)

#: Prose that QUOTES the literals deliberately — a plan about this very leak
#: (worktree 2939 is the live specimen). The escape hatch, not a leak.
QUOTED_DECISION = (
    'The harness emits ' + _close('decision') + ' mid-value and then ' + _INVOKE_CLOSER
    + ', which is what the guard matches on.'
)


# ---------------------------------------------------------------------------
# Fixtures and harness.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_reported_refusals():
    """Clear ``_REPORTED_REFUSALS`` around every test in this module.

    It is PROCESS-global memo state owned by the read-time path, so a refusal
    reported by one test would otherwise be suppressed in the next.
    """
    plan_tools._REPORTED_REFUSALS.clear()
    yield
    plan_tools._REPORTED_REFUSALS.clear()


class Harness:
    """The REAL plan-tools server over a temp worktree, driven by a Client."""

    def __init__(self, artifacts: TaskArtifacts) -> None:
        self.artifacts = artifacts
        self.server = plan_tools.create_server(artifacts)

    async def call(self, tool: str, arguments: dict[str, Any]):
        async with Client(self.server) as client:
            return await client.call_tool(tool, arguments)

    # -- plan.json access ------------------------------------------------

    @property
    def plan_path(self) -> Path:
        return self.artifacts.root / 'plan.json'

    def plan_bytes(self) -> bytes:
        return self.plan_path.read_bytes()

    def plan(self) -> dict[str, Any]:
        return json.loads(self.plan_path.read_text(encoding='utf-8'))

    async def seed_plan(self) -> None:
        """Create a CLEAN plan through the guard, as an architect would."""
        await self.call(
            'create_plan',
            {
                'task_id': 'test-1',
                'title': 'A clean plan',
                'analysis': 'Clean analysis prose describing the approach.',
                'files': ['orchestrator/src/orchestrator/mcp/plan_tools.py'],
            },
        )

    def store_damaged_rationale(self) -> None:
        """Poison the STORED ``design_decisions[0].rationale`` on disk.

        Damage that has ALREADY LANDED is 3692's population — the middleware
        never sees it, because the middleware only ever sees what is being sent
        now.
        """
        plan = self.plan()
        plan['design_decisions'] = [
            {'decision': 'A clean stored decision.', 'rationale': STORED_TRAILING_RATIONALE},
        ]
        self.artifacts.write_plan(plan)


@pytest.fixture()
def harness(tmp_path) -> Harness:
    """A plan-tools server over a temp worktree — mirrors ``test_plan_tools_server``."""
    artifacts = TaskArtifacts(tmp_path)
    artifacts.init('test-1', 'Test task', 'A test')
    return Harness(artifacts)


def _refusal(excinfo) -> dict[str, Any]:
    """The refusal payload, parsed off the ToolError.

    Measured for task 3689: a ``json.dumps`` payload raised from
    ``on_call_tool`` round-trips to the caller BYTE-INTACT, which is what lets
    ``repaired_call`` survive the boundary at all.
    """
    return json.loads(str(excinfo.value))


# ---------------------------------------------------------------------------
# (a)/(b) — PRD boundary rows B1/B2 on the real server.
# ---------------------------------------------------------------------------


class TestAbsorbedSiblingIsRejected:
    """``add_design_decision.decision`` swallowed its ``rationale`` sibling.

    Chosen because it is THE measured plan-tools specimen and because it
    repairs by construction: supplied is ``('decision',)``, recovered is
    ``{'rationale'}`` — in schema, and disjoint from supplied.
    """

    @pytest.mark.asyncio
    async def test_the_call_is_refused_machine_readably(self, harness: Harness):
        await harness.seed_plan()

        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_design_decision', {'decision': ABSORBED_RATIONALE})

        payload = _refusal(excinfo)
        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['outcome'] == 'rejected'
        assert payload['tool'] == 'add_design_decision'
        assert payload['field'] == 'decision'
        assert payload['misclose'] == _close('decision')
        assert payload['recovered_params'] == ['rationale']

    @pytest.mark.asyncio
    async def test_the_repaired_call_is_a_mechanical_retry(self, harness: Harness):
        """D5: the clean value is a PREFIX of what was sent, never a rewrite."""
        await harness.seed_plan()

        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_design_decision', {'decision': ABSORBED_RATIONALE})

        repaired = _refusal(excinfo)['repaired_call']
        assert repaired['decision'] == _DECISION_PROSE
        assert ABSORBED_RATIONALE.startswith(repaired['decision'])
        assert repaired['rationale'] == _RATIONALE_PROSE
        for name, value in repaired.items():
            if isinstance(value, str):
                assert detect(value) is None, f'{name} still carries envelope markup'

    @pytest.mark.asyncio
    async def test_the_tool_body_never_ran(self, harness: Harness):
        """What makes "reject writes nothing" TRUE rather than merely intended."""
        await harness.seed_plan()
        before = harness.plan_bytes()

        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert _refusal(excinfo)['error_type'] == 'mcp_markup_detected', (
            'the refusal must come from the GUARD: a pydantic "Missing '
            'required argument" also writes nothing, so without this pin '
            'the row would pass with no middleware registered at all'
        )
        assert harness.plan_bytes() == before, 'a rejected call must not touch plan.json'
        assert harness.plan()['design_decisions'] == []


# ---------------------------------------------------------------------------
# (c) — create_plan's INBOUND arguments, the delegation its comment makes.
# ---------------------------------------------------------------------------


class TestCreatePlanInboundArgumentsAreGuarded:
    """``_create_plan`` is deliberately unhooked from the read-time path.

    It overwrites plan.json wholesale, so there is no stored document to
    repair on the way in. Its comment delegates its INBOUND arguments to the
    write-time middleware — this leaf is that middleware, so the delegation is
    now kept.
    """

    @pytest.mark.asyncio
    async def test_a_poisoned_title_is_refused(self, harness: Harness):
        with pytest.raises(ToolError) as excinfo:
            await harness.call(
                'create_plan',
                {'task_id': 'test-1', 'title': ABSORBED_ANALYSIS, 'files': ['a.py']},
            )

        payload = _refusal(excinfo)
        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['tool'] == 'create_plan'
        assert payload['field'] == 'title'
        assert payload['repaired_call']['title'] == _TITLE_PROSE
        assert payload['repaired_call']['analysis'] == _ANALYSIS_PROSE

    @pytest.mark.asyncio
    async def test_no_plan_is_written(self, harness: Harness):
        with pytest.raises(ToolError) as excinfo:
            await harness.call(
                'create_plan',
                {'task_id': 'test-1', 'title': ABSORBED_ANALYSIS, 'files': ['a.py']},
            )

        assert _refusal(excinfo)['error_type'] == 'mcp_markup_detected', (
            'the refusal must come from the GUARD: a pydantic "Missing '
            'required argument" also writes nothing, so without this pin '
            'the row would pass with no middleware registered at all'
        )
        assert not harness.plan_path.exists(), (
            'create_plan overwrites plan.json wholesale — a rejected call must '
            'never reach that write'
        )


# ---------------------------------------------------------------------------
# (d)/(e)/(f) — the COMPOSE ruling, pinned as behaviour.
# ---------------------------------------------------------------------------


class TestComposesWithTheReadTimeRepair:
    """Two mechanisms, one boundary each — neither supersedes the other.

    The middleware guards INBOUND ARGUMENTS at the request layer; task 3692's
    read-time path repairs STORED STATE inside the tool body. Their populations
    are DISJOINT, which is why registering the guard does not retire 3692.
    """

    @pytest.mark.asyncio
    async def test_stored_damage_is_still_repaired_behind_the_guard(self, harness: Harness):
        """(d) The read-time path works unchanged with the guard in front of it."""
        await harness.seed_plan()
        harness.store_damaged_rationale()

        result = await harness.call(
            'add_design_decision',
            {'decision': 'A clean new decision.', 'rationale': 'A clean new rationale.'},
        )

        facts = result.data['markup_repairs']
        repaired = [f for f in facts if f['outcome'] == 'repaired']
        assert [(f['collection'], f['index'], f['field']) for f in repaired] == [
            ('design_decisions', 0, 'rationale')
        ]
        stored = harness.plan()['design_decisions'][0]['rationale']
        assert stored == _RATIONALE_PROSE
        assert detect(stored) is None

    @pytest.mark.asyncio
    async def test_a_rejection_never_becomes_a_second_fact(self, harness: Harness):
        """(e) No middleware-repaired value can ever reach storage.

        Under REJECT_WITH_REPAIR the guard NEVER forwards a repaired argument,
        so the read-time path can never re-report the same damage as a second
        fact. The disjointness is a CONSEQUENCE of the declared policy.
        """
        await harness.seed_plan()

        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert 'markup_repairs' not in _refusal(excinfo)
        plan = harness.plan()
        assert plan['design_decisions'] == []
        for decision in plan['design_decisions']:
            assert detect(decision['decision']) is None

    @pytest.mark.asyncio
    async def test_a_rejection_defers_the_stored_repair_and_never_loses_it(
        self, harness: Harness
    ):
        """(f) A rejected call short-circuits the tool body — a DEFERRAL, not a loss.

        Do not "fix" this by invoking the read-time repair from the middleware:
        that is how two mechanisms on one boundary become one tangled one.
        """
        await harness.seed_plan()
        harness.store_damaged_rationale()
        damaged = harness.plan_bytes()

        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert _refusal(excinfo)['error_type'] == 'mcp_markup_detected', (
            'the refusal must come from the GUARD: a pydantic "Missing '
            'required argument" also writes nothing, so without this pin '
            'the row would pass with no middleware registered at all'
        )
        assert harness.plan_bytes() == damaged, (
            'the tool body never ran, so the STORED damage is still there'
        )

        result = await harness.call(
            'add_design_decision',
            {'decision': 'A clean new decision.', 'rationale': 'A clean new rationale.'},
        )

        assert any(f['outcome'] == 'repaired' for f in result.data['markup_repairs'])
        assert harness.plan()['design_decisions'][0]['rationale'] == _RATIONALE_PROSE


# ---------------------------------------------------------------------------
# (g) — INV-1: the declaration is EXPLICIT and machine-checked.
# ---------------------------------------------------------------------------


class TestTheDeclarationIsMachineChecked:
    """INV-1: a policy is DECLARED at registration, never inferred per call."""

    def test_exactly_one_guard_is_registered_under_the_declared_policy(self, harness: Harness):
        guards = [
            m for m in harness.server.middleware if isinstance(m, MarkupGuardMiddleware)
        ]

        assert len(guards) == 1, 'one guard, one boundary — never two on one server'
        assert guards[0].policy is RepairPolicy.REJECT_WITH_REPAIR
        assert guards[0].exempt_tools == frozenset(), (
            'the empty exemption set is a DECLARATION, not an omission: no '
            'plan-tools tool has searching for envelope literals as its job'
        )

    def test_strict_input_validation_stays_off(self, harness: Harness):
        """Substrate fact 5: with it ON the middleware chain is never entered.

        Every required-parameter leak — which is the shape this server's
        largest victim takes — would become silently unrepairable, with no
        fact, no storm and no residue escalation.
        """
        assert harness.server.strict_input_validation is False


# ---------------------------------------------------------------------------
# (h) — the deliberate-quoting override on a metadata-less tool.
# ---------------------------------------------------------------------------


class TestDeliberateQuotingOverride:
    """An architect planning a task ABOUT this leak has to be able to say so.

    ``add_design_decision`` declares no ``metadata`` parameter, so this
    exercises ``_apply_override``'s DROP branch: the flag is stripped before
    dispatch rather than forwarded as an unexpected argument.
    """

    @pytest.mark.asyncio
    async def test_the_quoted_decision_lands_verbatim(self, harness: Harness):
        await harness.seed_plan()

        await harness.call(
            'add_design_decision',
            {
                'decision': QUOTED_DECISION,
                'rationale': 'Quoting the literals is the subject of the plan.',
                'metadata': {MARKUP_OVERRIDE_KEY: True},
            },
        )

        stored = harness.plan()['design_decisions'][-1]
        assert stored['decision'] == QUOTED_DECISION
        assert 'metadata' not in stored, (
            'the override is write-time-only control, never payload'
        )
