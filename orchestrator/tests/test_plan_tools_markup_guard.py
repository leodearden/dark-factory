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
import subprocess
from pathlib import Path
from typing import Any

import pytest
from escalation.models import Escalation
from fastmcp import Client
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import (
    ENVELOPE_LITERALS,
    MARKUP_OVERRIDE_KEY,
    detect,
    detect_for,
    repair,
)

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import markup_journal, plan_tools
from orchestrator.workflow import _is_gating_escalation

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


def journal_lines(root: Path) -> list[dict[str, Any]]:
    """Every plan-tools markup fact journalled under *root*, parsed.

    An absent file reads as no lines rather than raising: "the journal was
    never written" is an assertable outcome here, not an error.
    """
    path = markup_journal.journal_path(root, 'plan-tools')
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


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
def artifacts(tmp_path) -> TaskArtifacts:
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


@pytest.fixture()
def harness(artifacts: TaskArtifacts) -> Harness:
    """A plan-tools server over a temp worktree.

    Kept SEPARATE from ``artifacts`` because the residue rows below must
    install their fakes BEFORE ``create_server`` runs — the escalation sink is
    built once, at the registration site.
    """
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

        A CLEAN decision is stored first so the stored-state assertion has
        something to range over: against an empty list "no repaired value
        reached storage" is vacuously true and would hold with no guard at all.
        """
        await harness.seed_plan()
        await harness.call(
            'add_design_decision',
            {'decision': 'A clean stored decision.', 'rationale': 'A clean stored rationale.'},
        )

        with pytest.raises(ToolError) as excinfo:
            await harness.call('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert 'markup_repairs' not in _refusal(excinfo)
        decisions = harness.plan()['design_decisions']
        assert len(decisions) == 1, 'the refused call added nothing'
        for decision in decisions:
            assert detect(decision['decision']) is None
            assert detect(decision['rationale']) is None

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


# ---------------------------------------------------------------------------
# Residue preservation on the UNREPAIRABLE path.
# ---------------------------------------------------------------------------
#
# plan-tools owns 52 of the system's 95 unrepairable specimens
# (add_design_decision.decision x45, add_design_decision.rationale x4,
# add_reuse_item.how x3), so this is the path that decides whether
# REJECT_WITH_REPAIR is NON-DESTRUCTIVE here. Registering REJECT with no
# residue channel would convert "corrupt but present in plan.json" into
# "silently absent", which is a regression in data preservation dressed as a
# fix — and the middleware's own hint tells the caller its payload "is
# preserved verbatim in the escalation named above".

#: The committed corpus of REAL leaked calls (task 3688, boundary row B13).
CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / 'shared' / 'tests' / 'fixtures' / 'toolcall_markup_corpus.jsonl'
)


def _specimen(tool_use_id: str) -> dict[str, Any]:
    """The committed corpus record with this ``tool_use_id``.

    Keyed by id rather than by index or by "the shortest one", so a corpus
    refresh that adds records cannot silently repoint a test at a different
    payload. Read as JSON, which is also how the raw envelope literals reach
    this module WITHOUT appearing in its own source text — the corpus escapes
    every one of them as ``\\u003c``.
    """
    for line in CORPUS_PATH.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('tool_use_id') == tool_use_id:
            return record
    raise AssertionError(f'specimen {tool_use_id!r} is missing from {CORPUS_PATH}')


#: A REAL ``add_design_decision.decision`` payload whose own boundary cannot be
#: determined. A refusal is the one outcome an invented specimen can trivially
#: fake — anything sufficiently mangled refuses — so the assertion that this
#: guard never guesses is only worth something against input the harness really
#: produced.
UNREPAIRABLE_SPECIMEN = _specimen('toolu_01CX8okLpKgz5auenVYQzY23')
UNREPAIRABLE_DECISION: str = UNREPAIRABLE_SPECIMEN['value']

#: What ``add_design_decision`` declares, as the middleware resolves it LIVE
#: off the tool. Spelled here only so the specimen's unrepairability can be
#: re-derived independently of the server.
_DECISION_SCHEMA = ('decision', 'rationale')


class _FakeQueue:
    """Stands in for ``EscalationQueue``: records what was filed, hands ids back.

    A bare list would return ``None`` from ``submit``, which would let a sink
    that never propagated the queue's id pass the payload assertions by
    accident.
    """

    def __init__(
        self,
        *,
        submit_error: Exception | None = None,
        read_error: Exception | None = None,
    ) -> None:
        self.submitted: list[Escalation] = []
        self.pending: list[Escalation] = []
        #: Every ``get_by_task`` lookup, so "a residue record never dedups"
        #: is assertable as "it never even looked", not merely as a count.
        self.reads: list[tuple[str, str | None]] = []
        self.submit_error = submit_error
        self.read_error = read_error

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-{len(self.submitted) + 1}'

    def submit(self, escalation: Escalation) -> str:
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append(escalation)
        self.pending.append(escalation)
        return escalation.id

    def get_by_task(self, task_id: str, status: str | None = None) -> list[Escalation]:
        self.reads.append((task_id, status))
        if self.read_error is not None:
            raise self.read_error
        return [
            esc for esc in self.pending
            if esc.task_id == task_id and (status is None or esc.status == status)
        ]


class ResidueRig:
    """A real plan-tools server whose escalation channel points at a fake queue.

    ``records`` holds every record the MIDDLEWARE handed the sink, so the
    contracted keys are assertable directly rather than reconstructed from what
    the queue happens to have kept.
    """

    def __init__(
        self,
        harness: Harness,
        queue: _FakeQueue,
        records: list[dict[str, Any]],
        returns: list[str | None],
    ):
        self.harness = harness
        self.queue = queue
        self.records = records
        #: What the sink handed BACK for each record, in the same order. The
        #: middleware discards a storm's return value, so this is the only way
        #: to see that a deduped storm reported the still-open record's id.
        self.returns = returns

    def storm_records(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get('error_type') == 'mcp_markup_storm']

    async def refuse(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Drive one call that must be refused; return the refusal payload."""
        with pytest.raises(ToolError) as excinfo:
            await self.harness.call(tool, arguments)
        return _refusal(excinfo)


def build_residue_rig(
    monkeypatch, artifacts: TaskArtifacts, queue: _FakeQueue | None = None
) -> ResidueRig:
    """The REAL sink, with only its queue-facing seams faked.

    The two patched seams are the ones step-4 introduces: where this server
    files (``_markup_project_root``, normally derived from git) and what it
    files through (``_escalation_channel``, normally the lazily imported
    ``EscalationQueue``). The record BUILDER — attribution, level, category,
    the raw payload — is the real one, which is the part under test.

    ``_markup_project_root`` steers BOTH injected channels, so this also lands
    the task-4744 fact journal under ``artifacts.worktree`` (which is the
    ``tmp_path`` the fixtures are built over) instead of shelling out to
    ``git rev-parse`` per leak. Read it with :func:`journal_lines`.
    """
    queue = _FakeQueue() if queue is None else queue
    records: list[dict[str, Any]] = []
    returns: list[str | None] = []

    monkeypatch.setattr(plan_tools, '_markup_project_root', lambda worktree: artifacts.worktree)
    monkeypatch.setattr(plan_tools, '_escalation_channel', lambda root: (Escalation, queue))

    real_factory = plan_tools._markup_escalation_sink

    def recording_factory(sink_artifacts: TaskArtifacts):
        sink = real_factory(sink_artifacts)

        # ASYNC, because the real sink is: it does its blocking work on a
        # worker thread so a git subprocess and two fsync'd queue writes never
        # sit on the server's event loop. A sync wrapper here would hand the
        # middleware a coroutine it awaits into `returns` as an un-awaited
        # object, so the wrapper has to keep the same shape as the thing it
        # wraps.
        async def wrapper(record: dict[str, Any]):
            records.append(record)
            result = await sink(record)
            returns.append(result)
            return result

        return wrapper

    monkeypatch.setattr(plan_tools, '_markup_escalation_sink', recording_factory)
    return ResidueRig(Harness(artifacts), queue, records, returns)


class TestTheSpecimenIsReal:
    """Non-circular pins on the corpus row these rows are driven by."""

    def test_it_is_a_measured_plan_tools_unrepairable_payload(self):
        assert UNREPAIRABLE_SPECIMEN['expected_outcome'] == 'unrepairable'
        assert UNREPAIRABLE_SPECIMEN['tool'] == 'mcp__plan-tools__add_design_decision'
        assert UNREPAIRABLE_SPECIMEN['param'] == 'decision'
        assert not UNREPAIRABLE_SPECIMEN['truncated']

    def test_it_is_still_unrepairable_against_this_tool_s_own_schema(self):
        """A corpus row is scored against the tool it was CAPTURED on.

        Re-derived here against ``add_design_decision``'s live parameter set,
        so a schema drift that made this payload repairable would fail loudly
        instead of quietly turning the residue rows into rejection rows.
        """
        assert detect(UNREPAIRABLE_DECISION) is not None
        assert repair(
            UNREPAIRABLE_DECISION, 'decision', _DECISION_SCHEMA, ('decision',)
        ) is None


class TestUnrepairableResidueIsPreserved:
    """C2: unrepairable input is never guessed — refused, and never discarded."""

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_escalation_holding_the_payload(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(a) The hint promises a named record; an unwired sink makes it a lie."""
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        payload = await rig.refuse(
            'add_design_decision', {'decision': UNREPAIRABLE_DECISION}
        )

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        assert payload['outcome'] == 'unrepairable'
        assert payload['tool'] == 'add_design_decision'
        assert payload['field'] == 'decision'
        assert 'repaired_call' not in payload, (
            'there is no repair — offering one would invite a retry that '
            're-sends a guess'
        )
        assert isinstance(payload['escalation_id'], str) and payload['escalation_id']

    @pytest.mark.asyncio
    async def test_the_residue_record_carries_the_caller_payload_verbatim(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(b) INV-7's contracted keys, and the only surviving copy of the data.

        ``matched_pattern`` is the SCAN's pattern, not a repair's. On the
        reject and forward paths the two coincide; here ``repair`` returned
        ``None``, so there is no ``Repair`` to read one off and the guard
        publishes what ``_first_markup_argument`` actually matched on --
        ``detect_for(value, param)`` since task 4696 widened it. Asserting
        the blanket ``detect`` here would name a literal that merely TRAILS
        the leak (PRD 2.2) and would disagree with the fact for the same
        event, which already carries the widened value.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': UNREPAIRABLE_DECISION})

        assert len(rig.records) == 1
        record = rig.records[0]
        assert record['error_type'] == 'mcp_markup_unrepairable'
        assert record['category'] == 'mcp_markup_residue'
        assert record['owner'] == 'l2-escalation-watcher'
        assert record['level'] == 2
        assert record['tool'] == 'add_design_decision'
        assert record['field'] == 'decision'
        assert record['matched_pattern'] == detect_for(UNREPAIRABLE_DECISION, 'decision')
        assert record['raw_value'] == UNREPAIRABLE_DECISION

    @pytest.mark.asyncio
    async def test_the_filed_escalation_holds_the_payload_byte_for_byte(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """The record is only preserved if it reaches the QUEUE intact.

        Asserted against the exact string sent — not a prefix, not an excerpt:
        after the refusal this is the only copy that exists anywhere.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        payload = await rig.refuse(
            'add_design_decision', {'decision': UNREPAIRABLE_DECISION}
        )

        assert len(rig.queue.submitted) == 1
        esc = rig.queue.submitted[0]
        assert esc.id == payload['escalation_id']
        assert esc.category == 'mcp_markup_residue'
        assert esc.level == 2
        assert esc.agent_role == 'plan-tools-markup-guard'
        assert esc.severity == 'blocking'
        assert UNREPAIRABLE_DECISION in esc.detail
        assert str(artifacts.worktree) == esc.worktree
        assert "owner='l2-escalation-watcher'" in esc.detail, (
            'the middleware declares owner as INV-7\'s machine-readable owner '
            'and Escalation has no field for it, so dropping it here would '
            'silently discard a declared contract field'
        )

    @pytest.mark.asyncio
    async def test_the_residue_never_gates_the_leaking_task(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """The record is filed under a NON-TASK anchor, and that is deliberate.

        The middleware declares residue at ``level=2``, and a PENDING level>=2
        escalation carrying a live task id is a stop-the-line event in this
        repo: ``workflow._is_gating_escalation`` gates on ``level >= 2``,
        ``_check_escalations`` looks records up by task id alone, and
        ``_wait_for_resolution`` raises ``_StewardReescalated`` — which
        ``run()`` turns into ``_mark_blocked``.

        So filing under the leaking task's own id would turn ONE leaked tool
        call into a human-gated task halt, and would contradict the refusal
        hint this same guard ships, which tells the caller to resend from its
        own copy and carry on. Same posture ``markup_tripwire`` takes with its
        own anchor.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': UNREPAIRABLE_DECISION})

        esc = rig.queue.submitted[0]
        assert rig.harness.plan()['task_id'] == 'test-1'
        assert esc.task_id == plan_tools._MARKUP_RESIDUE_ANCHOR_TASK_ID
        assert esc.task_id != 'test-1'
        assert esc.id.startswith(f'esc-{plan_tools._MARKUP_RESIDUE_ANCHOR_TASK_ID}')
        # The record is gating-SHAPED and stays that way: level 2 is the
        # middleware's own declaration (INV-7) and is not re-decided here. The
        # ANCHOR is therefore the only thing keeping it out of the running
        # task's gate, because `_check_escalations` scopes that gate by task id
        # alone — which is exactly what these two lookups model.
        assert _is_gating_escalation(esc) is True
        assert esc not in rig.queue.get_by_task('test-1', status='pending'), (
            'the leaking task must not find this record at its own gate'
        )
        assert esc in rig.queue.get_by_task(
            plan_tools._MARKUP_RESIDUE_ANCHOR_TASK_ID, status='pending'
        ), 'it must still be visible to the L2 watcher under the anchor'

    @pytest.mark.asyncio
    async def test_the_subject_task_comes_from_the_plan(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(c) The middleware's own identity fields are structurally None here.

        ``_identity`` reads ``agent_id`` / ``project_root`` / ``project_id`` off
        the call's arguments, and NO plan-tools tool declares any of the three
        — so the sink is the only party that can attribute the record. Since
        the queue's ``task_id`` is the non-gating anchor, that attribution has
        to survive in the summary and the detail instead.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': UNREPAIRABLE_DECISION})

        assert rig.harness.plan()['task_id'] == 'test-1'
        esc = rig.queue.submitted[0]
        assert esc.summary.startswith('[test-1] ')
        assert "subject_task_id='test-1'" in esc.detail

    @pytest.mark.asyncio
    async def test_the_subject_task_falls_back_to_the_worktree_name(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(c) The create_plan-rejected-before-any-plan-exists case."""
        rig = build_residue_rig(monkeypatch, artifacts)

        await rig.refuse(
            'create_plan',
            {
                'task_id': 'test-1',
                'title': UNREPAIRABLE_DECISION,
                'analysis': 'Clean analysis prose.',
                'files': ['a.py'],
            },
        )

        assert not rig.harness.plan_path.exists()
        esc = rig.queue.submitted[0]
        assert esc.summary.startswith(f'[{artifacts.worktree.name}] ')
        assert f'subject_task_id={artifacts.worktree.name!r}' in esc.detail

    @pytest.mark.asyncio
    async def test_the_subject_task_survives_an_unreadable_plan(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """An unreadable plan costs ATTRIBUTION, never the record.

        The payload is the only copy that exists, so a plan that cannot be
        parsed must degrade to the worktree-name fallback rather than take the
        filing down with it.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()
        monkeypatch.setattr(
            TaskArtifacts, 'read_plan',
            lambda self: (_ for _ in ()).throw(ValueError('plan.json is not JSON')),
        )

        payload = await rig.refuse(
            'add_design_decision', {'decision': UNREPAIRABLE_DECISION}
        )

        assert len(rig.queue.submitted) == 1
        esc = rig.queue.submitted[0]
        assert esc.id == payload['escalation_id']
        assert esc.summary.startswith(f'[{artifacts.worktree.name}] ')
        assert UNREPAIRABLE_DECISION in esc.detail

    @pytest.mark.asyncio
    async def test_a_queue_failure_never_changes_the_refusal(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(d) A queue outage costs visibility, never a working guard.

        The refusal is already decided before escalation is attempted, so every
        failure mode degrades to a logged ``None`` plus an unchanged payload.

        ``matched_pattern`` is the SCAN's pattern, not a repair's. On the
        reject and forward paths the two coincide; here ``repair`` returned
        ``None``, so there is no ``Repair`` to read one off and the guard
        publishes what ``_first_markup_argument`` actually matched on --
        ``detect_for(value, param)`` since task 4696 widened it. Asserting
        the blanket ``detect`` here would name a literal that merely TRAILS
        the leak (PRD 2.2) and would disagree with the fact for the same
        event, which already carries the widened value.
        """
        queue = _FakeQueue(submit_error=OSError('queue is unwritable'))
        rig = build_residue_rig(monkeypatch, artifacts, queue)
        await rig.harness.seed_plan()

        payload = await rig.refuse(
            'add_design_decision', {'decision': UNREPAIRABLE_DECISION}
        )

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        assert payload['matched_pattern'] == detect_for(
            UNREPAIRABLE_DECISION, 'decision'
        )
        assert payload['escalation_id'] is None, (
            'the caller is better told nothing than pointed at a record that '
            'was never written'
        )

    @pytest.mark.asyncio
    async def test_the_unrepairable_path_writes_nothing(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(e) Refusing is what makes the residue the ONLY copy — so it must hold."""
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()
        before = rig.harness.plan_bytes()

        await rig.refuse('add_design_decision', {'decision': UNREPAIRABLE_DECISION})

        assert rig.harness.plan_bytes() == before
        assert rig.harness.plan()['design_decisions'] == []


# ---------------------------------------------------------------------------
# The storm escape (INV-4) and its dedup.
# ---------------------------------------------------------------------------
#
# The middleware routes storm records through the SAME injected sink as residue
# records (``_file_storm_escalation`` hands it ``{'error_type':
# 'mcp_markup_storm', ...}``), so without a branch the sink would file a
# residue-shaped record — one claiming to hold a caller payload it does not
# have — for a burst.

#: A SECOND real unrepairable ``add_design_decision.decision`` payload, so
#: "residue records never dedup" is asserted across two DISTINCT caller
#: payloads rather than the same one twice.
SECOND_UNREPAIRABLE_SPECIMEN = _specimen('toolu_0183rUJqrD3dMjmWAx71mKmP')
SECOND_UNREPAIRABLE_DECISION: str = SECOND_UNREPAIRABLE_SPECIMEN['value']


class _Clock:
    """A hand-cranked time source, so a 3600s window costs no wall clock."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def tune_storm(rig: ResidueRig, *, threshold: int, clock: _Clock) -> MarkupGuardMiddleware:
    """Point the REGISTERED guard's burst detector at a fake clock.

    Not a reach-around: ``StormCounter``'s reload-safety contract has the
    middleware pass ``threshold`` and ``window_seconds`` PER record() call
    precisely so a consumer can read them live, and the counters themselves are
    created lazily on the first event — so tuning them on the constructed guard
    before any call is exactly as supported as reading them from a config leaf.
    The alternative is sleeping through the real 3600s window.
    """
    guard = next(
        m for m in rig.harness.server.middleware if isinstance(m, MarkupGuardMiddleware)
    )
    guard._storm_threshold = threshold
    guard._storm_time_provider = clock
    return guard


class TestTheStormEscape:
    """A burst means the upstream serialization leak is running RIGHT NOW."""

    @pytest.mark.asyncio
    async def test_a_burst_files_exactly_one_storm_escalation(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(a) One alarm per window, naming the outcome that actually burst."""
        rig = build_residue_rig(monkeypatch, artifacts)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(3):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        storms = rig.storm_records()
        assert len(storms) == 1, 'the per-window rate limit is what keeps it one'
        assert storms[0]['count'] == 2
        assert storms[0]['threshold'] == 2
        assert storms[0]['window_seconds'] == 3600.0
        assert storms[0]['outcome'] == 'rejected'

        assert len(rig.queue.submitted) == 1
        filed = rig.queue.submitted[0]
        assert filed.category == plan_tools._MARKUP_STORM_CATEGORY
        assert filed.task_id == plan_tools._MARKUP_STORM_ANCHOR_TASK_ID, (
            'the burst is a property of the SERVER window, not of one task '
            "payload — pinning the anchor is what makes the dedup lookup stable"
        )
        assert 'plans/toolcall-markup-containment-prd.md' in filed.detail

    @pytest.mark.asyncio
    async def test_a_second_burst_folds_into_the_still_open_record(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(b) A leak running for hours must not file one record per window."""
        rig = build_residue_rig(monkeypatch, artifacts)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(2):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})
        # Past the window, so the counter may fire again — while the record
        # filed by the first burst is still OPEN in the queue.
        clock.advance(4_000.0)
        for _ in range(2):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert len(rig.storm_records()) == 2, 'two bursts really did fire'
        assert len(rig.queue.submitted) == 1, 'but only one record was filed'
        assert rig.returns[-1] == rig.queue.submitted[0].id, (
            'the deduped burst reports the OPEN record, so the caller-facing '
            'trail still points somewhere real'
        )
        assert rig.queue.reads == [
            (plan_tools._MARKUP_STORM_ANCHOR_TASK_ID, 'pending')
        ] * 2

    @pytest.mark.asyncio
    async def test_residue_records_never_dedup(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(c) Each residue record is the only surviving copy of a DIFFERENT payload.

        Folding them would destroy one. They share a task_id, so an
        anchor-style dedup would have collapsed them.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': UNREPAIRABLE_DECISION})
        await rig.refuse('add_design_decision', {'decision': SECOND_UNREPAIRABLE_DECISION})

        assert len(rig.queue.submitted) == 2
        first, second = rig.queue.submitted
        assert first.id != second.id
        assert UNREPAIRABLE_DECISION in first.detail
        assert SECOND_UNREPAIRABLE_DECISION in second.detail
        assert rig.queue.reads == [], (
            'a residue record must not even LOOK for an open record to fold '
            'into — the lookup itself is what would lose a payload'
        )

    @pytest.mark.asyncio
    async def test_a_read_failure_falls_through_to_filing(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(d) Losing duplicate suppression beats losing the alarm."""
        queue = _FakeQueue(read_error=OSError('queue is unreadable'))
        rig = build_residue_rig(monkeypatch, artifacts, queue)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(2):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert len(rig.queue.submitted) == 1
        assert rig.queue.submitted[0].category == plan_tools._MARKUP_STORM_CATEGORY

    @pytest.mark.asyncio
    async def test_a_storm_filing_failure_never_changes_the_refusal(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """(e) The burst summary reaches the caller even when the queue is down.

        The middleware folds ``storm`` into the payload BEFORE the sink is
        consulted, so the one tier whose callers can learn of the burst is not
        also the tier a queue outage silences.
        """
        queue = _FakeQueue(submit_error=OSError('queue is unwritable'))
        rig = build_residue_rig(monkeypatch, artifacts, queue)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})
        payload = await rig.refuse(
            'add_design_decision', {'decision': ABSORBED_RATIONALE}
        )

        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['outcome'] == 'rejected'
        assert payload['storm']['count'] == 2
        assert payload['storm']['threshold'] == 2
        assert payload['storm']['window_seconds'] == 3600.0
        assert payload['storm']['outcome'] == 'rejected'
        assert rig.queue.submitted == []


# ---------------------------------------------------------------------------
# The sink's own degraded paths — where the record lands, and what happens
# when it cannot land there.
# ---------------------------------------------------------------------------
#
# Every row above patches ``_markup_project_root`` and ``_escalation_channel``
# away, because the record BUILDER is what those rows are about. These are the
# rows that exercise the seams themselves: a residue record is the only
# surviving copy of a caller's payload, so a sink that resolves the wrong
# project root, or gives up permanently after one transient git failure, loses
# data exactly where this leaf promises none is lost.


def _git(*args: str, cwd: Path) -> None:
    """Run one git command in *cwd*, failing loudly."""
    subprocess.run(
        ['git', *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    )


class TestWhereTheRecordLands:
    """``_markup_project_root`` decides which queue every record reaches."""

    def test_a_plain_checkout_resolves_to_its_own_root(self, tmp_path):
        """The relative-``.git`` branch: git answers ``.git``, not a path.

        Resolved against the worktree it was run in, which is what makes the
        branch correct rather than merely non-crashing.
        """
        checkout = tmp_path / 'repo'
        checkout.mkdir()
        _git('init', cwd=checkout)

        assert plan_tools._markup_project_root(checkout) == checkout.resolve()

    def test_a_linked_worktree_resolves_to_the_MAIN_checkout(self, tmp_path):
        """The load-bearing case, and the reason it is not ``--show-toplevel``.

        Every task runs in a linked worktree, and its records belong in the ONE
        queue the fleet's watchers read — the main checkout's. ``--show-toplevel``
        would answer with the lane itself, filing each record into a directory
        that is deleted when the lane is reset.
        """
        checkout = tmp_path / 'repo'
        checkout.mkdir()
        _git('init', cwd=checkout)
        _git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit',
             '--allow-empty', '-m', 'root', cwd=checkout)
        lane = tmp_path / 'lane'
        _git('worktree', 'add', str(lane), '-b', 'task/1', cwd=checkout)

        assert plan_tools._markup_project_root(lane) == checkout.resolve()
        assert plan_tools._markup_project_root(lane) != lane.resolve()

    def test_a_git_failure_degrades_to_None(self, tmp_path):
        """Outside any repository there is no answer — and no guess either."""
        assert plan_tools._markup_project_root(tmp_path) is None

    def test_an_empty_answer_degrades_to_None(self, monkeypatch, tmp_path):
        """A git that succeeds while naming nothing is still no answer.

        ``Path('').parent`` is ``Path('.')``, so without the emptiness check
        this branch would silently file every record under the CWD.
        """
        monkeypatch.setattr(
            plan_tools.subprocess, 'run',
            lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout='  \n', stderr=''),
        )

        assert plan_tools._markup_project_root(tmp_path) is None


class TestTheSinkNeverGivesUpPermanently:
    """A TRANSIENT failure must not disable residue preservation for good."""

    @pytest.mark.asyncio
    async def test_a_failed_project_root_is_retried_on_the_next_record(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """Memoising a FAILURE would lose every later payload on this server.

        ``_markup_project_root`` shells out to git, so it can fail for reasons
        that have nothing to do with this worktree — a fork failure under load,
        an EINTR, a transient timeout, an index.lock storm. Caching that answer
        for the life of the server would turn one such blip into permanent,
        silent data loss.

        The outage is modelled as a STATE that clears between the two records,
        rather than as a fixed list of answers to pop. That seam feeds BOTH
        injected channels (task 4744 routes the fact journal through it too), so
        a per-call script would silently encode how many consumers there happen
        to be — and would start failing for a reason that has nothing to do with
        the memoization this row is about.
        """
        queue = _FakeQueue()
        rig = build_residue_rig(monkeypatch, artifacts, queue)
        await rig.harness.seed_plan()

        git_is_down = True
        asked: list[Path] = []

        def flaky_root(worktree: Path) -> Path | None:
            asked.append(worktree)
            return None if git_is_down else artifacts.worktree

        monkeypatch.setattr(plan_tools, '_markup_project_root', flaky_root)

        first = await rig.refuse(
            'add_design_decision', {'decision': UNREPAIRABLE_DECISION}
        )
        asked_during_the_outage = len(asked)
        git_is_down = False
        second = await rig.refuse(
            'add_design_decision', {'decision': SECOND_UNREPAIRABLE_DECISION}
        )

        assert first['escalation_id'] is None
        assert rig.returns[0] is None
        assert asked_during_the_outage > 0, 'the first record really did ask'
        assert len(asked) > asked_during_the_outage, (
            'the second record must have RE-ASKED git — a cached failure would '
            'have skipped the call entirely and lost this payload'
        )
        assert isinstance(second['escalation_id'], str) and second['escalation_id']
        assert len(rig.queue.submitted) == 1
        assert SECOND_UNREPAIRABLE_DECISION in rig.queue.submitted[0].detail

    @pytest.mark.asyncio
    async def test_a_failed_channel_is_retried_on_the_next_record(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """Same argument one seam further in: opening the queue can also blip."""
        queue = _FakeQueue()
        rig = build_residue_rig(monkeypatch, artifacts, queue)
        await rig.harness.seed_plan()

        channels: list[Any] = [None, (Escalation, queue)]
        monkeypatch.setattr(
            plan_tools, '_escalation_channel', lambda root: channels.pop(0)
        )

        first = await rig.refuse(
            'add_design_decision', {'decision': UNREPAIRABLE_DECISION}
        )
        second = await rig.refuse(
            'add_design_decision', {'decision': SECOND_UNREPAIRABLE_DECISION}
        )

        assert first['escalation_id'] is None
        assert channels == [], 'the second record must have re-opened the queue'
        assert isinstance(second['escalation_id'], str) and second['escalation_id']
        assert len(rig.queue.submitted) == 1

    def test_an_unopenable_queue_degrades_to_None(self, monkeypatch, tmp_path):
        """``_escalation_channel``'s own failure branch, at the seam it guards.

        A queue whose directory cannot be opened costs the record, never the
        refusal — so the branch must return ``None`` rather than propagate.
        """
        import escalation.queue as escalation_queue

        monkeypatch.setattr(
            escalation_queue, 'EscalationQueue',
            lambda path: (_ for _ in ()).throw(OSError('read-only filesystem')),
        )

        assert plan_tools._escalation_channel(tmp_path) is None


class TestAnUnrecognisedRecordKindIsStillFiled:
    """Silently discarding a record kind is the fail-soft this PRD ends."""

    @pytest.mark.asyncio
    async def test_it_is_filed_under_the_fallback_vocabulary(
        self, monkeypatch, artifacts: TaskArtifacts
    ):
        """A kind the middleware grows LATER reaches an operator regardless.

        Driven straight at the sink because there is, by construction, no way
        to make today's middleware emit tomorrow's record. The record carries
        neither ``category`` nor ``summary`` — the shape a future emitter is
        least likely to get right — so both fallbacks are exercised.
        """
        queue = _FakeQueue()
        monkeypatch.setattr(
            plan_tools, '_markup_project_root', lambda worktree: artifacts.worktree
        )
        monkeypatch.setattr(
            plan_tools, '_escalation_channel', lambda root: (Escalation, queue)
        )
        sink = plan_tools._markup_escalation_sink(artifacts)

        esc_id = await sink({
            'error_type': 'mcp_markup_something_new',
            'tool': 'add_design_decision',
            'field': 'decision',
            'raw_value': UNREPAIRABLE_DECISION,
        })

        assert len(queue.submitted) == 1
        esc = queue.submitted[0]
        assert esc.id == esc_id
        assert esc.category == plan_tools._ESCALATION_FALLBACK_CATEGORY
        assert plan_tools._ESCALATION_FALLBACK_SUMMARY in esc.summary
        assert UNREPAIRABLE_DECISION in esc.detail, (
            'an unfamiliar record still carries a payload — filing it without '
            'that payload would discard the very thing worth keeping'
        )
        assert queue.reads == [], 'only the storm kind dedups'


# ---------------------------------------------------------------------------
# THE DURABLE JOURNAL — task 4744's user-observable signal.
# ---------------------------------------------------------------------------
#
# Measured 2026-08-25, while fulfilling a plan-tools storm escalation's OWN
# instruction ("identify the leaking caller from the guard's log lines"):
#
#     journalctl --user --since 2026-08-22 | grep 'markup guard:'  ->  0 lines
#
# against 35 REAL plan-tools rejections in data/orchestrator/agent-transcripts/
# over the same span. plan-tools is a per-agent stdio subprocess whose stderr
# the CLI agent that spawned it consumes, so the per-call fact — the only record
# anywhere that names WHICH call leaked — reached no durable sink at all. The
# instruction was unfollowable by construction, and anyone following it
# correctly concluded "no evidence" and was wrong.
#
# These rows are the inverse of that measurement, driven through the REAL
# server: after a rejection, the leaking task is nameable from a durable
# artifact with no transcript mining.


class TestTheRejectionReachesADurableJournal:
    """One line per EVENT, carrying the identity the storm summary cannot."""

    @pytest.mark.asyncio
    async def test_a_rejected_call_is_journalled_with_its_task_id(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(a) THE user-observable signal, on the measured leak shape."""
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        (line,) = journal_lines(tmp_path)
        assert line['tool'] == 'add_design_decision'
        assert line['param'] == 'decision'
        assert line['outcome'] == 'rejected'
        assert line['subject_task_id'] == 'test-1', (
            "the seeded plan's own task_id — this is what lets an operator name "
            'the leaking agent without mining agent transcripts'
        )
        assert line['server'] == 'plan-tools'

    @pytest.mark.asyncio
    async def test_the_unrepairable_outcome_is_journalled_too(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(b) All three outcomes, not just the ones that reach the queue.

        The escalation channel sees an unrepairable record and a burst summary;
        it never sees an ordinary REJECTED call, which is what the 35 measured
        rejections were. The fact channel sees all of them.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        await rig.refuse('add_design_decision', {'decision': UNREPAIRABLE_DECISION})

        (line,) = journal_lines(tmp_path)
        assert line['outcome'] == 'unrepairable'
        assert line['tool'] == 'add_design_decision'
        assert line['subject_task_id'] == 'test-1'

    @pytest.mark.asyncio
    async def test_a_leak_before_any_plan_exists_still_lands(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(c) A ``create_plan`` refused before there is a plan to attribute to.

        The attribution thunk has nothing to read, so it falls to the worktree
        directory name — which in the fleet IS the task's lane id. Losing the
        record instead would be the opposite of the point.
        """
        rig = build_residue_rig(monkeypatch, artifacts)

        await rig.refuse(
            'create_plan',
            {'task_id': 'test-1', 'title': ABSORBED_ANALYSIS, 'files': ['a.py']},
        )

        assert not rig.harness.plan_path.exists(), 'no plan to attribute against'
        (line,) = journal_lines(tmp_path)
        assert line['tool'] == 'create_plan'
        assert line['subject_task_id'] == artifacts.worktree.name
        assert line['subject_task_id'], 'never empty, never None'

    @pytest.mark.asyncio
    async def test_a_burst_is_three_journal_lines_and_one_escalation(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(d) The journal is per-EVENT; the escalation is per-WINDOW.

        This is the whole division of labour. A storm record can only ever say
        "N calls leaked in this window" — its own fields are count / threshold /
        window_seconds / outcome / project, and ``project`` is structurally None
        on this boundary. Which caller leaked is a per-event fact, and the
        journal is where it now lives.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(3):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        lines = journal_lines(tmp_path)
        assert len(lines) == 3, 'one line per rejection, not one per window'
        assert {line['subject_task_id'] for line in lines} == {'test-1'}
        assert len(rig.queue.submitted) == 1, 'still exactly one burst alarm'

    @pytest.mark.asyncio
    async def test_a_journal_outage_never_changes_a_refusal(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(f) The journal is ADDITIVE: the outcome is decided before it runs.

        Forced here by making the journal path an existing DIRECTORY, so the
        append cannot open it.
        """
        markup_journal.journal_path(tmp_path, 'plan-tools').mkdir(parents=True)
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()

        payload = await rig.refuse(
            'add_design_decision', {'decision': ABSORBED_RATIONALE}
        )

        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['outcome'] == 'rejected'
        assert payload['field'] == 'decision'
        assert payload['repaired_call']['rationale'] == _RATIONALE_PROSE

    def test_the_registered_guard_declares_a_fact_sink(self, harness: Harness):
        """(e) INV-1, at the axis this task adds.

        The fact channel had no consumer, and the comment at the registration
        site said so. It has one now, so the wiring is a DECLARATION at the
        registration site rather than a global logging side effect.
        """
        (guard,) = [
            m for m in harness.server.middleware if isinstance(m, MarkupGuardMiddleware)
        ]

        assert guard._fact_sink is not None, (
            'without it the per-call record naming the leaking caller reaches '
            "only this subprocess's stderr, which nobody retains"
        )

    @pytest.mark.asyncio
    async def test_a_rejection_still_writes_nothing_to_the_plan(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """The journal is additive in the other direction too.

        "Reject writes nothing" is the contract the whole policy rests on; a new
        write-side channel is exactly the kind of change that could quietly
        breach it.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        await rig.harness.seed_plan()
        before = rig.harness.plan_bytes()

        await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        assert rig.harness.plan_bytes() == before
        assert len(journal_lines(tmp_path)) == 1, 'the record went to the journal'


# ---------------------------------------------------------------------------
# The storm record POINTS AT the journal (task 4744).
# ---------------------------------------------------------------------------


class TestTheStormRecordNamesTheJournal:
    """A durable artifact an operator cannot FIND is not durable.

    The storm escalation used to close with "identify the leaking caller from
    the guard's own log lines (grep the orchestrator logs for 'markup guard:')".
    On this boundary that instruction is unfollowable by construction — the
    lines go to a per-agent subprocess's stderr — so a correct reader concluded
    "no evidence" and was wrong. Wiring the journal without repointing the
    record would move the dead end rather than close it.
    """

    @pytest.mark.asyncio
    async def test_the_detail_names_the_journal_not_the_logs(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(a) The record tells the reader where the answer actually is."""
        rig = build_residue_rig(monkeypatch, artifacts)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(2):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        (filed,) = rig.queue.submitted
        assert f'{markup_journal.MARKUP_JOURNAL_DIRNAME}/plan-tools.jsonl' in filed.detail
        assert 'orchestrator logs' not in filed.detail, (
            'the instruction this task measured to be unfollowable must be '
            'RETIRED, not merely supplemented'
        )
        assert 'plans/toolcall-markup-containment-prd.md' in filed.detail, (
            'the standing PRD pointer stays'
        )

    @pytest.mark.asyncio
    async def test_the_suggested_action_names_the_journal(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(b) The one line an operator reads first."""
        rig = build_residue_rig(monkeypatch, artifacts)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(2):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        (filed,) = rig.queue.submitted
        assert f'{markup_journal.MARKUP_JOURNAL_DIRNAME}/plan-tools.jsonl' in (
            filed.suggested_action
        )
        assert "guard's log lines" not in filed.suggested_action

    @pytest.mark.asyncio
    async def test_following_the_records_own_instruction_now_succeeds(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path
    ):
        """(c) The direct inverse of the measurement that opened this task.

        The lines the burst was made of are actually THERE, one per rejection in
        that window, each naming the leaking task. Asserting the record's prose
        without this would pin an instruction that is merely better-worded.
        """
        rig = build_residue_rig(monkeypatch, artifacts)
        clock = _Clock()
        tune_storm(rig, threshold=2, clock=clock)
        await rig.harness.seed_plan()

        for _ in range(2):
            await rig.refuse('add_design_decision', {'decision': ABSORBED_RATIONALE})

        (filed,) = rig.queue.submitted
        named = markup_journal.journal_path(tmp_path, 'plan-tools')
        assert str(named.relative_to(tmp_path)) in filed.detail.replace('\\', '/')
        assert named.exists(), 'the record names a file that is really there'

        lines = journal_lines(tmp_path)
        assert len(lines) == 2, 'one line per rejection in the window'
        assert {line['subject_task_id'] for line in lines} == {'test-1'}
        assert {line['outcome'] for line in lines} == {'rejected'}
