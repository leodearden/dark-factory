"""MarkupGuardMiddleware registration on the REAL verdict-tools server (task 3690).

PRD ``plans/toolcall-markup-containment-prd.md`` section 4 contract C2, the
other half of the FORWARD_REPAIR pair. The escalation server's leaks were
SILENT — its swallowed parameters are optional, so the call landed with
``evidence=[]`` and nobody noticed. This server's were LOUD: ``submit_review_verdict``
declares FOUR required parameters, so an absorbed ``issues`` failed the call
outright. Nineteen such calls sit in the committed corpus.

Both shapes are the same bug and both strand something (INV-6): a lost
``submit_review_verdict`` strands a review gate.

TWO CONSTRAINTS THIS FILE ENCODES, both measured:

1. Every call goes through ``async with Client(server)``. Middleware is BYPASSED
   by ``tool.fn(...)``, ``await tool.run({...})`` and
   ``server._tool_manager.call_tool(...)``. The established idiom in
   ``orchestrator/tests/test_verdict_tools_server.py`` is ``await tool.run({...})``
   — a test written that way would pass while running none of the guard, so that
   file is deliberately NOT the template here.

2. Specimens come from the committed corpus and are keyed by ``tool_use_id``,
   never by index, so a corpus refresh cannot silently repoint an assertion at a
   different payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp.verdict_tools import _SINGLETON_ROLE_TOOLS, create_server

# ---------------------------------------------------------------------------
# The committed corpus.
# ---------------------------------------------------------------------------
#
# Read with a small module-local loader rather than by importing
# ``shared/tests/toolcall_markup_corpus_extract``: the orchestrator package
# cannot import another package's test tree. The format is one JSON object per
# line, which is the whole of what that helper does for a reader.

CORPUS_PATH = (
    Path(__file__).resolve().parents[2]
    / 'shared'
    / 'tests'
    / 'fixtures'
    / 'toolcall_markup_corpus.jsonl'
)


def load_corpus() -> list[dict[str, Any]]:
    records = []
    with CORPUS_PATH.open(encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    assert records, f'the committed corpus at {CORPUS_PATH} is empty'
    return records


def specimen(tool_use_id: str) -> dict[str, Any]:
    """The committed corpus record with this ``tool_use_id``."""
    for record in load_corpus():
        if record.get('tool_use_id') == tool_use_id:
            return record
    raise AssertionError(f'specimen {tool_use_id!r} is missing from {CORPUS_PATH}')


#: Recovers the REQUIRED list-typed ``issues`` — PRD boundary row B14's shape.
#: The parameter is absent from the wire entirely, so the call works only
#: because ``on_call_tool`` runs BEFORE pydantic validation.
ISSUES_SPECIMEN = 'toolu_013LntXLfTwEZG1akr93PVGc'

#: The role this file drives. Not in ``_SINGLETON_ROLE_TOOLS``, so
#: ``create_server`` takes the reviewer-panel branch and registers
#: ``submit_review_verdict``.
REVIEWER_ROLE = 'test_analyst'


@pytest.fixture()
def artifacts(tmp_path: Path) -> TaskArtifacts:
    """TaskArtifacts pointing at a temporary worktree.

    Four lines duplicated from ``test_verdict_tools_server.py`` because a
    pytest fixture is not importable across modules; promoting it to
    ``conftest.py`` would widen its blast radius across the whole orchestrator
    suite for two consumers.
    """
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def guard_of(server) -> MarkupGuardMiddleware:
    """The registered guard — asserting there is exactly one."""
    guards = [
        m for m in getattr(server, 'middleware', []) or []
        if isinstance(m, MarkupGuardMiddleware)
    ]
    assert len(guards) == 1, (
        f'expected exactly one MarkupGuardMiddleware on the verdict-tools '
        f'server, found {len(guards)}'
    )
    return guards[0]


# ---------------------------------------------------------------------------
# The B14 signal: a REQUIRED list-typed parameter, absorbed and restored.
# ---------------------------------------------------------------------------


class TestSubmitReviewVerdictForwardsRepaired:
    """A real leaked ``submit_review_verdict`` LANDS (INV-6).

    Today it fails twice over: no guard is registered, and even registered it
    would die on the list-typed coercion — ``issues`` is recovered as a
    verbatim ``str`` slice and pydantic rejects it with ``list_type``, so the
    tool body never runs and the review gate stays stranded.
    """

    async def _call(self, artifacts: TaskArtifacts):
        record = specimen(ISSUES_SPECIMEN)
        assert record['expected_outcome'] == 'repaired'
        assert record['expected_recovered'] == ['issues']
        # `issues` is ABSENT from the wire — the caller supplied only these
        # three, which is exactly what the specimen's own `supplied` records.
        assert sorted(record['supplied']) == ['reviewer', 'summary', 'verdict']

        server = create_server(artifacts, REVIEWER_ROLE)
        async with Client(server) as client:
            return await client.call_tool('submit_review_verdict', {
                'reviewer': REVIEWER_ROLE,
                'verdict': 'ISSUES_FOUND',
                record['param']: record['value'],
            })

    @pytest.mark.asyncio
    async def test_the_call_succeeds(self, artifacts: TaskArtifacts):
        """(a) The review gate is NOT stranded."""
        result = await self._call(artifacts)

        assert result.data['status'] == 'ok'
        assert result.data['role'] == REVIEWER_ROLE

    @pytest.mark.asyncio
    async def test_the_verdict_envelope_was_written(self, artifacts: TaskArtifacts):
        """(b) Not merely 'no error' — the artifact really landed."""
        await self._call(artifacts)

        envelope = artifacts.read_verdict(REVIEWER_ROLE)
        assert envelope is not None
        assert envelope['role'] == REVIEWER_ROLE

    @pytest.mark.asyncio
    async def test_the_recovered_issues_are_a_list_of_dicts(
        self, artifacts: TaskArtifacts
    ):
        """(b) THE row. A ``str`` here is the coercion gap, not a recovery."""
        await self._call(artifacts)

        envelope = artifacts.read_verdict(REVIEWER_ROLE)
        assert envelope is not None
        issues = envelope['verdict']['issues']

        assert isinstance(issues, list), (
            f'issues stored as {type(issues).__name__}, not a list'
        )
        assert issues, 'issues stored EMPTY: the payload was lost'
        assert all(isinstance(entry, dict) for entry in issues)

    @pytest.mark.asyncio
    async def test_the_issues_carry_their_real_content(self, artifacts: TaskArtifacts):
        """The recovery is the reviewer's real finding, not a placeholder."""
        await self._call(artifacts)

        envelope = artifacts.read_verdict(REVIEWER_ROLE)
        assert envelope is not None
        issues = envelope['verdict']['issues']

        assert any('suggested_fix' in entry for entry in issues)

    @pytest.mark.asyncio
    async def test_meta_reports_the_repair(self, artifacts: TaskArtifacts):
        """(c) NAMES only — the warning must not become a second copy."""
        result = await self._call(artifacts)

        assert result.meta is not None
        warning = result.meta['markup_repair']
        assert warning['outcome'] == 'repaired'
        assert warning['field'] == 'summary'
        assert 'issues' in warning['recovered_params']


# ---------------------------------------------------------------------------
# INV-1 and boundary row B15, on EVERY branch.
# ---------------------------------------------------------------------------


#: ``create_server`` has four registration branches. The guard must be attached
#: on the SHARED path so no branch can miss it — a per-branch registration is
#: exactly the silent gap this task exists to close, and it would be invisible
#: to a test that only ever built one role.
ALL_BRANCHES = (*sorted(_SINGLETON_ROLE_TOOLS), REVIEWER_ROLE)


class TestPolicyDeclaredAndStrictValidationOff:
    """INV-1 plus PRD boundary row B15, parameterised over all four branches."""

    @pytest.mark.parametrize('role', ALL_BRANCHES)
    def test_the_guard_is_registered(self, artifacts: TaskArtifacts, role: str):
        server = create_server(artifacts, role)

        assert guard_of(server) is not None

    @pytest.mark.parametrize('role', ALL_BRANCHES)
    def test_the_policy_is_forward_repair(self, artifacts: TaskArtifacts, role: str):
        """C2: a lost submit_review_verdict strands a review gate.

        A registration-time enum, never inferred per call from the shape of the
        damage or from a tool's name.
        """
        server = create_server(artifacts, role)

        assert guard_of(server).policy is RepairPolicy.FORWARD_REPAIR

    @pytest.mark.parametrize('role', ALL_BRANCHES)
    def test_nothing_is_exempt(self, artifacts: TaskArtifacts, role: str):
        """An exemption is a DECLARATION, so the empty set is asserted too.

        No tool on this server legitimately carries envelope literals as data.
        A future one would be named BARE (``submit_review_verdict``, never the
        agent-facing ``mcp__verdict-tools__submit_review_verdict`` spelling the
        specimen corpus records).
        """
        server = create_server(artifacts, role)

        assert guard_of(server).exempt_tools == frozenset()

    @pytest.mark.parametrize('role', ALL_BRANCHES)
    def test_strict_input_validation_stays_off(
        self, artifacts: TaskArtifacts, role: str
    ):
        """Row B15 — the one setting that disables this guard entirely.

        With it on the SDK jsonschema-validates BEFORE FastMCP's handler, the
        middleware chain is never entered, no ``markup_detected`` fact is
        emitted, and every required-parameter leak becomes silently
        unrepairable. Which is EVERY leak on this server: all four of
        ``submit_review_verdict``'s parameters are required.
        """
        server = create_server(artifacts, role)

        assert not getattr(server, 'strict_input_validation', False)

    def test_the_branches_are_all_of_them(self, artifacts: TaskArtifacts):
        """A guard on the parameterisation itself.

        If ``create_server`` grows a fifth branch, this fails rather than
        letting the new branch go unregistered and untested.
        """
        assert len(ALL_BRANCHES) == 4
        assert set(ALL_BRANCHES) - {REVIEWER_ROLE} == _SINGLETON_ROLE_TOOLS


# ---------------------------------------------------------------------------
# The regression pin: EVERY committed specimen, against the REAL server.
# ---------------------------------------------------------------------------

#: The corpus records tool names in the AGENT-FACING prefixed spelling; the
#: in-server name FastMCP dispatches on is the bare suffix.
VERDICT_TOOL = 'mcp__verdict-tools__submit_review_verdict'

REPLAY = [r for r in load_corpus() if r['tool'] == VERDICT_TOOL]

# A mis-typed filter would otherwise yield an empty, always-green
# parametrisation. Measured: 19 specimens, all `repaired`.
assert REPLAY, f'no specimens collected for {VERDICT_TOOL}'

#: Type-correct fillers for the OTHER arguments a specimen was sent with.
#: Replaying with the specimen's own ``supplied`` set is what makes the outcome
#: comparable: ``repair`` REFUSES a recovery whose name the caller already
#: supplied, so replaying with a smaller argument map would be a more permissive
#: call than the one that really happened.
FILLERS: dict[str, Any] = {
    'reviewer': REVIEWER_ROLE,
    'verdict': 'ISSUES_FOUND',
    'issues': [],
    'summary': '',
}


def replay_args(record: dict[str, Any]) -> dict[str, Any]:
    """The argument map this specimen really arrived with, damage included."""
    args = {name: FILLERS[name] for name in record['supplied']}
    args[record['param']] = record['value']
    return args


def replay_id(record: dict[str, Any]) -> str:
    """Name the parametrisation by ``tool_use_id`` — never by index."""
    return record['tool_use_id']


class TestCorpusReplayAgainstRealServer:
    """All 19 real leaked calls, replayed through the real verdict-tools server.

    Catches two things nothing before it can:

    (a) SCHEMA DRIFT. Each record carries the ``schema_params`` captured at
        extraction time, and the live tool has in fact MOVED since: every
        specimen's captured schema lists ``__unparsedToolInput``, which is not a
        parameter of ``submit_review_verdict`` at all. The real server is ground
        truth, and dropping that name is inert here — no specimen recovers it,
        so no outcome changes. (``repair`` validates recovered NAMES against the
        schema, so a specimen that DID target it would legitimately become
        unrepairable against the live tool, and that is the correct answer, not
        a reason to edit the committed fixture: it is task 3688's artefact and
        other suites replay it.)

    (b) BOTH TYPE SHAPES AT ONCE. 16 specimens recover the REQUIRED list-typed
        ``issues`` and one recovers the str-typed ``verdict``, so the
        schema-directed coercion is exercised across every real shape rather
        than the one the hand-picked specimen above covers. Two recover NOTHING
        while still repairing — a clean truncation with an empty tail — which is
        a distinct outcome from unrepairable and must stay one.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('record', REPLAY, ids=replay_id)
    async def test_specimen(self, artifacts: TaskArtifacts, record: dict[str, Any]):
        assert record['expected_outcome'] == 'repaired'
        assert '__unparsedToolInput' in record['schema_params']

        server = create_server(artifacts, REVIEWER_ROLE)
        async with Client(server) as client:
            result = await client.call_tool(
                'submit_review_verdict', replay_args(record)
            )

        # The review gate is NOT stranded (INV-6).
        assert result.data['status'] == 'ok'
        assert result.meta is not None
        warning = result.meta['markup_repair']
        assert warning['outcome'] == 'repaired'
        assert warning['recovered_params'] == sorted(record['expected_recovered'])

        envelope = artifacts.read_verdict(REVIEWER_ROLE)
        assert envelope is not None
        verdict = envelope['verdict']
        # The recovered values landed with their DECLARED types, not as the
        # verbatim str slices repair() hands back.
        if 'issues' in record['expected_recovered']:
            assert isinstance(verdict['issues'], list)
            assert all(isinstance(entry, dict) for entry in verdict['issues'])
        if 'verdict' in record['expected_recovered']:
            assert verdict['verdict'] in {'PASS', 'ISSUES_FOUND'}


# ---------------------------------------------------------------------------
# The unrepairable path — C2 L187 / INV-7 on THIS server.
# ---------------------------------------------------------------------------

#: The doubly-corrupted B5-class specimen (the same class as the on-disk
#: esc-3184-2 record). Borrowed from the ``escalate_info`` rows because the
#: corpus has NO unrepairable ``submit_review_verdict`` specimen — MEASURED: all
#: 19 of them are ``repaired``.
#:
#: The reuse is legitimate because unrepairability is a property of the VALUE's
#: own boundary, not of the target tool's schema, and the test below ASSERTS the
#: observed outcome rather than assuming it — so the reuse is self-verifying if
#: the corpus or ``repair`` ever moves.
UNREPAIRABLE_SPECIMEN = 'toolu_012YjuXbKZAMwNAo9WR4Pvjx'


class TestUnrepairableResidueIsPreserved:
    """A refusal on THIS server must not destroy the payload it refuses.

    The state at risk here is worth MORE than the escalation server's, not
    less: a lost ``submit_review_verdict`` strands a review gate (INV-6) AND
    destroys a reviewer's entire ``issues`` findings list — by construction text
    the agent cannot re-emit identically.

    Fails today: ``escalation_sink`` is deliberately unwired on this server, so
    ``escalation_id`` comes back null, no residue exists anywhere on disk, and
    the caller is nonetheless told its payload is safely stored.
    """

    @staticmethod
    def _value() -> str:
        record = specimen(UNREPAIRABLE_SPECIMEN)
        assert record['expected_outcome'] == 'unrepairable'
        assert record['expected_recovered'] == []
        return record['value']

    async def _refuse(self, artifacts: TaskArtifacts) -> dict[str, Any]:
        """Drive the specimen as ``submit_review_verdict.summary``."""
        server = create_server(artifacts, REVIEWER_ROLE)
        async with Client(server) as client:
            with pytest.raises(ToolError) as excinfo:
                await client.call_tool('submit_review_verdict', {
                    'reviewer': REVIEWER_ROLE,
                    'verdict': 'ISSUES_FOUND',
                    'issues': [],
                    'summary': self._value(),
                })
        return json.loads(str(excinfo.value))

    @staticmethod
    def _residue_files(artifacts: TaskArtifacts) -> list[Path]:
        return sorted(artifacts.root.glob('markup_residue-*.json'))

    @pytest.mark.asyncio
    async def test_the_call_is_refused(self, artifacts: TaskArtifacts):
        """(a) The boundary is a GUESS, so nothing is forwarded.

        Also the self-verifying half of the cross-tool specimen reuse: the
        observed outcome is asserted, not assumed.
        """
        payload = await self._refuse(artifacts)

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        assert payload['outcome'] == 'unrepairable'
        assert payload['tool'] == 'submit_review_verdict'
        # No repaired_call: offering one would invite a retry re-sending a guess.
        assert 'repaired_call' not in payload

    @pytest.mark.asyncio
    async def test_nothing_partial_was_written(self, artifacts: TaskArtifacts):
        """(b) The tool body never ran, so no half-written verdict envelope
        carrying the corrupted ``summary`` exists. Mirrors the escalation
        suite's ``test_nothing_partial_was_written_for_the_caller``."""
        await self._refuse(artifacts)

        assert artifacts.read_verdict(REVIEWER_ROLE) is None

    @pytest.mark.asyncio
    async def test_the_residue_survives_in_full(self, artifacts: TaskArtifacts):
        """(c) "The entire issues payload destroyed with no surviving copy
        anywhere on disk" — made FALSE.

        Verbatim and entire, deliberately unlike ``build_markup_block``'s
        200-char excerpt: this is the only surviving copy.
        """
        value = self._value()
        await self._refuse(artifacts)

        files = self._residue_files(artifacts)
        assert len(files) == 1, f'expected exactly one residue file, got {files!r}'
        stored = json.loads(files[0].read_text())
        assert stored['raw_value'] == value
        assert len(stored['raw_value']) == len(value) == 3525

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_residue(self, artifacts: TaskArtifacts):
        """(d) A bounced reviewer must be able to point an operator at its own
        preserved data — an id it cannot look up is no better than none."""
        payload = await self._refuse(artifacts)

        assert isinstance(payload['escalation_id'], str)
        assert (artifacts.root / payload['escalation_id']).is_file()

    @pytest.mark.asyncio
    async def test_the_residue_carries_its_routing_fields(
        self, artifacts: TaskArtifacts
    ):
        """(e) INV-7: a machine-readable owner plus the standing L2 bound, and
        the flat fields locating the leak."""
        await self._refuse(artifacts)

        stored = json.loads(self._residue_files(artifacts)[0].read_text())
        assert stored['category'] == 'mcp_markup_residue'
        assert stored['owner'] == 'l2-escalation-watcher'
        assert stored['level'] == 2
        assert stored['tool'] == 'submit_review_verdict'
        assert stored['field'] == 'summary'
        assert stored['matched_pattern']

    @pytest.mark.asyncio
    async def test_the_hint_makes_the_preservation_claim(
        self, artifacts: TaskArtifacts
    ):
        """(f) On this server preservation now ACTUALLY happened, so the
        preserved variant of the hint is the true one."""
        payload = await self._refuse(artifacts)

        assert 'preserved verbatim' in payload['hint']

    @pytest.mark.asyncio
    async def test_the_hint_tells_the_truth_when_nothing_could_be_written(
        self, tmp_path: Path
    ):
        """(f, negative) The guarantee is "the prose matches the id", NOT "the
        prose is always optimistic".

        With the artifacts root removed the residue write cannot land, and the
        caller must be told so rather than sent after a file that never
        existed.
        """
        import shutil

        worktree = tmp_path / 'vanishing'
        worktree.mkdir()
        gone = TaskArtifacts(worktree)
        gone.init('test-1', 'Test task', 'A test')
        shutil.rmtree(worktree)

        payload = await self._refuse(gone)

        assert payload['escalation_id'] is None
        assert 'preserved verbatim' not in payload['hint']
        assert 'nothing was preserved' in payload['hint'].lower()

    @pytest.mark.asyncio
    async def test_the_fact_still_fires(self, artifacts: TaskArtifacts, caplog):
        """(g) INV-2: EVERY outcome emits ``markup_detected``, including this
        one. A refusal that emitted no fact would be invisible to the storm
        counter and to any consumer watching the leak rate."""
        import logging

        with caplog.at_level(logging.INFO, logger='orchestrator.mcp.verdict_tools'):
            await self._refuse(artifacts)

        facts = [
            json.loads(rec.getMessage().split(' ', 1)[1])
            for rec in caplog.records
            if rec.getMessage().startswith('markup_detected ')
        ]
        assert len(facts) == 1, f'expected exactly one fact, got {facts!r}'
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['tool'] == 'submit_review_verdict'
