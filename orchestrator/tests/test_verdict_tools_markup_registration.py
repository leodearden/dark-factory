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
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import markup_journal, verdict_tools
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


def journal_lines(root: Path) -> list[dict[str, Any]]:
    """Every verdict-tools markup fact journalled under *root*, parsed.

    An absent file reads as no lines rather than raising: "the journal was
    never written" is an assertable outcome here, not an error.

    The path comes from :func:`markup_journal.journal_path` rather than being
    reconstructed, so a row cannot pass against a path the production wiring
    does not write. Copied from ``test_plan_tools_markup_guard.py`` with the
    server label changed — which sits in THIS directory, so the copy is a
    choice and not a constraint (the header's not-importable note is about
    ``shared/tests``, genuinely another package, and does not apply here). The
    honest reason is the one the ``artifacts`` fixture above already gives for
    its own duplication: promoting a helper to ``conftest.py`` widens its blast
    radius across the whole orchestrator suite, which two consumers do not yet
    justify. Both copies read the format ``markup_journal`` writes and
    ``test_markup_journal.py`` pins, so neither can drift alone without that
    file going red first.
    """
    path = markup_journal.journal_path(root, 'verdict-tools')
    if not path.exists():
        return []
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# The B14 signal: a REQUIRED list-typed parameter, absorbed and restored.
# ---------------------------------------------------------------------------


def seed_plan(artifacts: TaskArtifacts) -> None:
    """Write the plan a verdict is ABOUT — which is what attribution reads.

    ``verdict_tools._markup_subject_task_id`` reads ``plan.json``'s own
    ``task_id``, NOT ``metadata.json``'s, and in the fleet there always is one
    by the time a verdict is submitted: the verdict is about that plan's diff.
    The ``artifacts`` fixture writes only metadata (``TaskArtifacts.init``), so
    a row that wants the real attribution answer rather than
    ``markup_sink.resolve_subject``'s worktree-name fallback seeds it here.
    """
    artifacts.write_plan({
        'task_id': 'test-1',
        'title': 'Test task',
        'analysis': 'A test',
        'prerequisites': [],
        'steps': [],
    })


async def repaired_call(artifacts: TaskArtifacts):
    """Drive the ``ISSUES_SPECIMEN`` leak through the REAL server, repaired.

    Module level rather than a method because two suites need it: the B14
    recovery rows below, and the durable-journal rows, which want the one
    outcome that never touches the escalation channel (task 4917).
    """
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


class TestSubmitReviewVerdictForwardsRepaired:
    """A real leaked ``submit_review_verdict`` LANDS (INV-6).

    Today it fails twice over: no guard is registered, and even registered it
    would die on the list-typed coercion — ``issues`` is recovered as a
    verbatim ``str`` slice and pydantic rejects it with ``list_type``, so the
    tool body never runs and the review gate stays stranded.
    """

    async def _call(self, artifacts: TaskArtifacts):
        return await repaired_call(artifacts)

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
    async def test_the_fact_still_fires(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """(g) INV-2: EVERY outcome emits ``markup_detected``, including this
        one. A refusal that emitted no fact would be invisible to the storm
        counter and to any consumer watching the leak rate.

        READ OFF THE JOURNAL, not off ``caplog``. This row used to parse the
        ``orchestrator.mcp.verdict_tools`` logger, because the fact channel WAS
        a ``logger.info`` — a line that reached only a per-agent stdio
        subprocess's stderr. Task 4917 retires that emitter for the durable
        journal, so the channel this row reads changes while its guarantee does
        not: the fact must still fire on the outcome nothing else records
        per-event.

        Patching ``_markup_project_root`` steers the escalation sink to
        ``tmp_path`` as well, so this row's residue filing opens a real
        ``EscalationQueue`` under the temp tree. That is hermetic, and it is
        what makes this the ONE outcome where BOTH channels fire — so it is
        also the only place the shared-ladder guarantee can be pinned end to
        end. ``markup_sink.resolve_subject`` is deliberately one function
        rather than a copy per channel, on the stated grounds that a guard
        whose escalation and whose journal disagree about who leaked is worse
        than either alone; the two assertions at the bottom are what would
        catch that disagreement. The residue FLOOR (what happens when the queue
        cannot be opened at all) stays the business of the UNPATCHED rows
        above.
        """
        monkeypatch.setattr(
            verdict_tools, '_markup_project_root', lambda worktree: tmp_path,
        )
        seed_plan(artifacts)

        await self._refuse(artifacts)

        facts = journal_lines(tmp_path)
        assert len(facts) == 1, f'expected exactly one fact, got {facts!r}'
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['tool'] == 'submit_review_verdict'
        assert facts[0]['subject_task_id'] == 'test-1', (
            "the seeded plan's own task_id — the refused call is attributed "
            'exactly as a repaired one is, on the outcome where the caller is '
            'bounced and the journal line is the only per-event record left'
        )
        # ...and the residue record filed on this same outcome names the SAME
        # subject, which is what one shared ladder buys and two copies would
        # eventually stop buying.
        filed = sorted((tmp_path / 'data' / 'escalations').rglob('esc-*.json'))
        assert len(filed) == 1, f'expected exactly one residue record, got {filed!r}'
        record = json.loads(filed[0].read_text(encoding='utf-8'))
        assert record['summary'].startswith('[test-1] '), (
            f'the journal says test-1 leaked and the escalation says '
            f'{record["summary"]!r} — one shared attribution ladder is the '
            f'whole reason those cannot disagree'
        )


# ---------------------------------------------------------------------------
# The residue channel is the SHARED one (task 3690 review follow-up)
# ---------------------------------------------------------------------------


class TestVerdictToolsResidueChannelIsShared:
    """The sink files into the real escalation queue, not a doomed worktree file.

    The first cut of this leaf wired ``escalation_sink`` straight to
    ``TaskArtifacts.write_markup_residue``, which lands under the task
    worktree — a root the orchestrator destroys with
    ``git worktree remove --force`` at teardown, and which no watcher ever
    reads. These pin the corrected wiring: the queued escalation is the primary
    channel, and the worktree file is only the floor beneath a queue that
    cannot be opened.
    """

    def test_spec_declares_verdict_tools_own_anchors(self):
        """Not plan-tools' anchors, and not the middleware's defaults."""
        from orchestrator.mcp import plan_tools, verdict_tools

        spec = verdict_tools._MARKUP_SINK_SPEC
        assert spec.server_label == 'verdict-tools'
        assert spec.residue_anchor_task_id == 'verdict-tools-markup-residue'
        assert spec.storm_anchor_task_id == 'verdict-tools-markup-storm'
        # Distinct from plan-tools', so a reader can tell whose boundary spoke,
        # and so one server's open storm record cannot dedup the other's away.
        assert spec.residue_anchor_task_id != plan_tools._MARKUP_RESIDUE_ANCHOR_TASK_ID
        assert spec.storm_anchor_task_id != plan_tools._MARKUP_STORM_ANCHOR_TASK_ID
        # A NON-TASK anchor: at the level 2 the middleware declares, a pending
        # record under a live task id would halt the task whose verdict leaked.
        assert not spec.residue_anchor_task_id.isdigit()

    def test_spec_declares_where_ITS_callers_can_be_identified(self):
        """Task 4917: ``attribution_source`` is verdict-tools' OWN answer, and
        it now names verdict-tools' OWN journal.

        THIS ROW'S PREMISE EXPIRED, and the inversion is deliberate rather than
        a weakening. It used to assert that the journal directory must NOT
        appear here, on the written rationale that "naming a journal this
        boundary does not write would send an operator to an empty or missing
        file" — correct while the fact channel was a ``logger.info`` nobody
        retains, and false the moment task 4917 wired the journal. The
        GUARANTEE is unchanged: this boundary states its own answer, naming ITS
        file, rather than inheriting plan-tools'.
        """
        from orchestrator.mcp import plan_tools

        spec = verdict_tools._MARKUP_SINK_SPEC
        assert (
            f'{markup_journal.MARKUP_JOURNAL_DIRNAME}/verdict-tools.jsonl'
            in spec.attribution_source
        )
        assert 'data/orchestrator/agent-transcripts' not in spec.attribution_source, (
            'the transcript-mining instruction is RETIRED, not merely '
            'supplemented — leaving it would keep sending an operator down the '
            'expensive route when a one-line grep now answers the question'
        )
        assert 'orchestrator logs' not in spec.attribution_source, (
            'the unfollowable grep-the-logs sentence task 4744 retired must '
            'not come back'
        )
        assert spec.attribution_source != plan_tools._MARKUP_SINK_SPEC.attribution_source

    def test_the_field_is_required_so_a_new_server_must_decide(self):
        """A DEFAULT is what would let the next server inherit silently.

        Every other field on the spec is required for this reason; this one is
        the axis where inheriting plan-tools' answer is actively harmful, since
        the answer is a filesystem path only plan-tools writes.
        """
        from orchestrator.mcp import markup_sink

        with pytest.raises(TypeError):
            markup_sink.MarkupSinkSpec(  # type: ignore[call-arg]
                server_label='new-server',
                agent_role='new-server-markup-guard',
                residue_anchor_task_id='new-server-markup-residue',
                storm_anchor_task_id='new-server-markup-storm',
                refusal_consequence='Nothing was written.',
                storm_consequence='the leak keeps landing.',
            )

    @pytest.mark.asyncio
    async def test_residue_is_queued_as_an_escalation_not_dropped_in_the_worktree(
        self, tmp_path,
    ):
        """The primary channel is the queue, and the fallback is NOT taken."""
        from orchestrator.mcp import markup_sink, verdict_tools

        submitted: list[Any] = []
        fell_back: list[dict[str, Any]] = []

        class _Queue:
            def make_id(self, anchor: str) -> str:
                return f'esc-{anchor}-1'

            def get_by_task(self, anchor: str, status: str = '') -> list[Any]:
                return []

            def submit(self, esc: Any) -> str:
                submitted.append(esc)
                return esc.id

        from escalation.models import Escalation

        sink = markup_sink.make_escalation_sink(
            worktree=tmp_path,
            spec=verdict_tools._MARKUP_SINK_SPEC,
            subject_task_id=lambda: '3690',
            resolve_root=lambda worktree: tmp_path,
            open_channel=lambda root: (Escalation, _Queue()),
            last_resort=lambda record: fell_back.append(record) or 'residue.json',
        )
        record = {
            'error_type': markup_sink.MARKUP_RESIDUE_ERROR_TYPE,
            'tool': 'submit_review_verdict',
            'field': 'issues',
            'category': 'mcp_markup_residue',
            'summary': 'unrepairable markup',
            'owner': 'l2-escalation-watcher',
            'level': 2,
            'raw_value': 'THE ONLY SURVIVING COPY',
        }
        esc_id = await sink(record)

        assert esc_id == 'esc-verdict-tools-markup-residue-1'
        assert not fell_back, (
            'the worktree-local writer is the LAST RESORT — taking it while '
            'the queue is healthy is the doomed-storage bug this test pins'
        )
        assert len(submitted) == 1
        filed = submitted[0]
        assert filed.task_id == 'verdict-tools-markup-residue'
        assert filed.agent_role == 'verdict-tools-markup-guard'
        # INV-7: the middleware owns the vocabulary; the sink re-decides none of it.
        assert filed.category == 'mcp_markup_residue'
        assert filed.level == 2
        # The subject rides in the summary, because the task_id is the anchor.
        assert filed.summary.startswith('[3690] ')
        # ...and the payload survives verbatim, which is the whole point.
        assert 'THE ONLY SURVIVING COPY' in filed.detail
        assert "owner='l2-escalation-watcher'" in filed.detail
        assert 'verdict-tools' in filed.detail

    @pytest.mark.asyncio
    async def test_worktree_file_is_the_floor_when_the_queue_cannot_be_opened(
        self, tmp_path,
    ):
        """Strictly better than losing the payload; strictly worse than a queue."""
        from orchestrator.artifacts import TaskArtifacts
        from orchestrator.mcp import markup_sink, verdict_tools

        artifacts = TaskArtifacts(tmp_path)
        artifacts.root.mkdir(parents=True, exist_ok=True)
        sink = markup_sink.make_escalation_sink(
            worktree=tmp_path,
            spec=verdict_tools._MARKUP_SINK_SPEC,
            subject_task_id=lambda: '3690',
            resolve_root=lambda worktree: tmp_path,
            open_channel=lambda root: None,
            last_resort=artifacts.write_markup_residue,
        )
        locator = await sink({
            'error_type': markup_sink.MARKUP_RESIDUE_ERROR_TYPE,
            'tool': 'submit_review_verdict',
            'field': 'issues',
            'raw_value': 'THE ONLY SURVIVING COPY',
        })
        assert locator, 'a queue outage must not silently destroy the payload'
        # A bare filename, resolved against the artifacts root it was written to.
        assert 'THE ONLY SURVIVING COPY' in (artifacts.root / locator).read_text()

    @pytest.mark.asyncio
    async def test_a_sink_that_can_reach_nothing_returns_none_rather_than_lying(
        self, tmp_path,
    ):
        """The refusal hint must never promise a preservation that did not happen."""
        from orchestrator.mcp import markup_sink, verdict_tools

        sink = markup_sink.make_escalation_sink(
            worktree=tmp_path,
            spec=verdict_tools._MARKUP_SINK_SPEC,
            subject_task_id=lambda: '3690',
            resolve_root=lambda worktree: None,
            open_channel=lambda root: None,
            last_resort=None,
        )
        assert await sink({'error_type': 'mcp_markup_unrepairable'}) is None


# ---------------------------------------------------------------------------
# The fact reaches a DURABLE journal (task 4917).
# ---------------------------------------------------------------------------


class TestTheVerdictFactReachesADurableJournal:
    """One line per EVENT, carrying the identity the storm summary cannot.

    THE HEADLINE ROW HERE IS THE REPAIRED ONE, which is what makes this
    boundary different from plan-tools'. verdict-tools declares FORWARD_REPAIR,
    so a repaired call SUCCEEDS: the tool body runs, the verdict lands, the
    caller is never bounced, and ``escalation_sink`` is never consulted at all
    (it sees only unrepairable residue and window storms). The only
    caller-visible trace is a ``meta['markup_repair']`` block on a response
    nobody retains. So on this boundary the journal is not merely the BEST
    durable record of a repair — before task 4917 there was no other.

    It is also the cleanest rig: because the repaired path never touches the
    escalation channel, these rows need no fake queue and no clock.

    Every row steers the project root through the ``_markup_project_root``
    seam. That indirection is load-bearing, not convenience:
    ``make_fact_journal``'s default resolver is bound at module-DEFINITION
    time, so patching ``markup_sink.resolve_project_root`` afterwards would not
    reach a server ``create_server`` has already built — the journal would run
    a real ``git rev-parse`` against a bare ``tmp_path``, resolve nothing, and
    write no line whether or not the wiring landed.
    """

    @staticmethod
    def _steer(monkeypatch, tmp_path: Path) -> None:
        """Point BOTH injected channels at *tmp_path*, failing if the seam went.

        ``raising`` is left at its default TRUE deliberately. The step-1 rows
        that first drove this passed it as False because the seam did not exist
        yet; once the seam landed, that flag became the thing DISABLING the
        only check that it still does. With it, a renamed or inlined
        ``_markup_project_root`` would leave monkeypatch quietly creating an
        unused attribute — and while most rows here would then fail loudly (an
        empty journal), ``test_a_journal_outage_never_changes_the_outcome``
        would pass VACUOUSLY: its assertions are all that the call succeeded,
        which is equally true when the journal was never steered at the
        directory collision it means to force.
        ``test_plan_tools_markup_guard.py`` patches the identical seam the same
        way.
        """
        monkeypatch.setattr(
            verdict_tools, '_markup_project_root', lambda worktree: tmp_path,
        )

    @pytest.mark.asyncio
    async def test_a_repaired_call_is_journalled_with_its_task_id(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """(a) THE user-observable signal, on the measured leak shape."""
        self._steer(monkeypatch, tmp_path)
        seed_plan(artifacts)

        await repaired_call(artifacts)

        (line,) = journal_lines(tmp_path)
        assert line['tool'] == 'submit_review_verdict'
        assert line['param'] == 'summary'
        assert line['outcome'] == 'repaired'
        assert line['server'] == 'verdict-tools'
        assert line['subject_task_id'] == 'test-1', (
            "the seeded plan's own task_id — this is what lets an operator "
            'name the leaking agent without mining agent transcripts'
        )
        assert 'issues' in line['recovered_params']
        assert datetime.fromisoformat(line['ts'])

    @pytest.mark.asyncio
    async def test_the_journal_is_the_only_durable_trace_of_a_repair(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """(b) The boundary-specific row: FORWARD_REPAIR leaves nothing else.

        The call is not bounced, so no refusal payload carries the repair; the
        escalation channel is not consulted, so no residue record does either.
        Only the journal knows this happened at all.
        """
        self._steer(monkeypatch, tmp_path)

        result = await repaired_call(artifacts)

        assert result.data['status'] == 'ok'
        assert artifacts.read_verdict(REVIEWER_ROLE) is not None
        assert sorted(artifacts.root.glob('markup_residue-*.json')) == [], (
            'a repaired call never reaches the residue channel'
        )
        assert len(journal_lines(tmp_path)) == 1

    @pytest.mark.asyncio
    async def test_a_burst_is_one_line_per_event(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """(c) The journal is per-EVENT; the storm escalation is per-WINDOW.

        This is the whole division of labour. A storm record can only ever say
        "N calls leaked in this window" — its own fields are count / threshold
        / window_seconds / outcome / project, and ``project`` is structurally
        None on this boundary. WHICH caller leaked is a per-event fact.
        """
        self._steer(monkeypatch, tmp_path)
        seed_plan(artifacts)

        for _ in range(3):
            await repaired_call(artifacts)

        lines = journal_lines(tmp_path)
        assert len(lines) == 3, 'one line per repair, not one per window'
        assert {line['subject_task_id'] for line in lines} == {'test-1'}

    @pytest.mark.asyncio
    async def test_a_journal_outage_never_changes_the_outcome(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """(d) The journal is ADDITIVE: the outcome is decided before it runs.

        Forced here by making the journal path an existing DIRECTORY, so the
        append cannot open it. A review gate must not strand because a
        record-keeping file could not be written.
        """
        markup_journal.journal_path(tmp_path, 'verdict-tools').mkdir(parents=True)
        self._steer(monkeypatch, tmp_path)

        result = await repaired_call(artifacts)

        assert result.data['status'] == 'ok'
        assert artifacts.read_verdict(REVIEWER_ROLE) is not None
        assert result.meta is not None
        assert result.meta['markup_repair']['outcome'] == 'repaired'


# ---------------------------------------------------------------------------
# The storm record POINTS AT the journal (task 4917).
# ---------------------------------------------------------------------------


class TestTheStormRecordNamesTheJournal:
    """A durable artifact an operator cannot FIND is not durable.

    ``MarkupSinkSpec.attribution_source`` is the one string rendered into the
    burst alarm's body (``markup_sink.storm_detail``) and into its
    ``suggested_action``. Wiring the journal without repointing that string
    would move the dead end rather than close it: the record would still send a
    reader to ``data/orchestrator/agent-transcripts/`` to mine by hand for an
    answer that is now one grep away.
    """

    @staticmethod
    def _storm_detail() -> str:
        """The REAL rendered record, not the spec field read in isolation."""
        from orchestrator.mcp import markup_sink

        return markup_sink.storm_detail(
            {
                'count': 3,
                'threshold': 3,
                'window_seconds': 3600,
                'outcome': 'repaired',
                'project': None,
            },
            'test-1',
            verdict_tools._MARKUP_SINK_SPEC,
        )

    def test_the_storm_detail_names_the_journal(self):
        """(b) The body an operator reads, rendered through the real helper."""
        detail = self._storm_detail()

        assert f'{markup_journal.MARKUP_JOURNAL_DIRNAME}/verdict-tools.jsonl' in detail
        assert 'orchestrator logs' not in detail, (
            'the instruction task 4744 measured to be unfollowable must be '
            'RETIRED, not merely supplemented'
        )
        assert 'plans/toolcall-markup-containment-prd.md' in detail, (
            'the standing PRD pointer stays'
        )

    @pytest.mark.asyncio
    async def test_following_the_records_own_instruction_now_succeeds(
        self, monkeypatch, artifacts: TaskArtifacts, tmp_path: Path
    ):
        """(c) The end-to-end row, and the only one that catches the two halves
        drifting apart.

        Asserting the record's prose alone would pin an instruction that is
        merely better-worded. So this one FOLLOWS it: pull the path the record
        names out of its own body, open that exact file, and read the line.
        """
        monkeypatch.setattr(
            verdict_tools, '_markup_project_root', lambda worktree: tmp_path,
        )
        seed_plan(artifacts)

        await repaired_call(artifacts)

        named = [tok for tok in self._storm_detail().split() if tok.endswith('.jsonl')]
        assert len(named) == 1, (
            f'the record must name exactly one journal to open, got {named!r}'
        )
        path = tmp_path / named[0]
        assert path == markup_journal.journal_path(tmp_path, 'verdict-tools'), (
            'the instruction and the artifact must be the same path, which is '
            'why both are composed from MARKUP_JOURNAL_DIRNAME'
        )
        assert path.is_file(), (
            f'the record sends an operator to {named[0]}, which does not exist'
        )
        (line,) = [
            json.loads(entry)
            for entry in path.read_text(encoding='utf-8').splitlines()
            if entry.strip()
        ]
        assert line['subject_task_id'] == 'test-1'
