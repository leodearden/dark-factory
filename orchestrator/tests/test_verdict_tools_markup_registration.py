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
