"""MarkupGuardMiddleware registration on the REAL escalation server (task 3690).

PRD ``plans/toolcall-markup-containment-prd.md`` section 4 contract C2. Task
3689 built the guard and proved it against toy servers; this file proves it is
actually ATTACHED to the escalation server and that a real leaked call now
lands rather than losing its arguments silently.

THIS PATH IS LOAD-BEARING. The capability manifest's
``list-typed-evidence-recovery-pinned`` delivered_check greps for this exact
filename and for the string ``evidence`` inside it. Do not rename either.

TWO CONSTRAINTS THIS FILE ENCODES, both measured:

1. Every call goes through ``async with Client(server)``. Middleware is
   BYPASSED by ``tool.fn(...)``, ``await tool.run({...})`` and
   ``server._tool_manager.call_tool(...)`` — and the established idiom in
   ``escalation/tests/test_server.py`` is ``tool.fn(...)``. A test written
   that way would pass while running none of the guard.

2. Specimens come from the committed corpus and are keyed by ``tool_use_id``,
   never by index, so a corpus refresh cannot silently repoint an assertion at
   a different payload. They are REAL leaked payloads: a refusal is the one
   outcome an invented specimen can trivially fake.

The on-disk record ``esc-3184-2`` is the motivating specimen — stored
``suggested_action`` is ``''`` and ``evidence`` is ``[]`` while both swallowed
values sit legibly in the tail of ``detail`` — but it CANNOT be a fixture
here: ``data/`` is gitignored so it does not exist in this worktree, and it is
doubly corrupted (PRD boundary row B5) so it could never demonstrate a
successful recovery. The corpus supplies both shapes instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from shared.mcp_markup_middleware import MarkupGuardMiddleware, RepairPolicy
from shared.toolcall_markup import detect

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# The committed corpus.
# ---------------------------------------------------------------------------
#
# Read with a small module-local loader rather than by importing
# ``shared/tests/toolcall_markup_corpus_extract``: the escalation package
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


#: Recovers BOTH a list-typed (``evidence``) and a str-typed
#: (``suggested_action``) parameter in ONE call — the PRD section 9 gamma-1
#: signal in a single specimen.
GAMMA_1 = 'toolu_01Q1FPhhjWsxGhTEQRfvMaLa'

#: The four required parameters of ``escalate_info``. The gamma-1 specimen's
#: own ``supplied`` list records exactly these plus ``detail``, so this mirrors
#: what the leaking caller really put on the wire.
REQUIRED = {
    'task_id': '3441',
    'agent_role': 'implementer',
    'category': 'cleanup_needed',
    'summary': 'Eight near-duplicate entries restate the same gotcha.',
}


def build(tmp_path: Path):
    """The real server, built the established way, with the sweep off."""
    queue = EscalationQueue(tmp_path / 'esc')
    return queue, create_server(queue, startup_sweep=False)


def guard_of(server) -> MarkupGuardMiddleware:
    """The registered guard — asserting there is exactly one."""
    guards = [
        m for m in getattr(server, 'middleware', []) or []
        if isinstance(m, MarkupGuardMiddleware)
    ]
    assert len(guards) == 1, (
        f'expected exactly one MarkupGuardMiddleware on the escalation '
        f'server, found {len(guards)}'
    )
    return guards[0]


def error_payload(excinfo) -> dict[str, Any]:
    """The refusal payload, parsed off the ToolError."""
    return json.loads(str(excinfo.value))


# ---------------------------------------------------------------------------
# The B3 signal, extended to the list-typed parameter.
# ---------------------------------------------------------------------------


class TestLiveEscalateInfoLandsRepaired:
    """A real leaked ``escalate_info`` is FILED, with its arguments restored.

    This is the whole point of the FORWARD_REPAIR tier (C2 / INV-6): a lost
    ``escalate_info`` strands a task, so the call must get through. The
    ``evidence`` assertion is the field ``esc-3184-2`` lost silently and the
    capability the manifest carries as OPEN.
    """

    async def _call(self, tmp_path: Path):
        record = specimen(GAMMA_1)
        assert record['expected_outcome'] == 'repaired'
        queue, server = build(tmp_path)
        async with Client(server) as client:
            result = await client.call_tool(
                'escalate_info', {**REQUIRED, record['param']: record['value']}
            )
        return queue, result

    @pytest.mark.asyncio
    async def test_the_escalation_is_actually_filed(self, tmp_path: Path):
        """(a) Not merely 'no error' — a real id, really queued."""
        _, result = await self._call(tmp_path)

        assert result.data['status'] == 'queued'
        assert result.data['id'].startswith('esc-3441-')

    @pytest.mark.asyncio
    async def test_the_str_typed_recovery_reaches_the_record(self, tmp_path: Path):
        """(b) ``suggested_action`` — stored as '' by the unguarded server."""
        queue, result = await self._call(tmp_path)

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.suggested_action, 'suggested_action was stored empty'
        assert 'consider seeding this topic' in esc.suggested_action

    def _evidence_of(self, queue, result):
        esc = queue.get(result.data['id'])
        assert esc is not None
        return esc.evidence

    @pytest.mark.asyncio
    async def test_the_list_typed_recovery_reaches_the_record(self, tmp_path: Path):
        """(c) THE row. ``evidence`` must be a LIST of dicts.

        Not ``[]`` — which is what the unguarded server stores today, and what
        a guard that swallowed a decode failure would also store while
        reporting success. Not a ``str`` either, which is what a guard without
        the schema-directed coercion forwards.
        """
        queue, result = await self._call(tmp_path)
        evidence = self._evidence_of(queue, result)

        assert isinstance(evidence, list), (
            f'evidence stored as {type(evidence).__name__}, not a list'
        )
        assert evidence, 'evidence stored EMPTY: the payload was lost'
        assert all(isinstance(entry, dict) for entry in evidence)

    @pytest.mark.asyncio
    async def test_the_evidence_entries_carry_their_observations(self, tmp_path: Path):
        """The recovery is the caller's real data, not a shaped placeholder."""
        queue, result = await self._call(tmp_path)
        evidence = self._evidence_of(queue, result)

        assert any('observation' in entry for entry in evidence)

    @pytest.mark.asyncio
    async def test_the_stored_detail_is_the_clean_value(self, tmp_path: Path):
        """(d) The residue is gone, and it did not migrate into the record."""
        queue, result = await self._call(tmp_path)

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert detect(esc.detail) is None, 'the stored detail still trips detect()'
        assert 'consider seeding this topic' not in esc.detail

    @pytest.mark.asyncio
    async def test_meta_reports_both_recoveries(self, tmp_path: Path):
        """(e) NAMES only — the warning must not become a second copy."""
        _, result = await self._call(tmp_path)

        assert result.meta is not None
        warning = result.meta['markup_repair']
        assert warning['outcome'] == 'repaired'
        assert warning['field'] == 'detail'
        assert warning['recovered_params'] == ['evidence', 'suggested_action']


# ---------------------------------------------------------------------------
# INV-1 and PRD boundary row B15 — properties of the REGISTRATION itself.
# ---------------------------------------------------------------------------


class TestPolicyIsDeclaredAtRegistration:
    """INV-1: the tier is a registration-time enum, never inferred per call."""

    def test_the_guard_is_registered_at_all(self, tmp_path: Path):
        _, server = build(tmp_path)

        assert guard_of(server) is not None

    def test_the_policy_is_forward_repair(self, tmp_path: Path):
        """C2: a lost escalate_info strands a task, so this server forwards."""
        _, server = build(tmp_path)

        assert guard_of(server).policy is RepairPolicy.FORWARD_REPAIR

    def test_nothing_is_exempt(self, tmp_path: Path):
        """An exemption is a DECLARATION, so the empty set is asserted too.

        No tool on this server legitimately carries envelope literals as data
        — the ``scan_memory_content`` case that motivates exemptions lives on
        fused-memory. A future tool that does would be named BARE here.
        """
        _, server = build(tmp_path)

        assert guard_of(server).exempt_tools == frozenset()


class TestStrictInputValidationStaysOff:
    """PRD boundary row B15 — the one setting that disables this guard.

    With ``strict_input_validation=True`` the SDK jsonschema-validates BEFORE
    FastMCP's handler, the middleware chain is never entered, no
    ``markup_detected`` fact is emitted, and every required-parameter leak
    becomes silently unrepairable. Registration must not enable it.
    """

    def test_it_is_falsy(self, tmp_path: Path):
        _, server = build(tmp_path)

        assert not getattr(server, 'strict_input_validation', False)

    @pytest.mark.asyncio
    async def test_and_the_guard_really_does_run(self, tmp_path: Path):
        """The behavioural half: B15 off means a leak is SEEN.

        Asserting the flag alone would still pass on a server where the
        middleware was never added.
        """
        record = specimen(GAMMA_1)
        _, server = build(tmp_path)

        async with Client(server) as client:
            result = await client.call_tool(
                'escalate_info', {**REQUIRED, record['param']: record['value']}
            )

        assert result.meta is not None
        assert 'markup_repair' in result.meta
