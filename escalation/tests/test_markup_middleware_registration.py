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
from fastmcp.exceptions import ToolError
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

#: DOUBLY corrupted — the residue's own value carries a closing content tag, so
#: :func:`repair` cannot locate the boundary and refuses (PRD boundary row B5).
#: The same class as the on-disk ``esc-3184-2``, which is why that record could
#: never have demonstrated a recovery.
UNREPAIRABLE = 'toolu_012YjuXbKZAMwNAo9WR4Pvjx'

#: The category the middleware stamps on a residue record
#: (``mcp_markup_middleware._ESCALATION_CATEGORY``) and the machine-readable
#: owner that will exit the hold unprompted (``_ESCALATION_OWNER``, INV-7).
#: Spelled out here rather than imported: the point of these assertions is that
#: the REGISTRATION SITE carried the middleware's contract onto a real record,
#: and importing both sides of a contract lets both drift together.
RESIDUE_CATEGORY = 'mcp_markup_residue'
RESIDUE_OWNER = 'l2-escalation-watcher'

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


# ---------------------------------------------------------------------------
# The other half of C2: what happens when the boundary CANNOT be found.
# ---------------------------------------------------------------------------


class TestUnrepairableResidueIsPreserved:
    """A refusal must not DESTROY the payload it refuses (C2 L187, INV-7).

    PRD section 4 C2: "Unrepairable input is never guessed, under either
    policy: reject, and file an escalation carrying the full raw payload so
    nothing is discarded even if the caller never retries. That escalation
    names its owner and carries the standing L2 age bound (INV-7)."

    That second sentence is the half a BARE registration silently does not
    deliver. Measured on this server after step 8: with no ``escalation_sink``
    wired the middleware logs ``no escalation_sink is wired, so the residue of
    %r will not be preserved anywhere`` and returns ``None`` — the call is
    refused, ``escalation_id`` comes back null, and the caller's payload is
    gone. For a leak that is by construction an agent emitting text it cannot
    re-emit identically, "the caller can just retry" is not true.

    The refusal itself is CORRECT and is asserted here too: forwarding
    unparsed residue would deliver a call the caller never made while
    permanently dropping whatever arguments hide inside it.
    """

    async def _refuse(self, tmp_path: Path):
        """Drive the doubly-corrupted specimen and return (queue, payload)."""
        record = specimen(UNREPAIRABLE)
        assert record['expected_outcome'] == 'unrepairable'
        assert record['expected_recovered'] == []
        queue, server = build(tmp_path)
        async with Client(server) as client:
            with pytest.raises(ToolError) as excinfo:
                await client.call_tool(
                    'escalate_info', {**REQUIRED, record['param']: record['value']}
                )
        return queue, error_payload(excinfo)

    @staticmethod
    def _residue(queue) -> Any:
        """The one residue record on the queue — asserting there is exactly one."""
        residues = [
            esc for esc in queue.get_pending()
            if esc.category == RESIDUE_CATEGORY
        ]
        assert len(residues) == 1, (
            f'expected exactly one {RESIDUE_CATEGORY} record on the queue, '
            f'found {len(residues)}'
        )
        return residues[0]

    @pytest.mark.asyncio
    async def test_the_call_is_refused(self, tmp_path: Path):
        """(a) The boundary is a guess, so nothing is forwarded."""
        _, payload = await self._refuse(tmp_path)

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        assert payload['outcome'] == 'unrepairable'
        assert payload['tool'] == 'escalate_info'
        # No repaired_call: offering one would invite a retry re-sending a guess.
        assert 'repaired_call' not in payload

    @pytest.mark.asyncio
    async def test_nothing_partial_was_written_for_the_caller(self, tmp_path: Path):
        """(a) The refused call filed NO escalation of its own.

        The tool body never ran, so the caller's own ``task_id`` must have no
        record — a half-written escalation carrying the corrupted ``detail``
        would be the silent fail-soft this guard exists to end.
        """
        queue, _ = await self._refuse(tmp_path)

        assert queue.get_by_task(REQUIRED['task_id']) == []

    @pytest.mark.asyncio
    async def test_a_residue_escalation_is_queued(self, tmp_path: Path):
        """(b) The refusal is non-destructive because THIS record exists."""
        queue, _ = await self._refuse(tmp_path)

        residue = self._residue(queue)
        assert residue.category == RESIDUE_CATEGORY

    @pytest.mark.asyncio
    async def test_the_residue_carries_the_payload_in_full(self, tmp_path: Path):
        """(c) VERBATIM and ENTIRE — not an excerpt.

        Deliberately unlike ``build_markup_block``'s 200-char
        ``content_excerpt``: that is a diagnostic sitting beside a payload the
        caller still holds, while this is the only surviving copy. 3525
        characters of it, for this specimen.
        """
        record = specimen(UNREPAIRABLE)
        queue, _ = await self._refuse(tmp_path)

        residue = self._residue(queue)
        stored = residue.to_json()
        assert record['value'] in stored, (
            'the residue record does not contain the raw payload in full'
        )

    @pytest.mark.asyncio
    async def test_the_residue_names_the_leaking_call(self, tmp_path: Path):
        """(c) Plus the flat fields an operator needs to chase the leak."""
        queue, _ = await self._refuse(tmp_path)

        stored = self._residue(queue).to_json()
        assert 'escalate_info' in stored
        assert 'detail' in stored

    @pytest.mark.asyncio
    async def test_the_residue_is_born_at_l2_and_names_its_owner(self, tmp_path: Path):
        """(d) INV-7: a supervised consumer plus the standing age surfacing.

        This is a queue-backed handoff, so the bound is the L2 watcher's
        standing age surfacing rather than a deadline of this record's own —
        which is exactly what the ``level=2`` stamp buys. A residue record born
        at L0 would wait on a steward that has no idea it exists.
        """
        queue, _ = await self._refuse(tmp_path)

        residue = self._residue(queue)
        assert residue.level == 2
        assert RESIDUE_OWNER in residue.to_json(), (
            'the residue record does not name the owner that will exit the hold'
        )

    @pytest.mark.asyncio
    async def test_the_refusal_names_the_preserved_record(self, tmp_path: Path):
        """(e) A bounced caller can point an operator at its own data.

        Null today: ``_file_residue_escalation`` returns ``None`` when no sink
        is wired, and a sink reporting a non-``str`` is treated as reporting no
        id at all — the caller is better told nothing than pointed at a value
        it cannot look up.
        """
        queue, payload = await self._refuse(tmp_path)

        residue = self._residue(queue)
        assert payload['escalation_id'] == residue.id
