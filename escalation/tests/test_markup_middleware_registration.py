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

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation
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

        Asserted against the record READ BACK OFF THE QUEUE, so this is a
        write-then-read round trip and not merely what was handed to
        ``submit``: a payload this size crosses a JSON encode/decode, and the
        specimen is full of the embedded quotes and newlines that a lossy one
        would mangle.
        """
        record = specimen(UNREPAIRABLE)
        queue, _ = await self._refuse(tmp_path)

        residue = self._residue(queue)
        assert record['value'] in residue.detail, (
            'the residue record does not contain the raw payload in full'
        )
        assert len(record['value']) == record['original_length'] == 3525

    @pytest.mark.asyncio
    async def test_the_residue_names_the_leaking_call(self, tmp_path: Path):
        """(c) Plus the flat fields an operator needs to chase the leak."""
        queue, _ = await self._refuse(tmp_path)

        detail = self._residue(queue).detail
        assert "tool='escalate_info'" in detail
        assert "field='detail'" in detail
        assert 'matched_pattern=' in detail

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
        assert f'owner={RESIDUE_OWNER!r}' in residue.detail, (
            'the residue record does not name the owner that will exit the hold'
        )
        # level=2 and a born-at-L2 severity are ONE decision (models.py: an
        # escalation is born at L2 *when* its severity is in this set), so a
        # level=2 record carrying 'info' would be incoherent to every reader of
        # the consumer-per-level contract.
        assert residue.severity in BORN_AT_L2_SEVERITIES
        # And a born-at-L2 record must carry a SENTINEL role — the same
        # contract submit.py enforces at its argument boundary, and the reason
        # the server downgrades an agent-filed critical.
        assert residue.agent_role.startswith(('harness-', 'orchestrator-'))

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


# ---------------------------------------------------------------------------
# The regression pin: EVERY committed specimen, against the REAL server.
# ---------------------------------------------------------------------------

#: The corpus records tool names in the AGENT-FACING prefixed spelling; the
#: in-server name FastMCP dispatches on is the bare suffix.
ESCALATION_TOOLS = ('mcp__escalation__escalate_info', 'mcp__escalation__escalate_blocker')

REPLAY = [r for r in load_corpus() if r['tool'] in ESCALATION_TOOLS]

# A mis-typed filter would otherwise yield an empty, always-green
# parametrisation. Measured: 24 escalate_info + 1 escalate_blocker.
assert REPLAY, f'no specimens collected for {ESCALATION_TOOLS}'
assert {r['tool'] for r in REPLAY} == set(ESCALATION_TOOLS), (
    'at least one specimen must be collected for EACH tool'
)

#: Type-correct fillers for the OTHER arguments a specimen was sent with.
#: Replaying with the specimen's own ``supplied`` set is what makes the outcome
#: comparable: ``repair`` REFUSES a recovery whose name the caller already
#: supplied, so replaying with a smaller argument map would be a more permissive
#: call than the one that really happened.
FILLERS: dict[str, Any] = {
    'task_id': '3441',
    'agent_role': 'implementer',
    'category': 'cleanup_needed',
    'summary': 'Replayed committed corpus specimen.',
    'detail': '',
    'suggested_action': '',
    'severity': 'info',
    'worktree': '/tmp/worktree',
    'workflow_state': 'implementing',
    'evidence': [],
    'level': 0,
}


def replay_args(record: dict[str, Any]) -> dict[str, Any]:
    """The argument map this specimen really arrived with, damage included."""
    args = {name: FILLERS[name] for name in record['supplied']}
    args[record['param']] = record['value']
    if record['tool'].endswith('escalate_blocker'):
        args.setdefault('severity', 'blocking')
    return args


def replay_id(record: dict[str, Any]) -> str:
    """Name the parametrisation by ``tool_use_id`` — never by index."""
    return f"{record['tool'].split('__')[-1]}-{record['tool_use_id']}"


class TestCorpusReplayAgainstRealServer:
    """All 25 real leaked calls, replayed through the real escalation server.

    The hand-picked specimens above prove three shapes. This proves the whole
    measured population, which is what catches two things nothing before it can:

    (a) SCHEMA DRIFT. Each record carries the ``schema_params`` captured at
        extraction time; the live tools may have moved since, and a recovery
        target that is no longer a parameter legitimately becomes unrepairable
        (``repair`` validates recovered names against the schema). The real
        server is ground truth. MEASURED at this commit: zero divergence — every
        name in every specimen's captured schema is still a live parameter of
        the tool it names, so no specimen's outcome changes.

    (b) BOTH TYPE SHAPES AT ONCE. These specimens recover ``suggested_action``
        (str, 13 of them) and ``evidence`` (list, 4), so the schema-directed
        coercion is exercised across every real shape rather than the three the
        earlier tests hand-pick.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('record', REPLAY, ids=replay_id)
    async def test_specimen(self, tmp_path: Path, record: dict[str, Any]):
        bare = record['tool'].split('__')[-1]
        queue, server = build(tmp_path)

        async with Client(server) as client:
            if record['expected_outcome'] == 'unrepairable':
                with pytest.raises(ToolError) as excinfo:
                    await client.call_tool(bare, replay_args(record))
                payload = error_payload(excinfo)
                assert payload['error_type'] == 'mcp_markup_unrepairable'
                # The refusal preserved the payload rather than destroying it.
                residues = [
                    esc for esc in queue.get_pending()
                    if esc.category == RESIDUE_CATEGORY
                ]
                assert len(residues) == 1
                assert record['value'] in residues[0].detail
                assert payload['escalation_id'] == residues[0].id
                return

            result = await client.call_tool(bare, replay_args(record))

        # The escalation is FILED — the tier's whole purpose (INV-6).
        assert result.data['status'] in {'queued', 'dedup_skipped'}
        assert result.meta is not None
        warning = result.meta['markup_repair']
        assert warning['outcome'] == 'repaired'
        assert warning['recovered_params'] == sorted(record['expected_recovered'])

        esc = queue.get(result.data['id'])
        assert esc is not None
        # And the recovered values landed with their DECLARED types, not as the
        # verbatim str slices repair() hands back.
        if 'evidence' in record['expected_recovered']:
            assert isinstance(esc.evidence, list) and esc.evidence
            assert all(isinstance(entry, dict) for entry in esc.evidence)
        if 'suggested_action' in record['expected_recovered']:
            assert isinstance(esc.suggested_action, str)
            assert esc.suggested_action.strip()
        # The clean value is envelope-free wherever it landed.
        assert detect(esc.detail) is None
        assert detect(esc.suggested_action) is None


# ---------------------------------------------------------------------------
# INV-4: the burst alarm, on the real server.
# ---------------------------------------------------------------------------


#: The synthetic anchor every burst alarm is filed under
#: (``server._MARKUP_STORM_ANCHOR_TASK_ID``) and the category that identifies an
#: alarm AS one. Spelled out rather than imported for the same reason
#: ``RESIDUE_CATEGORY`` is: these tests assert the registration site really put
#: this vocabulary on a real record, and importing both sides of a contract lets
#: both drift together.
STORM_ANCHOR = 'mcp-markup-storm'
STORM_CATEGORY = 'mcp_markup_storm'

#: ``MarkupGuardMiddleware``'s default ``storm_threshold``. The escalation
#: server registers the guard without overriding it, so three unrepairable
#: calls in one window is what a burst IS here.
STORM_THRESHOLD = 3


class TestStormBurstAlarm:
    """The burst path against the REAL server (INV-4).

    ``_file_markup_storm`` is the half of the escalation server's sink that
    nothing exercised: the residue path was covered from step 9, the burst path
    by nothing at all. Everything it decides — the dedup predicate, the
    read-failure fall-through, ``level=1``/``severity='blocking'``, and the
    summary it builds — was therefore unverified.

    The dedup predicate is the one that matters. ``get_by_task(anchor,
    status='pending')`` answers "is anything pending on this anchor", NOT "is
    MY alarm already open", and a shared anchor is squatted in practice: the
    measured precedent is recorded in
    ``fused_memory/server/markup_tripwire.py``'s own docstring — the tripwire
    "filed nothing 2026-08-16..2026-08-19 while 41 rejections occurred" because
    another producer's record held its anchor open. A category-blind dedup
    turns that squatting into indefinite silence, and silence reads as calm.
    """

    @staticmethod
    async def _burst(server, calls: int = STORM_THRESHOLD) -> list[dict[str, Any]]:
        """Drive *calls* unrepairable calls; return each refusal payload.

        Unrepairable rather than repairable because that outcome refuses
        VISIBLY — the storm summary rides back on the ``ToolError`` — so the
        test can assert the burst fired from the caller's side as well as from
        the queue's, without depending on either one alone.
        """
        record = specimen(UNREPAIRABLE)
        payloads = []
        async with Client(server) as client:
            for _ in range(calls):
                with pytest.raises(ToolError) as excinfo:
                    await client.call_tool(
                        'escalate_info',
                        {**REQUIRED, record['param']: record['value']},
                    )
                payloads.append(error_payload(excinfo))
        return payloads

    @staticmethod
    def _alarms(queue) -> list[Any]:
        """Every pending burst alarm on the queue, in filing order."""
        return sorted(
            (esc for esc in queue.get_pending() if esc.category == STORM_CATEGORY),
            key=lambda esc: esc.id,
        )

    @pytest.mark.asyncio
    async def test_a_burst_files_exactly_one_alarm(self, tmp_path: Path):
        """(a) Three refusals in a window produce ONE alarm, on the anchor."""
        queue, server = build(tmp_path)
        payloads = await self._burst(server)

        # The caller-facing half: only the call that CROSSED the threshold
        # carries the burst summary, so the alarm is not announced early.
        assert [p.get('storm') is not None for p in payloads] == [False, False, True]
        assert payloads[-1]['storm']['count'] == STORM_THRESHOLD
        assert payloads[-1]['storm']['outcome'] == 'unrepairable'

        alarms = self._alarms(queue)
        assert len(alarms) == 1, f'expected one burst alarm, found {len(alarms)}'
        alarm = alarms[0]
        # Filed on the SYNTHETIC anchor, never on the leaking caller's task.
        assert alarm.task_id == STORM_ANCHOR
        assert alarm.agent_role == 'harness-markup-guard'
        # A rate alarm about a condition, not a hold on one caller's payload:
        # L1, and a severity that is NOT born-at-L2.
        assert alarm.level == 1
        assert alarm.severity == 'blocking'
        assert alarm.severity not in BORN_AT_L2_SEVERITIES
        assert str(STORM_THRESHOLD) in alarm.summary
        assert 'unrepairable' in alarm.summary

    @pytest.mark.asyncio
    async def test_a_second_burst_folds_into_the_open_alarm(self, tmp_path: Path):
        """(b) One open record per burst, not one per window (INV-4).

        The second burst is driven through a SECOND server on the same queue
        directory, because the middleware's own ``StormCounter`` rate-limits to
        one fire per window per instance — so a leak that outlives a restart is
        exactly how a second fire reaches the sink in production.
        """
        queue, server = build(tmp_path)
        await self._burst(server)
        first = self._alarms(queue)
        assert len(first) == 1

        _, restarted = build(tmp_path)
        await self._burst(restarted)

        alarms = self._alarms(queue)
        assert len(alarms) == 1, (
            f'a second burst must fold into the open alarm, found {len(alarms)}'
        )
        assert alarms[0].id == first[0].id

    @pytest.mark.asyncio
    async def test_an_unrelated_pending_record_does_not_suppress_the_alarm(
        self, tmp_path: Path
    ):
        """(c) Another producer's record on this anchor must NOT silence it.

        The anchor is a synthetic id, not a real task, so nothing reserves it:
        any producer filing SYSTEM-scoped records may land one there. If merely
        "something is pending here" counts as "my alarm is already open", an
        actively running leak files nothing for as long as that unrelated record
        stays open — the measured markup_tripwire failure, verbatim.
        """
        queue, server = build(tmp_path)
        squatter = queue.submit(Escalation(
            id=queue.make_id(STORM_ANCHOR),
            task_id=STORM_ANCHOR,
            agent_role='escalation-watcher',
            severity='blocking',
            category='escalation_cluster',
            summary='Unrelated cluster record filed on the same synthetic anchor.',
            level=1,
        ))

        await self._burst(server)

        alarms = self._alarms(queue)
        assert len(alarms) == 1, (
            f'an unrelated pending record on {STORM_ANCHOR!r} must not suppress '
            f'the burst alarm, found {len(alarms)} alarm(s)'
        )
        assert alarms[0].id != squatter
        assert alarms[0].category == STORM_CATEGORY

    @pytest.mark.asyncio
    async def test_an_alarm_from_another_guard_does_not_suppress_this_one(
        self, tmp_path: Path
    ):
        """(c') Nor does a record whose category matches but whose filer differs.

        Sibling guards on other servers file their own bursts under their own
        anchors, so this is the narrower squat: a same-anchor record carrying
        the storm category but filed by a different producer. Dedup means "MY
        alarm is already open", and the filer is half of that identity.
        """
        queue, server = build(tmp_path)
        foreign = queue.submit(Escalation(
            id=queue.make_id(STORM_ANCHOR),
            task_id=STORM_ANCHOR,
            agent_role='plan-tools-markup-guard',
            severity='blocking',
            category=STORM_CATEGORY,
            summary="Another guard's burst alarm, filed on this anchor.",
            level=1,
        ))

        await self._burst(server)

        mine = [esc for esc in self._alarms(queue) if esc.id != foreign]
        assert len(mine) == 1, (
            "another guard's alarm must not suppress this server's, found "
            f'{len(mine)}'
        )
        assert mine[0].agent_role == 'harness-markup-guard'
