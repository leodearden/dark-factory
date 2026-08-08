"""Tests for ``reconciliation.citation_repair.repair_memory_citation`` (task 3065).

``citation_repair`` owns the third half of the "a cited memory id must resolve"
invariant whose other two halves live in ``citation_verifier``: repairing a
dangling citation on a finding owned by a run that has ALREADY COMPLETED.

Why a separate mechanism at all — the premise these tests encode: a closed run's
recon-report state is gone. ``recon_report_state_ttl_seconds`` defaults to 300s
and ``ReconReportState.tick`` deletes the shadow-store rows at run quiescence, so
every ``cite_*`` tool answers ``run_id_unknown`` for a run older than a few
minutes. The journal's ``runs.stage_reports`` blob is the only durable home of
that run's findings, so it is what the repair path reads and rewrites.

Every test drives a REAL ``ReconciliationJournal`` (via
``_fm_helpers.build_journal_with_closed_run``) rather than a fake, because the
round-trip through ``StageReport`` parse-on-read / ``model_dump`` -on-write is
part of what is under test.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import FakeMemoryLookup, build_journal_with_closed_run

from fused_memory.reconciliation import citation_repair

RUN_ID = '06a4466d-cdc0-49ac-8e99-e6723be39392'
DANGLING = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
SUCCESSOR = '746b4ab9-ca3c-418b-982a-32b85bfcf94b'
SIBLING = '17085708-b888-472f-bbf3-0a06634fd4db'

# The live-record shape ``MemoryService.get_memory_by_id`` returns: the FULL raw
# Qdrant payload under 'metadata', plus a ready-to-read 'content' string.
SUCCESSOR_RECORD: dict[str, Any] = {
    'id': SUCCESSOR,
    'content': 'the surviving consolidated entry',
    'metadata': {
        'category': 'procedural_knowledge',
        'agent_id': 'recon-stage-memory_consolidator',
        'created_at': '2026-07-26T04:34:05Z',
        'kind': 'stage3_procedure',
        'supersedes': [SIBLING],
    },
}


# A raw, non-StageReport ``stage_reports`` entry, shaped like the ones
# ``harness`` actually writes. Note it carries ``failed_stage``, NOT ``stage``:
# ``journal.get_run`` discriminates on ``'stage' in v`` to decide whether to
# parse an entry as a StageReport, so a fixture using ``stage`` here would not
# be a raw entry at all — it would fail model validation on read-back.
RAW_ERROR_ENTRY: dict[str, Any] = {
    'error_type': 'CancelledError',
    'error_message': 'Run cancelled (timeout or external cancellation)',
    'failed_stage': 'integrity_check',
    'traceback': '',
}


def _finding(finding_id: str, cited: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        'finding_id': finding_id,
        'description': f'finding {finding_id}',
        'severity': 'medium',
        'cited_memories': cited,
        **extra,
    }


def _citation(memory_id: str, store: str = 'mem0') -> dict[str, Any]:
    return {
        'memory_id': memory_id,
        'store': store,
        'metadata_fingerprint': {
            'category': 'procedural_knowledge',
            'agent_id': 'recon-stage-memory_consolidator',
            'created_at': '2026-07-20T00:00:00Z',
        },
    }


def _dump(run: Any) -> dict[str, Any]:
    """A JSON-comparable snapshot of a run's whole ``stage_reports`` blob."""
    serialized = {
        key: value.model_dump(mode='json') if hasattr(value, 'model_dump') else value
        for key, value in run.stage_reports.items()
    }
    return json.loads(json.dumps(serialized, sort_keys=True, default=str))


class TestRepairHappyPath:
    """A confirmed-dangling citation on a completed run is re-pointed durably."""

    @pytest.mark.asyncio
    async def test_repairs_citation_and_records_provenance(self, tmp_path):
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[
                _finding('f-1', [_citation(DANGLING)]),
                _finding('f-2', [_citation(SUCCESSOR)]),
            ],
            extra_stage_reports={'_error': RAW_ERROR_ENTRY},
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['status'] == 'repaired'
            assert outcome['target_run_id'] == RUN_ID
            assert outcome['stage'] == 'memory_consolidator'
            assert outcome['finding_id'] == 'f-1'
            assert outcome['removed_memory_id'] == DANGLING
            assert outcome['replacement_memory_id'] == SUCCESSOR
            assert outcome['removed_count'] == 1
            assert outcome['deduped'] is False
            assert 'error' not in outcome

            # The PERSISTED blob, re-read through the journal — not the in-memory
            # model the call mutated. A repair that only edited the loaded object
            # would pass every assertion above and change nothing durable.
            after = _dump(await journal.get_run(RUN_ID))
            repaired = after['memory_consolidator']['items_flagged'][0]
            assert repaired['finding_id'] == 'f-1'

            cited_ids = [c['memory_id'] for c in repaired['cited_memories']]
            assert DANGLING not in cited_ids
            assert cited_ids == [SUCCESSOR]
            new_citation = repaired['cited_memories'][0]
            assert new_citation['store'] == 'mem0'
            # Fingerprint is derived from the live record the corroboration read
            # already fetched — the same {category, agent_id, created_at} shape
            # MemoryService.get_memory returns for a mem0 citation.
            assert new_citation['metadata_fingerprint'] == {
                'category': 'procedural_knowledge',
                'agent_id': 'recon-stage-memory_consolidator',
                'created_at': '2026-07-26T04:34:05Z',
            }

            assert len(repaired['citation_repairs']) == 1
            record = repaired['citation_repairs'][0]
            assert record['memory_id'] == DANGLING
            assert record['replacement_memory_id'] == SUCCESSOR
            assert record['store'] == 'mem0'
            assert record['reason'] == 'memory_not_found'
            assert record['repaired_by'] == 'run:caller-1'
            repaired_at = datetime.fromisoformat(record['repaired_at'])
            assert repaired_at.tzinfo is not None
            assert repaired_at.utcoffset().total_seconds() == 0

            # Nothing but the target finding moved: the sibling finding, the raw
            # non-StageReport entry, and every stat/summary field are untouched,
            # so the run's flagged_count and the judge's stat verification stay
            # exactly as the original run reported them.
            assert after['_error'] == before['_error']
            assert after['memory_consolidator']['items_flagged'][1] == (
                before['memory_consolidator']['items_flagged'][1]
            )
            assert after['memory_consolidator']['stats'] == (
                before['memory_consolidator']['stats']
            )
            untouched_keys = {
                k: v for k, v in repaired.items()
                if k not in {'cited_memories', 'citation_repairs'}
            }
            assert untouched_keys == {
                k: v for k, v in before['memory_consolidator']['items_flagged'][0].items()
                if k not in {'cited_memories', 'citation_repairs'}
            }
        finally:
            await journal.close()


class TestRepairCorroborationGates:
    """The repair is structurally incapable of rewriting a live claim.

    Each case asserts BOTH the structured refusal AND that the persisted blob is
    unchanged — a gate that returns an error after having already written would
    satisfy the first assertion alone.
    """

    @pytest.mark.asyncio
    async def test_victim_still_resolves_is_refused(self, tmp_path):
        """A citation that STILL resolves is never re-pointed.

        This is the gate that keeps the tool from being a provenance-
        falsification surface: without it, any Stage-1/2 agent could silently
        re-point a valid, live citation on a closed audit record.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            # The 'dangling' id is in fact alive.
            memory = FakeMemoryLookup(
                {DANGLING: {'id': DANGLING, 'metadata': {}}, SUCCESSOR: SUCCESSOR_RECORD}
            )

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'citation_not_dangling'
            assert outcome['error_type'] == 'ReconCitationNotDangling'
            assert 'status' not in outcome
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_replacement_not_found_is_refused(self, tmp_path):
        """A repair may never introduce a SECOND dangling id.

        Without this gate the tool becomes a generator of the very defect it
        exists to fix.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: None})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'replacement_not_found'
            assert outcome['error_type'] == 'ReconCitationReplacementNotFound'
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('raising_id', 'exc'),
        [
            (DANGLING, TimeoutError('qdrant read timed out')),
            (SUCCESSOR, TimeoutError('qdrant read timed out')),
        ],
        ids=['victim_lookup_raises', 'replacement_lookup_raises'],
    )
    async def test_raised_lookup_is_unknown_never_absent(self, tmp_path, raising_id, exc):
        """A RAISED backend read is 'unknown', not 'absent' — never a repair.

        The same no-silent-fail split ``verify_cited_memories`` already draws for
        this exact lookup, so the two halves of the invariant cannot disagree
        about what a backend timeout means. Collapsing a timeout into
        'confirmed absent' would let a transient Qdrant blip rewrite a live
        citation.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup(
                {DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD} | {raising_id: exc}
            )

            # Must not propagate: the caller gets a structured verdict, not a raise.
            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'verification_error'
            assert outcome['error_type'] == 'ReconCitationVerificationError'
            # The raised type is carried as a FACT, not folded into prose.
            assert outcome['exception_type'] == 'TimeoutError'
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()


class TestRepairResolutionErrors:
    """Shape and resolution refusals, every one leaving the blob untouched."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('bad_id', ['beacf7fc', '96cddd4d-edge', '', 'not-a-uuid'])
    @pytest.mark.parametrize('field', ['memory_id', 'replacement_memory_id'])
    async def test_non_canonical_uuid_refused_before_any_lookup(
        self, tmp_path, field, bad_id
    ):
        """A non-canonical id is rejected BEFORE the service is called.

        Same full-36-char requirement ``citation_verifier.is_concrete_memory_id``
        already enforces for a forwarding pointer: an 8-char prefix is not a
        valid id, and looking one up would just return not-found and read as
        'confirmed absent'.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            kwargs = {'memory_id': DANGLING, 'replacement_memory_id': SUCCESSOR}
            kwargs[field] = bad_id

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                store='mem0',
                repaired_by='run:caller-1',
                **kwargs,
            )

            assert outcome['error'] == 'invalid_uuid_shape'
            assert outcome['error_type'] == 'ReconReportInvalidUuid'
            assert memory.calls == []
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_graphiti_store_refused_before_any_lookup(self, tmp_path):
        """``store='graphiti'`` is refused, not silently mis-verified.

        ``get_memory_by_id`` is a Mem0/Qdrant point read, so every graphiti
        citation would resolve to None and be classified confirmed-dangling —
        the tool would destroy valid graph provenance. Verifying a graph
        citation needs a different primitive; refusing loudly leaves that as a
        clean separate extension rather than a silent wrong answer.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING, store='graphiti')])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='graphiti',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'unsupported_store'
            assert outcome['error_type'] == 'ReconCitationUnsupportedStore'
            assert 'get_memory_by_id' in outcome['hint']
            assert memory.calls == []
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_unknown_target_run(self, tmp_path):
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id='2b1d1b4e-0000-4000-8000-000000000000',
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'target_run_not_found'
            assert outcome['error_type'] == 'ReconCitationTargetRunNotFound'
            assert memory.calls == []
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_finding_absent_from_every_stage(self, tmp_path):
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-nope',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            # Reuses recon-report's existing spelling so a consumer branching on
            # cite_memory's errors sees ONE vocabulary, not two.
            assert outcome['error'] == 'finding_unknown'
            assert outcome['error_type'] == 'ReconReportFindingUnknown'
            assert memory.calls == []
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_finding_does_not_cite_the_memory(self, tmp_path):
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(SIBLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'citation_not_present'
            assert outcome['error_type'] == 'ReconCitationNotPresent'
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_raw_non_stage_report_entry_is_inert(self, tmp_path):
        """The cross-stage scan skips raw entries and finds the real stage.

        ``journal.get_run`` deliberately keeps ``_error`` / ``_resume`` entries
        as plain dicts. A scan that assumed every value was a StageReport would
        crash on exactly the runs most likely to need repair.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            # The finding lives in a LATER stage, behind the raw entry, so the
            # scan must both skip the raw value and keep going.
            findings=[],
            extra_stage_reports={
                '_error': RAW_ERROR_ENTRY,
                '_resume': {'resumed_at': '2026-07-26T04:00:00Z'},
                'integrity_check': {
                    'stage': 'integrity_check',
                    'started_at': '2026-07-26T04:30:00Z',
                    'completed_at': '2026-07-26T04:31:15Z',
                    'items_flagged': [_finding('f-1', [_citation(DANGLING)])],
                    'stats': {},
                },
            },
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['status'] == 'repaired'
            assert outcome['stage'] == 'integrity_check'
            after = _dump(await journal.get_run(RUN_ID))
            assert after['_error'] == RAW_ERROR_ENTRY
            assert after['_resume'] == {'resumed_at': '2026-07-26T04:00:00Z'}
            repaired = after['integrity_check']['items_flagged'][0]
            assert [c['memory_id'] for c in repaired['cited_memories']] == [SUCCESSOR]
        finally:
            await journal.close()


class TestRepairLiveRunRefusalAndDryRun:
    """Closed runs only, and a dry-run that shares the applied path's answer."""

    @pytest.mark.asyncio
    async def test_running_status_is_refused(self, tmp_path):
        """A run still in flight is refused before any lookup.

        ``update_run_stage_reports`` rewrites the WHOLE blob, and the harness
        calls it again at each stage boundary from its in-memory assembled
        report — so a journal-side repair on a live run would be silently
        clobbered at the next stage transition. A fix that appears to succeed
        and then evaporates is worse than a clean refusal.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            status='running',
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['error'] == 'run_still_live'
            assert outcome['error_type'] == 'ReconCitationRunStillLive'
            assert memory.calls == []
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_completed_run_still_in_process_is_refused(self, tmp_path):
        """A run the recon-report state still holds is live for this purpose.

        The journal row can read 'completed' while the in-process entry is
        still being written through, so ``live_run_ids`` (the caller's
        ``ReconReportState._state`` view) is the second half of the guard.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
                live_run_ids=frozenset({RUN_ID}),
            )

            assert outcome['error'] == 'run_still_live'
            assert outcome['error_type'] == 'ReconCitationRunStillLive'
            # The live case is not a gap: within a live run _resolve_finding is
            # already cross-stage, so the ordinary tools reach the finding.
            assert 'cite_memory' in outcome['hint']
            assert 'delete_finding' in outcome['hint']
            assert memory.calls == []
            assert _dump(await journal.get_run(RUN_ID)) == before
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_dry_run_matches_applied_outcome_without_writing(self, tmp_path):
        """``apply=False`` answers exactly what ``apply=True`` would, and writes nothing."""
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            before = _dump(await journal.get_run(RUN_ID))
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            spy = AsyncMock(wraps=journal.update_run_stage_reports)
            journal.update_run_stage_reports = spy

            call = dict(
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )
            dry = await citation_repair.repair_memory_citation(
                journal, memory, apply=False, **call
            )

            assert dry['status'] == 'dry_run'
            assert spy.await_count == 0
            assert _dump(await journal.get_run(RUN_ID)) == before
            # Both corroboration reads still happened — a dry-run that skipped
            # them would tell the operator nothing about whether the gates hold.
            assert [mid for _project, mid in memory.calls] == [DANGLING, SUCCESSOR]

            applied = await citation_repair.repair_memory_citation(
                journal, memory, apply=True, **call
            )

            assert applied['status'] == 'repaired'
            assert spy.await_count == 1
            # One code path: everything but the status verdict is identical.
            assert {k: v for k, v in dry.items() if k != 'status'} == {
                k: v for k, v in applied.items() if k != 'status'
            }
        finally:
            await journal.close()


class TestRepairDropOnlyAndIdempotency:
    """Drop-only, dedupe, repeat-call and multi-occurrence behaviour."""

    @pytest.mark.asyncio
    async def test_drop_only_when_no_replacement_given(self, tmp_path):
        """Omitting the replacement DROPS the dangling citation.

        The same shape ``verify_cited_memories`` applies to the current run's
        report — a claim that lost its backing loses the citation and gains a
        provenance record, rather than keeping a phantom id.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING), _citation(SUCCESSOR)])],
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=None,
                repaired_by='run:caller-1',
            )

            assert outcome['status'] == 'repaired'
            assert outcome['replacement_memory_id'] is None
            assert outcome['removed_count'] == 1
            # No replacement to resolve, so only the victim was looked up.
            assert [mid for _project, mid in memory.calls] == [DANGLING]

            repaired = _dump(await journal.get_run(RUN_ID))[
                'memory_consolidator'
            ]['items_flagged'][0]
            # The finding's OTHER citation is untouched.
            assert [c['memory_id'] for c in repaired['cited_memories']] == [SUCCESSOR]
            record = repaired['citation_repairs'][0]
            assert record['memory_id'] == DANGLING
            assert record['replacement_memory_id'] is None
            assert record['reason'] == 'memory_not_found'
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_replacement_already_cited_is_deduped(self, tmp_path):
        """A replacement the finding already cites is not appended twice."""
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING), _citation(SUCCESSOR)])],
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['status'] == 'repaired'
            assert outcome['deduped'] is True

            repaired = _dump(await journal.get_run(RUN_ID))[
                'memory_consolidator'
            ]['items_flagged'][0]
            cited_ids = [c['memory_id'] for c in repaired['cited_memories']]
            assert cited_ids == [SUCCESSOR]
            assert cited_ids.count(SUCCESSOR) == 1
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_repeat_repair_is_idempotent(self, tmp_path):
        """A retried repair adds no second provenance record.

        The same no-self-amplification property ``citation_verifier``'s
        tombstone path documents for retried sweeps: a repair that appended a
        further record on every pass would inflate provenance unbounded.
        """
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING)])],
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})
            call = dict(
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            first = await citation_repair.repair_memory_citation(journal, memory, **call)
            assert first['status'] == 'repaired'
            after_first = _dump(await journal.get_run(RUN_ID))

            second = await citation_repair.repair_memory_citation(journal, memory, **call)

            assert second['error'] == 'citation_not_present'
            assert _dump(await journal.get_run(RUN_ID)) == after_first
            repaired = after_first['memory_consolidator']['items_flagged'][0]
            assert len(repaired['citation_repairs']) == 1
        finally:
            await journal.close()

    @pytest.mark.asyncio
    async def test_duplicate_dangling_citations_all_removed(self, tmp_path):
        """One call removes EVERY occurrence and reports how many."""
        journal = await build_journal_with_closed_run(
            tmp_path,
            run_id=RUN_ID,
            findings=[_finding('f-1', [_citation(DANGLING), _citation(DANGLING)])],
        )
        try:
            memory = FakeMemoryLookup({DANGLING: None, SUCCESSOR: SUCCESSOR_RECORD})

            outcome = await citation_repair.repair_memory_citation(
                journal,
                memory,
                target_run_id=RUN_ID,
                finding_id='f-1',
                memory_id=DANGLING,
                store='mem0',
                replacement_memory_id=SUCCESSOR,
                repaired_by='run:caller-1',
            )

            assert outcome['removed_count'] == 2
            repaired = _dump(await journal.get_run(RUN_ID))[
                'memory_consolidator'
            ]['items_flagged'][0]
            assert [c['memory_id'] for c in repaired['cited_memories']] == [SUCCESSOR]
        finally:
            await journal.close()
