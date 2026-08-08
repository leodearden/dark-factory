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
            extra_stage_reports={'_error': {'stage': 'integrity_check', 'msg': 'boom'}},
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
