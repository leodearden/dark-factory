"""Tests for ``reconciliation.cross_run_citation_scan`` (task 5749).

The scan reads settled runs' findings back out of the journal and confirms each
mem0 citation with a Mem0 point read; a confirmed-absent one becomes a
``DanglingCitation`` that Stage 1 can hand straight to
``repair_memory_citation``. Every test drives a REAL ``ReconciliationJournal``,
because the ``stage_reports`` blob round-trip is where those findings live.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from _fm_helpers import FakeMemoryLookup, build_journal_with_closed_run

from fused_memory.models.reconciliation import (
    ReconciliationRun,
    RunStatus,
    RunType,
    StageId,
    StageReport,
)
from fused_memory.reconciliation import citation_repair
from fused_memory.reconciliation.cross_run_citation_scan import (
    CrossRunCitationScan,
    scan_cross_run_citations,
)
from fused_memory.reconciliation.dangling_citation import DanglingCitation

P = 'reify'
OTHER_PROJECT = 'dark_factory'

RUN_A = '06a4466d-cdc0-49ac-8e99-e6723be39392'
RUN_B = '3f1c7a52-0d4e-4b8e-a2b1-8c0f5b7d9e10'
RUN_C = 'c9d2e8f1-7a3b-4c5d-9e6f-1a2b3c4d5e6f'
RUN_LIVE = '9b8a7c6d-5e4f-4a3b-8c2d-1e0f9a8b7c6d'

FINDING_1 = 'a1a1a1a1-0000-4000-8000-000000000001'
FINDING_2 = 'a1a1a1a1-0000-4000-8000-000000000002'
FINDING_3 = 'a1a1a1a1-0000-4000-8000-000000000003'
FINDING_4 = 'a1a1a1a1-0000-4000-8000-000000000004'

ABSENT_1 = 'beacf7fc-b76a-4c0b-876d-f4cf6d906d42'
ABSENT_2 = 'e1c2d3b4-a5f6-4e7d-8c9b-0a1b2c3d4e5f'
ABSENT_3 = 'f0e1d2c3-b4a5-4968-87a6-b5c4d3e2f1a0'
ABSENT_4 = '0d1c2b3a-4f5e-4d6c-9b8a-7f6e5d4c3b2a'
LIVE = '746b4ab9-ca3c-418b-982a-32b85bfcf94b'
RAISING = '17085708-b888-472f-bbf3-0a06634fd4db'

LIVE_RECORD: dict[str, Any] = {'id': LIVE, 'content': 'still here', 'metadata': {}}

STATS_KEYS = {
    'stage3_cross_run_runs_scanned',
    'stage3_cross_run_citations_verified',
    'stage3_cross_run_dangling_citations',
    'stage3_cross_run_dangling_citations_deferred',
    'stage3_cross_run_citation_verification_errors',
    'stage3_cross_run_citations_unchecked',
    'stage3_cross_run_citation_scan_failed',
}


def _citation(memory_id: str, store: str = 'mem0') -> dict[str, Any]:
    return {'memory_id': memory_id, 'store': store}


def _finding(finding_id: str | None, *cited: Any) -> dict[str, Any]:
    finding: dict[str, Any] = {
        'description': f'finding {finding_id}',
        'severity': 'minor',
        'cited_memories': list(cited),
    }
    if finding_id is not None:
        finding['finding_id'] = finding_id
    return finding


async def _seed_run(
    journal: Any,
    run_id: str,
    *,
    started_at: datetime,
    status: str = 'completed',
    project_id: str = P,
    reports: dict[str, list[dict[str, Any]]],
) -> None:
    """Seed one run the way ``build_journal_with_closed_run`` does, at *started_at*."""
    await journal.start_run(
        ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type=RunType.full,
            trigger_reason='test',
            started_at=started_at,
            completed_at=None,
            status=RunStatus.running,
        )
    )
    await journal.update_run_stage_reports(
        run_id,
        {
            stage: StageReport(
                stage=StageId(stage),
                started_at=started_at,
                completed_at=started_at,
                items_flagged=findings,
            )
            for stage, findings in reports.items()
        },
    )
    if status != 'running':
        await journal.complete_run(run_id, status)


def _looked_up(memory: FakeMemoryLookup) -> list[str]:
    return [memory_id for _, memory_id in memory.calls]


def _later(minutes: int) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)


@pytest_asyncio.fixture
async def journal_a(tmp_path: Any):
    journal = await build_journal_with_closed_run(
        tmp_path,
        run_id=RUN_A,
        project_id=P,
        findings=[_finding(FINDING_1, _citation(ABSENT_1), _citation(LIVE))],
    )
    yield journal
    await journal.close()


class TestDetection:
    @pytest.mark.asyncio
    async def test_absent_citation_becomes_a_record(self, journal_a: Any) -> None:
        memory = FakeMemoryLookup({LIVE: LIVE_RECORD, ABSENT_1: None})
        scan = await scan_cross_run_citations(journal_a, memory, project_id=P)

        assert scan.dangling == (DanglingCitation(RUN_A, FINDING_1, ABSENT_1, 'mem0'),)
        assert scan.citations_verified == 1
        assert scan.dangling_total == 1
        assert scan.runs_scanned == 1
        assert scan.scan_failed is False

    @pytest.mark.asyncio
    async def test_raised_lookup_is_unknown_never_absent(self, tmp_path: Any) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[_finding(FINDING_1, _citation(RAISING))],
        )
        try:
            memory = FakeMemoryLookup({RAISING: TimeoutError('qdrant timed out')})
            scan = await scan_cross_run_citations(journal, memory, project_id=P)
        finally:
            await journal.close()

        assert scan.dangling == ()
        assert scan.dangling_total == 0
        assert scan.verification_errors == 1

    @pytest.mark.asyncio
    async def test_graphiti_and_non_uuid_citations_are_never_looked_up(
        self, tmp_path: Any,
    ) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[
                _finding(
                    FINDING_1,
                    _citation(ABSENT_1, store='graphiti'),
                    _citation('beacf7fc'),
                    _citation('not a uuid at all'),
                    'not even a dict',
                ),
            ],
        )
        try:
            memory = FakeMemoryLookup()
            scan = await scan_cross_run_citations(journal, memory, project_id=P)
        finally:
            await journal.close()

        assert memory.calls == []
        assert scan.dangling == ()
        assert scan.citations_verified == 0

    @pytest.mark.asyncio
    async def test_finding_without_finding_id_is_skipped(self, tmp_path: Any) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[_finding(None, _citation(ABSENT_1))],
        )
        try:
            memory = FakeMemoryLookup()
            scan = await scan_cross_run_citations(journal, memory, project_id=P)
        finally:
            await journal.close()

        assert scan.dangling == ()
        assert memory.calls == []


class TestRunSelection:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', ['running', 'interrupted'])
    async def test_unsettled_runs_are_not_read(self, journal_a: Any, status: str) -> None:
        await _seed_run(
            journal_a, RUN_LIVE, started_at=_later(5), status=status,
            reports={'memory_consolidator': [_finding(FINDING_2, _citation(ABSENT_2))]},
        )
        memory = FakeMemoryLookup()
        scan = await scan_cross_run_citations(journal_a, memory, project_id=P)

        assert ABSENT_2 not in _looked_up(memory)
        assert all(r.target_run_id != RUN_LIVE for r in scan.dangling)
        assert scan.runs_scanned == 1

    @pytest.mark.asyncio
    async def test_other_projects_runs_are_not_read(self, journal_a: Any) -> None:
        await _seed_run(
            journal_a, RUN_B, started_at=_later(5), project_id=OTHER_PROJECT,
            reports={'memory_consolidator': [_finding(FINDING_2, _citation(ABSENT_2))]},
        )
        memory = FakeMemoryLookup()
        scan = await scan_cross_run_citations(journal_a, memory, project_id=P)

        assert ABSENT_2 not in _looked_up(memory)
        assert all(r.target_run_id != RUN_B for r in scan.dangling)


class TestDedupAndOrder:
    @pytest.mark.asyncio
    async def test_same_id_twice_in_one_finding_gives_one_record(
        self, tmp_path: Any,
    ) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[
                _finding(FINDING_1, _citation(ABSENT_1), _citation(ABSENT_1.upper())),
            ],
        )
        try:
            scan = await scan_cross_run_citations(journal, FakeMemoryLookup(), project_id=P)
        finally:
            await journal.close()

        assert scan.dangling == (DanglingCitation(RUN_A, FINDING_1, ABSENT_1, 'mem0'),)

    @pytest.mark.asyncio
    async def test_same_absent_id_in_two_findings_gives_two_records_one_lookup(
        self, tmp_path: Any,
    ) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[
                _finding(FINDING_1, _citation(ABSENT_1)),
                _finding(FINDING_2, _citation(ABSENT_1)),
            ],
        )
        try:
            memory = FakeMemoryLookup()
            scan = await scan_cross_run_citations(journal, memory, project_id=P)
        finally:
            await journal.close()

        assert scan.dangling == (
            DanglingCitation(RUN_A, FINDING_1, ABSENT_1, 'mem0'),
            DanglingCitation(RUN_A, FINDING_2, ABSENT_1, 'mem0'),
        )
        assert _looked_up(memory) == [ABSENT_1]

    @pytest.mark.asyncio
    async def test_newest_run_first_then_stage_then_finding_order(
        self, journal_a: Any,
    ) -> None:
        await _seed_run(
            journal_a, RUN_B, started_at=_later(5),
            reports={
                'memory_consolidator': [
                    _finding(FINDING_2, _citation(ABSENT_2)),
                    _finding(FINDING_3, _citation(ABSENT_3)),
                ],
                'task_knowledge_sync': [_finding(FINDING_4, _citation(ABSENT_4))],
            },
        )
        memory = FakeMemoryLookup({LIVE: LIVE_RECORD})
        scan = await scan_cross_run_citations(journal_a, memory, project_id=P)

        assert scan.dangling == (
            DanglingCitation(RUN_B, FINDING_2, ABSENT_2, 'mem0'),
            DanglingCitation(RUN_B, FINDING_3, ABSENT_3, 'mem0'),
            DanglingCitation(RUN_B, FINDING_4, ABSENT_4, 'mem0'),
            DanglingCitation(RUN_A, FINDING_1, ABSENT_1, 'mem0'),
        )
        assert scan.runs_scanned == 2


class TestBounds:
    @pytest.mark.asyncio
    async def test_lookback_scans_only_the_most_recent_runs(self, journal_a: Any) -> None:
        await _seed_run(
            journal_a, RUN_B, started_at=_later(5),
            reports={'memory_consolidator': [_finding(FINDING_2, _citation(ABSENT_2))]},
        )
        await _seed_run(
            journal_a, RUN_C, started_at=_later(10),
            reports={'memory_consolidator': [_finding(FINDING_3, _citation(ABSENT_3))]},
        )
        memory = FakeMemoryLookup()
        scan = await scan_cross_run_citations(journal_a, memory, project_id=P, lookback=2)

        assert sorted(_looked_up(memory)) == sorted([ABSENT_2, ABSENT_3])
        assert {r.target_run_id for r in scan.dangling} == {RUN_B, RUN_C}
        assert scan.runs_scanned == 2

    @pytest.mark.asyncio
    async def test_max_ids_caps_distinct_lookups(self, tmp_path: Any) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[
                _finding(FINDING_1, _citation(ABSENT_1)),
                _finding(FINDING_2, _citation(ABSENT_2)),
                _finding(FINDING_3, _citation(ABSENT_1)),
                _finding(FINDING_4, _citation(ABSENT_3)),
            ],
        )
        try:
            memory = FakeMemoryLookup()
            scan = await scan_cross_run_citations(journal, memory, project_id=P, max_ids=2)
        finally:
            await journal.close()

        assert _looked_up(memory) == [ABSENT_1, ABSENT_2]
        assert scan.unchecked == 1
        assert scan.dangling_total == 3
        assert ABSENT_3 not in {r.memory_id for r in scan.dangling}

    @pytest.mark.asyncio
    async def test_max_findings_truncates_but_counts_all(self, tmp_path: Any) -> None:
        journal = await build_journal_with_closed_run(
            tmp_path, run_id=RUN_A, project_id=P,
            findings=[
                _finding(FINDING_1, _citation(ABSENT_1)),
                _finding(FINDING_2, _citation(ABSENT_2)),
                _finding(FINDING_3, _citation(ABSENT_3)),
            ],
        )
        try:
            scan = await scan_cross_run_citations(
                journal, FakeMemoryLookup(), project_id=P, max_findings=2,
            )
        finally:
            await journal.close()

        assert scan.dangling == (
            DanglingCitation(RUN_A, FINDING_1, ABSENT_1, 'mem0'),
            DanglingCitation(RUN_A, FINDING_2, ABSENT_2, 'mem0'),
        )
        assert scan.dangling_total == 3
        assert scan.deferred == 1


class _RaisingJournal:
    async def get_recent_runs(self, project_id: str, limit: int = 10) -> Any:
        raise sqlite3.OperationalError('database is locked')


class TestFailureAndStats:
    @pytest.mark.asyncio
    async def test_journal_failure_gives_empty_failed_scan(self) -> None:
        memory = FakeMemoryLookup()
        scan = await scan_cross_run_citations(_RaisingJournal(), memory, project_id=P)

        assert scan.scan_failed is True
        assert scan.dangling == ()
        assert scan.runs_scanned == 0
        assert memory.calls == []

    @pytest.mark.asyncio
    async def test_stats_keys_on_success_path(self, journal_a: Any) -> None:
        scan = await scan_cross_run_citations(
            journal_a, FakeMemoryLookup({LIVE: LIVE_RECORD}), project_id=P,
        )
        stats = scan.as_stats('stage3')
        assert set(stats) == STATS_KEYS
        assert all(type(v) is int for v in stats.values())
        assert stats['stage3_cross_run_dangling_citations'] == 1
        assert stats['stage3_cross_run_citation_scan_failed'] == 0

    @pytest.mark.asyncio
    async def test_stats_keys_on_failed_path(self) -> None:
        scan = await scan_cross_run_citations(
            _RaisingJournal(), FakeMemoryLookup(), project_id=P,
        )
        stats = scan.as_stats('stage3')
        assert set(stats) == STATS_KEYS
        assert all(type(v) is int for v in stats.values())
        assert stats['stage3_cross_run_citation_scan_failed'] == 1

    def test_scan_record_is_frozen(self) -> None:
        scan: Any = CrossRunCitationScan(
            dangling=(), runs_scanned=0, citations_verified=0, dangling_total=0,
            verification_errors=0, unchecked=0, scan_failed=False,
        )
        with pytest.raises(AttributeError):
            scan.runs_scanned = 1


class TestDetectRepairAgreement:
    """Every record the scan emits is one ``repair_memory_citation`` accepts."""

    @pytest.mark.asyncio
    async def test_every_record_passes_the_repair_dry_run(self, journal_a: Any) -> None:
        await _seed_run(
            journal_a, RUN_B, started_at=_later(5), status='failed',
            reports={'task_knowledge_sync': [_finding(FINDING_2, _citation(ABSENT_2))]},
        )
        memory = FakeMemoryLookup({LIVE: LIVE_RECORD})
        scan = await scan_cross_run_citations(journal_a, memory, project_id=P)
        assert {r.target_run_id for r in scan.dangling} == {RUN_A, RUN_B}

        for record in scan.dangling:
            outcome = await citation_repair.repair_memory_citation(
                journal_a,
                memory,
                target_run_id=record.target_run_id,
                finding_id=record.finding_id,
                memory_id=record.memory_id,
                store=record.store,
                replacement_memory_id=None,
                repaired_by='test',
                reason='memory_not_found',
                caller_project_id=P,
                apply=False,
            )
            assert outcome.get('status') == 'dry_run', (record, outcome)
