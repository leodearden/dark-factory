"""Integration-gate suite ο=2232: two-way boundary tests over the ledger +
write-policy seams (PRD plans/recon-reliability-prd.md §9, stream W5, B+H).

This file is the integration-gate LEAF for the recon-reliability PRD: it
drives the REAL, already-merged code across every §9 boundary scenario,
facing BOTH the producer and consumer side of each seam, against per-test
tmp SQLite. It adds NO production code — every seam exercised here was
built and merged by one of the seven W5 dependencies below.

Fake/instrumented boundaries only:
  - The LLM/agent turn itself (never invoked — every scenario drives a
    deterministic Python entry point directly).
  - The Mem0/Qdrant network (AsyncMock services; an optional real-Qdrant
    delta check in P4 self-skips via qdrant_skipif()).

Expected GREEN-on-arrival: all 7 dependencies (α/δ/ζ/η/ι/κ/λ =
2219/2222/2224/2225/2227/2228/2229) are status=done and merged, and each
seam's own per-module unit suite is already green. A genuine RED here
would indicate a cross-seam COMPOSITION gap in a dependency's module (not
something this test-only task fixes) — escalate rather than patch.

§9 scenario -> seam module -> dependency map
---------------------------------------------
  L1  Ledger writer-vs-GC race (concurrent UPSERT + gc() interleave)
        -> reconciliation/recon_ledger.py (ReconLedgerStore)          [α=2219]
  L2  Ledger UPSERT idempotency (N upserts, one row, last-write-wins)
        -> reconciliation/recon_ledger.py (ReconLedgerStore)          [α=2219]
  L3  Suppression round-trip, no Mem0 search
        -> reconciliation/flag_dedup.py (write_suppression_record,
           filter_suppressed)                                         [ι=2227]
  L4  GC terminal-referenced marker, single DELETE pass
        -> reconciliation/stages/task_knowledge_sync.py
           (_gc_recon_markers)                                        [κ=2228]
  P1  Interceptor update_task terminal-write rejection
        -> middleware/recon_write_policy.py + middleware/
           task_interceptor.py (TaskInterceptor.update_task)          [ζ=2224]
  P2  Interceptor set_task_status live-workflow rejection
        -> middleware/recon_write_policy.py + task_interceptor.py
           (TaskInterceptor.set_task_status)                          [ζ=2224]
  P3  Interceptor update_task stale-snapshot rejection
        -> middleware/recon_write_policy.py + task_interceptor.py
           (TaskInterceptor.update_task)                              [ζ=2224]
  P4  Dedup-exempt add_system_record permission + fresh-point routing
        -> server/tools.py (add_system_record gate) +
           services/memory_service.py (MemoryService.add_system_record) [δ=2222]
  D1  Deterministic per-cycle summary (Python-written, no LLM turn)
        -> reconciliation/summary_pool.py (write_cycle_summary)       [λ=2229]
  S1  Write-journal-derived stats override self-reported counters
        -> reconciliation/stats_verifier.py (verify_and_rewrite_stats)
           + reconciliation/stage_stats.py (derive_stage_stats)       [θ, merged support]
  E1  Execution-class declaration-layer enforcement at submit_task
        -> middleware/execution_class_guard.py (execution_class_error,
           inject_execution_class) + server/tools.py (submit_task)    [η=2225]

Two scope corrections from the ο task description (see plan.json design
decisions for the full rationale):
  - S1's "unknown key dropped" assertion is corrected to the retired
    ``memories_written`` counter being dropped from top-level stats (a
    truly-unknown key is left completely untouched by ``_apply_observed``,
    not dropped — asserting that would be a doomed premise).
  - E1 does NOT assert operational -> deterministic pure-gate routing
    (task 2085's job, not delivered by any of ο's dependencies); it is
    scoped to η's ratified declaration layer (reject invalid, accept +
    persist valid).

xdist-safety: every fixture below is keyed off a per-test ``tmp_path``
with no shared global/module state, so the suite is safe under this
project's ``-n auto --dist loadgroup`` addopts (pyproject.toml) without
needing per-test ``xdist_group`` markers. No live Neo4j/Qdrant is
required; an optional real-Qdrant delta check self-skips via
qdrant_skipif() (see test_recon_dedup_premise.py's precedent).
``asyncio_mode = "strict"``: every async test carries an explicit
``@pytest.mark.asyncio`` and every async fixture uses
``@pytest_asyncio.fixture``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.services.write_journal import WriteJournal

# Recon-stage agent_id used throughout this suite's interceptor/journal
# scenarios — matches test_recon_write_policy.py's AGENT_ID so the
# recon-stage task-write setup mirrors the established per-seam suites.
AGENT_ID = 'recon-stage-task_knowledge_sync'


# ---------------------------------------------------------------------------
# (a) Real ReconLedgerStore fixture (α=2219) — mirrors test_recon_ledger.py's
# `store` fixture (tmp_path + init/close).
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def ledger(tmp_path):
    """Real, initialized ReconLedgerStore on a per-test tmp_path SQLite DB."""
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    yield store
    await store.close()


# ---------------------------------------------------------------------------
# (b) AsyncMock MemoryService-like object with a REAL initialized
# ReconLedgerStore wired as `.recon_ledger` — mirrors test_flag_dedup.py's
# `ledger_memory_service` fixture / test_stages.py's `ledger_store` pattern.
# `.search` is explicitly an AsyncMock so "Mem0 search was never consulted"
# (L3) is directly observable via `.search.assert_not_called()`.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def recon_service(tmp_path):
    """AsyncMock service (.search/.add_memory/.add_system_record mockable)
    with a REAL initialized ReconLedgerStore attached as `.recon_ledger`."""
    store = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await store.initialize()
    service = AsyncMock()
    service.recon_ledger = store
    service.search = AsyncMock()
    try:
        yield service
    finally:
        await store.close()


# ---------------------------------------------------------------------------
# (c) TaskInterceptor boundary fixtures (ζ=2224) — copied from
# test_recon_write_policy.py:266-297 so P1/P2/P3 drive the REAL
# TaskInterceptor.update_task/set_task_status entry points.
# ---------------------------------------------------------------------------


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


# ---------------------------------------------------------------------------
# (d) WriteJournal fixture + _log_write helper — mirrors
# test_stage_stats.py:99-125, reused by S1's verify_and_rewrite_stats drive.
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def journal(tmp_path):
    j = WriteJournal(tmp_path / 'wj')
    await j.initialize()
    yield j
    await j.close()


async def _log_write(
    journal: WriteJournal,
    *,
    causation_id: str,
    operation: str,
    agent_id: str = AGENT_ID,
    result_summary: dict | str | None = None,
    success: bool = True,
) -> None:
    await journal.log_write_op(
        write_op_id=str(uuid.uuid4()),
        causation_id=causation_id,
        source='mcp_tool',
        operation=operation,
        project_id='test',
        agent_id=agent_id,
        result_summary=result_summary,
        success=success,
    )


# ---------------------------------------------------------------------------
# (e) _make_stage_report helper — builds a real StageReport
# (models/reconciliation.py) with sane defaults, used by D1/S1.
# ---------------------------------------------------------------------------


def _make_stage_report(**overrides) -> StageReport:
    """Build a real StageReport with sane defaults; every field overridable.

    ``stage`` defaults to the task_knowledge_sync stage (this suite's
    primary recon-stage identity, matching AGENT_ID); callers exercising a
    different stage (e.g. S1's memory_consolidator scenario) override it.
    """
    now = datetime(2026, 7, 9, 0, 0, 0, tzinfo=UTC)
    defaults: dict = {
        'stage': StageId.task_knowledge_sync,
        'started_at': now,
        'completed_at': now,
        'items_flagged': [],
        'stats': {},
        'llm_calls': 0,
        'tokens_used': 0,
    }
    defaults.update(overrides)
    return StageReport(**defaults)


# ---------------------------------------------------------------------------
# L1 + L2 — ReconLedgerStore UPSERT idempotency + writer-vs-GC race [α=2219]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLedgerUpsertAndRace:
    """L1 + L2 (α=2219): ReconLedgerStore UPSERT idempotency + writer-vs-GC race.

    Integration delta over test_recon_ledger.py's sequential UPSERT/gc
    coverage: L1 drives a CONCURRENT asyncio.gather() interleave of
    same-identity UPSERTs against an in-flight gc() pass — no existing
    per-seam unit test exercises this interleaving. The 5-column primary
    key plus aiosqlite's single-connection write serialization together
    guarantee the invariants below regardless of task-scheduling order.

    Driving harness (``_drive_l2_sequential_upserts`` /
    ``_drive_l1_concurrent_race`` / ``_count_rows``) lands in step-2 —
    this step only declares the §9 postconditions, so it is RED until
    then (the harness methods referenced below don't exist yet).
    """

    _PROJECT_L2 = 'proj-l2-upsert-idempotency'
    _PROJECT_L1 = 'proj-l1-writer-gc-race'

    async def test_l2_repeated_upsert_same_identity_leaves_one_row_last_write_wins(
        self, ledger,
    ) -> None:
        """L2: N upserts of the same 5-part identity leave exactly one row
        (SELECT COUNT(*)==1), whose payload_json/state equal the LAST
        upsert's values."""
        last_record = await self._drive_l2_sequential_upserts(ledger)

        count = await self._count_rows(ledger, self._PROJECT_L2)
        assert count == 1, f'Expected exactly one row after N upserts, got {count}'

        fetched = await ledger.get_by_identity(
            self._PROJECT_L2, 'stage1_flag_marker',
            task_id='T-idempotent', flag_type='flag_idempotent', run_id='',
        )
        assert fetched is not None, 'Expected the upserted identity to be readable back'
        assert fetched.payload_json == last_record.payload_json, (
            f'Expected last-write-wins payload {last_record.payload_json!r}, '
            f'got {fetched.payload_json!r}'
        )
        assert fetched.state == last_record.state, (
            f'Expected last-write-wins state {last_record.state!r}, got {fetched.state!r}'
        )

    async def test_l1_concurrent_upsert_interleaved_with_gc_preserves_live_marker(
        self, ledger,
    ) -> None:
        """L1: two concurrent same-identity UPSERTs interleaved with a
        gc() pass (via asyncio.gather) leave exactly one row for the
        racing identity, last-writer-wins on payload, AND a co-resident
        still-active, non-expired, non-terminal marker is NEVER deleted
        by the interleaved GC."""
        candidates, terminal_task_id, live_task_id = await self._drive_l1_concurrent_race(ledger)

        fetched_race = await ledger.get_by_identity(
            self._PROJECT_L1, 'stage1_flag_marker',
            task_id='T-race', flag_type='flag_race', run_id='',
        )
        assert fetched_race is not None, 'Expected the racing identity to survive the interleave'
        candidate_payloads = {c.payload_json for c in candidates}
        assert fetched_race.payload_json in candidate_payloads, (
            f'Expected the surviving row to match one of the concurrent '
            f'writers {candidate_payloads!r}, got {fetched_race.payload_json!r}'
        )

        terminal_marker = await ledger.get_by_identity(
            self._PROJECT_L1, 'stage1_flag_marker',
            task_id=terminal_task_id, flag_type='flag_terminal', run_id='',
        )
        assert terminal_marker is None, (
            'gc() must have actually run its terminal-delete pass during the '
            'interleave (precondition proving the live-marker survival below '
            'is not just a GC no-op)'
        )

        live_marker = await ledger.get_by_identity(
            self._PROJECT_L1, 'stage1_flag_marker',
            task_id=live_task_id, flag_type='flag_live', run_id='',
        )
        assert live_marker is not None, (
            'A co-resident still-active, non-expired, non-terminal marker '
            'must NEVER be deleted by an interleaved GC pass'
        )

        # Total surviving rows for the project: racing identity + live
        # marker (terminal marker was collected by the interleaved gc()).
        count = await self._count_rows(ledger, self._PROJECT_L1)
        assert count == 2, (
            f'Expected exactly 2 surviving rows (race + live) after the '
            f'interleaved gc() collected the terminal marker, got {count}'
        )

    # -- driving harness (task 2232 step-2) ---------------------------------

    async def _drive_l2_sequential_upserts(self, ledger: ReconLedgerStore) -> ReconLedgerRecord:
        """Sequentially UPSERT the SAME 5-part identity N=4 times, varying
        payload_json/state each call; return the LAST record written.

        Sequential (not concurrent) — ordering is deterministic, so the
        "last write" is unambiguous, unlike the L1 race below.
        """
        created_at = '2026-07-01T00:00:00+00:00'
        expires_at = '2099-01-01T00:00:00+00:00'
        last_record: ReconLedgerRecord | None = None
        for seq in range(4):
            last_record = ReconLedgerRecord(
                project_id=self._PROJECT_L2,
                record_kind='stage1_flag_marker',
                payload_json=json.dumps({'seq': seq}),
                state='active' if seq % 2 == 0 else 'addressed',
                created_at=created_at,
                task_id='T-idempotent',
                flag_type='flag_idempotent',
                run_id='',
                expires_at=expires_at,
            )
            await ledger.upsert(last_record)
        assert last_record is not None
        return last_record

    async def _drive_l1_concurrent_race(
        self, ledger: ReconLedgerStore,
    ) -> tuple[list[ReconLedgerRecord], str, str]:
        """Seed a terminal marker + a live marker, then concurrently UPSERT
        two versions of a THIRD (racing) identity while a gc() pass
        (referencing the terminal marker's task_id) runs interleaved via
        asyncio.gather().

        Returns ([race_v1, race_v2], terminal_task_id, live_task_id).
        """
        seeded_at = '2026-07-01T00:00:00+00:00'
        far_future = '2099-01-01T00:00:00+00:00'  # never TTL-expires in this test
        now_iso = '2026-07-09T00:00:00+00:00'  # canonical zero-padded UTC ISO-8601

        terminal_marker = ReconLedgerRecord(
            project_id=self._PROJECT_L1,
            record_kind='stage1_flag_marker',
            payload_json='{}',
            state='active',
            created_at=seeded_at,
            task_id='T-done',
            flag_type='flag_terminal',
            run_id='',
            expires_at=far_future,
        )
        live_marker = ReconLedgerRecord(
            project_id=self._PROJECT_L1,
            record_kind='stage1_flag_marker',
            payload_json='{}',
            state='active',
            created_at=seeded_at,
            task_id='T-live',
            flag_type='flag_live',
            run_id='',
            expires_at=far_future,
        )
        # Seed BEFORE the race so the interleaved gc() has real rows to
        # evaluate — these two are not part of the concurrent gather().
        await ledger.upsert(terminal_marker)
        await ledger.upsert(live_marker)

        race_v1 = ReconLedgerRecord(
            project_id=self._PROJECT_L1,
            record_kind='stage1_flag_marker',
            payload_json=json.dumps({'writer': 1}),
            state='active',
            created_at=seeded_at,
            task_id='T-race',
            flag_type='flag_race',
            run_id='',
            expires_at=far_future,
        )
        race_v2 = ReconLedgerRecord(
            project_id=self._PROJECT_L1,
            record_kind='stage1_flag_marker',
            payload_json=json.dumps({'writer': 2}),
            state='active',
            created_at=seeded_at,
            task_id='T-race',
            flag_type='flag_race',
            run_id='',
            expires_at=far_future,
        )

        # The two racing UPSERTs interleave with a gc() pass referencing
        # the terminal marker's task_id — a single aiosqlite connection
        # serializes the actual writes, so exactly one racing row and a
        # real terminal-delete both land regardless of scheduling order.
        await asyncio.gather(
            ledger.upsert(race_v1),
            ledger.upsert(race_v2),
            ledger.gc(self._PROJECT_L1, now_iso, terminal_task_ids=['T-done']),
        )

        return [race_v1, race_v2], 'T-done', 'T-live'

    async def _count_rows(self, ledger: ReconLedgerStore, project_id: str) -> int:
        """Raw SELECT COUNT(*) over recon_ledger for *project_id* (all
        record_kinds) — a direct check on the store's own connection,
        independent of get_by_identity's per-identity read path."""
        cursor = await ledger._db.execute(  # noqa: SLF001 — intentional direct-connection check
            'SELECT COUNT(*) FROM recon_ledger WHERE project_id = ?',
            (project_id,),
        )
        row = await cursor.fetchone()
        return row[0]


# ---------------------------------------------------------------------------
# L3 — Suppression round-trip: write_suppression_record -> filter_suppressed,
# no Mem0 search [ι=2227]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSuppressionRoundTrip:
    """L3 (ι=2227): producer ``write_suppression_record`` -> consumer
    ``filter_suppressed`` round-trip, proving the indexed
    ``list_suppressions`` ledger query fully replaces the retired
    project-wide Mem0 semantic search.

    Driving harness (``_drive_scoped_round_trip`` /
    ``_drive_blanket_round_trip``) lands in step-4 — this step only
    declares the §9 postconditions, so it is RED until then (the harness
    methods referenced below don't exist yet).
    """

    _PROJECT = 'proj-l3-suppression-round-trip'
    _TASK_ID = 501
    _OTHER_TASK_ID = 502
    _SUPPRESSED_FLAG_TYPE = 'missing_deliverable'
    _SURVIVING_FLAG_TYPE = 'human_review_required_deferred'

    async def test_scoped_suppression_drops_matching_flag_keeps_others(
        self, recon_service,
    ) -> None:
        """A SCOPED suppression for (task_id, flag_type) drops only that
        pair: a different flag_type for the same task, and the same
        flag_type for a different task, both survive. Mem0 is never
        searched — the indexed ledger query is the sole read path."""
        result = await self._drive_scoped_round_trip(recon_service)

        kept = {(f['task_id'], f['flag_type']) for f in result}
        assert (self._TASK_ID, self._SUPPRESSED_FLAG_TYPE) not in kept, (
            'Expected the scoped-suppressed (task_id, flag_type) pair to be dropped'
        )
        assert (self._TASK_ID, self._SURVIVING_FLAG_TYPE) in kept, (
            'Expected a differently-typed flag for the same task to survive'
        )
        assert (self._OTHER_TASK_ID, self._SUPPRESSED_FLAG_TYPE) in kept, (
            'Expected the same flag_type for a different task to survive'
        )
        recon_service.search.assert_not_called()

    async def test_blanket_suppression_drops_every_flag_type_for_task(
        self, recon_service,
    ) -> None:
        """A BLANKET suppression (``flag_types=None``) drops EVERY
        flag_type for that task_id, while a different task's matching
        flag_type survives untouched. Mem0 is never searched."""
        result = await self._drive_blanket_round_trip(recon_service)

        assert result == [
            {'task_id': self._OTHER_TASK_ID, 'flag_type': self._SUPPRESSED_FLAG_TYPE},
        ], (
            "Expected every flag_type for the blanket-suppressed task to be "
            "dropped, keeping only the other task's flag"
        )
        recon_service.search.assert_not_called()
