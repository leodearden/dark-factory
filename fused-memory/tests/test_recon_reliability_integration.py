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
  - The Mem0/Qdrant network (AsyncMock services throughout; no live
    Neo4j/Qdrant backend is exercised by this suite).

Expected GREEN-on-arrival: all 7 dependencies (α/δ/ζ/η/ι/κ/λ =
2219/2222/2224/2225/2227/2228/2229) are status=done and merged, and each
seam's own per-module unit suite is already green. A genuine RED here
would indicate a cross-seam COMPOSITION gap in a dependency's module (not
something this test-only task fixes) — escalate rather than patch.

§9 scenario -> seam module -> dependency map
---------------------------------------------
  L1  Ledger UPSERT/gc() interleave invariants (concurrent asyncio.gather;
      aiosqlite serializes writes, so this is not a genuine data race)
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
needing per-test ``xdist_group`` markers. No live Neo4j/Qdrant backend
is required or exercised by this suite (see test_recon_dedup_premise.py
for the repo's separate real-backend dedup-premise check, guarded by
qdrant_skipif()).
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

from fused_memory.middleware import recon_write_policy
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.reconciliation import StageId, StageReport
from fused_memory.models.scope import ProjectId, ProjectRoot, ProjectScope
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.flag_dedup import filter_suppressed, write_suppression_record
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore
from fused_memory.reconciliation.stages.task_knowledge_sync import _gc_recon_markers
from fused_memory.reconciliation.stats_verifier import verify_and_rewrite_stats
from fused_memory.reconciliation.summary_pool import write_cycle_summary
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.memory_service import MemoryService
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


def _scope(project_id: str, project_root: str) -> ProjectScope:
    """Build a ProjectScope from raw strings — mirrors test_stages.py's
    helper of the same name, used by L4's ``_gc_recon_markers`` drive."""
    return ProjectScope(ProjectId(project_id), ProjectRoot(project_root))


# Canonical zero-padded UTC ISO-8601 timestamps shared by every
# _make_ledger_record call below — ReconLedgerStore.gc() compares
# expires_at as lexicographic SQLite TEXT, so every timestamp in this
# suite must share this exact format/width/offset.
_SEEDED_AT = '2026-07-01T00:00:00+00:00'
_FAR_FUTURE_EXPIRES_AT = '2099-01-01T00:00:00+00:00'  # never TTL-expires in these tests


def _make_ledger_record(
    project_id: str,
    task_id: str,
    flag_type: str,
    *,
    record_kind: str = 'stage1_flag_marker',
    payload_json: str = '{}',
    state: str = 'active',
    created_at: str = _SEEDED_AT,
    run_id: str = '',
    expires_at: str | None = _FAR_FUTURE_EXPIRES_AT,
) -> ReconLedgerRecord:
    """Build a ReconLedgerRecord with this suite's canonical marker
    defaults (an 'active' stage1_flag_marker seeded at ``_SEEDED_AT`` that
    never TTL-expires). Callers pass only the varying identity
    (*project_id*/*task_id*/*flag_type*) plus whichever keyword(s) their
    scenario actually varies (e.g. *payload_json*/*state*), so the
    override — the point of each call site — is what stands out. Shared
    by L2's sequential-upsert drive, L1's terminal/live/interleaved
    markers, and L4's GC-pass seed. (D1 never constructs a record
    directly — it drives the write_cycle_summary producer instead.)"""
    return ReconLedgerRecord(
        project_id=project_id,
        record_kind=record_kind,
        payload_json=payload_json,
        state=state,
        created_at=created_at,
        task_id=task_id,
        flag_type=flag_type,
        run_id=run_id,
        expires_at=expires_at,
    )


async def _count_ledger_rows(
    ledger: ReconLedgerStore, project_id: str, *, record_kind: str | None = None,
) -> int:
    """Raw SELECT COUNT(*) over recon_ledger for *project_id*, optionally
    filtered to a single *record_kind* — a direct check on the store's own
    connection, independent of get_by_identity's per-identity read path.
    Shared by L1/L2 (all record_kinds) and D1 (record_kind='cycle_summary').

    ReconLedgerStore exposes only identity-scoped public reads
    (get_by_identity/list_suppressions/marker_task_ids) — no public
    row-count/list-by-project API — so this deliberately reaches into the
    store's private `_db` connection (noqa SLF001) and duplicates the
    `recon_ledger` table/column names owned by TABLE_SQL in
    reconciliation/recon_ledger.py. If either the `_db` attribute or that
    schema is ever renamed, this is the one place (and the only place in
    this suite) that needs updating to match.
    """
    db = ledger._db  # noqa: SLF001 — intentional direct-connection check; see docstring
    assert db is not None, 'Expected an initialized store with an open connection'
    if record_kind is None:
        cursor = await db.execute(
            'SELECT COUNT(*) FROM recon_ledger WHERE project_id = ?',
            (project_id,),
        )
    else:
        cursor = await db.execute(
            'SELECT COUNT(*) FROM recon_ledger WHERE project_id = ? AND record_kind = ?',
            (project_id, record_kind),
        )
    row = await cursor.fetchone()
    assert row is not None
    return row[0]


# ---------------------------------------------------------------------------
# L1 + L2 — ReconLedgerStore UPSERT idempotency + writer/GC interleave [α=2219]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLedgerUpsertAndGcInterleave:
    """L1 + L2 (α=2219): ReconLedgerStore UPSERT idempotency + an
    interleaved-invariants check against a concurrent gc() pass.

    Integration delta over test_recon_ledger.py's sequential UPSERT/gc
    coverage: L1 drives same-identity UPSERTs and a gc() pass concurrently
    via asyncio.gather() — no existing per-seam unit test exercises this
    interleaving. This is NOT a genuine data-race stressor: aiosqlite
    serializes every write on its single connection, so asyncio.gather()
    only varies which coroutine's await points are scheduled first, not
    whether writes can actually interleave at the storage layer. The
    invariants below (exactly one surviving row for the interleaved
    identity, the terminal marker deleted, the live marker kept)
    therefore hold by construction for any scheduling order — this class
    guards those invariants across the async interleave, it does not
    detect a race condition.
    """

    _PROJECT_L2 = 'proj-l2-upsert-idempotency'
    _PROJECT_L1 = 'proj-l1-upsert-gc-interleave'

    async def test_l2_repeated_upsert_same_identity_leaves_one_row_last_write_wins(
        self, ledger,
    ) -> None:
        """L2: N upserts of the same 5-part identity leave exactly one row
        (SELECT COUNT(*)==1), whose payload_json/state equal the LAST
        upsert's values."""
        last_record = await self._drive_l2_sequential_upserts(ledger)

        count = await _count_ledger_rows(ledger, self._PROJECT_L2)
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

    async def test_l1_upsert_interleaved_with_gc_preserves_live_marker(
        self, ledger,
    ) -> None:
        """L1: same-identity UPSERTs and a gc() pass, run concurrently via
        asyncio.gather(), leave exactly one row for the interleaved
        identity, last-writer-wins on payload, AND a co-resident
        still-active, non-expired, non-terminal marker is NEVER deleted by
        the interleaved GC — invariants that hold under any scheduling
        order (see class docstring; aiosqlite serializes the actual
        writes)."""
        candidates, terminal_task_id, live_task_id = await self._drive_l1_concurrent_interleave(ledger)

        fetched_interleaved = await ledger.get_by_identity(
            self._PROJECT_L1, 'stage1_flag_marker',
            task_id='T-interleave', flag_type='flag_interleave', run_id='',
        )
        assert fetched_interleaved is not None, 'Expected the interleaved identity to survive the interleave'
        candidate_payloads = {c.payload_json for c in candidates}
        assert fetched_interleaved.payload_json in candidate_payloads, (
            f'Expected the surviving row to match one of the concurrent '
            f'writers {candidate_payloads!r}, got {fetched_interleaved.payload_json!r}'
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

        # Total surviving rows for the project: interleaved identity + live
        # marker (terminal marker was collected by the interleaved gc()).
        count = await _count_ledger_rows(ledger, self._PROJECT_L1)
        assert count == 2, (
            f'Expected exactly 2 surviving rows (interleaved + live) after the '
            f'interleaved gc() collected the terminal marker, got {count}'
        )

    # -- driving harness ------------------------------------------------------

    async def _drive_l2_sequential_upserts(self, ledger: ReconLedgerStore) -> ReconLedgerRecord:
        """Sequentially UPSERT the SAME 5-part identity N=4 times, varying
        payload_json/state each call; return the LAST record written.

        Sequential (not concurrent) — ordering is deterministic, so the
        "last write" is unambiguous, unlike the L1 interleave below.
        """
        last_record: ReconLedgerRecord | None = None
        for seq in range(4):
            last_record = _make_ledger_record(
                self._PROJECT_L2, 'T-idempotent', 'flag_idempotent',
                payload_json=json.dumps({'seq': seq}),
                state='active' if seq % 2 == 0 else 'addressed',
            )
            await ledger.upsert(last_record)
        assert last_record is not None
        return last_record

    async def _drive_l1_concurrent_interleave(
        self, ledger: ReconLedgerStore,
    ) -> tuple[list[ReconLedgerRecord], str, str]:
        """Seed a terminal marker + a live marker, then UPSERT two versions
        of a THIRD (concurrently-upserted) identity while a gc() pass
        (given the terminal marker's task_id explicitly via
        ``terminal_task_ids`` — resolving which task_ids are terminal is
        L4's concern, not this store-level check) runs interleaved via
        asyncio.gather().

        Returns ([interleave_v1, interleave_v2], terminal_task_id, live_task_id).
        """
        now_iso = '2026-07-09T00:00:00+00:00'  # canonical zero-padded UTC ISO-8601

        terminal_marker = _make_ledger_record(self._PROJECT_L1, 'T-done', 'flag_terminal')
        live_marker = _make_ledger_record(self._PROJECT_L1, 'T-live', 'flag_live')
        # Seed BEFORE the interleave so the interleaved gc() has real rows
        # to evaluate — these two are not part of the concurrent gather().
        await ledger.upsert(terminal_marker)
        await ledger.upsert(live_marker)

        interleave_v1 = _make_ledger_record(
            self._PROJECT_L1, 'T-interleave', 'flag_interleave',
            payload_json=json.dumps({'writer': 1}),
        )
        interleave_v2 = _make_ledger_record(
            self._PROJECT_L1, 'T-interleave', 'flag_interleave',
            payload_json=json.dumps({'writer': 2}),
        )

        # The two concurrently-issued UPSERTs interleave with a gc() pass
        # referencing the terminal marker's task_id — a single aiosqlite
        # connection serializes the actual writes, so exactly one
        # surviving row and a real terminal-delete both land regardless of
        # scheduling order.
        await asyncio.gather(
            ledger.upsert(interleave_v1),
            ledger.upsert(interleave_v2),
            ledger.gc(self._PROJECT_L1, now_iso, terminal_task_ids=['T-done']),
        )

        return [interleave_v1, interleave_v2], 'T-done', 'T-live'


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

    # -- driving harness ------------------------------------------------------

    def _candidate_flags(self) -> list[dict]:
        """Three flags shared by both scenarios: the (task_id,
        suppressed-flag-type) pair under test, a differently-typed flag for
        the same task, and the same flag_type for a different task."""
        return [
            {'task_id': self._TASK_ID, 'flag_type': self._SUPPRESSED_FLAG_TYPE},
            {'task_id': self._TASK_ID, 'flag_type': self._SURVIVING_FLAG_TYPE},
            {'task_id': self._OTHER_TASK_ID, 'flag_type': self._SUPPRESSED_FLAG_TYPE},
        ]

    async def _drive_scoped_round_trip(self, recon_service) -> list[dict]:
        """Producer: write a SCOPED suppression for (_TASK_ID,
        _SUPPRESSED_FLAG_TYPE) via the real write_suppression_record.
        Consumer: filter_suppressed over the three candidate flags."""
        await write_suppression_record(
            recon_service,
            project_id=self._PROJECT,
            task_id=self._TASK_ID,
            flag_types=[self._SUPPRESSED_FLAG_TYPE],
        )
        return await filter_suppressed(recon_service, self._PROJECT, self._candidate_flags())

    async def _drive_blanket_round_trip(self, recon_service) -> list[dict]:
        """Producer: write a BLANKET suppression (flag_types=None) for
        _TASK_ID via the real write_suppression_record. Consumer:
        filter_suppressed over the three candidate flags."""
        await write_suppression_record(
            recon_service,
            project_id=self._PROJECT,
            task_id=self._TASK_ID,
            flag_types=None,
        )
        return await filter_suppressed(recon_service, self._PROJECT, self._candidate_flags())


# ---------------------------------------------------------------------------
# L4 — GC terminal-referenced marker, single DELETE pass [κ=2228]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGcTerminalReferenced:
    """L4 (κ=2228): one ``_gc_recon_markers`` pass deletes ONLY the
    terminal-referenced marker and KEEPS the live-referenced marker.
    """

    _PROJECT = 'proj-l4-gc-terminal-referenced'
    _TERMINAL_TASK_ID = 'T-done'
    _LIVE_TASK_ID = 'T-live'

    async def test_gc_pass_deletes_only_terminal_referenced_marker(
        self, ledger,
    ) -> None:
        """Given a marker referencing a now-terminal task and a marker
        referencing a still-live task, one GC pass deletes only the
        terminal-referenced marker, keeps the live-referenced marker, and
        its swept-count return value reflects exactly the one deleted row
        (proving the deletion happened in the single collapsed pass, not
        some other path)."""
        swept = await self._drive_gc_pass(ledger)

        assert swept == 1, f'Expected exactly 1 row swept (the terminal marker), got {swept}'

        terminal_marker = await ledger.get_by_identity(
            self._PROJECT, 'stage1_flag_marker',
            task_id=self._TERMINAL_TASK_ID, flag_type='flag_terminal', run_id='',
        )
        assert terminal_marker is None, 'Expected the terminal-referenced marker to be deleted'

        live_marker = await ledger.get_by_identity(
            self._PROJECT, 'stage1_flag_marker',
            task_id=self._LIVE_TASK_ID, flag_type='flag_live', run_id='',
        )
        assert live_marker is not None, 'Expected the live-referenced marker to be kept'

    # -- driving harness ------------------------------------------------------

    async def _drive_gc_pass(self, ledger: ReconLedgerStore) -> int:
        """Seed a terminal-referenced marker and a live-referenced marker
        directly on the real ledger, wire a taskmaster whose get_statuses
        reports the terminal/live split, and drive the real consumer
        _gc_recon_markers(memory_service, taskmaster, scope, run_id, now=)."""
        await ledger.upsert(_make_ledger_record(self._PROJECT, self._TERMINAL_TASK_ID, 'flag_terminal'))
        await ledger.upsert(_make_ledger_record(self._PROJECT, self._LIVE_TASK_ID, 'flag_live'))

        memory_service = AsyncMock()
        memory_service.recon_ledger = ledger
        taskmaster = AsyncMock()
        taskmaster.get_statuses = AsyncMock(
            return_value={self._TERMINAL_TASK_ID: 'done', self._LIVE_TASK_ID: 'in-progress'},
        )
        scope = _scope(self._PROJECT, '/tmp/' + self._PROJECT)

        return await _gc_recon_markers(
            memory_service, taskmaster, scope, 'r1',
            now=datetime(2026, 7, 9, 0, 0, 0, tzinfo=UTC),
        )


# ---------------------------------------------------------------------------
# P1 + P2 + P3 — TaskInterceptor two-way rejection round-trips [ζ=2224]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInterceptorReconWritePolicy:
    """P1 + P2 + P3 (ζ=2224): the REAL TaskInterceptor.update_task /
    set_task_status boundary rejects a recon-stage caller's write with the
    canonical ``{error, error_type, agent_id, task_id, op, ...}`` dict and
    never awaits the underlying taskmaster write — while a non-recon caller
    on the same conditions is never gated (recon-scoping negative, one per
    op, since both ops share the same top-level scoping check).
    """

    _TASK_ID = '1'

    def _assert_rejection_shape(self, result: dict, *, error_type: str, op: str) -> None:
        """The LLM-visible rejection dict carries the canonical
        {error(str), error_type, agent_id, task_id, op} shape."""
        assert result.get('error_type') == error_type, (
            f'Expected error_type={error_type!r}, got {result!r}'
        )
        assert isinstance(result.get('error'), str) and result['error'], (
            f'Expected a non-empty str "error" message, got {result!r}'
        )
        assert result.get('agent_id') == AGENT_ID
        assert result.get('task_id') == self._TASK_ID
        assert result.get('op') == op

    async def test_p1_update_task_on_terminal_task_rejects(
        self, interceptor, taskmaster,
    ) -> None:
        """update_task against a task whose live status is terminal
        (done) is rejected before the underlying taskmaster write."""
        result = await self._drive_p1_terminal_reject(interceptor, taskmaster)

        self._assert_rejection_shape(
            result, error_type='ReconTerminalWriteRejected', op='update_task',
        )
        taskmaster.update_task.assert_not_awaited()

    async def test_p1_non_recon_agent_id_not_gated(self, interceptor, taskmaster) -> None:
        """Recon-scoping negative: a non-recon/None agent_id on the same
        terminal-task condition is never gated — the write proceeds."""
        await self._drive_p1_non_recon(interceptor, taskmaster)

        taskmaster.update_task.assert_awaited_once()

    async def test_p2_set_task_status_with_live_workflow_rejects(
        self, interceptor, taskmaster, monkeypatch,
    ) -> None:
        """set_task_status when a live workflow is detected for the task
        is rejected before the underlying taskmaster write."""
        result = await self._drive_p2_live_workflow_reject(interceptor, taskmaster, monkeypatch)

        self._assert_rejection_shape(
            result, error_type='ReconLiveWorkflowWriteRejected', op='set_task_status',
        )
        taskmaster.set_task_status.assert_not_awaited()

    async def test_p2_non_recon_agent_id_not_gated(
        self, interceptor, taskmaster, monkeypatch,
    ) -> None:
        """Recon-scoping negative: a non-recon/None agent_id is never
        gated even when the detector reports a live workflow."""
        await self._drive_p2_non_recon(interceptor, taskmaster, monkeypatch)

        taskmaster.set_task_status.assert_awaited_once()

    async def test_p3_update_task_with_stale_snapshot_rejects(
        self, interceptor, taskmaster,
    ) -> None:
        """update_task carrying a snapshot_status that disagrees with the
        task's live (non-terminal) status is rejected before the write."""
        result = await self._drive_p3_stale_snapshot_reject(interceptor, taskmaster)

        self._assert_rejection_shape(
            result, error_type='ReconStaleSnapshotRejected', op='update_task',
        )
        taskmaster.update_task.assert_not_awaited()

    # -- driving harness ------------------------------------------------------

    async def _drive_p1_terminal_reject(self, interceptor, taskmaster) -> dict:
        taskmaster.get_task = AsyncMock(
            return_value={'id': self._TASK_ID, 'status': 'done', 'title': 'T'},
        )
        return await interceptor.update_task(
            self._TASK_ID, '/project', title='x', agent_id=AGENT_ID,
        )

    async def _drive_p1_non_recon(self, interceptor, taskmaster) -> None:
        taskmaster.get_task = AsyncMock(
            return_value={'id': self._TASK_ID, 'status': 'done', 'title': 'T'},
        )
        await interceptor.update_task(self._TASK_ID, '/project', title='x', agent_id=None)

    async def _drive_p2_live_workflow_reject(self, interceptor, taskmaster, monkeypatch) -> dict:
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        return await interceptor.set_task_status(
            self._TASK_ID, 'in-progress', '/project', agent_id=AGENT_ID,
        )

    async def _drive_p2_non_recon(self, interceptor, taskmaster, monkeypatch) -> None:
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        await interceptor.set_task_status(self._TASK_ID, 'in-progress', '/project', agent_id=None)

    async def _drive_p3_stale_snapshot_reject(self, interceptor, taskmaster) -> dict:
        taskmaster.get_task = AsyncMock(
            return_value={'id': self._TASK_ID, 'status': 'in-progress', 'title': 'T'},
        )
        return await interceptor.update_task(
            self._TASK_ID, '/project', metadata={'snapshot_status': 'pending'}, agent_id=AGENT_ID,
        )


# ---------------------------------------------------------------------------
# P4 — Dedup-exempt add_system_record permission + fresh-point routing [δ=2222]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDedupExemptPermission:
    """P4 (δ=2222): a non-recon agent_id is rejected at the MCP tool
    boundary before the underlying service is ever touched; a recon-stage
    agent_id is permitted and EVERY call is routed through the fresh-uuid
    ``mem0.add_system_record`` path (never the dedup ``mem0.add`` path) —
    so a fresh point lands on every call, not just the first.
    """

    _CONTENT = 'Stage 2 cycle summary for run r1'
    _PROJECT_ID = 'proj-p4-dedup-exempt'
    _CATEGORY = 'observations_and_summaries'

    async def test_non_recon_agent_rejected_before_service_touched(self) -> None:
        """A non-recon-stage caller gets the exact DedupExemptNotPermitted
        dict, and the gate fires before the underlying service method is
        ever called."""
        mock_service, result = await self._drive_reject_via_mcp_tool()

        assert result == {
            'error': 'dedup_exempt_write_not_permitted',
            'error_type': 'DedupExemptNotPermitted',
            'agent_id': 'claude-interactive',
        }, f'Expected the exact DedupExemptNotPermitted dict, got {result!r}'
        mock_service.add_system_record.assert_not_called()

    async def test_recon_stage_agent_permitted_fresh_point_every_call(self, mock_config) -> None:
        """A recon-stage agent_id is permitted, and calling
        add_system_record twice routes through mem0.add_system_record
        BOTH times — never the dedup mem0.add path — proving a fresh
        point lands on every call, not just the first."""
        service = await self._drive_fresh_point_routing(mock_config)

        assert service.mem0.add_system_record.await_count == 2, (  # type: ignore[attr-defined]
            'Expected mem0.add_system_record to be awaited once per call '
            '(fresh point every call, no caching/memoization across calls)'
        )
        service.mem0.add.assert_not_called()  # type: ignore[attr-defined]

    # -- driving harness ------------------------------------------------------

    async def _drive_reject_via_mcp_tool(self) -> tuple[AsyncMock, dict]:
        """Non-recon caller through the REAL MCP tool boundary
        (create_mcp_server + _tool_manager.call_tool), mirroring
        tests/server/test_add_system_record_gate.py."""
        mock_service = AsyncMock()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': self._CONTENT,
                'project_id': self._PROJECT_ID,
                'category': self._CATEGORY,
                'agent_id': 'claude-interactive',
            },
        )
        return mock_service, result

    async def _drive_fresh_point_routing(self, mock_config) -> MemoryService:
        """A real MemoryService with mem0 mocked; call the real
        add_system_record producer TWICE as a recon-stage caller."""
        service = MemoryService(mock_config)
        service.mem0 = AsyncMock()
        service.mem0.add_system_record = AsyncMock(return_value={'results': [{'id': 'sys-1'}]})
        service.mem0.add = AsyncMock(return_value={'results': [{'id': 'dedup-1'}]})

        for _ in range(2):
            await service.add_system_record(
                self._CONTENT,
                project_id=self._PROJECT_ID,
                agent_id=AGENT_ID,
                category=self._CATEGORY,
            )
        return service


# ---------------------------------------------------------------------------
# D1 — Deterministic per-cycle summary: one Python-written ledger row per
# (stage, run_id), idempotent on repeat [λ=2229]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDeterministicCycleSummary:
    """D1 (λ=2229): the deterministic Python ``write_cycle_summary`` writes
    EXACTLY ONE authoritative ``cycle_summary`` ledger row per (project_id,
    'cycle_summary', task_id='', flag_type=stage, run_id) identity, sourced
    directly from the stage's own ``StageReport`` — no LLM turn, no nonce,
    nothing to self-heal. A repeat call for the same identity is an
    idempotent UPSERT: last write wins, row count stays 1.

    Assertions are scoped to the LEDGER row (the authoritative copy) only —
    the best-effort Mem0 mirror is covered by λ's own unit suite
    (test_summary_pool.py::TestWriteCycleSummaryMirrorAndTrim) and is out of
    scope here. Per Rule 5 (and plan.json's design decision), this class
    deliberately has NO grep-source / import-introspection assertion that
    the retired nonce/verify/repair/reconstruct machinery is gone — a
    repo-wide ``retry_nonce`` grep would falsely hit residual prompt prose
    in ``prompts/stage2.py`` owned by a downstream task.
    """

    _PROJECT = 'proj-d1-cycle-summary'
    _STAGE = 'task_knowledge_sync'
    _RUN_ID = 'run-d1-cycle-summary'
    _RECON_POOL = 'stage2_cycle_summary'
    _TRIM_SOURCE = 'stage2_cycle_summary_trim'
    _CAP = 5
    _NOW = datetime(2026, 7, 9, 0, 5, 0, tzinfo=UTC)

    async def test_write_cycle_summary_writes_single_ledger_row_from_report(
        self, recon_service,
    ) -> None:
        """A single write_cycle_summary call for a completed stage writes
        exactly one cycle_summary ledger row, whose payload is derived
        verbatim from the StageReport (stage/run_id/started_at/completed_at/
        items_flagged_count/stats/llm_calls/tokens_used) — never an LLM
        turn."""
        report, wrote = await self._drive_single_write(recon_service)

        assert wrote is True, 'Expected write_cycle_summary to report the ledger upsert succeeded'
        count = await _count_ledger_rows(
            recon_service.recon_ledger, self._PROJECT, record_kind='cycle_summary',
        )
        assert count == 1, f'Expected exactly one cycle_summary row to exist, got {count}'

        record = await recon_service.recon_ledger.get_by_identity(
            self._PROJECT, 'cycle_summary', task_id='', flag_type=self._STAGE, run_id=self._RUN_ID,
        )
        assert record is not None, 'Expected the row to be readable via get_by_identity'
        assert record.task_id == ''
        assert record.flag_type == self._STAGE
        assert record.run_id == self._RUN_ID

        payload = json.loads(record.payload_json)
        assert payload['stage'] == self._STAGE
        assert payload['run_id'] == self._RUN_ID
        assert payload['started_at'] == report.started_at.isoformat()
        assert payload['completed_at'] == report.completed_at.isoformat()
        assert payload['items_flagged_count'] == len(report.items_flagged)
        assert payload['stats'] == report.stats
        assert payload['llm_calls'] == report.llm_calls
        assert payload['tokens_used'] == report.tokens_used

    async def test_repeat_call_same_identity_keeps_one_row_last_write_wins(
        self, recon_service,
    ) -> None:
        """Calling the writer twice for the same (stage, run_id) identity
        leaves exactly one ledger row, carrying the SECOND call's payload —
        an idempotent UPSERT, not an accumulating insert."""
        second_report = await self._drive_repeat_write(recon_service)

        count = await _count_ledger_rows(
            recon_service.recon_ledger, self._PROJECT, record_kind='cycle_summary',
        )
        assert count == 1, f'Expected exactly one row after a repeat write, got {count}'

        record = await recon_service.recon_ledger.get_by_identity(
            self._PROJECT, 'cycle_summary', task_id='', flag_type=self._STAGE, run_id=self._RUN_ID,
        )
        assert record is not None
        payload = json.loads(record.payload_json)
        assert payload['items_flagged_count'] == len(second_report.items_flagged), (
            'Expected the surviving row to carry the SECOND call payload (last write wins)'
        )
        assert payload['llm_calls'] == second_report.llm_calls
        assert payload['tokens_used'] == second_report.tokens_used

    # -- driving harness ------------------------------------------------------

    async def _drive_single_write(self, recon_service) -> tuple[StageReport, bool]:
        """Build a real StageReport and drive the real producer
        write_cycle_summary ONCE for this class's (project, stage, run_id)
        identity; return (report, wrote)."""
        report = _make_stage_report(
            stage=self._STAGE,
            items_flagged=[{'description': 'a'}, {'description': 'b'}],
            stats={'stage2_stage1_dups_suppressed': 1},
            llm_calls=4,
            tokens_used=1234,
        )
        wrote = await write_cycle_summary(
            recon_service,
            self._PROJECT,
            report,
            self._RUN_ID,
            stage=self._STAGE,
            recon_pool=self._RECON_POOL,
            trim_source=self._TRIM_SOURCE,
            cap=self._CAP,
            now=self._NOW,
        )
        return report, wrote

    async def _drive_repeat_write(self, recon_service) -> StageReport:
        """Drive the real producer write_cycle_summary TWICE for the same
        (project, stage, run_id) identity with differing report content;
        return the SECOND report (the one that must survive)."""
        first_report = _make_stage_report(
            stage=self._STAGE,
            items_flagged=[{'description': 'a'}],
            stats={'first': True},
            llm_calls=1,
            tokens_used=100,
        )
        await write_cycle_summary(
            recon_service,
            self._PROJECT,
            first_report,
            self._RUN_ID,
            stage=self._STAGE,
            recon_pool=self._RECON_POOL,
            trim_source=self._TRIM_SOURCE,
            cap=self._CAP,
            now=self._NOW,
        )

        second_report = _make_stage_report(
            stage=self._STAGE,
            items_flagged=[{'description': 'c'}, {'description': 'd'}, {'description': 'e'}],
            stats={'second': True},
            llm_calls=9,
            tokens_used=999,
        )
        await write_cycle_summary(
            recon_service,
            self._PROJECT,
            second_report,
            self._RUN_ID,
            stage=self._STAGE,
            recon_pool=self._RECON_POOL,
            trim_source=self._TRIM_SOURCE,
            cap=self._CAP,
            now=self._NOW,
        )
        return second_report


# ---------------------------------------------------------------------------
# S1 — Write-journal-derived stats override self-reported counters
# [θ, merged support]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestComputedStatsOverride:
    """S1 (θ, merged support): ``stats_verifier.verify_and_rewrite_stats``
    overrides a stage's self-reported write counters with the write-journal
    truth — the LLM's self-report survives only as a divergence signal under
    ``stats['_reported']``.

    Given the journal records 2 successful add_memory ops for the stage and
    the StageReport self-reports memories_added=5 (and the retired
    memories_written=5): journal truth wins (memories_added -> 2), both LLM
    originals are preserved under ``_reported``, and memories_written — an
    allowlisted LLM-reported key that is no longer in ``_COMPUTED_STAT_KEYS``
    — is DROPPED from top-level stats entirely. (Corrected from ο's task
    description's imprecise "unknown key dropped": a truly-unknown key is
    left completely untouched by ``_apply_observed``, not dropped — asserting
    that would be a doomed premise. See plan.json design decisions.) Mirrors
    the canonical green
    ``test_stats_verifier.py::test_verify_boundary_s1_computed_stats_override_llm_report``.
    """

    _STAGE = 'memory_consolidator'

    async def test_journal_truth_overrides_self_reported_stats(self, journal) -> None:
        """Given 2 real journal-observed add_memory ops and a self-reported
        memories_added=5 / memories_written=5: journal truth wins
        (memories_added -> 2), both LLM originals are preserved under
        _reported as the judge's divergence signal, and the retired
        memories_written counter is dropped from top-level stats entirely."""
        report = await self._drive_verify(journal)

        assert report.stats['memories_added'] == 2, (
            f"Expected journal truth (2) to win, got {report.stats.get('memories_added')!r}"
        )
        assert report.stats['_reported']['memories_added'] == 5, (
            "Expected the LLM's self-reported memories_added to survive under _reported"
        )
        assert 'memories_written' not in report.stats, (
            'Expected the retired memories_written counter to be dropped from top-level stats'
        )
        assert report.stats['_reported']['memories_written'] == 5, (
            "Expected the LLM's memories_written claim to survive under _reported"
        )

    # -- driving harness ------------------------------------------------------

    async def _drive_verify(self, journal: WriteJournal) -> StageReport:
        """Log 2 real successful add_memory ops for the stage (journal
        truth), build a StageReport that self-reports memories_added=5 /
        memories_written=5 (LLM claim), then drive the real
        verify_and_rewrite_stats and return the (in-place mutated)
        report."""
        run_id = str(uuid.uuid4())
        for i in range(2):
            await _log_write(
                journal,
                causation_id=run_id,
                operation='add_memory',
                agent_id=f'recon-stage-{self._STAGE}',
                result_summary={'memory_ids': [f'm{i}'], 'stores': ['mem0']},
            )

        report = _make_stage_report(
            stage=self._STAGE,
            stats={'memories_added': 5, 'memories_written': 5},
        )

        await verify_and_rewrite_stats(run_id, {self._STAGE: report}, journal)
        return report


# ---------------------------------------------------------------------------
# E1 — Execution-class declaration-layer enforcement at submit_task [η=2225]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecutionClassEnforcement:
    """E1 (η=2225): the REAL submit_task MCP tool boundary enforces
    metadata.execution_class for recon-stage callers ONLY — reject when
    missing, accept and PERSIST when valid — while a non-recon caller is
    never gated. Scoped to η's ratified declaration layer: this class does
    NOT assert operational -> deterministic pure-gate routing (task 2085's
    job, not delivered by any of ο's dependencies — see plan.json design
    decisions).

    Driven through the real submit_task tool (create_mcp_server +
    _tool_manager.call_tool, mirroring P4's MCP-boundary pattern) with an
    AsyncMock task_interceptor standing in for taskmaster plumbing, and
    resolve_main_checkout monkeypatched to a passthrough — mirrors
    test_submit_resolve_tools.py's passthrough_main_checkout fixture — so a
    synthetic project_root works without a real git working tree.
    Backstopped by a direct guard-level round-trip against
    execution_class_error / inject_execution_class (mirrors
    test_execution_class_guard.py) in case the MCP-boundary accept path
    ever needs deeper taskmaster plumbing than this AsyncMock provides.
    """

    _PROJECT_ROOT = '/project'

    async def test_recon_stage_missing_execution_class_rejected_before_interceptor(
        self, monkeypatch,
    ) -> None:
        """A recon-stage submit_task with NO metadata.execution_class is
        rejected with a ValidationError naming execution_class, and
        task_interceptor.submit_task is never awaited."""
        task_interceptor, result = await self._drive_reject_missing_class(monkeypatch)

        assert result.get('error_type') == 'ValidationError', (
            f'Expected error_type=ValidationError, got {result!r}'
        )
        assert 'execution_class' in result.get('error', ''), (
            f'Expected the error message to name execution_class, got {result!r}'
        )
        task_interceptor.submit_task.assert_not_awaited()

    @pytest.mark.parametrize('valid_class', ['code_tdd', 'operational', 'decision'])
    async def test_recon_stage_valid_execution_class_accepted_and_persisted(
        self, monkeypatch, valid_class,
    ) -> None:
        """A recon-stage submit_task WITH a valid execution_class is
        accepted, and the class is PERSISTED into the metadata forwarded
        to task_interceptor.submit_task."""
        task_interceptor = await self._drive_accept_and_persist(monkeypatch, valid_class)

        task_interceptor.submit_task.assert_awaited_once()
        call_kwargs = task_interceptor.submit_task.call_args.kwargs
        assert call_kwargs.get('metadata', {}).get('execution_class') == valid_class, (
            f'Expected execution_class={valid_class!r} persisted into the forwarded '
            f'metadata, got {call_kwargs.get("metadata")!r}'
        )

    async def test_non_recon_agent_id_not_enforced(self, monkeypatch) -> None:
        """Recon-scoping negative: a non-recon (None) agent_id with NO
        execution_class is never gated — the submission proceeds."""
        task_interceptor = await self._drive_non_recon_not_enforced(monkeypatch)

        task_interceptor.submit_task.assert_awaited_once()

    async def test_guard_level_backstop_reject_and_accept_matrix(self) -> None:
        """Backstop at the guard level (execution_class_error /
        inject_execution_class), independent of the MCP tool wiring:
        missing -> reject naming execution_class; valid -> accept (None)
        AND inject_execution_class persists it into a fresh metadata
        dict."""
        reject, accept, persisted = self._drive_guard_level_backstop()

        assert reject is not None
        assert reject.get('error_type') == 'ValidationError'
        assert 'execution_class' in reject.get('error', '')
        assert accept is None
        assert persisted.get('execution_class') == 'operational'

    # -- driving harness ------------------------------------------------------

    def _passthrough_main_checkout(self, monkeypatch) -> None:
        """Stub resolve_main_checkout to pass its argument through unchanged
        — mirrors test_submit_resolve_tools.py's passthrough_main_checkout
        fixture — so a synthetic project_root like '/project' works without
        a real git working tree."""
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
        )

    def _make_task_interceptor(self) -> AsyncMock:
        """AsyncMock task_interceptor standing in for taskmaster plumbing —
        mirrors test_submit_resolve_tools.py's task_interceptor fixture."""
        task_interceptor = AsyncMock()
        task_interceptor.submit_task = AsyncMock(
            return_value={'ticket': 'tkt_ABCDEFGHIJKLMNOPQRSTUVWXYZ'},
        )
        return task_interceptor

    async def _drive_reject_missing_class(self, monkeypatch) -> tuple[AsyncMock, dict]:
        """Recon-stage submit_task with NO metadata.execution_class through
        the REAL submit_task MCP tool boundary (create_mcp_server +
        _tool_manager.call_tool, mirroring P4's MCP-boundary pattern)."""
        self._passthrough_main_checkout(monkeypatch)
        task_interceptor = self._make_task_interceptor()
        server = create_mcp_server(AsyncMock(), task_interceptor=task_interceptor)

        result = await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': self._PROJECT_ROOT,
                'title': 'Recon stage task',
                'agent_id': AGENT_ID,
                'metadata': {},
            },
        )
        return task_interceptor, result

    async def _drive_accept_and_persist(self, monkeypatch, valid_class: str) -> AsyncMock:
        """Recon-stage submit_task WITH a valid metadata.execution_class
        through the REAL submit_task MCP tool boundary; the interceptor is
        awaited once with the class persisted into the forwarded metadata."""
        self._passthrough_main_checkout(monkeypatch)
        task_interceptor = self._make_task_interceptor()
        server = create_mcp_server(AsyncMock(), task_interceptor=task_interceptor)

        await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': self._PROJECT_ROOT,
                'title': 'Recon stage task',
                'agent_id': AGENT_ID,
                'metadata': {'execution_class': valid_class},
            },
        )
        return task_interceptor

    async def _drive_non_recon_not_enforced(self, monkeypatch) -> AsyncMock:
        """A non-recon (None) agent_id submit_task with NO
        metadata.execution_class is never gated — the interceptor call
        proceeds through the REAL submit_task MCP tool boundary."""
        self._passthrough_main_checkout(monkeypatch)
        task_interceptor = self._make_task_interceptor()
        server = create_mcp_server(AsyncMock(), task_interceptor=task_interceptor)

        await server._tool_manager.call_tool(
            'submit_task',
            {
                'project_root': self._PROJECT_ROOT,
                'title': 'Non-recon task',
                'metadata': {},
            },
        )
        return task_interceptor

    def _drive_guard_level_backstop(self) -> tuple[dict | None, dict | None, dict]:
        """Backstop at the guard level (execution_class_error /
        inject_execution_class), independent of the MCP tool wiring: missing
        -> reject naming execution_class; valid 'operational' -> accept
        (None) AND inject_execution_class persists it into a fresh metadata
        dict — mirrors test_execution_class_guard.py."""
        from fused_memory.middleware.execution_class_guard import (
            execution_class_error,
            inject_execution_class,
        )

        reject = execution_class_error({}, AGENT_ID, self._PROJECT_ROOT)
        accept = execution_class_error(
            {'execution_class': 'operational'}, AGENT_ID, self._PROJECT_ROOT,
        )
        persisted = inject_execution_class({'execution_class': 'operational'})
        return reject, accept, persisted
