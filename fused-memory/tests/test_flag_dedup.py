"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

import json
import uuid as _uuid_mod
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.flag_dedup import build_suppression_payload
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

# ---------------------------------------------------------------------------
# compute_flag_signature tests (step-1)
# ---------------------------------------------------------------------------


class TestComputeFlagSignature:
    """Tests for compute_flag_signature(flag) -> tuple[str, str] | None."""

    def test_returns_tuple_with_both_fields_int_task_id(self):
        """Returns ('123', 'missing_deliverable') when task_id is int and flag_type is str."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': 123, 'flag_type': 'missing_deliverable'}
        result = compute_flag_signature(flag)
        assert result == ('123', 'missing_deliverable')

    def test_returns_tuple_with_both_fields_str_task_id(self):
        """Returns ('123', 'missing_deliverable') when task_id is str."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': '123', 'flag_type': 'missing_deliverable'}
        result = compute_flag_signature(flag)
        assert result == ('123', 'missing_deliverable')

    def test_returns_none_when_task_id_missing(self):
        """Returns None when task_id is absent."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'flag_type': 'missing_deliverable'}
        assert compute_flag_signature(flag) is None

    def test_returns_none_when_flag_type_missing(self):
        """Returns None when flag_type is absent."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': '42'}
        assert compute_flag_signature(flag) is None

    def test_returns_none_when_both_missing(self):
        """Returns None when both task_id and flag_type are absent."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'description': 'some flag'}
        assert compute_flag_signature(flag) is None

    def test_returns_tuple_when_task_id_is_zero(self):
        """task_id=0 is a falsy-but-valid value; should produce a signature, not None."""
        from fused_memory.reconciliation.flag_dedup import compute_flag_signature

        flag = {'task_id': 0, 'flag_type': 'missing_deliverable'}
        result = compute_flag_signature(flag)
        assert result == ('0', 'missing_deliverable'), (
            'task_id=0 must not be silently discarded by a falsy check'
        )


# ---------------------------------------------------------------------------
# dedup_flags tests — no-signature path (step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_no_signature_flags_pass_through_unchanged():
    """Flags with no computable signature pass through unchanged; add_memory never called.

    After task-1654 Fix 2, compute_content_fingerprint_signature is tried as a
    fallback for null-task_id flags.  The four "no-sig" cases that survive both
    helpers are:
    - Has a non-None task_id but missing flag_type (content-fp returns None
      because task_id is not None; compute_flag_signature returns None because
      flag_type is missing).
    - Has task_id=None + flag_type but a blank/whitespace-only description
      (content-fp returns None because the normalised description is empty).
    - Has task_id=None with non-empty cited_tasks whose task_id is None
      (compute_flag_signature's cited_tasks scan yields no usable ids;
      content-fp returns None because cited_tasks technically present but task_id
      is None — both helpers return None).
    - Empty dict: trivially no-sig.

    No flag here has a computable signature, so the per-flag ledger path is
    never reached; recon_ledger is set to None to model a memory_service with
    no ledger attached (the degraded pass-through contract).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.recon_ledger = None
    flags = [
        # (1) has task_id but missing flag_type — compute_flag_signature None;
        #     content-fp None because task_id is not None.
        {'description': 'no flag_type present', 'task_id': '42'},
        # (2) task_id=None + flag_type but blank description — content-fp None.
        {'task_id': None, 'flag_type': 'missing_deliverable', 'description': '   '},
        # (3) task_id=None + cited_tasks whose task_id is None (no usable cited id
        #     for compute_flag_signature; content-fp also None: no non-None task_id
        #     in cited_tasks means the cited_tasks guard doesn't block, BUT then
        #     we'd need a description — omit it to force both helpers to None).
        {'task_id': None, 'flag_type': 'cross_project', 'cited_tasks': [{'project_id': 'x'}]},
        # (4) empty dict — trivially no-sig.
        {},
    ]
    original_flags = [dict(f) for f in flags]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # All flags returned unchanged
    assert result == original_flags
    # add_memory never called — no-signature flags never reach the marker write path
    memory_service.add_memory.assert_not_called()


# ---------------------------------------------------------------------------
# dedup_flags — prior marker found path (step-5)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Ledger-backed memory_service fixture (task 2227 prereq-1)
#
# Shared by every ledger-backed RED test that exercises filter_suppressed's
# indexed query, write_suppression_record's upsert, acknowledge's
# mark_addressed, or dedup_flags' marker UPSERT + completion sweep. Mirrors
# the `store` fixture in tests/test_recon_ledger.py (tmp_path + init/close).
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def ledger_memory_service(tmp_path):
    """AsyncMock memory_service (.search/.add_memory/.delete_memory mockable)
    with a REAL initialized ReconLedgerStore attached as `.recon_ledger`."""
    ledger = ReconLedgerStore(tmp_path / 'reconciliation.db')
    await ledger.initialize()
    service = AsyncMock()
    service.recon_ledger = ledger
    service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['mirror-id']))
    try:
        yield service
    finally:
        await ledger.close()


async def _seed_marker(
    ledger: ReconLedgerStore,
    project_id: str,
    task_id: str,
    flag_type: str,
    *,
    run_id: str = 'seed-run',
    last_seen_run_id: str | None = None,
    state: str = 'active',
    created_at: str = '2026-01-01T00:00:00+00:00',
    expires_at: str | None = '2099-01-01T00:00:00+00:00',
    extra_payload: dict | None = None,
) -> None:
    """Seed a stage1_flag_marker row directly via ledger.upsert (bypasses dedup_flags).

    Identity is (project_id, 'stage1_flag_marker', task_id, flag_type, run_id='') —
    dedup identity excludes run_id from the PK; run_id/last_seen_run_id ride in
    payload_json (see ReconLedgerRecord PK design decision in plan.json).
    """
    payload = {
        'source': 'stage1_flag_marker',
        'kind': 'stage1_flag_marker',
        'task_id': task_id,
        'flag_type': flag_type,
        'run_id': run_id,
        'last_seen_run_id': last_seen_run_id or run_id,
    }
    if extra_payload:
        payload.update(extra_payload)
    await ledger.upsert(ReconLedgerRecord(
        project_id=project_id,
        record_kind='stage1_flag_marker',
        payload_json=json.dumps(payload),
        state=state,
        created_at=created_at,
        task_id=task_id,
        flag_type=flag_type,
        run_id='',
        expires_at=expires_at,
    ))


async def _seed_suppression(
    ledger: ReconLedgerStore,
    project_id: str,
    task_id: str,
    flag_type: str = '',
    *,
    state: str = 'active',
    created_at: str = '2026-01-01T00:00:00+00:00',
) -> None:
    """Seed a stage1_flag_suppression row directly via ledger.upsert.

    flag_type='' (default) seeds a blanket/wildcard row; a non-empty
    flag_type seeds a scoped row.
    """
    payload: dict[str, Any] = {'kind': 'stage1_flag_suppression', 'task_id': task_id}
    if flag_type:
        payload['flag_types'] = [flag_type]
    await ledger.upsert(ReconLedgerRecord(
        project_id=project_id,
        record_kind='stage1_flag_suppression',
        payload_json=json.dumps(payload),
        state=state,
        created_at=created_at,
        task_id=str(task_id),
        flag_type=flag_type,
        run_id='',
        expires_at=None,
    ))


async def _get_marker(
    ledger: ReconLedgerStore, project_id: str, task_id: str, flag_type: str
) -> ReconLedgerRecord | None:
    """Read back a stage1_flag_marker row (fixed run_id='' identity)."""
    return await ledger.get_by_identity(project_id, 'stage1_flag_marker', task_id, flag_type, '')


@pytest.mark.asyncio
async def test_dedup_flags_hit_on_addressed_marker_does_not_suppress_flag(ledger_memory_service):
    """Pin the intended behavior for the addressed-marker/recurrence-detection
    interaction (task-2029 amendment round 2, reviewer finding: design).

    A ``stage1_flag_marker`` row in state='addressed' (set by
    acknowledge_flag_marker) is still found by ``ledger.get_by_identity`` —
    the ledger's identity lookup does not filter on state — exactly like any
    other prior marker. Confirmed here to be SAFE by construction rather than
    a masking bug: dedup_flags NEVER drops/suppresses a flag on a HIT (a HIT
    only annotates persisted_from_run/last_seen_run_id and re-upserts the
    row) — so a genuine recurrence is still surfaced to Stage 2 this cycle —
    and the re-upserted row carries a fresh payload with no addressed_by/
    addressed_run_id key, so the tag does not propagate forward: it
    self-clears (state reverts to 'active') on the very next recurrence
    rather than permanently marking future occurrences as pre-resolved.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger = ledger_memory_service.recon_ledger
    await _seed_marker(ledger, 'p', '42', 'missing_deliverable', run_id='r-ack', state='addressed')

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'recurred'}]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r2',
        flags=flags,
    )

    # The recurring flag is NOT suppressed — still surfaced to Stage 2 this cycle.
    assert len(result) == 1
    assert result[0]['last_seen_run_id'] == 'r2'
    # Annotated from the addressed marker (treated as an ordinary HIT).
    assert result[0]['persisted_from_run'] == 'r-ack'

    # The row is re-upserted: state reverts to 'active' and the new payload
    # carries NO addressed_by/addressed_run_id — the tag is transient and
    # self-clears on the next recurrence instead of permanently marking every
    # future occurrence as pre-resolved.
    row = await _get_marker(ledger, 'p', '42', 'missing_deliverable')
    assert row is not None
    assert row.state == 'active'
    payload = json.loads(row.payload_json)
    assert 'addressed_by' not in payload, (
        f're-upserted marker must not carry forward addressed_by; got {payload!r}'
    )
    assert 'addressed_run_id' not in payload


# ---------------------------------------------------------------------------
# dedup_flags — marker metadata shape + malformed run_id sentinel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marker_metadata_includes_source_and_kind(ledger_memory_service):
    """Stage1 flag marker ledger row must carry BOTH source and kind keys.

    Regression test for task-1659: earlier writes set source='stage1_flag_marker'
    but omitted kind='stage1_flag_marker', breaking dual-filter queries that key
    on both fields.  This test drives a fresh-signature write through dedup_flags
    and asserts both keys are present in the ledger row's payload.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [{'task_id': '7', 'flag_type': 'missing_deliverable', 'description': 'x'}]
    await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='proj',
        run_id='r99',
        flags=flags,
    )

    row = await _get_marker(ledger_memory_service.recon_ledger, 'proj', '7', 'missing_deliverable')
    assert row is not None
    payload = json.loads(row.payload_json)
    assert payload.get('source') == 'stage1_flag_marker', (
        f"payload.source must be 'stage1_flag_marker', got: {payload!r}"
    )
    assert payload.get('kind') == 'stage1_flag_marker', (
        f"payload.kind must be 'stage1_flag_marker', got: {payload!r}"
    )


@pytest.mark.parametrize(
    'malformed_run_id',
    [
        pytest.param(None, id='run_id-is-None'),
        pytest.param('', id='run_id-is-empty-string'),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_with_malformed_run_id_uses_sentinel(
    malformed_run_id, ledger_memory_service, caplog
):
    """When a prior stage1_flag_marker row exists but its payload run_id is
    falsy, dedup_flags must annotate the flag with persisted_from_run='unknown'
    — not the current run_id.

    Also asserts that the sentinel-collapse path emits a DEBUG log.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger = ledger_memory_service.recon_ledger
    await _seed_marker(
        ledger, 'p', '42', 'missing_deliverable',
        run_id=malformed_run_id if malformed_run_id else '',
        extra_payload={'run_id': malformed_run_id},
    )

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'unknown', (
        f"persisted_from_run must fall back to sentinel 'unknown' for any falsy run_id "
        f"in the prior marker's payload, but got {result[0].get('persisted_from_run')!r}."
    )
    assert result[0]['last_seen_run_id'] == 'r1'

    # The row is re-upserted with the current run_id even when the prior run_id
    # was malformed (annotation sentinel does not block the write).
    row = await _get_marker(ledger, 'p', '42', 'missing_deliverable')
    assert row is not None
    payload = json.loads(row.payload_json)
    assert payload['run_id'] == 'r1'

    # Sentinel-collapse path emits a DEBUG log.
    assert any(
        '42' in record.message
        and 'missing_deliverable' in record.message
        and ('unknown' in record.message or 'malformed' in record.message)
        and record.levelno == logging.DEBUG
        for record in caplog.records
    ), (
        f"Expected a DEBUG log mentioning task_id='42', flag_type='missing_deliverable', "
        f"and 'unknown'/'malformed', but got records: {[r.message for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# Ledger-write-is-authoritative / mirror-is-best-effort (task 2227)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_retires_mem0_marker_mirror(ledger_memory_service):
    """The best-effort Mem0 ``stage1_flag_marker`` mirror write is retired
    (task 2406, option (a)): the ``recon_ledger`` row remains the sole store
    for an ordinary recurring flag's marker — upserted exactly as before —
    but ``dedup_flags`` no longer calls ``add_memory`` to mirror it to Mem0.
    The ledger row is already reaped each cycle by
    ``ReconLedgerStore.gc()``, so no periodic collector is needed for it.

    A second cycle of the SAME signature is included (amendment,
    reviewer_comprehensive finding #1) to pin the read-FROM-ledger dedup
    path specifically: ``persisted_from_run`` must still be annotated from
    the ledger row the first cycle wrote, with the mirror staying uncalled
    across both cycles — there is no Mem0 fallback of any kind involved.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [{'task_id': '77', 'flag_type': 'missing_deliverable', 'description': 'x'}]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    assert len(result) == 1

    # (a) The ledger row is still upserted — the ledger remains authoritative.
    row = await _get_marker(ledger_memory_service.recon_ledger, 'p', '77', 'missing_deliverable')
    assert row is not None
    assert row.state == 'active'

    # (b) The best-effort Mem0 marker mirror is retired — never called.
    ledger_memory_service.add_memory.assert_not_called()

    # (c) A second cycle of the same signature reads persisted_from_run back
    # from the ledger row alone — the only possible source now that no Mem0
    # fallback exists — and the mirror stays uncalled across both cycles.
    result2 = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r2',
        flags=flags,
    )
    assert result2[0]['persisted_from_run'] == 'r1'
    assert result2[0]['last_seen_run_id'] == 'r2'
    ledger_memory_service.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_dedup_flags_marker_noop_when_ledger_unset():
    """When ``memory_service.recon_ledger`` is unset/``None`` (ledger disabled
    or not yet wired), the marker path degrades to a pure no-op (amendment,
    reviewer_comprehensive finding #2) rather than falling back to a Mem0
    mirror write — there is no longer a mirror fallback for markers. The flag
    is returned unchanged (no ``persisted_from_run`` can be computed without a
    ledger to read), ``dedup_flags`` does not raise, and ``add_memory`` is
    never called.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.recon_ledger = None

    flags = [{'task_id': '77', 'flag_type': 'missing_deliverable', 'description': 'x'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    assert len(result) == 1
    assert result[0]['task_id'] == '77'
    assert 'persisted_from_run' not in result[0]
    # last_seen_run_id requires no ledger read, so it is still stamped.
    assert result[0]['last_seen_run_id'] == 'r1'

    memory_service.add_memory.assert_not_called()


# ---------------------------------------------------------------------------
# Ledger read/write is ALSO best-effort, not just the Mem0 mirror (amendment
# round — reviewer_comprehensive findings #1/#2). dedup_flags' own docstring
# and public-API contract promise "best-effort (exceptions are logged, not
# raised)"; these pin that the ledger get_by_identity/json.loads/upsert
# sequence honors that contract, mirroring the mirror-write guard above.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_ledger_get_by_identity_exception_does_not_raise(
    ledger_memory_service, caplog
):
    """A raising recon_ledger.get_by_identity must not abort dedup_flags.

    Amendment (reviewer_comprehensive finding #1/#2): the ledger read/write
    path was unguarded, so any exception from it — not just the Mem0 mirror —
    would propagate out of dedup_flags and fail the whole Stage-1 batch. This
    pins the best-effort contract dedup_flags' own docstring promises.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger_memory_service.recon_ledger.get_by_identity = AsyncMock(
        side_effect=RuntimeError('get_by_identity boom')
    )

    flags = [{'task_id': '77', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # Does not raise; flag still returned with no ledger-derived annotation.
    assert len(result) == 1
    assert result[0]['task_id'] == '77'
    assert 'persisted_from_run' not in result[0]

    # WARNING log mentions the failure and task_id.
    assert any(
        '77' in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    ), f'Expected a WARNING log mentioning task_id=77, got: {[r.message for r in caplog.records]}'


@pytest.mark.asyncio
async def test_dedup_flags_ledger_upsert_exception_does_not_raise(
    ledger_memory_service, caplog
):
    """A raising recon_ledger.upsert must not abort dedup_flags — the flag
    is still returned (without a persisted ledger row) and a WARNING logged.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger_memory_service.recon_ledger.upsert = AsyncMock(side_effect=RuntimeError('upsert boom'))

    flags = [{'task_id': '78', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    assert len(result) == 1
    assert result[0]['task_id'] == '78'

    # No row was actually persisted, since the upsert raised.
    row = await _get_marker(ledger_memory_service.recon_ledger, 'p', '78', 'missing_deliverable')
    assert row is None

    assert any(
        '78' in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    ), f'Expected a WARNING log mentioning task_id=78, got: {[r.message for r in caplog.records]}'


@pytest.mark.asyncio
async def test_dedup_flags_corrupt_prior_payload_json_does_not_raise(
    ledger_memory_service, caplog
):
    """A prior stage1_flag_marker row with non-JSON payload_json (legacy row,
    partial write, external corruption) must not poison dedup for its flag —
    json.loads(prior.payload_json) is inside the same best-effort try/except
    as the rest of the ledger read/write sequence.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger = ledger_memory_service.recon_ledger
    await ledger.upsert(ReconLedgerRecord(
        project_id='p',
        record_kind='stage1_flag_marker',
        payload_json='{not valid json',
        state='active',
        created_at='2026-01-01T00:00:00+00:00',
        task_id='79',
        flag_type='missing_deliverable',
        run_id='',
        expires_at='2099-01-01T00:00:00+00:00',
    ))

    flags = [{'task_id': '79', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r2',
            flags=flags,
        )

    # Does not raise; flag returned without a persisted_from_run (the read
    # failed, so no prior run_id could be extracted).
    assert len(result) == 1
    assert result[0]['task_id'] == '79'
    assert 'persisted_from_run' not in result[0]

    assert any(
        '79' in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    ), f'Expected a WARNING log mentioning task_id=79, got: {[r.message for r in caplog.records]}'


@pytest.mark.asyncio
async def test_dedup_flags_multiple_flags_one_ledger_failure_does_not_poison_batch(
    ledger_memory_service, caplog
):
    """A single flag's ledger failure must not prevent OTHER flags in the same
    batch from being processed normally — the try/except is per-flag, not
    around the whole loop.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger = ledger_memory_service.recon_ledger
    real_get_by_identity = ledger.get_by_identity

    async def _flaky_get_by_identity(project_id, record_kind, task_id, flag_type, run_id):
        if task_id == '80':
            raise RuntimeError('get_by_identity boom for 80')
        return await real_get_by_identity(project_id, record_kind, task_id, flag_type, run_id)

    ledger.get_by_identity = AsyncMock(side_effect=_flaky_get_by_identity)

    flags = [
        {'task_id': '80', 'flag_type': 'missing_deliverable', 'description': 'poisoned'},
        {'task_id': '81', 'flag_type': 'missing_deliverable', 'description': 'healthy'},
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # Restore the real method before reading back rows for assertions — the
    # flaky wrapper above still raises for task_id '80'.
    ledger.get_by_identity = real_get_by_identity

    assert len(result) == 2

    # Task 80's ledger op failed — no row persisted.
    row_80 = await _get_marker(ledger_memory_service.recon_ledger, 'p', '80', 'missing_deliverable')
    assert row_80 is None

    # Task 81 was processed normally despite task 80's failure.
    row_81 = await _get_marker(ledger_memory_service.recon_ledger, 'p', '81', 'missing_deliverable')
    assert row_81 is not None


# ---------------------------------------------------------------------------
# TestFilterSuppressed (task-1186 step-1; rewritten onto the ledger at task
# 2227 step-4) — filter_suppressed reads the ReconLedgerStore's indexed
# list_suppressions(project_id) query; no Mem0 search.
# ---------------------------------------------------------------------------


class TestFilterSuppressed:
    """Tests for filter_suppressed(memory_service, project_id, flags).

    filter_suppressed reads memory_service.recon_ledger.list_suppressions(
    project_id) — an indexed (project_id, record_kind, state) query — instead
    of issuing a Mem0 search (task 2227 step-4).
    """

    @pytest.mark.asyncio
    async def test_empty_flags_returns_empty_no_io(self, ledger_memory_service):
        """(a) Empty flags input → empty result + zero I/O (ledger never queried)."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        result = await filter_suppressed(ledger_memory_service, 'p', [])
        assert result == []
        ledger_memory_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_suppression_rows_all_flags_pass_through(self, ledger_memory_service):
        """(b) No suppression rows in the ledger → all flags pass through unchanged."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 99, 'flag_type': 'stale_metadata'},
        ]
        result = await filter_suppressed(ledger_memory_service, 'p', flags)
        assert result == flags

    @pytest.mark.asyncio
    async def test_never_calls_memory_service_search(self, ledger_memory_service):
        """filter_suppressed must never call memory_service.search — the
        ledger's indexed query fully replaces the old project-wide Mem0 search."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        await _seed_suppression(ledger_memory_service.recon_ledger, 'p', '42', '')
        await filter_suppressed(ledger_memory_service, 'p', [{'task_id': 42, 'flag_type': 'x'}])
        ledger_memory_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_list_suppressions_exception_returns_flags_unchanged(
        self, ledger_memory_service, caplog
    ):
        """A raising recon_ledger.list_suppressions must not abort the caller.

        Amendment (reviewer_comprehensive finding #1/#2): filter_suppressed's
        docstring promises a conservative "no suppression in effect"
        pass-through contract; a ledger read failure must degrade the same
        way the old Mem0-search-exception path did, with a WARNING logged
        instead of the exception propagating into dedup_flags' batch loop.
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        ledger_memory_service.recon_ledger.list_suppressions = AsyncMock(
            side_effect=RuntimeError('list_suppressions boom')
        )

        flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 99, 'flag_type': 'stale_metadata'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await filter_suppressed(ledger_memory_service, 'p', flags)

        assert result == flags
        assert any(
            'list_suppressions' in record.message and record.levelno == logging.WARNING
            for record in caplog.records
        ), (
            f'Expected a WARNING log mentioning list_suppressions, got: '
            f'{[r.message for r in caplog.records]}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'flag_task_id,record_task_id',
        [
            pytest.param(42, 42, id='int-int'),
            pytest.param(42, '42', id='int-str'),
            pytest.param('42', 42, id='str-int'),
            pytest.param('42', '42', id='str-str'),
        ],
    )
    async def test_suppressed_flag_dropped_and_unrelated_kept(
        self, ledger_memory_service, flag_task_id, record_task_id
    ):
        """(d) blanket suppression row with matching task_id drops the flag;
        unrelated flags kept.

        Parametrized: int-int, int-str, str-int, str-str coercion combinations
        so that symmetric str() coercion is verified on both sides (the
        ledger's task_id column is always TEXT; _seed_suppression stringifies
        its task_id argument the same way write_suppression_record will).
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        await _seed_suppression(ledger_memory_service.recon_ledger, 'p', record_task_id, '')

        suppressed_flag = {'task_id': flag_task_id, 'flag_type': 'missing_deliverable'}
        unrelated_flag = {'task_id': 99, 'flag_type': 'stale_metadata'}
        flags = [suppressed_flag, unrelated_flag]

        result = await filter_suppressed(ledger_memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == unrelated_flag

    @pytest.mark.asyncio
    async def test_row_with_empty_task_id_ignored(self, ledger_memory_service):
        """(f) A suppression row with an empty task_id is ignored — it can
        never match a flag (mirrors the old producer-side empty-task_id guard)."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        await _seed_suppression(ledger_memory_service.recon_ledger, 'p', '', '')

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(ledger_memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'seeded_task_id,label',
        [
            (None, 'empty_set'),
            ('None', 'none_string_in_set'),
            ('42', 'other_valid_id_in_set'),
        ],
        ids=['empty_set', 'none_string_in_set', 'other_valid_id_in_set'],
    )
    async def test_flag_with_none_task_id_never_dropped(
        self, ledger_memory_service, seeded_task_id, label
    ):
        """(h) flag with task_id=None is never suppressed regardless of ledger contents.

        The consumer-side guard short-circuits to 'keep' when flag_tid is
        None, regardless of what rows exist in the ledger.  Three cases:
        - empty_set: trivially preserved (no suppression rows at all).
        - none_string_in_set: a row with the literal string task_id='None'
          exists; without the consumer guard, str(None) == 'None' would drop
          the flag — this is the key regression scenario.
        - other_valid_id_in_set: a real suppression row (task_id='42') must
          not suppress a flag whose task_id is None.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        if seeded_task_id is not None:
            await _seed_suppression(ledger_memory_service.recon_ledger, 'p', seeded_task_id, '')

        flag_with_none_task_id = {'task_id': None, 'flag_type': 'missing_deliverable'}
        result = await filter_suppressed(ledger_memory_service, 'p', [flag_with_none_task_id])

        assert result == [flag_with_none_task_id], (
            f'[{label}] Expected flag with task_id=None to be preserved but got: {result}'
        )

    # -----------------------------------------------------------------
    # Scoped (task_id, flag_type) suppression — task-1966, ledger-backed
    # -----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_scoped_row_drops_only_matching_flag_type(self, ledger_memory_service):
        """(a) A SCOPED suppression row drops only its flag_type, keeping
        other flag_types for the same task_id — the core fix."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        await _seed_suppression(
            ledger_memory_service.recon_ledger, 'p', '452', 'human_review_required_deferred'
        )

        suppressed_flag = {'task_id': 452, 'flag_type': 'human_review_required_deferred'}
        surviving_flag = {'task_id': 452, 'flag_type': 'live_workflow_recurrence_counter_needed'}
        result = await filter_suppressed(
            ledger_memory_service, 'p', [suppressed_flag, surviving_flag]
        )

        assert result == [surviving_flag]

    @pytest.mark.asyncio
    async def test_wildcard_row_still_blanket_drops_all_flag_types(self, ledger_memory_service):
        """(b) A blanket/wildcard row (flag_type='') still drops ALL
        flag_types for that task_id — backward-compat with the legacy shape."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        await _seed_suppression(ledger_memory_service.recon_ledger, 'p', '452', '')

        flag_a = {'task_id': 452, 'flag_type': 'human_review_required_deferred'}
        flag_b = {'task_id': 452, 'flag_type': 'live_workflow_recurrence_counter_needed'}
        result = await filter_suppressed(ledger_memory_service, 'p', [flag_a, flag_b])

        assert result == []

    @pytest.mark.asyncio
    async def test_absent_flag_type_kept_against_scoped_but_dropped_against_wildcard(
        self, ledger_memory_service, tmp_path
    ):
        """(c) A flag whose flag_type is None/absent cannot match a specific
        flag_type, so it is KEPT against a scoped row — but a wildcard row
        still drops it (no flag_type to check against)."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        flag_no_type = {'task_id': 452}  # flag_type key intentionally absent

        await _seed_suppression(
            ledger_memory_service.recon_ledger, 'p', '452', 'human_review_required_deferred'
        )
        result = await filter_suppressed(ledger_memory_service, 'p', [flag_no_type])
        assert result == [flag_no_type], 'flag_type=None cannot match a scoped allowlist'

        wildcard_ledger = ReconLedgerStore(tmp_path / 'wildcard_reconciliation.db')
        await wildcard_ledger.initialize()
        try:
            await _seed_suppression(wildcard_ledger, 'p', '452', '')
            wildcard_service = AsyncMock()
            wildcard_service.recon_ledger = wildcard_ledger
            result2 = await filter_suppressed(wildcard_service, 'p', [flag_no_type])
            assert result2 == [], 'wildcard row must still drop a flag with no flag_type'
        finally:
            await wildcard_ledger.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'flag_task_id,record_task_id',
        [
            pytest.param(452, 452, id='int-int'),
            pytest.param(452, '452', id='int-str'),
            pytest.param('452', 452, id='str-int'),
            pytest.param('452', '452', id='str-str'),
        ],
    )
    async def test_scoped_row_task_id_coercion(
        self, ledger_memory_service, flag_task_id, record_task_id
    ):
        """(d) task_id str/int coercion still applies with scoped rows."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        await _seed_suppression(
            ledger_memory_service.recon_ledger, 'p', record_task_id, 'human_review_required_deferred'
        )

        suppressed_flag = {'task_id': flag_task_id, 'flag_type': 'human_review_required_deferred'}
        result = await filter_suppressed(ledger_memory_service, 'p', [suppressed_flag])
        assert result == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'record_order',
        [
            pytest.param('scoped_first', id='scoped-then-wildcard'),
            pytest.param('wildcard_first', id='wildcard-then-scoped'),
        ],
    )
    async def test_scoped_and_wildcard_rows_for_same_task_id_union_to_blanket(
        self, ledger_memory_service, record_order
    ):
        """(e) One scoped + one wildcard row for the same task_id results in
        blanket suppression (union semantics: wildcard wins), regardless of
        upsert/iteration order — SQLite gives no ordering guarantee without
        ORDER BY, so both orders occur in practice.

        'wildcard-then-scoped' exercises the wildcard-already-present
        short-circuit branch (``if tid_str in suppressed and
        suppressed[tid_str] is None: continue``), which 'scoped-then-wildcard'
        never reaches (the scoped row's ``set`` is simply overwritten by the
        wildcard row in that order)."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        ledger = ledger_memory_service.recon_ledger
        if record_order == 'scoped_first':
            await _seed_suppression(ledger, 'p', '452', 'human_review_required_deferred')
            await _seed_suppression(ledger, 'p', '452', '')
        else:
            await _seed_suppression(ledger, 'p', '452', '')
            await _seed_suppression(ledger, 'p', '452', 'human_review_required_deferred')

        flag_a = {'task_id': 452, 'flag_type': 'human_review_required_deferred'}
        flag_b = {'task_id': 452, 'flag_type': 'live_workflow_recurrence_counter_needed'}
        result = await filter_suppressed(ledger_memory_service, 'p', [flag_a, flag_b])
        assert result == []


# ---------------------------------------------------------------------------
# task-1186 step-5 — integration: dedup_flags calls filter_suppressed FIRST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_calls_filter_suppressed_before_signature_dedup(ledger_memory_service):
    """Integration: dedup_flags calls filter_suppressed BEFORE the signature-dedup loop.

    A blanket suppression row for task_id=42 is seeded in the ledger. Two
    flags are processed: task_id=42 (suppressed) and task_id=99 (not
    suppressed), both with flag_type='missing_deliverable'.

    Expected outcomes:
    (a) Result has exactly one item — the task_id=99 flag.
    (b) task_id=42 flag was dropped.
    (c) A ledger marker row is created only for the surviving task_id=99 flag.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger = ledger_memory_service.recon_ledger
    await _seed_suppression(ledger, 'p', '42', '')

    flags = [
        {'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'suppressed'},
        {'task_id': 99, 'flag_type': 'missing_deliverable', 'description': 'survivor'},
    ]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) Only one flag survived
    assert len(result) == 1
    # (b) task_id=42 was dropped
    assert result[0].get('task_id') == 99, (
        f"Expected surviving flag task_id=99 but got {result[0].get('task_id')!r}"
    )
    # (c) A marker row exists only for the surviving task_id=99 flag
    assert await _get_marker(ledger, 'p', '99', 'missing_deliverable') is not None
    assert await _get_marker(ledger, 'p', '42', 'missing_deliverable') is None


@pytest.mark.asyncio
async def test_dedup_flags_distinct_flag_types_same_task_each_write(ledger_memory_service):
    """Guard: two DISTINCT flag_types for the same task_id are independent
    signatures — each gets its own ledger row (dedup identity is the FULL
    (task_id, flag_type) tuple, not task_id alone).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [
        {'task_id': 1970, 'flag_type': 'task_blocked_stale_escalations', 'description': 'a'},
        {'task_id': 1970, 'flag_type': 'other_flag_type', 'description': 'b'},
    ]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r_now',
        flags=flags,
    )

    assert len(result) == 2
    ledger = ledger_memory_service.recon_ledger
    assert await _get_marker(ledger, 'p', '1970', 'task_blocked_stale_escalations') is not None
    assert await _get_marker(ledger, 'p', '1970', 'other_flag_type') is not None


@pytest.mark.asyncio
async def test_dedup_flags_none_signature_flags_not_collapsed_when_repeated():
    """Guard: two flags with no computable signature (task_id=None, no
    cited_tasks, blank description) must both pass through unchanged — the
    in-batch memo must never trigger for None-signature flags.

    Green both before and after the task-1978 fix — pins that the memo is
    consulted only AFTER the None-signature pass-through, so repeated
    unsignable flags are never collapsed together.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])

    flag = {'task_id': None, 'flag_type': 'x', 'description': '   '}
    flags = [dict(flag), dict(flag)]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r_now',
        flags=flags,
    )

    assert len(result) == 2, (
        f'None-signature flags must never be collapsed by the in-batch memo; '
        f'expected 2 flags but got {len(result)}'
    )
    assert result == flags
    memory_service.add_memory.assert_not_called()


@pytest.mark.asyncio
async def test_dedup_flags_invalid_tid_flags_not_collapsed_when_repeated():
    """Guard: two flags sharing the same invalid task_id (fails
    _is_valid_marker_task_id) must both pass through unchanged — the in-batch
    memo must sit AFTER the invalid-tid guard so invalid-tid flags are never
    collapsed together.

    Green both before and after the task-1978 fix.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])

    flag = {'task_id': 'abc', 'flag_type': 'x', 'description': 'd'}
    flags = [dict(flag), dict(flag)]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r_now',
        flags=flags,
    )

    assert len(result) == 2, (
        f'Invalid-tid flags must never be collapsed by the in-batch memo; '
        f'expected 2 flags but got {len(result)}'
    )
    assert result == flags
    memory_service.add_memory.assert_not_called()


# ---------------------------------------------------------------------------
# build_suppression_payload tests (task-1185 step-1)
# ---------------------------------------------------------------------------


class TestBuildSuppressionPayload:
    """Unit tests for build_suppression_payload(task_id) -> SuppressionPayload.

    Pure dict assertions — no I/O, no async.
    """

    _CANONICAL = {
        'content': 'STAGE 1 FLAG SUPPRESSION task_id=42',
        'category': 'observations_and_summaries',
        'metadata': {'kind': 'stage1_flag_suppression', 'task_id': 42},
    }

    def test_full_payload_for_int_task_id(self):
        """Full payload dict equals the canonical schema literal for int input.

        Implicitly asserts: content, category, metadata.kind, metadata.task_id
        (int), and absence of project_id (not in canonical schema).
        """
        assert build_suppression_payload(42) == self._CANONICAL

    def test_coerces_str_task_id_to_int(self):
        """str task_id is coerced to int; resulting payload equals canonical schema."""
        assert build_suppression_payload('42') == self._CANONICAL

    def test_invalid_task_id_raises_descriptive_value_error(self):
        """Non-numeric task_id raises ValueError with function-name and bad-value context.

        The bare int() raises a context-free ValueError; the hardened implementation
        must wrap it with:
        - 'build_suppression_payload' in the error message (function-name context)
        - the bad value itself in the error message (bad-value context)
        - __cause__ set to the original ValueError/TypeError (from e chaining)
        """
        with pytest.raises(ValueError) as exc_info:
            build_suppression_payload('abc')

        error_message = str(exc_info.value)
        assert 'build_suppression_payload' in error_message, (
            f"Expected 'build_suppression_payload' in error message but got: {error_message!r}"
        )
        assert 'abc' in error_message, (
            f"Expected bad value 'abc' in error message but got: {error_message!r}"
        )
        assert exc_info.value.__cause__ is not None, (
            'Expected __cause__ to be set (from e chaining) but it was None'
        )
        assert isinstance(exc_info.value.__cause__, (ValueError, TypeError)), (
            f'Expected __cause__ to be ValueError or TypeError but got: {type(exc_info.value.__cause__)}'
        )


# ---------------------------------------------------------------------------
# build_suppression_payload — optional flag_types allowlist (task-1966 step-1)
#
# RED until step-2 adds the flag_types param.  All TestBuildSuppressionPayload
# assertions above (legacy shape) must remain green throughout.
# ---------------------------------------------------------------------------


class TestBuildSuppressionPayloadFlagTypes:
    """Tests for the OPTIONAL flag_types scoping allowlist (task-1966)."""

    def test_legacy_call_has_no_flag_types_key(self):
        """(a) build_suppression_payload(42) still equals the canonical dict —
        NO 'flag_types' key in metadata (backward compat)."""
        result = build_suppression_payload(42)
        assert result == {
            'content': 'STAGE 1 FLAG SUPPRESSION task_id=42',
            'category': 'observations_and_summaries',
            'metadata': {'kind': 'stage1_flag_suppression', 'task_id': 42},
        }
        assert 'flag_types' not in result['metadata']

    def test_scoped_call_includes_flag_types_in_metadata(self):
        """(b) Non-empty flag_types produces metadata.task_id (int-coerced) AND
        metadata.flag_types (list[str]); content is still the canonical
        non-empty 'STAGE 1 FLAG SUPPRESSION task_id=452...' string."""
        result = build_suppression_payload(452, flag_types=['human_review_required_deferred'])
        assert result['metadata']['task_id'] == 452
        assert isinstance(result['metadata']['task_id'], int)
        # .get() (not ['flag_types']) — the key is NotRequired in _SuppressionMetadata,
        # so a direct subscript trips pyright's reportTypedDictNotRequiredAccess.
        assert result['metadata'].get('flag_types') == ['human_review_required_deferred']
        assert isinstance(result['content'], str) and result['content']
        assert result['content'].startswith('STAGE 1 FLAG SUPPRESSION task_id=452')

    def test_flag_types_normalized_sorted_unique_str_coerced(self):
        """(c) Mixed/duplicate/unsorted flag_types normalize to sorted-unique,
        every element str-coerced."""
        result = build_suppression_payload(452, flag_types=['b', 'a', 'a'])
        flag_types = result['metadata'].get('flag_types')
        assert flag_types == ['a', 'b']
        assert flag_types is not None  # narrows list[str] | None for the iteration below
        assert all(isinstance(x, str) for x in flag_types)

    @pytest.mark.parametrize('flag_types', [None, []], ids=['none', 'empty_list'])
    def test_none_or_empty_flag_types_yields_legacy_shape(self, flag_types):
        """(d) flag_types=[] or None yields the legacy no-flag_types shape."""
        result = build_suppression_payload(452, flag_types=flag_types)
        assert 'flag_types' not in result['metadata']


# ---------------------------------------------------------------------------
# Round-trip schema validation tests (task-1185 step-5)
#
# FakeMemoryService: in-test stub generalising the FakeMem0 pattern already
# used in test_dedup_flags_two_consecutive_runs_no_predecessor_accumulation.
# Accepts writer-supplied content and metadata so it works for arbitrary
# writers (flag-marker writes AND suppression-record writes).
# Kept module-private — promoting to conftest is a future refactor.
# ---------------------------------------------------------------------------

class _MemoryResultStub:
    """Minimal stand-in for a Mem0 MemoryResult, accepts explicit content."""

    def __init__(self, id_: str, content: str, metadata: dict) -> None:
        self.id = id_
        self.content = content
        self.metadata = metadata


class _FakeMemoryService:
    """In-memory memory service stub: add_memory, search, delete_memory.

    Stores records keyed by UUID.  search() applies minimal filtering:

    * When ``query`` is provided it is matched against ``metadata['kind']``
      exactly — records whose kind does not equal query are excluded.  This
      catches drift between the producer's ``metadata.kind`` and the reader's
      search query (e.g. filter_suppressed passes ``query='stage1_flag_suppression'``
      which must match the kind the producer wrote).
    * When ``categories`` is provided, only records whose stored category is in
      the list are returned.  ``categories=None`` (kwarg absent) means no category
      filter — all records pass.  This enforces the producer→reader category
      contract so that a category mismatch is caught by tests using this fake.
    * ``limit`` truncates the result list when provided.
    * All other kwargs (``stores``, ``project_id``) are accepted and ignored.

    This mirrors the existing FakeMem0 in the regression test but accepts
    writer-supplied content and metadata.
    """

    def __init__(self) -> None:
        self._store: dict[str, _MemoryResultStub] = {}
        self._categories: dict[str, str] = {}

    async def add_memory(
        self, *, content: str, category: str, metadata: dict, **_kwargs
    ) -> AddMemoryResponse:
        id_ = str(_uuid_mod.uuid4())
        self._store[id_] = _MemoryResultStub(id_, content, metadata)
        self._categories[id_] = category
        return AddMemoryResponse(memory_ids=[id_])

    async def search(
        self,
        *,
        query: str = '',
        limit: int | None = None,
        categories: list[str] | None = None,
        **_kwargs,
    ) -> list[_MemoryResultStub]:
        results = list(self._store.values())
        if categories is not None:
            results = [r for r in results if self._categories.get(r.id) in categories]
        if query:
            results = [r for r in results if r.metadata.get('kind') == query]
        if limit is not None:
            results = results[:limit]
        return results

    async def delete_memory(self, *, memory_id: str, **_kwargs) -> None:
        self._store.pop(memory_id, None)
        self._categories.pop(memory_id, None)

    def count(self) -> int:
        return len(self._store)


@pytest.mark.asyncio
async def test_fake_memory_service_search_filters_by_categories_kwarg():
    """_FakeMemoryService enforces category routing when categories kwarg is provided.

    Asserts:
    (a) A record stored with category='observations_and_summaries' is found when
        searching with categories=['observations_and_summaries'] (matching category).
    (b) The same record is NOT found when searching with categories=['preferences_and_norms']
        (category mismatch → 0 results).
    (c) The same record IS found when no categories kwarg is provided (absent kwarg =
        no category filter).

    Pins the contract that _FakeMemoryService applies the categories filter
    (case (b) would return 1 instead of 0 if the filter were dropped).
    """
    fake = _FakeMemoryService()
    await fake.add_memory(
        content='x',
        category='observations_and_summaries',
        metadata={'kind': 'stage1_flag_suppression', 'task_id': 1},
    )

    # (a) Matching category → 1 result
    results_matching = await fake.search(
        query='stage1_flag_suppression',
        categories=['observations_and_summaries'],
    )
    assert len(results_matching) == 1, (
        f'Expected 1 result for matching category but got {len(results_matching)}'
    )

    # (b) Mismatched category → 0 results
    results_mismatch = await fake.search(
        query='stage1_flag_suppression',
        categories=['preferences_and_norms'],
    )
    assert len(results_mismatch) == 0, (
        f'Expected 0 results for mismatched category but got {len(results_mismatch)}'
    )

    # (c) No categories kwarg → no filter, 1 result
    results_no_filter = await fake.search(query='stage1_flag_suppression')
    assert len(results_no_filter) == 1, (
        f'Expected 1 result with no categories filter but got {len(results_no_filter)}'
    )


@pytest.mark.asyncio
async def test_suppression_record_round_trips_via_producer():
    """Round-trip schema contract (producer path): write_suppression_record writes
    exactly what filter_suppressed's search finds.

    Exercises ONLY the canonical producer path (int task_id → write_suppression_record).
    A regression in write_suppression_record's int(task_id) coercion would be caught here.

    Canonical schema (four-line contract):
      1. metadata.kind == 'stage1_flag_suppression'
      2. metadata.task_id == 42 (int)
      3. content == 'STAGE 1 FLAG SUPPRESSION task_id=42'
      4. category == 'observations_and_summaries'

    The search kwargs match filter_suppressed's actual call shape exactly so
    that any future drift between the producer and the search-call contract is
    caught here first.
    """
    from fused_memory.reconciliation.flag_dedup import write_suppression_record

    raw_task_id = 42
    expected_content = f'STAGE 1 FLAG SUPPRESSION task_id={raw_task_id}'
    fake = _FakeMemoryService()

    # Write via the canonical producer (coerces to int).
    await write_suppression_record(fake, project_id='dark_factory', task_id=raw_task_id)

    # Search with the kwargs filter_suppressed actually uses (task-1186)
    results = await fake.search(
        query='stage1_flag_suppression',
        project_id='dark_factory',
        categories=['observations_and_summaries'],
        stores=['mem0'],
        limit=500,
    )

    # (4) Exactly one result
    assert len(results) == 1, f'Expected 1 result but got {len(results)}: {results}'

    # (5) Canonical kind key
    assert results[0].metadata['kind'] == 'stage1_flag_suppression', (
        f'metadata.kind mismatch: {results[0].metadata}'
    )

    # (6) task_id is stored as int by the producer
    assert results[0].metadata['task_id'] == raw_task_id, (
        f'metadata.task_id mismatch: {results[0].metadata["task_id"]!r}'
    )

    # (7) Canonical content
    assert results[0].content == expected_content, (
        f'content mismatch: {results[0].content!r}'
    )


@pytest.mark.asyncio
async def test_legacy_str_suppression_record_round_trips_through_consumer_search():
    """Round-trip schema contract (legacy-str consumer path): a hand-authored or
    legacy record with str task_id is found by filter_suppressed's search.

    Exercises ONLY the legacy-str path: task_id stored as str '42' via
    fake.add_memory directly, modelling records written before int coercion was
    enforced.  The reader's str-coercion makes the record visible to filter_suppressed.

    Canonical schema assertions use str-coercion so the test accurately reflects
    what a str-task_id record actually looks like in Mem0.
    """
    raw_task_id = '42'
    tid_int = int(raw_task_id)
    expected_content = f'STAGE 1 FLAG SUPPRESSION task_id={tid_int}'
    fake = _FakeMemoryService()

    # Model the legacy/hand-authored str-task_id storage shape directly.
    await fake.add_memory(
        content=expected_content,
        category='observations_and_summaries',
        project_id='dark_factory',
        metadata={'kind': 'stage1_flag_suppression', 'task_id': raw_task_id},
    )

    # Search with the kwargs filter_suppressed actually uses (task-1186)
    results = await fake.search(
        query='stage1_flag_suppression',
        project_id='dark_factory',
        categories=['observations_and_summaries'],
        stores=['mem0'],
        limit=500,
    )

    # (4) Exactly one result
    assert len(results) == 1, f'Expected 1 result but got {len(results)}: {results}'

    # (5) Canonical kind key
    assert results[0].metadata['kind'] == 'stage1_flag_suppression', (
        f'metadata.kind mismatch: {results[0].metadata}'
    )

    # (6) task_id round-trips via str-coercion (legacy records store as str)
    assert str(results[0].metadata['task_id']) == str(tid_int), (
        f'metadata.task_id mismatch: {results[0].metadata["task_id"]!r}'
    )

    # (7) Canonical content
    assert results[0].content == expected_content, (
        f'content mismatch: {results[0].content!r}'
    )


@pytest.mark.asyncio
async def test_write_suppression_record_propagates_invalid_task_id():
    """Pins that ValueError from build_suppression_payload propagates out of the
    public write_suppression_record entry point unchanged.

    Regression pin: a future producer change that wraps build_suppression_payload
    in a swallow-all try/except would silently break invalid-task_id detection.
    The existing test_invalid_task_id_raises_descriptive_value_error covers
    build_suppression_payload directly; this test closes the public-API gap.
    """
    from fused_memory.reconciliation.flag_dedup import write_suppression_record

    fake = _FakeMemoryService()
    with pytest.raises(ValueError):
        await write_suppression_record(fake, project_id='p', task_id='abc')


# ---------------------------------------------------------------------------
# TestWriteSuppressionRecord (task-1185 step-3; rewritten onto the ledger at
# task 2227 step-6) — write_suppression_record upserts stage1_flag_suppression
# row(s) to memory_service.recon_ledger AND best-effort mirrors the same
# payload to Mem0 via add_memory.
# ---------------------------------------------------------------------------


class TestWriteSuppressionRecord:
    """Async tests for write_suppression_record(memory_service, *, project_id, task_id, causation_id).

    Uses the ``ledger_memory_service`` fixture (prereq-1) throughout — a REAL
    initialized ReconLedgerStore plus a mockable ``add_memory`` Mem0 mirror.
    The canonical mirror-payload, project_id, _source, and default-causation_id
    assertions are consolidated into one test that inspects the full kwargs
    dict; separate small tests cover the causation_id/str-coercion variants
    and the ledger-row invariants (task 2227 step-6).
    """

    @pytest.mark.asyncio
    async def test_canonical_mirror_call_kwargs(self, ledger_memory_service):
        """add_memory mirror called once with the full canonical kwargs dict.

        Asserts content, category, metadata (kind + int task_id), project_id,
        _source sentinel, and causation_id=None (default) in a single call.
        """
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        result = await write_suppression_record(
            ledger_memory_service, project_id='autopilot_video', task_id=42
        )

        ledger_memory_service.add_memory.assert_called_once_with(
            content='STAGE 1 FLAG SUPPRESSION task_id=42',
            category='observations_and_summaries',
            metadata={'kind': 'stage1_flag_suppression', 'task_id': 42},
            project_id='autopilot_video',
            causation_id=None,
            _source='stage1_flag_suppression',
        )
        assert result.memory_ids == ['mirror-id']

    @pytest.mark.asyncio
    async def test_passes_causation_id_when_provided(self, ledger_memory_service):
        """causation_id is forwarded to the add_memory mirror when explicitly provided."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(
            ledger_memory_service, project_id='p', task_id=42, causation_id='recon-run-99'
        )

        kwargs = ledger_memory_service.add_memory.call_args.kwargs
        assert kwargs['causation_id'] == 'recon-run-99'

    @pytest.mark.asyncio
    async def test_coerces_str_task_id(self, ledger_memory_service):
        """passing task_id='42' produces mirror metadata.task_id == 42 (int)
        and a ledger row keyed by the same coerced task_id."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(ledger_memory_service, project_id='p', task_id='42')

        kwargs = ledger_memory_service.add_memory.call_args.kwargs
        assert kwargs['metadata']['task_id'] == 42
        assert isinstance(kwargs['metadata']['task_id'], int)

        rows = await ledger_memory_service.recon_ledger.list_suppressions('p')
        assert rows[0].task_id == '42'

    @pytest.mark.asyncio
    async def test_forwards_flag_types_to_mirror_metadata(self, ledger_memory_service):
        """flag_types is forwarded into the mirror metadata.flag_types (task-1966 step-5)."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(
            ledger_memory_service,
            project_id='p',
            task_id=452,
            flag_types=['human_review_required_deferred'],
        )

        kwargs = ledger_memory_service.add_memory.call_args.kwargs
        assert kwargs['metadata']['flag_types'] == ['human_review_required_deferred']
        assert kwargs['metadata']['kind'] == 'stage1_flag_suppression'
        assert kwargs['metadata']['task_id'] == 452

    @pytest.mark.asyncio
    async def test_omitting_flag_types_produces_legacy_mirror_metadata(
        self, ledger_memory_service
    ):
        """Omitting flag_types produces mirror metadata with NO flag_types key (legacy)."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(ledger_memory_service, project_id='p', task_id=452)

        kwargs = ledger_memory_service.add_memory.call_args.kwargs
        assert 'flag_types' not in kwargs['metadata']

    # -----------------------------------------------------------------
    # Ledger-row invariants (task 2227 step-6)
    # -----------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_blanket_call_upserts_one_wildcard_row(self, ledger_memory_service):
        """task_id-only call upserts exactly one row: task_id='42', flag_type='',
        state='active', expires_at=None — readable via list_suppressions /
        is_suppressed."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(ledger_memory_service, project_id='p', task_id=42)

        rows = await ledger_memory_service.recon_ledger.list_suppressions('p')
        assert len(rows) == 1
        row = rows[0]
        assert row.task_id == '42'
        assert row.flag_type == ''
        assert row.state == 'active'
        assert row.expires_at is None

        assert (
            await ledger_memory_service.recon_ledger.is_suppressed('p', '42', 'anything')
            is True
        )

    @pytest.mark.asyncio
    async def test_flag_types_upserts_one_row_per_flag_type(self, ledger_memory_service):
        """flag_types=['a', 'b'] upserts one scoped row per flag_type."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(
            ledger_memory_service, project_id='p', task_id=7, flag_types=['a', 'b']
        )

        rows = await ledger_memory_service.recon_ledger.list_suppressions('p')
        assert len(rows) == 2
        assert {row.flag_type for row in rows} == {'a', 'b'}
        assert all(row.task_id == '7' for row in rows)

        assert await ledger_memory_service.recon_ledger.is_suppressed('p', '7', 'a') is True
        assert await ledger_memory_service.recon_ledger.is_suppressed('p', '7', 'c') is False

    @pytest.mark.asyncio
    async def test_repeated_identical_call_is_idempotent(self, ledger_memory_service):
        """A second identical call leaves the row count unchanged (UPSERT
        idempotency) — no duplicate row per recurrence."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(ledger_memory_service, project_id='p', task_id=42)
        await write_suppression_record(ledger_memory_service, project_id='p', task_id=42)

        rows = await ledger_memory_service.recon_ledger.list_suppressions('p')
        assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_written_rows_round_trip_through_filter_suppressed(
        self, ledger_memory_service
    ):
        """Rows written by write_suppression_record are read back by the
        ledger-backed filter_suppressed — end-to-end producer -> consumer
        round-trip."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_suppressed,
            write_suppression_record,
        )

        await write_suppression_record(ledger_memory_service, project_id='p', task_id=42)

        flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 99, 'flag_type': 'missing_deliverable'},
        ]
        result = await filter_suppressed(ledger_memory_service, 'p', flags)

        assert result == [{'task_id': 99, 'flag_type': 'missing_deliverable'}]

    @pytest.mark.asyncio
    async def test_invalid_task_id_raises_value_error_no_ledger_row(
        self, ledger_memory_service
    ):
        """Invalid task_id still raises ValueError (build_suppression_payload's
        guard is unchanged) and never writes a ledger row."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        with pytest.raises(ValueError):
            await write_suppression_record(
                ledger_memory_service, project_id='p', task_id='not-a-number'
            )

        rows = await ledger_memory_service.recon_ledger.list_suppressions('p')
        assert rows == []

    @pytest.mark.asyncio
    async def test_no_recon_ledger_degrades_to_mirror_only(self):
        """memory_service.recon_ledger unset/None skips the ledger write
        entirely and never raises — write_suppression_record degrades to a
        Mem0-only mirror write (mirrors filter_suppressed's pass-through
        contract when it finds no ledger to read)."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        svc = AsyncMock()
        svc.recon_ledger = None
        svc.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['supp-1']))

        result = await write_suppression_record(svc, project_id='p', task_id=42)

        svc.add_memory.assert_called_once()
        assert result.memory_ids == ['supp-1']


def test_write_suppression_record_importable_from_canonical_path():
    """Smoke test: write_suppression_record is importable from the path stage1.py advertises.

    If the helper is ever moved or renamed, this test fails CI rather than
    silently drifting the prompt's operator instructions
    (see STAGE1_SYSTEM_PROMPT ## Flag Suppression Check section).
    """
    from fused_memory.reconciliation.flag_dedup import write_suppression_record  # noqa: F401


# ---------------------------------------------------------------------------
# _extract_deduped_against_uuids helper (task-2047 step-1 / step-2)
# ---------------------------------------------------------------------------


class TestExtractDedupedAgainstUuids:
    """Unit tests for _extract_deduped_against_uuids(flag) -> list[str].

    Pure, sync, no-I/O helper that extracts the resolvable Mem0 memory
    UUID(s) a duplicate-detection finding cites as duplicates, from a
    documented set of candidate flag-dict fields (canonical
    'deduped_against' plus common aliases the LLM might use instead).
    """

    def test_reads_canonical_field_sorted_unique(self):
        """Canonical 'deduped_against' list is returned sorted-unique."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {'deduped_against': ['uuid-b', 'uuid-a', 'uuid-b']}
        assert _extract_deduped_against_uuids(flag) == ['uuid-a', 'uuid-b']

    def test_unions_alias_fields_with_canonical_field(self):
        """Values from alias fields are unioned together with the canonical field."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {
            'deduped_against': ['uuid-a'],
            'duplicate_memory_ids': ['uuid-b'],
            'duplicate_ids': ['uuid-c'],
            'memory_ids': ['uuid-d'],
            'cited_memory_ids': ['uuid-e'],
        }
        assert _extract_deduped_against_uuids(flag) == [
            'uuid-a', 'uuid-b', 'uuid-c', 'uuid-d', 'uuid-e',
        ]

    def test_accepts_single_str_value_not_wrapped_in_list(self):
        """A bare str value (not a list) on any candidate field is accepted as a 1-item result."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {'deduped_against': 'uuid-solo'}
        assert _extract_deduped_against_uuids(flag) == ['uuid-solo']

    def test_drops_none_and_blank_entries_and_coerces_non_str_to_str(self):
        """None/empty/whitespace-only entries are dropped; non-str entries are str-coerced."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {'deduped_against': [None, '', '   ', 'uuid-a', 123]}
        assert _extract_deduped_against_uuids(flag) == ['123', 'uuid-a']

    def test_drops_structured_elements_instead_of_stringifying(self):
        """A dict or nested-list element inside a candidate field is DROPPED,
        not str()-coerced into a junk 'UUID' (task-2047 amendment)."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {
            'deduped_against': [
                'uuid-a',
                {'k': 'v'},
                ['nested', 'list'],
                3.14,
                'uuid-b',
            ],
        }
        assert _extract_deduped_against_uuids(flag) == ['uuid-a', 'uuid-b']

    def test_returns_empty_list_when_no_candidate_field_present(self):
        """No candidate field present at all -> []."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {
            'task_id': None,
            'flag_type': 'duplicate_procedural_knowledge',
            'description': 'x',
        }
        assert _extract_deduped_against_uuids(flag) == []

    def test_returns_empty_list_when_candidate_fields_are_all_empty(self):
        """Candidate fields present but every value is empty/None -> []."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {'deduped_against': [], 'duplicate_ids': None, 'memory_ids': ''}
        assert _extract_deduped_against_uuids(flag) == []

    def test_returns_empty_list_for_bare_dict_with_unrelated_keys(self):
        """A flag dict carrying only unrelated keys -> [] (no KeyError/AttributeError)."""
        from fused_memory.reconciliation.flag_dedup import _extract_deduped_against_uuids

        flag = {'foo': 'bar', 'baz': [1, 2, 3]}
        assert _extract_deduped_against_uuids(flag) == []


# ---------------------------------------------------------------------------
# Step 7: confirm_task_absent fail-closed classifier (RED tests)
# ---------------------------------------------------------------------------


class TestConfirmTaskAbsent:
    """RED tests for confirm_task_absent(get_task_result) (step-7).

    confirm_task_absent is fail-closed: returns True ONLY when the result
    positively confirms the task does not exist (not-found error dict).
    Any present, inconclusive, or non-dict result returns False.
    """

    def test_true_for_not_found_error_dict(self):
        """Returns True when get_task returns the canonical not-found error dict.

        The sqlite backend raises TaskmasterError('TASKMASTER_TOOL_ERROR',
        'No tasks found for ID(s): N') on missing tasks; the server surfaces it
        as {'error': 'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): N',
            'error_type': 'TaskmasterError'}.
        """
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        not_found = {
            'error': 'TASKMASTER_TOOL_ERROR: No tasks found for ID(s): 3438',
            'error_type': 'TaskmasterError',
        }
        assert confirm_task_absent(not_found) is True, (
            'confirm_task_absent must return True for canonical not-found error dict'
        )

    def test_true_for_not_found_lowercase_variant(self):
        """Returns True for case-insensitive variant of not-found message."""
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        not_found = {
            'error': 'TASKMASTER_TOOL_ERROR: no tasks found for id(s): 42',
            'error_type': 'TaskmasterError',
        }
        assert confirm_task_absent(not_found) is True, (
            'confirm_task_absent must return True for case-insensitive not-found message'
        )

    def test_false_for_present_task_dict(self):
        """Returns False when get_task returns a valid task record (task is present)."""
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        task_record = {
            'id': '3438',
            'title': 'Some real task',
            'status': 'in-progress',
            'dependencies': [],
        }
        assert confirm_task_absent(task_record) is False, (
            'confirm_task_absent must return False when task record is present'
        )

    def test_false_for_generic_error_dict(self):
        """Returns False for a generic/inconclusive error dict (e.g. timeout, backend error)."""
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        timeout_error = {
            'error': 'Connection timeout reaching Taskmaster backend',
            'error_type': 'TimeoutError',
        }
        assert confirm_task_absent(timeout_error) is False, (
            'confirm_task_absent must return False for inconclusive errors (fail-closed)'
        )

        backend_error = {
            'error': 'TASKMASTER_UNAVAILABLE: backend not reachable',
            'error_type': 'TaskmasterError',
        }
        assert confirm_task_absent(backend_error) is False, (
            'confirm_task_absent must return False for TASKMASTER_UNAVAILABLE (inconclusive)'
        )

    def test_false_for_none(self):
        """Returns False for None input (fail-closed)."""
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        assert confirm_task_absent(None) is False, (
            'confirm_task_absent must return False for None (fail-closed)'
        )

    def test_false_for_empty_dict(self):
        """Returns False for empty dict (fail-closed)."""
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        assert confirm_task_absent({}) is False, (
            'confirm_task_absent must return False for empty dict (fail-closed)'
        )

    def test_false_for_non_dict_inputs(self):
        """Returns False for non-dict inputs (str, int, list) — fail-closed."""
        from fused_memory.reconciliation.flag_dedup import confirm_task_absent

        for bad in ['no tasks found for id(s)', 42, [], True]:
            assert confirm_task_absent(bad) is False, (
                f'confirm_task_absent must return False for non-dict {bad!r} (fail-closed)'
            )


# ---------------------------------------------------------------------------
# Step 9: filter_false_absence_flags (RED tests)
# ---------------------------------------------------------------------------


class TestFilterFalseAbsenceFlags:
    """RED tests for async filter_false_absence_flags(taskmaster, project_root, flags) (step-9).

    Keeps an absence-type flag ONLY when get_task POSITIVELY confirms absence.
    All other cases (present, inconclusive, error, no task_id) use fail-closed semantics.
    """

    @pytest.mark.asyncio
    async def test_absence_flag_for_present_task_is_dropped(self):
        """Absence flag whose task is PRESENT (get_task returns a record) is DROPPED."""
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={
            'id': '3438', 'title': 'Real task', 'status': 'in-progress',
        })
        project_root = '/proj'

        flags = [{'task_id': '3438', 'flag_type': 'task_absent', 'description': 'looks absent'}]
        result = await filter_false_absence_flags(taskmaster, project_root, flags)

        assert result == [], (
            'Absence flag for a present task must be dropped (fail-closed)'
        )

    @pytest.mark.asyncio
    async def test_absence_flag_for_confirmed_absent_task_is_kept(self):
        """Absence flag whose task is confirmed ABSENT is KEPT.

        The REAL backend (sqlite_task_backend.py:497-499) RAISES TaskmasterError
        with 'No tasks found for ID(s): N' on absence — it does NOT return a dict.
        Only the MCP server wrapper (server/tools.py:1827-1829) converts that
        exception to the {error, error_type} dict.  The production self.taskmaster
        is the RAW TaskBackendProtocol, so filter_false_absence_flags must KEEP the
        flag when the raised exception's str() contains the not-found phrase.
        """
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(
            side_effect=TaskmasterError('TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): 9999')
        )
        project_root = '/proj'

        flag = {'task_id': '9999', 'flag_type': 'task_absent', 'description': 'ghost task'}
        result = await filter_false_absence_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'Absence flag for a confirmed-absent task must be kept when '
            'get_task raises the not-found TaskmasterError (real backend behavior). '
            'RED: current impl treats ALL raises as inconclusive and drops the flag.'
        )

    @pytest.mark.asyncio
    async def test_absence_flag_with_inconclusive_lookup_is_dropped(self):
        """Absence flag whose lookup is INCONCLUSIVE (generic error) is DROPPED (fail-closed)."""
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={
            'error': 'Connection timeout', 'error_type': 'TimeoutError',
        })
        project_root = '/proj'

        flags = [{'task_id': '42', 'flag_type': 'phantom_task', 'description': 'maybe phantom'}]
        result = await filter_false_absence_flags(taskmaster, project_root, flags)

        assert result == [], (
            'Absence flag with inconclusive lookup must be dropped (fail-closed)'
        )

    @pytest.mark.asyncio
    async def test_absence_flag_when_get_task_raises_is_dropped(self):
        """Absence flag whose get_task raises a GENERIC exception is DROPPED (fail-closed).

        RuntimeError('backend down') does NOT contain 'No tasks found for ID(s)',
        so after normalizing to {error: str(exc), error_type: ...}, confirm_task_absent
        returns False → flag is dropped.
        """
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(side_effect=RuntimeError('backend down'))
        project_root = '/proj'

        flags = [{'task_id': '100', 'flag_type': 'orphaned_knowledge', 'description': 'orphan?'}]
        result = await filter_false_absence_flags(taskmaster, project_root, flags)

        assert result == [], (
            'Absence flag when get_task raises a generic exception must be dropped (fail-closed)'
        )

    @pytest.mark.asyncio
    async def test_absence_flag_when_get_task_raises_not_found_is_kept(self):
        """Absence flag whose get_task RAISES the not-found TaskmasterError is KEPT.

        This pins the fix for the production path: self.taskmaster is the RAW
        TaskBackendProtocol whose get_task RAISES TaskmasterError(
        'TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): N') on absence
        (sqlite_task_backend.py:497-499) rather than returning an {error} dict.

        The fix normalizes the exception as {error: str(exc), error_type: typename}
        and passes it to confirm_task_absent, which detects the not-found phrase and
        returns True → the flag is KEPT (task positively absent).

        RED against the current impl, which unconditionally drops every raise as
        inconclusive, making the KEEP path dead code in production.
        """
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(
            side_effect=TaskmasterError('TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): 9999')
        )
        project_root = '/proj'

        flags = [{'task_id': '9999', 'flag_type': 'task_absent', 'description': 'confirmed absent'}]
        result = await filter_false_absence_flags(taskmaster, project_root, flags)

        assert result == flags, (
            'Absence flag must be KEPT when get_task raises the not-found TaskmasterError '
            '(real sqlite backend behavior). '
            'RED: current impl treats ALL raises as inconclusive and drops this flag.'
        )

    @pytest.mark.asyncio
    async def test_non_absence_flag_is_kept_regardless_of_existence(self):
        """Non-absence flag is KEPT untouched regardless of get_task result."""
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        # get_task should NOT be called for non-absence flags
        taskmaster.get_task = AsyncMock(return_value={'id': '7', 'status': 'pending'})
        project_root = '/proj'

        non_absence_flag = {'task_id': '7', 'flag_type': 'missing_deliverable', 'description': 'no output'}
        result = await filter_false_absence_flags(taskmaster, project_root, [non_absence_flag])

        assert result == [non_absence_flag], (
            'Non-absence flag must pass through unchanged'
        )
        # get_task must NOT be called for non-absence flags
        taskmaster.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_absence_flag_without_task_id_is_kept_unchanged(self):
        """Absence flag missing task_id is KEPT unchanged (no task to evaluate)."""
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock()
        project_root = '/proj'

        no_id_flag = {'flag_type': 'task_absent', 'description': 'no task_id set'}
        result = await filter_false_absence_flags(taskmaster, project_root, [no_id_flag])

        assert result == [no_id_flag], (
            'Absence flag without task_id must be kept (cannot evaluate absence)'
        )
        taskmaster.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_flags_processed_correctly(self):
        """Mixed list: absence-present dropped, absence-absent kept, non-absence kept.

        The id-200 leg now RAISES the not-found TaskmasterError (real backend behavior)
        instead of returning an {error} dict (MCP wrapper behavior).
        filter_false_absence_flags must normalise the raised exception to the same dict
        shape confirm_task_absent uses, so the absent-task flag is kept.
        """
        from fused_memory.reconciliation.flag_dedup import filter_false_absence_flags

        project_root = '/proj'

        present_response = {'id': '100', 'title': 'Real', 'status': 'pending'}

        async def _mock_get_task(task_id, project_root, **_kw):
            if str(task_id) == '100':
                return present_response
            if str(task_id) == '200':
                # Real sqlite backend RAISES, not returns, on absence
                raise TaskmasterError('TASKMASTER_TOOL_ERROR', 'No tasks found for ID(s): 200')
            return {'error': 'unexpected', 'error_type': 'RuntimeError'}

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(side_effect=_mock_get_task)

        absent_for_present = {'task_id': '100', 'flag_type': 'task_absent', 'description': 'wrong'}
        absent_for_absent = {'task_id': '200', 'flag_type': 'phantom_task', 'description': 'ok'}
        non_absence = {'task_id': '999', 'flag_type': 'missing_deliverable', 'description': 'x'}
        no_id = {'flag_type': 'task_absent', 'description': 'no task_id'}

        flags = [absent_for_present, absent_for_absent, non_absence, no_id]
        result = await filter_false_absence_flags(taskmaster, project_root, flags)

        assert absent_for_present not in result, 'Flag for present task must be dropped'
        assert absent_for_absent in result, 'Flag for confirmed-absent task must be kept'
        assert non_absence in result, 'Non-absence flag must be kept'
        assert no_id in result, 'Absence flag without task_id must be kept'


# ---------------------------------------------------------------------------
# task-1654 step-1 — RED: compute_content_fingerprint_signature tests
# ---------------------------------------------------------------------------


class TestComputeContentFingerprintSignature:
    """Tests for compute_content_fingerprint_signature(flag) -> tuple[str, str] | None.

    This function is a NEW helper introduced in task-1654 (Fix 2) to route
    null-task_id findings that lack cited_tasks through a deterministic content
    fingerprint so dedup_flags can write/match a stage1_flag_marker and stop
    re-escalating each cycle.

    All tests import from flag_dedup; they will fail with ImportError until
    step-2 adds the implementation.
    """

    def test_returns_deterministic_fp_tuple_for_null_task_id_flag(self):
        """(a) null task_id + valid description + flag_type set -> non-None (fp:<hex>, flag_type)."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'An orphaned edge with no known task anchor',
        }
        result = compute_content_fingerprint_signature(flag)

        assert result is not None, 'Expected a 2-tuple, got None'
        assert isinstance(result, tuple) and len(result) == 2, (
            f'Expected 2-tuple, got {result!r}'
        )
        fp, ftype = result
        assert fp.startswith('fp:'), (
            f'Fingerprint must be prefixed with "fp:"; got {fp!r}'
        )
        assert ftype == 'stale_edge', (
            f'flag_type element must equal "stale_edge"; got {ftype!r}'
        )
        # Deterministic: calling again with the same flag returns the same result
        assert compute_content_fingerprint_signature(flag) == result, (
            'compute_content_fingerprint_signature must be deterministic'
        )

    def test_whitespace_and_case_variants_produce_identical_signature(self):
        """(b) Two descriptions differing only by whitespace/case -> identical signature."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag_a = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'orphaned edge found in graph',
        }
        flag_b = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': '  Orphaned   EDGE  found   in   Graph  ',  # extra spaces + mixed case
        }
        result_a = compute_content_fingerprint_signature(flag_a)
        result_b = compute_content_fingerprint_signature(flag_b)

        assert result_a is not None and result_b is not None
        assert result_a == result_b, (
            f'Whitespace/case variants must produce the same signature; '
            f'got {result_a!r} vs {result_b!r}'
        )

    def test_genuinely_different_descriptions_produce_different_signatures(self):
        """(c) Two genuinely different descriptions -> different signatures."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag_a = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'orphaned edge found in graph',
        }
        flag_b = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'completely different finding about missing memory',
        }
        result_a = compute_content_fingerprint_signature(flag_a)
        result_b = compute_content_fingerprint_signature(flag_b)

        assert result_a is not None and result_b is not None
        assert result_a != result_b, (
            f'Different descriptions must produce different signatures; '
            f'both returned {result_a!r}'
        )

    def test_blank_description_returns_none(self):
        """(d-a) Blank (whitespace-only) description -> None (not a meaningful dedup key)."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag = {'task_id': None, 'flag_type': 'stale_edge', 'description': '   '}
        assert compute_content_fingerprint_signature(flag) is None, (
            'Whitespace-only description must return None'
        )

    def test_missing_description_returns_none(self):
        """(d-b) Missing description key -> None."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag = {'task_id': None, 'flag_type': 'stale_edge'}
        assert compute_content_fingerprint_signature(flag) is None, (
            'Missing description must return None'
        )

    def test_with_top_level_task_id_returns_none(self):
        """(e-a) flag with non-None task_id -> None (defers to compute_flag_signature)."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag = {
            'task_id': 42,
            'flag_type': 'stale_edge',
            'description': 'Has a task_id — should use compute_flag_signature instead',
        }
        assert compute_content_fingerprint_signature(flag) is None, (
            'Non-None task_id must return None (defer to compute_flag_signature)'
        )

    def test_with_non_empty_cited_tasks_returns_none(self):
        """(e-b) flag with non-empty cited_tasks -> None (defers to compute_flag_signature)."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
        )

        flag = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'Has cited_tasks — compute_flag_signature fallback handles this',
            'cited_tasks': [{'project_id': 'reify', 'task_id': '3803'}],
        }
        assert compute_content_fingerprint_signature(flag) is None, (
            'Non-empty cited_tasks must return None (defer to compute_flag_signature)'
        )

    def test_null_flag_type_uses_sentinel_constant(self):
        """(f) flag_type=None + valid description -> 2nd element is _CONTENT_FP_FLAG_TYPE sentinel."""
        from fused_memory.reconciliation.flag_dedup import (
            _CONTENT_FP_FLAG_TYPE,
            compute_content_fingerprint_signature,
        )

        flag = {
            'task_id': None,
            'flag_type': None,
            'description': 'Finding without a flag_type — use sentinel',
        }
        result = compute_content_fingerprint_signature(flag)

        assert result is not None, 'Should still return a tuple even when flag_type is None'
        fp, ftype = result
        assert fp.startswith('fp:'), f'fp must start with "fp:"; got {fp!r}'
        assert ftype == _CONTENT_FP_FLAG_TYPE, (
            f'When flag_type is None, the 2nd element must be _CONTENT_FP_FLAG_TYPE '
            f'({_CONTENT_FP_FLAG_TYPE!r}), got {ftype!r}'
        )


# ---------------------------------------------------------------------------
# task-1654 step-3 — RED: dedup_flags cross-cycle dedup via content fingerprint
# ---------------------------------------------------------------------------


class TestDedupFlagsContentFingerprintPath:
    """Verify dedup_flags uses compute_content_fingerprint_signature as a fallback
    for null-task_id, no-cited-tasks flags so they stop re-escalating each cycle.

    Uses the ledger_memory_service fixture (task 2227): the fp: key is the
    ledger row's task_id column, exactly like a numeric task_id.
    """

    @pytest.mark.asyncio
    async def test_cycle1_miss_writes_marker_keyed_by_content_fingerprint(
        self, ledger_memory_service
    ):
        """Cycle 1 (no prior row): fp:… task_id now accepted — a ledger row is
        written keyed on the fp: value.

        Updated by task-1670 (Option A): the guard now accepts canonical
        fp:+32-hex keys. The flag is returned unchanged (no persisted_from_run
        annotation on a fresh signature).
        """
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'Stale edge d592ca46 has no known task anchor',
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None, 'Test setup: signature must be computable for this flag'
        expected_fp, expected_ftype = sig

        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        # (a) Flag returned unchanged — fresh signature does not annotate persisted_from_run
        assert len(result) == 1
        assert 'persisted_from_run' not in result[0], (
            f'Fresh signature: flag must not have persisted_from_run; got {result[0]}'
        )

        # (b) Ledger row written keyed on the fp: value
        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'p', expected_fp, expected_ftype
        )
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload['task_id'] == expected_fp
        assert payload['run_id'] == 'r1'

    @pytest.mark.asyncio
    async def test_cycle2_hit_annotates_flag_with_persisted_from_run(
        self, ledger_memory_service
    ):
        """Cycle 2 (prior fp:… row present): flag annotated, row re-upserted
        with the new run_id.

        Updated by task-1670 (Option A): the guard now accepts fp: keys.
        """
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'Stale edge d592ca46 has no known task anchor',
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None
        fp, ftype = sig

        ledger = ledger_memory_service.recon_ledger
        await _seed_marker(ledger, 'p', fp, ftype, run_id='r1')

        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r2',
            flags=[flag],
        )

        # (a) Flag annotated: persisted_from_run == r1 (cycle-1 run_id), last_seen_run_id == r2
        assert len(result) == 1
        assert result[0]['persisted_from_run'] == 'r1', (
            f'persisted_from_run must be "r1" (cycle-1 run_id); got {result[0].get("persisted_from_run")!r}'
        )
        assert result[0]['last_seen_run_id'] == 'r2', (
            f'last_seen_run_id must be "r2"; got {result[0].get("last_seen_run_id")!r}'
        )

        # (b) Row re-upserted (still exactly one row) with the new run_id
        row = await _get_marker(ledger, 'p', fp, ftype)
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload['task_id'] == fp
        assert payload['run_id'] == 'r2'


# ---------------------------------------------------------------------------
# task-2047 step-7 — RED: dedup_flags deduped_against enrichment (Gap 1)
# ---------------------------------------------------------------------------


class TestDedupFlagsDedupedAgainstEnrichment:
    """dedup_flags threads deduped_against UUIDs into fp:-keyed markers ONLY
    (task-2047 Gap 1): numeric-task_id markers are unaffected by scoping, and
    a fp: flag with no UUID-bearing fields writes a row with no
    deduped_against key (extraction correctly yields empty -> field omitted).

    Uses the ledger_memory_service fixture (task 2227) — the enrichment rides
    in the ledger row's payload_json (and the best-effort Mem0 mirror).
    """

    @pytest.mark.asyncio
    async def test_fresh_fp_row_carries_deduped_against(self, ledger_memory_service):
        """Fresh signature: a content-fingerprint flag carrying
        deduped_against=['m1','m2'] -> the ledger row's payload.deduped_against
        == ['m1', 'm2'] and payload.task_id is a canonical fp:+32hex key."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
            is_content_fingerprint_task_id,
        )

        flag = {
            'task_id': None,
            'flag_type': 'duplicate_procedural_knowledge',
            'description': 'Duplicate procedural knowledge about deploy steps',
            'deduped_against': ['m1', 'm2'],
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None, 'Test setup: signature must be computable for this flag'
        expected_fp, expected_ftype = sig

        await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        row = await _get_marker(ledger_memory_service.recon_ledger, 'p', expected_fp, expected_ftype)
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload.get('deduped_against') == ['m1', 'm2'], (
            f'Expected deduped_against=["m1","m2"]; got {payload.get("deduped_against")!r}'
        )
        assert is_content_fingerprint_task_id(payload.get('task_id')), (
            f"row task_id {payload.get('task_id')!r} must be a canonical fp:+32hex key"
        )

    @pytest.mark.asyncio
    async def test_recurring_fp_row_still_carries_deduped_against(self, ledger_memory_service):
        """Recurring signature: with a prior fp: row present, the re-upserted
        row ALSO carries payload.deduped_against == ['m1', 'm2']."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'duplicate_procedural_knowledge',
            'description': 'Duplicate procedural knowledge about deploy steps',
            'deduped_against': ['m1', 'm2'],
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None
        fp, ftype = sig

        ledger = ledger_memory_service.recon_ledger
        await _seed_marker(ledger, 'p', fp, ftype, run_id='r1')

        await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r2',
            flags=[flag],
        )

        row = await _get_marker(ledger, 'p', fp, ftype)
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload.get('deduped_against') == ['m1', 'm2'], (
            f'Re-upserted row must carry deduped_against; got {payload.get("deduped_against")!r}'
        )

    @pytest.mark.asyncio
    async def test_numeric_task_id_flag_scoped_out_no_deduped_against(self, ledger_memory_service):
        """Scoping: a numeric-task_id flag (task_id=42) carrying a duplicate_ids
        field must NOT get deduped_against in its ledger row — enrichment
        is fp:-only."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flag = {
            'task_id': 42,
            'flag_type': 'missing_deliverable',
            'duplicate_ids': ['m1', 'm2'],
        }

        await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        row = await _get_marker(ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable')
        assert row is not None
        payload = json.loads(row.payload_json)
        assert 'deduped_against' not in payload, (
            f'Numeric-task_id row must NOT carry deduped_against; got payload={payload!r}'
        )

    @pytest.mark.asyncio
    async def test_fp_flag_with_no_uuid_fields_no_deduped_against(self, ledger_memory_service):
        """fp: flag with no UUID-bearing fields -> row written with no
        deduped_against key (extraction correctly returns empty -> field omitted)."""
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'duplicate_procedural_knowledge',
            'description': 'Duplicate procedural knowledge with no cited UUIDs',
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None
        fp, ftype = sig

        await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        row = await _get_marker(ledger_memory_service.recon_ledger, 'p', fp, ftype)
        assert row is not None
        payload = json.loads(row.payload_json)
        assert 'deduped_against' not in payload, (
            f'fp: row with no UUID fields must NOT carry deduped_against; got payload={payload!r}'
        )

    @pytest.mark.asyncio
    async def test_alias_only_enrichment_logs_observability_notice(self, ledger_memory_service, caplog):
        """When deduped_against is populated ONLY from an alias field (the
        canonical 'deduped_against' field is absent), dedup_flags logs an INFO
        notice so alias-sourced enrichment is observable (task-2047 amendment)."""
        import logging

        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'duplicate_procedural_knowledge',
            'description': 'Duplicate procedural knowledge about deploy steps, alias only',
            'duplicate_ids': ['m1', 'm2'],
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None
        fp, ftype = sig

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.flag_dedup'):
            await dedup_flags(
                memory_service=ledger_memory_service,
                project_id='p',
                run_id='r1',
                flags=[flag],
            )

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any(
            'sourced only from alias' in m and fp in m and ftype in m for m in info_msgs
        ), f'Expected an alias-fallback INFO log naming task_id/flag_type; got {info_msgs!r}'

        row = await _get_marker(ledger_memory_service.recon_ledger, 'p', fp, ftype)
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload.get('deduped_against') == ['m1', 'm2'], (
            f'Alias-sourced enrichment must still be written; got {payload.get("deduped_against")!r}'
        )

    @pytest.mark.asyncio
    async def test_canonical_field_present_does_not_log_alias_fallback(self, ledger_memory_service, caplog):
        """When the canonical 'deduped_against' field is present (non-empty), no
        alias-fallback observability log fires — the log exists specifically to
        flag alias-ONLY enrichment (task-2047 amendment)."""
        import logging

        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'duplicate_procedural_knowledge',
            'description': 'Duplicate procedural knowledge about deploy steps, canonical present',
            'deduped_against': ['m1', 'm2'],
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None

        with caplog.at_level(logging.INFO, logger='fused_memory.reconciliation.flag_dedup'):
            await dedup_flags(
                memory_service=ledger_memory_service,
                project_id='p',
                run_id='r1',
                flags=[flag],
            )

        info_msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert not any('sourced only from alias' in m for m in info_msgs), (
            f'Canonical-field enrichment must NOT log the alias-fallback notice; got {info_msgs!r}'
        )


# ---------------------------------------------------------------------------
# task-1656 step-3 — RED: dedup_flags write-guard integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDedupFlagsWriteGuard:
    """Integration tests for the _is_valid_marker_task_id guard wired into
    dedup_flags' ledger-backed marker path.

    fp: MISS and HIT path coverage lives in TestDedupFlagsContentFingerprintPath
    (focused single-cycle variants) and test_cross_cycle_fp_roundtrip (end-to-end
    two-cycle variant) below.  This class focuses on the remaining contract:

    (a) Regression: numeric task_id '42' still writes its marker (guard does not
        over-reject valid integer keys).
    (b) Regression: comma-joined cited_tasks signature '12,15' still writes its
        marker (guard accepts the comma-joined shape).
    (c) Defense-in-depth: genuinely-invalid tid 'abc' short-circuits before any
        ledger I/O — no row is created, no Mem0 mirror add_memory call.
    (d) Cross-cycle round-trip: fp: marker written on cycle-1, found and used
        to annotate on cycle-2 HIT (persisted_from_run → no re-escalation).
    """

    async def test_numeric_task_id_still_writes_marker(self, ledger_memory_service):
        """(c) Regression: numeric task_id '42' passes guard → a ledger row is written."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flag = {'task_id': 42, 'flag_type': 'missing_deliverable'}

        await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='proj',
            run_id='r1',
            flags=[flag],
        )

        # Guard accepts numeric key → a ledger row is written
        row = await _get_marker(ledger_memory_service.recon_ledger, 'proj', '42', 'missing_deliverable')
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload.get('task_id') == '42', (
            f'Row task_id must be "42"; got {payload.get("task_id")!r}'
        )

    async def test_comma_joined_cited_tasks_signature_still_writes_marker(self, ledger_memory_service):
        """(d) Regression: comma-joined cited_tasks key '12,15' passes guard → a ledger row is written."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flag = {
            'task_id': None,
            'flag_type': 'cross_task_blocker',
            'cited_tasks': [{'task_id': 15}, {'task_id': 12}],
        }
        # compute_flag_signature produces ('12,15', 'cross_task_blocker') via cited_tasks fallback

        await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='proj',
            run_id='r1',
            flags=[flag],
        )

        # Guard accepts comma-joined integer key → a ledger row is written
        row = await _get_marker(ledger_memory_service.recon_ledger, 'proj', '12,15', 'cross_task_blocker')
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload.get('task_id') == '12,15', (
            f'Row task_id must be "12,15"; got {payload.get("task_id")!r}'
        )

    async def test_invalid_tid_write_guard_returns_flag_unchanged_no_io(
        self, ledger_memory_service
    ):
        """(c) dedup_flags' own _is_valid_marker_task_id guard rejects a
        genuinely-invalid tid 'abc' before any ledger I/O: the flag passes
        through unchanged, no ledger row is created, and no Mem0 mirror
        add_memory call is made.
        """
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flag = {'task_id': 'abc', 'flag_type': 'missing_deliverable', 'description': 'd'}

        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='proj',
            run_id='run-x',
            flags=[flag],
        )

        assert result == [flag]
        ledger_memory_service.add_memory.assert_not_called()

        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'proj', 'abc', 'missing_deliverable'
        )
        assert row is None

    async def test_cross_cycle_fp_roundtrip(self, ledger_memory_service):
        """(e2) Cross-cycle round-trip: cycle 1 writes fp: marker; cycle 2 detects it.

        Verifies the full ledger-backed dedup loop for fp: keys:
        - Cycle 1: dedup_flags UPSERTs a stage1_flag_marker row keyed on fp:…
          (a fresh signature — no persisted_from_run annotation).
        - Cycle 2: get_by_identity finds the cycle-1 row; the flag is annotated
          persisted_from_run="c1" → no re-escalation, and the SAME row is
          UPSERTed again (still exactly one row, refreshed to cycle 2).

        Uses compute_content_fingerprint_signature / _content_fingerprint to derive the
        expected fp: value (no hardcoded hashes — anti-drift).
        """
        from fused_memory.reconciliation.flag_dedup import (
            compute_content_fingerprint_signature,
            dedup_flags,
        )

        flag = {
            'task_id': None,
            'flag_type': 'stale_edge',
            'description': 'Round-trip test: stale edge with no task anchor abc999',
        }
        sig = compute_content_fingerprint_signature(flag)
        assert sig is not None
        fp, ftype = sig

        # ---- Cycle 1: fresh signature — no prior row ----
        result_c1 = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='proj',
            run_id='c1',
            flags=[flag],
        )

        assert 'persisted_from_run' not in result_c1[0], (
            f'Cycle 1 must not annotate flag with persisted_from_run; got {result_c1[0]}'
        )
        assert result_c1[0]['last_seen_run_id'] == 'c1'

        row_c1 = await _get_marker(ledger_memory_service.recon_ledger, 'proj', fp, ftype)
        assert row_c1 is not None
        payload_c1 = json.loads(row_c1.payload_json)
        assert payload_c1.get('task_id') == fp
        assert payload_c1['run_id'] == 'c1'

        # ---- Cycle 2: same signature recurs — HIT on the cycle-1 row ----
        result_c2 = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='proj',
            run_id='c2',
            flags=[flag],
        )

        # Cycle 2: flag annotated — dedup worked, no re-escalation
        assert result_c2[0]['persisted_from_run'] == 'c1', (
            f'Cycle 2 must annotate persisted_from_run="c1"; got {result_c2[0].get("persisted_from_run")!r}'
        )
        assert result_c2[0]['last_seen_run_id'] == 'c2'

        # Still exactly one row for this signature, refreshed to cycle 2.
        row_c2 = await _get_marker(ledger_memory_service.recon_ledger, 'proj', fp, ftype)
        assert row_c2 is not None
        payload_c2 = json.loads(row_c2.payload_json)
        assert payload_c2['run_id'] == 'c2'
        assert payload_c2['last_seen_run_id'] == 'c2'


# ---------------------------------------------------------------------------
# amend: normalizer parity — _normalize_content_description must stay aligned
# with recon_report._normalize_description (Suggestion 3).
# A shared test set pins both implementations to identical output so silent
# drift between the local copies is caught by CI.
# ---------------------------------------------------------------------------


class TestNormalizerParity:
    """Both _normalize_content_description (flag_dedup) and _normalize_description
    (recon_report) must produce identical output for the same inputs.

    The implementations are kept as local copies to avoid a server<-reconciliation
    import inversion.  This test class pins them to the same behaviour so that a
    future edit to one is immediately flagged when the other diverges.
    """

    # Shared inputs that exercise whitespace collapse, casefold, and
    # combinations of both.
    _CASES = [
        'Stale edge d592ca46 has no known task anchor',
        '  Leading  and   trailing  whitespace  ',
        'UPPERCASE DESCRIPTION',
        'MiXeD CaSe  with  extra  spaces',
        'single',
        '',
    ]

    def test_identical_output_for_all_cases(self):
        from fused_memory.reconciliation.flag_dedup import (
            _normalize_content_description,
        )
        from fused_memory.server.recon_report import _normalize_description

        for raw in self._CASES:
            fd_result = _normalize_content_description(raw)
            rr_result = _normalize_description(raw)
            assert fd_result == rr_result, (
                f'Normaliser drift detected for input {raw!r}:\n'
                f'  flag_dedup._normalize_content_description → {fd_result!r}\n'
                f'  recon_report._normalize_description       → {rr_result!r}'
            )


# ---------------------------------------------------------------------------
# task-1656 step-1 — RED: _is_valid_marker_task_id unit tests
# ---------------------------------------------------------------------------


class TestIsValidMarkerTaskId:
    """Unit tests for _is_valid_marker_task_id(tid: str) -> bool.

    The helper must ACCEPT bare non-negative integers and comma-joined
    lists of them (the cited_tasks fallback shape), and REJECT content-
    fingerprint keys ('fp:…'), the empty string, non-numeric strings,
    and malformed comma forms.
    """

    # ----- ACCEPT cases -----

    def test_accepts_bare_integer(self):
        """'42' — canonical single-task marker key."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('42') is True, "'42' must be accepted"

    def test_accepts_zero(self):
        """'0' — smallest valid non-negative integer."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('0') is True, "'0' must be accepted"

    def test_accepts_large_integer(self):
        """'99999' — large task id within normal range."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('99999') is True, "'99999' must be accepted"

    def test_accepts_comma_joined_two_ids(self):
        """'12,15' — the cited_tasks fallback shape (multi-task finding)."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('12,15') is True, "'12,15' must be accepted"

    def test_accepts_comma_joined_three_ids(self):
        """'1,2,3' — three tasks cited by one finding."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('1,2,3') is True, "'1,2,3' must be accepted"

    def test_accepts_comma_joined_with_spaces(self):
        """'12, 15' — components with surrounding whitespace must be stripped before check."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('12, 15') is True, "'12, 15' must be accepted (strip spaces)"

    # ----- REJECT cases -----

    def test_accepts_content_fingerprint_key(self):
        """'fp:9216e85ac497b68d93043b64684eb049' — canonical fp:+32 lowercase hex must be accepted."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        tid = 'fp:9216e85ac497b68d93043b64684eb049'
        assert _is_valid_marker_task_id(tid) is True, (
            f'{tid!r} must be accepted (canonical fp:+32 lowercase hex)'
        )

    def test_rejects_empty_string(self):
        """'' — falsy input must be rejected."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('') is False, "'' must be rejected"

    def test_rejects_non_numeric_string(self):
        """'abc' — plain non-numeric string must be rejected."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('abc') is False, "'abc' must be rejected"

    def test_rejects_trailing_comma(self):
        """'12,' — trailing comma produces an empty component, which is non-numeric."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('12,') is False, "'12,' must be rejected"

    def test_rejects_comma_with_fp_component(self):
        """'12,fp:abc' — mixed numeric + fp: component must be rejected."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('12,fp:abc') is False, "'12,fp:abc' must be rejected"

    def test_rejects_negative_integer_string(self):
        """'-1' — negative integer: leading minus is not a digit."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('-1') is False, "'-1' must be rejected"

    def test_rejects_float_string(self):
        """'1.5' — dot is not a digit (mirrors _looks_like_task_id dot-rejecting convention)."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('1.5') is False, "'1.5' must be rejected"

    # ----- Anti-drift: guard accept-pattern tied to the real emitter -----

    def test_accepts_anti_drift_roundtrip(self):
        """_is_valid_marker_task_id(_content_fingerprint(x)) must be True for any non-blank x.

        Ties the guard's accept-pattern to _content_fingerprint's actual output so that
        accept/emit drift is caught as a test failure rather than a silent dedup outage.
        """
        from fused_memory.reconciliation.flag_dedup import (
            _content_fingerprint,
            _is_valid_marker_task_id,
        )

        description = 'Any non-blank description for anti-drift validation'
        fp = _content_fingerprint(description)
        assert _is_valid_marker_task_id(fp) is True, (
            f'_content_fingerprint output {fp!r} must be accepted by '
            f'_is_valid_marker_task_id (anti-drift invariant)'
        )

    # ----- REJECT: malformed fp: forms -----

    def test_rejects_fp_empty_hex(self):
        """'fp:' (no hex digits) — prefix alone is not a valid fp: key."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        assert _is_valid_marker_task_id('fp:') is False, "'fp:' must be rejected (no hex digits)"

    def test_rejects_fp_too_short_hex(self):
        """'fp:' + 30 hex chars — too short (canonical format requires exactly 32 lowercase hex)."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        tid = 'fp:' + 'a' * 30
        assert _is_valid_marker_task_id(tid) is False, (
            f'{tid!r} must be rejected (30 hex chars, need 32)'
        )

    def test_rejects_fp_too_long_hex(self):
        """'fp:' + 64 hex chars — too long; the spec's /^fp:[0-9a-f]{64}$/ shape is wrong.

        The real emitter (_content_fingerprint) produces only 32 hex chars (digest[:32]).
        A 64-hex key would never be emitted, so accepting it would silently break dedup.
        """
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        tid = 'fp:' + 'a' * 64
        assert _is_valid_marker_task_id(tid) is False, (
            f'{tid!r} must be rejected (64 hex chars — too long; real emitter produces 32)'
        )

    def test_rejects_fp_uppercase_hex(self):
        """'fp:' + uppercase 32 hex chars — fp: keys must use lowercase hex (SHA-256 hexdigest)."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        tid = 'fp:' + 'A' * 32
        assert _is_valid_marker_task_id(tid) is False, (
            f'{tid!r} must be rejected (uppercase hex; fp: must be lowercase)'
        )

    def test_rejects_fp_nonhex_char(self):
        """'fp:' + 31 hex chars + 'g' — non-hex character in the body must be rejected."""
        from fused_memory.reconciliation.flag_dedup import _is_valid_marker_task_id

        tid = 'fp:' + 'a' * 31 + 'g'
        assert _is_valid_marker_task_id(tid) is False, (
            f'{tid!r} must be rejected (non-hex char "g" at position 35)'
        )


# ---------------------------------------------------------------------------
# is_content_fingerprint_task_id helper (task-2047 step-3 / step-4)
# ---------------------------------------------------------------------------


class TestIsContentFingerprintTaskId:
    """Unit tests for is_content_fingerprint_task_id(tid) -> bool.

    Unlike _is_valid_marker_task_id (which also accepts numeric and
    comma-joined tids), this helper is the fp:-ONLY gate: True iff *tid* is
    a canonical 'fp:' + exactly 32 lowercase hex digits key, False for
    every other shape (including otherwise-valid numeric/comma-joined
    marker keys).
    """

    def test_true_for_canonical_fp_key(self):
        """A canonical fp:+32-lowercase-hex key (real emitter output) -> True."""
        from fused_memory.reconciliation.flag_dedup import (
            _content_fingerprint,
            is_content_fingerprint_task_id,
        )

        tid = _content_fingerprint('x')
        assert is_content_fingerprint_task_id(tid) is True, (
            f'{tid!r} must be accepted (canonical fp:+32-hex key)'
        )

    def test_false_for_bare_numeric_tid(self):
        """A bare numeric tid ('42') is a valid marker key but NOT an fp: key -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('42') is False

    def test_false_for_comma_joined_numerics(self):
        """A comma-joined numeric tid ('12,15') is a valid marker key but NOT an fp: key -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('12,15') is False

    def test_false_for_empty_string(self):
        """Empty string -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('') is False

    def test_false_for_fp_prefix_with_no_hex(self):
        """'fp:' alone (no hex digits) -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('fp:') is False

    def test_false_for_fp_too_short_hex(self):
        """'fp:' + 30 hex chars (too short) -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('fp:' + 'a' * 30) is False

    def test_false_for_fp_too_long_hex(self):
        """'fp:' + 64 hex chars (too long) -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('fp:' + 'a' * 64) is False

    def test_false_for_fp_uppercase_hex(self):
        """'fp:' + uppercase hex (real emitter only produces lowercase) -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('fp:' + 'A' * 32) is False

    def test_false_for_arbitrary_string(self):
        """An arbitrary non-fp: string -> False."""
        from fused_memory.reconciliation.flag_dedup import is_content_fingerprint_task_id

        assert is_content_fingerprint_task_id('not-a-marker-key-at-all') is False

    def test_anti_drift_roundtrip_with_content_fingerprint(self):
        """is_content_fingerprint_task_id(_content_fingerprint(<desc>)) must be True
        for any non-blank description — ties the gate to the real emitter's output
        so accept/emit drift is caught as a test failure."""
        from fused_memory.reconciliation.flag_dedup import (
            _content_fingerprint,
            is_content_fingerprint_task_id,
        )

        description = 'Any non-blank description for anti-drift validation'
        fp = _content_fingerprint(description)
        assert is_content_fingerprint_task_id(fp) is True, (
            f'_content_fingerprint output {fp!r} must be accepted by '
            f'is_content_fingerprint_task_id (anti-drift invariant)'
        )


# ---------------------------------------------------------------------------
# task-1725 step-1 — RED: filter_terminal_metadata_flags basic tests
# ---------------------------------------------------------------------------


class TestFilterTerminalMetadataFlags:
    """RED tests for async filter_terminal_metadata_flags(taskmaster, project_root, flags).

    A stale_metadata/task_metadata_stale flag for a terminal (cancelled or done)
    task must be DROPPED before dedup_flags sees it.  Non-metadata flags for
    terminal tasks must pass through unchanged (scope guard).
    """

    @pytest.mark.asyncio
    async def test_cancelled_task_stale_metadata_flag_is_dropped(self):
        """stale_metadata flag for a CANCELLED task is DROPPED."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'cancelled'})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [], (
            'stale_metadata flag for a cancelled task must be dropped '
            '(task 1703 is cancelled — no execution-time need for cleaned metadata)'
        )

    @pytest.mark.asyncio
    async def test_active_task_stale_metadata_flag_is_kept(self):
        """stale_metadata flag for an ACTIVE (pending) task is KEPT."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'pending'})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'stale_metadata flag for an active (pending) task must be kept '
            '(task may still have execution-time need for metadata)'
        )

    @pytest.mark.asyncio
    async def test_non_metadata_flag_for_cancelled_task_is_kept(self):
        """Non-metadata flag (orphaned_knowledge) for a cancelled task is KEPT (scope guard).

        filter_terminal_metadata_flags targets ONLY stale_metadata-type flags;
        other flag types for terminal tasks must pass through untouched.
        """
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'cancelled'})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'orphaned_knowledge'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'orphaned_knowledge flag for a cancelled task must NOT be dropped; '
            'filter_terminal_metadata_flags only suppresses stale_metadata-type flags'
        )
        # get_task must NOT be called for non-metadata flags (out of scope)
        taskmaster.get_task.assert_not_called()

    # ---- edge cases added in step-3 ----------------------------------------

    @pytest.mark.asyncio
    async def test_done_task_stale_metadata_flag_is_dropped(self):
        """stale_metadata flag for a DONE task is DROPPED (done is terminal too)."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'done'})
        project_root = '/proj'

        flag = {'task_id': '42', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [], (
            'stale_metadata flag for a done task must be dropped '
            '(done is terminal — no future execution to consume the metadata)'
        )

    @pytest.mark.asyncio
    async def test_get_task_raising_keeps_flag(self):
        """get_task raising an exception → KEEP the flag (fail-safe)."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(side_effect=RuntimeError('backend down'))
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'get_task raising must KEEP the flag (fail-safe: only drop on positively '
            'confirmed terminal status)'
        )

    @pytest.mark.asyncio
    async def test_get_task_returning_non_dict_keeps_flag(self):
        """get_task returning a non-dict (e.g. None) → KEEP the flag (fail-safe)."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value=None)
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'Non-dict get_task result must KEEP the flag (fail-safe: unknown status)'
        )

    @pytest.mark.asyncio
    async def test_get_task_returning_dict_without_status_keeps_flag(self):
        """get_task returning a dict with no 'status' key → 'unknown' status → KEEP."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'id': '1703', 'title': 'Old task'})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'Missing status key should result in unknown status → KEEP the flag'
        )

    @pytest.mark.asyncio
    async def test_get_task_returning_non_terminal_status_keeps_flag(self):
        """get_task returning a non-terminal status ('in-progress') → KEEP."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'in-progress'})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'in-progress status is not terminal → flag must be kept'
        )

    @pytest.mark.asyncio
    async def test_nested_data_status_cancelled_drops_flag(self):
        """Status under {'data': {'status': 'cancelled'}} is extracted and causes DROP."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'data': {'status': 'cancelled'}})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [], (
            'Status nested under data.status must be extracted; cancelled → DROP'
        )

    @pytest.mark.asyncio
    async def test_passthrough_when_taskmaster_is_none(self):
        """No-op pass-through when taskmaster is None — get_task must not be called."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'cancelled'})
        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}

        result = await filter_terminal_metadata_flags(None, '/proj', [flag])

        assert result == [flag], 'None taskmaster must pass all flags through unchanged'
        taskmaster.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_passthrough_when_project_root_is_empty(self):
        """No-op pass-through when project_root is '' — get_task must not be called."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'cancelled'})
        flag = {'task_id': '1703', 'flag_type': 'stale_metadata'}

        result = await filter_terminal_metadata_flags(taskmaster, '', [flag])

        assert result == [flag], 'Empty project_root must pass all flags through unchanged'
        taskmaster.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_metadata_flag_with_no_task_id_passes_through(self):
        """stale_metadata flag with task_id=None passes through without get_task call."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock()
        project_root = '/proj'

        flag = {'task_id': None, 'flag_type': 'stale_metadata'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [flag], (
            'stale_metadata flag with task_id=None must pass through unchanged '
            '(no task to look up)'
        )
        taskmaster.get_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_task_metadata_stale_alias_also_dropped(self):
        """task_metadata_stale (the task-title spelling) is treated the same as stale_metadata."""
        from fused_memory.reconciliation.flag_dedup import filter_terminal_metadata_flags

        taskmaster = AsyncMock()
        taskmaster.get_task = AsyncMock(return_value={'status': 'cancelled'})
        project_root = '/proj'

        flag = {'task_id': '1703', 'flag_type': 'task_metadata_stale'}
        result = await filter_terminal_metadata_flags(taskmaster, project_root, [flag])

        assert result == [], (
            'task_metadata_stale alias must also be dropped for cancelled tasks'
        )


# ---------------------------------------------------------------------------
# task-1786 step-1 — filter_stale_count_snapshot_corrections
# ---------------------------------------------------------------------------


class TestFilterStaleCountSnapshotCorrections:
    """Tests for filter_stale_count_snapshot_corrections(flags) -> list[dict].

    The function drops flags that represent false 'off-by-N correction' findings
    on stale-by-design task-count snapshot edges.  A flag is dropped iff ALL THREE
    conditions hold:
      (a) correction language in description+suggested_action
      (b) ≥2 count-groups of arity ≥2 extractable from the combined text
      (c) first two groups (current, proposed) have equal arity, proposed ≥ current
          componentwise, and max componentwise delta ≤ STALE_SNAPSHOT_CADENCE_DELTA

    All other flags are returned unchanged (fail-open).

    Tests are RED until step-2 adds the symbols to flag_dedup.py.
    """

    def _make_flag(self, description: str, suggested_action: str = '') -> dict:
        return {
            'task_id': '999',
            'flag_type': 'count_snapshot_mismatch',
            'description': description,
            'suggested_action': suggested_action,
        }

    def test_drop_status_worded_incident(self):
        """(DROP) Status-worded incident: 634 done / 607 total → 635 done / 608 total, delta=1."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description=(
                'snapshot edge reports 634 done / 607 total but is off by 1; '
                'should be 635 done / 608 total'
            ),
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [], (
            'Status-worded incident (634/607 → 635/608, delta=1) must be dropped '
            f'but filter returned {result!r}'
        )

    def test_drop_bare_incident_with_stray_digit(self):
        """(DROP) Bare incident: 634/607 → 635/608, stray '1' in 'off by 1' must not confuse parser."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description=(
                'task count snapshot 634/607 is off by 1; should be 635/608'
            ),
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [], (
            "Bare incident (634/607 → 635/608) with stray '1' in 'off by 1' must be "
            f'dropped but filter returned {result!r}'
        )

    def test_drop_correction_language_in_suggested_action(self):
        """(DROP) Correction language in suggested_action (not description) triggers the gate."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='snapshot edge shows 634/607',
            suggested_action='should be 635/608 to match current task tree',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [], (
            'Correction language in suggested_action with delta=1 snapshot must be dropped '
            f'but filter returned {result!r}'
        )

    def test_keep_large_delta(self):
        """(KEEP) Large delta (634→800, delta=166) is not a stale snapshot — preserve as genuine."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='task count snapshot 634/607 is incorrect; should be 800/607',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Large delta (634→800) must be KEPT as a genuine integrity finding '
            f'but filter returned {result!r}'
        )

    def test_keep_count_decrease(self):
        """(KEEP) Count DECREASE (635→634) is a genuine integrity finding — preserve."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='task count snapshot 635/608 is wrong; should be 634/607',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Count DECREASE (635→634) must be KEPT (not stale drift, could be real error) '
            f'but filter returned {result!r}'
        )

    def test_keep_arity_mismatch_single_proposed(self):
        """(KEEP) Arity mismatch: current has arity-2 but proposed is single-number — fail-open."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='task count snapshot 634/607 is off by 1; should be 635',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Arity mismatch (634/607 → 635 alone) must be KEPT (fail-open) '
            f'but filter returned {result!r}'
        )

    def test_keep_no_correction_language(self):
        """(KEEP) No correction language — both snapshot groups present but no 'should be'."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='snapshot 634/607; next cycle shows 635/608',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            "No correction language → must be KEPT regardless of snapshots present "
            f'but filter returned {result!r}'
        )

    def test_keep_no_arity2_snapshot(self):
        """(KEEP) Single-number 'snapshot' — no arity≥2 count group extractable."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='task 634 is off by 1, should be 635',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Single-number count (no arity≥2 group) must be KEPT (fail-open) '
            f'but filter returned {result!r}'
        )

    def test_keep_benign_is_correct(self):
        """(KEEP) 'is correct' must NOT trigger the gate (it is not a correction finding)."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='snapshot 1505 done / 148 cancelled / 1653 total is correct',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            "'is correct' must not be treated as correction language; flag must be KEPT "
            f'but filter returned {result!r}'
        )

    def test_keep_incorrect_substring_word_boundary(self):
        """(KEEP / borderline) 'incorrect' is a trigger, but no arity≥2 snapshot present here."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        # 'incorrect' triggers language gate; but there is no arity≥2 snapshot → KEEP (fail-open)
        flag = self._make_flag(
            description='the edge count is incorrect',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            "'incorrect' triggers language gate but no snapshot pair → KEEP (fail-open) "
            f'but filter returned {result!r}'
        )

    def test_drop_at_boundary_delta_equals_constant(self):
        """(DROP) Delta exactly == STALE_SNAPSHOT_CADENCE_DELTA — must be dropped."""
        from fused_memory.reconciliation.flag_dedup import (
            STALE_SNAPSHOT_CADENCE_DELTA,
            filter_stale_count_snapshot_corrections,
        )
        delta = STALE_SNAPSHOT_CADENCE_DELTA
        current_a, current_b = 634, 607
        proposed_a, proposed_b = current_a + delta, current_b + delta
        flag = self._make_flag(
            description=(
                f'task count snapshot {current_a}/{current_b} is off by {delta}; '
                f'should be {proposed_a}/{proposed_b}'
            ),
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [], (
            f'Delta == STALE_SNAPSHOT_CADENCE_DELTA ({delta}) must be dropped '
            f'but filter returned {result!r}'
        )

    def test_keep_at_delta_above_constant(self):
        """(KEEP) Delta == STALE_SNAPSHOT_CADENCE_DELTA + 100 — must be kept."""
        from fused_memory.reconciliation.flag_dedup import (
            STALE_SNAPSHOT_CADENCE_DELTA,
            filter_stale_count_snapshot_corrections,
        )
        delta = STALE_SNAPSHOT_CADENCE_DELTA + 100
        current_a, current_b = 634, 607
        proposed_a, proposed_b = current_a + delta, current_b + delta
        flag = self._make_flag(
            description=(
                f'task count snapshot {current_a}/{current_b} is off by {delta}; '
                f'should be {proposed_a}/{proposed_b}'
            ),
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            f'Delta == STALE_SNAPSHOT_CADENCE_DELTA + 100 ({delta}) must be KEPT '
            f'but filter returned {result!r}'
        )

    def test_keep_reversed_order_phrasing(self):
        """(KEEP) Corrected-value-first phrasing — proposed comes before current in text.

        When the text reads 'should be 635/608 ... currently 634/607', groups[0] is the
        proposed value and groups[1] is the current value, so the delta computation
        (proposed - current) yields negative values.  The monotonic check rejects this
        and keeps the flag (fail-open).  This test pins the positional contract so any
        future re-ordering of group selection is caught.
        """
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='should be 635/608 but currently snapshot reads 634/607 — off by 1',
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Reversed-order phrasing (proposed before current) yields negative deltas '
            'and must be KEPT (fail-open); '
            f'filter returned {result!r}'
        )

    def test_keep_more_than_two_count_groups(self):
        """(KEEP) Three or more arity-≥2 count-groups → bail to KEEP (ambiguous text).

        A text with three or more count-pairs cannot be safely resolved via positional
        current/proposed assignment — groups[0]/groups[1] might be an incidental near-
        equal pair while the real discrepancy appears later.  The filter must KEEP the
        flag rather than risk a false DROP.
        """
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        # Three count-groups: 634/607 (near-equal to 635/608) then 700/500 (large gap).
        # Positional logic would compare 634/607 vs 635/608 (delta=1, would drop),
        # but the 700/500 group hints at a real discrepancy — flag must be KEPT.
        flag = self._make_flag(
            description=(
                'snapshot edge reports 634/607; should be 635/608; '
                'however the authoritative tree actually shows 700/500 — investigate'
            ),
        )
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Three arity-≥2 count-groups must cause a KEEP (bail-to-KEEP for ambiguous text); '
            f'filter returned {result!r}'
        )

    def test_non_matching_flags_pass_through_unchanged(self):
        """Unrelated flags (not count_snapshot_mismatch) are returned unchanged."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        benign = {'task_id': '42', 'flag_type': 'missing_deliverable', 'description': 'no deliverable'}
        result = filter_stale_count_snapshot_corrections([benign])
        assert result == [benign], (
            'Non-count-snapshot flag must pass through unchanged '
            f'but filter returned {result!r}'
        )

    def test_empty_list_returns_empty(self):
        """Empty input returns empty output."""
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        result = filter_stale_count_snapshot_corrections([])
        assert result == []

    def test_keep_collapsed_single_distinct_group(self):
        """(KEEP) Proposed value restated in both description and suggested_action collapses
        to one distinct group after dedup — must KEEP (fail-open) without raising IndexError.

        When both description and suggested_action mention *only* the proposed value
        (e.g. 'should be 635/608' appears in both fields), raw extraction yields
        groups=[(635,608),(635,608)] (len 2, so the len(groups)<2 early-out does NOT fire).
        After deduplication, unique_groups=[(635,608)] has len 1.  The pre-fix guard
        ``if len(unique_groups) > 2`` does not trigger, so execution falls through to
        ``current, proposed = unique_groups[0], unique_groups[1]`` → IndexError, which
        would abort the entire Stage 1 run.  Post-fix: ``if len(unique_groups) != 2``
        catches len==1 and bails to KEEP (fail-open).
        """
        from fused_memory.reconciliation.flag_dedup import filter_stale_count_snapshot_corrections

        flag = self._make_flag(
            description='The snapshot should be 635/608; currently incorrect.',
            suggested_action='Set it to 635/608.',
        )
        # Must not raise AND must return the flag unchanged (KEEP, fail-open)
        result = filter_stale_count_snapshot_corrections([flag])
        assert result == [flag], (
            'Collapsed-to-one-distinct-group case must be KEPT (fail-open), not raise IndexError; '
            f'filter returned {result!r}'
        )


# ---------------------------------------------------------------------------
# filter_blocked_snapshot_findings (task-1840 step-3 RED / step-4 GREEN)
# ---------------------------------------------------------------------------


class TestFilterBlockedSnapshotFindings:
    """Pure-function tests for filter_blocked_snapshot_findings(flags, project_id).

    All tests call the function directly — no mocks, no I/O.

    RED until step-4 adds filter_blocked_snapshot_findings to flag_dedup.py.
    """

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_flag(category: str, description: str, suggested_action: str = '') -> dict:
        return {
            'task_id': None,
            'flag_type': 'missing_knowledge',
            'category': category,
            'description': description,
            'suggested_action': suggested_action,
            'actionable': False,
        }

    # -----------------------------------------------------------------------
    # Blocked project — dropped cases
    # -----------------------------------------------------------------------

    def test_missing_knowledge_absence_finding_dropped_for_blocked_project(self):
        """Case (a): missing_knowledge finding about a task-count snapshot absence is DROPPED.

        The description uses the 'task-count snapshot' marker phrase (no raw numbers),
        which is the shape of an 'absence' finding from Stage 3.
        """
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='autopilot_video has no task-count snapshot temporal_fact edge',
            suggested_action='Add a count snapshot temporal_fact for autopilot_video',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert result == [], (
            "missing_knowledge finding about task-count snapshot absence must be DROPPED "
            f"for autopilot_video (snapshot writes blocked-by-design); got {result!r}"
        )

    def test_memory_stale_raw_count_finding_dropped_for_blocked_project(self):
        """Case (b): memory_stale finding quoting raw paired count snapshot is DROPPED.

        The description contains a raw paired count snapshot string (e.g. '607 done /
        148 cancelled') which is_count_snapshot() detects.
        """
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='memory_stale',
            description=(
                'task-count snapshot edge records 607 done / 148 cancelled '
                'but the tree shows more'
            ),
            suggested_action='Update the snapshot edge to reflect current counts',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert result == [], (
            "memory_stale finding quoting raw count snapshot must be DROPPED "
            f"for autopilot_video; got {result!r}"
        )

    # -----------------------------------------------------------------------
    # Blocked project — kept cases (category gate and signature gate)
    # -----------------------------------------------------------------------

    def test_missing_deliverable_finding_kept_for_blocked_project(self):
        """Case (c): missing_deliverable finding is KEPT (category gate passes only missing_knowledge/memory_stale)."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_deliverable',
            description='autopilot_video task 42 has no task-count snapshot temporal_fact',
            suggested_action='Add the deliverable',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert flag in result, (
            "missing_deliverable finding must be KEPT (not in suppressed categories); "
            f"got {result!r}"
        )

    def test_memory_contradiction_finding_kept_for_blocked_project(self):
        """Case (c): memory_contradiction finding is KEPT (category gate)."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='memory_contradiction',
            description='count snapshot contradicts task tree',
            suggested_action='Investigate',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert flag in result, (
            "memory_contradiction finding must be KEPT (category not in suppressed set); "
            f"got {result!r}"
        )

    def test_missing_knowledge_unrelated_to_snapshot_kept(self):
        """Case (d): missing_knowledge finding unrelated to snapshots is KEPT (signature gate)."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='task 5 has no design doc and no implementation notes',
            suggested_action='Add a design doc for task 5',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert flag in result, (
            "missing_knowledge finding unrelated to count snapshots must be KEPT; "
            f"got {result!r}"
        )

    def test_empty_list_returns_empty(self):
        """Case (e): empty list → empty list."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        result = filter_blocked_snapshot_findings([], project_id='autopilot_video')
        assert result == []

    # -----------------------------------------------------------------------
    # Non-blocked project — fail-open (all flags pass through)
    # -----------------------------------------------------------------------

    def test_snapshot_absence_finding_kept_for_non_blocked_project(self):
        """For reify (not in blocked set), same missing_knowledge absence finding is KEPT."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='reify has no task-count snapshot temporal_fact edge',
            suggested_action='Add a count snapshot temporal_fact for reify',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='reify')
        assert flag in result, (
            "For non-blocked project 'reify', missing_knowledge snapshot finding "
            f"must pass through unchanged (fail-open); got {result!r}"
        )

    def test_raw_count_finding_kept_for_non_blocked_project(self):
        """For reify, same memory_stale raw-count finding is KEPT (fail-open)."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='memory_stale',
            description='task-count snapshot edge records 607 done / 148 cancelled',
            suggested_action='Update the snapshot',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='reify')
        assert flag in result, (
            "For non-blocked project 'reify', memory_stale count snapshot finding "
            f"must pass through unchanged (fail-open); got {result!r}"
        )

    def test_both_snapshot_findings_survive_for_non_blocked_project(self):
        """For reify both the snapshot-absence and raw-count findings survive."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        absence_flag = self._make_flag(
            category='missing_knowledge',
            description='reify has no task-count snapshot temporal_fact edge',
        )
        stale_flag = self._make_flag(
            category='memory_stale',
            description='count snapshot edge records 607 done / 148 cancelled',
        )
        result = filter_blocked_snapshot_findings(
            [absence_flag, stale_flag], project_id='reify'
        )
        assert len(result) == 2, (
            f"Both findings must survive for non-blocked 'reify'; got {result!r}"
        )

    # -----------------------------------------------------------------------
    # Coverage-gap tests: suggested_action-only path + 'account snapshot' negative
    # -----------------------------------------------------------------------

    def test_marker_in_suggested_action_only_is_dropped(self):
        """_is_task_count_snapshot_finding detects via suggested_action when description is empty.

        The combined text is f'{description} {suggested_action}', so a marker phrase
        appearing only in suggested_action must still trigger suppression.
        """
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='',  # empty — marker only in suggested_action
            suggested_action='Add a task-count snapshot temporal_fact edge for autopilot_video',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert result == [], (
            "missing_knowledge finding whose marker phrase appears only in suggested_action "
            f"must be DROPPED for autopilot_video; got {result!r}"
        )

    def test_account_snapshot_phrase_not_dropped(self):
        """'account snapshot' must NOT be mistaken for a task-count snapshot finding.

        The bare substring 'count snapshot' was intentionally removed from the marker
        list to avoid this false positive.  A finding mentioning 'account snapshot'
        (with no task-count-snapshot marker and no raw count pairs) must be KEPT.
        """
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='autopilot_video account snapshot is missing from the knowledge graph',
            suggested_action='Add the account snapshot edge for project tracking',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='autopilot_video')
        assert flag in result, (
            "Finding about 'account snapshot' must NOT be suppressed — "
            f"'count snapshot' is a substring of 'account snapshot' but must not match; "
            f"got {result!r}"
        )

    # -----------------------------------------------------------------------
    # know_live registration (task 1943 — mirrors autopilot_video / task 1840)
    # -----------------------------------------------------------------------

    def test_missing_knowledge_absence_finding_dropped_for_know_live(self):
        """Case (a) mirrored for know_live: missing_knowledge absence finding is DROPPED."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='know_live has no task-count snapshot temporal_fact edge',
            suggested_action='Add a count snapshot temporal_fact for know_live',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='know_live')
        assert result == [], (
            "missing_knowledge finding about task-count snapshot absence must be DROPPED "
            f"for know_live (snapshot writes blocked-by-design); got {result!r}"
        )

    def test_memory_stale_raw_count_finding_dropped_for_know_live(self):
        """Case (b) mirrored for know_live: memory_stale raw-count finding is DROPPED."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='memory_stale',
            description=(
                'task-count snapshot edge records 607 done / 148 cancelled '
                'but the tree shows more'
            ),
            suggested_action='Update the snapshot edge to reflect current counts',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='know_live')
        assert result == [], (
            "memory_stale finding quoting raw count snapshot must be DROPPED "
            f"for know_live; got {result!r}"
        )

    def test_missing_knowledge_unrelated_to_snapshot_kept_for_know_live(self):
        """Case (d) mirrored for know_live: unrelated missing_knowledge finding is KEPT (signature gate)."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='missing_knowledge',
            description='task 5 has no design doc and no implementation notes',
            suggested_action='Add a design doc for task 5',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='know_live')
        assert flag in result, (
            "missing_knowledge finding unrelated to count snapshots must be KEPT; "
            f"got {result!r}"
        )

    def test_memory_contradiction_finding_kept_for_know_live(self):
        """Case (c) mirrored for know_live: memory_contradiction finding is KEPT (category gate)."""
        from fused_memory.reconciliation.flag_dedup import filter_blocked_snapshot_findings

        flag = self._make_flag(
            category='memory_contradiction',
            description='count snapshot contradicts task tree',
            suggested_action='Investigate',
        )
        result = filter_blocked_snapshot_findings([flag], project_id='know_live')
        assert flag in result, (
            "memory_contradiction finding must be KEPT (category not in suppressed set); "
            f"got {result!r}"
        )


# ---------------------------------------------------------------------------
# TestAcknowledgeFlagMarkerLedger / TestAcknowledgeResolvedFlagsLedger
# (task 2227 step-7) — acknowledge_flag_marker maps to
# memory_service.recon_ledger.mark_addressed; RED until step-8 rewrites
# acknowledge_flag_marker and acknowledge_resolved_flags onto the ledger, at
# which point these classes fold into TestAcknowledgeFlagMarkerDelete/Tag/
# ResolvedFlags below (replacing their Mem0 search+delete/tag bodies).
# ---------------------------------------------------------------------------


class TestAcknowledgeFlagMarkerLedger:
    """acknowledge_flag_marker(memory_service, ...) maps to
    memory_service.recon_ledger.mark_addressed — no Mem0 find_prior_memories
    search, no delete_memory fan-out."""

    @pytest.mark.asyncio
    async def test_seeded_marker_acknowledged_flips_to_addressed(self, ledger_memory_service):
        """A seeded stage1_flag_marker row is acknowledged: returns 1 and the
        row's state flips to 'addressed' with addressed_by/addressed_run_id
        stamped from run_id."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_flag_marker

        await _seed_marker(ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable')

        result = await acknowledge_flag_marker(
            ledger_memory_service,
            project_id='p',
            run_id='rk',
            task_id='42',
            flag_type='missing_deliverable',
        )
        assert result == 1

        row = await ledger_memory_service.recon_ledger.get_by_identity(
            'p', 'stage1_flag_marker', '42', 'missing_deliverable', ''
        )
        assert row is not None
        assert row.state == 'addressed'
        payload = json.loads(row.payload_json)
        assert payload['addressed_by'] == 'rk'
        assert payload['addressed_run_id'] == 'rk'

        ledger_memory_service.search.assert_not_called()
        ledger_memory_service.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_prior_row_returns_0_creates_nothing(self, ledger_memory_service):
        """Acknowledging a signature with no ledger row returns 0 and creates
        no row (mark_addressed no-op — no resurrection of a GC'd marker)."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_flag_marker

        result = await acknowledge_flag_marker(
            ledger_memory_service,
            project_id='p',
            run_id='rk',
            task_id='99',
            flag_type='missing_deliverable',
        )
        assert result == 0

        row = await ledger_memory_service.recon_ledger.get_by_identity(
            'p', 'stage1_flag_marker', '99', 'missing_deliverable', ''
        )
        assert row is None

        ledger_memory_service.search.assert_not_called()
        ledger_memory_service.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_task_id_returns_0_no_io(self, ledger_memory_service):
        """An invalid task_id short-circuits to 0 with no Mem0 I/O."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_flag_marker

        result = await acknowledge_flag_marker(
            ledger_memory_service,
            project_id='p',
            run_id='rk',
            task_id='not-a-number',
            flag_type='missing_deliverable',
        )
        assert result == 0

        ledger_memory_service.search.assert_not_called()
        ledger_memory_service.delete_memory.assert_not_called()
        ledger_memory_service.add_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('mode', ['delete', 'tag'])
    async def test_delete_and_tag_modes_both_mark_addressed(self, ledger_memory_service, mode):
        """mode='delete' and mode='tag' behave identically — both map to
        ledger.mark_addressed (the ledger has no delete; 'addressed' state is
        the durable acknowledgement)."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_flag_marker

        await _seed_marker(ledger_memory_service.recon_ledger, 'p', '7', 'stale_metadata')

        result = await acknowledge_flag_marker(
            ledger_memory_service,
            project_id='p',
            run_id='rk',
            task_id='7',
            flag_type='stale_metadata',
            mode=mode,
        )
        assert result == 1

        row = await ledger_memory_service.recon_ledger.get_by_identity(
            'p', 'stage1_flag_marker', '7', 'stale_metadata', ''
        )
        assert row.state == 'addressed'


class TestAcknowledgeResolvedFlagsLedger:
    """acknowledge_resolved_flags(memory_service, ...) de-dupes signatures and
    fans out to the ledger-backed acknowledge_flag_marker — no Mem0
    find_prior_memories search, no delete_memory fan-out."""

    @pytest.mark.asyncio
    async def test_duplicate_signature_acknowledged_once_and_summed(self, ledger_memory_service):
        """Two flags reducing to the same (task_id, flag_type) signature
        acknowledge exactly once; the single ack's count (1) is the total."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_resolved_flags

        await _seed_marker(ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable')

        resolved_flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
        ]
        total = await acknowledge_resolved_flags(ledger_memory_service, 'p', 'rk', resolved_flags)

        assert total == 1
        ledger_memory_service.search.assert_not_called()
        ledger_memory_service.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_distinct_signatures_summed(self, ledger_memory_service):
        """Two distinct (task_id, flag_type) signatures each with a seeded row
        acknowledge independently and the counts sum to 2."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_resolved_flags

        await _seed_marker(ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable')
        await _seed_marker(ledger_memory_service.recon_ledger, 'p', '7', 'stale_metadata')

        resolved_flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 7, 'flag_type': 'stale_metadata'},
        ]
        total = await acknowledge_resolved_flags(ledger_memory_service, 'p', 'rk', resolved_flags)

        assert total == 2

    @pytest.mark.asyncio
    async def test_unsignable_flags_skipped(self, ledger_memory_service):
        """A flag with no computable signature is skipped (no crash, no
        contribution to the total)."""
        from fused_memory.reconciliation.flag_dedup import acknowledge_resolved_flags

        await _seed_marker(ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable')

        resolved_flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {},  # unsignable: no task_id, no flag_type, no cited_tasks/description
        ]
        total = await acknowledge_resolved_flags(ledger_memory_service, 'p', 'rk', resolved_flags)

        assert total == 1


# ---------------------------------------------------------------------------
# acknowledge_resolved_flags(...) — step-5 RED tests.
#
# RED until step-6 adds acknowledge_resolved_flags to flag_dedup.py.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAcknowledgeResolvedFlags:
    """RED tests for acknowledge_resolved_flags(...) (step-5).

    RED until step-6 adds acknowledge_resolved_flags to flag_dedup.py.
    """

    async def test_dispatches_per_signable_flag_and_sums_count(self, monkeypatch):
        """One acknowledge_flag_marker call per signable flag; correct signature
        forwarded per flag (task_id, cited_tasks-fallback, content-fingerprint);
        no-signature flag skipped; total is the summed count.
        """
        import fused_memory.reconciliation.flag_dedup as flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import (
            acknowledge_resolved_flags,
            compute_content_fingerprint_signature,
        )

        calls: list[tuple[str, str, str]] = []

        async def _fake_ack(memory_service, *, project_id, run_id, task_id, flag_type, mode, log=None):
            calls.append((task_id, flag_type, mode))
            return 1

        monkeypatch.setattr(
            flag_dedup_mod, 'acknowledge_flag_marker', AsyncMock(side_effect=_fake_ack)
        )

        memory_service = AsyncMock()
        task_id_flag = {'task_id': 42, 'flag_type': 'x'}
        cited_tasks_flag = {'task_id': None, 'flag_type': 'y', 'cited_tasks': [{'task_id': 7}]}
        content_fp_flag = {'task_id': None, 'flag_type': None, 'description': 'orphan finding text'}
        no_sig_flag: dict = {}

        resolved_flags = [task_id_flag, cited_tasks_flag, content_fp_flag, no_sig_flag]

        result = await acknowledge_resolved_flags(
            memory_service, 'p', 'r1', resolved_flags, mode='delete',
        )

        assert result == 3
        assert len(calls) == 3, f'no-signature flag must be skipped (no call); got {calls!r}'
        assert ('42', 'x', 'delete') in calls
        assert ('7', 'y', 'delete') in calls
        fp_sig = compute_content_fingerprint_signature(content_fp_flag)
        assert fp_sig is not None
        assert (fp_sig[0], fp_sig[1], 'delete') in calls

    async def test_mode_forwarded_verbatim_tag(self, monkeypatch):
        """mode='tag' is forwarded verbatim to every acknowledge_flag_marker call."""
        import fused_memory.reconciliation.flag_dedup as flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import acknowledge_resolved_flags

        calls: list[tuple[str, str, str]] = []

        async def _fake_ack(memory_service, *, project_id, run_id, task_id, flag_type, mode, log=None):
            calls.append((task_id, flag_type, mode))
            return 1

        monkeypatch.setattr(
            flag_dedup_mod, 'acknowledge_flag_marker', AsyncMock(side_effect=_fake_ack)
        )

        memory_service = AsyncMock()
        resolved_flags = [{'task_id': 1, 'flag_type': 'a'}, {'task_id': 2, 'flag_type': 'b'}]

        result = await acknowledge_resolved_flags(
            memory_service, 'p', 'r1', resolved_flags, mode='tag',
        )

        assert result == 2
        assert calls == [('1', 'a', 'tag'), ('2', 'b', 'tag')]

    async def test_one_flag_failure_does_not_abort_batch(self, monkeypatch, caplog):
        """One flag's acknowledge_flag_marker raising is logged and does not abort
        the batch — the remaining flags are still processed.
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import acknowledge_resolved_flags

        calls: list[str] = []

        async def _fake_ack(memory_service, *, project_id, run_id, task_id, flag_type, mode, log=None):
            calls.append(task_id)
            if task_id == '2':
                raise RuntimeError('boom')
            return 1

        monkeypatch.setattr(
            flag_dedup_mod, 'acknowledge_flag_marker', AsyncMock(side_effect=_fake_ack)
        )

        memory_service = AsyncMock()
        resolved_flags = [
            {'task_id': 1, 'flag_type': 'a'},
            {'task_id': 2, 'flag_type': 'b'},
            {'task_id': 3, 'flag_type': 'c'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await acknowledge_resolved_flags(
                memory_service, 'p', 'r1', resolved_flags, mode='delete',
            )

        # All three flags were attempted despite the middle one raising.
        assert calls == ['1', '2', '3']
        # flag 2's exception is excluded from the count; 1 and 3 succeeded.
        assert result == 2
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('2' in m for m in warning_messages), (
            f'expected a WARNING mentioning the failed flag; got {warning_messages!r}'
        )

    async def test_duplicate_signatures_deduplicated_before_dispatch(self, monkeypatch):
        """Two flags reducing to the SAME (task_id, flag_type) signature — e.g.
        two stale_metadata findings on the same task — must result in exactly
        ONE acknowledge_flag_marker call, not two concurrent calls racing to
        delete/tag the same prior marker(s) (amendment round 2, reviewer
        finding: robustness). Without de-duplication the count would also be
        inflated (2 instead of 1 for the duplicated signature).
        """
        import fused_memory.reconciliation.flag_dedup as flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import acknowledge_resolved_flags

        calls: list[tuple[str, str, str]] = []

        async def _fake_ack(memory_service, *, project_id, run_id, task_id, flag_type, mode, log=None):
            calls.append((task_id, flag_type, mode))
            return 1

        monkeypatch.setattr(
            flag_dedup_mod, 'acknowledge_flag_marker', AsyncMock(side_effect=_fake_ack)
        )

        memory_service = AsyncMock()
        resolved_flags = [
            {'task_id': 42, 'flag_type': 'x', 'description': 'first finding'},
            {'task_id': 42, 'flag_type': 'x', 'description': 'second finding'},
            {'task_id': 7, 'flag_type': 'y'},
        ]

        result = await acknowledge_resolved_flags(
            memory_service, 'p', 'r1', resolved_flags, mode='delete',
        )

        assert calls == [('42', 'x', 'delete'), ('7', 'y', 'delete')], (
            f'expected exactly one call per de-duplicated signature; got {calls!r}'
        )
        assert result == 2


# ---------------------------------------------------------------------------
# _is_completion_flag pure predicate tests (task-2312 step-1)
# ---------------------------------------------------------------------------


class TestIsCompletionFlag:
    """Unit tests for _is_completion_flag(flag: dict) -> bool.

    The helper must return True ONLY when flag['flag_for_stage2'] is
    present AND explicitly false (bool False, or a case-insensitive 'false'
    string) — never on mere absence of the key. Absence is the shape every
    dedup MISS/HIT marker and every ordinary recurring flag has, so treating
    absence as "false" would misclassify recurring findings as one-time
    completions.
    """

    # ----- ACCEPT cases: present-and-explicitly-false -----

    def test_bool_false_is_completion_flag(self):
        """flag_for_stage2=False (bool) — the canonical completion signal."""
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x', 'flag_for_stage2': False}
        assert _is_completion_flag(flag) is True

    def test_string_false_lowercase_is_completion_flag(self):
        """flag_for_stage2='false' (str) is accepted case-insensitively."""
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x', 'flag_for_stage2': 'false'}
        assert _is_completion_flag(flag) is True

    def test_string_false_titlecase_is_completion_flag(self):
        """flag_for_stage2='False' (str, title-case) is accepted case-insensitively."""
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x', 'flag_for_stage2': 'False'}
        assert _is_completion_flag(flag) is True

    # ----- REJECT cases -----

    def test_absent_key_is_not_completion_flag(self):
        """Key absent entirely — MUST NOT be treated as a completion flag.

        Dedup-safety-critical case: dedup MISS markers never set
        flag_for_stage2, so gating on absence would delete markers that
        cross-cycle dedup depends on.
        """
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x'}
        assert _is_completion_flag(flag) is False

    def test_bool_true_is_not_completion_flag(self):
        """flag_for_stage2=True (bool) is a normal Stage-2-bound flag, not a completion marker."""
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x', 'flag_for_stage2': True}
        assert _is_completion_flag(flag) is False

    def test_string_true_is_not_completion_flag(self):
        """flag_for_stage2='true' (truthy string) is not a completion marker."""
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x', 'flag_for_stage2': 'true'}
        assert _is_completion_flag(flag) is False

    def test_none_value_is_not_completion_flag(self):
        """flag_for_stage2=None is not an explicit false — not a completion marker."""
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'task_id': 1, 'flag_type': 'x', 'flag_for_stage2': None}
        assert _is_completion_flag(flag) is False

    def test_non_flag_shaped_dict_is_not_completion_flag(self):
        """A dict with none of the usual flag fields (no task_id/flag_type/
        flag_for_stage2) must not crash and must return False.
        """
        from fused_memory.reconciliation.flag_dedup import _is_completion_flag

        flag = {'unexpected_key': 'unexpected_value'}
        assert _is_completion_flag(flag) is False


# ---------------------------------------------------------------------------
# dedup_flags — completion-marker same-cycle self-delete (task-2312 step-3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_completion_flag_sweeps_prior_without_writing_new_marker(
    ledger_memory_service,
):
    """flag_for_stage2=False marks a ONE-TIME completed-work finding: dedup_flags
    does NOT write a new stage1_flag_marker for it (task-2312 review amendment —
    an emit-then-self-delete design raced Mem0 read-after-write indexing lag and
    could leave a fresh orphan). Instead it only sweeps any PRIOR marker for the
    signature via acknowledge_flag_marker → ledger.mark_addressed — bypassing
    the persist-for-dedup MISS/HIT path entirely.

    A prior marker (simulating one left over from an earlier cycle) is found
    and swept to state='addressed'; no marker is (re-)written.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger = ledger_memory_service.recon_ledger
    await _seed_marker(ledger, 'p', '77', 'duplicate_flag_marker_cleanup', run_id='r0')

    flags = [{
        'task_id': 77,
        'flag_type': 'duplicate_flag_marker_cleanup',
        'description': 'cleaned up an orphaned duplicate flag marker',
        'flag_for_stage2': False,
    }]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) No new marker is (re-)written for a completion flag.
    ledger_memory_service.add_memory.assert_not_called()

    # (b) The pre-existing prior is swept to state='addressed' on the ledger.
    row = await _get_marker(ledger, 'p', '77', 'duplicate_flag_marker_cleanup')
    assert row is not None
    assert row.state == 'addressed'

    # (c) Flag annotated; persist-for-dedup fields are NOT set (bypassed).
    assert len(result) == 1
    assert result[0]['completion_marker_self_deleted'] is True
    assert result[0]['last_seen_run_id'] == 'r1'
    assert 'persisted_from_run' not in result[0]


@pytest.mark.asyncio
async def test_dedup_flags_completion_flag_with_no_priors_is_noop_sweep(
    ledger_memory_service,
):
    """A completion flag whose signature has no prior marker at all (the common
    case: nothing was ever persisted for it) is a clean no-op sweep — no
    add_memory, no ledger row created — and is still annotated
    completion_marker_self_deleted=True.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [{
        'task_id': 77,
        'flag_type': 'duplicate_flag_marker_cleanup',
        'description': 'cleaned up an orphaned duplicate flag marker',
        'flag_for_stage2': False,
    }]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    ledger_memory_service.add_memory.assert_not_called()

    row = await _get_marker(
        ledger_memory_service.recon_ledger, 'p', '77', 'duplicate_flag_marker_cleanup'
    )
    assert row is None

    assert len(result) == 1
    assert result[0]['completion_marker_self_deleted'] is True
    assert result[0]['last_seen_run_id'] == 'r1'
    assert 'persisted_from_run' not in result[0]


@pytest.mark.asyncio
async def test_dedup_flags_completion_flag_duplicate_signature_in_batch_swept_once(
    ledger_memory_service,
):
    """Revised task-2312 semantics under the ledger (seen_signatures removed,
    plan.json design decision): when the SAME (task_id, flag_type) completion
    signature is emitted twice in one items_flagged batch, BOTH occurrences
    self-annotate — the ledger sweep (acknowledge_flag_marker →
    ledger.mark_addressed) is idempotent, so re-running it for the second
    occurrence is harmless and no in-batch memo is needed to prevent it.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [
        {
            'task_id': 77,
            'flag_type': 'duplicate_flag_marker_cleanup',
            'description': 'cleaned up an orphaned duplicate flag marker',
            'flag_for_stage2': False,
        },
        {
            'task_id': 77,
            'flag_type': 'duplicate_flag_marker_cleanup',
            'description': 'same signature, re-evaluated within this run',
            'flag_for_stage2': False,
        },
    ]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    ledger_memory_service.add_memory.assert_not_called()

    assert len(result) == 2
    # Both occurrences: resolved directly by the completion branch — the
    # idempotent ledger sweep makes a second sweep for the same signature
    # harmless, so no in-batch memo is needed to skip it.
    for flag in result:
        assert flag.get('completion_marker_self_deleted') is True
        assert flag.get('last_seen_run_id') == 'r1'
        assert 'persisted_from_run' not in flag


@pytest.mark.asyncio
async def test_dedup_flags_recurring_flag_without_flag_for_stage2_unaffected(
    ledger_memory_service,
):
    """Dedup-safety guard: a flag WITHOUT flag_for_stage2 (an ordinary recurring
    finding) is completely unchanged by the completion-marker feature — its
    marker is UPSERTed to the ledger and never swept, proving cross-cycle
    dedup is not regressed by the completion branch.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [{'task_id': '99', 'flag_type': 'stale_metadata', 'description': 'bar'}]

    result = await dedup_flags(
        memory_service=ledger_memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]
    assert 'completion_marker_self_deleted' not in result[0]

    row = await _get_marker(ledger_memory_service.recon_ledger, 'p', '99', 'stale_metadata')
    assert row is not None
    assert row.state == 'active'

    ledger_memory_service.add_memory.assert_not_called()
    ledger_memory_service.delete_memory.assert_not_called()


# ---------------------------------------------------------------------------
# dedup_flags — ledger-backed marker path (task 2227 step-09)
#
# RED until step-10 rewrites dedup_flags' per-flag marker path onto a single
# memory_service.recon_ledger.upsert(...) keyed on (task_id, flag_type) with
# run_id='' in the identity (see plan.json design decisions), replacing the
# Mem0 search/write/confirm/delete chain exercised by the tests above.
# filter_suppressed (step-04) and acknowledge_flag_marker (step-08) are
# already ledger-backed, so the completion branch's sweep already flows
# through the ledger today; the tests below additionally pin that behaviour
# survives the upcoming per-flag marker-path rewrite.
# ---------------------------------------------------------------------------


class TestDedupFlagsLedgerMarkerPath:
    """dedup_flags' marker path becomes a single ledger UPSERT keyed on
    (task_id, flag_type) — no Mem0 search/confirm/delete round trip."""

    @pytest.mark.asyncio
    async def test_recurring_flag_first_run_upserts_single_marker_row(
        self, ledger_memory_service
    ):
        """A recurring flag's first run UPSERTs exactly one stage1_flag_marker
        row, with payload run_id/last_seen_run_id set from the current run,
        and the flag itself is annotated last_seen_run_id (no
        persisted_from_run — this is a fresh signature)."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]

        result = await dedup_flags(ledger_memory_service, 'p', 'r1', flags)

        assert len(result) == 1
        assert 'persisted_from_run' not in result[0]
        assert result[0]['last_seen_run_id'] == 'r1'

        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable'
        )
        assert row is not None
        assert row.state == 'active'
        payload = json.loads(row.payload_json)
        assert payload['run_id'] == 'r1'
        assert payload['last_seen_run_id'] == 'r1'

    @pytest.mark.asyncio
    async def test_recurring_flag_second_run_upserts_same_row(self, ledger_memory_service):
        """The same signature recurring on a later run UPSERTs the SAME row
        (still exactly one) with a refreshed run_id/last_seen_run_id, and the
        flag is annotated persisted_from_run from the FIRST run's write."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        await dedup_flags(ledger_memory_service, 'p', 'r1', flags)

        result = await dedup_flags(ledger_memory_service, 'p', 'r2', flags)

        assert len(result) == 1
        assert result[0]['persisted_from_run'] == 'r1'
        assert result[0]['last_seen_run_id'] == 'r2'

        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable'
        )
        assert row is not None
        payload = json.loads(row.payload_json)
        assert payload['run_id'] == 'r2'
        assert payload['last_seen_run_id'] == 'r2'

    @pytest.mark.asyncio
    async def test_same_signature_twice_in_one_call_collapses_to_one_row(
        self, ledger_memory_service
    ):
        """The SAME recurring signature emitted twice in ONE dedup_flags call
        collapses to exactly one ledger row — the ledger's UPSERT +
        read-after-write consistency makes the old seen_signatures in-batch
        memo unnecessary for this case."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'first'},
            {'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'second'},
        ]

        result = await dedup_flags(ledger_memory_service, 'p', 'r1', flags)

        assert len(result) == 2
        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'p', '42', 'missing_deliverable'
        )
        assert row is not None

    @pytest.mark.asyncio
    async def test_recurring_marker_path_never_touches_mem0_search_or_delete(
        self, ledger_memory_service
    ):
        """The ledger-backed marker path performs no Mem0 read-back/confirm
        loop and no per-prior delete_memory dance, across two recurring
        cycles of the same signature."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        await dedup_flags(ledger_memory_service, 'p', 'r1', flags)
        await dedup_flags(ledger_memory_service, 'p', 'r2', flags)

        ledger_memory_service.search.assert_not_called()
        ledger_memory_service.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_completion_flag_writes_no_marker_and_sweeps_prior_to_addressed(
        self, ledger_memory_service
    ):
        """A completion flag (flag_for_stage2=False) writes NO stage1_flag_marker
        row and sweeps a pre-seeded prior to state='addressed' via the ledger
        (acknowledge_flag_marker, already ledger-backed since step-08)."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        await _seed_marker(
            ledger_memory_service.recon_ledger, 'p', '77', 'duplicate_flag_marker_cleanup',
            run_id='r0',
        )

        flags = [{
            'task_id': 77,
            'flag_type': 'duplicate_flag_marker_cleanup',
            'flag_for_stage2': False,
        }]

        result = await dedup_flags(ledger_memory_service, 'p', 'r1', flags)

        assert len(result) == 1
        assert result[0]['completion_marker_self_deleted'] is True
        assert result[0]['last_seen_run_id'] == 'r1'
        assert 'persisted_from_run' not in result[0]
        ledger_memory_service.add_memory.assert_not_called()

        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'p', '77', 'duplicate_flag_marker_cleanup'
        )
        assert row is not None
        assert row.state == 'addressed'

    @pytest.mark.asyncio
    async def test_completion_flag_with_no_prior_is_noop_sweep_still_annotated(
        self, ledger_memory_service
    ):
        """A completion flag whose signature has no prior marker at all is a
        clean no-op sweep — still annotated completion_marker_self_deleted=True,
        and creates no row."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flags = [{
            'task_id': 77,
            'flag_type': 'duplicate_flag_marker_cleanup',
            'flag_for_stage2': False,
        }]

        result = await dedup_flags(ledger_memory_service, 'p', 'r1', flags)

        assert len(result) == 1
        assert result[0]['completion_marker_self_deleted'] is True
        assert result[0]['last_seen_run_id'] == 'r1'
        assert 'persisted_from_run' not in result[0]
        ledger_memory_service.add_memory.assert_not_called()

        row = await _get_marker(
            ledger_memory_service.recon_ledger, 'p', '77', 'duplicate_flag_marker_cleanup'
        )
        assert row is None

    @pytest.mark.asyncio
    async def test_duplicate_completion_signature_in_batch_self_annotates_each_occurrence(
        self, ledger_memory_service
    ):
        """Revised task-2312 semantics under the ledger (seen_signatures
        removed, plan.json design decision): the SAME completion signature
        emitted twice in one batch self-annotates BOTH occurrences — the
        ledger sweep (mark_addressed) is idempotent, so re-running it for the
        second occurrence is harmless and no in-batch memo is needed to
        prevent it."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flags = [
            {
                'task_id': 77,
                'flag_type': 'duplicate_flag_marker_cleanup',
                'flag_for_stage2': False,
                'description': 'first',
            },
            {
                'task_id': 77,
                'flag_type': 'duplicate_flag_marker_cleanup',
                'flag_for_stage2': False,
                'description': 'second',
            },
        ]

        result = await dedup_flags(ledger_memory_service, 'p', 'r1', flags)

        assert len(result) == 2
        for flag in result:
            assert flag.get('completion_marker_self_deleted') is True
            assert flag.get('last_seen_run_id') == 'r1'
            assert 'persisted_from_run' not in flag


# ---------------------------------------------------------------------------
# module-surface contract — dead compensation code removed (task 2227 step-11)
#
# RED until step-12 deletes confirm_marker_persisted, _write_and_confirm_marker,
# _marker_query, _CONFIRMATION_MISS_THRESHOLD, _CONFIRM_RETRY_DELAY_SECS and
# strips their now-dead references out of dedup_flags' source. _is_completion_flag
# must survive the cleanup (task-2312 completion-marker self-delete stays).
# ---------------------------------------------------------------------------


class TestModuleSurfaceCompensationsRemoved:
    """Pins that the Mem0 compensation chain is fully deleted from the module
    surface while the task-2312 completion-marker predicate survives."""

    def test_confirm_marker_persisted_removed(self):
        import fused_memory.reconciliation.flag_dedup as fd

        assert not hasattr(fd, 'confirm_marker_persisted'), (
            'confirm_marker_persisted must be deleted — the ledger UPSERT '
            'replaces the confirm-after-write compensation.'
        )

    def test_write_and_confirm_marker_removed(self):
        import fused_memory.reconciliation.flag_dedup as fd

        assert not hasattr(fd, '_write_and_confirm_marker'), (
            '_write_and_confirm_marker must be deleted — dedup_flags now '
            'performs a single ledger.upsert call.'
        )

    def test_marker_query_removed(self):
        import fused_memory.reconciliation.flag_dedup as fd

        assert not hasattr(fd, '_marker_query'), (
            '_marker_query must be deleted — no Mem0 search is issued for '
            'stage1_flag_marker rows anymore.'
        )

    def test_confirmation_miss_threshold_removed(self):
        import fused_memory.reconciliation.flag_dedup as fd

        assert not hasattr(fd, '_CONFIRMATION_MISS_THRESHOLD'), (
            '_CONFIRMATION_MISS_THRESHOLD must be deleted — the per-invocation '
            'circuit breaker has no ledger-backed equivalent (or need).'
        )

    def test_confirm_retry_delay_secs_removed(self):
        import fused_memory.reconciliation.flag_dedup as fd

        assert not hasattr(fd, '_CONFIRM_RETRY_DELAY_SECS'), (
            '_CONFIRM_RETRY_DELAY_SECS must be deleted along with the '
            'confirmation retry loop it configured.'
        )

    def test_is_completion_flag_preserved(self):
        import fused_memory.reconciliation.flag_dedup as fd

        assert hasattr(fd, '_is_completion_flag'), (
            '_is_completion_flag (task-2312) must survive the compensation '
            'cleanup — the completion-marker self-delete branch depends on it.'
        )

    def test_dedup_flags_source_free_of_compensation_machinery(self):
        import inspect

        import fused_memory.reconciliation.flag_dedup as fd

        source = inspect.getsource(fd.dedup_flags)
        for banned in (
            'seen_signatures',
            'confirmation_disabled',
            'consecutive_confirmation_misses',
            'confirm_marker_persisted',
            'limit=50',
        ):
            assert banned not in source, (
                f'dedup_flags source must not reference {banned!r} — the '
                'ledger UPSERT replaced the compensation chain that used it.'
            )

    def test_dedup_flags_source_retains_completion_marker_annotation(self):
        import inspect

        import fused_memory.reconciliation.flag_dedup as fd

        source = inspect.getsource(fd.dedup_flags)
        assert 'completion_marker_self_deleted' in source, (
            'dedup_flags source must still annotate completion_marker_self_deleted '
            '— the task-2312 completion branch must survive the cleanup.'
        )


# ---------------------------------------------------------------------------
# filter_already_tracked_systemic_patterns pure-helper tests (task 2416, step-1)
# ---------------------------------------------------------------------------


class TestAlreadyTrackedSystemicPatternHelpers:
    """Tests for the pure helpers backing filter_already_tracked_systemic_patterns
    (task 2416): ``_asserts_never_tracked(text) -> bool`` and
    ``_significant_terms(text) -> set[str]``.

    RED until step-2 adds both symbols to flag_dedup.py.
    """

    @pytest.mark.parametrize(
        'text',
        [
            'This idea was never converted to a tracked task.',
            'The recommendation was never converted to a task for follow-up.',
            'This recurring pattern is never tracked in the task tree.',
            'The suggestion was never filed as a task by anyone.',
            'There is no tracked task for this recurring pattern.',
            'NEVER CONVERTED TO A TRACKED TASK (shouting case still matches).',
        ],
        ids=[
            'never-converted-to-a-tracked-task',
            'was-never-converted-to-a-task',
            'never-tracked',
            'never-filed-as-a-task',
            'no-tracked-task',
            'case-insensitive',
        ],
    )
    def test_asserts_never_tracked_true_for_lexicon_phrases(self, text):
        """_asserts_never_tracked recognises every fixed never-tracked lexicon phrase."""
        from fused_memory.reconciliation.flag_dedup import _asserts_never_tracked

        assert _asserts_never_tracked(text) is True, (
            f'Expected _asserts_never_tracked to detect never-tracked language in {text!r}'
        )

    def test_asserts_never_tracked_false_for_unrelated_text(self):
        """Unrelated finding text (no never-tracked assertion) returns False."""
        from fused_memory.reconciliation.flag_dedup import _asserts_never_tracked

        text = 'Task 100 has no deliverable attached; consider closing it out.'
        assert _asserts_never_tracked(text) is False, (
            'Unrelated finding text must not be classified as never-tracked language'
        )

    def test_asserts_never_tracked_false_for_empty_text(self):
        """Empty string returns False (no phrase can match)."""
        from fused_memory.reconciliation.flag_dedup import _asserts_never_tracked

        assert _asserts_never_tracked('') is False

    def test_significant_terms_lowercases_splits_and_dedupes(self):
        """_significant_terms lowercases, splits on non-alphanumeric runs, and dedupes.

        'project_status_correction' must split into three distinct tokens
        (underscore is a non-alphanumeric separator), not survive as one
        underscore-joined token — this is what lets the finding's key terms
        overlap with a done task's title/description prose.
        """
        from fused_memory.reconciliation.flag_dedup import _significant_terms

        text = (
            'Diff the project_status_correction cache vs live get_statuses '
            'every cycle — Project_Status_Correction correction is needed.'
        )
        terms = _significant_terms(text)

        assert isinstance(terms, set)
        assert 'project' in terms
        assert 'status' in terms
        assert 'correction' in terms
        assert 'get' in terms
        assert 'statuses' in terms
        assert 'cycle' in terms
        assert 'cache' in terms
        # Case-insensitive: 'Cache' and 'cache' collapse to a single set entry.
        assert sum(1 for t in terms if t == 'cache') == 1

    def test_significant_terms_drops_stopwords_and_short_tokens(self):
        """Stopwords and tokens shorter than 3 chars are dropped."""
        from fused_memory.reconciliation.flag_dedup import _significant_terms

        text = 'This is a of to in on at it an idea for the cache.'
        terms = _significant_terms(text)

        for stopword in (
            'this', 'is', 'a', 'of', 'to', 'in', 'on', 'at', 'it', 'an', 'for', 'the',
        ):
            assert stopword not in terms, f'Stopword {stopword!r} must be dropped from {terms!r}'
        assert 'cache' in terms

    def test_significant_terms_empty_string_returns_empty_set(self):
        """Empty input returns an empty set (no I/O, no crash)."""
        from fused_memory.reconciliation.flag_dedup import _significant_terms

        assert _significant_terms('') == set()


# ---------------------------------------------------------------------------
# filter_already_tracked_systemic_patterns core-drop test (task 2416, step-3)
# ---------------------------------------------------------------------------


class TestFilterAlreadyTrackedSystemicPatterns:
    """Tests for async filter_already_tracked_systemic_patterns(taskmaster,
    dark_factory_root, flags) -> list[dict] (task 2416).

    Drops a systemic_pattern 'never tracked' finding when a done dark_factory
    task's title+description already covers its distinctive key terms —
    hardening against the e61b38f9/1938 false-positive incident (a finding
    claiming the 'diff project_status_correction cache vs live get_statuses
    every cycle' idea was never tracked, despite dark_factory task 1938
    (done, merged 2026-07-01) already implementing it).

    RED until step-4 adds filter_already_tracked_systemic_patterns to
    flag_dedup.py.
    """

    def _make_never_tracked_flag(self) -> dict:
        return {
            'task_id': None,
            'category': 'systemic_pattern',
            'flag_type': 'systemic_pattern',
            'description': (
                'This systemic pattern was never converted to a tracked task: diff '
                'the project_status_correction cache against live get_statuses every '
                'cycle to catch drift.'
            ),
            'suggested_action': (
                'File a task to diff the cache against live status each cycle.'
            ),
        }

    @pytest.mark.asyncio
    async def test_drop_when_done_task_already_implements_the_idea(self):
        """Core e61b38f9/1938 scenario: DROP when a done task covers the idea."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': '1938',
                    'status': 'done',
                    'title': (
                        'Diff project_status_correction cache against live '
                        'get_statuses every cycle'
                    ),
                    'description': (
                        'Implemented a periodic diff of the cached '
                        'project_status_correction value against a live '
                        'get_statuses call each cycle to catch drift and correct '
                        'stale cache entries before they propagate.'
                    ),
                },
            ],
        })

        result = await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        assert result == [], (
            'systemic_pattern never-tracked finding must be DROPPED when done '
            f'dark_factory task 1938 already covers the idea; got {result!r}'
        )

    # ---- scope / keep-guard edge cases (step-5) ----------------------------

    @pytest.mark.asyncio
    async def test_keep_when_no_done_task_covers_the_idea(self):
        """(KEEP) No done task's terms cover the finding — real signal preserved."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': '77',
                    'status': 'done',
                    'title': 'Rewrite the onboarding email template',
                    'description': (
                        'Refreshed wording and branding in the welcome email flow.'
                    ),
                },
            ],
        })

        result = await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        assert result == [flag], (
            'Finding must be KEPT when no done task covers its key terms (real '
            f'systemic signal must be preserved); got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_non_systemic_pattern_flag_is_kept_and_get_tasks_not_called(self):
        """(KEEP, scope guard) Non-systemic_pattern flag_type is out of scope.

        get_tasks must NOT be called — the candidate list is empty before any
        taskmaster I/O is attempted.
        """
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = {
            'task_id': '42',
            'category': 'stale_metadata',
            'flag_type': 'stale_metadata',
            'description': 'This idea was never converted to a tracked task.',
        }
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        result = await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        assert result == [flag], (
            f'Non-systemic_pattern flag must pass through unchanged; got {result!r}'
        )
        taskmaster.get_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_systemic_pattern_without_never_tracked_language_is_kept_and_get_tasks_not_called(
        self,
    ):
        """(KEEP, scope guard) systemic_pattern flag lacking never-tracked language.

        get_tasks must NOT be called — the candidate list is empty before any
        taskmaster I/O is attempted.
        """
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = {
            'task_id': None,
            'category': 'systemic_pattern',
            'flag_type': 'systemic_pattern',
            'description': (
                'Recurring pattern: agents keep re-deriving the same cache diff '
                'each cycle.'
            ),
        }
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        result = await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        assert result == [flag], (
            f'systemic_pattern flag without never-tracked language must be kept; got {result!r}'
        )
        taskmaster.get_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_candidate_with_too_few_key_terms_is_kept(self):
        """(KEEP) Fewer than min_key_terms distinctive terms — cannot match confidently.

        Even though a done task's text would otherwise fully cover this
        candidate's (tiny) term set, too few distinctive terms means the match
        cannot be trusted, so the finding must survive.
        """
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = {
            'task_id': None,
            'category': 'systemic_pattern',
            'flag_type': 'systemic_pattern',
            'description': 'This was never tracked: fix the cache.',
        }
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={
            'tasks': [
                {
                    'id': '1',
                    'status': 'done',
                    'title': 'fix the cache',
                    'description': 'fix the cache',
                },
            ],
        })

        result = await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        assert result == [flag], (
            'A candidate with fewer than min_key_terms distinctive terms must be '
            f'KEPT (cannot match confidently); got {result!r}'
        )

    # ---- done-only + fail-open edge cases (step-7) -------------------------

    @pytest.mark.asyncio
    async def test_get_tasks_called_with_done_status_only(self):
        """(a) get_tasks must be called with statuses=['done'].

        Only done/merged tasks can trigger suppression — a PENDING duplicate
        (like task 2412 in the e61b38f9 incident) must never be able to
        suppress the finding that motivated filing it.
        """
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        taskmaster.get_tasks.assert_called_once_with('/df', statuses=['done'])

    @pytest.mark.asyncio
    async def test_none_taskmaster_keeps_all_flags(self):
        """(b) taskmaster is None → no-op KEEP-all (degrade to pass-through)."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()

        result = await filter_already_tracked_systemic_patterns(None, '/df', [flag])

        assert result == [flag], (
            f'A None taskmaster must degrade to a no-op KEEP-all; got {result!r}'
        )

    @pytest.mark.asyncio
    async def test_none_dark_factory_root_keeps_all_flags_and_get_tasks_not_called(self):
        """(c) dark_factory_root is None → no-op KEEP-all, get_tasks NOT called."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        result = await filter_already_tracked_systemic_patterns(taskmaster, None, [flag])

        assert result == [flag], (
            f'A None dark_factory_root must degrade to a no-op KEEP-all; got {result!r}'
        )
        taskmaster.get_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_dark_factory_root_keeps_all_flags_and_get_tasks_not_called(self):
        """(c) dark_factory_root is '' → no-op KEEP-all, get_tasks NOT called."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(return_value={'tasks': []})

        result = await filter_already_tracked_systemic_patterns(taskmaster, '', [flag])

        assert result == [flag], (
            f"An empty-string dark_factory_root must degrade to a no-op KEEP-all; got {result!r}"
        )
        taskmaster.get_tasks.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_tasks_raising_keeps_all_flags(self):
        """(d) get_tasks raising → fail-open KEEP-all."""
        from fused_memory.reconciliation.flag_dedup import (
            filter_already_tracked_systemic_patterns,
        )

        flag = self._make_never_tracked_flag()
        taskmaster = AsyncMock()
        taskmaster.get_tasks = AsyncMock(side_effect=RuntimeError('backend down'))

        result = await filter_already_tracked_systemic_patterns(taskmaster, '/df', [flag])

        assert result == [flag], (
            f'get_tasks raising must fail-open to KEEP-all; got {result!r}'
        )

