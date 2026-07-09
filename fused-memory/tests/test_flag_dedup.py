"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

import json
import uuid as _uuid_mod
from collections import deque
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.flag_dedup import _marker_query, build_suppression_payload
from fused_memory.reconciliation.recon_ledger import ReconLedgerRecord, ReconLedgerStore

_STUB_ADD_MEMORY_RESPONSE = AddMemoryResponse(memory_ids=['stub-id'])

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

def _make_search_stub(
    *,
    suppression: list | None = None,
    marker: dict[tuple[str, str], list] | None = None,
) -> Callable[..., Awaitable[list]]:
    """Return an async callable suitable for ``AsyncMock(side_effect=...)``.

    Dispatches on ``kwargs.get('query', '')``:

    * ``query == 'stage1_flag_suppression'`` — the ``filter_suppressed`` sweep.
      Pop the front of the *suppression* queue.  Each entry in ``suppression``
      is the full list returned for that call (e.g. ``suppression=[[rec1], []]``
      means call 1 returns ``[rec1]``, call 2 returns ``[]``).

    * ``query == _marker_query(tid, ftype)`` for any configured ``(tid, ftype)``
      key — the ``find_prior_memories`` / ``confirm_marker_persisted`` call site.
      Dispatch uses equality against ``_marker_query(*key)`` (single source of
      truth; no regex parsing needed).  A single ordered queue per
      ``(task_id, flag_type)`` key (provided via ``marker``) services ALL calls
      that share that query:

        1. The pre-write ``find_prior_memories`` search (pop 1st entry)
        2. The post-write ``confirm_marker_persisted`` search (pop 2nd entry)
        3. Any confirmation retry (pop 3rd entry), etc.

      Populate the queue in call order: ``marker={('42', 'md'): [[prior], [new]]}``
      means the pre-write search returns ``[prior]`` and the confirmation returns
      ``[new]``.

    * Any other query — raises ``AssertionError`` with a clear diagnostic.

    On queue exhaustion, raises ``AssertionError`` naming the kind and how many
    entries were configured, rather than the cryptic ``StopAsyncIteration`` that
    an exhausted ``AsyncMock(side_effect=[...])`` would raise.
    """
    # Build mutable queues from the caller-supplied specs.
    suppression_queue: deque[list] = deque(suppression or [])
    suppression_configured: int = len(suppression_queue)

    marker_queues: dict[tuple[str, str], deque[list]] = {
        k: deque(v) for k, v in (marker or {}).items()
    }
    marker_configured: dict[tuple[str, str], int] = {
        k: len(v) for k, v in marker_queues.items()
    }
    # Build equality-dispatch map: canonical query string → (tid, ftype) key.
    # Keyed by _marker_query(*k) so production-format changes propagate here
    # automatically (no regex to keep in sync).
    marker_query_to_key: dict[str, tuple[str, str]] = {
        _marker_query(*k): k for k in marker_queues
    }

    async def _stub(**kwargs: object) -> list:
        query: str = str(kwargs.get('query', ''))

        if query == 'stage1_flag_suppression':
            if not suppression_queue:
                raise AssertionError(
                    f'_make_search_stub: suppression queue exhausted '
                    f'(configured {suppression_configured} entr'
                    f'{"y" if suppression_configured == 1 else "ies"} via '
                    f'suppression=[...]; add more entries to cover this call)'
                )
            return suppression_queue.popleft()

        key = marker_query_to_key.get(query)
        if key is not None:
            if key not in marker_queues:
                raise AssertionError(
                    f'_make_search_stub: unconfigured marker query for '
                    f'task_id={key[0]!r}, flag_type={key[1]!r}; '
                    f'configured keys: {list(marker_queues.keys())}'
                )
            q = marker_queues[key]
            if not q:
                n = marker_configured[key]
                raise AssertionError(
                    f'_make_search_stub: marker queue exhausted for '
                    f'task_id={key[0]!r}, flag_type={key[1]!r}; '
                    f'queue had {n} entr{"y" if n == 1 else "ies"} — '
                    f'add more via marker[{key!r}]=[...]'
                )
            return q.popleft()

        raise AssertionError(
            f"_make_search_stub: unrecognised query {query!r}; "
            f"expected 'stage1_flag_suppression' or the output of "
            f"_marker_query(tid, ftype) (e.g. {_marker_query('42', 'missing_deliverable')!r})"
        )

    return _stub


def _make_memory_result(metadata: dict | None) -> MagicMock:
    """Build a minimal mock MemoryResult; metadata may be None to model a malformed memory record."""
    r = MagicMock()
    r.metadata = metadata
    r.content = 'Stage 1 flag marker'
    return r


def _assert_valid_stage1_marker(
    call_kwargs: dict,
    *,
    task_id: str,
    flag_type: str,
    run_id: str,
) -> None:
    """Assert that an add_memory call_kwargs encodes a well-formed stage1_flag_marker.

    Centralises the marker-shape contract so that a schema change only needs to
    be updated here rather than in every test that writes or inspects a marker.
    """
    assert call_kwargs.get('category') == 'observations_and_summaries'
    meta = call_kwargs.get('metadata', {})
    assert meta.get('source') == 'stage1_flag_marker'
    assert meta.get('kind') == 'stage1_flag_marker'
    assert meta.get('task_id') == task_id
    assert meta.get('flag_type') == flag_type
    assert meta.get('run_id') == run_id
    assert meta.get('last_seen_run_id') == run_id


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
    payload = {'kind': 'stage1_flag_suppression', 'task_id': task_id}
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
async def test_dedup_flags_mirror_write_exception_does_not_raise_ledger_still_committed(
    ledger_memory_service, caplog
):
    """When the best-effort Mem0 mirror add_memory raises, dedup_flags does
    not raise and the ledger row (already committed before the mirror
    attempt) survives — only the mirror is best-effort, the ledger write is
    not gated on it.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    ledger_memory_service.add_memory = AsyncMock(side_effect=RuntimeError('mirror write failed'))

    flags = [{'task_id': '66', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=ledger_memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) Does NOT raise
    # (b) Returns flag unchanged (fresh signature — no persisted_from_run)
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]

    # (c) The ledger row is committed despite the mirror failure
    row = await _get_marker(ledger_memory_service.recon_ledger, 'p', '66', 'missing_deliverable')
    assert row is not None

    # (d) DEBUG log mentions the failure and task_id
    assert any(
        '66' in record.message and record.levelno == logging.DEBUG
        for record in caplog.records
    )


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
# TestConfirmMarkerPersisted (task-1400 step-1+) — post-write confirmation helper
# ---------------------------------------------------------------------------


class TestConfirmMarkerPersisted:
    """Tests for confirm_marker_persisted(memory_service, *, project_id, task_id,
    flag_type, run_id, log) — returns True if findable, False otherwise.

    step-1: test_returns_true_when_search_finds_marker
    step-3: test_miss_then_retry_finds_marker
    step-5: test_miss_after_retry_returns_false_and_warns_never_raises
    """

    @pytest.mark.asyncio
    async def test_returns_true_when_search_finds_marker(self):
        """Returns True when the confirmation search finds at least one matching marker.

        The bool contract (task-1413): confirm_marker_persisted answers "is the
        marker findable?" — True if at least one matching marker is returned by the
        initial search, False after retry-miss or unexpected error.  The caller
        only needs the presence sentinel; the id string is not used downstream.
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        confirmed_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'x',
            'run_id': 'r1',
        })
        confirmed_marker.id = 'canonical-XYZ'

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[confirmed_marker])

        result = await confirm_marker_persisted(
            memory_service,
            project_id='p',
            task_id='42',
            flag_type='x',
            run_id='r1',
            log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
        )

        assert result is True, (
            f"confirm_marker_persisted must return True when marker is findable; got {result!r}"
        )
        assert isinstance(result, bool), (
            f"confirm_marker_persisted must return bool, not {type(result)!r}"
        )

    @pytest.mark.asyncio
    async def test_miss_then_retry_finds_marker(self, caplog):
        """On a first-search miss, one WARNING is emitted and a retry search is performed.

        Step-3 RED: current impl only does one search → on miss returns None immediately,
        no WARNING emitted, no retry.

        search side_effect: [[] (miss), [marker with id='retry-canon']] (retry hit)

        Asserts:
        (a) Returns the canonical id from the retry result ('retry-canon').
        (b) memory_service.search was called exactly 2 times.
        (c) Exactly one WARNING emitted BEFORE the retry containing task_id AND flag_type.
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        retry_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'x',
            'run_id': 'r1',
        })
        retry_marker.id = 'retry-canon'

        memory_service = AsyncMock()
        # First call: miss ([]); second call (retry): hit
        memory_service.search = AsyncMock(side_effect=[[], [retry_marker]])

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await confirm_marker_persisted(
                memory_service,
                project_id='p',
                task_id='42',
                flag_type='x',
                run_id='r1',
                log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            )

        # (a) Returns True when retry finds the marker
        assert result is True, (
            f"Expected True (marker found on retry) but got {result!r}"
        )
        assert isinstance(result, bool), (
            f"confirm_marker_persisted must return bool, not {type(result)!r}"
        )
        # (b) Exactly 2 search calls (initial + 1 retry)
        assert memory_service.search.call_count == 2, (
            f"Expected 2 search calls but got {memory_service.search.call_count}"
        )
        # (c) One WARNING mentioning task_id AND flag_type before the retry
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any('42' in m and 'x' in m for m in warning_messages), (
            f"Expected WARNING containing task_id '42' and flag_type 'x' but got: {warning_messages}"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('search_side_effect,label', [
        pytest.param([[], []], 'both_miss', id='both_miss'),
        pytest.param(RuntimeError('search exploded'), 'exception', id='exception'),
    ])
    async def test_miss_after_retry_returns_false_and_warns_never_raises(
        self, search_side_effect, label, caplog
    ):
        """On double-miss (or search exception), returns False without raising; final WARNING emitted.

        Step-5 RED: current impl already handles both-miss path (should pass),
        but the exception path (side_effect=RuntimeError) needs verification that
        find_prior_memories degrades it to [] so helper still returns False not raises.

        Two parametrizations:
        - both_miss:  search returns [] twice → final WARNING + False
        - exception:  search raises RuntimeError → find_prior_memories catches it → []
                      → first miss → WARNING + retry → [] → final WARNING + False

        Asserts:
        (a) Returns False (bool).
        (b) Does NOT raise.
        (c) A final WARNING is emitted containing task_id AND flag_type
            with phrasing like 'could not confirm' (ops-greppable).
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        memory_service = AsyncMock()
        if isinstance(search_side_effect, list):
            memory_service.search = AsyncMock(side_effect=search_side_effect)
        else:
            # RuntimeError — find_prior_memories catches this and returns []
            memory_service.search = AsyncMock(side_effect=search_side_effect)

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await confirm_marker_persisted(
                memory_service,
                project_id='p',
                task_id='55',
                flag_type='stale_metadata',
                run_id='r2',
                log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            )

        # (a) Returns False (bool)
        assert result is False, f"[{label}] Expected False but got {result!r}"
        assert isinstance(result, bool), (
            f"[{label}] confirm_marker_persisted must return bool, not {type(result)!r}"
        )

        # (b) Does NOT raise (no exception propagated — verified by reaching here)

        # (c) Final WARNING mentioning task_id AND flag_type with 'could not confirm'
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            '55' in m and 'stale_metadata' in m and 'could not confirm' in m
            for m in warning_messages
        ), (
            f"[{label}] Expected final WARNING with 'could not confirm' + task_id + flag_type "
            f"but got: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_ignores_stale_prior_with_different_run_id(self, caplog):
        """Confirmation search must NOT accept a stale prior whose run_id differs from the current run.

        The confirmation's job is 'did MY write for THIS run land?': a prior from an
        earlier run must not masquerade as confirmation of the current write.

        Setup: memory_service.search returns ONLY a stale prior with run_id='r0' on
        BOTH the first call and the retry (side_effect=[[stale_prior],[stale_prior]]).
        Call confirm_marker_persisted with run_id='r1'.

        Asserts:
        (a) Returns None — the prior's run_id 'r0' != current 'r1', must NOT be accepted.
        (b) memory_service.search.call_count == 2 (miss → retry → miss).
        (c) Final 'could not confirm' WARNING fires containing task_id + flag_type.

        Fails until step-15 adds run_id to the confirmation kind filter so stale priors
        are not incorrectly matched.
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import confirm_marker_persisted

        stale_prior = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'x',
            'run_id': 'r0',  # OLD run — NOT the current run 'r1'
        })
        stale_prior.id = 'stale-prior'

        memory_service = AsyncMock()
        # Both calls return the stale prior (old run_id 'r0')
        memory_service.search = AsyncMock(side_effect=[[stale_prior], [stale_prior]])

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await confirm_marker_persisted(
                memory_service,
                project_id='p',
                task_id='42',
                flag_type='x',
                run_id='r1',
                log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            )

        # (a) Returns False — stale prior must NOT be accepted as confirmation of 'r1' write
        assert result is False, (
            f"Expected False (stale prior run_id='r0' must not confirm run_id='r1') "
            f"but got {result!r}"
        )
        assert isinstance(result, bool), (
            f"confirm_marker_persisted must return bool, not {type(result)!r}"
        )
        # (b) Both attempts made: miss → retry → miss
        assert memory_service.search.call_count == 2, (
            f"Expected 2 search calls (initial + 1 retry) but got {memory_service.search.call_count}"
        )
        # (c) Final 'could not confirm' WARNING fires containing task_id + flag_type
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(
            '42' in m and 'x' in m and 'could not confirm' in m
            for m in warning_messages
        ), (
            f"Expected final WARNING with 'could not confirm' + task_id '42' + flag_type 'x' "
            f"but got: {warning_messages}"
        )

    @pytest.mark.asyncio
    async def test_confirm_marker_retry_waits_for_configured_delay(self, monkeypatch):
        """On first-search miss, _sleep(_CONFIRM_RETRY_DELAY_SECS) is called before the retry.

        RED (step-1 task-1415): _sleep and _CONFIRM_RETRY_DELAY_SECS do not yet exist
        in flag_dedup — the import will FAIL with AttributeError until step-2 adds them.

        Test strategy:
        - Monkeypatch _CONFIRM_RETRY_DELAY_SECS to a sentinel value 0.123.
        - Install a recording coroutine as _sleep to capture calls and their order
          relative to memory_service.search calls.
        - search side_effect: [[] (miss), [retry_marker]] (miss then retry-hit).
        - Call confirm_marker_persisted and assert:
          (a) _sleep was called exactly once with 0.123 (the sentinel).
          (b) _sleep was called AFTER the first search miss and BEFORE the retry search
              (verify via ordered event log capturing both search and sleep events).
          (c) Function returns True (retry found the marker).
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import (
            _CONFIRM_RETRY_DELAY_SECS,  # noqa: F401 — AttributeError if absent (RED)
            _sleep,  # noqa: F401 — AttributeError if absent (RED)
            confirm_marker_persisted,
        )

        # Sentinel delay so the test pins the exact value forwarded to _sleep.
        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRM_RETRY_DELAY_SECS', 0.123)

        retry_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '77',
            'flag_type': 'stale_metadata',
            'run_id': 'rX',
        })
        retry_marker.id = 'retry-canonical'

        # Shared ordered log: each event is ('search', call_count) or ('sleep', delay).
        event_log: list[tuple[str, object]] = []

        memory_service = AsyncMock()

        # Wrap search to record events with a running counter.
        _search_call_count = 0

        async def recording_search(*args, **kwargs):
            nonlocal _search_call_count
            _search_call_count += 1
            n = _search_call_count
            event_log.append(('search', n))
            if n == 1:
                return []
            return [retry_marker]

        memory_service.search = recording_search

        # Recording sleep coroutine.
        async def recording_sleep(delay: float) -> None:
            event_log.append(('sleep', delay))

        monkeypatch.setattr(_flag_dedup_mod, '_sleep', recording_sleep)

        result = await confirm_marker_persisted(
            memory_service,
            project_id='p',
            task_id='77',
            flag_type='stale_metadata',
            run_id='rX',
            log=logging.getLogger('fused_memory.reconciliation.flag_dedup'),
        )

        # (a) Returns True — retry found the marker.
        assert result is True, f"Expected True (retry found marker) but got {result!r}"
        assert isinstance(result, bool), (
            f"confirm_marker_persisted must return bool, not {type(result)!r}"
        )

        # (b) _sleep was called exactly once with the sentinel value 0.123.
        sleep_events = [(name, val) for name, val in event_log if name == 'sleep']
        assert len(sleep_events) == 1, (
            f"Expected _sleep called exactly once; event_log: {event_log}"
        )
        assert sleep_events[0][1] == 0.123, (
            f"_sleep must be called with sentinel 0.123; got: {sleep_events[0][1]}"
        )

        # (c) Order: search-1 (miss) → sleep → search-2 (retry hit).
        assert event_log == [('search', 1), ('sleep', 0.123), ('search', 2)], (
            f"Expected ordered events [search-1, sleep, search-2]; got: {event_log}"
        )


# ---------------------------------------------------------------------------
# task-1412 — Confirmation circuit-breaker tests
# ---------------------------------------------------------------------------


class TestConfirmationCircuitBreaker:
    """Per-invocation circuit-breaker: trips after N consecutive confirmation
    misses and falls back to bool(memory_ids) gate for the remainder of the
    batch.  Counter resets to zero at the start of each dedup_flags invocation.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('flag_c_response,expect_c_deleted', [
        pytest.param(
            AddMemoryResponse(memory_ids=['x']), True,
            id='non_empty_memory_ids',
        ),
        pytest.param(
            AddMemoryResponse(memory_ids=[]), False,
            id='empty_memory_ids',
        ),
    ])
    async def test_hit_path_trips_after_threshold_consecutive_misses_and_falls_back_to_memory_ids(
        self, flag_c_response, expect_c_deleted, monkeypatch, caplog,
    ):
        """HIT path: after _CONFIRMATION_MISS_THRESHOLD consecutive confirmation
        misses the circuit-breaker trips; remaining flags skip confirm_marker_persisted
        and gate prior-deletion on bool(response.memory_ids) instead.

        Monkeypatches threshold to 2 so only 3 HIT-path flags are needed.
        Flags A and B each miss confirmation (2 searches each: initial + retry).
        After B's miss the counter reaches 2 → breaker trips.
        Flag C's confirmation call is entirely skipped (search.call_count stays at 8).
        Flag C's prior-deletion gate is driven off bool(flag_c_response.memory_ids).

        Asserts:
          (a) Exactly ONE circuit-breaker WARNING mentioning the threshold count '2'.
          (b) non-empty memory_ids: prior-C deleted (write_succeeded = True via memory_ids).
          (c) empty memory_ids: prior-C NOT deleted (write_succeeded = False).
          (d) search.call_count == 8 (no confirmation searches for flag-C).
          (e) priors A and B NOT deleted: their confirmed_id=None → write_succeeded=False
              (circuit-breaker only affects flag-C; A and B follow the normal path).
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        run_id = 'r1'

        prior_a = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '101',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_a.id = 'prior-a'

        prior_b = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '102',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_b.id = 'prior-b'

        prior_c = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '103',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_c.id = 'prior-c'

        # 10 search results (elements [8] and [9] are only reached without the
        # circuit-breaker, letting search.call_count==8 catch the failure):
        #   [0]  suppression filter → []
        #   [1]  flag-A pre-write   → [prior-a]  (HIT)
        #   [2]  flag-A confirmation miss         → counter = 1
        #   [3]  flag-A confirmation retry-miss
        #   [4]  flag-B pre-write   → [prior-b]  (HIT)
        #   [5]  flag-B confirmation miss         → counter = 2 → TRIP
        #   [6]  flag-B confirmation retry-miss
        #   [7]  flag-C pre-write   → [prior-c]  (HIT)
        #   [8]  (only reached without circuit-breaker: flag-C confirmation miss)
        #   [9]  (only reached without circuit-breaker: flag-C confirmation retry)
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],         # [0] suppression
            [prior_a],  # [1] A pre-write HIT
            [],         # [2] A confirmation miss
            [],         # [3] A confirmation retry
            [prior_b],  # [4] B pre-write HIT
            [],         # [5] B confirmation miss → counter = 1
            [],         # [6] B confirmation retry → counter = 2 → TRIP
            [prior_c],  # [7] C pre-write HIT
            [],         # [8] C confirmation miss  (without breaker)
            [],         # [9] C confirmation retry (without breaker)
        ])
        memory_service.add_memory = AsyncMock(side_effect=[
            _STUB_ADD_MEMORY_RESPONSE,  # flag-A
            _STUB_ADD_MEMORY_RESPONSE,  # flag-B
            flag_c_response,            # flag-C (parametrized)
        ])
        memory_service.delete_memory = AsyncMock(return_value=None)

        flags = [
            {'task_id': 101, 'flag_type': 'missing_deliverable', 'description': 'A'},
            {'task_id': 102, 'flag_type': 'missing_deliverable', 'description': 'B'},
            {'task_id': 103, 'flag_type': 'missing_deliverable', 'description': 'C'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id=run_id,
                flags=flags,
            )

        # (a) Exactly ONE circuit-breaker WARNING mentioning the count '2'
        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 1, (
            f"Expected exactly 1 circuit-breaker WARNING but got "
            f"{len(breaker_warnings)}: {breaker_warnings}\nAll WARNINGs: {all_warnings}"
        )
        assert '2' in breaker_warnings[0], (
            f"Breaker WARNING must mention the count '2'; got: {breaker_warnings[0]}"
        )

        # (b)/(c) Flag-C prior-deletion gated on bool(memory_ids) after trip
        deleted_ids = [
            c.kwargs.get('memory_id') for c in memory_service.delete_memory.call_args_list
        ]
        if expect_c_deleted:
            # non-empty memory_ids → write_succeeded = True → prior-c deleted
            assert 'prior-c' in deleted_ids, (
                f"non-empty memory_ids: expected prior-c deleted, got: {deleted_ids}"
            )
        else:
            # empty memory_ids → write_succeeded = False → prior-c NOT deleted
            assert 'prior-c' not in deleted_ids, (
                f"empty memory_ids: expected prior-c NOT deleted, got: {deleted_ids}"
            )

        # (d) Exactly 8 search calls — no confirmation for flag-C
        assert memory_service.search.call_count == 8, (
            f"Expected 8 search calls (1 suppression + 3 pre-write + 2+2 confirmations "
            f"for A+B, none for C), got: {memory_service.search.call_count}"
        )

        # (e) Priors A and B NOT deleted: confirmation missed → confirmed_id=None
        assert 'prior-a' not in deleted_ids, (
            f"prior-a must not be deleted (confirmed_id=None for A), got: {deleted_ids}"
        )
        assert 'prior-b' not in deleted_ids, (
            f"prior-b must not be deleted (confirmed_id=None for B), got: {deleted_ids}"
        )

        # All 3 flags returned and annotated as HIT-path
        assert len(result) == 3
        for f in result:
            assert 'persisted_from_run' in f, f"Expected HIT-path annotation: {f}"

    @pytest.mark.asyncio
    async def test_consecutive_misses_reset_on_successful_confirmation(
        self, monkeypatch, caplog,
    ):
        """Counter resets to 0 on any successful confirmation (strictly consecutive).

        Monkeypatches threshold to 3.  Five HIT-path flags with confirmation
        pattern: miss-miss-HIT-miss-miss.  After the HIT (flag-3) the counter
        resets to 0, so flag-4+flag-5 only accumulate 2 consecutive misses
        (< 3 → no trip).

        search side_effect (15 total):
          [0]  suppression → []
          [1]  flag-1 pre-write → [prior-1]  (HIT)
          [2]  flag-1 confirmation miss
          [3]  flag-1 confirmation retry
          [4]  flag-2 pre-write → [prior-2]  (HIT)
          [5]  flag-2 confirmation miss  → counter=2
          [6]  flag-2 confirmation retry
          [7]  flag-3 pre-write → [prior-3]  (HIT)
          [8]  flag-3 confirmation HIT   → counter=0 (reset)
          [9]  flag-4 pre-write → [prior-4]  (HIT)
          [10] flag-4 confirmation miss  → counter=1
          [11] flag-4 confirmation retry
          [12] flag-5 pre-write → [prior-5]  (HIT)
          [13] flag-5 confirmation miss  → counter=2 (<3, no trip)
          [14] flag-5 confirmation retry

        Without counter-reset (step-2's impl only): counter after flag-4 miss
        would be 3 → breaker trips → flag-5 skips confirmation → search.call_count
        drops to 13 and a breaker WARNING is emitted.

        Asserts:
          (a) NO circuit-breaker WARNING emitted.
          (b) search.call_count == 15 (all 5 flags confirmed; no short-circuit).
          (c) delete_memory.call_count == 1 (only flag-3: only flag whose confirmed_id
              is non-None; flags 1,2,4,5 miss confirmation → confirmed_id=None → not deleted).

        Will FAIL until step-4 adds `consecutive_confirmation_misses = 0` on hit.
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 3)

        run_id = 'r1'

        def _make_prior(task_id: str) -> MagicMock:
            p = _make_memory_result({
                'source': 'stage1_flag_marker', 'task_id': task_id,
                'flag_type': 'missing_deliverable', 'run_id': 'r0',
            })
            p.id = f'prior-{task_id}'
            return p

        # New marker from the current run for flag-3's confirmation search.
        new_3 = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '203',
            'flag_type': 'missing_deliverable', 'run_id': run_id,
        })
        new_3.id = 'new-3'

        priors = {tid: _make_prior(tid) for tid in ['201', '202', '203', '204', '205']}

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],                   # [0]  suppression
            [priors['201']],      # [1]  flag-1 pre-write HIT
            [],                   # [2]  flag-1 confirmation miss
            [],                   # [3]  flag-1 confirmation retry   → counter=1
            [priors['202']],      # [4]  flag-2 pre-write HIT
            [],                   # [5]  flag-2 confirmation miss
            [],                   # [6]  flag-2 confirmation retry   → counter=2
            [priors['203']],      # [7]  flag-3 pre-write HIT
            [new_3],              # [8]  flag-3 confirmation HIT     → counter=0 (reset)
            [priors['204']],      # [9]  flag-4 pre-write HIT
            [],                   # [10] flag-4 confirmation miss
            [],                   # [11] flag-4 confirmation retry   → counter=1
            [priors['205']],      # [12] flag-5 pre-write HIT
            [],                   # [13] flag-5 confirmation miss
            [],                   # [14] flag-5 confirmation retry   → counter=2 (<3)
        ])
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
        memory_service.delete_memory = AsyncMock(return_value=None)

        flags = [
            {'task_id': int(tid), 'flag_type': 'missing_deliverable', 'description': f'flag-{tid}'}
            for tid in ['201', '202', '203', '204', '205']
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id=run_id,
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (a) NO circuit-breaker WARNING (counter resets on flag-3 hit)
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 0, (
            f"Expected NO circuit-breaker WARNING but got: {breaker_warnings}\n"
            f"All WARNINGs: {all_warnings}"
        )

        # (b) All 5 confirmations were attempted — no short-circuit
        assert memory_service.search.call_count == 15, (
            f"Expected 15 search calls (1 suppression + 5 pre-write + "
            f"2+2+1+2+2 confirmations), got: {memory_service.search.call_count}"
        )

        # (c) Only flag-3's prior deleted (confirmed_id non-None only for flag-3)
        assert memory_service.delete_memory.call_count == 1, (
            f"Expected 1 deletion (only flag-3's prior) but got "
            f"{memory_service.delete_memory.call_count}"
        )
        deleted_ids = [
            c.kwargs.get('memory_id') for c in memory_service.delete_memory.call_args_list
        ]
        assert deleted_ids == ['prior-203'], (
            f"Expected only prior-203 deleted, got: {deleted_ids}"
        )

        # All 5 flags returned, all HIT-path annotated
        assert len(result) == 5
        for f in result:
            assert 'persisted_from_run' in f, f"Expected HIT annotation: {f}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize('flag_3_response,expect_noop_warning', [
        pytest.param(
            AddMemoryResponse(memory_ids=['x']), False,
            id='non_empty_fallback_no_warning',
        ),
        pytest.param(
            AddMemoryResponse(memory_ids=[]), True,
            id='empty_fallback_emits_warning',
        ),
    ])
    async def test_miss_path_shares_counter_and_falls_back_to_memory_ids(
        self, flag_3_response, expect_noop_warning, monkeypatch, caplog,
    ):
        """MISS path shares the same circuit-breaker counter as HIT path.

        Monkeypatches threshold to 2.  Three MISS-path flags (no priors found).
        Flags 1 and 2 each miss confirmation (2 searches each).  After flag-2's
        miss the counter reaches 2 → breaker trips.  Flag-3's confirmation call
        is entirely skipped; the "will not be detected next cycle" WARNING is
        driven off bool(response.memory_ids) rather than off confirmed_id.

        search side_effect (8 total; elements [8]/[9] only reached without breaker):
          [0]  suppression → []
          [1]  flag-1 pre-write → []  (MISS)
          [2]  flag-1 confirmation miss
          [3]  flag-1 confirmation retry     → counter=1
          [4]  flag-2 pre-write → []  (MISS)
          [5]  flag-2 confirmation miss
          [6]  flag-2 confirmation retry     → counter=2 → TRIP
          [7]  flag-3 pre-write → []  (MISS)
          [8]  (flag-3 confirmation miss — only without breaker)
          [9]  (flag-3 confirmation retry   — only without breaker)

        Parametrized on flag-3's add_memory response:
          non_empty_fallback_no_warning:  memory_ids=['x'] → no "will not be detected" WARNING.
          empty_fallback_emits_warning:   memory_ids=[]    → "will not be detected" WARNING fires.

        Asserts (both parametrizations):
          (i)  Exactly ONE circuit-breaker WARNING.
          (ii) flag-3's add_memory called once.
          (iii) search.call_count == 8 (no confirmation searches for flag-3).
          (iv) "will not be detected next cycle" WARNING presence matches memory_ids gate.

        Will FAIL until step-6 wires MISS branch through circuit-breaker.
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        memory_service = AsyncMock()
        # 10 elements: elements [8] and [9] only consumed without the circuit-breaker.
        memory_service.search = AsyncMock(side_effect=[
            [],   # [0]  suppression
            [],   # [1]  flag-1 pre-write MISS
            [],   # [2]  flag-1 confirmation miss
            [],   # [3]  flag-1 confirmation retry    → counter=1
            [],   # [4]  flag-2 pre-write MISS
            [],   # [5]  flag-2 confirmation miss
            [],   # [6]  flag-2 confirmation retry    → counter=2 → TRIP
            [],   # [7]  flag-3 pre-write MISS
            [],   # [8]  flag-3 confirmation miss  (without breaker only)
            [],   # [9]  flag-3 confirmation retry (without breaker only)
        ])
        memory_service.add_memory = AsyncMock(side_effect=[
            _STUB_ADD_MEMORY_RESPONSE,  # flag-1
            _STUB_ADD_MEMORY_RESPONSE,  # flag-2
            flag_3_response,            # flag-3 (parametrized)
        ])

        flags = [
            {'task_id': 301, 'flag_type': 'missing_deliverable', 'description': 'miss-1'},
            {'task_id': 302, 'flag_type': 'missing_deliverable', 'description': 'miss-2'},
            {'task_id': 303, 'flag_type': 'missing_deliverable', 'description': 'miss-3'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id='r1',
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (i) Exactly ONE circuit-breaker WARNING
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 1, (
            f"Expected 1 breaker WARNING but got {len(breaker_warnings)}: "
            f"{breaker_warnings}\nAll WARNINGs: {all_warnings}"
        )
        assert '2' in breaker_warnings[0], (
            f"Breaker WARNING must mention count '2': {breaker_warnings[0]}"
        )

        # (ii) flag-3's add_memory called once (write still happens even after trip)
        assert memory_service.add_memory.call_count == 3

        # (iii) Exactly 8 search calls — flag-3 confirmation short-circuited
        assert memory_service.search.call_count == 8, (
            f"Expected 8 search calls (1 suppression + 3 pre-write + 2+2 confirmations "
            f"for flags 1+2, none for flag-3), got: {memory_service.search.call_count}"
        )

        # (iv) "will not be detected next cycle" WARNING driven by memory_ids gate
        noop_warnings = [
            m for m in all_warnings
            if 'will not be detected' in m and '303' in m
        ]
        if expect_noop_warning:
            # empty memory_ids → bool([]) = False → WARNING fires for flag-3
            assert len(noop_warnings) >= 1, (
                f"Expected 'will not be detected' WARNING for task 303 "
                f"(empty memory_ids) but got none.\nAll WARNINGs: {all_warnings}"
            )
        else:
            # non-empty memory_ids → bool(['x']) = True → no WARNING for flag-3
            assert len(noop_warnings) == 0, (
                f"Expected NO 'will not be detected' WARNING for task 303 "
                f"(non-empty memory_ids) but got: {noop_warnings}"
            )

        # All 3 flags returned; MISS path so no persisted_from_run
        assert len(result) == 3
        for f in result:
            assert 'persisted_from_run' not in f, f"MISS path must not annotate: {f}"

    @pytest.mark.asyncio
    async def test_circuit_breaker_budget_is_fresh_per_dedup_flags_invocation(
        self, monkeypatch, caplog,
    ):
        """Per-invocation freshness: each dedup_flags call starts with counter=0.

        This is a regression pin.  The counter is function-local so it resets
        automatically at each call.  A future refactor that lifts it to module
        scope would break this test.

        Monkeypatches threshold to 2.  Calls dedup_flags TWICE on the same
        memory_service.  Each call has 3 MISS-path flags whose confirmations
        all miss.  Each call's confirmation pattern:
          flag-1: miss+retry → counter=1
          flag-2: miss+retry → counter=2 → TRIP
          flag-3: skipped (short-circuit)

        search side_effect: 8 elements per call × 2 calls = 16 total.
          Per-call pattern:
            [n+0] suppression → []
            [n+1] flag-1 pre-write → []
            [n+2] flag-1 confirmation miss
            [n+3] flag-1 confirmation retry    → counter=1
            [n+4] flag-2 pre-write → []
            [n+5] flag-2 confirmation miss
            [n+6] flag-2 confirmation retry    → counter=2 → TRIP
            [n+7] flag-3 pre-write → []

        Asserts:
          (a) Exactly TWO breaker WARNINGs across both calls (one per invocation).
          (b) Total search.call_count == 16 (each call's flag-3 short-circuits).
          (c) Second call's flag-1 confirmation IS reached (proves counter reset
              between invocations).  Evidence: call-2's pattern matches call-1's
              pattern (if counter carried over it would trip immediately on flag-1
              and total search count would be less).
          (d) Both calls return 3-element MISS-path flag lists (no annotation).
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        # 16 search results: 8 per call, same pattern repeated.
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            # --- call 1 ---
            [],   # [0]  suppression
            [],   # [1]  flag-1 pre-write MISS
            [],   # [2]  flag-1 confirmation miss
            [],   # [3]  flag-1 confirmation retry   → counter=1
            [],   # [4]  flag-2 pre-write MISS
            [],   # [5]  flag-2 confirmation miss
            [],   # [6]  flag-2 confirmation retry   → counter=2 → TRIP
            [],   # [7]  flag-3 pre-write MISS (no confirmation)
            # --- call 2 ---
            [],   # [8]  suppression
            [],   # [9]  flag-1 pre-write MISS  ← proves counter reset (fresh budget)
            [],   # [10] flag-1 confirmation miss
            [],   # [11] flag-1 confirmation retry  → counter=1
            [],   # [12] flag-2 pre-write MISS
            [],   # [13] flag-2 confirmation miss
            [],   # [14] flag-2 confirmation retry  → counter=2 → TRIP
            [],   # [15] flag-3 pre-write MISS (no confirmation)
        ])
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        flags = [
            {'task_id': 401, 'flag_type': 'missing_deliverable', 'description': 'f1'},
            {'task_id': 402, 'flag_type': 'missing_deliverable', 'description': 'f2'},
            {'task_id': 403, 'flag_type': 'missing_deliverable', 'description': 'f3'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result1 = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id='r1',
                flags=list(flags),  # copy so mutations don't bleed
            )
            result2 = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id='r2',
                flags=list(flags),
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (a) Exactly TWO breaker WARNINGs (one per invocation)
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 2, (
            f"Expected exactly 2 breaker WARNINGs (one per invocation) but got "
            f"{len(breaker_warnings)}: {breaker_warnings}"
        )

        # (b) Total 16 search calls (each invocation's flag-3 short-circuits)
        assert memory_service.search.call_count == 16, (
            f"Expected 16 total search calls (8 per invocation) but got: "
            f"{memory_service.search.call_count}"
        )

        # (c) Both calls returned 3-element MISS-path lists
        assert len(result1) == 3
        assert len(result2) == 3
        for f in result1 + result2:
            assert 'persisted_from_run' not in f, f"MISS path must not annotate: {f}"

    @pytest.mark.asyncio
    async def test_add_memory_exceptions_do_not_count_toward_threshold(
        self, monkeypatch, caplog,
    ):
        """Write failures (add_memory exceptions) do NOT advance the miss counter.

        The circuit-breaker targets the *confirmation* overhead: the extra search
        round-trip after a successful write.  When add_memory itself raises, the
        confirmation call is never reached so neither counter nor disabled flag is
        touched.  A batch where every add_memory raises therefore never trips the
        breaker — this pins that contract against an incorrect future change that
        would increment the counter on write failure.

        Setup: 3 MISS-path flags, all with add_memory raising RuntimeError.
        search side_effect: 1 suppression + 3 pre-write (no confirmation searches
        because add_memory always fails before _confirm_and_track is called).

        Asserts:
          (a) No breaker WARNING emitted (counter never advanced).
          (b) search.call_count == 4 (1 suppression + 3 pre-write; zero confirmations).
          (c) All 3 flags returned (exceptions logged, not raised).
          (d) A per-flag WARNING is emitted for each failed write.
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],   # [0] suppression
            [],   # [1] flag-1 pre-write MISS
            [],   # [2] flag-2 pre-write MISS
            [],   # [3] flag-3 pre-write MISS
            # No confirmation searches — add_memory always raises before them
        ])
        memory_service.add_memory = AsyncMock(
            side_effect=RuntimeError('Mem0 write failure'),
        )

        flags = [
            {'task_id': 501, 'flag_type': 'stale'},
            {'task_id': 502, 'flag_type': 'stale'},
            {'task_id': 503, 'flag_type': 'stale'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id='r1',
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (a) No breaker WARNING — counter never advanced via write failures
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 0, (
            f'Breaker must not trip on write-only failures; got: {breaker_warnings}'
        )

        # (b) Exactly 4 searches: 1 suppression + 3 pre-write; no confirmation searches
        assert memory_service.search.call_count == 4, (
            f'Expected 4 searches (1 suppression + 3 pre-write) but got: '
            f'{memory_service.search.call_count}'
        )

        # (c) All 3 flags returned (exceptions are logged, not raised)
        assert len(result) == 3

        # (d) Per-flag write-failure WARNINGs emitted (one per flag).
        # The unified _write_and_confirm_marker exception handler emits
        # 'flag_dedup: failed to write marker for task %s flag_type %s: %s'
        # for both HIT and MISS branches.
        write_fail_warnings = [
            m for m in all_warnings
            if 'failed to write marker' in m
        ]
        assert len(write_fail_warnings) == 3, (
            f'Expected 3 per-flag write-failure WARNINGs but got: {write_fail_warnings}'
        )

    # -----------------------------------------------------------------------
    # Fix 3 regression-pin tests (task-1413, step-1)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_breaker_stays_tripped_no_un_trip_invariant(
        self, monkeypatch, caplog,
    ):
        """Once tripped, a later flag whose confirmation WOULD succeed does NOT un-trip the breaker.

        Pins the invariant that ``confirmation_disabled`` is only ever set to True
        (never reset to False mid-batch once tripped).  The ``else`` branch in
        ``_confirm_and_track`` does NOT call ``confirm_marker_persisted`` and does NOT
        touch the counter, so a queued successful search result is never consumed.

        Setup: threshold=2, 3 HIT-path flags.
        - Flags 1 and 2 each miss confirmation → counter reaches 2 → TRIP.
        - Flag 3's pre-write HIT is returned but its confirmation search entries
          [8] and [9] would succeed — they are NOT consumed (breaker is tripped).

        Asserts:
          (a) search.call_count == 8 — sentinel entries [8] and [9] not consumed.
          (b) Exactly ONE breaker WARNING (``'tripped after'`` substring); no re-trip.
          (c) All three flags annotated with ``persisted_from_run`` (HIT-path).
          (d) delete_memory called only once with memory_id='prior-c'.
              (flags 1+2 active-miss → write_succeeded=False; flag 3 tripped-branch
               → write_succeeded=bool(memory_ids)=True)
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        run_id = 'r1'

        prior_a = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1101',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_a.id = 'prior-a'

        prior_b = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1102',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_b.id = 'prior-b'

        prior_c = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1103',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_c.id = 'prior-c'

        # Sentinel entries [8] and [9]: if the breaker could un-trip, flag-3's
        # confirm_marker_persisted would consume these and return a non-None id.
        # Asserting search.call_count==8 proves these were NOT consumed.
        would_succeed_marker = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1103',
            'flag_type': 'missing_deliverable', 'run_id': 'r1',
        })
        would_succeed_marker.id = 'would-succeed'

        # 10-element side_effect; entries [8] and [9] are non-consumption sentinels.
        #   [0]  suppression filter → []
        #   [1]  flag-1 pre-write   → [prior_a]  (HIT)
        #   [2]  flag-1 confirmation initial miss
        #   [3]  flag-1 confirmation retry miss  → counter = 1
        #   [4]  flag-2 pre-write   → [prior_b]  (HIT)
        #   [5]  flag-2 confirmation initial miss
        #   [6]  flag-2 confirmation retry miss  → counter = 2 → TRIP
        #   [7]  flag-3 pre-write   → [prior_c]  (HIT)
        #   [8]  (sentinel: would confirm flag-3 if NOT tripped — must not be consumed)
        #   [9]  (sentinel: would confirm flag-3 if NOT tripped — must not be consumed)
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],                      # [0] suppression
            [prior_a],               # [1] flag-1 pre-write HIT
            [],                      # [2] flag-1 confirmation initial miss
            [],                      # [3] flag-1 confirmation retry miss → counter=1
            [prior_b],               # [4] flag-2 pre-write HIT
            [],                      # [5] flag-2 confirmation initial miss
            [],                      # [6] flag-2 confirmation retry miss → counter=2 → TRIP
            [prior_c],               # [7] flag-3 pre-write HIT
            [would_succeed_marker],  # [8] sentinel: NOT consumed (breaker tripped)
            [would_succeed_marker],  # [9] sentinel: NOT consumed (breaker tripped)
        ])
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
        memory_service.delete_memory = AsyncMock(return_value=None)

        flags = [
            {'task_id': 1101, 'flag_type': 'missing_deliverable', 'description': 'A'},
            {'task_id': 1102, 'flag_type': 'missing_deliverable', 'description': 'B'},
            {'task_id': 1103, 'flag_type': 'missing_deliverable', 'description': 'C'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id=run_id,
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (a) Sentinel entries [8] and [9] NOT consumed — breaker stays tripped.
        # A count of 9 (or 10) would prove the breaker un-tripped and flag-3
        # consumed sentinel entry [8] (no retry needed when confirmation hits).
        assert memory_service.search.call_count == 8, (
            f'Expected 8 search calls (1 suppression + 3 pre-write + 2+2 confirmation '
            f'for flags 1+2, none for flag 3); got: {memory_service.search.call_count}. '
            f'Count of 9 or 10 would indicate the breaker failed to stay tripped.'
        )

        # (b) Exactly ONE trip WARNING — no re-trip on flag-3's tripped-branch pass.
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 1, (
            f'Expected exactly 1 breaker trip WARNING; got '
            f'{len(breaker_warnings)}: {breaker_warnings}'
        )

        # (c) All 3 flags annotated as HIT-path (annotation extracted before write).
        assert len(result) == 3
        for f in result:
            assert 'persisted_from_run' in f, f'Expected HIT-path annotation: {f}'

        # (d) Only prior-c deleted: flags 1+2 active-miss → write_succeeded=False;
        # flag-3 tripped-branch → write_succeeded=bool(memory_ids)=True.
        deleted_ids = [
            c.kwargs.get('memory_id') for c in memory_service.delete_memory.call_args_list
        ]
        assert deleted_ids == ['prior-c'], (
            f'Expected only prior-c to be deleted; got: {deleted_ids}'
        )

    @pytest.mark.asyncio
    async def test_hit_path_add_memory_exception_does_not_advance_miss_counter(
        self, monkeypatch, caplog,
    ):
        """HIT-path write failures do NOT advance the circuit-breaker miss counter.

        Complements ``test_add_memory_exceptions_do_not_count_toward_threshold``
        (which covers MISS-path flags).  Because HIT and MISS branches share the
        same ``_write_and_confirm_marker`` try/except, a future refactor that splits
        them could independently regress one without the other.

        When ``add_memory`` raises, ``_write_and_confirm_marker`` returns False
        immediately without calling ``_confirm_and_track``, so neither the miss
        counter nor ``confirmation_disabled`` is touched.

        Setup: threshold=2, 3 HIT-path flags, add_memory always raises RuntimeError.
        search side_effect: 1 suppression + 3 pre-write HITs; zero confirmation
        searches (add_memory always fails before _confirm_and_track is called).

        Asserts:
          (a) No breaker WARNING (``'tripped after'`` substring) — counter stays 0.
          (b) search.call_count == 4 (1 suppression + 3 pre-write only).
          (c) All 3 flags returned (exceptions logged, not raised).
          (d) 3 per-flag ``'failed to write marker'`` WARNINGs emitted.
          (e) All 3 flags annotated with ``persisted_from_run`` (extracted before write).
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        prior_a = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1201',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_a.id = 'prior-a'

        prior_b = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1202',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_b.id = 'prior-b'

        prior_c = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '1203',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_c.id = 'prior-c'

        # search side_effect: suppression + 3 pre-write HITs.
        # No confirmation searches because add_memory always raises before
        # _confirm_and_track is ever called.
        #   [0]  suppression filter → []
        #   [1]  flag-1 pre-write   → [prior_a]  (HIT)
        #   [2]  flag-2 pre-write   → [prior_b]  (HIT)
        #   [3]  flag-3 pre-write   → [prior_c]  (HIT)
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],        # [0] suppression
            [prior_a], # [1] flag-1 pre-write HIT
            [prior_b], # [2] flag-2 pre-write HIT
            [prior_c], # [3] flag-3 pre-write HIT
        ])
        memory_service.add_memory = AsyncMock(
            side_effect=RuntimeError('Mem0 write failure'),
        )

        flags = [
            {'task_id': 1201, 'flag_type': 'missing_deliverable', 'description': 'A'},
            {'task_id': 1202, 'flag_type': 'missing_deliverable', 'description': 'B'},
            {'task_id': 1203, 'flag_type': 'missing_deliverable', 'description': 'C'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id='r1',
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (a) No breaker WARNING — counter not advanced by HIT-path write failures.
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 0, (
            f'Breaker must not trip on HIT-path write-only failures; got: {breaker_warnings}'
        )

        # (b) Only 4 search calls: suppression + 3 pre-write; no confirmation.
        assert memory_service.search.call_count == 4, (
            f'Expected 4 searches (1 suppression + 3 pre-write) but got: '
            f'{memory_service.search.call_count}'
        )

        # (c) All 3 flags returned (exceptions are logged, not raised).
        assert len(result) == 3

        # (d) Per-flag write-failure WARNINGs emitted (one per flag).
        write_fail_warnings = [m for m in all_warnings if 'failed to write marker' in m]
        assert len(write_fail_warnings) == 3, (
            f'Expected 3 per-flag write-failure WARNINGs but got: {write_fail_warnings}'
        )

        # (e) All 3 flags annotated with persisted_from_run: annotation is extracted
        # from priors BEFORE the write attempt (lines 622-632 of flag_dedup.py), so
        # write failure does not suppress it.
        for f in result:
            assert 'persisted_from_run' in f, f'Expected HIT-path annotation: {f}'

    # -----------------------------------------------------------------------
    # Fix 2 disambiguation test (task-1413, step-4)
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    @pytest.mark.parametrize('branch', ['hit', 'miss'])
    async def test_active_vs_tripped_branch_warnings_are_disambiguated(
        self, branch, monkeypatch, caplog,
    ):
        """ACTIVE-branch and TRIPPED-branch per-flag WARNINGs use distinct wording.

        Pins the disambiguation contract (task-1413 Fix 2): when the breaker is
        ACTIVE (confirmation search attempted and missed), the per-flag WARNING
        must contain ``'could not be confirmed findable'``.  When the breaker is
        TRIPPED (no confirmation search attempted), the WARNING must contain
        ``'confirmation skipped (circuit-breaker open)'`` and
        ``'memory_ids gate failed'``.

        Setup: threshold=2, 3 flags (task_ids 901/902/903).
        - Flags 901 and 902: ACTIVE-branch confirmation miss → counter reaches 2 → TRIP.
        - Flag 903: TRIPPED branch; add_memory returns empty memory_ids → WARNING fires.

        Parametrized over branch='hit' (priors found on pre-write, consequence
        ``'skipping prior deletion'``) and branch='miss' (no priors, consequence
        ``'will not be detected next cycle'``).

        search side_effect (8 entries for both branches):
          [0]  suppression filter → []
          [1]  flag-901 pre-write → HIT/MISS
          [2]  flag-901 confirmation initial miss
          [3]  flag-901 confirmation retry miss → counter = 1
          [4]  flag-902 pre-write → HIT/MISS
          [5]  flag-902 confirmation initial miss
          [6]  flag-902 confirmation retry miss → counter = 2 → TRIP
          [7]  flag-903 pre-write → HIT/MISS (no confirmation — breaker tripped)

        Fails against the post-step-3 impl because _confirm_and_track emits the
        same ``miss_warning_msg`` in both ACTIVE and TRIPPED branches, so
        bucket_903 has ``'could not be confirmed findable'`` rather than
        ``'confirmation skipped (circuit-breaker open)'``.
        """
        import logging

        import fused_memory.reconciliation.flag_dedup as _flag_dedup_mod
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        monkeypatch.setattr(_flag_dedup_mod, '_CONFIRMATION_MISS_THRESHOLD', 2)

        run_id = 'r1'

        prior_a = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '901',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_a.id = 'prior-a'

        prior_b = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '902',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_b.id = 'prior-b'

        prior_c = _make_memory_result({
            'source': 'stage1_flag_marker', 'task_id': '903',
            'flag_type': 'missing_deliverable', 'run_id': 'r0',
        })
        prior_c.id = 'prior-c'

        # For 'hit': entries [1],[4],[7] return priors; for 'miss' they return [].
        if branch == 'hit':
            priors_901, priors_902, priors_903 = [prior_a], [prior_b], [prior_c]
        else:
            priors_901 = priors_902 = priors_903 = []

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],          # [0] suppression
            priors_901,  # [1] flag-901 pre-write
            [],          # [2] flag-901 confirmation initial miss
            [],          # [3] flag-901 confirmation retry miss → counter=1
            priors_902,  # [4] flag-902 pre-write
            [],          # [5] flag-902 confirmation initial miss
            [],          # [6] flag-902 confirmation retry miss → counter=2 → TRIP
            priors_903,  # [7] flag-903 pre-write (no confirmation — breaker tripped)
        ])
        # flag-903 gets empty memory_ids so the tripped-skip WARNING fires.
        memory_service.add_memory = AsyncMock(side_effect=[
            _STUB_ADD_MEMORY_RESPONSE,        # flag-901 (non-empty)
            _STUB_ADD_MEMORY_RESPONSE,        # flag-902 (non-empty)
            AddMemoryResponse(memory_ids=[]), # flag-903 (empty → tripped-skip WARNING)
        ])
        memory_service.delete_memory = AsyncMock(return_value=None)

        flags = [
            {'task_id': 901, 'flag_type': 'missing_deliverable'},
            {'task_id': 902, 'flag_type': 'missing_deliverable'},
            {'task_id': 903, 'flag_type': 'missing_deliverable'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id=run_id,
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        def per_flag(task_id_str: str) -> list[str]:
            """WARNINGs containing task_id_str, excluding the one-time trip line."""
            return [m for m in all_warnings if task_id_str in m and 'tripped after' not in m]

        bucket_901 = per_flag('901')
        bucket_902 = per_flag('902')
        bucket_903 = per_flag('903')

        # Exactly ONE trip WARNING in the whole batch.
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 1, (
            f'[branch={branch}] Expected exactly 1 trip WARNING; got: {breaker_warnings}'
        )

        # Flags 901 and 902 — ACTIVE-branch miss: confirmation was attempted and missed.
        # Must have 'could not be confirmed findable'; must NOT have tripped-skip wording.
        for tid, bucket in [('901', bucket_901), ('902', bucket_902)]:
            assert any('could not be confirmed findable' in m for m in bucket), (
                f'[branch={branch}] task {tid}: expected ACTIVE-miss WARNING '
                f"'could not be confirmed findable'; got: {bucket}"
            )
            assert not any(
                'confirmation skipped (circuit-breaker open)' in m for m in bucket
            ), (
                f'[branch={branch}] task {tid}: TRIPPED-skip wording must NOT appear '
                f'in ACTIVE-branch bucket; got: {bucket}'
            )

        # Flag 903 — TRIPPED-branch skip: no confirmation attempted; breaker open.
        # Must have tripped-skip wording; must NOT have active-miss wording.
        assert any(
            'confirmation skipped (circuit-breaker open)' in m and
            'memory_ids gate failed' in m
            for m in bucket_903
        ), (
            f'[branch={branch}] task 903: expected TRIPPED-skip WARNING with '
            f"'confirmation skipped (circuit-breaker open)' + "
            f"'memory_ids gate failed'; got: {bucket_903}"
        )
        assert not any('could not be confirmed findable' in m for m in bucket_903), (
            f'[branch={branch}] task 903: ACTIVE-miss wording must NOT appear in '
            f'TRIPPED-skip bucket; got: {bucket_903}'
        )

        # Branch-specific consequence suffix preserved in per-flag WARNINGs.
        if branch == 'hit':
            consequence = 'skipping prior deletion'
        else:
            consequence = 'will not be detected next cycle'

        for tid, bucket in [('901', bucket_901), ('902', bucket_902), ('903', bucket_903)]:
            assert any(consequence in m for m in bucket), (
                f'[branch={branch}] task {tid}: expected consequence '
                f'{consequence!r} in WARNING; got: {bucket}'
            )

    # -----------------------------------------------------------------------
    # task-1415 step-3 — default threshold behavior pin
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_default_threshold_trips_at_three_consecutive_misses(self, caplog):
        """Default _CONFIRMATION_MISS_THRESHOLD (no monkeypatch) trips at 3 consecutive misses.

        RED (step-3 task-1415): the current default is 5.  At threshold=5 the
        breaker does NOT trip after 3 misses, so flag-4 confirmation searches are
        still reached → search.call_count would be 13 instead of 11, and no breaker
        WARNING fires.  The test anchors the production default at 3.

        Setup: 4 MISS-path flags.  Each flag's confirmation always misses (initial
        miss + retry = 2 searches).  After flag-3's retry the counter reaches 3 →
        TRIP.  Flag-4's confirmation is entirely skipped.

        search side_effect (13 elements; [11] and [12] only reached without breaker):
          [0]   suppression → []
          [1]   flag-1 pre-write → []  (MISS)
          [2]   flag-1 confirmation initial miss
          [3]   flag-1 confirmation retry   → counter = 1
          [4]   flag-2 pre-write → []  (MISS)
          [5]   flag-2 confirmation initial miss
          [6]   flag-2 confirmation retry   → counter = 2
          [7]   flag-3 pre-write → []  (MISS)
          [8]   flag-3 confirmation initial miss
          [9]   flag-3 confirmation retry   → counter = 3 → TRIP
          [10]  flag-4 pre-write → []  (MISS; no confirmation)
          [11]  (flag-4 confirmation miss — only reached without breaker)
          [12]  (flag-4 confirmation retry — only reached without breaker)

        Asserts:
          (a) Exactly ONE circuit-breaker WARNING whose text contains
              'tripped after 3 consecutive' (anchoring both the event type
              and the exact count; a bare '3' would also match 13 or 30).
          (b) search.call_count == 11 (1 suppression + 4 pre-write +
              3×2 confirmations for flags 1-3 + 0 confirmations for flag-4).
          (c) All 4 flags returned (MISS-path, no 'persisted_from_run').
        """
        import logging

        from fused_memory.reconciliation.flag_dedup import dedup_flags

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=[
            [],   # [0]  suppression
            [],   # [1]  flag-1 pre-write MISS
            [],   # [2]  flag-1 confirmation initial miss
            [],   # [3]  flag-1 confirmation retry        → counter = 1
            [],   # [4]  flag-2 pre-write MISS
            [],   # [5]  flag-2 confirmation initial miss
            [],   # [6]  flag-2 confirmation retry        → counter = 2
            [],   # [7]  flag-3 pre-write MISS
            [],   # [8]  flag-3 confirmation initial miss
            [],   # [9]  flag-3 confirmation retry        → counter = 3 → TRIP
            [],   # [10] flag-4 pre-write MISS (no confirmation — breaker tripped)
            [],   # [11] flag-4 confirmation miss  (only reached without breaker)
            [],   # [12] flag-4 confirmation retry (only reached without breaker)
        ])
        memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['x']))

        flags = [
            {'task_id': 601, 'flag_type': 'missing_deliverable', 'description': 'f1'},
            {'task_id': 602, 'flag_type': 'missing_deliverable', 'description': 'f2'},
            {'task_id': 603, 'flag_type': 'missing_deliverable', 'description': 'f3'},
            {'task_id': 604, 'flag_type': 'missing_deliverable', 'description': 'f4'},
        ]

        with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await dedup_flags(
                memory_service=memory_service,
                project_id='p',
                run_id='r1',
                flags=flags,
            )

        all_warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]

        # (a) Exactly ONE breaker WARNING whose text contains 'tripped after 3 consecutive'
        #     (anchoring both the event type and the exact count rendered into the message).
        #     A bare '3' would match counts like 13 or 30; the longer substring is precise.
        breaker_warnings = [m for m in all_warnings if 'tripped after' in m]
        assert len(breaker_warnings) == 1, (
            f"Expected exactly 1 circuit-breaker WARNING but got "
            f"{len(breaker_warnings)}: {breaker_warnings}\nAll WARNINGs: {all_warnings}"
        )
        assert 'tripped after 3 consecutive' in breaker_warnings[0], (
            f"Breaker WARNING must contain 'tripped after 3 consecutive' "
            f"(the default threshold rendered into the message); "
            f"got: {breaker_warnings[0]!r}"
        )

        # (b) Exactly 11 search calls — no confirmation for flag-4.
        assert memory_service.search.call_count == 11, (
            f"Expected 11 search calls (1 suppression + 4 pre-write + 3×2 confirmations "
            f"for flags 1-3 + 0 for flag-4), got: {memory_service.search.call_count}; "
            f"if count=13, the default threshold is still 5 (not yet lowered to 3)"
        )

        # (c) All 4 flags returned; MISS path — no 'persisted_from_run' annotation.
        assert len(result) == 4, f"Expected 4 flags returned; got {len(result)}"
        for f in result:
            assert 'persisted_from_run' not in f, f"MISS path must not annotate: {f}"


# ---------------------------------------------------------------------------
# _marker_query builder (step-1 / step-2)
# ---------------------------------------------------------------------------


class TestMarkerQuery:
    """Unit tests for the _marker_query(tid, ftype) -> str builder."""

    def test_returns_canonical_format_for_simple_ids(self):
        """Returns the exact canonical string for straightforward id values."""
        from fused_memory.reconciliation.flag_dedup import _marker_query

        result = _marker_query('42', 'missing_deliverable')
        assert result == 'stage1 flag marker task 42 type missing_deliverable'


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
# _write_and_confirm_marker helper (step-4 / step-5)
# ---------------------------------------------------------------------------


import logging as _logging_mod  # noqa: E402 — needed for caplog logger name


@pytest.mark.asyncio
class TestWriteAndConfirmMarker:
    """Unit tests for the _write_and_confirm_marker module-level helper."""

    async def test_writes_canonical_payload_and_delegates_to_confirm_and_track(self, caplog):
        """Helper writes canonical payload and forwards result from confirm_and_track."""
        from fused_memory.reconciliation.flag_dedup import _write_and_confirm_marker

        memory_service = MagicMock()
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        confirm_calls = []

        async def stub_confirm(
            response_memory_ids,
            active_miss_warning_template,
            tripped_skip_warning_template,
            *,
            tid,
            ftype,
        ):
            confirm_calls.append((
                response_memory_ids,
                active_miss_warning_template,
                tripped_skip_warning_template,
                tid,
                ftype,
            ))
            return True

        log = _logging_mod.getLogger('fused_memory.reconciliation.flag_dedup')
        result = await _write_and_confirm_marker(
            memory_service,
            project_id='p',
            run_id='r1',
            tid='42',
            ftype='missing_deliverable',
            log=log,
            confirm_and_track=stub_confirm,
            active_miss_warning_template='active-miss-template-%s-%s',
            tripped_skip_warning_template='tripped-skip-template-%s-%s',
        )

        # Return value propagated verbatim from confirm_and_track (bool contract)
        assert result is True
        assert isinstance(result, bool), (
            f"_write_and_confirm_marker must return bool, not {type(result)!r}"
        )

        # add_memory called exactly once with canonical payload
        memory_service.add_memory.assert_called_once()
        kwargs = memory_service.add_memory.call_args.kwargs
        _assert_valid_stage1_marker(kwargs, task_id='42', flag_type='missing_deliverable', run_id='r1')
        assert kwargs['_source'] == 'stage1_flag_dedup'
        assert kwargs['causation_id'] == 'r1'
        assert kwargs['content'] == 'Stage 1 flag marker: task=42 type=missing_deliverable from run=r1'

        # confirm_and_track called with correct args (delegation contract)
        assert len(confirm_calls) == 1
        ids, active_tmpl, tripped_tmpl, c_tid, c_ftype = confirm_calls[0]
        assert ids == _STUB_ADD_MEMORY_RESPONSE.memory_ids
        assert active_tmpl == 'active-miss-template-%s-%s'
        assert tripped_tmpl == 'tripped-skip-template-%s-%s'
        assert c_tid == '42'
        assert c_ftype == 'missing_deliverable'

    async def test_returns_false_and_logs_unified_warning_on_add_memory_exception(self, caplog):
        """On add_memory exception: returns False, logs WARNING, does NOT call confirm_and_track."""
        from fused_memory.reconciliation.flag_dedup import _write_and_confirm_marker

        memory_service = MagicMock()
        memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write blew up'))

        confirm_and_track = AsyncMock()
        log = _logging_mod.getLogger('fused_memory.reconciliation.flag_dedup')

        with caplog.at_level(_logging_mod.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
            result = await _write_and_confirm_marker(
                memory_service,
                project_id='p',
                run_id='r2',
                tid='42',
                ftype='missing_deliverable',
                log=log,
                confirm_and_track=confirm_and_track,
                active_miss_warning_template='irrelevant-active',
                tripped_skip_warning_template='irrelevant-tripped',
            )

        assert result is False
        assert isinstance(result, bool), (
            f"_write_and_confirm_marker must return bool on exception, not {type(result)!r}"
        )
        confirm_and_track.assert_not_called()

        warning_msgs = [r.message for r in caplog.records if r.levelno >= _logging_mod.WARNING]
        assert len(warning_msgs) == 1
        assert '42' in warning_msgs[0]
        assert 'missing_deliverable' in warning_msgs[0]
        assert 'failed to write marker' in warning_msgs[0]

    async def test_deduped_against_included_in_metadata_when_provided(self, caplog):
        """deduped_against=[...] is included in the add_memory metadata payload,
        additive to the existing source/kind/task_id/flag_type/run_id/last_seen_run_id
        keys (task-2047 Gap 1)."""
        from fused_memory.reconciliation.flag_dedup import _write_and_confirm_marker

        memory_service = MagicMock()
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
        confirm_and_track = AsyncMock(return_value=True)
        log = _logging_mod.getLogger('fused_memory.reconciliation.flag_dedup')

        result = await _write_and_confirm_marker(
            memory_service,
            project_id='p',
            run_id='r1',
            tid='fp:9216e85ac497b68d93043b64684eb049',
            ftype='duplicate_procedural_knowledge',
            log=log,
            confirm_and_track=confirm_and_track,
            active_miss_warning_template='active-%s-%s',
            tripped_skip_warning_template='tripped-%s-%s',
            deduped_against=['uuid-a', 'uuid-b'],
        )

        assert result is True
        memory_service.add_memory.assert_called_once()
        kwargs = memory_service.add_memory.call_args.kwargs
        _assert_valid_stage1_marker(
            kwargs,
            task_id='fp:9216e85ac497b68d93043b64684eb049',
            flag_type='duplicate_procedural_knowledge',
            run_id='r1',
        )
        assert kwargs['metadata']['deduped_against'] == ['uuid-a', 'uuid-b'], (
            f"metadata must carry deduped_against; got {kwargs['metadata']!r}"
        )

    async def test_deduped_against_omitted_from_metadata_when_none_or_empty(self, caplog):
        """deduped_against=None (or []) must NOT add a 'deduped_against' metadata key —
        locks the additive/no-regression contract: the payload is otherwise
        byte-identical to the pre-task-2047 contract (task-2047 Gap 1)."""
        from fused_memory.reconciliation.flag_dedup import _write_and_confirm_marker

        log = _logging_mod.getLogger('fused_memory.reconciliation.flag_dedup')

        for deduped_against in (None, []):
            memory_service = MagicMock()
            memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
            confirm_and_track = AsyncMock(return_value=True)

            await _write_and_confirm_marker(
                memory_service,
                project_id='p',
                run_id='r1',
                tid='42',
                ftype='missing_deliverable',
                log=log,
                confirm_and_track=confirm_and_track,
                active_miss_warning_template='active-%s-%s',
                tripped_skip_warning_template='tripped-%s-%s',
                deduped_against=deduped_against,
            )

            kwargs = memory_service.add_memory.call_args.kwargs
            assert 'deduped_against' not in kwargs['metadata'], (
                f'deduped_against={deduped_against!r} must not add a metadata key; '
                f"got metadata={kwargs['metadata']!r}"
            )
            assert kwargs['metadata'] == {
                'source': 'stage1_flag_marker',
                'kind': 'stage1_flag_marker',
                'task_id': '42',
                'flag_type': 'missing_deliverable',
                'run_id': 'r1',
                'last_seen_run_id': 'r1',
            }, 'payload must be byte-identical to the pre-change contract when deduped_against is empty/None'


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
    _write_and_confirm_marker.

    fp: MISS and HIT path coverage lives in TestDedupFlagsContentFingerprintPath
    (focused single-cycle variants) and test_cross_cycle_fp_roundtrip (end-to-end
    two-cycle variant) below.  This class focuses on the remaining contract:

    (a) Regression: numeric task_id '42' still writes its marker (guard does not
        over-reject valid integer keys).
    (b) Regression: comma-joined cited_tasks signature '12,15' still writes its
        marker (guard accepts the comma-joined shape).
    (c) Defense-in-depth: genuinely-invalid tid 'abc' → _write_and_confirm_marker
        returns False, no add_memory call, circuit-breaker counter untouched.
    (d) Cross-cycle round-trip: fp: marker written on cycle-1 MISS, found and used
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

    async def test_invalid_tid_write_guard_returns_false_no_io(self):
        """(e) _write_and_confirm_marker with genuinely-invalid tid 'abc' returns False.

        The defense-in-depth guard in _write_and_confirm_marker must reject any tid
        that is not a valid numeric or canonical fp: key.  This test directly exercises
        the guard with tid='abc' (not numeric, not fp:) to verify:
        (i)  Returns False — write was blocked.
        (ii) add_memory is NOT called — no Mem0 I/O.
        (iii) confirm_and_track is NOT called — circuit-breaker counter is untouched.
        """
        import logging as _logging

        from fused_memory.reconciliation.flag_dedup import _write_and_confirm_marker

        memory_service = AsyncMock()
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        confirm_and_track = AsyncMock()

        result = await _write_and_confirm_marker(
            memory_service,
            project_id='proj',
            run_id='run-x',
            tid='abc',
            ftype='missing_deliverable',
            log=_logging.getLogger('fused_memory.reconciliation.flag_dedup'),
            confirm_and_track=confirm_and_track,
            active_miss_warning_template='active %s %s',
            tripped_skip_warning_template='tripped %s %s',
        )

        # (i) Returns False — guard blocked the write
        assert result is False, f'Expected False for invalid tid "abc"; got {result!r}'

        # (ii) add_memory not called
        memory_service.add_memory.assert_not_called()

        # (iii) confirm_and_track not called — circuit-breaker counter untouched
        confirm_and_track.assert_not_called()

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
async def test_dedup_flags_completion_flag_with_no_priors_is_noop_sweep():
    """A completion flag whose signature has no prior marker at all (the common
    case: nothing was ever persisted for it) is a clean no-op sweep — no
    add_memory, no delete_memory — and is still annotated
    completion_marker_self_deleted=True.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('77', 'duplicate_flag_marker_cleanup'): [[]]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{
        'task_id': 77,
        'flag_type': 'duplicate_flag_marker_cleanup',
        'description': 'cleaned up an orphaned duplicate flag marker',
        'flag_for_stage2': False,
    }]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    memory_service.add_memory.assert_not_called()
    memory_service.delete_memory.assert_not_called()

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

    ledger_memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        ledger_memory_service.add_memory.call_args.kwargs,
        task_id='99', flag_type='stale_metadata', run_id='r1',
    )
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

