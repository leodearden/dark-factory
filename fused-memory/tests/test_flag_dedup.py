"""Tests for fused_memory.reconciliation.flag_dedup module.

Tests cover compute_flag_signature, dedup_flags, and error-handling behavior.
"""
from __future__ import annotations

import uuid as _uuid_mod
from collections import deque
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from fused_memory.backends.task_backend_errors import TaskmasterError
from fused_memory.models.memory import AddMemoryResponse
from fused_memory.reconciliation.flag_dedup import _marker_query, build_suppression_payload

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
    """Flags with no computable signature pass through with exactly one I/O call (suppression filter); add_memory never called.

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
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
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
    # filter_suppressed issues exactly one project-scoped suppression search;
    # no per-flag searches because no flags have computable signatures.
    assert memory_service.search.call_count == 1
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


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_found_annotates_flag_no_write():
    """When a prior stage1_flag_marker exists the flag gets persisted_from_run/last_seen_run_id,
    a replacement marker is written, and the prior is deleted (atomic-replacement contract).

    Updated for task-1146: the old \"no write on HIT\" contract is replaced by
    atomic-replacement (write new marker → delete prior).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-42'

    # task-1400 step-16: confirmation must find a marker with the CURRENT run_id='r1'
    # (the prior has run_id='r0' and does not match the run_id-scoped confirmation kind filter)
    new_marker_r1 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_marker_r1.id = 'new-42-r1'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('42', 'missing_deliverable'): [[prior_marker], [new_marker_r1]]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) Flag is annotated with persisted_from_run and last_seen_run_id
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # (b) search was called 3 times: suppression filter + per-flag prior-marker + confirmation.
    #     call_args refers to the LAST call (confirmation search), which must mention
    #     task_id+flag_type and use project_id='p'.
    assert memory_service.search.call_count == 3  # 1 suppression + 1 per-flag + 1 confirmation
    # project_id must be passed as a kwarg (production code uses kwargs throughout)
    assert memory_service.search.call_args.kwargs['project_id'] == 'p'
    # query must strictly mention both the task_id and the flag_type (no permissive 'or')
    query = memory_service.search.call_args.kwargs.get('query', '')
    assert '42' in query and 'missing_deliverable' in query

    # (c) Replacement marker written and prior deleted (atomic-replacement)
    memory_service.add_memory.assert_called_once()
    memory_service.delete_memory.assert_called_once()


@pytest.mark.asyncio
async def test_dedup_flags_metadata_predicate_filters_non_matching_results():
    """When Mem0 search returns rows matching task_id but with wrong source or wrong
    flag_type, the metadata predicate filters them all out, so the flag is treated as
    fresh: not annotated, and a new stage1_flag_marker is written.

    Regression coverage for flag_dedup.py:77-86 — without this test, dropping the
    source/flag_type guards from the kind dict passed to find_prior_memory would
    silently start treating cross-source rows as prior markers.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    # Three rows whose task_id matches but whose source and/or flag_type do not.
    # They exercise both clauses of the kind conjunction independently:
    wrong_source = _make_memory_result({
        'source': 'targeted_reconciliation',  # wrong source
        'task_id': '42',
        'flag_type': 'missing_deliverable',
    })
    wrong_flag_type = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'stale_metadata',  # wrong flag_type
    })
    both_wrong = _make_memory_result({
        'source': 'other',  # wrong source
        'task_id': '42',
        'flag_type': 'unrelated',  # wrong flag_type
    })

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[wrong_source, wrong_flag_type, both_wrong]],
        marker={('42', 'missing_deliverable'): [
            [wrong_source, wrong_flag_type, both_wrong],  # pre-write search (all filtered → MISS)
            [],  # confirmation search (miss)
            [],  # confirmation retry (miss)
        ]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # Predicate rejected all rows — flag NOT annotated as prior-seen
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0], (
        'Predicate should have filtered all non-matching rows; flag must not be annotated'
    )

    # No-prior-marker write path exercised: a new marker is written
    memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='42', flag_type='missing_deliverable', run_id='r1',
    )

    # Four search calls: 1 suppression + 1 per-flag + 2 confirmation (miss+retry).
    assert memory_service.search.call_count == 4


# ---------------------------------------------------------------------------
# dedup_flags — no prior marker (fresh flag) path (step-7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_no_prior_marker_writes_new_marker():
    """When no prior stage1_flag_marker exists, the flag is not annotated and a new marker
    is written to Mem0.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # empty — no prior marker
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [{'task_id': '99', 'flag_type': 'stale_metadata', 'description': 'bar'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) Flag has NO persisted_from_run field — it's a fresh finding
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]

    # (b) add_memory called exactly once with the expected marker metadata
    memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='99', flag_type='stale_metadata', run_id='r1',
    )


@pytest.mark.asyncio
async def test_marker_metadata_includes_source_and_kind():
    """Stage1 flag marker add_memory call must carry BOTH source and kind keys.

    Regression test for task-1659: earlier writes set source='stage1_flag_marker'
    but omitted kind='stage1_flag_marker', breaking dual-filter queries that key
    on both fields.  This test drives a MISS path (no prior marker) through
    dedup_flags and asserts both keys are present in the written metadata.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # no prior marker
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [{'task_id': '7', 'flag_type': 'missing_deliverable', 'description': 'x'}]
    await dedup_flags(
        memory_service=memory_service,
        project_id='proj',
        run_id='r99',
        flags=flags,
    )

    memory_service.add_memory.assert_called_once()
    written_meta = memory_service.add_memory.call_args.kwargs.get('metadata', {})
    assert written_meta.get('source') == 'stage1_flag_marker', (
        f"metadata.source must be 'stage1_flag_marker', got: {written_meta!r}"
    )
    assert written_meta.get('kind') == 'stage1_flag_marker', (
        f"metadata.kind must be 'stage1_flag_marker', got: {written_meta!r}"
    )


# ---------------------------------------------------------------------------
# dedup_flags — exception handling (step-9)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_search_exception_does_not_raise_and_warns(caplog):
    """When memory_service.search raises, dedup_flags does not raise, returns flags unchanged,
    and logs a WARNING.

    Also pins the post-refactor marker-growth behavior: when search fails,
    find_prior_memory swallows the exception and returns None, so dedup_flags
    falls into the else-branch and writes one new marker per flag (was: zero in
    the prior wrap-both pattern where both search and add_memory shared a single
    try/except).  See the marker-growth caveat comment in the else-branch of
    dedup_flags for details on monotonic marker growth during a sustained outage.

    Two flags are passed so the 'one write per flag' contract is verified, not
    merely 'exactly one write ever'.  The assertions (d) and (e) below lock down
    this contract so a future refactor cannot silently flip the count in either
    direction (zero if the wrap-both pattern is restored, or >N if an extra
    refresh write is added per flag).
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [
        {'task_id': '55', 'flag_type': 'stale_metadata', 'description': 'test'},
        {'task_id': '66', 'flag_type': 'missing_deliverable', 'description': 'test2'},
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) Does NOT raise
    # (b) Returns both flags unchanged (no persisted_from_run)
    assert len(result) == 2
    assert 'persisted_from_run' not in result[0]
    assert 'persisted_from_run' not in result[1]
    # (c) WARNING log mentions the failure for both task_ids
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any('55' in m for m in warning_messages)
    assert any('66' in m for m in warning_messages)
    # (d) Exactly one marker write per flag when search fails — verifies the 'per flag'
    #     contract (two flags → two writes).  Catches: zero writes (wrap-both try/except
    #     restored) or multiple writes per flag (extra refresh write added).
    assert memory_service.add_memory.call_count == 2
    # (e) Each marker is correctly shaped on the failure path.
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args_list[0].kwargs,
        task_id='55', flag_type='stale_metadata', run_id='r1',
    )
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args_list[1].kwargs,
        task_id='66', flag_type='missing_deliverable', run_id='r1',
    )


# ---------------------------------------------------------------------------
# dedup_flags — malformed prior-marker run_id uses sentinel (step-1)
# ---------------------------------------------------------------------------

_VALID_FILTER_META = {
    'source': 'stage1_flag_marker',
    'task_id': '42',
    'flag_type': 'missing_deliverable',
}


@pytest.mark.parametrize(
    'prior_metadata',
    [
        # (a) 'run_id' key absent — .get('run_id', run_id) silently returns run_id, not 'unknown'
        pytest.param(
            {**_VALID_FILTER_META},
            id='run_id-key-absent',
        ),
        # (c) 'run_id' key present but value is None — .get returns None (not 'unknown')
        pytest.param(
            {**_VALID_FILTER_META, 'run_id': None},
            id='run_id-is-None',
        ),
        # (d) 'run_id' key present but value is '' — .get returns '' (not 'unknown')
        pytest.param(
            {**_VALID_FILTER_META, 'run_id': ''},
            id='run_id-is-empty-string',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_with_malformed_run_id_uses_sentinel(
    prior_metadata, caplog
):
    """When a prior stage1_flag_marker exists but has a missing/falsy run_id, dedup_flags
    must annotate the flag with persisted_from_run='unknown' — not the current run_id.

    Three malformed shapes parametrised:
    (a) 'run_id' key absent        → .get(key, run_id) silently returns 'r1' ≠ 'unknown'
    (c) run_id=None                → .get returns None ≠ 'unknown'
    (d) run_id=''                  → .get returns '' ≠ 'unknown'
    All three must produce persisted_from_run='unknown' with the sentinel fix applied.

    Also asserts that the sentinel-collapse path emits a DEBUG log for each malformed
    shape — this folds in what was previously a standalone test, giving observability
    coverage across all three shapes in a single parametrized suite.

    Note: the case where prior.metadata is None is intentionally omitted.  When
    metadata is None, the candidate filter ((r.metadata or {}).get('source') etc.)
    returns falsy values for all three checks, so the candidate is never selected
    as ``prior`` — the code under test (run_id extraction) is never reached.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result(prior_metadata)
    prior_marker.id = 'prior-malformed'

    # task-1400 step-16: confirmation kind filter now scopes by run_id='r1'.
    # The prior has a malformed run_id (None/''/ absent) so it does NOT match.
    # Supply a separate well-formed replacement marker with run_id='r1'.
    new_marker_r1 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_marker_r1.id = 'new-malformed-r1'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('42', 'missing_deliverable'): [[prior_marker], [new_marker_r1]]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.DEBUG, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'unknown', (
        f"persisted_from_run must fall back to sentinel 'unknown' for any falsy run_id "
        f"in prior marker metadata, but got {result[0].get('persisted_from_run')!r}."
    )
    assert result[0]['last_seen_run_id'] == 'r1'
    # Atomic-replacement: replacement marker is written and prior is deleted even
    # when run_id was malformed (annotation sentinel does not suppress the write).
    memory_service.add_memory.assert_called_once()
    memory_service.delete_memory.assert_called_once()

    # Sentinel-collapse path emits a DEBUG log — covers all three malformed shapes.
    # Loose enough to tolerate minor wording changes; strict enough to lock in the
    # observability intent (dashboards can grep for 'unknown'/'malformed').
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


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_None_metadata_writes_new_marker():
    """Locks in the candidate-filter rejection behavior — the `meta = r.metadata or {}`
    guard in find_prior_memory short-circuits when metadata is None, so a refactor that
    drops the `or {}` guard (e.g., direct attribute access on r.metadata) would silently
    regress this path. Test passes against current impl. Companion to
    test_dedup_flags_prior_marker_with_malformed_run_id_uses_sentinel, which intentionally
    omits this case (see its docstring).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    # Prior result with metadata=None — survives the search but is rejected by the
    # candidate filter because (r.metadata or {}).get('task_id', '') is empty and
    # does not match '42'; the task_id check runs first so the source/flag_type
    # checks are never evaluated.
    prior_result = _make_memory_result(None)

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_result])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=[{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}],
    )

    # Candidate was filtered out — flag is NOT annotated with persisted_from_run
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0], (
        f"Expected no persisted_from_run annotation but got {result[0].get('persisted_from_run')!r}"
    )

    # The dedup logic took the else-branch (novel flag) and wrote a new marker
    memory_service.add_memory.assert_called_once()

    # New marker has a well-formed stage1_flag_marker shape
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='42',
        flag_type='missing_deliverable',
        run_id='r1',
    )


# ---------------------------------------------------------------------------
# Step 3 (task-1146) — best-effort replace: HIT writes new marker then deletes prior
# NOTE: test names below use the historical 'atomic_replace' prefix; the contract
# was renamed 'best-effort replacement' in task-1165 (concurrent duplicates self-heal).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_prior_marker_atomic_replace_writes_new_and_deletes_prior():
    """On HIT, dedup_flags writes a new marker and deletes the prior, in that order.

    Pins the atomic-replacement contract introduced by task-1146:
    (a) flag is annotated with persisted_from_run='r0' and last_seen_run_id='r1'
    (b) add_memory called exactly once with a well-formed stage1_flag_marker for run_id='r1'
    (c) delete_memory called exactly once with memory_id='prior-123', store='mem0'
    (d) ordering invariant: add_memory call precedes delete_memory call in mock_calls
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-123'

    # task-1400 step-16: confirmation needs a marker with current run_id='r1'.
    # Prior has run_id='r0' and does not match the run_id-scoped confirmation filter.
    new_marker_r1 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_marker_r1.id = 'new-123-r1'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('42', 'missing_deliverable'): [[prior_marker], [new_marker_r1]]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) annotation
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # (b) add_memory called once with valid marker for run_id='r1'
    memory_service.add_memory.assert_called_once()
    _assert_valid_stage1_marker(
        memory_service.add_memory.call_args.kwargs,
        task_id='42', flag_type='missing_deliverable', run_id='r1',
    )

    # (c) delete_memory called once for the prior
    memory_service.delete_memory.assert_called_once()
    del_kwargs = memory_service.delete_memory.call_args.kwargs
    assert del_kwargs.get('memory_id') == 'prior-123'
    assert del_kwargs.get('store') == 'mem0'
    assert del_kwargs.get('project_id') == 'p'

    # (d) ordering: add_memory before delete_memory.
    # Use method_calls (records only actual method invocations, no attribute
    # access) with exact name matching via call[0] so the check is immune to
    # future child-mock interactions whose repr happens to contain 'add_memory'.
    method_names = [c[0] for c in memory_service.method_calls]
    add_idx = method_names.index('add_memory')
    del_idx = method_names.index('delete_memory')
    assert add_idx < del_idx, (
        f'add_memory (idx {add_idx}) must precede delete_memory (idx {del_idx})'
    )


# ---------------------------------------------------------------------------
# Step 5 (task-1146) — atomic-replace collapses multiple predecessors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_atomic_replace_handles_multiple_predecessors():
    """On HIT with multiple prior markers (past leakage), all priors are deleted.

    Simulates N=3 prior markers for the same (task_id, flag_type) that exist
    due to prior search-failure or top-N rank-eviction leakage.  dedup_flags
    must write exactly ONE replacement marker and delete all THREE priors.

    (a) add_memory called exactly once
    (b) delete_memory called exactly three times covering ids {p-1, p-2, p-3}
    (c) flag annotation uses the lowest-id-lex prior's run_id
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    def _prior(id_: str, run_id: str) -> MagicMock:
        r = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'missing_deliverable',
            'run_id': run_id,
            'last_seen_run_id': run_id,
        })
        r.id = id_
        return r

    prior1 = _prior('p-1', 'r0')   # lex-lowest id — annotation comes from this one
    prior2 = _prior('p-2', 'r-prev')
    prior3 = _prior('p-3', 'r-earlier')

    # task-1400 step-16: confirmation kind filter now scopes by run_id='r1'.
    # All priors have older run_ids so they won't match confirmation.
    # Supply a separate new marker with run_id='r1' for the confirmation element.
    new_marker_r1 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_marker_r1.id = 'new-multi-r1'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('42', 'missing_deliverable'): [[prior2, prior3, prior1], [new_marker_r1]]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # (a) exactly one write
    memory_service.add_memory.assert_called_once()

    # (b) exactly three deletes covering all prior ids
    assert memory_service.delete_memory.call_count == 3, (
        f'Expected 3 delete_memory calls but got {memory_service.delete_memory.call_count}'
    )
    deleted_ids = {
        call.kwargs.get('memory_id')
        for call in memory_service.delete_memory.call_args_list
    }
    assert deleted_ids == {'p-1', 'p-2', 'p-3'}, (
        f'Expected all three prior ids deleted but got {deleted_ids}'
    )

    # (c) annotation from lowest-id-lex prior (p-1 → run_id 'r0')
    assert result[0]['persisted_from_run'] == 'r0'


# ---------------------------------------------------------------------------
# Step 7 (task-1146) — write-failure skips delete (predecessor preserved)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_atomic_replace_skips_delete_if_write_fails(caplog):
    """When add_memory raises on the HIT path, delete_memory is never called.

    Pins the write-first ordering guarantee: if the replacement write fails,
    all priors must remain intact so the next cycle still has dedup state.

    (a) dedup_flags does NOT raise
    (b) delete_memory was NEVER called
    (c) flag IS annotated (annotation extracted BEFORE write attempt)
    (d) WARNING log mentions task_id, flag_type, and the failure
    """
    import logging as _logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-write-fail'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[prior_marker])
    memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write failed'))
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(_logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) does not raise
    # (b) delete_memory never called — prior preserved
    memory_service.delete_memory.assert_not_called()

    # (c) flag IS annotated (annotation extracted before write attempt)
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'

    # (d) WARNING log mentions task_id and flag_type
    warning_messages = [r.message for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any('42' in m for m in warning_messages), (
        f'Expected WARNING mentioning task 42 but got: {warning_messages}'
    )
    assert any('missing_deliverable' in m for m in warning_messages), (
        f'Expected WARNING mentioning flag_type but got: {warning_messages}'
    )


# ---------------------------------------------------------------------------
# Step 9 (task-1146) — per-prior delete failure logs warning and continues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_atomic_replace_per_prior_delete_failure_logs_warning_and_continues(caplog):
    """One failing delete does not abort the batch; both priors get a delete attempt.

    Configures two priors ('p-1', 'p-2'); delete_memory raises for 'p-1' but
    succeeds for 'p-2'.  Pins the contract that a per-prior delete error:
    (a) does NOT cause dedup_flags to raise
    (b) add_memory is called exactly once (write succeeded)
    (c) delete_memory is called exactly twice — both priors attempted
    (d) WARNING log mentions 'p-1' and the failure
    (e) flag is annotated normally
    """
    import logging as _logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    def _prior(id_: str, run_id: str) -> MagicMock:
        r = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'missing_deliverable',
            'run_id': run_id,
            'last_seen_run_id': run_id,
        })
        r.id = id_
        return r

    prior1 = _prior('p-1', 'r0')
    prior2 = _prior('p-2', 'r-prev')

    # task-1400 step-16: confirmation kind filter now scopes by run_id='r1'.
    # Both priors have older run_ids so they won't match confirmation.
    # Supply a well-formed new marker with run_id='r1'.
    new_marker_r1 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_marker_r1.id = 'new-del-fail-r1'

    def _delete_side_effect(**kwargs):
        if kwargs.get('memory_id') == 'p-1':
            raise RuntimeError('delete p-1 failed')
        # p-2 succeeds — return None (AsyncMock awaitable returns None by default)

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('42', 'missing_deliverable'): [[prior1, prior2], [new_marker_r1]]},
    ))
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(side_effect=_delete_side_effect)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(_logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) does not raise
    # (b) add_memory called once
    memory_service.add_memory.assert_called_once()

    # (c) delete_memory called twice — both priors attempted
    assert memory_service.delete_memory.call_count == 2, (
        f'Expected 2 delete calls but got {memory_service.delete_memory.call_count}'
    )

    # (d) WARNING log mentions 'p-1'
    warning_messages = [r.message for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any('p-1' in m for m in warning_messages), (
        f'Expected WARNING mentioning p-1 but got: {warning_messages}'
    )

    # (e) flag annotated normally
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


@pytest.mark.asyncio
async def test_dedup_flags_add_memory_exception_does_not_raise_and_warns(caplog):
    """When memory_service.add_memory raises, dedup_flags does not raise, returns flag unchanged,
    and logs a WARNING.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # no prior marker
    memory_service.add_memory = AsyncMock(side_effect=RuntimeError('write failed'))

    flags = [{'task_id': '66', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) Does NOT raise
    # (b) Returns flag unchanged
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]
    # (c) WARNING log mentions failure and task_id
    assert any(
        '66' in record.message and record.levelno >= logging.WARNING
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# Step 11 (task-1146) — two-consecutive-runs no-accumulation regression test
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# task-1165 step-5 — HIT path: deterministic prior-marker selection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_hit_prior_selection_is_deterministic_across_search_orders():
    """HIT path: prior selection must be deterministic regardless of search return order.

    Builds three priors with ids 'aaa', 'mmm', 'zzz' and distinct run_ids.
    Runs dedup_flags twice with different search return orders:
    - Run 1: search returns [zzz, aaa, mmm]
    - Run 2: search returns [mmm, zzz, aaa]

    Both runs must:
    - annotate persisted_from_run='r-aaa' (lowest-id-lex prior wins)
    - have the FIRST delete_memory call target memory_id='aaa'

    Today this fails because priors[0] reflects search order.
    """

    def _prior(id_: str, run_id: str) -> MagicMock:
        r = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '99',
            'flag_type': 'stale_metadata',
            'run_id': run_id,
            'last_seen_run_id': run_id,
        })
        r.id = id_
        return r

    prior_aaa = _prior('aaa', 'r-aaa')
    prior_mmm = _prior('mmm', 'r-mmm')
    prior_zzz = _prior('zzz', 'r-zzz')

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    flags = [{'task_id': '99', 'flag_type': 'stale_metadata', 'description': 'test'}]

    # task-1400 step-16: confirmation kind filter now scopes by run_id.
    # All priors have their own run_ids (r-aaa, r-mmm, r-zzz) — none match the
    # current run_id ('r1' or 'r2').  Supply well-formed new markers for each run.
    new_marker_r1 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '99',
        'flag_type': 'stale_metadata',
        'run_id': 'r1',
    })
    new_marker_r1.id = 'new-det-r1'

    new_marker_r2 = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '99',
        'flag_type': 'stale_metadata',
        'run_id': 'r2',
    })
    new_marker_r2.id = 'new-det-r2'

    # Run 1: search returns [zzz, aaa, mmm] — aaa is NOT first
    memory_service_1 = AsyncMock()
    memory_service_1.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('99', 'stale_metadata'): [[prior_zzz, prior_aaa, prior_mmm], [new_marker_r1]]},
    ))
    memory_service_1.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service_1.delete_memory = AsyncMock(return_value=None)

    result_1 = await dedup_flags(
        memory_service=memory_service_1,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # Run 2: search returns [mmm, zzz, aaa] — different order
    memory_service_2 = AsyncMock()
    memory_service_2.search = AsyncMock(side_effect=_make_search_stub(
        suppression=[[]],
        marker={('99', 'stale_metadata'): [[prior_mmm, prior_zzz, prior_aaa], [new_marker_r2]]},
    ))
    memory_service_2.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service_2.delete_memory = AsyncMock(return_value=None)

    result_2 = await dedup_flags(
        memory_service=memory_service_2,
        project_id='p',
        run_id='r2',
        flags=flags,
    )

    # Both runs should annotate with 'r-aaa' (lowest lex id = 'aaa')
    assert result_1[0]['persisted_from_run'] == 'r-aaa', (
        f"Run 1: expected persisted_from_run='r-aaa' but got {result_1[0].get('persisted_from_run')!r}"
    )
    assert result_2[0]['persisted_from_run'] == 'r-aaa', (
        f"Run 2: expected persisted_from_run='r-aaa' but got {result_2[0].get('persisted_from_run')!r}"
    )

    # First delete in each run must target 'aaa'
    first_delete_1 = memory_service_1.delete_memory.call_args_list[0].kwargs.get('memory_id')
    assert first_delete_1 == 'aaa', (
        f"Run 1: expected first delete to target 'aaa' but got {first_delete_1!r}"
    )
    first_delete_2 = memory_service_2.delete_memory.call_args_list[0].kwargs.get('memory_id')
    assert first_delete_2 == 'aaa', (
        f"Run 2: expected first delete to target 'aaa' but got {first_delete_2!r}"
    )


# ---------------------------------------------------------------------------
# task-1165 step-3 — MISS path: respects add_memory response memory_ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'add_memory_response,expect_noop_warning',
    [
        pytest.param(
            'empty',  # AddMemoryResponse(memory_ids=[])
            True,
            id='empty-memory_ids-warns',
        ),
        pytest.param(
            'non_empty',  # AddMemoryResponse(memory_ids=['new-marker-id'])
            False,
            id='non-empty-memory_ids-no-warn',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_miss_respects_add_memory_response_memory_ids(
    add_memory_response, expect_noop_warning, caplog
):
    """MISS path: dedup_flags must inspect add_memory's return value.

    When add_memory returns an empty memory_ids list on MISS:
    - a WARNING must be emitted containing task_id and flag_type
      (the flag won't be detectable next cycle)

    When add_memory returns a non-empty memory_ids list on MISS:
    - no no-op WARNING should be emitted

    In both cases:
    - flag returns unannotated (no persisted_from_run)
    - add_memory is called exactly once
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    if add_memory_response == 'empty':
        response = AddMemoryResponse(memory_ids=[])
    else:
        response = AddMemoryResponse(memory_ids=['new-marker-id'])

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=[])  # MISS
    memory_service.add_memory = AsyncMock(return_value=response)

    flags = [{'task_id': '77', 'flag_type': 'stale_metadata', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # Flag is NOT annotated on MISS path regardless of write outcome
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]

    # add_memory called exactly once
    memory_service.add_memory.assert_called_once()

    # Check WARNING for no-op case
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    if expect_noop_warning:
        assert any('77' in m for m in warning_messages), (
            f'Expected WARNING mentioning task_id=77 but got: {warning_messages}'
        )
        assert any('stale_metadata' in m for m in warning_messages), (
            f'Expected WARNING mentioning flag_type but got: {warning_messages}'
        )
    else:
        noop_warnings = [m for m in warning_messages if 'no memory_ids' in m or 'returned no memory_ids' in m]
        assert not noop_warnings, (
            f'Unexpected no-op WARNING on non-empty memory_ids MISS path: {noop_warnings}'
        )


# ---------------------------------------------------------------------------
# task-1165 step-1 / task-1400 amend — HIT path: delete gated on confirmation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'add_memory_response,expect_delete,expect_noop_warning',
    [
        pytest.param(
            'empty',  # AddMemoryResponse(memory_ids=[])
            False,
            True,
            id='empty-memory_ids-skips-delete-and-warns',
        ),
        pytest.param(
            'non_empty',  # AddMemoryResponse(memory_ids=['new-marker-id'])
            True,
            False,
            id='non-empty-memory_ids-proceeds-to-delete',
        ),
    ],
)
@pytest.mark.asyncio
async def test_dedup_flags_hit_delete_gated_on_confirmation_not_memory_ids(
    add_memory_response, expect_delete, expect_noop_warning, caplog
):
    """HIT path: prior deletion is gated on post-write confirmation, not memory_ids.

    When add_memory is a silent no-op (confirmation misses, empty memory_ids):
    - delete_memory must NOT be called (priors preserved for next cycle)
    - a WARNING must be emitted containing task_id and flag_type

    When add_memory succeeds and the marker is confirmed findable (non-empty memory_ids
    is a proxy; the real gate is whether confirm_marker_persisted returns a canonical id):
    - delete_memory MUST be called once for the prior
    - no no-op WARNING should be emitted
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'prior-hit-resp-test'

    if add_memory_response == 'empty':
        response = AddMemoryResponse(memory_ids=[])
    else:
        response = AddMemoryResponse(memory_ids=['new-marker-id'])

    # task-1400 step-16: confirmation kind filter now scopes by run_id='r1'.
    # 'empty': add_memory is a no-op → no new marker with run_id='r1' indexed
    #          → confirmation misses (only stale prior with run_id='r0' present)
    #          → confirmed_id=None → write_succeeded=False → delete skipped.
    # 'non_empty': add_memory wrote a marker with run_id='r1'
    #              → supply a fresh marker with run_id='r1' for confirmation.
    #              Prior has run_id='r0' and does NOT match confirmation filter.
    if add_memory_response == 'empty':
        search_stub = _make_search_stub(
            suppression=[[]],
            marker={('42', 'missing_deliverable'): [
                [prior_marker],
                [],   # confirmation search (miss)
                [],   # confirmation retry (miss)
            ]},
        )
    else:
        new_marker_r1 = _make_memory_result({
            'source': 'stage1_flag_marker',
            'task_id': '42',
            'flag_type': 'missing_deliverable',
            'run_id': 'r1',  # current run — matches confirmation kind filter
        })
        new_marker_r1.id = 'new-hit-resp-r1'
        search_stub = _make_search_stub(
            suppression=[[]],
            marker={('42', 'missing_deliverable'): [[prior_marker], [new_marker_r1]]},
        )

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=search_stub)
    memory_service.add_memory = AsyncMock(return_value=response)
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # Flag is still annotated with persisted_from_run regardless of write outcome
    assert len(result) == 1
    assert result[0]['persisted_from_run'] == 'r0'
    assert result[0]['last_seen_run_id'] == 'r1'

    # add_memory always called once
    memory_service.add_memory.assert_called_once()

    if expect_delete:
        # Confirmation succeeded: delete should proceed
        memory_service.delete_memory.assert_called_once()
        del_kwargs = memory_service.delete_memory.call_args.kwargs
        assert del_kwargs.get('memory_id') == 'prior-hit-resp-test'
    else:
        # Confirmation missed (no findable marker): delete must be skipped
        memory_service.delete_memory.assert_not_called()

    # Check WARNING for no-op case (task-1400: WARNING now comes from confirmation failure)
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    if expect_noop_warning:
        assert any('42' in m for m in warning_messages), (
            f'Expected WARNING mentioning task_id=42 but got: {warning_messages}'
        )
        assert any('missing_deliverable' in m for m in warning_messages), (
            f'Expected WARNING mentioning flag_type but got: {warning_messages}'
        )
    else:
        # No no-op warning expected when confirmation succeeded
        noop_warnings = [m for m in warning_messages if 'no memory_ids' in m or 'returned no memory_ids' in m]
        assert not noop_warnings, (
            f'Unexpected no-op WARNING on confirmed path: {noop_warnings}'
        )


# ---------------------------------------------------------------------------
# TestFilterSuppressed (task-1186 step-1) — filter_suppressed behavior
# ---------------------------------------------------------------------------


class TestFilterSuppressed:
    """Tests for filter_suppressed(memory_service, project_id, flags).

    All tests import filter_suppressed from flag_dedup; they will fail with
    ImportError until the implementation is added in step-2.
    """

    @pytest.mark.asyncio
    async def test_empty_flags_returns_empty_no_io(self):
        """(a) Empty flags input → empty result + zero I/O."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        result = await filter_suppressed(memory_service, 'p', [])
        assert result == []
        memory_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_search_results_all_flags_pass_through(self):
        """(b) Empty search results → all flags pass through unchanged."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[])
        flags = [
            {'task_id': 42, 'flag_type': 'missing_deliverable'},
            {'task_id': 99, 'flag_type': 'stale_metadata'},
        ]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert result == flags

    @pytest.mark.asyncio
    async def test_search_called_with_canonical_kwargs(self):
        """(c) search called with canonical kwargs (project_id, categories, stores, limit, query).

        limit=501 is used internally (not 500) so that genuine overflow can be
        detected without false positives at the boundary — see filter_suppressed
        docstring for details.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[])
        flags = [{'task_id': 1, 'flag_type': 'missing_deliverable'}]
        await filter_suppressed(memory_service, 'p', flags)

        memory_service.search.assert_called_once()
        kwargs = memory_service.search.call_args.kwargs
        assert kwargs.get('project_id') == 'p'
        assert kwargs.get('categories') == ['observations_and_summaries']
        assert kwargs.get('stores') == ['mem0']
        assert kwargs.get('limit') == 501  # sentinel: request one extra to detect overflow
        assert 'stage1_flag_suppression' in kwargs.get('query', '')

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
        self, flag_task_id, record_task_id
    ):
        """(d) suppression record with matching task_id drops the flag; unrelated flags kept.

        Parametrized: int-int, int-str, str-int, str-str coercion combinations
        so that symmetric str() coercion is verified on both sides.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        suppression_record = _make_memory_result({
            'kind': 'stage1_flag_suppression',
            'task_id': record_task_id,
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[suppression_record])

        suppressed_flag = {'task_id': flag_task_id, 'flag_type': 'missing_deliverable'}
        unrelated_flag = {'task_id': 99, 'flag_type': 'stale_metadata'}
        flags = [suppressed_flag, unrelated_flag]

        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == unrelated_flag

    @pytest.mark.asyncio
    async def test_wrong_kind_not_treated_as_suppression(self):
        """(e) result with metadata.kind != 'stage1_flag_suppression' is NOT treated as suppression.

        Canonical-schema enforcement: vector-search near-miss must be rejected.
        The flag must pass through unchanged.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        near_miss_record = _make_memory_result({
            'kind': 'some_other_kind',
            'task_id': 42,
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[near_miss_record])

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    async def test_correct_kind_but_no_task_id_key_ignored(self):
        """(f) result with metadata.kind correct but no task_id key → ignored, flag passes through."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        no_task_id_record = _make_memory_result({
            'kind': 'stage1_flag_suppression',
            # 'task_id' key intentionally absent
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[no_task_id_record])

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    async def test_metadata_none_guard_does_not_crash(self):
        """(g) result with metadata=None → defensive (r.metadata or {}) guard, doesn't crash, flag passes through."""
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        null_metadata_record = _make_memory_result(None)
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=[null_metadata_record])

        flags = [{'task_id': 42, 'flag_type': 'missing_deliverable'}]
        result = await filter_suppressed(memory_service, 'p', flags)
        assert len(result) == 1
        assert result[0] == flags[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize('suppression_records,label', [
        ([], 'empty_set'),
        (
            [_make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': 'None'})],
            'none_string_in_set',
        ),
        (
            [_make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': 42})],
            'other_valid_id_in_set',
        ),
    ], ids=['empty_set', 'none_string_in_set', 'other_valid_id_in_set'])
    async def test_flag_with_none_task_id_never_dropped(self, suppression_records, label):
        """(h) flag with task_id=None is never suppressed regardless of suppression set contents.

        The consumer-side guard short-circuits to 'keep' when flag_tid is None,
        regardless of what is in the suppressed_task_ids set.  This is symmetric
        with the producer-side suppression-record guard that skips None/empty
        task_ids when building the suppressed set.

        Three cases are parametrized to express the invariant directly:
        - empty_set: trivially preserved (no suppression records at all)
        - none_string_in_set: 'None' as a literal string passes the producer
          guard; without the consumer guard, str(None) == 'None' would drop
          the flag — this is the key regression scenario.
        - other_valid_id_in_set: a real suppression id (42) must not suppress
          a flag whose task_id is None.
        """
        from fused_memory.reconciliation.flag_dedup import filter_suppressed

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(return_value=suppression_records)

        # Flag whose task_id key is present but set to Python None.
        flag_with_None_task_id = {'task_id': None, 'flag_type': 'missing_deliverable'}
        flags = [flag_with_None_task_id]

        result = await filter_suppressed(memory_service, 'p', flags)

        # The flag must always be preserved — task_id=None is not a valid suppression target.
        assert len(result) == 1, (
            f'[{label}] Expected flag with task_id=None to be preserved but result was: {result}'
        )
        assert result[0] == flag_with_None_task_id, (
            f'[{label}] Preserved flag differs from input: {result[0]}'
        )


# ---------------------------------------------------------------------------
# task-1186 step-3 — filter_suppressed: search exception → pass-through + WARNING
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppressed_search_exception_returns_flags_unchanged_and_warns(caplog):
    """Search exception in filter_suppressed → conservative pass-through + WARNING.

    When memory_service.search raises, filter_suppressed must:
    (a) NOT raise
    (b) return both flags unchanged
    (c) emit at least one WARNING log mentioning the failure and the function name
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=RuntimeError('Mem0 down'))

    flags = [
        {'task_id': '55', 'flag_type': 'stale_metadata'},
        {'task_id': '66', 'flag_type': 'missing_deliverable'},
    ]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await filter_suppressed(memory_service, 'p', flags)

    # (a) Does NOT raise
    # (b) Returns both flags unchanged
    assert result == flags
    # (c) At least one WARNING log mentions failure and filter_suppressed
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'filter_suppressed' in m
        for m in warning_messages
    ), f'Expected WARNING mentioning filter_suppressed but got: {warning_messages}'


# ---------------------------------------------------------------------------
# task-1188 step-3 — filter_suppressed: saturation WARNING at limit=500
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppressed_warns_when_search_results_saturate_limit(caplog):
    """filter_suppressed emits WARNING when search returns > 500 records (true overflow).

    The search uses limit=501 internally so that exactly-500-result sets are
    not false-positive warned.  501 results confirm the real set is larger than
    500 and genuine truncation is occurring.  Operators need a WARNING so
    dashboards can alert on incomplete coverage.

    Asserts:
    (a) At least one WARNING is emitted.
    (b) The WARNING message mentions the project_id ('proj_saturation').
    (c) The WARNING message contains '500', 'saturat', or 'truncat' to signal the
        saturation condition without pinning exact wording.
    (d) The non-suppressed flag (task_id=9999) is preserved in the result — a
        regression where saturation swallowed all flags would be caught here.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    records_501 = [
        _make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': i})
        for i in range(501)
    ]
    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=records_501)

    flags = [{'task_id': 9999, 'flag_type': 'missing_deliverable'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await filter_suppressed(memory_service, 'proj_saturation', flags)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        'proj_saturation' in m for m in warning_messages
    ), f'Expected WARNING mentioning project_id but got: {warning_messages}'
    assert any(
        ('500' in m or 'saturat' in m or 'truncat' in m)
        for m in warning_messages
    ), f'Expected WARNING signalling saturation condition but got: {warning_messages}'
    # (d) Non-suppressed flag must pass through even at the saturation boundary.
    assert result == flags, (
        f'Expected non-suppressed flag to be preserved at saturation but got: {result}'
    )


@pytest.mark.asyncio
async def test_filter_suppressed_does_not_warn_when_search_results_below_limit(caplog):
    """filter_suppressed does NOT emit saturation WARNING when results count is <= 500.

    Negative test: prevents accidental always-on logging.  500 results (the
    effective business limit) must NOT trigger the saturation signal — only
    501+ (genuine overflow) should warn.  This eliminates the false-positive
    boundary case that would otherwise fire whenever a project has exactly 500
    active suppression records with no truncation occurring.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    records_500 = [
        _make_memory_result({'kind': 'stage1_flag_suppression', 'task_id': i})
        for i in range(500)
    ]
    memory_service = AsyncMock()
    memory_service.search = AsyncMock(return_value=records_500)

    flags = [{'task_id': 9999, 'flag_type': 'missing_deliverable'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        await filter_suppressed(memory_service, 'proj_below', flags)

    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        ('500' in m or 'saturat' in m or 'truncat' in m)
        for m in warning_messages
    ), f'Unexpected saturation WARNING with 500 results: {warning_messages}'


# ---------------------------------------------------------------------------
# task-1186 step-5 — integration: dedup_flags calls filter_suppressed FIRST
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_calls_filter_suppressed_before_signature_dedup():
    """Integration: dedup_flags calls filter_suppressed BEFORE the signature-dedup loop.

    The mock search side_effect tracks call order:
    - Call 1 (suppression query): returns a record suppressing task_id=42.
    - Call 2+ (per-flag prior-marker queries): returns [].

    Two flags: task_id=42 (suppressed) and task_id=99 (not suppressed), both
    with flag_type='missing_deliverable'.

    Expected outcomes:
    (a) Result has exactly one item — the task_id=99 flag.
    (b) task_id=42 flag was dropped.
    (c) add_memory called exactly once (MISS path for task_id=99 only).
    (d) search called exactly 4 times: 1 suppression + 1 per-flag-marker for task_id=99
        + 2 confirmation searches (first + retry, both miss — no stage1_flag_marker written
        yet from the perspective of the confirmation search).
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    suppression_record = _make_memory_result({
        'kind': 'stage1_flag_suppression',
        'task_id': 42,
    })

    call_count = [0]

    async def _search_side_effect(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call is the filter_suppressed project-scoped query
            return [suppression_record]
        # Subsequent calls: per-flag prior-marker (MISS) and confirmation searches
        return []

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=_search_side_effect)
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

    flags = [
        {'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'suppressed'},
        {'task_id': 99, 'flag_type': 'missing_deliverable', 'description': 'survivor'},
    ]

    result = await dedup_flags(
        memory_service=memory_service,
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
    # (c) add_memory called exactly once — only for task_id=99 MISS path
    assert memory_service.add_memory.call_count == 1, (
        f'Expected exactly 1 add_memory call (task_id=99 MISS) but got '
        f'{memory_service.add_memory.call_count}'
    )
    # (d) search called exactly 4 times: 1 suppression + 1 per-flag-marker + 2 confirmation
    assert memory_service.search.call_count == 4, (
        f'Expected 4 search calls (1 suppression + 1 per-flag + 2 confirmation) but got '
        f'{memory_service.search.call_count}'
    )


@pytest.mark.asyncio
async def test_dedup_flags_two_consecutive_runs_no_predecessor_accumulation():
    """Regression: two successive dedup_flags calls for the same flag leave exactly 1 marker.

    Uses a FakeMem0 in-memory store (self-contained, no external mocks) to
    exercise the full add/search/delete cycle deterministically.

    Run 1 (run_id='r1'): no prior exists → MISS branch → 1 marker stored.
    Run 2 (run_id='r2'): prior found → HIT branch → write replacement, delete prior
                         → still exactly 1 marker stored, with run_id='r2'.
    """
    import uuid as _uuid

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    class _MemoryRecord:
        """Minimal stand-in for a Mem0 MemoryResult."""
        def __init__(self, id_: str, metadata: dict) -> None:
            self.id = id_
            self.metadata = metadata
            self.content = (
                f"Stage 1 flag marker: task={metadata.get('task_id')} "
                f"type={metadata.get('flag_type')} from run={metadata.get('run_id')}"
            )

    class FakeMem0:
        """In-memory Mem0 stub with add_memory, search, and delete_memory."""

        def __init__(self) -> None:
            self._store: dict[str, _MemoryRecord] = {}

        async def add_memory(self, *, metadata: dict, **_kwargs) -> AddMemoryResponse:
            id_ = str(_uuid.uuid4())
            self._store[id_] = _MemoryRecord(id_, metadata)
            return AddMemoryResponse(memory_ids=[id_])

        async def search(self, *, query: str = '', **_kwargs) -> list[_MemoryRecord]:
            # Return all stored records (deterministic; query not used for filtering)
            return list(self._store.values())

        async def delete_memory(self, *, memory_id: str, **_kwargs) -> None:
            self._store.pop(memory_id, None)

        def count(self) -> int:
            return len(self._store)

        def latest_run_id(self) -> str | None:
            """Return the run_id of the single stored marker (for assertion)."""
            records = list(self._store.values())
            if not records:
                return None
            return records[0].metadata.get('run_id')

    fake = FakeMem0()
    flag = {'task_id': '42', 'flag_type': 'missing_deliverable', 'description': 'foo'}

    # Run 1 — MISS branch: no prior, writes one marker
    await dedup_flags(
        memory_service=fake,
        project_id='p',
        run_id='r1',
        flags=[flag],
    )
    assert fake.count() == 1, (
        f'After run 1: expected 1 marker but got {fake.count()}'
    )

    # Run 2 — HIT branch: prior found, replacement written, prior deleted
    await dedup_flags(
        memory_service=fake,
        project_id='p',
        run_id='r2',
        flags=[flag],
    )
    assert fake.count() == 1, (
        f'After run 2: expected 1 marker (no accumulation) but got {fake.count()}'
    )
    assert fake.latest_run_id() == 'r2', (
        f"Surviving marker must have run_id='r2' but got {fake.latest_run_id()!r}"
    )


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
# filter_suppressed end-to-end test (task-1185 step-6)
#
# Closes the producer→reader contract end-to-end for both int and str
# task_id storage variants.  filter_suppressed is already on main from
# sibling task-1186; this test pins that what the producer writes is exactly
# what the reader correctly acts on.
#
# This test would have caught a producer that wrote metadata.kind under a
# different key, or metadata.task_id under a non-str-coercible type, because
# filter_suppressed requires BOTH fields to be present and correct before
# adding task_id to the suppressed set.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_filter_suppressed_drops_flag_written_by_producer():
    """Producer→reader contract (producer path): flag written by write_suppression_record
    is dropped by filter_suppressed.

    Exercises ONLY the canonical producer path (int task_id → write_suppression_record).
    A regression in write_suppression_record's int(task_id) coercion would be caught here.

    Test body:
      (1) build a FakeMemoryService;
      (2) write the suppression record via write_suppression_record (int task_id);
      (3) call filter_suppressed with a matching flag and an unrelated flag;
      (4) assert the suppressed flag is dropped;
      (5) assert the unrelated flag is preserved.
    """
    from fused_memory.reconciliation.flag_dedup import filter_suppressed, write_suppression_record

    task_id = 42
    fake = _FakeMemoryService()
    await write_suppression_record(fake, project_id='dark_factory', task_id=task_id)

    flags = [
        {'task_id': task_id, 'flag_type': 'missing_deliverable'},  # should be dropped
        {'task_id': 99, 'flag_type': 'stale_metadata'},             # should be kept
    ]

    result = await filter_suppressed(fake, 'dark_factory', flags)

    # (4) Suppressed flag is dropped
    assert len(result) == 1, (
        f'Expected 1 surviving flag but got {len(result)}: {result}'
    )
    # (5) Unrelated flag is preserved
    assert result[0]['task_id'] == 99, (
        f"Expected surviving flag task_id=99 but got {result[0]['task_id']!r}"
    )


@pytest.mark.asyncio
async def test_filter_suppressed_drops_flag_for_legacy_str_consumer_record():
    """Producer→reader contract (legacy-str consumer path): a hand-authored record
    with str task_id still causes filter_suppressed to drop the matching flag.

    Exercises ONLY the legacy-str path: task_id stored as str '42' via
    fake.add_memory directly.  The reader's str-coercion handles both int and str
    suppression records, so a legacy str record must suppress the corresponding flag.

    Test body:
      (1) build a FakeMemoryService;
      (2) write a legacy str-task_id record directly via fake.add_memory;
      (3) call filter_suppressed with a matching flag and an unrelated flag;
      (4) assert the suppressed flag is dropped;
      (5) assert the unrelated flag is preserved.
    """
    from fused_memory.reconciliation.flag_dedup import filter_suppressed

    raw_task_id = '42'
    tid_int = int(raw_task_id)
    fake = _FakeMemoryService()

    # Model the legacy/hand-authored str-task_id storage shape directly.
    await fake.add_memory(
        content=f'STAGE 1 FLAG SUPPRESSION task_id={tid_int}',
        category='observations_and_summaries',
        project_id='dark_factory',
        metadata={'kind': 'stage1_flag_suppression', 'task_id': raw_task_id},
    )

    flags = [
        {'task_id': tid_int, 'flag_type': 'missing_deliverable'},  # should be dropped
        {'task_id': 99, 'flag_type': 'stale_metadata'},             # should be kept
    ]

    result = await filter_suppressed(fake, 'dark_factory', flags)

    # (4) Suppressed flag is dropped
    assert len(result) == 1, (
        f'Expected 1 surviving flag but got {len(result)}: {result}'
    )
    # (5) Unrelated flag is preserved
    assert result[0]['task_id'] == 99, (
        f"Expected surviving flag task_id=99 but got {result[0]['task_id']!r}"
    )


# ---------------------------------------------------------------------------
# write_suppression_record tests (task-1185 step-3)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_memory_service():
    """AsyncMock memory_service with add_memory returning a stub AddMemoryResponse."""
    svc = AsyncMock()
    svc.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['supp-1']))
    return svc


class TestWriteSuppressionRecord:
    """Async tests for write_suppression_record(memory_service, *, project_id, task_id, causation_id).

    Uses the ``mock_memory_service`` fixture to avoid per-test boilerplate.
    The canonical-payload, project_id, _source, and default-causation_id
    assertions are consolidated into one test that inspects the full kwargs
    dict; separate small tests cover the causation_id and str-coercion variants.
    """

    @pytest.mark.asyncio
    async def test_canonical_call_kwargs(self, mock_memory_service):
        """add_memory called once with the full canonical kwargs dict.

        Asserts content, category, metadata (kind + int task_id), project_id,
        _source sentinel, and causation_id=None (default) in a single call.
        """
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        result = await write_suppression_record(
            mock_memory_service, project_id='autopilot_video', task_id=42
        )

        mock_memory_service.add_memory.assert_called_once_with(
            content='STAGE 1 FLAG SUPPRESSION task_id=42',
            category='observations_and_summaries',
            metadata={'kind': 'stage1_flag_suppression', 'task_id': 42},
            project_id='autopilot_video',
            causation_id=None,
            _source='stage1_flag_suppression',
        )
        assert result.memory_ids == ['supp-1']

    @pytest.mark.asyncio
    async def test_passes_causation_id_when_provided(self, mock_memory_service):
        """causation_id is forwarded to add_memory when explicitly provided."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(
            mock_memory_service, project_id='p', task_id=42, causation_id='recon-run-99'
        )

        kwargs = mock_memory_service.add_memory.call_args.kwargs
        assert kwargs['causation_id'] == 'recon-run-99'

    @pytest.mark.asyncio
    async def test_coerces_str_task_id(self, mock_memory_service):
        """passing task_id='42' produces metadata.task_id == 42 (int)."""
        from fused_memory.reconciliation.flag_dedup import write_suppression_record

        await write_suppression_record(mock_memory_service, project_id='p', task_id='42')

        kwargs = mock_memory_service.add_memory.call_args.kwargs
        assert kwargs['metadata']['task_id'] == 42
        assert isinstance(kwargs['metadata']['task_id'], int)


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
# task-1400 step-9 — dedup_flags HIT path: deletes priors only when confirmed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_hit_path_no_delete_when_confirmation_misses(caplog):
    """HIT path: when confirmation misses (marker not findable), priors are NOT deleted
    and a WARNING is emitted.

    Prior: [prior id='aaa']; add_memory returns memory_ids=['returned-id'];
    confirmation: search side_effect = [[], []] (both miss — not findable).

    Asserts:
    (a) delete_memory NOT called (priors preserved — self-healing intact).
    (b) WARNING mentioning task_id AND flag_type that the replacement could not be confirmed.
    (c) Flag IS annotated with persisted_from_run (annotation extracted before write).

    Fails until step-10 switches HIT gate from memory_ids to confirmed_id.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'aaa'

    memory_service = AsyncMock()
    # suppression=[], pre-write HIT=[prior], confirmation miss=[], confirmation retry=[]
    memory_service.search = AsyncMock(side_effect=[[], [prior_marker], [], []])
    memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['returned-id']))
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) delete_memory NOT called — priors preserved when replacement unfindable
    memory_service.delete_memory.assert_not_called()

    # (b) WARNING that replacement could not be confirmed
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        ('42' in m or 'missing_deliverable' in m) and
        ('could not confirm' in m or 'cannot confirm' in m or 'not confirmed' in m
         or 'skipping' in m or 'skip' in m)
        for m in warning_messages
    ), (
        f"Expected WARNING about unconfirmed replacement for task 42 but got: {warning_messages}"
    )

    # (c) Flag IS annotated (annotation extracted before write attempt)
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


@pytest.mark.asyncio
async def test_dedup_flags_hit_path_deletes_when_confirmed(caplog):
    """HIT path: when confirmation succeeds (marker findable), all priors are deleted.

    Converse of the above: add_memory returns memory_ids=['returned-id'] (different
    from canonical); confirmation search returns [new_marker id='canon-new'].
    Priors must be deleted exactly as before (existing atomic-replace contract).

    Fails until step-10 switches HIT gate from memory_ids to confirmed_id.
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'aaa'

    new_confirmed_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    new_confirmed_marker.id = 'canon-new'

    memory_service = AsyncMock()
    # suppression=[], pre-write HIT=[prior], confirmation hit=[new_marker]
    memory_service.search = AsyncMock(side_effect=[[], [prior_marker], [new_confirmed_marker]])
    memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['returned-DIFFERENT']))
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=flags,
    )

    # Prior deleted (atomic-replace contract holds when confirmed)
    memory_service.delete_memory.assert_called_once()
    del_kwargs = memory_service.delete_memory.call_args.kwargs
    assert del_kwargs.get('memory_id') == 'aaa'

    # Flag annotated
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


# ---------------------------------------------------------------------------
# task-1400 step-11 — end-to-end integration: confirmation wired, full call sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_end_to_end_confirmation_wired():
    """Integration regression: full call sequence with one MISS-happy and one HIT-happy flag.

    Verifies the complete wired path so a later refactor cannot silently desync
    the search-call sequence.

    Flags:
    - flag_A: has task_id but no flag_type → no-signature under both helpers
      (task-1654: content-fp returns None because task_id is not None), pass-through
    - flag_B: MISS-happy (no prior); confirmation search finds the new marker
    - flag_C: HIT-happy (prior exists); confirmation search confirms replacement

    Search call sequence:
    1. suppression sweep (filter_suppressed) → []
    2. flag_B pre-write search → [] (MISS)
    3. flag_B confirmation → [b_marker] (found)
    4. flag_C pre-write search → [c_prior] (HIT)
    5. flag_C confirmation → [c_new_marker] (found)

    Asserts:
    (a) flag_A returned unchanged (no-signature pass-through)
    (b) flag_B NOT annotated (MISS → no persisted_from_run)
    (c) flag_C annotated with persisted_from_run='r0' and last_seen_run_id='r1'
    (d) add_memory called exactly twice (once for MISS, once for HIT replacement)
    (e) delete_memory called exactly once (for C's prior)
    (f) search called exactly 5 times (suppression + 2 pre-write + 2 confirmation)
    (g) filter_suppressed still issues exactly one project-scoped sweep
    """
    from fused_memory.reconciliation.flag_dedup import dedup_flags

    b_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '10',
        'flag_type': 'stale_metadata',
        'run_id': 'r1',
    })
    b_marker.id = 'b-canon'

    c_prior = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '20',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    c_prior.id = 'c-prior'

    c_new_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '20',
        'flag_type': 'missing_deliverable',
        'run_id': 'r1',
    })
    c_new_marker.id = 'c-canon'

    memory_service = AsyncMock()
    memory_service.search = AsyncMock(side_effect=[
        [],          # (1) suppression sweep
        [],          # (2) flag_B pre-write MISS
        [b_marker],  # (3) flag_B confirmation hit
        [c_prior],   # (4) flag_C pre-write HIT
        [c_new_marker],  # (5) flag_C confirmation hit
    ])
    memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
    memory_service.delete_memory = AsyncMock(return_value=None)

    # flag_A: has task_id but no flag_type → no-sig under both helpers (task-1654:
    # content-fp returns None because task_id is not None; compute_flag_signature
    # returns None because flag_type is missing).
    flag_A = {'task_id': '5', 'description': 'no-signature: missing flag_type'}
    flag_B = {'task_id': '10', 'flag_type': 'stale_metadata', 'description': 'B'}
    flag_C = {'task_id': '20', 'flag_type': 'missing_deliverable', 'description': 'C'}

    result = await dedup_flags(
        memory_service=memory_service,
        project_id='p',
        run_id='r1',
        flags=[flag_A, flag_B, flag_C],
    )

    # (a) flag_A unchanged (no-signature — has task_id but missing flag_type)
    assert len(result) == 3
    assert result[0] == flag_A

    # (b) flag_B not annotated (MISS)
    assert 'persisted_from_run' not in result[1]

    # (c) flag_C annotated (HIT)
    assert result[2].get('persisted_from_run') == 'r0'
    assert result[2].get('last_seen_run_id') == 'r1'

    # (d) add_memory called twice
    assert memory_service.add_memory.call_count == 2, (
        f"Expected 2 add_memory calls but got {memory_service.add_memory.call_count}"
    )

    # (e) delete_memory called once (for C's prior)
    memory_service.delete_memory.assert_called_once()
    assert memory_service.delete_memory.call_args.kwargs.get('memory_id') == 'c-prior'

    # (f) search called exactly 5 times
    assert memory_service.search.call_count == 5, (
        f"Expected 5 search calls but got {memory_service.search.call_count}"
    )


# ---------------------------------------------------------------------------
# task-1400 step-7 — dedup_flags MISS path: confirmation called after write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_miss_path_confirmed_marker_no_noop_warning(caplog):
    """MISS path: when confirmation succeeds (even if returned id != add_memory response id),
    no 'will not be detected next cycle' WARNING is emitted.

    search side_effect:
      [1] [] → suppression filter (no suppression)
      [2] [] → per-flag pre-write MISS (no prior)
      [3] [confirmation_marker id='canon-1'] → post-write confirmation hit

    add_memory returns memory_ids=['returned-DIFFERENT'] (different from canonical id).

    Asserts:
    (a) add_memory called once.
    (b) search called 3 times (suppression + pre-write + confirmation).
    (c) NO 'recurring flag will not be detected next cycle' WARNING (marker confirmed findable).
    (d) Flag is NOT annotated (MISS path → no persisted_from_run).

    Fails until step-8 wires confirmation into MISS branch.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    confirmation_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '77',
        'flag_type': 'stale_metadata',
        'run_id': 'r1',
    })
    confirmation_marker.id = 'canon-1'

    memory_service = AsyncMock()
    # [suppression filter=[], pre-write miss=[], confirmation hit=[marker]]
    memory_service.search = AsyncMock(side_effect=[[], [], [confirmation_marker]])
    memory_service.add_memory = AsyncMock(
        return_value=AddMemoryResponse(memory_ids=['returned-DIFFERENT'])
    )

    flags = [{'task_id': '77', 'flag_type': 'stale_metadata', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) add_memory called once
    memory_service.add_memory.assert_called_once()

    # (b) search called 3 times: suppression + pre-write + confirmation
    assert memory_service.search.call_count == 3, (
        f"Expected 3 search calls (suppression+pre-write+confirmation) but got "
        f"{memory_service.search.call_count}"
    )

    # (c) No 'will not be detected next cycle' WARNING (marker IS confirmed)
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any('will not be detected next cycle' in m for m in warning_messages), (
        f"Expected no 'will not be detected next cycle' WARNING when confirmation "
        f"succeeds, but got: {warning_messages}"
    )

    # (d) Flag not annotated (MISS path)
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]


@pytest.mark.asyncio
async def test_dedup_flags_miss_path_confirmation_miss_emits_noop_warning(caplog):
    """MISS path: when confirmation misses (double-miss), the 'will not be detected next cycle'
    WARNING fires — driven off confirmation, not memory_ids.

    search side_effect:
      [1] [] → suppression filter
      [2] [] → pre-write MISS
      [3] [] → confirmation miss
      [4] [] → confirmation retry miss

    add_memory returns memory_ids=['stub-id'] (non-empty — old guard would have
    suppressed the WARNING; confirmation-driven guard must still emit it).

    Fails until step-8 wires confirmation into MISS branch.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    memory_service = AsyncMock()
    # suppression + pre-write miss + confirmation miss + confirmation retry miss
    memory_service.search = AsyncMock(side_effect=[[], [], [], []])
    memory_service.add_memory = AsyncMock(return_value=AddMemoryResponse(memory_ids=['stub-id']))

    flags = [{'task_id': '88', 'flag_type': 'missing_deliverable', 'description': 'test'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # add_memory called once
    memory_service.add_memory.assert_called_once()

    # WARNING driven by confirmation failure (not just empty memory_ids)
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        ('88' in m or 'missing_deliverable' in m) and
        ('detected next cycle' in m or 'could not confirm' in m or 'will not be detected' in m)
        for m in warning_messages
    ), (
        f"Expected WARNING about confirmation failure for task 88 but got: {warning_messages}"
    )

    # Flag not annotated (MISS path)
    assert len(result) == 1
    assert 'persisted_from_run' not in result[0]


# ---------------------------------------------------------------------------
# task-1400 step-14(B) — HIT path silent no-op does not wipe priors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dedup_flags_hit_path_silent_noop_does_not_wipe_priors(caplog):
    """HIT path: when add_memory is a silent no-op the stale prior must NOT
    masquerade as confirmation, and priors must be preserved.

    Models the deceptive no-op: add_memory returns non-empty memory_ids (as if
    the write succeeded), but the new marker was never indexed by Mem0.  The
    confirmation search can only find the surviving stale prior (run_id='r0').

    With run_id scoping in the confirmation kind filter (step-15):
    - confirmation finds only the prior (run_id='r0', != 'r1') → miss
    - confirmed_id = None → write_succeeded = False → priors NOT deleted
    - flag annotated with persisted_from_run='r0' and last_seen_run_id='r1'

    search side_effect: [[], [prior], [prior], [prior]]
      [1] suppression sweep        → []
      [2] per-flag pre-write HIT   → [prior run_id='r0']
      [3] confirmation first       → [prior run_id='r0'] (stale only, no new marker)
      [4] confirmation retry       → [prior run_id='r0'] (still stale only)

    Asserts:
    (a) delete_memory NOT called (priors preserved — silent no-op protection).
    (b) WARNING fires that the replacement could not be confirmed.
    (c) Flag annotated with persisted_from_run='r0' and last_seen_run_id='r1'
        (annotation extracted before write attempt).

    Fails until step-15 adds run_id to the confirmation kind filter so that
    the stale prior with run_id='r0' is not wrongly accepted as proof that the
    run_id='r1' write landed.
    """
    import logging

    from fused_memory.reconciliation.flag_dedup import dedup_flags

    prior_marker = _make_memory_result({
        'source': 'stage1_flag_marker',
        'task_id': '42',
        'flag_type': 'missing_deliverable',
        'run_id': 'r0',
        'last_seen_run_id': 'r0',
    })
    prior_marker.id = 'aaa'

    memory_service = AsyncMock()
    # suppression=[], pre-write HIT=[prior], confirmation=[prior], retry=[prior]
    memory_service.search = AsyncMock(
        side_effect=[[], [prior_marker], [prior_marker], [prior_marker]]
    )
    memory_service.add_memory = AsyncMock(
        return_value=AddMemoryResponse(memory_ids=['returned-id'])
    )
    memory_service.delete_memory = AsyncMock(return_value=None)

    flags = [{'task_id': 42, 'flag_type': 'missing_deliverable', 'description': 'foo'}]

    with caplog.at_level(logging.WARNING, logger='fused_memory.reconciliation.flag_dedup'):
        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=flags,
        )

    # (a) delete_memory NOT called — priors preserved (silent no-op protection)
    memory_service.delete_memory.assert_not_called()

    # (b) WARNING fires that replacement could not be confirmed
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(
        ('42' in m or 'missing_deliverable' in m) and
        ('could not confirm' in m or 'cannot confirm' in m or 'not confirmed' in m
         or 'skipping' in m or 'skip' in m)
        for m in warning_messages
    ), (
        f"Expected WARNING about unconfirmed replacement but got: {warning_messages}"
    )

    # (c) Flag annotated (annotation extracted before write attempt, from the prior)
    assert len(result) == 1
    assert result[0].get('persisted_from_run') == 'r0'
    assert result[0].get('last_seen_run_id') == 'r1'


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

    Mirrors the _make_search_stub / _make_memory_result / AddMemoryResponse
    stub pattern from existing dedup_flags tests above.  The search stub is
    keyed on the fp:… marker query string, computed from the test flag's
    description at test setup time.
    """

    @pytest.mark.asyncio
    async def test_cycle1_miss_writes_marker_keyed_by_content_fingerprint(self):
        """Cycle 1 (no prior marker): fp:… task_id now accepted — marker is written.

        Updated by task-1670 (Option A): the guard now accepts canonical fp:+32-hex keys.
        find_prior_memories is called (MISS → []), then _write_and_confirm_marker writes
        a stage1_flag_marker with the fp: task_id in its metadata.  The flag is returned
        unchanged (no persisted_from_run annotation on the MISS path).
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

        # Confirmation stub: prior search → [] (MISS); confirm search → [written marker].
        written_marker = _make_memory_result({
            'source': 'stage1_flag_marker',
            'kind': 'stage1_flag_marker',
            'task_id': expected_fp,
            'flag_type': expected_ftype,
            'run_id': 'r1',
        })
        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            marker={(expected_fp, expected_ftype): [[], [written_marker]]},
        ))
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        result = await dedup_flags(
            memory_service=memory_service,
            project_id='p',
            run_id='r1',
            flags=[flag],
        )

        # (a) Flag returned unchanged — MISS path does not annotate with persisted_from_run
        assert len(result) == 1
        assert 'persisted_from_run' not in result[0], (
            f'MISS path: flag must not have persisted_from_run; got {result[0]}'
        )

        # (b) add_memory called once with fp: key in marker metadata
        memory_service.add_memory.assert_called_once()
        _assert_valid_stage1_marker(
            memory_service.add_memory.call_args.kwargs,
            task_id=expected_fp,
            flag_type=expected_ftype,
            run_id='r1',
        )

    @pytest.mark.asyncio
    async def test_cycle2_hit_annotates_flag_with_persisted_from_run(self):
        """Cycle 2 (prior fp:… marker present): flag annotated, replacement written, prior deleted.

        Updated by task-1670 (Option A): the guard now accepts fp: keys.
        find_prior_memories returns the cycle-1 marker (HIT); the flag is annotated
        with persisted_from_run=r1 / last_seen_run_id=r2; a replacement marker is
        written; and the prior is deleted (best-effort replacement pattern).
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

        # Cycle-1 fp: prior marker (was written in cycle 1).
        fp_prior = _make_memory_result({
            'source': 'stage1_flag_marker',
            'kind': 'stage1_flag_marker',
            'task_id': fp,
            'flag_type': ftype,
            'run_id': 'r1',
            'last_seen_run_id': 'r1',
        })
        fp_prior.id = 'fp-prior-r1'

        # Replacement marker returned by confirm search (run_id=r2).
        replacement = _make_memory_result({
            'source': 'stage1_flag_marker',
            'kind': 'stage1_flag_marker',
            'task_id': fp,
            'flag_type': ftype,
            'run_id': 'r2',
        })
        replacement.id = 'fp-replacement-r2'

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            marker={(fp, ftype): [[fp_prior], [replacement]]},
        ))
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
        memory_service.delete_memory = AsyncMock(return_value=None)

        result = await dedup_flags(
            memory_service=memory_service,
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

        # (b) Replacement written
        memory_service.add_memory.assert_called_once()
        meta = memory_service.add_memory.call_args.kwargs.get('metadata', {})
        assert meta.get('task_id') == fp, (
            f'Replacement marker task_id must be {fp!r}; got {meta.get("task_id")!r}'
        )

        # (c) Prior deleted after confirmed replacement
        memory_service.delete_memory.assert_called_once_with(
            memory_id='fp-prior-r1',
            store='mem0',
            project_id='p',
            causation_id='r2',
            _source='stage1_flag_dedup',
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

    async def test_numeric_task_id_still_writes_marker(self):
        """(c) Regression: numeric task_id '42' passes guard → add_memory called."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flag = {'task_id': 42, 'flag_type': 'missing_deliverable'}

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            marker={('42', 'missing_deliverable'): [[], [_make_memory_result({
                'source': 'stage1_flag_marker',
                'task_id': '42',
                'flag_type': 'missing_deliverable',
                'run_id': 'r1',
            })]]},
        ))
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        await dedup_flags(
            memory_service=memory_service,
            project_id='proj',
            run_id='r1',
            flags=[flag],
        )

        # Guard accepts numeric key → add_memory called exactly once
        memory_service.add_memory.assert_called_once()
        meta = memory_service.add_memory.call_args.kwargs.get('metadata', {})
        assert meta.get('task_id') == '42', (
            f'Marker task_id must be "42"; got {meta.get("task_id")!r}'
        )

    async def test_comma_joined_cited_tasks_signature_still_writes_marker(self):
        """(d) Regression: comma-joined cited_tasks key '12,15' passes guard → add_memory called."""
        from fused_memory.reconciliation.flag_dedup import dedup_flags

        flag = {
            'task_id': None,
            'flag_type': 'cross_task_blocker',
            'cited_tasks': [{'task_id': 15}, {'task_id': 12}],
        }
        # compute_flag_signature produces ('12,15', 'cross_task_blocker') via cited_tasks fallback

        memory_service = AsyncMock()
        memory_service.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            marker={('12,15', 'cross_task_blocker'): [[], [_make_memory_result({
                'source': 'stage1_flag_marker',
                'task_id': '12,15',
                'flag_type': 'cross_task_blocker',
                'run_id': 'r1',
            })]]},
        ))
        memory_service.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        await dedup_flags(
            memory_service=memory_service,
            project_id='proj',
            run_id='r1',
            flags=[flag],
        )

        # Guard accepts comma-joined integer key → add_memory called exactly once
        memory_service.add_memory.assert_called_once()
        meta = memory_service.add_memory.call_args.kwargs.get('metadata', {})
        assert meta.get('task_id') == '12,15', (
            f'Marker task_id must be "12,15"; got {meta.get("task_id")!r}'
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

    async def test_cross_cycle_fp_roundtrip(self):
        """(e2) Cross-cycle round-trip: cycle 1 writes fp: marker; cycle 2 detects it.

        Verifies the full Stage-1-internal dedup loop for fp: keys:
        - Cycle 1 (MISS): dedup_flags writes a stage1_flag_marker with task_id=fp:…
        - Cycle 2 (HIT): find_prior_memories finds the cycle-1 marker; flag is annotated
          with persisted_from_run → no re-escalation on cycle 2+.

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

        # ---- Cycle 1: MISS — no prior marker ----
        written_c1 = _make_memory_result({
            'source': 'stage1_flag_marker',
            'kind': 'stage1_flag_marker',
            'task_id': fp,
            'flag_type': ftype,
            'run_id': 'c1',
            'last_seen_run_id': 'c1',
        })
        written_c1.id = 'fp-marker-c1'

        ms_c1 = AsyncMock()
        ms_c1.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            marker={(fp, ftype): [[], [written_c1]]},
        ))
        ms_c1.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)

        result_c1 = await dedup_flags(
            memory_service=ms_c1,
            project_id='proj',
            run_id='c1',
            flags=[flag],
        )

        # Cycle 1: flag returned unchanged (MISS — no prior)
        assert 'persisted_from_run' not in result_c1[0], (
            f'Cycle 1 MISS must not annotate flag; got {result_c1[0]}'
        )
        # Marker was written
        ms_c1.add_memory.assert_called_once()
        written_meta = ms_c1.add_memory.call_args.kwargs.get('metadata', {})
        assert written_meta.get('task_id') == fp

        # ---- Cycle 2: HIT — prior fp: marker found via written_c1 ----
        replacement_c2 = _make_memory_result({
            'source': 'stage1_flag_marker',
            'kind': 'stage1_flag_marker',
            'task_id': fp,
            'flag_type': ftype,
            'run_id': 'c2',
        })
        replacement_c2.id = 'fp-marker-c2'

        ms_c2 = AsyncMock()
        ms_c2.search = AsyncMock(side_effect=_make_search_stub(
            suppression=[[]],
            # Prior = written_c1 marker; confirm = replacement_c2
            marker={(fp, ftype): [[written_c1], [replacement_c2]]},
        ))
        ms_c2.add_memory = AsyncMock(return_value=_STUB_ADD_MEMORY_RESPONSE)
        ms_c2.delete_memory = AsyncMock(return_value=None)

        result_c2 = await dedup_flags(
            memory_service=ms_c2,
            project_id='proj',
            run_id='c2',
            flags=[flag],
        )

        # Cycle 2: flag annotated — dedup worked, no re-escalation
        assert result_c2[0]['persisted_from_run'] == 'c1', (
            f'Cycle 2 must annotate persisted_from_run="c1"; got {result_c2[0].get("persisted_from_run")!r}'
        )
        assert result_c2[0]['last_seen_run_id'] == 'c2'

        # Replacement written and prior deleted
        ms_c2.add_memory.assert_called_once()
        ms_c2.delete_memory.assert_called_once_with(
            memory_id='fp-marker-c1',
            store='mem0',
            project_id='proj',
            causation_id='c2',
            _source='stage1_flag_dedup',
        )


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

