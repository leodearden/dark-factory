"""Tests for the recon Mem0 deletion tombstone
(fused_memory.reconciliation.mem0_tombstone) — task 3041.

Two units live in that module:

- :func:`is_protected_mirror_record` — the shared precision predicate every
  marker-GC sweep consults at its one choke point (``_sweep_stale_mem0_pool``),
  so no marker sweep can ever take a ``kind='cycle_summary'`` /
  ``record_type='ledger_stamp'`` mirror no matter how loose its Qdrant payload
  filter is.
- :func:`record_mem0_deletion_tombstone` — writes the queryable audit row that
  makes a designed eviction distinguishable from silent data loss (the defining
  signature of the recon-gate-165 / esc-165-1 finding).
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.mem0_tombstone import (
    MEM0_TOMBSTONE_TTL_DAYS,
    RECORD_KIND_MEM0_TOMBSTONE,
    is_protected_mirror_record,
    record_mem0_deletion_tombstone,
)

_LOGGER = 'fused_memory.reconciliation.mem0_tombstone'


class TestIsProtectedMirrorRecord:
    """The shared protected-mirror predicate consulted by every marker sweep."""

    @pytest.mark.parametrize(
        'metadata',
        [
            {'kind': 'cycle_summary'},
            {'record_type': 'ledger_stamp'},
            {'kind': 'cycle_summary', 'record_type': 'ledger_stamp'},
            # A mirror that ALSO carries an over-broad sweep's match key is
            # still protected — this is the concrete collateral-loss shape.
            {'kind': 'cycle_summary', 'flag_for_stage2': True},
            {'record_type': 'ledger_stamp', 'source': 'stage1_flag_marker'},
        ],
    )
    def test_protected_payloads(self, metadata):
        assert is_protected_mirror_record(metadata) is True

    @pytest.mark.parametrize(
        'metadata',
        [
            {'source': 'stage1_flag_marker'},
            {'flag_for_stage2': True},
            {'source': 'stage2_persistence_marker'},
            {'kind': 'flag_marker'},
            {'record_type': 'narrative'},
            {},
        ],
    )
    def test_ordinary_marker_payloads_are_not_protected(self, metadata):
        assert is_protected_mirror_record(metadata) is False

    @pytest.mark.parametrize(
        'metadata',
        [None, 'not-a-dict', 42, ['kind', 'cycle_summary'], object()],
    )
    def test_malformed_input_returns_false_without_raising(self, metadata):
        """A marker sweep must never crash on a weird payload."""
        assert is_protected_mirror_record(metadata) is False

    def test_module_constants(self):
        assert RECORD_KIND_MEM0_TOMBSTONE == 'mem0_tombstone'
        assert isinstance(MEM0_TOMBSTONE_TTL_DAYS, int)
        assert MEM0_TOMBSTONE_TTL_DAYS > 0


_NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)
_VICTIM_METADATA = {
    'kind': 'cycle_summary',
    'record_type': 'ledger_stamp',
    'source': 'cycle_summary_mirror',
    'recon_pool': 'stage1_cycle_summary',
    'run_id': 'run-victim',
    'stage': 'memory_consolidator',
    'content': 'should NOT be copied into the tombstone',
}


def _svc_with_ledger() -> tuple[AsyncMock, AsyncMock]:
    ledger = AsyncMock()
    ledger.upsert = AsyncMock(return_value=None)
    memory_service = AsyncMock()
    memory_service.recon_ledger = ledger
    return memory_service, ledger


class TestRecordMem0DeletionTombstone:
    """The tombstone write itself: one idempotent, TTL'd ledger row per victim."""

    @pytest.mark.asyncio
    async def test_writes_exactly_one_row_with_the_expected_identity(self):
        memory_service, ledger = _svc_with_ledger()

        result = await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at='2026-07-28T00:00:00+00:00',
            deleter='stage1_cycle_summary_trim',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        assert result is True
        assert ledger.upsert.await_count == 1
        record = ledger.upsert.await_args.args[0]
        assert record.project_id == 'dark_factory'
        assert record.record_kind == 'mem0_tombstone'
        # The auditor's single known key — the memory uuid — is sufficient for
        # the existing 5-part get_by_identity lookup.
        assert record.task_id == 'mem-victim-uuid'
        assert record.flag_type == ''
        assert record.run_id == ''
        assert record.state == 'deleted'

    @pytest.mark.asyncio
    async def test_timestamps_use_the_canonical_isoformat_the_gc_pass_requires(self):
        memory_service, ledger = _svc_with_ledger()

        await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at='2026-07-28T00:00:00+00:00',
            deleter='stage1_cycle_summary_trim',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        record = ledger.upsert.await_args.args[0]
        assert record.created_at == _NOW.isoformat()
        assert record.expires_at == (
            _NOW + timedelta(days=MEM0_TOMBSTONE_TTL_DAYS)
        ).isoformat()
        # The ledger's gc() compares expires_at as plain lexicographic TEXT, so
        # the canonical +00:00 offset form is load-bearing.
        assert record.created_at.endswith('+00:00')
        assert record.expires_at.endswith('+00:00')

    @pytest.mark.asyncio
    async def test_payload_carries_deleter_and_victim_identity(self):
        memory_service, ledger = _svc_with_ledger()

        await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at='2026-07-28T00:00:00+00:00',
            deleter='stage1_cycle_summary_trim',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        payload = json.loads(ledger.upsert.await_args.args[0].payload_json)
        assert isinstance(payload, dict)
        assert payload['deleter'] == 'stage1_cycle_summary_trim'
        assert payload['deleting_run_id'] == 'run-deleter'
        assert payload['deleted_at'] == _NOW.isoformat()
        assert payload['kind'] == 'cycle_summary'
        assert payload['record_type'] == 'ledger_stamp'
        assert payload['source'] == 'cycle_summary_mirror'
        assert payload['recon_pool'] == 'stage1_cycle_summary'
        # The victim's OWN run_id, distinct from the deleting run — the precise
        # ambiguity that made the original finding unreadable.
        assert payload['run_id'] == 'run-victim'
        assert payload['created_at'] == '2026-07-28T00:00:00+00:00'
        # Only identifying keys are copied; the record's content is not.
        assert 'content' not in payload

    @pytest.mark.asyncio
    async def test_naive_now_is_normalized_to_utc(self):
        memory_service, ledger = _svc_with_ledger()
        naive = datetime(2026, 7, 30, 12, 0, 0)

        await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at=None,
            deleter='stage1_flag_marker_gc_sweep',
            deleting_run_id='run-deleter',
            now=naive,
        )

        record = ledger.upsert.await_args.args[0]
        assert record.created_at == _NOW.isoformat()
        assert record.expires_at == (
            _NOW + timedelta(days=MEM0_TOMBSTONE_TTL_DAYS)
        ).isoformat()

    @pytest.mark.asyncio
    async def test_missing_victim_metadata_still_writes_a_tombstone(self):
        """A tombstone with no victim payload is still better than none."""
        memory_service, ledger = _svc_with_ledger()

        result = await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=None,
            victim_created_at=None,
            deleter='stage1_flag_marker_gc_sweep',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        assert result is True
        payload = json.loads(ledger.upsert.await_args.args[0].payload_json)
        assert payload['deleter'] == 'stage1_flag_marker_gc_sweep'
        assert payload['kind'] is None

    @pytest.mark.asyncio
    async def test_no_recon_ledger_attribute_returns_false_without_raising(self):
        """Mirrors the getattr(memory_service, 'recon_ledger', None) precedent."""
        memory_service = AsyncMock(spec=[])  # no recon_ledger attribute at all

        result = await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at=None,
            deleter='stage1_cycle_summary_trim',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_none_recon_ledger_returns_false_without_raising(self):
        memory_service = AsyncMock()
        memory_service.recon_ledger = None

        result = await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at=None,
            deleter='stage1_cycle_summary_trim',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_failure_returns_false_logs_one_warning_and_does_not_raise(
        self, caplog
    ):
        memory_service, ledger = _svc_with_ledger()
        ledger.upsert = AsyncMock(side_effect=RuntimeError('sqlite is on fire'))

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            result = await record_mem0_deletion_tombstone(
                memory_service,
                'dark_factory',
                'mem-victim-uuid',
                victim_metadata=_VICTIM_METADATA,
                victim_created_at=None,
                deleter='stage1_cycle_summary_trim',
                deleting_run_id='run-deleter',
                now=_NOW,
            )

        assert result is False
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'mem-victim-uuid' in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_default_now_is_used_when_omitted(self):
        memory_service, ledger = _svc_with_ledger()

        result = await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at=None,
            deleter='stage1_cycle_summary_trim',
            deleting_run_id='run-deleter',
        )

        assert result is True
        record = ledger.upsert.await_args.args[0]
        assert datetime.fromisoformat(record.created_at).tzinfo is not None
