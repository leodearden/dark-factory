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
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from fused_memory.reconciliation.mem0_tombstone import (
    MEM0_TOMBSTONE_TTL_DAYS,
    RECORD_KIND_MEM0_TOMBSTONE,
    is_protected_mirror_record,
    record_mem0_deletion_tombstone,
    record_mem0_deletion_tombstones,
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

    def test_discriminators_are_single_sourced_not_copied(self):
        """Every module's view of the two discriminators must be ONE literal.

        The guard above is an OR over ``kind`` and ``record_type``, and the
        pool trim in summary_pool matches on the same two literals from the
        other side of an import edge this module cannot traverse back. They
        were originally duplicated here and "kept in sync BY CONVENTION", with
        nothing pinning the copies equal — so an edit to summary_pool's copy
        alone would have silently disabled half this guard (a ledger_stamp
        mirror would stop being protected while still being trimmed), which is
        indistinguishable from the silent loss task 3041 exists to make
        impossible.

        They now live in the import-free leaf ``recon_pool_map``. This test
        fails loudly if anyone re-duplicates them (reviewer finding
        duplication, task 3041 amendment pass).
        """
        from fused_memory.reconciliation import mem0_tombstone, recon_pool_map, summary_pool

        assert recon_pool_map.CYCLE_SUMMARY_KIND == 'cycle_summary'
        assert recon_pool_map.CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP == 'ledger_stamp'
        assert recon_pool_map.CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE == 'narrative'

        # The predicate's inputs and the trim's filter/eviction-order inputs
        # must be the SAME value, not two that merely happen to match today.
        assert (
            mem0_tombstone._KIND_CYCLE_SUMMARY
            == summary_pool._KIND_CYCLE_SUMMARY
            == recon_pool_map.CYCLE_SUMMARY_KIND
        )
        assert (
            mem0_tombstone._RECORD_TYPE_LEDGER_STAMP
            == summary_pool.CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP
            == recon_pool_map.CYCLE_SUMMARY_RECORD_TYPE_LEDGER_STAMP
        )
        assert (
            summary_pool.CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE
            == recon_pool_map.CYCLE_SUMMARY_RECORD_TYPE_NARRATIVE
        )


# Probe run in a FRESH interpreter by TestReconPoolMapIsImportFreeLeaf: import
# the package the leaf sits in, snapshot the loaded ``fused_memory.*`` modules,
# import the leaf, snapshot again. The delta is exactly what importing the leaf
# costs — measured, not read off the source text.
_LEAF_IMPORT_PROBE = """
import json
import sys

import fused_memory.reconciliation

base = {m for m in sys.modules if m.startswith('fused_memory')}

import fused_memory.reconciliation.recon_pool_map  # noqa: F401

after = {m for m in sys.modules if m.startswith('fused_memory')}
print(json.dumps({'base': sorted(base), 'delta': sorted(after - base)}))
"""


class TestReconPoolMapIsImportFreeLeaf:
    """``recon_pool_map`` must stay a genuine import-free leaf.

    That leaf-ness is what lets BOTH sides of the one-way
    ``summary_pool -> mem0_tombstone`` edge reach the same cycle_summary
    discriminator literals (see
    :meth:`TestIsProtectedMirrorRecord.test_discriminators_are_single_sourced_not_copied`),
    instead of keeping private copies "in sync by convention". Re-introducing
    an intra-package import here re-creates the circular import documented in
    ``recon_pool_map``'s own docstring and would force those copies back.

    This asserts on OBSERVED RUNTIME IMPORTS, not on source text. It replaces
    an ``inspect.getsource`` substring guard that was both brittle (a doc
    cleanup touching the module's own near-miss prose about ``from
    fused_memory`` broke the build without any real regression) and incomplete
    (relative, dynamic and line-broken imports all re-created the cycle while
    keeping it green). Same class of test this repo removed once already in
    da8e5a4c96.
    """

    def test_importing_the_leaf_loads_no_other_fused_memory_module(self):
        proc = subprocess.run(
            [sys.executable, '-c', _LEAF_IMPORT_PROBE],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # (a) A probe that died on import must fail LOUDLY. Checking this
        # first is what stops an empty delta from vacuously passing.
        assert proc.returncode == 0, (
            'the leaf-import probe did not run to completion — this test must '
            'never pass vacuously.\n'
            f'returncode: {proc.returncode}\nstdout: {proc.stdout}\n'
            f'stderr: {proc.stderr}'
        )
        stdout_lines = [line for line in proc.stdout.splitlines() if line.strip()]
        assert stdout_lines, f'probe produced no output.\nstderr: {proc.stderr}'
        observed = json.loads(stdout_lines[-1])

        # (b) Importing the leaf pulls in NO other fused_memory module. Catches
        # absolute, relative AND dynamic imports alike.
        assert observed['delta'] == ['fused_memory.reconciliation.recon_pool_map'], (
            'recon_pool_map is no longer an import-free leaf; importing it now '
            f'also loads {sorted(set(observed["delta"]) - {"fused_memory.reconciliation.recon_pool_map"})}. '
            'It is imported by BOTH summary_pool and mem0_tombstone to '
            'single-source the cycle_summary discriminators — see its module '
            'docstring for the circular import this prevents.'
        )

        # (c) The two package __init__s the leaf sits under stay import-free
        # too (1 and 6 lines today). If a future task legitimately makes a
        # parent non-leaf, THIS is the intended signal.
        assert observed['base'] == ['fused_memory', 'fused_memory.reconciliation'], (
            'a package __init__ above recon_pool_map stopped being '
            f'import-free; merely importing fused_memory.reconciliation now '
            f'loads {observed["base"]}.'
        )


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


class TestTheReversePointer:
    """`absorbed_by` — the survivor id that ate this record (task 3133).

    The recon-gate-165 audit dead-end that motivated this: a consolidation
    survivor correctly carried the FORWARD pointer
    (`consolidated_from=<victim>`), but probing the DEAD id returned
    `{'found': false}` with no tombstone at all. From the victim id alone
    — the only id an auditor chasing a broken reference actually holds —
    that deletion was indistinguishable from silent data loss.

    Recorded as a first-class field rather than smuggled through
    `victim_metadata`: `_VICTIM_IDENTITY_KEYS` is deliberately an
    identity-only projection of what the VICTIM recorded about itself, and
    a canonical id is not that. Routing it there would make the tombstone
    misreport its own provenance and silently widen a projection whose
    narrowness is a stated invariant.
    """

    @pytest.mark.asyncio
    async def test_the_canonical_that_absorbed_the_victim_is_recorded(self):
        memory_service, ledger = _svc_with_ledger()

        await record_mem0_deletion_tombstone(
            memory_service,
            'dark_factory',
            'mem-victim-uuid',
            victim_metadata=_VICTIM_METADATA,
            victim_created_at='2026-07-28T00:00:00+00:00',
            deleter='consolidate_memories',
            deleting_run_id='run-deleter',
            absorbed_by='canonical-uuid',
            now=_NOW,
        )

        payload = json.loads(ledger.upsert.await_args.args[0].payload_json)
        assert payload['absorbed_by'] == 'canonical-uuid'
        # It joins the existing fields rather than displacing any of them.
        assert payload['deleter'] == 'consolidate_memories'
        assert payload['deleting_run_id'] == 'run-deleter'
        assert payload['deleted_at'] == _NOW.isoformat()
        assert payload['created_at'] == '2026-07-28T00:00:00+00:00'
        assert payload['kind'] == 'cycle_summary'

    @pytest.mark.asyncio
    async def test_absent_is_written_as_an_explicit_none(self):
        """PRESENT-with-None, never omitted. The key must distinguish
        "reaped by a GC sweep, nothing absorbed it" from a tombstone
        written before this field existed — and only presence can carry
        that, which is why the assertion is on the KEY, not on falsiness.
        Matches how the identity projection already writes explicit `None`
        for victim keys the record did not have."""
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
        assert 'absorbed_by' in payload
        assert payload['absorbed_by'] is None

    @pytest.mark.asyncio
    async def test_the_batch_form_stamps_every_victim_with_the_same_absorber(self):
        """Batch-LEVEL by construction: every victim in one consolidation is
        absorbed by the same canonical, so this is a property of the call,
        not of each row."""
        memory_service, ledger = _svc_with_ledger()
        victims = [
            {
                'id': f'victim-{i}',
                'metadata': _VICTIM_METADATA,
                'created_at': '2026-07-28T00:00:00+00:00',
            }
            for i in range(3)
        ]

        written = await record_mem0_deletion_tombstones(
            memory_service,
            'dark_factory',
            victims,
            deleter='consolidate_memories',
            deleting_run_id='run-deleter',
            absorbed_by='canonical-uuid',
            now=_NOW,
        )

        assert written == 3
        records = ledger.upsert_many.await_args.args[0]
        payloads = [json.loads(r.payload_json) for r in records]
        assert [p['absorbed_by'] for p in payloads] == ['canonical-uuid'] * 3
        assert [r.task_id for r in records] == ['victim-0', 'victim-1', 'victim-2']

    @pytest.mark.asyncio
    async def test_the_existing_sweeps_need_no_edit(self):
        """BACK-COMPAT, called in the exact shape `summary_pool` and
        `stages/task_knowledge_sync` use. The two live sweeps absorb
        nothing, so the keyword is keyword-only with a `None` default and
        the change is purely additive."""
        memory_service, ledger = _svc_with_ledger()
        victims = [
            {
                'id': 'victim-0',
                'metadata': _VICTIM_METADATA,
                'created_at': '2026-07-28T00:00:00+00:00',
            }
        ]

        written = await record_mem0_deletion_tombstones(
            memory_service,
            'dark_factory',
            victims,
            deleter='stage1_flag_marker_gc_sweep',
            deleting_run_id='run-deleter',
            now=_NOW,
        )

        assert written == 1
        payload = json.loads(ledger.upsert_many.await_args.args[0][0].payload_json)
        assert payload['absorbed_by'] is None

    @pytest.mark.asyncio
    async def test_the_writer_stays_fail_safe_with_the_new_arg(self):
        """No ledger wired => 0/False, no raise. This runs inside a delete
        path; the new field must not have given it a way to blow up."""
        memory_service = AsyncMock()
        memory_service.recon_ledger = None

        assert (
            await record_mem0_deletion_tombstones(
                memory_service,
                'dark_factory',
                [{'id': 'victim-0', 'metadata': {}, 'created_at': None}],
                deleter='consolidate_memories',
                deleting_run_id='run-deleter',
                absorbed_by='canonical-uuid',
            )
            == 0
        )
        assert (
            await record_mem0_deletion_tombstone(
                memory_service,
                'dark_factory',
                'victim-0',
                victim_metadata=None,
                victim_created_at=None,
                deleter='consolidate_memories',
                deleting_run_id='run-deleter',
                absorbed_by='canonical-uuid',
            )
            is False
        )
