"""Tests for ``fused_memory/maintenance/topic_slug_lockstep.py``, the write
boundary and two-store lockstep behind ``scripts/normalize_topic_slugs.py``
(task 4878).

Every store is a double: ``rename_one`` and ``rename_group`` are driven with
injected memory services and MCP clients, and no test loads the script.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from _topic_slug_migration_doubles import (
    fail_writes_moving_back_to,
    gate_task,
    mcp_client,
    refuse_writes_to,
    result_row,
    write_boundary_service,
)

from fused_memory.maintenance.topic_slug_gates import GateGroup, pair_gate_blocks
from fused_memory.maintenance.topic_slug_lockstep import (
    WRITE_REASON,
    WRITE_SOURCE,
    describe_partial_ungated_group,
    rename_group,
    rename_one,
)
from fused_memory.maintenance.topic_slug_plan import Rename
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY

# ===========================================================================
# rename_one — the single write boundary
# ===========================================================================

def _rename(**overrides) -> Rename:
    fields = {
        'project_id': 'dark_factory',
        'memory_id': 'm1',
        'old_topic': 'legacy_topic',
        'new_topic': 'legacy-topic',
    }
    fields.update(overrides)
    return Rename(**fields)


def _call_names(service: AsyncMock) -> list[str]:
    return [name for name, _a, _kw in service.mock_calls if name]


class TestRenameOne:
    """``rename_one(memory_service, rename, *, apply)``.

    The only function here that touches a store.  Order is load-bearing and
    copied from ``retro_stamp_topics.stamp_one``: live re-read, decide, then
    write.  The re-read is not a formality — the plan's ids come off a scroll
    that may be minutes old on a corpus orchestrators are writing to, so a
    record consolidated away since must become a report line rather than a
    Qdrant ``set_payload`` that acknowledges a write to nothing.
    """

    @pytest.mark.asyncio
    async def test_reads_live_before_it_writes(self):
        service = write_boundary_service()
        await rename_one(service, _rename(), apply=True)
        assert _call_names(service) == ['get_memory_by_id', 'update_memory']

    @pytest.mark.asyncio
    async def test_the_write_is_metadata_only_and_carries_only_topic(self):
        """A migration must not touch content, and must not touch a sibling key.

        ``metadata_mode='merge'`` with a patch containing exactly ``topic``
        leaves every other metadata key alone.  A wider patch would let a
        normalization silently clobber ``canonical``, ``supersedes`` or the
        consolidation bookkeeping on ~141 records at once.
        """
        service = write_boundary_service()
        await rename_one(service, _rename(), apply=True)
        service.update_memory.assert_awaited_once()
        kwargs = service.update_memory.await_args.kwargs
        assert kwargs['memory_id'] == 'm1'
        assert kwargs['project_id'] == 'dark_factory'
        assert kwargs['metadata_patch'] == {'topic': 'legacy-topic'}
        assert kwargs['metadata_mode'] == 'merge'
        assert 'content' not in kwargs
        assert 'messages' not in kwargs

    @pytest.mark.asyncio
    async def test_the_write_is_attributed_to_this_sweep(self):
        """``_source`` / ``reason`` so the amendment-storm alarm can tell.

        A bulk run under the default ``mcp_tool`` source looks exactly like
        the runaway rewrite that alarm exists to catch.
        """
        service = write_boundary_service()
        await rename_one(service, _rename(), apply=True)
        kwargs = service.update_memory.await_args.kwargs
        assert kwargs['_source'] == WRITE_SOURCE == 'normalize_topic_slugs'
        assert kwargs['reason'] == WRITE_REASON
        assert '4878' in WRITE_REASON

    @pytest.mark.asyncio
    async def test_no_delete_is_ever_issued(self):
        """A topic rename is a metadata patch, never a delete-and-recreate.

        Re-creating would mint a new memory id, orphaning every reference to
        the old one and destroying ``created_at`` — the field consolidation
        uses to pick a canonical.
        """
        service = write_boundary_service()
        await rename_one(service, _rename(), apply=True)
        service.delete_memory.assert_not_awaited()
        assert 'delete_memory' not in _call_names(service)

    @pytest.mark.asyncio
    async def test_a_successful_write_is_renamed(self):
        service = write_boundary_service()
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'renamed'
        assert result['memory_id'] == 'm1'
        assert result['old_topic'] == 'legacy_topic'
        assert result['new_topic'] == 'legacy-topic'
        assert result['response']['status'] == 'updated'

    @pytest.mark.asyncio
    async def test_dry_run_reads_and_decides_but_never_writes(self):
        """The rehearsal must exercise the SAME decision path as the apply.

        A dry run that skipped the live re-read would rehearse a different
        function from the one ``--apply`` runs, and its report would be a
        prediction rather than a rehearsal.
        """
        service = write_boundary_service()
        result = await rename_one(service, _rename(), apply=False)
        assert result['outcome'] == 'would_rename'
        service.get_memory_by_id.assert_awaited_once()
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_vanished_record_is_memory_not_found(self):
        """Consolidated away between the scroll and the write.

        Measured precedent: retro_stamp's gate 3036 id ``19705df4`` no longer
        resolves. Writing to it anyway would be acknowledged by the backend
        and would appear in the report as a successful rename.
        """
        service = write_boundary_service(record=None)
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'memory_not_found'
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_topic_that_moved_since_the_plan_is_refused(self):
        """Somebody else rewrote it — this plan row is stale, so do not write.

        Applying anyway would overwrite a live value with a fold of a value
        that no longer exists, which is a data loss the report would score as
        a success.
        """
        service = write_boundary_service(record={'id': 'm1', 'metadata': {'topic': 'something_else'}})
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'topic_moved_since_plan'
        assert result['existing_topic'] == 'something_else'
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_already_normalized_costs_zero_writes(self):
        """A second run plans zero WRITES, not merely zero net effect.

        A no-op ``update_memory`` would still journal a write op, still count
        toward the amendment-storm alarm, and would inflate the report's
        renamed count with records that gained nothing.
        """
        service = write_boundary_service(record={'id': 'm1', 'metadata': {'topic': 'legacy-topic'}})
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'already_normalized'
        service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_error_envelope_is_update_failed_not_renamed(self):
        """``update_memory`` reports some rejections by RETURNING, not raising.

        A caller guarding only against exceptions would score a refused write
        as a rename — and then the residue probe would find the legacy slug
        still populated with no explanation in the report.
        """
        service = write_boundary_service(update_response={
            'error': 'memory not found', 'error_type': 'MemoryNotFound',
        })
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'update_failed'
        assert result['error_type'] == 'MemoryNotFound'
        assert result['error'] == 'memory not found'
        assert result['response'] == {
            'error': 'memory not found', 'error_type': 'MemoryNotFound',
        }

    @pytest.mark.asyncio
    async def test_a_raising_read_is_one_row_not_an_unwound_sweep(self):
        """Every await is individually guarded.

        One backend hiccup must cost one report row, not the whole run — the
        artifact is written after the loop, so an exception escaping here
        would discard everything the sweep had already learned.
        """
        service = write_boundary_service()
        service.get_memory_by_id.side_effect = RuntimeError('qdrant timeout')
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'rename_error'
        assert 'RuntimeError' in result['error']
        assert 'qdrant timeout' in result['error']

    @pytest.mark.asyncio
    async def test_a_raising_write_is_one_row_too(self):
        service = write_boundary_service()
        service.update_memory.side_effect = RuntimeError('set_payload failed')
        result = await rename_one(service, _rename(), apply=True)
        assert result['outcome'] == 'rename_error'
        assert 'set_payload failed' in result['error']

    @pytest.mark.asyncio
    async def test_the_memory_id_is_never_changed(self):
        """Whatever the outcome, the row reports the id it was handed.

        The report is how an operator re-addresses a failed row; an id that
        drifted between the plan and the artifact makes it unusable.
        """
        for record in (None, {'id': 'm1', 'metadata': {'topic': 'other_x'}}):
            result = await rename_one(
                write_boundary_service(record=record), _rename(), apply=True,
            )
            assert result['memory_id'] == 'm1'

    @pytest.mark.asyncio
    async def test_every_outcome_carries_what_it_needs_to_be_actionable(self):
        """No follow-up query should be needed to act on a report row."""
        result = await rename_one(write_boundary_service(), _rename(), apply=True)
        for key in ('project_id', 'memory_id', 'old_topic', 'new_topic', 'outcome'):
            assert key in result


# ===========================================================================
# rename_group — the two-store lockstep
# ===========================================================================
#
# ``metadata.x_recon_consolidation_gate`` is a Tier-C block on TASK metadata,
# not Mem0 metadata: a genuinely different store, reached through the MCP
# server rather than through a backend call.  Renaming the memories without
# moving that block leaves the gate's closure scroll pointed at a slug no
# record carries any more — the uncloseable-gate trap re-created at a NEW
# address, which is strictly worse than leaving the snake_case slug alone.
#
# So the halves are planned as a pair and refused as a pair.  Every test below
# exists to pin one edge of that atomicity.

def _stateful_service(records: dict[str, dict]) -> MagicMock:
    """A ``MemoryService`` double that REMEMBERS what was written to it.

    The gate lockstep's most important property — that a refused gate patch
    leaves the memory half at its ORIGINAL value — is a statement about the
    store's end state, not about a call list.  A stateless double can only
    show the calls; it cannot show that the corpus came back to where it
    started, which is the whole claim.
    """
    service = MagicMock()

    async def _get(*, project_id, memory_id):
        record = records.get(memory_id)
        return None if record is None else {
            'id': memory_id, 'metadata': dict(record['metadata']),
        }

    async def _update(*, memory_id, project_id, metadata_patch, **kwargs):
        record = records.get(memory_id)
        if record is None:
            return {'error_type': 'not_found', 'error': memory_id}
        record['metadata'].update(metadata_patch)
        return {'status': 'updated', 'id': memory_id, 'metadata_patched': True}

    service.get_memory_by_id = AsyncMock(side_effect=_get)
    service.update_memory = AsyncMock(side_effect=_update)
    return service


def _store(*topics: tuple[str, str]) -> dict[str, dict]:
    return {mid: {'metadata': {'topic': topic}} for mid, topic in topics}


def _one_group(renames, gate_tasks=()) -> GateGroup:
    """The single :class:`GateGroup` a writer test drives, unpacked."""
    groups, _skips = pair_gate_blocks(renames, list(gate_tasks))
    assert len(groups) == 1, groups
    return groups[0]


class TestGateLockstepWriter:
    """``rename_group(memory_service, group, *, apply, client)``.

    The pair is ALL-OR-NOTHING.  Memory renames land first; the gate patch
    follows only once they ALL succeeded; and if either half fails the group
    ends at its original state, so no gate is ever left scrolling a slug no
    record carries.
    """

    @pytest.mark.asyncio
    async def test_memory_renames_land_first_then_the_gate_patch(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = mcp_client()
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert [r['outcome'] for r in outcome.results] == ['renamed']
        assert records['m1']['metadata']['topic'] == 'gate-topic'
        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_patched'
        # The write is a NARROW merge of the one block, never a whole-blob
        # replace: `update_task` refuses any replace carrying `done_provenance`.
        (tool, payload), _ = client.call_tool.call_args
        assert tool == 'update_task'
        assert payload['id'] == '4220'
        assert payload['metadata_mode'] == 'merge'
        assert payload['metadata'] == {
            GATE_METADATA_KEY: {'topic': 'gate-topic'},
        }

    @pytest.mark.asyncio
    async def test_a_dry_run_writes_neither_half(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = mcp_client()
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=False, client=client)

        assert [r['outcome'] for r in outcome.results] == ['would_rename']
        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'would_patch_gate'
        assert records['m1']['metadata']['topic'] == 'gate_topic'
        service.update_memory.assert_not_awaited()
        client.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_memory_failure_withholds_the_gate_patch(self):
        """The other direction of the pair: a failed half holds the whole group."""
        records = _store(('m1', 'gate_topic'))  # m2 is absent -> memory_not_found
        service = _stateful_service(records)
        client = mcp_client()
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic'),
             Rename('dark_factory', 'm2', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert {r['outcome'] for r in outcome.results} >= {'memory_not_found'}
        client.call_tool.assert_not_awaited()
        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_lockstep_failed'
        assert outcome.gate_row['half'] == 'memory'
        assert outcome.held_entry is None, 'a held gated pair is reported by its gate row'

    @pytest.mark.asyncio
    async def test_a_refused_gate_patch_undoes_the_memory_half(self):
        """The dangerous direction, and the reason the group exists at all.

        ``update_task`` reports some rejections by RETURNING an error envelope
        inside a successful JSON-RPC frame, so a caller guarding only against
        exceptions would score a refused gate move as a completed one — and
        leave the gate scrolling a slug the records no longer carry.
        """
        records = _store(('m1', 'gate_topic'), ('m2', 'gate_topic'))
        service = _stateful_service(records)
        client = mcp_client(result={'success': False, 'error': 'done_provenance_via_update_task'})
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic'),
             Rename('dark_factory', 'm2', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_lockstep_failed'
        assert outcome.gate_row['half'] == 'gate'
        assert 'done_provenance_via_update_task' in str(outcome.gate_row['error'])
        # Net effect: the slug is exactly where it started in BOTH stores.
        assert records['m1']['metadata']['topic'] == 'gate_topic'
        assert records['m2']['metadata']['topic'] == 'gate_topic'
        assert [r['outcome'] for r in outcome.results] == [
            'gate_lockstep_failed', 'gate_lockstep_failed']

    @pytest.mark.asyncio
    async def test_a_raising_gate_patch_undoes_the_memory_half_too(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = mcp_client(error=RuntimeError('mcp down'))
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_lockstep_failed'
        assert 'mcp down' in str(outcome.gate_row['error'])
        assert records['m1']['metadata']['topic'] == 'gate_topic'

    @pytest.mark.asyncio
    async def test_a_failed_undo_is_named_on_the_row_not_swallowed(self):
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        client = mcp_client(error=RuntimeError('mcp down'))
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        calls = {'n': 0}
        forward = service.update_memory.side_effect

        async def _fail_the_undo(**kwargs):
            calls['n'] += 1
            if calls['n'] > 1:
                raise RuntimeError('undo failed')
            return await forward(**kwargs)

        service.update_memory = AsyncMock(side_effect=_fail_the_undo)

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_lockstep_failed'
        assert outcome.gate_row['undo_failures']
        assert 'undo failed' in str(outcome.gate_row['undo_failures'])

    @pytest.mark.asyncio
    async def test_an_ungated_group_never_reaches_the_client(self):
        records = _store(('m1', 'plain_topic'))
        service = _stateful_service(records)
        client = mcp_client()
        group = _one_group(
            [Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic')])

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert [r['outcome'] for r in outcome.results] == ['renamed']
        assert (outcome.gate_row, outcome.held_entry) == (None, None)
        client.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_ungated_group_runs_without_a_client_at_all(self):
        """A run that reaches no gate must not require an MCP handshake."""
        records = _store(('m1', 'plain_topic'))
        group = _one_group(
            [Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic')])

        outcome = await rename_group(
            _stateful_service(records), group, apply=True, client=None)

        assert [r['outcome'] for r in outcome.results] == ['renamed']
        assert outcome.gate_row is None

    @pytest.mark.asyncio
    async def test_a_gated_group_with_no_client_is_a_lockstep_failure(self):
        """Unable to move the gate is the same verdict as refused to move it."""
        records = _store(('m1', 'gate_topic'))
        service = _stateful_service(records)
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=True, client=None)

        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_lockstep_failed'
        assert records['m1']['metadata']['topic'] == 'gate_topic'

    @pytest.mark.asyncio
    async def test_an_already_normalized_record_is_not_undone(self):
        """Undo reverses THIS run's writes; it does not rewrite what it found."""
        records = _store(('m1', 'gate-topic'), ('m2', 'gate_topic'))
        service = _stateful_service(records)
        client = mcp_client(error=RuntimeError('mcp down'))
        group = _one_group(
            [Rename('dark_factory', 'm1', 'gate_topic', 'gate-topic'),
             Rename('dark_factory', 'm2', 'gate_topic', 'gate-topic')],
            [gate_task('4220', 'gate_topic')])

        outcome = await rename_group(
            service, group, apply=True, client=client)

        assert outcome.gate_row is not None
        assert outcome.gate_row['outcome'] == 'gate_lockstep_failed'
        assert records['m1']['metadata']['topic'] == 'gate-topic'
        assert records['m2']['metadata']['topic'] == 'gate_topic'


class TestUngatedGroupRollback:
    """An ungated slug moves all-or-nothing too.

    A split slug cannot be finished by re-running: the collision guard sees
    the moved members as incumbents on the new slug.  So when a member is
    left on the legacy value, the members this run moved are rolled back.
    """

    @staticmethod
    def _plain_group() -> GateGroup:
        return _one_group([
            Rename('dark_factory', 'm1', 'plain_topic', 'plain-topic'),
            Rename('dark_factory', 'm2', 'plain_topic', 'plain-topic'),
        ])

    @pytest.mark.asyncio
    async def test_a_refused_member_rolls_the_moved_members_back(self):
        records = _store(('m1', 'plain_topic'), ('m2', 'plain_topic'))
        service = _stateful_service(records)
        refuse_writes_to(service, 'm2')

        outcome = await rename_group(
            service, self._plain_group(), apply=True, client=None)

        assert records['m1']['metadata']['topic'] == 'plain_topic'
        assert records['m2']['metadata']['topic'] == 'plain_topic'
        assert [(r['memory_id'], r['outcome'], r.get('undone')) for r in outcome.results] == [
            ('m1', 'ungated_partial_rename', True),
            ('m2', 'update_failed', None),
        ]
        assert outcome.held_entry is not None
        assert {key: value for key, value in outcome.held_entry.items() if key != 'note'} == {
            'reason': 'ungated_partial_rename',
            'project_id': 'dark_factory',
            'old_topic': 'plain_topic',
            'new_topic': 'plain-topic',
            'moved_memory_ids': ['m1'],
            'failed_memory_ids': ['m2'],
            'failed_outcomes': ['update_failed'],
            'on_new_slug_memory_ids': [],
            'undo_failures': [],
        }

    @pytest.mark.asyncio
    async def test_a_failed_rollback_is_named_not_swallowed(self):
        records = _store(('m1', 'plain_topic'), ('m2', 'plain_topic'))
        service = _stateful_service(records)
        refuse_writes_to(service, 'm2')
        fail_writes_moving_back_to(service, 'm1', 'plain_topic')

        outcome = await rename_group(
            service, self._plain_group(), apply=True, client=None)

        assert records['m1']['metadata']['topic'] == 'plain-topic'
        (m1_row,) = [r for r in outcome.results if r['memory_id'] == 'm1']
        assert m1_row['undone'] is False
        assert outcome.held_entry is not None
        assert [failure['memory_id'] for failure in outcome.held_entry['undo_failures']] == ['m1']
        assert 'undo failed' in str(outcome.held_entry['undo_failures'])
        assert outcome.held_entry['on_new_slug_memory_ids'] == ['m1']

    @pytest.mark.asyncio
    async def test_a_member_found_on_the_new_slug_is_named_as_still_there(self):
        """Another writer moved m1 between plan and apply: the hold cannot put it back.

        The undo reverses only this run's writes, so nothing is rolled back and
        the slug stays split.  The entry says so structurally, rather than
        leaving an empty ``undo_failures`` to read as "whole again".
        """
        records = _store(('m1', 'plain-topic'), ('m2', 'plain_topic'))
        service = _stateful_service(records)
        refuse_writes_to(service, 'm2')

        outcome = await rename_group(
            service, self._plain_group(), apply=True, client=None)

        assert records['m1']['metadata']['topic'] == 'plain-topic'
        assert records['m2']['metadata']['topic'] == 'plain_topic'
        entry = outcome.held_entry
        assert entry is not None
        assert (entry['moved_memory_ids'], entry['failed_memory_ids'],
                entry['on_new_slug_memory_ids'], entry['undo_failures']) == (
            [], ['m2'], ['m1'], [])

    @pytest.mark.asyncio
    async def test_a_vanished_straggler_does_not_roll_back_the_rest(self):
        """A member that is gone splits nothing, so the good write stands."""
        records = _store(('m1', 'plain_topic'))  # m2 is absent -> memory_not_found
        service = _stateful_service(records)

        outcome = await rename_group(
            service, self._plain_group(), apply=True, client=None)

        assert records['m1']['metadata']['topic'] == 'plain-topic'
        assert [(r['memory_id'], r['outcome']) for r in outcome.results] == [
            ('m1', 'renamed'), ('m2', 'memory_not_found'),
        ]
        assert outcome.held_entry is None

    @pytest.mark.asyncio
    async def test_a_dry_run_predicts_the_hold_and_writes_nothing(self):
        records = _store(('m1', 'plain_topic'), ('m2', 'plain_topic'))
        service = _stateful_service(records)
        forward = service.get_memory_by_id.side_effect

        async def _read(**kwargs):
            if kwargs['memory_id'] == 'm2':
                raise RuntimeError('read failed')
            return await forward(**kwargs)

        service.get_memory_by_id = AsyncMock(side_effect=_read)

        outcome = await rename_group(
            service, self._plain_group(), apply=False, client=None)

        assert [(r['memory_id'], r['outcome']) for r in outcome.results] == [
            ('m1', 'would_rename'), ('m2', 'rename_error'),
        ]
        entry = outcome.held_entry
        assert entry is not None
        assert (entry['reason'], entry['moved_memory_ids'], entry['failed_memory_ids']) == (
            'ungated_partial_rename', ['m1'], ['m2'])
        assert (entry['on_new_slug_memory_ids'], entry['undo_failures']) == ([], [])
        service.update_memory.assert_not_awaited()


class TestDescribePartialUngatedGroup:
    """``describe_partial_ungated_group(group, results)``.

    Pure.  An ungated slug that only partly moved is listed with what moved
    and what was left on the legacy value; any other group yields ``None``.
    """

    @staticmethod
    def _plain_group(gate_tasks=(), *, members: int = 2) -> GateGroup:
        return _one_group([
            Rename('dark_factory', f'm{n}', 'plain_topic', 'plain-topic')
            for n in range(1, members + 1)
        ], gate_tasks)

    @staticmethod
    def _plain_results(*outcomes: str) -> list[dict]:
        return [
            result_row(outcome, memory_id=f'm{n}', old_topic='plain_topic', new_topic='plain-topic')
            for n, outcome in enumerate(outcomes, start=1)
        ]

    def test_a_partly_moved_ungated_group_names_both_halves(self):
        entry = describe_partial_ungated_group(
            self._plain_group(), self._plain_results('renamed', 'update_failed'))

        assert entry is not None
        assert {key: value for key, value in entry.items() if key != 'note'} == {
            'reason': 'ungated_partial_rename',
            'project_id': 'dark_factory',
            'old_topic': 'plain_topic',
            'new_topic': 'plain-topic',
            'moved_memory_ids': ['m1'],
            'failed_memory_ids': ['m2'],
            'failed_outcomes': ['update_failed'],
        }

    def test_a_rehearsed_rename_counts_as_moved(self):
        entry = describe_partial_ungated_group(
            self._plain_group(), self._plain_results('would_rename', 'rename_error'))

        assert entry is not None
        assert entry['moved_memory_ids'] == ['m1']
        assert entry['failed_memory_ids'] == ['m2']
        assert entry['failed_outcomes'] == ['rename_error']

    def test_a_member_found_on_the_new_slug_splits_it_without_having_moved(self):
        entry = describe_partial_ungated_group(
            self._plain_group(), self._plain_results('already_normalized', 'update_failed'))

        assert entry is not None
        assert entry['moved_memory_ids'] == []
        assert entry['failed_memory_ids'] == ['m2']

    @pytest.mark.parametrize(('gate_tasks', 'outcomes'), [
        pytest.param([gate_task('4220', 'plain_topic')], ('renamed', 'update_failed'), id='gated'),
        pytest.param([], ('renamed', 'already_normalized'), id='all_stood'),
        pytest.param([], ('update_failed', 'rename_error'), id='none_stood'),
        pytest.param([], ('renamed', 'memory_not_found'), id='straggler_vanished'),
        pytest.param([], ('renamed', 'topic_moved_since_plan'), id='straggler_moved_elsewhere'),
    ])
    def test_no_entry_unless_an_ungated_slug_is_split(self, gate_tasks, outcomes):
        assert describe_partial_ungated_group(
            self._plain_group(gate_tasks), self._plain_results(*outcomes)) is None

    def test_only_the_members_left_on_the_legacy_value_are_listed_as_failed(self):
        entry = describe_partial_ungated_group(
            self._plain_group(members=3),
            self._plain_results('renamed', 'update_failed', 'memory_not_found'))

        assert entry is not None
        assert entry['moved_memory_ids'] == ['m1']
        assert entry['failed_memory_ids'] == ['m2']
        assert entry['failed_outcomes'] == ['update_failed']
