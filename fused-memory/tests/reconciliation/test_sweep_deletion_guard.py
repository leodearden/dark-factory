"""Tests for fused_memory.reconciliation.sweep_deletion_guard (task 5271).

``filter_benign_sweep_deletion_flags`` drops a Stage-1 evidentiary-anchor
deletion-pattern flag whose swept Mem0 ids were all tombstoned by a documented
recon sweep.  Incident: solar_challenge_platform run 09f2829f / finding
c4639ec8, whose three "new occurrences" all carried stage1/stage2
cycle-summary-trim tombstones from the same run.
"""

from __future__ import annotations

import copy
import logging
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.sweep_deletion_guard import (
    filter_benign_sweep_deletion_flags,
)

_PROJECT = 'solar_challenge_platform'
_SWEPT_A = '445c97ac-14d7-4956-9e46-e4475eecff16'
_SWEPT_B = '9529e396-761a-4914-93cc-19ec00c74ab8'
_SWEPT_C = '90b08c6e-c091-412c-8d7a-2265854c489c'
_STAMP_1 = 'fb667ebc-1964-4e64-a7b1-bbf92c396a34'
_STAMP_2 = 'e8ba2f02-03fa-44f3-89f2-e89fb73b039a'
_TRACKER = '867b12ed-c77b-4a49-a213-debf2bb40989'
_TRIM_RUN = 'ea597ad1-f634-4b33-a074-0aa755c40b4a'
_UNTOMBSTONED = '252e1cd8-5b0e-4c3a-9f4e-2d1c0b9a8e7f'


class _FakeTombstoneReader:
    """Answers get_mem0_deletion_tombstone from a dict and records every call."""

    def __init__(
        self,
        tombstones: dict[str, Any] | None = None,
        *,
        raising: frozenset[str] = frozenset(),
    ) -> None:
        self._tombstones = tombstones or {}
        self._raising = raising
        self.calls: list[tuple[str, str]] = []

    async def get_mem0_deletion_tombstone(self, project_id: str, memory_id: str) -> Any:
        self.calls.append((project_id, memory_id))
        if memory_id in self._raising:
            raise RuntimeError('recon ledger locked')
        return self._tombstones.get(memory_id)


def _tombstone(deleter: str, deleting_run_id: str = _TRIM_RUN) -> dict[str, Any]:
    return {
        'deleter': deleter,
        'deleting_run_id': deleting_run_id,
        'deleted_at': '2026-09-20T10:00:00+00:00',
        'kind': 'cycle_summary',
        'record_type': 'ledger_stamp',
    }


def _incident_tombstones() -> dict[str, Any]:
    return {
        _SWEPT_A: _tombstone('stage1_cycle_summary_trim'),
        _SWEPT_B: _tombstone('stage1_cycle_summary_trim'),
        _SWEPT_C: _tombstone('stage2_cycle_summary_trim'),
    }


def _deleted(memory_id: str, *, cascaded: list[str] | None = None) -> ReconciliationEvent:
    payload: dict[str, Any] = {'memory_id': memory_id, 'store': 'mem0'}
    if cascaded is not None:
        payload['cascaded_child_ids'] = cascaded
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=EventType.memory_deleted,
        source=EventSource.agent,
        project_id=_PROJECT,
        timestamp=datetime(2026, 9, 20, 10, 0, tzinfo=UTC),
        payload=payload,
    )


def _deletion_flag(
    swept: tuple[str, ...] = (_SWEPT_A, _SWEPT_B, _SWEPT_C),
    *,
    flag_type: str = 'mem0_evidentiary_anchor_deletion_pattern',
) -> dict[str, Any]:
    """Finding c4639ec8, incident-shaped: swept ids appear ONLY in the prose."""
    occurrences = '; '.join(
        f'mem0 {m} deleted +0.065s after the ledger-stamp write' for m in swept
    )
    return {
        'task_id': '165',
        'flag_type': flag_type,
        'category': 'memory_integrity',
        'description': (
            f'Evidentiary-anchor deletion pattern recurs: {occurrences}. '
            f'Ledger stamps {_STAMP_1} and {_STAMP_2} remain; all in run {_TRIM_RUN}.'
        ),
        'cited_memories': [
            {'memory_id': m, 'store': 'mem0'} for m in (_STAMP_1, _STAMP_2, _TRACKER)
        ],
    }


def _swept_by_id(flag: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {s['memory_id']: s for s in flag['sweep_deletion_provenance']['swept']}


@pytest.mark.asyncio
async def test_incident_all_swept_ids_benign_is_dropped():
    """(1) Every swept id was a cycle-summary trim -> DROP."""
    reader = _FakeTombstoneReader(_incident_tombstones())
    events = [_deleted(_SWEPT_A), _deleted(_SWEPT_B), _deleted(_SWEPT_C)]

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [_deletion_flag()], events=events,
    )

    assert result == [], (
        'all three "new occurrences" carry documented trim tombstones, so the '
        f'flag describes a designed sweep and must be DROPPED; got {result!r}'
    )


@pytest.mark.asyncio
async def test_untombstoned_deletion_keeps_the_flag():
    """(2) A deleted id with no tombstone is an unexplained deletion -> KEEP."""
    tombstones = _incident_tombstones()
    del tombstones[_SWEPT_C]
    reader = _FakeTombstoneReader(tombstones)
    flag = _deletion_flag()
    events = [_deleted(_SWEPT_A), _deleted(_SWEPT_B), _deleted(_SWEPT_C)]

    result = await filter_benign_sweep_deletion_flags(reader, _PROJECT, [flag], events=events)

    assert result == [flag]
    provenance = flag['sweep_deletion_provenance']
    assert provenance['decision'] == 'kept_unexplained_deletion'
    assert _swept_by_id(flag)[_SWEPT_C]['classification'] == 'untombstoned'


@pytest.mark.asyncio
async def test_undocumented_deleter_keeps_the_flag():
    """(3) A tombstone from a deleter outside the documented set -> KEEP."""
    tombstones = _incident_tombstones()
    tombstones[_SWEPT_B] = _tombstone('unregistered_manual_sweep')
    reader = _FakeTombstoneReader(tombstones)
    flag = _deletion_flag()

    result = await filter_benign_sweep_deletion_flags(reader, _PROJECT, [flag], events=[])

    assert result == [flag]
    assert flag['sweep_deletion_provenance']['decision'] == 'kept_unexplained_deletion'
    assert _swept_by_id(flag)[_SWEPT_B]['classification'] == 'undocumented_deleter'
    assert _swept_by_id(flag)[_SWEPT_B]['deleter'] == 'unregistered_manual_sweep'


@pytest.mark.asyncio
async def test_mixed_flag_is_kept_and_records_the_benign_ids():
    """(4) The bundled-252e1cd8 shape: two benign plus one untombstoned -> KEEP."""
    reader = _FakeTombstoneReader({
        _SWEPT_A: _tombstone('stage1_cycle_summary_trim'),
        _SWEPT_B: _tombstone('stage2_cycle_summary_trim', 'run-trim-2'),
    })
    flag = _deletion_flag((_SWEPT_A, _SWEPT_B, _UNTOMBSTONED))
    events = [_deleted(_SWEPT_A), _deleted(_SWEPT_B), _deleted(_UNTOMBSTONED)]

    result = await filter_benign_sweep_deletion_flags(reader, _PROJECT, [flag], events=events)

    assert result == [flag], 'a partially-benign claim must not be cleared'
    swept = _swept_by_id(flag)
    assert swept[_SWEPT_A] == {
        'memory_id': _SWEPT_A,
        'deleter': 'stage1_cycle_summary_trim',
        'deleting_run_id': _TRIM_RUN,
        'classification': 'benign',
    }
    assert swept[_SWEPT_B]['deleter'] == 'stage2_cycle_summary_trim'
    assert swept[_SWEPT_B]['deleting_run_id'] == 'run-trim-2'
    assert swept[_SWEPT_B]['classification'] == 'benign'
    assert swept[_UNTOMBSTONED]['classification'] == 'untombstoned'
    assert swept[_UNTOMBSTONED]['deleter'] is None


@pytest.mark.asyncio
async def test_benign_tombstone_counts_as_swept_without_a_deletion_event():
    """(5) A tombstone alone marks an id swept, even from an earlier cycle."""
    reader = _FakeTombstoneReader(_incident_tombstones())

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [_deletion_flag()], events=[],
    )

    assert result == []


@pytest.mark.asyncio
async def test_prior_cycle_untombstoned_deletion_is_invisible_to_the_gate():
    """(5b) The documented visibility limit: outside this cycle's buffer, an
    untombstoned deletion reads like a run id, so a flag bundling it with
    benign ids is DROPPED.  Closing this gap must update the module docstring
    and the Stage-1 prompt, which state the limit."""
    reader = _FakeTombstoneReader({
        _SWEPT_A: _tombstone('stage1_cycle_summary_trim'),
        _SWEPT_B: _tombstone('stage2_cycle_summary_trim'),
    })
    flag = _deletion_flag((_SWEPT_A, _SWEPT_B, _UNTOMBSTONED))

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [flag], events=[_deleted(_SWEPT_A), _deleted(_SWEPT_B)],
    )

    assert result == []


@pytest.mark.asyncio
async def test_flag_naming_no_swept_id_is_kept():
    """(6) Run ids and present records are neither tombstoned nor deleted -> KEEP."""
    reader = _FakeTombstoneReader({})
    flag = _deletion_flag(())

    result = await filter_benign_sweep_deletion_flags(reader, _PROJECT, [flag], events=[])

    assert result == [flag]
    assert flag['sweep_deletion_provenance'] == {
        'swept': [],
        'decision': 'kept_no_swept_ids',
    }


@pytest.mark.asyncio
async def test_raising_lookup_reads_as_no_tombstone_and_warns(caplog):
    """(7) A raising tombstone read is untombstoned, KEEPs, and is loud."""
    reader = _FakeTombstoneReader(_incident_tombstones(), raising=frozenset({_SWEPT_A}))
    flag = _deletion_flag()

    with caplog.at_level(logging.WARNING):
        result = await filter_benign_sweep_deletion_flags(
            reader, _PROJECT, [flag], events=[_deleted(_SWEPT_A)],
        )

    assert result == [flag]
    assert _swept_by_id(flag)[_SWEPT_A]['classification'] == 'untombstoned'
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any(_SWEPT_A in w for w in warnings), (
        f'a swallowed lookup error must WARN naming the id; got {warnings!r}'
    )


@pytest.mark.asyncio
async def test_non_dict_tombstone_reads_as_no_tombstone():
    """(8) A non-dict tombstone value is not a tombstone."""
    tombstones = _incident_tombstones()
    tombstones[_SWEPT_A] = MagicMock()
    reader = _FakeTombstoneReader(tombstones)
    flag = _deletion_flag()

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [flag], events=[_deleted(_SWEPT_A)],
    )

    assert result == [flag]
    assert _swept_by_id(flag)[_SWEPT_A]['classification'] == 'untombstoned'


@pytest.mark.parametrize('flag_type', [
    'Deletion_Pattern-mem0 evidentiary anchor',
    'mem0_evidentiary_anchor_full_mirror_loss',
    'cycle_summary_evidentiary_anchor_mirror_deletion_confirmed',
])
@pytest.mark.asyncio
async def test_observed_and_paraphrased_spellings_are_candidates(flag_type):
    """(9) Every measured Stage-1 spelling, and a reworded one, is a candidate."""
    reader = _FakeTombstoneReader(_incident_tombstones())

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [_deletion_flag(flag_type=flag_type)], events=[],
    )

    assert result == [], f'{flag_type!r} must be a candidate and DROP; got {result!r}'


@pytest.mark.asyncio
async def test_non_candidates_pass_through_with_zero_lookups():
    """(10) Other families are never looked up or annotated."""
    reader = _FakeTombstoneReader(_incident_tombstones())
    flags = [
        {'task_id': '1', 'flag_type': 'memory_duplicate', 'description': f'mem0 {_SWEPT_A}'},
        {'task_id': '2', 'flag_type': 'missing_deliverable', 'description': 'b'},
    ]
    snapshot = copy.deepcopy(flags)

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, flags, events=[_deleted(_SWEPT_A)],
    )

    assert result == snapshot
    assert flags == snapshot, 'a non-candidate must never be annotated'
    assert reader.calls == [], 'a batch with no candidates must do zero I/O'


@pytest.mark.parametrize(('has_reader', 'project_id'), [
    (False, _PROJECT),
    (True, ''),
    (False, ''),
])
@pytest.mark.asyncio
async def test_falsy_dependencies_degrade_to_a_no_op(has_reader, project_id):
    """(11) Falsy memory_service / project_id -> unchanged pass-through."""
    reader = _FakeTombstoneReader(_incident_tombstones())
    flags = [_deletion_flag()]
    snapshot = copy.deepcopy(flags)

    result = await filter_benign_sweep_deletion_flags(
        reader if has_reader else None, project_id, flags, events=[],
    )

    assert result == snapshot
    assert reader.calls == []


@pytest.mark.asyncio
async def test_each_distinct_id_is_looked_up_once_per_call():
    """(12) Two flags naming the same ids share one lookup per id."""
    reader = _FakeTombstoneReader(_incident_tombstones())

    await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [_deletion_flag(), _deletion_flag()], events=[],
    )

    looked_up = [memory_id for _, memory_id in reader.calls]
    assert len(looked_up) == len(set(looked_up)), (
        f'every distinct id must be looked up exactly once; got {looked_up!r}'
    )
    assert {_SWEPT_A, _SWEPT_B, _SWEPT_C} <= set(looked_up)
    assert {project for project, _ in reader.calls} == {_PROJECT}


@pytest.mark.asyncio
async def test_input_list_is_not_mutated_and_order_is_preserved():
    """(13) A new list, survivors in input order, input list untouched."""
    reader = _FakeTombstoneReader({
        **_incident_tombstones(),
        _UNTOMBSTONED: None,
    })
    first = {'task_id': '1', 'flag_type': 'stale_metadata', 'description': 'a'}
    benign = _deletion_flag()
    middle = {'task_id': '2', 'flag_type': 'missing_deliverable', 'description': 'b'}
    unexplained = _deletion_flag((_UNTOMBSTONED,))
    flags = [first, benign, middle, unexplained]
    identities = [id(f) for f in flags]

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, flags, events=[_deleted(_UNTOMBSTONED)],
    )

    assert result == [first, middle, unexplained]
    assert result is not flags
    assert [id(f) for f in flags] == identities, 'the input list must not be mutated'


@pytest.mark.asyncio
async def test_cascaded_child_deletions_count_as_swept():
    """A cascaded child's deletion is visible through cascaded_child_ids."""
    reader = _FakeTombstoneReader({})
    flag = _deletion_flag((_UNTOMBSTONED,))

    result = await filter_benign_sweep_deletion_flags(
        reader, _PROJECT, [flag], events=[_deleted(_SWEPT_A, cascaded=[_UNTOMBSTONED])],
    )

    assert result == [flag]
    assert flag['sweep_deletion_provenance']['decision'] == 'kept_unexplained_deletion'
    assert _swept_by_id(flag)[_UNTOMBSTONED]['classification'] == 'untombstoned'
