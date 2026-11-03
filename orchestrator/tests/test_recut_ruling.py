"""The re-cut ruling marker contract (orchestrator.recut_ruling), task 3154."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from orchestrator.recut_ruling import (
    RECUT_EXECUTED_KEY,
    RECUT_RULING_KEY,
    GateVerdict,
    ParkedIncarnation,
    RecutRuling,
    RecutRulingUnreadable,
    RecutRulingViolation,
    RecutState,
    carries_recut_ruling,
    enforce_recut_ruling,
    execution_record,
    read_recut_ruling,
    ruling_to_enforce,
)

RULING_ID = 'esc-2896-3/D11-A'
RULED_AT = '2026-08-30T00:00:00Z'
RULING = RecutRuling(ruling_id=RULING_ID, ruled_at=datetime(2026, 8, 30, tzinfo=UTC))
EXECUTED_AT = datetime(2026, 9, 1, 12, 30, tzinfo=UTC)
PARKED = ParkedIncarnation(
    parked_branch='task/2896-20260901T123000Z',
    parked_tip='a' * 40,
    quarantined_worktree='/pool/.worktrees-quarantine/2896-20260901T123000Z',
    relocated_meta_roots=('/pool/.task-meta/_lane-3',),
)


def _task(metadata: Any = None) -> dict[str, Any]:
    task: dict[str, Any] = {'id': '2896'}
    if metadata is not None:
        task['metadata'] = metadata
    return task


def _ruled(**extra: Any) -> dict[str, Any]:
    return {
        RECUT_RULING_KEY: {'ruling_id': RULING_ID, 'ruled_at': RULED_AT},
        **extra,
    }


class TestCarriesRecutRuling:
    def test_no_metadata_is_not_carried(self):
        assert carries_recut_ruling(_task()) is False

    def test_metadata_without_key_is_not_carried(self):
        assert carries_recut_ruling(_task({'files': ['a.py']})) is False

    @pytest.mark.parametrize('value', [
        {'ruling_id': RULING_ID, 'ruled_at': RULED_AT}, 'garbage', None, 7,
    ])
    def test_key_presence_carries_any_value(self, value: object):
        assert carries_recut_ruling(_task({RECUT_RULING_KEY: value})) is True

    def test_json_string_metadata_is_read(self):
        assert carries_recut_ruling(_task(json.dumps(_ruled()))) is True

    def test_unreadable_blob_is_carried_so_the_gate_can_skip_loudly(self):
        assert carries_recut_ruling(_task('not json')) is True


class TestReadRecutRuling:
    def test_absent_is_silent(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger='orchestrator.recut_ruling'):
            reading = read_recut_ruling(_task({'files': []}))
        assert reading.state is RecutState.ABSENT
        assert reading.ruling is None
        assert caplog.records == []

    def test_unreadable_blob_skips_with_warning(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger='orchestrator.recut_ruling'):
            reading = read_recut_ruling(_task('not json'))
        assert reading.state is RecutState.SKIP
        assert '2896' in caplog.text
        assert 'no ruling was read' in caplog.text

    @pytest.mark.parametrize('metadata', [_ruled(), json.dumps(_ruled())],
                             ids=['dict', 'json-string'])
    def test_pending(self, metadata: object):
        reading = read_recut_ruling(_task(metadata))
        assert reading.state is RecutState.PENDING
        assert reading.ruling == RULING

    @pytest.mark.parametrize('ruled_at', ['2026-08-30T00:00:00', '2026-08-30'],
                             ids=['naive-is-utc', 'date-only'])
    def test_ruled_at_shapes(self, ruled_at: str):
        reading = read_recut_ruling(
            _task({RECUT_RULING_KEY: {'ruling_id': RULING_ID, 'ruled_at': ruled_at}})
        )
        assert reading.state is RecutState.PENDING
        assert reading.ruling == RULING

    def test_satisfied_by_matching_execution_record(self):
        reading = read_recut_ruling(
            _task(_ruled(**{RECUT_EXECUTED_KEY: {'ruling_id': RULING_ID}}))
        )
        assert reading.state is RecutState.SATISFIED
        assert reading.ruling == RULING

    def test_newer_ruling_is_pending_again(self):
        reading = read_recut_ruling(
            _task(_ruled(**{RECUT_EXECUTED_KEY: {'ruling_id': 'esc-2896-1/OLD'}}))
        )
        assert reading.state is RecutState.PENDING

    @pytest.mark.parametrize(('metadata', 'field'), [
        pytest.param({RECUT_RULING_KEY: 'esc-2896-3'}, RECUT_RULING_KEY, id='non-dict'),
        pytest.param({RECUT_RULING_KEY: {'ruled_at': RULED_AT}}, 'ruling_id',
                     id='missing-ruling-id'),
        pytest.param({RECUT_RULING_KEY: {'ruling_id': '  ', 'ruled_at': RULED_AT}},
                     'ruling_id', id='blank-ruling-id'),
        pytest.param({RECUT_RULING_KEY: {'ruling_id': 3, 'ruled_at': RULED_AT}},
                     'ruling_id', id='non-str-ruling-id'),
        pytest.param({RECUT_RULING_KEY: {'ruling_id': RULING_ID}}, 'ruled_at',
                     id='missing-ruled-at'),
        pytest.param({RECUT_RULING_KEY: {'ruling_id': RULING_ID, 'ruled_at': 'last week'}},
                     'ruled_at', id='unparseable-ruled-at'),
        pytest.param(_ruled(**{RECUT_EXECUTED_KEY: 'done'}), RECUT_EXECUTED_KEY,
                     id='non-dict-executed'),
        pytest.param(_ruled(**{RECUT_EXECUTED_KEY: {'executed_at': RULED_AT}}),
                     RECUT_EXECUTED_KEY, id='executed-without-ruling-id'),
        pytest.param(_ruled(**{RECUT_EXECUTED_KEY: {'ruling_id': 9}}),
                     RECUT_EXECUTED_KEY, id='executed-non-str-ruling-id'),
    ])
    def test_malformed_names_the_field(self, metadata: dict[str, Any], field: str):
        reading = read_recut_ruling(_task(metadata))
        assert reading.state is RecutState.MALFORMED
        assert field in reading.problem


class TestRulingToEnforce:
    def test_pending_returns_the_ruling(self):
        assert ruling_to_enforce(_task(_ruled())) == RULING

    @pytest.mark.parametrize('metadata', [
        pytest.param(None, id='absent'),
        pytest.param(_ruled(**{RECUT_EXECUTED_KEY: {'ruling_id': RULING_ID}}),
                     id='satisfied'),
        pytest.param('not json', id='skip'),
    ])
    def test_nothing_to_enforce(self, metadata: object):
        assert ruling_to_enforce(_task(metadata)) is None

    def test_malformed_raises(self):
        with pytest.raises(RecutRulingUnreadable, match='ruled_at'):
            ruling_to_enforce(_task({RECUT_RULING_KEY: {'ruling_id': RULING_ID}}))


class TestExecutionRecord:
    @pytest.mark.parametrize('parked', [PARKED, None], ids=['parked', 'nothing-parked'])
    def test_record_satisfies_the_ruling(self, parked: ParkedIncarnation | None):
        record = execution_record(RULING, parked, EXECUTED_AT)
        metadata = json.loads(json.dumps({**_ruled(), RECUT_EXECUTED_KEY: record}))
        assert read_recut_ruling(_task(metadata)).state is RecutState.SATISFIED

    def test_record_fields(self):
        record = execution_record(RULING, PARKED, EXECUTED_AT)
        assert record == {
            'ruling_id': RULING_ID,
            'ruled_at': RULING.ruled_at.isoformat(),
            'executed_at': EXECUTED_AT.isoformat(),
            'parked_branch': PARKED.parked_branch,
            'parked_tip': PARKED.parked_tip,
            'quarantined_worktree': PARKED.quarantined_worktree,
            'relocated_meta_roots': list(PARKED.relocated_meta_roots),
        }

    def test_record_with_nothing_parked(self):
        record = execution_record(RULING, None, EXECUTED_AT)
        assert record['parked_branch'] is None
        assert record['parked_tip'] is None
        assert record['quarantined_worktree'] is None
        assert record['relocated_meta_roots'] == []


class TestRecutRulingViolation:
    def test_is_a_runtime_error_naming_ruling_and_evidence(self):
        evidence = ('branch task/2896 exists', 'lane _lane-3 holds task/2896')
        err = RecutRulingViolation(RULING, evidence)
        assert isinstance(err, RuntimeError)
        assert err.ruling == RULING
        assert err.evidence == evidence
        text = str(err)
        assert RULING_ID in text
        assert RULING.ruled_at.isoformat() in text
        for item in evidence:
            assert item in text


class _RecordingOps:
    """A PriorIncarnationOps that records each call into a shared list."""

    def __init__(
        self,
        calls: list[tuple[Any, ...]],
        *,
        parked: ParkedIncarnation | None = PARKED,
        park_error: Exception | None = None,
        evidence: tuple[str, ...] = (),
    ) -> None:
        self.calls = calls
        self.parked = parked
        self.park_error = park_error
        self.evidence = evidence

    async def park_prior_incarnation(
        self, branch_name: str, *, reason: str,
    ) -> ParkedIncarnation | None:
        self.calls.append(('park', branch_name, reason))
        if self.park_error is not None:
            raise self.park_error
        return self.parked

    async def prior_incarnation_evidence(self, branch_name: str) -> tuple[str, ...]:
        self.calls.append(('evidence', branch_name))
        return self.evidence


def _persist(
    calls: list[tuple[Any, ...]], *, result: bool = True, error: Exception | None = None,
) -> AsyncMock:
    async def _record(patch: dict[str, Any]) -> bool:
        calls.append(('persist', patch))
        if error is not None:
            raise error
        return result

    return AsyncMock(side_effect=_record)


@pytest.mark.asyncio
class TestEnforceRecutRuling:
    @pytest.mark.parametrize('metadata', [
        pytest.param(None, id='absent'),
        pytest.param('not json', id='skip'),
        pytest.param(_ruled(**{RECUT_EXECUTED_KEY: {'ruling_id': RULING_ID}}),
                     id='satisfied'),
    ])
    async def test_nothing_to_execute_proceeds(self, metadata: object):
        calls: list[tuple[Any, ...]] = []
        result = await enforce_recut_ruling(
            _task(metadata), ops=_RecordingOps(calls), persist=_persist(calls),
            now=EXECUTED_AT,
        )
        assert result.verdict is GateVerdict.PROCEED
        assert calls == []

    async def test_malformed_refuses_without_touching_anything(self):
        calls: list[tuple[Any, ...]] = []
        result = await enforce_recut_ruling(
            _task({RECUT_RULING_KEY: {'ruling_id': RULING_ID}}),
            ops=_RecordingOps(calls), persist=_persist(calls), now=EXECUTED_AT,
        )
        assert result.verdict is GateVerdict.REFUSED and result.refused
        assert 'ruled_at' in result.reason
        assert calls == []

    async def test_pending_parks_then_verifies_then_persists(self):
        calls: list[tuple[Any, ...]] = []
        result = await enforce_recut_ruling(
            _task(_ruled()), ops=_RecordingOps(calls), persist=_persist(calls),
            now=EXECUTED_AT,
        )
        record = execution_record(RULING, PARKED, EXECUTED_AT)
        assert [call[0] for call in calls] == ['park', 'evidence', 'persist']
        assert calls[0][1] == '2896' and RULING_ID in calls[0][2]
        assert calls[1] == ('evidence', '2896')
        assert calls[2] == ('persist', {RECUT_EXECUTED_KEY: record})
        merged = _task({**_ruled(), **calls[2][1]})
        assert read_recut_ruling(merged).state is RecutState.SATISFIED
        assert result.verdict is GateVerdict.EXECUTED and result.executed
        assert result.parked == PARKED
        assert result.ruling == RULING

    async def test_integer_task_id_is_the_bare_branch_name(self):
        calls: list[tuple[Any, ...]] = []
        await enforce_recut_ruling(
            {'id': 2896, 'metadata': _ruled()}, ops=_RecordingOps(calls),
            persist=_persist(calls), now=EXECUTED_AT,
        )
        assert calls[0][1] == '2896'

    async def test_nothing_parked_still_records_execution(self):
        calls: list[tuple[Any, ...]] = []
        result = await enforce_recut_ruling(
            _task(_ruled()), ops=_RecordingOps(calls, parked=None),
            persist=_persist(calls), now=EXECUTED_AT,
        )
        assert calls[-1] == (
            'persist', {RECUT_EXECUTED_KEY: execution_record(RULING, None, EXECUTED_AT)},
        )
        assert result.verdict is GateVerdict.EXECUTED
        assert result.parked is None

    async def test_park_failure_refuses_and_records_nothing(self):
        calls: list[tuple[Any, ...]] = []
        result = await enforce_recut_ruling(
            _task(_ruled()),
            ops=_RecordingOps(calls, park_error=RuntimeError('worktree move failed')),
            persist=_persist(calls), now=EXECUTED_AT,
        )
        assert result.verdict is GateVerdict.REFUSED
        assert RULING_ID in result.reason and 'worktree move failed' in result.reason
        assert [call[0] for call in calls] == ['park']

    async def test_surviving_evidence_refuses_and_records_nothing(self):
        calls: list[tuple[Any, ...]] = []
        evidence = ('branch task/2896 exists at abc', 'lane /pool/_lane-1 is assigned to 2896')
        result = await enforce_recut_ruling(
            _task(_ruled()), ops=_RecordingOps(calls, evidence=evidence),
            persist=_persist(calls), now=EXECUTED_AT,
        )
        assert result.verdict is GateVerdict.REFUSED
        for item in evidence:
            assert item in result.reason
        assert [call[0] for call in calls] == ['park', 'evidence']

    @pytest.mark.parametrize(('result', 'error'), [
        pytest.param(False, None, id='returns-false'),
        pytest.param(True, RuntimeError('store down'), id='raises'),
    ])
    async def test_persist_failure_refuses_naming_the_parked_ref(
        self, result: bool, error: Exception | None,
    ):
        calls: list[tuple[Any, ...]] = []
        gate = await enforce_recut_ruling(
            _task(_ruled()), ops=_RecordingOps(calls),
            persist=_persist(calls, result=result, error=error), now=EXECUTED_AT,
        )
        assert gate.verdict is GateVerdict.REFUSED
        assert PARKED.parked_branch is not None
        assert PARKED.parked_branch in gate.reason
        assert 're-dispatch is safe' in gate.reason
