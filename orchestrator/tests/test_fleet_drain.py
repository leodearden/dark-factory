"""Tests for orchestrator.fleet_drain -- the restart-drain request contract (task 5371).

The request file is written by ``scripts/restart-all-orchestrators.sh --drain``
and read by every orchestrator's merge-heartbeat pass; this module is the
reader, the honour decision, and the one-``fleet_drain``-event-per-request
tracker.  Every refusal reason, the honoured case, the absent file, and the
three event outcomes are pinned through the module's public functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.fleet_drain import (
    DrainEventTracker,
    DrainIdentity,
    DrainOutcome,
    DrainParticipant,
    DrainPass,
    DrainRefusal,
    DrainRequest,
    FleetDrainEvent,
    MalformedDrainRequest,
    drain_request_path,
    evaluate_drain_request,
    read_drain_request,
    sweep_alive,
)

UNIT = 'orchestrator-dark-factory.service'
INVOCATION = 'caaef715c93445dbb0c1cc608a9f4ef8'
REQUESTED_TS = 1_791_240_000
SWEEP_PID = 12345
IDENTITY = DrainIdentity(unit=UNIT, invocation_id=INVOCATION, max_age_secs=14_400.0)


def _alive(_pid: int) -> bool:
    return True


def _dead(_pid: int) -> bool:
    return False


def _request_fields(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        'unit': UNIT,
        'invocation_id': INVOCATION,
        'sweep_pid': SWEEP_PID,
        'requested_ts': REQUESTED_TS,
    }
    fields.update(overrides)
    return fields


def _write_request(tmp_path: Path, payload: object) -> Path:
    path = drain_request_path(tmp_path, UNIT)
    path.write_text(json.dumps(payload))
    return path


def _verdict(tmp_path: Path, payload: object, *, now: float = REQUESTED_TS + 30.0,
             identity: DrainIdentity = IDENTITY, pid_alive=_alive):
    read = read_drain_request(_write_request(tmp_path, payload))
    return evaluate_drain_request(read, identity, now=now, pid_alive=pid_alive)


def _verify(task_id: str, started_ts: float = REQUESTED_TS - 100.0) -> dict[str, object]:
    return {
        'task_id': task_id, 'host': 'local', 'kind': 'verify',
        'started_ts': started_ts, 'deadline_ts': started_ts + 10_800.0,
    }


class TestDrainRequestPath:
    def test_is_unit_dot_drain_json_in_the_fleet_dir(self, tmp_path: Path) -> None:
        assert drain_request_path(tmp_path, UNIT) == tmp_path / f'{UNIT}.drain.json'

    @pytest.mark.parametrize('unit', ['', '   ', '../escape', '/abs/unit', 'a/b', '..'])
    def test_refuses_a_unit_that_is_not_a_bare_file_name(self, tmp_path: Path, unit: str) -> None:
        with pytest.raises(ValueError, match='drain request'):
            drain_request_path(tmp_path, unit)


class TestReadDrainRequest:
    def test_absent_file_is_no_request(self, tmp_path: Path) -> None:
        assert read_drain_request(drain_request_path(tmp_path, UNIT)) is None

    def test_well_formed_file_reads_as_the_record(self, tmp_path: Path) -> None:
        read = read_drain_request(_write_request(tmp_path, _request_fields()))
        assert read == DrainRequest(
            unit=UNIT, invocation_id=INVOCATION, sweep_pid=SWEEP_PID, requested_ts=REQUESTED_TS,
        )

    @pytest.mark.parametrize('payload', [
        ['not', 'an', 'object'],
        _request_fields(sweep_pid='12345'),
        _request_fields(sweep_pid=True),
        _request_fields(sweep_pid=0),
        _request_fields(unit=7),
        {k: v for k, v in _request_fields().items() if k != 'invocation_id'},
        _request_fields(extra='key'),
    ])
    def test_wrong_shape_is_malformed_and_echoes_a_parseable_requested_ts(
        self, tmp_path: Path, payload: object,
    ) -> None:
        read = read_drain_request(_write_request(tmp_path, payload))
        assert isinstance(read, MalformedDrainRequest)
        assert read.refusal is DrainRefusal.MALFORMED
        expected_ts = REQUESTED_TS if isinstance(payload, dict) else None
        assert read.requested_ts == expected_ts
        assert read.problem

    @pytest.mark.parametrize('text', ['{not json', '', '\xff'])
    def test_unparseable_text_is_malformed_with_no_requested_ts(
        self, tmp_path: Path, text: str,
    ) -> None:
        path = drain_request_path(tmp_path, UNIT)
        path.write_text(text, encoding='latin-1')
        read = read_drain_request(path)
        assert isinstance(read, MalformedDrainRequest)
        assert read.requested_ts is None

    def test_float_requested_ts_is_malformed_and_not_echoed(self, tmp_path: Path) -> None:
        read = read_drain_request(_write_request(tmp_path, _request_fields(requested_ts=1.5)))
        assert isinstance(read, MalformedDrainRequest)
        assert read.requested_ts is None


class TestEvaluateDrainRequest:
    def test_absent_file_is_no_verdict(self) -> None:
        assert evaluate_drain_request(None, IDENTITY, now=0.0, pid_alive=_alive) is None

    def test_a_request_for_this_incarnation_is_honoured(self, tmp_path: Path) -> None:
        verdict = _verdict(tmp_path, _request_fields())
        assert verdict is not None
        assert verdict.honoured
        assert verdict.refused is None
        assert verdict.requested_ts == REQUESTED_TS
        assert verdict.heartbeat_block(admission_halted=True) == {
            'requested_ts': REQUESTED_TS, 'admission_halted': True, 'refused': None,
        }

    def test_malformed(self, tmp_path: Path) -> None:
        verdict = _verdict(tmp_path, _request_fields(sweep_pid='x'))
        assert verdict is not None and verdict.refused is DrainRefusal.MALFORMED
        assert verdict.heartbeat_block(admission_halted=False) == {
            'requested_ts': REQUESTED_TS, 'admission_halted': False, 'refused': 'malformed',
        }

    def test_unit_mismatch(self, tmp_path: Path) -> None:
        verdict = _verdict(tmp_path, _request_fields(unit='orchestrator-reify.service'))
        assert verdict is not None and verdict.refused is DrainRefusal.UNIT_MISMATCH

    def test_invocation_mismatch(self, tmp_path: Path) -> None:
        verdict = _verdict(tmp_path, _request_fields(invocation_id='an-older-incarnation'))
        assert verdict is not None and verdict.refused is DrainRefusal.INVOCATION_MISMATCH

    def test_an_unset_invocation_id_never_honours(self, tmp_path: Path) -> None:
        identity = DrainIdentity(unit=UNIT, invocation_id='', max_age_secs=14_400.0)
        verdict = _verdict(tmp_path, _request_fields(invocation_id=''), identity=identity)
        assert verdict is not None and verdict.refused is DrainRefusal.INVOCATION_MISMATCH

    def test_sweep_dead(self, tmp_path: Path) -> None:
        verdict = _verdict(tmp_path, _request_fields(), pid_alive=_dead)
        assert verdict is not None and verdict.refused is DrainRefusal.SWEEP_DEAD

    def test_expired_one_second_past_the_lease_bound(self, tmp_path: Path) -> None:
        at_bound = _verdict(tmp_path, _request_fields(), now=REQUESTED_TS + 14_400.0)
        past_bound = _verdict(tmp_path, _request_fields(), now=REQUESTED_TS + 14_401.0)
        assert at_bound is not None and at_bound.honoured
        assert past_bound is not None and past_bound.refused is DrainRefusal.EXPIRED

    def test_one_reason_only_in_contract_order(self, tmp_path: Path) -> None:
        verdict = _verdict(
            tmp_path, _request_fields(unit='other.service', invocation_id='other'),
            now=REQUESTED_TS + 1e9, pid_alive=_dead,
        )
        assert verdict is not None and verdict.refused is DrainRefusal.UNIT_MISMATCH

    def test_identity_reads_orch_unit_and_invocation_id(self) -> None:
        identity = DrainIdentity.from_environ(
            {'ORCH_UNIT': UNIT, 'INVOCATION_ID': INVOCATION}, max_age_secs=60.0,
        )
        assert identity == DrainIdentity(unit=UNIT, invocation_id=INVOCATION, max_age_secs=60.0)
        assert DrainIdentity.from_environ({}, max_age_secs=60.0).invocation_id == ''


class TestSweepAlive:
    def test_this_process_is_alive(self) -> None:
        import os
        assert sweep_alive(os.getpid()) is True

    def test_a_pid_beyond_the_kernel_range_is_dead(self) -> None:
        assert sweep_alive(2**62) is False


class TestDrainEventTracker:
    """Exactly one ``fleet_drain`` event per honoured request, keyed by requested_ts + sweep_pid."""

    def _honoured(self, tmp_path: Path, **overrides: object):
        verdict = _verdict(tmp_path, _request_fields(**overrides))
        assert verdict is not None and verdict.honoured
        return verdict

    def test_drained_once_with_the_union_of_awaited_verifies(self, tmp_path: Path) -> None:
        tracker = DrainEventTracker()
        verdict = self._honoured(tmp_path)
        assert tracker.observe(verdict, [_verify('a')], now=REQUESTED_TS + 10.0) == ()
        assert tracker.observe(verdict, [_verify('b')], now=REQUESTED_TS + 20.0) == ()

        (event,) = tracker.observe(verdict, [], now=REQUESTED_TS + 45.0)

        assert event.as_payload() == {
            'unit': UNIT, 'requested_ts': REQUESTED_TS, 'sweep_pid': SWEEP_PID,
            'waited_secs': 45.0, 'outcome': 'drained', 'refused': None,
            'merge_verifies_awaited': [_verify('a'), _verify('b')],
            'merge_verifies_killed': [],
        }
        assert tracker.observe(verdict, [], now=REQUESTED_TS + 60.0) == ()
        assert tracker.shutdown([], now=REQUESTED_TS + 70.0) == ()

    def test_verifies_killed_when_shutdown_arrives_mid_verify(self, tmp_path: Path) -> None:
        tracker = DrainEventTracker()
        verdict = self._honoured(tmp_path)
        tracker.observe(verdict, [_verify('a')], now=REQUESTED_TS + 10.0)

        (event,) = tracker.shutdown([_verify('a'), _verify('c')], now=REQUESTED_TS + 90.0)

        payload = event.as_payload()
        assert payload['outcome'] == DrainOutcome.VERIFIES_KILLED == 'verifies_killed'
        assert payload['waited_secs'] == 90.0
        assert payload['merge_verifies_killed'] == [_verify('a'), _verify('c')]
        assert payload['merge_verifies_awaited'] == [_verify('a'), _verify('c')]
        assert tracker.observe(verdict, [], now=REQUESTED_TS + 95.0) == ()

    def test_shutdown_with_nothing_in_flight_reports_drained(self, tmp_path: Path) -> None:
        tracker = DrainEventTracker()
        verdict = self._honoured(tmp_path)
        tracker.observe(verdict, [_verify('a')], now=REQUESTED_TS + 10.0)

        (event,) = tracker.shutdown([], now=REQUESTED_TS + 20.0)

        assert event.outcome is DrainOutcome.DRAINED

    def test_abandoned_with_the_refusal_when_the_request_stops_being_honoured(
        self, tmp_path: Path,
    ) -> None:
        tracker = DrainEventTracker()
        tracker.observe(self._honoured(tmp_path), [_verify('a')], now=REQUESTED_TS + 10.0)
        dead = _verdict(tmp_path, _request_fields(), pid_alive=_dead)

        (event,) = tracker.observe(dead, [_verify('a')], now=REQUESTED_TS + 40.0)

        assert event.outcome is DrainOutcome.ABANDONED
        assert event.as_payload()['refused'] == 'sweep_dead'
        assert event.as_payload()['merge_verifies_killed'] == []
        assert tracker.shutdown([_verify('a')], now=REQUESTED_TS + 50.0) == ()

    def test_abandoned_with_no_refusal_when_the_file_disappears(self, tmp_path: Path) -> None:
        tracker = DrainEventTracker()
        tracker.observe(self._honoured(tmp_path), [_verify('a')], now=REQUESTED_TS + 10.0)

        (event,) = tracker.observe(None, [_verify('a')], now=REQUESTED_TS + 40.0)

        assert event.outcome is DrainOutcome.ABANDONED
        assert event.as_payload()['refused'] is None

    def test_a_replacement_request_abandons_the_old_and_tracks_the_new(
        self, tmp_path: Path,
    ) -> None:
        tracker = DrainEventTracker()
        tracker.observe(self._honoured(tmp_path), [_verify('a')], now=REQUESTED_TS + 10.0)
        newer = self._honoured(tmp_path, requested_ts=REQUESTED_TS + 30, sweep_pid=999)

        abandoned, drained = tracker.observe(newer, [], now=REQUESTED_TS + 40.0)

        assert (abandoned.outcome, abandoned.requested_ts) == (DrainOutcome.ABANDONED, REQUESTED_TS)
        assert (drained.outcome, drained.requested_ts) == (DrainOutcome.DRAINED, REQUESTED_TS + 30)
        assert drained.waited_secs == 10.0

    def test_a_refused_request_never_emits(self, tmp_path: Path) -> None:
        tracker = DrainEventTracker()
        refused = _verdict(tmp_path, _request_fields(unit='other.service'))
        assert tracker.observe(refused, [], now=REQUESTED_TS + 1.0) == ()
        assert tracker.observe(None, [], now=REQUESTED_TS + 2.0) == ()
        assert tracker.shutdown([_verify('a')], now=REQUESTED_TS + 3.0) == ()


class _FakeLane:
    """An ``AdmissionPort`` whose snapshot reports what it was told to hold."""

    def __init__(
        self, *, in_flight: list[dict[str, object]] | None = None, depth: int = 0,
        obeys: bool = True,
    ) -> None:
        self.in_flight = in_flight or []
        self.depth = depth
        self.obeys = obeys
        self.halted = False
        self.halt_reasons: list[str] = []

    def halt_admission(self, reason: str) -> None:
        self.halt_reasons.append(reason)
        self.halted = self.obeys

    def resume_admission(self) -> None:
        self.halted = False

    def snapshot(self) -> dict[str, object]:
        return {
            'depth': self.depth,
            'restart_drain': {
                'admission_halted': self.halted, 'verifies_in_flight': list(self.in_flight),
            },
        }


@pytest.mark.asyncio
class TestDrainParticipant:
    """The unit side of the drain, driven through its two public calls."""

    NOW = REQUESTED_TS + 30.0

    def _participant(
        self, tmp_path: Path, lane: _FakeLane | None, events: list[FleetDrainEvent],
        *, pid_alive=_alive,
    ) -> DrainParticipant:
        return DrainParticipant(
            lane=lambda: lane,
            fleet_dir=lambda: tmp_path,
            identity=lambda: IDENTITY,
            emit=events.append,
            clock=lambda: self.NOW,
            pid_alive=pid_alive,
        )

    async def test_an_honoured_request_halts_admission_and_drains_once(self, tmp_path: Path) -> None:
        lane, events = _FakeLane(depth=3), []
        participant = self._participant(tmp_path, lane, events)
        _write_request(tmp_path, _request_fields())

        reading = await participant.on_heartbeat_pass()
        await participant.on_heartbeat_pass()

        assert reading == DrainPass(
            drain={'requested_ts': REQUESTED_TS, 'admission_halted': True, 'refused': None},
            verifies_in_flight=[], depth=3,
        )
        assert lane.halted is True
        assert lane.halt_reasons[0] == f'fleet restart drain requested_ts={REQUESTED_TS}'
        assert [(e.outcome, e.waited_secs) for e in events] == [(DrainOutcome.DRAINED, 30.0)]

    async def test_the_acknowledgement_is_read_back_from_the_lane(self, tmp_path: Path) -> None:
        lane = _FakeLane(obeys=False)
        _write_request(tmp_path, _request_fields())

        reading = await self._participant(tmp_path, lane, []).on_heartbeat_pass()

        assert reading.drain == {
            'requested_ts': REQUESTED_TS, 'admission_halted': False, 'refused': None,
        }

    async def test_a_refused_request_halts_nothing_and_emits_nothing(self, tmp_path: Path) -> None:
        lane, events = _FakeLane(), []
        _write_request(tmp_path, _request_fields(invocation_id='the-previous-incarnation'))

        reading = await self._participant(tmp_path, lane, events).on_heartbeat_pass()

        assert lane.halt_reasons == []
        assert reading.drain == {
            'requested_ts': REQUESTED_TS, 'admission_halted': False,
            'refused': 'invocation_mismatch',
        }
        assert events == []

    async def test_a_withdrawn_request_resumes_admission_on_the_next_pass(
        self, tmp_path: Path,
    ) -> None:
        lane, events = _FakeLane(in_flight=[_verify('a')]), []
        participant = self._participant(tmp_path, lane, events)
        path = _write_request(tmp_path, _request_fields())
        await participant.on_heartbeat_pass()
        assert lane.halted is True

        path.unlink()
        reading = await participant.on_heartbeat_pass()

        assert lane.halted is False
        assert reading.drain is None
        assert [(e.outcome, e.refused) for e in events] == [(DrainOutcome.ABANDONED, None)]

    async def test_with_no_lane_an_honoured_request_is_trivially_halted(
        self, tmp_path: Path,
    ) -> None:
        _write_request(tmp_path, _request_fields())

        reading = await self._participant(tmp_path, None, []).on_heartbeat_pass()

        assert reading == DrainPass(
            drain={'requested_ts': REQUESTED_TS, 'admission_halted': True, 'refused': None},
            verifies_in_flight=[], depth=None,
        )

    async def test_shutdown_mid_verify_reports_what_it_kills(self, tmp_path: Path) -> None:
        lane, events = _FakeLane(in_flight=[_verify('held')]), []
        participant = self._participant(tmp_path, lane, events)
        _write_request(tmp_path, _request_fields())
        reading = await participant.on_heartbeat_pass()
        assert reading.verifies_in_flight == [_verify('held')]

        participant.on_shutdown()

        (event,) = events
        assert event.outcome is DrainOutcome.VERIFIES_KILLED
        assert event.as_payload()['merge_verifies_killed'] == [_verify('held')]

    async def test_a_blank_unit_names_no_request(self, tmp_path: Path) -> None:
        participant = DrainParticipant(
            lane=lambda: None, fleet_dir=lambda: tmp_path,
            identity=lambda: DrainIdentity(unit='', invocation_id=INVOCATION, max_age_secs=1.0),
            emit=lambda _event: None,
        )
        with pytest.raises(ValueError, match='drain request'):
            await participant.on_heartbeat_pass()
