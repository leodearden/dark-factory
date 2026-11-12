"""The lane-buffered requests a deep chain holds for its tip verify (task 7720).

Unit tests for ``merge_lane/chain_reservation.py::ChainReservations`` driven by
a hand-advanced clock and a recording release callback. The contract lives in
``plans/deep-merge-ahead-prd.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from _merge_worker_drive import make_merge_request

from orchestrator.config import OrchestratorConfig
from orchestrator.merge_lane.chain_reservation import ChainReservation, ChainReservations
from orchestrator.merge_lane.types import MergeRequest


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Releases:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


@pytest.fixture
def releases() -> _Releases:
    return _Releases()


@pytest.fixture
def reservations(clock: _Clock, releases: _Releases) -> ChainReservations:
    return ChainReservations(monotonic=clock, on_release=releases)


@pytest.fixture
def config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


def _request(task_id: str, tmp_path: Path, config: OrchestratorConfig) -> MergeRequest:
    return make_merge_request(task_id, f'task/{task_id}', tmp_path, config)


@pytest.fixture
def requests(tmp_path: Path, config: OrchestratorConfig) -> dict[str, MergeRequest]:
    return {
        name: _request(name, tmp_path, config)
        for name in ('S', 'F1', 'F2', 'F3', 'O', 'X')
    }


def test_reserve_candidates_holds_only_the_candidates(reservations, requests) -> None:
    owner = requests['S']
    links = [requests['F1'], requests['F2'], requests['F3']]
    reservations.reserve_candidates(owner, links)
    for link in links:
        assert reservations.holds(link.request_id)
    assert not reservations.holds(owner.request_id)
    assert not reservations.holds(requests['X'].request_id)


def test_confirm_narrows_to_the_built_links(reservations, releases, requests) -> None:
    owner = requests['S']
    reservations.reserve_candidates(owner, [requests['F1'], requests['F2'], requests['F3']])
    reservations.confirm(
        owner.request_id,
        link_request_ids={requests['F1'].request_id, requests['F2'].request_id},
        tip='T',
        timeout_secs=100,
    )
    assert reservations.holds(requests['F1'].request_id)
    assert reservations.holds(requests['F2'].request_id)
    assert not reservations.holds(requests['F3'].request_id)
    assert releases.calls == 1


def test_release_returns_the_record_once(reservations, releases, clock, requests) -> None:
    owner = requests['S']
    reservations.reserve_candidates(owner, [requests['F1'], requests['F2']])
    reservations.confirm(
        owner.request_id,
        link_request_ids={requests['F1'].request_id, requests['F2'].request_id},
        tip='T',
        timeout_secs=100,
    )
    calls_before = releases.calls

    record = reservations.release(owner.request_id)

    assert isinstance(record, ChainReservation)
    assert record.owner_request_id == owner.request_id
    assert record.owner_task_id == 'S'
    assert record.link_task_ids == ('F1', 'F2')
    assert record.tip == 'T'
    assert record.deadline == clock.now + 100
    assert releases.calls == calls_before + 1
    assert not reservations.holds(requests['F1'].request_id)

    assert reservations.release(owner.request_id) is None
    assert releases.calls == calls_before + 1


def test_release_of_an_unknown_owner_fires_nothing(reservations, releases) -> None:
    assert reservations.release('mr-nobody') is None
    assert releases.calls == 0


def test_expire_releases_only_past_the_deadline(
    reservations, releases, clock, requests, caplog
) -> None:
    owner = requests['S']
    reservations.reserve_candidates(owner, [requests['F1'], requests['F2']])
    reservations.confirm(
        owner.request_id,
        link_request_ids={requests['F1'].request_id, requests['F2'].request_id},
        tip='T',
        timeout_secs=100,
    )
    calls_before = releases.calls
    clock.now += 99
    assert reservations.expire() == []
    assert releases.calls == calls_before

    clock.now += 2
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
        expired = reservations.expire()

    assert [record.owner_task_id for record in expired] == ['S']
    assert expired[0].link_task_ids == ('F1', 'F2')
    assert not reservations.holds(requests['F1'].request_id)
    assert releases.calls == calls_before + 1
    warnings = [
        rec for rec in caplog.records
        if rec.name == 'orchestrator.merge_lane' and rec.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    for task_id in ('S', 'F1', 'F2'):
        assert task_id in message
    assert reservations.expire() == []


def test_a_provisional_reservation_never_expires(reservations, clock, requests) -> None:
    owner = requests['S']
    reservations.reserve_candidates(owner, [requests['F1']])
    clock.now += 1e9
    assert reservations.expire() == []
    assert reservations.holds(requests['F1'].request_id)


def test_double_reservation_is_refused_naming_both_owners(reservations, requests) -> None:
    reservations.reserve_candidates(requests['S'], [requests['F1'], requests['F2']])
    with pytest.raises(ValueError) as excinfo:
        reservations.reserve_candidates(requests['O'], [requests['F3'], requests['F2']])
    message = str(excinfo.value)
    assert requests['F2'].request_id in message
    assert "by task S's chain" in message
    assert "task O's chain cannot" in message
    assert [r['owner_task_id'] for r in reservations.snapshot()] == ['S']
    assert not reservations.holds(requests['F3'].request_id)


def test_snapshot_lists_every_live_reservation(reservations, clock, requests) -> None:
    reservations.reserve_candidates(requests['S'], [requests['F1'], requests['F2']])
    reservations.reserve_candidates(requests['O'], [requests['F3']])
    reservations.confirm(
        requests['S'].request_id,
        link_request_ids={requests['F1'].request_id, requests['F2'].request_id},
        tip='T',
        timeout_secs=100,
    )
    clock.now += 40

    rows = sorted(reservations.snapshot(), key=lambda row: row['owner_task_id'])

    assert rows == [
        {'owner_task_id': 'O', 'link_task_ids': ['F3'], 'tip': None, 'seconds_left': None},
        {'owner_task_id': 'S', 'link_task_ids': ['F1', 'F2'], 'tip': 'T', 'seconds_left': 60.0},
    ]


def test_clear_drops_everything_quietly(reservations, clock, requests, caplog) -> None:
    reservations.reserve_candidates(requests['S'], [requests['F1']])
    reservations.confirm(
        requests['S'].request_id,
        link_request_ids={requests['F1'].request_id},
        tip='T',
        timeout_secs=1,
    )
    clock.now += 10
    with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
        reservations.clear()
    assert reservations.snapshot() == []
    assert not reservations.holds(requests['F1'].request_id)
    assert reservations.expire() == []
    assert not [rec for rec in caplog.records if rec.levelno >= logging.WARNING]
