"""A lease body whose fields do not carry LeaseHolder's declared types is UNREADABLE.

Every lease reader (lease_status, reap_stale_leases, claim_lease, the
``lease-claim`` CLI) treats such a body as an unknown holder -- never an
exception, and never a fake holder.
"""

from __future__ import annotations

import dataclasses
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator import session_registry as sr

_NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)
_DEAD_PID = 2**31 - 1
_PAST_TTL = sr.LEASE_HEARTBEAT_TTL + timedelta(minutes=1)
_WITHIN_TTL = sr.LEASE_HEARTBEAT_TTL - timedelta(minutes=1)

# `True` is deliberate: bool IS an int in Python, so without a type check a
# JSON `true` pid reads as live pid 1 (init) and the lease can never be reaped.
_NON_INT_PIDS = pytest.mark.parametrize(
    'bad',
    ['1234', None, 12.0, True, [1234]],
    ids=['str', 'null', 'float', 'bool', 'list'],
)


def _seed_raw_lease(root: Path, name: str, body: object, *, age: timedelta) -> Path:
    path = sr.lease_path_for_name(name, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body), encoding='utf-8')
    ts = (_NOW - age).timestamp()
    os.utime(path, (ts, ts))
    return path


def _body(pid: object) -> dict[str, object]:
    return {'session_slug': 'watcher-df-100', 'pid': pid, 'start_ts': _NOW.isoformat()}


def _claimant() -> sr.LeaseHolder:
    return sr.LeaseHolder(session_slug='watcher-df-200', pid=os.getpid(), start_ts=_NOW.isoformat())


@_NON_INT_PIDS
def test_from_dict_rejects_a_non_int_pid(bad: object) -> None:
    with pytest.raises(TypeError, match='pid') as excinfo:
        sr.LeaseHolder.from_dict(_body(bad))
    assert repr(bad) in str(excinfo.value)


@_NON_INT_PIDS
def test_constructor_rejects_a_non_int_pid(bad: object) -> None:
    with pytest.raises(TypeError, match='pid') as excinfo:
        dataclasses.replace(_claimant(), pid=bad)
    assert repr(bad) in str(excinfo.value)


@pytest.mark.parametrize('pid', [0, -1, os.getpid(), 2**70], ids=['zero', 'negative', 'self', 'oversized'])
def test_any_int_pid_is_accepted_and_round_trips(pid: int) -> None:
    holder = sr.LeaseHolder(session_slug='watcher-df-100', pid=pid, start_ts=_NOW.isoformat())
    assert sr.LeaseHolder.from_json(holder.to_json()) == holder


@_NON_INT_PIDS
def test_lease_status_reports_a_non_int_pid_body_as_unreadable(tmp_path: Path, bad: object) -> None:
    _seed_raw_lease(tmp_path, 'watcher-df', _body(bad), age=timedelta(minutes=42))

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.state == 'unreadable'
    assert status.holder_slug is None
    assert status.holder_pid is None
    assert status.holder_pid_alive is False
    assert status.holder_record_state is sr.HolderRecordState.UNLINKED
    assert int(status.heartbeat_age_secs) == 2520


@_NON_INT_PIDS
def test_reap_stale_leases_reaps_a_non_int_pid_body_and_finishes_the_sweep(
    tmp_path: Path, bad: object
) -> None:
    _seed_raw_lease(tmp_path, 'a-bad-pid', _body(bad), age=_PAST_TTL)
    _seed_raw_lease(tmp_path, 'b-bad-pid-fresh', _body(bad), age=_WITHIN_TTL)
    _seed_raw_lease(tmp_path, 'z-dead-stale', _body(_DEAD_PID), age=_PAST_TTL)

    reaped = sr.reap_stale_leases(root=tmp_path, now=_NOW)

    assert {r.lease_name: r.reason for r in reaped} == {
        'a-bad-pid': 'corrupt',
        'z-dead-stale': 'stale_pid',
    }
    remaining = {p.stem for p in sr.leases_dir(root=tmp_path).glob('*.lease')}
    assert remaining == {'b-bad-pid-fresh'}


@_NON_INT_PIDS
def test_claim_lease_within_ttl_on_a_non_int_pid_body_stands_down_on_an_unknown_holder(
    tmp_path: Path, bad: object
) -> None:
    path = _seed_raw_lease(tmp_path, 'watcher-df', _body(bad), age=_WITHIN_TTL)
    seeded = path.read_text(encoding='utf-8')

    claim = sr.claim_lease(
        'watcher-df', holder=_claimant(), policy=sr.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.acquired is False
    assert claim.decision is sr.LeaseDecision.STAND_DOWN
    assert claim.holder is None
    assert claim.holder_alive is False
    assert '<unknown>' in claim.message
    assert path.read_text(encoding='utf-8') == seeded


@_NON_INT_PIDS
def test_claim_lease_past_ttl_reclaims_a_non_int_pid_body(tmp_path: Path, bad: object) -> None:
    path = _seed_raw_lease(tmp_path, 'watcher-df', _body(bad), age=_PAST_TTL)
    claimant = _claimant()

    claim = sr.claim_lease(
        'watcher-df', holder=claimant, policy=sr.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.acquired is True
    assert sr.LeaseHolder.from_json(path.read_text(encoding='utf-8')) == claimant


@pytest.mark.parametrize('bad', ['1234', True], ids=['str', 'bool'])
def test_main_lease_claim_on_a_non_int_pid_body_stands_down_instead_of_failing_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad: object,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    path = _seed_raw_lease(tmp_path, 'watcher-df', _body(bad), age=timedelta(0))
    os.utime(path, None)

    rc = sr.main(
        ['lease-claim', '--name', 'watcher-df', '--slug', 'watcher-df-200', '--pid', str(os.getpid())]
    )

    assert rc == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == 'decision=stand-down'
    assert '<unknown>' in lines[1]
    assert lines[2] == 'holder_liveness=held'


_NON_STR_FIELDS = pytest.mark.parametrize(
    ('field', 'bad'),
    [('session_slug', 7), ('start_ts', 3), ('record_slug', 5)],
    ids=['session_slug', 'start_ts', 'record_slug'],
)


def _body_with(field: str, bad: object) -> dict[str, object]:
    return {**_body(os.getpid()), field: bad}


@_NON_STR_FIELDS
def test_from_dict_rejects_a_non_str_field(field: str, bad: object) -> None:
    with pytest.raises(TypeError, match=field) as excinfo:
        sr.LeaseHolder.from_dict(_body_with(field, bad))
    assert repr(bad) in str(excinfo.value)


@_NON_STR_FIELDS
def test_lease_status_reports_a_non_str_field_body_as_unreadable(
    tmp_path: Path, field: str, bad: object
) -> None:
    _seed_raw_lease(tmp_path, 'watcher-df', _body_with(field, bad), age=timedelta(minutes=5))

    status = sr.lease_status('watcher-df', root=tmp_path, now=_NOW)

    assert status.state == 'unreadable'
    assert status.holder_slug is None
    assert status.holder_record_slug is None
    assert status.holder_record_state is sr.HolderRecordState.UNLINKED


def test_claim_lease_within_ttl_on_a_non_int_record_slug_body_stands_down(tmp_path: Path) -> None:
    _seed_raw_lease(tmp_path, 'watcher-df', _body_with('record_slug', 5), age=_WITHIN_TTL)

    claim = sr.claim_lease(
        'watcher-df', holder=_claimant(), policy=sr.LeasePolicy.STAND_DOWN, root=tmp_path, now=_NOW
    )

    assert claim.decision is sr.LeaseDecision.STAND_DOWN
    assert claim.holder is None
    assert '<unknown>' in claim.message


@pytest.mark.parametrize('present', [True, False], ids=['null', 'missing'])
def test_falsy_record_slug_still_reads_as_unlinked(present: bool) -> None:
    body = _body(os.getpid())
    if present:
        body['record_slug'] = None

    assert sr.LeaseHolder.from_dict(body).record_slug == ''
