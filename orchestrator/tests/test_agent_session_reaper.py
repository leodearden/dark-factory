"""Boot reaper for zombie ``agent_session.json`` sidecars (task 3729, PRD γ/D7).

A sidecar is reaped only when its owner pid is dead AND its task's status is
known and is not 'in-progress' (plans/session-resume-eligibility-seam-prd.md
D7, B10).  The pure decision lives in ``orchestrator.agent_session_reaper``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.agent_session_reaper import (
    ReapTally,
    ReapVerdict,
    SessionSidecarClaim,
    owner_alive,
    read_sidecar_claim,
    reap_verdict,
)

_DEAD_PID = 2**31 - 1
_NOW = datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _v2_payload(**overrides: object) -> dict:
    payload: dict = {
        'session_id': 'sess-1',
        'role': 'implementer',
        'started_at': (_NOW - timedelta(seconds=3600)).isoformat(),
        'owner_pid': 4242,
        'task_id': '42',
        'resume_count': 0,
        'schema_version': 2,
    }
    payload.update(overrides)
    return payload


def _claim(owner_pid: object) -> SessionSidecarClaim:
    return SessionSidecarClaim(
        entry=Path('/wt/42'),
        sidecar_path=Path('/wt/42/.task/agent_session.json'),
        task_id='42',
        owner_pid=owner_pid,
        age_secs=None,
    )


class TestReapVerdict:
    @pytest.mark.parametrize('status', ['pending', 'blocked', 'done'])
    def test_dead_owner_and_known_non_in_progress_status_reaps(self, status: str):
        assert reap_verdict(owner_alive=False, status=status) is ReapVerdict.REAP

    def test_dead_owner_and_in_progress_task_keeps(self):
        assert (
            reap_verdict(owner_alive=False, status='in-progress')
            is ReapVerdict.KEEP_IN_PROGRESS
        )

    @pytest.mark.parametrize('status', ['pending', 'in-progress'])
    def test_live_owner_keeps_whatever_the_status(self, status: str):
        assert (
            reap_verdict(owner_alive=True, status=status)
            is ReapVerdict.KEEP_OWNER_ALIVE
        )

    def test_dead_owner_and_unknown_status_keeps(self):
        assert (
            reap_verdict(owner_alive=False, status=None)
            is ReapVerdict.KEEP_STATUS_UNKNOWN
        )


class TestReadSidecarClaim:
    def test_v2_sidecar_reads_its_own_identity_pid_and_age(self, tmp_path: Path):
        entry = tmp_path / '42'
        path = _write_json(entry / '.task' / 'agent_session.json', _v2_payload())

        claim = read_sidecar_claim(entry, path, fallback_task_id='99', now=_NOW)

        assert claim is not None
        assert claim.task_id == '42'
        assert claim.owner_pid == 4242
        assert claim.age_secs == 3600.0
        assert claim.entry == entry
        assert claim.sidecar_path == path

    def test_v1_sidecar_uses_the_fallback_task_id(self, tmp_path: Path):
        payload = _v2_payload()
        del payload['task_id']
        path = _write_json(tmp_path / 'agent_session.json', payload)

        claim = read_sidecar_claim(tmp_path, path, fallback_task_id='77', now=_NOW)

        assert claim is not None
        assert claim.task_id == '77'

    def test_v1_sidecar_without_a_fallback_is_unresolvable(self, tmp_path: Path):
        payload = _v2_payload()
        del payload['task_id']
        path = _write_json(tmp_path / 'agent_session.json', payload)

        assert read_sidecar_claim(tmp_path, path, fallback_task_id=None, now=_NOW) is None

    def test_integer_task_id_is_normalised_to_str(self, tmp_path: Path):
        path = _write_json(tmp_path / 'agent_session.json', _v2_payload(task_id=42))

        claim = read_sidecar_claim(tmp_path, path, fallback_task_id=None, now=_NOW)

        assert claim is not None
        assert claim.task_id == '42'

    @pytest.mark.parametrize(
        'started_at', [None, 1234, 'not-a-timestamp'], ids=['missing', 'non-str', 'unparseable'],
    )
    def test_bad_started_at_gives_no_age_but_still_a_claim(
        self, tmp_path: Path, started_at: object,
    ):
        payload = _v2_payload(started_at=started_at)
        if started_at is None:
            del payload['started_at']
        path = _write_json(tmp_path / 'agent_session.json', payload)

        claim = read_sidecar_claim(tmp_path, path, fallback_task_id=None, now=_NOW)

        assert claim is not None
        assert claim.age_secs is None

    def test_naive_started_at_is_treated_as_utc(self, tmp_path: Path):
        naive = (_NOW - timedelta(seconds=90)).replace(tzinfo=None).isoformat()
        path = _write_json(tmp_path / 'agent_session.json', _v2_payload(started_at=naive))

        claim = read_sidecar_claim(tmp_path, path, fallback_task_id=None, now=_NOW)

        assert claim is not None
        assert claim.age_secs == 90.0

    def test_corrupt_json_returns_none(self, tmp_path: Path):
        path = tmp_path / 'agent_session.json'
        path.write_text('{not json')

        assert read_sidecar_claim(tmp_path, path, fallback_task_id='1', now=_NOW) is None

    def test_json_list_returns_none(self, tmp_path: Path):
        path = _write_json(tmp_path / 'agent_session.json', [_v2_payload()])

        assert read_sidecar_claim(tmp_path, path, fallback_task_id='1', now=_NOW) is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        path = tmp_path / 'agent_session.json'

        assert read_sidecar_claim(tmp_path, path, fallback_task_id='1', now=_NOW) is None

    def test_event_data_names_path_task_pid_and_age(self, tmp_path: Path):
        path = _write_json(tmp_path / 'agent_session.json', _v2_payload())

        claim = read_sidecar_claim(tmp_path, path, fallback_task_id=None, now=_NOW)

        assert claim is not None
        assert claim.event_data() == {
            'sidecar_path': str(path),
            'task_id': '42',
            'owner_pid': 4242,
            'age_secs': 3600.0,
        }


class TestOwnerAlive:
    def test_this_process_is_alive(self):
        assert owner_alive(_claim(os.getpid())) is True

    @pytest.mark.parametrize(
        'owner_pid', [_DEAD_PID, 0, None, '4242'], ids=['dead', 'v1-default', 'missing', 'str'],
    )
    def test_dead_or_unusable_pid_reads_dead(self, owner_pid: object):
        assert owner_alive(_claim(owner_pid)) is False


class TestReapTally:
    def test_counts_each_verdict_and_carries_unresolvable_and_error(self):
        tally = ReapTally.from_verdicts(
            [
                ReapVerdict.REAP,
                ReapVerdict.REAP,
                ReapVerdict.KEEP_OWNER_ALIVE,
                ReapVerdict.KEEP_IN_PROGRESS,
                ReapVerdict.KEEP_STATUS_UNKNOWN,
                ReapVerdict.KEEP_STATUS_UNKNOWN,
                ReapVerdict.KEEP_STATUS_UNKNOWN,
            ],
            unresolvable=4,
            status_read_error=None,
        )

        assert tally.reaped == 2
        assert tally.kept_owner_alive == 1
        assert tally.kept_in_progress == 1
        assert tally.kept_status_unknown == 3
        assert tally.unresolvable == 4
        assert tally.status_read_error is None

    def test_event_data_exposes_exactly_the_counts_and_the_error(self):
        tally = ReapTally.from_verdicts([], unresolvable=0, status_read_error='boom')

        assert tally.event_data() == {
            'reaped': 0,
            'kept_owner_alive': 0,
            'kept_in_progress': 0,
            'kept_status_unknown': 0,
            'unresolvable': 0,
            'status_read_error': 'boom',
        }
