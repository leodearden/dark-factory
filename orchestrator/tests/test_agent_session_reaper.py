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
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared import transcript_archive as transcript_archive_module

from orchestrator.agent_session_reaper import (
    ReapTally,
    ReapVerdict,
    SessionSidecarClaim,
    owner_alive,
    read_sidecar_claim,
    reap_verdict,
)
from orchestrator.artifacts import AgentSession, TaskArtifacts
from orchestrator.config import TranscriptArchiveConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.lane_lifecycle import LaneLifecycle

_DEAD_PID = 2**31 - 1
_NOW = datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _v2_payload(**overrides: object) -> dict:
    session = AgentSession(
        session_id='sess-1',
        role='implementer',
        started_at=(_NOW - timedelta(seconds=3600)).isoformat(),
        owner_pid=4242,
        task_id='42',
    )
    return {**session.to_dict(), **overrides}


_V2_ONLY_KEYS = ('task_id', 'resume_count', 'schema_version')


def _as_v1(payload: dict) -> dict:
    """The same sidecar as a pre-v2 writer left it: without the keys v2 added."""
    return {k: v for k, v in payload.items() if k not in _V2_ONLY_KEYS}


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
        path = _write_json(tmp_path / 'agent_session.json', _as_v1(_v2_payload()))

        claim = read_sidecar_claim(tmp_path, path, fallback_task_id='77', now=_NOW)

        assert claim is not None
        assert claim.task_id == '77'

    def test_v1_sidecar_without_a_fallback_is_unresolvable(self, tmp_path: Path):
        path = _write_json(tmp_path / 'agent_session.json', _as_v1(_v2_payload()))

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

    def test_uncoercible_resume_count_returns_none(self, tmp_path: Path):
        path = _write_json(
            tmp_path / 'agent_session.json', _v2_payload(resume_count='many'),
        )

        assert read_sidecar_claim(tmp_path, path, fallback_task_id='1', now=_NOW) is None

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
            clear_failed=0,
            unresolvable=4,
            status_read_error=None,
        )

        assert tally.reaped == 2
        assert tally.clear_failed == 0
        assert tally.kept_owner_alive == 1
        assert tally.kept_in_progress == 1
        assert tally.kept_status_unknown == 3
        assert tally.unresolvable == 4
        assert tally.status_read_error is None

    def test_a_failed_clear_is_not_counted_reaped(self):
        tally = ReapTally.from_verdicts(
            [ReapVerdict.REAP] * 3,
            clear_failed=1,
            unresolvable=0,
            status_read_error=None,
        )

        assert tally.reaped == 2
        assert tally.clear_failed == 1

    def test_more_failed_clears_than_reap_verdicts_is_rejected(self):
        with pytest.raises(ValueError, match='clear_failed'):
            ReapTally.from_verdicts(
                [ReapVerdict.REAP, ReapVerdict.KEEP_IN_PROGRESS],
                clear_failed=2,
                unresolvable=0,
                status_read_error=None,
            )

    def test_event_data_exposes_exactly_the_counts_and_the_error(self):
        tally = ReapTally.from_verdicts(
            [], clear_failed=0, unresolvable=0, status_read_error='boom',
        )

        assert tally.event_data() == {
            'reaped': 0,
            'clear_failed': 0,
            'kept_owner_alive': 0,
            'kept_in_progress': 0,
            'kept_status_unknown': 0,
            'unresolvable': 0,
            'status_read_error': 'boom',
        }


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """A Harness with mocked internals and a REAL event store, mirroring
    test_transcript_archive_startup_sweeper.py's fixture.
    """
    mock_orch_config.transcript_archive = TranscriptArchiveConfig()
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler.get_tasks = AsyncMock(return_value=[])
    h.scheduler.set_task_status = AsyncMock()
    h.scheduler.get_task = AsyncMock(return_value={})
    h.scheduler.get_status = AsyncMock(return_value=None)
    h.scheduler._dispatched = set()
    h.scheduler.is_deterministic = MagicMock(return_value=False)
    _serve_statuses(h, {})

    h.git_ops.worktree_base = (tmp_path / '.worktrees').resolve()
    h.git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    h.git_ops.mark_pool_storage_present()
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.quarantine_worktree = AsyncMock(return_value=None)
    h.git_ops._lane_lifecycle = LaneLifecycle(
        h.git_ops.worktree_base, quarantine_worktree=h.git_ops.quarantine_worktree,
    )
    h.git_ops._is_registered_worktree = AsyncMock(return_value=True)
    h.event_store = EventStore(tmp_path / 'runs.db', run_id='r1')
    yield h
    transcript_archive_module._reset_archival_failures()


def _serve_statuses(
    harness: Harness, statuses: dict[str, str], err: Exception | None = None,
) -> AsyncMock:
    """Answer get_statuses from *statuses*, omitting ids it does not know."""

    async def _get_statuses(ids: list[str] | None = None):
        if err is not None:
            return {}, err
        return {k: v for k, v in statuses.items() if ids is None or k in ids}, None

    mock = AsyncMock(side_effect=_get_statuses)
    harness.scheduler.get_statuses = mock
    return mock


def _seed_sidecar(
    base: Path,
    entry_name: str,
    *,
    owner_pid: int,
    task_id: str | None = None,
    root: Literal['legacy', 'meta'] = 'legacy',
) -> Path:
    """Write a v2 sidecar for *entry_name* (v1 when *task_id* is None)."""
    entry = base / entry_name
    entry.mkdir(parents=True, exist_ok=True)
    payload = AgentSession(
        session_id=f'sess-{entry_name}',
        role='implementer',
        started_at=(datetime.now(UTC) - timedelta(hours=176)).isoformat(),
        owner_pid=owner_pid,
        task_id=task_id,
    ).to_dict()
    if task_id is None:
        payload = _as_v1(payload)
    if root == 'legacy':
        path = entry / '.task' / 'agent_session.json'
    else:
        path = TaskArtifacts.meta_root_for(base, entry_name) / 'agent_session.json'
    return _write_json(path, payload)


async def _boot(harness: Harness):
    """Run step 2c's boot recovery scan, which the reaper hangs off."""
    return await harness._recover_crashed_tasks()


def _summary(harness: Harness) -> dict:
    assert harness.event_store is not None
    rows = harness.event_store.fetch_events_by_type(EventType.sidecar_reap_summary)
    assert len(rows) == 1
    return rows[0]['data']


def _reaped_rows(harness: Harness) -> list[dict]:
    assert harness.event_store is not None
    return harness.event_store.fetch_events_by_type(EventType.sidecar_reaped)


def _awaited_ids(mock: AsyncMock) -> list[str]:
    call = mock.await_args
    assert call is not None
    return call.kwargs['ids'] if 'ids' in call.kwargs else call.args[0]


@pytest.mark.asyncio
class TestZombieSidecarReap:
    async def test_dead_owner_and_pending_task_is_reaped_from_both_roots(
        self, harness: Harness,
    ):
        base = harness.git_ops.worktree_base
        legacy = _seed_sidecar(base, '61', owner_pid=_DEAD_PID, task_id='61')
        meta = _seed_sidecar(base, '61', owner_pid=_DEAD_PID, task_id='61', root='meta')
        _serve_statuses(harness, {'61': 'pending'})
        cleanup = AsyncMock()
        harness.git_ops.cleanup_worktree = cleanup

        await _boot(harness)

        assert not legacy.exists()
        assert not meta.exists()
        assert '61' not in harness._recovered_sessions
        assert '61' not in harness._preserved_worktrees
        cleanup.assert_any_await(base / '61', '61')
        rows = _reaped_rows(harness)
        assert len(rows) == 1
        assert rows[0]['task_id'] == '61'
        data = rows[0]['data']
        assert set(data) == {'sidecar_path', 'task_id', 'owner_pid', 'age_secs', 'status'}
        assert data['sidecar_path'] == str(meta)
        assert data['task_id'] == '61'
        assert data['owner_pid'] == _DEAD_PID
        assert isinstance(data['age_secs'], float)
        assert data['status'] == 'pending'

    async def test_dead_owner_but_in_progress_task_is_kept(self, harness: Harness):
        """PRD §9 B10: the second conjunct is load-bearing."""
        base = harness.git_ops.worktree_base
        sidecar = _seed_sidecar(base, '62', owner_pid=_DEAD_PID, task_id='62')
        _serve_statuses(harness, {'62': 'in-progress'})

        await _boot(harness)

        assert sidecar.exists()
        assert '62' in harness._recovered_sessions
        assert _reaped_rows(harness) == []
        assert _summary(harness)['kept_in_progress'] == 1

    async def test_live_owner_is_kept_and_adopted(self, harness: Harness):
        base = harness.git_ops.worktree_base
        sidecar = _seed_sidecar(base, '64', owner_pid=os.getpid(), task_id='64')
        _serve_statuses(harness, {'64': 'pending'})

        await _boot(harness)

        assert sidecar.exists()
        assert '64' in harness._recovered_sessions
        assert _summary(harness)['kept_owner_alive'] == 1

    async def test_status_omitted_from_the_response_is_kept(self, harness: Harness):
        base = harness.git_ops.worktree_base
        sidecar = _seed_sidecar(base, '65', owner_pid=_DEAD_PID, task_id='65')
        _serve_statuses(harness, {})

        await _boot(harness)

        assert sidecar.exists()
        assert _reaped_rows(harness) == []
        assert _summary(harness)['kept_status_unknown'] == 1

    async def test_status_read_error_reaps_nothing_and_is_reported(
        self, harness: Harness,
    ):
        base = harness.git_ops.worktree_base
        sidecar = _seed_sidecar(base, '66', owner_pid=_DEAD_PID, task_id='66')
        _serve_statuses(harness, {'66': 'pending'}, err=RuntimeError('boom'))

        await _boot(harness)

        assert sidecar.exists()
        assert _reaped_rows(harness) == []
        summary = _summary(harness)
        assert summary['reaped'] == 0
        assert summary['kept_status_unknown'] == 1
        assert summary['status_read_error'] is not None
        assert 'boom' in summary['status_read_error']

    async def test_lane_sidecar_reaps_by_its_own_task_id_and_v1_is_unresolvable(
        self, harness: Harness,
    ):
        base = harness.git_ops.worktree_base
        v2 = _seed_sidecar(base, '_lane-0', owner_pid=_DEAD_PID, task_id='63', root='meta')
        v1 = _seed_sidecar(base, '_lane-1', owner_pid=_DEAD_PID, root='meta')
        _serve_statuses(harness, {'63': 'pending'})

        await _boot(harness)

        assert not v2.exists()
        assert v1.exists()
        assert [row['task_id'] for row in _reaped_rows(harness)] == ['63']
        summary = _summary(harness)
        assert summary['reaped'] == 1
        assert summary['unresolvable'] == 1

    async def test_v1_sidecar_on_a_cold_worktree_reaps_under_the_entry_name(
        self, harness: Harness,
    ):
        base = harness.git_ops.worktree_base
        sidecar = _seed_sidecar(base, '67', owner_pid=_DEAD_PID)
        _serve_statuses(harness, {'67': 'pending'})

        await _boot(harness)

        assert not sidecar.exists()
        rows = _reaped_rows(harness)
        assert [row['task_id'] for row in rows] == ['67']
        assert rows[0]['data']['task_id'] == '67'
        assert _summary(harness)['unresolvable'] == 0

    async def test_an_entry_that_cannot_be_read_costs_only_itself(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch,
    ):
        base = harness.git_ops.worktree_base
        _seed_sidecar(base, '81', owner_pid=_DEAD_PID, task_id='81')
        sibling = _seed_sidecar(base, '82', owner_pid=_DEAD_PID, task_id='82')
        _serve_statuses(harness, {'81': 'pending', '82': 'pending'})
        real_resolve = harness._resolve_recovery_artifact

        def _resolve(entry: Path, name: str) -> Path:
            if entry.name == '81':
                raise PermissionError(13, 'Permission denied', str(entry))
            return real_resolve(entry, name)

        monkeypatch.setattr(harness, '_resolve_recovery_artifact', _resolve)

        tally = await harness._reap_zombie_sidecars()

        assert not sibling.exists()
        assert (tally.reaped, tally.unresolvable) == (1, 1)
        assert _summary(harness)['unresolvable'] == 1

    async def test_a_sidecar_that_cannot_be_cleared_costs_only_itself(
        self, harness: Harness,
    ):
        base = harness.git_ops.worktree_base
        _seed_sidecar(base, '83', owner_pid=_DEAD_PID, task_id='83', root='meta')
        (base / '83' / '.task' / 'agent_session.json').mkdir(parents=True)
        sibling = _seed_sidecar(base, '84', owner_pid=_DEAD_PID, task_id='84')
        _serve_statuses(harness, {'83': 'pending', '84': 'pending'})

        tally = await harness._reap_zombie_sidecars()

        assert not sibling.exists()
        assert [row['task_id'] for row in _reaped_rows(harness)] == ['84']
        assert (tally.reaped, tally.clear_failed) == (1, 1)
        summary = _summary(harness)
        assert (summary['reaped'], summary['clear_failed']) == (1, 1)

    async def test_mixed_boot_makes_one_bulk_read_of_the_dead_owner_ids(
        self, harness: Harness,
    ):
        base = harness.git_ops.worktree_base
        _seed_sidecar(base, '71', owner_pid=_DEAD_PID, task_id='71')
        _seed_sidecar(base, '72', owner_pid=_DEAD_PID, task_id='72')
        _seed_sidecar(base, '73', owner_pid=_DEAD_PID, task_id='73')
        _seed_sidecar(base, '74', owner_pid=os.getpid(), task_id='74')
        get_statuses = _serve_statuses(
            harness,
            {'71': 'pending', '72': 'done', '73': 'in-progress', '74': 'pending'},
        )

        await _boot(harness)

        assert get_statuses.await_count == 1
        assert sorted(_awaited_ids(get_statuses)) == ['71', '72', '73']
        assert _summary(harness) == {
            'reaped': 2,
            'clear_failed': 0,
            'kept_owner_alive': 1,
            'kept_in_progress': 1,
            'kept_status_unknown': 0,
            'unresolvable': 0,
            'status_read_error': None,
        }

    async def test_clean_boot_skips_the_read_and_still_reports_zero(
        self, harness: Harness,
    ):
        get_statuses = _serve_statuses(harness, {})

        await _boot(harness)

        get_statuses.assert_not_awaited()
        assert _summary(harness) == {
            'reaped': 0,
            'clear_failed': 0,
            'kept_owner_alive': 0,
            'kept_in_progress': 0,
            'kept_status_unknown': 0,
            'unresolvable': 0,
            'status_read_error': None,
        }


def _seed_plan_worktree(base: Path, task_id: str) -> Path:
    """A cold worktree whose plan has completed work, so recovery pre-loads it."""
    plan = {
        'task_id': task_id,
        'title': 'Test Task',
        'steps': [
            {'id': 'step-1', 'description': 'Step 1', 'status': 'done', 'commit': 'abc0'},
            {'id': 'step-2', 'description': 'Step 2', 'status': 'pending', 'commit': None},
        ],
    }
    entry = base / task_id
    _write_json(entry / '.task' / 'plan.json', plan)
    return entry


@pytest.mark.asyncio
class TestReaperWiredIntoCrashRecovery:
    """Where the reaper hangs off step 2c, and where it must NOT run."""

    async def test_runs_once_before_the_first_cleanup(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch,
    ):
        """Before the per-entry loop, so no sidecar is adopted and then cleared."""
        order: list[str] = []
        real_reap = harness._reap_zombie_sidecars

        async def _reap() -> ReapTally:
            order.append('reap')
            return await real_reap()

        monkeypatch.setattr(harness, '_reap_zombie_sidecars', _reap)

        async def _cleanup(*a, **k):
            order.append('cleanup')

        harness.git_ops.cleanup_worktree = AsyncMock(side_effect=_cleanup)
        (harness.git_ops.worktree_base / '71').mkdir(parents=True, exist_ok=True)

        await _boot(harness)

        assert order.count('reap') == 1
        assert order.index('reap') < order.index('cleanup')

    async def test_not_run_when_pool_storage_absent(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch,
    ):
        base = harness.git_ops.worktree_base
        sidecar = _seed_sidecar(base, '72', owner_pid=_DEAD_PID, task_id='72')
        get_statuses = _serve_statuses(harness, {'72': 'pending'})
        monkeypatch.setattr(harness.git_ops, 'pool_in_use', MagicMock(return_value=True))
        monkeypatch.setattr(
            harness.git_ops, 'pool_storage_present', MagicMock(return_value=False),
        )
        monkeypatch.setattr(harness, '_file_pool_storage_absent_escalation', MagicMock())

        await _boot(harness)

        assert sidecar.exists()
        get_statuses.assert_not_awaited()

    async def test_reaper_failure_is_a_partial_outcome_and_plans_still_recover(
        self, harness: Harness, monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            harness, '_reap_zombie_sidecars',
            AsyncMock(side_effect=RuntimeError('reaper bug')),
        )
        _seed_plan_worktree(harness.git_ops.worktree_base, '73')

        failures = await _boot(harness)

        assert [f.exception_type for f in failures.failures] == ['RuntimeError']
        assert '73' in harness._recovered_plans
