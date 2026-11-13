"""The digest wires the warm-lane map↔record divergence cross-check (task 2987).

Imports no merge-lane cluster module: state is seeded through the public
warm-lane pool API on an unpatched, pool-enabled ``Harness``.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _orch_helpers import HermeticMcpSession
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.lane_record_divergence import (
    LANE_RECORD_DIVERGENCE,
    PERSISTENT_DIVERGENCE_MIN_AGE,
    STREAKS_FILENAME,
    LaneDivergence,
)

_T0 = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _pool_harness(
    tmp_path: Path, queue: EscalationQueue, *, warm_lane_pool: bool = True,
) -> Harness:
    harness = Harness(OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=2,
        git=GitConfig(warm_lane_pool=warm_lane_pool),
        digest_every_n_escalations=1,
        digest_ewa_threshold=999.0,
        digest_dir=str(tmp_path / 'digests'),
    ))
    harness._escalation_queue = queue
    # The digest's stale-lane census reads task statuses over MCP.
    harness.scheduler._mcp_session = HermeticMcpSession()
    return harness


def _lane(harness: Harness) -> Path:
    return harness.git_ops.worktree_base / '_lane-0'


def _assign(harness: Harness) -> None:
    pool = harness.git_ops.warm_lane_pool
    assert pool is not None
    pool.note_assignment('101', _lane(harness))


def _desync(harness: Harness) -> None:
    """Leave the durable record ASSIGNED:101 with no map entry."""
    _assign(harness)
    pool = harness.git_ops.warm_lane_pool
    assert pool is not None
    pool.drop_assignment('101')


def _divergence_alarms(queue: EscalationQueue) -> list[Escalation]:
    return [e for e in queue.get_pending() if e.root_cause == LANE_RECORD_DIVERGENCE]


def _advance(
    harness: Harness, digest_dir: Path, *, now: datetime = _T0,
) -> tuple[LaneDivergence, ...] | None:
    return harness._advance_lane_record_divergence_and_alarm(digest_dir, now=now)


class TestAdvanceLaneRecordDivergenceAndAlarm:
    def test_coherent_pool_has_no_divergence(self, tmp_path, queue):
        harness = _pool_harness(tmp_path, queue)
        _assign(harness)

        assert _advance(harness, tmp_path / 'digests') == ()
        assert _divergence_alarms(queue) == []

    def test_persistent_desync_files_one_alarm(self, tmp_path, queue):
        harness = _pool_harness(tmp_path, queue)
        _desync(harness)
        digest_dir = tmp_path / 'digests'

        first = _advance(harness, digest_dir)
        assert first is not None
        (divergence,) = first
        assert divergence.lane == '_lane-0'
        assert divergence.map_task_ids == ()
        assert divergence.record_task_id == '101'
        assert _divergence_alarms(queue) == []

        second = _advance(harness, digest_dir, now=_T0 + PERSISTENT_DIVERGENCE_MIN_AGE)
        assert second is not None
        assert [d.streak.consecutive_digests for d in second] == [2]
        assert len(_divergence_alarms(queue)) == 1

        _advance(harness, digest_dir, now=_T0 + 2 * PERSISTENT_DIVERGENCE_MIN_AGE)
        assert len(_divergence_alarms(queue)) == 1

    def test_streak_survives_a_restart(self, tmp_path, queue):
        digest_dir = tmp_path / 'digests'
        before_restart = _pool_harness(tmp_path, queue)
        _desync(before_restart)
        _advance(before_restart, digest_dir)

        after_restart = _pool_harness(tmp_path, queue)
        divergences = _advance(after_restart, digest_dir, now=_T0 + timedelta(minutes=5))

        assert divergences is not None
        assert [d.streak.consecutive_digests for d in divergences] == [2]

    def test_repaired_map_clears_divergence_and_streaks(self, tmp_path, queue):
        harness = _pool_harness(tmp_path, queue)
        _desync(harness)
        digest_dir = tmp_path / 'digests'
        _advance(harness, digest_dir)

        _assign(harness)

        assert _advance(harness, digest_dir) == ()
        assert json.loads((digest_dir / STREAKS_FILENAME).read_text()) == {}

    def test_pool_less_host_has_no_divergence(self, tmp_path, queue):
        harness = _pool_harness(tmp_path, queue, warm_lane_pool=False)

        assert _advance(harness, tmp_path / 'digests') == ()

    @pytest.mark.skipif(os.geteuid() == 0, reason='root reads a 0o000 file')
    def test_unreadable_records_report_unavailable_and_keep_streaks(self, tmp_path, queue):
        harness = _pool_harness(tmp_path, queue)
        _desync(harness)
        digest_dir = tmp_path / 'digests'
        _advance(harness, digest_dir)
        streaks_before = (digest_dir / STREAKS_FILENAME).read_bytes()
        record = next(harness.git_ops.worktree_base.glob('*/_lane-0.json'))

        record.chmod(0o000)
        try:
            assert _advance(harness, digest_dir) is None
        finally:
            record.chmod(0o600)

        assert (digest_dir / STREAKS_FILENAME).read_bytes() == streaks_before


def _newest_digest(tmp_path: Path) -> str:
    return sorted((tmp_path / 'digests').glob('digest-*.md'))[-1].read_text()


def _divergence_section(md: str) -> list[str]:
    lines = md.splitlines()
    (start,) = [
        i for i, line in enumerate(lines)
        if line.startswith('## ') and LANE_RECORD_DIVERGENCE in line
    ]
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith('## ')),
        len(lines),
    )
    return lines[start + 1:end]


async def _write_digest(harness: Harness) -> str:
    harness._escalation_event_count += 1
    await harness._maybe_write_digest()
    return _newest_digest(harness.config.project_root)


@pytest.mark.asyncio
class TestDigestCarriesLaneRecordDivergence:
    async def test_desync_seen_an_hour_ago_appears_in_each_digest_and_alarms_once(
        self, tmp_path, queue,
    ):
        harness = _pool_harness(tmp_path, queue)
        _desync(harness)
        _advance(
            harness, tmp_path / 'digests',
            now=datetime.now(UTC) - PERSISTENT_DIVERGENCE_MIN_AGE,
        )

        for _ in range(2):
            section = _divergence_section(await _write_digest(harness))
            assert any('_lane-0' in line and '101' in line for line in section)

        assert len(_divergence_alarms(queue)) == 1

    async def test_clean_digest_shows_none(self, tmp_path, queue):
        harness = _pool_harness(tmp_path, queue)
        _assign(harness)

        section = _divergence_section(await _write_digest(harness))

        assert section[0] == '_none_'

    @pytest.mark.skipif(os.geteuid() == 0, reason='root reads a 0o000 file')
    async def test_unreadable_records_still_write_a_digest_marked_unavailable(
        self, tmp_path, queue,
    ):
        harness = _pool_harness(tmp_path, queue)
        _desync(harness)
        record = next(harness.git_ops.worktree_base.glob('*/_lane-0.json'))

        record.chmod(0o000)
        try:
            section = _divergence_section(await _write_digest(harness))
        finally:
            record.chmod(0o600)

        assert 'unavailable' in section[0]
