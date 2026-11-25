"""PlanCapAlarm: one deduped L2 record per plan-cap episode (task 9944).

Only Leo can upgrade the ChatGPT plan, so a cap files ONE level-2 record.
Each probe that re-confirms the cap refreshes that record in place, and
recovery leaves it open for Leo to close.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest
from shared.project_layout import escalation_queue_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.middleware import _folded_escalation
from fused_memory.services.plan_cap_alarm import ANCHOR_TASK_ID, PlanCapAlarm
from fused_memory.services.plan_route import (
    PlanRouteSite,
    PlanRouteSnapshot,
    PlanRouteStateName,
)
from fused_memory.services.provider_billing import plan_cap_of

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.skipif(
        not _folded_escalation.HAS_ESCALATION,
        reason='escalation package unavailable (minimal env)',
    ),
]

_SINCE = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
_RESETS = _SINCE + timedelta(days=3)
_NEXT = _SINCE + timedelta(minutes=15)
_RECOVERED = _SINCE + timedelta(hours=2)
_RECAPPED = _SINCE + timedelta(hours=5)
_PROXY_BODY = '{"error": {"type": "usage_limit_reached", "message": "weekly limit reached"}}'
_ALARM_LOGGER = 'fused_memory.services.plan_cap_alarm'


def _snapshot(
    *, capped_since: datetime = _SINCE, resets_at: datetime | None = _RESETS, judge_calls: int = 3,
) -> PlanRouteSnapshot:
    return PlanRouteSnapshot(
        state=PlanRouteStateName.CAPPED,
        proxy_host='127.0.0.1:8317',
        capped_since=capped_since,
        resets_at=resets_at,
        next_probe_at=_NEXT,
        affected_calls_by_site=MappingProxyType({
            PlanRouteSite.WRITE_TRIAGE_JUDGE: judge_calls, PlanRouteSite.GRAPHITI: 1,
        }),
        exhaustion=plan_cap_of(_PROXY_BODY.encode()),
    )


def _recovered_snapshot() -> PlanRouteSnapshot:
    return PlanRouteSnapshot(state=PlanRouteStateName.OK, proxy_host='127.0.0.1:8317')


class _ParkedOrInFlight:
    def __init__(self, count: int = 4) -> None:
        self.count = count

    async def __call__(self) -> int:
        return self.count


def _alarm(
    root: Callable[[], str | None], writes: _ParkedOrInFlight | None = None,
) -> PlanCapAlarm:
    return PlanCapAlarm(
        resolve_project_root=root,
        count_parked_or_in_flight=writes or _ParkedOrInFlight(),
        clock=lambda: _RECOVERED,
    )


def _records(root: Path) -> list[dict]:
    records = (json.loads(p.read_text()) for p in escalation_queue_dir(root).rglob('esc-*.json'))
    return [r for r in records if r['category'] == 'plan_route_capped']


def _resolve(root: Path, escalation_id: str) -> None:
    from escalation.queue import EscalationQueue

    EscalationQueue(escalation_queue_dir(root)).resolve(
        escalation_id, 'Leo upgraded the plan', resolution_class='actionable',
    )


class TestTheFirstCap:
    @pytest.mark.asyncio
    async def test_files_one_level_two_blocking_record(self, tmp_path: Path) -> None:
        alarm = _alarm(lambda: str(tmp_path))

        await alarm.on_capped(_snapshot())

        [record] = _records(tmp_path)
        assert record['level'] == 2
        assert record['severity'] == 'blocking'
        assert record['task_id'] == ANCHOR_TASK_ID == 'plan-route-capped'
        assert record['status'] == 'pending'
        assert alarm.escalation_id() == record['id']
        detail = record['detail'].splitlines()
        for line in (
            'billing_source=plan_cap',
            f'capped_since={_SINCE.isoformat()}',
            f'resets_at={_RESETS.isoformat()}',
            f'next_probe_at={_NEXT.isoformat()}',
            'sites_affected=graphiti:1,write_triage_judge:3',
            'parked_or_in_flight_writes=4',
            'judge_fail_opens=3',
        ):
            assert line in detail, (line, detail)
        assert any(
            line.startswith('provider_message=') and 'weekly limit reached' in line
            for line in detail
        )
        assert record['suggested_action'] == (
            f'upgrade the ChatGPT plan, or wait for the reset at {_RESETS.isoformat()}; '
            'parked writes drain automatically when the cap clears'
        )

    @pytest.mark.asyncio
    async def test_an_unknown_reset_says_so(self, tmp_path: Path) -> None:
        await _alarm(lambda: str(tmp_path)).on_capped(_snapshot(resets_at=None))

        [record] = _records(tmp_path)
        assert 'resets_at=unknown' in record['detail'].splitlines()

    @pytest.mark.asyncio
    async def test_a_restarted_alarm_adopts_and_refreshes_the_open_record(
        self, tmp_path: Path,
    ) -> None:
        first = _alarm(lambda: str(tmp_path))
        await first.on_capped(_snapshot())

        restarted = _alarm(lambda: str(tmp_path))
        await restarted.on_capped(_snapshot(capped_since=_RECAPPED))

        [record] = _records(tmp_path)
        assert restarted.escalation_id() == record['id'] == first.escalation_id()
        assert record['dedupe_count'] == 1
        assert f'capped_since={_RECAPPED.isoformat()}' in record['detail'].splitlines()
        assert _RECAPPED.isoformat() in record['summary']


class TestASecondCapWhileTheRecordIsOpen:
    @pytest.mark.asyncio
    async def test_refreshes_the_open_record_with_the_new_episode(self, tmp_path: Path) -> None:
        writes = _ParkedOrInFlight(1)
        alarm = _alarm(lambda: str(tmp_path), writes)
        await alarm.on_capped(_snapshot())
        await alarm.on_recovered(_recovered_snapshot())
        writes.count = 2

        await alarm.on_capped(_snapshot(capped_since=_RECAPPED))

        [record] = _records(tmp_path)
        assert record['id'] == alarm.escalation_id()
        assert record['dedupe_count'] == 1
        detail = record['detail'].splitlines()
        assert f'capped_since={_RECAPPED.isoformat()}' in detail
        assert 'parked_or_in_flight_writes=2' in detail
        assert _RECAPPED.isoformat() in record['summary']


class TestAReconfirmedCap:
    @pytest.mark.asyncio
    async def test_refreshes_the_one_record_in_place(self, tmp_path: Path) -> None:
        writes = _ParkedOrInFlight(4)
        alarm = _alarm(lambda: str(tmp_path), writes)
        await alarm.on_capped(_snapshot())
        writes.count = 9

        await alarm.on_cap_confirmed(_snapshot(judge_calls=7))

        [record] = _records(tmp_path)
        assert record['id'] == alarm.escalation_id()
        assert record['dedupe_count'] == 1
        assert 'parked_or_in_flight_writes=9' in record['detail'].splitlines()
        assert 'judge_fail_opens=7' in record['detail'].splitlines()

    @pytest.mark.asyncio
    async def test_a_record_resolved_meanwhile_is_filed_afresh(self, tmp_path: Path) -> None:
        alarm = _alarm(lambda: str(tmp_path))
        await alarm.on_capped(_snapshot())
        first = alarm.escalation_id()
        assert first is not None
        _resolve(tmp_path, first)

        await alarm.on_cap_confirmed(_snapshot())

        pending = [r for r in _records(tmp_path) if r['status'] == 'pending']
        assert [r['id'] for r in pending] == [alarm.escalation_id()]
        assert alarm.escalation_id() != first


class TestRecovery:
    @pytest.mark.asyncio
    async def test_leaves_the_record_open_for_leo(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        alarm = _alarm(lambda: str(tmp_path))
        await alarm.on_capped(_snapshot())
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=_ALARM_LOGGER):
            await alarm.on_recovered(_recovered_snapshot())

        [record] = _records(tmp_path)
        assert record['status'] == 'pending'
        [line] = [r.getMessage() for r in caplog.records if r.name == _ALARM_LOGGER]
        assert _RECOVERED.isoformat() in line
        assert record['id'] in line


class TestAQueueThatCannotBeOpened:
    @pytest.mark.asyncio
    async def test_files_nothing_raises_nothing_and_the_next_probe_files(
        self, tmp_path: Path,
    ) -> None:
        roots: list[str | None] = [None]
        alarm = _alarm(lambda: roots[0])

        await alarm.on_capped(_snapshot())
        assert alarm.escalation_id() is None
        assert _records(tmp_path) == []

        roots[0] = str(tmp_path)
        await alarm.on_cap_confirmed(_snapshot())

        [record] = _records(tmp_path)
        assert alarm.escalation_id() == record['id']
