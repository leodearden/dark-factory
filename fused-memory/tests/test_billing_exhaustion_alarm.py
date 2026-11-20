"""BillingExhaustionAlarm: one born-at-L2 page per exhausted-credits episode (task 10922).

Exhausted OpenAI credits need a human to pay, so the alarm pages at L2 under
a sentinel role, files ONE record however many writes fail, resolves that
record itself on the first successful OpenAI-backed write, and re-latches
onto it after a restart.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from _fm_helpers import make_insufficient_quota_error
from shared.project_layout import escalation_queue_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.middleware import _folded_escalation
from fused_memory.services.billing_exhaustion_alarm import BillingExhaustionAlarm
from fused_memory.services.provider_billing import (
    BillingExhaustion,
    classify_billing_exhaustion,
)

pytestmark = [
    pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT),
    pytest.mark.skipif(
        not _folded_escalation.HAS_ESCALATION,
        reason='escalation package unavailable (minimal env)',
    ),
]

_FIRST_SEEN = datetime(2026, 11, 19, 6, 50, tzinfo=UTC)


def _exhaustion() -> BillingExhaustion:
    exhaustion = classify_billing_exhaustion(make_insufficient_quota_error())
    assert exhaustion is not None
    return exhaustion


def _ticking_clock() -> Callable[[], datetime]:
    ticks: Iterator[datetime] = (_FIRST_SEEN + timedelta(minutes=n) for n in range(1000))
    return lambda: next(ticks)


def _alarm(tmp_path, **kwargs) -> BillingExhaustionAlarm:
    kwargs.setdefault('clock', _ticking_clock())
    return BillingExhaustionAlarm(resolve_project_root=lambda: str(tmp_path), **kwargs)


def _records(tmp_path) -> list[dict]:
    """Every record on disk, the archive tier a resolve moves it to included."""
    queue_dir = escalation_queue_dir(tmp_path)
    return [json.loads(p.read_text()) for p in sorted(queue_dir.rglob('esc-*.json'))]


def _pending(tmp_path) -> list[dict]:
    return [r for r in _records(tmp_path) if r['status'] == 'pending']


class TestThePage:
    @pytest.mark.asyncio
    async def test_files_one_born_at_l2_critical_record_naming_the_billing_cause(self, tmp_path):
        from escalation.models import HARNESS_SENTINEL_ROLE_PREFIXES

        await _alarm(tmp_path).record_exhausted(_exhaustion(), site='durable_queue:add_episode')

        [record] = _records(tmp_path)
        assert record['category'] == 'provider_billing_exhausted'
        assert record['severity'] == 'critical'
        assert record['level'] == 2
        assert record['agent_role'].startswith(HARNESS_SENTINEL_ROLE_PREFIXES)
        assert 'credits exhausted' in record['summary'].lower()
        assert 'rate limit' not in record['summary'].lower()
        assert 'rate limit' not in record['suggested_action'].lower()
        detail = record['detail']
        assert 'provider=openai' in detail
        assert 'OPENAI_API_KEY' in detail
        assert f'first_seen={_FIRST_SEEN.isoformat()}' in detail
        assert 'first_site=durable_queue:add_episode' in detail
        assert 'parked' in detail.lower()

    @pytest.mark.asyncio
    async def test_concurrent_failures_file_once(self, tmp_path):
        alarm = _alarm(tmp_path)

        await asyncio.gather(*(
            alarm.record_exhausted(_exhaustion(), site=f'durable_queue:op{n}')
            for n in range(5)
        ))

        [record] = _records(tmp_path)
        state = alarm.state()
        assert state is not None
        assert state.first_seen == _FIRST_SEEN
        assert state.first_site == 'durable_queue:op0'
        assert state.escalation_id == record['id']
        for field in dataclasses.fields(state):
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(state, field.name, 'other')


class TestRecovery:
    @pytest.mark.asyncio
    async def test_recovery_resolves_the_page_and_a_later_exhaustion_is_a_new_episode(
        self, tmp_path,
    ):
        alarm = _alarm(tmp_path)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        [first] = _records(tmp_path)

        await alarm.record_recovered(site='mem0_write')

        [resolved] = _records(tmp_path)
        assert resolved['id'] == first['id']
        assert resolved['status'] == 'resolved'
        assert alarm.state() is None

        await alarm.record_exhausted(_exhaustion(), site='mem0_write')
        [second] = _pending(tmp_path)
        assert second['id'] != first['id']

    @pytest.mark.asyncio
    async def test_recovery_without_an_exhaustion_writes_nothing(self, tmp_path):
        await _alarm(tmp_path).record_recovered(site='mem0_write')

        assert _records(tmp_path) == []


class TestRestart:
    @pytest.mark.asyncio
    async def test_a_fresh_alarm_relatches_onto_the_open_page_and_resolves_it(self, tmp_path):
        await _alarm(tmp_path).record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        [open_page] = _records(tmp_path)

        restarted = _alarm(tmp_path)
        await restarted.restore()

        state = restarted.state()
        assert state is not None
        assert state.escalation_id == open_page['id']
        await restarted.record_recovered(site='durable_queue')
        assert [r['status'] for r in _records(tmp_path)] == ['resolved']

    @pytest.mark.asyncio
    async def test_a_relatched_alarm_does_not_page_again(self, tmp_path):
        await _alarm(tmp_path).record_exhausted(_exhaustion(), site='durable_queue:add_episode')

        restarted = _alarm(tmp_path)
        await restarted.restore()
        await restarted.record_exhausted(_exhaustion(), site='mem0_write')

        assert len(_records(tmp_path)) == 1


class TestNoProjectRoot:
    @pytest.mark.asyncio
    async def test_a_missing_root_warns_and_never_raises(self, caplog):
        alarm = BillingExhaustionAlarm(resolve_project_root=lambda: None)

        with caplog.at_level(logging.DEBUG):
            await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        assert any(
            r.levelno == logging.WARNING and r.name == 'fused_memory.services.billing_exhaustion_alarm'
            for r in caplog.records
        )
