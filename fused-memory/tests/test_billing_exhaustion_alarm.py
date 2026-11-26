"""BillingExhaustionAlarm: one born-at-L2 page per exhausted-credits episode (task 10922).

Exhausted OpenAI credits need a human to pay, so the alarm pages at L2 under
a sentinel role, files ONE record however many writes fail, resolves that
record itself on the first successful OpenAI-backed write, and re-latches
onto it after a restart.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta

import pytest
from _fm_helpers import make_insufficient_quota_error, make_usage_limit_reached_error
from shared.project_layout import escalation_queue_dir
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.middleware import _folded_escalation
from fused_memory.services.billing_exhaustion_alarm import (
    REPAGE_COOLDOWN,
    BillingExhaustionAlarm,
)
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
_ALARM_LOGGER = 'fused_memory.services.billing_exhaustion_alarm'


def _exhaustion(**error_kwargs) -> BillingExhaustion:
    exhaustion = classify_billing_exhaustion(make_insufficient_quota_error(**error_kwargs))
    assert exhaustion is not None
    return exhaustion


def _ticking_clock() -> Callable[[], datetime]:
    ticks: Iterator[datetime] = (_FIRST_SEEN + timedelta(minutes=n) for n in range(1000))
    return lambda: next(ticks)


class _Clock:
    """A clock that moves only when the test moves it."""

    def __init__(self) -> None:
        self.now = _FIRST_SEEN

    def __call__(self) -> datetime:
        return self.now

    def advance(self, by: timedelta) -> None:
        self.now += by


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
        assert '(insufficient_quota)' in record['summary']
        detail = record['detail']
        assert 'provider=openai' in detail
        assert 'OPENAI_API_KEY' in detail
        assert 'code=insufficient_quota' in detail
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


    @pytest.mark.asyncio
    async def test_the_page_names_the_code_the_provider_sent(self, tmp_path):
        exhaustion = _exhaustion(code='credit_balance_exhausted')

        await _alarm(tmp_path).record_exhausted(exhaustion, site='mem0_write')

        [record] = _records(tmp_path)
        assert '(credit_balance_exhausted)' in record['summary']
        assert 'code=credit_balance_exhausted' in record['detail']
        assert '(insufficient_quota)' not in record['summary']


class TestOnlyApiKeyCreditsPage:
    @pytest.mark.asyncio
    async def test_a_plan_cap_files_nothing(self, tmp_path):
        plan_cap = classify_billing_exhaustion(make_usage_limit_reached_error())
        assert plan_cap is not None
        alarm = _alarm(tmp_path)

        await alarm.record_exhausted(plan_cap, site='durable_queue:add_episode')

        assert alarm.state() is None
        assert _records(tmp_path) == []


class TestAPageThatCouldNotBeFiled:
    @pytest.mark.asyncio
    async def test_the_next_exhaustion_files_it_exactly_once(self, tmp_path):
        alarm = _alarm(tmp_path)
        # A file where the queue directory goes, so the queue cannot be opened.
        blocked = escalation_queue_dir(tmp_path)
        blocked.parent.mkdir(parents=True, exist_ok=True)
        blocked.write_text('')

        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')

        state = alarm.state()
        assert state is not None
        assert state.escalation_id is None

        blocked.unlink()
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        [record] = _records(tmp_path)
        state = alarm.state()
        assert state is not None
        assert state.escalation_id == record['id']
        assert state.first_site == 'durable_queue:add_episode'


class TestRecovery:
    @pytest.mark.asyncio
    async def test_recovery_resolves_the_page_and_a_later_exhaustion_is_a_new_episode(
        self, tmp_path,
    ):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        [first] = _records(tmp_path)

        await alarm.record_recovered(site='mem0_write')

        [resolved] = _records(tmp_path)
        assert resolved['id'] == first['id']
        assert resolved['status'] == 'resolved'
        assert alarm.state() is None

        clock.advance(REPAGE_COOLDOWN)
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')
        [second] = _pending(tmp_path)
        assert second['id'] != first['id']

    @pytest.mark.asyncio
    async def test_recovery_without_an_exhaustion_writes_nothing(self, tmp_path):
        await _alarm(tmp_path).record_recovered(site='mem0_write')

        assert _records(tmp_path) == []


class TestTheRepageCooldown:
    """A balance still settling after a top-up must not page Leo again."""

    @pytest.mark.asyncio
    async def test_a_recover_exhaust_recover_burst_files_one_page(self, tmp_path):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')

        for _ in range(3):
            clock.advance(timedelta(minutes=1))
            await alarm.record_recovered(site='mem0_write')
            clock.advance(timedelta(minutes=1))
            await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        await alarm.record_recovered(site='mem0_write')

        assert [r['status'] for r in _records(tmp_path)] == ['resolved']

    @pytest.mark.asyncio
    async def test_an_exhaustion_that_outlasts_the_cooldown_pages(self, tmp_path):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        await alarm.record_recovered(site='mem0_write')
        clock.advance(timedelta(minutes=1))
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')
        assert _pending(tmp_path) == []

        clock.advance(REPAGE_COOLDOWN)
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        [page] = _pending(tmp_path)
        state = alarm.state()
        assert state is not None
        assert state.escalation_id == page['id']
        assert state.first_seen == _FIRST_SEEN + timedelta(minutes=1)


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


def _close(tmp_path, page_id: str) -> None:
    from escalation.queue import EscalationQueue

    EscalationQueue(escalation_queue_dir(tmp_path)).resolve(
        page_id, 'acknowledged, topping up', resolution_class='actionable',
    )


@contextlib.contextmanager
def _unopenable_queue(tmp_path) -> Iterator[None]:
    """Set the queue directory aside and put a file in its place meanwhile."""
    queue_dir = escalation_queue_dir(tmp_path)
    aside = tmp_path / 'queue-set-aside'
    queue_dir.rename(aside)
    queue_dir.write_text('')
    try:
        yield
    finally:
        queue_dir.unlink()
        aside.rename(queue_dir)


class TestAPageClosedWhileCreditsAreStillOut:
    """A latch must not keep claiming a page a human has already closed."""

    @pytest.mark.asyncio
    async def test_a_page_resolved_while_still_exhausted_re_pages_after_the_cooldown(
        self, tmp_path,
    ):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        [first] = _pending(tmp_path)
        _close(tmp_path, first['id'])

        clock.advance(timedelta(minutes=1))
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')
        assert _pending(tmp_path) == []

        clock.advance(REPAGE_COOLDOWN)
        await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        [second] = _pending(tmp_path)
        assert second['id'] != first['id']
        state = alarm.state()
        assert state is not None
        assert state.escalation_id == second['id']
        assert state.first_seen == _FIRST_SEEN
        assert state.first_site == 'durable_queue:add_episode'
        assert _FIRST_SEEN.isoformat() in second['summary']

    @pytest.mark.asyncio
    async def test_concurrent_observations_after_the_cooldown_re_page_once(
        self, tmp_path, caplog,
    ):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        [first] = _pending(tmp_path)
        _close(tmp_path, first['id'])
        clock.advance(REPAGE_COOLDOWN)
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger=_ALARM_LOGGER):
            await asyncio.gather(*(
                alarm.record_exhausted(_exhaustion(), site=f'durable_queue:op{n}')
                for n in range(5)
            ))

        [second] = _pending(tmp_path)
        # One re-check, not five that each fold into the page the first filed.
        [latch_move] = [
            r.getMessage() for r in caplog.records
            if r.name == _ALARM_LOGGER and first['id'] in r.getMessage()
        ]
        assert second['id'] in latch_move

    @pytest.mark.asyncio
    async def test_a_re_check_that_cannot_read_the_queue_keeps_the_page_and_retries(
        self, tmp_path,
    ):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        [first] = _pending(tmp_path)
        _close(tmp_path, first['id'])
        clock.advance(REPAGE_COOLDOWN)

        with _unopenable_queue(tmp_path):
            await alarm.record_exhausted(_exhaustion(), site='mem0_write')
        state = alarm.state()
        assert state is not None
        assert state.escalation_id == first['id']

        await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        [second] = _pending(tmp_path)
        assert second['id'] != first['id']
        state = alarm.state()
        assert state is not None
        assert state.escalation_id == second['id']

    @pytest.mark.asyncio
    async def test_a_page_still_pending_after_the_cooldown_is_not_re_filed(self, tmp_path):
        clock = _Clock()
        alarm = _alarm(tmp_path, clock=clock)
        await alarm.record_exhausted(_exhaustion(), site='durable_queue:add_episode')
        state = alarm.state()
        assert state is not None
        paged = state.escalation_id

        for _ in range(3):
            clock.advance(REPAGE_COOLDOWN)
            await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        assert len(_records(tmp_path)) == 1
        state = alarm.state()
        assert state is not None
        assert state.escalation_id == paged

    @pytest.mark.asyncio
    async def test_a_relatched_page_closed_after_a_restart_re_pages_at_once(self, tmp_path):
        clock = _Clock()
        await _alarm(tmp_path, clock=clock).record_exhausted(
            _exhaustion(), site='durable_queue:add_episode',
        )
        [first] = _pending(tmp_path)

        restarted = _alarm(tmp_path, clock=clock)
        await restarted.restore()
        _close(tmp_path, first['id'])
        await restarted.record_exhausted(_exhaustion(), site='mem0_write')

        [second] = _pending(tmp_path)
        assert second['id'] != first['id']
        state = restarted.state()
        assert state is not None
        assert state.escalation_id == second['id']


class TestNoProjectRoot:
    @pytest.mark.asyncio
    async def test_a_missing_root_warns_and_never_raises(self, caplog):
        alarm = BillingExhaustionAlarm(resolve_project_root=lambda: None)

        with caplog.at_level(logging.DEBUG):
            await alarm.record_exhausted(_exhaustion(), site='mem0_write')

        assert any(
            r.levelno == logging.WARNING and r.name == _ALARM_LOGGER
            for r in caplog.records
        )
