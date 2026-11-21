"""The sanctioned fake clock: ``shared.testing_fake_clock.FakeClock`` and the ``fake_clock`` fixture."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from shared.testing_fake_clock import FakeClock


def _readings(clock: FakeClock) -> tuple[float, float, float, float]:
    return clock(), clock.monotonic(), clock.time(), clock.now


class TestFakeClock:
    def test_every_reading_starts_at_the_given_start(self):
        assert _readings(FakeClock(start=10.0)) == (10.0, 10.0, 10.0, 10.0)

    def test_the_default_start_is_zero(self):
        assert _readings(FakeClock()) == (0.0, 0.0, 0.0, 0.0)

    def test_advance_moves_every_reading(self):
        clock = FakeClock(start=10.0)
        clock.advance(2.5)
        assert _readings(clock) == (12.5, 12.5, 12.5, 12.5)

    def test_time_only_moves_forward(self):
        clock = FakeClock(start=12.5)
        with pytest.raises(ValueError, match='forward') as raised:
            clock.advance(-1)
        assert 'seconds=-1' in str(raised.value)
        assert 'now=12.5' in str(raised.value)
        assert clock.now == 12.5

    def test_sleep_advances_and_records_and_is_injectable_as_a_sleep_callable(self):
        clock = FakeClock()

        def retry(operation: Callable[[], bool], sleep: Callable[[float], None]) -> int:
            attempts = 1
            while not operation():
                sleep(0.5)
                attempts += 1
            return attempts

        outcomes = iter([False, False, True])
        assert retry(lambda: next(outcomes), clock.sleep) == 3
        clock.sleep(3)
        assert clock.sleeps == (0.5, 0.5, 3)
        assert clock.now == 4.0

    def test_a_negative_sleep_is_refused(self):
        clock = FakeClock()
        with pytest.raises(ValueError, match='forward'):
            clock.sleep(-0.1)
        assert clock.sleeps == ()

    async def test_async_sleep_advances_records_and_lets_a_ready_task_run(self):
        clock = FakeClock()
        ran: list[str] = []

        async def ready() -> None:
            ran.append('ready')

        task = asyncio.create_task(ready())
        await clock.async_sleep(4)
        assert ran == ['ready']
        assert clock.now == 4.0
        assert clock.sleeps == (4,)
        await task

    def test_it_is_a_time_provider(self):
        clock = FakeClock()

        def debounce(window: float, now: Callable[[], float]) -> Callable[[], bool]:
            last: list[float] = []

            def should_fire() -> bool:
                current = now()
                if last and current - last[-1] < window:
                    return False
                last.append(current)
                return True

            return should_fire

        should_fire = debounce(1.0, clock)
        assert should_fire()
        clock.advance(0.5)
        assert not should_fire()
        clock.advance(0.5)
        assert should_fire()


class TestFakeClockFixture:
    """Each requesting test gets its own clock, so neither assertion depends on test order."""

    def test_a_requester_gets_a_fake_clock_at_zero(self, fake_clock: FakeClock):
        assert isinstance(fake_clock, FakeClock)
        assert fake_clock.now == 0.0
        fake_clock.advance(5)

    def test_another_requester_does_not_share_its_state(self, fake_clock: FakeClock):
        assert fake_clock.now == 0.0
        fake_clock.advance(7)
