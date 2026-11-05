"""One served read: a cached, single-flight, bounded-wait read that keeps its last good value."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from dashboard.data.datum import Datum, DatumState, unknown_datum
from dashboard.data.served_read import ServedRead

T0 = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)
BOUND = 30


class Reads:
    """A read the test scripts: each call answers the next scripted Datum, optionally held."""

    def __init__(self, *answers: Datum[int]) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.release: asyncio.Event | None = None

    def hold(self) -> asyncio.Event:
        self.release = asyncio.Event()
        return self.release

    async def __call__(self) -> Datum[int]:
        self.calls += 1
        answer = self.answers.pop(0)
        if self.release is not None:
            await self.release.wait()
        return answer


def fresh(value: int, as_of: datetime = T0) -> Datum[int]:
    return Datum(value, as_of, DatumState.FRESH, None, BOUND)


def failed(reason: str) -> Datum[int]:
    return unknown_datum(reason, BOUND)


def make(ttl: list[float]) -> ServedRead[str, int]:
    return ServedRead(ttl_seconds=lambda: ttl[0], freshness_bound_seconds=BOUND)


async def test_a_fresh_read_is_served_as_of_its_own_measurement_and_cached_inside_its_ttl():
    served = make([60.0])
    read = Reads(fresh(1, as_of=T0 - timedelta(seconds=2)))

    first = await served.serve('root', read, now=T0)
    second = await served.serve('root', read, now=T0 + timedelta(seconds=1))

    assert first == fresh(1, as_of=T0 - timedelta(seconds=2))
    assert second == first
    assert read.calls == 1


async def test_concurrent_cold_serves_of_one_key_share_one_read():
    served = make([60.0])
    read = Reads(fresh(1), fresh(2))
    release = read.hold()

    pending = asyncio.gather(
        served.serve('root', read, now=T0), served.serve('root', read, now=T0),
    )
    await asyncio.sleep(0.01)
    release.set()

    assert await pending == [fresh(1), fresh(1)]
    assert read.calls == 1


async def test_a_failed_read_before_any_success_is_unknown_with_the_reads_own_reason():
    served = make([60.0])

    datum = await served.serve('root', Reads(failed('substrate offline')), now=T0)

    assert datum == unknown_datum('substrate offline', BOUND)


async def test_a_failed_read_after_a_success_serves_the_last_good_value_stale_with_its_reason():
    ttl = [60.0]
    served = make(ttl)
    read = Reads(fresh(1), failed('substrate offline'))
    await served.serve('root', read, now=T0)
    ttl[0] = 0.0

    datum = await served.serve('root', read, now=T0 + timedelta(seconds=5))

    assert datum == Datum(1, T0, DatumState.STALE, 'substrate offline', BOUND)
    assert read.calls == 2


async def test_a_lower_bound_read_is_served_and_kept_as_the_last_good_value():
    ttl = [60.0]
    served = make(ttl)
    partial = Datum(4, T0, DatumState.LOWER_BOUND, 'one table missing', BOUND)
    read = Reads(partial, failed('substrate offline'))

    assert await served.serve('root', read, now=T0) == partial
    ttl[0] = 0.0
    later = await served.serve('root', read, now=T0 + timedelta(seconds=5))

    assert later == Datum(4, T0, DatumState.STALE, 'substrate offline', BOUND)


async def test_a_read_slower_than_the_wait_is_unknown_before_any_success_and_lands_for_the_next_serve():
    served = make([60.0])
    read = Reads(fresh(1))
    release = read.hold()

    waited = await served.serve('root', read, now=T0, wait_seconds=0.05)
    release.set()
    landed = await served.serve('root', read, now=T0 + timedelta(seconds=1))

    assert waited.state is DatumState.UNKNOWN
    assert waited.value is None
    assert 'in flight' in (waited.reason or '')
    assert '0.05' in (waited.reason or '')
    assert landed == fresh(1)
    assert read.calls == 1


async def test_a_read_slower_than_the_wait_serves_the_last_good_value_stale_naming_the_wait():
    ttl = [60.0]
    served = make(ttl)
    read = Reads(fresh(1), fresh(2, as_of=T0 + timedelta(seconds=5)))
    await served.serve('root', read, now=T0)
    ttl[0] = 0.0
    release = read.hold()

    waited = await served.serve('root', read, now=T0 + timedelta(seconds=5), wait_seconds=0.05)
    ttl[0] = 60.0
    release.set()
    landed = await served.serve('root', read, now=T0 + timedelta(seconds=6))

    assert (waited.value, waited.as_of, waited.state) == (1, T0, DatumState.STALE)
    assert 'in flight' in (waited.reason or '') and '0.05' in (waited.reason or '')
    assert landed == fresh(2, as_of=T0 + timedelta(seconds=5))
    assert read.calls == 2


async def test_a_cached_value_past_its_bound_at_the_serving_instant_is_served_stale():
    served = make([60.0])
    read = Reads(fresh(1))
    await served.serve('root', read, now=T0)

    datum = await served.serve('root', read, now=T0 + timedelta(seconds=BOUND + 1))

    assert (datum.value, datum.as_of, datum.state) == (1, T0, DatumState.STALE)
    assert datum.reason
    assert read.calls == 1


async def test_clear_forgets_both_the_cached_value_and_the_last_good_value():
    served = make([60.0])
    read = Reads(fresh(1), failed('substrate offline'))
    await served.serve('root', read, now=T0)

    served.clear()
    datum = await served.serve('root', read, now=T0 + timedelta(seconds=1))

    assert datum == unknown_datum('substrate offline', BOUND)
    assert read.calls == 2


async def test_keys_are_served_independently():
    served = make([60.0])
    await served.serve('a', Reads(fresh(1)), now=T0)

    other = await served.serve('b', Reads(failed('b is down')), now=T0)
    same = await served.serve('a', Reads(failed('never asked')), now=T0)

    assert other == unknown_datum('b is down', BOUND)
    assert same == fresh(1)
