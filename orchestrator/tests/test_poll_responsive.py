"""Public contract of ``_poll_responsive.poll_responsive``.

``poll_responsive`` polls a predicate until it returns something truthy, under
``wait_responsive``'s loop-responsive deadline. Pinned here: the first truthy
value comes back verbatim, the predicate is checked before any sleep, a give-up
is ``wait_responsive``'s labelled failure (with ``max_wall_s`` forwarded), a
predicate's own exception propagates as itself, and an async predicate is
rejected with a TypeError naming the label.

Hermetic and sub-second: real waits on tiny named bounds, no patched privates.
"""

from __future__ import annotations

import re

import pytest
from _poll_responsive import poll_responsive

SLICE = 0.02
BUDGET = 0.5
TINY_WALL = 0.05
WALL_CAP = 15.0


class _PredicateFailed(RuntimeError):
    """Sentinel raised by a predicate, to prove it propagates unwrapped."""


class _CountingPredicate:
    """Returns ``values[i]`` on its i-th call (the last value thereafter)."""

    def __init__(self, *values: object) -> None:
        self._values = values
        self.calls = 0

    def __call__(self) -> object:
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return value


def _reported(message: str, field: str) -> float:
    match = re.search(rf'{field}=([0-9.]+)s', message)
    assert match is not None, f'give-up message does not report {field}=: {message}'
    return float(match.group(1))


@pytest.mark.asyncio
class TestPollResponsiveReturnsTheFirstTruthyValue:
    async def test_an_already_truthy_predicate_is_called_once_and_returned_verbatim(
        self,
    ) -> None:
        ready = ['x']
        predicate = _CountingPredicate(ready)

        got = await poll_responsive(
            predicate, timeout=BUDGET, label='already-truthy', max_wall_s=WALL_CAP
        )

        assert got is ready, 'the truthy value must come back verbatim, not a copy'
        assert predicate.calls == 1, (
            f'an already-truthy predicate must be checked before any sleep and '
            f'not again; it was called {predicate.calls} times'
        )

    async def test_a_predicate_truthy_on_its_third_call_returns_that_value(
        self,
    ) -> None:
        predicate = _CountingPredicate(None, 0, 'third')

        got = await poll_responsive(
            predicate,
            timeout=BUDGET,
            label='third-call',
            interval=SLICE,
            max_wall_s=WALL_CAP,
        )

        assert got == 'third'
        assert predicate.calls == 3


@pytest.mark.asyncio
class TestPollResponsiveGivesUpThroughWaitResponsive:
    async def test_a_never_truthy_predicate_fails_loudly_by_label(self) -> None:
        with pytest.raises(pytest.fail.Exception) as excinfo:
            await poll_responsive(
                lambda: False,
                timeout=BUDGET,
                label='never-truthy-poll',
                interval=SLICE,
                max_wall_s=WALL_CAP,
            )

        message = str(excinfo.value)
        assert 'never-truthy-poll' in message, f'label missing from: {message}'
        assert 'loop-responsive budget was exhausted' in message, message

    async def test_max_wall_s_is_forwarded_as_the_hard_cap(self) -> None:
        with pytest.raises(pytest.fail.Exception) as excinfo:
            await poll_responsive(
                lambda: False,
                timeout=BUDGET * 100,
                label='tiny-wall',
                interval=SLICE,
                max_wall_s=TINY_WALL,
            )

        message = str(excinfo.value)
        assert 'hard wall-clock cap was reached' in message, message
        assert _reported(message, 'cap') == pytest.approx(TINY_WALL)


@pytest.mark.asyncio
class TestPollResponsivePropagatesPredicateErrors:
    async def test_a_predicate_exception_propagates_as_itself(self) -> None:
        def _explodes() -> bool:
            raise _PredicateFailed('boom')

        with pytest.raises(_PredicateFailed, match='boom'):
            await poll_responsive(
                _explodes, timeout=BUDGET, label='explodes', max_wall_s=WALL_CAP
            )


@pytest.mark.asyncio
class TestPollResponsiveRejectsAnAsyncPredicate:
    async def test_an_async_predicate_raises_type_error_naming_the_label(self) -> None:
        """A coroutine object is always truthy, so accepting one would return at
        once with the condition unchecked: a silent false green."""

        async def _async_ready() -> bool:
            return False

        with pytest.raises(TypeError, match='async-predicate') as excinfo:
            await poll_responsive(
                _async_ready, timeout=BUDGET, label='async-predicate', max_wall_s=WALL_CAP
            )

        assert 'must be synchronous' in str(excinfo.value), str(excinfo.value)
