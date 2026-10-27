"""The tests that test_async_mock_orphan_enforcement.py runs in a nested pytest session.

Not collected by the suite (no ``test_`` prefix): the first test deliberately
calls an AsyncMock without awaiting it.  ``_KEEP`` holds that orphan so it
survives to the test's teardown without depending on garbage-collector timing.
"""
from unittest.mock import AsyncMock

import pytest

_KEEP: list[object] = []


def test_leaves_an_async_mock_call_unawaited() -> None:
    _KEEP.append(AsyncMock(name='planted_notifier')())


@pytest.mark.asyncio
async def test_awaits_its_async_mock_call() -> None:
    assert await AsyncMock(name='awaited_notifier', return_value=7)() == 7
