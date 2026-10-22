"""Zero-argument test doubles that return a fixed value and record the thread of every call, one of them blocking until released."""

from __future__ import annotations

import asyncio
import threading
from typing import Generic, TypeVar

T = TypeVar('T')

_CEILING_SECS = 30.0


class ThreadRecordingCall(Generic[T]):
    def __init__(self, returns: T) -> None:
        self.returns = returns
        self.threads: list[int] = []

    @property
    def calls(self) -> int:
        return len(self.threads)

    def __call__(self) -> T:
        self.threads.append(threading.get_ident())
        return self.returns


class BlockingCall(ThreadRecordingCall[T]):
    def __init__(self, returns: T) -> None:
        super().__init__(returns)
        self._entered = threading.Event()
        self._released = threading.Event()
        self._returned = threading.Event()

    def __call__(self) -> T:
        returns = super().__call__()
        self._entered.set()
        self._released.wait(timeout=_CEILING_SECS)
        self._returned.set()
        return returns

    def release(self) -> None:
        self._released.set()

    @property
    def blocked(self) -> bool:
        return self._entered.is_set() and not self._returned.is_set()

    async def entered(self) -> None:
        if not await asyncio.to_thread(self._entered.wait, _CEILING_SECS):
            raise AssertionError(f'no call entered within {_CEILING_SECS:g}s')
