"""A zero-argument test double that returns a fixed value and records the thread of every call."""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar('T')


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
