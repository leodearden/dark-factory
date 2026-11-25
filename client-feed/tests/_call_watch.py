"""Count the calls to one function, from every thread, while a block is open.

Calls are matched by code object, so a call through any binding of the function counts, not only one through
the name a test happens to patch.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import FrameType, FunctionType
from typing import Any

_Profiler = Callable[[FrameType, str, Any], object]


class Calls:
    """How many times the watched function was entered."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def _counting(self, function: FunctionType, previous: _Profiler | None) -> _Profiler:
        code = function.__code__

        def profile(frame: FrameType, event: str, arg: Any) -> None:
            if event == 'call' and frame.f_code is code:
                with self._lock:
                    self._count += 1
            if previous is not None:
                previous(frame, event, arg)

        return profile


@contextmanager
def watching(function: FunctionType) -> Iterator[Calls]:
    """Count `function`'s calls on this thread and on every thread started until the block exits."""
    calls = Calls()
    previous_sys, previous_threading = sys.getprofile(), threading.getprofile()
    sys.setprofile(calls._counting(function, previous_sys))
    threading.setprofile(calls._counting(function, previous_threading))
    try:
        yield calls
    finally:
        sys.setprofile(previous_sys)
        threading.setprofile(previous_threading)
