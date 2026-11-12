"""The pause an arm-harness retry or poll takes, injected so a test drives time without waiting."""

from collections.abc import Awaitable, Callable

Sleep = Callable[[float], Awaitable[object]]
