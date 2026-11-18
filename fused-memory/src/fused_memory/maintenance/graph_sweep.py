"""The best-effort per-graph loop shared by the read-only sweeps, and when a sweep is whole.

A graph whose read fails is recorded as a ``GraphFailure`` — an unknown, never a
zero — and the sweep moves on; ``BaseException`` (Ctrl-C, cancellation) is
deliberately not caught, so it still aborts the sweep. ``sweep_complete`` is the
one rule for whether a sweep's numbers can be trusted as a whole.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


class GraphResult(Protocol):
    """One graph's read, which knows whether it saw that graph whole."""

    @property
    def complete(self) -> bool: ...


@dataclass(frozen=True)
class GraphFailure:
    """A graph the sweep tried to read and could not.

    Recorded as structure rather than folded into a log line, because the two
    fields answer different operator questions: ``group_id`` is what a re-run
    is scoped to, and ``error`` names the exception TYPE as well as its message,
    which is what says whether to retry or to go and fix something.
    """

    group_id: str
    error: str


@dataclass(frozen=True)
class SweepReads(Generic[T]):
    """What ``read_each_graph`` read: the successes in key order, the failures, and the loop's duration."""

    results: tuple[T, ...]
    failures: tuple[GraphFailure, ...]
    elapsed_ms: float


def sweep_complete(results: Sequence[GraphResult], failures: Sequence[GraphFailure]) -> bool:
    """Whether a sweep's numbers can be trusted as a whole.

    False when any graph could not be read, when any read was not whole, or when
    no graph was read at all: a sweep over nothing proves nothing, so it must not
    answer like one that read every graph clean.
    """
    return bool(results) and not failures and all(result.complete for result in results)


async def read_each_graph(
    graph_keys: Iterable[str],
    read_one: Callable[[str], Awaitable[T]],
    *,
    what: str,
) -> SweepReads[T]:
    """Read every graph in *graph_keys* order; a failed read costs that graph alone.

    Args:
        graph_keys: The graphs to read, in the order to read them.
        read_one: Reads one graph by key.
        what: Names the sweep in the failure log line.

    Returns:
        The successful reads in key order, one ``GraphFailure`` per graph whose
        read raised, and how long the loop took.
    """
    start = time.monotonic()
    results: list[T] = []
    failures: list[GraphFailure] = []
    for key in graph_keys:
        try:
            results.append(await read_one(key))
        except Exception as exc:
            logger.error('%s failed for graph %r', what, key, exc_info=True)
            failures.append(GraphFailure(group_id=key, error=f'{type(exc).__name__}: {exc}'))
    return SweepReads(
        results=tuple(results),
        failures=tuple(failures),
        elapsed_ms=(time.monotonic() - start) * 1000,
    )
