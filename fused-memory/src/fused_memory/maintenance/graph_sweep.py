"""The best-effort per-graph loop shared by the read-only sweeps.

A graph whose read fails is recorded as a ``GraphFailure`` — an unknown, never a
zero — and the sweep moves on; ``BaseException`` (Ctrl-C, cancellation) is
deliberately not caught, so it still aborts the sweep.
"""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar('T')


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


async def read_each_graph(
    graph_keys: Iterable[str],
    read_one: Callable[[str], Awaitable[T]],
    *,
    what: str,
) -> tuple[tuple[T, ...], tuple[GraphFailure, ...]]:
    """Read every graph in *graph_keys* order; a failed read costs that graph alone.

    Args:
        graph_keys: The graphs to read, in the order to read them.
        read_one: Reads one graph by key.
        what: Names the sweep in the failure log line.

    Returns:
        The successful reads in key order, and one ``GraphFailure`` per graph
        whose read raised.
    """
    results: list[T] = []
    failures: list[GraphFailure] = []
    for key in graph_keys:
        try:
            results.append(await read_one(key))
        except Exception as exc:
            logger.error('%s failed for graph %r', what, key, exc_info=True)
            failures.append(GraphFailure(group_id=key, error=f'{type(exc).__name__}: {exc}'))
    return tuple(results), tuple(failures)
