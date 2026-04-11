"""Async utility helpers for asyncio patterns used across fused-memory.

This module contains synchronous helpers that support asyncio callsites — the
helpers themselves do not await anything, but they are designed for use alongside
asyncio.gather() and related constructs.

Convention: two-tier check for asyncio.gather(return_exceptions=True) callsites
=================================================================================
When a caller uses ``asyncio.gather(*coros, return_exceptions=True)``, both
ordinary Exceptions and bare BaseExceptions (CancelledError, KeyboardInterrupt,
SystemExit) are captured as result values rather than raised immediately.

The standard two-tier guard pattern separates these concerns:

  Pass 1 (delegated to :func:`propagate_cancellations`):
      Re-raise the first bare BaseException (cancellation signal) found in the
      results.  This must happen BEFORE any per-result logging or accumulation,
      so that structured-concurrency shutdown signals are never silently swallowed.

  Pass 2 (implemented inline at each callsite):
      Handle regular Exception subclasses according to local semantics — which
      vary by site:
        - ``MemoryService.get_entity``:      log all Exceptions, raise the first.
        - ``GraphitiBackend.rebuild_entity_summaries``: accumulate as per-entity
          error detail entries, continue to next entity.
        - ``ContextAssembler.assemble``:     degrade each failed fetch to an empty
          context list, log a warning.

  Pass 2 is intentionally NOT shared because the semantics differ per site.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def propagate_cancellations(results: Sequence[Any]) -> None:
    """Re-raise the first bare BaseException found in *results*, if any.

    This is Pass 1 of the two-tier check convention for
    ``asyncio.gather(return_exceptions=True)`` callsites.  Call it immediately
    after gather returns, before any per-result logging or accumulation, to
    ensure that structured-concurrency signals (CancelledError, KeyboardInterrupt,
    SystemExit) are never silently converted into per-item errors.

    Args:
        results: The list (or other Sequence) returned by
            ``asyncio.gather(..., return_exceptions=True)``.  Non-exception
            values are ignored.

    Returns:
        None — always.  If a bare BaseException is found it is raised instead
        of returned.

    Raises:
        BaseException: The first element of *results* that is an instance of
            ``BaseException`` but NOT an instance of ``Exception``.  In CPython
            3.8+ this means: ``asyncio.CancelledError``, ``KeyboardInterrupt``,
            ``SystemExit``, and any other direct BaseException subclass that is
            not also an Exception subclass.

    Note on Python's exception hierarchy:
        In Python 3.8, ``asyncio.CancelledError`` was moved from being a
        subclass of ``Exception`` to a direct subclass of ``BaseException``.
        This makes the ``not isinstance(r, Exception)`` guard the correct way to
        distinguish cancellation signals from application-level errors — the same
        pattern used by asyncio internals (e.g. ``Task.__step``).
    """
    for r in results:
        if isinstance(r, BaseException) and not isinstance(r, Exception):
            raise r
