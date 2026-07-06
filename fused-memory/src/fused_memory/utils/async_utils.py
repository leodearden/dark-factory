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

  Pass 2 (choose a named policy, or implement inline for a bespoke site):
      Handle regular Exception subclasses according to local semantics. Two
      shared, named Pass-2 policies cover the common cases:
        - :func:`gather_or_raise`: log every captured Exception at WARNING,
          then raise the positionally-first one (the ``MemoryService.get_entity``
          semantics).
        - :func:`gather_collect`: return the per-item value-or-Exception list
          for the caller to classify (the ``GraphitiBackend.rebuild_entity_summaries``
          / ``ContextAssembler.assemble`` / best-effort-sweep semantics).

  For a rare bespoke site that doesn't fit either named policy,
  :func:`propagate_cancellations` remains exported so Pass 1 can still be
  called directly ahead of hand-rolled Pass-2 logic.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Iterable, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def propagate_cancellations(results: Sequence[Any]) -> None:  # Sequence, not list: callers may wrap/slice gather's return value
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


async def gather_collect(coros: Iterable[Awaitable[Any]]) -> list:
    """Gather *coros*, then return per-item value-or-Exception for the caller.

    This is the "collect" Pass-2 policy: after Pass 1
    (:func:`propagate_cancellations`) confirms no bare BaseException (cancellation
    signal) was captured, the raw per-item results — values and captured
    Exceptions alike — are returned as-is for the caller to classify (e.g. via
    ``isinstance(r, Exception)`` accumulation). This matches the semantics used
    by ``GraphitiBackend.rebuild_entity_summaries``, ``ContextAssembler.assemble``,
    and best-effort sweep sites.

    Args:
        coros: An iterable of awaitables to gather.

    Returns:
        The list returned by ``asyncio.gather(*coros, return_exceptions=True)``,
        in input order — each element is either the coroutine's result value or
        the Exception it raised.

    Raises:
        BaseException: If any coroutine raises a bare BaseException (cancellation
            signal), it is re-raised via Pass 1 (:func:`propagate_cancellations`)
            before this function returns, i.e. before any Pass-2 accumulation.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    propagate_cancellations(results)
    return results


async def gather_or_raise(
    coros: Iterable[Awaitable[Any]],
    *,
    label: str,
    logger: logging.Logger = logger,
) -> list:
    """Gather *coros*; log every captured Exception, then raise the first.

    This is the "log-all-then-raise-first" Pass-2 policy (the
    ``MemoryService.get_entity`` semantics): after Pass 1
    (:func:`propagate_cancellations`) confirms no bare BaseException
    (cancellation signal) was captured, every captured ordinary Exception is
    logged at WARNING — so no failure is silently dropped — and then the
    positionally-first Exception (input/gather order) is raised.

    Args:
        coros: An iterable of awaitables to gather.
        label: Required operation name, prefixed onto each WARNING log line
            (e.g. ``'get_entity'``), so call sites remain identifiable in logs.
        logger: Logger to use for the WARNING lines. Defaults to this module's
            logger; inject a different logger for deterministic test capture.

    Returns:
        The list returned by ``asyncio.gather(*coros, return_exceptions=True)``,
        in input order, if no Exception was captured.

    Raises:
        BaseException: If any coroutine raises a bare BaseException (cancellation
            signal), it is re-raised via Pass 1 (:func:`propagate_cancellations`)
            before this function returns, i.e. before any Pass-2 logging.
        Exception: The positionally-first captured Exception, once every
            captured Exception has been logged at WARNING.
    """
    results = await asyncio.gather(*coros, return_exceptions=True)
    propagate_cancellations(results)
    first_exc = next((r for r in results if isinstance(r, Exception)), None)
    if first_exc is not None:
        for r in results:
            if isinstance(r, Exception):
                logger.warning('%s: %s: %s', label, type(r).__name__, r)
        raise first_exc
    return results
