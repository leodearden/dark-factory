"""The sqlite3 classification a reconciliation failure is recorded and logged
with, and the best-effort await that logs a bookkeeping operation's failure with it
instead of raising.

The sqlite3 module sets ``sqlite_errorname``/``sqlite_errorcode`` on the errors
it raises; plans/recon-sqlite-database-locked-rca-2026-09-16.md says what they
mean.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Mapping
from typing import TypeVar

_T = TypeVar('_T')


def sqlite_classification(exc: BaseException) -> dict[str, object]:
    """*exc*'s sqlite3 error name and code, each None for an exception the
    sqlite3 module did not raise."""
    return {
        'sqlite_errorname': getattr(exc, 'sqlite_errorname', None),
        'sqlite_errorcode': getattr(exc, 'sqlite_errorcode', None),
    }


def sqlite_log_fields(run_id: str, classified: Mapping[str, object]) -> str:
    """The ``run_id=… sqlite_errorname=… sqlite_errorcode=…`` fields of a
    classified failure line; *classified* is :func:`sqlite_classification`'s
    dict or a record that embeds it."""
    return (
        f"run_id={run_id} sqlite_errorname={classified['sqlite_errorname']} "
        f"sqlite_errorcode={classified['sqlite_errorcode']}"
    )


async def await_best_effort(
    operation: Awaitable[_T],
    *,
    what: str,
    run_id: str,
    logger: logging.Logger,
    level: int,
) -> _T | None:
    """Await *operation* and return its value; an Exception is logged on
    *logger* at *level* as ``<what> failed: …`` with its classification, and
    None is returned instead. Cancellation propagates."""
    try:
        return await operation
    except Exception as err:
        logger.log(
            level, '%s failed: %r (%s)',
            what, err, sqlite_log_fields(run_id, sqlite_classification(err)),
        )
    return None
