"""Shared timestamp parsing primitive — parse-or-warn with sortable fallback.

Provides :func:`parse_timestamp_or_warn`, the canonical unification target
for the "parse-ISO-timestamp-or-fall-back-safely" logic duplicated across the
escalation and dashboard packages.  Callers that migrate onto this primitive
get a uniform contract: always returns a tz-aware :class:`datetime`, never
silently drops a malformed record.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

__all__ = ['parse_timestamp_or_warn']

logger = logging.getLogger(__name__)


def parse_timestamp_or_warn(
    raw: object,
    *,
    fallback: datetime | None = None,
    context: str = '',
) -> tuple[datetime, bool]:
    """Parse an ISO-8601 timestamp string, normalizing naive results to UTC.

    Parameters
    ----------
    raw:
        The value to parse.  Must be a :class:`str` and a valid ISO-8601
        datetime for the success path.
    fallback:
        Caller-supplied sentinel returned on failure instead of the default
        ``datetime.min.replace(tzinfo=UTC)``.  When *None* (default) the
        module-level sortable floor sentinel is used.
    context:
        Free-form label included in the WARNING message on failure, e.g.
        ``'dedupe.find_parent'``.  Makes log triage easier in production.

    Returns
    -------
    tuple[datetime, bool]
        ``(dt, True)`` on success — *dt* is always tz-aware.
        ``(sentinel, False)`` on failure — WARNING is emitted with *context*
        and ``repr(raw)``.  The sentinel sorts *before* any real timestamp so
        that "oldest-first" folds always include malformed records rather than
        silently dropping them.
    """
    sentinel = fallback if fallback is not None else datetime.min.replace(tzinfo=UTC)
    ctx_tag = f' [{context}]' if context else ''

    if not isinstance(raw, str):
        logger.warning(
            'parse_timestamp_or_warn: non-string/None timestamp%s; got %r — using fallback',
            ctx_tag,
            raw,
        )
        return (sentinel, False)

    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning(
            'parse_timestamp_or_warn: unparseable timestamp%s; got %r — using fallback',
            ctx_tag,
            raw,
        )
        return (sentinel, False)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return (dt, True)
