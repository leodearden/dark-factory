"""The instant a caller-supplied task tree was read, against which its claimant heartbeats are aged."""

import logging
from collections.abc import Mapping
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def resolve_tree_read_instant(
    fetched_at: datetime | None, *, event: str, extra: Mapping[str, object]
) -> datetime:
    """Return *fetched_at* when it is timezone-aware, otherwise wall clock.

    Whoever hands over a task tree also owes the instant it was read. Every
    ``heartbeat_at`` in the tree is aged against that instant (task 4115),
    because reconciliation reads the tree minutes before it uses it and the
    claimant heartbeat TTL is only 10 minutes. A missing instant, and a naive
    one whose timezone cannot be known, both fall back to ``now(UTC)``. That
    fallback brings back the pre-4115 ageing error, so it is logged at WARNING
    as *event*, with *extra* plus a ``reason`` of ``'not_supplied'`` or
    ``'naive_datetime'``.
    """
    if fetched_at is not None and fetched_at.tzinfo is not None:
        return fetched_at
    reason = 'not_supplied' if fetched_at is None else 'naive_datetime'
    logger.warning(event, extra={**extra, 'reason': reason})
    return datetime.now(UTC)
