"""One reading of a Mem0 record's ``created_at``, shared by every script that orders or ages records."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def parse_created_at(created_at: Any) -> datetime | None:
    """The aware instant *created_at* names, or None when it is absent or not ISO 8601.

    A naive timestamp is read as UTC, never as the host's local time, so the
    same record ages the same way on every host.
    """
    if not isinstance(created_at, str) or not created_at:
        return None
    try:
        moment = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)
