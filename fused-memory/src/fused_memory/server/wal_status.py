"""Shared bookkeeping for the periodic WAL-checkpoint loop.

A single module-level dict so both ``server/main.py`` (writer — periodic
checkpoint loop) and ``server/tools.py`` (reader — the ``get_wal_status``
MCP tool) can share state without a circular import. The dashboard polls
``get_wal_status`` to surface checkpoint health.

Schema, per entry::

    {
        'ts': iso8601 string of the most recent checkpoint attempt,
        'busy': int — non-zero means readers/writers blocked TRUNCATE,
        'log': int — WAL frames present at checkpoint time,
        'checkpointed': int — frames copied to main DB this pass,
        'detail': str | None — error message or per-project summary,
    }

A missing entry means the loop has not yet checkpointed that store (e.g.
within the first ``_CHECKPOINT_INTERVAL`` after process start). The
dashboard renders that as ``no_data`` rather than ``stale``.
"""

from __future__ import annotations

CHECKPOINT_STATUS: dict[str, dict[str, object]] = {}
