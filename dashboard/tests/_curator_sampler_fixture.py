"""The stores and MCP leg that dashboard.data.metrics::_sample_curator reads, for tests.

``ACCOUNT_EVENTS_SCHEMA`` is the ``account_events`` table of
shared/src/shared/cost_store.py::_SCHEMA.  ``TICKETS_SCHEMA`` is a stand-in for
fused-memory/src/fused_memory/middleware/ticket_store.py::TABLE_SQL that keeps
only the columns the sampler selects, plus a key.
"""

from __future__ import annotations

import json

import httpx
from _dashboard_helpers import mcp_init_response, mcp_notify_response, mcp_tool_response

TICKETS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS tickets (
    id          TEXT PRIMARY KEY,
    project_id  TEXT,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    resolved_at TEXT
);
"""

ACCOUNT_EVENTS_SCHEMA = """\
CREATE TABLE IF NOT EXISTS account_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    account_name TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    project_id   TEXT,
    run_id       TEXT,
    details      TEXT,
    created_at   TEXT NOT NULL
);
"""


class ListTicketsHandler:
    """Mock MCP handler that returns a fixed count for list_tickets calls."""

    def __init__(self, count: int = 2, project_id: str = 'test-project'):
        self.count = count
        self.project_id = project_id
        self.calls: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        method = body.get('method', '')
        request_id = body.get('id', 1)

        if method == 'initialize':
            return mcp_init_response(request_id)
        if method.startswith('notifications/'):
            return mcp_notify_response()

        self.calls.append(body)
        return mcp_tool_response(
            {'count': self.count, 'tickets': [], 'project_id': self.project_id},
            request_id,
        )
