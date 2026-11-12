"""_sample_curator's active time must survive how the gate actually logs caps.

An all-capped window that never closes makes every ticket's capped overlap
equal its raw time, so p50/p90/p99_active_ms read 0 (task 6540).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiosqlite
import httpx
import pytest
from _curator_sampler_fixture import ACCOUNT_EVENTS_SCHEMA, TICKETS_SCHEMA, ListTicketsHandler

from dashboard.config import DashboardConfig
from dashboard.data.memory import reset_sessions
from dashboard.data.metrics import _sample_curator

_ONE_MINUTE_TICKET_MS = 60_000


async def _sample_one_minute_ticket(
    tmp_path: Path,
    *,
    now: datetime,
    cap_events: Sequence[tuple[str, str, datetime, dict | None]],
    account_count: int,
) -> dict:
    """Run _sample_curator over real runs.db/tickets.db files.

    tickets.db holds one 'created' ticket, created now-30min and resolved
    now-29min.  The gate reports ``account_count`` configured accounts.
    """
    tickets_path = tmp_path / 'tickets.db'
    conn_sync = sqlite3.connect(str(tickets_path))
    conn_sync.executescript(TICKETS_SCHEMA)
    created = now - timedelta(minutes=30)
    resolved = created + timedelta(milliseconds=_ONE_MINUTE_TICKET_MS)
    conn_sync.execute(
        'INSERT INTO tickets (id, project_id, status, created_at, resolved_at) VALUES (?, ?, ?, ?, ?)',
        ('t1', 'proj', 'created', created.isoformat(), resolved.isoformat()),
    )
    conn_sync.commit()
    conn_sync.close()

    runs_path = tmp_path / 'runs.db'
    conn_sync = sqlite3.connect(str(runs_path))
    conn_sync.executescript(ACCOUNT_EVENTS_SCHEMA)
    for account, event_type, ts, details in cap_events:
        conn_sync.execute(
            'INSERT INTO account_events (account_name, event_type, details, created_at)'
            ' VALUES (?, ?, ?, ?)',
            (account, event_type, None if details is None else json.dumps(details), ts.isoformat()),
        )
    conn_sync.commit()
    conn_sync.close()

    config = DashboardConfig(
        project_root=tmp_path,
        fused_memory_urls=['http://localhost:18765'],
        known_project_roots=[tmp_path],
    )
    gate = {
        'paused': False,
        'paused_reason': None,
        'soonest_open_at': None,
        'account_count': account_count,
    }
    transport = httpx.MockTransport(ListTicketsHandler(count=0))

    reset_sessions()
    with patch('dashboard.data.metrics.get_curator_state', new=AsyncMock(return_value=gate)):
        async with httpx.AsyncClient(transport=transport) as http_client:
            tickets_conn = await aiosqlite.connect(str(tickets_path))
            tickets_conn.row_factory = aiosqlite.Row
            runs_conn = await aiosqlite.connect(str(runs_path))
            runs_conn.row_factory = aiosqlite.Row
            try:
                return await _sample_curator(
                    http_client, config, tickets_conn, [runs_conn], now=now
                )
            finally:
                await tickets_conn.close()
                await runs_conn.close()


@pytest.mark.asyncio
async def test_sample_curator_restart_redetected_caps_do_not_zero_active_ms(tmp_path: Path):
    """The production shape measured in runs.db on 2026-11-10.

    Each orchestrator restart re-logs cap_hit (same resets_at) for an account
    that is still capped, and one resume probe later writes ONE 'resumed'.
    acc-b's cap ends at its resets_at with no 'resumed' at all.  Both caps are
    over well before the ticket ran, so its full minute is active time.
    """
    now = datetime.now(UTC)
    reset = {'resets_at': (now - timedelta(minutes=90)).isoformat()}
    cap_events = []
    for account in ('acc-a', 'acc-b'):
        cap_events.append((account, 'cap_hit', now - timedelta(hours=3), reset))
        cap_events.append((account, 'cap_hit', now - timedelta(hours=2), reset))
    cap_events.append(('acc-a', 'resumed', now - timedelta(minutes=88), None))

    result = await _sample_one_minute_ticket(
        tmp_path, now=now, cap_events=cap_events, account_count=2
    )

    assert result['p50_active_ms'] == _ONE_MINUTE_TICKET_MS
    assert result['capped_now'] == 0


@pytest.mark.asyncio
async def test_sample_curator_unseen_configured_account_keeps_active_ms(tmp_path: Path):
    """Two of three configured accounts are genuinely capped; the third never was.

    capped_now and the all-capped windows must describe the same three-account
    universe, so neither reports 'every account capped' and the ticket's full
    minute stays active.
    """
    now = datetime.now(UTC)
    reset = {'resets_at': (now + timedelta(days=2)).isoformat()}
    cap_events = [
        (account, 'cap_hit', now - timedelta(hours=2), reset) for account in ('acc-a', 'acc-b')
    ]

    result = await _sample_one_minute_ticket(
        tmp_path, now=now, cap_events=cap_events, account_count=3
    )

    assert result['capped_now'] == 0
    assert result['p50_active_ms'] == _ONE_MINUTE_TICKET_MS
