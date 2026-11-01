"""The ticket-status vocabulary and the answers resolve_ticket builds from it (task 6705)."""

from __future__ import annotations

import pytest
import pytest_asyncio

from fused_memory.config.schema import CuratorConfig
from fused_memory.middleware.curation_deferral import CurationRetryPolicy, DeferReason
from fused_memory.middleware.ticket_store import (
    OPEN_TICKET_STATUSES,
    TERMINAL_TICKET_STATUSES,
    ResolveReason,
    TicketStatus,
    TicketStore,
)


@pytest_asyncio.fixture
async def store(tmp_path):
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    yield ticket_store
    await ticket_store.close()


class TestTicketStatusVocabulary:
    def test_status_values(self):
        assert {s.value for s in TicketStatus} == {
            'pending', 'awaiting_curation', 'created', 'combined', 'refused', 'failed', 'cancelled',
        }

    def test_open_and_terminal_partition_the_statuses(self):
        assert OPEN_TICKET_STATUSES == frozenset({TicketStatus.PENDING, TicketStatus.AWAITING_CURATION})
        assert isinstance(OPEN_TICKET_STATUSES, frozenset)
        assert isinstance(TERMINAL_TICKET_STATUSES, frozenset)
        assert TERMINAL_TICKET_STATUSES == frozenset(TicketStatus) - OPEN_TICKET_STATUSES
        assert OPEN_TICKET_STATUSES.isdisjoint(TERMINAL_TICKET_STATUSES)

    def test_resolve_reason_values(self):
        assert {r.value for r in ResolveReason} == {
            'wait_timeout', 'wait_interrupted', 'store_unavailable', 'unknown_ticket',
            'ticket_store not configured',
        }

    @pytest.mark.asyncio
    async def test_a_stored_row_status_parses_into_the_enum(self, store):
        ticket = await store.submit('dark_factory', '{}')
        assert TicketStatus((await store.get(ticket))['status']) is TicketStatus.PENDING

        await store.mark_awaiting_curation(
            ticket,
            reason=DeferReason.ALL_ACCOUNTS_CAPPED,
            policy=CurationRetryPolicy.from_config(CuratorConfig()),
        )
        assert TicketStatus((await store.get(ticket))['status']) is TicketStatus.AWAITING_CURATION
