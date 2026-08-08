"""Payload-aware retry classification for the durable write queue (task 3586).

A not-found failure naming THE VERY UUID the queued operation exists to create
is provably unsatisfiable: no number of retries can make the backend find a node
that this write was supposed to bring into existence. Such an item must
dead-letter on attempt 1 rather than burn the full ``max_attempts`` budget (and,
after a ``replay_dead``, burn it again — the mechanism behind the recorded
55-retry item in esc-3561-3).

WHY THIS FILE IMPORTS THE REAL ``graphiti_core.errors``, WHERE
``test_durable_queue.py:1533-1545`` (task 3585) DELIBERATELY DOES NOT.
Those two positions are consistent, not contradictory — they assert different
things:

  * 3585 asserts class-NAME membership in ``DEFAULT_TRANSIENT_ERROR_NAMES``.
    ``_is_transient`` matches by ``__name__`` walked over ``type(exc).__mro__``,
    so a module-local class answering to the name genuinely IS the contract, and
    importing upstream would exercise the identical code path while adding the
    dependency the name-based scheme exists to avoid.

  * This file asserts the upstream MESSAGE TEXT — ``f'node {uuid} not found'``
    (graphiti_core 0.28.2, ``errors.py:54-59``) and ``f'edge {uuid} not found'``
    (``:22-27``). No module-local stand-in can supply that: a stand-in would
    only echo a literal typed here, so the pin would be worthless and would keep
    passing silently after an upstream reword. Every expectation below is built
    FROM the constructed exception (``str(NodeNotFoundError(u))``), never from a
    hand-copied string, so a graphiti-core upgrade that rewords the message
    turns these tests RED — the intended alarm. Production, meanwhile, degrades
    to today's ordinary retry policy, never to permanent failure.

The parse route is forced: ``NodeNotFoundError`` / ``EdgeNotFoundError`` expose
no ``.uuid`` attribute, no group_id and no node label. The message is the only
carrier of the uuid.

pytest runs with ``asyncio_mode='strict'``, so every async test needs an
explicit ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import asyncio  # noqa: F401  (used by later steps' e2e tests)
import logging  # noqa: F401  (used by the caplog assertions in step-09)
import uuid as uuid_mod
from unittest.mock import AsyncMock  # noqa: F401  (used by the e2e tests)

import pytest  # noqa: F401
import pytest_asyncio  # noqa: F401
from _fm_helpers import poll_until

# The REAL upstream classes — see the module docstring for why this import is
# correct here and deliberately absent from test_durable_queue.py.
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError

import fused_memory.services.durable_queue as dq_module
from fused_memory.services.durable_queue import DurableWriteQueue


def _uuid() -> str:
    """A fresh uuid4 string, as ``MemoryService`` mints for each episode
    (``memory_service.py:2516``)."""
    return str(uuid_mod.uuid4())


async def _poll_until_dead(
    q: DurableWriteQueue,
    *,
    group_id: str | None = None,
    expected_dead: int,
    timeout: float = 20.0,
) -> None:
    """Poll queue stats until at least *expected_dead* items reach status='dead'.

    Mirrors ``test_durable_queue.py:1207-1236``. Fixed sleeps are unsafe here:
    the suite runs under ``-n auto --dist loadgroup`` with a 60s per-test
    timeout that kills the whole worker, and the generous 20s budget means a
    regression fails on the observed COUNT rather than by timing out.
    """
    last_counts: dict = {}

    async def _dead_count_reached():
        stats = await q.get_stats(group_id=group_id)
        last_counts.update(stats['counts'])
        return stats['counts'].get('dead', 0) >= expected_dead

    try:
        await poll_until(_dead_count_reached, timeout=timeout, interval=0.05)
    except AssertionError as exc:
        raise AssertionError(
            f'Timed out waiting for {expected_dead} dead item(s) '
            f'(group_id={group_id!r}); last counts={last_counts}'
        ) from exc


def _queue(tmp_path, execute, **overrides) -> DurableWriteQueue:
    """The shared config, matching ``test_durable_queue.py:1571-1586`` exactly
    so attempt counts here are directly comparable to task 3585's: plain budget
    5, extended transient budget 12, sub-second backoff."""
    kwargs = dict(
        data_dir=tmp_path / 'queue',
        execute_write=execute,
        workers_per_group=1,
        semaphore_limit=5,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_delay_seconds=0.05,
        write_timeout_seconds=2.0,
        transient_max_attempts=12,
    )
    kwargs.update(overrides)
    return DurableWriteQueue(**kwargs)


class TestUpstreamNotFoundMessageFormat:
    """Pin graphiti_core's not-found message text (0.28.2, errors.py:22-27 and
    :54-59) by deriving every expectation FROM the constructed exception.

    Never assert against a hand-copied literal like ``f'node {u} not found'``:
    that would keep passing after an upstream reword while the parser silently
    stopped recognising the real thing. Building the input with
    ``str(NodeNotFoundError(u))`` makes such an upgrade turn this test RED,
    which is the whole point of the pin.
    """

    def test_parses_node_not_found_uuid(self):
        """NodeNotFoundError('<uuid>') -> 'node <uuid> not found' -> '<uuid>'."""
        u = _uuid()
        message = str(NodeNotFoundError(u))
        assert dq_module._parse_not_found_uuid(message) == u, (
            f'Failed to recover the uuid from graphiti_core '
            f'NodeNotFoundError({u!r}), whose message is {message!r}. If '
            f'graphiti-core reworded this, update _NOT_FOUND_MESSAGE_RE — the '
            f'exception carries no .uuid attribute, so the message is the only '
            f'carrier.'
        )

    def test_parses_edge_not_found_uuid(self):
        """EdgeNotFoundError('<uuid>') -> 'edge <uuid> not found' -> '<uuid>'."""
        u = _uuid()
        message = str(EdgeNotFoundError(u))
        assert dq_module._parse_not_found_uuid(message) == u, (
            f'Failed to recover the uuid from graphiti_core '
            f'EdgeNotFoundError({u!r}), whose message is {message!r}.'
        )


class TestParserFailsOpen:
    """Anything that is not EXACTLY the pinned format returns None.

    None routes the caller back to the ordinary retry policy, so an upstream
    reword — or any similarly-worded message from elsewhere — degrades to
    today's retry behaviour and NEVER to permanent failure. This is the
    direction the fail-open invariant mandates: a missed permanent failure
    costs four extra retries; a false permanent failure discards a write.
    """

    def test_fused_memory_own_not_found_messages_do_not_match(self):
        """fused-memory defines its OWN unrelated NodeNotFoundError
        (backends/graphiti_client.py:219). Its messages signal a CONFIRMED
        absence in a different lookup and must not be parsed as upstream's."""
        u = _uuid()
        for message in (
            f'Entity node not found: {u}',                        # :1086, :1983, :2006
            f'Episodic node not found in group g1: {u}',          # :876
            "No entity found with name: 'Foo'",                   # :2034
            f"Entity node not found in source_graph 'src': {u}",  # cross_graph_move.py:342
        ):
            assert dq_module._parse_not_found_uuid(message) is None, (
                f'fused-memory\'s own not-found text {message!r} must not parse '
                f'as graphiti_core\'s pinned format.'
            )

    @pytest.mark.parametrize(
        'message',
        [
            'uuid abc-123 not found',   # task 3585's test literal — must NOT match
            '',
            'node not found',           # no uuid at all
            'node a b not found',       # \S+ must not span the space
            'Node {u} not found',       # wrong case
            'NODE x not found',
            'node x not found.',        # trailing punctuation breaks the anchor
            'prefix node x not found',  # unanchored prefix
            'node x not found suffix',  # unanchored suffix
            'nodes x not found',
            'edgy x not found',
        ],
    )
    def test_non_matching_shapes_return_none(self, message):
        assert dq_module._parse_not_found_uuid(message) is None, (
            f'{message!r} must not parse as the pinned not-found format; '
            f'got {dq_module._parse_not_found_uuid(message)!r}.'
        )
