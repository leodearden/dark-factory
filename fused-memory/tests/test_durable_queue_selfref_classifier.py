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

import ast
import json
import logging
import uuid as uuid_mod
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import poll_until

# The REAL upstream classes — see the module docstring for why this import is
# correct here and deliberately absent from test_durable_queue.py.
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError

import fused_memory.services.durable_queue as dq_module
from fused_memory.services.durable_queue import DurableWriteQueue, QueueItem


def _uuid() -> str:
    """A fresh uuid4 string, as ``MemoryService`` mints for each episode
    (``memory_service.py:2516``)."""
    return str(uuid_mod.uuid4())


# A FIXED uuid, used ONLY inside ``@pytest.mark.parametrize`` lists. A
# parametrized string becomes part of the test id, and pytest-xdist aborts a run
# whose workers collect different ids — so a fresh uuid4 there would be a
# flake generator. Every assertion that pins upstream's FORMAT still uses a
# fresh ``_uuid()`` built through the real exception; these entries only pin
# shapes that must NOT match, where the uuid's value is irrelevant.
_FIXED_UUID = '00000000-0000-4000-8000-000000000000'


def _enqueued_operations() -> set[str]:
    """Operation strings ``MemoryService`` actually enqueues, read from the
    producer's SOURCE rather than restated here.

    Parses ``memory_service.py`` (a sibling of the module under test) and
    collects the literal ``operation=`` keyword of every
    ``self.durable_queue.enqueue(...)`` call. Deliberately a partial view: a
    producer expressed some other way — e.g. the dict-shaped batch enqueue at
    ``memory_service.py:2463`` that supplies ``mem0_classify_and_add`` — is not
    seen here, which is why the caller asserts a SUBSET and never equality.
    """
    source = Path(dq_module.__file__).with_name('memory_service.py').read_text()
    operations: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'enqueue'):
            continue
        target = func.value
        if not (isinstance(target, ast.Attribute) and target.attr == 'durable_queue'):
            continue
        for kw in node.keywords:
            if kw.arg != 'operation':
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                operations.add(kw.value.value)
    return operations


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


def _item(
    operation: str,
    payload,
    *,
    item_id: int = 1,
    group_id: str = 'proj1',
    attempts: int = 0,
    max_attempts: int = 5,
) -> QueueItem:
    """A QueueItem built straight from a row tuple — no DB needed.

    The 12-tuple order is QueueItem.__slots__ (durable_queue.py:114-118):
    id, group_id, operation, payload, callback_type, status, attempts,
    max_attempts, next_retry_at, created_at, completed_at, error.
    *payload* may be a dict (serialised here, as enqueue does) or a raw string,
    so the malformed-JSON cases can be expressed directly.
    """
    payload_text = payload if isinstance(payload, str) else json.dumps(payload)
    return QueueItem((
        item_id, group_id, operation, payload_text, None, 'in_flight',
        attempts, max_attempts, 0.0, 0.0, None, None,
    ))


def _queue(tmp_path, execute, **overrides) -> DurableWriteQueue:
    """The shared config, matching ``test_durable_queue.py:1571-1586`` exactly
    so attempt counts here are directly comparable to task 3585's: plain budget
    5, extended transient budget 12, sub-second backoff."""
    # Annotated: without it pyright joins the heterogeneous literal values into
    # a single narrow type and rejects the ** unpacking below.
    kwargs: dict[str, Any] = dict(
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
            f'Node {_FIXED_UUID} not found',  # wrong case, well-formed uuid
            'NODE x not found',
            'node x not found.',        # trailing punctuation breaks the anchor
            'prefix node x not found',  # unanchored prefix
            'node x not found suffix',  # unanchored suffix
            'node x not found\n',       # $ would match here; the pattern uses \Z
            'nodes x not found',
            'edgy x not found',
        ],
    )
    def test_non_matching_shapes_return_none(self, message):
        assert dq_module._parse_not_found_uuid(message) is None, (
            f'{message!r} must not parse as the pinned not-found format; '
            f'got {dq_module._parse_not_found_uuid(message)!r}.'
        )

    def test_trailing_newline_does_not_match(self):
        """The anchoring claim, enforced against the REAL message text.

        Python's ``$`` matches immediately before a single trailing newline, so
        a pattern anchored with ``$`` would accept ``'node <uuid> not found\\n'``
        and hand back the uuid — treating a message that is NOT the pinned shape
        as proof of permanent failure. That is the fail-CLOSED direction, the one
        that discards a write. ``\\Z`` is what makes "fully anchored" true.
        """
        u = _uuid()
        message = str(NodeNotFoundError(u)) + '\n'
        assert dq_module._parse_not_found_uuid(message) is None, (
            f'{message!r} must not parse: it is not exactly the pinned shape, '
            f'and the fail-open invariant forbids concluding "permanent" from '
            f'anything else. Use \\Z, not $, to anchor _NOT_FOUND_MESSAGE_RE.'
        )


class TestIdentityUuidResolution:
    """``_identity_uuid`` resolves the uuid the operation exists to CREATE —
    deliberately not "any uuid appearing in the payload".

    A payload may legitimately REFERENCE other nodes' uuids, and those genuinely
    can be transiently invisible or belong to another graph (the task's causes 3
    and 4), so they must keep their retry budget. Only the operation's own
    minted identity licenses the "retrying cannot possibly help" conclusion.
    """

    def test_every_mapped_operation_is_one_the_producer_enqueues(self):
        """Each key must name an operation ``MemoryService`` really enqueues.

        Deliberately DERIVED from the producer's source rather than restated: an
        assertion like ``set(map) == {'add_episode'}`` would only echo the
        constant it is testing, and would be edited to match whatever the
        constant became. This one goes RED if a key is misspelled or names an
        operation the producer stopped enqueueing — the drift that would make
        the rule silently dead.

        A SUBSET, not equality: the map claiming fewer operations than exist is
        the safe direction (those fall open to the ordinary policy), and
        ``_enqueued_operations`` sees only the keyword-argument enqueue sites.
        """
        enqueued = _enqueued_operations()
        mapped = set(dq_module.DEFAULT_IDENTITY_PAYLOAD_KEYS)
        assert mapped <= enqueued, (
            f'DEFAULT_IDENTITY_PAYLOAD_KEYS names {sorted(mapped - enqueued)}, '
            f'which no self.durable_queue.enqueue(operation=...) call site in '
            f'memory_service.py produces (found: {sorted(enqueued)}). Either the '
            f'key is stale/misspelled — in which case the permanent rule can '
            f'never fire and the doomed-retry loop is back — or the producer now '
            f'enqueues it some other way, in which case update '
            f'_enqueued_operations().'
        )

    def test_resolves_the_add_episode_identity(self, tmp_path):
        u = _uuid()
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_episode', {'uuid': u, 'content': 'c', 'group_id': 'g'})
        assert q._identity_uuid(item) == u

    def test_unmapped_operation_returns_none_even_with_a_uuid_key(self, tmp_path):
        """The MAP decides, not the key name.

        add_memory_graphiti has no caller-minted identity, so a 'uuid' key
        appearing in its payload would be a REFERENCE to some other node — a
        node that genuinely may be in flight or in another graph.
        """
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_memory_graphiti', {'uuid': _uuid(), 'content': 'c'})
        assert q._identity_uuid(item) is None, (
            'An unmapped operation must resolve to None even when its payload '
            'happens to contain a uuid key.'
        )

    @pytest.mark.parametrize(
        'payload',
        [
            '{not json',        # unparseable
            '[]',               # parses, but not a dict
            '"str"',            # parses, but not a dict
            '123',              # parses, but not a dict
            'null',             # parses, but not a dict
            '{}',               # dict, key missing
            '{"content": "c"}',  # dict, key missing
            '{"uuid": null}',
            '{"uuid": ""}',     # empty string is not an identity
            '{"uuid": 123}',    # non-str
            '{"uuid": {"a": 1}}',
            '{"uuid": ["u"]}',
        ],
    )
    def test_fails_open_on_unusable_payloads(self, tmp_path, payload):
        """Every unusable shape resolves to None, which falls back to today's
        policy. Failing open here is the safe direction."""
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_episode', payload)
        assert q._identity_uuid(item) is None, (
            f'payload {payload!r} must resolve to None; got '
            f'{q._identity_uuid(item)!r}.'
        )

    def test_constructor_override_fully_replaces_the_default(self, tmp_path):
        """identity_payload_keys= is the test seam and the generic-component
        property: it REPLACES the default map rather than extending it."""
        u = _uuid()
        q = _queue(
            tmp_path, AsyncMock(),
            identity_payload_keys={'custom_op': 'node_id'},
        )
        custom = _item('custom_op', {'node_id': u, 'content': 'c'})
        assert q._identity_uuid(custom) == u

        episode = _item('add_episode', {'uuid': _uuid(), 'content': 'c'})
        assert q._identity_uuid(episode) is None, (
            'An explicit identity_payload_keys= must fully replace the default '
            'map, not merge with it.'
        )

    def test_empty_map_disables_the_rule_entirely(self, tmp_path):
        """``identity_payload_keys={}`` is the documented OFF-SWITCH.

        durable_queue.py's REINSTATEMENT CONDITION tells a future operator that
        this is how the permanent rule is disabled if Graphiti ever moves to a
        clustered, read-routed backend where a self-referential not-found could
        be genuinely transient. An advertised escape hatch that nothing
        exercises is an escape hatch that quietly stops working, so pin it.
        """
        q = _queue(tmp_path, AsyncMock(), identity_payload_keys={})
        item = _item('add_episode', {'uuid': _uuid(), 'content': 'c'})
        assert q._identity_uuid(item) is None, (
            'identity_payload_keys={} must express "no operation has a '
            'self-identity"; {} is falsy but is NOT None, so the constructor '
            'must not silently fall back to the default map.'
        )
        assert q._classify_failure(item, NodeNotFoundError('x')) == (
            'normal', item.max_attempts
        ), 'with the map emptied, every failure returns to the ordinary policy.'


class TestClassifyFailure:
    """``_classify_failure(item, exc) -> (classification, attempts_limit)``.

    Permanent maps to a limit of 1 rather than a separate boolean, so the call
    site keeps ``died = new_attempts >= limit`` as one uniform expression — and
    correctly re-kills an item arriving with ``attempts > 0``, which
    ``_recover_in_flight`` produces: crash recovery returns an ``in_flight`` row
    to ``pending`` while PRESERVING its attempt count. (``replay_dead`` zeroes
    ``attempts``; what it re-supplies is the identical poison payload.)
    """

    @pytest.mark.parametrize('exc_class', [NodeNotFoundError, EdgeNotFoundError])
    def test_self_referential_not_found_is_permanent_at_attempt_one(
        self, tmp_path, exc_class
    ):
        """The proof: the node reported missing IS the node this write exists to
        create, so no number of retries can make it appear.

        Parametrized over BOTH upstream classes because ``_NOT_FOUND_MESSAGE_RE``
        accepts ``node`` and ``edge`` alike; without the edge case a regression
        narrowing the classifier to nodes only would stay green here, caught
        solely at the parser level.
        """
        u = _uuid()
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_episode', {'uuid': u, 'content': 'c', 'group_id': 'proj1'})
        assert q._classify_failure(item, exc_class(u)) == ('permanent', 1), (
            f'{exc_class.__name__}({u!r}) names this item\'s own identity uuid '
            f'and must classify permanent at attempt 1.'
        )

    def test_referenced_uuid_not_found_keeps_the_full_budget(self, tmp_path):
        """CONTROL — the rule keys off the IDENTITY field, not off "a uuid
        appearing somewhere in the payload".

        Here the missing uuid is one the payload merely references. That node
        may be in another graph or still in flight (the task's causes 3 and 4),
        both of which retrying can genuinely resolve.
        """
        identity, referenced = _uuid(), _uuid()
        q = _queue(tmp_path, AsyncMock())
        item = _item(
            'add_episode',
            {'uuid': identity, 'referenced_uuid': referenced, 'content': 'c'},
        )
        assert q._classify_failure(item, NodeNotFoundError(referenced)) == (
            'normal', item.max_attempts
        ), (
            'A not-found naming a uuid the payload merely REFERENCES must keep '
            'its retry budget — only the identity uuid is provably doomed.'
        )

    def test_transient_error_keeps_the_extended_budget(self, tmp_path):
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_episode', {'uuid': _uuid(), 'content': 'c'})
        assert q._classify_failure(item, ConnectionResetError('reset')) == (
            'transient', 12
        )

    def test_unrelated_error_is_normal(self, tmp_path):
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_episode', {'uuid': _uuid(), 'content': 'c'})
        assert q._classify_failure(item, RuntimeError('boom')) == ('normal', 5)

    def test_unrecognised_message_falls_open_at_the_policy_layer(self, tmp_path):
        """Fail-open is enforced at the POLICY layer, not just in the parser.

        This is fused-memory's own not-found text (graphiti_client.py:1983),
        naming the very identity uuid — yet it must still get the plain budget,
        because the message is not the pinned format we can reason about.
        """
        u = _uuid()
        q = _queue(tmp_path, AsyncMock())
        item = _item('add_episode', {'uuid': u, 'content': 'c'})
        assert q._classify_failure(item, Exception(f'Entity node not found: {u}')) == (
            'normal', 5
        )

    def test_permanent_wins_over_an_operator_readded_transient_name(self, tmp_path):
        """PRECEDENCE: permanent is decided BEFORE transient.

        transient_error_names is operator-tunable and test_config_schema.py
        deliberately permits ADDING names back (it asserts a superset, not
        equality). If transient were checked first, re-adding NodeNotFoundError
        would hand a provably-doomed write the extended 12-attempt budget,
        resurrecting exactly the loop this task exists to kill. The
        payload-derived proof must outrank the name-based guess.
        """
        u = _uuid()
        q = _queue(
            tmp_path, AsyncMock(),
            transient_error_names={'NodeNotFoundError'},
        )
        item = _item('add_episode', {'uuid': u, 'content': 'c'})
        assert q._is_transient(NodeNotFoundError(u)), (
            'precondition: the operator override really did make this name '
            'transient'
        )
        assert q._classify_failure(item, NodeNotFoundError(u)) == ('permanent', 1), (
            'An operator re-adding a not-found name must not be able to '
            'resurrect a retry budget for a provably self-referential failure.'
        )


class TestSelfReferentialNotFoundDeadLettersImmediately:
    """The acceptance signal, end-to-end through a real queue and real SQLite.

    Everything above tests the policy functions in isolation; this class asserts
    what an operator actually observes: the attempt count on the dead row, and
    how many times the backend was hit.
    """

    @pytest.mark.asyncio
    async def test_self_referential_not_found_dies_on_attempt_one(self, tmp_path):
        """ACCEPTANCE: one backend round-trip, then dead. Not five."""
        u = _uuid()
        execute = AsyncMock(side_effect=NodeNotFoundError(u))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={'uuid': u, 'name': 'ep', 'content': 'c', 'group_id': 'proj1'},
            )
            await _poll_until_dead(q, group_id='proj1', expected_dead=1, timeout=20.0)

            dead = await q.get_dead_items()
            assert len(dead) == 1
            assert dead[0]['attempts'] == 1, (
                'A not-found naming the episode\'s OWN uuid is unsatisfiable by '
                f'construction and must dead-letter on attempt 1; got '
                f'{dead[0]["attempts"]} — it is still burning the retry budget.'
            )
            stats = await q.get_stats(group_id='proj1')
            assert stats['counts'].get('dead', 0) == 1
            assert execute.await_count == 1, (
                'The backend must be hit exactly once — sparing the four futile '
                f'round-trips is the point; got {execute.await_count}.'
            )
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_referenced_uuid_not_found_keeps_the_full_budget(self, tmp_path):
        """CONTROL: a not-found naming a merely-REFERENCED uuid retries fully.

        Causes 3 (the node lives in another graph) and 4 (a concurrent write
        creating it is still in flight) are genuinely retryable, and this is the
        assertion that fails if the rule is ever widened to "any uuid in the
        payload".
        """
        identity, referenced = _uuid(), _uuid()
        execute = AsyncMock(side_effect=NodeNotFoundError(referenced))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={
                    'uuid': identity, 'referenced_uuid': referenced,
                    'name': 'ep', 'content': 'c', 'group_id': 'proj1',
                },
            )
            await _poll_until_dead(q, group_id='proj1', expected_dead=1, timeout=20.0)

            dead = await q.get_dead_items()
            assert dead[0]['attempts'] == 5, (
                'A not-found naming a REFERENCED uuid must keep the plain '
                f'max_attempts budget (5); got {dead[0]["attempts"]}.'
            )
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_unrecognised_message_falls_open_end_to_end(self, tmp_path):
        """FAIL-OPEN: fused-memory's own not-found text, naming the identity
        uuid, still gets the full budget — an unrecognised message degrades to
        today's behaviour, never to permanent failure."""
        u = _uuid()
        execute = AsyncMock(side_effect=Exception(f'Entity node not found: {u}'))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            await q.enqueue(
                group_id='proj1', operation='add_episode',
                payload={'uuid': u, 'name': 'ep', 'content': 'c', 'group_id': 'proj1'},
            )
            await _poll_until_dead(q, group_id='proj1', expected_dead=1, timeout=20.0)

            dead = await q.get_dead_items()
            assert dead[0]['attempts'] == 5, (
                'An unrecognised message shape must fall open to the ordinary '
                f'budget (5); got {dead[0]["attempts"]}.'
            )
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_unmapped_operation_keeps_the_full_budget(self, tmp_path):
        """An operation with no declared identity keeps its retries even when a
        'uuid' key happens to sit in its payload."""
        u = _uuid()
        execute = AsyncMock(side_effect=NodeNotFoundError(u))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            await q.enqueue(
                group_id='proj1', operation='add_memory_graphiti',
                payload={'uuid': u, 'content': 'c', 'group_id': 'proj1'},
            )
            await _poll_until_dead(q, group_id='proj1', expected_dead=1, timeout=20.0)

            dead = await q.get_dead_items()
            assert dead[0]['attempts'] == 5, (
                'An unmapped operation must keep the plain budget (5); got '
                f'{dead[0]["attempts"]}.'
            )
        finally:
            await q.close()


class TestDeadLetterLogNamesOperationAndGroup:
    """The dead-letter WARNING must name the operation, the group_id and the
    classification.

    These are the fields a triager needs first, and before this change they had
    to be re-read from SQLite via get_dead_items — a step that is impossible
    once the row has been deleted, which is precisely when the log line is all
    that survives.
    """

    @pytest.mark.asyncio
    async def test_dead_letter_warning_names_operation_and_group(
        self, tmp_path, caplog
    ):
        execute = AsyncMock(side_effect=RuntimeError('boom'))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            with caplog.at_level(
                logging.WARNING, logger='fused_memory.services.durable_queue'
            ):
                await q.enqueue(
                    group_id='proj-alpha', operation='add_episode',
                    payload={'content': 'c', 'group_id': 'proj-alpha', 'name': 'ep'},
                )
                await _poll_until_dead(
                    q, group_id='proj-alpha', expected_dead=1, timeout=20.0
                )

            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert any(
                'add_episode' in r.getMessage() and 'proj-alpha' in r.getMessage()
                for r in warnings
            ), (
                'expected the dead-letter WARNING to name both the operation and '
                'the group_id; got: '
                f'{[r.getMessage() for r in warnings]}'
            )
            # The pre-existing fields must survive the enrichment.
            assert any(
                '5 attempts' in r.getMessage() and 'RuntimeError' in r.getMessage()
                for r in warnings
            ), (
                'the attempt count and error text must still be present; got: '
                f'{[r.getMessage() for r in warnings]}'
            )
        finally:
            await q.close()

    @pytest.mark.asyncio
    async def test_dead_letter_warning_names_the_classification(
        self, tmp_path, caplog
    ):
        """"Doomed from attempt 1" must be distinguishable from "exhausted its
        budget" without a second lookup."""
        u = _uuid()
        execute = AsyncMock(side_effect=NodeNotFoundError(u))
        q = _queue(tmp_path, execute)
        await q.initialize()
        try:
            with caplog.at_level(
                logging.WARNING, logger='fused_memory.services.durable_queue'
            ):
                await q.enqueue(
                    group_id='proj-alpha', operation='add_episode',
                    payload={'uuid': u, 'content': 'c', 'group_id': 'proj-alpha'},
                )
                await _poll_until_dead(
                    q, group_id='proj-alpha', expected_dead=1, timeout=20.0
                )

            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert any('permanent' in r.getMessage() for r in warnings), (
                'expected the WARNING to label the classification "permanent" so '
                'a triager can tell a doomed write from an exhausted one; got: '
                f'{[r.getMessage() for r in warnings]}'
            )
        finally:
            await q.close()
