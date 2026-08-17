"""Live-FalkorDB integration tests for the hardened fulltext query (task 3334).

Context: task 3334 / Graphiti durable-queue dead-letter 9950, which failed the
``add_episode`` write path with::

    RediSearch: Syntax error at offset 122 near note

Every other test in this task asserts against *strings*.  This file is the only
one that proves RediSearch actually **accepts** what we build — and, just as
importantly, that stock upstream still **rejects** the reproduction, so the
override is demonstrably load-bearing rather than decorative.

Skips automatically when FalkorDB is not reachable, and is deselected by the
default ``-m 'not integration'`` addopts.

WHY THE INDEX-READINESS BARRIER EXISTS (do not remove it)
--------------------------------------------------------
FalkorDB builds full-text indices **asynchronously**.  Querying before the index
is ready silently returns success for a query RediSearch would otherwise reject.
Measured while characterising this bug: a known-unparseable query issued
immediately after ``CALL db.idx.fulltext.createNodeIndex(...)`` across six fresh
graphs gave FAIL, OK, OK, FAIL, FAIL, FAIL — i.e. **2 of 6 runs falsely reported
success**.  ``CALL db.indexes()`` shows why: the ``status`` column reads
``'[Indexing] 1/1: UNDER CONSTRUCTION'`` for a short window after creation before
becoming ``'OPERATIONAL'``.

Without the barrier this file is a false-GREEN generator: the hardening test
would pass vacuously about a third of the time even if the fix were broken, and
the upstream-baseline test would flake red and get quarantined.  With the barrier
the same cases were stable 5/5.

The barrier itself lives in ``_fm_helpers.await_index_operational`` (task 3377
extracted it out of this file so every live-index module shares one
implementation); ``tests/test_falkor_index_barrier_guard.py`` enforces that this
module keeps routing through it and never re-forks a local copy.

Verified against FalkorDB module v41800 (4.18.0).
"""

from __future__ import annotations

import contextlib
import re

import pytest
import pytest_asyncio
from _fm_helpers import (
    FALKOR_HOST,
    FALKOR_PORT,
    await_index_operational,
    falkor_skipif,
    unique_graph_name,
)
from falkordb.asyncio import FalkorDB
from graphiti_core.driver.falkordb_driver import FalkorDriver

from fused_memory.backends.falkor_fulltext import escape_group_id
from fused_memory.backends.graphiti_client import _MultiTenantFalkorDriver

TEST_GRAPH: str = unique_graph_name('3334_falkor_fulltext')

TEST_GROUP_ID: str = 'dark_factory'

# The RediSearch parser names the token *preceding* the fault, so the reported
# word is an artefact of where the bad token happens to sit in the content.
# Matching it loosely is deliberate: pinning the literal ``note`` would couple
# this assertion to incidental content position rather than to the failure.
SYNTAX_ERROR_RE = re.compile(r'Syntax error at offset \d+ near \w+')

# Contents reproduced from the dead-letter and from characterising its siblings.
# Several reduce to the empty-query sentinel rather than an executable query;
# the test handles both outcomes explicitly.
ADVERSARIAL_CONTENT: list[str] = [
    'Please note _ is the Python throwaway variable',
    'for _ in range(10): pass  # note the idiom',
    r'See the note: \_ escaped underscore in markdown',
    'note `` empty code span',
    'the and of',
    '',
    '!!!',
    'café — naïve 日本語 note',
    ' '.join(f'word{i}' for i in range(200)),  # over-length → '' sentinel
]


pytestmark = [
    falkor_skipif(),
    pytest.mark.timeout(60),
    pytest.mark.integration,
]


@pytest_asyncio.fixture
async def live_fulltext_graph():
    """Throwaway graph with an OPERATIONAL fulltext index on Entity(name, group_id)."""
    client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
    with contextlib.suppress(Exception):
        stale = client.select_graph(TEST_GRAPH)
        await stale.delete()

    graph = client.select_graph(TEST_GRAPH)
    await graph.query(
        'CREATE (:Entity {name: $n, group_id: $g})',
        {'n': 'alpha note beta', 'g': TEST_GROUP_ID},
    )
    await graph.query(
        "CALL db.idx.fulltext.createNodeIndex('Entity', 'name', 'group_id')"
    )
    # MANDATORY — see module docstring. Without this the tests below are a
    # false-green generator.
    await await_index_operational(graph)
    try:
        yield graph
    finally:
        with contextlib.suppress(Exception):
            await graph.delete()
        with contextlib.suppress(Exception):
            await client.aclose()


async def _run_fulltext(graph, query: str):
    """Execute ``query`` against the fixture's fulltext index."""
    return await graph.query(
        'CALL db.idx.fulltext.queryNodes($label, $q) YIELD node RETURN node.name',
        {'label': 'Entity', 'q': query},
    )


def _hardened_driver() -> _MultiTenantFalkorDriver:
    """A driver instance for query-building only — no connection is opened."""
    return object.__new__(_MultiTenantFalkorDriver)


class TestHardenedQueryIsAcceptedByRediSearch:
    """The payoff assertion: RediSearch parses what we build, for hostile content."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('content', ADVERSARIAL_CONTENT)
    async def test_hardened_query_parses_for_adversarial_corpus(
        self, live_fulltext_graph, content: str
    ) -> None:
        """Every reproduced failure content must build a query RediSearch accepts.

        Content that legitimately reduces to the empty-query sentinel (``''``)
        is never executed — that is graphiti's documented "no query" path, and
        the callers short-circuit before touching the driver.
        """
        query = _hardened_driver().build_fulltext_query(content, [TEST_GROUP_ID], 128)
        if query == '':
            return
        await _run_fulltext(live_fulltext_graph, query)

    @pytest.mark.asyncio
    async def test_stock_upstream_driver_still_rejects_the_repro(
        self, live_fulltext_graph
    ) -> None:
        """BASELINE — proves the override is load-bearing, not decorative.

        Stock ``FalkorDriver.build_fulltext_query`` emits the lone ``_`` and
        RediSearch rejects the result.  If a future graphiti-core fixes this
        upstream, THIS test fails and tells us the override can be retired —
        which is the loud signal we want, rather than silently carrying a patch
        for a bug that no longer exists.
        """
        stock = object.__new__(FalkorDriver)
        broken = stock.build_fulltext_query(
            'Please note _ is the Python throwaway variable', [TEST_GROUP_ID], 128
        )
        assert ' _ ' in broken, (
            'upstream no longer emits the bare underscore — re-verify whether '
            f'this override is still needed (got {broken!r})'
        )
        with pytest.raises(Exception, match=SYNTAX_ERROR_RE) as exc_info:
            await _run_fulltext(live_fulltext_graph, broken)
        assert SYNTAX_ERROR_RE.search(str(exc_info.value))

    @pytest.mark.asyncio
    async def test_hardened_query_still_returns_matches(
        self, live_fulltext_graph
    ) -> None:
        """Hardening must not cost recall: the repro still matches the seeded node.

        Dropping the unparseable token has to leave a *working* query behind, not
        merely a parseable one — otherwise "fixed" would mean "silently returns
        nothing", which is the degradation this project's norms forbid.
        """
        query = _hardened_driver().build_fulltext_query(
            'Please note _ is the Python throwaway variable', [TEST_GROUP_ID], 128
        )
        result = await _run_fulltext(live_fulltext_graph, query)
        assert result.result_set, f'hardened query matched nothing: {query!r}'

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'token', ['_x', 'A_B_C', 'café', '日本語1', 'x—y', '```a```']
    )
    async def test_alnum_bearing_tokens_survive_and_parse(
        self, live_fulltext_graph, token: str
    ) -> None:
        """Tokens carrying an alphanumeric character are kept AND parse.

        This is the other half of the invariant: the predicate must not be so
        aggressive that it quietly strips legitimate (especially non-Latin)
        search terms.
        """
        query = _hardened_driver().build_fulltext_query(
            f'alpha {token} beta', [TEST_GROUP_ID], 128
        )
        assert token in query, f'{token!r} was dropped from {query!r}'
        await _run_fulltext(live_fulltext_graph, query)


class TestGroupFilterEscapingAgainstLiveParser:
    """Pin the secondary hardening against the real parser, not just a literal.

    Upstream's comment claims quoting the group_id handles "special characters
    like hyphens".  These tests measure that claim rather than trusting it.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize('gid', ['a-b', 'q"uote'])
    async def test_unescaped_group_id_is_rejected(
        self, live_fulltext_graph, gid: str
    ) -> None:
        """Bare quoting is NOT sufficient — this is the premise for escaping.

        Matching the specific parser error (rather than bare ``Exception``) keeps
        this honest: a connection blip or a typo'd procedure name would otherwise
        satisfy the assertion and let the premise rot unnoticed.
        """
        with pytest.raises(Exception, match=SYNTAX_ERROR_RE):
            await _run_fulltext(live_fulltext_graph, f'(@group_id:"{gid}") (alpha)')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('gid', ['a-b', 'q"uote', 'x\\y'])
    async def test_escaped_group_id_parses(
        self, live_fulltext_graph, gid: str
    ) -> None:
        """``escape_group_id`` makes each of the same group_ids parseable."""
        await _run_fulltext(
            live_fulltext_graph, f'(@group_id:"{escape_group_id(gid)}") (alpha)'
        )

    @pytest.mark.asyncio
    async def test_ordinary_group_id_is_unaffected_end_to_end(
        self, live_fulltext_graph
    ) -> None:
        """The common case still matches the seeded node after escaping.

        ``dark_factory`` round-trips unchanged, so escaping cannot have shifted
        behaviour for any group_id that exists today.
        """
        query = _hardened_driver().build_fulltext_query('note', [TEST_GROUP_ID], 128)
        assert f'(@group_id:"{TEST_GROUP_ID}")' in query
        result = await _run_fulltext(live_fulltext_graph, query)
        assert result.result_set
