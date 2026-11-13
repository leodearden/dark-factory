"""The paginated, census-proven FalkorDB read primitive and its result vocabulary.

It imports no graphiti_core, so a read-only seam can page a whole-graph read
without loading the graphiti driver.
"""

import logging
from dataclasses import dataclass

__all__ = [
    'DEFAULT_READ_PAGE_SIZE',
    'INCOMPLETE_CENSUS_UNAVAILABLE',
    'INCOMPLETE_KINDS',
    'INCOMPLETE_PAGE_CAP',
    'INCOMPLETE_SHORT_READ',
    'INCOMPLETE_STRUCTURAL_KINDS',
    'INCOMPLETE_STRUCTURAL_REFUSAL',
    'MAX_READ_PAGES',
    'RESULTSET_SIZE',
    'PagedRead',
    'census_count',
    'paged_ro_query',
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paginated whole-graph reads (tasks 4340, 4869)
# ---------------------------------------------------------------------------
# plans/falkordb-resultset-cap-audit.md is the one place for the dated row
# counts, the measured paging cost, the per-read classification of every read
# routed through this primitive, the open residual, and the ticket
# cross-references.  Update it there, not here.

RESULTSET_SIZE = 10000
"""FalkorDB's server-wide result-set ceiling: every query returns at most this
many rows, silently truncated — no error, no marker.

This is an ASSUMPTION about server configuration, not a fact this module
controls: fused-memory/docker/docker-compose.yml pins ``RESULTSET_SIZE`` in
FALKORDB_ARGS (task 6848), but a server started any other way may differ.
Measured 2026-08-17 against localhost:6379::

    GRAPH.CONFIG GET RESULTSET_SIZE  ->  10000

That it is an assumption is exactly why ``paged_ro_query`` does not trust it
alone and cross-checks against a server-side census (see the guards there).
"""

DEFAULT_READ_PAGE_SIZE = 5000
"""Rows per page: half the assumed cap, so a short page really is end-of-data.

The page loop breaks on a page shorter than the requested size, reasoning
"the data is exhausted".  That reasoning is sound only while the server
cannot be what shortened the page — i.e. only while the page size stays
strictly below the cap.  Half the cap buys a full page of margin.
"""

MAX_READ_PAGES = 1000
"""Hard bound on pages per enumeration, so a pathological corpus (or a store
that never returns a short page) cannot spin forever.

At the default page size that is 5,000,000 rows, against a live maximum near
32,000 — normal use never approaches it.  Hitting it is REPORTED
(``complete=False`` + a WARNING naming the numbers), never silently swallowed:
a page cap that truncates in silence would just be the defect this module
exists to fix, moved one layer up.
"""

# The four fail-closed paths of ``paged_ro_query``, as a typed discriminator
# on ``PagedRead.incomplete_kind``.  A caller reads these off a returned
# PagedRead to learn WHY a read was incomplete, so they are part of the
# contract.
#
# They exist so nobody has to sniff ``PagedRead.reason``, which is diagnostic
# prose written for an operator reading a log and is deliberately NOT a stable
# interface — any wording improvement would silently break a substring match,
# and the branch it broke is the one that stops a fabricated empty from being
# written back over real summaries.

INCOMPLETE_STRUCTURAL_REFUSAL = 'structural_refusal'
"""Guard 1: ``page_size >= resultset_size``. Zero queries issued, zero rows.

The returned emptiness is FABRICATED — no read was attempted — which is why
it must never be confused with a genuinely empty corpus.
"""

INCOMPLETE_PAGE_CAP = 'page_cap'
"""Guard 2: ``max_pages`` exhausted while the last page was still full.

The rows are a PREFIX of the corpus in ``ORDER BY`` order, not a sample.
"""

INCOMPLETE_CENSUS_UNAVAILABLE = 'census_unavailable'
"""Guard 3: no usable count. The rows were fetched; only the PROOF is missing."""

INCOMPLETE_SHORT_READ = 'short_read'
"""Guard 4: ``rows_seen < expected_rows`` — the enumeration is SHORT."""

INCOMPLETE_STRUCTURAL_KINDS = frozenset(
    {INCOMPLETE_STRUCTURAL_REFUSAL, INCOMPLETE_PAGE_CAP}
)
"""The DETERMINISTIC, non-transient incompleteness kinds.

Both are fully determined by configuration and code: they reproduce exactly
on retry and can never be caused by a concurrent write.  That is what makes
them safe to RAISE on, where the empirical kinds are not — a census
disagreeing by a handful of rows is the expected signature of a live graph
being written to mid-read.  The raise lives in
``fused_memory/backends/graphiti_client.py::apply_incompleteness_policy``.

Callers should branch on membership in this set rather than on a specific
kind, so a future fifth structural path is covered by construction.
"""

INCOMPLETE_KINDS = INCOMPLETE_STRUCTURAL_KINDS | {
    INCOMPLETE_CENSUS_UNAVAILABLE,
    INCOMPLETE_SHORT_READ,
}
"""Every incompleteness kind: the vocabulary ``incomplete_kind`` is drawn from."""


@dataclass(frozen=True)
class PagedRead:
    """Result of a paginated read: the rows, plus whether they are all of them.

    ``reason`` is None exactly when ``complete`` is True.  ``expected_rows`` is
    None when the census probe could not produce a usable count — an
    unavailable proof, which is itself a reason to report incomplete.

    ``incomplete_kind`` is one of the four module-level ``INCOMPLETE_*``
    constants, and is None exactly when ``complete`` is True.  It exists so a
    caller can branch on WHY a read was incomplete without parsing ``reason``,
    whose wording is diagnostic prose aimed at an operator reading a log and
    is deliberately NOT a stable interface.  The distinction matters most for
    ``INCOMPLETE_STRUCTURAL_REFUSAL``, where the returned emptiness is
    fabricated rather than read: without a typed discriminator that is
    indistinguishable from a genuinely empty graph, and the difference decides
    whether a caller writes an empty summary back over a real one.

    The field is last and defaulted so the five original fields — none of
    which have defaults — keep working under positional construction.
    """

    rows: list[list]
    complete: bool
    rows_seen: int
    expected_rows: int | None
    reason: str | None
    incomplete_kind: str | None = None


async def census_count(graph, cypher: str, params: dict | None = None) -> int | None:
    """Return the single integer a ``count(...)`` probe reports, or None.

    EVERY "the store did not say" shape collapses to None — no rows, a null
    result set, a row with no columns, a NULL value, a non-integer — because
    the caller treats None as fail-closed and there is nothing to gain from
    distinguishing flavours of missing evidence.

    A single-row aggregate can never be truncated by the row cap it is being
    used to detect, which is what makes this a proof rather than one more
    heuristic.
    """
    result = await graph.ro_query(cypher, params)
    rows = getattr(result, 'result_set', None) or []
    if not rows:
        return None
    first = rows[0]
    if first is None or len(first) == 0:
        return None
    try:
        return int(first[0])
    except (TypeError, ValueError):
        return None


_SKIP_PLACEHOLDER = '{skip}'
_LIMIT_PLACEHOLDER = '{limit}'


def _render_page_bounds(page_template: str, *, skip: int, limit: int) -> str:
    """Substitute the two SKIP/LIMIT placeholders, leaving every other brace alone.

    Deliberately NOT ``str.format``.  Cypher map patterns carry literal braces
    — ``MATCH (n:Entity {name: $name})``, a shape ordinary Cypher reads use
    routinely — and ``str.format`` raises ``KeyError``/``ValueError`` on
    one of those BEFORE any query is issued, from a traceback that points at a
    formatting call rather than at the offending template.  Since more reads
    are meant to be routed through this primitive, that trap would be one map
    literal away for the next author, and it would need a brace-doubling
    convention nobody has to remember here.  Two literal replacements have no
    such failure mode.

    Both placeholders are REQUIRED.  A template missing ``{skip}`` would
    re-fetch the same offset every iteration until ``max_pages``, then report
    a page-cap shortfall — a real authoring bug wearing the costume of a
    server truncation.  Raising is how it stays legible.

    Bounds are coerced to ``int`` before substitution, so nothing but a
    validated integer this module computed can reach the Cypher text.
    """
    for placeholder in (_SKIP_PLACEHOLDER, _LIMIT_PLACEHOLDER):
        if placeholder not in page_template:
            raise ValueError(
                f'page_template is missing the {placeholder} placeholder, so '
                f'its pages could never advance: {page_template!r}'
            )
    return page_template.replace(_SKIP_PLACEHOLDER, str(int(skip))).replace(
        _LIMIT_PLACEHOLDER, str(int(limit))
    )


async def paged_ro_query(
    graph,
    page_template: str,
    census_cypher: str,
    *,
    params: dict | None = None,
    page_size: int = DEFAULT_READ_PAGE_SIZE,
    resultset_size: int = RESULTSET_SIZE,
    max_pages: int = MAX_READ_PAGES,
) -> PagedRead:
    """Read every row a whole-graph query matches, past the server's row cap.

    ``page_template`` must be a Cypher string with ``{skip}`` and ``{limit}``
    placeholders and a TOTAL ``ORDER BY``; ``census_cypher`` must be the
    identical MATCH/WHERE returning ``count(*)``, so the two numbers describe
    the same population.

    The ``ORDER BY`` is load-bearing, not cosmetic: every page is a SEPARATE
    query, and SKIP/LIMIT with no total order gives the store no obligation to
    return rows in the same order twice — so ``SKIP n`` on page 2 can skip rows
    page 1 never returned (silently dropped, permanently) or re-return rows it
    did (harmlessly deduped, which is what makes the drop so easy to miss).

    SKIP/LIMIT bounds are substituted as validated ints rather than bound as
    ``$`` parameters: parameterised SKIP/LIMIT is not portable across Cypher
    implementations, and these are integers this module computes, never caller
    data, so there is no injection surface.  Substitution goes through
    ``_render_page_bounds``, which replaces the two placeholders LITERALLY
    rather than running ``str.format`` over the whole template — so a template
    containing a Cypher map pattern (``{name: $name}``) is safe, and no
    brace-doubling convention has to be remembered.  See that helper.

    The read fails CLOSED on four independent paths, each logging a WARNING
    that names the numbers as structured facts rather than prose:

      STRUCTURAL (fires on the numbers alone, before any evidence)
        1. ``page_size >= resultset_size`` — refuse to enumerate at all and
           return NO rows.  See the guard body for why a partial list is worse
           than none.
        2. ``max_pages`` exhausted while the last page was still full.

      EMPIRICAL (compares fetched against what the server says exists)
        3. The census probe returned no usable count.
        4. ``rows_seen < expected_rows`` — the enumeration is SHORT.

    BOTH kinds are kept, and neither subsumes the other.  ``resultset_size``
    is an assumption about server configuration; if the live server is ever
    configured BELOW it, the structural check passes and the short-page break
    lies exactly as an unpaginated query does today — the identical silent
    truncation, undetected — and only the census catches it.  Conversely the
    structural check fails FAST, before any query, with a specific operator
    action.  They fail differently and usefully.

    Completeness is ``rows_seen >= expected_rows``, NOT ``==``: a corpus that
    grew between the census probe and the last page is not a truncation, and
    on a graph under continuous write, growth is the common case.

    WHY THE TWO KINDS ARE NAMED SEPARATELY on ``PagedRead.incomplete_kind``,
    and why callers branch on ``INCOMPLETE_STRUCTURAL_KINDS`` rather than on
    an individual kind: the split is not a taxonomy for its own sake, it is
    the line between "safe to raise on" and "must not raise on".

      The STRUCTURAL kinds are DETERMINISTIC and non-transient.  They are
      properties of the configuration and of this code — a page size at or
      above the cap, a page budget too small for the corpus.  They reproduce
      exactly on retry, they can never be caused by a concurrent write, and
      neither is reachable at the shipped defaults.  A caller that treats
      them as fatal will never be woken by a passing graph.

      The EMPIRICAL kinds are transient-capable.  A census disagreeing by a
      handful of rows is the *expected* signature of a live graph being
      written to mid-read, not evidence of a defect, which is precisely why
      the design warns rather than raises on them: raising would take down
      the reconciliation rebuild for something that self-heals next cycle.

    KNOWN RESIDUAL, left open deliberately (task 4340): a materially-short
    ``INCOMPLETE_SHORT_READ`` still returns a partial collection that the
    ``force=True`` rebuild path writes back.  Tightening guard 4 into a raise
    is the WRONG fix — it fires on a single concurrently-invalidated edge, so
    it would take down the live rebuild for exactly the transient described
    above.  Full statement, the affected call site, and the ticket that closes
    it are in plans/falkordb-resultset-cap-audit.md.

    Args:
        graph: FalkorDB graph handle exposing ``async ro_query(cypher, params)``.
        page_template: Page query with ``{skip}``/``{limit}`` placeholders,
            both required. Other braces are left untouched (see
            ``_render_page_bounds``); no escaping is needed.
        census_cypher: Single-row ``count(*)`` over the identical MATCH/WHERE.
        params: Optional bound parameters, passed to every query.
        page_size: Rows per page. Must stay strictly below ``resultset_size``.
        resultset_size: Assumed server row cap (see ``RESULTSET_SIZE``).
        max_pages: Hard bound on pages fetched.

    Returns:
        PagedRead. ``reason`` is None exactly when ``complete`` is True.
    """
    page_size = int(page_size)
    resultset_size = int(resultset_size)
    max_pages = int(max_pages)

    # Guard 1 (structural). At or above the cap there is nothing trustworthy
    # to return, so return nothing. The short-page break reasons "this page
    # was not full, therefore the data is exhausted" — sound only while the
    # server cannot be what shortened it. At or above the cap those two causes
    # are indistinguishable, and a partial list would simply invite the caller
    # to use it anyway, recreating the silently-short-collection defect one
    # layer up. The comparison is >= and not > deliberately: equality is
    # arithmetically safe on a server configured at exactly RESULTSET_SIZE,
    # but that constant is an assumption, so equality leaves zero margin and a
    # server configured one row lower silently re-opens the truncation.
    if page_size >= resultset_size:
        reason = (
            f'page_size={page_size} is at or above the assumed server '
            f'resultset_size={resultset_size}, so a short page cannot be '
            f'distinguished from a server-truncated one; refusing to '
            f'enumerate. Re-run with a page size well below the cap '
            f'(default {DEFAULT_READ_PAGE_SIZE}).'
        )
        logger.warning('paged_ro_query: %s', reason)
        return PagedRead(
            rows=[],
            complete=False,
            rows_seen=0,
            expected_rows=None,
            reason=reason,
            incomplete_kind=INCOMPLETE_STRUCTURAL_REFUSAL,
        )

    # Census FIRST, before paging, so the target is fixed up front rather than
    # inferred from the same pages whose completeness is in question.
    expected = await census_count(graph, census_cypher, params)
    if expected is None:
        # Guard 3 (empirical). An unavailable proof is not a passing proof —
        # but keep paging: the rows are still worth returning, only the proof
        # is missing.
        logger.warning(
            'paged_ro_query: census probe returned no usable count '
            '(cypher=%r); the enumeration cannot be proven complete and will '
            'be reported incomplete even if every row was fetched',
            census_cypher,
        )

    rows: list[list] = []
    skip = 0
    paged_to_the_end = False
    for _ in range(max_pages):
        page_cypher = _render_page_bounds(page_template, skip=skip, limit=page_size)
        result = await graph.ro_query(page_cypher, params)
        page = list(getattr(result, 'result_set', None) or [])
        rows.extend(page)
        if len(page) < page_size:
            paged_to_the_end = True
            break
        skip += len(page)

    rows_seen = len(rows)

    reason: str | None = None
    kind: str | None = None
    if not paged_to_the_end:
        # Guard 2 (structural). Hitting the page cap on a still-full page is
        # REPORTED, never swallowed: a page cap that truncated in silence
        # would just be this module's own defect, one layer up.
        kind = INCOMPLETE_PAGE_CAP
        reason = (
            f'max_pages={max_pages} exhausted at page_size={page_size} with '
            f'rows_seen={rows_seen} and the last page still full, so more '
            f'rows almost certainly remain unread'
        )
        logger.warning('paged_ro_query: %s', reason)
    elif expected is None:
        kind = INCOMPLETE_CENSUS_UNAVAILABLE
        reason = (
            f'census probe returned no usable count, so the {rows_seen} rows '
            f'fetched cannot be proven to be all of them'
        )
    elif rows_seen < expected:
        # Guard 4 (empirical). Name BOTH candidate causes: an operator reading
        # this log must not be misled into chasing a phantom truncation when
        # the graph was simply written to mid-read.
        kind = INCOMPLETE_SHORT_READ
        reason = (
            f'enumeration is SHORT: fetched rows_seen={rows_seen} but the '
            f'census reports expected_rows={expected}. Most likely a server '
            f'result-set cap below the assumed resultset_size={resultset_size}; '
            f'the benign alternative is concurrent invalidation of edges '
            f'between the census probe and the last page'
        )
        logger.warning('paged_ro_query: %s', reason)

    return PagedRead(
        rows=rows,
        complete=reason is None,
        rows_seen=rows_seen,
        expected_rows=expected,
        reason=reason,
        incomplete_kind=kind,
    )
