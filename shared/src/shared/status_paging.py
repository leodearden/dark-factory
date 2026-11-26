"""The one walk that assembles a complete ``get_statuses`` map from its pages.

``get_statuses`` may answer with a slice of the map and a pagination
envelope. This module turns those slices back into one map, or refuses: a
read whose completeness cannot be proven is never handed back as complete.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ['STATUSES_SAFE_PAGE_SIZE', 'StatusPage', 'StatusPagingError', 'walk_status_pages']

STATUSES_SAFE_PAGE_SIZE = 2000
"""How many statuses one ``get_statuses`` page may carry.

The server's own documented bound (``_STATUSES_AUTO_PAGE_LIMIT``), and a
measured one rather than a round number: 2 000 entries is ~46 KB against the
~62 KB documented-safe MCP tool-response envelope, while 3 000 is ~69 KB —
past the wall.  An oversized page is NOT clamped server-side; it is rejected
WHOLESALE, which is the incident paging exists to close.

Held here, at the module that speaks the paged side of the wire contract,
rather than imported from the server: no consumer of this walk depends on the
fused-memory package.
"""

K = TypeVar('K')


class StatusPagingError(ValueError):
    """A paged ``get_statuses`` read whose completeness could not be proven."""


@dataclass(frozen=True, slots=True)
class StatusPage(Generic[K]):
    """One ``get_statuses`` response: the slice it carried, and its envelope as sent.

    ``pagination`` is None when the response carried no envelope, which the
    tool's contract defines as a COMPLETE answer rather than a missing field.
    Any other value is passed through unread, so the walk alone decides
    whether it can be trusted.

    *statuses* may be SHORTER than *delivered* when a page reader drops
    entries it cannot use.  The walk's ``returned`` cross-check counts
    *delivered*, what the server actually sent: counting the kept entries
    would turn one unusable entry into a refusal of the whole read.
    """

    statuses: Mapping[K, str]
    delivered: int
    pagination: object


async def walk_status_pages(
    page_fn: Callable[[int], Awaitable[StatusPage[K]]],
    label: str,
) -> dict[K, str]:
    """Assemble the COMPLETE status map from repeated *page_fn* calls at increasing offsets.

    *page_fn* reads one page from ONE source: a walk that switched sources
    mid-walk would tile pages from two different states of the world.
    *label* names the read in every refusal.

    ADVANCE BY ``returned``, THE COUNT SERVED.  The envelope comes from
    ``fused_memory/server/tools.py::_pagination_meta``.  It passes the
    caller's REQUESTED ``page_size`` through verbatim, sets ``returned`` to
    the number of entries it actually sliced, and derives
    ``has_more = offset + returned < total``.  That identity is why
    ``offset + returned`` is the only advance that tiles the population
    without gaps: a walk that stops on the server's ``has_more`` has to step
    exactly as the server counted.  Advancing by the requested size would skip
    the difference whenever a server serves fewer entries than asked, which
    produces a silently incomplete map.

    ``get_statuses``' own docstring says the opposite: "Advance by
    ``pagination['page_size']`` (what was actually served)".  Its
    implementation does not do that.  Do not switch this walk back to
    ``page_size`` because of that docstring.

    ``returned`` is cross-checked against the entries the page actually
    delivered, because it is the counter the walk advances on.  A page
    clipped in flight that still claims its full count would skip the clipped
    entries and terminate normally.  A page that under-reports would re-read
    entries already held.  Neither is detectable afterwards, so the read is
    refused.

    TRUNCATION IS LOUD: every failure below raises :class:`StatusPagingError`
    and discards the pages in hand, and an exception from *page_fn*
    propagates the same way.  A short map is indistinguishable from a full
    one once it has been returned, so a partial map is never returned.

    There is no changed-``total`` coherence check, because this walk stops on
    the SERVER's ``has_more``, computed against the population at the instant
    of each page, not on a client-side ``offset >= total`` comparison that a
    shrinking total would silently satisfy early.
    """
    merged: dict[K, str] = {}
    offset = 0
    pages = 0
    page_budget: int | None = None
    while True:
        page = await page_fn(offset)
        merged.update(page.statuses)
        pages += 1

        meta = page.pagination
        if meta is None:
            # No envelope: the response IS the whole map.  Looping blind here
            # would re-request the same entries forever.
            break
        if not isinstance(meta, Mapping):
            raise StatusPagingError(
                f'get_statuses pagination for {label} is unverifiable at '
                f'offset {offset}: the envelope is a {type(meta).__name__}, not a mapping'
            )

        has_more = meta.get('has_more')
        returned = meta.get('returned')
        total = meta.get('total')
        if not isinstance(has_more, bool) or not isinstance(returned, int):
            # Without these two, completeness is UNVERIFIABLE — and an
            # unverifiable read must not be reported as a complete one.
            raise StatusPagingError(
                f'get_statuses pagination for {label} is unverifiable at '
                f'offset {offset}: has_more={has_more!r}, returned={returned!r}'
            )
        if returned != page.delivered:
            raise StatusPagingError(
                f'get_statuses pagination for {label} inconsistent at '
                f'offset {offset}: server claims returned={returned} but sent '
                f'{page.delivered} status(es)'
            )
        if not has_more:
            break
        if returned <= 0:
            # More entries are owed, but the page that would carry them cannot
            # advance the offset, so this server cannot be paged past.
            raise StatusPagingError(
                f'get_statuses pagination for {label} cannot advance at '
                f'offset {offset}: more entries remain but returned={returned}'
            )

        if page_budget is None:
            # BOUND THE WALK.  ``has_more`` is the loop's only terminator and
            # it is server-reported, so a server that never lowers it would
            # spin here forever.  Derived from the FIRST page's ``returned``
            # because that is the rate the walk actually advances at; the +2
            # covers the final partial page plus one page of slack.
            if not isinstance(total, int):
                raise StatusPagingError(
                    f'get_statuses pagination for {label} reports no usable '
                    f'total at offset {offset}: total={total!r}'
                )
            page_budget = math.ceil(total / returned) + 2
        if pages >= page_budget:
            raise StatusPagingError(
                f'get_statuses pagination for {label} exceeded its '
                f'{page_budget}-page budget at offset {offset} '
                f'(total={total}); refusing to keep walking'
            )
        offset += returned
    return merged
