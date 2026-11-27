"""A ``get_statuses`` MCP session double that pages its answer, for the orchestrator's readers of that tool."""

from __future__ import annotations

from shared.testing_mcp_stub import tool_reply_body


class PagingStatusesSession:
    """A ``get_statuses`` server paging like ``fused_memory/server/tools.py::_pagination_meta``.

    With ``page_size`` it serves ``min(page_size, cap)`` entries from
    ``offset``.  Without it, a population larger than *cap* is TRUNCATED to
    *cap* entries and marked ``auto_paginated``; any other unpaged request is
    answered with the whole map and no envelope, which means complete.
    """

    def __init__(
        self,
        population: dict[str, str],
        *,
        cap: int | None = None,
        fail_at_offset: int | None = None,
        first_page_claims_returned: int | None = None,
    ) -> None:
        self._population = population
        self._cap = cap
        self._fail_at_offset = fail_at_offset
        self._first_page_claims_returned = first_page_claims_returned
        self.requests: list[dict] = []

    async def call_tool(self, name: str, arguments: dict, *, timeout: float = 15) -> dict:
        assert name == 'get_statuses', name
        self.requests.append(dict(arguments))
        offset = arguments.get('offset', 0)
        if offset == self._fail_at_offset:
            raise OSError(f'connection reset at offset {offset}')
        ids = arguments.get('ids')
        items = [(tid, status) for tid, status in self._population.items()
                 if ids is None or tid in ids]
        return tool_reply_body(self._payload(items, arguments.get('page_size'), offset))

    def _payload(self, items: list[tuple[str, str]], requested: int | None, offset: int) -> dict:
        if requested is not None:
            served = requested if self._cap is None else min(requested, self._cap)
            echoed, marker = requested, {}
        elif self._cap is None or len(items) <= self._cap:
            return {'statuses': dict(items)}
        else:
            served, echoed, marker = self._cap, self._cap, {'auto_paginated': True}
        window = dict(items[offset:offset + served])
        returned = len(window)
        if offset == 0 and self._first_page_claims_returned is not None:
            returned = self._first_page_claims_returned
        return {
            'statuses': window,
            'pagination': {
                'total': len(items),
                'offset': offset,
                'page_size': echoed,
                'returned': returned,
                'has_more': offset + len(window) < len(items),
                **marker,
            },
        }


def numbered_statuses(count: int) -> dict[str, str]:
    return {str(tid): 'done' if tid % 3 else 'pending' for tid in range(1, count + 1)}
