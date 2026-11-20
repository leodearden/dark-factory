"""Records how each MCP ``httpx.AsyncClient`` was constructed (task 4023).

``follow_redirects=True`` is defence in depth that never shows on the wire:
every call site builds a slash-less URL, so no redirect ever reaches the
server. The contract is therefore pinned at construction, through the
``patch('httpx.AsyncClient', ...)`` seam ``shared.mcp_post.open_mcp_client``
documents. The client built is REAL, so a test using this recorder MUST point
the call site at a ``shared.testing_mcp_stub.StubMcpServer``.

A ``_``-prefixed sibling because two test modules use it and test modules must
not import each other; ``conftest.py`` records that convention.
"""

from __future__ import annotations

import httpx

__all__ = ['RecordingClientFactory']

#: Captured at import, BEFORE any ``patch('httpx.AsyncClient', ...)`` is
#: active, so :class:`RecordingClientFactory` can build a real client without
#: recursing into its own patch.
_REAL_ASYNC_CLIENT = httpx.AsyncClient


class RecordingClientFactory:
    """Stands in for ``httpx.AsyncClient``: records ctor kwargs, builds a real client."""

    def __init__(self) -> None:
        #: One dict per ``httpx.AsyncClient(...)`` construction.
        self.ctor_kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.ctor_kwargs.append(dict(kwargs))
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    def assert_follows_redirects(self):
        assert self.ctor_kwargs, 'no httpx.AsyncClient was constructed at all'
        for kwargs in self.ctor_kwargs:
            assert kwargs.get('follow_redirects') is True, (
                'the MCP client must be constructed with follow_redirects=True '
                f'(defence in depth against a re-introduced slash); got {kwargs!r}'
            )
