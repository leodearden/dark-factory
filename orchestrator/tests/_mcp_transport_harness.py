"""MockTransport harness reproducing the live fused-memory MCP server (task 4023).

WHY A ``_``-PREFIXED SIBLING RATHER THAN A ``test_`` MODULE.  These helpers are
imported cross-file: ``test_mcp_post_transport.py`` and
``test_suggestion_triage.py`` both drive the same fake server.  Importing them
from a ``test_``-prefixed module would couple one test file's collection to
another's module-level side effects (that file also computes ``REPO_ROOT`` and
the sweep constants), and pytest would import the same module under two names.
``_``-prefixed, uniquely-named sibling modules are the established convention
here — ``_verify_config_corpus.py``, ``_orch_helpers.py``, ``_mcp_url_scan.py``
— and ``conftest.py`` puts ``_TESTS_DIR`` on ``sys.path`` at import time, which
is what makes a bare ``from _mcp_transport_harness import ...`` resolve.  They
are NOT in ``conftest.py`` because non-fixture helpers imported from a conftest
collide across sibling subprojects under ``sys.modules['conftest']`` — see that
file's docstring.

WHAT THE HANDLER REPRODUCES.  Both live rejections, measured by curl against
127.0.0.1:8002 while writing this (see escalation esc-4023-2):

  * ``POST /mcp/``            -> 307, Location: /mcp   (trailing slash)
  * ``POST /mcp`` w/o Accept  -> 406 "Client must accept application/json"
  * ``POST /mcp`` w/ Accept   -> 200 + a real JSON-RPC result

Both matter, because fixing only the slash moves the failure from a silent 307
no-op to a silent 406 no-op — still a lost write.

``httpx.MockTransport`` keeps httpx's REAL redirect semantics, so
``follow_redirects`` genuinely decides the outcome while no socket is opened.
That is what lets these tests assert WHERE a POST landed rather than merely
that ``.post`` was called — the distinction that ~3 months of silently
discarded writes turned on.
"""

from __future__ import annotations

import json

import httpx

__all__ = [
    'MCP_PATH',
    'MCP_PATH_WITH_SLASH',
    'RecordingClientFactory',
    'RecordingMcpServer',
]

#: Captured at import, BEFORE any ``patch('httpx.AsyncClient', ...)`` is
#: active, so :class:`RecordingClientFactory` can build a real client without
#: recursing into its own patch.
_REAL_ASYNC_CLIENT = httpx.AsyncClient

MCP_PATH = '/mcp'
MCP_PATH_WITH_SLASH = '/mcp/'


class RecordingMcpServer:
    """``httpx.MockTransport`` handler reproducing the live fused-memory server."""

    def __init__(self):
        #: (path, json_body) for every request the server actually PROCESSED.
        self.delivered: list[tuple[str, dict]] = []
        #: Requests refused with a 307 because of the trailing slash.
        self.redirected: list[str] = []
        #: Requests refused with a 406 because the Accept header was missing.
        self.not_acceptable: list[str] = []
        #: (path, Accept) for EVERY request that arrived, including the ones
        #: refused above.  Recorded pre-triage so the Accept assertion stays
        #: non-vacuous even while the slash defect still absorbs the request.
        self.seen: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((path, request.headers.get('accept', '')))

        if path == MCP_PATH_WITH_SLASH:
            self.redirected.append(path)
            return httpx.Response(
                307,
                headers={'Location': str(request.url.copy_with(path=MCP_PATH))},
            )

        if path != MCP_PATH:
            return httpx.Response(404, text=f'no route for {path}')

        # The live server refuses a POST that does not accept application/json.
        if 'application/json' not in request.headers.get('accept', ''):
            self.not_acceptable.append(path)
            return httpx.Response(
                406,
                json={
                    'jsonrpc': '2.0',
                    'id': 'server-error',
                    'error': {
                        'code': -32600,
                        'message': 'Not Acceptable: Client must accept application/json',
                    },
                },
            )

        body = json.loads(request.content)
        self.delivered.append((path, body))
        return httpx.Response(
            200,
            json={'jsonrpc': '2.0', 'id': 1, 'result': {'content': []}},
        )

    # -- assertion helpers --------------------------------------------------

    def tool_calls(self, tool_name: str) -> list[dict]:
        """Bodies of every DELIVERED ``tools/call`` for *tool_name*."""
        return [
            body for _path, body in self.delivered
            if body.get('method') == 'tools/call'
            and body.get('params', {}).get('name') == tool_name
        ]

    def assert_every_request_accepted_json(self):
        """Every request that ARRIVED must have declared it accepts JSON.

        Checks ``seen`` rather than ``not_acceptable`` so this stays a real
        assertion even before the slash fix lands: a request absorbed by the
        307 never reaches the 406 branch, so an empty ``not_acceptable`` would
        pass vacuously.
        """
        assert self.seen, 'no request reached the server at all'
        bad = [(p, a) for p, a in self.seen if 'application/json' not in a]
        assert bad == [], (
            'every MCP POST must send Accept: application/json (the live server '
            f'answers 406 without it); these did not: {bad!r}'
        )


class RecordingClientFactory:
    """Stands in for ``httpx.AsyncClient``: records ctor kwargs, injects transport.

    Uses the same ``patch('httpx.AsyncClient', lambda *a, **k: ...)``
    constructor seam that ``test_workflow_completion_memory.py`` already relies
    on — widened to RECORD the kwargs, which is what makes the
    ``follow_redirects=True`` assertion possible at all.
    """

    def __init__(self, server: RecordingMcpServer):
        self._server = server
        #: One dict per ``httpx.AsyncClient(...)`` construction.
        self.ctor_kwargs: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.ctor_kwargs.append(dict(kwargs))
        kwargs.setdefault('transport', httpx.MockTransport(self._server.handler))
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    def assert_follows_redirects(self):
        assert self.ctor_kwargs, 'no httpx.AsyncClient was constructed at all'
        for kwargs in self.ctor_kwargs:
            assert kwargs.get('follow_redirects') is True, (
                'the MCP client must be constructed with follow_redirects=True '
                f'(defence in depth against a re-introduced slash); got {kwargs!r}'
            )
