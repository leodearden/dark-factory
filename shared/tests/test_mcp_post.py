"""Tests for shared.mcp_post — the slash-less MCP endpoint + loud response check.

Task 4023.  Five raw ``httpx`` POSTs in the orchestrator targeted
``f'{base}/mcp/'`` on a bare ``AsyncClient``.  The fused-memory server
307-redirects ``/mcp/`` -> ``/mcp``, and a bare client does NOT follow
redirects — so every one of those writes was silently discarded for months
while the HTTP exchange itself reported perfect success.

TDD pair 1: ``mcp_endpoint_url`` canonicalization (GREEN on impl step-2).
TDD pair 2: ``check_mcp_post_response`` loud-and-never-raising (GREEN on step-4).
"""
from __future__ import annotations

import json
from pathlib import Path

from shared.mcp_post import mcp_endpoint_url

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pair 1 — mcp_endpoint_url (step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


def test_endpoint_url_has_no_trailing_slash():
    """The canonical endpoint is ``/mcp`` — the trailing slash is the defect."""
    assert mcp_endpoint_url('http://localhost:8002') == 'http://localhost:8002/mcp'


def test_endpoint_url_never_ends_in_slash_for_any_base():
    """No base shape may produce a trailing slash — that is what 307s."""
    for base in (
        'http://localhost:8002',
        'http://localhost:8002/',
        'http://127.0.0.1:8102',
        'https://memory.example.test',
    ):
        url = mcp_endpoint_url(base)
        assert not url.endswith('/'), f'{base!r} produced a redirect-triggering {url!r}'
        assert url.endswith('/mcp')


def test_endpoint_url_collapses_a_base_that_already_ends_in_slash():
    """A configured base ending in ``/`` must not yield ``//mcp``.

    Mirrors ``McpSession.__init__``'s ``base_url.rstrip('/')`` — the rstrip is
    load-bearing, not cosmetic.
    """
    assert mcp_endpoint_url('http://localhost:8002/') == 'http://localhost:8002/mcp'
    assert '//mcp' not in mcp_endpoint_url('http://localhost:8002/')
    # Multiple trailing slashes collapse too.
    assert mcp_endpoint_url('http://localhost:8002///') == 'http://localhost:8002/mcp'


def test_endpoint_url_matches_the_form_mcp_json_declares():
    """The result must equal the slash-less URL ``.mcp.json`` already declares.

    ``.mcp.json`` is the convergence target: the Claude CLI's own MCP client
    reaches fused-memory at ``http://127.0.0.1:8002/mcp`` with no trailing
    slash, and so must every raw POST.
    """
    declared = json.loads((REPO_ROOT / '.mcp.json').read_text())['mcpServers']
    urls = [
        server['url']
        for server in declared.values()
        if isinstance(server, dict) and str(server.get('url', '')).endswith('/mcp')
    ]
    assert urls, 'expected .mcp.json to declare at least one HTTP MCP server'
    for url in urls:
        base = url[: -len('/mcp')].rstrip('/')
        assert mcp_endpoint_url(base) == url, (
            f'helper must reproduce the declared endpoint {url!r} exactly'
        )
