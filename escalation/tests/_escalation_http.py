"""The SINGLE construction site of the escalation capability headers in tests.

Every call in ``escalation/tests/`` that sends ``X-Escalation-Levels`` /
``X-Escalation-Identity`` over real HTTP goes through here. Task 3736 deduped
the SERVER-LIFECYCLE half of this harness into ``conftest.py``
(``serve_escalation_mcp`` / ``serve_escalation_mcp_module``); task 4345 folded
the last two copies of the CALL half — near-twin ``_call_over_http`` bodies in
``test_capability_guard_http.py`` and ``test_status_authority_gate.py`` — into
this module (INV-5). The single-source property is not a convention anyone has
to remember: it is asserted by an AST scan in
``test_escalation_http_helper.py``.

WHY A SIBLING MODULE AND NOT A CONFTEST FIXTURE. ``conftest.py`` is for
FIXTURES, and a fixture is unreachable from the module-level ``async def``
per-tool partials that call this. The house convention for a non-fixture test
helper is a uniquely-named, underscore-prefixed sibling module (so pytest never
collects it) — ``_fm_helpers.py``, ``_orch_helpers.py``, ``_dashboard_helpers.py``
— made importable by the ``sys.path.insert(0, str(_TESTS_DIR))`` in the
subproject conftest, which ``tests/scripts/test_pytest_workspace_collection.py``
enforces. A bare ``from conftest import ...`` stays forbidden: the ``conftest``
module name collides across subprojects in ``sys.modules`` under the repo-wide
``--import-mode=importlib``.

WHY THE HEADER NAMES ARE LITERALS HERE. They are deliberately NOT imported from
``escalation.server._LEVELS_HEADER`` / ``_IDENTITY_HEADER``. This looks like the
next INV-5 fold and is not one: this module is the CLIENT half of a two-way
boundary contract whose whole purpose is to prove the client-side header
protocol and the server-side parser stay in agreement. A test that sources the
wire name from the code under test can no longer catch a rename on either side —
it would follow the server anywhere and go green on a break. The lockstep is
pinned the correct way instead, by driving the REAL client constant
``orchestrator.harness._WATCHER_ESCALATION_HEADERS`` against the running server
(``test_capability_guard_http.py::TestWatcherConstantLockstep``). Note also the
case difference that makes the coupling non-trivial even if someone tried: the
server constants are lowercased (``'x-escalation-levels'``, because
``get_http_headers()`` lowercases) while the client sends
``'X-Escalation-Levels'``. Do not "deduplicate" these literals against the
server constants.

THE ``None`` / ``''`` DISTINCTION. A value of ``None`` omits the header
ENTIRELY; any non-None value — including ``''`` — is sent verbatim. The gate is
``is not None``, never truthiness, and the difference is observable end-to-end:
``server.py::stamp_triage`` does ``if identity is not None: triaged_by =
identity``, so an omitted header leaves the tool arg intact while an empty one
stamps ``triaged_by == ''``. Both halves are pinned in
``test_escalation_http_helper.py``.
"""

from __future__ import annotations

from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


def capability_headers(
    *,
    levels: str | None = None,
    identity: str | None = None,
) -> dict[str, str]:
    """Build the capability request headers for an escalation HTTP call.

    *levels* / *identity*, when not None, become the literal
    ``X-Escalation-Levels`` / ``X-Escalation-Identity`` request headers; when
    None the header is omitted entirely (never sent as an empty string), so a
    header-less call exercises the exact same default-open path a real
    header-less client would hit.

    This is the one place those names are constructed — see the module
    docstring for why they are literals and not imports from the server.
    """
    headers: dict[str, str] = {}
    if levels is not None:
        headers['X-Escalation-Levels'] = levels
    if identity is not None:
        headers['X-Escalation-Identity'] = identity
    return headers


async def escalation_http_call(
    base_url: str,
    tool_name: str,
    *,
    levels: str | None = None,
    identity: str | None = None,
    **tool_kwargs: Any,
) -> dict[str, Any]:
    """Call *tool_name* on a running escalation server over real HTTP.

    Drives a RUNNING FastMCP escalation server rather than calling
    ``tool.fn(...)`` in-process, because
    ``fastmcp.server.dependencies.get_http_headers()`` only resolves real
    request headers under an ASGI request context — an in-process call always
    sees ``{}``, which would make the capability gate untestable.

    *levels* / *identity* are forwarded to :func:`capability_headers`; every
    other keyword is passed through to the tool. Returns ``result.data``.

    The per-tool partials in the consumer modules (``_resolve_over_http`` /
    ``_promote_over_http`` / ``_stamp_triage_over_http``) are one-liners over
    this, so the header construction cannot drift between them.
    """
    headers = capability_headers(levels=levels, identity=identity)
    transport = StreamableHttpTransport(f'{base_url}/mcp/', headers=headers)
    async with Client(transport) as client:
        result = await client.call_tool(tool_name, tool_kwargs)
        return result.data
