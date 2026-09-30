"""The unnarrowed ``fetch_tasks`` callers' budget must be deliverable.

Companion to ``test_tasks_budget.py::test_tasks_budget_is_structurally
_deliverable`` and ``test_healthz_deadline.py::test_healthz_budget_is
_structurally_deliverable``, and deliberately written in the same idiom:
assert on the SHIPPED constants (no monkeypatching, no fakes, no event loop,
no ``DashboardConfig``), derive every count from an introspectable roster
rather than hard-coding it, and parse the caller-side ceiling out of the
source that actually ships rather than restating it.

The bug class this closes is arithmetic, not behavioural: a budget whose
parts do not fit inside its whole cannot deliver its own degraded payload.
That is the class ``test_tasks_budget.py`` and ``test_healthz_deadline.py``
already close for the Tasks tab and /healthz; this file extends it to the
``fetch_tasks`` callers that had no whole-operation bound AT ALL. Three of
them — ``orchestrator.discover_orchestrators``, ``merge_queue.load_task_titles``
and ``app._load_task_cards`` — wedged three dashboard endpoints for 19.8 h
behind a hung MCP seam, because ``fetch_tasks``' *timeout* is a per-HTTP-request
budget and never bounded the operation as a whole.

``discover_orchestrators`` has since left that population entirely: task 5587
removed its task fetch, so its per-root and whole-loop budgets, and the two
assertions that checked them, went with it. That is the STRONGEST available
resolution of its share of the incident — a call that does not exist cannot
hang — and not a relaxation. The endpoint is still swept by
``test_dashboard_endpoints_survive_hung_mcp.py``, which asserts it answers
200 while the seam hangs. The browser-abort ceiling this file used to read out
of ``data.js`` went with (d); ``test_tasks_budget.py`` assertion (c) still
enforces it for the one whole-handler deadline that remains.

None of the remaining assertions can be satisfied by loosening a constant in
one place: widening a call-site constant walks into (b) and raising the shared
default walks into (e). That mutual constraint is the point.
"""

from __future__ import annotations

from dashboard.api import escalations
from dashboard.data import merge_queue, tasks


def test_fetch_tasks_whole_operation_budget_is_structurally_deliverable():
    """Every layer of the fetch_tasks whole-operation budget must fit.

    Three independent arithmetic facts, each of which can regress on its own:

    (a) ONE URL's cold MCP session fits inside the shared whole-operation
        bound — with the post count DERIVED from ``COLD_SESSION_POSTS``, so a
        fourth handshake post fails here instead of silently overrunning;
    (b) each call site's own constant only ever TIGHTENS the shared default,
        never widens it;
    (e) the shared default never creeps toward ``mcp_tool_call``'s own 10 s
        default — this work is only ever allowed to tighten.

    (c) and (d) retired with ``discover_orchestrators``' task fetch (task
    5587). They bounded a per-root share and a whole-loop deadline that no
    longer exist; the letters are left un-reused so a reader comparing this
    against an older revision can see what went rather than mis-reading a
    renumbered assertion as the old one.
    """
    per_call = tasks.DEFAULT_PER_CALL_TIMEOUT
    posts = tasks.COLD_SESSION_POSTS
    whole = tasks.DEFAULT_WHOLE_OPERATION_BUDGET

    # (0) NON-VACUITY.
    assert len(posts) > 0, (
        'COLD_SESSION_POSTS must enumerate the real JSON-RPC posts a cold '
        'MCP session performs — an empty roster makes assertion (a) '
        'vacuously true'
    )

    # (a) one URL's cold session fits the whole-operation bound.
    #
    # This is a PER-URL claim and nothing more. fetch_tasks does NOT make one
    # cold session per call: its ``_refresh`` delegates to
    # ``first_success(config.fused_memory_urls, ...)``, and
    # ``mcp_fanout.first_success`` walks its URLs strictly IN ORDER, falling
    # through to the next only after the current one fails. So an N-URL
    # deployment's true cold worst case is N times the left-hand side here.
    assert per_call * len(posts) <= whole, (
        f'{per_call}s * {len(posts)} cold-session posts '
        f'({", ".join(posts)}) exceeds the whole-operation budget of '
        f'{whole}s — ONE URL cannot even complete its own MCP handshake '
        'inside the bound its callers enforce. NOTE what this does and does '
        'NOT claim: it is a PER-URL sum. fetch_tasks fans out over '
        'config.fused_memory_urls IN ORDER, so an N-URL deployment costs up '
        f'to N * {per_call * len(posts)}s cold. The budget deliberately CAPS '
        'that fan-out rather than accommodating it — capping exactly this '
        'residual is why the caller-side wait_for layer exists, so the two '
        'layers are complementary, not redundant. The fan-out width is an '
        'operator deployment variable, not a shipped constant, and is '
        'deliberately NOT asserted here.'
    )

    # (b) a call-site constant may only ever tighten the shared default.
    for label, site_budget in (
        ('merge_queue._TASK_TITLES_BUDGET', merge_queue._TASK_TITLES_BUDGET),
        ('escalations._TASK_CARDS_BUDGET', escalations._TASK_CARDS_BUDGET),
    ):
        assert site_budget <= whole, (
            f'{label} ({site_budget}s) exceeds the shared '
            f'DEFAULT_WHOLE_OPERATION_BUDGET ({whole}s) — a call site may '
            'only ever TIGHTEN the shared default, never widen it, or the '
            'one place the arithmetic is derived stops being authoritative'
        )

    # (e) standing guard: never raise the shared whole-operation budget.
    assert whole < 10, (
        f'DEFAULT_WHOLE_OPERATION_BUDGET ({whole}s) must stay strictly below '
        "mcp_tool_call's 10s default — this work may only ever tighten a "
        'budget, never widen one'
    )
