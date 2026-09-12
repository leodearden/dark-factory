"""The three unnarrowed ``fetch_tasks`` callers' budget must be deliverable.

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
three ``fetch_tasks`` callers that had no whole-operation bound AT ALL —
``orchestrator.discover_orchestrators``, ``merge_queue.load_task_titles``
and ``app._load_task_cards``. Those three wedged three dashboard endpoints
for 19.8 h behind a hung MCP seam, because ``fetch_tasks``' *timeout* is a
per-HTTP-request budget and never bounded the operation as a whole.

None of these assertions can be satisfied by loosening a constant in one
place: widening a call-site constant walks into (b), raising the shared
default walks into (e), and raising the orchestrators loop budget walks into
(d). That mutual constraint is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

from dashboard import app
from dashboard.data import merge_queue, orchestrator, tasks

_DATA_JS = Path(tasks.__file__).parent.parent / 'static' / 'redux' / 'data.js'
_ABORT_MS_RE = re.compile(r'DEFAULT_TIMEOUT_MS\s*=\s*(\d+)')


def _browser_abort_ms():
    """The browser-side fetch abort, in ms, read from the shipped data.js."""
    source = _DATA_JS.read_text(encoding='utf-8')
    match = _ABORT_MS_RE.search(source)
    # A rename of the JS constant must fail LOUDLY here rather than silently
    # skipping the ceiling check — a check that quietly stops checking is
    # indistinguishable from a passing one.
    assert match is not None, (
        f'could not find DEFAULT_TIMEOUT_MS in {_DATA_JS} — if the constant '
        'was renamed, update _ABORT_MS_RE; do not delete this assertion, or '
        'the handler budget loses its only ceiling'
    )
    return int(match.group(1))


def test_fetch_tasks_whole_operation_budget_is_structurally_deliverable():
    """Every layer of the fetch_tasks whole-operation budget must fit.

    Five independent arithmetic facts, each of which can regress on its own:

    (a) ONE URL's cold MCP session fits inside the shared whole-operation
        bound — with the post count DERIVED from ``COLD_SESSION_POSTS``, so a
        fourth handshake post fails here instead of silently overrunning;
    (b) each call site's own constant only ever TIGHTENS the shared default,
        never widens it;
    (c) at least one orchestrator root fits inside the whole-loop budget, so
        ``discover_orchestrators`` can never return an all-degraded payload
        by arithmetic alone;
    (d) that whole-loop budget fits inside the browser's fetch abort, so the
        partial payload the deadline produces is actually deliverable;
    (e) the shared default never creeps toward ``mcp_tool_call``'s own 10 s
        default — this work is only ever allowed to tighten.
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
        ('orchestrator._ORCHESTRATORS_PER_ROOT_BUDGET',
         orchestrator._ORCHESTRATORS_PER_ROOT_BUDGET),
        ('merge_queue._TASK_TITLES_BUDGET', merge_queue._TASK_TITLES_BUDGET),
        ('app._TASK_CARDS_BUDGET', app._TASK_CARDS_BUDGET),
    ):
        assert site_budget <= whole, (
            f'{label} ({site_budget}s) exceeds the shared '
            f'DEFAULT_WHOLE_OPERATION_BUDGET ({whole}s) — a call site may '
            'only ever TIGHTEN the shared default, never widen it, or the '
            'one place the arithmetic is derived stops being authoritative'
        )

    # (c) at least one root always fits inside the whole-loop budget.
    per_root = orchestrator._ORCHESTRATORS_PER_ROOT_BUDGET
    total = orchestrator._ORCHESTRATORS_TOTAL_BUDGET
    assert per_root <= total, (
        f'per-root budget {per_root}s exceeds the whole-loop orchestrators '
        f'budget {total}s — the first root alone would exhaust the loop, so '
        'every payload would be all-degraded by arithmetic'
    )

    # (d) the degraded payload must survive the browser's abort.
    abort_ms = _browser_abort_ms()
    assert total * 1000 < abort_ms, (
        f'whole-loop orchestrators budget {total}s ({total * 1000}ms) is not '
        f'strictly below the browser fetch abort of {abort_ms}ms (data.js '
        'DEFAULT_TIMEOUT_MS) — the degraded payload would be aborted before '
        'it could be rendered, which is the 15s-behind-a-5s-caller bug that '
        'test_healthz_deadline.py exists to prevent'
    )

    # (e) standing guard: never raise the shared whole-operation budget.
    assert whole < 10, (
        f'DEFAULT_WHOLE_OPERATION_BUDGET ({whole}s) must stay strictly below '
        "mcp_tool_call's 10s default — this work may only ever tighten a "
        'budget, never widen one'
    )
