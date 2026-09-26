"""The Tasks-tab fetch budget must be structurally deliverable.

Companion to ``test_healthz_deadline.py::test_healthz_budget_is_structurally
_deliverable``, and deliberately written in the same idiom: assert on the
SHIPPED constants (no monkeypatching, no fakes, no event loop), derive every
count from an introspectable roster rather than hard-coding it, and parse the
caller-side ceiling out of the source that actually ships rather than
restating it.

The bug class this closes is arithmetic, not behavioural: a budget whose
parts do not fit inside its whole cannot deliver its own degraded payload.
/healthz shipped exactly that for a while — a 15 s handler behind a
``curl --max-time 5`` caller, measured delivering a 503 at 50.6 s. The Tasks
tab has the same shape: ``collect_tasks_with_counts`` fans out per project,
each project issues several MCP calls, and the only real caller is
``data.js``'s own ``fetch`` abort. If the handler budget ever creeps past
that abort, the partial payload the deadline exists to produce is thrown
away by the browser before it can be rendered — a strictly worse outcome
than not having a deadline at all, and invisible in every behavioural test
(they all call the collector directly, never through the browser's abort).

None of these assertions can be satisfied by loosening a constant in one
place: raising ``_TASKS_TOTAL_BUDGET`` to accommodate a slower per-project
budget walks into (c), and tightening ``data.js``'s abort to accommodate
(c) walks into the behavioural suite. That mutual constraint is the point.
"""

from __future__ import annotations

import re
from pathlib import Path

from dashboard.data import active_tasks, task_snapshot, tasks

# The tightest real caller of /api/v2/dashboard/tasks is the dashboard's own
# poll loop, whose fetch wrapper aborts at DEFAULT_TIMEOUT_MS. Parsed out of
# the shipped source rather than hard-coded, for the same reason the healthz
# test derives its probe count from _healthz_db_targets(): a hard-coded copy
# silently stops tracking reality the moment the real value moves.
_DATA_JS = (
    Path(active_tasks.__file__).parent.parent / 'static' / 'redux' / 'data.js'
)
_ABORT_MS_RE = re.compile(r'DEFAULT_TIMEOUT_MS\s*=\s*(\d+)')


def _browser_abort_ms() -> int:
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


def test_tasks_budget_is_structurally_deliverable() -> None:
    """Every layer of the Tasks-tab budget must fit inside the layer above it.

    Seven independent arithmetic facts, each of which can regress on its own:

    (a) the per-project MCP calls fit inside the per-project budget — with the
        call count DERIVED from ``PER_PROJECT_MCP_CALLS``, so adding a fourth
        bounded operation without raising the budget fails here, and with the
        per-call term taken from the Tasks-tab-LOCAL
        ``task_snapshot.PER_CALL_TIMEOUT`` rather than the shared default;
    (b) one project fits inside the whole-handler budget, so the handler can
        never return an all-degraded payload by arithmetic alone;
    (c) the whole-handler budget fits inside the browser's fetch abort, so the
        partial payload the deadline produces is actually deliverable;
    (d) the per-request budget never creeps up toward ``mcp_tool_call``'s own
        10 s default;
    (e) ISOLATION: the Tasks tab's wider per-call timeout is its OWN constant
        and the shared ``tasks.DEFAULT_PER_CALL_TIMEOUT`` is untouched;
    (f) CONCURRENCY: the per-root walk is parallel but BOUNDED;
    (g) at least one project's budget fits inside the whole.

    WHAT THIS FILE DOES NOT GUARANTEE (read (g) carefully before adding an
    assertion that looks like it should): it does NOT prove every configured
    root is served on every render.  With N roots, a concurrency of W and a
    per-project budget B, the worst case is ``ceil(N / W) * B``, which for the
    incident's 9 roots exceeds ``_TASKS_TOTAL_BUDGET`` — deliberately, because
    driving that worst case under the total would mean either a per-project
    budget too small for a 5 000-task tree or a total budget above the
    browser's abort in (c).  The tail is made FAIR by ROTATION of the walk's
    start offset (``active_tasks._rotated_project_roots``), not by arithmetic:
    a root starved on one render leads the next one.  Asserting the worst case
    here would encode a target the design deliberately does not meet, so (g)
    asserts only the weaker fact that is actually true.
    """
    per_call = task_snapshot.PER_CALL_TIMEOUT
    shared_per_call = tasks.DEFAULT_PER_CALL_TIMEOUT
    calls = task_snapshot.PER_PROJECT_MCP_CALLS
    per_project = active_tasks._TASKS_PER_PROJECT_BUDGET
    total = active_tasks._TASKS_TOTAL_BUDGET
    concurrency = active_tasks._TASKS_ROOT_CONCURRENCY

    assert len(calls) > 0, (
        'PER_PROJECT_MCP_CALLS must enumerate the real per-project bounded '
        'operations '
        '— an empty roster makes assertion (a) vacuously true'
    )

    # (a) the parts fit the whole, per project.
    assert per_call * len(calls) <= per_project, (
        f'{per_call}s * {len(calls)} per-project MCP calls '
        f'({", ".join(calls)}) exceeds the per-project budget of '
        f'{per_project}s — a project cannot complete its own calls inside '
        'its own deadline'
    )

    # (b) at least one project always fits inside the whole-handler budget.
    assert per_project <= total, (
        f'per-project budget {per_project}s exceeds the whole-handler budget '
        f'{total}s — the first project alone would exhaust the handler, so '
        'every payload would be all-degraded by arithmetic'
    )

    # (c) the handler's degraded payload must survive the browser's abort.
    abort_ms = _browser_abort_ms()
    assert total * 1000 < abort_ms, (
        f'whole-handler budget {total}s ({total * 1000}ms) is not strictly '
        f'below the browser fetch abort of {abort_ms}ms (data.js '
        'DEFAULT_TIMEOUT_MS) — the degraded payload would be aborted before '
        'it could be rendered, which is the 15s-behind-a-5s-caller bug that '
        'test_healthz_deadline.py exists to prevent'
    )

    # (d) standing guard: never raise the per-request budget to the ceiling.
    assert per_call < 10, (
        f'PER_CALL_TIMEOUT ({per_call}s) must stay strictly below '
        "mcp_tool_call's 10s default — a per-call budget that reaches the "
        'server-side default stops being a budget at all, and (a) would then '
        'force a per-project budget wider than (c) permits'
    )

    # (e) ISOLATION: the Tasks tab widened its OWN constant, not the shared one.
    assert shared_per_call == 2.0, (
        f'tasks.DEFAULT_PER_CALL_TIMEOUT is {shared_per_call}s, expected 2.0 '
        '— the Tasks tab needs a WIDER per-call timeout than the shared '
        'default (measured 2026-09-07: per-root wall max 2.876s against a '
        '2.0s default, which marked dark-factory/reify/autopilot-video '
        'OFFLINE on a cold render and shipped 208 of 3045 active rows). It '
        'must NOT buy that by raising this shared constant: '
        'DEFAULT_PER_CALL_TIMEOUT feeds DEFAULT_WHOLE_OPERATION_BUDGET, which '
        'merge_queue._TASK_TITLES_BUDGET and escalations._TASK_CARDS_BUDGET '
        'both bind BY REFERENCE — so a bump here silently widens route '
        'budgets this work must not touch (task 4788 territory). Widen '
        'task_snapshot.PER_CALL_TIMEOUT instead.'
    )
    assert per_call > shared_per_call, (
        f'PER_CALL_TIMEOUT ({per_call}s) is not wider than the shared '
        f'default ({shared_per_call}s) — if the Tasks tab does not actually '
        'need a wider budget, delete the local constant and use the shared '
        'one rather than keeping a same-valued alias that hides the coupling'
    )

    # (f) CONCURRENCY: parallel, but bounded.
    assert concurrency > 1, (
        f'_TASKS_ROOT_CONCURRENCY is {concurrency} — at 1 this is the '
        'SEQUENTIAL walk that caused the starvation this constant exists to '
        'fix: with N roots the wall clock is the SUM of the per-root costs, '
        'so the tail roots consistently exhausted _TASKS_TOTAL_BUDGET and '
        'were reported degraded on every render. Concurrency > 1 is the '
        'entire point of the constant.'
    )
    assert concurrency <= 8, (
        f'_TASKS_ROOT_CONCURRENCY is {concurrency} — the walk must stay '
        'BOUNDED. burndown.py records the measurement: a full fan-out over '
        'every root hits a SINGLE fused-memory server where the requests '
        'serialise server-side, on the SAME shared httpx client the 3s render '
        'polls use, which is "a live httpx.PoolTimeout risk for request-path '
        'handlers" (see burndown._SNAPSHOT_PAGE_SIZE and '
        '_fetch_snapshot_tasks). An unbounded gather trades this endpoint\'s '
        'latency for every other endpoint\'s availability.'
    )

    # (g) at least one project's budget fits inside the whole. See the
    # docstring: this is deliberately the WEAK form. Fairness across the tail
    # comes from rotation, not from arithmetic.
    assert total / per_project >= 1, (
        f'whole-handler budget {total}s is smaller than one project\'s '
        f'{per_project}s — no root could ever complete, so every render '
        'would be all-degraded regardless of concurrency or rotation'
    )
