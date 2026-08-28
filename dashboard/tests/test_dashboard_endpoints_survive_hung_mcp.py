"""No dashboard endpoint may hang when the fetch_tasks MCP seam hangs.

This is the ACCEPTANCE PROBE for task 4788 — verification by probe, not by
reading. Three endpoints (``/api/v2/dashboard/orchestrators``,
``/merge-queue``, ``/escalations``) once wedged for 19.8 h behind a hung
fused-memory seam, and ``/tasks`` — which reaches the SAME ``fetch_tasks``
through ``collect_tasks_with_counts`` — survived, because it alone wrapped
the call in ``asyncio.wait_for``. That asymmetry is what this file pins.

Why the per-request timeout was never enough: ``fetch_tasks``' *timeout* is
threaded into ``mcp_tool_call`` and thence ``client.post``, so it bounds
connect/read/write and pool acquisition ONLY. The incident's hang was inside
httpcore's connection lock, where no outbound socket is ever opened and that
timeout therefore never fires. Only an enclosing ``asyncio.wait_for`` — whose
cancellation IS delivered into an awaiting lock — bounds it.

Three properties keep this probe from quietly stopping checking:

* the route list is DERIVED from ``app.routes``, never hard-coded, so a new
  endpoint is swept automatically and a shrinking sweep fails loudly;
* the hang stub is ``await asyncio.Event().wait()`` on an event nothing ever
  sets — it has no duration at all, so no budget value can make pre-fix code
  pass, unlike a slow-sleep stub;
* a NON-VACUITY assertion requires that all three formerly-wedging modules
  actually reached the hang, so the sweep cannot pass by never exercising one.
"""

from __future__ import annotations

import asyncio

import pytest

# Budgets are shrunk to this so the whole sweep is fast. The stub never
# returns, so every bounded call site pays exactly this and then degrades.
_TINY_BUDGET = 0.05

_TARGET_MODULES = {
    'dashboard.app',
    'dashboard.data.orchestrator',
    'dashboard.data.merge_queue',
}


def _dashboard_get_paths() -> list[str]:
    """Every GET route under /api/v2/dashboard/, read off the live app."""
    from dashboard.app import app

    return sorted({
        route.path
        for route in app.routes
        if getattr(route, 'path', '').startswith('/api/v2/dashboard/')
        and 'GET' in (getattr(route, 'methods', None) or set())
    })


@pytest.fixture()
def hung_mcp(monkeypatch, tmp_path):
    """Hang every fetch_tasks binding; shrink every budget; clear every cache.

    Returns the ``reached`` list of ``(module_name, project_root)`` pairs the
    stub recorded, so a caller can assert the hang was genuinely exercised.
    """
    from dashboard.app import _analytics_cache_clear, _task_cards_cache_clear
    from dashboard.data import active_tasks, merge_queue, orchestrator, tasks

    reached: list[tuple[str, str]] = []

    def _make_stub(module_name: str):
        async def _hang(client, config, project_root):
            reached.append((module_name, str(project_root)))
            await asyncio.Event().wait()  # nothing ever sets it

        return _hang

    # Patch the binding in EVERY module that imported the name by value —
    # patching dashboard.data.tasks.fetch_tasks alone would miss all four.
    for module_name in (
        'dashboard.app',
        'dashboard.data.orchestrator',
        'dashboard.data.merge_queue',
        'dashboard.data.active_tasks',
    ):
        monkeypatch.setattr(
            f'{module_name}.fetch_tasks', _make_stub(module_name),
        )

    import dashboard.app as _app

    monkeypatch.setattr(_app, '_TASK_CARDS_BUDGET', _TINY_BUDGET)
    monkeypatch.setattr(merge_queue, '_TASK_TITLES_BUDGET', _TINY_BUDGET)
    monkeypatch.setattr(
        orchestrator, '_ORCHESTRATORS_PER_ROOT_BUDGET', _TINY_BUDGET,
    )
    monkeypatch.setattr(
        orchestrator, '_ORCHESTRATORS_TOTAL_BUDGET', _TINY_BUDGET,
    )
    # The Tasks tab is already compliant; shrink it too so it does not
    # dominate the sweep's wall time.
    monkeypatch.setattr(active_tasks, '_TASKS_PER_PROJECT_BUDGET', _TINY_BUDGET)
    monkeypatch.setattr(active_tasks, '_TASKS_TOTAL_BUDGET', _TINY_BUDGET)

    # A warm entry would be served without ever reaching the hang.
    tasks._fetch_tasks_cache_clear()
    tasks._fetch_statuses_cache_clear()
    merge_queue._task_titles_cache_clear()
    _task_cards_cache_clear()
    _analytics_cache_clear()

    # Force the preconditions so the three target endpoints actually REACH
    # their call site — otherwise the probe would pass vacuously.
    proj = tmp_path / 'probe_root'
    (proj / '.taskmaster').mkdir(parents=True)
    monkeypatch.setattr(
        orchestrator,
        'find_running_orchestrators',
        lambda: [{
            'pid': 4788, 'prd': str(proj / 'prd.md'), 'config_path': None,
            'running': True, 'started': 'Aug27',
        }],
    )
    monkeypatch.setattr(
        _app,
        'build_escalation_queues',
        lambda config: {
            'subsections': [
                {'id': str(proj), 'kind': 'orchestrator', 'label': 'probe',
                 'items': []},
            ],
        },
    )

    yield reached

    # Leave no hang-stubbed entry behind for the next test in the session.
    tasks._fetch_tasks_cache_clear()
    tasks._fetch_statuses_cache_clear()
    merge_queue._task_titles_cache_clear()
    _task_cards_cache_clear()
    _analytics_cache_clear()


def test_every_dashboard_endpoint_survives_a_hung_fetch_tasks(client, hung_mcp):
    """Every GET endpoint answers 200 while the fetch_tasks seam hangs."""
    paths = _dashboard_get_paths()

    # The sweep is derived, and may not silently shrink. 14 is the count at
    # the time this probe was written; a NEW endpoint raises it, and a route
    # that disappears must be noticed rather than quietly dropped.
    assert len(paths) >= 14, (
        f'only {len(paths)} GET routes discovered under /api/v2/dashboard/ '
        f'({paths}) — the sweep has shrunk below the 14 that existed when '
        'this probe was written; a check that quietly stops checking is '
        'indistinguishable from a passing one'
    )

    for path in paths:
        # TestClient.get is SYNCHRONOUS and blocks for as long as the handler
        # does, so a regression here shows up as a hung or failed call — the
        # 19.8 h symptom in miniature.
        resp = client.get(path)
        assert resp.status_code == 200, (
            f'{path} returned {resp.status_code} while the fetch_tasks MCP '
            'seam was hung — a hung dependency must degrade this endpoint, '
            'never wedge or 500 it'
        )

    # NON-VACUITY: the probe must actually have exercised a hang in each of
    # the three modules that wedged, or it proves nothing at all.
    modules_reached = {module for module, _ in hung_mcp}
    assert modules_reached >= _TARGET_MODULES, (
        f'the hang was never reached in {sorted(_TARGET_MODULES - modules_reached)} '
        f'(reached: {sorted(modules_reached)}) — the sweep passed without '
        'exercising the defect, so it proves nothing. Check the endpoint '
        'preconditions in the hung_mcp fixture.'
    )
