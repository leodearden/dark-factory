"""Pin tests for the lock-display unification (task 1508).

Verifies that:
  - data.js has dropped FILE_LOCKS from both the window.DF_DATA default and
    the /api/v2/dashboard/tasks endpoint key mapping.
  - tab_tasks.jsx reads lock state from D.SCHEDULER (not FILE_LOCKS).
  - tab_overview.jsx shows a Modules lock column derived from D.SCHEDULER.modules.
  - tabs.jsx LocksCell uses D.SCHEDULER instead of DF.FILE_LOCKS.

Tests use the TestClient static-asset fetch pattern from test_index_html.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# step-5: data.js drops FILE_LOCKS default + endpoint mapping
# ---------------------------------------------------------------------------


def test_data_js_drops_file_locks_default_and_endpoint_mapping(client):
    resp = client.get('/static/redux/data.js')
    assert resp.status_code == 200
    body = resp.text

    # FILE_LOCKS must be gone from the window.DF_DATA initialiser and the
    # /api/v2/dashboard/tasks endpoint mapping.
    assert 'FILE_LOCKS' not in body, (
        "data.js still references FILE_LOCKS — remove the default and endpoint key"
    )

    # The tasks endpoint mapping must still carry the three surviving keys.
    assert "'ACTIVE_TASKS'" in body or '"ACTIVE_TASKS"' in body
    assert "'TASKS_OFFLINE'" in body or '"TASKS_OFFLINE"' in body
    assert "'TASKS_OFFLINE_PROJECTS'" in body or '"TASKS_OFFLINE_PROJECTS"' in body

    # Sanity: SCHEDULER (unified source) must still be initialised.
    assert 'SCHEDULER' in body, "data.js must keep SCHEDULER as the unified lock source"


# ---------------------------------------------------------------------------
# step-7: tab_tasks.jsx rewired to read locks from D.SCHEDULER
# ---------------------------------------------------------------------------


def test_tab_tasks_jsx_uses_scheduler_for_lock_display(client):
    resp = client.get('/static/redux/tab_tasks.jsx')
    assert resp.status_code == 200
    body = resp.text

    # (a) Old FILE_LOCKS lookups must be gone.
    assert 'FILE_LOCKS' not in body, (
        "tab_tasks.jsx still references FILE_LOCKS"
    )
    assert 'fileLocks' not in body, (
        "tab_tasks.jsx still references fileLocks"
    )

    # (b) Unified source must be used.
    assert 'D.SCHEDULER' in body or 'DF_T.SCHEDULER' in body, (
        "tab_tasks.jsx must read locks from D.SCHEDULER or DF_T.SCHEDULER"
    )
    assert '.rows' in body, "tab_tasks.jsx must access .rows from SCHEDULER"
    assert '.modules' in body, "tab_tasks.jsx must access .modules from SCHEDULER"

    # (c) Section label updated.
    assert 'Module locks' in body, (
        "tab_tasks.jsx must label the lock section 'Module locks'"
    )

    # (d) holder_project qualifier present (project-qualified join).
    assert 'holder_project' in body, (
        "tab_tasks.jsx must reference holder_project for the scheduler join"
    )

    # (e) Lock chip CSS palette classes preserved.
    assert 'lock-free' in body
    assert 'lock-mine' in body
    assert 'lock-taken' in body


# ---------------------------------------------------------------------------
# step-9: tab_overview.jsx gains Modules lock column from D.SCHEDULER.modules
# ---------------------------------------------------------------------------


def test_tab_overview_jsx_shows_module_locks_from_scheduler(client):
    resp = client.get('/static/redux/tab_overview.jsx')
    assert resp.status_code == 200
    body = resp.text

    # (a) The Orchestrators table must have a Modules column header.
    assert 'Modules' in body, (
        "tab_overview.jsx Orchestrators table must have a 'Modules' column header"
    )

    # (b) The column must derive held/contended counts from D.SCHEDULER.modules.
    assert 'D.SCHEDULER' in body, (
        "tab_overview.jsx must reference D.SCHEDULER for module lock counts"
    )
    assert '.modules' in body, (
        "tab_overview.jsx must reference .modules from SCHEDULER"
    )

    # (c) FILE_LOCKS must not be referenced.
    assert 'FILE_LOCKS' not in body, (
        "tab_overview.jsx must not reference FILE_LOCKS"
    )


# ---------------------------------------------------------------------------
# step-11: tabs.jsx LocksCell rewired to read locks from D.SCHEDULER
# ---------------------------------------------------------------------------


def test_tabs_jsx_locks_cell_uses_scheduler_not_file_locks(client):
    resp = client.get('/static/redux/tabs.jsx')
    assert resp.status_code == 200
    body = resp.text

    # (a) No stale FILE_LOCKS lookups.
    assert 'DF.FILE_LOCKS' not in body, (
        "tabs.jsx LocksCell still reads from DF.FILE_LOCKS — remove this dependency"
    )
    assert 'fileLocks' not in body, (
        "tabs.jsx still has a fileLocks variable — remove FILE_LOCKS dependency"
    )

    # (b) LocksCell must reference DF.SCHEDULER and use rows/modules arrays.
    assert 'DF.SCHEDULER' in body or 'D.SCHEDULER' in body, (
        "tabs.jsx LocksCell must read from DF.SCHEDULER (the unified lock source)"
    )
    assert '.rows' in body, "tabs.jsx must access .rows from SCHEDULER"
    assert '.modules' in body, "tabs.jsx must access .modules from SCHEDULER"

    # (c) LockChip props carry a holder resolved through the scheduler join —
    #     holder_project qualifier must be present (same join as tab_tasks.jsx).
    assert 'holder_project' in body, (
        "tabs.jsx LockChip must receive holder_project from the scheduler module entry"
    )

    # (d) Lock chip CSS palette classes preserved.
    assert 'lock-free' in body
    assert 'lock-mine' in body
    assert 'lock-taken' in body
