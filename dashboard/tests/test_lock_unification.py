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
