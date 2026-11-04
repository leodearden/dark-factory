"""The /prds fetch budget must be structurally deliverable (boundary row 12a).

Written in the idiom of ``test_tasks_budget.py``: shipped constants only, no
fakes and no event loop, the browser's fetch abort parsed out of the shipped
``data.js``, and every name read is public. The /prds handler acquires each
root's snapshot unit and lineage index inside a per-root share of a
whole-handler deadline, so the same arithmetic must hold here: each part
fits the layer above it, and the whole fits the browser's abort.
"""

from __future__ import annotations

import re
from pathlib import Path

from dashboard.api import prds
from dashboard.data import task_lineage, task_snapshot, tasks

_DATA_JS = Path(prds.__file__).parent.parent / 'static' / 'redux' / 'data.js'
_ABORT_MS_RE = re.compile(r'DEFAULT_TIMEOUT_MS\s*=\s*(\d+)')


def _browser_abort_ms() -> int:
    """The browser-side fetch abort, in ms, read from the shipped data.js."""
    match = _ABORT_MS_RE.search(_DATA_JS.read_text(encoding='utf-8'))
    assert match is not None, (
        f'could not find DEFAULT_TIMEOUT_MS in {_DATA_JS}; if the constant was renamed, '
        'update _ABORT_MS_RE rather than dropping the ceiling check'
    )
    return int(match.group(1))


def test_prds_budget_is_structurally_deliverable() -> None:
    budget = prds.PRDS_BUDGET
    snapshot_reads = task_snapshot.PER_CALL_TIMEOUT * len(task_snapshot.PER_PROJECT_MCP_CALLS)

    assert snapshot_reads <= budget.per_project, (
        f'the snapshot unit costs up to {snapshot_reads}s per root, beyond the '
        f'{budget.per_project}s per-root share'
    )
    assert budget.per_project > task_lineage.LINEAGE_REQUEST_WAIT_SECONDS, (
        'a slow lineage read must land as stale or unknown lineage, never as a '
        'budget-degraded root'
    )
    assert budget.per_project <= budget.total
    assert budget.total * 1000 < _browser_abort_ms(), (
        f'the {budget.total}s /prds deadline would be aborted by the browser before '
        'its degraded payload could render'
    )
    assert 1 < budget.root_concurrency <= 8


def test_the_lineage_read_releases_its_slot_before_the_next_refresh() -> None:
    cold_read = len(tasks.COLD_SESSION_POSTS) * task_lineage.LINEAGE_PER_CALL_TIMEOUT
    assert task_lineage.LINEAGE_TTL_SECONDS < task_lineage.LINEAGE_FRESHNESS_BOUND_SECONDS
    assert cold_read <= task_lineage.LINEAGE_READ_TIMEOUT < task_lineage.LINEAGE_TTL_SECONDS
    assert task_lineage.LINEAGE_PER_CALL_TIMEOUT < 10


def test_a_stand_in_view_states_the_loosest_input_bound() -> None:
    loosest = max(task_snapshot.FRESHNESS_BOUND_SECONDS, task_lineage.LINEAGE_FRESHNESS_BOUND_SECONDS)
    assert loosest == prds.STAND_IN_VIEW_BOUND_SECONDS


def test_the_open_question_1_figures_are_pinned() -> None:
    assert task_lineage.LINEAGE_TTL_SECONDS == 300
    assert task_lineage.LINEAGE_FRESHNESS_BOUND_SECONDS == 900
    assert task_lineage.LINEAGE_PER_CALL_TIMEOUT == 6.0
    assert prds.PRDS_BUDGET.total == 20.0
    assert prds.PRDS_BUDGET.root_concurrency == 4
