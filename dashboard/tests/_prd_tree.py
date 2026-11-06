"""The PRD view's census-join fixture tree, the one source both of its suites read.

``test_prd_view.py`` builds the pure view's inputs from :func:`prd_tree_tasks`,
and ``_boundary_payloads.py::prd_tree`` serves the same tasks through
CannedMCP to the endpoint suite, so neither suite can drift from the other.

PRD A: 1 done, 2 ready and pending :data:`READY_PENDING_DAYS` days, 3 a
follow-up of 1, 4 a done follow-up of 3 carrying two links, 5 cancelled.
PRD B: 6 done only. Then 7 with no link, 8 done with no PRD, 9 a
high-priority cross-project follow-up from source 'review', and 10 blocked
under a foreign PRD with no priority recorded.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Any, NamedTuple

PRD_A = 'plans/a-prd.md'
PRD_B = 'plans/b-prd.md'
FOREIGN_PRD = '/elsewhere/plans/x-prd.md'
READY_PENDING_DAYS = 40


class TreeTask(NamedTuple):
    status: str
    priority: str | None
    metadata: Mapping[str, Any]


def prd_tree_tasks(at: datetime) -> Mapping[int, TreeTask]:
    """The tree by task id, with task 2 pending since :data:`READY_PENDING_DAYS` before *at*."""
    pending_since = (at - timedelta(days=READY_PENDING_DAYS)).isoformat()
    return MappingProxyType({
        1: TreeTask('done', 'medium', {'prd_path': PRD_A}),
        2: TreeTask('pending', 'medium', {'prd_path': PRD_A, 'pending_since': pending_since}),
        3: TreeTask('in-progress', 'medium', {'spawned_from': 1}),
        4: TreeTask('done', 'medium', {'spawned_from': 3, 'cross_repo_refile_of': 'reify:1'}),
        5: TreeTask('cancelled', 'medium', {'prd_path': PRD_A}),
        6: TreeTask('done', 'medium', {'prd_path': PRD_B}),
        7: TreeTask('pending', 'medium', {}),
        8: TreeTask('done', 'medium', {}),
        9: TreeTask('pending', 'high', {'spawned_from': 'reify:5569', 'source': 'review'}),
        10: TreeTask('blocked', None, {'prd_path': FOREIGN_PRD}),
    })
