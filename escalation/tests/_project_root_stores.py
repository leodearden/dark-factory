"""Store layout shared by the suites that drive the γ2 ``project_root`` guard.

``test_mismatch_streak.py`` drives ``MismatchStreak`` directly and
``test_project_root_assertion.py`` drives it through ``create_server``, as does
``test_store_identity_boundary.py`` over a real MCP session. All three
must agree on which tools the guard covers, on the arguments each guarded tool
is driven with, on what counts as the storm escape's output and on how a
project store and a reconciliation store are laid out, so that lives here once.

Plain factories rather than ``conftest.py`` fixtures, like
``_pending_tool_fixtures.py``: each suite keeps its own thin fixtures over
these, so a reader of one suite sees every name it uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.store_identity import StoreIdentity

PROJECT_ID = 'proj'

# The reads plans/escalation-store-ambiguity-prd.md §6.3 guards.
GUARDED_READS = (
    'get_pending_escalations',
    'get_escalation',
    'get_task_escalation_history',
    'get_task_escalations',
)
# The writes the same §6.3 guards.
GUARDED_WRITES = ('escalate_info', 'escalate_blocker')
# The record transitions the §6.3 amendment of task 11553 guards.
GUARDED_RESOLVES = ('resolve_issue',)


def guarded_arguments(
    name: str, *, task_id: str, escalation_id: str, filed_task_id: str | None = None,
) -> dict[str, Any]:
    """A fresh, minimal valid argument set, ``project_root`` aside, for guarded tool *name*.

    Reads ask about *task_id* or *escalation_id*, filings file under
    *filed_task_id* (default *task_id*), and ``resolve_issue`` closes
    *escalation_id*. A guarded tool with no row here raises ``KeyError``.
    """
    filing = {
        'task_id': filed_task_id or task_id,
        'agent_role': 'implementer',
        'category': 'design_concern',
        'summary': 's',
    }
    return {
        'get_pending_escalations': {'task_id': task_id},
        'get_escalation': {'escalation_id': escalation_id},
        'get_task_escalation_history': {'task_id': task_id},
        'get_task_escalations': {'task_id': task_id},
        'escalate_info': filing,
        'escalate_blocker': filing,
        'resolve_issue': {'escalation_id': escalation_id, 'resolution': 'r', 'action': 'close_only'},
    }[name]


def storm_records(queue: EscalationQueue) -> list[Escalation]:
    """The pending records a mismatch storm files: blocking ``infra_issue`` L1s."""
    return [e for e in queue.get_pending() if e.category == 'infra_issue' and e.level == 1]


def make_root(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def project_queue(project_root: Path) -> EscalationQueue:
    return EscalationQueue(project_root / 'esc')


def recon_queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'recon')


def project_identity(
    queue: EscalationQueue, project_root: Path, *, project_id: str = PROJECT_ID,
) -> StoreIdentity:
    return StoreIdentity(
        kind='project', queue_dir=queue.queue_dir, project_id=project_id,
        project_root=project_root,
    )


def recon_identity(queue: EscalationQueue) -> StoreIdentity:
    return StoreIdentity(
        kind='reconciliation', queue_dir=queue.queue_dir, project_id=None, project_root=None,
    )
