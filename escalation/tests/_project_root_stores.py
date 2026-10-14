"""Store layout shared by the two suites of the γ2 ``project_root`` guard.

``test_mismatch_streak.py`` drives ``MismatchStreak`` directly and
``test_project_root_assertion.py`` drives it through ``create_server``. Both
must agree on what counts as the storm escape's output and on how a project
store and a reconciliation store are laid out, so that lives here once.

Plain factories rather than ``conftest.py`` fixtures, like
``_pending_tool_fixtures.py``: each suite keeps its own thin fixtures over
these, so a reader of one suite sees every name it uses.
"""

from __future__ import annotations

from pathlib import Path

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.store_identity import StoreIdentity

PROJECT_ID = 'proj'


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


def project_identity(queue: EscalationQueue, project_root: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='project', queue_dir=queue.queue_dir, project_id=PROJECT_ID,
        project_root=project_root,
    )


def recon_identity(queue: EscalationQueue) -> StoreIdentity:
    return StoreIdentity(
        kind='reconciliation', queue_dir=queue.queue_dir, project_id=None, project_root=None,
    )
