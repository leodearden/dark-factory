"""Per-consult bulk reads for the already-landed gate, and its call contract.

``AlreadyLandedTickView`` is one bulk read of local branches per consult.

The view may only make the gate ABSTAIN. It concludes nothing on its own except
"this branch had no ref when the view was read and main carries no merge marker
for it", which means there is nothing the gate could flip to done. Every other
case, including a branch the view lists, a branch main has a marker for, and a
view whose enumeration failed, sends the gate back to its fresh per-candidate
ref read. So a stale view, for example one read just before warm-lane
provisioning re-created a branch whose earlier incarnation left a marker on
main, can only cost a missed auto-done, never cause a false one (task 4900; see
``orchestrator/src/orchestrator/git_ops.py::GitOps.find_merge_marker``'s
branch-existence gate).

``PendingEscalationsByTask`` is one read of the pending escalation store per
consult, grouped by task id (task 8073). For a task id it lists exactly the rows
``escalation/src/escalation/queue.py::EscalationQueue.get_by_task`` returns with
``status='pending'``, and a task id it lacks has no pending rows. It is up to
one consult old when the gate judges a candidate, so the any-level escalation
veto (task 3534) and the arbitration hold decide on the store as it was when
the consult began. The snapshot can only over-hold: before any done-flip the
gate re-reads that task's pending rows and abstains on any record the snapshot
lacked, or on a failed read, so a record filed during the consult still vetoes
the flip. A record resolved during the consult holds its task one consult
longer.

``AlreadyLandedGate`` is the gate's call contract, the signature the
scheduler calls and the harness implements. It lives here, beside the
snapshots it carries, so the scheduler can type the gate without importing
the harness. Every keyword defaults to ``None``, meaning the gate does that
read itself.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

if TYPE_CHECKING:
    from escalation.models import Escalation

    from orchestrator.git_ops import GitOps

PendingEscalationsByTask: TypeAlias = 'Mapping[str, Sequence[Escalation]]'


@dataclass(frozen=True)
class AlreadyLandedTickView:
    """Local branch names at read time; ``None`` when git could not enumerate them."""

    local_branches: frozenset[str] | None


async def read_already_landed_tick_view(git_ops: GitOps) -> AlreadyLandedTickView:
    return AlreadyLandedTickView(local_branches=await git_ops.list_local_branches())


async def branch_absent_and_unmarked(
    view: AlreadyLandedTickView, git_ops: GitOps, branch: str,
) -> bool:
    if view.local_branches is None or branch in view.local_branches:
        return False
    return await git_ops.find_merge_marker(branch, gate_on_existing_ref=False) is None


class AlreadyLandedGate(Protocol):
    """The scheduler-to-harness call contract of the already-landed dispatch gate."""

    async def __call__(
        self,
        task_id: str,
        /,
        *,
        task: dict | None = None,
        tick_view: AlreadyLandedTickView | None = None,
        pending_escalations: PendingEscalationsByTask | None = None,
    ) -> bool: ...
