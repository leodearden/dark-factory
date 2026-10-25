"""One bulk read of local branches per dispatch tick, for the already-landed gate.

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

``AlreadyLandedGate`` is the gate's call contract, the signature the
scheduler calls and the harness implements. It lives here, beside the view it
carries, so the scheduler can type the gate without importing the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from orchestrator.git_ops import GitOps


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
    ) -> bool: ...
