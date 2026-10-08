"""The observed facts of one no-progress verify abort, and their two durable renderings.

Abort trigger 3 of the merge worker's in-flight verify fires when no progress
signal (a live remote dispatch, or new content under the merge worktree) was
seen for a whole budget. ``NoProgressAbort`` records what was observed at that
firing; ``event_data`` renders it as the ``merge_verify_progress_abort`` event
payload and ``terminal_reason`` as the blocked reason of a cap-out. Both state
observations only: nothing here can tell a dead verify from a slow one, so
neither rendering claims to.

Every firing counts as a strike towards ``max_strikes``, including on a remote
lease whose dispatch had already returned. Since task 4579 such an abort means
the post-dispatch LOCAL work wrote nothing under the merge worktree for a whole
budget, which is the stalled-local-work class the cap bounds; the remote leg's
health says nothing about it, and not counting would turn a stalled cross-check
into an unbounded requeue loop holding the verifier host. ``dispatch_returned``
makes that population measurable before any separate cap is designed.
"""

from __future__ import annotations

import dataclasses

from orchestrator.event_store import EventStore, EventType


@dataclasses.dataclass(frozen=True)
class NoProgressAbort:
    task_id: str
    request_id: str
    runner: str
    is_local: bool
    no_progress_secs: float
    budget_secs: float
    strike: int
    max_strikes: int
    dispatch_seen_live: bool

    def __post_init__(self) -> None:
        if self.strike < 1 or self.no_progress_secs < self.budget_secs:
            raise ValueError(
                f'NoProgressAbort for task {self.task_id} requires strike >= 1 '
                f'and no_progress_secs >= budget_secs; got strike={self.strike}, '
                f'no_progress_secs={self.no_progress_secs}, '
                f'budget_secs={self.budget_secs}'
            )

    @property
    def capped(self) -> bool:
        return self.strike >= self.max_strikes

    @property
    def dispatch_returned(self) -> bool | None:
        return None if self.is_local else self.dispatch_seen_live

    def event_data(self) -> dict:
        return {
            'request_id': self.request_id,
            'lease_kind': 'local' if self.is_local else 'remote',
            'runner': self.runner,
            'no_progress_secs': self.no_progress_secs,
            'budget_secs': self.budget_secs,
            'strike': self.strike,
            'max_strikes': self.max_strikes,
            'capped': self.capped,
            'dispatch_returned': self.dispatch_returned,
        }

    def terminal_reason(self) -> str:
        """The cap-out's blocked reason, built from occurrence-stable facts only.

        The measured duration and the runner name stay in ``event_data``: the
        reason feeds a retry-thrash signature that a jittery number would
        defeat, and a review-category substring match that a hostname could
        trip.
        """
        return (
            f'no in-flight verify progress observed for the {self.budget_secs:g}s '
            f'budget on {self.strike} consecutive attempts ({self._lease_clause()})'
        )

    def _lease_clause(self) -> str:
        if self.is_local:
            return 'local lease'
        if self.dispatch_seen_live:
            return 'remote lease, dispatch had returned'
        return 'remote lease, no dispatch seen in flight'


def emit_no_progress_abort(event_store: EventStore | None, abort: NoProgressAbort) -> None:
    if event_store is None:
        return
    event_store.emit(
        EventType.merge_verify_progress_abort,
        task_id=abort.task_id,
        phase='merge',
        data=abort.event_data(),
    )
