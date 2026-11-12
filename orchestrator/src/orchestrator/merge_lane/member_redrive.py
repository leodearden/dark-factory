"""Hand parked train members back when no train will land them (tasks 1867, 4737).

A member parked merge-deferred waits for a train to land it.  When that train
derails, is dissolved, or is lost with its process, :meth:`MemberRedriver.redrive`
decides each still-parked member's fate from git.  A branch already on main
goes done with found_on_main evidence when the landing is attributable, and
otherwise is escalated and goes back to pending.  An unlanded member goes back
to pending for a fresh solo merge, first stripped of any unlanded stack base
when the caller asks (task 5618).
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from orchestrator.branch_stack import UnstackOutcome
from orchestrator.merge_lane.landing_evidence import (
    file_unattributed_landing_escalation,
    validate_landing_evidence,
)
from orchestrator.scheduler import StaleEvidenceRejection
from orchestrator.task_status import TaskStatus

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.git_ops import GitOps

logger = logging.getLogger('orchestrator.merge_lane')

StatusCheck = Callable[[list[str]], Awaitable[Mapping[str, str]]]
RedriveMember = Callable[[str, bool, str | None], Awaitable[None]]


class MemberRedriveOutcome(enum.StrEnum):
    """What the redriver decided for one member, not that member's final status:
    the ``redrive_member`` callback may still withhold a done-write downstream."""

    NOT_PARKED = 'not_parked'
    NOT_ON_MAIN = 'not_on_main'
    ON_MAIN_ATTRIBUTED = 'on_main_attributed'
    ON_MAIN_UNATTRIBUTED = 'on_main_unattributed'
    PROVENANCE_CONFLICT_MEMOIZED = 'provenance_conflict_memoized'
    STALE_EVIDENCE = 'stale_evidence'
    ERRORED = 'errored'
    HELD_BY_LIVE_TRAIN = 'held_by_live_train'


class ProvenanceConflictMemo(Protocol):
    """The slice of orchestrator/src/orchestrator/provenance_conflict.py::ProvenanceConflictSink
    the re-drive uses."""

    def should_skip(self, task_id: str) -> bool: ...

    def record_from_rejection(
        self, exc: StaleEvidenceRejection, *, gate_source: str,
    ) -> object: ...


@dataclass(frozen=True)
class MemberRedriver:
    git_ops: GitOps
    provenance_conflict_sink: ProvenanceConflictMemo | None
    escalation_queue: EscalationQueue | None
    agent_role: str

    async def redrive(
        self,
        label: str,
        member_ids: Sequence[str],
        *,
        status_check: StatusCheck,
        redrive_member: RedriveMember,
        strip_unlanded_base: bool,
    ) -> dict[str, MemberRedriveOutcome]:
        """Re-drive every member of *member_ids* still merge-deferred.  Never raises
        for one member: its failure is that member's outcome and the rest still run.

        A landing found on main is validated in DISCOVERY mode
        (``orchestrator/src/orchestrator/merge_lane/landing_evidence.py::validate_landing_evidence``);
        an unattributable one files a 'provenance_unattributed' escalation
        instead of fabricating provenance.  A ``StaleEvidenceRejection`` of the
        done-write (task 2677) goes to the provenance-conflict sink, and a
        member that sink has memoized is skipped before any write.
        """
        statuses = await status_check(list(member_ids))
        outcomes: dict[str, MemberRedriveOutcome] = {}
        for mid in member_ids:
            if statuses.get(mid) != TaskStatus.MERGE_DEFERRED:
                outcomes[mid] = MemberRedriveOutcome.NOT_PARKED
                continue
            try:
                outcomes[mid] = await self._redrive_parked(
                    label, mid, redrive_member, strip_unlanded_base=strip_unlanded_base,
                )
            except StaleEvidenceRejection as exc:
                self._record_stale_evidence(label, mid, exc)
                outcomes[mid] = MemberRedriveOutcome.STALE_EVIDENCE
            except Exception:
                logger.exception(
                    'Train %s: re-drive failed for member %s — '
                    'continuing with remaining members', label, mid,
                )
                outcomes[mid] = MemberRedriveOutcome.ERRORED
        return outcomes

    async def _redrive_parked(
        self, label: str, mid: str, redrive_member: RedriveMember, *, strip_unlanded_base: bool,
    ) -> MemberRedriveOutcome:
        branch = f'{self.git_ops.config.branch_prefix}{mid}'
        if await self.git_ops.is_ancestor(branch, self.git_ops.config.main_branch):
            return await self._redrive_on_main(label, mid, branch, redrive_member)
        if strip_unlanded_base:
            await self._unstack(label, mid, branch)
        await redrive_member(mid, False, None)
        logger.info(
            'Train %s: member %s not on main — '
            'flipped to pending for solo-merge re-dispatch', label, mid,
        )
        return MemberRedriveOutcome.NOT_ON_MAIN

    async def _redrive_on_main(
        self, label: str, mid: str, branch: str, redrive_member: RedriveMember,
    ) -> MemberRedriveOutcome:
        sink = self.provenance_conflict_sink
        if sink is not None and sink.should_skip(mid):
            logger.info(
                'Train %s: member %s already memoized as a provenance conflict '
                '— skipping re-drive attempt', label, mid,
            )
            return MemberRedriveOutcome.PROVENANCE_CONFLICT_MEMOIZED
        verdict = await validate_landing_evidence(
            self.git_ops, mid, branch,
            branch_tip_sha=await self.git_ops.resolve_branch_sha(branch),
            pattern_template=self.git_ops.config.commit_citation_pattern,
        )
        if verdict.accepted:
            await redrive_member(mid, True, verdict.evidence_sha)
            logger.info(
                'Train %s: member %s already on main — marked done '
                '(found_on_main, sha=%s)', label, mid, verdict.evidence_sha,
            )
            return MemberRedriveOutcome.ON_MAIN_ATTRIBUTED
        file_unattributed_landing_escalation(
            self.escalation_queue, mid, branch, verdict, agent_role=self.agent_role,
        )
        await redrive_member(mid, False, None)
        logger.info(
            'Train %s: member %s on main but landing evidence unattributable '
            '(reason=%s) — escalated, flipped to pending for solo-merge re-dispatch',
            label, mid, verdict.reason,
        )
        return MemberRedriveOutcome.ON_MAIN_UNATTRIBUTED

    async def _unstack(self, label: str, mid: str, branch: str) -> None:
        """Strip an unlanded stack base from *mid* before its re-drive.  Never raises,
        so the re-drive always runs: a member left merge-deferred with no
        workflow is stranded.  A member that cannot be un-stacked is re-driven
        anyway, and its solo merge admission un-stacks it again."""
        try:
            result = await self.git_ops.unstack_from_unlanded_base(branch)
        except Exception:
            logger.exception(
                'Train %s: un-stacking member %s raised — re-driving it anyway; '
                'its solo merge admission un-stacks it again', label, mid,
            )
            return
        if result.outcome is UnstackOutcome.UNSTACKED:
            logger.info('Train %s: member %s un-stacked before re-drive', label, mid)
        elif result.stops_merge:
            logger.warning(
                'Train %s: member %s re-driven still stacked; its solo merge '
                'admission will block with: %s', label, mid, result.merge_block_reason(),
            )

    def _record_stale_evidence(self, label: str, mid: str, exc: StaleEvidenceRejection) -> None:
        if self.provenance_conflict_sink is not None:
            self.provenance_conflict_sink.record_from_rejection(
                exc, gate_source='coalesce-redrive',
            )
        logger.warning(
            'Train %s: member %s hit done_evidence_stale — evidence %s (%s) '
            'predates reopen_at %s; filed provenance_conflict escalation '
            'instead of re-driving',
            label, mid, exc.evidence_commit, exc.evidence_committed_at, exc.reopen_at,
        )
