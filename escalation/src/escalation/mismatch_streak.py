"""Per-root streak of refused ``project_root`` assertions on one escalation server.

The INV-4 storm escape for the γ2 ``project_root`` guard: a caller that keeps
asserting a root this store does not serve earns ONE ``infra_issue`` L1 in the
server's own queue (``plans/escalation-store-ambiguity-prd.md`` §6.4).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from escalation.dedupe import file_or_fold_l1

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue
    from escalation.store_identity import StoreIdentity

logger = logging.getLogger(__name__)

DEFAULT_MISMATCH_STREAK_THRESHOLD = 5

_SUBJECT_PREFIX = 'mcp-project-root-mismatch'
_AGENT_ROLE = 'harness-project-root-guard'
_CATEGORY = 'infra_issue'
_FINDING_CATEGORY = 'project_root_mismatch'
_LOG_LABEL = 'project_root mismatch storm escape'
_SUGGESTED_ACTION = (
    'find the caller asserting that root and point it at the escalation endpoint that serves it'
)


class MismatchStreak:
    """Counts refusals per resolved asserted root; every *threshold*-th one fires and re-arms."""

    def __init__(
        self, queue: EscalationQueue, *, threshold: int = DEFAULT_MISMATCH_STREAK_THRESHOLD,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
            raise ValueError(
                f'mismatch_streak threshold must be a positive int; got {threshold!r}'
            )
        self._queue = queue
        self._threshold = threshold
        self._streaks: dict[Path, int] = {}
        self._lock = threading.Lock()

    def record_mismatch(self, identity: StoreIdentity, asserted_root: str, refusal: str) -> bool:
        """Count one refusal of *asserted_root*; True iff this call completed a streak."""
        resolved = Path(asserted_root).resolve()
        with self._lock:
            count = self._streaks.get(resolved, 0) + 1
            fire = count >= self._threshold
            if fire:
                self._streaks.pop(resolved, None)
            else:
                self._streaks[resolved] = count
        logger.warning(
            'project_root assertion refused: mismatch_streak for %s at %d/%d',
            resolved, count, self._threshold,
        )
        if not fire:
            return False
        self._file(identity, asserted_root, resolved, refusal)
        return True

    def _file(
        self, identity: StoreIdentity, asserted_root: str, resolved: Path, refusal: str,
    ) -> None:
        summary = (
            f'project_root {asserted_root} mismatched {self._threshold} consecutive times '
            f'on escalation store {identity.queue_dir} '
            f'(kind={identity.kind}, project={identity.project_id or "n/a"})'
        )
        detail = '\n'.join([
            f'asserted_root: {asserted_root}',
            f'resolved: {resolved}',
            f'threshold: {self._threshold}',
            f'queue_dir: {identity.queue_dir}',
            f'kind: {identity.kind}',
            f'project_id: {identity.project_id}',
            f'wired project_root: {identity.project_root}',
            f'suggested_action: {_SUGGESTED_ACTION}',
            f'refusal: {refusal}',
        ])
        minted = file_or_fold_l1(
            self._queue,
            project_id=identity.project_id or identity.kind,
            subject=f'{_SUBJECT_PREFIX}-{hashlib.sha256(bytes(resolved)).hexdigest()[:16]}',
            category=_CATEGORY,
            finding_category=_FINDING_CATEGORY,
            agent_role=_AGENT_ROLE,
            summary=summary,
            detail=detail,
            log=logger,
            log_label=_LOG_LABEL,
        )
        logger.error(
            'project_root %s mismatched %d consecutive times on escalation store %s: %s',
            asserted_root,
            self._threshold,
            identity.queue_dir,
            'filed a NEW infra_issue L1' if minted else 'folded into the open record or failed',
        )
