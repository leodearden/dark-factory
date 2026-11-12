"""Per-root burst counter of refused ``project_root`` assertions on one escalation server.

The INV-4 storm escape for the γ2 ``project_root`` guard: a caller that keeps
asserting a root this store does not serve earns ONE ``infra_issue`` L1 in the
server's own queue (``plans/escalation-store-ambiguity-prd.md`` §6.4).
"""

from __future__ import annotations

import hashlib
import logging
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from shared.storm_counter import KeyedStormCounters

from escalation.dedupe import file_or_fold_l1

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue
    from escalation.store_identity import StoreIdentity

logger = logging.getLogger(__name__)

DEFAULT_MISMATCH_STREAK_THRESHOLD = 5
DEFAULT_MISMATCH_STREAK_WINDOW_SECONDS = 3600.0

_SUBJECT_PREFIX = 'mcp-project-root-mismatch'
_AGENT_ROLE = 'harness-project-root-guard'
_CATEGORY = 'infra_issue'
_FINDING_CATEGORY = 'project_root_mismatch'
_LOG_LABEL = 'project_root mismatch storm escape'
_SUGGESTED_ACTION = (
    'find the caller asserting that root and point it at the escalation endpoint that serves it'
)


def _require_positive_int(threshold: int) -> int:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 1:
        raise ValueError(f'mismatch_streak threshold must be a positive int; got {threshold!r}')
    return threshold


def _require_positive_finite(window_seconds: float) -> float:
    if (
        isinstance(window_seconds, bool)
        or not isinstance(window_seconds, (int, float))
        or not math.isfinite(window_seconds)
        or window_seconds <= 0
    ):
        raise ValueError(
            f'mismatch_streak window_seconds must be a positive finite number; '
            f'got {window_seconds!r}'
        )
    return float(window_seconds)


class MismatchStreak:
    """Fires when one resolved asserted root is refused *threshold* times within *window_seconds*.

    A root still being refused fires again at most once per window, and that
    fire folds into the root's open record. A root with no refusal left in the
    window is forgotten, so memory holds only roots refused in the last window.
    """

    def __init__(
        self,
        queue: EscalationQueue,
        *,
        threshold: int = DEFAULT_MISMATCH_STREAK_THRESHOLD,
        window_seconds: float = DEFAULT_MISMATCH_STREAK_WINDOW_SECONDS,
        time_provider: Callable[[], float] = time.monotonic,
    ) -> None:
        self._queue = queue
        self._threshold = _require_positive_int(threshold)
        self._window_seconds = _require_positive_finite(window_seconds)
        self._now = time_provider
        self._bursts: KeyedStormCounters[Path] = KeyedStormCounters()
        self._lock = threading.Lock()

    @property
    def tracked_roots(self) -> frozenset[Path]:
        """The resolved roots currently counted.

        Public for the reason ``shared.boundary_storm_escape::BoundaryStormEscape.tracked_keys``
        is: the roots are caller-supplied, so forgetting the dormant ones is
        this object's memory bound and belongs in its interface.
        """
        with self._lock:
            return self._bursts.tracked_keys

    def record_mismatch(self, identity: StoreIdentity, asserted_root: str, refusal: str) -> bool:
        """Count one refusal of *asserted_root*; True iff this call fired."""
        resolved = Path(asserted_root).resolve()
        with self._lock:
            burst = self._bursts.record(
                resolved,
                threshold=self._threshold,
                window_seconds=self._window_seconds,
                now=self._now(),
            )
        logger.warning(
            'project_root assertion refused: %s on escalation store %s',
            resolved, identity.queue_dir,
        )
        if burst is None:
            return False
        self._file(identity, asserted_root, resolved, refusal, burst['count'])
        return True

    def _file(
        self,
        identity: StoreIdentity,
        asserted_root: str,
        resolved: Path,
        refusal: str,
        refusals_in_window: int,
    ) -> None:
        within = f'{self._window_seconds:g}s'
        summary = (
            f'project_root {asserted_root} refused {refusals_in_window} times within {within} '
            f'on escalation store {identity.queue_dir} '
            f'(kind={identity.kind}, project={identity.project_id or "n/a"})'
        )
        detail = '\n'.join([
            f'asserted_root: {asserted_root}',
            f'resolved: {resolved}',
            f'refusals_in_window: {refusals_in_window}',
            f'threshold: {self._threshold}',
            f'window_seconds: {self._window_seconds:g}',
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
            'project_root %s refused %d times within %s on escalation store %s: %s',
            asserted_root,
            refusals_in_window,
            within,
            identity.queue_dir,
            'filed a NEW infra_issue L1' if minted else 'folded into the open record or failed',
        )
