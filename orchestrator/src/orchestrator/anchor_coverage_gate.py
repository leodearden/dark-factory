"""The I-6 anchor-coverage arming gate and its two INV-4 storm escapes.

plans/scheduler-dispatch-scoring-and-lock-layer-prd.md D12.  Once per scoring
tick the gate takes the candidates' ``pending_since`` readings and decides
whether the WHOLE tick scores on C2 (armed) or on the legacy formula.  The
coverage only chooses the formula; it is never a score term (I-5).

Two disjoint signals each reach a human as one deduped L1 per episode:

(a) a streak of consecutive unarmed ticks, a fleet stuck on the legacy formula;
(b) on armed ticks, a sustained share of candidates scored at age 0 because
    their anchor was absent or unparseable, the blind spot (a) cannot see.

The gate owns that state and the filing text.  Its caller delivers the
verdict: it emits the event and files the escalation.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from orchestrator.pending_age import AnchorReading, AnchorStatus
from orchestrator.sentinel_l1 import SentinelL1

# (a) 12 is the fm-read-storm precedent: it clears a routine fused-memory
#     restart yet catches a fleet stuck on the legacy formula.
STREAK_ESCALATION_THRESHOLD: int = 12
STREAK_SENTINEL: str = '__anchor_coverage_low__'
# (b) 0.02 is 10x the 0.2% residual measured live on 2026-10-12 (three tasks
#     stripped by task 4302) and well under the 10% the arming gate tolerates.
ANCHORLESS_RATE_THRESHOLD: float = 0.02
# No rate is believed below this many sampled candidates.
ANCHORLESS_MIN_SAMPLES: int = 500
# Rolling window of per-candidate samples, so a fix decays the rate instead
# of a lifetime ratio re-filing after the operator acted.  Above the floor.
ANCHORLESS_WINDOW: int = 4096
ANCHORLESS_SENTINEL: str = '__anchorless_rate__'
_IDS_SHOWN: int = 10


@dataclass(frozen=True)
class CoverageLow:
    """One unarmed tick, shaped as the ``scoring_anchor_coverage_low`` event payload."""

    coverage: float
    threshold: float
    candidates: int
    with_anchor: int
    consecutive_ticks: int


@dataclass(frozen=True)
class GateVerdict:
    armed: bool
    coverage_low: CoverageLow | None = None
    escalation: SentinelL1 | None = None


class _StatusWindow:
    """The latest *size* anchor statuses, with a per-status count kept in step."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._statuses: deque[AnchorStatus] = deque()
        self.counts: Counter[AnchorStatus] = Counter()

    def __len__(self) -> int:
        return len(self._statuses)

    def extend(self, statuses: Iterable[AnchorStatus]) -> None:
        for status in statuses:
            if len(self._statuses) == self._size:
                self.counts[self._statuses.popleft()] -= 1
            self._statuses.append(status)
            self.counts[status] += 1


class AnchorCoverageGate:
    def __init__(self) -> None:
        # Bumped by an unarmed tick and zeroed by an armed one; an idle tick
        # never scores, so it leaves the streak alone.
        self._unarmed_streak = 0
        # Statuses of candidates scored on ARMED ticks only: escape (b)'s sample.
        self._armed_window = _StatusWindow(ANCHORLESS_WINDOW)

    def observe(
        self, task_ids: Sequence[str], readings: Sequence[AnchorReading], *, threshold: float,
    ) -> GateVerdict:
        """Decide one tick from its readings, non-empty and parallel to *task_ids*."""
        covered = sum(r.covered for r in readings)
        coverage = covered / len(readings)
        if coverage >= threshold:
            self._unarmed_streak = 0
            self._armed_window.extend(r.status for r in readings)
            return GateVerdict(armed=True, escalation=self._anchorless_rate_escalation(task_ids, readings))
        self._unarmed_streak += 1
        low = CoverageLow(coverage, threshold, len(readings), covered, self._unarmed_streak)
        escalation = None
        if self._unarmed_streak >= STREAK_ESCALATION_THRESHOLD:
            escalation = _streak_escalation(low, _anchorless_ids(task_ids, readings))
        return GateVerdict(armed=False, coverage_low=low, escalation=escalation)

    def _anchorless_rate_escalation(
        self, task_ids: Sequence[str], readings: Sequence[AnchorReading],
    ) -> SentinelL1 | None:
        sampled = len(self._armed_window)
        if sampled < ANCHORLESS_MIN_SAMPLES:
            return None
        absent = self._armed_window.counts[AnchorStatus.ABSENT]
        malformed = self._armed_window.counts[AnchorStatus.MALFORMED]
        rate = (absent + malformed) / sampled
        if rate < ANCHORLESS_RATE_THRESHOLD:
            return None
        return _rate_escalation(absent, malformed, sampled, rate, _anchorless_ids(task_ids, readings))


def _anchorless_ids(task_ids: Sequence[str], readings: Sequence[AnchorReading]) -> list[str]:
    """The first few ids whose anchor was unusable, for an escalation detail."""
    ids = [tid for tid, r in zip(task_ids, readings, strict=True) if not r.covered]
    return ids[:_IDS_SHOWN]


def _streak_escalation(low: CoverageLow, ids: list[str]) -> SentinelL1:
    streak, coverage, threshold = low.consecutive_ticks, low.coverage, low.threshold
    return SentinelL1(
        STREAK_SENTINEL,
        category='infra_issue',
        summary=(
            f'Dispatch scored on the legacy formula for {streak} consecutive ticks: '
            f'pending_since anchor coverage {coverage:.1%} < {threshold:.1%}'
        ),
        detail=(
            f'For {streak} consecutive dispatch ticks only {coverage:.1%} of the '
            f'candidates carried a parseable metadata.pending_since, below '
            f'scoring.new_scoring_min_anchor_coverage ({threshold:.1%}), so every '
            f'such tick scored on the legacy formula (PRD D12, I-6).  Anchorless '
            f'candidates on the latest tick include: {", ".join(ids)}.\n\n'
            f'The anchor is written by fused-memory: stamp_pending_since on every '
            f'* -> pending landing, plus the one-shot v4 -> v5 back-fill (task 3816, '
            f'fused_memory/backends/sqlite_task_backend.py).  Resolve by running a '
            f'fused-memory that carries that code, or by re-stamping the stripped '
            f'tasks: an already-pending task is stamped only by a fresh * -> pending '
            f'landing.  The streak clears itself on the first armed tick.'
        ),
        suggested_action='restore_pending_since_anchors',
    )


def _rate_escalation(absent: int, malformed: int, sampled: int, rate: float, ids: list[str]) -> SentinelL1:
    return SentinelL1(
        ANCHORLESS_SENTINEL,
        category='risk_identified',
        summary=(
            f'{rate:.1%} of armed-tick dispatch candidates scored at age 0 for want '
            f'of a usable pending_since anchor'
        ),
        detail=(
            f'{absent + malformed} of the last {sampled} candidates scored on armed '
            f'ticks ({rate:.1%}, threshold {ANCHORLESS_RATE_THRESHOLD:.1%}) had no '
            f'usable metadata.pending_since and so scored at age 0: {absent} ABSENT '
            f'(the key is missing, as after a whole-blob metadata replace), '
            f'{malformed} MALFORMED (unparseable).  Affected candidates on the latest '
            f'tick include: {", ".join(ids)}.\n\n'
            f'The anchor is written by fused-memory stamp_pending_since on every '
            f'* -> pending landing (task 3816); an already-pending task is re-stamped '
            f'only by a fresh * -> pending landing.  Find the writer that drops or '
            f'garbles the key.  The rate is measured over a rolling window of '
            f'{ANCHORLESS_WINDOW} samples, so it decays once the anchors are restored.'
        ),
        suggested_action='find_the_writer_dropping_pending_since',
    )
