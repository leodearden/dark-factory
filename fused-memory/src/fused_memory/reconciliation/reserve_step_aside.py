"""Whether reconciliation steps aside now, because the allowance reserve refuses its work.

Reconciliation is background work (plans/allowance-reserve-by-account-prd.md D13), so while the
reserve is the only open account in its protected window, a drain site asks first (§6.4) and
defers instead of spending a cycle waiting in the gate. This module holds that policy: the
question, the edge on which a refusal ends, and a deferral log quiet enough to run every tick.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from shared.account_reserve import ReserveRefusal

from fused_memory.reconciliation import RECONCILIATION_WORK_CLASS

if TYPE_CHECKING:
    from shared.usage_gate import UsageGate

logger = logging.getLogger(__name__)

DEFERRAL_LOG_INTERVAL_SECS = 600.0


class StepAsideSite(StrEnum):
    PROJECT_LOOP = 'project_loop'
    BACKLOG_CHUNK = 'backlog_chunk'
    JUDGE = 'judge'


@dataclass(frozen=True)
class ReserveCheck:
    refusal: ReserveRefusal | None
    refusal_ended: bool

    @property
    def refused(self) -> bool:
        return self.refusal is not None


class ReserveStepAside:
    """Asks *gate* whether it refuses reconciliation's class; synchronous, and claims nothing."""

    def __init__(
        self, gate: UsageGate | None, *, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._gate = gate
        self._clock = clock
        self._refused = False
        self._last_logged: dict[str, float] = {}

    def check(self, project_id: str, *, site: StepAsideSite) -> ReserveCheck:
        refusal = None if self._gate is None else self._gate.reserve_refusal(RECONCILIATION_WORK_CLASS)
        if refusal is not None:
            self._refused = True
            self._log_deferral(project_id, site, refusal)
            return ReserveCheck(refusal=refusal, refusal_ended=False)
        ended = self._refused
        self._refused = False
        if ended:
            self._last_logged.clear()
        return ReserveCheck(refusal=None, refusal_ended=ended)

    def _log_deferral(self, project_id: str, site: StepAsideSite, refusal: ReserveRefusal) -> None:
        now = self._clock()
        last = self._last_logged.get(project_id)
        if last is not None and now - last < DEFERRAL_LOG_INTERVAL_SECS:
            return
        self._last_logged[project_id] = now
        logger.info(
            'Reconciliation for project %r steps aside at %s: the allowance reserve %r refuses '
            'background work until %s; buffered events wait',
            project_id, site.value, refusal.account, refusal.release_at.isoformat(),
        )
