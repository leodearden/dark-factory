"""When the main-tip sweep builds warm and when it builds cold.

The sweep builds WARM (CoW-seeded from the warm-lane base) by default, and
periodically COLD as the ground-truth control (Leo ruling 2026-09-23, reify
esc-7423-8, superseding task 2567's always-cold sweep).  The cold control bounds
the exposure to a poisoned warm base to about one interval.  Only a cold sweep
that reached a verdict resets the clock.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

logger = logging.getLogger(__name__)


class SweepSeedMode(StrEnum):
    WARM = 'warm'  # seeded from the warm-lane base; falls back to cold on failure
    COLD = 'cold'  # a fresh build from nothing: the ground truth


class ColdVerdictLedger(Protocol):
    def load_main_sweep_last_cold_verdict(self, project_id: str) -> datetime | None: ...

    def save_main_sweep_cold_verdict(
        self, project_id: str, *, verdict_at: datetime, swept_sha: str,
    ) -> None: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


class MainSweepColdControl:
    """Chooses each sweep's seed mode and remembers the last cold verdict.

    Every state it cannot vouch for (no history, an unreadable ledger, a clock
    reading earlier than the last cold verdict) chooses COLD.
    """

    def __init__(
        self,
        ledger: ColdVerdictLedger | None,
        project_id: str,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._ledger = ledger
        self._project_id = project_id
        self._clock = clock
        self._last_cold_verdict_at = self._load_last_cold_verdict()

    @property
    def last_cold_verdict_at(self) -> datetime | None:
        return self._last_cold_verdict_at

    def seed_mode(self, cold_interval_secs: float) -> SweepSeedMode:
        last = self._last_cold_verdict_at
        if cold_interval_secs <= 0 or last is None:
            return SweepSeedMode.COLD
        now = self._clock()
        if now < last or now - last >= timedelta(seconds=cold_interval_secs):
            return SweepSeedMode.COLD
        return SweepSeedMode.WARM

    def record_verdict(self, mode: SweepSeedMode, swept_sha: str) -> None:
        if mode is not SweepSeedMode.COLD:
            return
        self._last_cold_verdict_at = self._clock()
        if self._ledger is None:
            return
        try:
            self._ledger.save_main_sweep_cold_verdict(
                self._project_id,
                verdict_at=self._last_cold_verdict_at,
                swept_sha=swept_sha,
            )
        except Exception:
            logger.warning(
                'main-tip sweep: could not persist the cold verdict for %s; '
                'after a restart the next sweep will run cold',
                swept_sha[:12], exc_info=True,
            )

    def _load_last_cold_verdict(self) -> datetime | None:
        if self._ledger is None:
            return None
        try:
            return self._ledger.load_main_sweep_last_cold_verdict(self._project_id)
        except Exception:
            logger.warning(
                'main-tip sweep: could not read the last cold verdict; '
                'the next sweep will run cold',
                exc_info=True,
            )
            return None
