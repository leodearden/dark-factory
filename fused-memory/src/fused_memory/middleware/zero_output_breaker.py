"""A circuit breaker that opens after consecutive zero-output LLM timeouts and stays open for a cooldown."""

from __future__ import annotations

from shared.streak import KeyedStreaks


class ZeroOutputBreaker:
    """Opens once the timeout run reaches the threshold; every timeout after the cooldown re-opens it."""

    def __init__(self, *, threshold: int, cooldown_seconds: float) -> None:
        self._threshold = threshold
        self._cooldown_seconds = cooldown_seconds
        self._timeouts: KeyedStreaks[None] = KeyedStreaks(threshold=threshold)
        self._open_until: float | None = None

    def is_open(self, now: float) -> bool:
        return self._open_until is not None and now < self._open_until

    def record_timeout(self, now: float) -> bool:
        """Count one zero-output timeout; return True when it opens the breaker."""
        if self._timeouts.failure(None).count < self._threshold:
            return False
        self._open_until = now + self._cooldown_seconds
        return True

    def close(self) -> None:
        """End the timeout run and close the breaker, on any real LLM success."""
        self._timeouts.success(None)
        self._open_until = None

    @property
    def consecutive_timeouts(self) -> int:
        return self._timeouts.count(None)

    @property
    def cooldown_seconds(self) -> float:
        return self._cooldown_seconds
