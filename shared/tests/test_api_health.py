"""Tests for ApiHealthGate — the C4 provider-degradation circuit breaker.

Contract: plans/server-side-api-error-handling-prd.md C4 (task 3325 / mu).
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from shared.api_health import ApiHealthGate, Closed, GateState, GateStats, Open, Probing

_WALL = datetime(2026, 7, 31, 12, 0, 0, tzinfo=UTC)


def _stats(
    *,
    failures: int = 8,
    completions: int = 8,
    distinct_tasks: int = 3,
    failure_rate: float = 1.0,
    window_secs: float = 600.0,
) -> GateStats:
    return GateStats(
        failures=failures,
        completions=completions,
        distinct_tasks=distinct_tasks,
        failure_rate=failure_rate,
        window_secs=window_secs,
    )


class TestGateSurface:
    """step-5: the constructed gate exposes PRD C4's defaults and starts Closed."""

    def test_defaults_match_prd_c4_config_block(self) -> None:
        """Defaults mirror `api_health.*` so config maps 1:1 with no translation layer."""
        gate = ApiHealthGate()
        assert gate.window_secs == 600.0
        assert gate.min_failures == 8
        assert gate.min_distinct_tasks == 3
        assert gate.failure_rate_threshold == 0.5
        assert gate.close_after_successes == 2

    def test_fresh_gate_is_closed_and_not_degraded(self) -> None:
        gate = ApiHealthGate()
        assert isinstance(gate.state(), Closed)
        assert gate.is_degraded is False

    @pytest.mark.parametrize('variant', [Closed, Open, Probing])
    def test_variants_are_gate_state_subclasses(self, variant: type) -> None:
        """isinstance(state, GateState) is the discriminator consumers hold onto."""
        assert issubclass(variant, GateState)

    def test_states_are_frozen(self) -> None:
        """state() hands out an immutable snapshot — a consumer cannot mutate gate state."""
        for state in (
            Closed(),
            Open(since=_WALL, stats=_stats()),
            Probing(since=_WALL, stats=_stats(), consecutive_successes=1),
        ):
            with pytest.raises(dataclasses.FrozenInstanceError):
                state.wat = 1  # type: ignore[attr-defined]

    def test_gate_stats_is_frozen(self) -> None:
        stats = _stats()
        with pytest.raises(dataclasses.FrozenInstanceError):
            stats.failures = 99  # type: ignore[misc]
