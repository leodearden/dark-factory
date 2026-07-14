"""Tests for orchestrator.background_service (W10-η).

PRD ``plans/harness-supervision-prd.md`` §5.3 (LR-1/2/3): one reusable seam
collapsing the eleven background-loop/service lifecycles in harness.py so
the recurring shutdown-hang class (survey 2.3; tasks 108/161/162/169/875/
1080) becomes structurally impossible.

Steps covered:
  step-1   RED   — BackoffPolicy constant-delay contract
  step-2   GREEN — BackoffPolicy + DEFAULT_BACKOFF_SECS
  step-3   RED   — BackgroundService.start() idempotency
  step-4   GREEN — BackgroundService dataclass + idempotent start()
  step-5   RED   — BackgroundService loop contract (S2 / LR-1)
  step-6   GREEN — BackgroundService._loop() canonical sleep-first body
  step-7   RED   — BackgroundService.stop() contract
  step-8   GREEN — BackgroundService.stop()
  step-9   RED   — LifecycleRegistry.register()/start_all() (LR-3 start side)
  step-10  GREEN — LifecycleRegistry register/start_all
  step-11  RED   — LifecycleRegistry.stop_all() (S1 / LR-2)
  step-12  GREEN — LifecycleRegistry.stop_all()
  step-13  RED   — ManagedService adapter + LifecycleService Protocol
  step-14  GREEN — ManagedService + LifecycleService

This module imports ``orchestrator.background_service`` symbols LOCALLY
inside each test (not at module scope) so a not-yet-implemented symbol
never breaks collection of the rest of the file during the RED steps —
mirrors test_merge_queue_lifecycle_registry.py's / test_item_lifecycle.py's
convention.

This project's pyproject.toml does NOT set ``asyncio_mode = "auto"``, so
pytest-asyncio's default "strict" mode applies: every coroutine test needs
an explicit ``@pytest.mark.asyncio`` marker, applied PER TEST (never at
class or module level, since a class-level mark would error the sync tests
sharing that class) — mirrors test_merge_queue_resource_audit.py.
"""

from __future__ import annotations


class TestBackoffPolicy:
    """step-1: BackoffPolicy constant-delay contract."""

    def test_constant_delay_regardless_of_attempt(self) -> None:
        from orchestrator.background_service import BackoffPolicy

        policy = BackoffPolicy(60.0)

        assert policy.delay_for(0) == 60.0
        assert policy.delay_for(1) == 60.0
        assert policy.delay_for(5) == 60.0
        assert policy.delay_for(100) == 60.0

    def test_default_backoff_secs_matches_harness_parity_constant(self) -> None:
        from orchestrator.background_service import DEFAULT_BACKOFF_SECS
        from orchestrator.harness import _BG_LOOP_FAILURE_BACKOFF_SECS

        assert DEFAULT_BACKOFF_SECS == _BG_LOOP_FAILURE_BACKOFF_SECS
        assert DEFAULT_BACKOFF_SECS == 60.0
