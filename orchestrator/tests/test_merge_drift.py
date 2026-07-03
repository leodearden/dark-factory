"""Tests for orchestrator.merge_drift: extracted drift-check detective
subsystem (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_drift`` exists and exports the
   full closure of moved symbols (the drift-check runner and its cadence
   gate).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_drift``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   drift-check's WARNING-level fail-open messages.
3. Reach-back / string-path monkeypatch routing — the existing test suite
   monkeypatches drift-check dependencies by STRING PATH
   ``orchestrator.merge_queue.<name>``.  A moved function must resolve a
   monkeypatched-or-staying sibling via a function-local deferred import so
   those patches stay effective even though the function body now lives in
   this module.  Each ``TestReachBackRouting`` test below patches BOTH
   namespaces (merge_drift-local naive vs. merge_queue reach-back target)
   with CONTRASTING behaviour.

   Correction to this module's own extraction-time docstring: reading the
   extracted ``_run_drift_check`` body confirms it does NOT call
   ``_run_cold_shadow_verify`` / ``_run_shadow_compare`` — drift-check and
   shadow-compare are independent sibling detectives (as the module
   docstring below already states).  The reach-back cluster that actually
   needs coverage here is the verify-pool-construction cluster
   (``build_merge_verify_spec`` / ``VerifyRunnerPool`` / ``LocalRunner`` /
   ``run_scoped_verification``), mirroring merge_shadow's
   ``_run_cold_shadow_verify`` contract — confirmed necessary by
   ``LocalRunner``'s own docstring and by the existing
   ``TestRunDriftCheck`` tests in ``test_merge_queue_multihost_wiring.py``,
   which construct a real ``HostAllocator`` and patch
   ``orchestrator.merge_queue.run_scoped_verification`` directly.
4. Shim re-export identity (added in a later step, once merge_queue.py's
   shim swap lands).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.verify import VerifyResult


def test_merge_drift_exports_moved_public_symbols() -> None:
    from orchestrator.merge_drift import _maybe_run_drift_check, _run_drift_check

    for name, obj in {
        '_run_drift_check': _run_drift_check,
        '_maybe_run_drift_check': _maybe_run_drift_check,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_drift_logger_name_is_merge_queue() -> None:
    """merge_drift emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_drift`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in test_merge_queue_multihost_wiring.py keep capturing the
    moved drift-check's WARNING-level fail-open messages after relocation.
    """
    import orchestrator.merge_drift as merge_drift

    assert merge_drift.logger.name == 'orchestrator.merge_queue'


@pytest.mark.asyncio
class TestReachBackRouting:
    """Reach-back / string-path monkeypatch routing contract.

    See the module docstring above for why this diverges from the original
    plan's "(4) _run_drift_check → _run_cold_shadow_verify / _run_shadow_compare"
    description: that call does not exist in the extracted body.
    """

    async def test_maybe_run_drift_check_reachback_to_run_drift_check(self) -> None:
        """(5) _maybe_run_drift_check must resolve _run_drift_check via
        orchestrator.merge_queue, not the co-located merge_drift copy."""
        from orchestrator.merge_drift import _maybe_run_drift_check

        git_ops = MagicMock()
        req = MagicMock()
        req.config.enabled_verify_runners = ['laptop']
        req.config.verify_drift_check_every_n_lands = 1

        worker = MagicMock()
        worker._drift_land_count = 0
        worker._drift_check_tasks = set()
        worker._ensure_host_allocator = MagicMock(return_value=MagicMock())

        naive = AsyncMock(side_effect=AssertionError(
            'naive merge_drift._run_drift_check must not be called'
        ))
        reachback = AsyncMock(return_value=None)

        with (
            patch('orchestrator.merge_drift._run_drift_check', naive),
            patch('orchestrator.merge_queue._run_drift_check', reachback),
        ):
            await _maybe_run_drift_check(worker, git_ops, req, 'commit-sha')
            assert len(worker._drift_check_tasks) == 1, (
                'expected exactly one drift-check task to be scheduled'
            )
            task = next(iter(worker._drift_check_tasks))
            await task

        naive.assert_not_called()
        reachback.assert_awaited_once()

    async def test_run_drift_check_reachback_to_verify_pool_deps(
        self, tmp_path: Path,
    ) -> None:
        """_run_drift_check must resolve build_merge_verify_spec,
        VerifyRunnerPool, LocalRunner, and run_scoped_verification via
        orchestrator.merge_queue, not the co-located merge_drift imports.

        Uses a real HostAllocator + a fake remote runner (mirrors
        TestRunDriftCheck in test_merge_queue_multihost_wiring.py) so the
        allocator-branch construction path is exercised end to end; a
        local/remote agreement is observed via the verdict_parity_ok event,
        which only fires if DriftDetector.check ever gets a real LocalRunner
        talking to the merge_queue-patched run_scoped_verification.
        """
        from orchestrator.event_store import EventStore, EventType
        from orchestrator.merge_drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-drift-reachback'
        req.task_files = ['src/foo.py']
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=tmp_path)

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        allocator = HostAllocator([fake_remote], quarantine=set())

        class _FakeEventStore(EventStore):
            def __init__(self) -> None:
                object.__init__(self)
                self.emitted: list = []

            def emit(self, event_type, *, task_id=None, data=None, **kw) -> None:  # type: ignore[override]
                self.emitted.append(event_type)

        event_store = _FakeEventStore()

        with (
            patch(
                'orchestrator.merge_drift.build_merge_verify_spec',
                MagicMock(side_effect=AssertionError('naive build_merge_verify_spec used')),
            ),
            patch(
                'orchestrator.merge_drift.VerifyRunnerPool',
                MagicMock(side_effect=AssertionError('naive VerifyRunnerPool used')),
            ),
            patch(
                'orchestrator.merge_drift.LocalRunner',
                MagicMock(side_effect=AssertionError('naive LocalRunner used')),
            ),
            patch(
                'orchestrator.merge_drift.run_scoped_verification',
                AsyncMock(side_effect=AssertionError('naive run_scoped_verification used')),
            ),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', None, event_store, set(),
                allocator=allocator,
            )

        assert EventType.verdict_parity_ok in event_store.emitted, (
            f'expected the orchestrator.merge_queue-patched dependency chain to '
            f'govern _run_drift_check and reach a local/remote parity agreement, '
            f'emitted event types: {event_store.emitted!r}'
        )


def test_merge_queue_reexports_identical_objects() -> None:
    """merge_queue re-exports the SAME objects from merge_drift (shim identity).

    Covers both of the 2 moved names.

    RED (pre-shim): merge_queue.py still defines its own independent copies
    of these names (the duplicate definitions left in place by the EXPAND
    step), so ``getattr(merge_queue, name) is getattr(merge_drift, name)``
    fails for every name — two distinct objects that merely share a name.
    """
    import orchestrator.merge_drift as merge_drift
    import orchestrator.merge_queue as merge_queue

    moved_names = [
        '_run_drift_check',
        '_maybe_run_drift_check',
    ]

    for name in moved_names:
        mq_obj = getattr(merge_queue, name)
        md_obj = getattr(merge_drift, name)
        assert mq_obj is md_obj, (
            f'{name}: orchestrator.merge_queue.{name} and '
            f'orchestrator.merge_drift.{name} must be the identical object'
        )
