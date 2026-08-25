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

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
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


class TestDriftCheckStatePersistence:
    """Unit tests for DriftCheckState + _load_drift_check_state + _save_drift_check_state.

    Fix 1a (task 2886): the drift-check cadence counter must be PERSISTED so it
    survives the ~8h fleet redeploy that resets the in-memory worker counter
    (which is why the drift check has NEVER fired).  The persistence primitive
    is an EXACT mirror of merge_shadow's ShadowCompareState +
    _load_shadow_compare_state/_save_shadow_compare_state (fail-safe JSON), the
    same contract asserted by
    test_merge_queue_warm_cold_shadow.py::TestShadowCompareStatePersistence.

    RED (pre-impl): DriftCheckState / _load_drift_check_state /
    _save_drift_check_state do not exist yet in orchestrator.merge_drift.
    """

    def test_drift_check_state_defaults_land_count_zero(self) -> None:
        from orchestrator.merge_drift import DriftCheckState

        assert DriftCheckState().land_count == 0

    def test_load_returns_default_when_file_missing(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import DriftCheckState, _load_drift_check_state

        state = _load_drift_check_state(tmp_path / 'nonexistent.json')
        assert state == DriftCheckState(land_count=0)

    def test_load_returns_default_on_corrupt_json(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import DriftCheckState, _load_drift_check_state

        path = tmp_path / 'drift.json'
        path.write_text('{ not valid json !!!')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_load_returns_default_on_empty_json_object(self, tmp_path: Path) -> None:
        # Missing keys → fail-safe default (mirror shadow's empty-object test).
        from orchestrator.merge_drift import DriftCheckState, _load_drift_check_state

        path = tmp_path / 'drift.json'
        path.write_text('{}')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_load_returns_default_on_wrong_typed_key(self, tmp_path: Path) -> None:
        # int("not-an-int") raises ValueError → fail-safe default.
        from orchestrator.merge_drift import DriftCheckState, _load_drift_check_state

        path = tmp_path / 'drift.json'
        path.write_text('{"land_count": "not-an-int"}')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_load_returns_default_on_null_key(self, tmp_path: Path) -> None:
        # int(None) raises TypeError → fail-safe default.
        from orchestrator.merge_drift import DriftCheckState, _load_drift_check_state

        path = tmp_path / 'drift.json'
        path.write_text('{"land_count": null}')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_round_trip_preserves_count(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import (
            DriftCheckState,
            _load_drift_check_state,
            _save_drift_check_state,
        )

        path = tmp_path / 'drift.json'
        _save_drift_check_state(path, DriftCheckState(land_count=19))
        assert _load_drift_check_state(path).land_count == 19

    def test_round_trip_count_zero(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import (
            DriftCheckState,
            _load_drift_check_state,
            _save_drift_check_state,
        )

        path = tmp_path / 'drift.json'
        original = DriftCheckState(land_count=0)
        _save_drift_check_state(path, original)
        assert _load_drift_check_state(path) == original

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import DriftCheckState, _save_drift_check_state

        path = tmp_path / 'a' / 'b' / 'c' / 'drift.json'
        _save_drift_check_state(path, DriftCheckState(land_count=5))
        assert path.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import DriftCheckState, _save_drift_check_state

        path = tmp_path / 'drift.json'
        _save_drift_check_state(path, DriftCheckState(land_count=3))
        data = json.loads(path.read_text())
        assert 'land_count' in data
        assert data['land_count'] == 3


@pytest.mark.asyncio
async def test_maybe_run_drift_check_guards_against_non_positive_every_n() -> None:
    """0 or negative verify_drift_check_every_n_lands must degrade to a no-op.

    ``OrchestratorConfig`` enforces ``ge=1`` at construction (see
    test_config.py::test_orchestrator_config_verify_drift_check_every_n_lands_rejects_zero),
    but ``req.config`` is frequently a ``MagicMock``/hand-built stand-in in
    tests (as elsewhere in this file), so the function itself must not rely
    solely on that upstream validation — a 0-or-negative value must not raise
    ``ZeroDivisionError`` via ``worker._drift_land_count % every_n``.
    """
    from orchestrator.merge_drift import _maybe_run_drift_check

    git_ops = MagicMock()
    req = MagicMock()
    req.config.enabled_verify_runners = ['laptop']
    req.config.verify_drift_check_every_n_lands = 0

    worker = MagicMock()
    worker._drift_land_count = 0
    worker._drift_check_tasks = set()

    await _maybe_run_drift_check(worker, git_ops, req, 'commit-sha')

    assert worker._drift_land_count == 0, 'land count must not advance on the disabled path'
    assert len(worker._drift_check_tasks) == 0, 'no drift-check task should be scheduled'


def _build_drift_worker(state_path: Path) -> MagicMock:
    """A MagicMock SpeculativeMergeWorker wired for _maybe_run_drift_check.

    Fresh in-memory ``_drift_land_count=0`` (models the post-restart reset) but a
    caller-supplied ``_drift_state_path`` so the PERSISTED cadence can be shared
    across simulated restarts (fix 1a).
    """
    w = MagicMock()
    w._drift_land_count = 0
    w._drift_check_tasks = set()
    w._drift_state_path = state_path
    w._ensure_host_allocator = MagicMock(return_value=MagicMock())
    return w


@pytest.mark.asyncio
class TestDriftCheckCadencePersistence:
    """Fix 1a (task 2886): the drift-check cadence must use a PERSISTED counter
    so it survives the ~8h fleet redeploy that resets the in-memory worker
    counter — the root cause of the drift check having NEVER fired.

    RED (pre step-4): ``_maybe_run_drift_check`` keys the cadence off the
    in-memory ``worker._drift_land_count`` and never touches
    ``_drift_state_path``, so (a) no state file is written and (b) a fresh
    (restarted) worker resets the count to 0 and the carried-over cadence
    never fires.
    """

    async def test_persisted_counter_drives_cadence_and_is_saved(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_drift import (
            _load_drift_check_state,
            _maybe_run_drift_check,
        )

        state_path = tmp_path / 'drift_check_state.json'
        git_ops = MagicMock()
        req = MagicMock()
        req.config.enabled_verify_runners = ['laptop']
        req.config.verify_drift_check_every_n_lands = 3

        reachback = AsyncMock(return_value=None)
        worker = _build_drift_worker(state_path)
        with patch('orchestrator.merge_queue._run_drift_check', reachback):
            # Land 1: no fire; persisted count advances to 1.
            await _maybe_run_drift_check(worker, git_ops, req, 'c1')
            assert len(worker._drift_check_tasks) == 0
            assert _load_drift_check_state(state_path).land_count == 1
            # Land 2: no fire; persisted count advances to 2.
            await _maybe_run_drift_check(worker, git_ops, req, 'c2')
            assert len(worker._drift_check_tasks) == 0
            assert _load_drift_check_state(state_path).land_count == 2
            # Land 3: fires (3 % 3 == 0), keyed off the PERSISTED count.
            await _maybe_run_drift_check(worker, git_ops, req, 'c3')
            assert len(worker._drift_check_tasks) == 1
            for t in list(worker._drift_check_tasks):
                await t
        assert reachback.await_count == 1

    async def test_cadence_survives_simulated_restart(self, tmp_path: Path) -> None:
        from orchestrator.merge_drift import _maybe_run_drift_check

        state_path = tmp_path / 'drift_check_state.json'
        git_ops = MagicMock()
        req = MagicMock()
        req.config.enabled_verify_runners = ['laptop']
        req.config.verify_drift_check_every_n_lands = 3

        reachback = AsyncMock(return_value=None)
        with patch('orchestrator.merge_queue._run_drift_check', reachback):
            # Worker A observes 2 lands (no fire), then the process restarts.
            worker_a = _build_drift_worker(state_path)
            await _maybe_run_drift_check(worker_a, git_ops, req, 'c1')
            await _maybe_run_drift_check(worker_a, git_ops, req, 'c2')
            assert len(worker_a._drift_check_tasks) == 0
            assert reachback.await_count == 0

            # RESTART: a FRESH worker with the in-memory counter reset to 0 but
            # the SAME persisted state path.  The 3rd land must still fire
            # because the persisted count (2) carried over — proving the
            # cadence does not depend on the in-memory _drift_land_count.
            worker_b = _build_drift_worker(state_path)
            assert worker_b._drift_land_count == 0  # in-memory counter reset
            await _maybe_run_drift_check(worker_b, git_ops, req, 'c3')
            assert len(worker_b._drift_check_tasks) == 1
            for t in list(worker_b._drift_check_tasks):
                await t
        assert reachback.await_count == 1


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

    async def test_run_drift_check_full_gate_when_task_files_none(
        self, tmp_path: Path,
    ) -> None:
        """Fix 1b (task 2886): with req.task_files=None + enabled Lever-C
        runners — the branch that USED to derive dispatching-host task_files —
        _run_drift_check must now build a FULL-GATE spec (task_files=None) and
        NOT invoke _derive_task_files_from_git.

        Complements TestDriftCheckFullGateSpec's explicit-task_files case by
        covering the previously-deriving branch.  Still exercises reach-back
        routing (the build_merge_verify_spec spy only captures calls when
        _run_drift_check resolves it via orchestrator.merge_queue).  Before
        fix 1b this asserted the derived files flowed into the spec; that
        derivation path is gone.
        """
        import orchestrator.verify_runner as _vr
        from orchestrator.event_store import EventStore
        from orchestrator.merge_drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-drift-derive'
        req.task_files = None
        req.module_configs = []
        # enabled_verify_runners is a read-only property derived from
        # verify_runners — must be populated at construction, not assigned.
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[  # type: ignore[arg-type]
                {'name': 'laptop', 'ssh_host': 'laptop.local', 'git_remote': 'origin'},
            ],
        )

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

        spec_calls = []
        orig_build_spec = _vr.build_merge_verify_spec

        def spy_build_spec(config, module_configs, task_files, **kw):
            spec_calls.append(task_files)
            return orig_build_spec(config, module_configs, task_files, **kw)

        derive_spy = AsyncMock(return_value=['derived/from/mq.py'])
        with (
            patch(
                'orchestrator.merge_queue.build_merge_verify_spec',
                side_effect=spy_build_spec,
            ),
            patch(
                'orchestrator.merge_queue._derive_task_files_from_git',
                derive_spy,
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

        assert spec_calls, 'expected build_merge_verify_spec to be called at least once'
        assert spec_calls[0] is None, (
            f'drift spec must be FULL-GATE (task_files=None) even when '
            f'req.task_files is None; got task_files={spec_calls[0]!r}'
        )
        derive_spy.assert_not_awaited()


@pytest.mark.asyncio
class TestDriftCheckFullGateSpec:
    """Fix 1b (task 2886, PRD §8δ): the drift check must re-dispatch a
    FULL-GATE spec (task_files=None) to both hosts, NOT the scoped/no-source
    spec that produced the (trivial) pass under investigation.

    Re-dispatching the SAME scoped spec trivially passes on both hosts and
    structurally cannot catch the trivial-pass divergence class — so the drift
    check must force the complete workspace gate on the local trust-anchor AND
    the eligible remote.

    RED (pre step-6): _run_drift_check scopes the spec to req.task_files (or the
    git-derived task_files) instead of passing task_files=None.
    """

    async def test_run_drift_check_builds_full_gate_spec(self, tmp_path: Path) -> None:
        import orchestrator.verify_runner as _vr
        from orchestrator.merge_drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_throwaway_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-drift-fullgate'
        # Scoping signal PRESENT: an explicit task_files list that the pre-fix
        # code would tuple-ify and pass straight into build_merge_verify_spec.
        req.task_files = ['src/foo.py']
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[  # type: ignore[arg-type]
                {'name': 'laptop', 'ssh_host': 'laptop.local', 'git_remote': 'origin'},
            ],
        )

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        allocator = HostAllocator([fake_remote], quarantine=set())

        spec_calls: list = []
        orig_build_spec = _vr.build_merge_verify_spec

        def spy_build_spec(config, module_configs, task_files, **kw):
            spec_calls.append(task_files)
            return orig_build_spec(config, module_configs, task_files, **kw)

        with (
            patch(
                'orchestrator.merge_queue.build_merge_verify_spec',
                side_effect=spy_build_spec,
            ),
            # A full-gate drift spec must NOT derive/scope task_files at all.
            patch(
                'orchestrator.merge_queue._derive_task_files_from_git',
                AsyncMock(side_effect=AssertionError(
                    'drift check must NOT derive task_files for a full-gate spec'
                )),
            ),
            patch(
                'orchestrator.merge_queue.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', None, None, set(),
                allocator=allocator,
            )

        assert spec_calls, 'expected build_merge_verify_spec to be called at least once'
        assert spec_calls[0] is None, (
            f'drift spec must be FULL-GATE (task_files=None) so both hosts run the '
            f'complete suite and CAN diverge; got task_files={spec_calls[0]!r}'
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


# ---------------------------------------------------------------------------
# task 3018 (steps 11-12): a LIVE drift-check throwaway worktree must survive a
# concurrent periodic reap.
#
# Sibling of TestColdShadowVerifyHoldsLaneLease in test_merge_shadow.py — the
# drift detective creates its throwaway `_merge-<uuid>` worktree through the
# very same `create_throwaway_verify_worktree` primitive and runs a full merge
# verify inside it, so it inherits the identical exposure: the tree is neither
# registered in `_owned_merge_worktrees` (the reap's owned-ledger skip misses
# it) nor touched by `_touch_owned_merge_worktrees` (its measured age is real
# elapsed time), leaving the per-lane merge-verify flock consulted by
# `remove_merge_worktree_guarded` as its only liveness protection.
#
# Deliberately uses a REAL git repo + REAL throwaway worktree (not the
# MagicMock git_ops the reach-back tests above use) so `lane_lock_path(wt)`
# names a real file and the flock contention is genuine — a mocked git_ops
# cannot exercise a real lock.  The pool half is built the way the existing
# TestReachBackRouting tests build it: a real HostAllocator plus a fake remote,
# which is what gets DriftDetector.check past its `local is None or remote is
# None` INCONCLUSIVE early-return and into the LocalRunner that reaches back to
# the patched `orchestrator.merge_queue.run_scoped_verification`.
# ---------------------------------------------------------------------------


async def _setup_drift_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _drift_head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


@pytest.fixture
def drift_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_drift_repo(repo))
    return repo


@pytest.fixture
def drift_git_ops(drift_git_repo: Path) -> GitOps:
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(git_config, drift_git_repo)


@pytest.mark.asyncio
class TestDriftCheckHoldsLaneLease:
    """_run_drift_check must hold merge_verify_lease(lane_dir=wt) (task 3018).

    The lane flock is the codebase's canonical "this tree is live" signal — it
    is exactly what `remove_merge_worktree_guarded`'s acquire-then-remove C1
    primitive consults, and therefore what the periodic reap routes its
    removals through.  Holding it across the drift verify makes the throwaway
    tree's protection independent of how long that verify runs, rather than
    resting on an age heuristic.
    """

    async def test_live_drift_check_worktree_survives_concurrent_reap(
        self, drift_git_ops: GitOps, drift_git_repo: Path,
    ) -> None:
        """A periodic reap firing mid drift-check must be REFUSED, and the tree
        must still be cleaned up once the check returns.

        Three assertions, in order of load-bearingness:

        (a) the simulated sweep's outcome is ``'skipped_lease_held'`` — a
            concurrent periodic reap CANNOT delete the checkout out from under
            a running drift verify.  This is the assertion that goes RED today:
            with no lease held, the sweep acquires the uncontended
            ``<wt>.lock`` and returns ``'removed'``.
        (b) the worktree still existed at that moment (so the skip is real, not
            an artefact of the tree already being gone).
        (c) AFTER `_run_drift_check` returns the worktree is GONE — pinning
            that the lease is released BEFORE the existing
            ``finally: cleanup_merge_worktree(wt)``.  If the lease wrapped the
            finally too, the guarded removal's NON-BLOCKING acquire would fail
            against OURSELVES and return ``'skipped_lease_held'``, leaking the
            very tree the finally exists to remove — a worse leak than the one
            being fixed.  (c) passes vacuously in the RED state, which is why
            (a) leads.

        `_run_drift_check` catches and logs every exception per its docstring
        contract ("this detective control never crashes the worker"), so these
        assertions read the outcome RECORDED by the fake rather than relying on
        the function raising.
        """
        from orchestrator.merge_drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        head = await _drift_head_sha(drift_git_ops.project_root)
        recorded: list[tuple[str, bool, Path]] = []

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )

        async def _reaping_scoped(worktree: Path, *args, **kwargs) -> VerifyResult:
            # Simulate the task-3018 periodic sweep firing WHILE the drift
            # verify is running, via the very primitive
            # reap_orphaned_merge_worktrees routes its removals through.
            outcome = await drift_git_ops.remove_merge_worktree_guarded(
                worktree, reason='periodic-reap-sim',
            )
            recorded.append((outcome, worktree.exists(), worktree))
            return pass_result

        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        allocator = HostAllocator([fake_remote], quarantine=set())

        req = MagicMock()
        req.task_id = 'task-3018-drift-lease'
        req.task_files = ['src/foo.py']
        req.module_configs = []
        req.config = OrchestratorConfig(project_root=drift_git_repo)

        with patch(
            'orchestrator.merge_queue.run_scoped_verification', _reaping_scoped,
        ):
            await _run_drift_check(
                drift_git_ops, req, head, None, None, set(), allocator=allocator,
            )

        assert recorded, (
            'the patched run_scoped_verification never ran, so the concurrent '
            'reap was never simulated — the test proves nothing'
        )
        outcome, existed_mid_verify, wt = recorded[0]
        assert outcome == 'skipped_lease_held', (
            f'a periodic reap firing during a LIVE drift check must be refused '
            f'by the lane flock, got {outcome!r} — the throwaway worktree at '
            f'{wt} would have been deleted out from under the running verify'
        )
        assert existed_mid_verify is True, (
            f'the throwaway worktree {wt} must still exist at the moment the '
            f'reap is refused (otherwise the skip is vacuous)'
        )
        assert not wt.exists(), (
            f'the lease must be released BEFORE the finally-block '
            f'cleanup_merge_worktree, else the cleanup skips on our own lease '
            f'and leaks {wt}'
        )
