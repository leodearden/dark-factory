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

    async def test_run_drift_check_reachback_for_task_file_derivation(
        self, tmp_path: Path,
    ) -> None:
        """_run_drift_check must resolve _derive_task_files_from_git via
        orchestrator.merge_queue (not a merge_drift-local binding) when
        task_files is None and Lever C verify runners are enabled.

        This exercises the derivation branch that test_run_drift_check_reachback_to_verify_pool_deps
        above does not: that test always supplies an explicit task_files list.
        Mirrors test_dispatching_host_derives_task_files_when_enabled_runners in
        test_merge_queue_multihost_wiring.py, which covers the identical gate for
        _run_post_merge_verify's own dispatching-host derivation path.
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

        with (
            patch(
                'orchestrator.merge_queue.build_merge_verify_spec',
                side_effect=spy_build_spec,
            ),
            patch(
                'orchestrator.merge_queue._derive_task_files_from_git',
                AsyncMock(return_value=['derived/from/mq.py']),
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
        assert spec_calls[0] == ('derived/from/mq.py',), (
            f'expected the orchestrator.merge_queue-patched _derive_task_files_from_git '
            f'to flow into the built spec, got task_files={spec_calls[0]!r}'
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
