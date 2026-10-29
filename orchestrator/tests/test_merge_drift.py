"""Tests for orchestrator.merge_lane.drift: extracted drift-check detective
subsystem (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_lane.drift`` exists and exports the
   full closure of moved symbols (the drift-check runner and its cadence
   gate).
2. Logger-name — the module logs under the lane's one logger name,
   ``orchestrator.merge_lane`` (task 5037), so ``caplog`` assertions
   filtered to that name keep capturing the moved
   drift-check's WARNING-level fail-open messages.
3. Verify-pool construction — ``TestDriftCheckFullGateSpecNoDerivation``
   spies on ``build_merge_verify_spec`` where ``_run_drift_check`` looks it
   up, and the local trust-anchor verifies through the injected
   :class:`~orchestrator.merge_lane.ports.VerifyPort`.  ``_run_drift_check``
   does NOT call ``_run_cold_shadow_verify`` / ``_run_shadow_compare`` —
   drift-check and shadow-compare are independent sibling detectives.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _merge_lane_fakes import FakeVerifier, hangs_until
from _orch_helpers import wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig, VerifyRunnerConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane.drift import (
    DriftCheckState,
    _load_drift_check_state,
    _maybe_run_drift_check,
    _run_drift_check,
    _save_drift_check_state,
)
from orchestrator.merge_lane.worker import PRODUCTION_VERIFIER
from orchestrator.verify import VerifyResult
from orchestrator.verify_runner import HostAllocator

# Lever C explicitly OFF for the drift tests that are not about it (task 4537).
#
# conftest.py::_isolate_orch_config is AUTOUSE and pins ORCH_CONFIG_PATH at this
# repo's OWN dark-factory-orchestrator.yaml, which DECLARES verify_env_probes.
# A real OrchestratorConfig built in a test therefore INHERITS them, and
# orchestrator/src/orchestrator/merge_lane/drift.py::_run_drift_check calls
# check_env with no `local_run` — so the
# local fingerprint is captured through verify_runner.py::env_probe_local_run,
# i.e. real create_subprocess_exec calls against whatever is on PATH (npx, uv,
# node, python3, plus the hardcoded rustc/cargo/sccache), none of them
# timeout-bounded and npx reaching the npm registry on a cold cache.
#
# check_env probes the remote FIRST, so these tests' MagicMock remote fails the
# capture before any of that happens.  Relying on that would be relying on an
# accident of the fake, and it still logs a WARNING traceback per test.  Saying
# "no probes" out loud keeps them hermetic on purpose and by construction — the
# guard order in check_env makes an empty list return before it touches a host.
_LEVER_C_OFF: list = []


def test_merge_drift_exports_moved_public_symbols() -> None:
    from orchestrator.merge_lane.drift import _maybe_run_drift_check, _run_drift_check

    for name, obj in {
        '_run_drift_check': _run_drift_check,
        '_maybe_run_drift_check': _maybe_run_drift_check,
    }.items():
        assert obj is not None, f'{name} must not be None'


def test_merge_drift_logger_name_is_merge_lane() -> None:
    """The drift module logs under the lane's one name, 'orchestrator.merge_lane'.

    Every merge-lane module shares it (task 5037), so a
    ``caplog.at_level(..., logger='orchestrator.merge_lane')`` filter
    captures this module's records too.
    """
    import orchestrator.merge_lane.drift as merge_drift

    assert merge_drift.logger.name == 'orchestrator.merge_lane'


class TestDriftCheckStatePersistence:
    """Unit tests for DriftCheckState + _load_drift_check_state + _save_drift_check_state.

    Fix 1a (task 2886): the drift-check cadence counter must be PERSISTED so it
    survives the ~8h fleet redeploy that resets the in-memory worker counter
    (which is why the drift check has NEVER fired).  The persistence primitive
    is modelled on merge_shadow's ShadowCompareState +
    _load_shadow_compare_state/_save_shadow_compare_state (fail-safe JSON), the
    same contract asserted by
    test_merge_queue_warm_cold_shadow.py::TestShadowCompareStatePersistence.
    """

    def test_drift_check_state_defaults_land_count_zero(self) -> None:
        assert DriftCheckState().land_count == 0

    def test_load_returns_default_when_file_missing(self, tmp_path: Path) -> None:
        state = _load_drift_check_state(tmp_path / 'nonexistent.json')
        assert state == DriftCheckState(land_count=0)

    def test_load_returns_default_on_corrupt_json(self, tmp_path: Path) -> None:
        path = tmp_path / 'drift.json'
        path.write_text('{ not valid json !!!')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_load_returns_default_on_empty_json_object(self, tmp_path: Path) -> None:
        # Missing keys → fail-safe default (mirror shadow's empty-object test).
        path = tmp_path / 'drift.json'
        path.write_text('{}')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_load_returns_default_on_wrong_typed_key(self, tmp_path: Path) -> None:
        # int("not-an-int") raises ValueError → fail-safe default.
        path = tmp_path / 'drift.json'
        path.write_text('{"land_count": "not-an-int"}')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_load_returns_default_on_null_key(self, tmp_path: Path) -> None:
        # int(None) raises TypeError → fail-safe default.
        path = tmp_path / 'drift.json'
        path.write_text('{"land_count": null}')
        assert _load_drift_check_state(path) == DriftCheckState(land_count=0)

    def test_round_trip_preserves_count(self, tmp_path: Path) -> None:
        path = tmp_path / 'drift.json'
        _save_drift_check_state(path, DriftCheckState(land_count=19))
        assert _load_drift_check_state(path).land_count == 19

    def test_round_trip_count_zero(self, tmp_path: Path) -> None:
        path = tmp_path / 'drift.json'
        original = DriftCheckState(land_count=0)
        _save_drift_check_state(path, original)
        assert _load_drift_check_state(path) == original

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / 'a' / 'b' / 'c' / 'drift.json'
        _save_drift_check_state(path, DriftCheckState(land_count=5))
        assert path.exists()

    def test_save_writes_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / 'drift.json'
        _save_drift_check_state(path, DriftCheckState(land_count=3))
        data = json.loads(path.read_text())
        assert 'land_count' in data
        assert data['land_count'] == 3

    # -- task 6648: a due check that could not get its host slots is deferred --

    def test_drift_check_state_defaults_no_deferrals(self) -> None:
        assert DriftCheckState().consecutive_deferrals == 0

    def test_load_old_format_file_keeps_land_count_and_no_deferrals(
        self, tmp_path: Path,
    ) -> None:
        # A state file written before deferrals existed must not reset the cadence.
        path = tmp_path / 'drift.json'
        path.write_text('{"land_count": 7}')
        assert _load_drift_check_state(path) == DriftCheckState(
            land_count=7, consecutive_deferrals=0,
        )

    def test_round_trip_preserves_consecutive_deferrals(self, tmp_path: Path) -> None:
        path = tmp_path / 'drift.json'
        original = DriftCheckState(land_count=12, consecutive_deferrals=3)
        _save_drift_check_state(path, original)
        assert _load_drift_check_state(path) == original

    def test_due_on_cadence_multiple_or_after_a_deferral(self) -> None:
        assert DriftCheckState(land_count=6).is_due(3) is True
        assert DriftCheckState(land_count=7).is_due(3) is False
        assert DriftCheckState(land_count=7, consecutive_deferrals=1).is_due(3) is True

    def test_transitions_return_new_values(self) -> None:
        original = DriftCheckState(land_count=4, consecutive_deferrals=2)
        assert original.after_land() == DriftCheckState(land_count=5, consecutive_deferrals=2)
        assert original.after_deferral() == DriftCheckState(land_count=4, consecutive_deferrals=3)
        assert original.after_start() == DriftCheckState(land_count=4, consecutive_deferrals=0)
        assert original == DriftCheckState(land_count=4, consecutive_deferrals=2)


@pytest.mark.asyncio
async def test_maybe_run_drift_check_guards_against_non_positive_every_n(
    tmp_path: Path,
) -> None:
    """0 or negative verify_drift_check_every_n_lands must degrade to a no-op.

    ``OrchestratorConfig`` enforces ``ge=1`` at construction (see
    test_config.py::test_orchestrator_config_verify_drift_check_every_n_lands_rejects_zero),
    but ``req.config`` is frequently a ``MagicMock``/hand-built stand-in in
    tests (as elsewhere in this file), so the function itself must not rely
    solely on that upstream validation — a 0-or-negative value must not raise
    ``ZeroDivisionError`` via ``worker._drift_land_count % every_n``.
    """
    from orchestrator.merge_lane.drift import (
        _load_drift_check_state,
        _maybe_run_drift_check,
    )

    git_ops = MagicMock()
    req = MagicMock()
    req.config.enabled_verify_runners = ['laptop']
    req.config.verify_drift_check_every_n_lands = 0

    state_path = tmp_path / 'drift_check_state.json'
    worker = _build_drift_worker(state_path)

    await _maybe_run_drift_check(worker, git_ops, req, 'commit-sha', verifier=PRODUCTION_VERIFIER)

    assert _load_drift_check_state(state_path).land_count == 0, (
        'land count must not advance on the disabled path'
    )
    assert len(worker._drift_check_tasks) == 0, 'no drift-check task should be scheduled'


def _build_drift_worker(
    state_path: Path,
    *,
    allocator: HostAllocator | None = None,
    event_store: EventStore | None = None,
) -> MagicMock:
    """A MagicMock SpeculativeMergeWorker wired for _maybe_run_drift_check.

    Fresh in-memory ``_drift_land_count=0`` (models the post-restart reset) but a
    caller-supplied ``_drift_state_path`` so the PERSISTED cadence can be shared
    across simulated restarts (fix 1a).  *allocator* and *event_store*, when
    given, are what a spawned drift check acquires from and emits to.

    The private attributes below are the function-under-test's OWN parameter
    surface: ``_maybe_run_drift_check`` takes the worker and reads them, so a
    stand-in must carry them.  Retiring these needs a production signature
    change, which task 5031 forbids.
    """
    w = MagicMock()
    w._drift_land_count = 0
    w._drift_check_tasks = set()
    w._drift_state_path = state_path
    w._ensure_host_allocator = MagicMock(
        return_value=MagicMock() if allocator is None else allocator,
    )
    w._event_store = event_store
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
        from orchestrator.merge_lane.drift import (
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
        with patch('orchestrator.merge_lane.drift._run_drift_check', reachback):
            # Land 1: no fire; persisted count advances to 1.
            await _maybe_run_drift_check(worker, git_ops, req, 'c1', verifier=PRODUCTION_VERIFIER)
            assert len(worker._drift_check_tasks) == 0
            assert _load_drift_check_state(state_path).land_count == 1
            # Land 2: no fire; persisted count advances to 2.
            await _maybe_run_drift_check(worker, git_ops, req, 'c2', verifier=PRODUCTION_VERIFIER)
            assert len(worker._drift_check_tasks) == 0
            assert _load_drift_check_state(state_path).land_count == 2
            # Land 3: fires (3 % 3 == 0), keyed off the PERSISTED count.
            await _maybe_run_drift_check(worker, git_ops, req, 'c3', verifier=PRODUCTION_VERIFIER)
            assert len(worker._drift_check_tasks) == 1
            for t in list(worker._drift_check_tasks):
                await t
        assert reachback.await_count == 1

    async def test_cadence_survives_simulated_restart(self, tmp_path: Path) -> None:
        from orchestrator.merge_lane.drift import (
            _load_drift_check_state,
            _maybe_run_drift_check,
        )

        state_path = tmp_path / 'drift_check_state.json'
        git_ops = MagicMock()
        req = MagicMock()
        req.config.enabled_verify_runners = ['laptop']
        req.config.verify_drift_check_every_n_lands = 3

        reachback = AsyncMock(return_value=None)
        with patch('orchestrator.merge_lane.drift._run_drift_check', reachback):
            # Worker A observes 2 lands (no fire), then the process restarts.
            worker_a = _build_drift_worker(state_path)
            await _maybe_run_drift_check(worker_a, git_ops, req, 'c1', verifier=PRODUCTION_VERIFIER)
            await _maybe_run_drift_check(worker_a, git_ops, req, 'c2', verifier=PRODUCTION_VERIFIER)
            assert len(worker_a._drift_check_tasks) == 0
            assert reachback.await_count == 0

            # RESTART: a FRESH worker with the in-memory counter reset to 0 but
            # the SAME persisted state path.  The 3rd land must still fire
            # because the persisted count (2) carried over — proving the
            # cadence does not depend on the in-memory _drift_land_count.
            worker_b = _build_drift_worker(state_path)
            # The persisted count (2) is what must carry over; the fresh
            # worker's in-memory counter is vestigial (merge_drift.py:407-420).
            assert _load_drift_check_state(state_path).land_count == 2
            await _maybe_run_drift_check(worker_b, git_ops, req, 'c3', verifier=PRODUCTION_VERIFIER)
            assert len(worker_b._drift_check_tasks) == 1
            for t in list(worker_b._drift_check_tasks):
                await t
        assert reachback.await_count == 1


@pytest.mark.asyncio
class TestDriftCheckFullGateSpecNoDerivation:
    """The drift check re-dispatches a FULL-GATE spec and never derives
    task_files, on BOTH the explicit-task_files and the task_files=None
    branch (the latter is the one that USED to derive dispatching-host files).
    """

    @pytest.mark.parametrize('task_files', [None, ['src/foo.py']])
    async def test_run_drift_check_builds_full_gate_spec(
        self, tmp_path: Path, task_files: list[str] | None,
    ) -> None:
        """Fix 1b (task 2886, PRD §8δ): the drift check must re-dispatch a
        FULL-GATE spec (task_files=None) to both hosts, NOT the scoped/
        no-source spec that produced the (trivial) pass under investigation.
        Re-dispatching the SAME scoped spec trivially passes on both hosts and
        structurally cannot catch the trivial-pass divergence class.

        Both branches are covered by the one parameter:

        * ``task_files=None``   — the branch that USED to derive
          dispatching-host task_files.  It must build a full-gate spec and NOT
          invoke ``_derive_task_files_from_git``.
        * ``['src/foo.py']``    — a scoping signal PRESENT, which the pre-fix
          code would tuple-ify straight into ``build_merge_verify_spec``.

        Both assert the identical outcome (``spec_calls[0] is None``, no
        derivation), which is why they are one test.

        The ``build_merge_verify_spec`` spy is patched where ``_run_drift_check``
        looks it up (``orchestrator.merge_lane.drift``).
        """
        import orchestrator.verify_runner as _vr
        from orchestrator.event_store import EventStore
        from orchestrator.merge_lane.drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-drift-fullgate'
        req.task_files = task_files
        req.module_configs = []
        # enabled_verify_runners is a read-only property derived from
        # verify_runners — must be populated at construction, not assigned.
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[  # type: ignore[arg-type]
                {'name': 'laptop', 'ssh_host': 'laptop.local', 'git_remote': 'origin',
                 'df_checkout_path': '/df'},
            ],
            verify_env_probes=_LEVER_C_OFF,
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

        spec_calls: list = []
        orig_build_spec = _vr.build_merge_verify_spec

        def spy_build_spec(config, module_configs, task_files, **kw):
            spec_calls.append(task_files)
            return orig_build_spec(config, module_configs, task_files, **kw)

        derive_spy = AsyncMock(return_value=['derived/from/mq.py'])
        with (
            patch(
                'orchestrator.merge_lane.drift.build_merge_verify_spec',
                side_effect=spy_build_spec,
            ),
            # A full-gate drift spec must NOT derive/scope task_files at all.
            patch(
                'orchestrator.merge_lane.worker._derive_task_files_from_git',
                derive_spy,
            ),
            patch(
                'orchestrator.merge_lane.worker.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', None, event_store, set(),
                allocator=allocator,
                verifier=PRODUCTION_VERIFIER,
            )

        assert spec_calls, 'expected build_merge_verify_spec to be called at least once'
        assert spec_calls[0] is None, (
            f'drift spec must be FULL-GATE (task_files=None) so both hosts run the '
            f'complete suite and CAN diverge; got task_files={spec_calls[0]!r}'
        )
        derive_spy.assert_not_awaited()


# ---------------------------------------------------------------------------
# task 4537 step-15: env-divergence detection wired into the LIVE drift path
# ---------------------------------------------------------------------------

_ENV_PROBE_CONFIG: Any = [{'key': 'pyright', 'argv': ['npx', 'pyright', '--version']}]

# Measured on the two real hosts 2026-08-20: the workstation and leo-laptop
# resolve DIFFERENT unpinned pyright builds through bare `npx pyright`.
_LOCAL_PROBE_OUTPUT = {'npx pyright --version': 'pyright 1.1.403\n'}
_REMOTE_PROBE_OUTPUT_MATCHING = dict(_LOCAL_PROBE_OUTPUT)
_REMOTE_PROBE_OUTPUT_SKEWED = {'npx pyright --version': 'pyright 1.1.377\n'}


def _make_env_probe_seam(outputs):
    """async (argv, *, cwd=None) probe seam driven by a joined-argv → stdout map.

    Records every argv it is handed on ``.issued`` so a test can assert that
    the OPERATOR-declared probe reached the host, rather than some hardcoded
    set.  An unmapped command returns rc=127 — the 'command not found' rc
    capture_env_fingerprint normalises a local ENOENT to — so the Rust probes
    read as uniformly-absent on both sides instead of raising.
    """
    issued: list[list[str]] = []

    async def run(argv, *, cwd=None):
        issued.append(list(argv))
        joined = ' '.join(argv)
        if joined in outputs:
            return (0, outputs[joined], '')
        return (127, '', f'{argv[0]}: command not found')

    recording: Any = run
    recording.issued = issued
    return recording


async def _unscoped_seam_tripwire(argv, *, cwd=None):
    """Stand-in for ``_default_subprocess_run`` that must never be called.

    ``check_env`` routes its LOCAL capture through
    ``verify_runner.env_probe_local_run`` (the ``_target_subprocess_env``
    scrub), NOT through ``capture_env_fingerprint``'s own
    ``_default_subprocess_run`` default, which passes no ``env=`` and so
    inherits the orchestrator daemon's ``uv run`` venv activation.  Patching
    the fallback with this keeps the wiring tests off the real machine either
    way, and turns a routing regression into a legible failure instead of a
    surprise `npx pyright --version` against the npm registry.
    """
    raise AssertionError(
        'the local env probe was issued through _default_subprocess_run — the '
        "unscoped seam that inherits the daemon's venv activation, under which "
        f'`python3` resolves .venv/bin/python3 — for argv={argv!r}'
    )


class _RecordingEventStore(EventStore):
    """Records (event_type, data) pairs — the drift path's None-safe sink.

    Subclasses EventStore (skipping its __init__, so no sqlite file is opened)
    rather than duck-typing it: ``_run_drift_check``'s ``event_store``
    parameter is declared ``EventStore | None``, and pyright over ``tests`` is
    clause 2 of this repo's merge-verify type gate.  Same shape as the
    pre-existing ``_FakeEventStore`` stand-ins in this file.
    """

    def __init__(self) -> None:
        object.__init__(self)
        self.emitted: list[tuple] = []

    def emit(
        self,
        event_type: EventType,
        *,
        task_id: str | None = None,
        phase: str | None = None,
        role: str | None = None,
        data: dict | None = None,
        cost_usd: float | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.emitted.append((event_type, data or {}))

    def types(self) -> list:
        return [t for t, _ in self.emitted]

    def data_for(self, event_type) -> list[dict]:
        return [d for t, d in self.emitted if t == event_type]


class _RecordingEscalationQueue:
    """EscalationQueue stand-in with faithful has_open_l1 dedup semantics."""

    def __init__(self) -> None:
        self.submitted: list = []
        self._seq = 0

    def make_id(self, task_id):
        self._seq += 1
        return f'esc-{task_id}-{self._seq}'

    def submit(self, esc):
        self.submitted.append(esc)

    def has_open_l1(self, task_id, *, category=None):
        return any(
            e.task_id == task_id
            and e.level == 1
            and e.status == 'pending'
            and (category is None or e.category == category)
            for e in self.submitted
        )


@pytest.mark.asyncio
class TestDriftCheckEnvDivergenceWiring:
    """_run_drift_check actually RUNS the env check — the wiring 4537 exists for.

    EnvFingerprint / capture_env_fingerprint / compare_env_fingerprints were
    tested for months with zero production callers.  These tests assert BOTH
    arms through the live path: a matching pair emits the faithful outcome and
    files nothing, a perturbed pair emits divergence naming the dimension and
    the probe and files exactly one escalation.  A test of the matching arm
    alone would be passed by a detector that never fires.
    """

    async def _run(self, tmp_path, *, remote_output, probes=_ENV_PROBE_CONFIG):
        """Drive _run_drift_check with a 2-host pool and controllable probes.

        Returns (event_store, escalation_queue, local_seam, remote_seam).
        """
        from orchestrator.merge_lane.drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-env-drift'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path, verify_env_probes=probes,
        )

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        remote_seam = _make_env_probe_seam(remote_output)
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        fake_remote.env_probe_run = AsyncMock(return_value=remote_seam)
        allocator = HostAllocator([fake_remote], quarantine=set())

        local_seam = _make_env_probe_seam(_LOCAL_PROBE_OUTPUT)
        event_store = _RecordingEventStore()
        escalation_queue = _RecordingEscalationQueue()

        with (
            patch(
                'orchestrator.merge_lane.worker.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
            # The LOCAL host is probed through check_env's scoped seam
            # (_target_subprocess_env + cwd=$HOME); patching it keeps the test
            # off the real machine.  The unscoped fallback is patched with a
            # tripwire so a routing regression fails loudly rather than
            # silently spawning the probes for real.
            patch('orchestrator.verify_runner.env_probe_local_run', local_seam),
            patch(
                'orchestrator.verify_runner._default_subprocess_run',
                _unscoped_seam_tripwire,
            ),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', escalation_queue, event_store, set(),
                allocator=allocator, verifier=PRODUCTION_VERIFIER,
            )
        return event_store, escalation_queue, local_seam, remote_seam

    async def test_matching_hosts_emit_env_parity_and_file_nothing(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.event_store import EventType

        event_store, queue, _, _ = await self._run(
            tmp_path, remote_output=_REMOTE_PROBE_OUTPUT_MATCHING,
        )
        assert EventType.verify_env_parity_ok in event_store.types()
        assert EventType.verify_env_divergence not in event_store.types()
        assert queue.submitted == []

    async def test_matching_hosts_still_run_the_verdict_drift_check(
        self, tmp_path: Path,
    ) -> None:
        """The pre-existing alarm keeps firing alongside the new one."""
        from orchestrator.event_store import EventType

        event_store, _, _, _ = await self._run(
            tmp_path, remote_output=_REMOTE_PROBE_OUTPUT_MATCHING,
        )
        assert EventType.verdict_parity_ok in event_store.types()

    async def test_skewed_probe_emits_divergence_naming_dimension_and_probe(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.event_store import EventType

        event_store, _, _, _ = await self._run(
            tmp_path, remote_output=_REMOTE_PROBE_OUTPUT_SKEWED,
        )
        payloads = event_store.data_for(EventType.verify_env_divergence)
        assert len(payloads) == 1
        data = payloads[0]
        assert tuple(data['drift_dimensions']) == ('extra_probes',)
        assert data['diverging_probes'] == {
            'pyright': {'local': 'pyright 1.1.403', 'remote': 'pyright 1.1.377'}
        }
        assert EventType.verify_env_parity_ok not in event_store.types()

    async def test_skewed_probe_files_exactly_one_escalation(
        self, tmp_path: Path,
    ) -> None:
        _, queue, _, _ = await self._run(
            tmp_path, remote_output=_REMOTE_PROBE_OUTPUT_SKEWED,
        )
        assert len(queue.submitted) == 1
        esc = queue.submitted[0]
        assert esc.category == 'verify_env_divergence'
        assert esc.level == 1
        blob = esc.summary + esc.detail
        assert 'pyright' in blob
        assert 'pyright 1.1.403' in blob
        assert 'pyright 1.1.377' in blob

    async def test_skewed_probe_does_not_quarantine_the_remote(
        self, tmp_path: Path,
    ) -> None:
        """Fail-safe: env drift must not bench a verify host.

        quarantine_set alone would NOT discriminate here — it is fed only by
        detector.check()'s result, so a check_env that quarantined the remote
        would leave it empty anyway (check() would then find no eligible
        remote and go INCONCLUSIVE).  Mutation-tested: adding a quarantine
        call to check_env leaves the set assertion green.  The load-bearing
        assertion is that verdict_parity_ok is STILL emitted, which is only
        possible if the remote was still eligible when check() ran.
        """
        quarantine_set: set[str] = set()
        from orchestrator.merge_lane.drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()
        req = MagicMock()
        req.task_id = 'task-env-drift-noquarantine'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_env_probes=_ENV_PROBE_CONFIG,
        )
        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        fake_remote.env_probe_run = AsyncMock(
            return_value=_make_env_probe_seam(_REMOTE_PROBE_OUTPUT_SKEWED)
        )
        allocator = HostAllocator([fake_remote], quarantine=set())
        event_store = _RecordingEventStore()
        with (
            patch(
                'orchestrator.merge_lane.worker.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
            patch(
                'orchestrator.verify_runner.env_probe_local_run',
                _make_env_probe_seam(_LOCAL_PROBE_OUTPUT),
            ),
            patch(
                'orchestrator.verify_runner._default_subprocess_run',
                _unscoped_seam_tripwire,
            ),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', _RecordingEscalationQueue(),
                event_store, quarantine_set, allocator=allocator, verifier=PRODUCTION_VERIFIER,
            )
        from orchestrator.event_store import EventType
        assert quarantine_set == set()
        assert EventType.verify_env_divergence in event_store.types(), (
            'the env check must actually have fired, or the no-quarantine '
            'assertion below is vacuous'
        )
        assert EventType.verdict_parity_ok in event_store.types(), (
            'the remote was benched by the env check: the verdict drift check '
            f'found no eligible remote.  emitted: {event_store.types()!r}'
        )

    async def test_operator_declared_argv_is_what_reaches_both_hosts(
        self, tmp_path: Path,
    ) -> None:
        """The probe set comes from config, not from a hardcoded Rust list."""
        _, _, local_seam, remote_seam = await self._run(
            tmp_path,
            remote_output=_REMOTE_PROBE_OUTPUT_MATCHING,
            probes=[{'key': 'uv', 'argv': ['uv', '--version']}],
        )
        assert ['uv', '--version'] in local_seam.issued
        assert ['uv', '--version'] in remote_seam.issued
        assert ['npx', 'pyright', '--version'] not in remote_seam.issued

    async def test_empty_probe_list_emits_no_env_event_and_probes_nothing(
        self, tmp_path: Path,
    ) -> None:
        """The shipped default is inert: no env event, no host touched."""
        from orchestrator.event_store import EventType

        event_store, queue, local_seam, remote_seam = await self._run(
            tmp_path, remote_output=_REMOTE_PROBE_OUTPUT_MATCHING, probes=[],
        )
        assert EventType.verify_env_parity_ok not in event_store.types()
        assert EventType.verify_env_divergence not in event_store.types()
        assert queue.submitted == []
        assert local_seam.issued == []
        assert remote_seam.issued == []
        # ...and the pre-existing verdict check is untouched by all of that.
        assert EventType.verdict_parity_ok in event_store.types()

    async def test_unprobeable_remote_spawns_no_local_subprocess(
        self, tmp_path: Path,
    ) -> None:
        """A remote whose probe seam is unusable must cost ZERO local spawns.

        merge_drift's ``_run_drift_check`` calls ``check_env`` with NO
        ``local_run``, so the local fingerprint is captured through
        ``verify_runner.env_probe_local_run`` — real
        ``create_subprocess_exec`` calls against whatever is on PATH (rustc,
        cargo, sccache, plus every declared probe; ``npx`` reaches the npm
        registry on a cold cache, and nothing applies a timeout).

        That has to be reached only when a comparison is actually possible.
        The remote here is a bare MagicMock — the shape every drift test that
        predates Lever C uses — so its auto-mocked ``env_probe_run()`` returns
        a non-awaitable and the remote capture cannot succeed.  Probing the
        remote FIRST makes that discovery free; probing the local host first
        spends seven unpinned external commands to build a fingerprint that is
        then thrown away.
        """
        from orchestrator.merge_lane.drift import _run_drift_check
        from orchestrator.verify_runner import HostAllocator

        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-env-drift-no-local-spawn'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_env_probes=_ENV_PROBE_CONFIG,
        )

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        allocator = HostAllocator([fake_remote], quarantine=set())

        local_seam = _make_env_probe_seam(_LOCAL_PROBE_OUTPUT)
        event_store = _RecordingEventStore()

        with (
            patch(
                'orchestrator.merge_lane.worker.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
            patch('orchestrator.verify_runner.env_probe_local_run', local_seam),
            # Both local seams are stubbed: the assertion below is "no local
            # spawn happened", and a stub on only ONE of them would let a
            # regression satisfy it by spawning through the other for real.
            patch(
                'orchestrator.verify_runner._default_subprocess_run',
                _unscoped_seam_tripwire,
            ),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', _RecordingEscalationQueue(), event_store,
                set(), allocator=allocator, verifier=PRODUCTION_VERIFIER,
            )

        assert local_seam.issued == [], (
            f'the local host was probed {len(local_seam.issued)} time(s) for a '
            f'comparison that could never complete: {local_seam.issued!r} — on '
            f'the production path these are unpinned, untimed external binaries'
        )


# ---------------------------------------------------------------------------
# task 4537 step-17: the NEW env check must never break the OLD verdict check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEnvCheckCannotBreakVerdictDriftCheck:
    """A broken env probe must not take the pre-existing verdict alarm with it.

    The verdict drift check is load-bearing: it is what catches a remote PASS
    against a local trust-anchor FAIL, the split that lands unverified code on
    main.  The env check is a brand-new detective bolted in front of it.  If
    the two shared one try block, any failure in the new code — an ssh probe
    hanging up, a malformed probe argv, a bug in check_env itself — would skip
    detector.check() entirely and silently disarm the older alarm.

    So the env check gets its OWN try/except, and this test is the proof.
    """

    async def test_env_check_exception_still_runs_the_verdict_check(
        self, tmp_path: Path, caplog,
    ) -> None:
        import logging

        from orchestrator.event_store import EventType
        from orchestrator.merge_lane.drift import _run_drift_check
        from orchestrator.verify_runner import DriftDetector, HostAllocator

        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-env-check-blast-radius'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_env_probes=_ENV_PROBE_CONFIG,
        )

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        fake_remote.env_probe_run = AsyncMock(
            return_value=_make_env_probe_seam(_REMOTE_PROBE_OUTPUT_MATCHING)
        )
        allocator = HostAllocator([fake_remote], quarantine=set())
        event_store = _RecordingEventStore()

        async def _explode(*args, **kwargs):
            raise RuntimeError('env probe seam blew up')

        with (
            patch(
                'orchestrator.merge_lane.worker.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
            patch.object(DriftDetector, 'check_env', _explode),
            caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', _RecordingEscalationQueue(),
                event_store, set(), allocator=allocator, verifier=PRODUCTION_VERIFIER,
            )

        assert EventType.verdict_parity_ok in event_store.types(), (
            'an exception from the NEW env check swallowed the PRE-EXISTING '
            f'verdict drift check; emitted: {event_store.types()!r}'
        )

    async def test_env_check_exception_is_logged_at_warning(
        self, tmp_path: Path, caplog,
    ) -> None:
        """Swallowed must not mean silent — a broken probe has to be visible."""
        import logging

        from orchestrator.merge_lane.drift import _run_drift_check
        from orchestrator.verify_runner import DriftDetector, HostAllocator

        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(
            return_value=Path('/repo/_throwaway')
        )
        git_ops.cleanup_merge_worktree = AsyncMock()

        req = MagicMock()
        req.task_id = 'task-env-check-logged'
        req.task_files = None
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_env_probes=_ENV_PROBE_CONFIG,
        )

        pass_result = VerifyResult(
            passed=True, test_output='', lint_output='', type_output='', summary='ok',
        )
        fake_remote = MagicMock()
        fake_remote.name = 'laptop'
        fake_remote.is_local = False
        fake_remote.run_merge_verify = AsyncMock(return_value=pass_result)
        fake_remote.env_probe_run = AsyncMock(
            return_value=_make_env_probe_seam(_REMOTE_PROBE_OUTPUT_MATCHING)
        )
        allocator = HostAllocator([fake_remote], quarantine=set())

        async def _explode(*args, **kwargs):
            raise RuntimeError('env probe seam blew up')

        with (
            patch(
                'orchestrator.merge_lane.worker.run_scoped_verification',
                AsyncMock(return_value=pass_result),
            ),
            patch.object(DriftDetector, 'check_env', _explode),
            caplog.at_level(logging.WARNING, logger='orchestrator.verify_runner'),
        ):
            await _run_drift_check(
                git_ops, req, 'commit-sha', _RecordingEscalationQueue(),
                _RecordingEventStore(), set(), allocator=allocator, verifier=PRODUCTION_VERIFIER,
            )

        env_warnings = [
            r for r in caplog.records
            if r.name == 'orchestrator.verify_runner'
            and r.levelno == logging.WARNING
            and 'Env drift check failed' in r.getMessage()
        ]
        assert len(env_warnings) == 1, (
            'expected exactly one env-specific WARNING naming the env check; '
            f'saw: {[r.getMessage() for r in caplog.records]!r}'
        )
        assert env_warnings[0].exc_info is not None, (
            'the warning must carry the traceback (exc_info=True) or a broken '
            'probe is undiagnosable from the journal'
        )
        assert req.task_id in env_warnings[0].getMessage()


# ---------------------------------------------------------------------------
# task 3018 (steps 11-12): a LIVE drift-check throwaway worktree must survive a
# concurrent periodic reap.
#
# Sibling of TestColdShadowVerifyHoldsLaneLease in test_merge_shadow.py — the
# drift detective creates its throwaway `_merge-<uuid>` worktree through the
# very same `create_ephemeral_verify_worktree` primitive and runs a full merge
# verify inside it, so it inherits the identical exposure: the tree is neither
# registered in `_owned_merge_worktrees` (the reap's owned-ledger skip misses
# it) nor touched by `_touch_owned_merge_worktrees` (its measured age is real
# elapsed time), leaving the per-lane merge-verify flock consulted by
# `remove_merge_worktree_guarded` as its only liveness protection.
#
# Deliberately uses a REAL git repo + REAL throwaway worktree (not the
# MagicMock git_ops the cadence tests above use) so `lane_lock_path(wt)`
# names a real file and the flock contention is genuine — a mocked git_ops
# cannot exercise a real lock.  The pool half is built the way
# TestDriftCheckFullGateSpecNoDerivation builds it: a real HostAllocator plus a
# fake remote, which is what gets DriftDetector.check past its `local is None
# or remote is None` INCONCLUSIVE early-return and into the LocalRunner, which
# verifies through the injected VerifyPort and so through the patched
# `orchestrator.merge_lane.worker.run_scoped_verification`.
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
        from orchestrator.merge_lane.drift import _run_drift_check
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
        req.config = OrchestratorConfig(
            project_root=drift_git_repo, verify_env_probes=_LEVER_C_OFF,
        )

        with patch(
            'orchestrator.merge_lane.worker.run_scoped_verification', _reaping_scoped,
        ):
            await _run_drift_check(
                drift_git_ops, req, head, None, None, set(), allocator=allocator,
                verifier=PRODUCTION_VERIFIER,
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


# ---------------------------------------------------------------------------
# task 6648: each host lease is released when its own leg ends
#
# Slot state is read ONLY through HostAllocator.host_states(), the read the
# heartbeat publishes as hosts[...].slot_state.  The local leg is parked
# through the injected VerifyPort (FakeVerifier + hangs_until), so no module
# name is patched.
# ---------------------------------------------------------------------------

_WAIT_SECS = 5.0
_PASS = VerifyResult(passed=True, test_output='', lint_output='', type_output='', summary='ok')


async def _until(predicate: Callable[[], bool]) -> None:
    """Poll *predicate* until it holds, bounded so a RED run fails instead of hanging."""
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(_poll(), _WAIT_SECS)


@dataclasses.dataclass
class _DriftScene:
    """``_run_drift_check`` over MagicMock git_ops and a REAL allocator with a 'laptop' double."""

    git_ops: MagicMock
    req: MagicMock
    remote: MagicMock
    allocator: HostAllocator
    event_store: _RecordingEventStore

    @classmethod
    def build(
        cls, tmp_path: Path, *, remote_verify: AsyncMock | None = None, **config: Any,
    ) -> _DriftScene:
        git_ops = MagicMock()
        git_ops.create_ephemeral_verify_worktree = AsyncMock(return_value=tmp_path / '_throwaway')
        git_ops.cleanup_merge_worktree = AsyncMock()
        req = MagicMock()
        req.task_id = 'task-6648-drift'
        req.module_configs = []
        req.config = OrchestratorConfig(
            project_root=tmp_path,
            verify_runners=[
                VerifyRunnerConfig(
                    name='laptop', ssh_host='laptop.local', git_remote='origin',
                    df_checkout_path='/df',
                ),
            ],
            verify_env_probes=_LEVER_C_OFF,
            **config,
        )
        remote = MagicMock()
        remote.name = 'laptop'
        remote.is_local = False
        remote.run_merge_verify = remote_verify or AsyncMock(return_value=_PASS)
        allocator = HostAllocator([remote], quarantine=set())
        return cls(git_ops, req, remote, allocator, _RecordingEventStore())

    def run(self, verifier: FakeVerifier, **options: Any) -> Coroutine[Any, Any, None]:
        return _run_drift_check(
            self.git_ops, self.req, 'drift-sha', None, self.event_store, set(),
            verifier=verifier, allocator=self.allocator, **options,
        )

    def slot(self, name: str) -> str:
        return next(h['slot_state'] for h in self.allocator.host_states() if h['name'] == name)

    def parity_events(self) -> int:
        return self.event_store.types().count(EventType.verdict_parity_ok)


def _gated_remote(gate: asyncio.Event, entered: asyncio.Event) -> AsyncMock:
    async def _body(*_args: Any, **_kwargs: Any) -> VerifyResult:
        entered.set()
        await gate.wait()
        return _PASS
    return AsyncMock(side_effect=_body)


@pytest.mark.asyncio
class TestDriftCheckReleasesEachLeaseAtItsLegsEnd:
    """Each host slot frees the moment its own leg ends, and is released exactly once."""

    async def test_laptop_slot_is_free_while_the_local_leg_still_runs(
        self, tmp_path: Path,
    ) -> None:
        scene = _DriftScene.build(tmp_path)
        local_gate, local_entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(default=hangs_until(local_gate, entered=local_entered))
        drift = asyncio.create_task(scene.run(verifier))
        try:
            await wait_responsive(
                local_entered.wait(), timeout=_WAIT_SECS, label='drift local leg entered',
            )
            await _until(lambda: scene.slot('laptop') == 'free')
            assert scene.slot('local') == 'busy'
        finally:
            local_gate.set()
            await asyncio.wait_for(drift, _WAIT_SECS)
        assert (scene.slot('local'), scene.slot('laptop')) == ('free', 'free')
        assert scene.parity_events() == 1

    async def test_lease_released_at_leg_end_is_not_released_again_by_the_safety_net(
        self, tmp_path: Path,
    ) -> None:
        # HostAllocator.release keys on slot STATE, not lease identity: a second
        # release of the drift check's laptop lease would free (or un-PARK) the
        # slot the merge lane has since taken.
        scene = _DriftScene.build(tmp_path)
        local_gate, local_entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(default=hangs_until(local_gate, entered=local_entered))
        drift = asyncio.create_task(scene.run(verifier))
        try:
            await wait_responsive(
                local_entered.wait(), timeout=_WAIT_SECS, label='drift local leg entered',
            )
            await _until(lambda: scene.slot('laptop') == 'free')
            assert scene.allocator.acquire_remote() is not None
        finally:
            local_gate.set()
            await asyncio.wait_for(drift, _WAIT_SECS)
        assert scene.slot('laptop') == 'busy'

    async def test_local_slot_is_free_while_the_remote_leg_still_runs(
        self, tmp_path: Path,
    ) -> None:
        remote_gate, remote_entered = asyncio.Event(), asyncio.Event()
        scene = _DriftScene.build(
            tmp_path, remote_verify=_gated_remote(remote_gate, remote_entered),
        )
        drift = asyncio.create_task(scene.run(FakeVerifier()))
        try:
            await wait_responsive(
                remote_entered.wait(), timeout=_WAIT_SECS, label='drift remote leg entered',
            )
            await _until(lambda: scene.slot('local') == 'free')
            assert scene.slot('laptop') == 'busy'
        finally:
            remote_gate.set()
            await asyncio.wait_for(drift, _WAIT_SECS)
        assert (scene.slot('local'), scene.slot('laptop')) == ('free', 'free')
        assert scene.parity_events() == 1


# ---------------------------------------------------------------------------
# task 6648: a check that cannot get both host slots defers, visibly
# ---------------------------------------------------------------------------


def _deferrals(scene: _DriftScene) -> list[dict]:
    return scene.event_store.data_for(EventType.drift_check_deferred)


def _deferral_logs(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.levelno == level and 'deferred' in r.getMessage()
    ]


@pytest.mark.asyncio
class TestDriftCheckDefersWhenAHostSlotIsBusy:
    """A due check whose slots stay busy takes nothing, runs nothing, and stays owed."""

    async def test_busy_remote_defers_without_running_either_leg(self, tmp_path: Path) -> None:
        scene = _DriftScene.build(tmp_path)
        assert scene.allocator.acquire_remote() is not None
        verifier = FakeVerifier()
        state_path = tmp_path / 'drift_check_state.json'

        await scene.run(verifier, state_path=state_path, host_settle_secs=0)

        scene.remote.run_merge_verify.assert_not_awaited()
        assert verifier.entered_count == 0
        (deferral,) = _deferrals(scene)
        assert deferral['merge_sha'] == 'drift-sha'
        assert deferral['consecutive_deferrals'] == 1
        assert deferral['hosts'] == scene.allocator.host_states()
        assert {h['name']: h['slot_state'] for h in deferral['hosts']}['laptop'] == 'busy'
        assert _load_drift_check_state(state_path).consecutive_deferrals == 1
        assert (scene.slot('local'), scene.slot('laptop')) == ('free', 'busy')
        scene.git_ops.create_ephemeral_verify_worktree.assert_not_awaited()
        scene.git_ops.cleanup_merge_worktree.assert_not_awaited()

    async def test_a_check_waiting_for_its_slots_has_built_no_worktree(
        self, tmp_path: Path,
    ) -> None:
        scene = _DriftScene.build(tmp_path)
        merge_lease = scene.allocator.acquire_remote()
        assert merge_lease is not None
        drift = asyncio.create_task(
            scene.run(FakeVerifier(), host_settle_secs=5, host_poll_secs=0.01),
        )
        try:
            await asyncio.sleep(0.05)
            scene.git_ops.create_ephemeral_verify_worktree.assert_not_awaited()
        finally:
            await scene.allocator.release(merge_lease)
            await asyncio.wait_for(drift, 2 * _WAIT_SECS)
        scene.git_ops.create_ephemeral_verify_worktree.assert_awaited_once_with('drift-sha')
        scene.git_ops.cleanup_merge_worktree.assert_awaited_once()
        assert scene.parity_events() == 1

    async def test_worktree_failure_frees_both_slots_it_acquired(self, tmp_path: Path) -> None:
        scene = _DriftScene.build(tmp_path)
        scene.git_ops.create_ephemeral_verify_worktree.side_effect = OSError('disk full')

        await scene.run(FakeVerifier(), host_settle_secs=0)

        assert (scene.slot('local'), scene.slot('laptop')) == ('free', 'free')
        scene.remote.run_merge_verify.assert_not_awaited()
        scene.git_ops.cleanup_merge_worktree.assert_not_awaited()
        assert _deferrals(scene) == []

    async def test_both_slots_busy_defers_and_leaves_both_holders_untouched(
        self, tmp_path: Path,
    ) -> None:
        scene = _DriftScene.build(tmp_path)
        assert scene.allocator.acquire_remote() is not None
        assert scene.allocator.acquire_local(lambda: MagicMock()) is not None

        await scene.run(
            FakeVerifier(), state_path=tmp_path / 'drift_check_state.json', host_settle_secs=0,
        )

        assert len(_deferrals(scene)) == 1
        assert (scene.slot('local'), scene.slot('laptop')) == ('busy', 'busy')

    async def test_busy_local_rolls_back_the_remote_it_acquired(self, tmp_path: Path) -> None:
        scene = _DriftScene.build(tmp_path)
        assert scene.allocator.acquire_local(lambda: MagicMock()) is not None

        await scene.run(
            FakeVerifier(), state_path=tmp_path / 'drift_check_state.json', host_settle_secs=0,
        )

        assert len(_deferrals(scene)) == 1
        assert (scene.slot('local'), scene.slot('laptop')) == ('busy', 'free')

    async def test_slot_freed_within_the_settle_window_lets_the_check_run(
        self, tmp_path: Path,
    ) -> None:
        scene = _DriftScene.build(tmp_path)
        merge_lease = scene.allocator.acquire_remote()
        assert merge_lease is not None
        state_path = tmp_path / 'drift_check_state.json'

        async def _merge_lands_shortly() -> None:
            await asyncio.sleep(0.05)
            await scene.allocator.release(merge_lease)

        freer = asyncio.create_task(_merge_lands_shortly())
        await asyncio.wait_for(
            scene.run(
                FakeVerifier(), state_path=state_path,
                host_settle_secs=5, host_poll_secs=0.01,
            ),
            2 * _WAIT_SECS,
        )
        await freer

        assert scene.parity_events() == 1
        assert _deferrals(scene) == []
        assert _load_drift_check_state(state_path).consecutive_deferrals == 0

    async def test_consecutive_deferrals_log_info_once_and_emit_every_time(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        scene = _DriftScene.build(tmp_path)
        assert scene.allocator.acquire_remote() is not None
        state_path = tmp_path / 'drift_check_state.json'

        with caplog.at_level(logging.DEBUG, logger='orchestrator.merge_lane'):
            for _ in range(2):
                await scene.run(FakeVerifier(), state_path=state_path, host_settle_secs=0)

        assert [d['consecutive_deferrals'] for d in _deferrals(scene)] == [1, 2]
        assert len(_deferral_logs(caplog, logging.INFO)) == 1
        assert len(_deferral_logs(caplog, logging.DEBUG)) == 1, (
            'the repeat deferral must still be logged, at DEBUG -- without it the '
            'INFO-once assertion above could pass on a capture that saw nothing'
        )

    async def test_a_check_that_runs_after_deferrals_clears_the_counter(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        scene = _DriftScene.build(tmp_path)
        state_path = tmp_path / 'drift_check_state.json'
        _save_drift_check_state(
            state_path, DriftCheckState(land_count=5, consecutive_deferrals=2),
        )

        with caplog.at_level(logging.INFO, logger='orchestrator.merge_lane'):
            await scene.run(FakeVerifier(), state_path=state_path, host_settle_secs=0)

        assert scene.parity_events() == 1
        assert _load_drift_check_state(state_path) == DriftCheckState(land_count=5)
        assert any(
            'after 2 deferred' in message
            for message in _deferral_logs(caplog, logging.INFO)
        )


# ---------------------------------------------------------------------------
# task 6648: the land hook re-arms a deferred check and runs one at a time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDriftCheckLandHookReArmsAndCoalesces:
    """_maybe_run_drift_check: a deferral keeps the check owed; one check runs at a time."""

    @staticmethod
    def _lands(scene: _DriftScene, state_path: Path, verifier: FakeVerifier):
        worker = _build_drift_worker(
            state_path, allocator=scene.allocator, event_store=scene.event_store,
        )

        async def land(sha: str) -> asyncio.Task[None] | None:
            return await _maybe_run_drift_check(
                worker, scene.git_ops, scene.req, sha,
                verifier=verifier, host_settle_secs=0,
            )
        return land

    async def test_deferred_check_re_arms_at_the_next_landing(self, tmp_path: Path) -> None:
        scene = _DriftScene.build(tmp_path, verify_drift_check_every_n_lands=3)
        state_path = tmp_path / 'drift_check_state.json'
        land = self._lands(scene, state_path, FakeVerifier())

        assert await land('c1') is None
        assert await land('c2') is None
        merge_lease = scene.allocator.acquire_remote()
        assert merge_lease is not None
        deferred = await land('c3')
        assert deferred is not None
        await asyncio.wait_for(deferred, _WAIT_SECS)
        assert _load_drift_check_state(state_path) == DriftCheckState(
            land_count=3, consecutive_deferrals=1,
        )

        await scene.allocator.release(merge_lease)
        re_armed = await land('c4')
        assert re_armed is not None, 'a deferred check must fire at the next landing'
        await asyncio.wait_for(re_armed, _WAIT_SECS)
        assert scene.parity_events() == 1
        assert _load_drift_check_state(state_path) == DriftCheckState(land_count=4)

        assert await land('c5') is None

    async def test_a_due_landing_while_a_check_is_in_flight_spawns_nothing(
        self, tmp_path: Path,
    ) -> None:
        scene = _DriftScene.build(tmp_path, verify_drift_check_every_n_lands=1)
        state_path = tmp_path / 'drift_check_state.json'
        gate, entered = asyncio.Event(), asyncio.Event()
        verifier = FakeVerifier(default=hangs_until(gate, entered=entered))
        land = self._lands(scene, state_path, verifier)

        first = await land('c1')
        assert first is not None
        try:
            await wait_responsive(
                entered.wait(), timeout=_WAIT_SECS, label='first drift check entered its verify',
            )
            assert await land('c2') is None
        finally:
            gate.set()
            await asyncio.wait_for(first, _WAIT_SECS)
        assert verifier.entered_count == 1
        assert _load_drift_check_state(state_path) == DriftCheckState(land_count=2)

    async def test_hook_returns_none_when_not_due_or_disabled(self, tmp_path: Path) -> None:
        worker = _build_drift_worker(tmp_path / 'drift_check_state.json')
        no_runners = MagicMock()
        no_runners.config.enabled_verify_runners = []
        disabled = MagicMock()
        disabled.config.enabled_verify_runners = ['laptop']
        disabled.config.verify_drift_check_every_n_lands = 0
        off_cadence = MagicMock()
        off_cadence.config.enabled_verify_runners = ['laptop']
        off_cadence.config.verify_drift_check_every_n_lands = 3

        for req in (no_runners, disabled, off_cadence):
            spawned = await _maybe_run_drift_check(
                worker, MagicMock(), req, 'sha', verifier=PRODUCTION_VERIFIER,
            )
            assert spawned is None
