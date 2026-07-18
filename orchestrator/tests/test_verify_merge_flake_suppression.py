"""Tests for the merge-gate single flake-retry gate (PRD task α).

`plans/cpu-load-robust-verify-prd.md` task α: give role='merge' verify the
per-node-id isolated-rerun-confirm gate that run_main_tip_sweep /
confirm_main_tip_failure_is_real already have, so a CPU-starvation-induced red
on an unrelated test does not block a correct, code-complete task from landing.
Critically, α resolves to a pass/fail verdict INLINE (returns a VerifyResult) —
it MUST NOT raise VerifyInfraError, because the merge path (merge_queue.py) has
no VerifyInfraError handler.

Test structure mirrors the confirm-gate test suite in
test_verify_main_tip_sweep.py (fake on-disk project layout so the node-id ->
subproject existence mapping runs against real files; run_verification
monkeypatched so no real subprocess runs).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_cmd import (
    ToolKind,
    parse_config_command,
    render,
    with_pytest_timeout,
)


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    """A real minimal OrchestratorConfig (never a bare MagicMock — the
    check_bare_magicmock_config lint gate). run_verification is fully
    monkeypatched in these tests, so only project_root/git are load-bearing.
    """
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _materialize(worktree: Path, *relpaths: str) -> None:
    """Create empty files at *relpaths* under *worktree* so the node-id ->
    subproject existence mapping (`(worktree / prefix / file).exists()` etc.)
    runs against real files without a real git checkout.
    """
    for rel in relpaths:
        p = worktree / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('def test_x():\n    pass\n')


# A merge-role subproject whose pytest command mirrors dark_factory's real
# per-subproject test_command shape (uv run --project X --directory X pytest ...).
_ORCH_TEST_CMD = (
    'uv run --project orchestrator --directory orchestrator '
    'pytest tests/ --tb=short -q'
)


def _orch_module_config() -> ModuleConfig:
    return ModuleConfig(
        prefix='orchestrator',
        test_command=_ORCH_TEST_CMD,
        lint_command='uv run --project orchestrator ruff check src/',
        type_check_command='uv run --project orchestrator pyright src/',
    )


def _result(passed: bool, *, category: str = '') -> VerifyResult:
    return VerifyResult(
        passed=passed,
        test_output='',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category=category or ('' if passed else 'test_failure'),
    )


class _FakeEventStore:
    """Records emit() calls without touching sqlite."""

    def __init__(self) -> None:
        self.emits: list[tuple] = []

    def emit(self, event_type, *, task_id=None, data=None, **kwargs) -> None:
        self.emits.append((event_type, task_id, data))


class _FakeEscalationQueue:
    """Records submit()/get_by_task()/make_id() for storm-escalation tests.

    *open_l2* controls the dedup path: when truthy, get_by_task returns it so
    the filer treats an open L2 as already present and does NOT re-submit.
    """

    def __init__(self, open_l2=None) -> None:
        self.submitted: list = []
        self.get_by_task_calls: list = []
        self._open_l2 = open_l2

    def make_id(self, task_id: str) -> str:
        return f'esc-{task_id}-1'

    def get_by_task(self, task_id, *, status=None, level=None):
        self.get_by_task_calls.append((task_id, status, level))
        return self._open_l2

    def submit(self, esc) -> None:
        self.submitted.append(esc)


_SUPPRESSED_IDS = ['orchestrator/tests/test_x.py::test_y']
_MERGE_SHA = 'd' * 40


def _failing_with_logs() -> VerifyResult:
    return VerifyResult(
        passed=False,
        test_output=_B1_TEST_OUTPUT,
        lint_output='',
        type_output='',
        summary='fail',
        category='test_failure',
        worktree_log_paths=['/wt/verify.log'],
        archive_log_paths=['/archive/verify.log'],
    )


class TestWithPytestTimeout:
    """with_pytest_timeout(cmd, secs) appends a `--timeout <secs>` flag to a
    structured pytest command's base_flags (PRD task α; INV-5 reuse).

    Copies with_junitxml's structure exactly: same ToolKind.PYTEST +
    cmd.raw is None guard, same `replace(cmd, base_flags=(*cmd.base_flags,
    '--timeout', str(secs)))` shape. A structured command's per-test timeout
    override is REQUIRED because the pyproject `timeout=60` default lives in
    `[tool.pytest.ini_options]`, NOT addopts, so the serial-recovery
    `-o addopts=` does not clear it — without an explicit `--timeout=300` the
    confirm run could itself starve into a false non-suppression.
    """

    def test_structured_single_invocation_appends_timeout_flag(self):
        cmd = parse_config_command('pytest tests/x.py')
        result = render(with_pytest_timeout(cmd, 300))
        assert result == 'pytest --timeout 300 tests/x.py'

    def test_base_flags_end_with_timeout_pair(self):
        cmd = parse_config_command('pytest tests/x.py')
        mutated = with_pytest_timeout(cmd, 300)
        assert mutated.base_flags[-2:] == ('--timeout', '300')
        assert mutated.raw is None

    def test_secs_rendered_as_string(self):
        """secs is an int at the call site but must render as a bare token."""
        cmd = parse_config_command('pytest tests/x.py')
        mutated = with_pytest_timeout(cmd, 300)
        assert '300' in mutated.base_flags
        assert 300 not in mutated.base_flags  # str, never int

    def test_raw_retained_chain_returned_unchanged(self):
        """A recognised-but-unstructurable pytest chain (raw is not None) is
        NEVER regex-rewritten — mirrors with_junitxml's deliberate divergence
        from apply_pytest_numprocesses/serial_pytest. Byte-identical no-op.
        """
        raw = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.raw == raw

        result = with_pytest_timeout(cmd, 300)
        assert result == cmd
        assert render(result) == raw

    def test_non_pytest_command_returned_unchanged(self):
        cmd = parse_config_command('ruff check .')
        assert with_pytest_timeout(cmd, 300) == cmd

    def test_pyright_command_returned_unchanged(self):
        cmd = parse_config_command('pyright')
        assert with_pytest_timeout(cmd, 300) == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert cmd.tool is ToolKind.OPAQUE
        assert with_pytest_timeout(cmd, 300) == cmd

    def test_noop_returns_same_object_identity(self):
        """A no-op returns the SAME object (is), so _with_pytest_timeout_str's
        `rewritten is parsed` short-circuit detects it and returns the input
        string byte-identically.
        """
        cmd = parse_config_command('cargo test --workspace')
        assert with_pytest_timeout(cmd, 300) is cmd


class TestWithPytestTimeoutStr:
    """_with_pytest_timeout_str(cmd, secs) — verify.py string wrapper around
    parse_config_command -> with_pytest_timeout -> render (mirrors
    _serial_pytest_str). Returns cmd unchanged when None / non-pytest / OPAQUE.
    """

    def test_roundtrips_real_pytest_command(self):
        from orchestrator.verify import _with_pytest_timeout_str

        cmd = (
            'uv run --project orchestrator --directory orchestrator '
            'pytest tests/ --tb=short -q'
        )
        result = _with_pytest_timeout_str(cmd, 300)
        assert result is not None
        assert '--timeout 300' in result

    def test_none_input_returns_none(self):
        from orchestrator.verify import _with_pytest_timeout_str

        assert _with_pytest_timeout_str(None, 300) is None

    def test_non_pytest_input_returned_unchanged(self):
        from orchestrator.verify import _with_pytest_timeout_str

        cmd = 'cargo test --workspace'
        assert _with_pytest_timeout_str(cmd, 300) == cmd

    def test_opaque_input_returned_unchanged(self):
        from orchestrator.verify import _with_pytest_timeout_str

        cmd = 'mypy src/'
        assert _with_pytest_timeout_str(cmd, 300) == cmd

    def test_composes_with_serial_pytest_str(self):
        """The α confirm command is
        _with_pytest_timeout_str(_serial_pytest_str(scoped)) — both flags
        must survive composition (serial recovery + generous per-test timeout).
        """
        from orchestrator.verify import _serial_pytest_str, _with_pytest_timeout_str

        base = (
            'uv run --project orchestrator --directory orchestrator '
            'pytest tests/ --tb=short -q'
        )
        composed = _with_pytest_timeout_str(_serial_pytest_str(base), 300)
        assert composed is not None
        assert '-p no:xdist' in composed
        assert '-o addopts=' in composed
        assert '--timeout 300' in composed


# --- Node-id fixtures for the confirm gate ------------------------------------

# B1: a real FAILED node-id plus an xdist `node down`-preceding node-id, both
# owned by orchestrator/tests/test_x.py so they group into one isolated re-run.
_B1_FAILED_ID = 'orchestrator/tests/test_x.py::test_y'
_B1_CRASH_ID = 'orchestrator/tests/test_x.py::test_z'
_B1_TEST_OUTPUT = (
    f'FAILED {_B1_FAILED_ID}\n'
    f'{_B1_CRASH_ID}\n'
    '[gw3] node down: Not properly terminated\n'
)

# B3: a bare whole-file collection ERROR (no ::nodeid) — _extract_failing_test_ids
# yields the file target, which re-errors at collection on the isolated re-run.
_B3_TEST_OUTPUT = 'ERROR orchestrator/tests/test_x.py\n'


class TestConfirmMergeVerifyFlakeSuppressible:
    """confirm_merge_verify_flake_suppressible(config, failing_result, *,
    worktree, module_configs) -> list[str] | None — the PURE gate (PRD task α).

    SAME-TREE (re-runs in the given merge worktree, no fresh probe worktree),
    single-shot per node-id group, returns the suppressed node-ids on a
    confirmed flake or None (fail-closed to red) otherwise. NEVER raises.
    """

    def _run(self, config, failing_result, worktree, module_configs):
        from orchestrator.verify import confirm_merge_verify_flake_suppressible

        return asyncio.run(
            confirm_merge_verify_flake_suppressible(
                config, failing_result, worktree=worktree, module_configs=module_configs,
            )
        )

    # -- B1: suppress a confirmed flake -----------------------------------

    def test_b1_suppresses_when_isolated_rerun_passes(self, tmp_path: Path) -> None:
        """FAILED + node-down node-ids both pass on isolated re-run -> returns
        the extracted node-id list, and the isolated ModuleConfig carries the
        serial + generous-timeout recovery command with null lint/type."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output=_B1_TEST_OUTPUT, lint_output='',
            type_output='', summary='fail', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify_module, 'run_verification', rv):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result == [_B1_FAILED_ID, _B1_CRASH_ID], result

        rv.assert_awaited()
        # The isolated re-run's ModuleConfig: serial + generous timeout, null gates.
        called_mc = rv.call_args.args[2]
        assert '-p no:xdist' in called_mc.test_command, called_mc.test_command
        assert '--timeout 300' in called_mc.test_command, called_mc.test_command
        assert _B1_FAILED_ID in called_mc.test_command, called_mc.test_command
        assert called_mc.lint_command is None
        assert called_mc.type_check_command is None
        # Merge-role, single-shot, cold-timeout semantics.
        assert rv.call_args.kwargs['is_merge_verify'] is True
        assert rv.call_args.kwargs['max_retries'] == 0
        assert rv.call_args.kwargs['role'] == 'merge'

    # -- B2: never mask a still-real red ----------------------------------

    def test_b2_no_suppress_when_isolated_rerun_still_fails(self, tmp_path: Path) -> None:
        """The isolated re-run still FAILS -> None (merge stays red)."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output=_B1_TEST_OUTPUT, lint_output='',
            type_output='', summary='fail', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(False))
        with patch.object(verify_module, 'run_verification', rv):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result is None
        rv.assert_awaited()

    # -- B3: whole-file collection error re-errors -> not suppressed ------

    def test_b3_collection_error_not_suppressed(self, tmp_path: Path) -> None:
        """A bare `ERROR file.py` collection error yields a whole-file target;
        the isolated re-run re-errors (not passed) -> None."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output=_B3_TEST_OUTPUT, lint_output='',
            type_output='', summary='collection error', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(False))
        with patch.object(verify_module, 'run_verification', rv):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result is None

    # -- fail-closed: no recoverable node-id, no re-run at all -------------

    def test_no_node_id_returns_none_without_rerun(self, tmp_path: Path) -> None:
        """Opaque/lint/type failure output (no pytest node-id) -> None WITHOUT
        calling run_verification (cheap early-out, fail-closed to red)."""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output='',
            lint_output='src/foo.py:12:5: F401 unused import',
            type_output='', summary='lint_failure', category='lint_failure',
        )

        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify_module, 'run_verification', rv):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result is None
        rv.assert_not_awaited()

    # -- fail-closed: node-id maps to no subproject -----------------------

    def test_unmapped_node_id_returns_none(self, tmp_path: Path) -> None:
        """A node-id whose file exists under no given subproject -> None.
        (No file materialized on disk -> the existence mapping fails.)"""
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output='FAILED some/other/tests/test_q.py::test_z\n',
            lint_output='', type_output='', summary='fail', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify_module, 'run_verification', rv):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result is None

    # -- fail-closed: infra-sentinel re-run is never trusted --------------

    def test_infra_transient_rerun_category_returns_none(self, tmp_path: Path) -> None:
        """An isolated re-run whose category is in INFRA_TRANSIENT_CATEGORIES
        is never trusted as confirmation, even paired with passed=True -> None."""
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output=_B1_TEST_OUTPUT, lint_output='',
            type_output='', summary='fail', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(True, category='pytest_internalerror'))
        with patch.object(verify_module, 'run_verification', rv):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result is None


class TestApplyMergeFlakeSuppression:
    """apply_merge_flake_suppression(failing_result, *, worktree, config,
    module_configs, merge_sha, event_store=None, escalation_queue=None,
    task_id=None, _confirm=...) — the merge-verify result handler (PRD task α).

    On a confirmed flake: emits the merge_flake_suppressed fact, bumps the
    storm streak, and returns a PASSED VerifyResult (category
    'merge_flake_suppressed') so the merge proceeds into the unscoped gate.
    On a non-confirmation: returns the ORIGINAL failing result unchanged, with
    no fact and no streak bump.
    """

    def _apply(self, failing, tmp_path, *, confirm, event_store=None,
               escalation_queue=None, task_id='2768'):
        from orchestrator import verify as verify_module

        return asyncio.run(
            verify_module.apply_merge_flake_suppression(
                failing,
                worktree=tmp_path,
                config=_make_config(tmp_path),
                module_configs=[_orch_module_config()],
                merge_sha=_MERGE_SHA,
                event_store=event_store,
                escalation_queue=escalation_queue,
                task_id=task_id,
                _confirm=confirm,
            )
        )

    def test_b1_suppression_returns_passed_result_and_emits_fact(self, tmp_path: Path) -> None:
        from orchestrator.event_store import EventType

        failing = _failing_with_logs()
        confirm = AsyncMock(return_value=list(_SUPPRESSED_IDS))
        es = _FakeEventStore()

        result = self._apply(failing, tmp_path, confirm=confirm, event_store=es)

        # Suppressed -> a PASSED result with the sentinel category.
        assert result.passed is True
        assert result.timed_out is False
        assert result.category == 'merge_flake_suppressed'
        # replace() preserves the original log paths (durable evidence).
        assert result.worktree_log_paths == ['/wt/verify.log']
        assert result.archive_log_paths == ['/archive/verify.log']

        # Exactly one structured fact, carrying the suppressed ids + sha + when.
        assert len(es.emits) == 1
        event_type, task_id, data = es.emits[0]
        assert event_type == EventType.merge_flake_suppressed
        assert task_id == '2768'
        assert data['node_ids'] == _SUPPRESSED_IDS
        assert data['merge_sha'] == _MERGE_SHA
        assert data['measured_at']  # non-empty ISO timestamp

    def test_b2_no_suppression_returns_original_unchanged_no_emit(self, tmp_path: Path) -> None:
        failing = _failing_with_logs()
        confirm = AsyncMock(return_value=None)
        es = _FakeEventStore()

        result = self._apply(failing, tmp_path, confirm=confirm, event_store=es)

        # The SAME object is returned unchanged (merge stays red).
        assert result is failing
        assert result.passed is False
        assert es.emits == []

    def test_suppression_tolerates_none_event_store(self, tmp_path: Path) -> None:
        """A suppression with event_store=None must not crash (None-safe emit)."""
        failing = _failing_with_logs()
        confirm = AsyncMock(return_value=list(_SUPPRESSED_IDS))

        result = self._apply(failing, tmp_path, confirm=confirm, event_store=None)

        assert result.passed is True
        assert result.category == 'merge_flake_suppressed'

    def test_confirm_called_with_config_and_failing_and_worktree(self, tmp_path: Path) -> None:
        """The injected _confirm receives (config, failing_result) positionally
        and worktree/module_configs as kwargs (the pure-gate contract)."""
        failing = _failing_with_logs()
        confirm = AsyncMock(return_value=None)

        self._apply(failing, tmp_path, confirm=confirm)

        confirm.assert_awaited_once()
        assert confirm.call_args.args[1] is failing
        assert confirm.call_args.kwargs['worktree'] == tmp_path
        assert 'module_configs' in confirm.call_args.kwargs
