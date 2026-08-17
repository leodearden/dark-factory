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
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult
from orchestrator.verify_cmd import (
    ToolKind,
    parse_config_command,
    render,
    with_pytest_timeout,
)
from orchestrator.verify_runner import (
    LocalRunner,
    MergeVerifySpec,
    UnscopedTypecheckSpec,
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


def _module_config(prefix: str) -> ModuleConfig:
    """A merge-role subproject whose commands mirror dark_factory's real
    per-subproject shape (``uv run --project X --directory X pytest ...``).

    THE single definition of the fixture's command shape — `_orch_module_config`
    is just this at ``prefix='orchestrator'``, so a change to the shape lands in
    exactly one place and the single- and multi-subproject fixtures cannot
    silently diverge.
    """
    return ModuleConfig(
        prefix=prefix,
        test_command=(
            f'uv run --project {prefix} --directory {prefix} '
            'pytest tests/ --tb=short -q'
        ),
        lint_command=f'uv run --project {prefix} ruff check src/',
        type_check_command=f'uv run --project {prefix} pyright src/',
    )


def _orch_module_config() -> ModuleConfig:
    """The default single-subproject fixture used by most cases here."""
    return _module_config('orchestrator')


def _fmt_log(call) -> str:
    """Render a mocked ``logger.<level>(fmt, *args)`` call to its final text,
    so a substring assertion sees what an operator would actually read."""
    args = call.args
    if not args:
        return ''
    return (args[0] % args[1:]) if len(args) > 1 else str(args[0])


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

    # -- task-3290: node-id -> subproject mapping DELEGATES to the shared
    # -- helper (_group_node_ids_by_subproject), plus the characterization
    # -- guards the unification must not lose.
    # ----------------------------------------------------------------------

    @staticmethod
    def _b1_failing() -> VerifyResult:
        return VerifyResult(
            passed=False, test_output=_B1_TEST_OUTPUT, lint_output='',
            type_output='', summary='fail', category='test_failure',
        )

    def test_delegates_node_id_mapping_to_shared_helper(self, tmp_path: Path) -> None:
        """The gate maps node-ids via the SHARED
        `_group_node_ids_by_subproject` helper — not a private inline copy.

        The helper takes DICT-shaped module_configs, so the gate must hand it
        the `{mc.prefix: mc}` dict it already builds (dict construction from a
        prefix-deduped list preserves order, keeping candidate iteration — and
        therefore first-wins ambiguity resolution — byte-identical). The
        `log_label` must name THIS call site so an operator can attribute the
        helper's log lines to the merge gate rather than the sweep.

        RED today: the inline copy never calls the helper, so the spy records
        zero calls. The file IS materialized so the inline path would
        otherwise succeed — the failure is the missing delegation, not a
        mapping miss.

        SCOPE: this is a deliberately STRUCTURAL pin — it asserts on a private
        helper's call signature, so a rename/re-ordering of
        `_group_node_ids_by_subproject`'s parameters breaks it by design. It
        exists to keep the third inline copy from creeping back (task 3290 is
        the de-duplication); the *behavioral* contract lives in the end-to-end
        cases below (ambiguity first-wins, unmapped -> not suppressing) and in
        the B1/B2/B3 suppress/still-fails cases above.
        """
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)
        mc = _orch_module_config()

        spy = MagicMock(return_value={'orchestrator': [_B1_FAILED_ID, _B1_CRASH_ID]})
        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, '_group_node_ids_by_subproject', spy),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = self._run(config, self._b1_failing(), tmp_path, [mc])

        assert spy.call_count == 1, (
            f'Expected exactly one delegation to _group_node_ids_by_subproject, '
            f'got {spy.call_count} (call_args_list={spy.call_args_list!r})'
        )
        call = spy.call_args
        assert call.args[0] == tmp_path, call.args[0]
        assert call.args[1] == {'orchestrator': mc}, call.args[1]
        assert call.args[2] == [_B1_FAILED_ID, _B1_CRASH_ID], call.args[2]
        assert call.kwargs['log_label'] == 'confirm_merge_verify_flake_suppressible', (
            call.kwargs
        )
        # The helper's groups drive the re-run, which confirms green -> suppress.
        assert result == [_B1_FAILED_ID, _B1_CRASH_ID], result

    def test_helper_returning_none_fails_closed_without_rerunning(
        self, tmp_path: Path
    ) -> None:
        """`None` from the helper (an unmappable node-id) fails CLOSED to red
        end-to-end THROUGH the delegation boundary: no isolated re-run is
        attempted and the gate returns None.

        RED today: the inline copy ignores the spy and maps the materialized
        file successfully, so the gate re-runs and suppresses.
        """
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        spy = MagicMock(return_value=None)
        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, '_group_node_ids_by_subproject', spy),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = self._run(config, self._b1_failing(), tmp_path, [_orch_module_config()])

        assert result is None, f'Expected None (fail-closed to red), got {result!r}'
        rv.assert_not_awaited()

    def test_helper_returning_empty_dict_fails_closed(self, tmp_path: Path) -> None:
        """`{}` from the helper ALSO fails closed — the `not groups` (not
        `is None`) guard.

        An empty dict must never fall through the `for prefix, group_node_ids
        in groups.items()` loop: that would exit having run ZERO isolated
        re-runs and then `return node_ids` — a full suppression verdict on
        zero evidence, letting a genuinely red merge land. That is precisely
        the merge-masks-a-real-red failure this gate exists to prevent, so the
        guard is pinned rather than left to a future simplification.

        DEFENSIVE-ONLY — do NOT read this as reachable behavior. In production
        the gate returns early on empty *node_ids*, and the helper documents
        `{}` as its empty-input-only return, so `{}` cannot arrive here today;
        the case is only reachable by patching the helper out, exactly as done
        below. It pins the guard's PRESENCE so a future "simplify to `is None`"
        edit — or a helper change that starts returning `{}` for a
        non-empty input — cannot quietly turn zero evidence into a suppression.

        RED today: no such guard exists (the inline loop cannot produce {}).
        """
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'orchestrator/tests/test_x.py')
        config = _make_config(tmp_path)

        spy = MagicMock(return_value={})
        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, '_group_node_ids_by_subproject', spy),
            patch.object(verify_module, 'run_verification', rv),
        ):
            result = self._run(config, self._b1_failing(), tmp_path, [_orch_module_config()])

        assert result is None, (
            f'Expected None: an empty group dict is zero evidence, not '
            f'"all groups clean" — got {result!r}'
        )
        rv.assert_not_awaited()

    def test_ambiguous_node_id_uses_first_by_list_order_and_warns(
        self, tmp_path: Path
    ) -> None:
        """CHARACTERIZATION (green before AND after the unification).

        A bare subproject-relative node-id present under TWO given
        subprojects resolves deterministically to the FIRST by
        *module_configs* list order, and logs a WARNING naming the
        ambiguity. This is the sole guard on the list->dict conversion's
        order-preservation claim: `{mc.prefix: mc for mc in module_configs}`
        preserves insertion order, so first-wins must stay byte-identical
        after the gate starts iterating the dict inside the shared helper.
        """
        from orchestrator import verify as verify_module

        _materialize(tmp_path, 'alpha/tests/test_dup.py', 'beta/tests/test_dup.py')
        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output='FAILED tests/test_dup.py::test_dup\n',
            lint_output='', type_output='', summary='fail', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, 'run_verification', rv),
            patch.object(verify_module, 'logger') as mock_logger,
        ):
            result = self._run(
                config, failing, tmp_path, [_module_config('alpha'), _module_config('beta')],
            )

        assert result == ['tests/test_dup.py::test_dup'], result
        rv.assert_awaited()
        cmd = rv.call_args.args[2].test_command
        assert 'alpha/tests/test_dup.py::test_dup' in cmd, (
            f'Expected FIRST-by-list-order (alpha) attribution, got {cmd!r}'
        )
        assert 'beta/tests/test_dup.py::test_dup' not in cmd, cmd

        warned = any(
            'tests/test_dup.py::test_dup' in _fmt_log(call)
            for call in mock_logger.warning.call_args_list
        )
        assert warned, (
            f'Expected a WARNING naming the ambiguous node-id; got '
            f'calls={mock_logger.warning.call_args_list!r}'
        )

    def test_unmapped_node_id_logs_not_suppressing(self, tmp_path: Path) -> None:
        """CHARACTERIZATION (green before AND after the unification).

        The unmapped path emits an INFO carrying the operator-facing verdict
        vocabulary "not suppressing". The SHARED helper only knows the neutral
        "unconfirmable" — only the CALLER knows whether that means "file an
        alarm" (the sweep) or "stay red" (this gate) — so the unification must
        add a caller-side INFO rather than silently changing what an operator
        greps merge-lane logs for.

        That ONE line must also name the offending node-id: after the
        unification the helper names the node-id on a line of its own, so a
        caller line carrying only the verdict would force an operator to
        correlate two lines to answer "which test failed to map?".
        """
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        failing = VerifyResult(
            passed=False, test_output='FAILED some/other/tests/test_q.py::test_z\n',
            lint_output='', type_output='', summary='fail', category='test_failure',
        )

        rv = AsyncMock(return_value=_result(True))
        with (
            patch.object(verify_module, 'run_verification', rv),
            patch.object(verify_module, 'logger') as mock_logger,
        ):
            result = self._run(config, failing, tmp_path, [_orch_module_config()])

        assert result is None
        rendered = [_fmt_log(call) for call in mock_logger.info.call_args_list]
        logged = any(
            'not suppressing' in msg and 'some/other/tests/test_q.py::test_z' in msg
            for msg in rendered
        )
        assert logged, (
            f'Expected ONE self-contained INFO log naming both the unmapped '
            f'node-id and the "not suppressing" verdict; got {rendered!r}'
        )


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


class TestSuppressionStormStreak:
    """INV-4 storm detector: _bump_suppression_streak_and_maybe_escalate files
    exactly ONE born-at-L2 escalation once the module-global suppression streak
    reaches the threshold, then resets. Only suppressions bump it (task α, B4).
    """

    def _reset(self):
        from orchestrator import verify as verify_module

        verify_module._merge_flake_suppression_streak = 0
        return verify_module

    def test_storm_files_one_l2_escalation_at_threshold_and_resets(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        q = _FakeEscalationQueue(open_l2=None)

        for _ in range(threshold):
            vm._bump_suppression_streak_and_maybe_escalate(q, '2768', _MERGE_SHA)

        assert len(q.submitted) == 1, q.submitted
        esc = q.submitted[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.agent_role.startswith('orchestrator-')
        assert esc.category == 'merge_flake_suppression_storm'
        # Counter cleared so the next window starts fresh (B4).
        assert vm._merge_flake_suppression_streak == 0

    def test_below_threshold_files_nothing(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        q = _FakeEscalationQueue(open_l2=None)

        for _ in range(threshold - 1):
            vm._bump_suppression_streak_and_maybe_escalate(q, '2768', _MERGE_SHA)

        assert q.submitted == []
        assert vm._merge_flake_suppression_streak == threshold - 1

    def test_dedup_skips_submit_when_open_l2_exists(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD
        q = _FakeEscalationQueue(open_l2=object())  # a truthy open L2

        for _ in range(threshold):
            vm._bump_suppression_streak_and_maybe_escalate(q, '2768', _MERGE_SHA)

        assert q.submitted == []
        # get_by_task consulted for dedup on the same sentinel, at level=2.
        assert q.get_by_task_calls, q.get_by_task_calls
        assert q.get_by_task_calls[-1][2] == 2
        assert vm._merge_flake_suppression_streak == 0

    def test_none_queue_is_tolerated(self) -> None:
        vm = self._reset()
        threshold = vm._MERGE_FLAKE_SUPPRESSION_STREAK_THRESHOLD

        for _ in range(threshold):
            vm._bump_suppression_streak_and_maybe_escalate(None, '2768', _MERGE_SHA)

        # No crash; window resets without filing (no queue to file into).
        assert vm._merge_flake_suppression_streak == 0

    def test_non_suppression_does_not_bump_streak(self, tmp_path: Path) -> None:
        vm = self._reset()
        confirm = AsyncMock(return_value=None)

        asyncio.run(
            vm.apply_merge_flake_suppression(
                _failing_with_logs(),
                worktree=tmp_path,
                config=_make_config(tmp_path),
                module_configs=[_orch_module_config()],
                merge_sha=_MERGE_SHA,
                escalation_queue=_FakeEscalationQueue(),
                _confirm=confirm,
            )
        )

        assert vm._merge_flake_suppression_streak == 0


# --- LocalRunner.run_merge_verify integration (step-9/10) ---------------------

_MERGE_VERIFY_SHA = 'abc123'


def _merge_spec() -> MergeVerifySpec:
    return MergeVerifySpec(
        verify_commands=(),
        unscoped_typecheck=UnscopedTypecheckSpec(commands=()),
        task_files=None,
        verify_env={},
        cold_timeout_secs=60.0,
    )


def _clean_unscoped_gate() -> MagicMock:
    """An unscoped-typecheck gate result that is not broken (falls through)."""
    return MagicMock(
        broken=False, timed_out=False, failing_subprojects=[], timed_out_subprojects=[],
    )


def _broken_unscoped_gate() -> MagicMock:
    """A broken unscoped-typecheck gate (turns a passed scoped result red)."""
    return MagicMock(
        broken=True, timed_out=False, failing_subprojects=['orchestrator'],
        timed_out_subprojects=[], detail='type error line 10',
    )


def _make_hook_runner(
    tmp_path: Path,
    *,
    scoped_result: VerifyResult,
    unscoped_gate: MagicMock | None = None,
    event_store=None,
    escalation_queue=None,
) -> tuple[LocalRunner, AsyncMock, AsyncMock]:
    """A LocalRunner whose scoped phase fails (so the α hook fires), with a
    fake unscoped gate + optional event_store / escalation_queue threaded in.
    """
    run_scoped = AsyncMock(return_value=scoped_result)
    run_unscoped = AsyncMock(
        return_value=unscoped_gate if unscoped_gate is not None else _clean_unscoped_gate()
    )
    runner = LocalRunner(
        merge_wt=tmp_path,
        config=_make_config(tmp_path),
        module_configs=[_orch_module_config()],
        task_files=None,
        run_scoped=run_scoped,
        run_unscoped=run_unscoped,
        task_id='2768',
        event_store=event_store,
        escalation_queue=escalation_queue,
    )
    return runner, run_scoped, run_unscoped


class TestLocalRunnerMergeFlakeSuppressionHook:
    """LocalRunner.run_merge_verify wires apply_merge_flake_suppression into the
    `not scoped.passed` branch (PRD task α, steps 9/10).

    RED until step-10 adds event_store/escalation_queue to LocalRunner.__init__
    and calls verify.apply_merge_flake_suppression on the scoped-fail branch
    (resolved via the verify module so this monkeypatch takes effect).
    """

    # -- (a) suppression is NOT a bypass of the unscoped typecheck gate ----

    def test_suppressed_result_proceeds_into_clean_unscoped_gate(self, tmp_path: Path) -> None:
        """apply_* returns a PASSED (suppressed) result -> run_merge_verify runs
        the unscoped gate and, when it is clean, returns the suppressed pass."""
        from orchestrator import verify as verify_module

        runner, _run_scoped, run_unscoped = _make_hook_runner(
            tmp_path, scoped_result=_result(False),
        )
        suppressed = _result(True, category='merge_flake_suppressed')
        apply = AsyncMock(return_value=suppressed)

        with patch.object(verify_module, 'apply_merge_flake_suppression', apply):
            result = asyncio.run(runner.run_merge_verify(_MERGE_VERIFY_SHA, _merge_spec()))

        apply.assert_awaited_once()
        run_unscoped.assert_awaited_once()  # suppression does NOT skip the gate
        assert result.passed is True
        assert result.category == 'merge_flake_suppressed'

    def test_suppressed_result_still_gated_by_broken_unscoped(self, tmp_path: Path) -> None:
        """Even after suppression, a BROKEN unscoped gate turns the merge red —
        proving suppression is not a bypass of the unscoped typecheck gate."""
        from orchestrator import verify as verify_module
        from orchestrator.verify_runner import UNSCOPED_TYPECHECK_FAILED_CATEGORY

        runner, _run_scoped, run_unscoped = _make_hook_runner(
            tmp_path, scoped_result=_result(False), unscoped_gate=_broken_unscoped_gate(),
        )
        apply = AsyncMock(return_value=_result(True, category='merge_flake_suppressed'))

        with patch.object(verify_module, 'apply_merge_flake_suppression', apply):
            result = asyncio.run(runner.run_merge_verify(_MERGE_VERIFY_SHA, _merge_spec()))

        run_unscoped.assert_awaited_once()
        assert result.passed is False
        assert result.category == UNSCOPED_TYPECHECK_FAILED_CATEGORY

    # -- (b) non-suppression is byte-identical to today's short-circuit ----

    def test_non_suppression_returns_failing_and_skips_unscoped(self, tmp_path: Path) -> None:
        """apply_* returns the ORIGINAL failing result -> run_merge_verify
        returns it unchanged and does NOT call the unscoped gate."""
        from orchestrator import verify as verify_module

        failing = _result(False, category='test_failure')
        runner, _run_scoped, run_unscoped = _make_hook_runner(
            tmp_path, scoped_result=failing,
        )
        apply = AsyncMock(side_effect=lambda fr, **kw: fr)  # non-confirmation: unchanged

        with patch.object(verify_module, 'apply_merge_flake_suppression', apply):
            result = asyncio.run(runner.run_merge_verify(_MERGE_VERIFY_SHA, _merge_spec()))

        apply.assert_awaited_once()
        assert result is failing
        assert result.passed is False
        run_unscoped.assert_not_awaited()

    # -- scoped PASS path is byte-identical (hook never fires) --------------

    def test_scoped_pass_does_not_invoke_suppression_hook(self, tmp_path: Path) -> None:
        """A passing scoped phase never reaches the α hook (byte-identical)."""
        from orchestrator import verify as verify_module

        runner, _run_scoped, run_unscoped = _make_hook_runner(
            tmp_path, scoped_result=_result(True),
        )
        apply = AsyncMock(return_value=_result(True, category='merge_flake_suppressed'))

        with patch.object(verify_module, 'apply_merge_flake_suppression', apply):
            result = asyncio.run(runner.run_merge_verify(_MERGE_VERIFY_SHA, _merge_spec()))

        apply.assert_not_awaited()
        run_unscoped.assert_awaited_once()
        assert result.passed is True

    # -- (c) event_store/escalation_queue thread into the hook call --------

    def test_init_threads_event_store_and_escalation_queue_into_hook(self, tmp_path: Path) -> None:
        """LocalRunner.__init__ accepts event_store=/escalation_queue= and passes
        them (plus worktree/config/module_configs/merge_sha/task_id) to the hook."""
        from orchestrator import verify as verify_module

        es = _FakeEventStore()
        eq = _FakeEscalationQueue()
        failing = _result(False)
        runner, _run_scoped, _run_unscoped = _make_hook_runner(
            tmp_path, scoped_result=failing, event_store=es, escalation_queue=eq,
        )
        apply = AsyncMock(side_effect=lambda fr, **kw: fr)

        with patch.object(verify_module, 'apply_merge_flake_suppression', apply):
            asyncio.run(runner.run_merge_verify(_MERGE_VERIFY_SHA, _merge_spec()))

        apply.assert_awaited_once()
        # The failing scoped result is passed positionally as the first arg.
        assert apply.call_args.args[0] is failing
        kwargs = apply.call_args.kwargs
        assert kwargs['event_store'] is es
        assert kwargs['escalation_queue'] is eq
        assert kwargs['worktree'] == tmp_path
        assert kwargs['merge_sha'] == _MERGE_VERIFY_SHA
        assert kwargs['task_id'] == '2768'
        assert kwargs['module_configs'] == [_orch_module_config()]
        assert 'config' in kwargs

    def test_init_defaults_event_store_and_escalation_queue_to_none(self, tmp_path: Path) -> None:
        """Omitting event_store/escalation_queue defaults both to None and
        threads None into the hook (byte-identical CLI/test construction)."""
        from orchestrator import verify as verify_module

        failing = _result(False)
        run_scoped = AsyncMock(return_value=failing)
        run_unscoped = AsyncMock(return_value=_clean_unscoped_gate())
        # Construct WITHOUT event_store/escalation_queue (default None).
        runner = LocalRunner(
            merge_wt=tmp_path,
            config=_make_config(tmp_path),
            module_configs=[_orch_module_config()],
            task_files=None,
            run_scoped=run_scoped,
            run_unscoped=run_unscoped,
            task_id='2768',
        )
        apply = AsyncMock(side_effect=lambda fr, **kw: fr)

        with patch.object(verify_module, 'apply_merge_flake_suppression', apply):
            asyncio.run(runner.run_merge_verify(_MERGE_VERIFY_SHA, _merge_spec()))

        kwargs = apply.call_args.kwargs
        assert kwargs['event_store'] is None
        assert kwargs['escalation_queue'] is None
