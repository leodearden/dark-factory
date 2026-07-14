"""Tests for orchestrator.verify's plan-authoritative execution (task κ).

verify-scope-inversion PRD task κ (plans/verify-scope-inversion-prd.md):
``run_scoped_verification`` becomes derive→execute→aggregate, EXECUTING the
``VerifyPlan`` ``derive_verify_plan`` produces instead of re-deriving scope via
the hand-mirrored ``scope_module_config`` decision tree (deleted in step-6).
This changes WHO decides verify scope, not WHAT is decided — every golden
below pins that the plan-driven execution reproduces the pre-refactor scope
decisions byte-identically.

GOLDEN fixtures:

- ``ROOT_CONFTEST_DIFF`` / ``DATA_MODULE_DIFF`` are the W7 corpus (task-1077
  commit cb7277926d, task-1852 commit 7c9b316260), reused directly from
  ``test_verify_plan.py`` so both suites share one provenance-pinned source of
  truth rather than a second hand-copied literal.
- ``STRUCTURAL_FILE_DIFF`` / ``SOURCE_ONLY_ZERO_PYTEST_DIFF`` /
  ``FALLBACK_SUBPROJECT_DIFF`` / ``UNREGISTERED_PATH_DIFF`` are new shapes
  this task's goldens need that test_verify_plan.py's corpus doesn't already
  cover (D2 structural widening, the zero-pytest SKIPPED shape, task
  2344/2355 subproject rescoping, and the scoped-fallback-never-global-fanout
  boundary row respectively).

Spy helpers:

- :func:`_run_verification_spy` / :func:`_executed_module_configs` — the
  module-config-level spy (patches ``orchestrator.verify.run_verification``),
  modeled on ``TestRunScopedVerificationPlan``'s
  ``mock_run_verification.await_args.args[2]`` pattern (test_verify.py),
  generalised to capture every call in order instead of just the last.
- :func:`_run_cmd_spy` — the raw-shell-string-level spy (patches
  ``orchestrator.verify._run_cmd``), modeled on
  ``TestRunScopedVerificationSkipsUntouched``'s ``fake_run_cmd`` (test_verify.py).

Autouse fixtures (``_neutralize_verify_admission``, ``_clear_probe_cache``,
``_mock_merge_queue_verification``, ...) live in ``orchestrator/tests/conftest.py``
and apply to this module automatically — nothing to import or opt into here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from test_verify import _canned_passing_result, _real_worktree_reader
from test_verify_plan import (  # noqa: F401 — reused by this module's byte-identical goldens (steps 3/7/9)
    DATA_MODULE_DIFF,
    ROOT_CONFTEST_DIFF,
)

from orchestrator import verify, verify_plan
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_cmd import ToolKind, render

# ---------------------------------------------------------------------------
# GOLDEN diff shapes (task κ corpus) — see module docstring for provenance.
# ---------------------------------------------------------------------------

# A Protocol-defining source file (D2): file-scoped pyright cannot verify
# cross-file Protocol conformance, so a STRUCTURAL file must widen pyright to
# the unscoped package-wide command in BOTH the module-config and fallback
# paths. Unlike ROOT_CONFTEST_DIFF/DATA_MODULE_DIFF (path-only — classify_file
# never needs their content), STRUCTURAL detection is content-based, so a
# real worktree file with STRUCTURAL_FILE_CONTENT must be written for this
# path in each test that uses it.
STRUCTURAL_FILE_DIFF: list[str] = ['mymod/interfaces.py']
STRUCTURAL_FILE_CONTENT: str = (
    'from typing import Protocol\n\n\nclass Foo(Protocol):\n    def method(self) -> None: ...\n'
)

# A plain source file with no collectable test alongside it — the
# verify_plan.py:318-322 "no collectable test files touched" SKIPPED pytest
# shape (zero pytest invocations for this module; never a fabricated rc=5
# "no tests ran" run).
SOURCE_ONLY_ZERO_PYTEST_DIFF: list[str] = ['mymod/helpers.py']

# Fallback-subproject (cockpit-shaped, tasks 2344/2355): every touched file
# lives under a single top-level directory that carries its own
# pyproject.toml, so the fallback TEST command scopes to run *inside* that
# subproject (`cd cockpit && uv run pytest tests/test_c3.py`) and TYPE/LINT
# rescope into its own uv context. A test using this shape must also create
# `<worktree>/cockpit/pyproject.toml` — _single_subproject_prefix's
# discriminator — for the subproject-scoping branch to fire.
FALLBACK_SUBPROJECT_DIFF: list[str] = ['cockpit/tests/test_c3.py']

# Unregistered-path diff (only tests/scripts/ — boundary row 9): no
# module_configs prefix matches, so this drives the plan-driven FALLBACK
# branch (scoped commands only) — never the whole-repo global fan-out chain,
# the wall-clock-costly path this task must not regress into.
UNREGISTERED_PATH_DIFF: list[str] = ['tests/scripts/test_deploy.py']


# ---------------------------------------------------------------------------
# Spy helpers
# ---------------------------------------------------------------------------


def _run_verification_spy() -> AsyncMock:
    """AsyncMock stand-in for ``orchestrator.verify.run_verification``.

    Returns a canned passing ``VerifyResult`` for every call — never spawns a
    real subprocess. Patch via
    ``patch.object(verify, 'run_verification', new=_run_verification_spy())``
    and recover the ordered list of executed ``ModuleConfig``(s) afterward via
    :func:`_executed_module_configs`.
    """
    return AsyncMock(return_value=_canned_passing_result())


def _executed_module_configs(mock: AsyncMock) -> list[ModuleConfig]:
    """The ordered list of ``ModuleConfig``(s) *mock* was awaited with.

    *mock* is a spy built by :func:`_run_verification_spy`.
    ``run_verification``'s signature is
    ``(worktree, config, module_config=None, *, ...)`` — ``module_config`` is
    its 3rd positional argument at every ``run_scoped_verification`` call site
    that passes one (the module-config and fallback execution branches); the
    force_workspace/global/no-scope branches call it with only
    ``(worktree, config)`` and are excluded here, since there is no
    ``ModuleConfig`` to compare against a plan run in that case.
    """
    return [
        call.args[2]
        for call in mock.await_args_list
        if len(call.args) > 2 and call.args[2] is not None
    ]


def _run_cmd_spy() -> tuple[list[str], object]:
    """A ``_run_cmd`` fake recording every raw shell command string invoked.

    Mirrors ``TestRunScopedVerificationSkipsUntouched``'s ``fake_run_cmd``
    (test_verify.py) — patch via
    ``patch('orchestrator.verify._run_cmd', side_effect=<the returned fake>)``.
    Every call is a canned pass: ``(rc=0, '', timed_out=False)``.

    Returns ``(calls, fake)``: *calls* accumulates the raw command strings in
    call order; *fake* is the coroutine function to hand to ``patch(...)``.
    """
    calls: list[str] = []

    async def fake_run_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
        calls.append(cmd)
        return 0, '', False

    return calls, fake_run_cmd


# ---------------------------------------------------------------------------
# step-1 / step-2: A1 plan-authority spy (module-config path)
# ---------------------------------------------------------------------------


class TestModuleConfigPlanAuthority:
    """A1: the module-config branch is DRIVEN BY the plan, not scope_module_config.

    RED today: run_scoped_verification's module-config branch derives
    derive_verify_plan(...) only for the diagnostic VerifyResult.plan record
    (_safe_derive_verify_plan_dict) — the ModuleConfig(s) actually handed to
    run_verification are still built by the hand-mirrored scope_module_config
    decision tree. GREEN once step-2 rewires execution through a
    plan→ModuleConfig bridge instead, so scope_module_config is no longer
    consulted at all.
    """

    @pytest.mark.asyncio
    async def test_executed_commands_are_driven_by_the_plan(self, tmp_path: Path):
        """A touched test file + a plain source file under one registered module."""
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        (tmp_path / 'mymod' / 'tests' / 'test_thing.py').write_text('def test_thing(): pass\n')
        (tmp_path / 'mymod' / 'helpers.py').write_text('def helper():\n    return 1\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(
                prefix='mymod',
                test_command='uv run --directory mymod pytest tests/',
                lint_command='uv run --directory mymod ruff check .',
                type_check_command='uv run --directory mymod pyright',
            ),
        ]
        task_files = ['mymod/tests/test_thing.py', 'mymod/helpers.py']
        existing_files = [f for f in task_files if (tmp_path / f).exists()]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=task_files,
            )

        # The plan-authoritative execution never consults a hand-mirrored
        # scope decision tree — that IS the inversion (task κ). The twin
        # (scope_module_config) is deleted entirely (step-6); see
        # TestDeleteTheTwinInvariant for that invariant directly.

        expected_plan = verify_plan.derive_verify_plan(
            existing_files, module_configs, config, _real_worktree_reader(tmp_path),
        )

        executed = _executed_module_configs(mock_run_verification)
        # No module executed that the plan didn't include.
        assert {mc.prefix for mc in executed} == {
            run.module_prefix for run in expected_plan.runs
        }
        assert len(executed) == 1
        executed_mc = executed[0]

        by_tool = {run.cmd.tool: run for run in expected_plan.runs if run.cmd is not None}

        pytest_run = by_tool[ToolKind.PYTEST]
        assert pytest_run.scope_kind is verify_plan.ScopeKind.FILE_SCOPED
        assert pytest_run.cmd is not None
        assert executed_mc.test_command == render(pytest_run.cmd)

        ruff_run = by_tool[ToolKind.RUFF]
        assert ruff_run.scope_kind is verify_plan.ScopeKind.FILE_SCOPED
        assert ruff_run.cmd is not None
        assert executed_mc.lint_command == render(ruff_run.cmd)

        pyright_run = by_tool[ToolKind.PYRIGHT]
        assert pyright_run.scope_kind is verify_plan.ScopeKind.FILE_SCOPED
        assert pyright_run.cmd is not None
        assert executed_mc.type_check_command == render(pyright_run.cmd)

        # VerifyResult.plan is the plan that DROVE execution above, not an
        # independently re-derived diagnostic mirror.
        assert result.plan == expected_plan.to_dict()


# ---------------------------------------------------------------------------
# step-3: module-path byte-identical scope goldens
# ---------------------------------------------------------------------------


class TestModuleConfigScopeGoldens:
    """Byte-identical goldens: the plan-driven executed ModuleConfig commands
    reproduce ``scope_module_config``'s exact pre-refactor decisions, module-config
    path. Each golden derives its expected string the SAME way
    ``scope_module_config`` itself would have produced it (the verbatim mc
    command for a widen/full-suite trigger; ``verify._scope_to_keyword`` for a
    file-scoped one) — not merely self-consistency with the plan (that's
    step-1/2's A1 spy) — so a future divergence between the plan-driven bridge
    and ``scope_module_config``'s historical behaviour is caught here directly.
    """

    @pytest.mark.asyncio
    async def test_conftest_touched_uses_full_suite_verbatim(self, tmp_path: Path):
        """(a) ROOT_CONFTEST_DIFF (task-1077) -> test_command == mc.test_command verbatim (D1)."""
        conftest_path = ROOT_CONFTEST_DIFF[0]
        full = tmp_path / conftest_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('# root\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(
                prefix='orchestrator',
                test_command=(
                    'uv run --project orchestrator --directory orchestrator '
                    'pytest tests/ --tb=short -q'
                ),
            ),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[conftest_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command == module_configs[0].test_command, (
            f'conftest must trigger the verbatim full suite: {executed[0].test_command!r}'
        )

    @pytest.mark.asyncio
    async def test_data_module_touched_uses_full_suite_verbatim(self, tmp_path: Path):
        """(b) DATA_MODULE_DIFF (task-1852) -> test_command == mc.test_command verbatim (D1)."""
        data_path = DATA_MODULE_DIFF[0]
        full = tmp_path / data_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('ALLOWLIST = set()\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='shared', test_command='uv run --directory shared pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[data_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command == module_configs[0].test_command, (
            f'a test-data module must trigger the verbatim full suite: {executed[0].test_command!r}'
        )
        assert 'silent_fallthrough_allowlist.py' not in (executed[0].test_command or ''), (
            'the data file must never appear as a pytest target'
        )

    @pytest.mark.asyncio
    async def test_structural_file_unscopes_type_check_verbatim(self, tmp_path: Path):
        """(c) STRUCTURAL_FILE_DIFF -> type_check_command == mc.type_check_command
        verbatim, INCLUDING --directory (D2)."""
        struct_path = STRUCTURAL_FILE_DIFF[0]
        full = tmp_path / struct_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(STRUCTURAL_FILE_CONTENT)

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(
                prefix='mymod',
                type_check_command='uv run --project mymod --directory mymod pyright src/ tests/',
            ),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[struct_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].type_check_command == module_configs[0].type_check_command, (
            f'a structural file must trigger the verbatim unscoped type check '
            f'(--directory preserved): {executed[0].type_check_command!r}'
        )

    @pytest.mark.asyncio
    async def test_source_only_diff_skips_pytest_entirely(self, tmp_path: Path):
        """(d) SOURCE_ONLY_ZERO_PYTEST_DIFF -> test_command is None (verify_plan.py:318-322)."""
        source_path = SOURCE_ONLY_ZERO_PYTEST_DIFF[0]
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='mymod', test_command='uv run --directory mymod pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[source_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command is None, (
            f'a source-only diff with no collectable tests must never fabricate '
            f'a pytest run: {executed[0].test_command!r}'
        )

    @pytest.mark.asyncio
    async def test_plain_test_file_matches_scope_to_keyword(self, tmp_path: Path):
        """(e) a plain touched test file -> test_command == _scope_to_keyword's own output (FILE_SCOPED)."""
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        test_path = 'mymod/tests/test_thing.py'
        (tmp_path / test_path).write_text('def test_thing(): pass\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='mymod', test_command='uv run --directory mymod pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[test_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        expected = verify._scope_to_keyword(module_configs[0].test_command, 'pytest', [test_path])
        assert executed[0].test_command == expected, (
            f'expected the _scope_to_keyword-scoped string {expected!r}, got '
            f'{executed[0].test_command!r}'
        )


# ---------------------------------------------------------------------------
# step-5: delete-the-twin invariant
# ---------------------------------------------------------------------------


class TestDeleteTheTwinInvariant:
    """task κ deletes ``scope_module_config`` — the hand-mirrored twin the two
    drift notes (verify.py / verify_plan.py) document — once the plan is the
    SOLE decision tree for module scope.

    RED today: ``scope_module_config`` still exists (removed in step-6, along
    with both drift-note passages).
    """

    def test_scope_module_config_no_longer_exists(self):
        assert not hasattr(verify, 'scope_module_config'), (
            'scope_module_config is the hand-mirrored twin the drift notes '
            'document — task κ deletes it once derive_verify_plan is the sole '
            'decision tree for module scope (step-6)'
        )

    def test_verify_py_drift_note_is_gone(self):
        source = Path(verify.__file__).read_text()
        assert 'has_structural/has_conftest/has_test_data scan is a SEPARATE decision' not in source, (
            "verify.py's scope_module_config drift note must be removed once "
            'the twin it documents is deleted (step-6)'
        )

    def test_verify_plan_drift_note_is_gone(self):
        source = Path(verify_plan.__file__).read_text()
        assert 'mirrored BY HAND in verify.scope_module_config' not in source, (
            "verify_plan._derive_module_runs' drift note must be removed once "
            'scope_module_config (the hand-mirrored twin) is deleted (step-6)'
        )

    @pytest.mark.asyncio
    async def test_classify_file_runs_exactly_once_per_touched_file(self, tmp_path: Path):
        """The module-config execution path does not independently re-classify:
        classify_file runs exactly once per distinct touched .py file — proof
        there is no second (hand-mirrored) decision tree consuming it.
        """
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        (tmp_path / 'mymod' / 'tests' / 'test_thing.py').write_text('def test_thing(): pass\n')
        (tmp_path / 'mymod' / 'helpers.py').write_text('def helper():\n    return 1\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(
                prefix='mymod',
                test_command='uv run --directory mymod pytest tests/',
                lint_command='uv run --directory mymod ruff check .',
                type_check_command='uv run --directory mymod pyright',
            ),
        ]
        task_files = ['mymod/tests/test_thing.py', 'mymod/helpers.py']

        calls: list[str] = []
        real_classify_file = verify_plan.classify_file

        def counting_classify_file(path, content):
            calls.append(path)
            return real_classify_file(path, content)

        mock_run_verification = _run_verification_spy()
        with (
            patch.object(verify, 'run_verification', new=mock_run_verification),
            patch.object(verify_plan, 'classify_file', side_effect=counting_classify_file),
        ):
            await run_scoped_verification(
                tmp_path, config, module_configs, task_files=task_files,
            )

        assert len(calls) == len(set(calls)) == len(task_files), (
            f'classify_file must run exactly once per distinct touched .py file '
            f'on the module-config path — no independent re-classification by a '
            f'second (hand-mirrored) decision tree: calls={calls!r}'
        )
