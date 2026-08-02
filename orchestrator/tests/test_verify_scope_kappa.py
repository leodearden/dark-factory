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

import logging
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from test_verify import _canned_passing_result, _real_worktree_reader, _write_guard_script
from test_verify_plan import (  # noqa: F401 — reused by this module's byte-identical goldens (steps 3/7/9)
    DATA_MODULE_DIFF,
    ROOT_CONFTEST_DIFF,
)

from orchestrator import verify, verify_plan
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_cmd import ToolKind, VerifyCmd, render

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

# Mixed root+subproject (task 2368): a root-level conftest.py alongside the
# cockpit subproject test file above — mirrors
# TestBuildFallbackConfigSubprojectScoped
# .test_mixed_root_conftest_plus_subproject_scopes_to_subproject_and_root_owning_tests
# (test_verify.py). File order matches that test exactly.
MIXED_ROOT_SUBPROJECT_DIFF: list[str] = [*FALLBACK_SUBPROJECT_DIFF, 'conftest.py']

# The real dark_factory root-level fleet chain (orchestrator/config.yaml) —
# reused verbatim from TestBuildFallbackConfigSubprojectScoped (test_verify.py)
# so the step-7 subproject-rescoping goldens below (d)/(e) exercise the SAME
# realistic multi-clause commands tasks 2344/2355/2368 were fixed against.
# _FLEET_TYPE_COMMAND is itself already a multi-clause OPAQUE chain (no
# 'pytest'/'cargo' token — see verify_cmd._parse_chain), so (d)/(e) already
# exercise _scope_to_keyword's OPAQUE first-clause scoping for TYPE, ahead of
# any subproject rescoping. _FLEET_LINT_COMMAND, in contrast, is a SINGLE
# ruff-check clause — well-formed/never-OPAQUE even parsed whole — so a
# second, genuinely multi-clause OPAQUE lint variant
# (_FLEET_LINT_COMMAND_OPAQUE) is defined separately below for golden (f).
_FLEET_TEST_COMMAND: str = (
    'cd shared && uv run pytest tests/ && cd ../escalation && uv run pytest tests/ '
    '&& cd ../orchestrator && uv run pytest tests/ && cd ../fused-memory && uv run pytest tests/ '
    '&& cd ../dashboard && uv run pytest tests/'
)
_FLEET_LINT_COMMAND: str = 'uv run ruff check shared escalation fused-memory orchestrator dashboard'
_FLEET_TYPE_COMMAND: str = (
    'cd fused-memory && npx pyright && cd ../orchestrator && npx pyright '
    '&& cd ../dashboard && npx pyright'
)

# The real dark_factory lint_command verbatim (orchestrator/config.yaml) — a
# genuine multi-clause OPAQUE chain (unlike _FLEET_LINT_COMMAND above). Reused
# from TestBuildFallbackConfigWithNonDefaultCommands
# .test_fallback_lint_reprojects_repo_root_file_to_ruff_bearing_context
# (test_verify.py) for the OPAQUE-fleet-chain golden (f) below.
_FLEET_LINT_COMMAND_OPAQUE: str = (
    'uv run ruff check shared escalation fused-memory orchestrator dashboard '
    '&& python3 fused-memory/scripts/check_bare_magicmock_config.py '
    'shared/tests escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
)


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
        """A touched test file + a plain source file under one registered module.

        Since task 3294 this MIXED shape full-suites pytest at the default
        role='task' (any touched SOURCE/STRUCTURAL file under the prefix runs
        the owning module's whole test_command), so this pins the FULL_SUITE
        arm of the plan→ModuleConfig mapping alongside lint/pyright's
        FILE_SCOPED arms. The class's contract — executed == planned, with no
        hand-mirrored decision tree in between — is unchanged; only which arm
        the pytest slot exercises moved.
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
        # Task 3294: this MIXED shape (production file + co-committed test)
        # full-suites pytest at role='task', so this now pins the FULL_SUITE
        # arm of the plan→ModuleConfig mapping. The expectation is the
        # VERBATIM configured command, not render(cmd): a FULL_SUITE slot is
        # rendered by _executed_module_configs_from_plan as `getattr(mc, attr)`,
        # while render() normalises `--directory` into a leading `cd` — so the
        # render-based form would fail on a correct mapping.
        assert pytest_run.scope_kind is verify_plan.ScopeKind.FULL_SUITE
        assert pytest_run.cmd is not None
        assert executed_mc.test_command == module_configs[0].test_command

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
    async def test_source_only_diff_floors_pytest_to_full_suite_at_task_role(
        self, tmp_path: Path,
    ):
        """(d) SOURCE_ONLY_ZERO_PYTEST_DIFF at role='task' -> test_command == the
        owning module's verbatim full suite (λ, task 2589 R3: the task-role pytest
        floor). Pre-λ this fabricated ZERO pytest signal (test_command was None);
        see test_source_only_diff_skips_pytest_entirely_at_merge_role below for the
        preserved legacy SKIPPED shape (R4)."""
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
                tmp_path, config, module_configs, task_files=[source_path], role='task',
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command == module_configs[0].test_command, (
            f'a source-only diff at task role must full-suite the owning module\'s '
            f'pytest (the task-role floor): {executed[0].test_command!r}'
        )

    @pytest.mark.asyncio
    async def test_source_only_diff_skips_pytest_entirely_at_merge_role(self, tmp_path: Path):
        """(d) R4 rollback golden: the SAME diff at role='merge' + merge_verify_breadth=
        'scoped' preserves the pre-λ legacy shape -> test_command is None
        (verify_plan.py's pytest else-branch SKIPPED). The task-role floor (R3) never
        widens the merge gate itself — that widening is the separate, knob-gated
        merge_verify_breadth='full' path."""
        source_path = SOURCE_ONLY_ZERO_PYTEST_DIFF[0]
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        config = OrchestratorConfig(project_root=tmp_path, merge_verify_breadth='scoped')
        module_configs = [
            ModuleConfig(prefix='mymod', test_command='uv run --directory mymod pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[source_path], role='merge',
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command is None, (
            f'merge role + scoped breadth must preserve the legacy SKIPPED shape: '
            f'{executed[0].test_command!r}'
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

    @pytest.mark.asyncio
    async def test_file_scoped_command_with_trailing_flags_matches_scope_to_keyword(
        self, tmp_path: Path,
    ):
        """(f) FILE_SCOPED command with flags trailing the target (dark_factory's
        real ``... pytest tests/ --tb=short -q`` shape) -> test_command ==
        _scope_to_keyword's own output: first-clause scoped, with the
        trailing flags DROPPED (reviewer_comprehensive correctness finding,
        verify_plan.py:207) — not preserved by scoping the whole parsed
        command, which would both change the command actually run in the
        merge gate for every real flag-bearing config AND (for a
        value-taking flag, see the next golden) misread the flag's value as
        an extra scope target.
        """
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        test_path = 'mymod/tests/test_thing.py'
        (tmp_path / test_path).write_text('def test_thing(): pass\n')

        test_command = 'uv run --directory mymod pytest tests/ --tb=short -q'
        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [ModuleConfig(prefix='mymod', test_command=test_command)]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[test_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        expected = verify._scope_to_keyword(test_command, 'pytest', [test_path])
        assert executed[0].test_command == expected, (
            f'expected the _scope_to_keyword-scoped string {expected!r}, got '
            f'{executed[0].test_command!r}'
        )
        # The original command's trailing ` tests/ --tb=short -q` (target plus
        # flags positioned after the matched 'pytest' keyword) must be dropped
        # entirely — not merely have its `tests/` target replaced while
        # `--tb=short -q` survives.
        assert executed[0].test_command == 'uv run pytest mymod/tests/test_thing.py', (
            f'trailing flags/targets after the matched keyword must be dropped, '
            f'matching _scope_to_keyword, not preserved: {executed[0].test_command!r}'
        )

        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pytest:', executed[0].test_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_file_scoped_command_with_value_taking_flag_after_target_is_dropped(
        self, tmp_path: Path,
    ):
        """(g) FILE_SCOPED lint command with a value-taking flag AFTER the
        target (``'ruff check src/ --select E'``) -> the whole
        ``'--select E'`` tail is dropped, matching _scope_to_keyword — not
        scoped as though ``'E'`` were an extra target (the latent hazard the
        reviewer flagged: scoping the whole parsed command would replace
        BOTH ``src/`` and ``E`` with the touched file, leaving a dangling
        valueless ``--select``).
        """
        (tmp_path / 'mymod').mkdir(parents=True)
        touched = 'mymod/thing.py'
        (tmp_path / touched).write_text('def helper():\n    return 1\n')

        lint_command = 'ruff check src/ --select E'
        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [ModuleConfig(prefix='mymod', lint_command=lint_command)]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[touched],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        expected = verify._scope_to_keyword(lint_command, 'ruff check', [touched])
        assert executed[0].lint_command == expected, (
            f'expected the _scope_to_keyword-scoped string {expected!r}, got '
            f'{executed[0].lint_command!r}'
        )
        assert '--select' not in (executed[0].lint_command or ''), (
            f'a dangling valueless --select must never survive scoping: '
            f'{executed[0].lint_command!r}'
        )

        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'lint:', executed[0].lint_command, executed[0].prefix,
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


# ---------------------------------------------------------------------------
# step-7: fallback-path plan authority + byte-identical goldens
# ---------------------------------------------------------------------------


def _make_cockpit_worktree(tmp_path: Path) -> None:
    """Write cockpit/pyproject.toml so ``_single_subproject_prefix`` recognises it."""
    cockpit = tmp_path / 'cockpit'
    cockpit.mkdir(parents=True, exist_ok=True)
    (cockpit / 'pyproject.toml').write_text('[project]\nname = "cockpit"\n')


def _plan_run(plan: dict, reason_prefix: str) -> dict:
    """The single ``PlannedRun`` dict in *plan*['runs'] whose reason starts with *reason_prefix*."""
    matches = [r for r in plan['runs'] if r['reason'].startswith(reason_prefix)]
    assert len(matches) == 1, f'expected exactly one {reason_prefix!r} run in {plan!r}'
    return matches[0]


def _render_plan_cmd(cmd_dict: dict) -> str:
    """Render a ``PlannedRun.to_dict()['cmd']`` payload back to a shell string.

    Reconstructs a ``VerifyCmd`` from the plain-dict form ``VerifyResult.plan``
    carries, so a golden can assert the plan's recorded command renders to the
    same string that was actually executed — independent of whether the
    executed plan represents it as a raw pass-through or a fully structured
    ``VerifyCmd``.
    """
    return render(VerifyCmd(
        tool=ToolKind(cmd_dict['tool']),
        uv_project=cmd_dict['uv_project'],
        cwd_rel=cmd_dict['cwd_rel'],
        base_flags=tuple(cmd_dict['base_flags']),
        targets=tuple(cmd_dict['targets']),
        env=dict(cmd_dict['env']),
        wrappers=tuple(cmd_dict['wrappers']),
        raw=cmd_dict['raw'],
    ))


def _assert_plan_run_matches_executed(
    plan: dict, reason_prefix: str, executed_cmd: str | None, executed_prefix: str,
) -> None:
    """Assert *plan*'s *reason_prefix* run faithfully records what was executed.

    A1 for the fallback path: ``VerifyResult.plan`` must be the EXECUTED plan
    — ``module_prefix`` plus a rendered ``cmd`` matching the ACTUAL
    subproject/rescoped command that ran — not an independently re-derived
    diagnostic mirror that ignores subproject/OPAQUE-chain rescoping.
    """
    run = _plan_run(plan, reason_prefix)
    assert run['module_prefix'] == executed_prefix, (
        f'{reason_prefix!r} run recorded module_prefix={run["module_prefix"]!r}, '
        f'expected the EXECUTED prefix {executed_prefix!r}'
    )
    if executed_cmd is None:
        assert run['cmd'] is None, (
            f'{reason_prefix!r} run recorded a cmd but nothing executed: {run["cmd"]!r}'
        )
    else:
        assert run['cmd'] is not None, (
            f'{reason_prefix!r} run recorded no cmd, but {executed_cmd!r} executed'
        )
        rendered = _render_plan_cmd(run['cmd'])
        assert rendered == executed_cmd, (
            f'{reason_prefix!r} plan cmd renders to {rendered!r}, '
            f'expected the EXECUTED command {executed_cmd!r}'
        )


class TestFallbackPlanAuthorityGoldens:
    """A1 + byte-identical goldens for the fallback (no-``module_configs``) branch.

    Mirrors ``TestModuleConfigScopeGoldens`` (module-config path) but for
    ``run_scoped_verification``'s OTHER scoping branch: ``derive_verify_plan``'s
    fallback branch (``_derive_fallback_runs``) is a pure, subproject-blind
    D1/D2 decision record, while ``_build_fallback_config`` (today, still
    independently invoked) additionally applies filesystem-dependent
    RENDERING — subproject cd-scoping (task 2344), mixed root+subproject
    scoping (task 2368), TYPE/LINT uv-context rescoping (task 2355), and
    OPAQUE fleet-chain first-clause scoping (``_scope_to_keyword``) — that the
    fallback plan does not model at all.

    (a)/(b)/(c) below involve no subproject/OPAQUE-chain shape and are
    already GREEN today (regression pins); (d)/(e)/(f) pin the actual gap
    and are RED until step-8 makes ``VerifyResult.plan`` the EXECUTED plan
    on this branch too.
    """

    @pytest.mark.asyncio
    async def test_fallback_conftest_uses_parent_directory(self, tmp_path: Path):
        """(a) ROOT_CONFTEST_DIFF (task-1077) -> test_command == 'pytest <parent-dir>'.

        Already GREEN today: no subproject/OPAQUE-chain shape is involved, so
        the diagnostic plan and the executed command already agree.
        """
        conftest_path = ROOT_CONFTEST_DIFF[0]
        full = tmp_path / conftest_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('# root\n')

        config = OrchestratorConfig(project_root=tmp_path, test_command='pytest')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=[conftest_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command == 'pytest orchestrator/tests', (
            f'expected the conftest parent-dir target, got {executed[0].test_command!r}'
        )
        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pytest:', executed[0].test_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_fallback_root_conftest_maps_to_dot(self, tmp_path: Path):
        """(a) A root-level conftest.py -> test_command == 'pytest .', never 'pytest conftest.py'.

        Already GREEN today (see class docstring).
        """
        (tmp_path / 'conftest.py').write_text('# root conftest\n')

        config = OrchestratorConfig(project_root=tmp_path, test_command='pytest')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=['conftest.py'],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command == 'pytest .'
        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pytest:', executed[0].test_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_bare_pytest_data_module_skips_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        """(b) DATA_MODULE_DIFF on bare pytest -> test_command is None + task-1852 WARNING.

        Already GREEN today (see class docstring).
        """
        data_path = DATA_MODULE_DIFF[0]
        full = tmp_path / data_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('ALLOWLIST = set()\n')

        config = OrchestratorConfig(project_root=tmp_path, test_command='pytest')

        mock_run_verification = _run_verification_spy()
        with (
            patch.object(verify, 'run_verification', new=mock_run_verification),
            caplog.at_level(logging.WARNING, logger='orchestrator.verify'),
        ):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=[data_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].test_command is None
        assert any(
            'test-tree data module' in r.getMessage() and r.levelno >= logging.WARNING
            for r in caplog.records
        ), f'expected the task-1852 WARNING; got: {[r.getMessage() for r in caplog.records]}'
        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pytest:', executed[0].test_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_structural_file_widens_pyright_unscoped(self, tmp_path: Path):
        """(c) STRUCTURAL_FILE_DIFF -> unscoped pyright end-to-end (the D2 gap
        _build_fallback_config closes). Mirrors
        TestRunScopedVerificationPlan.test_fallback_path_closes_d2_gap_end_to_end
        (test_verify.py), which this golden must not regress.

        Already GREEN today (see class docstring).
        """
        struct_path = STRUCTURAL_FILE_DIFF[0]
        full = tmp_path / struct_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(STRUCTURAL_FILE_CONTENT)

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command='pytest', lint_command='ruff check', type_check_command='pyright',
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=[struct_path],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert struct_path not in (executed[0].type_check_command or ''), (
            f'structural file must trigger the unscoped type check: '
            f'{executed[0].type_check_command!r}'
        )
        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pyright:', executed[0].type_check_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_single_subproject_diff_scopes_to_cockpit(self, tmp_path: Path):
        """(d) FALLBACK_SUBPROJECT_DIFF (cockpit-shaped, tasks 2344/2355) -> TEST scoped
        to cockpit alone, TYPE/LINT rescoped into cockpit's own uv context.

        RED today: the fallback plan (_derive_fallback_runs) does not model
        subproject rescoping at all — it records the fleet chain running
        verbatim (TEST) / unscoped (LINT/TYPE) against the flat file list,
        while execution actually narrows everything to cockpit alone.
        """
        _make_cockpit_worktree(tmp_path)
        test_path = FALLBACK_SUBPROJECT_DIFF[0]
        full = tmp_path / test_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def test_x():\n    pass\n')

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command=_FLEET_TEST_COMMAND,
            lint_command=_FLEET_LINT_COMMAND,
            type_check_command=_FLEET_TYPE_COMMAND,
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=FALLBACK_SUBPROJECT_DIFF,
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].prefix == 'cockpit'
        assert executed[0].test_command == 'cd cockpit && uv run pytest tests/test_c3.py'
        assert executed[0].lint_command == (
            'uv run --project cockpit ruff check cockpit/tests/test_c3.py'
        )
        assert executed[0].type_check_command == (
            'uv run --project cockpit npx pyright cockpit/tests/test_c3.py'
        )
        # False-block guard: no OTHER fleet subproject leaks into the scoped commands.
        for other in ('shared', 'escalation', 'fused-memory', 'dashboard'):
            assert other not in executed[0].test_command
            assert other not in executed[0].type_check_command

        assert result.plan is not None
        for prefix, cmd in (
            ('pytest:', executed[0].test_command),
            ('lint:', executed[0].lint_command),
            ('pyright:', executed[0].type_check_command),
        ):
            _assert_plan_run_matches_executed(result.plan, prefix, cmd, executed[0].prefix)

    @pytest.mark.asyncio
    async def test_mixed_root_and_subproject_scopes_narrowly(self, tmp_path: Path):
        """(e) MIXED_ROOT_SUBPROJECT_DIFF (task 2368) -> TEST scoped to cockpit's own
        tests plus the root-owning tests/scripts/ suite, never the fleet chain.

        RED today: the fallback plan does not model mixed root+subproject
        rescoping either — same gap as (d), different shape.
        """
        _make_cockpit_worktree(tmp_path)
        for f in MIXED_ROOT_SUBPROJECT_DIFF:
            full = tmp_path / f
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text('def test_x():\n    pass\n' if 'test_' in f else '# conftest\n')

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command=_FLEET_TEST_COMMAND,
            lint_command=_FLEET_LINT_COMMAND,
            type_check_command=_FLEET_TYPE_COMMAND,
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=MIXED_ROOT_SUBPROJECT_DIFF,
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].prefix == 'cockpit'
        assert executed[0].test_command == (
            'cd cockpit && uv run pytest tests/test_c3.py '
            '&& cd .. && uv run --project shared pytest tests/scripts/'
        )
        for other in ('escalation', 'fused-memory', 'dashboard'):
            assert f'cd ../{other}' not in executed[0].test_command

        assert result.plan is not None
        for prefix, cmd in (
            ('pytest:', executed[0].test_command),
            ('lint:', executed[0].lint_command),
            ('pyright:', executed[0].type_check_command),
        ):
            _assert_plan_run_matches_executed(result.plan, prefix, cmd, executed[0].prefix)

    @pytest.mark.asyncio
    async def test_opaque_fleet_chain_lint_type_scoped_to_first_clause(self, tmp_path: Path):
        """(f) UNREGISTERED_PATH_DIFF against the REAL OPAQUE fleet lint/type chains ->
        LINT/TYPE scope to their first clause (``_scope_to_keyword``); the
        OPAQUE TEST chain runs verbatim (P1).

        RED today for LINT/TYPE: the fallback plan records the WHOLE
        untouched multi-clause chain (parses OPAQUE at the full-string
        level), while execution truncates-then-parses the first clause only,
        producing a completely different, file-scoped string. TEST is
        already GREEN today — P1 means neither layer scopes it.

        The two chains diverge on what happens to the TAIL (task 3061). LINT
        is a SIBLING-CHECKER chain, so its trailing
        ``check_bare_magicmock_config.py`` clause is preserved unscoped and
        verbatim. TYPE is a ``cd``-sequenced SAME-TOOL FAN-OUT, so its tail
        is still dropped — preserving it would run two more subprojects
        unscoped and leave a ``cd ../orchestrator`` that misresolves once
        ``strip_cwd`` has removed the leading ``cd``.

        LINT also exercises the fallback path's reprojection end-to-end: the
        scoped head is a bare ``uv run``, and ``_reproject_str`` must still
        inject ``--project shared`` into it despite the appended tail (task
        2036 — the depless workspace-root project cannot spawn ruff).
        """
        test_path = UNREGISTERED_PATH_DIFF[0]
        full = tmp_path / test_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def test_x():\n    pass\n')

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command=_FLEET_TEST_COMMAND,
            lint_command=_FLEET_LINT_COMMAND_OPAQUE,
            type_check_command=_FLEET_TYPE_COMMAND,
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=UNREGISTERED_PATH_DIFF,
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert executed[0].prefix == '__fallback__'
        # P1: the OPAQUE fleet TEST chain is never scoped/mutated — verbatim.
        assert executed[0].test_command == _FLEET_TEST_COMMAND
        # LINT: ruff is file-scoped AND reprojected into the fallback uv
        # project, while the trailing sibling-checker clause survives verbatim.
        assert executed[0].lint_command == (
            f'uv run --project shared ruff check {test_path}'
            ' && python3 fused-memory/scripts/check_bare_magicmock_config.py '
            'shared/tests escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
        )
        assert 'check_bare_magicmock_config' in (executed[0].lint_command or '')
        # TYPE: a cd-sequenced same-tool fan-out still truncates at the keyword.
        assert executed[0].type_check_command == f'npx pyright {test_path}'
        assert 'orchestrator' not in (executed[0].type_check_command or '')
        assert 'dashboard' not in (executed[0].type_check_command or '')

        assert result.plan is not None
        for prefix, cmd in (
            ('pytest:', executed[0].test_command),
            ('lint:', executed[0].lint_command),
            ('pyright:', executed[0].type_check_command),
        ):
            _assert_plan_run_matches_executed(result.plan, prefix, cmd, executed[0].prefix)


# ---------------------------------------------------------------------------
# step-9/step-10: preservation of the non-scoping branches
# ---------------------------------------------------------------------------


class TestNonScopingBranchesPreserved:
    """Guards pinning that the derive→execute rewrite (task κ) leaves the
    NON-scoping branches of ``run_scoped_verification`` untouched: the
    docs-only TRIVIAL short-circuit (+ its merge-role pipeline-guard
    override, at both call sites), ``force_workspace``, and the
    scoped-fallback-never-global-fanout boundary row
    (verify-scope-inversion-prd.md boundary row 9). None of these branches
    are driven by the plan->execution bridge (steps 1-8) — they sit
    strictly outside it — so a regression here would mean the rewrite
    widened its own footprint into territory task κ was never meant to
    touch.
    """

    # -- (a) docs-only TRIVIAL short-circuit, both branches, both roles --

    @pytest.mark.asyncio
    @pytest.mark.parametrize('role', ['task', 'merge'])
    async def test_docs_only_module_configs_branch_trivially_passes(
        self, tmp_path: Path, role: Literal['task', 'merge'],
    ):
        """(a) .md-only diff, module_configs branch -> TRIVIAL, zero
        run_verification calls, at BOTH role='task' and role='merge' (no
        guard script present, so the merge-role override never fires).
        """
        (tmp_path / 'skills').mkdir()
        (tmp_path / 'skills' / 'foo.md').write_text('# foo\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='mymod', test_command='uv run --directory mymod pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=['skills/foo.md'], role=role,
            )

        assert result.passed
        assert 'No source files' in result.summary
        assert mock_run_verification.await_count == 0, (
            f'docs-only diff must execute zero commands (role={role!r}); '
            f'got {mock_run_verification.await_count} call(s)'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('role', ['task', 'merge'])
    async def test_docs_only_no_module_configs_branch_trivially_passes(
        self, tmp_path: Path, role: Literal['task', 'merge'],
    ):
        """(a) .md-only diff, no-module_configs branch — the mirror of the
        module_configs short-circuit above (verify.py's "Mirror the same
        docs-only short-circuit as the module_configs branch" comment) ->
        TRIVIAL, zero run_verification calls, at BOTH role='task' and
        role='merge'.
        """
        (tmp_path / 'skills').mkdir()
        (tmp_path / 'skills' / 'foo.md').write_text('# foo\n')

        config = OrchestratorConfig(project_root=tmp_path)

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=['skills/foo.md'], role=role,
            )

        assert result.passed
        assert 'No source files' in result.summary
        assert mock_run_verification.await_count == 0, (
            f'docs-only diff must execute zero commands (role={role!r}); '
            f'got {mock_run_verification.await_count} call(s)'
        )

    # -- (b) merge-role pipeline-guard full-gate override, both call sites --

    @pytest.mark.asyncio
    async def test_pipeline_guard_override_fires_module_configs_branch(self, tmp_path: Path):
        """(b) role='merge', config-only diff, guard exits 0 -> full gate
        override still fires at the module_configs call site (mirrors
        TestMergeGuardModuleConfigs.test_guard_exit_0_merge_overrides_trivial_pass,
        test_verify.py — kept green by this task, pinned again here through
        the new module's spy helpers).
        """
        (tmp_path / 'orchestrator').mkdir(parents=True)
        (tmp_path / 'orchestrator' / 'config.yaml').write_text('foo: bar\n')
        _write_guard_script(tmp_path, exit_code=0)

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [ModuleConfig(prefix='orchestrator', test_command='__orch_cmd__')]

        calls, fake_run_cmd = _run_cmd_spy()
        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_scoped_verification(
                tmp_path, config, module_configs,
                task_files=['orchestrator/config.yaml'], role='merge',
            )

        assert result.passed
        assert 'No source files' not in result.summary, (
            f'expected the trivial pass to be overridden; got: {result.summary!r}'
        )
        assert '__orch_cmd__' in ' | '.join(calls), (
            f'expected the per-subproject fan-out to run after the override; calls={calls!r}'
        )

    @pytest.mark.asyncio
    async def test_pipeline_guard_override_fires_no_module_configs_branch(self, tmp_path: Path):
        """(b) role='merge', config-only diff, guard exits 0 -> full gate
        override still fires at the no-module_configs call site (mirrors
        TestMergeGuardNoModuleConfigs.test_guard_exit_0_merge_overrides_trivial_pass,
        test_verify.py).
        """
        (tmp_path / 'scripts').mkdir(parents=True)
        (tmp_path / 'scripts' / 'verify.sh').write_text('#!/usr/bin/env bash\n')
        _write_guard_script(tmp_path, exit_code=0)

        config = OrchestratorConfig(project_root=tmp_path, test_command='__scope_all_cmd__')

        calls, fake_run_cmd = _run_cmd_spy()
        with patch('orchestrator.verify._run_cmd', side_effect=fake_run_cmd):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=['scripts/verify.sh'], role='merge',
            )

        assert result.passed
        assert 'No source files' not in result.summary, (
            f'expected the trivial pass to be overridden; got: {result.summary!r}'
        )
        assert '__scope_all_cmd__' in ' | '.join(calls), (
            f'expected the full-gate command to run after the override; calls={calls!r}'
        )

    # -- (c) force_workspace bypasses scoping AND plan derivation --

    @pytest.mark.asyncio
    async def test_force_workspace_bypasses_scoping_and_plan(self, tmp_path: Path):
        """(c) force_workspace=True -> the global run_verification call runs
        (scoping bypassed entirely — no ModuleConfig is built from the plan)
        and VerifyResult.plan is never derived (mirrors
        TestRunScopedVerificationForceWorkspace, test_verify.py).
        """
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        (tmp_path / 'mymod' / 'tests' / 'test_thing.py').write_text('def test_thing(): pass\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='mymod', test_command='uv run --directory mymod pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs,
                task_files=['mymod/tests/test_thing.py'], force_workspace=True,
            )

        assert result.passed
        assert mock_run_verification.await_count == 1
        assert _executed_module_configs(mock_run_verification) == [], (
            'force_workspace must bypass ALL scoping — no ModuleConfig should '
            'be built from the plan'
        )
        assert result.plan is None, (
            'force_workspace bypasses derive_verify_plan entirely — '
            'VerifyResult.plan must stay unset'
        )

    # -- (d) unregistered-path diff never falls through to the whole-repo fan-out --

    @pytest.mark.asyncio
    async def test_unregistered_path_diff_never_falls_through_to_global_fanout(
        self, tmp_path: Path,
    ):
        """(d) UNREGISTERED_PATH_DIFF, role='task' -> scoped commands only;
        the whole-repo fan-out chain never appears in the executed commands
        (verify-scope-inversion-prd.md boundary row 9, "the wall-clock win",
        asserted structurally: exactly ONE scoped ModuleConfig executes, not
        a per-fleet-subproject fan-out and not the unscoped global command).
        """
        test_path = UNREGISTERED_PATH_DIFF[0]
        full = tmp_path / test_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def test_x():\n    pass\n')

        config = OrchestratorConfig(
            project_root=tmp_path,
            test_command=_FLEET_TEST_COMMAND,
            lint_command=_FLEET_LINT_COMMAND,
            type_check_command=_FLEET_TYPE_COMMAND,
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [], task_files=UNREGISTERED_PATH_DIFF,
                # role='task' is the default
            )

        assert result.passed
        assert mock_run_verification.await_count == 1, (
            f'expected exactly one scoped call, not a per-subproject fan-out '
            f'or an extra global fallthrough; got {mock_run_verification.await_count}'
        )
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1, (
            f'expected exactly one scoped ModuleConfig, not a per-subproject '
            f'fan-out: {executed!r}'
        )
        assert executed[0].prefix == '__fallback__', (
            f'the diff must route through the plan-driven fallback branch, '
            f'not the unscoped global branch: {executed[0].prefix!r}'
        )
        for other in ('escalation', 'fused-memory', 'dashboard'):
            assert other not in (executed[0].lint_command or ''), (
                f'whole-repo fleet chain leaked into lint_command: '
                f'{executed[0].lint_command!r}'
            )
            assert other not in (executed[0].type_check_command or ''), (
                f'whole-repo fleet chain leaked into type_check_command: '
                f'{executed[0].type_check_command!r}'
            )
        assert result.plan is not None


# ---------------------------------------------------------------------------
# step-11/step-12: VerifyResult.plan folds execution-time module skips
# ---------------------------------------------------------------------------


class TestExecutedPlanRecordsModuleSkips:
    """A1, multi-module shape (module-config branch): ``VerifyResult.plan``
    is the plan that actually drove the gathered ``run_verification`` calls
    for a run spanning SEVERAL registered modules, not a re-derivation that
    could diverge from execution — and a registered module skipped at
    execution time (no files under its prefix, so it is dropped from
    ``scoped`` by :func:`verify._executed_module_configs_from_plan`) is
    recorded as an explicit ``SKIPPED`` ``PlannedRun`` with a reason, never
    silently absent from the record.
    """

    @pytest.mark.asyncio
    async def test_untouched_module_skip_is_recorded_with_reason(self, tmp_path: Path):
        """Two registered modules, only one touched -> the untouched one is
        dropped from execution but still appears in result.plan as an
        explicit SKIPPED run (reason: 'no files under prefix')."""
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        (tmp_path / 'mymod' / 'tests' / 'test_thing.py').write_text('def test_thing(): pass\n')
        (tmp_path / 'othermod').mkdir(parents=True)

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='mymod', test_command='uv run --directory mymod pytest tests/'),
            ModuleConfig(prefix='othermod', test_command='uv run --directory othermod pytest tests/'),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs,
                task_files=['mymod/tests/test_thing.py'],
            )

        assert result.passed
        # Only the touched module actually ran — othermod is dropped from
        # execution entirely (verify._executed_module_configs_from_plan's
        # "no files under prefix" collapse).
        executed = _executed_module_configs(mock_run_verification)
        assert [mc.prefix for mc in executed] == ['mymod']

        assert result.plan is not None
        othermod_runs = [r for r in result.plan['runs'] if r['module_prefix'] == 'othermod']
        assert len(othermod_runs) == 1, (
            f'the untouched, dropped-from-execution module must still appear '
            f'exactly once in the record, as a single explicit skip: '
            f'{result.plan["runs"]!r}'
        )
        assert othermod_runs[0]['scope_kind'] == 'skipped'
        assert othermod_runs[0]['cmd'] is None
        assert othermod_runs[0]['reason'], 'a skip must carry a non-empty reason, never a silent drop'

    @pytest.mark.asyncio
    async def test_plan_equals_the_plan_that_drove_every_executed_module(self, tmp_path: Path):
        """Multi-module run, BOTH modules touched with DIFFERENT shapes (one
        FULL_SUITE via conftest, one FILE_SCOPED) -> every pytest PlannedRun
        in result.plan matches the corresponding executed ModuleConfig's
        test_command exactly — the attached plan is the plan that actually
        drove the gathered run_verification calls, not an independent
        re-derivation that happens to agree only by coincidence.
        """
        (tmp_path / 'alpha' / 'tests').mkdir(parents=True)
        (tmp_path / 'alpha' / 'tests' / 'conftest.py').write_text('# conftest\n')
        (tmp_path / 'beta' / 'tests').mkdir(parents=True)
        (tmp_path / 'beta' / 'tests' / 'test_thing.py').write_text('def test_thing(): pass\n')

        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='alpha', test_command='uv run --directory alpha pytest tests/'),
            ModuleConfig(prefix='beta', test_command='uv run --directory beta pytest tests/'),
        ]
        task_files = ['alpha/tests/conftest.py', 'beta/tests/test_thing.py']

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=task_files,
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        executed_by_prefix = {mc.prefix: mc for mc in executed}
        assert set(executed_by_prefix) == {'alpha', 'beta'}

        assert result.plan is not None
        pytest_runs = [r for r in result.plan['runs'] if r['reason'].startswith('pytest:')]
        assert {r['module_prefix'] for r in pytest_runs} == {'alpha', 'beta'}
        for run in pytest_runs:
            executed_mc = executed_by_prefix[run['module_prefix']]
            assert run['cmd'] is not None, f'both modules touched a file; nothing should be skipped: {run!r}'
            if run['scope_kind'] == 'full_suite':
                # FULL_SUITE (alpha's conftest) keeps the ORIGINAL mc's
                # verbatim command — never render(run['cmd']), which would
                # normalize e.g. `--directory` to a `cd` prefix.
                original = next(mc for mc in module_configs if mc.prefix == run['module_prefix'])
                assert executed_mc.test_command == original.test_command
            else:
                assert run['scope_kind'] == 'file_scoped'
                assert executed_mc.test_command == _render_plan_cmd(run['cmd'])


# ---------------------------------------------------------------------------
# reviewer_comprehensive remediation (verify.py:3660, behavior_change): pin
# the dataclasses.replace field-preservation as intended.
# ---------------------------------------------------------------------------


class TestExecutedModuleConfigsPreservesPerModuleFields:
    """``_executed_module_configs_from_plan`` builds the executed ``ModuleConfig``
    via ``dataclasses.replace(mc, ...)`` rather than the deleted
    ``scope_module_config``'s hand-listed ``ModuleConfig(prefix=..., ...)``
    reconstruction, which OMITTED (reset to defaults) every field other than
    the three commands and ``lock_depth``/``max_per_module``/
    ``module_overrides``. This means per-module timeout knobs, ``verify_env``,
    and ``scope_cargo`` now survive onto the scoped config where they
    previously did not — documented as intentional in this function's own
    docstring (and arguably a latent-bug fix for ``scope_cargo``, since
    ``_apply_cargo_scope`` now reads the real value instead of a reset
    default) but previously unpinned by any dedicated regression test.
    """

    def test_per_module_timeout_env_and_scope_cargo_survive_onto_executed_config(self):
        mc = ModuleConfig(
            prefix='mymod',
            test_command='uv run --directory mymod pytest tests/',
            verify_command_timeout_secs=123.0,
            verify_cold_command_timeout_secs=456.0,
            verify_env={'MYVAR': 'myvalue'},
            scope_cargo=True,
        )
        touched = 'mymod/tests/test_thing.py'
        plan = verify_plan.derive_verify_plan([touched], [mc], None, lambda _path: None)
        executed = verify._executed_module_configs_from_plan([mc], plan)

        assert len(executed) == 1
        executed_mc = executed[0]
        assert executed_mc.verify_command_timeout_secs == 123.0, (
            'a per-module verify_command_timeout_secs override must survive onto '
            'the executed config, not reset to the config default'
        )
        assert executed_mc.verify_cold_command_timeout_secs == 456.0, (
            'a per-module verify_cold_command_timeout_secs override must survive '
            'onto the executed config, not reset to the config default'
        )
        assert executed_mc.verify_env == {'MYVAR': 'myvalue'}, (
            'a per-module verify_env override must survive onto the executed config'
        )
        assert executed_mc.scope_cargo is True, (
            'scope_cargo must survive onto the executed config so _apply_cargo_scope '
            'reads the real value, not a reset default'
        )


# ---------------------------------------------------------------------------
# step-14/step-15: module-config OPAQUE &&-chain scoping (reviewer_comprehensive
# remediation, verify_plan.py:253)
# ---------------------------------------------------------------------------


class TestModuleConfigOpaqueChainScoping:
    """Every ``TestModuleConfigScopeGoldens``/``TestModuleConfigPlanAuthority``
    golden uses a single-clause command, but a REAL subproject
    ``lint_command``/``type_check_command`` is an ``&&``-chain that parses
    OPAQUE (or, for a chained ``pytest``, a recognised-but-unstructurable
    chain) — a shape none of those goldens exercise. Before this class, that
    left the module-config path's FILE_SCOPED derivation silently UNPINNED
    for the one command shape that actually occurs in production, so a
    regression (the whole unscoped chain, INCLUDING any trailing clause,
    executing verbatim instead of being file-scoped to the touched file)
    slipped through.

    Each golden below asserts the executed command is FIRST-CLAUSE scoped
    EXACTLY as the deleted ``scope_module_config`` twin produced it — derived
    independently via ``verify._scope_to_keyword`` (the exact helper the old
    twin used, and the fallback path still uses today — see golden (f) of
    ``TestFallbackPlanAuthorityGoldens``) — not merely self-consistency with
    the plan.

    RED today: ``_derive_module_runs`` stores the raw-retained chain
    unchanged (``strip_cwd(scope_to(parse_config_command(x), files))``
    no-ops on an OPAQUE/raw-retained command — see ``verify_cmd.scope_to``'s
    P1 guard) and ``_executed_module_configs_from_plan`` renders it verbatim
    (``render()==raw``), so the whole unscoped chain executes instead of a
    file-scoped first clause.

    Task 3061 refined what "first-clause scoped" means for the TAIL. A
    SIBLING-CHECKER clause (a different tool, no ``cd`` sequencing — the
    ``python3 .../check_*.py <dir>`` gates every subproject chains after
    ``ruff check``) is now PRESERVED unscoped and verbatim: it asserts a
    whole-directory invariant, so dropping it made the gate invisible to
    scoped pre-merge verify. A SAME-TOOL FAN-OUT (the root config's ``cd X
    && npx pyright`` chain) keeps being truncated — see
    ``verify_cmd.split_chain_tail``'s gate.
    """

    @pytest.mark.asyncio
    async def test_lint_real_subproject_chain_scopes_to_first_clause(self, tmp_path: Path):
        """(1) LINT — the real subproject shape (config.yaml-style): an
        OPAQUE ``ruff check`` clause followed by a
        ``check_bare_magicmock_config.py`` sibling-checker clause -> the ruff
        clause is file-scoped (its unscoped ``src/ tests/`` targets dropped)
        while the trailing checker is PRESERVED unscoped and verbatim
        (task 3061).
        """
        (tmp_path / 'escalation' / 'src').mkdir(parents=True)
        touched = 'escalation/src/thing.py'
        (tmp_path / touched).write_text('def helper():\n    return 1\n')

        lint_command = (
            'uv run --project escalation --directory escalation ruff check src/ tests/ '
            '&& python3 fused-memory/scripts/check_bare_magicmock_config.py escalation/tests'
        )
        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [ModuleConfig(prefix='escalation', lint_command=lint_command)]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[touched],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        expected = verify._scope_to_keyword(lint_command, 'ruff check', [touched])
        assert executed[0].lint_command == expected, (
            f'expected the first-clause-scoped string {expected!r}, got '
            f'{executed[0].lint_command!r}'
        )
        assert 'check_bare_magicmock_config' in (executed[0].lint_command or ''), (
            f'the trailing sibling-checker clause must be PRESERVED (unscoped and '
            f'verbatim — it asserts a whole-directory invariant) while the ruff '
            f'clause is file-scoped: {executed[0].lint_command!r}'
        )
        assert 'src/ tests/' not in (executed[0].lint_command or ''), (
            f'the unscoped src/ tests/ targets must not survive scoping: '
            f'{executed[0].lint_command!r}'
        )

        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'lint:', executed[0].lint_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_fused_memory_real_lint_chain_runs_asyncmock_checker(self, tmp_path: Path):
        """(1b) TASK-2920 REPRODUCTION — the checker that caught it post-merge now runs.

        Task 2920's asyncmock-assertion violation landed on main and was only
        caught by the post-merge full-config run, because the scoped
        pre-merge lint silently truncated fused-memory's REAL 3-segment
        ``lint_command`` at ``ruff check`` and dropped both sibling gates.
        With the trailing clauses preserved, ``check_asyncmock_assertion_style.py``
        executes on the pre-merge scoped path — i.e. the same violation would
        now be caught before the merge, not after.
        """
        (tmp_path / 'fused-memory' / 'tests').mkdir(parents=True)
        touched = 'fused-memory/tests/test_harness.py'
        (tmp_path / touched).write_text('def test_harness(): pass\n')

        # Verbatim fused-memory/orchestrator.yaml:11.
        lint_command = (
            'uv run --project fused-memory --directory fused-memory ruff check src/ tests/ '
            '&& python3 fused-memory/scripts/check_bare_magicmock_config.py fused-memory/tests '
            '&& python3 fused-memory/scripts/check_asyncmock_assertion_style.py fused-memory/tests'
        )
        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [ModuleConfig(prefix='fused-memory', lint_command=lint_command)]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[touched],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        assert 'check_asyncmock_assertion_style.py' in (executed[0].lint_command or ''), (
            f'task 2920 acceptance: the asyncmock-assertion gate must run on the '
            f'scoped pre-merge path, not only post-merge: {executed[0].lint_command!r}'
        )
        assert 'check_bare_magicmock_config.py' in (executed[0].lint_command or '')
        assert 'src/ tests/' not in (executed[0].lint_command or ''), (
            f'the ruff clause itself must still be file-scoped: '
            f'{executed[0].lint_command!r}'
        )
        assert touched in (executed[0].lint_command or '')

        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'lint:', executed[0].lint_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_type_non_structural_chain_scopes_to_first_clause(self, tmp_path: Path):
        """(2) TYPE — a non-structural diff against a genuine multi-clause
        ``npx pyright`` chain -> first-clause scoped. A plain
        (non-Protocol/TypedDict) source file, so D2 does not widen this to
        FULL_SUITE — this pins the FILE_SCOPED pyright branch specifically.
        """
        tmp_path.joinpath('mymod').mkdir(parents=True)
        touched = 'mymod/helpers.py'
        (tmp_path / touched).write_text('def helper():\n    return 1\n')

        type_check_command = (
            'cd fused-memory && npx pyright && cd ../orchestrator && npx pyright'
        )
        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [
            ModuleConfig(prefix='mymod', type_check_command=type_check_command),
        ]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[touched],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        expected = verify._scope_to_keyword(type_check_command, 'pyright', [touched])
        assert executed[0].type_check_command == expected, (
            f'expected the first-clause-scoped string {expected!r}, got '
            f'{executed[0].type_check_command!r}'
        )

        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pyright:', executed[0].type_check_command, executed[0].prefix,
        )

    @pytest.mark.asyncio
    async def test_pytest_chained_command_scopes_to_first_clause(self, tmp_path: Path):
        """(3) TEST — a touched collectable test file against a genuine
        multi-clause pytest chain (recognised-but-unstructurable — see
        ``verify_cmd._parse_chain``) -> first-clause scoped.
        """
        (tmp_path / 'mymod' / 'tests').mkdir(parents=True)
        touched = 'mymod/tests/test_thing.py'
        (tmp_path / touched).write_text('def test_thing(): pass\n')

        test_command = 'cd a && uv run pytest tests/ && cd b && uv run pytest tests/'
        config = OrchestratorConfig(project_root=tmp_path)
        module_configs = [ModuleConfig(prefix='mymod', test_command=test_command)]

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, module_configs, task_files=[touched],
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        expected = verify._scope_to_keyword(test_command, 'pytest', [touched])
        assert executed[0].test_command == expected, (
            f'expected the first-clause-scoped string {expected!r}, got '
            f'{executed[0].test_command!r}'
        )

        assert result.plan is not None
        _assert_plan_run_matches_executed(
            result.plan, 'pytest:', executed[0].test_command, executed[0].prefix,
        )
