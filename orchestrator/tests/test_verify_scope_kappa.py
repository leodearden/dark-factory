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
        LINT/TYPE scope to their first clause (``_scope_to_keyword``, dropping the
        rest); the OPAQUE TEST chain runs verbatim (P1).

        RED today for LINT/TYPE: the fallback plan records the WHOLE
        untouched multi-clause chain (parses OPAQUE at the full-string
        level), while execution truncates-then-parses the first clause only,
        producing a completely different, file-scoped string. TEST is
        already GREEN today — P1 means neither layer scopes it.
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
        # LINT/TYPE scope to the first clause, dropping the rest of the chain.
        assert executed[0].lint_command == f'uv run --project shared ruff check {test_path}'
        assert 'check_bare_magicmock_config' not in (executed[0].lint_command or '')
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
