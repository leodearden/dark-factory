"""Tests for orchestrator.verify's role-differentiated scope policy (task λ, 2589).

verify-scope-inversion PRD task λ (plans/verify-scope-inversion-prd.md): ``role``
becomes the policy fork ``derive_verify_plan`` always should have had. This
module pins the END-TO-END ``run_scoped_verification`` wiring for the two
behaviour changes λ delivers on top of task κ's plan-authoritative execution:

- The task-role pytest floor (R3, step-4): a source-only diff under a
  registered module now runs that module's full ``test_command`` at
  role='task' instead of the pre-λ zero-pytest SKIPPED.
- The broad merge gate (R1, steps 5/6/8): role='merge' +
  ``merge_verify_breadth='full'`` expands execution to EVERY registered
  module (``config.module_configs_or_empty``), not just the modules the
  diff's own task/train touches — closing the "only the touched modules are
  protected" gap the task-role floor deliberately leaves open (R1: the floor
  never widens beyond owning modules; only this knob-gated gate does).

``test_verify_plan.py``'s ``TestDeriveVerifyPlanMergeBreadth`` pins the same
policy fork at the pure ``derive_verify_plan`` layer; this module pins it
through the full derive-then-EXECUTE pipeline, where a module untouched by
the diff must actually be expanded from the registry before it can execute
at all (``run_scoped_verification``'s job, not ``derive_verify_plan``'s —
see verify_plan.py's design-decision docstring on the module-config branch).

Spy helpers (``_run_verification_spy`` / ``_executed_module_configs``) are
reused directly from ``test_verify_scope_kappa.py`` rather than
re-implemented, so both suites share one provenance-pinned execution-spy
implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import yaml
from test_verify_scope_kappa import _executed_module_configs, _run_verification_spy

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_plan import ScopeKind, derive_verify_plan, parse_config_command

# ---------------------------------------------------------------------------
# step-7: role x merge_verify_breadth end-to-end execution goldens (RED)
# ---------------------------------------------------------------------------


def _two_module_registry(
    tmp_path: Path, *, breadth: Literal['scoped', 'full'] = 'full',
) -> tuple[ModuleConfig, ModuleConfig, OrchestratorConfig]:
    """A 2-module registry (modA/modB), ``merge_verify_breadth=`` *breadth*.

    Mirrors a real dark_factory ``config.module_configs_or_empty`` registry —
    modB is NEVER passed to ``run_scoped_verification`` as part of the
    *module_configs* argument, only discoverable via the registry, so a test
    that observes modB execute is exercising the merge+full expansion
    (step-8), not merely echoing its own *module_configs* argument back.

    *breadth* defaults to ``'full'`` (this module's usual shape, unchanged
    for every pre-existing call site); pass ``'scoped'`` to build the SAME
    2-module registry for a legacy-byte-identical rollback golden (R4).
    """
    mod_a = ModuleConfig(
        prefix='moda',
        test_command='uv run --directory moda pytest tests/',
        lint_command='uv run --directory moda ruff check src/',
        type_check_command='uv run --directory moda pyright src/',
    )
    mod_b = ModuleConfig(
        prefix='modb',
        test_command='uv run --directory modb pytest tests/',
        lint_command='uv run --directory modb ruff check src/',
        type_check_command='uv run --directory modb pyright src/',
    )
    config = OrchestratorConfig(project_root=tmp_path, merge_verify_breadth=breadth)
    config._module_configs = {'moda': mod_a, 'modb': mod_b}
    return mod_a, mod_b, config


class TestRoleBreadthExecutionGoldens:
    """End-to-end ``run_scoped_verification`` execution, role x ``merge_verify_breadth``.

    modB is registered (``config._module_configs``) but never touched by the
    diff and never passed as part of the *module_configs* argument — only
    discoverable via the registry — so these goldens exercise the FULL
    plan-derive-then-EXECUTE pipeline, not just ``derive_verify_plan`` in
    isolation (that's ``TestDeriveVerifyPlanMergeBreadth``, test_verify_plan.py).
    """

    @pytest.mark.asyncio
    async def test_merge_role_full_breadth_executes_every_registered_module(
        self, tmp_path: Path,
    ):
        """(a) role='merge': BOTH modA (touched) AND modB (untouched,
        registry-only) execute, each with its OWN verbatim full-suite
        commands — the closed source-only/untouched-module hole (boundary
        row 1 shape)."""
        mod_a, mod_b, config = _two_module_registry(tmp_path)
        source_path = 'moda/helpers.py'
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], task_files=[source_path], role='merge',
            )

        assert result.passed
        executed = {mc.prefix: mc for mc in _executed_module_configs(mock_run_verification)}
        assert set(executed) == {'moda', 'modb'}, (
            f'expected BOTH registered modules to execute under merge+full; '
            f'got {set(executed)!r}'
        )
        assert executed['moda'].test_command == mod_a.test_command
        assert executed['moda'].lint_command == mod_a.lint_command
        assert executed['moda'].type_check_command == mod_a.type_check_command
        assert executed['modb'].test_command == mod_b.test_command
        assert executed['modb'].lint_command == mod_b.lint_command
        assert executed['modb'].type_check_command == mod_b.type_check_command

    @pytest.mark.asyncio
    async def test_task_role_same_diff_executes_only_the_touched_module(self, tmp_path: Path):
        """(b) role='task', the SAME diff: ONLY modA executes (the task-role
        floor, R3) — modB never runs even though it's in the registry (R1:
        the floor never widens beyond owning modules). Control — already
        green pre-step-8; pins that the merge+full expansion never leaks
        into role='task'."""
        mod_a, _mod_b, config = _two_module_registry(tmp_path)
        source_path = 'moda/helpers.py'
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], task_files=[source_path], role='task',
            )

        assert result.passed
        executed = {mc.prefix: mc for mc in _executed_module_configs(mock_run_verification)}
        assert set(executed) == {'moda'}, (
            f'expected ONLY the touched module to execute at task role; got {set(executed)!r}'
        )
        assert executed['moda'].test_command == mod_a.test_command, (
            f"expected the task-role floor to full-suite the owning module's pytest: "
            f'{executed["moda"].test_command!r}'
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize('role', ['task', 'merge'])
    async def test_docs_only_diff_zero_run_verification_calls(
        self, tmp_path: Path, role: Literal['task', 'merge'],
    ):
        """(c) A docs-only diff stays TRIVIAL at BOTH roles — zero
        run_verification calls, breadth-independent (R2). Control — already
        green pre-step-8."""
        mod_a, _mod_b, config = _two_module_registry(tmp_path)
        (tmp_path / 'skills').mkdir()
        (tmp_path / 'skills' / 'foo.md').write_text('# foo\n')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], task_files=['skills/foo.md'], role=role,
            )

        assert result.passed
        assert 'No source files' in result.summary
        assert mock_run_verification.await_count == 0, (
            f'docs-only diff must execute zero commands (role={role!r}); '
            f'got {mock_run_verification.await_count} call(s)'
        )


# ---------------------------------------------------------------------------
# step-9: force_workspace x merge_verify_breadth + train routing goldens (RED, T1)
# ---------------------------------------------------------------------------


class TestForceWorkspaceBreadthExecutionGoldens:
    """``force_workspace`` x ``merge_verify_breadth`` execution goldens (T1).

    ``force_workspace`` bypasses ALL file-scoping (see
    ``run_scoped_verification``'s "Workspace (train-member override)"
    docstring paragraph) — ``merge_verify_breadth`` forks WHAT that
    bypassed-scoping path executes, not WHETHER it executes: breadth='full'
    replaces the single OPAQUE global ``&&``-chain with a per-module
    full-suite fan-out across every REGISTERED module (mirroring the
    module_configs-branch merge+full expansion ``TestRoleBreadthExecutionGoldens``
    pins above); breadth='scoped' (the shipped default) stays byte-identical
    to the pre-λ legacy single global call (R4).

    This is the ``merge_verify_workspace=True`` routing
    ``verify_runner.LocalRunner._run`` threads ``role='merge'`` into — the
    production DF merge gate leaves ``merge_verify_workspace`` at its
    default ``False`` (module-config ``role='merge'`` branch instead), so
    this fork exercises a currently-off-by-default but supported routing.
    """

    @pytest.mark.asyncio
    async def test_force_workspace_merge_full_executes_every_registered_module_per_module(
        self, tmp_path: Path,
    ):
        """(a) force_workspace=True + role='merge' + breadth='full': every
        REGISTERED module executes with its OWN verbatim per-module
        commands — the opaque global chain is replaced by a per-module
        fan-out, NOT a single module-config-less global run_verification
        call."""
        mod_a, mod_b, config = _two_module_registry(tmp_path, breadth='full')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], force_workspace=True, role='merge',
            )

        assert result.passed
        executed = {mc.prefix: mc for mc in _executed_module_configs(mock_run_verification)}
        assert set(executed) == {'moda', 'modb'}, (
            f'expected BOTH registered modules to execute per-module under '
            f'force_workspace merge+full; got {set(executed)!r}'
        )
        assert executed['moda'].test_command == mod_a.test_command
        assert executed['moda'].lint_command == mod_a.lint_command
        assert executed['moda'].type_check_command == mod_a.type_check_command
        assert executed['modb'].test_command == mod_b.test_command
        assert executed['modb'].lint_command == mod_b.lint_command
        assert executed['modb'].type_check_command == mod_b.type_check_command

    @pytest.mark.asyncio
    async def test_force_workspace_merge_full_falls_back_when_no_module_has_commands(
        self, tmp_path: Path,
    ):
        """Robustness regression (reviewer amendment): force_workspace=True +
        role='merge' + breadth='full', but EVERY registered module has zero
        configured commands (all lint/pyright/test None).
        ``_derive_full_suite_runs`` always emits 3 (SKIPPED) runs per
        module, so ``scoped`` is never literally empty here — every module
        survives with all-None commands. Gathering over all-None
        ModuleConfigs would silently report passed=True with nothing
        actually executed, so this must fall back to the single legacy
        global ``run_verification`` call instead — mirroring the
        module_configs/task_files branch's ``if not scoped:`` guard
        (loud-over-silent-degradation, not a vacuous pass)."""
        mod_a = ModuleConfig(prefix='moda')
        mod_b = ModuleConfig(prefix='modb')
        config = OrchestratorConfig(project_root=tmp_path, merge_verify_breadth='full')
        config._module_configs = {'moda': mod_a, 'modb': mod_b}

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], force_workspace=True, role='merge',
            )

        assert result.passed
        assert mock_run_verification.await_count == 1, (
            f'expected exactly ONE legacy global fallback call when no registered '
            f'module has any configured command; got '
            f'{mock_run_verification.await_count} call(s)'
        )
        assert _executed_module_configs(mock_run_verification) == [], (
            'expected the fallback call to carry NO ModuleConfig (opaque global command)'
        )

    @pytest.mark.asyncio
    async def test_force_workspace_merge_scoped_stays_single_opaque_global_call(
        self, tmp_path: Path,
    ):
        """(b) force_workspace=True + breadth='scoped' (the shipped
        default): byte-identical legacy — exactly ONE global
        run_verification call carrying NO ModuleConfig (the opaque
        workspace command), never a per-module fan-out (R4 rollback
        golden)."""
        mod_a, _mod_b, config = _two_module_registry(tmp_path, breadth='scoped')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], force_workspace=True, role='merge',
            )

        assert result.passed
        assert mock_run_verification.await_count == 1, (
            f'expected exactly ONE opaque global call under breadth=scoped; '
            f'got {mock_run_verification.await_count} call(s)'
        )
        assert _executed_module_configs(mock_run_verification) == [], (
            'expected the legacy call to carry NO ModuleConfig (opaque global command)'
        )

    @pytest.mark.asyncio
    async def test_train_tip_shaped_call_widens_to_every_registered_module(
        self, tmp_path: Path,
    ):
        """(c) Train-tip-shaped call: force_workspace=False (as when
        merge_verify_workspace is off), role='merge', module_configs=[modA]
        only (union of the tip's own touched modules), breadth='full' —
        widens to ALL registered modules per-module (one broad verify of
        the tip; boundary row 7 plan shape) via the already-green step-8
        module_configs-branch expansion. Regression pin, not new behaviour —
        grouped here because it completes this class's T1 "train routing"
        coverage alongside (a)/(b)'s force_workspace fork."""
        mod_a, mod_b, config = _two_module_registry(tmp_path, breadth='full')
        source_path = 'moda/helpers.py'
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], task_files=[source_path],
                force_workspace=False, role='merge',
            )

        assert result.passed
        executed = {mc.prefix: mc for mc in _executed_module_configs(mock_run_verification)}
        assert set(executed) == {'moda', 'modb'}, (
            f'expected the train-tip call to widen to every registered module; '
            f'got {set(executed)!r}'
        )


# ---------------------------------------------------------------------------
# task 3294: the mixed-diff widening, end-to-end through the plan→execution bridge
# ---------------------------------------------------------------------------

# orchestrator/orchestrator.yaml:5/7/8, verbatim. The REAL config, not the
# synthetic 'uv run --directory moda pytest tests/' shape the rest of this
# module uses: the widening has to be exercised against the config the
# task-3033 incident actually ran under, or these goldens would only show that
# SOME command widens rather than that THE command the incident narrowed does.
#
# These are hand-copied, so they can drift.
# ``test_module_command_constants_are_the_live_orchestrator_yaml`` is what
# keeps them honest — it pins all three byte-identically to the live yaml. If
# you edit one of these, that test is where the pin lives.
_ORCH_TEST_COMMAND: str = (
    'uv run --project orchestrator --directory orchestrator pytest tests/ '
    '--tb=short -q --timeout=300'
)
_ORCH_LINT_COMMAND: str = (
    'uv run --project orchestrator --directory orchestrator ruff check src/ tests/'
    ' && python3 fused-memory/scripts/check_bare_magicmock_config.py orchestrator/tests'
)
_ORCH_TYPE_CHECK_COMMAND: str = (
    'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
)

# The task-3033 attempt-1 diff shape: the module's hottest production file plus
# a co-committed test file. Pre-3294 the plan file-scoped this to the test file
# alone (36 items instead of ~13188) and the regression in a DIFFERENT consumer
# of workflow.py was structurally invisible.
_MIXED_3033_DIFF: list[str] = [
    'orchestrator/src/orchestrator/workflow.py',
    'orchestrator/tests/test_new_thing.py',
]

# The STRUCTURAL content the real workflow.py carries (``class
# _McpLike(Protocol)``), so the file classifies STRUCTURAL rather than SOURCE —
# the case a SOURCE-only widening predicate would miss.
_STRUCTURAL_WORKFLOW_CONTENT: str = (
    'from typing import Protocol\n\n\n'
    'class _McpLike(Protocol):\n'
    '    def call(self) -> None: ...\n'
)


def _live_orchestrator_yaml() -> dict[str, object]:
    """Load the REAL ``orchestrator/orchestrator.yaml`` from the repo root.

    Reuses the repo's established "read a real yaml config in a test" idiom
    (``test_config.py``, ``test_chronic_flake.py``,
    ``test_harness_service_restart.py``): ``Path(__file__).parents[N]`` +
    ``yaml.safe_load(path.read_text())`` — no config-loader fake, no reach
    through ``OrchestratorConfig``/``discover_module_configs``.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return yaml.safe_load(
        (repo_root / 'orchestrator' / 'orchestrator.yaml').read_text(encoding='utf-8'),
    )


def _orchestrator_registry(tmp_path: Path) -> tuple[ModuleConfig, OrchestratorConfig]:
    """A single-module registry carrying orchestrator.yaml's REAL commands."""
    mc = ModuleConfig(
        prefix='orchestrator',
        test_command=_ORCH_TEST_COMMAND,
        lint_command=_ORCH_LINT_COMMAND,
        type_check_command=_ORCH_TYPE_CHECK_COMMAND,
    )
    config = OrchestratorConfig(project_root=tmp_path)
    config._module_configs = {'orchestrator': mc}
    return mc, config


def _write_3033_diff(tmp_path: Path) -> None:
    """Materialise the mixed diff under *tmp_path* so the existence filter keeps it.

    workflow.py's content carries a ``Protocol`` subclass so it classifies
    STRUCTURAL exactly as the real file does (it defines ``class
    _McpLike(Protocol)``) — the case a SOURCE-only widening predicate would
    miss.
    """
    src = tmp_path / 'orchestrator' / 'src' / 'orchestrator'
    src.mkdir(parents=True, exist_ok=True)
    (src / 'workflow.py').write_text(_STRUCTURAL_WORKFLOW_CONTENT)
    tests = tmp_path / 'orchestrator' / 'tests'
    tests.mkdir(parents=True, exist_ok=True)
    (tests / 'test_new_thing.py').write_text('def test_new_thing(): pass\n')


class TestTaskRoleMixedDiffFullSuiteExecution:
    """Task 3294: the mixed-diff widening, against the real orchestrator config.

    WHY THIS CLASS EXISTS. Task 3033's attempt-1 diff touched
    ``workflow.py`` plus a co-committed test file. Pre-3294 the plan took the
    collectable-tests branch and narrowed the TEST leg to that one file — 36
    items instead of the module's ~13188 — so a regression in a DIFFERENT
    consumer of ``workflow.py`` was structurally invisible and reached a
    debugger. The fix is a decision-layer reorder; this class pins that the
    fix is real all the way out to what runs.

    Three things are pinned here, and they are deliberately different kinds
    of assertion:

    - (a)/(b) THE BRIDGE. ``test_verify_plan.py``'s
      ``TestDeriveVerifyPlanTaskRoleFloorMixedDiff`` pins the decision at the
      pure ``derive_verify_plan`` layer; task κ made the ``VerifyPlan``
      authoritative, but ``_executed_module_configs_from_plan`` is what
      actually renders it into the ``ModuleConfig``(s) handed to
      ``run_verification``. So these drive the REAL
      ``run_scoped_verification`` with the task-3033-shaped diff and pin what
      EXECUTES, plus that ``result.plan`` records the same thing an operator
      reads (PRD A1).
    - (c1) PROVENANCE. (a)/(b) assert against the hand-copied command
      constants above, so a pin ties those constants byte-identically to the
      live ``orchestrator/orchestrator.yaml``. Without it the suite could
      stay green against a copy that no longer matches the config.
    - (c2) GRANULARITY. The widened plan reproduces the owning module's own
      ``test_command`` UNNARROWED and targets its whole test PACKAGE (PRD
      Resolved decision 3) — derived from the plan and asserted path-free
      (every target a directory, none a file), so the claim does not rest on
      any individual test module's filename.
    """

    @pytest.mark.asyncio
    async def test_task_3033_shaped_mixed_diff_executes_owning_module_full_suite(
        self, tmp_path: Path,
    ):
        """(a) The EXECUTED ModuleConfig carries the module's verbatim
        full-suite test_command — not a narrowed one — and the command never
        names the co-committed test file."""
        mc, config = _orchestrator_registry(tmp_path)
        _write_3033_diff(tmp_path)

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mc], task_files=_MIXED_3033_DIFF, role='task',
            )

        assert result.passed
        executed = _executed_module_configs(mock_run_verification)
        assert len(executed) == 1
        executed_mc = executed[0]
        assert executed_mc.prefix == 'orchestrator'
        # The verbatim full-suite command, byte-for-byte — a FULL_SUITE slot
        # is rendered by _executed_module_configs_from_plan as getattr(mc, attr).
        assert executed_mc.test_command == _ORCH_TEST_COMMAND
        assert executed_mc.test_command is not None
        assert 'test_new_thing.py' not in executed_mc.test_command

    @pytest.mark.asyncio
    async def test_result_plan_records_full_suite_for_the_mixed_diff(
        self, tmp_path: Path,
    ):
        """(b) PRD A1: the record an operator reads matches what ran. The
        plan's orchestrator pytest run is 'full_suite' with no
        scoped_targets, and its reason explains that the co-committed test
        did not narrow it."""
        mc, config = _orchestrator_registry(tmp_path)
        _write_3033_diff(tmp_path)

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mc], task_files=_MIXED_3033_DIFF, role='task',
            )

        assert result.plan is not None
        pytest_runs = [
            r for r in result.plan['runs']
            if r['module_prefix'] == 'orchestrator' and r['reason'].startswith('pytest:')
        ]
        assert len(pytest_runs) == 1
        pytest_run = pytest_runs[0]
        assert pytest_run['scope_kind'] == 'full_suite'
        assert pytest_run['scoped_targets'] == []
        assert 'narrow' in pytest_run['reason'].lower()

    def test_module_command_constants_are_the_live_orchestrator_yaml(self):
        """(c1) PROVENANCE. The three module-command constants above are
        hand-copied; this is what keeps them honest.

        (a) and (b) drive the widening from ``_ORCH_TEST_COMMAND`` and assert
        the executed command against it — which is only worth anything if that
        literal really is the command the task-3033 incident ran under. Without
        this pin, the live ``test_command`` could be narrowed (an ``--ignore``,
        a single-file target) and the widened FULL_SUITE run would silently
        stop collecting the regressed module while this suite stayed green.
        A drift guard, so GREEN on arrival by construction.
        """
        data = _live_orchestrator_yaml()
        yaml_ref = 'orchestrator/orchestrator.yaml'
        assert data['test_command'] == _ORCH_TEST_COMMAND, (
            f'_ORCH_TEST_COMMAND has drifted from {yaml_ref}:test_command. This '
            f"suite's whole task-3033 story depends on driving the widening with "
            f'the command the incident actually ran under — re-copy it verbatim '
            f'and re-check that the widened run still targets the whole tests/ '
            f'package'
        )
        assert data['lint_command'] == _ORCH_LINT_COMMAND, (
            f'_ORCH_LINT_COMMAND has drifted from {yaml_ref}:lint_command; '
            f're-copy it verbatim'
        )
        assert data['type_check_command'] == _ORCH_TYPE_CHECK_COMMAND, (
            f'_ORCH_TYPE_CHECK_COMMAND has drifted from {yaml_ref}:'
            f'type_check_command — it is what makes workflow.py classify '
            f'STRUCTURAL here (content is only read when type_check_command is '
            f'set); re-copy it verbatim'
        )

    def test_widened_plan_targets_the_owning_modules_whole_test_package(self):
        """(c2) The widened run reproduces the owning module's OWN command,
        unnarrowed, and targets its whole test PACKAGE.

        Derived from the plan, not from a local literal: the ``ModuleConfig``
        is built from the yaml-LOADED commands, so these assertions track the
        live config rather than this module's copy of it. Pure — the
        ``worktree_reader`` seam makes it a straight ``derive_verify_plan``
        call with no tmp_path, no subprocess, no mocking.

        Anti-tautology: pre-3294 this same diff took the collectable-tests
        branch and the cmd was scoped to ``orchestrator/tests/test_new_thing.py``,
        so ``cmd == parse_config_command(<live test_command>)`` genuinely
        discriminates. Package granularity (PRD Resolved decision 3) is
        asserted path-free — every target is a directory, none is a file — so
        the claim survives any legitimate rename in the targeted package.
        """
        data = _live_orchestrator_yaml()
        yaml_test_command = data['test_command']
        yaml_lint_command = data['lint_command']
        yaml_type_check_command = data['type_check_command']
        assert isinstance(yaml_test_command, str)
        assert isinstance(yaml_lint_command, str)
        assert isinstance(yaml_type_check_command, str)

        mc = ModuleConfig(
            prefix='orchestrator',
            test_command=yaml_test_command,
            lint_command=yaml_lint_command,
            type_check_command=yaml_type_check_command,
        )

        def reader(path: str) -> str | None:
            if path == 'orchestrator/src/orchestrator/workflow.py':
                return _STRUCTURAL_WORKFLOW_CONTENT
            return None

        plan = derive_verify_plan(_MIXED_3033_DIFF, [mc], None, reader, role='task')
        run = next(
            (
                r for r in plan.runs
                if r.module_prefix == 'orchestrator' and r.reason.startswith('pytest:')
            ),
            None,
        )
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        assert run.cmd is not None
        # parse_config_command, never render() — render normalises --directory
        # into a leading `cd`, so a render-based comparison would fail against
        # a --directory-form input even when the behaviour is correct.
        assert run.cmd == parse_config_command(yaml_test_command), (
            f"the widened slot must reproduce the owning module's own "
            f'test_command UNNARROWED; got {run.cmd!r}'
        )
        assert run.cmd.targets, 'a FULL_SUITE pytest run must still name its package'
        for target in run.cmd.targets:
            assert target.endswith('/'), (
                f'{target!r} is not a directory-style target — the widened run '
                f'must select the whole test PACKAGE (PRD Resolved decision 3), '
                f'not individual files'
            )
            assert not target.endswith('.py')
        assert not any('test_new_thing.py' in t for t in run.cmd.targets)
        assert run.scoped_targets == ()
