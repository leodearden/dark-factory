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

Task 4536 extends this module one layer outward, to the REMOTE leg. The
classes above pin that ``run_scoped_verification`` delegates the
effective-set DECISION to ``verify_plan.effective_merge_module_configs``
(task 3787 γ); ``TestRemoteSpecAuthoritativeModuleRegistry`` below pins
WHERE that helper's registry comes FROM when the merge runs off-host —
i.e. for the Lever C consumer, ``verify_runner.run_merge_verify_on_worktree``
(the CLI ``verify-merge`` subcommand's host entry). The helper reads
``config.module_configs_or_empty``, which on that leg used to be the REMOTE
host's own ``_discover_module_configs`` walk, so a stale/divergent laptop
checkout silently decided the merge — both DROPPING modules the spec named
(the task-2822 false-green class) and INJECTING modules it never did. The
dispatcher's spec is authoritative for the module SET, so these tests drive
the real ``build_merge_verify_spec`` → ``run_merge_verify_on_worktree`` →
``run_scoped_verification`` round trip with a dispatcher and a host whose
registries deliberately DISAGREE, and observe which side actually executed.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from test_verify_scope_kappa import _executed_module_configs, _run_verification_spy

from orchestrator import verify, verify_plan
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_scoped_verification
from orchestrator.verify_plan import ScopeKind, derive_verify_plan, parse_config_command
from orchestrator.verify_runner import (
    MergeVerifySpec,
    build_merge_verify_spec,
    run_merge_verify_on_worktree,
)

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


def _orchestrator_module_config() -> ModuleConfig:
    """A ModuleConfig carrying orchestrator.yaml's REAL, LIVE commands.

    Loaded from the yaml rather than hand-copied into constants here: the
    provenance these tests need ("the widening is exercised against the config
    the task-3033 incident actually ran under") then holds BY CONSTRUCTION,
    with no drift pin to maintain and no coupling to command values — such as
    the lint chain's members or pytest's --timeout — that this suite does not
    care about and that routinely, harmlessly change.

    The REAL commands, not the synthetic 'uv run --directory moda pytest
    tests/' shape the rest of this module uses: against a toy command these
    goldens would only show that SOME command widens, not that THE command the
    incident narrowed does.
    """
    data = _live_orchestrator_yaml()
    test_command = data['test_command']
    lint_command = data['lint_command']
    type_check_command = data['type_check_command']
    assert isinstance(test_command, str)
    assert isinstance(lint_command, str)
    assert isinstance(type_check_command, str)
    return ModuleConfig(
        prefix='orchestrator',
        test_command=test_command,
        lint_command=lint_command,
        # Load-bearing only in being NON-EMPTY: content is read (so workflow.py
        # can classify STRUCTURAL) exactly when a type_check_command is set.
        type_check_command=type_check_command,
    )


def _orchestrator_registry(tmp_path: Path) -> tuple[ModuleConfig, OrchestratorConfig]:
    """A single-module registry carrying orchestrator.yaml's REAL commands."""
    mc = _orchestrator_module_config()
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

    Two things are pinned here, and they are deliberately different kinds of
    assertion:

    - (a)/(b) THE BRIDGE. ``test_verify_plan.py``'s
      ``TestDeriveVerifyPlanTaskRoleFloorMixedDiff`` pins the decision at the
      pure ``derive_verify_plan`` layer; task κ made the ``VerifyPlan``
      authoritative, but ``_executed_module_configs_from_plan`` is what
      actually renders it into the ``ModuleConfig``(s) handed to
      ``run_verification``. So these drive the REAL
      ``run_scoped_verification`` with the task-3033-shaped diff and pin what
      EXECUTES, plus that ``result.plan`` records the same thing an operator
      reads (PRD A1).
    - (c) GRANULARITY. The widened plan reproduces the owning module's own
      ``test_command`` UNNARROWED and targets its whole test PACKAGE (PRD
      Resolved decision 3) — derived from the plan and asserted path-free (no
      target is a file), so the claim does not rest on any individual test
      module's filename.

    Every case runs against the LIVE ``orchestrator/orchestrator.yaml``
    commands (``_orchestrator_module_config``), so the provenance the
    task-3033 story needs — the widening exercised against the config the
    incident actually ran under — holds by construction rather than by a
    hand-copied literal and a drift pin.
    """

    @pytest.mark.asyncio
    async def test_task_3033_shaped_mixed_diff_executes_owning_module_full_suite(
        self, tmp_path: Path,
    ):
        """(a) The EXECUTED ModuleConfig carries the module's verbatim
        full-suite test_command — not a narrowed one — and the command never
        names the co-committed test file."""
        mc, config = _orchestrator_registry(tmp_path)
        # Precondition for the STRUCTURAL half of the story: content is read,
        # so workflow.py can classify STRUCTURAL, exactly when this is set.
        assert mc.type_check_command
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
        # The module's OWN command, byte-for-byte — a FULL_SUITE slot is
        # rendered by _executed_module_configs_from_plan as getattr(mc, attr),
        # where a FILE_SCOPED one would have narrowed it to the touched test.
        assert executed_mc.test_command == mc.test_command
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

    def test_widened_plan_targets_the_owning_modules_whole_test_package(self):
        """(c) The widened run reproduces the owning module's OWN command,
        unnarrowed, and targets its whole test PACKAGE.

        Derived from the plan and from the live yaml, never from a local
        literal, so these assertions track the real config. Pure — the
        ``worktree_reader`` seam makes it a straight ``derive_verify_plan``
        call with no tmp_path, no subprocess, no mocking.

        Anti-tautology: pre-3294 this same diff took the collectable-tests
        branch and the cmd was scoped to ``orchestrator/tests/test_new_thing.py``,
        so ``cmd == parse_config_command(<live test_command>)`` genuinely
        discriminates. Package granularity (PRD Resolved decision 3) is
        asserted path-free — no target is a FILE — so the claim survives any
        legitimate rename in the targeted package, and does not pin the
        config's cosmetic choice of a trailing slash.
        """
        mc = _orchestrator_module_config()
        assert mc.test_command is not None

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
        assert run.cmd == parse_config_command(mc.test_command), (
            f"the widened slot must reproduce the owning module's own "
            f'test_command UNNARROWED; got {run.cmd!r}'
        )
        assert run.cmd.targets, 'a FULL_SUITE pytest run must still name its package'
        for target in run.cmd.targets:
            # Package, not file. Asserted as "has no file extension" rather
            # than "ends with /": `pytest tests` and `pytest tests/` are the
            # same selection to pytest, and the trailing slash is a cosmetic
            # property of how the config happens to be spelled.
            assert '.' not in PurePosixPath(target.rstrip('/')).name, (
                f'{target!r} names a FILE — the widened run must select the '
                f'whole test PACKAGE (PRD Resolved decision 3)'
            )
        assert not any('test_new_thing.py' in t for t in run.cmd.targets)
        assert run.scoped_targets == ()


# ---------------------------------------------------------------------------
# task 3787 (flake-ledger γ) step-3: run_scoped_verification delegates the
# "effective merge module set" decision to the ONE shared helper (INV-5)
# ---------------------------------------------------------------------------


def _effective_helper_spy(returns: list[ModuleConfig]):
    """A stand-in for ``verify_plan.effective_merge_module_configs`` returning
    *returns*, recording every ``(config, module_configs)`` it was called with.

    Returns ``(spy, calls)``. Patch onto the ``verify_plan`` MODULE object —
    verify.py does ``from orchestrator import verify_plan`` and resolves the
    attribute at CALL time, so patching there intercepts; patching the name
    re-imported into some other module would install the spy off the
    resolution path and pass vacuously.
    """
    calls: list[tuple[object, list[ModuleConfig]]] = []

    def _spy(config, module_configs):
        calls.append((config, list(module_configs)))
        return list(returns)

    return _spy, calls


class TestRunScopedVerificationDelegatesEffectiveModuleConfigs:
    """``run_scoped_verification`` asks ``verify_plan.effective_merge_module_configs``
    which modules a merge covers — it does NOT reimplement the expansion inline.

    Flake-ledger PRD §8.2 / task 3787 (γ), INV-5: there must be exactly ONE
    implementation of "the effective merge module set" in the tree, because γ
    also resolves it at the merge-request boundary (merge_queue) so the local
    runner, the wire spec and the suppression gate all receive the identical
    set BY CONSTRUCTION. Two copies of the expression would be two things
    asserted to agree — which is the drift §8.2 exists to remove.

    The spy returns modB — the REGISTERED-but-untouched module, which is
    neither the passed set (modA) nor, on its own, the registry — so an
    execution set equal to the spy's return can only have come THROUGH the
    helper, not from an inline re-expansion that happens to look similar.

    Note ``derive_verify_plan``'s own internal breadth fork (a FULL_SUITE-vs-
    file-scoped decision about HOW to run a module) is a separate concern and
    stays untouched here; this is about WHICH modules arrive.
    """

    # -- (a) the module_configs branch (verify.py site B) ---------------------

    @pytest.mark.asyncio
    async def test_module_configs_branch_executes_exactly_the_helpers_return(
        self, tmp_path: Path, monkeypatch,
    ):
        """(a) role='merge' + breadth='full': the helper is consulted ONCE with
        ``(config, <passed set>)`` and the EXECUTED module set is its return —
        not the registry, and not the passed set."""
        mod_a, mod_b, config = _two_module_registry(tmp_path, breadth='full')
        source_path = 'moda/helpers.py'
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        spy, calls = _effective_helper_spy([mod_b])
        monkeypatch.setattr(verify_plan, 'effective_merge_module_configs', spy)

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], task_files=[source_path], role='merge',
            )

        assert result.passed
        assert len(calls) == 1, f'expected exactly one helper call; got {len(calls)}'
        called_config, called_modules = calls[0]
        assert called_config is config
        assert called_modules == [mod_a], (
            f'the helper must be handed the PASSED (task-scoped) set; got '
            f'{[m.prefix for m in called_modules]!r}'
        )
        executed = {mc.prefix for mc in _executed_module_configs(mock_run_verification)}
        assert executed == {'modb'}, (
            f"expected execution to follow the helper's return exactly; got {executed!r}"
        )

    # -- (b) the force_workspace branch (verify.py site A) --------------------

    @pytest.mark.asyncio
    async def test_force_workspace_branch_fans_out_over_exactly_the_helpers_return(
        self, tmp_path: Path, monkeypatch,
    ):
        """(b) The same delegation on the ``force_workspace`` per-module
        full-suite fan-out — the second site that used to re-expand inline."""
        mod_a, mod_b, config = _two_module_registry(tmp_path, breadth='full')

        spy, calls = _effective_helper_spy([mod_b])
        monkeypatch.setattr(verify_plan, 'effective_merge_module_configs', spy)

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a], force_workspace=True, role='merge',
            )

        assert result.passed
        assert len(calls) == 1, f'expected exactly one helper call; got {len(calls)}'
        called_config, called_modules = calls[0]
        assert called_config is config
        assert called_modules == [mod_a]
        executed = {mc.prefix for mc in _executed_module_configs(mock_run_verification)}
        assert executed == {'modb'}, (
            f"expected the fan-out to cover exactly the helper's return; got {executed!r}"
        )

    # -- (c) role='task' control: the helper is never consulted ---------------

    @pytest.mark.asyncio
    @pytest.mark.parametrize('force_workspace', [False, True])
    async def test_task_role_never_consults_the_helper(
        self, tmp_path: Path, monkeypatch, force_workspace: bool,
    ):
        """(c) Breadth is merge-role-gated ONLY. At role='task' the helper is
        not called at all and execution stays task-scoped — the train-member
        override (workflow.py) must not pick up the broad gate by accident."""
        mod_a, mod_b, config = _two_module_registry(tmp_path, breadth='full')
        source_path = 'moda/helpers.py'
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        spy, calls = _effective_helper_spy([mod_b])
        monkeypatch.setattr(verify_plan, 'effective_merge_module_configs', spy)

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a],
                task_files=None if force_workspace else [source_path],
                force_workspace=force_workspace,
                role='task',
            )

        assert result.passed
        assert calls == [], (
            f'the helper must never be consulted at role="task"; got {len(calls)} call(s)'
        )
        executed = {mc.prefix for mc in _executed_module_configs(mock_run_verification)}
        assert 'modb' not in executed, (
            f'role="task" must stay task-scoped; got {executed!r}'
        )

    # -- (d) VALUE-PRESERVATION goldens with the REAL helper ------------------

    @pytest.mark.asyncio
    @pytest.mark.parametrize('force_workspace', [False, True])
    async def test_real_helper_full_breadth_still_executes_every_registered_module(
        self, tmp_path: Path, force_workspace: bool,
    ):
        """(d1) No spy: breadth='full' + role='merge' still executes EVERY
        registered module on BOTH branches, exactly as before the refactor."""
        mod_a, _mod_b, config = _two_module_registry(tmp_path, breadth='full')
        source_path = 'moda/helpers.py'
        full = tmp_path / source_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text('def helper():\n    return 1\n')

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_scoped_verification(
                tmp_path, config, [mod_a],
                task_files=None if force_workspace else [source_path],
                force_workspace=force_workspace,
                role='merge',
            )

        assert result.passed
        executed = {mc.prefix for mc in _executed_module_configs(mock_run_verification)}
        assert executed == {'moda', 'modb'}, (
            f'value-preservation: merge+full must still cover the whole registry; '
            f'got {executed!r}'
        )

    @pytest.mark.asyncio
    async def test_real_helper_scoped_breadth_still_executes_only_the_touched_module(
        self, tmp_path: Path,
    ):
        """(d2) The R4 rollback golden: breadth='scoped' + role='merge' still
        executes only the touched module. The helper is the identity here, so
        this path is byte-identical to the legacy one."""
        mod_a, _mod_b, config = _two_module_registry(tmp_path, breadth='scoped')
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
        executed = {mc.prefix for mc in _executed_module_configs(mock_run_verification)}
        assert executed == {'moda'}, (
            f'R4 rollback: breadth="scoped" must stay task-scoped; got {executed!r}'
        )

    @pytest.mark.asyncio
    async def test_real_helper_full_breadth_empty_registry_falls_back_to_passed_set(
        self, tmp_path: Path,
    ):
        """(d3) breadth='full' with an EMPTY registry still falls back to the
        passed module_configs rather than executing nothing — the safe degrade,
        pinned end-to-end through execution and not just at the helper."""
        mod_a, _mod_b, config = _two_module_registry(tmp_path, breadth='full')
        config._module_configs = {}
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
        executed = {mc.prefix for mc in _executed_module_configs(mock_run_verification)}
        assert executed == {'moda'}, (
            f'safe degrade: an empty registry must fall back to the passed set, '
            f'never execute nothing; got {executed!r}'
        )


# ---------------------------------------------------------------------------
# task 4536: the REMOTE leg — the SPEC, not the host's discovery, is the
# authoritative source of the merge-verify module SET
# ---------------------------------------------------------------------------


# The nine real dark_factory module prefixes, as the live registry would carry
# them — so the dispatcher side of the divergence is not a toy 1-vs-1 shape but
# the actual set the production merge gate widens to under breadth='full'.
_DARK_FACTORY_PREFIXES = (
    'cockpit', 'dashboard', 'escalation', 'fused-memory', 'orchestrator',
    'sampler', 'scripts', 'shared', 'tests/scripts',
)


def _tagged_module(prefix: str, tag: Literal['DISP', 'HOST']) -> ModuleConfig:
    """A ModuleConfig whose three commands are all provenance-tagged *tag*.

    The tag is what lets an assertion distinguish "the spec's set won" from
    "the two sets happened to union" on an OVERLAPPING prefix: a module named
    by BOTH registries can only be attributed by whose command it carries.
    """
    return ModuleConfig(
        prefix=prefix,
        test_command=f'{tag}_TEST --directory {prefix} pytest tests/',
        lint_command=f'{tag}_LINT --directory {prefix} ruff check src/',
        type_check_command=f'{tag}_TYPE --directory {prefix} pyright src/',
    )


def _divergent_dispatcher_and_host(
    tmp_path: Path,
    *,
    dispatcher_prefixes: tuple[str, ...],
    host_prefixes: tuple[str, ...],
    workspace: bool,
) -> tuple[list[ModuleConfig], MergeVerifySpec, OrchestratorConfig]:
    """A DISPATCHER (spec producer) and a REMOTE HOST whose registries disagree.

    Mirrors ``_two_module_registry``'s shape (real ``OrchestratorConfig`` +
    the blessed direct ``config._module_configs = {...}`` assignment) but
    builds a PAIR of configs, so an observed execution proves WHICH SIDE won
    rather than echoing the test's own argument back — the same reasoning
    ``_two_module_registry``'s "modB is registry-only, never passed in"
    docstring gives, lifted to the local/remote axis.

    Returns ``(dispatcher_module_configs, spec, host_config)``. The spec comes
    from the REAL ``build_merge_verify_spec`` rather than hand-authored
    ``VerifyCommand`` tuples, so these tests exercise the genuine
    producer→wire→consumer round trip and pick up ``global_verify_command``
    (INV-1, task 2883) and the ``merge_verify_workspace`` /
    ``merge_verify_breadth`` projection (fix (a), task 2822) for free. That is
    also the production ordering post-task-3787-γ: ``merge_queue``'s
    ``_merge_boundary_module_configs`` resolves the effective set ONCE at the
    merge-request boundary and hands it to this same producer, so a
    nine-module dispatcher spec is exactly what the live merge gate ships.

    Fixture requirements, each of which bites if dropped:

    - ``merge_verify_breadth`` AND ``merge_verify_workspace`` are pinned
      EXPLICITLY on BOTH configs. ``OrchestratorConfig`` is a ``BaseSettings``
      whose bare defaults can be widened by a settings source (see the comment
      at test_verify_runner.py's ``test_spec_profile_overrides_host_config``).
    - The HOST is built at ``merge_verify_breadth='scoped'`` on purpose: the
      full fan-out these tests observe can then only have come from the SPEC's
      breadth via fix (a), never from the host's own knob.
    - Both configs share ``project_root=tmp_path`` and every file in
      *task_files* EXISTS on disk — verify.py's reuse guard re-walks the
      filesystem when project_root differs, and the scoped path filters
      *task_files* down to files that exist.
    - Global commands are tagged on BOTH sides too, so the zero-module (INV-1)
      case can attribute the global gate as well.
    """
    (tmp_path / 'f.py').write_text('x = 1\n')
    task_files = ('f.py',)

    dispatcher_modules = [_tagged_module(p, 'DISP') for p in dispatcher_prefixes]
    dispatcher = OrchestratorConfig(
        project_root=tmp_path,
        merge_verify_breadth='full',
        merge_verify_workspace=workspace,
        test_command='DISP_GLOBAL_TEST pytest',
        lint_command='DISP_GLOBAL_LINT ruff check',
        type_check_command='DISP_GLOBAL_TYPE pyright',
    )
    dispatcher._module_configs = {mc.prefix: mc for mc in dispatcher_modules}
    spec = build_merge_verify_spec(dispatcher, dispatcher_modules, task_files)

    host_modules = [_tagged_module(p, 'HOST') for p in host_prefixes]
    host = OrchestratorConfig(
        project_root=tmp_path,
        merge_verify_breadth='scoped',
        merge_verify_workspace=False,
        test_command='HOST_GLOBAL_TEST pytest',
        lint_command='HOST_GLOBAL_LINT ruff check',
        type_check_command='HOST_GLOBAL_TYPE pyright',
    )
    host._module_configs = {mc.prefix: mc for mc in host_modules}
    return dispatcher_modules, spec, host


def _passing_unscoped_gate() -> AsyncMock:
    """A passing stand-in for ``merge_queue._run_unscoped_typechecks``.

    Injected so ``LocalRunner.run_merge_verify``'s second phase never reaches
    the real gate; ``run_scoped`` is deliberately NOT injected, so the REAL
    ``run_scoped_verification`` runs and these tests observe genuine routing.
    """
    return AsyncMock(return_value=MagicMock(
        broken=False, timed_out=False, failing_subprojects=[], timed_out_subprojects=[],
    ))


class TestRemoteSpecAuthoritativeModuleRegistry:
    """The merge-verify module SET comes from the SPEC on the remote leg.

    Task 4536. ``run_merge_verify_on_worktree`` reconstructs the dispatcher's
    module_configs from ``spec.verify_commands`` and passes them positionally
    — but under ``merge_verify_breadth='full'``
    ``verify_plan.effective_merge_module_configs`` prefers
    ``config.module_configs_or_empty``, which on this leg is the REMOTE host's
    own ``_discover_module_configs`` walk. So the passed set was thrown away
    and a divergent laptop checkout decided the merge.

    Each test below pins BOTH halves of the infidelity, which are distinct
    failure modes and would need distinct bugs to reappear:

    - DROPPED: a module the spec named must still execute (the task-2822
      false-green class — the merge is vouched for by a gate that never ran
      on 7 of the 9 modules).
    - INJECTED: a module ONLY the host knows about must NOT execute (a merge
      red attributable to a subproject the dispatching side never scoped).

    Attribution is by command tag, never by prefix alone: the OVERLAPPING
    prefix in case (a) is registered by both sides, so only the ``DISP_*``
    command proves the spec's registry won rather than the two sets unioning.
    """

    @pytest.mark.asyncio
    async def test_live_routing_executes_the_spec_module_set_not_the_hosts(
        self, tmp_path: Path,
    ):
        """(a) dark_factory's LIVE routing — ``merge_verify_workspace=False``,
        so the ``if module_configs:`` site (NOT the force_workspace fan-out the
        task description names; ``dark-factory-orchestrator.yaml`` sets
        ``merge_verify_breadth: full`` but leaves ``merge_verify_workspace`` at
        its ``False`` default).

        Observed on main before the fix: exactly ``['cockpit', 'stale-only']``
        — the HOST's two, with ``HOST_*`` commands.
        """
        _disp_modules, spec, host = _divergent_dispatcher_and_host(
            tmp_path,
            dispatcher_prefixes=_DARK_FACTORY_PREFIXES,
            host_prefixes=('cockpit', 'stale-only'),
            workspace=False,
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_merge_verify_on_worktree(
                tmp_path, host, spec, run_unscoped=_passing_unscoped_gate(),
            )

        assert result.passed
        executed = {mc.prefix: mc for mc in _executed_module_configs(mock_run_verification)}
        assert set(executed) == set(_DARK_FACTORY_PREFIXES), (
            f'the SPEC names the module set a remote merge covers; expected '
            f'{set(_DARK_FACTORY_PREFIXES)!r}, got {set(executed)!r}'
        )
        for prefix, mc in sorted(executed.items()):
            assert mc.test_command == f'DISP_TEST --directory {prefix} pytest tests/', (
                f'{prefix!r} executed a non-dispatcher test command: {mc.test_command!r}'
            )
            assert mc.lint_command == f'DISP_LINT --directory {prefix} ruff check src/', (
                f'{prefix!r} executed a non-dispatcher lint command: {mc.lint_command!r}'
            )
            assert mc.type_check_command == f'DISP_TYPE --directory {prefix} pyright src/', (
                f'{prefix!r} executed a non-dispatcher type command: '
                f'{mc.type_check_command!r}'
            )
        assert 'stale-only' not in executed, (
            'a host-only module the spec never named must NOT execute — the fix '
            'must stop phantom INJECTION, not merely restore the DROPPED modules'
        )

    @pytest.mark.asyncio
    async def test_force_workspace_routing_executes_the_spec_module_set_not_the_hosts(
        self, tmp_path: Path,
    ):
        """(b) The force_workspace per-module fan-out site. The DISPATCHER
        carries ``merge_verify_workspace=True``, so the spec ships it and fix
        (a) applies it onto the host config — the host's own knob is False, so
        reaching this routing at all already proves the spec drove the profile.

        Observed on main before the fix: exactly ``['hostonly1', 'hostonly2']``.
        """
        _disp_modules, spec, host = _divergent_dispatcher_and_host(
            tmp_path,
            dispatcher_prefixes=_DARK_FACTORY_PREFIXES,
            host_prefixes=('hostonly1', 'hostonly2'),
            workspace=True,
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_merge_verify_on_worktree(
                tmp_path, host, spec, run_unscoped=_passing_unscoped_gate(),
            )

        assert result.passed
        executed = {mc.prefix: mc for mc in _executed_module_configs(mock_run_verification)}
        assert set(executed) == set(_DARK_FACTORY_PREFIXES), (
            f'the force_workspace fan-out must cover the SPEC\'s registry; expected '
            f'{set(_DARK_FACTORY_PREFIXES)!r}, got {set(executed)!r}'
        )
        for prefix, mc in sorted(executed.items()):
            assert mc.test_command == f'DISP_TEST --directory {prefix} pytest tests/'
            assert mc.lint_command == f'DISP_LINT --directory {prefix} ruff check src/'
            assert mc.type_check_command == f'DISP_TYPE --directory {prefix} pyright src/'
        assert not {'hostonly1', 'hostonly2'} & set(executed), (
            f'host-only modules the spec never named must NOT execute; got {set(executed)!r}'
        )

    @pytest.mark.asyncio
    async def test_zero_module_spec_runs_the_shipped_global_gate_not_the_hosts_modules(
        self, tmp_path: Path,
    ):
        """(c) INV-1 (task 2883) on the workspace path. A dispatcher with ZERO
        module configs but real global commands ships
        ``global_verify_command``; fix (a)'s ``config_update`` already applies
        those globals onto the host config. But the force_workspace fan-out
        consults the REGISTRY before the helper's ``or module_configs``
        fallback can apply — so a host that registers modules runs ITS OWN
        modules instead of the global gate the spec shipped.

        Observed on main before the fix: TWO calls carrying the HOST's module
        configs. An outright INV-1 hole, structurally invisible to reify (the
        only Lever C project today) only because reify leaves
        ``merge_verify_workspace`` False, where a zero-module spec never enters
        the ``if module_configs:`` branch at all.
        """
        _disp_modules, spec, host = _divergent_dispatcher_and_host(
            tmp_path,
            dispatcher_prefixes=(),
            host_prefixes=('hostonly1', 'hostonly2'),
            workspace=True,
        )
        assert spec.verify_commands == (), 'fixture precondition: a zero-module spec'
        assert spec.global_verify_command is not None, (
            'fixture precondition: build_merge_verify_spec must ship the '
            'dispatching globals for a zero-module spec (INV-1, task 2883)'
        )

        mock_run_verification = _run_verification_spy()
        with patch.object(verify, 'run_verification', new=mock_run_verification):
            result = await run_merge_verify_on_worktree(
                tmp_path, host, spec, run_unscoped=_passing_unscoped_gate(),
            )

        assert result.passed
        assert mock_run_verification.await_count == 1, (
            f'a zero-module spec must run the ONE opaque global gate it shipped, '
            f'never the host\'s own modules; got '
            f'{mock_run_verification.await_count} call(s)'
        )
        assert _executed_module_configs(mock_run_verification) == [], (
            'expected the global gate call to carry NO ModuleConfig (opaque global '
            'command); a non-empty list means the host\'s registry drove execution'
        )
        effective_config = mock_run_verification.await_args.args[1]
        assert effective_config.test_command == 'DISP_GLOBAL_TEST pytest'
        assert effective_config.lint_command == 'DISP_GLOBAL_LINT ruff check'
        assert effective_config.type_check_command == 'DISP_GLOBAL_TYPE pyright'
