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
from test_verify_scope_kappa import _executed_module_configs, _run_verification_spy

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import run_scoped_verification

# ---------------------------------------------------------------------------
# step-7: role x merge_verify_breadth end-to-end execution goldens (RED)
# ---------------------------------------------------------------------------


def _two_module_registry(tmp_path: Path) -> tuple[ModuleConfig, ModuleConfig, OrchestratorConfig]:
    """A 2-module registry (modA/modB), ``merge_verify_breadth='full'``.

    Mirrors a real dark_factory ``config.module_configs_or_empty`` registry —
    modB is NEVER passed to ``run_scoped_verification`` as part of the
    *module_configs* argument, only discoverable via the registry, so a test
    that observes modB execute is exercising the merge+full expansion
    (step-8), not merely echoing its own *module_configs* argument back.
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
    config = OrchestratorConfig(project_root=tmp_path, merge_verify_breadth='full')
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
