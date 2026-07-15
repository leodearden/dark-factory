"""Tests for orchestrator.verify's reverse-dependency test widening (task 2607).

Closes the merge-verify blind spot where a task's diff scoped to
orchestrator/ SOURCE never runs escalation/'s coupled cross-package
merge_queue tests — module_configs is resolved from the TASK's own touched
modules only, so escalation's ModuleConfig is never a candidate for an
orchestrator-only diff. This has caused RED-main fix-forward 3x (1736->1761,
2173->2038, 2435->2604), each patched reactively; this task closes it
structurally.

Two layers, each with its own test class in this file:

- ``verify._reverse_dependency_module_configs`` (step-5/step-6): the impure
  wrapper that builds worktree-bound ``list_pkg_tests``/``read_content``
  callables, calls the pure ``verify_plan.reverse_dependent_test_targets``,
  and renders each coupled dependent into an executable, pytest-only
  ``ModuleConfig``.
- ``run_scoped_verification`` wiring (step-7 onward): integration tests
  asserting the widened ModuleConfig's pytest command actually executes (or
  correctly does NOT) via the ``_run_cmd`` recording-spy idiom
  (test_verify.py:3745).

A NEW file (rather than adding to test_verify.py) per the plan's design
decision: test_verify.py contains a fragile fixture
(``TestRunScopedVerificationForwardsWorktreeToFallback``) that replaces
``_build_fallback_config`` with a signature-locked fake with no ``**kwargs``
catch-all — a dedicated file avoids that hazard.
"""

from __future__ import annotations

from pathlib import Path

from orchestrator import verify
from orchestrator.config import ModuleConfig, OrchestratorConfig

# escalation/tests/test_server.py's real shape: a guarded, indented import of
# orchestrator.merge_queue (escalation/tests/test_server.py:31-34).
_TEST_SERVER_CONTENT = (
    'from __future__ import annotations\n'
    '\n'
    'import pytest\n'
    '\n'
    'try:\n'
    '    from orchestrator.merge_queue import SpeculativeMergeWorker\n'
    'except ImportError:\n'
    '    SpeculativeMergeWorker = None\n'
)

_TEST_UNRELATED_CONTENT = (
    'from __future__ import annotations\n'
    '\n'
    'def test_unrelated():\n'
    '    assert True\n'
)

_ESCALATION_TEST_COMMAND = (
    'uv run --project escalation --directory escalation pytest tests/ --tb=short -q'
)
_ORCHESTRATOR_TEST_COMMAND = (
    'uv run --project orchestrator --directory orchestrator pytest tests/ --tb=short -q'
)


def _build_worktree(tmp_path: Path) -> Path:
    """A tmp_path worktree: orchestrator source + escalation's importing/unrelated tests."""
    (tmp_path / 'orchestrator' / 'src' / 'orchestrator').mkdir(parents=True)
    (tmp_path / 'orchestrator' / 'src' / 'orchestrator' / 'merge_queue.py').write_text(
        'class SpeculativeMergeWorker:\n    pass\n',
    )
    (tmp_path / 'escalation' / 'tests').mkdir(parents=True)
    (tmp_path / 'escalation' / 'tests' / 'test_server.py').write_text(_TEST_SERVER_CONTENT)
    (tmp_path / 'escalation' / 'tests' / 'test_unrelated.py').write_text(_TEST_UNRELATED_CONTENT)
    return tmp_path


def _build_config(tmp_path: Path) -> OrchestratorConfig:
    """An OrchestratorConfig with escalation+orchestrator ModuleConfigs set post-construction.

    ``_module_configs`` is a Pydantic PrivateAttr (config.py:2682), not a
    constructor kwarg — set via direct attribute assignment, the established
    idiom (test_verify.py:5558 et al.).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    config._module_configs = {
        'escalation': ModuleConfig(
            prefix='escalation',
            test_command=_ESCALATION_TEST_COMMAND,
            lint_command=None,
            type_check_command=None,
        ),
        'orchestrator': ModuleConfig(
            prefix='orchestrator',
            test_command=_ORCHESTRATOR_TEST_COMMAND,
            lint_command='uv run --project orchestrator --directory orchestrator ruff check src/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/'
            ),
        ),
    }
    return config


class TestReverseDependencyModuleConfigs:
    """verify._reverse_dependency_module_configs(existing_files, config, worktree, already_scoped, content_cache=None).

    RED until step-6 implements the helper (AttributeError: module
    'orchestrator.verify' has no attribute '_reverse_dependency_module_configs').
    """

    def test_orchestrator_source_change_widens_to_scoped_escalation_pytest(self, tmp_path: Path):
        """An orchestrator-source-only diff widens to ONE escalation ModuleConfig.

        pytest-only (lint/type None) and scoped to the coupled test file —
        worktree-root-relative (no '--directory'), matching
        _scope_to_keyword's cwd-stripped output shape.
        """
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)

        result = verify._reverse_dependency_module_configs(
            ['orchestrator/src/orchestrator/merge_queue.py'],
            config,
            worktree,
            already_scoped={'orchestrator'},
        )

        assert len(result) == 1
        mc = result[0]
        assert mc.prefix == 'escalation'
        assert mc.lint_command is None
        assert mc.type_check_command is None
        assert mc.test_command is not None
        assert 'escalation/tests/test_server.py' in mc.test_command
        assert 'test_unrelated.py' not in mc.test_command
        assert '--directory' not in mc.test_command

    def test_already_scoped_escalation_dedupes_to_empty(self, tmp_path: Path):
        """already_scoped already containing 'escalation' -> no widening (dedup)."""
        worktree = _build_worktree(tmp_path)
        config = _build_config(tmp_path)

        result = verify._reverse_dependency_module_configs(
            ['orchestrator/src/orchestrator/merge_queue.py'],
            config,
            worktree,
            already_scoped={'orchestrator', 'escalation'},
        )

        assert result == []
