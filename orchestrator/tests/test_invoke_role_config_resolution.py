"""Tests for TaskWorkflow._invoke's per-role config resolution (routing
alpha, task 2531 / plans/adaptive-model-routing-prd.md).

``_invoke`` used to resolve per-role config with
``role_key = role.name.split('_')[0]``, which yields ``'simple'`` for
``simple_task`` -- a key that matches no field on any of the six routing
submodels (ModelsConfig/BudgetsConfig/TurnsConfig/EffortConfig/
TimeoutsConfig/BackendsConfig), so simple_task silently fell back to its
AgentRole dataclass defaults / the getattr fallbacks and could never be
tuned via ``models.simple_task`` et al. in orchestrator.yaml.

This module locks two things across that retirement:

- TestInvokeResolvesSimpleTaskBySubmodelField (RED until workflow.py's
  ``_invoke`` switches to full-name lookup): simple_task's OWN
  ``.simple_task`` submodel fields reach the ``invoke_with_cap_retry``
  boundary.
- TestInvokeReviewerVariantOverrideStaysGreen (regression lock, green
  throughout): a reviewer VARIANT role (e.g. ``reviewer_comprehensive``)
  whose full name matches no submodel field still resolves every field to
  the shared ``.reviewer`` config value via the pre-existing
  ``if role.name.startswith('reviewer')`` override -- which transitions
  from redundant (under the old split-derivation, 'reviewer_comprehensive'.
  split('_')[0] == 'reviewer' already matched) to load-bearing (under
  full-name lookup, the override is the ONLY thing that still resolves it).

Mirrors the TaskWorkflow-construction + invoke_with_cap_retry-patch pattern
from test_agent_capability_wiring.py's ``_invoke_probe`` (including its
git_repo/git_ops/task_assignment fixture trio) -- per-test-file duplication
of this small fixture set is the established convention (see that file's
own fixture-block comment / test_steward_outcome.py's ``_make_workflow``
docstring).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE, SIMPLE_TASK
from orchestrator.config import (
    BackendsConfig,
    BudgetsConfig,
    EffortConfig,
    GitConfig,
    ModelsConfig,
    OrchestratorConfig,
    TimeoutsConfig,
    TurnsConfig,
    load_config,
)
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# ---------------------------------------------------------------------------
# Fixtures -- mirrors test_agent_capability_wiring.py's git_repo/git_ops/
# task_assignment fixtures. git_ops is deliberately built from its own fixed
# GitConfig (not from the per-test OrchestratorConfig) since each test below
# constructs a DIFFERENT OrchestratorConfig (distinct routing-submodel
# sentinels) but they all share the same underlying git repo/worktree setup.
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )
    return GitOps(git_config, git_repo)


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


async def _invoke_probe(role, config, git_ops, task_assignment):
    """Build a TaskWorkflow, patch invoke_with_cap_retry, invoke workflow._invoke(role, ...).

    Mirrors test_agent_capability_wiring.py::_invoke_probe. Returns
    (call_kwargs, workflow) -- the kwargs invoke_with_cap_retry was awaited
    with, and the workflow instance.
    """
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path

    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
    )
    workflow.artifacts = None

    with patch(
        'orchestrator.workflow.invoke_with_cap_retry',
        new_callable=AsyncMock,
        return_value=AgentResult(success=True, output=''),
    ) as mock_cap_retry:
        await workflow._invoke(role, 'p', cwd)

    assert mock_cap_retry.await_count == 1, 'invoke_with_cap_retry must be called once'
    assert mock_cap_retry.await_args is not None, 'await_args must be set after one await'
    return mock_cap_retry.await_args.kwargs, workflow


def _base_config_kwargs(git_repo: Path) -> dict:
    return dict(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.mark.asyncio
class TestInvokeResolvesSimpleTaskBySubmodelField:
    """RED: role_key = role.name.split('_')[0] yields 'simple' for
    SIMPLE_TASK ('simple_task'.split('_')[0]) -- a key that matches no
    field on any routing submodel -- so these distinct per-field sentinels
    never reach the invoke_with_cap_retry boundary today; _invoke instead
    falls back to SIMPLE_TASK's AgentRole dataclass defaults / the getattr
    fallback literals. Fails until workflow.py's _invoke switches to
    full-name lookup (role_key = role.name).
    """

    async def test_simple_task_resolves_own_submodel_fields(
        self, git_repo, git_ops, task_assignment,
    ):
        config = OrchestratorConfig(
            **_base_config_kwargs(git_repo),
            models=ModelsConfig(simple_task='haiku'),
            budgets=BudgetsConfig(simple_task=9.9),
            max_turns=TurnsConfig(simple_task=7),
            effort=EffortConfig(simple_task='low'),
            timeouts=TimeoutsConfig(simple_task=123.0),
            backends=BackendsConfig(simple_task='codex'),
        )

        call_kwargs, _workflow = await _invoke_probe(
            SIMPLE_TASK, config, git_ops, task_assignment,
        )

        assert call_kwargs.get('model') == 'haiku', (
            f"Expected model=='haiku' (models.simple_task) but got "
            f"{call_kwargs.get('model')!r}. _invoke must resolve simple_task's "
            "OWN submodel field, not fall back via a split('_')[0] derivation."
        )
        assert call_kwargs.get('max_budget_usd') == 9.9, (
            f"Expected max_budget_usd==9.9 (budgets.simple_task) but got "
            f"{call_kwargs.get('max_budget_usd')!r}."
        )
        assert call_kwargs.get('max_turns') == 7, (
            f"Expected max_turns==7 (max_turns.simple_task) but got "
            f"{call_kwargs.get('max_turns')!r}."
        )
        assert call_kwargs.get('effort') == 'low', (
            f"Expected effort=='low' (effort.simple_task) but got "
            f"{call_kwargs.get('effort')!r}."
        )
        assert call_kwargs.get('timeout_seconds') == 123.0, (
            f"Expected timeout_seconds==123.0 (timeouts.simple_task) but got "
            f"{call_kwargs.get('timeout_seconds')!r}."
        )
        assert call_kwargs.get('backend') == 'codex', (
            f"Expected backend=='codex' (backends.simple_task) but got "
            f"{call_kwargs.get('backend')!r}."
        )


@pytest.mark.asyncio
class TestInvokeReviewerVariantOverrideStaysGreen:
    """Regression lock: a reviewer VARIANT role's full name (e.g.
    'reviewer_comprehensive') matches no submodel field under full-name
    lookup -- only the bare 'reviewer' key exists -- so the pre-existing
    `if role.name.startswith('reviewer')` override in _invoke is what keeps
    resolving every field to the shared `.reviewer` config value. Must stay
    green both BEFORE and AFTER the role_key derivation changes (before:
    the override is redundant with the split-derivation's own
    'reviewer_comprehensive'.split('_')[0] == 'reviewer' match; after: the
    override is the only thing keeping this correct).
    """

    async def test_reviewer_variant_resolves_reviewer_submodel_fields(
        self, git_repo, git_ops, task_assignment,
    ):
        config = OrchestratorConfig(
            **_base_config_kwargs(git_repo),
            models=ModelsConfig(reviewer='reviewer-sentinel-model'),
            budgets=BudgetsConfig(reviewer=42.0),
            max_turns=TurnsConfig(reviewer=99),
            effort=EffortConfig(reviewer='max'),
            timeouts=TimeoutsConfig(reviewer=555.0),
            backends=BackendsConfig(reviewer='gemini'),
        )

        call_kwargs, _workflow = await _invoke_probe(
            REVIEWER_COMPREHENSIVE, config, git_ops, task_assignment,
        )

        assert call_kwargs.get('model') == 'reviewer-sentinel-model'
        assert call_kwargs.get('max_budget_usd') == 42.0
        assert call_kwargs.get('max_turns') == 99
        assert call_kwargs.get('effort') == 'max'
        assert call_kwargs.get('timeout_seconds') == 555.0
        assert call_kwargs.get('backend') == 'gemini'


# ---------------------------------------------------------------------------
# Capstone -- PRD boundary test 4 (plans/adaptive-model-routing-prd.md):
# setting models.simple_task via a REAL hot reload must both (a) land in the
# reload's `applied` disposition -- not `restart_required` -- and (b) flip
# the model the NEXT simple_task _invoke resolves. This is the end-to-end,
# user-observable signal the whole task exists to deliver; it only passes
# with step-2 (submodel field exists, so it round-trips through
# RELOADABLE_FIELDS/YAML) and step-4 (full-name lookup, so _invoke actually
# reads simple_task's own field) both in place.
#
# Mirrors test_config_reload_integration_gate.py's _make_reload_harness /
# TestS2AllowlistedNextSpawn pattern (a real on-disk orchestrator.yaml read
# via load_config() + Harness.reload_config(), never mocking load_config) --
# duplicated module-locally per that file's documented
# per-test-file-duplication convention (see this file's own top-of-module
# docstring) -- but asserts at the invoke_with_cap_retry boundary (this
# file's own established pattern, per _invoke_probe above) rather than a
# StubInvoke.invoke_agent capture.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCapstoneHotReloadFlipsSimpleTaskModel:
    """PRD boundary test 4: a real hot reload of models.simple_task applies
    (not restart_required) and is observed by the next simple_task _invoke.
    """

    async def test_reload_models_simple_task_applies_and_flips_next_invoke(
        self, tmp_path: Path, monkeypatch, git_repo, git_ops, task_assignment,
    ) -> None:
        config_path = tmp_path / 'orchestrator.yaml'
        config_path.write_text(yaml.safe_dump({}))
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))

        config = load_config()
        assert config.models.simple_task == 'sonnet', (
            'must start from the stock default so the reload below is an '
            'observable change'
        )

        harness = Harness(config)
        # Inject a mock RunStore (no persistence side effects) and a real
        # EventStore pointing at a tmp DB -- same injection convention as
        # test_config_reload_integration_gate.py::_make_reload_harness.
        harness._run_store = MagicMock(spec=RunStore)
        harness._run_id = 'run-test-0001'
        harness.event_store = EventStore(tmp_path / 'events.db', 'run-test-0001')

        # Edit the on-disk YAML: models.simple_task 'sonnet' -> 'haiku'.
        data = yaml.safe_load(config_path.read_text()) or {}
        data.setdefault('models', {})['simple_task'] = 'haiku'
        config_path.write_text(yaml.safe_dump(data))

        report = await harness.reload_config()

        assert report['error'] is None
        assert report['applied'].get('models.simple_task') == {
            'old': 'sonnet', 'new': 'haiku',
        }, (
            f"Expected 'models.simple_task' in the applied disposition (not "
            f"restart_required); got applied={report['applied']!r} "
            f"restart_required={report['restart_required']!r}"
        )
        assert 'models.simple_task' not in report['restart_required']
        assert harness.config.models.simple_task == 'haiku'

        # workflow shares harness.config BY REFERENCE (I3 identity
        # preservation) -- the reload mutated it in place, so the workflow
        # observes 'haiku' on its NEXT _invoke with no extra wiring.
        workflow = TaskWorkflow(
            assignment=task_assignment,
            config=harness.config,
            git_ops=git_ops,
            scheduler=FakeScheduler(),  # type: ignore[arg-type]
            briefing=FakeBriefing(),  # type: ignore[arg-type]
            mcp=FakeMcp(),  # type: ignore[arg-type]
        )
        workflow.artifacts = None

        wt_info = await git_ops.create_worktree(task_assignment.task_id)

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ) as mock_cap_retry:
            await workflow._invoke(SIMPLE_TASK, 'p', wt_info.path)

        assert mock_cap_retry.await_count == 1
        assert mock_cap_retry.await_args is not None, 'await_args must be set after one await'
        assert mock_cap_retry.await_args.kwargs.get('model') == 'haiku', (
            'the next simple_task _invoke after a hot reload must resolve '
            'the freshly-reloaded models.simple_task value, not a stale one'
        )
