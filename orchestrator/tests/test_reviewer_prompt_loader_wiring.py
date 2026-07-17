"""RED until TaskWorkflow accepts a ``prompt_store`` kwarg and defines
``_resolve_role_system_prompt(role, model)`` — the per-invocation
prompt-artifact-loader helper that ``_invoke`` will call at the
``system_prompt`` build seam, right after the router resolves ``model``
(task 2493 step-3/step-4). Mirrors the curator sibling's
``TestCuratorPromptLoaderWiringSingle`` (task 2494,
``fused-memory/tests/test_task_curator.py``), but exercises the resolution
helper directly rather than through the full ``_invoke``/``invoke_with_cap_retry``
boundary — ``_resolve_role_system_prompt`` only touches ``self._prompt_store``
and the passed-in ``role``/``model``, so a full worktree/git-repo harness
(as in ``test_invoke_role_config_resolution.py``) is unnecessary here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from shared.prompt_artifact import ArtifactProvenance, PromptArtifactStore, compose_prompt

from orchestrator.agents.roles import (
    _REVIEWER_PROMPT_HARNESS_VERSION,
    AgentRole,
    REVIEWER_COMPREHENSIVE,
)
from orchestrator.config import OrchestratorConfig
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow


def _provenance_kwargs(**overrides: Any) -> dict[str, Any]:
    # dict[str, Any] is intentional: ArtifactProvenance's fields have
    # heterogeneous types and spreading a concrete dict[str, <union>] defeats
    # pyright's per-parameter type checking at the ** call site (mirrors
    # fused-memory/tests/test_task_curator.py's _prompt_artifact_provenance_kwargs).
    kwargs: dict[str, Any] = dict(
        optimizer_model='claude-opus-4',
        corpus_hash='sha256:deadbeef',
        split_seed=42,
        held_out_TEST_score=0.9,
        accept_delta=0.05,
        git_sha='abc1234',
        date='2026-07-17',
        harness_version=_REVIEWER_PROMPT_HARNESS_VERSION,
    )
    kwargs.update(overrides)
    return kwargs


def _make_workflow(
    *, tmp_path: Path, prompt_store: PromptArtifactStore | None = None,
) -> TaskWorkflow:
    """Minimal TaskWorkflow builder for ``_resolve_role_system_prompt`` tests.

    ``_resolve_role_system_prompt`` only touches ``self._prompt_store`` and
    the ``role``/``model`` passed to it -- git_ops/scheduler/briefing are
    never invoked -- so this intentionally skips the real-git-repo
    ``_make_workflow`` convention used by ``_invoke``-exercising tests (e.g.
    ``test_invoke_role_config_resolution.py``) and passes bare ``MagicMock``s
    for the collaborators this helper never calls.
    """
    assignment = TaskAssignment(
        task_id='2493',
        task={'id': '2493', 'title': 'X', 'status': 'pending', 'metadata': {}},
        modules=[],
    )
    config = OrchestratorConfig(project_root=tmp_path)
    return TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=MagicMock(),
        scheduler=MagicMock(),
        briefing=MagicMock(),
        mcp=None,
        prompt_store=prompt_store,
    )


class TestResolveRoleSystemPromptUnpinned:
    """(a) Nothing pinned -> the in-code baseline (byte-identical to today)."""

    def test_nothing_pinned_returns_in_code_constant(self, tmp_path):
        store = PromptArtifactStore(tmp_path / 'artifacts')
        workflow = _make_workflow(tmp_path=tmp_path, prompt_store=store)

        result = workflow._resolve_role_system_prompt(REVIEWER_COMPREHENSIVE, 'opus')

        assert result == REVIEWER_COMPREHENSIVE.prompt_spec.in_code_constant


class TestResolveRoleSystemPromptPinnedPerModel:
    """(b) A pin for one model overrides HEURISTICS there; a different,
    unpinned model still resolves to the in-code baseline -- proving the
    *passed* ``model`` (the router-resolved executor_model) is the resolve()
    key, not some fixed/static model.
    """

    def test_pinned_model_gets_override_other_model_stays_baseline(self, tmp_path):
        store = PromptArtifactStore(tmp_path / 'artifacts')
        workflow = _make_workflow(tmp_path=tmp_path, prompt_store=store)
        spec = REVIEWER_COMPREHENSIVE.prompt_spec

        heuristics = 'PINNED: prefer flagging more suggestions.'
        provenance = ArtifactProvenance(**_provenance_kwargs())
        store.pin(
            spec.prompt_id, 'opus', _REVIEWER_PROMPT_HARNESS_VERSION,
            heuristics=heuristics, provenance=provenance,
        )

        pinned_result = workflow._resolve_role_system_prompt(REVIEWER_COMPREHENSIVE, 'opus')
        assert pinned_result == compose_prompt(spec.contract, heuristics)
        assert pinned_result.startswith(spec.contract)

        other_model_result = workflow._resolve_role_system_prompt(REVIEWER_COMPREHENSIVE, 'sonnet')
        assert other_model_result == spec.in_code_constant


class TestResolveRoleSystemPromptNoSpec:
    """(c) A role with prompt_spec=None (every role but the reviewer(s), today)
    returns role.system_prompt verbatim -- the loader is never consulted.
    """

    def test_role_without_prompt_spec_returns_system_prompt_verbatim(self, tmp_path):
        store = PromptArtifactStore(tmp_path / 'artifacts')
        workflow = _make_workflow(tmp_path=tmp_path, prompt_store=store)
        role = AgentRole(name='dummy_role', system_prompt='VERBATIM PROMPT TEXT')
        assert role.prompt_spec is None  # premise: opt-in field defaults to None

        result = workflow._resolve_role_system_prompt(role, 'opus')

        assert result == 'VERBATIM PROMPT TEXT'
