"""Merge-skew β (task 2383, M2 of plans/merge-skew-attribution-prd.md): wire
alpha's classify_merge_failure_disposition into the production merge-gate
failure path + every I4 surface.

Boundary rows as tests:
  row1 (steps 1-2):   MAIN_RED — _classify_main_health_red stamps disposition.
  I4 surfacing (steps 3-4): _render_skew_surfaces pure helper.
  rows 2/3/4 (steps 5-6):   _classify_disposition_for_outcome async wrapper.
  I4 on the real outcome (steps 7-8): _run_post_merge_verify wiring.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps
from orchestrator.merge_disposition import MergeFailureDisposition, SkewEvidence
from orchestrator.merge_queue import (
    MAIN_HEALTH_RED_REASON_PREFIX,
    MergeOutcome,
    MergeRequest,
    _classify_main_health_red,
    _run_post_merge_verify,
)
from orchestrator.verify import VerifyResult

MAIN_SHA = 'cafecafe1234567890deadbeef'

COMPILE_ERROR_RESULT = VerifyResult(
    passed=False,
    test_output='',
    lint_output='',
    type_output='error TS2322: StatusBar.tsx:12',
    summary='tsc failed',
    cause_hint='error TS2322: StatusBar.tsx',
    category='compile_error',
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_probe_cache():
    """Clear the process-wide _PROBE_CACHE between tests (mirrors
    test_merge_queue_main_health.py's fixture)."""
    from orchestrator.verify import _PROBE_CACHE
    _PROBE_CACHE.clear()
    yield
    _PROBE_CACHE.clear()


def _make_config(tmp_path: Path) -> OrchestratorConfig:
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


def _make_git_ops(tmp_path: Path) -> GitOps:
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path
    git_ops.cleanup_merge_worktree = AsyncMock(return_value=None)
    git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)
    return git_ops


def _make_req(
    task_id: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=f'task/{task_id}',
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane='normal',
    )


# ---------------------------------------------------------------------------
# Step-1/2 [boundary row 1 — MAIN_RED]
# ---------------------------------------------------------------------------


class TestClassifyMainHealthRedSetsDisposition:
    """_classify_main_health_red must stamp disposition=MAIN_RED on the
    outcome it returns when the preexisting probe confirms True (I1: probe
    order preserved; the classifier is never invoked for this bucket), while
    the fix-main reason prefix and dedupe_fingerprint stay unchanged."""

    def test_disposition_is_main_red(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(True, MAIN_SHA)),
            ):
                return await _classify_main_health_red(git_ops, req, COMPILE_ERROR_RESULT)

        outcome = asyncio.run(_run())
        assert outcome is not None
        assert outcome.disposition == MergeFailureDisposition.MAIN_RED, (
            f'Expected disposition=MAIN_RED; got {outcome.disposition!r}'
        )
        assert outcome.reason.startswith(MAIN_HEALTH_RED_REASON_PREFIX), (
            f'fix-main reason prefix must stay unchanged; got {outcome.reason!r}'
        )
        from orchestrator.workflow import compute_preexisting_main_break_fingerprint
        expected_fp = compute_preexisting_main_break_fingerprint(
            'compile_error', 'error TS2322: StatusBar.tsx', MAIN_SHA,
        )
        assert outcome.dedupe_fingerprint == expected_fp, (
            f'dedupe_fingerprint must stay unchanged; '
            f'expected={expected_fp!r} got={outcome.dedupe_fingerprint!r}'
        )

    def test_negative_classification_stays_indeterminate(self, tmp_path: Path) -> None:
        """When probe returns (False, ''), _classify_main_health_red returns
        None (falls through) — disposition is not this function's concern."""
        config = _make_config(tmp_path)
        git_ops = _make_git_ops(tmp_path)
        req = _make_req('99', tmp_path / 'task-wt', config)
        (tmp_path / 'task-wt').mkdir()

        async def _run() -> MergeOutcome | None:
            with patch(
                'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                new=AsyncMock(return_value=(False, '')),
            ):
                return await _classify_main_health_red(git_ops, req, COMPILE_ERROR_RESULT)

        outcome = asyncio.run(_run())
        assert outcome is None


# ---------------------------------------------------------------------------
# Step-3/4 [I4 surfacing content] — _render_skew_surfaces pure helper
# ---------------------------------------------------------------------------


class TestRenderSkewSurfaces:
    """_render_skew_surfaces(disposition, evidence) is a pure helper: only
    INTEGRATION_SKEW (with non-None evidence) yields a non-empty reason_suffix
    + failure_diagnostic dict; every other disposition/evidence combination
    returns ('', None)."""

    def test_integration_skew_with_evidence_renders_surfaces(self) -> None:
        from orchestrator.merge_queue import _render_skew_surfaces

        evidence = SkewEvidence(
            implicated_commits=('abc123deadbeef',),
            failing_tests=('tests/test_foo.py::test_bar',),
            overlap_files=('a/b.py',),
        )
        reason_suffix, failure_diagnostic = _render_skew_surfaces(
            MergeFailureDisposition.INTEGRATION_SKEW, evidence,
        )

        assert 'integration_skew' in reason_suffix, reason_suffix
        assert 'abc123deadbeef' in reason_suffix, reason_suffix
        assert 'a/b.py' in reason_suffix, reason_suffix
        assert 'port landed commit' in reason_suffix, reason_suffix
        assert 'do not hunt your own diff' in reason_suffix, reason_suffix

        assert failure_diagnostic is not None
        assert all(isinstance(v, str) for v in failure_diagnostic.values()), (
            f'failure_diagnostic must be dict[str,str]; got {failure_diagnostic!r}'
        )
        joined = ' '.join(failure_diagnostic.values())
        assert 'integration_skew' in joined, joined
        assert 'abc123deadbeef' in joined, joined
        assert 'a/b.py' in joined, joined
        assert 'tests/test_foo.py::test_bar' in joined, joined

    @pytest.mark.parametrize('disposition', [
        MergeFailureDisposition.MAIN_RED,
        MergeFailureDisposition.BRANCH_BUG,
        MergeFailureDisposition.INDETERMINATE,
    ])
    def test_non_skew_dispositions_render_nothing(
        self, disposition: MergeFailureDisposition,
    ) -> None:
        from orchestrator.merge_queue import _render_skew_surfaces

        evidence = SkewEvidence(
            implicated_commits=('abc123',),
            failing_tests=('t',),
            overlap_files=('f.py',),
        )
        assert _render_skew_surfaces(disposition, evidence) == ('', None)

    def test_integration_skew_with_none_evidence_renders_nothing(self) -> None:
        from orchestrator.merge_queue import _render_skew_surfaces

        assert _render_skew_surfaces(
            MergeFailureDisposition.INTEGRATION_SKEW, None,
        ) == ('', None)
