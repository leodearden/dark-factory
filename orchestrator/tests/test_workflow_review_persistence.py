"""Tests for cross-dispatch review/amendment persistence (task 2749).

Two mechanisms drive ``_execute_verify_review_loop``:

1. A tree-hash-keyed verdict cache (``review_state.json`` via
   ``TaskArtifacts``): on entering REVIEW, if a non-blocking verdict is
   already recorded for HEAD's committed tree hash, the reviewer invocation
   is SKIPPED entirely and the loop takes the DONE path WITHOUT re-routing
   suggestions.
2. Task-lifetime counters (``amendment_rounds_total`` /
   ``review_cycles_total``): the loop seeds its per-dispatch locals from the
   persisted totals so ``max_amendment_rounds`` / ``max_review_cycles`` bound
   the whole task lifetime, not each dispatch.

These tests drive the loop with ``_execute_iterations`` /
``_verify_debugfix_loop`` / ``_review`` stubbed, mirroring the
``_make_workflow`` harness in ``test_workflow.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _orch_helpers import pydantic_spec

from orchestrator.artifacts import ReviewAggregation, TaskArtifacts
from orchestrator.config import OrchestratorConfig
from orchestrator.workflow import (
    TaskWorkflow,
    WorkflowOutcome,
)


def _make_workflow(
    *,
    tmp_path: Path,
    task_id: str = '2749',
    modules: list[str] | None = None,
    max_amendment_rounds: int = 1,
    max_review_cycles: int = 2,
) -> TaskWorkflow:
    """Minimal TaskWorkflow harness for driving _execute_verify_review_loop."""
    assignment = MagicMock()
    assignment.task_id = task_id
    assignment.task = {'id': task_id, 'title': 'T', 'description': 'd'}
    assignment.modules = modules if modules is not None else ['src/foo']

    _spec = pydantic_spec(OrchestratorConfig)
    config = MagicMock(spec_set=_spec)
    config.fused_memory.project_id = 'dark_factory'
    config.fused_memory.url = 'http://localhost:8002'
    config.max_review_cycles = max_review_cycles
    config.max_amendment_rounds = max_amendment_rounds
    config.lock_depth = 2
    config.project_root = tmp_path / 'proj'
    # Real reviewer model string so _reviewer_config_fingerprint() (task 2749
    # amendment) yields a deterministic, non-None digest instead of choking on
    # an unserialisable auto-child MagicMock.  The fingerprint reads the
    # reviewer* → 'reviewer' collapsed config key (the resolver's _config_key).
    config.models = MagicMock()
    config.models.reviewer = 'sonnet'

    scheduler = MagicMock()
    git_ops = MagicMock()

    wf = TaskWorkflow(
        assignment=assignment,
        config=config,
        git_ops=git_ops,
        scheduler=scheduler,
        briefing=MagicMock(),
        mcp=MagicMock(),
    )
    worktree = tmp_path / 'wt'
    worktree.mkdir(parents=True, exist_ok=True)
    artifacts = TaskArtifacts(worktree)
    artifacts.init(task_id, 'T', 'd')
    wf.artifacts = artifacts
    wf.worktree = worktree

    # Loop sub-steps default to success; the REVIEW branch is what each test
    # exercises, so stub EXECUTE/VERIFY to DONE.
    wf._execute_iterations = AsyncMock(return_value=WorkflowOutcome.DONE)
    wf._verify_debugfix_loop = AsyncMock(return_value=WorkflowOutcome.DONE)

    # Tree-hash source: a stable committed tree hash.
    wf.git_ops.get_head_tree_hash = AsyncMock(return_value='TREE1')

    # Spies on the REVIEW-branch side effects.
    wf._review = AsyncMock(
        return_value=ReviewAggregation(
            has_blocking_issues=False,
            blocking_issues=[],
            suggestions=[],
            reviews={},
        )
    )
    wf._route_review_suggestions_to_curator = AsyncMock()
    wf._write_suggestions_to_memory = AsyncMock()
    wf._amend = AsyncMock(return_value=True)
    return wf


@pytest.mark.asyncio
class TestVerdictCacheSkip:
    """A pre-recorded non-blocking verdict for the current tree SKIPs REVIEW."""

    async def test_cached_verdict_skips_review_and_routing(
        self, tmp_path: Path
    ):
        wf = _make_workflow(tmp_path=tmp_path)
        # Simulate a prior dispatch that already recorded a suggestions_only
        # verdict (and routed its suggestions) for this committed tree — under
        # the SAME reviewer config (fingerprint matches), so the skip fires.
        assert wf.artifacts is not None
        wf.artifacts.record_review_verdict(
            'TREE1', 'suggestions_only', True,
            reviewer_fingerprint=wf._reviewer_config_fingerprint(),
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        # The skip path re-mints nothing: no reviewer, no re-routing, no amend.
        cast(AsyncMock, wf._review).assert_not_called()
        cast(AsyncMock, wf._route_review_suggestions_to_curator).assert_not_called()
        cast(AsyncMock, wf._write_suggestions_to_memory).assert_not_called()
        cast(AsyncMock, wf._amend).assert_not_called()


@pytest.mark.asyncio
class TestReviewerConfigFingerprint:
    """A cached verdict is honoured only when the reviewer-config fingerprint
    still matches (task 2749 amendment) — a roster/model change forces a
    fresh review even on a byte-identical committed tree.
    """

    async def test_stale_fingerprint_forces_review(self, tmp_path: Path):
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None
        # A verdict recorded under a DIFFERENT reviewer config (e.g. a model
        # re-pin) — the fingerprint no longer matches the current config.
        wf.artifacts.record_review_verdict(
            'TREE1', 'suggestions_only', True,
            reviewer_fingerprint='STALE_CONFIG_FINGERPRINT',
        )
        # Current review: non-blocking, no suggestions → PASS.
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=False,
                blocking_issues=[],
                suggestions=[],
                reviews={'analyst': {}},
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        # The stale fingerprint must NOT short-circuit — the reviewer runs.
        cast(AsyncMock, wf._review).assert_awaited_once()
        # …and the verdict is re-recorded under the CURRENT fingerprint.
        rec = wf.artifacts.get_cached_verdict('TREE1')
        assert rec is not None
        assert rec['reviewer_fingerprint'] == wf._reviewer_config_fingerprint()

    async def test_missing_fingerprint_forces_review(self, tmp_path: Path):
        # A pre-amendment record (no reviewer_fingerprint field) can never
        # satisfy the positive-match requirement → full review, not a skip.
        wf = _make_workflow(tmp_path=tmp_path)
        assert wf.artifacts is not None
        wf.artifacts.record_review_verdict('TREE1', 'suggestions_only', True)
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=False,
                blocking_issues=[],
                suggestions=[],
                reviews={'analyst': {}},
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        cast(AsyncMock, wf._review).assert_awaited_once()

    async def test_fingerprint_stable_across_calls(self, tmp_path: Path):
        # The digest must be deterministic within a fixed config, or the
        # cross-dispatch cache would never hit.
        wf = _make_workflow(tmp_path=tmp_path)
        fp1 = wf._reviewer_config_fingerprint()
        fp2 = wf._reviewer_config_fingerprint()
        assert fp1 is not None
        assert fp1 == fp2

    async def test_model_repin_changes_fingerprint(self, tmp_path: Path):
        # A per-role model re-pin (config.models change) must change the
        # digest so a byte-identical tree re-reviews.
        wf = _make_workflow(tmp_path=tmp_path)
        fp_before = wf._reviewer_config_fingerprint()
        wf.config.models.reviewer = 'opus'
        fp_after = wf._reviewer_config_fingerprint()
        assert fp_before is not None and fp_after is not None
        assert fp_before != fp_after

    async def test_metadata_override_changes_fingerprint(self, tmp_path: Path):
        # A metadata.model_overrides pin on a reviewer role must change the
        # digest (the reviewer's named scenario).
        wf = _make_workflow(tmp_path=tmp_path)
        fp_before = wf._reviewer_config_fingerprint()
        wf.task['metadata'] = {
            'model_overrides': {'reviewer_comprehensive': 'opus'}
        }
        fp_after = wf._reviewer_config_fingerprint()
        assert fp_before is not None and fp_after is not None
        assert fp_before != fp_after


_OUT_OF_SCOPE_SUGGESTION = {
    'reviewer': 'analyst',
    'severity': 'suggestion',
    'location': 'zzz/other.py:5',  # not under modules=['src/foo'] → out of scope
    'category': 'coverage',
    'description': 'Missing edge case',
    'suggested_fix': 'Add a test',
}


@pytest.mark.asyncio
class TestVerdictRecording:
    """The non-blocking DONE path records a tree-hash-keyed verdict."""

    async def test_suggestions_only_verdict_recorded(self, tmp_path: Path):
        wf = _make_workflow(tmp_path=tmp_path)
        # One OUT-OF-scope suggestion: no amend (nothing in-scope), routed to
        # the curator, DONE.
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=False,
                blocking_issues=[],
                suggestions=[_OUT_OF_SCOPE_SUGGESTION],
                reviews={'analyst': {}},
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        cast(AsyncMock, wf._route_review_suggestions_to_curator).assert_awaited_once()
        cast(AsyncMock, wf._amend).assert_not_called()
        assert wf.artifacts is not None
        rec = wf.artifacts.get_cached_verdict('TREE1')
        assert rec is not None
        assert rec['verdict'] == 'suggestions_only'
        assert rec['suggestions_routed'] is True

    async def test_pass_verdict_recorded(self, tmp_path: Path):
        wf = _make_workflow(tmp_path=tmp_path)
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=False,
                blocking_issues=[],
                suggestions=[],
                reviews={'analyst': {}},
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        cast(AsyncMock, wf._write_suggestions_to_memory).assert_awaited_once()
        assert wf.artifacts is not None
        rec = wf.artifacts.get_cached_verdict('TREE1')
        assert rec is not None
        assert rec['verdict'] == 'PASS'
        assert rec['suggestions_routed'] is False

    async def test_blocking_verdict_not_recorded(self, tmp_path: Path):
        wf = _make_workflow(tmp_path=tmp_path, max_review_cycles=1)
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=True,
                blocking_issues=[{
                    'reviewer': 'analyst',
                    'category': 'bug',
                    'description': 'boom',
                }],
                suggestions=[],
                reviews={'analyst': {}},
            )
        )
        wf._escalate_review_issues = MagicMock()
        wf._replan = AsyncMock()

        outcome = await wf._execute_verify_review_loop()

        # max_review_cycles=1 → escalate on the first blocking cycle,
        # before any replan.
        assert outcome == WorkflowOutcome.ESCALATED
        wf._replan.assert_not_called()
        # Blocking verdicts are NEVER cached.
        assert wf.artifacts is not None
        assert wf.artifacts.get_cached_verdict('TREE1') is None


_IN_SCOPE_SUGGESTION = {
    'reviewer': 'analyst',
    'severity': 'suggestion',
    'location': 'src/foo/bar.py:42',  # under modules=['src/foo'] → in scope
    'category': 'coverage',
    'description': 'Missing edge case',
    'suggested_fix': 'Add a test',
}


@pytest.mark.asyncio
class TestLifetimeCounters:
    """The loop seeds its per-dispatch locals from the persisted totals, so
    the amendment/review caps bound the WHOLE task lifetime.
    """

    async def test_amendment_counter_seeded_and_persisted(self, tmp_path: Path):
        # Fresh task (amendment_rounds_total=0), one in-scope suggestion.
        wf = _make_workflow(tmp_path=tmp_path, max_amendment_rounds=1)
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=False,
                blocking_issues=[],
                suggestions=[_IN_SCOPE_SUGGESTION],
                reviews={'analyst': {}},
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        # Exactly one amend, then the cap is hit on the re-loop.
        assert cast(AsyncMock, wf._amend).await_count == 1
        # The increment was persisted as a task-lifetime total.
        assert wf.artifacts is not None
        assert wf.artifacts.get_amendment_rounds_total() == 1

    async def test_amendment_cap_exhausted_across_lifetime(self, tmp_path: Path):
        # Pre-seed the persisted total at the cap — a prior dispatch already
        # used the one allowed amendment round.
        wf = _make_workflow(tmp_path=tmp_path, max_amendment_rounds=1)
        assert wf.artifacts is not None
        wf.artifacts.set_review_counters(amendment_rounds_total=1)
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=False,
                blocking_issues=[],
                suggestions=[_IN_SCOPE_SUGGESTION],
                reviews={'analyst': {}},
            )
        )

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.DONE
        # Cap already exhausted for the task lifetime → no fresh amendment.
        cast(AsyncMock, wf._amend).assert_not_called()
        cast(AsyncMock, wf._route_review_suggestions_to_curator).assert_awaited_once()

    async def test_review_cycle_cap_exhausted_across_lifetime(
        self, tmp_path: Path
    ):
        # Pre-seed the persisted review-cycle total at the cap; a blocking
        # review must escalate without a fresh replan.
        wf = _make_workflow(tmp_path=tmp_path, max_review_cycles=1)
        assert wf.artifacts is not None
        wf.artifacts.set_review_counters(review_cycles_total=1)
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=True,
                blocking_issues=[{
                    'reviewer': 'analyst',
                    'category': 'bug',
                    'description': 'boom',
                }],
                suggestions=[],
                reviews={'analyst': {}},
            )
        )
        wf._escalate_review_issues = MagicMock()
        wf._replan = AsyncMock()

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.ESCALATED
        wf._replan.assert_not_called()
        assert wf.artifacts is not None
        assert wf.artifacts.get_review_cycles_total() >= 1
