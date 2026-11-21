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
``_verify_debugfix_loop`` / ``_review`` stubbed, via the shared
``orchestrator/tests/_review_loop_harness.py::review_loop_workflow`` harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from _review_loop_harness import review_loop_workflow

from orchestrator.artifacts import ReviewAggregation
from orchestrator.review_suggestions.disposition import SuggestionDisposition
from orchestrator.workflow import WorkflowOutcome


@pytest.mark.asyncio
class TestVerdictCacheSkip:
    """A pre-recorded non-blocking verdict for the current tree SKIPs REVIEW."""

    async def test_cached_verdict_skips_review_and_routing(
        self, tmp_path: Path
    ):
        wf = review_loop_workflow(tmp_path)
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
        wf = review_loop_workflow(tmp_path)
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
        wf = review_loop_workflow(tmp_path)
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
        wf = review_loop_workflow(tmp_path)
        fp1 = wf._reviewer_config_fingerprint()
        fp2 = wf._reviewer_config_fingerprint()
        assert fp1 is not None
        assert fp1 == fp2

    async def test_model_repin_changes_fingerprint(self, tmp_path: Path):
        # A per-role model re-pin (config.models change) must change the
        # digest so a byte-identical tree re-reviews.
        wf = review_loop_workflow(tmp_path)
        fp_before = wf._reviewer_config_fingerprint()
        wf.config.models.reviewer = 'opus'
        fp_after = wf._reviewer_config_fingerprint()
        assert fp_before is not None and fp_after is not None
        assert fp_before != fp_after

    async def test_metadata_override_changes_fingerprint(self, tmp_path: Path):
        # A metadata.model_overrides pin on a reviewer role must change the
        # digest (the reviewer's named scenario).
        wf = review_loop_workflow(tmp_path)
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
        wf = review_loop_workflow(tmp_path)
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
        wf = review_loop_workflow(tmp_path)
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
        wf = review_loop_workflow(tmp_path, max_review_cycles=1)
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

        outcome = await wf._execute_verify_review_loop()

        # max_review_cycles=1 → escalate on the first blocking cycle,
        # before any replan.
        assert outcome == WorkflowOutcome.ESCALATED
        cast(AsyncMock, wf._replan).assert_not_called()
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
        wf = review_loop_workflow(tmp_path, max_amendment_rounds=1)
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
        wf = review_loop_workflow(tmp_path, max_amendment_rounds=1)
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
        wf = review_loop_workflow(tmp_path, max_review_cycles=1)
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

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.ESCALATED
        cast(AsyncMock, wf._replan).assert_not_called()
        assert wf.artifacts is not None
        assert wf.artifacts.get_review_cycles_total() >= 1

    async def test_cap_exit_suppresses_suggestions_settled_in_a_prior_amendment_round(
        self, tmp_path: Path
    ):
        # A prior dispatch spent the amendment round and the review cycle, so
        # this blocking review hits the cap exit with a re-flag of a concern
        # that round settled.  The cap exit must suppress it exactly as the
        # DONE exit would, rather than file it again and count it.
        wf = review_loop_workflow(tmp_path, max_review_cycles=1)
        wf.config.suppress_resettled_review_suggestions = True
        assert wf.artifacts is not None
        wf.artifacts.set_review_counters(amendment_rounds_total=1, review_cycles_total=1)
        settled = {
            'severity': 'suggestion',
            'location': 'src/foo/helpers.py:3',
            'category': 'naming',
            'description': 'helper name hides its side effect',
            'suggested_fix': 'rename helper',
        }
        archive = wf.artifacts.review_archive_dir('amend-1')
        archive.mkdir()
        (archive / 'analyst.json').write_text(
            json.dumps({'verdict': 'ISSUES_FOUND', 'issues': [settled]})
        )
        reflag = {
            **settled,
            'reviewer': 'analyst',
            'description': 'helper is still named as if it were pure',
        }
        fresh = {
            'reviewer': 'analyst',
            'severity': 'suggestion',
            'location': 'src/foo/batch.py:9',
            'category': 'coverage',
            'description': 'no test covers the empty batch',
            'suggested_fix': 'add an empty-batch test',
        }
        wf._review = AsyncMock(
            return_value=ReviewAggregation(
                has_blocking_issues=True,
                blocking_issues=[{
                    'reviewer': 'analyst',
                    'category': 'bug',
                    'description': 'boom',
                }],
                suggestions=[reflag, fresh],
                reviews={'analyst': {}},
            )
        )
        wf._invoke = AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                timed_out=False,
                structured_output={'decisions': [
                    {'index': 0, 'decision': 'settled'},
                    {'index': 1, 'decision': 'not_settled'},
                ]},
            )
        )
        queue = MagicMock()
        queue.make_id.return_value = 'esc-2749-1'
        wf.escalation_queue = queue
        route = cast(AsyncMock, wf._route_review_suggestions_to_curator)
        route.return_value = SuggestionDisposition.CURATOR

        outcome = await wf._execute_verify_review_loop()

        assert outcome == WorkflowOutcome.ESCALATED
        route.assert_awaited_once()
        assert route.await_args is not None
        assert route.await_args.args[0].suggestions == [fresh]
        esc = queue.submit.call_args[0][0]
        assert fresh['description'] in esc.detail
        assert reflag['description'] not in esc.detail
        assert '1 suggestion(s) (+1 scoped out' in esc.summary
