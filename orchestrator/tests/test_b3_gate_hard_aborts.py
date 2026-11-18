"""Tests for the entries orchestrator.b3_gate.check_proposal must abort, or must
deliberately not abort, whatever their freshness.

Covers: the post-merge red-main hard-abort, the explicit stop-instruction
hard-abort and the captured-output exclusion from that scan, the typed
block_class dual-read, and the merge-completion consumption contract.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from _agent_result_fakes import make_agent_result
from _b3_gate_fixtures import (
    RECORDED_HEAD_SHA,
    RECORDED_MAIN_SHA,
    fake_git_fresh,
    fake_git_never_called,
    low_risk_entry,
)

# ---------------------------------------------------------------------------
# step-1 (task 1680): post-merge red-main hard-abort in check_proposal
# ---------------------------------------------------------------------------

class TestPostMergeRedMainAbort:
    """check_proposal hard-ABORTs any entry whose block_reason starts with
    POST_MERGE_RED_MAIN_REASON_PREFIX — before risk_label and git checks.
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)
    # Prefix used in all cases (constant not yet defined — tests fail initially)
    _PM_REASON = 'Post-merge unscoped type-check failed on main: foo.py (error E001)'

    def test_low_risk_post_merge_reason_aborts_without_git(self):
        """Case 1: low-risk entry with post-merge block_reason -> ABORT, no git called."""
        from orchestrator.b3_gate import ABORT, POST_MERGE_RED_MAIN_REASON_PREFIX, check_proposal
        entry = low_risk_entry(block_reason=self._PM_REASON)
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        reason_lower = result['reason'].lower()
        assert 'post-merge' in reason_lower or 'red-main' in reason_lower, (
            f'expected "post-merge" or "red-main" in reason: {result["reason"]!r}'
        )
        # Sanity: prefix match is correct
        assert self._PM_REASON.startswith(POST_MERGE_RED_MAIN_REASON_PREFIX)

    def test_human_review_post_merge_reason_also_aborts(self):
        """Case 2: risk_label='human-review-required' with post-merge reason -> ABORT.

        Documents that the post-merge abort branch fires regardless of risk_label.
        """
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(
            risk_label='human-review-required',
            block_reason=self._PM_REASON,
        )
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, (
            f'expected ABORT for human-review-required post-merge entry, got {result}'
        )

    def test_non_post_merge_reason_not_aborted_by_post_merge_branch(self):
        """Case 3 (regression): block_reason='verify exhausted' does NOT trigger the
        post-merge abort — reaches FRESH exactly as before this change.
        """
        from orchestrator.b3_gate import FRESH, check_proposal
        entry = low_risk_entry(block_reason='verify exhausted')
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, (
            f'non-post-merge reason must NOT be caught by post-merge branch; got {result}'
        )

    def test_boundary_low_risk_post_merge_entry_aborts_without_git(self, tmp_path):
        """Case 4 (end-to-end boundary): build a genuine _build_entry output with
        block_reason=POST_MERGE_RED_MAIN_REASON_PREFIX + ' ...', stamp head/main sha,
        then assert check_proposal aborts without consulting git.
        """
        from orchestrator.b3_gate import ABORT, POST_MERGE_RED_MAIN_REASON_PREFIX, check_proposal
        from orchestrator.dry_run_unblock import _build_entry

        reason = POST_MERGE_RED_MAIN_REASON_PREFIX + ' on main: foo.py'
        agent_result = make_agent_result(
            success=True,
            structured_output={
                'proposal_text': 'Fix the type error in foo.py',
                'risk_label': 'low',
                'files_referenced': ['foo.py'],
            },
        )
        entry = _build_entry(agent_result, reason=reason, budget_usd=5.0)
        # Mirror T1 sha-stamping (dry_run_unblock.py)
        entry['head_sha'] = RECORDED_HEAD_SHA
        entry['main_sha'] = RECORDED_MAIN_SHA

        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, (
            f'expected ABORT for post-merge low-risk boundary entry, got {result}'
        )

    def test_coherence_prefix_matches_merge_queue_constant(self):
        """Case 5 (coherence): b3_gate re-declares the prefix; assert it stays in sync
        with the canonical merge_queue.POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX.
        """
        from orchestrator.b3_gate import POST_MERGE_RED_MAIN_REASON_PREFIX
        from orchestrator.merge_lane import POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX
        assert POST_MERGE_RED_MAIN_REASON_PREFIX == POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX, (
            f'b3_gate prefix {POST_MERGE_RED_MAIN_REASON_PREFIX!r} '
            f'!= merge_queue prefix {POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX!r}'
        )

    def test_production_format_reason_aborts_without_git(self):
        """Case 6 (producer→gate contract): block_reason built with the exact merge_queue
        production template (PREFIX + ': post-merge unscoped type-check failed for ...')
        must ABORT.  Locks the ': ' separator format used by merge_queue.py:653-658 so a
        future regression that drops the prefix from the template is caught here.
        """
        from orchestrator.b3_gate import ABORT, POST_MERGE_RED_MAIN_REASON_PREFIX, check_proposal

        # Exact production format from merge_queue.py:
        #   f'{POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX}: '
        #   f'post-merge unscoped type-check failed for '
        #   f'{", ".join(pyright_result.failing_subprojects)} '
        #   f'on {advanced_sha[:12]}. {pyright_result.detail}'
        production_reason = (
            POST_MERGE_RED_MAIN_REASON_PREFIX
            + ': post-merge unscoped type-check failed for orchestrator'
            + ' on abc123456789. error: Type "int | None" is not assignable to "int"'
        )
        entry = low_risk_entry(block_reason=production_reason)
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, (
            f'expected ABORT for production-format post-merge reason, got {result}'
        )


# ---------------------------------------------------------------------------
# step-3 (task 2509): explicit stop-instruction hard-abort in check_proposal
# ---------------------------------------------------------------------------

class TestStopInstructionAbort:
    """check_proposal hard-ABORTs any entry whose proposal_text, block_reason,
    or extra_texts carries an explicit stop instruction — before risk_label
    and git checks (task 2509, reconciliation finding 0aac21b4: an autonomous
    /unblock session self-authorized around an explicit "do not apply"
    instruction; this mechanical gate closes that hole).
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_stop_instruction_in_proposal_text_aborts_without_git(self):
        """Case (a): proposal_text carries 'do not apply' -> ABORT, no git called."""
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(proposal_text='Fix it — do not apply until reviewed')
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'stop instruction' in result['reason'].lower(), (
            f'expected "stop instruction" in reason: {result["reason"]!r}'
        )

    def test_stop_instruction_in_block_reason_aborts_without_git(self):
        """Case (b): block_reason carries 'human rehearsal' -> ABORT, no git called."""
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(block_reason='this fix requires human rehearsal before merging')
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'stop instruction' in result['reason'].lower(), (
            f'expected "stop instruction" in reason: {result["reason"]!r}'
        )

    def test_stop_instruction_in_extra_texts_aborts_without_git(self):
        """Case (c): a new extra_texts kwarg carries the phrase (proposal_text/
        block_reason are benign) -> ABORT, no git called."""
        from orchestrator.b3_gate import ABORT, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
            extra_texts=['Task description: please do not apply this change yet.'],
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'stop instruction' in result['reason'].lower(), (
            f'expected "stop instruction" in reason: {result["reason"]!r}'
        )

    def test_stop_instruction_wins_over_otherwise_fresh_entry(self):
        """Precedence: an otherwise-FRESH low-risk entry that ALSO carries a stop
        phrase in proposal_text -> ABORT (stop wins over fresh), and still consumes
        no git (fires before the P1/P2 freshness checks)."""
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(proposal_text='Fix it — do not apply until reviewed')
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, (
            f'stop instruction must win over an otherwise-fresh entry: {result}'
        )

    def test_benign_extra_texts_does_not_abort_regression(self):
        """Regression: benign extra_texts under a fresh low-risk entry -> FRESH (unchanged)."""
        from orchestrator.b3_gate import FRESH, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
            extra_texts=['deploy the service'],
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'


class TestStopScanExcludesCapturedOutput:
    """check_proposal's step-1c scan skips the fenced captured output of each
    verify failure-report section in block_reason, and still scans every
    authored text: proposal_text, extra_texts, and the rest of block_reason
    (task 6933). The failure-report stream cases cover every captured section
    shape failure_report emits (task 5357).
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    _RATCHET_RED_BLOCK_REASON = (
        'Post-merge verification failed: 1 new failing test(s) not present on main: '
        'tests.test_merge_lane_ratchet::test_baseline_matches_a_fresh_measurement'
        '@merge_lane_ratchet'
        '\n\n## Failure Cause\n\nFAILED tests/test_merge_lane_ratchet.py::'
        'test_baseline_matches_a_fresh_measurement@merge_lane_ratchet'
        '\n\n## Test Failures\n\n```\n'
        'E     RESIDUAL, so you do not trust more than is true: `git rebase` and a '
        'CLEAN merge do not run pre-commit at all, and the merge worker advances '
        'main with `git update-ref`, which runs no hooks.\n'
        '1 failed, 25096 passed'
        '\n```'
        '\n\nintegration_skew: port landed commit(s) 071b84b48f touching '
        'scripts/merge_lane_metrics.py — do not hunt your own diff'
    )

    def _merge_verify_red_entry(self, block_reason):
        return low_risk_entry(
            block_class='merge_verify_red',
            block_reason=block_reason,
        )

    def test_ratchet_readme_in_fenced_pytest_output_does_not_trip_the_stop_scan(self):
        from orchestrator.b3_gate import FRESH, check_proposal
        from orchestrator.stop_instruction import detect_stop_instruction

        assert detect_stop_instruction(self._RATCHET_RED_BLOCK_REASON) == 'do not run'

        result = check_proposal(
            self._merge_verify_red_entry(self._RATCHET_RED_BLOCK_REASON),
            worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'
        assert 'stop instruction' not in result['reason']

    def test_do_not_merge_in_proposal_text_still_aborts_beside_captured_output(self):
        from orchestrator.b3_gate import ABORT, check_proposal

        entry = {
            **self._merge_verify_red_entry(self._RATCHET_RED_BLOCK_REASON),
            'proposal_text': 'Regenerate the baseline, but do not merge until a human has looked.',
        }
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'stop instruction' in result['reason']
        assert "'do not merge'" in result['reason']

    @pytest.mark.parametrize('block_reason', [
        'Blocked: hold for human sign-off.\n\n## Test Failures\n\n```\nbenign output\n```',
        '## Test Failures\n\n```\nbenign output\n```\n\nhold for human sign-off',
    ], ids=['prose-before-fence', 'prose-after-fence'])
    def test_stop_phrase_in_block_reason_prose_outside_a_fence_still_aborts(self, block_reason):
        from orchestrator.b3_gate import ABORT, check_proposal

        result = check_proposal(
            self._merge_verify_red_entry(block_reason),
            worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'hold for human' in result['reason']

    def test_unterminated_fence_excludes_nothing(self):
        from orchestrator.b3_gate import ABORT, check_proposal

        truncated = (
            'Post-merge verification failed\n\n## Test Failures\n\n```\n'
            'E     a CLEAN merge do not run pre-commit at all'
        )
        result = check_proposal(
            self._merge_verify_red_entry(truncated),
            worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'do not run' in result['reason']

    @pytest.mark.parametrize('block_reason', [
        pytest.param(
            'Blocked.\n\n```\nhold for human sign-off\n```',
            id='fence-without-a-failure-report-heading',
        ),
        pytest.param(
            'Post-merge verification failed\n\n```\nE  expected:\n```\nE  got\n```'
            '\n\nhold for human sign-off'
            '\n\n## Lint Issues\n\n```\nx.py:1:1: E501 line too long\n```',
            id='unheaded-fences-around-prose',
        ),
        pytest.param(
            'Post-merge verification failed\n\n## Test Failures\n\n```\nE  expected:\n```\nE  got\n```'
            '\n\nhold for human sign-off'
            '\n\n## Lint Issues\n\n```\nx.py:1:1: E501 line too long\n```',
            id='stray-fence-inside-captured-output',
        ),
    ])
    def test_stray_or_unheaded_fences_never_hide_authored_prose(self, block_reason):
        from orchestrator.b3_gate import ABORT, check_proposal

        result = check_proposal(
            self._merge_verify_red_entry(block_reason),
            worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'hold for human' in result['reason']

    @pytest.mark.parametrize('stream', [
        {'test_output': (
            'E     a CLEAN merge do not run pre-commit at all\n'
            'FAILED tests/test_merge_lane_ratchet.py::test_baseline_matches_a_fresh_measurement\n'
            '1 failed'
        )},
        {'lint_output': 'x.py:1:1: E501 do not run this by hand'},
        {'type_output': 'x.py:1: error: do not run this by hand'},
        {'test_output': 'collected 3 items\nsetup note: do not run this by hand\n3 errors in 1.2s'},
        {'type_output': 'x.py:1: warning: do not run this by hand'},
        {'lint_output': '\n'.join(
            [f'x.py:{i}:1: E501 line too long' for i in range(80)]
            + ['x.py:999:1: E501 do not run this by hand']
        )},
    ], ids=[
        'test_output', 'lint_output', 'type_output',
        'test_output_without_marker', 'type_output_without_error',
        'truncated_lint_output',
    ])
    def test_verify_failure_report_streams_are_fenced_and_skipped(self, stream):
        from orchestrator.b3_gate import FRESH, check_proposal
        from orchestrator.stop_instruction import detect_stop_instruction
        from orchestrator.verify import VerifyResult

        streams = {'test_output': '', 'lint_output': '', 'type_output': '', **stream}
        report = VerifyResult(
            passed=False, summary='Failures: tests failed',
            test_leg_failed=bool(streams['test_output']), **streams,
        )
        block_reason = 'Post-merge verification failed: tests failed\n\n' + report.failure_report()

        assert '```' in block_reason
        assert detect_stop_instruction(block_reason) == 'do not run'

        result = check_proposal(
            self._merge_verify_red_entry(block_reason),
            worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'


# ---------------------------------------------------------------------------
# step-3 (task 2138): typed block_class dual-read path
# ---------------------------------------------------------------------------

class TestCheckProposalBlockClass:
    """check_proposal dual-reads a typed 'block_class' key when present,
    falling back to the legacy block_reason-prefix/status-presence sniffs
    only when block_class is absent (B3 legacy path — TestPostMergeRedMainAbort
    above and
    test_b3_gate.py::TestCheckProposalMechanical::test_status_key_present_aborts
    carry NO block_class and must keep passing unchanged).

    PRD: plans/verify-plan-prd.md task ζ, invariants B2/B3/B4.
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_post_merge_red_main_block_class_aborts_without_git(self):
        """B2: block_class=post_merge_red_main hard-aborts before risk/git —
        even when block_reason carries NO post-merge prose (proves the typed
        field alone drives the abort, not the prose prefix) — for both a
        low-risk and a human-review-required entry.
        """
        from orchestrator.b3_gate import ABORT, check_proposal
        for risk_label in ('low', 'human-review-required'):
            entry = low_risk_entry(
                block_class='post_merge_red_main',
                block_reason='verify exhausted',
                risk_label=risk_label,
            )
            result = check_proposal(
                entry, worktree='/tmp', category='task_failure',
                run_git=fake_git_never_called, now=self._NOW,
            )
            assert result['verdict'] == ABORT, (
                f'risk_label={risk_label!r}: expected ABORT, got {result}'
            )

    def test_merge_verify_red_low_risk_is_gateable_not_auto_aborted(self):
        """B4: block_class=merge_verify_red is NOT the highest-blast-radius
        class — a low-risk entry reaches FRESH exactly like today, with
        fresh git and category=None.
        """
        from orchestrator.b3_gate import FRESH, check_proposal
        entry = low_risk_entry(block_class='merge_verify_red')
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'

    def test_new_path_failure_entry_aborts_via_risk_label_not_status(self):
        """New-path failure death: a real producer failure shape always sets
        risk_label='human-review-required', so step 2 catches it — the
        status-sniff is not what fires. Asserts the reason names risk_label.
        """
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(
            block_class='agent_failure',
            status='investigation_failed',
            risk_label='human-review-required',
        )
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'risk_label' in result['reason'], (
            f'expected the risk_label check (step 2) to fire, not the '
            f'status sniff: {result["reason"]!r}'
        )

    def test_status_presence_does_not_abort_typed_path(self):
        """The 'status'-key presence sniff (legacy step 3) must NOT fire when
        block_class is present — a stray/legacy 'status' key on an otherwise
        low-risk, fresh, typed entry must not cause a false abort.
        """
        from orchestrator.b3_gate import FRESH, check_proposal
        entry = low_risk_entry(
            block_class='merge_verify_red',
            status='x',
            risk_label='low',
        )
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, (
            f'status-key presence must not abort the typed path, got {result}'
        )

    def test_agent_failure_low_risk_fresh_entry_is_fresh(self):
        """Regression: block_class=agent_failure (the common case) behaves
        exactly like today's low-risk fresh entry.
        """
        from orchestrator.b3_gate import FRESH, check_proposal
        entry = low_risk_entry(block_class='agent_failure')
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'


# ---------------------------------------------------------------------------
# step-5 (task 2633): merge-completion consumption-gate contract
# ---------------------------------------------------------------------------

class TestMergeCompletionConsumptionContract:
    """Pins the merge-stage completion mode's CONSUMPTION-side acceptance
    criteria against b3_gate.check_proposal (task 2633).

    These are CONTRACT/regression tests, not tests driving new b3_gate code:
    the (a)+(b)+(c) eligibility gate is enforced at GENERATION time by the
    dry_run_unblock.run_dry_run_unblock risk_label clamp (see
    orchestrator.merge_lane.completion.merge_completion_eligible and
    TestMergeCompletionRiskClamp in test_dry_run_unblock.py) —
    check_proposal itself is UNCHANGED by task 2633. This class pins that
    the EXISTING check_proposal behaviour already delivers the three
    consumption outcomes the merge-completion mode depends on, so a future
    refactor of check_proposal cannot silently break merge-completion
    consumption without failing a test here:
      (i)   an ineligible (clamped) MERGE_VERIFY_RED entry aborts;
      (ii)  an eligible (low-risk, fresh) MERGE_VERIFY_RED entry is
            consumable end-to-end; and
      (iii) the POST_MERGE_RED_MAIN hard-abort (task 1680) is retained
            VERBATIM.
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_clamped_merge_verify_red_entry_aborts(self):
        """(i) The step-4 clamp shape — MERGE_VERIFY_RED, risk_label clamped
        to 'human-review-required' by the generation-time (b)+(c) gate —
        aborts at step 2 (risk_label != 'low'), never reaching git.
        """
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(
            block_class='merge_verify_red',
            risk_label='human-review-required',
        )
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'

    def test_eligible_merge_verify_red_entry_is_fresh(self):
        """(ii) An eligible mechanical proposal — block_class=merge_verify_red,
        risk_label='low' (the (b)+(c) gate passed at generation, so the clamp
        never fired), matching HEAD, unchanged files_referenced footprint,
        category='task_failure' (the real category post-merge-verify blocks
        file, workflow.py:8608) — is consumable end-to-end at the gate.
        """
        from orchestrator.b3_gate import FRESH, check_proposal
        entry = low_risk_entry(block_class='merge_verify_red')
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'

    def test_post_merge_red_main_entry_hard_aborts(self):
        """(iii) task 1680 retained VERBATIM: a POST_MERGE_RED_MAIN entry
        hard-aborts before risk_label/git checks even with risk_label='low'
        (low_risk_entry()'s default) — the merge-completion mode must never
        weaken this highest-blast-radius safety rail.
        """
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(block_class='post_merge_red_main')
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'post-merge red-main fix-forward class' in result['reason'], (
            f"expected the task-1680 reason text, got: {result['reason']!r}"
        )
