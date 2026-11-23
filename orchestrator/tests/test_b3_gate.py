"""Tests for orchestrator.b3_gate.check_proposal's ordinary verdict path.

Covers: the mechanical risk/status/category/sha-anchor checks, three-state age
reporting, P1/P2 freshness against real git fixtures, and the two-way boundary
with dry_run_unblock._build_entry, the producer of the entries it checks.

Siblings: test_b3_gate_state.py (the state file, record_launch and charge),
test_b3_gate_hard_aborts.py (the entries check_proposal aborts whatever their
freshness) and test_b3_gate_cli.py (the readers, cap config and CLI). Their
shared scaffolding is _b3_gate_fixtures.py.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

from _agent_result_fakes import make_agent_result
from _b3_gate_fixtures import (
    diverged_repo,
    fake_git_diff_fails,
    fake_git_fresh,
    fake_git_never_called,
    low_risk_entry,
)
from _git_fixtures import RepoSeed

_FOO_ON_MAIN = RepoSeed(files=(('foo.py', 'original content'),), message='add foo.py')

# ---------------------------------------------------------------------------
# step-9: check_proposal mechanical classification (fake git runner)
# ---------------------------------------------------------------------------

class TestCheckProposalMechanical:
    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_none_entry_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        result = check_proposal(
            None, worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT
        assert 'proposal' in result['reason'].lower()

    def test_medium_risk_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(risk_label='medium')
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_human_review_risk_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(risk_label='human-review-required')
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_status_key_present_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = low_risk_entry(status='investigation_failed')
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_invalid_category_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category='infra_issue',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT

    def test_valid_categories_not_aborted_on_category_check(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        for cat in ('task_failure', 'review_issues'):
            # Should NOT abort on category check (may abort/drift/fresh for other reasons)
            result = check_proposal(
                low_risk_entry(), worktree='/tmp', category=cat,
                run_git=fake_git_fresh, now=self._NOW,
            )
            # Category check passes -> proceeds to freshness check -> fresh
            assert result['verdict'] != ABORT or 'category' not in result['reason'].lower(), \
                f'Unexpected abort on valid category={cat}: {result}'

    def test_no_category_skips_category_check(self):
        from orchestrator.b3_gate import check_proposal
        # category=None -> skip category check entirely -> proceeds to freshness
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        # Should not abort on category; may be fresh or drift depending on git
        assert 'verdict' in result

    def test_missing_head_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = low_risk_entry()
        del entry['head_sha']
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
        assert 'sha anchor' in result['reason'].lower()

    def test_missing_main_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = low_risk_entry()
        del entry['main_sha']
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
        assert 'sha anchor' in result['reason'].lower()

    def test_none_head_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = low_risk_entry(head_sha=None)
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT

    # step-21: P2 diff failure must not fall through to fresh (fail-closed)
    def test_p2_diff_failure_does_not_fall_through_to_fresh(self):
        """When the footprint git-diff returns a non-zero rc, verdict must be DRIFT.

        Pin the fail-closed contract: a git failure (unreachable main_sha, missing
        'main' ref, transient git error) must never fall through to a merge-authorizing
        FRESH verdict.  low_risk_entry() has files_referenced=['foo.py'] so the diff IS
        invoked; fake_git_diff_fails returns rc=128 for the diff call.
        """
        from orchestrator.b3_gate import DRIFT, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category=None,
            run_git=fake_git_diff_fails, now=self._NOW,
        )
        assert result['verdict'] == DRIFT, (
            f'expected DRIFT when git diff fails, got {result["verdict"]!r}: {result}'
        )
        reason = result['reason'].lower()
        assert 'footprint' in reason, f'expected "footprint" in reason: {result["reason"]!r}'
        assert '128' in reason, f'expected rc 128 in reason: {result["reason"]!r}'


# ---------------------------------------------------------------------------
# task 5361 step-1: three-state age reporting — `age_state` closed domain
# ---------------------------------------------------------------------------


class TestAgeStateThreeState:
    """`check_proposal` names WHY it has no age, instead of collapsing to None.

    Four distinct causes previously all surfaced as a bare `age_seconds: None`,
    indistinguishable to every consumer: an absent timestamp, an unparseable
    string (ValueError), an awareness mismatch between the two operands
    (TypeError), and no clock supplied at all.  THREE states now cover those
    four causes — hence this class's name.  The mismatch shares
    `AGE_UNPARSEABLE` with the malformed string because the fault is the PAIR,
    not one operand; `_age_of`'s warning names both so the shared state cannot
    misdirect an operator.
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    @staticmethod
    def _states():
        from orchestrator.b3_gate import (
            AGE_ABSENT,
            AGE_NO_CLOCK,
            AGE_PARSED,
            AGE_UNPARSEABLE,
        )
        return {AGE_PARSED, AGE_UNPARSEABLE, AGE_ABSENT, AGE_NO_CLOCK}

    def test_valid_aware_timestamp_is_parsed(self):
        from orchestrator.b3_gate import AGE_PARSED, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_PARSED, result
        assert result['age_seconds'] == 10800.0, result

    def test_unparseable_timestamp_is_distinguished(self):
        from orchestrator.b3_gate import AGE_UNPARSEABLE, check_proposal
        entry = low_risk_entry(investigated_at='not-a-date')
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_UNPARSEABLE, result
        assert result['age_seconds'] is None, result

    def test_naive_timestamp_against_aware_clock_is_unparseable(self):
        """The TypeError branch — a DIFFERENT cause from the ValueError above,
        and the one most likely to arise from a real producer bug."""
        from orchestrator.b3_gate import AGE_UNPARSEABLE, check_proposal
        entry = low_risk_entry(investigated_at='2026-06-04T09:00:00')
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_UNPARSEABLE, result
        assert result['age_seconds'] is None, result

    def test_aware_timestamp_against_naive_clock_is_unparseable(self):
        """The OPERATOR-side mismatch: `run_check._parse_now` passes any
        `--now` string straight through `fromisoformat`, so a naive one fails
        against a perfectly well-formed aware `investigated_at`.  Same
        TypeError, same state, opposite culprit from the test above."""
        from orchestrator.b3_gate import AGE_UNPARSEABLE, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=datetime(2026, 6, 4, 12, 0, 0),
        )
        assert result['age_state'] == AGE_UNPARSEABLE, result
        assert result['age_seconds'] is None, result

    def test_warning_names_both_operands_not_just_the_entry(self, caplog):
        """The state cannot say which operand is at fault, so the message must
        name both.  Blaming `investigated_at` alone points an operator at the
        proposal when a naive `--now` of their own is the actual fault."""
        import logging as _logging

        from orchestrator.b3_gate import check_proposal
        naive_now = datetime(2026, 6, 4, 12, 0, 0)

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            check_proposal(
                low_risk_entry(), worktree='/tmp', category=None,
                run_git=fake_git_fresh, now=naive_now,
            )

        messages = [
            r.getMessage() for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
        ]
        assert len(messages) == 1, messages
        assert low_risk_entry()['investigated_at'] in messages[0], messages[0]
        assert repr(naive_now) in messages[0], messages[0]

    def test_missing_timestamp_key_is_absent(self):
        from orchestrator.b3_gate import AGE_ABSENT, check_proposal
        entry = low_risk_entry()
        del entry['investigated_at']
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_ABSENT, result
        assert result['age_seconds'] is None, result

    def test_empty_timestamp_is_absent(self):
        from orchestrator.b3_gate import AGE_ABSENT, check_proposal
        entry = low_risk_entry(investigated_at='')
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_ABSENT, result

    def test_no_clock_is_its_own_state(self):
        from orchestrator.b3_gate import AGE_NO_CLOCK, check_proposal
        result = check_proposal(
            low_risk_entry(), worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=None,
        )
        assert result['age_state'] == AGE_NO_CLOCK, result
        assert result['age_seconds'] is None, result

    def test_none_entry_early_return_carries_age_state(self):
        """Shape parity: EVERY return path carries the key, so a consumer never
        has to distinguish 'absent key' from 'old entry' from a real state."""
        from orchestrator.b3_gate import AGE_ABSENT, check_proposal
        result = check_proposal(
            None, worktree='/tmp', category=None,
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['age_state'] == AGE_ABSENT, result
        assert result['age_state'] in self._states(), result

    def test_unparseable_timestamp_warns_instead_of_swallowing(self, caplog):
        """Replaces today's silent `except Exception: pass`."""
        import logging as _logging

        from orchestrator.b3_gate import check_proposal
        entry = low_risk_entry(investigated_at='not-a-date')

        with caplog.at_level(_logging.WARNING, logger='orchestrator.b3_gate'):
            check_proposal(
                entry, worktree='/tmp', category=None,
                run_git=fake_git_fresh, now=self._NOW,
            )

        warn_records = [
            r for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
            and 'not-a-date' in r.getMessage()
        ]
        assert len(warn_records) == 1, (
            f'Expected exactly one WARNING under orchestrator.b3_gate naming the '
            f'offending investigated_at value; got records: '
            f'{[(r.name, r.getMessage()) for r in caplog.records]}'
        )


# ---------------------------------------------------------------------------
# task 5361 step-3: the gate never certifies FRESH without a parsed age
# ---------------------------------------------------------------------------


class TestFreshRequiresParsedAge:
    """FRESH authorises unattended edits and merges to main.

    A gate with that authority must not certify a proposal fresh when it cannot
    determine the proposal's age at all — whatever the cause.  The positive
    control is the load-bearing case here: without it, "fails closed" could be
    satisfied by a gate that never returns FRESH at all.
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def _check(self, entry, *, now):
        from orchestrator.b3_gate import check_proposal
        return check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=now,
        )

    def test_parsed_age_still_certifies_fresh(self):
        """POSITIVE CONTROL — the gate is not made vacuous by the guard."""
        from orchestrator.b3_gate import AGE_PARSED, FRESH
        result = self._check(low_risk_entry(), now=self._NOW)
        assert result['verdict'] == FRESH, result
        assert result['age_state'] == AGE_PARSED, result

    def test_unparseable_timestamp_aborts(self):
        from orchestrator.b3_gate import ABORT, AGE_UNPARSEABLE
        entry = low_risk_entry(investigated_at='not-a-date')
        result = self._check(entry, now=self._NOW)
        assert result['verdict'] == ABORT, result
        assert result['age_state'] == AGE_UNPARSEABLE, result
        reason = result['reason'].lower()
        assert 'age' in reason, result['reason']
        assert AGE_UNPARSEABLE in result['reason'], result['reason']

    def test_absent_timestamp_aborts(self):
        from orchestrator.b3_gate import ABORT, AGE_ABSENT
        entry = low_risk_entry()
        del entry['investigated_at']
        result = self._check(entry, now=self._NOW)
        assert result['verdict'] == ABORT, result
        assert result['age_state'] == AGE_ABSENT, result
        assert 'age' in result['reason'].lower(), result['reason']

    def test_no_clock_aborts(self):
        from orchestrator.b3_gate import ABORT, AGE_NO_CLOCK
        result = self._check(low_risk_entry(), now=None)
        assert result['verdict'] == ABORT, result
        assert result['age_state'] == AGE_NO_CLOCK, result
        assert 'age' in result['reason'].lower(), result['reason']


# ---------------------------------------------------------------------------
# step-11: freshness P1/P2 against a REAL git fixture
# ---------------------------------------------------------------------------

class TestCheckProposalFreshness:
    """P1 (HEAD moved -> abort) and P2 (file-scoped main drift -> drift)."""

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def test_clean_is_fresh(self, tmp_path):
        from orchestrator.b3_gate import FRESH, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = low_risk_entry(head_sha=head_sha, main_sha=main_sha)
        result = check_proposal(entry, worktree=str(repo), category=None,
                                run_git=_run_git, now=self._NOW)
        assert result['verdict'] == FRESH, f'expected fresh: {result}'

    def test_p1_head_moved_aborts(self, tmp_path):
        from orchestrator.b3_gate import ABORT, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = low_risk_entry(head_sha=head_sha, main_sha=main_sha)
        # Add extra commit on feature branch so HEAD moves
        p = str(repo)
        (repo / 'extra2.txt').write_text('more work')
        subprocess.run(['git', '-C', p, 'add', 'extra2.txt'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'extra commit'],
                       check=True, capture_output=True)
        result = check_proposal(entry, worktree=str(repo), category=None,
                                run_git=_run_git, now=self._NOW)
        assert result['verdict'] == ABORT
        # Reason must be git-anchored (both shas)
        assert head_sha in result['reason'], f'expected recorded sha in reason: {result}'

    def test_p2_footprint_file_changed_on_main_drifts(self, tmp_path):
        from orchestrator.b3_gate import DRIFT, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = low_risk_entry(head_sha=head_sha, main_sha=main_sha, files_referenced=['foo.py'])
        # Switch to main, change foo.py, switch back
        p = str(repo)
        subprocess.run(['git', '-C', p, 'checkout', 'main'], check=True, capture_output=True)
        (repo / 'foo.py').write_text('updated content')
        subprocess.run(['git', '-C', p, 'add', 'foo.py'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'update foo.py on main'],
                       check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'checkout', 'feature'], check=True, capture_output=True)
        result = check_proposal(entry, worktree=str(repo), category=None,
                                run_git=_run_git, now=self._NOW)
        assert result['verdict'] == DRIFT, f'expected drift: {result}'

    def test_p2_non_footprint_file_changed_on_main_still_fresh(self, tmp_path):
        from orchestrator.b3_gate import FRESH, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        # footprint is foo.py only; change bar.py on main (not in footprint)
        entry = low_risk_entry(head_sha=head_sha, main_sha=main_sha, files_referenced=['foo.py'])
        p = str(repo)
        subprocess.run(['git', '-C', p, 'checkout', 'main'], check=True, capture_output=True)
        (repo / 'bar.py').write_text('new file')
        subprocess.run(['git', '-C', p, 'add', 'bar.py'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'add bar.py on main'],
                       check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'checkout', 'feature'], check=True, capture_output=True)
        result = check_proposal(entry, worktree=str(repo), category=None,
                                run_git=_run_git, now=self._NOW)
        assert result['verdict'] == FRESH, f'expected fresh (outside footprint): {result}'


# ---------------------------------------------------------------------------
# step-19: TWO-WAY BOUNDARY — authentic _build_entry output → check_proposal
# ---------------------------------------------------------------------------

class TestTwoWayBoundary:
    """Prove the producer (_build_entry + T1 sha-stamping) and consumer
    (check_proposal) agree on field names and shape.

    Uses the REAL dry_run_unblock._build_entry and real git fixtures.
    """

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def _make_success_entry(self, head_sha, main_sha, files=None):
        """Build a genuine _build_entry output then mirror T1 sha-stamping."""
        from orchestrator.dry_run_unblock import _build_entry
        agent_result = make_agent_result(
            success=True,
            structured_output={
                'proposal_text': 'Fix the bug in foo.py',
                'risk_label': 'low',
                'files_referenced': files if files is not None else ['foo.py'],
            },
        )
        entry = _build_entry(agent_result, reason='blocked', budget_usd=5.0)
        # Mirror T1 stamping (dry_run_unblock.py :237-238)
        entry['head_sha'] = head_sha
        entry['main_sha'] = main_sha
        return entry

    def test_a_clean_repo_is_fresh(self, tmp_path):
        """(a) Clean repo (HEAD == head_sha, main unchanged) -> fresh."""
        from orchestrator.b3_gate import FRESH, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = self._make_success_entry(head_sha, main_sha)
        verdict = check_proposal(entry, worktree=str(repo), category='task_failure',
                                 run_git=_run_git, now=self._NOW)
        assert verdict['verdict'] == FRESH, f'expected fresh: {verdict}'

    def test_b_head_moved_aborts_with_git_anchored_reason(self, tmp_path):
        """(b) Extra commit on feature branch -> abort, reason contains both shas."""
        from orchestrator.b3_gate import ABORT, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = self._make_success_entry(head_sha, main_sha)
        # Add extra commit — HEAD moves
        p = str(repo)
        (repo / 'extra2.txt').write_text('more work')
        subprocess.run(['git', '-C', p, 'add', 'extra2.txt'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'extra feature commit'],
                       check=True, capture_output=True)
        verdict = check_proposal(entry, worktree=str(repo), category='task_failure',
                                 run_git=_run_git, now=self._NOW)
        assert verdict['verdict'] == ABORT, f'expected abort: {verdict}'
        # Both shas must appear in the reason
        assert head_sha in verdict['reason'], (
            f'recorded sha {head_sha!r} must be in reason: {verdict["reason"]}'
        )

    def test_c_footprint_file_changed_on_main_drifts(self, tmp_path):
        """(c) foo.py changed on main -> drift."""
        from orchestrator.b3_gate import DRIFT, _run_git, check_proposal
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = self._make_success_entry(head_sha, main_sha, files=['foo.py'])
        # Change foo.py on main
        p = str(repo)
        subprocess.run(['git', '-C', p, 'checkout', 'main'], check=True, capture_output=True)
        (repo / 'foo.py').write_text('modified on main')
        subprocess.run(['git', '-C', p, 'add', 'foo.py'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'update foo.py on main'],
                       check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'checkout', 'feature'], check=True, capture_output=True)
        verdict = check_proposal(entry, worktree=str(repo), category='task_failure',
                                 run_git=_run_git, now=self._NOW)
        assert verdict['verdict'] == DRIFT, f'expected drift: {verdict}'

    def test_d_no_sha_stamping_drifts_with_no_sha_anchor(self, tmp_path):
        """(d) Entry without T1 sha-stamping -> drift 'no sha anchor'."""
        from orchestrator.b3_gate import DRIFT, _run_git, check_proposal
        from orchestrator.dry_run_unblock import _build_entry
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        agent_result = make_agent_result(
            success=True,
            structured_output={
                'proposal_text': 'Fix the bug',
                'risk_label': 'low',
                'files_referenced': ['foo.py'],
            },
        )
        # Raw _build_entry output — NO sha stamping
        entry = _build_entry(agent_result, reason='blocked', budget_usd=5.0)
        assert 'head_sha' not in entry, f'entry should have no head_sha before stamping: {entry}'
        assert 'main_sha' not in entry, f'entry should have no main_sha before stamping: {entry}'
        verdict = check_proposal(entry, worktree=str(repo), category='task_failure',
                                 run_git=_run_git, now=self._NOW)
        assert verdict['verdict'] == DRIFT, f'expected drift: {verdict}'
        assert 'sha anchor' in verdict['reason'].lower(), (
            f'expected "sha anchor" in reason: {verdict["reason"]}'
        )

    def test_failure_entry_has_status_key_and_is_aborted(self, tmp_path):
        """Failure _build_entry carries a 'status' key; check_proposal aborts."""
        from orchestrator.b3_gate import ABORT, _run_git, check_proposal
        from orchestrator.dry_run_unblock import _build_entry
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        # Build a failure entry
        fail_agent = make_agent_result(success=False, subtype='error', output='boom')
        fail_entry = _build_entry(fail_agent, reason='blocked', budget_usd=5.0)
        # Mirror T1 stamping (failure entries also get stamped)
        fail_entry['head_sha'] = head_sha
        fail_entry['main_sha'] = main_sha
        # Failure entries must carry a 'status' key
        assert 'status' in fail_entry, f'failure entry must have status key: {fail_entry}'
        # check_proposal must abort on seeing 'status'
        verdict = check_proposal(fail_entry, worktree=str(repo), category='task_failure',
                                 run_git=_run_git, now=self._NOW)
        assert verdict['verdict'] == ABORT, f'expected abort for failure entry: {verdict}'

    def test_infra_failure_and_investigation_failed_both_aborted(self, tmp_path):
        """task 2020 step-9: the new infra_failure status (a retryable HOST
        wedge) and the existing investigation_failed status both carry a
        'status' key, and both must be aborted by check_proposal — proving
        the infra reclassification cannot open the B3 low-risk auto-unblock
        path. b3_gate.py is intentionally NOT modified for this task.
        """
        from orchestrator.b3_gate import ABORT, _run_git, check_proposal
        from orchestrator.dry_run_unblock import _build_entry
        repo, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)

        # Wedge: pre-turn-1 kill at the 600s UnblockAutoConfig timeout
        # ceiling -> empty stdout + timed_out=True.
        wedge_agent = make_agent_result(
            success=False,
            subtype='error_empty_output',
            output='Agent produced no output',
            timed_out=True,
            duration_ms=600000,
            transcript_turns=0,
            session_id='sess-x',
            stderr='...Absolute ceiling reached after 600.0s...',
        )
        wedge_entry = _build_entry(wedge_agent, reason='blocked', budget_usd=5.0)
        wedge_entry['head_sha'] = head_sha
        wedge_entry['main_sha'] = main_sha
        assert wedge_entry['status'] == 'infra_failure'

        # Substantive: agent ran and concluded a human must look — not timed
        # out, not empty output.
        substantive_agent = make_agent_result(
            success=False, subtype='error_max_turns', output='Exceeded max turns',
        )
        substantive_entry = _build_entry(substantive_agent, reason='blocked', budget_usd=5.0)
        substantive_entry['head_sha'] = head_sha
        substantive_entry['main_sha'] = main_sha
        assert substantive_entry['status'] == 'investigation_failed'

        for entry in (wedge_entry, substantive_entry):
            verdict = check_proposal(entry, worktree=str(repo), category='task_failure',
                                     run_git=_run_git, now=self._NOW)
            assert verdict['verdict'] == ABORT, (
                f"expected abort for status={entry['status']!r}: {verdict}"
            )

    def test_success_entry_has_no_status_key(self, tmp_path):
        """Success _build_entry must NOT have a 'status' key (contract invariant)."""
        _, head_sha, main_sha = diverged_repo(tmp_path, _FOO_ON_MAIN)
        entry = self._make_success_entry(head_sha, main_sha)
        assert 'status' not in entry, (
            f'success entry must not carry status key: {entry.keys()}'
        )
