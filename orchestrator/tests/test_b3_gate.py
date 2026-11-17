"""Tests for orchestrator.b3_gate module.

Covers: check_proposal mechanical classification, freshness P1/P2 against
real git fixtures, and the two-way boundary test against the real _build_entry
producer from dry_run_unblock. The state file, record_launch and charge are
covered in test_b3_gate_state.py; the readers, cap config and CLI in
test_b3_gate_cli.py.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime

import pytest
from _b3_gate_fixtures import (
    LOW_RISK_ENTRY,
    diverge_feature_branch,
    fake_git_diff_fails,
    fake_git_fresh,
    fake_git_never_called,
    make_agent_result,
)
from _git_fixtures import seed_repo

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
        entry = {**LOW_RISK_ENTRY, 'risk_label': 'medium'}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_human_review_risk_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = {**LOW_RISK_ENTRY, 'risk_label': 'human-review-required'}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_status_key_present_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = {**LOW_RISK_ENTRY, 'status': 'investigation_failed'}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == ABORT

    def test_invalid_category_aborts(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        result = check_proposal(
            LOW_RISK_ENTRY, worktree='/tmp', category='infra_issue',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT

    def test_valid_categories_not_aborted_on_category_check(self):
        from orchestrator.b3_gate import ABORT, check_proposal
        for cat in ('task_failure', 'review_issues'):
            # Should NOT abort on category check (may abort/drift/fresh for other reasons)
            result = check_proposal(
                LOW_RISK_ENTRY, worktree='/tmp', category=cat,
                run_git=fake_git_fresh, now=self._NOW,
            )
            # Category check passes -> proceeds to freshness check -> fresh
            assert result['verdict'] != ABORT or 'category' not in result['reason'].lower(), \
                f'Unexpected abort on valid category={cat}: {result}'

    def test_no_category_skips_category_check(self):
        from orchestrator.b3_gate import check_proposal
        # category=None -> skip category check entirely -> proceeds to freshness
        result = check_proposal(
            LOW_RISK_ENTRY, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        # Should not abort on category; may be fresh or drift depending on git
        assert 'verdict' in result

    def test_missing_head_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = {**LOW_RISK_ENTRY}
        del entry['head_sha']
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
        assert 'sha anchor' in result['reason'].lower()

    def test_missing_main_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = {**LOW_RISK_ENTRY}
        del entry['main_sha']
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT
        assert 'sha anchor' in result['reason'].lower()

    def test_none_head_sha_drifts(self):
        from orchestrator.b3_gate import DRIFT, check_proposal
        entry = {**LOW_RISK_ENTRY, 'head_sha': None}
        result = check_proposal(entry, worktree='/tmp', category=None,
                                run_git=fake_git_never_called, now=self._NOW)
        assert result['verdict'] == DRIFT

    # step-21: P2 diff failure must not fall through to fresh (fail-closed)
    def test_p2_diff_failure_does_not_fall_through_to_fresh(self):
        """When the footprint git-diff returns a non-zero rc, verdict must be DRIFT.

        Pin the fail-closed contract: a git failure (unreachable main_sha, missing
        'main' ref, transient git error) must never fall through to a merge-authorizing
        FRESH verdict.  LOW_RISK_ENTRY has files_referenced=['foo.py'] so the diff IS
        invoked; fake_git_diff_fails returns rc=128 for the diff call.
        """
        from orchestrator.b3_gate import DRIFT, check_proposal
        result = check_proposal(
            LOW_RISK_ENTRY, worktree='/tmp', category=None,
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
            LOW_RISK_ENTRY, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_PARSED, result
        assert result['age_seconds'] == 10800.0, result

    def test_unparseable_timestamp_is_distinguished(self):
        from orchestrator.b3_gate import AGE_UNPARSEABLE, check_proposal
        entry = {**LOW_RISK_ENTRY, 'investigated_at': 'not-a-date'}
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
        entry = {**LOW_RISK_ENTRY, 'investigated_at': '2026-06-04T09:00:00'}
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
            LOW_RISK_ENTRY, worktree='/tmp', category=None,
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
                LOW_RISK_ENTRY, worktree='/tmp', category=None,
                run_git=fake_git_fresh, now=naive_now,
            )

        messages = [
            r.getMessage() for r in caplog.records
            if r.levelno == _logging.WARNING and r.name == 'orchestrator.b3_gate'
        ]
        assert len(messages) == 1, messages
        assert LOW_RISK_ENTRY['investigated_at'] in messages[0], messages[0]
        assert repr(naive_now) in messages[0], messages[0]

    def test_missing_timestamp_key_is_absent(self):
        from orchestrator.b3_gate import AGE_ABSENT, check_proposal
        entry = {**LOW_RISK_ENTRY}
        del entry['investigated_at']
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_ABSENT, result
        assert result['age_seconds'] is None, result

    def test_empty_timestamp_is_absent(self):
        from orchestrator.b3_gate import AGE_ABSENT, check_proposal
        entry = {**LOW_RISK_ENTRY, 'investigated_at': ''}
        result = check_proposal(
            entry, worktree='/tmp', category=None,
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['age_state'] == AGE_ABSENT, result

    def test_no_clock_is_its_own_state(self):
        from orchestrator.b3_gate import AGE_NO_CLOCK, check_proposal
        result = check_proposal(
            LOW_RISK_ENTRY, worktree='/tmp', category=None,
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
        entry = {**LOW_RISK_ENTRY, 'investigated_at': 'not-a-date'}

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
        result = self._check(LOW_RISK_ENTRY, now=self._NOW)
        assert result['verdict'] == FRESH, result
        assert result['age_state'] == AGE_PARSED, result

    def test_unparseable_timestamp_aborts(self):
        from orchestrator.b3_gate import ABORT, AGE_UNPARSEABLE
        entry = {**LOW_RISK_ENTRY, 'investigated_at': 'not-a-date'}
        result = self._check(entry, now=self._NOW)
        assert result['verdict'] == ABORT, result
        assert result['age_state'] == AGE_UNPARSEABLE, result
        reason = result['reason'].lower()
        assert 'age' in reason, result['reason']
        assert AGE_UNPARSEABLE in result['reason'], result['reason']

    def test_absent_timestamp_aborts(self):
        from orchestrator.b3_gate import ABORT, AGE_ABSENT
        entry = {**LOW_RISK_ENTRY}
        del entry['investigated_at']
        result = self._check(entry, now=self._NOW)
        assert result['verdict'] == ABORT, result
        assert result['age_state'] == AGE_ABSENT, result
        assert 'age' in result['reason'].lower(), result['reason']

    def test_no_clock_aborts(self):
        from orchestrator.b3_gate import ABORT, AGE_NO_CLOCK
        result = self._check(LOW_RISK_ENTRY, now=None)
        assert result['verdict'] == ABORT, result
        assert result['age_state'] == AGE_NO_CLOCK, result
        assert 'age' in result['reason'].lower(), result['reason']


# ---------------------------------------------------------------------------
# step-11: freshness P1/P2 against a REAL git fixture
# ---------------------------------------------------------------------------

class TestCheckProposalFreshness:
    """P1 (HEAD moved -> abort) and P2 (file-scoped main drift -> drift)."""

    _NOW = datetime(2026, 6, 4, 12, 0, 0, tzinfo=UTC)

    def _make_entry(self, head_sha, main_sha, files=None):
        return {
            'proposal_text': 'Fix it',
            'risk_label': 'low',
            'files_referenced': files or ['foo.py'],
            'block_reason': 'test',
            'investigated_at': '2026-06-04T09:00:00+00:00',
            'head_sha': head_sha,
            'main_sha': main_sha,
        }

    def _setup_repo(self, tmp_path):
        """Create repo: main at initial commit, feature branch diverges; add foo.py."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        seed_repo(repo)
        # Create foo.py on main so it has a history
        (repo / 'foo.py').write_text('original content')
        p = str(repo)
        subprocess.run(['git', '-C', p, 'add', 'foo.py'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'add foo.py'],
                       check=True, capture_output=True)
        main_sha = subprocess.run(
            ['git', '-C', p, 'rev-parse', 'main'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        # Diverge feature branch
        head_sha = diverge_feature_branch(repo)
        assert head_sha != main_sha
        return repo, head_sha, main_sha

    def test_clean_is_fresh(self, tmp_path):
        from orchestrator.b3_gate import FRESH, _run_git, check_proposal
        repo, head_sha, main_sha = self._setup_repo(tmp_path)
        entry = self._make_entry(head_sha, main_sha)
        result = check_proposal(entry, worktree=str(repo), category=None,
                                run_git=_run_git, now=self._NOW)
        assert result['verdict'] == FRESH, f'expected fresh: {result}'

    def test_p1_head_moved_aborts(self, tmp_path):
        from orchestrator.b3_gate import ABORT, _run_git, check_proposal
        repo, head_sha, main_sha = self._setup_repo(tmp_path)
        entry = self._make_entry(head_sha, main_sha)
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
        repo, head_sha, main_sha = self._setup_repo(tmp_path)
        entry = self._make_entry(head_sha, main_sha, files=['foo.py'])
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
        repo, head_sha, main_sha = self._setup_repo(tmp_path)
        # footprint is foo.py only; change bar.py on main (not in footprint)
        entry = self._make_entry(head_sha, main_sha, files=['foo.py'])
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

    def _setup_boundary_repo(self, tmp_path):
        """Create git repo: main with foo.py, feature branch diverges.

        Returns (repo_path, head_sha, main_sha).
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        seed_repo(repo)
        p = str(repo)
        # Add foo.py on main so it exists in the diff history
        (repo / 'foo.py').write_text('original content')
        subprocess.run(['git', '-C', p, 'add', 'foo.py'], check=True, capture_output=True)
        subprocess.run(['git', '-C', p, 'commit', '-m', 'add foo.py'],
                       check=True, capture_output=True)
        main_sha = subprocess.run(
            ['git', '-C', p, 'rev-parse', 'main'],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        # Diverge feature branch from main
        head_sha = diverge_feature_branch(repo)
        assert head_sha != main_sha
        return repo, head_sha, main_sha

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
        repo, head_sha, main_sha = self._setup_boundary_repo(tmp_path)
        entry = self._make_success_entry(head_sha, main_sha)
        verdict = check_proposal(entry, worktree=str(repo), category='task_failure',
                                 run_git=_run_git, now=self._NOW)
        assert verdict['verdict'] == FRESH, f'expected fresh: {verdict}'

    def test_b_head_moved_aborts_with_git_anchored_reason(self, tmp_path):
        """(b) Extra commit on feature branch -> abort, reason contains both shas."""
        from orchestrator.b3_gate import ABORT, _run_git, check_proposal
        repo, head_sha, main_sha = self._setup_boundary_repo(tmp_path)
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
        repo, head_sha, main_sha = self._setup_boundary_repo(tmp_path)
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
        repo, head_sha, main_sha = self._setup_boundary_repo(tmp_path)
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
        repo, head_sha, main_sha = self._setup_boundary_repo(tmp_path)
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
        repo, head_sha, main_sha = self._setup_boundary_repo(tmp_path)

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
        _, head_sha, main_sha = self._setup_boundary_repo(tmp_path)
        entry = self._make_success_entry(head_sha, main_sha)
        assert 'status' not in entry, (
            f'success entry must not carry status key: {entry.keys()}'
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
        entry = {**LOW_RISK_ENTRY, 'block_reason': self._PM_REASON}
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
        entry = {
            **LOW_RISK_ENTRY,
            'risk_label': 'human-review-required',
            'block_reason': self._PM_REASON,
        }
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
        entry = {**LOW_RISK_ENTRY, 'block_reason': 'verify exhausted'}
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
        entry['head_sha'] = 'aaabbbccc'
        entry['main_sha'] = 'dddeeefff'

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
        entry = {**LOW_RISK_ENTRY, 'block_reason': production_reason}
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
        entry = {**LOW_RISK_ENTRY, 'proposal_text': 'Fix it — do not apply until reviewed'}
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
        entry = {**LOW_RISK_ENTRY, 'block_reason': 'this fix requires human rehearsal before merging'}
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
            LOW_RISK_ENTRY, worktree='/tmp', category='task_failure',
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
        entry = {**LOW_RISK_ENTRY, 'proposal_text': 'Fix it — do not apply until reviewed'}
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
            LOW_RISK_ENTRY, worktree='/tmp', category=None,
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
        return {
            **LOW_RISK_ENTRY,
            'block_class': 'merge_verify_red',
            'block_reason': block_reason,
        }

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
    and test_status_key_present_aborts above carry NO block_class and must
    keep passing unchanged).

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
            entry = {
                **LOW_RISK_ENTRY,
                'block_class': 'post_merge_red_main',
                'block_reason': 'verify exhausted',
                'risk_label': risk_label,
            }
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
        entry = {**LOW_RISK_ENTRY, 'block_class': 'merge_verify_red'}
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
        entry = {
            **LOW_RISK_ENTRY,
            'block_class': 'agent_failure',
            'status': 'investigation_failed',
            'risk_label': 'human-review-required',
        }
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
        entry = {
            **LOW_RISK_ENTRY,
            'block_class': 'merge_verify_red',
            'status': 'x',
            'risk_label': 'low',
        }
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
        entry = {**LOW_RISK_ENTRY, 'block_class': 'agent_failure'}
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
        entry = {
            **LOW_RISK_ENTRY,
            'block_class': 'merge_verify_red',
            'risk_label': 'human-review-required',
        }
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
        entry = {**LOW_RISK_ENTRY, 'block_class': 'merge_verify_red'}
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_fresh, now=self._NOW,
        )
        assert result['verdict'] == FRESH, f'expected FRESH, got {result}'

    def test_post_merge_red_main_entry_hard_aborts(self):
        """(iii) task 1680 retained VERBATIM: a POST_MERGE_RED_MAIN entry
        hard-aborts before risk_label/git checks even with risk_label='low'
        (LOW_RISK_ENTRY's default) — the merge-completion mode must never
        weaken this highest-blast-radius safety rail.
        """
        from orchestrator.b3_gate import ABORT, check_proposal
        entry = {**LOW_RISK_ENTRY, 'block_class': 'post_merge_red_main'}
        result = check_proposal(
            entry, worktree='/tmp', category='task_failure',
            run_git=fake_git_never_called, now=self._NOW,
        )
        assert result['verdict'] == ABORT, f'expected ABORT, got {result}'
        assert 'post-merge red-main fix-forward class' in result['reason'], (
            f"expected the task-1680 reason text, got: {result['reason']!r}"
        )
