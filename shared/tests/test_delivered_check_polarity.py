"""Tests for ``shared.delivered_check_polarity`` — the authoring-time
delivered-check polarity lint (task 3500).

THE PARITY LAYER (this file's first two classes). ``build_grep_argv`` and
``interpret_grep_rc`` are the single source of truth for grep-check
semantics: ``orchestrator.delivered_checks._run_grep_check`` delegates to
them (step-4) so the authoring-time gate and the runtime gate can never
drift on POSIX-ERE-via-``git grep -E`` vs Python ``re``, on the ``-e``
separator that keeps a leading-dash pattern from being parsed as a git
option, on the ``--`` pathspec placement, or on the ``rc >= 2 → ERRORED``
boundary. Every argv assertion below is byte-for-byte the argv
``orchestrator/tests/test_delivered_check_gate.py::TestRunnerGrepKind``
already pins through the public ``run_delivered_check`` entry point; the
two suites are deliberate mirrors, and a divergence should redden both.

THE NON-VACUITY RULE (this file's later classes). A sound
``delivered_check`` must FAIL at the authoring tree and PASS once its
producer lands. At authoring time the reference tree is free — the task
has not been implemented yet, so HEAD *is* the pre-task tree — which
collapses the whole classification into one measured 2x2 over
``(expect, matches-HEAD)``. Those tests build a REAL throwaway git repo
in ``tmp_path`` and shell out to the same ``git grep`` the runtime gate
uses, because the rule is a measurement of the repo rather than a
heuristic over the check's name (design decision #1).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from shared.delivered_check_polarity import (
    CheckOutcome,
    build_grep_argv,
    evaluate_grep_at_tree,
    interpret_grep_rc,
    lint_delivered_checks,
)

# ---------------------------------------------------------------------------
# TestBuildGrepArgv (task 3500 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestBuildGrepArgv:
    """``git -C <root> grep -E -e <pattern> <ref> [-- <paths...>]``."""

    def test_argv_omits_dashdash_when_paths_empty(self):
        assert build_grep_argv('FooBar', [], project_root='/proj', ref='main') == [
            'git', '-C', '/proj', 'grep', '-E', '-e', 'FooBar', 'main',
        ]

    def test_argv_appends_dashdash_and_paths_when_present(self):
        assert build_grep_argv(
            'FooBar', ['src/a.py', 'src/b.py'], project_root='/proj', ref='main'
        ) == [
            'git', '-C', '/proj', 'grep', '-E', '-e', 'FooBar', 'main',
            '--', 'src/a.py', 'src/b.py',
        ]

    def test_argv_omits_dashdash_when_paths_is_none(self):
        """``DeliveredCheckMeta.paths`` defaults to ``[]``, but the raw dict a
        lint sees may carry ``None``; both mean "no pathspec"."""
        assert build_grep_argv('FooBar', None, project_root='/proj', ref='main') == [
            'git', '-C', '/proj', 'grep', '-E', '-e', 'FooBar', 'main',
        ]

    def test_project_root_is_stringified(self):
        """``project_root`` is ``str | Path`` at both call sites."""
        argv = build_grep_argv('X', [], project_root=Path('/proj/sub'), ref='HEAD')

        assert argv[2] == '/proj/sub'
        assert all(isinstance(part, str) for part in argv)

    def test_ref_is_positional_before_the_pathspec_separator(self):
        argv = build_grep_argv('X', ['a'], project_root='/proj', ref='HEAD')

        assert argv.index('HEAD') < argv.index('--')

    @pytest.mark.parametrize('pattern', ['-foo', '--force', '-e', '-E'])
    def test_pattern_starting_with_dash_lands_after_the_e_separator(self, pattern):
        """The explicit ``-e`` separator (reviewer_comprehensive amendment)
        keeps a pattern beginning with ``'-'`` from being parsed by ``git
        grep`` as an option instead of the search pattern."""
        argv = build_grep_argv(pattern, [], project_root='/proj', ref='main')

        assert argv == ['git', '-C', '/proj', 'grep', '-E', '-e', pattern, 'main']
        # The pattern occupies the slot IMMEDIATELY after '-e' — never earlier.
        assert argv[argv.index('-e') + 1] == pattern


# ---------------------------------------------------------------------------
# TestCheckOutcome / TestInterpretGrepRc (task 3500 — step-1 RED / step-2 GREEN)
# ---------------------------------------------------------------------------


class TestCheckOutcome:
    """The three-valued verdict enum."""

    def test_members_are_distinct(self):
        members = {CheckOutcome.PASS, CheckOutcome.FAIL, CheckOutcome.ERRORED}

        assert len(members) == 3

    def test_all_three_members_exist(self):
        assert CheckOutcome.PASS is not None
        assert CheckOutcome.FAIL is not None
        assert CheckOutcome.ERRORED is not None


class TestInterpretGrepRc:
    """rc → verdict, pinned to ``_run_grep_check``'s mapping exactly."""

    @pytest.mark.parametrize('rc', [2, 3, 127, 128, 129])
    @pytest.mark.parametrize('expect', ['present', 'absent', None])
    def test_rc_ge_2_is_errored_for_every_expect(self, rc, expect):
        """A git error is ERRORED regardless of polarity — including a
        pathspec that matches nothing in the tree, which exits >= 2 and so
        yields ERRORED rather than FAIL."""
        assert interpret_grep_rc(rc, expect) is CheckOutcome.ERRORED

    @pytest.mark.parametrize(
        ('rc', 'expected'),
        [(0, CheckOutcome.PASS), (1, CheckOutcome.FAIL)],
    )
    def test_expect_present(self, rc, expected):
        assert interpret_grep_rc(rc, 'present') is expected

    @pytest.mark.parametrize(
        ('rc', 'expected'),
        [(0, CheckOutcome.FAIL), (1, CheckOutcome.PASS)],
    )
    def test_expect_absent(self, rc, expected):
        assert interpret_grep_rc(rc, 'absent') is expected

    @pytest.mark.parametrize('rc', [0, 1, 2, 128])
    def test_expect_none_behaves_exactly_like_absent(self, rc):
        """Pins ``_run_grep_check``'s existing ELSE-branch semantics: it
        computes ``matched if expect == 'present' else not matched``, so any
        non-``'present'`` value — including ``None`` — takes the absent arm.
        Recorded here so the delegation refactor cannot quietly change it."""
        assert interpret_grep_rc(rc, None) is interpret_grep_rc(rc, 'absent')


# ---------------------------------------------------------------------------
# The real-git fixture tree (task 3500 — step-5)
# ---------------------------------------------------------------------------
#
# THE 2x2 IS A MEASUREMENT, NOT A HEURISTIC (design decision #1), so it can
# only be tested against a real tree. Every test below builds a throwaway git
# repo in `tmp_path` and shells out to the same `git grep` the runtime gate
# uses — modelled on
# `orchestrator/tests/test_delivered_check_gate_e2e.py::_init_git_repo`.


def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', '-C', str(root), *args], check=True, capture_output=True, text=True
    )


def _init_git_repo(root: Path, files: dict[str, str]) -> Path:
    """Initialize a real repo at *root* on branch main, seeded with *files*
    and committed, so ``HEAD`` resolves.

    ``HEAD`` — not a historical SHA — is the whole point: at AUTHORING time
    the task has not been implemented yet, so HEAD *is* the pre-task tree
    (design decision #2). These fixtures therefore stand in for "the repo as
    the author sees it at ``commit_planning`` time".
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'init', '-b', 'main', str(root)], check=True, capture_output=True, text=True
    )
    _run_git(root, 'config', 'user.email', 'polarity-test@example.com')
    _run_git(root, 'config', 'user.name', 'Polarity Test')
    _run_git(root, 'config', 'commit.gpgsign', 'false')
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding='utf-8')
    _run_git(root, 'add', '-A')
    _run_git(root, 'commit', '--no-verify', '-m', 'seed the authoring tree')
    return root


#: The authoring tree every 2x2 test measures against. ``existing_symbol``
#: is present on a LIVE CODE line (not a comment, not a manifest
#: ``pattern:`` line) so the plain ``vacuous_present`` code is what step-8's
#: refinements will leave in place; ``src/producer.py`` is the file a
#: producer task would declare in ``metadata.files``.
_SEED_TREE = {
    'README.md': '# polarity fixture repo\n',
    'src/producer.py': (
        'def existing_symbol():\n'
        '    return 1\n'
    ),
}


@pytest.fixture
def authoring_repo(tmp_path: Path) -> Path:
    """A real git repo standing in for the tree at ``commit_planning`` time."""
    return _init_git_repo(tmp_path / 'repo', _SEED_TREE)


def _grep_check(**overrides: object) -> dict[str, object]:
    """A well-formed ``metadata.delivered_checks`` grep entry."""
    check: dict[str, object] = {
        'name': 'cap',
        'kind': 'grep',
        'pattern': 'existing_symbol',
        'expect': 'present',
        'paths': ['src/producer.py'],
    }
    check.update(overrides)
    return check


# ---------------------------------------------------------------------------
# TestEvaluateGrepAtTree (task 3500 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestEvaluateGrepAtTree:
    """``evaluate_grep_at_tree`` runs the SHARED argv against a real tree.

    It is ``build_grep_argv`` + ``subprocess.run`` + ``interpret_grep_rc``
    and nothing else, so a verdict it reaches is by construction the verdict
    ``orchestrator.delivered_checks._run_grep_check`` would reach against
    the same tree (design decision #5).
    """

    def test_present_pattern_that_matches_head_is_pass(self, authoring_repo):
        assert (
            evaluate_grep_at_tree(
                'existing_symbol',
                ['src/producer.py'],
                expect='present',
                repo_root=authoring_repo,
                ref='HEAD',
            )
            is CheckOutcome.PASS
        )

    def test_present_pattern_that_does_not_match_head_is_fail(self, authoring_repo):
        assert (
            evaluate_grep_at_tree(
                'not_yet_landed_symbol',
                ['src/producer.py'],
                expect='present',
                repo_root=authoring_repo,
                ref='HEAD',
            )
            is CheckOutcome.FAIL
        )

    def test_absent_pattern_that_matches_head_is_fail(self, authoring_repo):
        assert (
            evaluate_grep_at_tree(
                'existing_symbol',
                ['src/producer.py'],
                expect='absent',
                repo_root=authoring_repo,
                ref='HEAD',
            )
            is CheckOutcome.FAIL
        )

    def test_absent_pattern_that_does_not_match_head_is_pass(self, authoring_repo):
        assert (
            evaluate_grep_at_tree(
                'not_yet_landed_symbol',
                ['src/producer.py'],
                expect='absent',
                repo_root=authoring_repo,
                ref='HEAD',
            )
            is CheckOutcome.PASS
        )

    def test_empty_paths_searches_the_whole_tree(self, authoring_repo):
        assert (
            evaluate_grep_at_tree(
                'polarity fixture repo',
                [],
                expect='present',
                repo_root=authoring_repo,
                ref='HEAD',
            )
            is CheckOutcome.PASS
        )

    def test_non_repo_root_is_errored_not_a_verdict(self, tmp_path):
        """FAIL OPEN ON INFRASTRUCTURE (design decision #3). A directory that
        exists but is not a git repo is exactly the shape the existing suites
        commit against (``test_task_tools.py`` against ``/project``,
        ``real_task_stack`` against a bare ``tmp_path``), so this path must
        be ERRORED — never a silent PASS, and never a FAIL that would read as
        a verdict."""
        not_a_repo = tmp_path / 'not-a-repo'
        not_a_repo.mkdir()

        for expect in ('present', 'absent'):
            assert (
                evaluate_grep_at_tree(
                    'existing_symbol', [], expect=expect, repo_root=not_a_repo, ref='HEAD'
                )
                is CheckOutcome.ERRORED
            )

    def test_missing_root_is_errored_and_never_raises(self, tmp_path):
        assert (
            evaluate_grep_at_tree(
                'existing_symbol',
                [],
                expect='present',
                repo_root=tmp_path / 'does-not-exist',
                ref='HEAD',
            )
            is CheckOutcome.ERRORED
        )

    def test_unresolvable_ref_is_errored(self, authoring_repo):
        assert (
            evaluate_grep_at_tree(
                'existing_symbol',
                [],
                expect='present',
                repo_root=authoring_repo,
                ref='no-such-ref',
            )
            is CheckOutcome.ERRORED
        )


# ---------------------------------------------------------------------------
# TestNonVacuityRule (task 3500 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestNonVacuityRule:
    """THE ONE RULE, one test per 2x2 cell.

    A sound ``delivered_check`` must FAIL at the authoring tree and PASS
    once its producer lands (SCOPE item 4's 0→N / N→0 transition). A check
    that is already green the day it is written can never signal anything,
    so evaluating it against HEAD at authoring time decides the whole
    classification::

        ================  ===============  ==============
                          expect: present  expect: absent
        ================  ===============  ==============
        matches HEAD      REJECT           healthy
        no match at HEAD  healthy          REJECT
        ================  ===============  ==============
    """

    def test_present_that_already_matches_is_rejected_as_vacuous(self, authoring_repo):
        """Cell (a): the check passes the day it is written, so landing the
        producer cannot change its verdict — it gates nothing."""
        findings = lint_delivered_checks(
            [_grep_check(name='producer-symbol')],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        finding = findings[0]
        assert finding.check_name == 'producer-symbol'
        assert finding.severity == 'reject'
        assert finding.code == 'vacuous_present'

    def test_present_that_does_not_match_is_healthy(self, authoring_repo):
        """Cell (b): the MAJORITY case — a forward-looking check waiting on a
        producer that has not landed yet. This must NEVER be flagged; a false
        reject here would block legitimate planning on a hard gate."""
        findings = lint_delivered_checks(
            [_grep_check(pattern='not_yet_landed_symbol')],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_absent_that_still_matches_is_healthy(self, authoring_repo):
        """Cell (c): the removal has not happened yet, which is precisely the
        state a sound ``expect='absent'`` check is authored in."""
        findings = lint_delivered_checks(
            [_grep_check(expect='absent')],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_absent_that_does_not_match_is_rejected_as_vacuous(self, authoring_repo):
        """Cell (d): nothing is left to remove, so the check is green on the
        day it is written and its dependent is gated on nothing."""
        findings = lint_delivered_checks(
            [_grep_check(pattern='not_yet_landed_symbol', expect='absent')],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].severity == 'reject'
        assert findings[0].code == 'vacuous_absent'

    def test_rejection_message_states_the_invariant_in_the_authors_terms(
        self, authoring_repo
    ):
        """The message is the author's only feedback at a hard reject, so it
        must say what a sound check looks like, not merely that this one is
        bad."""
        findings = lint_delivered_checks(
            [_grep_check()], files=['src/producer.py'], repo_root=authoring_repo, ref='HEAD'
        )

        message = findings[0].message.lower()
        assert 'authoring' in message
        assert 'already' in message

    def test_script_kind_yields_no_finding(self, authoring_repo):
        """The 2x2 is a statement about grep polarity; a script check has no
        ``expect`` to invert."""
        findings = lint_delivered_checks(
            [{'name': 'cap', 'kind': 'script', 'script': 'scripts/x.sh', 'timeout_secs': 5}],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_every_check_is_evaluated_independently(self, authoring_repo):
        """A batch mixes healthy and vacuous checks; the lint reports one
        finding per offender and leaves the healthy ones alone."""
        findings = lint_delivered_checks(
            [
                _grep_check(name='vacuous'),
                _grep_check(name='healthy', pattern='not_yet_landed_symbol'),
                _grep_check(name='also-vacuous', pattern='return 1'),
            ],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert [f.check_name for f in findings] == ['vacuous', 'also-vacuous']
        assert {f.code for f in findings} == {'vacuous_present'}


# ---------------------------------------------------------------------------
# TestMode1PolarityInversionSpecimen (task 3500 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestMode1PolarityInversionSpecimen:
    """The measured 5799 → 5919 wedge, reproduced.

    Task 5799 authored ``expect='present'`` checks for a pattern its own
    diff was scoped to REMOVE, and 5919 sat blocked behind them forever.
    The prior plan proposed catching this with a ``{pre-state, still,
    unchanged, old, legacy, before}`` vocabulary matched against the check
    NAME; design decision #1 deleted that entirely, because the vacuity rule
    subsumes it: a pattern the task will remove is NECESSARILY present at
    authoring time — that is exactly what makes it removable — so the check
    already passes and the 2x2 rejects it.

    This test therefore names the check something the deleted vocabulary
    would never have matched, to assert the catch depends on no name tokens
    at all.
    """

    def test_a_5799_shaped_check_is_rejected_with_no_name_vocabulary(
        self, authoring_repo
    ):
        check = _grep_check(
            # Deliberately NOT '*-still-*' / '*-pre-state-*' / '*-legacy-*':
            # the deleted name heuristic would have passed this straight
            # through. The measurement catches it regardless.
            name='producer-emits-symbol',
            pattern='existing_symbol',
            expect='present',
            paths=['src/producer.py'],
        )

        findings = lint_delivered_checks(
            # The task's own declared files name the very file it will edit —
            # as 5799's did.
            [check],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].severity == 'reject'
        assert findings[0].code == 'vacuous_present'
        assert findings[0].check_name == 'producer-emits-symbol'


# ---------------------------------------------------------------------------
# TestLintFailsOpenOnInfrastructure (task 3500 — step-5 RED / step-6 GREEN)
# ---------------------------------------------------------------------------


class TestLintFailsOpenOnInfrastructure:
    """FAIL CLOSED ON A VERDICT, FAIL OPEN ON INFRASTRUCTURE (decision #3).

    "Structurally unable to accept an unvalidated check" means the
    validation is unconditionally APPLIED — it cannot mean blocking whenever
    git is unavailable, which would convert an availability failure into a
    total planning outage (a worse wedge than the one being fixed). So an
    unevaluable check is its own REPORTED disposition: never a silent pass,
    and never a reject.
    """

    def test_unevaluable_check_yields_an_errored_finding_not_a_reject(self, tmp_path):
        not_a_repo = tmp_path / 'not-a-repo'
        not_a_repo.mkdir()

        findings = lint_delivered_checks(
            [_grep_check(name='cap-a')],
            files=['src/producer.py'],
            repo_root=not_a_repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].check_name == 'cap-a'
        assert findings[0].severity == 'errored'
        assert findings[0].code == 'unevaluable'

    def test_unevaluable_is_never_silently_passed(self, tmp_path):
        """The failure mode this guards: swallowing the error would let an
        unvalidated check through while REPORTING nothing, which is the
        silent-degradation shape the project's loud-over-silent norm forbids
        — and would make the gate's own coverage invisible."""
        not_a_repo = tmp_path / 'not-a-repo'
        not_a_repo.mkdir()

        findings = lint_delivered_checks(
            [_grep_check(name='a'), _grep_check(name='b', expect='absent')],
            files=[],
            repo_root=not_a_repo,
            ref='HEAD',
        )

        assert [f.check_name for f in findings] == ['a', 'b']
        assert {f.severity for f in findings} == {'errored'}

    def test_lint_does_not_raise_on_a_missing_root(self, tmp_path):
        findings = lint_delivered_checks(
            [_grep_check()],
            files=['src/producer.py'],
            repo_root=tmp_path / 'does-not-exist',
            ref='HEAD',
        )

        assert [f.severity for f in findings] == ['errored']
