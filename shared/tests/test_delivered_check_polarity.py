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
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from shared.capability_manifest import MECHANICAL_CHECK_KINDS, DeliveredCheckMeta
from shared.delivered_check_polarity import (
    CheckFinding,
    CheckOutcome,
    build_grep_argv,
    build_path_argv,
    evaluate_grep_at_tree,
    evaluate_path_at_tree,
    extract_delivered_checks,
    interpret_grep_rc,
    interpret_path_listing,
    lint_delivered_checks,
    polarity_error,
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


# ---------------------------------------------------------------------------
# TestSelfReferentialRefinement (task 3500 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------
#
# THE REFINEMENTS ARE MESSAGES, NOT GATES. At authoring time a
# self-referential match and a comment-only match are both just sub-species
# of "a check that already matches", so the 2x2 has ALREADY rejected them by
# the time a refinement runs (design decision #1). What the refinement buys
# is legibility: `vacuous_present` tells an author their check is green on
# day one, `vacuous_present_self_referential` tells them WHY — the only
# thing their pattern matches is the descriptor that declares it.

#: The measured t2863 specimen's shape: match #1 of the capability token
#: under `plans/` was the descriptor's OWN `pattern:` line.
_SELF_REF_TREE = {
    'plans/foo-prd.md': (
        '# Foo PRD\n'
        '\n'
        'The capability lands as foo_capability_token in the scheduler.\n'
    ),
    'plans/foo-prd.capability-manifest.yaml': (
        'capabilities:\n'
        '  - name: foo-cap\n'
        '    delivered_check:\n'
        '      kind: grep\n'
        '      pattern: foo_capability_token\n'
        '      expect: present\n'
    ),
    'src/scheduler.py': 'def dispatch():\n    return None\n',
}


class TestSelfReferentialRefinement:
    """An ``expect='present'`` pattern whose only matches are the descriptor
    that declares it, plus the PRD that describes it.

    This is the emptiest possible check: it is satisfied by its own
    existence. It stays a REJECT — the 2x2 already decided that — but earns
    the sharper ``vacuous_present_self_referential`` code so the author is
    told the actual problem rather than left to rediscover it.
    """

    @pytest.fixture
    def self_ref_repo(self, tmp_path: Path) -> Path:
        return _init_git_repo(tmp_path / 'repo', _SELF_REF_TREE)

    def _check(self, **overrides: object) -> dict[str, object]:
        check: dict[str, object] = {
            'name': 'foo-cap',
            'kind': 'grep',
            'pattern': 'foo_capability_token',
            'expect': 'present',
            'paths': [],
            # Threaded through by the stamper, which knows which sidecar it
            # is copying from (step-18).
            'manifest_path': 'plans/foo-prd.capability-manifest.yaml',
        }
        check.update(overrides)
        return check

    def test_matches_confined_to_the_descriptor_family_get_the_sharper_code(
        self, self_ref_repo
    ):
        findings = lint_delivered_checks(
            [self._check()], files=['src/scheduler.py'], repo_root=self_ref_repo, ref='HEAD'
        )

        assert len(findings) == 1
        assert findings[0].severity == 'reject'
        assert findings[0].code == 'vacuous_present_self_referential'

    def test_message_names_the_self_match(self, self_ref_repo):
        findings = lint_delivered_checks(
            [self._check()], files=['src/scheduler.py'], repo_root=self_ref_repo, ref='HEAD'
        )

        assert 'plans/foo-prd.capability-manifest.yaml' in findings[0].message

    def test_without_a_manifest_path_it_falls_back_to_the_unrefined_code(
        self, self_ref_repo
    ):
        """``commit_planning`` has no sidecar to point at, so the refinement
        is simply unavailable there. Falling back to plain ``vacuous_present``
        keeps the REJECT — losing the refinement must never lose the
        rejection."""
        findings = lint_delivered_checks(
            [self._check(manifest_path=None)],
            files=['src/scheduler.py'],
            repo_root=self_ref_repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].severity == 'reject'
        assert findings[0].code == 'vacuous_present'

    def test_a_match_outside_the_family_defeats_the_refinement(self, tmp_path):
        """One live match in real code and the check is no longer merely
        self-referential — it is an ordinary vacuous check."""
        tree = dict(_SELF_REF_TREE)
        tree['src/scheduler.py'] = 'foo_capability_token = 1\n'
        repo = _init_git_repo(tmp_path / 'repo', tree)

        findings = lint_delivered_checks(
            [self._check()], files=['src/scheduler.py'], repo_root=repo, ref='HEAD'
        )

        assert len(findings) == 1
        assert findings[0].code == 'vacuous_present'


# ---------------------------------------------------------------------------
# TestCommentOnlyRefinement (task 3500 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestCommentOnlyRefinement:
    """An ``expect='present'`` pattern whose every match line is a comment.

    The measured t2792 specimen: the sole ``archive_task_transcripts`` match
    in ``git_ops.py`` is a comment. A comment cannot be a capability, so a
    check satisfied only by comments is asserting that someone wrote the
    word down — which is true before the producer lands and stays true if it
    never does.
    """

    @pytest.mark.parametrize(
        ('marker', 'rel_path'),
        [
            ('#', 'src/mod.py'),
            ('//', 'src/mod.ts'),
            ('*', 'src/mod.c'),
            ('"""', 'src/doc.py'),
            ("'''", 'src/doc2.py'),
        ],
    )
    def test_every_match_on_a_comment_line_gets_the_sharper_code(
        self, tmp_path, marker, rel_path
    ):
        repo = _init_git_repo(
            tmp_path / 'repo',
            {
                rel_path: (
                    f'    {marker} archive_task_transcripts is handled elsewhere\n'
                    'def unrelated():\n'
                    '    return 0\n'
                ),
            },
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'archival-cap',
                    'kind': 'grep',
                    'pattern': 'archive_task_transcripts',
                    'expect': 'present',
                    'paths': [rel_path],
                }
            ],
            files=[rel_path],
            repo_root=repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].severity == 'reject'
        assert findings[0].code == 'vacuous_present_comment_only'

    def test_one_live_code_line_among_the_comments_falls_back_to_plain_vacuous(
        self, tmp_path
    ):
        """The CONTROL. The refinement claims "every match is a comment"; a
        single live match makes that claim false, and over-claiming it would
        put a wrong diagnosis in front of the author on a hard reject."""
        repo = _init_git_repo(
            tmp_path / 'repo',
            {
                'src/mod.py': (
                    '# archive_task_transcripts is handled elsewhere\n'
                    'def archive_task_transcripts():\n'
                    '    return 0\n'
                ),
            },
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'archival-cap',
                    'kind': 'grep',
                    'pattern': 'archive_task_transcripts',
                    'expect': 'present',
                    'paths': ['src/mod.py'],
                }
            ],
            files=['src/mod.py'],
            repo_root=repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].code == 'vacuous_present'


# ---------------------------------------------------------------------------
# TestRefinementPrecedence (task 3500 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestRefinementPrecedence:
    """AT MOST ONE finding per check, with a deterministic winner.

    Both refinements can hold at once (a manifest ``pattern:`` line reached
    through a markdown bullet in the sibling PRD is self-referential AND
    comment-shaped). Emitting two findings would double-count one defect in
    the reject payload, so the order is pinned: self-referential is the more
    specific diagnosis and wins.
    """

    def test_self_reference_outranks_comment_only(self, tmp_path):
        repo = _init_git_repo(
            tmp_path / 'repo',
            {
                # Every match line here is BOTH inside the descriptor family
                # and comment-shaped (a markdown bullet starts with '*').
                'plans/foo-prd.md': '# Foo PRD\n\n* foo_capability_token lands here\n',
                'plans/foo-prd.capability-manifest.yaml': (
                    'capabilities:\n'
                    '  - name: foo-cap\n'
                    '    delivered_check:\n'
                    '      # pattern: foo_capability_token\n'
                    '      kind: grep\n'
                ),
                'src/scheduler.py': 'def dispatch():\n    return None\n',
            },
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'foo-cap',
                    'kind': 'grep',
                    'pattern': 'foo_capability_token',
                    'expect': 'present',
                    'paths': [],
                    'manifest_path': 'plans/foo-prd.capability-manifest.yaml',
                }
            ],
            files=['src/scheduler.py'],
            repo_root=repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].code == 'vacuous_present_self_referential'


# ---------------------------------------------------------------------------
# TestFilenameShaped (task 3500 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------
#
# MODE 3 is the ONE class the 2x2 cannot see: at authoring time the file the
# pattern names does not exist yet, so nothing matches, the check evaluates
# FAIL, and it looks like an ordinary healthy forward-looking check. It only
# reveals itself once the producer lands and the check STILL fails, because
# the module never mentions its own name. Design decision #4 takes SCOPE
# item 3's SECOND sanctioned option for it (reject a filename-shaped grep)
# rather than its first (a new `kind='path'`), which is filed as a follow-up.


class TestFilenameShaped:
    """An ``expect='present'`` pattern that matches a tracked PATH but no
    file CONTENT.

    The measured task-3536 specimen: ``test_workflow_merge_gating_strand``
    is a tracked module under ``orchestrator/tests/`` and ``git grep`` for
    it there yields ZERO content matches — a test module does not mention
    its own name.
    """

    @pytest.fixture
    def strand_repo(self, tmp_path: Path) -> Path:
        return _init_git_repo(
            tmp_path / 'repo',
            {
                'orchestrator/tests/test_workflow_merge_gating_strand.py': (
                    'import pytest\n\n\ndef test_strand():\n    assert True\n'
                ),
                'orchestrator/tests/test_other.py': 'def test_other():\n    assert True\n',
            },
        )

    def test_pattern_matching_only_a_tracked_path_is_rejected(self, strand_repo):
        findings = lint_delivered_checks(
            [
                {
                    'name': 'gating-strand',
                    'kind': 'grep',
                    'pattern': 'test_workflow_merge_gating_strand',
                    'expect': 'present',
                    'paths': ['orchestrator/tests'],
                }
            ],
            files=['orchestrator/tests'],
            repo_root=strand_repo,
            ref='HEAD',
        )

        assert len(findings) == 1
        assert findings[0].severity == 'reject'
        assert findings[0].code == 'filename_shaped'

    def test_message_tells_the_author_to_assert_a_symbol_inside_the_file(
        self, strand_repo
    ):
        findings = lint_delivered_checks(
            [
                {
                    'name': 'gating-strand',
                    'kind': 'grep',
                    'pattern': 'test_workflow_merge_gating_strand',
                    'expect': 'present',
                    'paths': ['orchestrator/tests'],
                }
            ],
            files=['orchestrator/tests'],
            repo_root=strand_repo,
            ref='HEAD',
        )

        message = findings[0].message.lower()
        assert 'inside' in message
        assert 'filename' in message or 'path' in message

    def test_message_steers_a_file_existence_capability_to_kind_path(self, strand_repo):
        """Since task 4743 a file-existence capability has its own kind, so
        the detective rule must name the prescriptive fix — including the
        exact ``paths`` entry to write — not only the grep workaround."""
        findings = lint_delivered_checks(
            [
                {
                    'name': 'gating-strand',
                    'kind': 'grep',
                    'pattern': 'test_workflow_merge_gating_strand',
                    'expect': 'present',
                    'paths': ['orchestrator/tests'],
                }
            ],
            files=['orchestrator/tests'],
            repo_root=strand_repo,
            ref='HEAD',
        )

        message = findings[0].message
        assert "kind='path'" in message
        assert "'orchestrator/tests/test_workflow_merge_gating_strand.py'" in message

    def test_no_content_match_and_no_path_match_is_healthy(self, strand_repo):
        """The CONTROL, and the reason this rule is narrow: an ordinary
        forward-looking check has zero content matches too. Only the PATH
        coincidence separates the two."""
        findings = lint_delivered_checks(
            [
                {
                    'name': 'forward-looking',
                    'kind': 'grep',
                    'pattern': 'not_yet_landed_symbol',
                    'expect': 'present',
                    'paths': ['orchestrator/tests'],
                }
            ],
            files=['orchestrator/tests'],
            repo_root=strand_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_absent_polarity_is_never_filename_shaped(self, strand_repo):
        """The rule is scoped to ``expect='present'``. The absent polarity
        with the same pattern is not MODE 3 at all — it is the 2x2's own
        cell (d), because a pattern with no content match today has nothing
        left for the task to remove. Pinned so the MODE 3 rule cannot leak
        across polarities and relabel an ordinary vacuous check."""
        findings = lint_delivered_checks(
            [
                {
                    'name': 'strand-removed',
                    'kind': 'grep',
                    'pattern': 'test_workflow_merge_gating_strand',
                    'expect': 'absent',
                    'paths': ['orchestrator/tests'],
                }
            ],
            files=['orchestrator/tests'],
            repo_root=strand_repo,
            ref='HEAD',
        )

        assert [f.code for f in findings] == ['vacuous_absent']


# ---------------------------------------------------------------------------
# TestAbsentOverbroad (task 3500 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestAbsentOverbroad:
    """MODE 2/2b is a WARN, never a reject.

    An ``expect='absent'`` pattern that also matches files the task does not
    own will keep failing after the task lands, wedging its dependent. But
    it is genuinely UNDECIDABLE at authoring time — task 3534's pattern
    legitimately matched inside the very file it owned — and this gate is
    hard-blocking, so a false reject here costs more than a missed catch.
    """

    @pytest.fixture
    def two_file_repo(self, tmp_path: Path) -> Path:
        return _init_git_repo(
            tmp_path / 'repo',
            {
                'src/owned.py': 'legacy_token = 1\n',
                'src/elsewhere.py': 'legacy_token = 2\n',
            },
        )

    def _check(self) -> dict[str, object]:
        return {
            'name': 'legacy-gone',
            'kind': 'grep',
            'pattern': 'legacy_token',
            'expect': 'absent',
            'paths': [],
        }

    def test_matches_outside_the_declared_files_warn(self, two_file_repo):
        findings = lint_delivered_checks(
            [self._check()], files=['src/owned.py'], repo_root=two_file_repo, ref='HEAD'
        )

        assert len(findings) == 1
        assert findings[0].severity == 'warn'
        assert findings[0].code == 'absent_overbroad'

    def test_detail_names_the_out_of_scope_files(self, two_file_repo):
        findings = lint_delivered_checks(
            [self._check()], files=['src/owned.py'], repo_root=two_file_repo, ref='HEAD'
        )

        joined = '\n'.join(findings[0].detail)
        assert 'src/elsewhere.py' in joined
        # The file the task DOES own is not the author's problem.
        assert 'src/owned.py' not in joined

    def test_all_matches_inside_the_declared_files_is_healthy(self, two_file_repo):
        findings = lint_delivered_checks(
            [self._check()],
            files=['src/owned.py', 'src/elsewhere.py'],
            repo_root=two_file_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_a_declared_directory_covers_the_files_beneath_it(self, two_file_repo):
        """``metadata.files`` routinely names a DIRECTORY; treating that as
        covering nothing would warn on every task that declares scope
        coarsely — a false positive on the common shape."""
        findings = lint_delivered_checks(
            [self._check()], files=['src'], repo_root=two_file_repo, ref='HEAD'
        )

        assert findings == []

    def test_a_warn_is_never_promoted_to_a_reject(self, two_file_repo):
        """Pins the disposition itself, not just the code: MODE 2 is
        undecidable, so this finding must stay out of the reject payload no
        matter how the codes are renamed later."""
        findings = lint_delivered_checks(
            [self._check()], files=[], repo_root=two_file_repo, ref='HEAD'
        )

        assert [f.severity for f in findings] == ['warn']


# ---------------------------------------------------------------------------
# TestExtractDeliveredChecks (task 3500 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


class TestExtractDeliveredChecks:
    """BENIGN-ABSENT, mirroring
    ``fused_memory.middleware.lock_charter_guard.extract_files`` exactly.

    The extractor runs on the wire path of ``commit_planning``, where
    ``metadata`` arrives as a dict OR a JSON string OR nothing at all. Every
    malformed shape resolves to "no checks to lint", never to an exception:
    a gate that can raise on a metadata shape it did not anticipate would
    take down planning for a defect it was not even built to catch.
    """

    def test_none_is_no_checks(self):
        assert extract_delivered_checks(None) == []

    def test_empty_string_is_no_checks(self):
        assert extract_delivered_checks('') == []

    def test_unparseable_json_string_is_no_checks(self):
        assert extract_delivered_checks('{not json at all') == []

    def test_json_string_decoding_to_a_non_object_is_no_checks(self):
        assert extract_delivered_checks('[1, 2, 3]') == []
        assert extract_delivered_checks('"a string"') == []
        assert extract_delivered_checks('42') == []

    @pytest.mark.parametrize('metadata', [[], ['a'], 42, 3.5, True, object()])
    def test_a_non_str_non_dict_is_no_checks(self, metadata):
        assert extract_delivered_checks(metadata) == []

    def test_dict_without_the_key_is_no_checks(self):
        assert extract_delivered_checks({'files': ['a.py']}) == []

    @pytest.mark.parametrize(
        'value', [None, 'grep', 42, {'name': 'cap'}, True]
    )
    def test_delivered_checks_that_is_not_a_list_is_no_checks(self, value):
        assert extract_delivered_checks({'delivered_checks': value}) == []

    def test_a_valid_json_string_blob_round_trips(self):
        blob = (
            '{"files": ["a.py"], "delivered_checks": '
            '[{"name": "cap", "kind": "grep", "pattern": "X", "expect": "present"}]}'
        )

        assert extract_delivered_checks(blob) == [
            {'name': 'cap', 'kind': 'grep', 'pattern': 'X', 'expect': 'present'}
        ]

    def test_non_dict_entries_inside_the_list_are_filtered_out(self):
        metadata = {
            'delivered_checks': [
                {'name': 'keeper', 'kind': 'grep'},
                'a bare string',
                None,
                42,
                ['nested', 'list'],
            ]
        }

        assert extract_delivered_checks(metadata) == [{'name': 'keeper', 'kind': 'grep'}]

    def test_a_dict_metadata_is_used_directly(self):
        checks = [{'name': 'cap', 'kind': 'script'}]

        assert extract_delivered_checks({'delivered_checks': checks}) == checks


# ---------------------------------------------------------------------------
# TestPolarityError (task 3500 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


def _reject(name: str, code: str = 'vacuous_present') -> CheckFinding:
    return CheckFinding(
        check_name=name,
        severity='reject',
        code=code,
        message=f'{name} is vacuous',
        detail=(f'{name}-site.py:1: token',),
    )


class TestPolarityError:
    """The reject payload, shaped like ``lock_charter_error`` on purpose.

    ``commit_planning`` already rejects a batch atomically for a lock-charter
    violation, and MCP callers handle that error through one code path.
    Mirroring its ``{error, error_type, <detail>, hint}`` convention means
    ``DeliveredCheckPolarityViolation`` needs no new caller handling at all.
    """

    def test_error_type_is_the_new_violation(self):
        payload = polarity_error([_reject('cap-a')], task_id='3500')

        assert payload['error_type'] == 'DeliveredCheckPolarityViolation'

    def test_key_set_matches_the_lock_charter_convention(self):
        payload = polarity_error([_reject('cap-a')], task_id='3500')

        assert set(payload) == {'error', 'error_type', 'checks', 'hint'}

    def test_error_names_the_task_in_the_parenthetical_convention(self):
        payload = polarity_error([_reject('cap-a')], task_id='3500')

        assert '(task 3500)' in payload['error']

    def test_error_names_every_reject_check_and_code(self):
        findings = [
            _reject('cap-a', 'vacuous_present'),
            _reject('cap-b', 'vacuous_absent'),
            _reject('cap-c', 'filename_shaped'),
        ]

        error = polarity_error(findings, task_id='3500')['error']

        for name in ('cap-a', 'cap-b', 'cap-c'):
            assert name in error
        for code in ('vacuous_present', 'vacuous_absent', 'filename_shaped'):
            assert code in error

    def test_checks_carries_one_structured_entry_per_reject(self):
        payload = polarity_error(
            [_reject('cap-a'), _reject('cap-b', 'vacuous_absent')], task_id='3500'
        )

        assert payload['checks'] == [
            {
                'name': 'cap-a',
                'code': 'vacuous_present',
                'message': 'cap-a is vacuous',
                'detail': ['cap-a-site.py:1: token'],
            },
            {
                'name': 'cap-b',
                'code': 'vacuous_absent',
                'message': 'cap-b is vacuous',
                'detail': ['cap-b-site.py:1: token'],
            },
        ]

    def test_warn_and_errored_findings_are_excluded(self):
        """A WARN is undecidable (MODE 2) and an ERRORED is an availability
        failure. Letting either into the reject payload would turn the
        deliberate non-blocking dispositions into blocking ones — the exact
        inversion decisions #3 and #8's WARN choice exist to prevent."""
        findings = [
            _reject('cap-a'),
            CheckFinding('cap-warn', 'warn', 'absent_overbroad', 'too broad'),
            CheckFinding('cap-err', 'errored', 'unevaluable', 'git said no'),
        ]

        payload = polarity_error(findings, task_id='3500')

        assert [c['name'] for c in payload['checks']] == ['cap-a']
        assert 'cap-warn' not in payload['error']
        assert 'cap-err' not in payload['error']

    def test_hint_prescribes_asserting_the_new_symbol_positively(self):
        """All three measured repairs took the same shape: assert the symbol
        the producer INTRODUCES with expect=present, rather than banning the
        one it removes. Naming that remedy is the difference between a reject
        an author can act on and one they can only be annoyed by."""
        hint = polarity_error([_reject('cap-a')], task_id='3500')['hint']

        assert 'new symbol' in hint.lower()
        assert 'expect=present' in hint

    def test_hint_names_kind_path_for_a_file_existence_capability(self):
        hint = polarity_error([_reject('cap-a')], task_id='3500')['hint']

        assert "kind='path'" in hint

    def test_hint_states_the_invariant(self):
        hint = polarity_error([_reject('cap-a')], task_id='3500')['hint'].lower()

        assert 'authoring' in hint
        assert 'lands' in hint or 'producer' in hint

    def test_task_id_is_optional(self):
        """``lock_charter_error``'s ``task_id`` is optional and the
        parenthetical simply disappears; mirroring that keeps the two
        payloads interchangeable for a caller that has no task id yet."""
        payload = polarity_error([_reject('cap-a')], task_id=None)

        assert '(task' not in payload['error']
        assert payload['error_type'] == 'DeliveredCheckPolarityViolation'


# ---------------------------------------------------------------------------
# TestLintImmunity (task 3500 — step-9 RED / step-10 GREEN)
# ---------------------------------------------------------------------------


class TestLintImmunity:
    """``lint_delivered_checks`` NEVER raises, on any input.

    Both wire points depend on this unconditionally and for opposite
    reasons: ``commit_planning`` would turn an exception into a planning
    outage, and the capability-manifest stamper is contractually
    never-raising, so an exception there would abort the very status flip it
    was called from.
    """

    def test_empty_batch(self, authoring_repo):
        assert lint_delivered_checks([], files=[], repo_root=authoring_repo) == []

    def test_script_kind_costs_no_subprocess(self, authoring_repo):
        """The short-circuit must come BEFORE the git call, not after it: a
        batch of script checks is common and should cost nothing."""
        with patch(
            'shared.delivered_check_polarity.subprocess.run',
            side_effect=AssertionError('lint must not shell out for a script check'),
        ):
            findings = lint_delivered_checks(
                [{'name': 'cap', 'kind': 'script', 'script': 'scripts/x.sh'}],
                files=[],
                repo_root=authoring_repo,
            )

        assert findings == []

    @pytest.mark.parametrize('name', [None, '', 42, ['cap'], {'a': 1}])
    def test_missing_or_non_string_name_is_skipped_before_any_git_call(
        self, authoring_repo, name
    ):
        """A finding is addressed to a check BY NAME; without a usable one
        there is nothing a caller could report or an author could fix, so the
        entry is skipped rather than reported under a name that does not
        exist."""
        with patch(
            'shared.delivered_check_polarity.subprocess.run',
            side_effect=AssertionError('lint must not shell out for a nameless check'),
        ):
            findings = lint_delivered_checks(
                [_grep_check(name=name)], files=[], repo_root=authoring_repo
            )

        assert findings == []

    @pytest.mark.parametrize('entry', [None, 'a string', 42, ['a', 'list'], object()])
    def test_a_non_dict_entry_is_skipped(self, authoring_repo, entry):
        findings = lint_delivered_checks(
            [entry, _grep_check(name='real')], files=[], repo_root=authoring_repo
        )

        assert [f.check_name for f in findings] == ['real']

    def test_non_string_paths_entries_are_dropped_not_crashed_on(self, authoring_repo):
        """A non-string in ``paths`` would be handed straight to
        ``subprocess`` as an argv element and raise ``TypeError``. Dropping
        it keeps the check EVALUABLE against the paths that are usable —
        degrading to a reject-or-clean verdict rather than to an exception."""
        findings = lint_delivered_checks(
            [_grep_check(name='cap', paths=['src/producer.py', 42, None])],
            files=['src/producer.py'],
            repo_root=authoring_repo,
        )

        assert [(f.check_name, f.code) for f in findings] == [('cap', 'vacuous_present')]

    def test_non_string_files_entries_are_dropped_not_crashed_on(self, tmp_path):
        repo = _init_git_repo(
            tmp_path / 'repo',
            {'src/owned.py': 'legacy_token = 1\n', 'src/other.py': 'legacy_token = 2\n'},
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'legacy-gone',
                    'kind': 'grep',
                    'pattern': 'legacy_token',
                    'expect': 'absent',
                    'paths': [],
                }
            ],
            # Deliberately ill-typed entries: the assertion below is that they
            # are dropped rather than crashed on, so the checker is told this
            # violation is the subject of the test, not an accident.
            files=['src/owned.py', 42, None],  # type: ignore[list-item]
            repo_root=repo,
        )

        assert [(f.check_name, f.severity) for f in findings] == [('legacy-gone', 'warn')]

    @pytest.mark.parametrize('pattern', [None, 42, '', ['x']])
    def test_a_missing_or_non_string_pattern_is_skipped(self, authoring_repo, pattern):
        """``kind='grep'`` without a usable pattern is a schema defect, not a
        polarity defect — and there is nothing to grep for."""
        findings = lint_delivered_checks(
            [_grep_check(pattern=pattern)], files=[], repo_root=authoring_repo
        )

        assert findings == []

    def test_checks_is_not_required_to_be_a_list(self, authoring_repo):
        """A tuple, a generator, anything iterable — the extractor's output
        is a list, but the lint is also called directly."""
        findings = lint_delivered_checks(
            (c for c in [_grep_check(name='gen')]),
            files=['src/producer.py'],
            repo_root=authoring_repo,
        )

        assert [f.check_name for f in findings] == ['gen']


# ---------------------------------------------------------------------------
# kind='path' — the second kind that carries an ``expect`` (task 4743)
# ---------------------------------------------------------------------------


def _path_check(**overrides: object) -> dict[str, object]:
    """A well-formed ``metadata.delivered_checks`` path entry.

    Against :data:`_SEED_TREE`, ``src/producer.py`` exists and
    ``src/new_module.py`` does not.
    """
    check: dict[str, object] = {
        'name': 'cap',
        'kind': 'path',
        'expect': 'present',
        'paths': ['src/producer.py'],
    }
    check.update(overrides)
    return check


class TestBuildPathArgv:
    """The argv ``orchestrator/tests/test_delivered_check_gate.py::TestRunnerPathKind``
    pins through ``run_delivered_check`` — mirrored here byte for byte."""

    def test_argv_is_one_ls_tree_probe_for_one_path(self):
        assert build_path_argv('a/one.py', project_root='/proj', ref='main') == [
            'git',
            '-C',
            '/proj',
            'ls-tree',
            '-r',
            '--full-tree',
            '--name-only',
            'main',
            '--',
            'a/one.py',
        ]

    def test_project_root_is_stringified(self):
        assert build_path_argv('a', project_root=Path('/proj'), ref='HEAD')[2] == '/proj'


class TestInterpretPathListing:
    """``ls-tree`` exits 0 whether or not the path exists, so existence is
    read from STDOUT and rc separates an answer from a git error."""

    @pytest.mark.parametrize('expect', ['present', 'absent', None])
    @pytest.mark.parametrize('stdout', ['', 'a/one.py\n'])
    def test_nonzero_rc_is_errored_whatever_stdout_says(self, expect, stdout):
        assert interpret_path_listing(128, stdout, expect) is CheckOutcome.ERRORED

    @pytest.mark.parametrize(
        ('expect', 'stdout', 'expected'),
        [
            ('present', 'a/one.py\n', CheckOutcome.PASS),
            ('present', '', CheckOutcome.FAIL),
            ('present', '   \n', CheckOutcome.FAIL),
            ('absent', '', CheckOutcome.PASS),
            ('absent', 'a/one.py\n', CheckOutcome.FAIL),
            ('absent', '   \n', CheckOutcome.PASS),
            (None, '', CheckOutcome.PASS),
        ],
    )
    def test_rc_zero_reads_existence_from_stdout(self, expect, stdout, expected):
        assert interpret_path_listing(0, stdout, expect) is expected


class TestEvaluatePathAtTree:
    """The shared primitive against a real tree: conjunctive over ``paths``,
    exactly as ``orchestrator.delivered_checks._run_path_check`` is."""

    @pytest.mark.parametrize(
        ('expect', 'paths', 'expected'),
        [
            ('present', ['src/producer.py'], CheckOutcome.PASS),
            ('present', ['src/new_module.py'], CheckOutcome.FAIL),
            ('absent', ['src/new_module.py'], CheckOutcome.PASS),
            ('absent', ['src/producer.py'], CheckOutcome.FAIL),
            ('present', ['src'], CheckOutcome.PASS),
        ],
    )
    def test_single_path_cells(self, authoring_repo, expect, paths, expected):
        outcome = evaluate_path_at_tree(
            paths, expect=expect, repo_root=authoring_repo, ref='HEAD'
        )

        assert outcome is expected

    def test_present_is_conjunctive(self, authoring_repo):
        outcome = evaluate_path_at_tree(
            ['src/producer.py', 'src/new_module.py'],
            expect='present',
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert outcome is CheckOutcome.FAIL

    def test_non_repo_root_is_errored_not_a_verdict(self, tmp_path):
        outcome = evaluate_path_at_tree(
            ['src/producer.py'], expect='absent', repo_root=tmp_path, ref='HEAD'
        )

        assert outcome is CheckOutcome.ERRORED

    def test_unresolvable_ref_is_errored(self, authoring_repo):
        outcome = evaluate_path_at_tree(
            ['src/new_module.py'],
            expect='absent',
            repo_root=authoring_repo,
            ref='no-such-ref',
        )

        assert outcome is CheckOutcome.ERRORED


class TestNonVacuityRulePathKind:
    """The same 2x2, read as existence: "matches the authoring tree" means
    every listed path already exists there."""

    def test_present_path_that_already_exists_is_rejected_as_vacuous(self, authoring_repo):
        findings = lint_delivered_checks(
            [_path_check(name='producer-file')],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert [(f.check_name, f.severity, f.code) for f in findings] == [
            ('producer-file', 'reject', 'vacuous_present')
        ]
        message = findings[0].message
        assert "kind='path'" in message
        assert 'src/producer.py' in message
        assert 'already' in message.lower()

    def test_present_path_the_task_will_create_is_healthy(self, authoring_repo):
        findings = lint_delivered_checks(
            [_path_check(paths=['src/new_module.py'])],
            files=['src/new_module.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_present_with_one_path_still_missing_is_healthy(self, authoring_repo):
        """Conjunctive: the check stays red until the missing path lands, so
        it still observes the task's change."""
        findings = lint_delivered_checks(
            [_path_check(paths=['src/producer.py', 'src/new_module.py'])],
            files=['src'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_absent_path_the_task_will_delete_is_healthy(self, authoring_repo):
        findings = lint_delivered_checks(
            [_path_check(expect='absent')],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert findings == []

    def test_absent_path_that_is_already_gone_is_rejected_as_vacuous(self, authoring_repo):
        findings = lint_delivered_checks(
            [_path_check(expect='absent', paths=['src/new_module.py'])],
            files=['src/new_module.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert [(f.severity, f.code) for f in findings] == [('reject', 'vacuous_absent')]
        assert "kind='path'" in findings[0].message

    def test_unevaluable_path_check_is_errored_not_rejected(self, tmp_path):
        findings = lint_delivered_checks(
            [_path_check()], files=[], repo_root=tmp_path, ref='HEAD'
        )

        assert [(f.severity, f.code) for f in findings] == [('errored', 'unevaluable')]

    @pytest.mark.parametrize('paths', [None, [], 'src/producer.py', [42, None]])
    def test_no_usable_paths_is_skipped_before_any_git_call(self, authoring_repo, paths):
        """A path check without a usable entry is a SCHEMA defect the metadata
        validator owns; there is nothing to probe."""
        with patch(
            'shared.delivered_check_polarity.subprocess.run',
            side_effect=AssertionError('lint must not shell out without a path'),
        ):
            findings = lint_delivered_checks(
                [_path_check(paths=paths)], files=[], repo_root=authoring_repo
            )

        assert findings == []

    @pytest.mark.parametrize('kind', [['path'], {'k': 'path'}, 7, None])
    def test_an_unhashable_or_non_string_kind_never_raises(self, authoring_repo, kind):
        findings = lint_delivered_checks(
            [_path_check(kind=kind)], files=[], repo_root=authoring_repo
        )

        assert findings == []


class TestEveryPolarityKindIsLinted:
    """Every mechanical kind that carries an ``expect`` is evaluated by the
    2x2 — a new one added to the schema must be wired here, not skipped.

    The kind vocabulary is read from
    :data:`shared.capability_manifest.MECHANICAL_CHECK_KINDS`, its single
    home; the only kind exempted is one the schema itself forbids an
    ``expect`` on, which the last test measures rather than asserts.
    """

    _VACUOUS_AT_SEED_TREE = {
        'grep': _grep_check(),
        'path': _path_check(),
    }
    _NO_EXPECT_KINDS = frozenset({'script'})

    def test_the_two_tables_partition_the_mechanical_vocabulary(self):
        linted = set(self._VACUOUS_AT_SEED_TREE)
        assert linted.isdisjoint(self._NO_EXPECT_KINDS)
        assert linted | self._NO_EXPECT_KINDS == set(MECHANICAL_CHECK_KINDS)

    @pytest.mark.parametrize('kind', sorted(_VACUOUS_AT_SEED_TREE))
    def test_a_vacuous_descriptor_of_every_polarity_kind_is_rejected(
        self, authoring_repo, kind
    ):
        findings = lint_delivered_checks(
            [self._VACUOUS_AT_SEED_TREE[kind]],
            files=['src/producer.py'],
            repo_root=authoring_repo,
            ref='HEAD',
        )

        assert [(f.severity, f.code) for f in findings] == [('reject', 'vacuous_present')]

    @pytest.mark.parametrize('kind', sorted(_NO_EXPECT_KINDS))
    def test_an_exempt_kind_really_cannot_carry_an_expect(self, kind):
        with pytest.raises(ValidationError, match='expect must not be set'):
            DeliveredCheckMeta.model_validate(
                {
                    'name': 'cap',
                    'kind': kind,
                    'script': 'scripts/x.sh',
                    'timeout_secs': 5,
                    'expect': 'present',
                }
            )
