"""Tests for the diagnostic refinements of a ``vacuous_present`` reject
(self-referential, comment-only, and their precedence) and for the rules
the 2x2 cannot see (MODE 3 ``filename_shaped`` in both shapes, MODE 2
``absent_overbroad``), all measured through ``lint_delivered_checks``
against a real throwaway git repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from polarity_fixture_repo import init_git_repo

from shared.delivered_check_polarity import lint_delivered_checks
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
        return init_git_repo(tmp_path / 'repo', _SELF_REF_TREE)

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
        repo = init_git_repo(tmp_path / 'repo', tree)

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
        repo = init_git_repo(
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
        repo = init_git_repo(
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
# TestMarkdownCommentRule (task 6037)
# ---------------------------------------------------------------------------


class TestMarkdownCommentRule:
    """In markdown only an HTML comment is a comment. A heading or a bullet
    can BE the capability: task 6036 adjudicated task 3199's
    ``'^## Decision table'`` check as correct."""

    def _lint_one_file(self, tmp_path: Path, rel_path: str, body: str, pattern: str):
        repo = init_git_repo(tmp_path / 'repo', {rel_path: body})
        return lint_delivered_checks(
            [{'name': 'doc-cap', 'kind': 'grep', 'pattern': pattern, 'expect': 'present',
              'paths': [rel_path]}],
            files=[rel_path],
            repo_root=repo,
            ref='HEAD',
        )

    def test_a_markdown_heading_is_content_not_a_comment(self, tmp_path):
        findings = self._lint_one_file(
            tmp_path,
            'docs/report.md',
            '# Report\n\n## Decision table\n\n| a | b |\n',
            '^## Decision table',
        )

        assert [f.code for f in findings] == ['vacuous_present']

    def test_a_markdown_bullet_is_content_not_a_comment(self, tmp_path):
        findings = self._lint_one_file(
            tmp_path,
            'docs/notes.md',
            '* archive_task_transcripts lands here\n',
            'archive_task_transcripts',
        )

        assert [f.code for f in findings] == ['vacuous_present']

    def test_an_html_comment_in_markdown_is_a_comment(self, tmp_path):
        findings = self._lint_one_file(
            tmp_path,
            'docs/notes.md',
            '<!-- archive_task_transcripts is handled elsewhere -->\n',
            'archive_task_transcripts',
        )

        assert [f.code for f in findings] == ['vacuous_present_comment_only']


# ---------------------------------------------------------------------------
# TestRefinementPrecedence (task 3500 — step-7 RED / step-8 GREEN)
# ---------------------------------------------------------------------------


class TestRefinementPrecedence:
    """AT MOST ONE finding per check, with a deterministic winner.

    Both refinements can hold at once (a manifest ``pattern:`` line reached
    through an HTML comment in the sibling PRD is self-referential AND
    comment-shaped). Emitting two findings would double-count one defect in
    the reject payload, so the order is pinned: self-referential is the more
    specific diagnosis and wins.
    """

    def test_self_reference_outranks_comment_only(self, tmp_path):
        repo = init_git_repo(
            tmp_path / 'repo',
            {
                # Every match line here is BOTH inside the descriptor family
                # and comment-shaped (an HTML comment, markdown's only one).
                'plans/foo-prd.md': (
                    '# Foo PRD\n\n<!-- foo_capability_token lands here -->\n'
                ),
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
        return init_git_repo(
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
        the detective rule must name the prescriptive fix, not only the grep
        workaround."""
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
        assert 'creates' in message

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

    def _lint_strand_scope(self, repo: Path, pattern: str):
        return lint_delivered_checks(
            [{'name': 'gating-strand', 'kind': 'grep', 'pattern': pattern,
              'expect': 'present', 'paths': ['orchestrator/tests']}],
            files=['orchestrator/tests'],
            repo_root=repo,
            ref='HEAD',
        )

    @pytest.mark.filterwarnings('error::FutureWarning')
    def test_a_posix_class_pattern_matching_a_tracked_filename_is_rejected(
        self, strand_repo
    ):
        findings = self._lint_strand_scope(
            strand_repo, 'test_workflow_merge_gating_[[:alpha:]]+'
        )

        assert [(f.severity, f.code) for f in findings] == [('reject', 'filename_shaped')]

    @pytest.mark.filterwarnings('error::FutureWarning')
    def test_a_posix_class_the_filename_does_not_satisfy_is_healthy(self, strand_repo):
        """The CONTROL: the translation is semantic, not permissive."""
        findings = self._lint_strand_scope(
            strand_repo, 'test_workflow_merge_gating_[[:digit:]]+'
        )

        assert findings == []

    def test_an_escape_git_reads_otherwise_names_no_filename(self, tmp_path):
        r"""git reads ``\d`` as a literal ``d`` and Python as a digit, so a
        digit in a tracked filename is evidence of nothing."""
        repo = init_git_repo(tmp_path / 'repo', {'src/test_7.py': 'def run():\n    pass\n'})

        findings = lint_delivered_checks(
            [{'name': 'digit-escape', 'kind': 'grep', 'pattern': r'test_\d',
              'expect': 'present', 'paths': ['src']}],
            files=['src'],
            repo_root=repo,
            ref='HEAD',
        )

        assert findings == []

    def test_the_filename_rule_lists_only_the_corpus_the_grep_searches(self, tmp_path):
        """From ``git -C sub``, the grep reads only ``sub/`` and prints names
        relative to it, so the filename rule must list exactly that."""
        repo = init_git_repo(
            tmp_path / 'repo',
            {
                'other/test_widget_strand.py': 'def run():\n    pass\n',
                'sub/a.py': 'def f():\n    return 0\n',
                'sub/test_gadget_strand.py': 'def run():\n    pass\n',
            },
        )

        def lint(pattern: str):
            return lint_delivered_checks(
                [{'name': 'strand', 'kind': 'grep', 'pattern': pattern,
                  'expect': 'present', 'paths': []}],
                files=[],
                repo_root=repo / 'sub',
                ref='HEAD',
            )

        assert lint('test_widget_strand') == []
        gadget = lint('test_gadget_strand')
        assert [(f.severity, f.code) for f in gadget] == [('reject', 'filename_shaped')]
        assert gadget[0].detail == ('test_gadget_strand.py',)


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
        return init_git_repo(
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
# MODE 3 in its real shape: the pattern names a file the task DECLARES
# ---------------------------------------------------------------------------

#: Task 3536's own descriptor and metadata.files, verbatim. The check was
#: authored at 2486d95548; the module it names was created later, by 3536
#: itself (cf60e3b90e), so no tracked path could match at authoring.
_T3536_FILES = [
    'orchestrator/src/orchestrator/workflow.py',
    'orchestrator/tests/test_workflow_merge_gating_strand.py',
    'orchestrator/tests/test_repend_state_machine.py',
    'orchestrator/tests/test_workflow_e2e.py',
]
_T3536_CHECK = {
    'name': 'no-steward-less-escalated-exit-at-merge-entry',
    'kind': 'grep',
    'pattern': 'test_workflow_merge_gating_strand',
    'expect': 'present',
    'paths': ['orchestrator/tests/'],
}


class TestDeclaredFilenameShaped:
    """The rule, exactly: a ``kind='grep'`` ``expect='present'`` check that
    is healthy under the 2x2 is rejected as ``filename_shaped`` when its
    pattern FULLY matches the basename or the stem of a file in the task's
    ``metadata.files`` that lies inside the check's ``paths`` scope."""

    @pytest.fixture
    def pre_3536_repo(self, tmp_path: Path) -> Path:
        """The authoring tree as 3536's author saw it: the test directory
        exists, the strand module does not."""
        return init_git_repo(
            tmp_path / 'repo',
            {
                'orchestrator/src/orchestrator/workflow.py': 'def run():\n    pass\n',
                'orchestrator/tests/test_workflow_e2e.py': 'def test_e2e():\n    pass\n',
            },
        )

    def test_the_3536_descriptor_is_rejected_at_its_authoring_tree(self, pre_3536_repo):
        findings = lint_delivered_checks(
            [_T3536_CHECK], files=_T3536_FILES, repo_root=pre_3536_repo
        )

        assert [(f.check_name, f.severity, f.code) for f in findings] == [
            ('no-steward-less-escalated-exit-at-merge-entry', 'reject', 'filename_shaped')
        ]
        assert findings[0].detail == ('orchestrator/tests/test_workflow_merge_gating_strand.py',)
        message = findings[0].message
        assert "kind='path'" in message
        assert "'orchestrator/tests/test_workflow_merge_gating_strand.py'" in message
        assert 'cannot state that a file exists' in message
        assert 'coincidence' in message

    def test_the_3536_descriptor_replayed_against_its_real_authoring_commit(self):
        repo_root = Path(__file__).resolve().parents[2]
        probe = subprocess.run(
            ['git', '-C', str(repo_root), 'cat-file', '-e', '2486d95548^{commit}'],
            capture_output=True,
        )
        if probe.returncode != 0:
            pytest.skip('task 3536 authoring commit is not in this clone')

        findings = lint_delivered_checks(
            [_T3536_CHECK], files=_T3536_FILES, repo_root=repo_root, ref='2486d95548'
        )

        assert [(f.severity, f.code) for f in findings] == [('reject', 'filename_shaped')]

    def test_the_basename_with_its_suffix_is_the_same_defect(self, pre_3536_repo):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': r'test_workflow_merge_gating_strand\.py'}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert [f.code for f in findings] == ['filename_shaped']

    @pytest.mark.parametrize(
        'pattern',
        [
            pytest.param('TestNoStrandExitProperty', id='a-symbol-inside-the-file'),
            pytest.param('merge_gating', id='a-substring-of-the-stem'),
            pytest.param('test_workflow_merge_gating_strand_v2', id='a-longer-name'),
        ],
    )
    def test_a_pattern_that_is_not_the_whole_name_is_healthy(self, pre_3536_repo, pattern):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': pattern}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert findings == []

    def test_an_alternation_with_a_symbol_arm_is_healthy(self, tmp_path):
        """Task 5324's exact descriptor: one arm is the module's stem, the
        other a class the producer defines, so the check is a sound 0->N gate."""
        repo = init_git_repo(
            tmp_path / 'repo', {'escalation/src/escalation/server.py': 'import os\n'}
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'prefix-resolver-http-client-exists',
                    'kind': 'grep',
                    'pattern': 'prefix_resolver_client|class PrefixResolverClient',
                    'expect': 'present',
                    'paths': ['escalation/src/escalation/'],
                }
            ],
            files=[
                'escalation/src/escalation/server.py',
                'escalation/src/escalation/prefix_resolver_client.py',
                'fused-memory/src/fused_memory/reconciliation/harness.py',
                'escalation/tests/test_uuid_prefix_guard_registration.py',
            ],
            repo_root=repo,
        )

        assert findings == []

    @pytest.mark.parametrize(
        'pattern',
        [
            pytest.param(r'test_workflow_merge_gating_strand|test_workflow_merge_gating_strand\.py',
                         id='every-arm-names-the-file'),
            pytest.param('(test_workflow_merge_gating_strand|test_workflow_merge_gating_strand.py)',
                         id='a-group-wrapping-the-whole-pattern'),
        ],
    )
    def test_an_alternation_whose_every_arm_names_the_file_is_rejected(
        self, pre_3536_repo, pattern
    ):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': pattern}], files=_T3536_FILES, repo_root=pre_3536_repo
        )

        assert [f.code for f in findings] == ['filename_shaped']

    def test_a_symbol_arm_inside_a_wrapping_group_keeps_the_check_healthy(
        self, pre_3536_repo
    ):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': '(test_workflow_merge_gating_strand|TestNoStrand)'}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert findings == []

    def test_a_bar_inside_a_bracket_is_not_an_alternation(self, pre_3536_repo):
        """One arm, which still names the file: splitting inside the bracket
        would have produced two unparseable halves and hidden the defect."""
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': 'test_workflow_merge_gating_stran[d|]'}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert [f.code for f in findings] == ['filename_shaped']

    def test_a_declared_file_outside_the_checks_scope_is_healthy(self, tmp_path):
        """A wiring check: the grep reads another file for an import of the
        declared module, which is content, not the module's existence."""
        repo = init_git_repo(
            tmp_path / 'repo',
            {'fused-memory/src/fused_memory/server/tools.py': 'import os\n'},
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'guard-wired',
                    'kind': 'grep',
                    'pattern': 'recurring_gate_guard',
                    'expect': 'present',
                    'paths': ['fused-memory/src/fused_memory/server/tools.py'],
                }
            ],
            files=['fused-memory/src/fused_memory/middleware/recurring_gate_guard.py'],
            repo_root=repo,
        )

        assert findings == []

    def test_a_multi_path_wiring_check_is_healthy(self, tmp_path):
        """Task 3659's exact descriptor and files: the declared module sits
        under ONE of the two paths, and the check went green through a real
        import in the OTHER (agents/briefing.py) — a wiring check."""
        repo = init_git_repo(
            tmp_path / 'repo',
            {
                'orchestrator/src/orchestrator/agents/briefing.py': 'import os\n',
                'shared/src/shared/__init__.py': '',
            },
        )

        findings = lint_delivered_checks(
            [
                {
                    'name': 'templates-single-home-in-shared',
                    'kind': 'grep',
                    'pattern': 'briefing_queries',
                    'expect': 'present',
                    'paths': ['orchestrator/src/orchestrator/agents/', 'shared/src/shared/'],
                }
            ],
            files=[
                'shared/src/shared/briefing_queries.py',
                'shared/tests/test_briefing_queries.py',
                'shared/tests/silent_fallthrough_allowlist.py',
                'orchestrator/src/orchestrator/agents/briefing.py',
                'orchestrator/src/orchestrator/mcp_lifecycle.py',
                'orchestrator/src/orchestrator/workflow.py',
                'orchestrator/tests/_workflow_helpers.py',
                'orchestrator/tests/test_briefing.py',
                'orchestrator/tests/test_briefing_project_scope.py',
                'orchestrator/tests/test_mcp_retry.py',
                'orchestrator/tests/test_reviewer_verdict_routing.py',
                'orchestrator/tests/test_merger_disposition_verdict.py',
                'scripts/legibility/digest.py',
                'scripts/tests/test_legibility_digest.py',
            ],
            repo_root=repo,
        )

        assert findings == []

    def test_every_path_containing_the_declared_file_is_still_mode_3(self, pre_3536_repo):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'paths': ['orchestrator/', 'orchestrator/tests/']}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert [f.code for f in findings] == ['filename_shaped']

    def test_an_unscoped_check_covers_every_declared_file(self, pre_3536_repo):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'paths': []}], files=_T3536_FILES, repo_root=pre_3536_repo
        )

        assert [f.code for f in findings] == ['filename_shaped']

    def test_expect_absent_is_never_filename_shaped(self, pre_3536_repo):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'expect': 'absent'}], files=_T3536_FILES, repo_root=pre_3536_repo
        )

        assert [f.code for f in findings] == ['vacuous_absent']

    @pytest.mark.filterwarnings('error::FutureWarning')
    def test_a_posix_class_pattern_naming_the_declared_file_is_rejected(
        self, pre_3536_repo
    ):
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': 'test_workflow_merge_gating_[[:alpha:]]+'}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert [f.code for f in findings] == ['filename_shaped']
        assert findings[0].detail == ('orchestrator/tests/test_workflow_merge_gating_strand.py',)

    @pytest.mark.filterwarnings('error::FutureWarning')
    def test_a_posix_class_the_declared_name_does_not_satisfy_is_healthy(
        self, pre_3536_repo
    ):
        """The CONTROL: the translation is semantic, not permissive."""
        findings = lint_delivered_checks(
            [{**_T3536_CHECK, 'pattern': 'test_workflow_merge_gating_[[:digit:]]+'}],
            files=_T3536_FILES,
            repo_root=pre_3536_repo,
        )

        assert findings == []


# ---------------------------------------------------------------------------
# Match sites survive unusual paths (task 6037)
# ---------------------------------------------------------------------------


class TestMatchSitesSurviveUnusualPaths:
    """A match site is read from ``git grep -n -z`` records, so a ':' in a
    path cannot split the record and a non-ASCII path arrives unquoted,
    comparable with the descriptor family and the declared files."""

    def test_a_colon_in_the_path_keeps_the_comment_only_refinement(self, tmp_path):
        repo = init_git_repo(
            tmp_path / 'repo', {'src/a:b.py': '# colon_token is described here\n'}
        )

        findings = lint_delivered_checks(
            [{'name': 'c', 'kind': 'grep', 'pattern': 'colon_token', 'expect': 'present',
              'paths': []}],
            files=[],
            repo_root=repo,
            ref='HEAD',
        )

        assert [f.code for f in findings] == ['vacuous_present_comment_only']
        assert any(site.startswith('src/a:b.py:1:') for site in findings[0].detail)

    def test_a_colon_in_the_path_is_named_by_absent_overbroad(self, tmp_path):
        repo = init_git_repo(
            tmp_path / 'repo',
            {'src/owned.py': 'legacy_token = 1\n', 'src/we:ird.py': 'legacy_token = 2\n'},
        )

        findings = lint_delivered_checks(
            [{'name': 'legacy-gone', 'kind': 'grep', 'pattern': 'legacy_token',
              'expect': 'absent', 'paths': []}],
            files=['src/owned.py'],
            repo_root=repo,
            ref='HEAD',
        )

        assert [(f.severity, f.code) for f in findings] == [('warn', 'absent_overbroad')]
        assert 'src/we:ird.py' in '\n'.join(findings[0].detail)

    def test_an_undecodable_name_degrades_only_its_own_record(self, tmp_path):
        undecodable = 'src/' + os.fsdecode(b'\xff') + '.py'
        repo = init_git_repo(
            tmp_path / 'repo',
            {
                'src/owned.py': 'legacy_token = 1\n',
                'src/elsewhere.py': 'legacy_token = 2\n',
                undecodable: 'legacy_token = 3\n',
            },
        )

        findings = lint_delivered_checks(
            [{'name': 'legacy-gone', 'kind': 'grep', 'pattern': 'legacy_token',
              'expect': 'absent', 'paths': []}],
            files=['src/owned.py'],
            repo_root=repo,
            ref='HEAD',
        )

        assert [(f.severity, f.code) for f in findings] == [('warn', 'absent_overbroad')]
        assert 'src/elsewhere.py' in '\n'.join(findings[0].detail)

    def test_a_non_ascii_sidecar_path_is_still_self_referential(self, tmp_path):
        manifest = 'plans/café-prd.capability-manifest.yaml'
        repo = init_git_repo(
            tmp_path / 'repo',
            {
                manifest: (
                    'capabilities:\n'
                    '  - name: c\n'
                    '    delivered_check:\n'
                    '      pattern: cafe_token\n'
                ),
                'src/x.py': 'def f():\n    return 0\n',
            },
        )

        findings = lint_delivered_checks(
            [{'name': 'c', 'kind': 'grep', 'pattern': 'cafe_token', 'expect': 'present',
              'paths': [], 'manifest_path': manifest}],
            files=[],
            repo_root=repo,
            ref='HEAD',
        )

        assert [f.code for f in findings] == ['vacuous_present_self_referential']
