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
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shared.delivered_check_polarity import (
    CheckOutcome,
    build_grep_argv,
    interpret_grep_rc,
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
