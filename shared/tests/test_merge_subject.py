"""Tests for shared.merge_subject: the merge-commit subject's writer and its readers.

Contract: plans/review-all-s14-shared-leaves-prd.md (decision 5, invariant I1).
The format table is ported from orchestrator/tests/test_git_ops.py::TestMergeSubject
and the refusals from ::TestMergeMarkerIsExactSubject (task 5765).
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
from pathlib import Path

import pytest

from shared.merge_subject import (
    MergeSubject,
    format_merge_subject,
    merge_subject_pattern,
    parse_merge_subject,
    rendered_prefix,
)

_BRANCHES = ['feat/12', 'task/v1.0', 'rel.v1+(x)/5', 'a/b/c', 'ünï/7', '\x00BRANCH\x00']
_MAINS = ['trunk', 'main', 'release/2.0']

_NOT_A_MERGE_SUBJECT = [
    'Revert "Merge feat/12 into trunk"',
    'Merge feat/12 into trunk (retry)',
    'Merge feat/12 into trunk ',
    ' Merge feat/12 into trunk',
    "Merge branch 'feat/12' into trunk",
    'Merge feat/12 into',
    'Merge  into trunk',
    'Merge a\tb into trunk',
    'fix: x\n\nMerge feat/12 into trunk',
    'Merge feat/12 into trunk\n',
    'Merge feat/12 into trunk\n\nbody',
    'impl(12): x',
    '',
]

_NOT_A_REF_TOKEN = [
    ('branch', '', 'main'),
    ('branch', 'a b', 'main'),
    ('main_branch', 'task/1', ''),
    ('main_branch', 'task/1', 'ma\tin'),
]


class TestFormatMergeSubject:
    @pytest.mark.parametrize(
        ('branch', 'main_branch', 'expected'),
        [
            ('task/1', 'main', 'Merge task/1 into main'),
            ('task/123', 'main', 'Merge task/123 into main'),
            ('task/v1.0', 'develop', 'Merge task/v1.0 into develop'),
            ('feat/12', 'trunk', 'Merge feat/12 into trunk'),
        ],
    )
    def test_reproduces_the_git_ops_subject(self, branch: str, main_branch: str, expected: str) -> None:
        assert format_merge_subject(branch, main_branch) == expected


class TestRoundTrip:
    @pytest.mark.parametrize('main_branch', _MAINS)
    @pytest.mark.parametrize('branch', _BRANCHES)
    def test_parse_reads_back_what_format_writes(self, branch: str, main_branch: str) -> None:
        parsed = parse_merge_subject(format_merge_subject(branch, main_branch))
        assert parsed == MergeSubject(branch, main_branch)

    @pytest.mark.parametrize('field', ['branch', 'main_branch'])
    def test_parsed_subject_is_frozen(self, field: str) -> None:
        parsed = parse_merge_subject(format_merge_subject('feat/12', 'trunk'))
        assert parsed is not None
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(parsed, field, 'other')


class TestParseRefusals:
    @pytest.mark.parametrize('subject', _NOT_A_MERGE_SUBJECT)
    def test_returns_none(self, subject: str) -> None:
        assert parse_merge_subject(subject) is None


class TestRefTokenInvariant:
    @pytest.mark.parametrize(('field', 'branch', 'main_branch'), _NOT_A_REF_TOKEN)
    def test_construction_refuses(self, field: str, branch: str, main_branch: str) -> None:
        offending = branch if field == 'branch' else main_branch
        with pytest.raises(ValueError) as refused:
            MergeSubject(branch, main_branch)
        message = str(refused.value)
        assert re.search(rf'\b{field}\b', message), message
        assert repr(offending) in message

    @pytest.mark.parametrize(('field', 'branch', 'main_branch'), _NOT_A_REF_TOKEN)
    def test_format_refuses_what_parse_could_never_read_back(
        self, field: str, branch: str, main_branch: str
    ) -> None:
        with pytest.raises(ValueError) as constructed:
            MergeSubject(branch, main_branch)
        with pytest.raises(ValueError) as formatted:
            format_merge_subject(branch, main_branch)
        assert str(formatted.value) == str(constructed.value)


# Every POSIX ERE special character, so an unescaped renderer selects the near
# misses below and a renderer escaping too much trips grep's stray-backslash
# warning.
_WILD_PREFIX = 'a.b*c+d?e(f)g{1}h|i^j$k[l]m\\n/'
_WILD = format_merge_subject(_WILD_PREFIX + '5', 'trunk')

_LANDED_INTO_TRUNK = {
    'Merge feat/12 into trunk',
    'Merge feat/123 into trunk',
    'Merge feat/1 into trunk',
    'Merge feat/v1.0 into trunk',
}

_CORPUS = (
    'seed',
    *sorted(_LANDED_INTO_TRUNK),
    'Merge feat/12 into main',
    'Merge task/12 into trunk',
    'Merge feat/12 into trunk (retry)',
    'Revert "Merge feat/12 into trunk"',
    'Merge feat/12: legacy subject',
    'Merge feat/12:no-space',
    "Merge branch 'feat/12' into trunk",
    'impl(12): x',
    _WILD,
    'Merge aXb*c+d?e(f)g{1}h|i^j$k[l]m\\n/5 into trunk',
    'Merge abbbc+d?e(f)g{1}h|i^j$k[l]m\\n/5 into trunk',
)


def _is_subject_line(text: str) -> bool:
    return not any(ch.isspace() and ch != ' ' for ch in text)


# merge_subject_pattern's Python domain is one git %s subject line; these are the
# inputs outside it where the pattern and parse_merge_subject part ways.
_OUTSIDE_THE_PYTHON_DOMAIN = [
    'Merge feat/a\tb into trunk',
    'Merge feat/12 into trunk\n',
    'Merge feat/12 into trunk\n\nbody',
]

_SUBJECT_LINES = [
    subject
    for subject in (*_CORPUS, *_NOT_A_MERGE_SUBJECT, 'Merge feat/ into trunk')
    if _is_subject_line(subject)
]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def subject_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'subjects'
    subprocess.run(['git', 'init', '-q', '-b', 'trunk', str(repo)], check=True, capture_output=True)
    _git(repo, 'config', 'user.email', 'merge-subject-test@example.com')
    _git(repo, 'config', 'user.name', 'Merge Subject Test')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    for subject in _CORPUS:
        _git(repo, 'commit', '-q', '--allow-empty', '--no-verify', '--cleanup=verbatim', '-m', subject)
    return repo


def _git_selected(repo: Path, pattern: str) -> set[str]:
    return set(_git(repo, 'log', '-E', f'--grep={pattern}', '--format=%s', 'trunk').splitlines())


def _python_selected(pattern: str) -> set[str]:
    return {subject for subject in _CORPUS if re.search(pattern, subject)}


class TestMergeSubjectPatternAcrossEngines:
    @pytest.mark.parametrize(
        ('kwargs', 'expected'),
        [
            (dict(task_id='12', main_branch='trunk'), {'Merge feat/12 into trunk'}),
            (dict(task_id='12'), {'Merge feat/12 into trunk', 'Merge feat/12 into main'}),
            (dict(main_branch='trunk'), _LANDED_INTO_TRUNK),
            (dict(), _LANDED_INTO_TRUNK | {'Merge feat/12 into main'}),
            (
                dict(task_id='12', main_branch='trunk', legacy_colon_form=True),
                {'Merge feat/12 into trunk', 'Merge feat/12: legacy subject'},
            ),
            (
                dict(legacy_colon_form=True),
                _LANDED_INTO_TRUNK | {'Merge feat/12 into main', 'Merge feat/12: legacy subject'},
            ),
            (dict(branch_prefix=_WILD_PREFIX, task_id='5', main_branch='trunk'), {_WILD}),
        ],
        ids=[
            'task-and-main',
            'task-any-main',
            'any-task-main',
            'any-task-any-main',
            'task-and-main-or-legacy',
            'any-or-legacy',
            'every-ere-special-in-prefix',
        ],
    )
    def test_git_and_python_select_the_same_subjects(
        self, subject_repo: Path, kwargs: dict, expected: set[str]
    ) -> None:
        pattern = merge_subject_pattern(**{'branch_prefix': 'feat/', **kwargs})
        by_git = _git_selected(subject_repo, pattern)
        by_python = _python_selected(pattern)
        assert by_git == by_python == expected, (
            f'pattern={pattern!r}\n  git:      {sorted(by_git)}\n'
            f'  python:   {sorted(by_python)}\n  expected: {sorted(expected)}'
        )

    @pytest.mark.parametrize('subject', _SUBJECT_LINES)
    def test_on_a_subject_line_pattern_accepts_exactly_what_parse_accepts(self, subject: str) -> None:
        parsed = parse_merge_subject(subject)
        parsed_as_feat = (
            parsed is not None
            and parsed.branch.startswith('feat/')
            and len(parsed.branch) > len('feat/')
        )
        matched = re.search(merge_subject_pattern(branch_prefix='feat/'), subject) is not None
        assert matched == parsed_as_feat

    @pytest.mark.parametrize('subject', _OUTSIDE_THE_PYTHON_DOMAIN)
    def test_tab_or_newline_is_selected_by_the_pattern_but_refused_by_parse(self, subject: str) -> None:
        assert re.search(merge_subject_pattern(branch_prefix='feat/'), subject)
        assert parse_merge_subject(subject) is None


def _bash_line_arm_verdict(head: str, task_id: str, into: str, subject: str) -> bool:
    proc = subprocess.run(
        ['bash', '-c', 'grep -qE -- "^$1$2$3" <<<"$4"', '_', head, task_id, into, subject],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode in (0, 1), f'grep exited {proc.returncode}: {proc.stderr}'
    assert proc.stderr == '', proc.stderr
    return proc.returncode == 0


class TestRenderedPrefixInBash:
    @pytest.mark.parametrize(
        ('branch_prefix', 'task_id', 'expected'),
        [
            (
                'feat/',
                '12',
                {'Merge feat/12 into trunk', 'Merge feat/12 into main', 'Merge feat/12 into trunk (retry)'},
            ),
            (_WILD_PREFIX, '5', {_WILD}),
        ],
        ids=['feat', 'every-ere-special-in-prefix'],
    )
    def test_bash_grep_line_arm_agrees_with_python(
        self, branch_prefix: str, task_id: str, expected: set[str]
    ) -> None:
        head, into = rendered_prefix(branch_prefix)
        selected = set()
        for subject in _CORPUS:
            verdict = _bash_line_arm_verdict(head, task_id, into, subject)
            assert verdict == (re.search('^' + head + task_id + into, subject) is not None), subject
            if verdict:
                selected.add(subject)
        assert selected == expected


class TestPatternArgumentsShareTheRefTokenInvariant:
    @pytest.mark.parametrize(
        ('field', 'kwargs'),
        [
            ('task_id', dict(branch_prefix='feat/', task_id='')),
            ('main_branch', dict(branch_prefix='feat/', main_branch='a b')),
            ('branch_prefix', dict(branch_prefix='fe at/')),
        ],
    )
    def test_refuses_and_names_the_argument(self, field: str, kwargs: dict) -> None:
        with pytest.raises(ValueError, match=rf'\b{field}\b'):
            merge_subject_pattern(**kwargs)

    def test_rendered_prefix_refuses_whitespace(self) -> None:
        with pytest.raises(ValueError, match=r'\bbranch_prefix\b'):
            rendered_prefix('fe at/')

    def test_empty_branch_prefix_is_accepted(self) -> None:
        assert rendered_prefix('') == ('Merge ', ' into ')
        assert re.search(merge_subject_pattern(branch_prefix=''), 'Merge feat/12 into trunk')
