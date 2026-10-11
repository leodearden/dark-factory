"""Tests for shared.merge_subject: the merge-commit subject's writer and its readers.

Contract: plans/review-all-s14-shared-leaves-prd.md (decision 5, invariant I1).
The format table is ported from orchestrator/tests/test_git_ops.py::TestMergeSubject
and the refusals from ::TestMergeMarkerIsExactSubject (task 5765).
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from shared.merge_subject import MergeSubject, format_merge_subject, parse_merge_subject

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
