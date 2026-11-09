"""The reviewer issue contract and its intake check (task 5494).

check_issues() is the one definition of an on-contract issue: the verdict-tools
intake normalises against it and the review gate's aggregation flags against it.
"""

from __future__ import annotations

import copy

import pytest

from orchestrator.review_issue_contract import (
    Deviation,
    IssueDeviation,
    IssueField,
    IssueIntake,
    IssueSeverity,
    check_issues,
)

_JOURNAL_FIELD_CAP = 200


def _issue(**overrides: object) -> dict[str, object]:
    issue: dict[str, object] = {
        'severity': 'blocking',
        'location': 'pkg/mod.py:3',
        'category': 'correctness',
        'description': 'the thing is broken',
        'suggested_fix': 'fix the thing',
    }
    issue.update(overrides)
    return issue


def _without(field: IssueField, **overrides: object) -> dict[str, object]:
    issue = _issue(**overrides)
    del issue[field]
    return issue


def _only_deviation(intake: IssueIntake) -> IssueDeviation:
    assert len(intake.deviations) == 1, intake.deviations
    return intake.deviations[0]


class TestOnContractIssue:
    def test_comes_back_equal_with_no_deviations(self):
        issue = _issue(title='a title', file='pkg/mod.py', line=3)

        intake = check_issues([issue])

        assert intake.issues == (issue,)
        assert intake.deviations == ()
        assert intake.unusable == ()
        assert intake.normalised == ()

    def test_every_contract_field_is_named_by_the_vocabulary(self):
        assert {str(field) for field in IssueField} == set(_issue())

    def test_severity_vocabulary_is_blocking_and_suggestion(self):
        assert [str(severity) for severity in IssueSeverity] == ['blocking', 'suggestion']


class TestSeverityVocabulary:
    def test_off_vocabulary_severity_is_recorded_as_a_suggestion(self):
        intake = check_issues([_issue(severity='high')])

        assert intake.issues[0]['severity'] == IssueSeverity.SUGGESTION
        assert _only_deviation(intake) == IssueDeviation(
            index=0,
            field=IssueField.SEVERITY,
            deviation=Deviation.OFF_VOCABULARY,
            received='high',
            recorded_as='suggestion',
        )
        assert intake.unusable == ()
        assert intake.normalised == intake.deviations

    def test_case_and_whitespace_variant_is_recorded_as_its_member_and_flagged(self):
        intake = check_issues([_issue(severity=' Blocking ')])

        assert intake.issues[0]['severity'] == 'blocking'
        deviation = _only_deviation(intake)
        assert deviation.deviation is Deviation.OFF_VOCABULARY
        assert deviation.recorded_as == 'blocking'
        assert not deviation.unusable


class TestLocationFallback:
    @pytest.mark.parametrize(
        ('extras', 'recorded'),
        [
            ({'file': 'pkg/mod.py', 'line': 12}, 'pkg/mod.py:12'),
            ({'file': 'pkg/mod.py', 'line': '12'}, 'pkg/mod.py:12'),
            ({'file': 'pkg/mod.py'}, 'pkg/mod.py'),
            ({'file': 'pkg/mod.py', 'line': True}, 'pkg/mod.py'),
            ({}, ''),
        ],
    )
    def test_missing_location_is_derived_from_file_and_line(self, extras, recorded):
        issue = _without(IssueField.LOCATION, **extras)

        intake = check_issues([issue])

        assert intake.issues[0]['location'] == recorded
        assert intake.issues[0] == {**issue, 'location': recorded}
        assert _only_deviation(intake) == IssueDeviation(
            index=0,
            field=IssueField.LOCATION,
            deviation=Deviation.MISSING,
            received=None,
            recorded_as=recorded,
        )


class TestOptionalFields:
    @pytest.mark.parametrize('field', [IssueField.CATEGORY, IssueField.SUGGESTED_FIX])
    @pytest.mark.parametrize('absent_form', ['absent', None, '', '   '])
    def test_missing_optional_field_is_recorded_blank_and_flagged(self, field, absent_form):
        issue = _without(field) if absent_form == 'absent' else _issue(**{field: absent_form})

        intake = check_issues([issue])

        assert intake.issues[0][field] == ''
        deviation = _only_deviation(intake)
        assert (deviation.field, deviation.deviation, deviation.recorded_as) == (
            field, Deviation.MISSING, '',
        )

    def test_present_non_string_optional_field_is_recorded_as_its_str(self):
        intake = check_issues([_issue(location=42)])

        assert intake.issues[0]['location'] == '42'
        deviation = _only_deviation(intake)
        assert (deviation.field, deviation.deviation, deviation.received, deviation.recorded_as) == (
            IssueField.LOCATION, Deviation.NOT_A_STRING, 42, '42',
        )


class TestUnusableIssues:
    @pytest.mark.parametrize(
        ('issue', 'field', 'deviation'),
        [
            (_without(IssueField.SEVERITY), IssueField.SEVERITY, Deviation.MISSING),
            (_issue(severity=None), IssueField.SEVERITY, Deviation.MISSING),
            (_issue(severity='  '), IssueField.SEVERITY, Deviation.MISSING),
            (_issue(severity=3), IssueField.SEVERITY, Deviation.NOT_A_STRING),
            (_without(IssueField.DESCRIPTION), IssueField.DESCRIPTION, Deviation.MISSING),
            (_issue(description=None), IssueField.DESCRIPTION, Deviation.MISSING),
            (_issue(description=''), IssueField.DESCRIPTION, Deviation.MISSING),
            ('a prose finding', None, Deviation.NOT_AN_OBJECT),
        ],
    )
    def test_unusable_issue_is_reported_and_omitted(self, issue, field, deviation):
        intake = check_issues([issue])

        assert intake.issues == ()
        assert len(intake.unusable) == 1
        found = intake.unusable[0]
        assert (found.index, found.field, found.deviation, found.recorded_as) == (
            0, field, deviation, None,
        )
        assert found.unusable

    def test_indices_are_positional_in_the_input(self):
        good_first = _issue(description='first')
        good_last = _issue(description='last')

        intake = check_issues([good_first, _without(IssueField.DESCRIPTION), good_last])

        assert intake.issues == (good_first, good_last)
        assert [(d.index, d.field) for d in intake.unusable] == [
            (1, IssueField.DESCRIPTION),
        ]


def test_the_input_issues_are_not_mutated():
    issues = [
        _issue(severity='HIGH'),
        _without(IssueField.LOCATION, file='pkg/mod.py', line=1),
        _issue(category=None),
        _without(IssueField.DESCRIPTION),
    ]
    before = copy.deepcopy(issues)

    check_issues(issues)

    assert issues == before


class TestDescribe:
    def test_names_the_field_path_received_value_and_recording(self):
        intake = check_issues([_issue(), _issue(severity='high')])

        text = intake.deviations[0].describe()

        assert text.startswith('issues[1].severity')
        assert repr('high') in text
        assert "'suggestion'" in text

    def test_names_the_issue_path_for_a_non_object(self):
        intake = check_issues([_issue(), 'a prose finding'])

        text = intake.unusable[0].describe()

        assert text.startswith('issues[1]')
        assert not text.startswith('issues[1].')
        assert repr('a prose finding') in text

    def test_long_values_are_shortened_to_fit_the_journal_field_cap(self):
        long_value = 'x' * 1000
        intake = check_issues([
            _issue(location=['a' * 500]),
            _issue(severity=long_value),
        ])

        for deviation in intake.deviations:
            text = deviation.describe()
            assert len(text) < _JOURNAL_FIELD_CAP, text
            assert long_value not in text
