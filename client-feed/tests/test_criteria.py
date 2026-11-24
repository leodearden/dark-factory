"""Acceptance criteria: the criteria file, the collection map, the junit join and the criteria events (PRD §7.7)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from client_feed.contract import Criterion, CriterionKind
from client_feed.criteria import CriteriaError, CriteriaProblem, parse_criteria

EXEMPLAR = Path(__file__).parent / 'fixtures' / 'acceptance' / 'criteria.yaml'
NAME = 'acceptance/criteria.yaml'

CRITERION: dict[str, Any] = {
    'id': 'C-1',
    'prd': 'plans/ledger-prd.md',
    'capability': 'Ledger',
    'statement': 'Every entry shows the running balance after it.',
    'kind': 'success',
}


def _criteria_text(**overrides: Any) -> str:
    return yaml.safe_dump({'format': 1, 'criteria': [CRITERION], **overrides})


def _refusal(text: str) -> CriteriaError:
    with pytest.raises(CriteriaError) as caught:
        parse_criteria(text, name=NAME)
    assert caught.value.problem is CriteriaProblem.INVALID_CRITERIA
    assert NAME in str(caught.value)
    return caught.value


# the criteria file


def test_the_exemplar_declares_its_criteria_in_file_order() -> None:
    criteria = parse_criteria(EXEMPLAR.read_text(), name=NAME)

    assert all(isinstance(criterion, Criterion) for criterion in criteria)
    assert [criterion.id for criterion in criteria] == ['CF-1', 'CF-2']
    assert {criterion.prd for criterion in criteria} == {'plans/client-feed-prd.md'}
    assert [criterion.kind for criterion in criteria] == [
        CriterionKind.SUCCESS,
        CriterionKind.FAILURE,
    ]


def test_an_empty_declaration_parses_to_no_criteria() -> None:
    assert parse_criteria(_criteria_text(criteria=[]), name=NAME) == ()


@pytest.mark.parametrize(
    'text',
    [
        _criteria_text(format=2),
        yaml.safe_dump({'criteria': [CRITERION]}),
        _criteria_text(owner='leo'),
        _criteria_text(criteria=[{**CRITERION, 'id': 'C:12'}]),
        _criteria_text(criteria=[{**CRITERION, 'kind': 'maybe'}]),
        _criteria_text(criteria=[{**CRITERION, 'prd': '../x.md'}]),
        yaml.safe_dump([CRITERION]),
        '{format: 1, criteria: [',
    ],
    ids=[
        'format-2',
        'missing-format',
        'unknown-top-level-key',
        'colon-id',
        'unknown-kind',
        'escaping-prd',
        'not-a-mapping',
        'not-yaml',
    ],
)
def test_an_invalid_criteria_file_is_refused_by_name(text: str) -> None:
    _refusal(text)


def test_a_duplicate_id_is_refused_naming_the_id() -> None:
    second = {**CRITERION, 'statement': 'A balance is never rounded before the last entry.'}

    refusal = _refusal(_criteria_text(criteria=[CRITERION, second]))

    assert 'C-1' in str(refusal)
