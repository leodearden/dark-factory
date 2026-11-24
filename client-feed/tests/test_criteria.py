"""Acceptance criteria: the criteria file, the collection map, the junit join and the criteria events (PRD §7.7)."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from client_feed.contract import Criterion, CriterionKind
from client_feed.criteria import (
    CriteriaError,
    CriteriaProblem,
    collection_map_path,
    criterion_bindings,
    parse_criteria,
    read_collection_map,
)

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


# the collection map

MAP_NAME = 'r.xml.tests.json'


def _write_map(tmp_path: Path, content: Any) -> Path:
    path = tmp_path / MAP_NAME
    path.write_text(json.dumps(content))
    return path


def _map_refusal(path: Path, problem: CriteriaProblem) -> CriteriaError:
    with pytest.raises(CriteriaError) as caught:
        read_collection_map(path, name=MAP_NAME)
    assert caught.value.problem is problem
    assert MAP_NAME in str(caught.value)
    return caught.value


@pytest.mark.parametrize(
    ('junit', 'expected'),
    [
        (
            Path('d/attempt-1.alpha.junit-20261001T100000_000000Z.xml.gz'),
            Path('d/attempt-1.alpha.junit-20261001T100000_000000Z.xml.gz.tests.json'),
        ),
        (Path('r.xml'), Path('r.xml.tests.json')),
    ],
)
def test_the_map_sits_beside_its_junit(junit: Path, expected: Path) -> None:
    assert collection_map_path(junit) == expected


def test_a_missing_map_is_not_recorded(tmp_path: Path) -> None:
    assert read_collection_map(tmp_path / MAP_NAME, name=MAP_NAME) is None


def test_an_empty_map_marks_nothing(tmp_path: Path) -> None:
    collection_map = read_collection_map(_write_map(tmp_path, {}), name=MAP_NAME)

    assert collection_map is not None
    assert dict(collection_map) == {}
    assert criterion_bindings(collection_map) == {}


def test_bindings_are_keyed_by_the_junit_case_id(tmp_path: Path) -> None:
    content = {
        'tests/sub/test_shapes.py::test_plain': {'defects': [], 'criterion': 'C-1'},
        'tests/sub/test_shapes.py::TestGroup::test_method': {'defects': [], 'criterion': 'C-2'},
        'tests/sub/test_shapes.py::test_param[a::b]': {'defects': [], 'criterion': 'C-3'},
        'tests/sub/test_shapes.py::test_param[x/y]': {'defects': ['df-1'], 'criterion': 'C-3'},
        'tests/sub/test_shapes.py::test_unbound': {'defects': ['df-2'], 'criterion': None},
    }
    collection_map = read_collection_map(_write_map(tmp_path, content), name=MAP_NAME)
    assert collection_map is not None

    assert criterion_bindings(collection_map) == {
        'tests.sub.test_shapes::test_plain': 'C-1',
        'tests.sub.test_shapes.TestGroup::test_method': 'C-2',
        'tests.sub.test_shapes::test_param[a::b]': 'C-3',
        'tests.sub.test_shapes::test_param[x/y]': 'C-3',
    }


def test_no_raw_path_is_ever_a_binding_key(tmp_path: Path) -> None:
    content = {
        'tests/sub/test_shapes.py::test_param[/home/u/secret/x]': {
            'defects': [],
            'criterion': 'C-1',
        },
    }
    collection_map = read_collection_map(_write_map(tmp_path, content), name=MAP_NAME)
    assert collection_map is not None

    assert criterion_bindings(collection_map) == {'tests.sub.test_shapes::test_param[[path]]': 'C-1'}


@pytest.mark.parametrize(
    'text',
    [
        '{"tests/test_a.py::test_a": ',
        '[]',
        json.dumps({'tests/test_a.py::test_a': {'criterion': 'C-1'}}),
        json.dumps({'tests/test_a.py::test_a': {'defects': [], 'criterion': 'C-1', 'kind': 'x'}}),
        json.dumps({'tests/test_a.py::test_a': {'defects': [], 'criterion': 7}}),
    ],
    ids=['not-json', 'a-list', 'missing-defects', 'extra-key', 'numeric-criterion'],
)
def test_a_malformed_map_is_refused_by_name(tmp_path: Path, text: str) -> None:
    path = tmp_path / MAP_NAME
    path.write_text(text)

    _map_refusal(path, CriteriaProblem.INVALID_MAP)


def test_a_symlinked_map_is_refused_and_never_followed(tmp_path: Path) -> None:
    target = tmp_path / 'elsewhere.json'
    target.write_text(json.dumps({'tests/test_a.py::test_a': {'defects': [], 'criterion': 'C-1'}}))
    (tmp_path / MAP_NAME).symlink_to(target)

    _map_refusal(tmp_path / MAP_NAME, CriteriaProblem.UNREADABLE)
