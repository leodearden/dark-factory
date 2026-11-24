"""The defect-tests collection map: where it sits beside its junit, how it is read, and the case id of each entry."""

import json
from pathlib import Path
from typing import Any

import pytest

from client_feed.sources.collection_map import (
    CollectionMapError,
    MapEntry,
    MapProblem,
    case_entries,
    collection_map_path,
    read_collection_map,
)

MAP_NAME = 'r.xml.tests.json'


def _write_map(tmp_path: Path, content: Any) -> Path:
    path = tmp_path / MAP_NAME
    path.write_text(json.dumps(content))
    return path


def _read(tmp_path: Path, content: Any) -> tuple[tuple[str, MapEntry], ...]:
    collection_map = read_collection_map(_write_map(tmp_path, content), name=MAP_NAME)
    assert collection_map is not None
    return case_entries(collection_map)


def _map_refusal(path: Path, problem: MapProblem) -> CollectionMapError:
    with pytest.raises(CollectionMapError) as caught:
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


def test_an_empty_map_has_no_entries(tmp_path: Path) -> None:
    assert _read(tmp_path, {}) == ()


def test_entries_are_keyed_by_the_junit_case_id_in_map_order(tmp_path: Path) -> None:
    content = {
        'tests/sub/test_shapes.py::test_plain': {'defects': [], 'criterion': 'C-1'},
        'tests/sub/test_shapes.py::TestGroup::test_method': {'defects': [], 'criterion': 'C-2'},
        'tests/sub/test_shapes.py::test_param[a::b]': {'defects': [], 'criterion': 'C-3'},
        'tests/sub/test_shapes.py::test_param[x/y]': {'defects': ['df-1'], 'criterion': 'C-3'},
        'tests/sub/test_shapes.py::test_unbound': {'defects': ['df-2'], 'criterion': None},
    }

    assert _read(tmp_path, content) == (
        ('tests.sub.test_shapes::test_plain', MapEntry(defects=(), criterion='C-1')),
        ('tests.sub.test_shapes.TestGroup::test_method', MapEntry(defects=(), criterion='C-2')),
        ('tests.sub.test_shapes::test_param[a::b]', MapEntry(defects=(), criterion='C-3')),
        ('tests.sub.test_shapes::test_param[x/y]', MapEntry(defects=('df-1',), criterion='C-3')),
        ('tests.sub.test_shapes::test_unbound', MapEntry(defects=('df-2',), criterion=None)),
    )


def test_no_raw_path_is_ever_a_case_id(tmp_path: Path) -> None:
    content = {
        'tests/sub/test_shapes.py::test_param[/home/u/secret/x]': {'defects': [], 'criterion': 'C-1'},
    }

    ((case, _),) = _read(tmp_path, content)

    assert case == 'tests.sub.test_shapes::test_param[[path]]'


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

    _map_refusal(path, MapProblem.INVALID_MAP)


def test_a_map_that_is_no_mapping_is_refused_at_the_whole_map(tmp_path: Path) -> None:
    path = tmp_path / MAP_NAME
    path.write_text('[]')

    refusal = _map_refusal(path, MapProblem.INVALID_MAP)

    assert refusal.detail.startswith('<map>: ')


def test_a_symlinked_map_is_refused_and_never_followed(tmp_path: Path) -> None:
    target = tmp_path / 'elsewhere.json'
    target.write_text(json.dumps({'tests/test_a.py::test_a': {'defects': [], 'criterion': 'C-1'}}))
    (tmp_path / MAP_NAME).symlink_to(target)

    _map_refusal(tmp_path / MAP_NAME, MapProblem.UNREADABLE)
