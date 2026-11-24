"""Acceptance criteria: the criteria file, the collection map, the junit join and the criteria events (PRD §7.7)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from _archive_files import junit_xml, passed, red, write_junit

from client_feed.contract import (
    CaseOutcome,
    CaseResult,
    Criterion,
    CriterionKind,
    CriterionOutcome,
    CriterionResult,
)
from client_feed.criteria import (
    BoundReport,
    CriteriaError,
    CriteriaProblem,
    CriteriaRun,
    MapEntry,
    collection_map_path,
    criterion_bindings,
    join_criteria,
    parse_criteria,
    read_collection_map,
    verify_criteria,
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


# the join

DECLARED = tuple(
    Criterion.model_validate({**CRITERION, 'id': criterion_id}) for criterion_id in ('C-1', 'C-2', 'C-3')
)


def _case(case_id: str, outcome: CaseOutcome) -> CaseResult:
    return CaseResult(
        id=case_id, outcome=outcome, duration_ms=None, criterion=None, exception_type=None
    )


def _report(cases: list[CaseResult], marked: dict[str, str]) -> BoundReport:
    """A report whose collection map marks each nodeid in `marked` with its criterion."""
    collection_map = {
        nodeid: MapEntry(defects=(), criterion=criterion) for nodeid, criterion in marked.items()
    }
    return BoundReport(cases=tuple(cases), criteria=criterion_bindings(collection_map))


def _outcomes(results: tuple[CriterionResult, ...]) -> dict[str, CriterionOutcome]:
    return {result.id: result.outcome for result in results}


@pytest.mark.criterion('CF-1')
def test_b20_a_criterion_reads_its_bound_tests() -> None:
    report = _report(
        [
            _case('tests.test_a::test_ok', CaseOutcome.PASSED),
            _case('tests.test_a::test_bad', CaseOutcome.FAILED),
            _case('tests.test_a::test_free', CaseOutcome.PASSED),
        ],
        {'tests/test_a.py::test_ok': 'C-1', 'tests/test_a.py::test_bad': 'C-2'},
    )

    assert join_criteria(DECLARED, [report]) == (
        CriterionResult(id='C-1', outcome=CriterionOutcome.PASSED, tests=('tests.test_a::test_ok',)),
        CriterionResult(id='C-2', outcome=CriterionOutcome.FAILED, tests=('tests.test_a::test_bad',)),
        CriterionResult(id='C-3', outcome=CriterionOutcome.NOT_RUN, tests=()),
    )


def test_an_errored_bound_case_fails_its_criterion() -> None:
    report = _report(
        [_case('tests.test_a::test_a', CaseOutcome.ERRORED)], {'tests/test_a.py::test_a': 'C-1'}
    )

    assert _outcomes(join_criteria(DECLARED, [report]))['C-1'] is CriterionOutcome.FAILED


def test_a_bound_test_with_no_case_leaves_its_criterion_not_run() -> None:
    report = _report(
        [_case('tests.test_a::test_other', CaseOutcome.PASSED)], {'tests/test_a.py::test_a': 'C-1'}
    )

    (c1, *_) = join_criteria(DECLARED, [report])

    assert c1 == CriterionResult(
        id='C-1', outcome=CriterionOutcome.NOT_RUN, tests=('tests.test_a::test_a',)
    )


def test_a_skipped_bound_test_leaves_its_criterion_not_run() -> None:
    report = _report(
        [
            _case('tests.test_a::test_a', CaseOutcome.SKIPPED),
            _case('tests.test_a::test_b', CaseOutcome.PASSED),
        ],
        {'tests/test_a.py::test_a': 'C-1', 'tests/test_a.py::test_b': 'C-1'},
    )

    assert _outcomes(join_criteria(DECLARED, [report]))['C-1'] is CriterionOutcome.NOT_RUN


def test_one_red_case_of_a_shared_id_fails_its_criterion() -> None:
    report = _report(
        [
            _case('tests.test_a::test_a', CaseOutcome.PASSED),
            _case('tests.test_a::test_a', CaseOutcome.FAILED),
        ],
        {'tests/test_a.py::test_a': 'C-1'},
    )

    assert _outcomes(join_criteria(DECLARED, [report]))['C-1'] is CriterionOutcome.FAILED


def test_passed_needs_every_bound_test_to_have_passed_cases() -> None:
    every = _report(
        [
            _case('tests.test_a::test_a', CaseOutcome.PASSED),
            _case('tests.test_a::test_b', CaseOutcome.PASSED),
        ],
        {'tests/test_a.py::test_a': 'C-1', 'tests/test_a.py::test_b': 'C-1'},
    )
    one_missing = _report(
        [_case('tests.test_a::test_a', CaseOutcome.PASSED)],
        {'tests/test_a.py::test_a': 'C-1', 'tests/test_a.py::test_b': 'C-1'},
    )

    assert _outcomes(join_criteria(DECLARED, [every]))['C-1'] is CriterionOutcome.PASSED
    assert _outcomes(join_criteria(DECLARED, [one_missing]))['C-1'] is CriterionOutcome.NOT_RUN


def test_a_binding_to_an_undeclared_criterion_is_ignored() -> None:
    report = _report(
        [_case('tests.test_a::test_a', CaseOutcome.FAILED)], {'tests/test_a.py::test_a': 'C-9'}
    )

    assert join_criteria(DECLARED, [report]) == tuple(
        CriterionResult(id=criterion.id, outcome=CriterionOutcome.NOT_RUN, tests=())
        for criterion in DECLARED
    )


def test_the_join_is_per_report() -> None:
    bound = _report(
        [_case('tests.test_a::test_a', CaseOutcome.PASSED)], {'tests/test_a.py::test_a': 'C-1'}
    )
    unbound = _report([_case('tests.test_a::test_a', CaseOutcome.FAILED)], {})

    (c1, *_) = join_criteria(DECLARED, [bound, unbound])

    assert c1 == CriterionResult(
        id='C-1', outcome=CriterionOutcome.PASSED, tests=('tests.test_a::test_a',)
    )


def test_bound_test_ids_are_sorted_and_unique() -> None:
    cases = [
        _case('tests.test_a::test_b', CaseOutcome.PASSED),
        _case('tests.test_a::test_a', CaseOutcome.PASSED),
    ]
    marked = {'tests/test_a.py::test_b': 'C-1', 'tests/test_a.py::test_a': 'C-1'}

    (c1, *_) = join_criteria(DECLARED, [_report(cases, marked), _report(cases, marked)])

    assert c1.tests == ('tests.test_a::test_a', 'tests.test_a::test_b')


# a verify run's criteria

SHA = 'a' * 40
RUN_AT = datetime(2026, 10, 1, 10, 5, tzinfo=UTC)


def _archived_junit(root: Path, content: bytes, *, minute: int, qualifiers: str) -> Path:
    when = datetime(2026, 10, 1, 10, minute, tzinfo=UTC)
    return write_junit(root, when, content, qualifiers=qualifiers).relative_to(root)


def _map_beside(root: Path, junit: Path, marked: dict[str, str]) -> Path:
    path = root / collection_map_path(junit)
    path.write_text(
        json.dumps({nodeid: {'defects': [], 'criterion': c} for nodeid, c in marked.items()})
    )
    return path


def _verify(root: Path, junits: list[Path]) -> CriteriaRun | None:
    return verify_criteria(root, junits, declared=DECLARED, commit=SHA, run_at=RUN_AT)


def test_a_verify_run_joins_its_archived_junit_with_the_map_beside_it(tmp_path: Path) -> None:
    junit = _archived_junit(
        tmp_path, junit_xml(passed('test_a'), red('test_b', 'failure')), minute=1, qualifiers='alpha'
    )
    _map_beside(tmp_path, junit, {'tests/test_a.py::test_a': 'C-1', 'tests/test_a.py::test_b': 'C-2'})

    assert _verify(tmp_path, [junit]) == CriteriaRun(
        commit=SHA,
        run_at=RUN_AT,
        results=(
            CriterionResult(id='C-1', outcome=CriterionOutcome.PASSED, tests=('tests.test_a::test_a',)),
            CriterionResult(id='C-2', outcome=CriterionOutcome.FAILED, tests=('tests.test_a::test_b',)),
            CriterionResult(id='C-3', outcome=CriterionOutcome.NOT_RUN, tests=()),
        ),
    )


def test_each_module_junit_is_joined_with_its_own_map(tmp_path: Path) -> None:
    alpha = _archived_junit(tmp_path, junit_xml(passed('test_a')), minute=1, qualifiers='alpha')
    beta = _archived_junit(
        tmp_path, junit_xml(red('test_a', 'failure')), minute=2, qualifiers='beta'
    )
    _map_beside(tmp_path, alpha, {'tests/test_a.py::test_a': 'C-1'})
    _map_beside(tmp_path, beta, {'tests/test_a.py::test_a': 'C-2'})

    run = _verify(tmp_path, [alpha, beta])

    assert run is not None
    assert _outcomes(run.results) == {
        'C-1': CriterionOutcome.PASSED,
        'C-2': CriterionOutcome.FAILED,
        'C-3': CriterionOutcome.NOT_RUN,
    }


def test_a_junit_with_no_map_binds_nothing_while_its_sibling_binds(tmp_path: Path) -> None:
    alpha = _archived_junit(tmp_path, junit_xml(passed('test_a')), minute=1, qualifiers='alpha')
    beta = _archived_junit(
        tmp_path, junit_xml(red('test_a', 'failure')), minute=2, qualifiers='beta'
    )
    _map_beside(tmp_path, alpha, {'tests/test_a.py::test_a': 'C-1'})

    run = _verify(tmp_path, [alpha, beta])

    assert run is not None
    assert _outcomes(run.results)['C-1'] is CriterionOutcome.PASSED


def test_a_run_with_no_map_records_no_criteria(tmp_path: Path) -> None:
    junit = _archived_junit(tmp_path, junit_xml(passed('test_a')), minute=1, qualifiers='alpha')

    assert _verify(tmp_path, [junit]) is None


def test_a_malformed_archived_map_is_refused_relative_to_the_root(tmp_path: Path) -> None:
    junit = _archived_junit(tmp_path, junit_xml(passed('test_a')), minute=1, qualifiers='alpha')
    (tmp_path / collection_map_path(junit)).write_text('[]')

    with pytest.raises(CriteriaError) as caught:
        _verify(tmp_path, [junit])

    assert caught.value.problem is CriteriaProblem.INVALID_MAP
    assert str(collection_map_path(junit)) in str(caught.value)
    assert str(tmp_path) not in str(caught.value)
