"""``state.py::validate_return`` over the fixture returns.

Form checks never consult the store; store contradictions are read through a
``StoreReader`` on the fixture fake store, whose request journal shows which
validations read it.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_fixture import FIXTURE_DIR, FakeStore, load_state, running_fake_store

state = load_state()

RETURNS = FIXTURE_DIR / 'returns'
PROJECT_ROOT = '/review-all-fixture/project'
PROGRAM = state.parse_program(json.loads((FIXTURE_DIR / 'program.json').read_text()))

Mutation = Callable[[dict[str, Any]], None]


def _text(name: str) -> str:
    return (RETURNS / name).read_text()


def _data(name: str) -> dict[str, Any]:
    return json.loads(_text(name))


def _set(key: str, value: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        data[key] = value
    return mutate


def _set_in(section: str, index: int, key: str, value: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        data[section][index][key] = value
    return mutate


def _drop(key: str) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        del data[key]
    return mutate


def _mutated(name: str, *mutations: Mutation) -> str:
    data = _data(name)
    for mutate in mutations:
        mutate(data)
    return json.dumps(data)


def _tools_calls(store: FakeStore) -> int:
    return sum(1 for entry in store.requests() if entry['body'].get('method') == 'tools/call')


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture(scope='module')
def reader(store: FakeStore) -> Any:
    return state.StoreReader(store.url, PROJECT_ROOT)


def _validate(text: str, reader: Any, ledger: Any = ()) -> Any:
    return state.validate_return(text, program=PROGRAM, ledger=ledger, store=reader)


def test_an_author_return_parses_into_typed_values(reader: Any) -> None:
    check = _validate(_text('S1.author.json'), reader)
    assert check.ok, check.rejections
    assert check.rejections == ()
    value = check.value
    assert isinstance(value, state.PrdReturn)
    assert value.pass_ is state.Pass.AUTHOR
    questions = {q.id: q for q in value.open_questions}
    assert questions['Q3'].lead_rulable is True
    assert questions['Q1'].lead_rulable is False
    assert questions['Q1'].source is state.QuestionSource.AUTHOR
    assert value.touches[0] == 'src/fixture/alpha.py'


def test_a_fold_amendment_carries_the_critic_question(reader: Any) -> None:
    check = _validate(_text('S1.author.amend.json'), reader)
    assert check.ok, check.rejections
    questions = {q.id: q for q in check.value.open_questions}
    assert questions['Q2'].source is state.QuestionSource.CRITIC


def test_an_investigate_return_names_its_item(reader: Any) -> None:
    check = _validate(_text('S1.investigate-Q2.json'), reader)
    assert check.ok, check.rejections
    assert check.value.item == 'Q2'
    assert check.value.rulings_in_session[0].id == 'R1'


@pytest.mark.parametrize('name', ['S2.author.json', 'S3.author.json'])
def test_the_other_author_returns_are_valid(name: str, reader: Any) -> None:
    check = _validate(_text(name), reader)
    assert check.ok, check.rejections


FORM_CASES: list[tuple[str, Mutation, str]] = [
    ('S1.author.json', _set('schema', 'x'), 'schema'),
    ('S1.author.json', _set('pass', 'review'), 'pass'),
    ('S1.decompose.json', _set_in('lead_actions', 0, 'op', 'delete'), 'lead_actions[0].op'),
    ('S1.author.json', _set('reviewer_mood', 'calm'), 'reviewer_mood'),
    ('S1.author.json', _set('filed', []), 'filed'),
    ('S1.decompose.json', _set('touches', []), 'touches'),
    ('S1.investigate-Q2.json', _set('prd_path', 'plans/review-all-fixture-s1-prd.md'), 'prd_path'),
    ('S1.author.json', _set('run_id', 'review-all-other-20261001'), 'run_id'),
    ('S1.author.json', _set('stream', 'S9'), 'stream'),
    ('S1.decompose.json', _set_in('held', 0, 'waits_on', 'S9'), 'held[0].waits_on'),
    ('S1.decompose.json', _set_in('moves', 0, 'from', '/abs/x.py'), 'moves[0].from'),
    ('S1.decompose.json', _set_in('moves', 0, 'from', '../x.py'), 'moves[0].from'),
    ('S1.author.json', _set('commits', ['abc']), 'commits'),
    ('S1.author.json', _set_in('open_questions', 0, 'source', 'reviewer'), 'open_questions[0].source'),
    ('S1.author.json', _drop('touches'), 'touches'),
    ('S1.decompose.json', _drop('moves'), 'moves'),
    ('S1.investigate-Q2.json', _drop('item'), 'item'),
    ('S1.investigate-Q2.json', _set('rulings_in_session', []), 'rulings_in_session'),
    ('S1.decompose.json', _set_in('filed', 0, 'task_id', '101'), 'filed[0].task_id'),
    ('S1.author.json', _set_in('open_questions', 0, 'needs_human', 'yes'),
     'open_questions[0].needs_human'),
]


@pytest.mark.parametrize(
    ('name', 'mutate', 'field'), FORM_CASES, ids=[f'{case[0]}:{case[2]}' for case in FORM_CASES],
)
def test_a_form_error_is_rejected_naming_its_field(
    name: str, mutate: Mutation, field: str, reader: Any,
) -> None:
    check = _validate(_mutated(name, mutate), reader)
    assert not check.ok
    assert check.value is None
    assert all(r.kind is state.RejectionClass.FORM for r in check.rejections)
    assert field in [r.field for r in check.rejections], check.rejections


def test_notes_over_the_cap_are_a_form_error_and_read_no_store(
    store: FakeStore, reader: Any,
) -> None:
    before = _tools_calls(store)
    check = _validate(_text('S1.decompose.notes-over-cap.json'), reader)
    assert check.value is None
    assert check.measured_at is None
    [rejection] = check.rejections
    assert (rejection.kind, rejection.field) == (state.RejectionClass.FORM, 'notes')
    assert '601' in rejection.reason and '600' in rejection.reason
    assert str(rejection).startswith('form: notes: ')
    assert _tools_calls(store) == before


def test_a_file_that_does_not_parse_is_a_form_error(reader: Any) -> None:
    check = _validate('not json', reader)
    assert [(r.kind, r.field) for r in check.rejections] == [
        (state.RejectionClass.FORM, '<file>'),
    ]
    assert check.value is None


RUN_ID = 'review-all-fixture-20261019'
AUTHORED = (
    {'seq': 1, 'at': '2026-10-19T12:00:00Z', 'by': 'lead', 'op': 'stream_status', 'stream': 'S1',
     'status': 'authored', 'prd_path': 'plans/review-all-fixture-s1-prd.md'},
    {'seq': 2, 'at': '2026-10-19T12:00:00Z', 'by': 'lead', 'op': 'stream_status', 'stream': 'S2',
     'status': 'authored', 'prd_path': 'plans/review-all-fixture-s2-prd.md'},
)


def _contradictions(check: Any) -> dict[str, list[str]]:
    assert check.value is None
    assert check.rejections
    assert all(r.kind is state.RejectionClass.CONTRADICTION for r in check.rejections)
    found: dict[str, list[str]] = {}
    for rejection in check.rejections:
        found.setdefault(rejection.field, []).append(rejection.reason)
    return found


def _set_filed(index: int, **fields: Any) -> Mutation:
    def mutate(data: dict[str, Any]) -> None:
        data['filed'][index].update(fields)
    return mutate


def test_a_decompose_return_consistent_with_the_store_is_accepted(reader: Any) -> None:
    check = _validate(_text('S1.decompose.json'), reader, AUTHORED)
    assert check.ok, check.rejections
    assert check.measured_at is not None
    assert [f.task_id for f in check.value.filed] == [101, 102]


def test_a_release_onto_a_store_dependency_is_accepted(reader: Any) -> None:
    check = _validate(_text('S2.decompose.json'), reader, AUTHORED)
    assert check.ok, check.rejections
    [released] = check.value.released
    assert (released.task_id, released.onto) == (113, (106,))


def test_a_decompose_before_its_author_return_is_a_contradiction(reader: Any) -> None:
    found = _contradictions(_validate(_text('S1.decompose.json'), reader, ()))
    assert any('S1' in reason and 'author' in reason for reason in found['pass'])


def test_a_filed_id_absent_from_the_store_is_a_contradiction(reader: Any) -> None:
    found = _contradictions(_validate(_text('S1.decompose.absent-filed.json'), reader, AUTHORED))
    assert any('999' in reason for reason in found['filed[0].task_id'])


def test_a_filed_key_outside_the_streams_keys_after_amendments(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _set_filed(1, keys=['fk-0000000000b1']))
    found = _contradictions(_validate(text, reader, AUTHORED))
    assert any('fk-0000000000b1' in reason for reason in found['filed[1].keys[0]'])
    amended = (*AUTHORED, {
        'seq': 3, 'at': '2026-10-19T12:00:00Z', 'by': 'lead', 'op': 'amend_stream',
        'stream': 'S1', 'field': 'keys', 'change': 'add', 'value': 'fk-0000000000b1',
        'ruling': 'D1',
    })
    assert _validate(text, reader, amended).ok


def test_a_filed_task_of_another_run_is_a_contradiction(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _set_filed(0, task_id=201))
    reasons = _contradictions(_validate(text, reader, AUTHORED))['filed[0].task_id']
    assert any(RUN_ID in reason and 'None' in reason for reason in reasons)


def test_an_unadopted_filed_task_must_carry_the_streams_prd_path(reader: Any) -> None:
    unadopted = _mutated('S1.decompose.json', _set_filed(0, task_id=108, gate_dep=False))
    reasons = _contradictions(_validate(unadopted, reader, AUTHORED))['filed[0].task_id']
    assert any('prd_path' in reason for reason in reasons)
    adopted = _mutated('S1.decompose.json', _set_filed(0, task_id=108, gate_dep=False, adopted=True))
    assert _validate(adopted, reader, AUTHORED).ok


def test_a_held_leaf_must_be_deferred_and_waiting(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _set_in('held', 0, 'task_id', 101))
    reasons = _contradictions(_validate(text, reader, AUTHORED))['held[0].task_id']
    assert any('deferred' in reason for reason in reasons)
    assert any('x_seam_wait' in reason for reason in reasons)


def test_a_held_leaf_must_wait_on_the_stream_the_store_records(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _set_in('held', 0, 'waits_on', 'S3'))
    reasons = _contradictions(_validate(text, reader, AUTHORED))['held[0].waits_on']
    assert any('S2' in reason and 'S3' in reason for reason in reasons)


def test_a_gate_dep_the_gate_does_not_hold_is_a_contradiction(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _set_filed(1, gate_dep=True))
    reasons = _contradictions(_validate(text, reader, AUTHORED))['filed[1].gate_dep']
    assert any('102' in reason and '110' in reason for reason in reasons)


def test_a_gate_dep_without_a_gate_is_a_contradiction(tmp_path: Path) -> None:
    data = json.loads((FIXTURE_DIR / 'store.json').read_text())
    data['tasks'] = [task for task in data['tasks'] if task['id'] != 110]
    gateless = tmp_path / 'store.json'
    gateless.write_text(json.dumps(data))
    with running_fake_store(store=gateless) as running:
        check = _validate(
            _text('S1.decompose.json'), state.StoreReader(running.url, PROJECT_ROOT), AUTHORED,
        )
    reasons = _contradictions(check)['filed[0].gate_dep']
    assert f'no human_gate task carries trigger_chain.run_id = {RUN_ID}' in reasons


def _released(task_id: int, onto: list[int]) -> Mutation:
    return _set('released', [
        {'task_id': task_id, 'onto': onto, 'contract_as_delivered': 'C1 as delivered'},
    ])


def test_a_release_of_an_unreleased_task_is_a_contradiction(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _released(103, [101]))
    reasons = _contradictions(_validate(text, reader, AUTHORED))['released[0].task_id']
    assert any('103' in reason and 'x_seam_released' in reason for reason in reasons)


def test_a_release_onto_a_task_it_does_not_depend_on_is_a_contradiction(reader: Any) -> None:
    text = _mutated('S1.decompose.json', _released(113, [102]))
    reasons = _contradictions(_validate(text, reader, AUTHORED))['released[0].onto[0]']
    assert any('102' in reason and '113' in reason for reason in reasons)


def test_several_contradictions_are_reported_together(reader: Any) -> None:
    text = _mutated(
        'S1.decompose.json',
        _set_in('held', 0, 'waits_on', 'S3'),
        _set_filed(1, gate_dep=True),
    )
    found = _contradictions(_validate(text, reader, AUTHORED))
    assert {'held[0].waits_on', 'filed[1].gate_dep'} <= set(found)
