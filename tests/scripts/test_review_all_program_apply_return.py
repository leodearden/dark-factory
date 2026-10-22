"""``skills/review-all/scripts/program/apply_return.py``, run as a ``-I -S`` subprocess outside the repo.

The store is the fixture fake store; each test's home is initialised from the
fixture program, and fixture returns are copied into its inbox.
"""
from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from review_all_program_cli import FIXTURE_RETURNS, RUN_ID, ProgramEnv, load_program_module
from review_all_program_fixture import FIXTURE_DIR, FakeStore, load_state, running_fake_store

state = load_state()

S1_PRD = 'plans/review-all-fixture-s1-prd.md'
S1_AUTHOR_DIGEST = 'S1 author: 0 filed, 0 held, 2 questions, 0 actions → returns/S1.author.json'


@pytest.fixture(scope='module')
def store() -> Iterator[FakeStore]:
    with running_fake_store() as running:
        yield running


@pytest.fixture
def env(tmp_path: Path, store: FakeStore) -> ProgramEnv:
    env = ProgramEnv.under(tmp_path, store.url)
    proc = env.run('record.py', 'init', FIXTURE_DIR / 'program.json')
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return env


def _output(proc: Any) -> str:
    return proc.stdout + proc.stderr


def _ledger(env: ProgramEnv) -> list[dict[str, Any]]:
    """The ledger lines with their clock stamp dropped."""
    return [
        {key: value for key, value in line.items() if key != 'at'}
        for line in state.read_ledger(state.Home(env.home))
    ]


def _inbox(env: ProgramEnv) -> list[str]:
    return sorted(path.name for path in (env.home / 'inbox').iterdir())


def _applied(env: ProgramEnv, fixture_return: str) -> None:
    proc = env.run('apply_return.py', env.put_inbox(fixture_return))
    assert proc.returncode == 0, _output(proc)


def _question(qid: str, **fields: Any) -> dict[str, Any]:
    return {'op': 'question', 'stream': 'S1', 'id': qid, 'return': 'returns/S1.author.json', **fields}


def test_b2_an_author_return_is_moved_and_recorded(env: ProgramEnv) -> None:
    proc = env.run('apply_return.py', env.put_inbox('S1.author.json'))
    assert proc.returncode == 0, _output(proc)
    applied = env.home / 'returns' / 'S1.author.json'
    assert applied.read_bytes() == (FIXTURE_RETURNS / 'S1.author.json').read_bytes()
    assert _inbox(env) == []
    by = {'by': 'apply_return.py'}
    assert _ledger(env) == [
        {'seq': 1, **by, 'op': 'stream_status', 'stream': 'S1', 'status': 'authored',
         'prd_path': S1_PRD, 'return': 'returns/S1.author.json'},
        {'seq': 2, **by, **_question(
            'Q1', question='Does alpha or beta own seam contract C1?',
            options=('alpha owns C1', 'beta owns C1'), recommendation='alpha owns C1',
            needs_human=True, lead_rulable=False, source='author')},
        {'seq': 3, **by, **_question(
            'Q3', question='Name the new module new_name.py or renamed.py?',
            options=('new_name.py', 'renamed.py'), recommendation='new_name.py',
            needs_human=False, lead_rulable=True, source='author')},
        {'seq': 4, **by, 'op': 'xref', 'from': 'S1', 'to': 'S2',
         'text': "S2 consumes C2 once S1's alpha task 101 lands.",
         'return': 'returns/S1.author.json'},
    ]
    assert not (env.home / 'rulings.jsonl').exists()
    digest, *rest = proc.stdout.splitlines()
    assert digest == S1_AUTHOR_DIGEST
    (needs_human,) = rest
    assert 'S1:Q1' in needs_human
    assert 'Does alpha or beta own seam contract C1?' in needs_human
    assert 'alpha owns C1' in needs_human


def test_b3_a_form_rejection_names_the_field_and_writes_nothing(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    env.put_inbox('S1.decompose.notes-over-cap.json')
    before = env.snapshot()
    from_home = dataclasses.replace(env, cwd=env.home)
    proc = from_home.run('apply_return.py', 'inbox/S1.decompose.json')
    assert proc.returncode == 1
    assert 'form: notes:' in _output(proc)
    assert '601' in _output(proc)
    assert env.snapshot() == before


def test_a_return_under_another_streams_name_is_a_form_rejection(env: ProgramEnv) -> None:
    misnamed = env.put_inbox('S1.author.json', as_stem='S2.author')
    before = env.snapshot()
    proc = env.run('apply_return.py', misnamed)
    assert proc.returncode == 1
    form_lines = [line for line in _output(proc).splitlines() if line.startswith('form: <file>:')]
    assert form_lines, _output(proc)
    assert 'S1.author' in form_lines[0] and 'S2.author' in form_lines[0]
    assert env.snapshot() == before


def test_check_reports_the_digest_and_writes_nothing(env: ProgramEnv) -> None:
    inbox_file = env.put_inbox('S1.author.json')
    before = env.snapshot()
    proc = env.run('apply_return.py', '--check', inbox_file)
    assert proc.returncode == 0, _output(proc)
    assert proc.stdout.splitlines()[0] == f'check ok: {S1_AUTHOR_DIGEST}'
    assert env.snapshot() == before


def _form_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith('form: ')]


def test_check_reports_the_same_form_rejection_as_apply(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = env.put_inbox('S1.decompose.notes-over-cap.json')
    before = env.snapshot()
    checked = env.run('apply_return.py', '--check', inbox_file)
    assert checked.returncode == 1
    applied = env.run('apply_return.py', inbox_file)
    assert applied.returncode == 1
    assert _form_lines(_output(checked)) == _form_lines(_output(applied))
    assert any(line.startswith('form: notes:') for line in _form_lines(_output(checked)))
    assert env.snapshot() == before


def test_a_file_outside_the_inbox_is_a_usage_error(env: ProgramEnv) -> None:
    stray = env.cwd / 'S1.author.json'
    shutil.copyfile(FIXTURE_RETURNS / 'S1.author.json', stray)
    proc = env.run('apply_return.py', stray)
    assert proc.returncode == 2
    assert 'inbox' in proc.stderr


def test_an_uninitialised_home_names_record_py_init(env: ProgramEnv) -> None:
    orphan = env.homes_root / 'review-all-other-20261019' / 'inbox' / 'S1.author.json'
    orphan.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE_RETURNS / 'S1.author.json', orphan)
    proc = env.run('apply_return.py', orphan)
    assert proc.returncode == 1
    assert 'record.py init' in _output(proc)


def _contradictions(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith('contradiction: ')]


def test_b3_a_store_contradiction_stops_for_the_lead_and_writes_nothing(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = env.put_inbox('S1.decompose.absent-filed.json')
    before = env.snapshot()
    applied = env.run('apply_return.py', inbox_file)
    assert applied.returncode == 1
    (absent,) = [line for line in _contradictions(_output(applied)) if 'filed[0].task_id:' in line]
    assert '999' in absent
    assert 'stop for the lead' in _output(applied).splitlines()[-1]
    assert env.snapshot() == before
    checked = env.run('apply_return.py', '--check', inbox_file)
    assert checked.returncode == 1
    assert _contradictions(_output(checked)) == _contradictions(_output(applied))
    assert env.snapshot() == before


def test_a_decompose_before_its_author_return_is_a_contradiction(env: ProgramEnv) -> None:
    inbox_file = env.put_inbox('S1.decompose.json')
    before = env.snapshot()
    proc = env.run('apply_return.py', inbox_file)
    assert proc.returncode == 1
    assert any(line.startswith('contradiction: pass:') for line in _output(proc).splitlines())
    assert env.snapshot() == before


def _tools_calls(store: FakeStore) -> list[str]:
    return [
        entry['body']['params']['name'] for entry in store.requests()
        if entry['body'].get('method') == 'tools/call'
    ]


def test_b4_a_decompose_return_records_its_status_and_numbered_actions(
    env: ProgramEnv, store: FakeStore,
) -> None:
    _applied(env, 'S1.author.json')
    calls_before = len(_tools_calls(store))
    proc = env.run('apply_return.py', env.put_inbox('S1.decompose.json'))
    assert proc.returncode == 0, _output(proc)
    by = {'by': 'apply_return.py', 'return': 'returns/S1.decompose.json'}
    assert _ledger(env)[4:] == [
        {'seq': 5, 'op': 'stream_status', 'stream': 'S1', 'status': 'decomposed',
         'prd_path': S1_PRD, **by},
        {'seq': 6, 'op': 'action', 'id': 'A1', 'lead_op': 'note', 'task': 102, 'target': None,
         'text': 'Route callers only after 101 lands.', 'needs_human': False, **by},
        {'seq': 7, 'op': 'action', 'id': 'A2', 'lead_op': 'edge', 'task': 102, 'target': (106,),
         'text': '102 consumes C2 from S2 α.', 'needs_human': False, **by},
    ]
    assert proc.stdout.splitlines()[0] == (
        'S1 decompose: 2 filed, 1 held, 0 questions, 2 actions → returns/S1.decompose.json'
    )
    assert set(_tools_calls(store)[calls_before:]) <= {'get_task', 'find_tasks_by_metadata'}
    assert len(_tools_calls(store)) > calls_before


def _inbox_variant(
    env: ProgramEnv, fixture_return: str, mutate: Callable[[dict[str, Any]], None],
    as_stem: str | None = None,
) -> Path:
    data = json.loads((FIXTURE_RETURNS / fixture_return).read_text())
    mutate(data)
    target = env.put_inbox(fixture_return, as_stem=as_stem)
    target.write_text(json.dumps(data, indent=2))
    return target


def _with_lead_note(data: dict[str, Any]) -> None:
    data['lead_actions'] = [
        {'op': 'note', 'task': 105, 'target': None, 'text': 'Watch C1.', 'needs_human': False},
    ]


def test_action_ids_stay_dense_across_returns(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    _applied(env, 'S1.decompose.json')
    proc = env.run('apply_return.py', _inbox_variant(env, 'S2.author.json', _with_lead_note))
    assert proc.returncode == 0, _output(proc)
    actions = [line for line in _ledger(env) if line['op'] == 'action']
    assert [line['id'] for line in actions] == ['A1', 'A2', 'A3']
    assert actions[-1]['return'] == 'returns/S2.author.json'


def test_a_second_application_of_the_same_return_is_refused(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    _applied(env, 'S1.decompose.json')
    again = env.put_inbox('S1.decompose.json')
    before = env.snapshot()
    proc = env.run('apply_return.py', again)
    assert proc.returncode == 1
    assert 'already applied' in _output(proc)
    assert 'returns/S1.decompose.json' in _output(proc)
    assert env.snapshot() == before
    amended = env.run('apply_return.py', again, '--amend')
    assert amended.returncode == 1
    assert 'already applied' in _output(amended)
    assert env.snapshot() == before


def test_different_content_for_an_applied_return_needs_amend(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    changed = _inbox_variant(env, 'S1.author.json', lambda data: data.update(notes='Revised.'))
    before = env.snapshot()
    proc = env.run('apply_return.py', changed)
    assert proc.returncode == 1
    assert '--amend' in _output(proc)
    assert env.snapshot() == before


def _fixture_store_factory(store: FakeStore) -> Callable[[Any], Any]:
    program_state = load_program_module('state')
    return lambda program: program_state.StoreReader(store.url, program.project_root)


def test_apply_returns_typed_outcomes_for_eta(env: ProgramEnv, store: FakeStore) -> None:
    apply_return = load_program_module('apply_return')
    factory = _fixture_store_factory(store)
    applied = apply_return.apply(env.put_inbox('S1.author.json'), store_factory=factory)
    assert isinstance(applied, apply_return.Applied)
    assert applied.returns_file.exists()
    assert applied.digest == S1_AUTHOR_DIGEST
    refused = apply_return.apply(env.put_inbox('S1.author.json'), store_factory=factory)
    assert isinstance(refused, apply_return.Refused)
    assert refused.kind is apply_return.RefusalKind.APPLICATION


def _rulings(env: ProgramEnv) -> list[dict[str, Any]]:
    return [dict(ruling) for ruling in state.read_rulings(state.Home(env.home))]


def _pending(env: ProgramEnv) -> list[str]:
    questions = load_program_module('questions')
    home = state.Home(env.home)
    recorded = questions.recorded_questions(state.read_ledger(home))
    return [str(ref) for ref in questions.pending_items(recorded, state.read_rulings(home))]


def _amended_author(env: ProgramEnv) -> subprocess.CompletedProcess[str]:
    return env.run('apply_return.py', env.put_inbox('S1.author.amend.json'), '--amend')


def _ruled(env: ProgramEnv, by: str, *args: str) -> None:
    proc = env.run(
        'record.py', 'rule', '--run', RUN_ID, '--by', by, '--question', 'q?', '--ruling', 'r',
        '--source', 'sitting 1', '--option', 'a', '--option', 'b', '--consequence', 'c', *args,
    )
    assert proc.returncode == 0, _output(proc)


def test_b16_an_investigate_return_applies_as_one_minted_ruling(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    assert _amended_author(env).returncode == 0
    _ruled(env, 'human', '--answers', 'S1:Q1')
    _ruled(env, 'lead', '--answers', 'S1:Q3')
    _ruled(env, 'human', '--answers', 'S1:Q2', '--reask')
    ledger_before = _ledger(env)
    proc = env.run('apply_return.py', env.put_inbox('S1.investigate-Q2.json'))
    assert proc.returncode == 0, _output(proc)
    assert (env.home / 'returns' / 'S1.investigate-Q2.json').exists()
    assert _inbox(env) == []
    assert _ledger(env) == ledger_before
    *_, minted = _rulings(env)
    assert len(_rulings(env)) == 4
    source = minted.pop('source')
    assert 'returns/S1.investigate-Q2.json' in source and 'R1' in source
    assert not minted.pop('reask', False)
    minted.pop('date')
    assert minted == {
        'id': 'D4', 'ruled_by': 'human', 'stream': 'S1', 'item': 'Q2',
        'question': 'Should the alpha shim be retired before S2 releases C1?',
        'ruling': 'hold until S2 releases C1',
        'options': ('retire now', 'hold until S2 releases C1'),
        'consequence': '103 stays held on S2 C1', 'keys': (), 'supersedes': None,
        'answers': ('S1:Q2',),
    }
    assert _pending(env) == []
    digest = proc.stdout.splitlines()[0]
    assert 'D4' in digest and 'S1:Q2' in digest


def test_an_investigate_return_on_an_unrecorded_item_is_a_contradiction(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = _inbox_variant(
        env, 'S1.investigate-Q2.json', lambda data: data.update(item='Q9'),
        as_stem='S1.investigate-Q9',
    )
    before = env.snapshot()
    proc = env.run('apply_return.py', inbox_file)
    assert proc.returncode == 1
    (line,) = [line for line in _contradictions(_output(proc)) if line.startswith('contradiction: item:')]
    assert 'S1:Q9' in line
    assert env.snapshot() == before


def _without_options(data: dict[str, Any]) -> None:
    data['item'] = 'Q1'
    data['rulings_in_session'][0]['options'] = []


def test_a_session_ruling_the_rulings_file_would_refuse_is_a_form_rejection(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = _inbox_variant(
        env, 'S1.investigate-Q2.json', _without_options, as_stem='S1.investigate-Q1',
    )
    before = env.snapshot()
    proc = env.run('apply_return.py', inbox_file)
    assert proc.returncode == 1
    (line,) = [line for line in _form_lines(_output(proc)) if 'rulings_in_session[0]' in line]
    assert 'options' in line
    assert env.snapshot() == before


def test_b16_amend_appends_only_the_delta_and_keeps_the_first_file(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    unamended = env.run('apply_return.py', env.put_inbox('S1.author.amend.json'))
    assert unamended.returncode == 1
    assert '--amend' in _output(unamended)
    proc = _amended_author(env)
    assert proc.returncode == 0, _output(proc)
    returns = env.home / 'returns'
    assert (returns / 'S1.author.json').read_bytes() == (FIXTURE_RETURNS / 'S1.author.json').read_bytes()
    assert (returns / 'S1.author.amend-1.json').read_bytes() == (
        FIXTURE_RETURNS / 'S1.author.amend.json').read_bytes()
    assert _inbox(env) == []
    amendment = {'by': 'apply_return.py', 'return': 'returns/S1.author.amend-1.json'}
    assert _ledger(env)[4:] == [
        {'seq': 5, 'op': 'stream_status', 'stream': 'S1', 'status': 'authored',
         'prd_path': S1_PRD, **amendment},
        {'seq': 6, 'op': 'question', 'stream': 'S1', 'id': 'Q2',
         'question': 'Should the alpha shim be retired before S2 releases C1?',
         'options': ('retire now', 'hold until S2 releases C1'),
         'recommendation': 'hold until S2 releases C1', 'needs_human': True,
         'lead_rulable': False, 'source': 'critic', **amendment},
    ]
    assert [line for line in proc.stdout.splitlines() if 'needs_human' in line and 'S1:Q2' in line]
    questions = load_program_module('questions')
    recorded = questions.recorded_questions(state.read_ledger(state.Home(env.home)))
    assert sorted(str(ref) for ref in recorded) == ['S1:Q1', 'S1:Q2', 'S1:Q3']


def test_an_amendment_may_not_move_the_streams_prd(env: ProgramEnv) -> None:
    _applied(env, 'S1.author.json')
    inbox_file = _inbox_variant(
        env, 'S1.author.amend.json', lambda data: data.update(prd_path='plans/elsewhere-prd.md'),
    )
    before = env.snapshot()
    proc = env.run('apply_return.py', inbox_file, '--amend')
    assert proc.returncode == 1
    assert any(line.startswith('contradiction: prd_path:') for line in _output(proc).splitlines())
    assert env.snapshot() == before


def _with_session_ruling(data: dict[str, Any]) -> None:
    data['rulings_in_session'] = [{
        'id': 'R7', 'question': 'Does S2 keep its own retry budget?', 'ruling': 'it does',
        'options': ['it does', 'S1 owns it'], 'consequence': 'S2 keeps budget.py',
    }]


def test_an_author_returns_session_rulings_become_minted_human_rulings(env: ProgramEnv) -> None:
    proc = env.run('apply_return.py', _inbox_variant(env, 'S2.author.json', _with_session_ruling))
    assert proc.returncode == 0, _output(proc)
    (ruling,) = _rulings(env)
    assert (ruling['id'], ruling['ruled_by'], ruling['stream']) == ('D1', 'human', 'S2')
    assert 'R7' in ruling['source']
    assert 'answers' not in ruling


S2_PRD = 'plans/review-all-fixture-s2-prd.md'
LOST = 'open_questions, xrefs, lead_actions, moves'
_TIMESTAMP = re.compile(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ')


def _from_store(env: ProgramEnv, stream: str) -> subprocess.CompletedProcess[str]:
    return env.run('apply_return.py', '--from-store', stream, '--run', RUN_ID)


@pytest.fixture
def authored(env: ProgramEnv) -> ProgramEnv:
    """A home whose S1 and S2 author returns are applied."""
    _applied(env, 'S1.author.json')
    _applied(env, 'S2.author.json')
    return env


def _note(env: ProgramEnv) -> str:
    (note,) = [line['text'] for line in _ledger(env) if line['op'] == 'note']
    return note


def test_from_store_records_the_decompose_the_store_shows(authored: ProgramEnv, store: FakeStore) -> None:
    before = authored.snapshot()
    calls_before = len(_tools_calls(store))
    proc = _from_store(authored, 'S2')
    assert proc.returncode == 0, _output(proc)
    status, note = _ledger(authored)[-2:]
    assert status == {'seq': status['seq'], 'by': 'apply_return.py', 'op': 'stream_status',
                      'stream': 'S2', 'status': 'decomposed', 'prd_path': S2_PRD}
    assert note['op'] == 'note' and note['seq'] == status['seq'] + 1
    text = note['text']
    assert 'filed 105, 106' in text and 'held none' in text and 'released 113' in text
    assert LOST in text
    assert _TIMESTAMP.search(text)
    after = authored.snapshot()
    assert sorted(name for name in after if name.startswith('returns/')) == sorted(
        name for name in before if name.startswith('returns/'))
    assert after.get('rulings.jsonl') == before.get('rulings.jsonl')
    assert proc.stdout.splitlines()[0] == (
        f'S2 decompose from store: 2 filed, 0 held, 1 released; lost: {LOST}'
    )
    called = set(_tools_calls(store)[calls_before:])
    assert 'find_tasks_by_metadata' in called
    assert called <= {'find_tasks_by_metadata', 'get_task'}


def test_from_store_refuses_a_stream_already_decomposed(authored: ProgramEnv) -> None:
    assert _from_store(authored, 'S2').returncode == 0
    before = authored.snapshot()
    proc = _from_store(authored, 'S2')
    assert proc.returncode == 1
    assert 'already decomposed' in _output(proc)
    assert authored.snapshot() == before


def test_from_store_refuses_a_stream_whose_author_return_is_not_applied(env: ProgramEnv) -> None:
    before = env.snapshot()
    proc = _from_store(env, 'S1')
    assert proc.returncode == 1
    assert 'author return is not applied' in _output(proc)
    assert env.snapshot() == before


def test_from_store_refuses_while_the_inbox_holds_a_usable_return(authored: ProgramEnv) -> None:
    authored.put_inbox('S2.decompose.json')
    before = authored.snapshot()
    proc = _from_store(authored, 'S2')
    assert proc.returncode == 1
    assert 'usable return' in _output(proc)
    assert authored.snapshot() == before


@pytest.mark.parametrize(
    ('content', 'why'),
    [(json.dumps({'stream': 'S2', 'pass': 'decompose'}).ljust(40), 'under 100'),
     ('not json ' * 20, 'does not parse')],
    ids=['under-100-bytes', 'unparsable'],
)
def test_from_store_proceeds_past_a_degenerate_inbox_file_and_names_it(
    content: str, why: str, authored: ProgramEnv,
) -> None:
    leftover = authored.home / 'inbox' / 'S2.decompose.json'
    leftover.write_text(content)
    proc = _from_store(authored, 'S2')
    assert proc.returncode == 0, _output(proc)
    assert leftover.read_text() == content
    for text in (_note(authored), proc.stdout):
        assert 'inbox/S2.decompose.json' in text and why in text


def test_from_store_reports_a_held_leaf(authored: ProgramEnv, tmp_path: Path) -> None:
    data = json.loads((FIXTURE_DIR / 'store.json').read_text())
    (row,) = [task for task in data['tasks'] if task['id'] == 105]
    row['status'] = 'deferred'
    row['metadata']['x_seam_wait'] = {'stream': 'S1', 'contract': 'C9', 'release_owner': 'S1'}
    held_store = tmp_path / 'store-held.json'
    held_store.write_text(json.dumps(data))
    with running_fake_store(store=held_store) as other:
        proc = _from_store(dataclasses.replace(authored, store_url=other.url), 'S2')
    assert proc.returncode == 0, _output(proc)
    assert proc.stdout.splitlines()[0].startswith('S2 decompose from store: 1 filed, 1 held,')
    assert '105 on S1 C9' in _note(authored)


@pytest.mark.parametrize(
    'args', [('--from-store', 'S2'), ('--from-store', 'S2', '--run', RUN_ID, 'inbox/S2.decompose.json')],
    ids=['without-run', 'with-a-file'],
)
def test_from_store_usage_errors(args: tuple[str, ...], authored: ProgramEnv) -> None:
    assert authored.run('apply_return.py', *args).returncode == 2
