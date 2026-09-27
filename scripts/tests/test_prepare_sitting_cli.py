"""Tests for scripts/sitting/prepare_sitting.py — the prepare-sitting composer and its read-only guarantee (task 5376)."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
from escalation.models import Escalation
from orchestrator.session_registry import (
    DecisionRecord,
    SessionRecord,
    Status,
    normalize_escalations_dir,
    normalize_project_token,
)
from sitting import ledger as ledger_mod
from sitting import preparation as prep_mod
from sitting import prepare_sitting as mod
from sitting.inventory import escalation_key, key_str
from sitting.preparation import Manual, Standing

NOW = '2026-09-26T09:00:00+00:00'
LATER = '2026-09-26T10:00:00+00:00'
BUCKETS = ('live', 'standing', 'closeable', 'report_only')
JSON_KEYS = {
    'generated_at', 'queues_scanned', 'shortfalls', 'live', 'standing', 'closeable', 'report_only', 'done',
    'docket_recommended',
}
EXPECTED_NUMBERS = {
    'esc-100-1': 1, 'esc-400-1': 2, 'esc-500-1': 3, 'esc-200-1': 4, 'esc-300-1': 5, 'dec-600': 6, 'esc-700-1': 7,
}
GATE_NAMES = ('ruling_is_leos_own', 'ruling_names_this_record', 'ruling_was_executed', 'session_terminated',
              'record_is_not_a_pin', 'sideways_check_ran')


@dataclass
class Env:
    tmp: Path
    df: Path
    fleet: Path
    sessions: Path
    handover: Path
    state: Path

    @property
    def queue(self) -> Path:
        return self.df / 'data' / 'escalations'

    @property
    def preparation(self) -> Path:
        return self.state / 'preparation.json'

    @property
    def ledger(self) -> Path:
        return self.state / 'ledger.json'

    def key(self, esc_id: str) -> str:
        return key_str(escalation_key(normalize_escalations_dir(self.queue), esc_id))

    def args(self, *, project_roots=None, decisions_root=None, sessions_root=None) -> list[str]:
        roots = project_roots if project_roots is not None else [self.df]
        argv = [arg for root in roots for arg in ('--project-root', str(root))]
        return [
            *argv,
            '--decisions-root', str(decisions_root or self.fleet),
            '--sessions-root', str(sessions_root or self.sessions),
            '--handover', str(self.handover),
            '--preparation', str(self.preparation),
            '--now', NOW,
        ]


def _esc(queue: Path, *, subdir: str = '', **fields) -> Escalation:
    fields.setdefault('task_id', fields['id'].split('-')[1])
    fields.setdefault('agent_role', 'escalation-watcher-auto')
    fields.setdefault('severity', 'blocking')
    fields.setdefault('category', 'design_concern')
    fields.setdefault('summary', f"summary of {fields['id']}")
    fields.setdefault('level', 2)
    fields.setdefault('options', ['close it', 'hold it'])
    esc = Escalation(**fields)
    target = queue / subdir if subdir else queue
    target.mkdir(parents=True, exist_ok=True)
    (target / f'{esc.id}.json').write_text(esc.to_json())
    return esc


def _options() -> list[dict]:
    return [
        {'label': 'A', 'text': 'close as ruled', 'ramification': 'the task resumes on the ruled scope'},
        {'label': 'B', 'text': 'hold for a re-ruling', 'ramification': 'the task stays blocked'},
    ]


def _prep_payload(env: Env, esc_id: str, **overrides) -> dict:
    payload = {
        'item': json.loads(env.key(esc_id)),
        'question': f'Close {esc_id} against the option-A ruling Leo already made?',
        'options': _options(),
        'recommendation': {'option': 'A', 'evidence_chain': 'the task description opens RULED 2026-09-20'},
        'on_apply': f'{esc_id} resolves; its task is stamped x_ruling',
        'prepared_at': '2026-09-26T05:40:00+00:00',
        'prepared_by': 'nightly-fable',
    }
    return {**payload, **overrides}


def _held(evidence: str, source_kind: str = '') -> dict:
    return {'held': True, 'evidence': evidence, 'source_kind': source_kind}


@pytest.fixture
def env(tmp_path, make_tasks_db) -> Env:
    df = tmp_path / 'src' / 'dark-factory'
    e = Env(tmp_path, df, tmp_path / 'fleet', tmp_path / 'fleet' / 'sessions', tmp_path / 'handover.md',
            tmp_path / 'state')
    e.queue.mkdir(parents=True)
    (df / '.taskmaster' / 'tasks').mkdir(parents=True)
    e.sessions.mkdir(parents=True)
    e.state.mkdir()
    e.handover.write_text('# handover\n\nnothing relevant here\n')
    make_tasks_db([
        {'id': 100, 'status': 'blocked'},
        {'id': 200, 'status': 'deferred', 'metadata': {'x_coalesced_into': 201}},
        {'id': 201, 'status': 'pending', 'title': 'the carrier the fold landed on'},
        {'id': 300, 'status': 'blocked'},
        {'id': 400, 'status': 'blocked'},
        {'id': 500, 'status': 'blocked'},
        {'id': 700, 'status': 'blocked'},
    ], directory=df / '.taskmaster' / 'tasks')

    _esc(e.queue, id='esc-100-1', severity='critical', timestamp='2026-09-20T09:00:00+00:00')
    _esc(e.queue, id='esc-200-1', timestamp='2026-09-21T09:00:00+00:00')
    _esc(e.queue, id='esc-300-1', timestamp='2026-09-22T09:00:00+00:00', pin_declared_by=['leo'])
    _esc(e.queue, id='esc-400-1', timestamp='2026-09-10T09:00:00+00:00', members=['esc-400-0'],
         root_cause='design-concern:400:retarget')
    _esc(e.queue, subdir='archive/2026-09-11', id='esc-400-0', level=1, status='resolved',
         resolution='Leo ruled option A on esc-400-1: retarget task 400')
    _esc(e.queue, id='esc-500-1', timestamp='2026-09-15T09:00:00+00:00')
    _esc(e.queue, id='esc-700-1', severity='info', timestamp='2026-09-23T09:00:00+00:00')

    decisions = e.fleet / 'decisions'
    decisions.mkdir(parents=True)
    record = DecisionRecord(id='dec-600', project='dark_factory', text='keep the nightly cap?',
                            filed_at='2026-09-18T09:00:00+00:00', options=['keep', 'lift'])
    (decisions / 'dec-600.json').write_text(record.to_json())
    unrelated = SessionRecord(session_slug='session-x-1', status=Status.EXITED, start_ts='2026-09-21T00:00:00+00:00')
    (e.sessions / 'session-x-1').mkdir()
    (e.sessions / 'session-x-1' / 'record.json').write_text(unrelated.to_json())

    prep_mod.record(e.preparation, [
        _prep_payload(e, 'esc-400-1', gate_facts={
            'ruling': _held('Leo 2026-09-20: esc-400-1 option A', 'task_description'),
            'executed': _held('commit abc123 retargeted task 400'),
            'session_terminated': _held('session unblock-df-400-1 ended 2026-09-20'),
            'pins_recovery': [],
        }),
        _prep_payload(e, 'esc-500-1', gate_facts={
            'ruling': _held('Leo 2026-09-14: esc-500-1 option A', 'commit_message'),
            'pins_recovery': [],
        }),
        _prep_payload(e, 'esc-700-1', options=[], on_apply='', recommendation={'no_lean': 'held by Leo'}, standing={
            'kind': 'hold', 'owner': 'Leo',
            'release': {'manual': 'Leo lifts the hold'}, 'evidence': 'Leo HOLD 2026-09-24 in the sitting',
        }),
    ])
    return e


def _run(capsys, *argv: str) -> tuple[int, str, str]:
    rc = mod.main(list(argv))
    out, err = capsys.readouterr()
    return rc, out, err


def _classify(capsys, env: Env, *extra: str, **arg_overrides) -> dict:
    rc, out, _ = _run(capsys, 'brief', *env.args(**arg_overrides), '--json', *extra)
    assert rc == 0
    return json.loads(out)


def _record_id(row: dict) -> str:
    return row['escalation_id'] or row['decision_id']


def _rows(data: dict, bucket: str) -> dict[str, dict]:
    return {_record_id(row): row for row in data[bucket]}


def _numbers(data: dict) -> dict[str, int]:
    return {_record_id(row): row['number'] for bucket in (*BUCKETS, 'done') for row in data[bucket]}


def _saved_ledger(path: Path) -> ledger_mod.Ledger:
    saved = ledger_mod.load(path)
    assert saved is not None, f'no ledger at {path}'
    return saved


def _resolve_in_fixture(env: Env, esc_id: str) -> None:
    path = env.queue / f'{esc_id}.json'
    esc = Escalation.from_json(path.read_text())
    esc.status = 'resolved'
    esc.resolution = 'Leo answered B'
    path.write_text(esc.to_json())


class TestBuckets:
    def test_every_open_item_lands_in_exactly_one_bucket(self, env, capsys):
        data = _classify(capsys, env)

        placed = [_record_id(row) for bucket in BUCKETS for row in data[bucket]]
        assert sorted(placed) == sorted(EXPECTED_NUMBERS)
        assert {bucket: set(_rows(data, bucket)) for bucket in BUCKETS} == {
            'live': {'esc-100-1', 'dec-600'},
            'standing': {'esc-200-1', 'esc-300-1', 'esc-700-1'},
            'closeable': set(),
            'report_only': {'esc-400-1', 'esc-500-1'},
        }

    def test_standing_names_where_the_classification_came_from(self, env, capsys):
        standing = _rows(_classify(capsys, env), 'standing')

        assert {esc_id: row['source'] for esc_id, row in standing.items()} == {
            'esc-200-1': 'ownership', 'esc-300-1': 'pin', 'esc-700-1': 'preparation',
        }
        assert standing['esc-200-1']['standing']['kind'] == 'owned'
        assert 'task 201' in standing['esc-200-1']['standing']['owner']
        assert standing['esc-200-1']['release']['observed'] == 'pending'
        assert standing['esc-200-1']['release']['released'] is False
        assert standing['esc-700-1']['release'] is None

    def test_a_ledger_standing_entry_routes_to_standing(self, env, capsys):
        assert _run(capsys, 'brief', *env.args(), '--ledger', str(env.ledger))[0] == 0
        held = ledger_mod.set_standing(
            _saved_ledger(env.ledger), env.key('esc-100-1'),
            Standing('hold', 'Leo', Manual('Leo lifts it'), 'Leo HOLD 2026-09-26 in the terminal'),
        )
        ledger_mod.save(env.ledger, held)

        standing = _rows(_classify(capsys, env, '--ledger', str(env.ledger)), 'standing')

        assert standing['esc-100-1']['source'] == 'ledger'
        assert standing['esc-100-1']['number'] == 1

    def test_the_text_brief_numbers_the_questions_and_foots_the_standing(self, env, capsys):
        rc, out, _ = _run(capsys, 'brief', *env.args())

        assert rc == 0
        assert out.startswith('# Sitting brief')
        decisions = out.split('## Decisions needed', 1)[1].split('## Standing / no action', 1)[0]
        assert [int(n) for n in re.findall(r'^### (\d+)\. ', decisions, flags=re.MULTILINE)] == [1, 2, 3, 6]
        footer = out.split('## Standing / no action', 1)[1].split('## Done', 1)[0]
        assert all(f'**{n}.**' in footer for n in (4, 5, 7))


class TestRecommendOnly:
    def test_by_default_a_would_be_close_is_only_recommended(self, env, capsys):
        data = _classify(capsys, env)

        assert data['closeable'] == []
        would_be = _rows(data, 'report_only')['esc-400-1']
        assert would_be['demoted_by'] == 'recommend-only'
        assert would_be['missed_gates'] == []
        assert [gate['name'] for gate in would_be['gates']] == list(GATE_NAMES)
        assert all(gate['held'] for gate in would_be['gates'])

    def test_the_text_shows_recommended_closes_with_their_six_gates(self, env, capsys):
        rc, out, _ = _run(capsys, 'brief', *env.args())

        closes = out.split('## Recommended closes (recommend-only)', 1)[1]
        assert rc == 0
        assert 'esc-400-1' in closes
        assert all(name in closes for name in GATE_NAMES)

    def test_a_missed_carveout_names_the_missed_gates(self, env, capsys):
        missed = _rows(_classify(capsys, env), 'report_only')['esc-500-1']
        rc, out, _ = _run(capsys, 'brief', *env.args())

        assert missed['missed_gates'] == ['ruling_was_executed', 'session_terminated']
        assert missed['demoted_by'] == ''
        assert re.search(r'\*\*3\.\*\*.*esc-500-1.*ruling_was_executed, session_terminated', out)

    def test_apply_closes_with_agent_facts_yields_closeable_items_and_their_payloads(self, env, capsys):
        data = _classify(capsys, env, '--apply-closes')

        closeable = _rows(data, 'closeable')
        assert set(closeable) == {'esc-400-1'}
        assert set(_rows(data, 'report_only')) == {'esc-500-1'}
        resolve, close = closeable['esc-400-1']['payloads']
        assert resolve['tool'] == 'resolve_issue'
        assert resolve['args']['escalation_id'] == 'esc-400-1'
        assert 'Leo 2026-09-20: esc-400-1 option A' in resolve['args']['resolution']
        assert close['tool'] == 'cli:session_registry'
        argv = close['args'][-1]
        assert argv[argv.index('close-decision') + 1:argv.index('close-decision') + 5] == [
            '--id', f'{normalize_project_token(env.df.name)}-esc-400-1', '--state', 'answered',
        ]
        assert argv[argv.index('--project') + 1] == closeable['esc-400-1']['project']
        assert argv[argv.index('--escalations-dir') + 1] == normalize_escalations_dir(env.queue)
        assert 'gate 6 sideways_check_ran: held' in argv[argv.index('--evidence') + 1]

    def test_apply_closes_alone_closes_nothing_without_recorded_agent_facts(self, env, capsys):
        env.preparation.unlink()

        data = _classify(capsys, env, '--apply-closes')

        assert data['closeable'] == []
        assert set(_rows(data, 'report_only')) == {'esc-400-1'}
        assert 'ruling_is_leos_own' in _rows(data, 'report_only')['esc-400-1']['missed_gates']


class TestCloseCollision:
    """The reviewer's regression, EXECUTED: a close payload never lands on another project's record at a shared id."""

    def _seed_other_project(self, env: Env, decision_id: str) -> Path:
        other_queue = env.tmp / 'src' / 'know-live' / 'data' / 'escalations'
        record = DecisionRecord(id=decision_id, project='know_live', text="know_live's own unrelated gate",
                                filed_at='2026-09-19T09:00:00+00:00', escalation_id='esc-400-1',
                                escalations_dir=normalize_escalations_dir(other_queue))
        path = env.fleet / 'decisions' / f'{decision_id}.json'
        path.write_text(record.to_json())
        return path

    def _decision_files(self, env: Env) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in (env.fleet / 'decisions').iterdir() if not p.name.endswith('.lock')}

    def _apply_the_close(self, env: Env, capsys) -> list[subprocess.CompletedProcess[str]]:
        """Run the closeable esc-400-1's registry argvs in order, stopping at the first non-zero exit as apply does."""
        data = _classify(capsys, env, '--apply-closes')
        (row,) = [r for r in data['closeable'] if r['escalation_id'] == 'esc-400-1']
        (payload,) = [p for p in row['payloads'] if p['tool'] == 'cli:session_registry']
        fleet_env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(env.fleet)}
        runs: list[subprocess.CompletedProcess[str]] = []
        for argv in payload['args']:
            runs.append(subprocess.run(argv, env=fleet_env, capture_output=True, text=True, timeout=60, check=False))
            if runs[-1].returncode != 0:
                break
        return runs

    def test_a_bare_id_collision_leaves_the_other_record_and_files_under_the_derived_id(self, env, capsys):
        self._seed_other_project(env, 'esc-400-1')
        before = self._decision_files(env)
        project = normalize_project_token(env.df.name)
        derived = f'{project}-esc-400-1'

        runs = self._apply_the_close(env, capsys)

        assert [run.returncode for run in runs] == [0, 0], [run.stderr for run in runs]
        after = self._decision_files(env)
        closed = DecisionRecord.from_json(after.pop(f'{derived}.json'))
        assert after == before
        assert closed.state == 'answered'
        assert closed.closing_evidence
        assert datetime.fromisoformat(closed.closed_at).tzinfo is not None
        assert (closed.project, closed.escalations_dir) == (project, normalize_escalations_dir(env.queue))

    def test_a_forced_derived_id_collision_is_refused_loudly(self, env, capsys):
        other = self._seed_other_project(env, f'{normalize_project_token(env.df.name)}-esc-400-1')
        before = self._decision_files(env)

        write, close = self._apply_the_close(env, capsys)

        assert write.returncode == 0
        assert close.returncode != 0
        assert 'close-decision refused' in close.stderr
        assert 'know_live' in close.stderr
        assert self._decision_files(env) == before
        untouched = DecisionRecord.from_json(other.read_text())
        assert (untouched.state, untouched.closing_evidence, untouched.closed_at) == ('open', '', '')


def _snapshot(*roots: Path) -> dict[Path, tuple[bytes, int]]:
    return {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for root in roots for path in root.rglob('*') if path.is_file()
    }


class TestReadOnly:
    def test_no_subcommand_writes_a_store_of_record(self, env, capsys):
        queueless = env.tmp / 'src' / 'no-queue'
        queueless.mkdir(parents=True)
        roots = [env.df, queueless]
        answers = env.state / 'prep-in.json'
        answers.write_text(json.dumps([_prep_payload(env, 'esc-100-1')]))
        before = _snapshot(env.df, env.fleet)

        for argv in (
            ['brief', *env.args(project_roots=roots)],
            ['brief', *env.args(project_roots=roots), '--json'],
            ['brief', *env.args(project_roots=roots), '--docket-json'],
            ['brief', *env.args(project_roots=roots), '--apply-closes'],
            ['brief', *env.args(project_roots=roots), '--ledger', str(env.ledger)],
            ['record', '--preparation', str(env.preparation), '--from', str(answers)],
            ['resolve-answers', *env.args(project_roots=roots), '--ledger', str(env.ledger), '--answer', '1=A'],
            ['summary', '--ledger', str(env.ledger)],
            ['new-sitting', '--ledger', str(env.ledger), '--now', NOW],
        ):
            assert _run(capsys, *argv)[0] == 0, argv

        assert _snapshot(env.df, env.fleet) == before
        assert not [path for root in (env.df, env.fleet) for path in root.rglob('*.lock')]
        assert not (queueless / 'data').exists()


class TestLedgerNumbering:
    def test_without_a_ledger_numbering_is_the_deterministic_sort_and_nothing_is_written(self, env, capsys):
        state_before = sorted(env.state.iterdir())

        first, second = _classify(capsys, env), _classify(capsys, env)

        assert _numbers(first) == _numbers(second) == EXPECTED_NUMBERS
        assert sorted(env.state.iterdir()) == state_before

    def test_a_ledger_keeps_numbers_and_moves_a_resolved_item_to_done(self, env, capsys):
        before = _classify(capsys, env, '--ledger', str(env.ledger))
        _resolve_in_fixture(env, 'esc-100-1')

        after = _classify(capsys, env, '--ledger', str(env.ledger))
        rc, out, _ = _run(capsys, 'brief', *env.args(), '--ledger', str(env.ledger))

        assert _numbers(before) == EXPECTED_NUMBERS
        assert _numbers(after) == EXPECTED_NUMBERS
        assert [(_record_id(row), row['number']) for row in after['done']] == [('esc-100-1', 1)]
        assert rc == 0
        assert '**1.**' in out.split('## Done', 1)[1]

    def test_new_items_take_the_next_number_not_a_freed_one(self, env, capsys):
        _classify(capsys, env, '--ledger', str(env.ledger))
        _resolve_in_fixture(env, 'esc-100-1')
        _esc(env.queue, id='esc-800-1', severity='urgent', timestamp='2026-09-26T08:00:00+00:00')

        numbers = _numbers(_classify(capsys, env, '--ledger', str(env.ledger)))

        assert numbers['esc-800-1'] == 8
        assert numbers['esc-100-1'] == 1


class TestNewSitting:
    def test_creates_an_empty_sitting_and_resets_an_existing_one(self, env, capsys):
        assert _run(capsys, 'new-sitting', '--ledger', str(env.ledger), '--now', NOW)[0] == 0
        assert _saved_ledger(env.ledger).entries == {}

        _classify(capsys, env, '--ledger', str(env.ledger))
        assert _saved_ledger(env.ledger).entries

        assert _run(capsys, 'new-sitting', '--ledger', str(env.ledger), '--now', LATER)[0] == 0
        reset = _saved_ledger(env.ledger)
        assert reset.entries == {}
        assert reset.started_at == LATER

    def test_a_seed_carries_the_nightly_numbers_forward(self, env, capsys):
        nightly = env.state / 'ledger-nightly.json'
        _classify(capsys, env, '--ledger', str(nightly))
        session = env.state / 'ledger-session.json'

        assert _run(capsys, 'new-sitting', '--ledger', str(session), '--seed', str(nightly), '--now', LATER)[0] == 0
        assert _numbers(_classify(capsys, env, '--ledger', str(session))) == EXPECTED_NUMBERS


class TestRecord:
    def test_record_from_a_file_merges_into_the_store(self, env, capsys):
        source = env.state / 'in.json'
        source.write_text(json.dumps([_prep_payload(env, 'esc-100-1')]))

        rc, _, _ = _run(capsys, 'record', '--preparation', str(env.preparation), '--from', str(source))

        store = prep_mod.load(env.preparation)
        assert rc == 0
        assert {json.loads(key)[-1] for key in store.entries} == {'esc-100-1', 'esc-400-1', 'esc-500-1', 'esc-700-1'}

    def test_record_from_stdin(self, env, capsys, monkeypatch):
        monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps([_prep_payload(env, 'esc-100-1')])))

        rc, _, _ = _run(capsys, 'record', '--preparation', str(env.preparation), '--from', '-')

        assert rc == 0
        assert prep_mod.load(env.preparation).get(escalation_key(normalize_escalations_dir(env.queue), 'esc-100-1'))

    def test_an_invalid_entry_exits_2_with_the_message_and_writes_nothing(self, env, capsys):
        fresh = env.state / 'fresh' / 'preparation.json'
        source = env.state / 'bad.json'
        bad = _prep_payload(env, 'esc-100-1', recommendation={'option': 'Z', 'evidence_chain': 'x'})
        source.write_text(json.dumps([_prep_payload(env, 'esc-500-1'), bad]))

        rc, _, err = _run(capsys, 'record', '--preparation', str(fresh), '--from', str(source))

        assert rc == 2
        assert "'Z'" in err
        assert not fresh.exists()


class TestResolveAnswers:
    def _brief(self, env, capsys):
        assert _run(capsys, 'brief', *env.args(), '--ledger', str(env.ledger))[0] == 0

    def _resolve(self, env, capsys, *answers: str) -> tuple[int, str, str]:
        tokens = [arg for answer in answers for arg in ('--answer', answer)]
        return _run(capsys, 'resolve-answers', *env.args(), '--ledger', str(env.ledger), *tokens)

    def _rounds(self, env, esc_id: str) -> int:
        return _saved_ledger(env.ledger).entries[env.key(esc_id)].answer_rounds

    def test_all_resolved_prints_the_echo_table_and_exits_0(self, env, capsys):
        self._brief(env, capsys)

        rc, out, _ = self._resolve(env, capsys, '1=B', '3=A:keep the cap')

        assert rc == 0
        assert re.search(r'\| 1 \| esc-100-1 \| B \| hold it \|', out)
        assert re.search(r'\| 3 \| esc-500-1 \| A \| close as ruled \| keep the cap \|', out)
        assert 'item 3 resolution_turns=1' in out
        assert (self._rounds(env, 'esc-100-1'), self._rounds(env, 'esc-500-1')) == (1, 1)

    def test_an_unresolved_token_asks_back_with_exit_3_and_still_counts_the_round(self, env, capsys):
        self._brief(env, capsys)

        rc, out, _ = self._resolve(env, capsys, '1=Z', '99=A', 'C:')

        assert rc == 3
        assert 'ASK BACK' in out
        assert "'C:'" in out and '99' in out and "'Z'" in out
        assert self._rounds(env, 'esc-100-1') == 1

    def test_a_standing_item_is_asked_back_not_resolved(self, env, capsys):
        self._brief(env, capsys)

        rc, out, _ = self._resolve(env, capsys, '5=A')

        assert rc == 3
        assert 'item 5 is standing (pin' in out

    def test_a_missing_ledger_is_a_configuration_error(self, env, capsys):
        assert self._resolve(env, capsys, '1=A')[0] == 2


class TestSummary:
    def test_prints_the_sitting_summary(self, env, capsys):
        _run(capsys, 'brief', *env.args(), '--ledger', str(env.ledger))
        _run(capsys, 'resolve-answers', *env.args(), '--ledger', str(env.ledger), '--answer', '1=B')

        rc, out, _ = _run(capsys, 'summary', '--ledger', str(env.ledger))

        summary = json.loads(out)
        assert rc == 0
        assert summary['items_answered'] == 1
        assert summary['answer_rounds'] == 1
        assert summary['items_presented'] == len(EXPECTED_NUMBERS)

    def test_a_missing_ledger_exits_2(self, env, capsys):
        assert _run(capsys, 'summary', '--ledger', str(env.ledger))[0] == 2


class TestProjectScope:
    def test_project_scopes_through_the_token_fold(self, env, capsys):
        reify = env.tmp / 'src' / 'reify'
        _esc(reify / 'data' / 'escalations', id='esc-900-1')
        roots = [env.df, reify]

        everything = _classify(capsys, env, project_roots=roots)
        scoped = _classify(capsys, env, '--project', 'df', project_roots=roots)

        assert 'esc-900-1' in _numbers(everything)
        assert set(_numbers(scoped)) == set(EXPECTED_NUMBERS)


class TestMachineOutputs:
    def test_json_top_level_keys_are_stable(self, env, capsys):
        data = _classify(capsys, env)

        assert set(data) == JSON_KEYS
        assert data['generated_at'] == NOW
        assert data['queues_scanned'] == [normalize_escalations_dir(env.queue)]
        assert data['shortfalls'] == []
        assert data['docket_recommended'] is None

    def test_docket_json_emits_one_row_per_numbered_question(self, env, capsys):
        rc, out, _ = _run(capsys, 'brief', *env.args(), '--docket-json')

        rows = json.loads(out)
        assert rc == 0
        assert [row['item'] for row in rows] == [1, 2, 3, 6]
        assert {row['esc_id'] for row in rows} == {'esc-100-1', 'esc-400-1', 'esc-500-1', 'dec-600'}

    def test_the_brief_states_when_the_docket_page_is_recommended(self, env, capsys):
        _, quiet, _ = _run(capsys, 'brief', *env.args())
        _, loud, _ = _run(capsys, 'brief', *env.args(), '--multi-sitting')
        data = _classify(capsys, env, '--multi-sitting')

        assert 'docket page recommended' not in quiet
        assert 'docket page recommended: the decisions span more than one sitting' in loud
        assert data['docket_recommended'] == 'the decisions span more than one sitting'

    def test_six_numbered_questions_cross_the_docket_threshold(self, env, capsys):
        for n in (1, 2):
            _esc(env.queue, id=f'esc-10{n}-1', timestamp='2026-09-25T09:00:00+00:00')

        data = _classify(capsys, env)

        assert data['docket_recommended'] == '6 decisions (threshold 6)'


class TestFailOpen:
    @pytest.mark.parametrize('broken', ['queue', 'decisions', 'sessions', 'corrupt'])
    def test_each_shortfall_is_stated_and_the_brief_still_renders(self, env, capsys, broken):
        overrides: dict = {}
        if broken == 'queue':
            empty = env.tmp / 'src' / 'empty-project'
            empty.mkdir(parents=True)
            overrides['project_roots'] = [env.df, empty]
            expected = str(empty / 'data' / 'escalations')
        elif broken == 'decisions':
            overrides['decisions_root'] = env.tmp / 'no-fleet'
            expected = str(env.tmp / 'no-fleet')
        elif broken == 'sessions':
            overrides['sessions_root'] = env.tmp / 'no-sessions'
            expected = str(env.tmp / 'no-sessions')
        else:
            (env.queue / 'esc-999-1.json').write_text('{"id": "esc-999-1", "task_')
            expected = 'esc-999-1.json'

        rc, out, _ = _run(capsys, 'brief', *env.args(**overrides))
        data = _classify(capsys, env, **overrides)

        assert rc == 0
        assert out.startswith('# Sitting brief')
        assert expected in out.split('## Shortfalls', 1)[1]
        assert any(expected in shortfall['path'] for shortfall in data['shortfalls'])
