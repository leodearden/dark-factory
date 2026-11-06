"""Tests for scripts/sitting/apply.py — the sitting's store-writing verbs (task 8425).

The apply rows run against two real escalation servers on two ports, one per
fixture project root, each queue holding an ``esc-100-1`` and an ``esc-200-1``,
so a call reaching the wrong server would act on a same-named record.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.authority import COORDINATOR_IDENTITY, EXECUTOR_IDENTITY
from escalation.classify import classify_resolver_tier
from escalation.models import Escalation
from orchestrator.session_registry import DecisionState, list_decisions, normalize_escalations_dir
from sitting import apply as mod
from sitting import ledger as ledger_mod
from sitting import payloads, prepare_sitting
from sitting import preparation as prep_mod
from sitting.inventory import escalation_key, key_str
from sitting.ports import QueuePort
from sitting_escalation_servers import RecordingToolCaller, ServedQueue, serve_project_queue

STEP_NAMES = ['stamp_triage', 'resolve_issue', 'update_task', 'close_decision', 'add_memory', 'add_memory']


def _pending(queue: Path, esc_id: str, now: datetime, **fields) -> Escalation:
    fields.setdefault('task_id', esc_id.split('-')[1])
    fields.setdefault('agent_role', 'escalation-watcher-auto')
    fields.setdefault('severity', 'blocking')
    fields.setdefault('category', 'design_concern')
    fields.setdefault('summary', f'summary of {esc_id}')
    fields.setdefault('level', 2)
    fields.setdefault('options', ['close it', 'hold it'])
    fields.setdefault('timestamp', (now - timedelta(days=2)).isoformat())
    esc = Escalation(id=esc_id, **fields)
    queue.mkdir(parents=True, exist_ok=True)
    (queue / f'{esc_id}.json').write_text(esc.to_json())
    return esc


def _project(tmp: Path, name: str, make_tasks_db) -> Path:
    root = tmp / 'src' / name
    (root / 'data' / 'escalations').mkdir(parents=True)
    (root / '.taskmaster' / 'tasks').mkdir(parents=True)
    make_tasks_db([{'id': 100, 'status': 'blocked'}, {'id': 200, 'status': 'blocked'},
                   {'id': 300, 'status': 'blocked'}], directory=root / '.taskmaster' / 'tasks')
    return root


@dataclass
class Sitting:
    tmp: Path
    alpha: ServedQueue
    beta: ServedQueue
    fleet: Path
    state: Path
    now: datetime
    extra_roots: tuple[Path, ...] = ()

    @property
    def ledger(self) -> Path:
        return self.state / 'ledger.json'

    @property
    def preparation(self) -> Path:
        return self.state / 'preparation.json'

    @property
    def roots(self) -> list[Path]:
        return [self.alpha.queue_dir.parents[1], self.beta.queue_dir.parents[1], *self.extra_roots]

    def stores(self) -> list[str]:
        return [
            *[arg for root in self.roots for arg in ('--project-root', str(root))],
            '--decisions-root', str(self.fleet),
            '--sessions-root', str(self.fleet / 'sessions'),
            '--preparation', str(self.preparation),
            '--now', self.now.isoformat(),
        ]

    def key(self, queue: Path, esc_id: str) -> str:
        return key_str(escalation_key(normalize_escalations_dir(queue), esc_id))

    def number(self, queue: Path, esc_id: str) -> int:
        loaded = ledger_mod.load(self.ledger)
        assert loaded is not None
        return loaded.entries[self.key(queue, esc_id)].number

    def number_brief(self, capsys) -> None:
        assert prepare_sitting.main(['new-sitting', '--ledger', str(self.ledger), '--now', self.now.isoformat()]) == 0
        assert prepare_sitting.main(['brief', *self.stores(), '--ledger', str(self.ledger)]) == 0
        capsys.readouterr()

    def sources(self) -> prepare_sitting.Sources:
        return prepare_sitting.read_sources(
            project_roots=[str(root) for root in self.roots], decisions_root=self.fleet,
            sessions_root=self.fleet / 'sessions', handover_override=None, preparation_path=self.preparation,
            project=None, now=self.now,
        )

    def answer(self, number: int, option: str, note: str = '') -> ledger_mod.Answer:
        return prepare_sitting.parse_answer_token(f'{number}={option}' + (f':{note}' if note else ''),
                                                  'terminal', self.now.isoformat())

    def apply(self, answers, actions, *, fused: RecordingToolCaller | None = None, connect=None) -> mod.ApplyReport:
        return mod.run_apply(
            self.sources(), self.ledger, answers, actions, now=self.now,
            fused_memory=fused if fused is not None else _fused(), connect=connect or mod.coordinator_session,
            fleet_root=self.fleet,
        )


def _fused() -> RecordingToolCaller:
    return RecordingToolCaller({'get_task': {'id': '100', 'metadata': {}}})


@pytest.fixture
def sitting(tmp_path, make_tasks_db, monkeypatch, capsys) -> Iterator[Sitting]:
    now = datetime.now(UTC)
    fleet = tmp_path / 'fleet'
    (fleet / 'sessions').mkdir(parents=True)
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(fleet))
    monkeypatch.delenv(prep_mod.NIGHTLY_CONFINEMENT_ENV, raising=False)
    alpha_root, beta_root = (_project(tmp_path, name, make_tasks_db) for name in ('alpha', 'beta'))
    with contextlib.ExitStack() as stack:
        alpha = stack.enter_context(serve_project_queue(alpha_root, 'alpha'))
        beta = stack.enter_context(serve_project_queue(beta_root, 'beta'))
        state = tmp_path / 'state'
        state.mkdir()
        s = Sitting(tmp_path, alpha, beta, fleet, state, now)
        for queue in (alpha.queue_dir, beta.queue_dir):
            _pending(queue, 'esc-100-1', now)
            _pending(queue, 'esc-200-1', now)
        prep_mod.record(s.preparation, [{
            'item': json.loads(s.key(alpha.queue_dir, 'esc-100-1')),
            'question': 'Close esc-100-1 against the option-A ruling Leo already made?',
            'options': [
                {'label': 'A', 'text': 'close as ruled', 'ramification': 'the task resumes on the ruled scope'},
                {'label': 'B', 'text': 'hold for a re-ruling', 'ramification': 'the task stays blocked'},
            ],
            'recommendation': {'option': 'A', 'evidence_chain': 'the task description opens RULED 2026-09-20'},
            'on_apply': 'esc-100-1 resolves; its task is stamped x_ruling',
            'prepared_at': (now - timedelta(hours=1)).isoformat(),
            'prepared_by': 'nightly-fable',
        }])
        s.number_brief(capsys)
        yield s


def _record(served: ServedQueue, esc_id: str) -> Escalation:
    esc = served.queue.get(esc_id)
    assert esc is not None
    return esc


def _rewrite(served: ServedQueue, esc_id: str, **fields) -> None:
    record = _record(served, esc_id).to_dict()
    record.update(fields)
    (served.queue_dir / f'{esc_id}.json').write_text(json.dumps(record))


def _decision_for(fleet: Path, queue: Path, esc_id: str):
    matches = [record for record in list_decisions(fleet)
               if record.escalation_id == esc_id
               and record.escalations_dir == normalize_escalations_dir(queue)]
    assert len(matches) == 1, matches
    return matches[0]


def _snapshot(s: Sitting) -> dict[str, object]:
    return {
        'ledger': s.ledger.read_bytes(),
        'records': {
            (str(served.queue_dir), esc_id): _record(served, esc_id).to_dict()
            for served in (s.alpha, s.beta) for esc_id in ('esc-100-1', 'esc-200-1')
        },
        'decisions': sorted(p.name for p in (s.fleet / 'decisions').glob('*.json')),
    }


class TestApplyAcrossTwoPorts:
    """Row 6: the coordinator applies answers on two queues, each through the server serving it."""

    def test_each_answer_is_applied_on_its_own_queue(self, sitting):
        s = sitting
        _rewrite(s.alpha, 'esc-100-1', policy_class='landing_provenance', claimed_by=EXECUTOR_IDENTITY,
                 claimed_at=(s.now - timedelta(hours=7)).isoformat())
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')
        fused = _fused()

        report = s.apply([s.answer(alpha_n, 'A', 'as ruled'), s.answer(beta_n, 'A')],
                         {str(alpha_n): 'resume', str(beta_n): 'close_only'}, fused=fused)

        assert report.exit_code == 0, report.rows
        for served, esc_id in ((s.alpha, 'esc-100-1'), (s.beta, 'esc-200-1')):
            record = _record(served, esc_id)
            assert record.status != 'pending'
            assert record.resolved_by == COORDINATOR_IDENTITY
            assert classify_resolver_tier(record.resolved_by) == 'human'
            agreed = payloads.parse_agreed_marker(record.triage_note or '')
            assert isinstance(agreed, payloads.AgreedMarker)
            assert (agreed.answer, agreed.answer_rounds) == ('A', 1)
        assert payloads.parse_agreed_marker(_record(s.alpha, 'esc-100-1').triage_note).agreed is True
        assert isinstance(payloads.parse_prepared_marker(_record(s.alpha, 'esc-100-1').triage_note),
                          payloads.PreparedMarker)
        assert payloads.parse_prepared_marker(_record(s.beta, 'esc-200-1').triage_note) is None
        for served, esc_id in ((s.alpha, 'esc-200-1'), (s.beta, 'esc-100-1')):
            untouched = _record(served, esc_id)
            assert (untouched.status, untouched.triage_note or '') == ('pending', '')

    def test_the_fused_memory_writes_and_decision_closes_land_per_item(self, sitting):
        s = sitting
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')
        fused = _fused()

        report = s.apply([s.answer(alpha_n, 'A'), s.answer(beta_n, 'B', 'hold it')],
                         {str(alpha_n): 'resume', str(beta_n): 'close_only'}, fused=fused)

        assert report.exit_code == 0, report.rows
        updates = {(args['id'], args['project_root']): args for name, args in fused.calls if name == 'update_task'}
        assert set(updates) == {('100', str(s.roots[0])), ('200', str(s.roots[1]))}
        assert all(payloads.X_RULING_KEY in args['metadata'] for args in updates.values())
        assert fused.names().count('add_memory') == 4
        assert {args['project_id'] for name, args in fused.calls if name == 'add_memory'} == {'alpha', 'beta'}
        for queue, esc_id in ((s.alpha.queue_dir, 'esc-100-1'), (s.beta.queue_dir, 'esc-200-1')):
            decision = _decision_for(s.fleet, queue, esc_id)
            assert decision.state == DecisionState.ANSWERED
            assert decision.closing_evidence

    def test_one_row_per_item_names_every_step(self, sitting):
        s = sitting
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')

        report = s.apply([s.answer(alpha_n, 'A'), s.answer(beta_n, 'A')],
                         {str(alpha_n): 'resume', str(beta_n): 'resume'})

        rows = {row['number']: row for row in report.rows}
        assert set(rows) == {alpha_n, beta_n}
        assert (rows[alpha_n]['escalation_id'], rows[alpha_n]['queue_dir']) == (
            'esc-100-1', normalize_escalations_dir(s.alpha.queue_dir))
        assert (rows[beta_n]['escalation_id'], rows[beta_n]['queue_dir']) == (
            'esc-200-1', normalize_escalations_dir(s.beta.queue_dir))
        for row in rows.values():
            assert [step['step'] for step in row['steps']] == STEP_NAMES
            assert {step['outcome'] for step in row['steps']} == {'ok'}
            assert json.loads(json.dumps(row, sort_keys=True)) == row

    def test_the_ledger_is_saved_with_the_counted_turn(self, sitting):
        s = sitting
        alpha_n = s.number(s.alpha.queue_dir, 'esc-100-1')

        assert s.apply([s.answer(alpha_n, 'A')], {str(alpha_n): 'resume'}).exit_code == 0

        saved = ledger_mod.load(s.ledger)
        assert ledger_mod.resolution_turns(saved, s.key(s.alpha.queue_dir, 'esc-100-1')) == 1


class TestPreflightRefusals:
    """Rows 4 and 5: a refused item stops the whole batch before anything is written."""

    def _batch(self, s: Sitting):
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')
        answers = [s.answer(alpha_n, 'A'), s.answer(beta_n, 'A')]
        return alpha_n, answers, {str(alpha_n): 'resume', str(beta_n): 'resume'}

    def test_a_record_no_longer_pending_is_refused_and_nothing_is_written(self, sitting):
        s = sitting
        alpha_n, answers, actions = self._batch(s)
        _rewrite(s.alpha, 'esc-100-1', status='dismissed', resolved_by=EXECUTOR_IDENTITY,
                 resolved_at=(s.now - timedelta(minutes=1)).isoformat(), resolution='closed by the executor')
        before = _snapshot(s)
        fused = _fused()

        report = s.apply(answers, actions, fused=fused)

        assert report.exit_code == 4
        (refused,) = [row for row in report.rows if row.get('refusal')]
        assert refused['number'] == alpha_n
        assert refused['refusal']['code'] == 'not_pending'
        assert refused['refusal']['status'] == 'dismissed'
        assert _snapshot(s) == before
        assert fused.calls == []

    def test_a_fresh_executor_claim_is_refused_and_nothing_is_written(self, sitting):
        s = sitting
        alpha_n, answers, actions = self._batch(s)
        claimed_at = (s.now - timedelta(minutes=10)).isoformat()
        _rewrite(s.alpha, 'esc-100-1', policy_class='landing_provenance', claimed_by=EXECUTOR_IDENTITY,
                 claimed_at=claimed_at)
        before = _snapshot(s)
        fused = _fused()

        report = s.apply(answers, actions, fused=fused)

        assert report.exit_code == 4
        (refused,) = [row for row in report.rows if row.get('refusal')]
        assert refused['number'] == alpha_n
        assert refused['refusal'] == {
            'code': 'claimed_by_executor', 'policy_class': 'landing_provenance', 'claimed_at': claimed_at,
        }
        assert _snapshot(s) == before
        assert fused.calls == []

    def test_a_queue_no_server_serves_is_refused(self, sitting, make_tasks_db):
        s = sitting
        gamma = _project(s.tmp, 'gamma', make_tasks_db)
        _pending(gamma / 'data' / 'escalations', 'esc-300-1', s.now)
        s.extra_roots = (gamma,)
        assert prepare_sitting.main(['brief', *s.stores(), '--ledger', str(s.ledger)]) == 0
        gamma_n = s.number(gamma / 'data' / 'escalations', 'esc-300-1')

        report = s.apply([s.answer(gamma_n, 'A')], {str(gamma_n): 'resume'})

        assert report.exit_code == 4
        (refused,) = report.rows
        assert refused['refusal']['code'] == 'no_port'


class _ClaimBeforeResolve:
    """Lands an executor claim on *esc_id* just before the coordinator's resolve reaches the server."""

    def __init__(self, session, served: ServedQueue, esc_id: str) -> None:
        self._session, self._served, self._esc_id = session, served, esc_id

    def call_tool(self, name, arguments):
        if name == 'resolve_issue' and arguments['escalation_id'] == self._esc_id:
            claimed = self._served.queue.claim_policy_class(
                self._esc_id, policy_class='landing_provenance', claimed_by=EXECUTOR_IDENTITY,
                refuse=lambda esc: None,
            )
            assert claimed.record is not None
        return self._session.call_tool(name, arguments)


class TestServerSideRefusal:
    def test_a_claim_landing_after_preflight_is_recorded_as_refused(self, sitting):
        s = sitting
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')
        alpha_queue = normalize_escalations_dir(s.alpha.queue_dir)

        @contextlib.contextmanager
        def racing(port: QueuePort):
            with mod.coordinator_session(port) as session:
                yield _ClaimBeforeResolve(session, s.alpha, 'esc-100-1') if port.queue_dir == alpha_queue else session

        report = s.apply([s.answer(alpha_n, 'A'), s.answer(beta_n, 'A')],
                         {str(alpha_n): 'resume', str(beta_n): 'resume'}, connect=racing)

        assert report.exit_code == 4
        rows = {row['number']: row for row in report.rows}
        outcomes = {step['step']: step for step in rows[alpha_n]['steps']}
        assert outcomes['resolve_issue']['outcome'] == 'refused'
        assert outcomes['resolve_issue']['code'] == 'claimed_by_executor'
        assert [step['outcome'] for step in rows[alpha_n]['steps'][2:]] == ['skipped'] * 4
        assert _record(s.alpha, 'esc-100-1').status == 'pending'
        assert {step['outcome'] for step in rows[beta_n]['steps']} == {'ok'}


class TestCli:
    def _run(self, capsys, *argv: str) -> tuple[int, str, str]:
        rc = mod.main(list(argv))
        out, err = capsys.readouterr()
        return rc, out, err

    def _apply_args(self, s: Sitting, *tokens: str) -> list[str]:
        return ['apply', *s.stores(), '--ledger', str(s.ledger), *tokens]

    @pytest.mark.parametrize('case', ['no_action', 'park', 'unanswered_action'])
    def test_invalid_actions_exit_2_and_write_nothing(self, sitting, capsys, case):
        s = sitting
        n = s.number(s.alpha.queue_dir, 'esc-100-1')
        other = s.number(s.beta.queue_dir, 'esc-200-1')
        tokens = {
            'no_action': ['--answer', f'{n}=A'],
            'park': ['--answer', f'{n}=A', '--action', f'{n}=park'],
            'unanswered_action': ['--answer', f'{n}=A', '--action', f'{n}=resume', '--action', f'{other}=resume'],
        }[case]
        before = _snapshot(s)

        rc, _, err = self._run(capsys, *self._apply_args(s, *tokens))

        assert rc == 2
        assert err.strip()
        assert _snapshot(s) == before

    def test_a_token_naming_no_item_asks_back_and_applies_nothing(self, sitting, capsys):
        s = sitting
        before = _snapshot(s)['records']

        rc, out, _ = self._run(capsys, *self._apply_args(s, '--answer', '999=A', '--action', '999=resume'))

        assert rc == 3
        assert 'ASK BACK' in out
        assert _snapshot(s)['records'] == before

    def test_a_refusal_prints_one_json_row_per_refused_item(self, sitting, capsys):
        s = sitting
        n = s.number(s.alpha.queue_dir, 'esc-100-1')
        _rewrite(s.alpha, 'esc-100-1', status='dismissed', resolved_by=EXECUTOR_IDENTITY,
                 resolved_at=s.now.isoformat(), resolution='closed by the executor')

        rc, out, _ = self._run(capsys, *self._apply_args(s, '--answer', f'{n}=A', '--action', f'{n}=resume'))

        assert rc == 4
        (row,) = [json.loads(line) for line in out.splitlines() if line.strip()]
        assert (row['number'], row['refusal']['code']) == (n, 'not_pending')

    @pytest.mark.parametrize('verb', ['apply', 'audit', 'policy-state'])
    def test_every_verb_is_refused_under_nightly_confinement(self, tmp_path, monkeypatch, capsys, verb):
        monkeypatch.setenv(prep_mod.NIGHTLY_CONFINEMENT_ENV, str(tmp_path))

        rc, _, err = self._run(capsys, verb, '--help')

        assert rc == 2
        assert prep_mod.NIGHTLY_CONFINEMENT_ENV in err
