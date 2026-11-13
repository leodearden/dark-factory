"""Tests for scripts/sitting/apply.py — the sitting's store-writing verbs (task 8425).

The apply rows run against two real escalation servers on two ports, one per
fixture project root, each queue holding an ``esc-100-1`` and an ``esc-200-1``,
so a call reaching the wrong server would act on a same-named record.
"""
from __future__ import annotations

import contextlib
import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.authority import COORDINATOR_IDENTITY, EXECUTOR_IDENTITY, ClassSwitch
from escalation.classify import classify_resolver_tier
from escalation.models import Escalation
from orchestrator.session_registry import (
    AuditOutcome,
    AuditVerdict,
    DecisionRecord,
    DecisionState,
    close_decision_with_evidence,
    decision_path_for_id,
    list_decisions,
    normalize_escalations_dir,
    record_audit_outcome,
    reopen_decision,
    write_decision,
)
from shared.mcp_session import SyncMcpSession
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from sitting import apply as mod
from sitting import audits, payloads, prepare_sitting
from sitting import ledger as ledger_mod
from sitting import preparation as prep_mod
from sitting.inventory import AUDIT_KIND, audit_key, decision_key, escalation_key, key_str
from sitting.ports import QueuePort
from sitting_escalation_servers import RecordingToolCaller, ServedQueue, serve_project_queue

from escalation import policy_state

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
        return self.number_of(self.key(queue, esc_id))

    def number_of(self, key: str) -> int:
        loaded = ledger_mod.load(self.ledger)
        assert loaded is not None
        return loaded.entries[key].number

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
            fused_memory=lambda: contextlib.nullcontext(fused if fused is not None else _fused()),
            connect=connect or mod.coordinator_session, fleet_root=self.fleet,
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

        assert report.exit_code == 0, report.json_rows()
        for served, esc_id in ((s.alpha, 'esc-100-1'), (s.beta, 'esc-200-1')):
            record = _record(served, esc_id)
            assert record.status != 'pending'
            assert record.resolved_by == COORDINATOR_IDENTITY
            assert classify_resolver_tier(record.resolved_by) == 'human'
            agreed = payloads.parse_agreed_marker(record.triage_note or '')
            assert isinstance(agreed, payloads.AgreedMarker)
            assert (agreed.answer, agreed.answer_rounds) == ('A', 1)
        alpha_marker = payloads.parse_agreed_marker(_record(s.alpha, 'esc-100-1').triage_note or '')
        assert isinstance(alpha_marker, payloads.AgreedMarker) and alpha_marker.agreed is True
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

        assert report.exit_code == 0, report.json_rows()
        updates = {(args['id'], args['project_root']): args for name, args in fused.calls if name == 'update_task'}
        assert set(updates) == {('100', str(s.roots[0])), ('200', str(s.roots[1]))}
        assert all(payloads.X_RULING_KEY in args['metadata'] for args in updates.values())
        assert fused.names().count('add_memory') == 4
        assert {args['project_id'] for name, args in fused.calls if name == 'add_memory'} == {'alpha', 'beta'}
        for queue, esc_id in ((s.alpha.queue_dir, 'esc-100-1'), (s.beta.queue_dir, 'esc-200-1')):
            decision = _decision_for(s.fleet, queue, esc_id)
            assert decision.state == DecisionState.ANSWERED
            assert decision.closing_evidence

    def _update_task_step(self, s: Sitting, fused: RecordingToolCaller) -> tuple[int, dict]:
        alpha_n = s.number(s.alpha.queue_dir, 'esc-100-1')

        report = s.apply([s.answer(alpha_n, 'A')], {str(alpha_n): 'resume'}, fused=fused)

        (row,) = report.json_rows()
        assert json.loads(json.dumps(row, sort_keys=True)) == row
        (step,) = [step for step in row['steps'] if step['step'] == 'update_task']
        return report.exit_code, step

    def test_a_superseded_string_ruling_is_disclosed_on_its_step(self, sitting):
        prior = 'RULED 2026-09-20: scope A'
        fused = RecordingToolCaller({'get_task': {'id': '100', 'metadata': {payloads.X_RULING_KEY: prior}}})

        exit_code, step = self._update_task_step(sitting, fused)

        assert exit_code == 0
        assert step == {'step': 'update_task', 'outcome': 'ok', 'supersedes': prior}

    def test_a_superseded_structured_ruling_is_disclosed_as_its_json_text(self, sitting):
        prior = {'esc_id': 'esc-100-0', 'action': 'resume', 'text': 'scope A', 'by': 'leo', 'at': '2026-09-20'}
        fused = RecordingToolCaller({'get_task': {'id': '100', 'metadata': {payloads.X_RULING_KEY: prior}}})

        exit_code, step = self._update_task_step(sitting, fused)

        assert exit_code == 0
        assert step['outcome'] == 'ok'
        assert json.loads(step['supersedes']) == prior

    def test_with_no_prior_ruling_the_step_names_nothing_superseded(self, sitting):
        exit_code, step = self._update_task_step(sitting, _fused())

        assert exit_code == 0
        assert step == {'step': 'update_task', 'outcome': 'ok'}

    def test_a_failed_update_discloses_nothing_superseded(self, sitting):
        fused = RecordingToolCaller({'get_task': {'id': '100', 'metadata': {payloads.X_RULING_KEY: 'RULED: A'}}},
                                    fail=frozenset({'update_task'}))

        exit_code, step = self._update_task_step(sitting, fused)

        assert exit_code == mod.EXIT_STEP_FAILED
        assert step['outcome'] == 'failed'
        assert 'supersedes' not in step

    def test_an_unreadable_prior_ruling_fails_the_step_and_writes_no_ruling(self, sitting):
        fused = RecordingToolCaller({'get_task': {'error': 'database is locked'}})

        exit_code, step = self._update_task_step(sitting, fused)

        assert exit_code == mod.EXIT_STEP_FAILED
        assert step['outcome'] == 'failed'
        assert 'database is locked' in step['detail']
        assert 'update_task' not in fused.names()

    def test_one_row_per_item_names_every_step(self, sitting):
        s = sitting
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')

        report = s.apply([s.answer(alpha_n, 'A'), s.answer(beta_n, 'A')],
                         {str(alpha_n): 'resume', str(beta_n): 'resume'})

        rows = {row['number']: row for row in report.json_rows()}
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
        assert saved is not None
        assert ledger_mod.resolution_turns(saved, s.key(s.alpha.queue_dir, 'esc-100-1')) == 1


SENTINEL_TASK = 'dashboard-watchdog-restart-ceiling'
"""The literal of scripts/dashboard-watchdog.py::ESCALATION_TASK_SENTINEL; that file is not importable by name."""


class TestSentinelTaskId:
    def test_an_escalation_naming_a_sentinel_applies_without_touching_a_task(self, sitting, capsys):
        s = sitting
        esc_id = f'esc-{SENTINEL_TASK}-1'
        _pending(s.alpha.queue_dir, esc_id, s.now, task_id=SENTINEL_TASK, agent_role='harness-dashboard-watchdog')
        assert prepare_sitting.main(['brief', *s.stores(), '--ledger', str(s.ledger)]) == 0
        capsys.readouterr()
        n = s.number(s.alpha.queue_dir, esc_id)
        fused = _fused()

        report = s.apply([s.answer(n, 'A')], {str(n): 'close_only'}, fused=fused)

        assert report.exit_code == 0, report.json_rows()
        (row,) = report.json_rows()
        outcomes = [(step['step'], step['outcome']) for step in row['steps']]
        assert outcomes == [('stamp_triage', 'ok'), ('resolve_issue', 'ok'), ('update_task', 'skipped'),
                            ('close_decision', 'ok'), ('add_memory', 'ok'), ('add_memory', 'ok')]
        assert 'sentinel' in row['steps'][2]['detail']
        assert not {'get_task', 'update_task'} & set(fused.names())
        memories = [args for name, args in fused.calls if name == 'add_memory']
        assert len(memories) == 2 and all(args['entities'] == [] for args in memories)
        assert _record(s.alpha, esc_id).resolved_by == COORDINATOR_IDENTITY
        decision = _decision_for(s.fleet, s.alpha.queue_dir, esc_id)
        assert decision.state == DecisionState.ANSWERED
        assert decision.closing_evidence


class TestARegistryStepThatCannotComplete:
    @pytest.mark.parametrize('error', [
        subprocess.TimeoutExpired(['python3'], mod.REGISTRY_TIMEOUT_SECS),
        FileNotFoundError(2, 'No such file or directory', 'python3'),
    ], ids=['timeout', 'no_interpreter'])
    def test_it_fails_its_own_item_and_the_batch_still_reports_and_saves(self, sitting, monkeypatch, error):
        s = sitting
        alpha_n, beta_n = s.number(s.alpha.queue_dir, 'esc-100-1'), s.number(s.beta.queue_dir, 'esc-200-1')
        alpha_queue = normalize_escalations_dir(s.alpha.queue_dir)
        real_run = subprocess.run

        def run(argv, **kwargs):
            if alpha_queue in argv:
                raise error
            return real_run(argv, **kwargs)

        monkeypatch.setattr(mod.subprocess, 'run', run)

        report = s.apply([s.answer(alpha_n, 'A'), s.answer(beta_n, 'A')],
                         {str(alpha_n): 'resume', str(beta_n): 'resume'})

        assert report.exit_code == mod.EXIT_STEP_FAILED
        rows = {row['number']: row for row in report.json_rows()}
        assert [(step['step'], step['outcome']) for step in rows[alpha_n]['steps']] == [
            ('stamp_triage', 'ok'), ('resolve_issue', 'ok'), ('update_task', 'ok'),
            ('close_decision', 'failed'), ('add_memory', 'skipped'), ('add_memory', 'skipped'),
        ]
        assert type(error).__name__ in rows[alpha_n]['steps'][3]['detail']
        assert {step['outcome'] for step in rows[beta_n]['steps']} == {'ok'}
        saved = ledger_mod.load(s.ledger)
        assert saved is not None
        for queue, esc_id in ((s.alpha.queue_dir, 'esc-100-1'), (s.beta.queue_dir, 'esc-200-1')):
            assert ledger_mod.resolution_turns(saved, s.key(queue, esc_id)) == 1


DECISION_STEPS = [('stamp_triage', 'skipped'), ('resolve_issue', 'skipped'), ('update_task', 'skipped'),
                  ('close_decision', 'ok'), ('add_memory', 'ok'), ('add_memory', 'ok')]


class TestDecisionItems:
    """A decision item acts on no escalation, even one its DecisionRecord links to; it closes only its record."""

    def _file(self, s: Sitting, decision_id: str, esc_id: str | None) -> None:
        record = DecisionRecord(
            id=decision_id, project='alpha', text=f'{decision_id}: which scope?',
            filed_at=(s.now - timedelta(days=1)).isoformat(), escalation_id=esc_id,
            escalations_dir=normalize_escalations_dir(s.alpha.queue_dir), options=['scope A', 'scope B'],
        )
        assert write_decision(record, root=s.fleet)

    def _rebrief(self, s: Sitting, capsys) -> None:
        assert prepare_sitting.main(['brief', *s.stores(), '--ledger', str(s.ledger)]) == 0
        capsys.readouterr()

    def _decision(self, s: Sitting, capsys, decision_id: str, esc_id: str | None) -> int:
        self._file(s, decision_id, esc_id)
        self._rebrief(s, capsys)
        return s.number_of(key_str(decision_key(decision_id)))

    def _assert_closed_only(self, s: Sitting, row: dict, decision_id: str) -> None:
        assert [(step['step'], step['outcome']) for step in row['steps']] == DECISION_STEPS
        closed = _decision_of(s.fleet, decision_id)
        assert closed.state == DecisionState.ANSWERED
        assert closed.closing_evidence

    def test_a_plain_decision_answered_beside_an_escalation_closes_its_record(self, sitting, capsys):
        s = sitting
        decision_n = self._decision(s, capsys, 'dec-alpha-scope', None)
        esc_n = s.number(s.alpha.queue_dir, 'esc-100-1')
        fused = _fused()

        report = s.apply([s.answer(decision_n, 'A'), s.answer(esc_n, 'A')], {str(esc_n): 'resume'}, fused=fused)

        assert report.exit_code == 0, report.json_rows()
        rows = {row['number']: row for row in report.json_rows()}
        self._assert_closed_only(s, rows[decision_n], 'dec-alpha-scope')
        assert {step['outcome'] for step in rows[esc_n]['steps']} == {'ok'}
        assert _record(s.alpha, 'esc-100-1').resolved_by == COORDINATOR_IDENTITY

    def test_a_decision_whose_escalation_is_already_resolved_closes_without_touching_it(self, sitting, capsys):
        s = sitting
        resolved = _pending(s.alpha.queue_dir, 'esc-300-1', s.now, status='resolved', resolved_by=EXECUTOR_IDENTITY,
                            resolved_at=s.now.isoformat(), resolution='closed by the executor')
        decision_n = self._decision(s, capsys, 'dec-alpha-300', 'esc-300-1')
        fused = _fused()

        report = s.apply([s.answer(decision_n, 'B', 'hold the scope')], {}, fused=fused)

        assert report.exit_code == 0, report.json_rows()
        (row,) = report.json_rows()
        assert row['escalation_id'] == 'esc-300-1'
        self._assert_closed_only(s, row, 'dec-alpha-300')
        assert _record(s.alpha, 'esc-300-1').to_dict() == resolved.to_dict()
        assert not {'get_task', 'update_task'} & set(fused.names())

    def test_a_second_decision_on_one_escalation_closes_its_record_and_the_escalation_resolves_once(
        self, sitting, capsys,
    ):
        s = sitting
        decision_ids = ('dec-alpha-100-a', 'dec-alpha-100-b')
        for decision_id in decision_ids:
            self._file(s, decision_id, 'esc-100-1')
        self._rebrief(s, capsys)
        loaded = ledger_mod.load(s.ledger)
        assert loaded is not None
        (second,) = [d for d in decision_ids if key_str(decision_key(d)) in loaded.entries]
        decision_n, esc_n = s.number_of(key_str(decision_key(second))), s.number(s.alpha.queue_dir, 'esc-100-1')

        report = s.apply([s.answer(esc_n, 'A'), s.answer(decision_n, 'A')], {str(esc_n): 'resume'})

        assert report.exit_code == 0, report.json_rows()
        rows = {row['number']: row for row in report.json_rows()}
        assert {step['outcome'] for step in rows[esc_n]['steps']} == {'ok'}
        assert rows[esc_n]['decision_id'] in decision_ids and rows[esc_n]['decision_id'] != second
        self._assert_closed_only(s, rows[decision_n], second)
        assert _decision_of(s.fleet, rows[esc_n]['decision_id']).state == DecisionState.ANSWERED
        assert _record(s.alpha, 'esc-100-1').resolved_by == COORDINATOR_IDENTITY

    def test_an_action_given_for_a_decision_item_exits_2_and_writes_nothing(self, sitting, capsys):
        s = sitting
        decision_n = self._decision(s, capsys, 'dec-alpha-plain', None)
        before = _snapshot(s)

        report = s.apply([s.answer(decision_n, 'A')], {str(decision_n): 'resume'})

        assert report.exit_code == mod.EXIT_CONFIG
        assert any('is a decision, not an escalation' in message for message in report.messages)
        assert _snapshot(s) == before


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
        (refused,) = [row for row in report.json_rows() if row.get('refusal')]
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
        (refused,) = [row for row in report.json_rows() if row.get('refusal')]
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
        (refused,) = report.json_rows()
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
        rows = {row['number']: row for row in report.json_rows()}
        outcomes = {step['step']: step for step in rows[alpha_n]['steps']}
        assert outcomes['resolve_issue']['outcome'] == 'refused'
        assert outcomes['resolve_issue']['code'] == 'claimed_by_executor'
        assert [step['outcome'] for step in rows[alpha_n]['steps'][2:]] == ['skipped'] * 4
        refused = _record(s.alpha, 'esc-100-1')
        assert refused.status == 'pending'
        assert {step['outcome'] for step in rows[beta_n]['steps']} == {'ok'}

    def test_the_markers_stamped_before_a_refused_resolve_stay_and_a_rerun_appends_a_newer_pair(self, sitting):
        s = sitting
        alpha_n = s.number(s.alpha.queue_dir, 'esc-100-1')
        alpha_queue = normalize_escalations_dir(s.alpha.queue_dir)

        @contextlib.contextmanager
        def racing(port: QueuePort):
            with mod.coordinator_session(port) as session:
                yield _ClaimBeforeResolve(session, s.alpha, 'esc-100-1') if port.queue_dir == alpha_queue else session

        assert s.apply([s.answer(alpha_n, 'A')], {str(alpha_n): 'resume'}, connect=racing).exit_code == 4

        note = _record(s.alpha, 'esc-100-1').triage_note or ''
        assert isinstance(payloads.parse_prepared_marker(note), payloads.PreparedMarker)
        agreed = payloads.parse_agreed_marker(note)
        assert isinstance(agreed, payloads.AgreedMarker)
        assert (agreed.answer, agreed.agreed) == ('A', True)
        _rewrite(s.alpha, 'esc-100-1', claimed_at=(s.now - timedelta(days=1)).isoformat())

        assert s.apply([s.answer(alpha_n, 'B', 'hold after all')], {str(alpha_n): 'close_only'}).exit_code == 0

        resolved = _record(s.alpha, 'esc-100-1')
        assert resolved.resolved_by == COORDINATOR_IDENTITY
        note = resolved.triage_note or ''
        assert note.count(payloads.AGREED_MARKER) == 2
        latest = payloads.parse_agreed_marker(note)
        assert isinstance(latest, payloads.AgreedMarker)
        assert (latest.answer, latest.agreed) == ('B', False)


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


LANDING = 'landing_provenance'
FOUND_EVIDENCE = 'found_on_main: commit abc1234 is-ancestor of main per merge-base'


def _decision_of(fleet: Path, decision_id: str) -> DecisionRecord:
    return DecisionRecord.from_json(decision_path_for_id(decision_id, root=fleet).read_text())


@dataclass
class AuditBench:
    """One project queue, a fleet, and a nightly ledger numbering executor closes as ``audit N``."""

    root: Path
    fleet: Path
    ledger: Path
    now: datetime

    @property
    def queue(self) -> Path:
        return self.root / 'data' / 'escalations'

    @property
    def queue_dir(self) -> str:
        return normalize_escalations_dir(self.queue)

    def file_close(self, esc_id: str, decision_id: str) -> DecisionRecord:
        opened = DecisionRecord(id=decision_id, project='alpha', text=f'close {esc_id}?',
                                filed_at=(self.now - timedelta(days=1)).isoformat(),
                                escalation_id=esc_id, escalations_dir=self.queue_dir)
        assert write_decision(opened, root=self.fleet)
        closed = close_decision_with_evidence(decision_id, 'answered', FOUND_EVIDENCE, root=self.fleet,
                                              expected_project='alpha', expected_escalations_dir=self.queue_dir)
        assert closed is not None
        return closed

    def executor_close(self, n: int, *, verdict: AuditVerdict | None = None) -> DecisionRecord:
        """An executor close written as the server would leave it, optionally already audited."""
        esc_id = f'esc-{500 + n}-1'
        _pending(self.queue / 'archive' / '2026-09-25', esc_id, self.now, category='provenance_unattributed',
                 status='resolved', resolved_by=EXECUTOR_IDENTITY, policy_class=LANDING,
                 claimed_by=EXECUTOR_IDENTITY, claimed_at=(self.now - timedelta(hours=1)).isoformat())
        record = self.file_close(esc_id, f'dec-exec-{n}')
        if verdict is not None:
            assert record_audit_outcome(record.id, verdict, '', root=self.fleet, expected_project='alpha',
                                        expected_escalations_dir=self.queue_dir,
                                        audited_at=self.now.isoformat(), policy_class=LANDING,
                                        closed_at=record.closed_at) is not None
        return record

    def number(self, key: str, *, kinds: frozenset[str] = frozenset({AUDIT_KIND})) -> int:
        current = ledger_mod.load(self.ledger) or ledger_mod.new_sitting(self.now.isoformat())
        numbered = ledger_mod.assign(current, [key, *(k for k, e in current.entries.items()
                                                      if e.state != 'done' and json.loads(k)[0] in kinds)],
                                     lambda k: 0, self.now.isoformat(), kinds=kinds)
        ledger_mod.save(self.ledger, numbered)
        return numbered.entries[key].number

    def number_audit(self, record: DecisionRecord) -> int:
        return self.number(key_str(audit_key(record.id)))

    def run(self, capsys, *answers: str) -> tuple[int, list[dict], str]:
        tokens = [arg for answer in answers for arg in ('--answer', answer)]
        rc = mod.main(['audit', '--ledger', str(self.ledger), *tokens,
                       '--decisions-root', str(self.fleet), '--now', self.now.isoformat()])
        out, err = capsys.readouterr()
        rows = [json.loads(line) for line in out.splitlines() if line.startswith('{')]
        return rc, rows, out + err

    def state(self) -> ClassSwitch:
        return policy_state.load_class_states(Path(self.queue_dir))[LANDING]

    def go_live(self) -> None:
        policy_state.set_class_state(Path(self.queue_dir), LANDING, ClassSwitch.LIVE)


@pytest.fixture
def bench(tmp_path, monkeypatch) -> AuditBench:
    monkeypatch.delenv(prep_mod.NIGHTLY_CONFINEMENT_ENV, raising=False)
    root = tmp_path / 'src' / 'alpha'
    (root / 'data' / 'escalations').mkdir(parents=True)
    fleet = tmp_path / 'fleet'
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(fleet))
    return AuditBench(root, fleet, tmp_path / 'state' / 'ledger-nightly.json', datetime.now(UTC))


def _executor_session(served: ServedQueue) -> SyncMcpSession:
    return SyncMcpSession(served.base_url, client_name='sitting-test-executor',
                          headers={mod.IDENTITY_HEADER: EXECUTOR_IDENTITY})


def _executor_resolve(served: ServedQueue, esc_id: str) -> dict:
    with _executor_session(served) as executor:
        claimed = executor.call_tool('claim_policy_class', {'escalation_id': esc_id, 'policy_class': LANDING})
        assert 'error' not in claimed, claimed
        return executor.call_tool('resolve_issue', {
            'escalation_id': esc_id, 'resolution': FOUND_EVIDENCE, 'action': 'close_only',
            'resolution_class': 'actionable',
        })


class TestAuditRow8:
    def test_a_wrong_audit_records_the_outcome_and_flips_the_class_to_shadow(self, bench, capsys):
        with serve_project_queue(bench.root, 'alpha') as served:
            bench.go_live()
            for esc_id in ('esc-500-1', 'esc-501-1'):
                _pending(served.queue_dir, esc_id, bench.now, category='provenance_unattributed',
                         agent_role='task-implementer')
            assert 'error' not in _executor_resolve(served, 'esc-500-1')
            record = bench.file_close('esc-500-1', 'dec-exec-0')
            n = bench.number_audit(record)

            rc, rows, text = bench.run(capsys, f'{n}=wrong:missed sibling')

            assert rc == 0, text
            assert _decision_of(bench.fleet, record.id).current_audit == AuditOutcome(
                AuditVerdict.WRONG, 'missed sibling', bench.now.isoformat(), LANDING, closed_at=record.closed_at)
            assert bench.state() == ClassSwitch.SHADOW
            assert _executor_resolve(served, 'esc-501-1')['code'] == 'level_forbidden'
            (row,) = rows
            assert {k: row[k] for k in ('number', 'decision_id', 'policy_class', 'queue_dir',
                                        'class_state_before', 'class_state_after')} == {
                'number': n, 'decision_id': record.id, 'policy_class': LANDING, 'queue_dir': bench.queue_dir,
                'class_state_before': 'live', 'class_state_after': 'shadow',
            }


class TestAuditVerdicts:
    def test_an_ok_audit_leaves_the_class_live(self, bench, capsys):
        bench.go_live()
        record = bench.executor_close(1)
        n = bench.number_audit(record)

        rc, _, text = bench.run(capsys, f'{n}=ok')

        assert rc == 0, text
        outcome = _decision_of(bench.fleet, record.id).current_audit
        assert outcome is not None and outcome.verdict == AuditVerdict.OK
        assert bench.state() == ClassSwitch.LIVE

    @pytest.mark.parametrize(('earlier_wrong', 'expected'), [(0, ClassSwitch.LIVE), (1, ClassSwitch.SHADOW)])
    def test_at_twenty_audits_only_a_rate_above_five_percent_flips(self, bench, capsys, earlier_wrong, expected):
        bench.go_live()
        for n in range(19):
            bench.executor_close(n, verdict=AuditVerdict.WRONG if n < earlier_wrong else AuditVerdict.OK)
        record = bench.executor_close(19)
        number = bench.number_audit(record)

        rc, _, text = bench.run(capsys, f'{number}=wrong')

        assert rc == 0, text
        assert bench.state() == expected

    def test_a_wrong_on_a_shadow_class_writes_no_state(self, bench, capsys):
        record = bench.executor_close(1)
        n = bench.number_audit(record)

        rc, _, text = bench.run(capsys, f'{n}=wrong')

        assert rc == 0, text
        outcome = _decision_of(bench.fleet, record.id).current_audit
        assert outcome is not None and outcome.verdict == AuditVerdict.WRONG
        assert not policy_state.policy_state_path(Path(bench.queue_dir)).exists()


class TestAuditRefusals:
    def _written(self, bench: AuditBench, record: DecisionRecord) -> tuple[bytes, bytes]:
        return bench.ledger.read_bytes(), decision_path_for_id(record.id, root=bench.fleet).read_bytes()

    def test_an_already_audited_close_is_refused(self, bench, capsys):
        bench.go_live()
        record = bench.executor_close(1)
        n = bench.number_audit(record)
        assert record_audit_outcome(record.id, AuditVerdict.OK, '', root=bench.fleet, expected_project='alpha',
                                    expected_escalations_dir=bench.queue_dir,
                                    audited_at=bench.now.isoformat(), policy_class=LANDING,
                                    closed_at=record.closed_at) is not None
        before = self._written(bench, record)

        rc, rows, _ = bench.run(capsys, f'{n}=wrong')

        assert rc == 4
        assert rows[0]['refusal']['code'] == 'already_audited'
        assert self._written(bench, record) == before
        assert bench.state() == ClassSwitch.LIVE

    def test_a_reopened_close_closed_again_is_audited_in_its_own_right(self, bench, capsys):
        bench.go_live()
        record = bench.executor_close(1, verdict=AuditVerdict.OK)
        assert reopen_decision(record.id, bench.fleet, expected_project='alpha',
                               expected_escalations_dir=bench.queue_dir) is not None
        closed_again = close_decision_with_evidence(record.id, 'answered', FOUND_EVIDENCE, root=bench.fleet,
                                                    expected_project='alpha', expected_escalations_dir=bench.queue_dir)
        assert closed_again is not None
        n = bench.number_audit(record)

        rc, rows, text = bench.run(capsys, f'{n}=wrong:the second close missed it too')

        assert rc == 0, text
        (row,) = rows
        assert (row['outcome'], row['verdict']) == ('ok', 'wrong')
        current = _decision_of(bench.fleet, record.id).current_audit
        assert current is not None
        assert (current.verdict, current.closed_at) == (AuditVerdict.WRONG, closed_again.closed_at)

    def test_a_close_that_no_longer_joins_to_an_executor_close_is_refused(self, bench, capsys):
        record = bench.executor_close(1)
        n = bench.number_audit(record)
        archived = bench.queue / 'archive' / '2026-09-25' / 'esc-501-1.json'
        archived.write_text(json.dumps({**json.loads(archived.read_text()), 'resolved_by': COORDINATOR_IDENTITY}))
        before = self._written(bench, record)

        rc, rows, _ = bench.run(capsys, f'{n}=wrong')

        assert rc == 4
        assert rows[0]['refusal']['code'] == 'not_an_executor_close'
        assert self._written(bench, record) == before

    def test_a_close_whose_escalation_aged_out_is_refused_and_writes_nothing(self, bench, capsys):
        bench.go_live()
        record = bench.executor_close(1)
        n = bench.number_audit(record)
        (bench.queue / 'archive' / '2026-09-25' / 'esc-501-1.json').unlink()
        state_file = policy_state.policy_state_path(Path(bench.queue_dir))
        decision_before = decision_path_for_id(record.id, root=bench.fleet).read_bytes()
        state_before = state_file.read_bytes()

        rc, rows, _ = bench.run(capsys, f'{n}=wrong')

        assert rc == 4
        assert rows[0]['refusal'] == {'code': 'escalation_aged_out', 'escalation_id': 'esc-501-1',
                                      'queue_dir': bench.queue_dir}
        assert decision_path_for_id(record.id, root=bench.fleet).read_bytes() == decision_before
        assert state_file.read_bytes() == state_before

    def test_a_question_item_is_asked_back(self, bench, capsys):
        record = bench.executor_close(1)
        bench.number_audit(record)
        question = bench.number(key_str(escalation_key(bench.queue_dir, 'esc-900-1')), kinds=frozenset({'esc'}))
        before = self._written(bench, record)

        rc, _, text = bench.run(capsys, f'{question}=ok')

        assert rc == 3
        assert 'ASK BACK' in text
        assert self._written(bench, record) == before

    def test_an_audit_prefixed_token_is_asked_back_and_the_slot_grammar_records(self, bench, capsys):
        record = bench.executor_close(1)
        n = bench.number_audit(record)
        before = self._written(bench, record)

        rc, _, text = bench.run(capsys, f'audit {n}=ok')

        assert rc == 3
        assert 'ASK BACK' in text
        assert self._written(bench, record) == before
        assert audits.answer_slot(n).startswith(f'{n}=ok|')
        rc, _, text = bench.run(capsys, f'{n}=ok')
        assert rc == 0, text

    def test_a_verdict_outside_ok_and_wrong_is_asked_back(self, bench, capsys):
        record = bench.executor_close(1)
        n = bench.number_audit(record)
        before = self._written(bench, record)

        rc, _, text = bench.run(capsys, f'{n}=maybe')

        assert rc == 3
        assert 'ASK BACK' in text
        assert self._written(bench, record) == before


class TestPolicyStateVerb:
    """The policy-state verb only reads through load_class_states and writes through set_class_state."""

    @pytest.fixture
    def queue(self, tmp_path, monkeypatch) -> Path:
        monkeypatch.delenv(prep_mod.NIGHTLY_CONFINEMENT_ENV, raising=False)
        queue = tmp_path / 'src' / 'alpha' / 'data' / 'escalations'
        queue.mkdir(parents=True)
        return queue

    def _run(self, capsys, *argv: str) -> tuple[int, str, str]:
        try:
            rc = mod.main(['policy-state', *argv])
        except SystemExit as exc:
            assert isinstance(exc.code, int)
            rc = exc.code
        out, err = capsys.readouterr()
        return rc, out, err

    def test_without_a_class_it_prints_every_class_switch(self, queue, capsys):
        rc, out, _ = self._run(capsys, '--queue-dir', str(queue))

        assert rc == 0
        assert json.loads(out) == {name: str(switch) for name, switch in policy_state.load_class_states(queue).items()}

    def test_a_class_walks_off_shadow_live_shadow(self, queue, capsys):
        policy_state.policy_state_path(queue).write_text(json.dumps({'classes': {LANDING: 'off'}}))

        for before, after in (('off', 'shadow'), ('shadow', 'live'), ('live', 'shadow')):
            rc, out, err = self._run(capsys, '--queue-dir', str(queue), '--class', LANDING, '--state', after)

            assert rc == 0, err
            assert json.loads(out) == {'queue_dir': normalize_escalations_dir(queue), 'policy_class': LANDING,
                                       'before': before, 'after': after}
            assert policy_state.load_class_states(queue)[LANDING] == ClassSwitch(after)

    def test_a_move_outside_the_transitions_is_refused_and_writes_nothing(self, queue, capsys):
        policy_state.set_class_state(queue, LANDING, ClassSwitch.LIVE)
        before = policy_state.policy_state_path(queue).read_bytes()

        rc, out, err = self._run(capsys, '--queue-dir', str(queue), '--class', LANDING, '--state', 'off')

        assert rc == 4
        assert 'live -> off' in out + err
        assert policy_state.policy_state_path(queue).read_bytes() == before

    def test_a_move_to_the_current_switch_is_a_no_op(self, queue, capsys):
        policy_state.set_class_state(queue, LANDING, ClassSwitch.LIVE)

        rc, out, _ = self._run(capsys, '--queue-dir', str(queue), '--class', LANDING, '--state', 'live')

        assert rc == 0
        row = json.loads(out)
        assert row['before'] == row['after'] == 'live'

    @pytest.mark.parametrize('argv', [
        ['--class', 'no_such_class', '--state', 'shadow'],
        ['--class', LANDING, '--state', 'paused'],
        ['--class', LANDING],
        ['--state', 'shadow'],
    ])
    def test_an_unknown_class_or_state_or_a_lone_flag_exits_2(self, queue, capsys, argv):
        rc, _, _ = self._run(capsys, '--queue-dir', str(queue), *argv)

        assert rc == 2
        assert not policy_state.policy_state_path(queue).exists()

    @pytest.mark.parametrize('argv', [[], ['--class', LANDING, '--state', 'live']])
    def test_an_unreadable_state_file_exits_2_naming_it(self, queue, capsys, argv):
        path = policy_state.policy_state_path(queue)
        path.write_text('{not json')

        rc, _, err = self._run(capsys, '--queue-dir', str(queue), *argv)

        assert rc == 2
        assert str(path) in err
        assert path.read_text() == '{not json'

    def test_a_queue_that_does_not_exist_exits_2_and_is_not_created(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv(prep_mod.NIGHTLY_CONFINEMENT_ENV, raising=False)
        absent = tmp_path / 'absent' / 'data' / 'escalations'

        rc, _, err = self._run(capsys, '--queue-dir', str(absent), '--class', LANDING, '--state', 'live')

        assert rc == 2
        assert err.strip()
        assert not absent.exists()
