"""Tests for orchestrator.evals.prompt_opt.curator_corpus -- the T5 curator
replay corpus builder (tickets.db -> frontier-adjudicated + human-spot-
checked labeled CuratorReplayItems, split 2:1:7).

See plans/tier1-prompt-optimization-prd.md T5. All tests run against
synthetic fixtures (``_curator_replay_fixtures.py``) or in-memory data --
never the real gitignored ``data/reconciliation/tickets.db`` and never a
real LLM call (the frontier proposer is always dependency-injected).
"""

from __future__ import annotations

from pathlib import Path

from _curator_replay_fixtures import make_synthetic_tickets_db

from orchestrator.evals.prompt_opt.curator_corpus import (
    read_curator_decisions,
    recover_recorded_action,
)


class TestRecoverRecordedAction:
    """recover_recorded_action(status, result_json, task_id) -- unit-level,
    no DB involved. result_json.action is authoritative when present since
    status='combined' is written for BOTH drop and combine."""

    def test_created_status_with_no_result_json_recovers_create(self) -> None:
        recovered = recover_recorded_action('created', None, 'task-1')
        assert recovered == ('create', None, None)

    def test_combined_status_with_combine_action_extracts_target_fingerprint_and_id(self) -> None:
        result_json = '{"id": "task-9", "action": "combine", "target_fingerprint": "Fix the thing"}'
        recovered = recover_recorded_action('combined', result_json, None)
        assert recovered == ('combine', 'Fix the thing', 'task-9')

    def test_combined_status_with_drop_action_recovers_drop_not_combine(self) -> None:
        """The critical disambiguation: status='combined' alone cannot tell
        drop from combine -- only result_json['action'] can."""
        result_json = '{"id": "task-5", "action": "drop"}'
        recovered = recover_recorded_action('combined', result_json, None)
        assert recovered is not None
        action, _fingerprint, target_id = recovered
        assert action == 'drop'
        assert target_id == 'task-5'

    def test_failed_status_with_no_result_json_is_unactionable(self) -> None:
        assert recover_recorded_action('failed', None, None) is None

    def test_pending_status_with_no_result_json_is_unactionable(self) -> None:
        assert recover_recorded_action('pending', None, None) is None

    def test_malformed_result_json_is_unactionable(self) -> None:
        assert recover_recorded_action('combined', 'not valid json{', None) is None

    def test_target_id_falls_back_to_ticket_task_id_when_result_json_omits_id(self) -> None:
        result_json = '{"action": "combine", "target_fingerprint": "Fix the thing"}'
        recovered = recover_recorded_action('combined', result_json, 'task-42')
        assert recovered == ('combine', 'Fix the thing', 'task-42')

    def test_create_action_never_carries_a_target(self) -> None:
        # Even if a task_id is set on the ticket row (the newly-created
        # task's own id), 'create' has no "target being combined into".
        recovered = recover_recorded_action('created', None, 'task-1')
        assert recovered is not None
        _action, fingerprint, target_id = recovered
        assert fingerprint is None
        assert target_id is None


class TestReadCuratorDecisions:
    """read_curator_decisions(db_path) -- read-only stdlib sqlite3 reader."""

    def test_created_row_recovers_create_action(self, tmp_path: Path) -> None:
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {'ticket_id': 't1', 'status': 'created', 'task_id': 'task-1',
             'candidate': {'title': 'Add the widget'}},
        ])
        decisions = read_curator_decisions(db_path)
        assert len(decisions) == 1
        assert decisions[0].ticket_id == 't1'
        assert decisions[0].action == 'create'

    def test_combined_row_with_combine_result_action_recovers_combine(self, tmp_path: Path) -> None:
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {
                'ticket_id': 't2', 'status': 'combined',
                'result': {'id': 'task-9', 'action': 'combine', 'target_fingerprint': 'Fix the thing'},
            },
        ])
        decisions = read_curator_decisions(db_path)
        assert len(decisions) == 1
        d = decisions[0]
        assert d.action == 'combine'
        assert d.target_fingerprint == 'Fix the thing'
        assert d.target_id == 'task-9'

    def test_combined_row_with_drop_result_action_recovers_drop(self, tmp_path: Path) -> None:
        """Proves drop and combine are disambiguated via result_json.action,
        not status (both persist status='combined')."""
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {'ticket_id': 't3', 'status': 'combined', 'result': {'id': 'task-5', 'action': 'drop'}},
        ])
        decisions = read_curator_decisions(db_path)
        assert len(decisions) == 1
        assert decisions[0].action == 'drop'
        assert decisions[0].target_id == 'task-5'

    def test_failed_row_is_skipped(self, tmp_path: Path) -> None:
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {'ticket_id': 't1', 'status': 'created', 'task_id': 'task-1'},
            {'ticket_id': 't4', 'status': 'failed', 'reason': 'llm-failed'},
        ])
        decisions = read_curator_decisions(db_path)
        assert {d.ticket_id for d in decisions} == {'t1'}

    def test_candidate_fields_are_parsed_from_candidate_json(self, tmp_path: Path) -> None:
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {
                'ticket_id': 't1', 'status': 'created', 'task_id': 'task-1',
                'candidate': {
                    'title': 'Add the widget', 'description': 'A widget for the gizmo',
                    'files_to_modify': ['a.py', 'b.py'],
                },
            },
        ])
        decisions = read_curator_decisions(db_path)
        assert decisions[0].candidate['title'] == 'Add the widget'
        assert decisions[0].candidate['description'] == 'A widget for the gizmo'
        assert decisions[0].candidate['files_to_modify'] == ['a.py', 'b.py']

    def test_mixed_batch_recovers_every_actionable_row(self, tmp_path: Path) -> None:
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {'ticket_id': 't1', 'status': 'created', 'task_id': 'task-1'},
            {'ticket_id': 't2', 'status': 'combined',
             'result': {'id': 'task-9', 'action': 'combine', 'target_fingerprint': 'Fix the thing'}},
            {'ticket_id': 't3', 'status': 'combined', 'result': {'id': 'task-5', 'action': 'drop'}},
            {'ticket_id': 't4', 'status': 'failed', 'reason': 'llm-failed'},
        ])
        decisions = read_curator_decisions(db_path)
        by_id = {d.ticket_id: d.action for d in decisions}
        assert by_id == {'t1': 'create', 't2': 'combine', 't3': 'drop'}

    def test_is_read_only(self, tmp_path: Path) -> None:
        """Calling the reader twice yields identical results -- no mutation."""
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', [
            {'ticket_id': 't1', 'status': 'created', 'task_id': 'task-1'},
        ])
        first = read_curator_decisions(db_path)
        second = read_curator_decisions(db_path)
        assert [d.ticket_id for d in first] == [d.ticket_id for d in second]
        assert len(first) == 1
