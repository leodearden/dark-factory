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

import pytest
from _curator_replay_fixtures import make_synthetic_tickets_db

from orchestrator.evals.prompt_opt.curator_corpus import (
    CuratorCorpusManifest,
    CuratorReplayItem,
    FrontierLabel,
    RecordedDecision,
    _parse_frontier_label,
    audit_curator_corpus,
    build_curator_corpus,
    read_curator_decisions,
    recover_recorded_action,
    select_spot_check_subset,
)
from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog
from orchestrator.evals.reviewer_trial.mining import assign_split


def _decision(ticket_id: str, action: str) -> RecordedDecision:
    return RecordedDecision(
        ticket_id=ticket_id, candidate={'title': ticket_id}, action=action,
        target_fingerprint=None, target_id=None,
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

    def test_refused_status_recovers_refuse(self) -> None:
        """task 3126 RED: a deterministic refusal is recoverable eval signal.

        Refusals are the ONE action the corpus can recover unambiguously --
        drop and combine both persist status='combined' and need result_json to
        disambiguate, but only the deterministic guards ever emit 'refused'.
        Skipping the row would drop the refuse decision out of the very corpus
        that trains the curator on it."""
        result_json = '{"id": null, "action": "refuse", "created": false, '\
                      '"justification": "cancelled-premise-blocklist: e: r"}'
        recovered = recover_recorded_action('refused', result_json, None)
        assert recovered == ('refuse', None, None)

    def test_refused_status_with_no_result_json_recovers_refuse(self) -> None:
        """Status alone is sufficient -- 'refused' is unambiguous."""
        assert recover_recorded_action('refused', None, None) == ('refuse', None, None)

    def test_refuse_is_not_a_proposable_frontier_action(self) -> None:
        """A frontier model must never be able to PROPOSE a refusal: refusal is
        an operator-curated deterministic verdict, not a judgement call. This
        pins that 'refuse' stays out of the frontier label enum even though
        recover_recorded_action now recovers it from persisted rows."""
        from orchestrator.evals.prompt_opt.curator_corpus import (
            _FRONTIER_CURATOR_LABEL_SCHEMA,
            _parse_frontier_label,
        )

        assert 'refuse' not in _FRONTIER_CURATOR_LABEL_SCHEMA['properties']['action']['enum']
        assert _parse_frontier_label({'action': 'refuse', 'justification': 'x'}) is None

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


def _decisions_for_actions(counts: dict[str, int]) -> list[RecordedDecision]:
    decisions = []
    for action, count in counts.items():
        for i in range(count):
            decisions.append(_decision(f'{action}-{i}', action))
    return decisions


class TestSelectSpotCheckSubset:
    """select_spot_check_subset -- the Open-Q Sec9 tactical decision (human
    spot-check subset size): deterministic, action-stratified sampling
    bounded to keep human review effort tractable."""

    def test_subset_is_a_subset_of_input_and_never_exceeds_it(self) -> None:
        decisions = _decisions_for_actions({'create': 20, 'combine': 10, 'drop': 5})
        subset = select_spot_check_subset(decisions, seed=1)
        all_ids = {d.ticket_id for d in decisions}
        assert set(subset) <= all_ids
        assert len(subset) <= len(decisions)

    def test_every_present_action_gets_representation(self) -> None:
        decisions = _decisions_for_actions({'create': 20, 'combine': 10, 'drop': 3})
        subset = set(select_spot_check_subset(decisions, seed=1))
        by_action = {
            action: [d.ticket_id for d in decisions if d.action == action]
            for action in ('create', 'combine', 'drop')
        }
        for action, ids in by_action.items():
            assert subset.intersection(ids), f'{action} stratum has no spot-check representation'

    def test_deterministic_for_a_fixed_seed(self) -> None:
        decisions = _decisions_for_actions({'create': 20, 'combine': 10, 'drop': 5})
        first = select_spot_check_subset(decisions, seed=7)
        second = select_spot_check_subset(decisions, seed=7)
        assert first == second

    def test_varies_with_seed(self) -> None:
        decisions = _decisions_for_actions({'create': 40, 'combine': 40, 'drop': 40})
        first = select_spot_check_subset(decisions, seed=1)
        second = select_spot_check_subset(decisions, seed=2)
        assert first != second

    def test_default_sized_subset_is_bounded_by_fraction(self) -> None:
        decisions = _decisions_for_actions({'create': 100})
        subset = select_spot_check_subset(decisions, seed=1, fraction=0.2, minimum=5)
        assert len(subset) == 20

    def test_result_is_capped_to_bound_human_effort(self) -> None:
        decisions = _decisions_for_actions({'create': 100, 'combine': 100, 'drop': 100})
        subset = select_spot_check_subset(decisions, seed=1, fraction=0.5, minimum=5, cap=10)
        assert len(subset) <= 10

    def test_empty_input_returns_empty(self) -> None:
        assert select_spot_check_subset([], seed=1) == []

    def test_stratify_by_action_false_samples_flatly(self) -> None:
        decisions = _decisions_for_actions({'create': 30, 'combine': 30, 'drop': 30})
        subset = select_spot_check_subset(
            decisions, seed=1, fraction=0.1, minimum=2, stratify_by_action=False,
        )
        # Flat sampling still respects subset/size invariants even though
        # it does not guarantee per-action representation.
        assert set(subset) <= {d.ticket_id for d in decisions}
        assert len(subset) <= len(decisions)


def _make_item(
    ticket_id: str = 't1',
    gold_action: str = 'create',
    split: str | None = None,
) -> CuratorReplayItem:
    return CuratorReplayItem(
        ticket_id=ticket_id,
        candidate={'title': f'Candidate {ticket_id}'},
        recorded_action='create',
        recorded_target_fingerprint=None,
        recorded_target_id=None,
        gold_action=gold_action,
        gold_target_fingerprint=None,
        gold_target_id=None,
        split=split,
        provenance={'kind': 'test'},
    )


class TestCuratorReplayItem:
    def test_holds_candidate_recorded_and_gold_fields(self) -> None:
        item = CuratorReplayItem(
            ticket_id='t1',
            candidate={'title': 'Add the widget'},
            recorded_action='combine',
            recorded_target_fingerprint='Old title',
            recorded_target_id='task-1',
            gold_action='create',
            gold_target_fingerprint=None,
            gold_target_id=None,
            split='train',
            provenance={'source_db': '/tmp/tickets.db'},
        )
        assert item.ticket_id == 't1'
        assert item.candidate == {'title': 'Add the widget'}
        assert item.recorded_action == 'combine'
        assert item.recorded_target_fingerprint == 'Old title'
        assert item.recorded_target_id == 'task-1'
        assert item.gold_action == 'create'
        assert item.split == 'train'
        assert item.provenance == {'source_db': '/tmp/tickets.db'}

    def test_split_and_provenance_default(self) -> None:
        item = CuratorReplayItem(
            ticket_id='t1', candidate={}, recorded_action='create',
            recorded_target_fingerprint=None, recorded_target_id=None,
            gold_action='create', gold_target_fingerprint=None, gold_target_id=None,
        )
        assert item.split is None
        assert item.provenance == {}


class TestCuratorCorpusManifestSaveLoad:
    def test_save_and_load_round_trips_items_losslessly(self, tmp_path: Path) -> None:
        items = [
            _make_item('t1', gold_action='create', split='train'),
            _make_item('t2', gold_action='drop', split='test'),
        ]
        manifest = CuratorCorpusManifest(items=items, split_seed=42)
        manifest_path = tmp_path / 'manifest.json'

        manifest.save(manifest_path)
        loaded = CuratorCorpusManifest.load(manifest_path)

        assert loaded.split_seed == 42
        assert len(loaded.items) == 2
        loaded_by_id = {item.ticket_id: item for item in loaded.items}
        assert loaded_by_id['t1'].gold_action == 'create'
        assert loaded_by_id['t1'].split == 'train'
        assert loaded_by_id['t2'].gold_action == 'drop'
        assert loaded_by_id['t2'].split == 'test'
        assert loaded_by_id['t1'].candidate == {'title': 'Candidate t1'}
        assert loaded_by_id['t1'].provenance == {'kind': 'test'}

    def test_save_writes_manifest_and_per_item_annotation_files(self, tmp_path: Path) -> None:
        manifest = CuratorCorpusManifest(items=[_make_item('t1', split='train')])
        manifest_path = tmp_path / 'manifest.json'

        manifest.save(manifest_path)

        assert manifest_path.exists()
        assert (tmp_path / 'annotations' / 't1.json').exists()

    def test_get_item_returns_none_when_missing(self) -> None:
        manifest = CuratorCorpusManifest(items=[_make_item('t1')])
        assert manifest.get_item('t1') is not None
        assert manifest.get_item('nonexistent') is None


class TestCuratorItemSplitAssignment:
    """Split assignment reuses reviewer_trial.mining.assign_split (stable
    per-id hash) -- deterministic, and disjoint/exhaustive across items."""

    def test_split_assignment_is_deterministic_for_a_fixed_seed(self) -> None:
        ticket_ids = [f'tkt_{i}' for i in range(30)]
        first = assign_split(ticket_ids, seed='2496-test-seed')
        second = assign_split(ticket_ids, seed='2496-test-seed')
        assert first == second

    def test_split_assignment_partitions_items_disjointly_and_exhaustively(self) -> None:
        ticket_ids = [f'tkt_{i}' for i in range(30)]
        assignment = assign_split(ticket_ids, seed='2496-test-seed')
        items = [_make_item(tid, split=assignment[tid]) for tid in ticket_ids]

        manifest = CuratorCorpusManifest(items=items, split_seed=None)
        by_split: dict[str, set[str]] = {'train': set(), 'selection': set(), 'test': set()}
        for item in manifest.items:
            assert item.split in by_split
            by_split[item.split].add(item.ticket_id)

        # exhaustive: every id appears in exactly one split
        all_assigned = by_split['train'] | by_split['selection'] | by_split['test']
        assert all_assigned == set(ticket_ids)
        # disjoint: no id appears in more than one split
        assert not (by_split['train'] & by_split['selection'])
        assert not (by_split['train'] & by_split['test'])
        assert not (by_split['selection'] & by_split['test'])


class TestParseFrontierLabel:
    """_parse_frontier_label -- pure mapping from a frontier structured-output
    dict to a FrontierLabel, never raising on malformed input."""

    def test_valid_data_maps_to_frontier_label(self) -> None:
        label = _parse_frontier_label({
            'action': 'combine', 'target_fingerprint': 'X', 'target_id': 'Y',
            'justification': 'why',
        })
        assert label == FrontierLabel(
            action='combine', target_fingerprint='X', target_id='Y', justification='why',
        )

    def test_create_action_with_no_target_fields(self) -> None:
        label = _parse_frontier_label({'action': 'create', 'justification': 'new work'})
        assert label == FrontierLabel(action='create', justification='new work')

    def test_missing_action_key_returns_none(self) -> None:
        assert _parse_frontier_label({'not_action': 'x'}) is None

    def test_invalid_action_value_returns_none(self) -> None:
        assert _parse_frontier_label({'action': 'bogus', 'justification': 'x'}) is None

    def test_non_dict_input_returns_none(self) -> None:
        assert _parse_frontier_label('not a dict') is None
        assert _parse_frontier_label(None) is None
        assert _parse_frontier_label(['issues']) is None


async def _fake_frontier_proposer(candidate: dict) -> FrontierLabel:
    """Deterministic fake: ALWAYS proposes 'drop' at a synthetic target,
    regardless of what the ticket's live curator historically recorded.
    Proves gold labels come from the injected proposer, never the recorded
    decision (PRD D-6: decisions != ground truth)."""
    return FrontierLabel(action='drop', target_id='synthetic-target-1', justification='fake-frontier')


class TestBuildCuratorCorpus:
    @pytest.mark.asyncio
    async def test_returns_manifest_with_frontier_gold_labels_not_recorded_actions(
        self, tmp_path: Path,
    ) -> None:
        rows = [
            {'ticket_id': f't{i}', 'status': 'created', 'task_id': f'task-{i}'}
            for i in range(20)
        ] + [
            {
                'ticket_id': f'c{i}', 'status': 'combined',
                'result': {'id': f'task-c{i}', 'action': 'combine', 'target_fingerprint': f'Target {i}'},
            }
            for i in range(10)
        ]
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', rows)

        manifest, log = await build_curator_corpus(
            db_path, n=20, seed=1, spot_check_size=10, frontier_proposer=_fake_frontier_proposer,
        )

        assert 0 < len(manifest.items) <= 20
        # Gold label ALWAYS comes from the fake proposer ('drop'), never the
        # recorded decision -- and at least one item's recorded action was
        # NOT 'drop', so this also proves gold != recorded in practice.
        assert all(item.gold_action == 'drop' for item in manifest.items)
        assert all(item.gold_target_id == 'synthetic-target-1' for item in manifest.items)
        assert any(item.recorded_action != item.gold_action for item in manifest.items)

    @pytest.mark.asyncio
    async def test_every_item_gets_a_2_1_7_split(self, tmp_path: Path) -> None:
        rows = [{'ticket_id': f't{i}', 'status': 'created', 'task_id': f'task-{i}'} for i in range(30)]
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', rows)

        manifest, _log = await build_curator_corpus(
            db_path, n=30, seed=1, spot_check_size=10, frontier_proposer=_fake_frontier_proposer,
        )

        assert all(item.split in ('train', 'selection', 'test') for item in manifest.items)

    @pytest.mark.asyncio
    async def test_flags_a_non_empty_human_spot_check_subset(self, tmp_path: Path) -> None:
        rows = [{'ticket_id': f't{i}', 'status': 'created', 'task_id': f'task-{i}'} for i in range(30)]
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', rows)

        _manifest, log = await build_curator_corpus(
            db_path, n=30, seed=1, spot_check_size=10, frontier_proposer=_fake_frontier_proposer,
        )

        assert log.spot_check_subset()

    @pytest.mark.asyncio
    async def test_adjudication_log_covers_every_item(self, tmp_path: Path) -> None:
        rows = [{'ticket_id': f't{i}', 'status': 'created', 'task_id': f'task-{i}'} for i in range(30)]
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', rows)

        manifest, log = await build_curator_corpus(
            db_path, n=30, seed=1, spot_check_size=10, frontier_proposer=_fake_frontier_proposer,
        )

        coverage = log.coverage(item.ticket_id for item in manifest.items)
        assert coverage.ok

    @pytest.mark.asyncio
    async def test_refused_rows_are_excluded_from_the_labeled_corpus(
        self, tmp_path: Path,
    ) -> None:
        """A deterministic refusal is provenance, never a labeled corpus item.

        `recover_recorded_action` recovers status='refused' -> 'refuse' so the
        verdict stays readable, but every gold label comes from
        *frontier_proposer* (PRD D-6) and no proposer can emit 'refuse' -- its
        schema is drop/combine/create. A refused row admitted to the corpus
        would therefore carry a definitionally-wrong gold label, take a
        round-robin stratum's share of a bounded n, and spend human spot-check
        budget -- all for a candidate the curator LLM never sees in production
        (both guards short-circuit pre-LLM)."""
        rows = [
            {'ticket_id': f't{i}', 'status': 'created', 'task_id': f'task-{i}'}
            for i in range(6)
        ] + [
            {
                'ticket_id': f'r{i}', 'status': 'refused', 'task_id': None,
                'result': {
                    'id': None, 'action': 'refuse', 'created': False,
                    'justification': 'cancelled-premise-blocklist: e: r',
                },
            }
            for i in range(6)
        ]
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', rows)
        proposed: list[dict] = []

        async def _tracking_proposer(candidate: dict) -> FrontierLabel:
            proposed.append(candidate)
            return await _fake_frontier_proposer(candidate)

        manifest, log = await build_curator_corpus(
            db_path, n=12, seed=1, spot_check_size=10, frontier_proposer=_tracking_proposer,
        )

        ticket_ids = {item.ticket_id for item in manifest.items}
        assert ticket_ids == {f't{i}' for i in range(6)}, (
            'refused rows must not reach the labeled corpus'
        )
        assert all(item.recorded_action != 'refuse' for item in manifest.items)
        # No frontier label was even PROPOSED for a refusal (no wasted call,
        # and no wrong gold label), and none reached the spot-check budget.
        assert len(proposed) == 6
        assert not any(entry.diff_id.startswith('r') for entry in log.spot_check_subset())

    @pytest.mark.asyncio
    async def test_no_real_db_or_llm_touched_beyond_the_injected_fake(self, tmp_path: Path) -> None:
        """Fully hermetic: only the passed-in synthetic db_path is read, and
        the only 'frontier' call is the injected fake (no real invoke_agent)."""
        rows = [{'ticket_id': 't1', 'status': 'created', 'task_id': 'task-1'}]
        db_path = make_synthetic_tickets_db(tmp_path / 'tickets.db', rows)
        calls = []

        async def _tracking_proposer(candidate: dict) -> FrontierLabel:
            calls.append(candidate)
            return await _fake_frontier_proposer(candidate)

        manifest, _log = await build_curator_corpus(
            db_path, n=10, seed=1, spot_check_size=10, frontier_proposer=_tracking_proposer,
        )

        assert len(calls) == len(manifest.items) == 1


class TestAuditCuratorCorpus:
    """audit_curator_corpus(manifest, adjudication_log, min_items) reports
    ok=True only when every corpus-integrity signal holds, naming which
    check(s) failed otherwise (mirrors reviewer_trial.mining.audit_corpus)."""

    @staticmethod
    def _make_item(ticket_id: str, split: str | None, gold_action: str | None) -> CuratorReplayItem:
        return CuratorReplayItem(
            ticket_id=ticket_id,
            candidate={'title': f'Candidate {ticket_id}'},
            recorded_action='create',
            recorded_target_fingerprint=None,
            recorded_target_id=None,
            gold_action=gold_action,  # type: ignore[arg-type]
            gold_target_fingerprint=None,
            gold_target_id=None,
            split=split,
        )

    @classmethod
    def _passing_manifest_and_log(cls, n: int = 30) -> tuple[CuratorCorpusManifest, AdjudicationLog]:
        ticket_ids = [f't{i}' for i in range(n)]
        splits = assign_split(ticket_ids, seed='curator-audit-test')

        items = [cls._make_item(tid, splits[tid], 'create') for tid in ticket_ids]
        manifest = CuratorCorpusManifest(items=items, split_seed=None)

        log = AdjudicationLog()
        for i, tid in enumerate(ticket_ids):
            log.append(tid, frontier_proposal=[{'action': 'create'}], in_spot_check_subset=(i == 0))

        return manifest, log

    def test_passes_when_all_signals_hold(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)

        report = audit_curator_corpus(manifest, log, min_items=30)

        assert report.ok is True
        assert report.failures == []
        assert report.item_count == 30

    def test_fails_when_below_min_items(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=10)

        report = audit_curator_corpus(manifest, log, min_items=50)

        assert report.ok is False
        assert 'item_count' in report.failures

    def test_fails_when_split_missing(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        manifest.items[0].split = None

        report = audit_curator_corpus(manifest, log, min_items=30)

        assert report.ok is False
        assert 'missing_split' in report.failures

    def test_fails_when_split_ratio_off(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        for item in manifest.items:
            item.split = 'train'  # every item in one bucket -> ratio way off 2:1:7

        report = audit_curator_corpus(manifest, log, min_items=30)

        assert report.ok is False
        assert 'split_ratio' in report.failures

    def test_fails_when_gold_label_missing(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        manifest.items[0].gold_action = None  # type: ignore[assignment]

        report = audit_curator_corpus(manifest, log, min_items=30)

        assert report.ok is False
        assert 'missing_gold_label' in report.failures

    def test_fails_when_adjudication_coverage_incomplete(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        log.entries.pop()  # last ticket_id now has no adjudication entry

        report = audit_curator_corpus(manifest, log, min_items=30)

        assert report.ok is False
        assert 'adjudication_coverage' in report.failures

    def test_fails_when_spot_check_subset_empty(self) -> None:
        manifest, log = self._passing_manifest_and_log(n=30)
        for entry in log.entries:
            entry.in_spot_check_subset = False

        report = audit_curator_corpus(manifest, log, min_items=30)

        assert report.ok is False
        assert 'spot_check_subset_empty' in report.failures

    def test_never_raises_on_empty_manifest(self) -> None:
        report = audit_curator_corpus(CuratorCorpusManifest(items=[]), AdjudicationLog(), min_items=50)

        assert report.ok is False
        assert report.item_count == 0
        assert 'item_count' in report.failures
