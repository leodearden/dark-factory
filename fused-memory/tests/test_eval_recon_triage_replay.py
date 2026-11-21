"""Tests for eval_recon_triage_replay.py — the read-only recon replay of write-triage attaches.

Reached only through the public surface. The scripts are loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``. Derivations come
from synthetic verdict rows through the scoring module's ``corpus_derivations``.
"""
from __future__ import annotations

import asyncio
import dataclasses
import functools
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig, ReconciliationConfig
from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_MEMORY_WRITES,
    STAGE1_DISALLOWED,
    StageResult,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_payload_format import format_memories
from fused_memory.reconciliation.stages.base import recon_core_mcp_servers
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)
from fused_memory.server.write_triage import OUTCOME_AMENDED, OUTCOME_CONTESTED, OUTCOME_RESTATED

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'eval_recon_triage_replay.py', mod_name='eval_recon_triage_replay',
    )


@functools.cache
def _scoring() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'recon_triage_replay_scoring.py', mod_name='recon_triage_replay_scoring',
    )


_CORPUS_STRATUM = {
    OUTCOME_CONTESTED: 'contested-census', OUTCOME_AMENDED: 'amended-sample',
    OUTCOME_RESTATED: 'restated-sample',
}


def _item(entry: str, outcome: str, project: str) -> dict[str, Any]:
    return {
        'band': 'judge', 'cross_category': False, 'earlier_ids': [], 'entry_id': entry,
        'outcome': outcome, 'parent_edited_after_attach': False, 'project': project,
        'stratum': _CORPUS_STRATUM[outcome], 'superseded_marks': [], 'supersedes_target': False,
        'target_id': f'P-{entry}', 'via_sibling': False,
    }


def _vote(entry: str, **answers: Any) -> dict[str, Any]:
    row = {
        'entry_id': entry, 'target_id': f'P-{entry}', 'basis': 'current',
        'rater': 'opus-primary', 'role': 'primary', 'batch': 'b1', **answers,
    }
    if answers.get('conflict') == 'YES':
        row['conflict_quote'] = {'parent': 'the record says A', 'child': 'it says B'}
    return row


_DRAW_PROJECTS = ('p1', 'p2')


def _draw_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per project: 6 decided contests, 4 decided amendments, 1 unclear contest; 2 off-allocation."""
    items: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    for project in _DRAW_PROJECTS:
        for index in range(6):
            entry = f'C-{project}-{index}'
            items.append(_item(entry, OUTCOME_CONTESTED, project))
            verdicts.append(_vote(entry, conflict='YES'))
        for index in range(4):
            entry = f'A-{project}-{index}'
            items.append(_item(entry, OUTCOME_AMENDED, project))
            verdicts.append(_vote(entry, home='NO', conflict='NO'))
        entry = f'C-{project}-unclear'
        items.append(_item(entry, OUTCOME_CONTESTED, project))
        verdicts.append(_vote(entry, conflict='UNCLEAR', unclear_reason='cut at the claim'))
    for index in range(2):
        entry = f'C-scoping-{index}'
        items.append(_item(entry, OUTCOME_CONTESTED, 'scoping_intake'))
        verdicts.append(_vote(entry, conflict='YES'))
    return items, verdicts


_ALLOCATION = {'contested|p1': 3, 'contested|p2': 3, 'amended|p1': 2, 'amended|p2': 2}


def _draw(*, seed: int = 7161, allocation: dict[str, int] = _ALLOCATION,
          touched: dict[str, str] | None = None) -> list[dict[str, Any]]:
    items, verdicts = _draw_corpus()
    return _mod().draw_sample(
        items, _scoring().corpus_derivations(items, verdicts), allocation=allocation, seed=seed,
        touched=touched or {},
    )


def _headline_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {row['entry_id'] for row in rows if row['headline']}


class TestTheDraw:
    def test_it_draws_exactly_the_allocation_from_decided_items(self) -> None:
        rows = _draw()
        drawn: dict[str, int] = {}
        for row in rows:
            drawn[row['stratum']] = drawn.get(row['stratum'], 0) + 1
        assert drawn == _ALLOCATION
        assert not any(row['entry_id'].endswith('unclear') for row in rows)

    def test_off_allocation_projects_are_never_drawn(self) -> None:
        assert not any(row['project'] == 'scoping_intake' for row in _draw())

    def test_it_is_deterministic_for_a_seed_and_moves_with_it(self) -> None:
        assert _draw() == _draw()
        assert _headline_ids(_draw(seed=1)) != _headline_ids(_draw())

    def test_an_allocation_a_stratum_cannot_meet_is_refused_naming_it(self) -> None:
        with pytest.raises(ValueError) as refused:
            _draw(allocation={'contested|p1': 7})
        message = str(refused.value)
        assert 'contested|p1' in message
        assert '7' in message
        assert '6' in message

    def test_every_row_carries_the_sample_contract(self) -> None:
        for row in _draw():
            assert set(row) == {
                'entry_id', 'target_id', 'outcome', 'project', 'stratum', 'headline',
                'live_recon_state',
            }
            assert row['stratum'] == _scoring().sample_stratum(row['outcome'], row['project'])
            assert row['target_id'] == f'P-{row["entry_id"]}'


class TestTheTouchedSubset:
    def test_touched_children_are_flagged_and_never_move_the_draw(self) -> None:
        plain = _draw()
        drawn = sorted(_headline_ids(plain))
        undrawn = sorted(
            row['entry_id'] for row in _draw_corpus()[0]
            if row['project'] in _DRAW_PROJECTS and not row['entry_id'].endswith('unclear')
            and row['entry_id'] not in set(drawn)
        )
        touched = {drawn[0]: 'missing', undrawn[0]: 'relinked', 'not-in-corpus': 'missing'}
        rows = _draw(touched=touched)
        assert _headline_ids(rows) == set(drawn)
        by_id = {row['entry_id']: row for row in rows}
        assert by_id[drawn[0]]['headline'] is True
        assert by_id[drawn[0]]['live_recon_state'] == 'missing'
        assert by_id[undrawn[0]]['headline'] is False
        assert by_id[undrawn[0]]['live_recon_state'] == 'relinked'
        assert 'not-in-corpus' not in by_id
        assert len(rows) == len(plain) + 1
        assert all(
            row['live_recon_state'] is None for row in rows if row['entry_id'] not in touched
        )

    def test_touched_states_are_read_from_the_soak_bundle_records(self) -> None:
        records = [
            {'id': 'a', 'state': 'relinked', 'shape': 'contested'},
            {'id': 'b', 'state': 'missing', 'shape': 'sighting'},
        ]
        assert _mod().touched_states(records) == {'a': 'relinked', 'b': 'missing'}


class TestTheHeader:
    def test_it_records_the_seed_the_allocation_and_each_inputs_sha256(self, tmp_path: Path) -> None:
        inputs = {}
        for name in ('items', 'verdicts', 'texts'):
            path = tmp_path / f'{name}.json'
            path.write_text(f'{name} bytes')
            inputs[name] = path
        header = _mod().sample_header(seed=7161, allocation=_ALLOCATION, inputs=inputs)
        assert header['seed'] == 7161
        assert header['allocation'] == _ALLOCATION
        assert header['inputs'] == {
            name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in inputs.items()
        }


class TestTheDefaultAllocation:
    def test_it_draws_160_headline_children_over_four_projects(self) -> None:
        sample_stratum = _scoring().sample_stratum
        projects = ('dark_factory', 'reify', 'solar_challenge', 'solar_challenge_platform')
        assert dict(_mod().DEFAULT_ALLOCATION) == {
            sample_stratum(outcome, project): count
            for project in projects
            for outcome, count in (
                (OUTCOME_CONTESTED, 20), (OUTCOME_AMENDED, 10), (OUTCOME_RESTATED, 10),
            )
        }
        assert sum(_mod().DEFAULT_ALLOCATION.values()) == 160
        assert _mod().SAMPLE_SEED == 7161


# ── the worklist ───────────────────────────────────────────────────────────


def _worklist_rows() -> list[dict[str, Any]]:
    """Two projects' sample rows: 5 in p1 across the outcomes, 3 in p2."""
    shapes = [
        ('p1', OUTCOME_CONTESTED), ('p1', OUTCOME_CONTESTED), ('p1', OUTCOME_AMENDED),
        ('p1', OUTCOME_RESTATED), ('p1', OUTCOME_RESTATED),
        ('p2', OUTCOME_CONTESTED), ('p2', OUTCOME_AMENDED), ('p2', OUTCOME_RESTATED),
    ]
    return [
        {'entry_id': f'E{index}', 'target_id': f'P{index}', 'outcome': outcome, 'project': project,
         'stratum': _scoring().sample_stratum(outcome, project), 'headline': True,
         'live_recon_state': None}
        for index, (project, outcome) in enumerate(shapes)
    ]


def _texts_root(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    """A RatingScratch-layout texts.json: each pair on the current basis, one also at attach time."""
    records = [
        {'basis': 'current', 'child_text': f'child text of {row["entry_id"]}',
         'entry_id': row['entry_id'], 'parent_text': f'parent text of {row["target_id"]}',
         'target_id': row['target_id']}
        for row in rows
    ]
    records.append({**records[0], 'basis': 'attach_time', 'child_text': 'the attach-time child'})
    (tmp_path / 'texts.json').write_text(json.dumps(records))
    return tmp_path


def _texts(tmp_path: Path, rows: list[dict[str, Any]]) -> Any:
    root = _texts_root(tmp_path, rows)
    return _mod().load_texts(root, {row['entry_id']: row['project'] for row in rows})


class TestTheBatchPlan:
    def test_each_batch_is_one_project_of_at_most_size_rows(self) -> None:
        for batch in _mod().plan_batches(_worklist_rows(), size=2, seed=7161):
            assert {row['project'] for row in batch.rows} == {batch.project}
            assert 1 <= len(batch.rows) <= 2

    def test_every_row_lands_in_exactly_one_batch(self) -> None:
        batches = _mod().plan_batches(_worklist_rows(), size=2, seed=7161)
        placed = [row['entry_id'] for batch in batches for row in batch.rows]
        assert sorted(placed) == sorted(row['entry_id'] for row in _worklist_rows())

    def test_batch_names_are_deterministic_and_unique(self) -> None:
        first = _mod().plan_batches(_worklist_rows(), size=2, seed=7161)
        again = _mod().plan_batches(list(reversed(_worklist_rows())), size=2, seed=7161)
        assert [batch.name for batch in first] == [batch.name for batch in again]
        assert first == again
        assert len({batch.name for batch in first}) == len(first)

    def test_order_within_a_project_follows_the_seed(self) -> None:
        def order(seed: int) -> list[str]:
            return [row['entry_id'] for batch in _mod().plan_batches(
                _worklist_rows(), size=8, seed=seed) for row in batch.rows]

        orders = {tuple(order(seed)) for seed in range(6)}
        assert len(orders) > 1


class TestTheWorklistPayload:
    def _payload(self, tmp_path: Path) -> tuple[Any, str]:
        rows = _worklist_rows()
        batch = next(b for b in _mod().plan_batches(rows, size=8, seed=7161) if b.project == 'p1')
        payload = _mod().render_worklist_payload(
            batch, _texts(tmp_path, rows), project_id='p1', project_root='/srv/p1',
        )
        return batch, payload

    def test_it_opens_with_the_stage_1_header(self, tmp_path: Path) -> None:
        _, payload = self._payload(tmp_path)
        assert payload.splitlines()[:2] == [
            '## Reconciliation Run — Stage 1: Memory Consolidation', '## Project: p1',
        ]

    def test_each_child_and_parent_renders_as_stage_1_renders_a_record(self, tmp_path: Path) -> None:
        batch, payload = self._payload(tmp_path)
        blocks = payload.split('\n#### ')
        for row in batch.rows:
            [block] = [block for block in blocks if block.startswith(f'child {row["entry_id"]}')]
            contested = row['outcome'] == OUTCOME_CONTESTED
            metadata: dict[str, Any] = {
                PARENT_ID_KEY: row['target_id'],
                'kind': SIGHTING_KIND if row['outcome'] == OUTCOME_RESTATED else AMENDMENT_KIND,
            }
            if contested:
                metadata[CONTESTED_METADATA_KEY] = True
            child_line, _ = format_memories([{
                'id': row['entry_id'], 'memory': f'child text of {row["entry_id"]}',
                'metadata': metadata,
            }])
            parent_line, _ = format_memories([{
                'id': row['target_id'], 'memory': f'parent text of {row["target_id"]}',
                'metadata': {},
            }])
            assert child_line in block.splitlines()
            assert parent_line in block.splitlines()
            assert f'triage_outcome={row["outcome"]}' in block
            assert f'{CONTESTED_METADATA_KEY}={"true" if contested else "false"}' in block

    def test_it_states_the_soak_caveat_and_the_project_root(self, tmp_path: Path) -> None:
        _, payload = self._payload(tmp_path)
        assert '2026-10-19' in payload
        assert 'Use project_root="/srv/p1"' in payload

    def test_it_holds_no_text_from_another_batch(self, tmp_path: Path) -> None:
        batch, payload = self._payload(tmp_path)
        mine = {row['entry_id'] for row in batch.rows}
        for row in _worklist_rows():
            if row['entry_id'] not in mine:
                assert f'child text of {row["entry_id"]}' not in payload
                assert row['entry_id'] not in payload

    def test_it_reads_the_current_basis_only(self, tmp_path: Path) -> None:
        _, payload = self._payload(tmp_path)
        assert 'the attach-time child' not in payload


class TestTheReplayContract:
    def test_the_system_prompt_is_stage_1_s_verbatim_plus_the_vocabulary(self) -> None:
        prompt = _mod().REPLAY_SYSTEM_PROMPT
        assert prompt.startswith(STAGE1_SYSTEM_PROMPT)
        addendum = prompt[len(STAGE1_SYSTEM_PROMPT):]
        for decision in _scoring().Decision:
            assert decision.value in addendum

    def test_the_deny_list_keeps_every_store_write_out(self) -> None:
        denied = set(_mod().REPLAY_DISALLOWED)
        assert denied >= set(STAGE1_DISALLOWED)
        assert denied >= set(DISALLOW_MEMORY_WRITES)
        assert 'mcp__fused-memory__propose_consolidation' in denied

    def test_the_output_schema_enumerates_exactly_the_decisions(self) -> None:
        item = _mod().REPLAY_OUTPUT_SCHEMA['properties']['decisions']['items']
        assert set(item['properties']['decision']['enum']) == {
            decision.value for decision in _scoring().Decision
        }
        assert set(item['required']) >= {'child_id', 'decision', 'relabel_to', 'reason', 'evidence'}


# ── parsing a batch reply ──────────────────────────────────────────────────


def _p1_batch() -> Any:
    """p1's batch: E0 and E1 contested, E2 amended, E3 and E4 restated."""
    return next(b for b in _mod().plan_batches(_worklist_rows(), size=8, seed=7161) if b.project == 'p1')


_GOOD = {
    'E0': ('contest_upheld', None), 'E1': ('detach', None), 'E2': ('relabel', OUTCOME_CONTESTED),
    'E3': ('relabel', OUTCOME_AMENDED), 'E4': ('keep', None),
}


def _reply(choices: dict[str, tuple[str, str | None]] = _GOOD, **extra: Any) -> dict[str, Any]:
    return {
        'decisions': [
            {'child_id': child, 'decision': decision, 'relabel_to': relabel_to,
             'reason': f'why {child}', 'evidence': [f'read {child}']}
            for child, (decision, relabel_to) in choices.items()
        ],
        **extra,
    }


def _refused(report: dict[str, Any]) -> str:
    parsed = _mod().parse_batch_reply(_p1_batch(), report)
    assert isinstance(parsed, _mod().BatchFailure), parsed
    assert parsed.batch == _p1_batch().name
    return parsed.reason


class TestParsingABatchReply:
    def test_a_well_formed_reply_gives_one_row_per_child(self) -> None:
        batch = _p1_batch()
        parsed = _mod().parse_batch_reply(batch, _reply(flagged_items=[]))
        assert isinstance(parsed, _mod().BatchDecisions)
        assert parsed.batch == batch.name
        targets = {row['entry_id']: row['target_id'] for row in batch.rows}
        assert sorted(parsed.rows, key=lambda row: row['entry_id']) == [
            {'entry_id': child, 'target_id': targets[child], 'decision': decision,
             'relabel_to': relabel_to, 'reason': f'why {child}', 'evidence': [f'read {child}']}
            for child, (decision, relabel_to) in sorted(_GOOD.items())
        ]

    def test_a_missing_child_fails_the_batch(self) -> None:
        choices = {child: choice for child, choice in _GOOD.items() if child != 'E3'}
        assert 'E3' in _refused(_reply(choices))

    def test_a_child_outside_the_batch_fails_it(self) -> None:
        assert 'E9' in _refused(_reply({**_GOOD, 'E9': ('keep', None)}))

    def test_a_duplicate_child_fails_the_batch(self) -> None:
        report = _reply()
        report['decisions'].append(dict(report['decisions'][0]))
        assert 'E0' in _refused(report)

    def test_an_unknown_decision_fails_the_batch_naming_child_and_value(self) -> None:
        reason = _refused(_reply({**_GOOD, 'E4': ('merge', None)}))
        assert 'E4' in reason
        assert 'merge' in reason

    def test_a_decision_the_outcome_does_not_allow_fails_the_batch(self) -> None:
        reason = _refused(_reply({**_GOOD, 'E0': ('keep', None)}))
        assert 'E0' in reason
        assert 'keep' in reason

    def test_a_relabel_without_a_target_fails_the_batch(self) -> None:
        reason = _refused(_reply({**_GOOD, 'E3': ('relabel', None)}))
        assert 'E3' in reason
        assert 'relabel' in reason

    def test_a_malformed_reason_or_evidence_fails_the_batch(self) -> None:
        report = _reply()
        report['decisions'][1]['evidence'] = 'not a list'
        assert 'E1' in _refused(report)

    @pytest.mark.parametrize('report', [
        {},
        {'summary': 'I could not finish', 'flagged_items': [], 'stats': {'raw_text_fallback': True}},
        {'decisions': 'none'},
        {'decisions': ['not an object']},
    ])
    def test_a_reply_without_a_decisions_list_fails_the_batch(self, report: dict[str, Any]) -> None:
        assert _refused(report)


# ── asking recon ───────────────────────────────────────────────────────────


def _worklists(tmp_path: Path, *, size: int = 2) -> list[Any]:
    """p1's 5 rows and p2's 3 in batches of at most *size*, each rendered as a worklist."""
    rows = _worklist_rows()
    return _mod().worklists(
        _mod().plan_batches(rows, size=size, seed=7161), _texts(tmp_path, rows),
        project_roots={'p1': '/srv/p1', 'p2': '/srv/p2'},
    )


def _p1_worklist(tmp_path: Path) -> Any:
    [worklist] = [
        worklist for worklist in _worklists(tmp_path, size=8) if worklist.batch.project == 'p1'
    ]
    return worklist


class TestTheWorklists:
    def test_each_batch_renders_with_its_projects_root(self, tmp_path: Path) -> None:
        rows = _worklist_rows()
        texts = _texts(tmp_path, rows)
        batches = _mod().plan_batches(rows, size=2, seed=7161)
        worklists = _mod().worklists(batches, texts, project_roots={'p1': '/srv/p1', 'p2': '/srv/p2'})
        assert [worklist.batch for worklist in worklists] == batches
        for worklist in worklists:
            root = f'/srv/{worklist.batch.project}'
            assert worklist.payload == _mod().render_worklist_payload(
                worklist.batch, texts, project_id=worklist.batch.project, project_root=root,
            )

    def test_a_project_with_no_root_is_refused_naming_it(self, tmp_path: Path) -> None:
        rows = _worklist_rows()
        with pytest.raises(_mod().ReplayRefused, match='p2'):
            _mod().worklists(
                _mod().plan_batches(rows, size=8, seed=7161), _texts(tmp_path, rows),
                project_roots={'p1': '/srv/p1'},
            )

    def test_a_child_with_no_rated_text_is_refused_naming_it(self, tmp_path: Path) -> None:
        rows = _worklist_rows()
        texts = dict(_texts(tmp_path, rows))
        del texts['E3', 'P3']
        with pytest.raises(_mod().ReplayRefused, match='E3'):
            _mod().worklists(
                _mod().plan_batches(rows, size=8, seed=7161), texts,
                project_roots={'p1': '/srv/p1', 'p2': '/srv/p2'},
            )


def _mcp_json(tmp_path: Path) -> Path:
    path = tmp_path / 'mcp' / '.mcp.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        'mcpServers': {'fused-memory': {'type': 'http', 'url': 'http://127.0.0.1:9/mcp'}},
    }))
    return path


class _RecordingStage:
    """A run_stage_via_cli stand-in: records each call's arguments and returns *result*."""

    def __init__(self, result: StageResult) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result = result

    async def __call__(self, **arguments: Any) -> StageResult:
        self.calls.append(arguments)
        return self._result


class TestTheStageCliAsker:
    def _asker(self, tmp_path: Path, stage: _RecordingStage, **extra: Any) -> Any:
        return _mod().StageCliAsker(
            run_stage=stage, config=ReconciliationConfig(), model='sonnet',
            cwd=Path('/srv/explore'), config_base=tmp_path / 'configs',
            mcp_candidates=[_mcp_json(tmp_path)], **extra,
        )

    def test_it_runs_stage_1_once_per_batch_under_the_replay_contract(self, tmp_path: Path) -> None:
        stage = _RecordingStage(StageResult(report=_reply(), cost_usd=0.42, success=True, model='sonnet'))
        gate = object()
        worklist = _p1_worklist(tmp_path)
        reply = asyncio.run(self._asker(tmp_path, stage, usage_gate=gate)(worklist))
        [call] = stage.calls
        assert call['system_prompt'] == _mod().REPLAY_SYSTEM_PROMPT
        assert call['payload'] == worklist.payload
        assert call['disallowed_tools'] == list(_mod().REPLAY_DISALLOWED)
        assert call['mcp_config'] == {'mcpServers': recon_core_mcp_servers([_mcp_json(tmp_path)])}
        assert call['output_schema'] == dict(_mod().REPLAY_OUTPUT_SCHEMA)
        json.dumps(call['output_schema'])
        assert call['model'] == 'sonnet'
        assert call['cwd'] == Path('/srv/explore')
        assert call['usage_gate'] is gate
        assert isinstance(call['config'], ReconciliationConfig)
        assert call['session_id']
        assert call['config_dir'].path.parent == tmp_path / 'configs'
        assert reply == _mod().AskReply(report=_reply(), cost_usd=0.42, error='', model='sonnet')

    def test_every_ask_gets_a_fresh_session_and_config_dir(self, tmp_path: Path) -> None:
        stage = _RecordingStage(StageResult(report=_reply(), success=True, model='sonnet'))
        asker = self._asker(tmp_path, stage)
        worklist = _p1_worklist(tmp_path)
        asyncio.run(asker(worklist))
        asyncio.run(asker(worklist))
        first, second = stage.calls
        assert first['session_id'] != second['session_id']
        assert first['config_dir'].path != second['config_dir'].path

    def test_a_failed_run_hands_back_its_error_and_cost(self, tmp_path: Path) -> None:
        stage = _RecordingStage(StageResult(cost_usd=0.1, error='CLI exited 1', model='sonnet'))
        reply = asyncio.run(self._asker(tmp_path, stage)(_p1_worklist(tmp_path)))
        assert reply == _mod().AskReply(report={}, cost_usd=0.1, error='CLI exited 1', model='sonnet')

    def test_its_identity_names_the_arm_and_model(self, tmp_path: Path) -> None:
        stage = _RecordingStage(StageResult())
        identity = self._asker(tmp_path, stage).identity
        assert identity['asker'] == 'stage'
        assert identity['model'] == 'sonnet'
        assert identity['system_prompt_sha256'] == hashlib.sha256(
            _mod().REPLAY_SYSTEM_PROMPT.encode()).hexdigest()


class TestTheFakeAsker:
    def test_it_is_pure_in_the_payload_and_answers_only_allowed_actions(self, tmp_path: Path) -> None:
        worklist = _p1_worklist(tmp_path)
        reply = asyncio.run(_mod().FakeAsker()(worklist))
        assert reply == asyncio.run(_mod().FakeAsker()(worklist))
        assert isinstance(_mod().parse_batch_reply(worklist.batch, reply.report), _mod().BatchDecisions)
        assert (reply.cost_usd, reply.error, reply.model) == (0.0, '', 'fake')
        assert _mod().FakeAsker().identity == {'asker': 'fake'}


# ── the run loop ───────────────────────────────────────────────────────────


class _CountingAsker:
    """Answers through the fake arm at a cost; fails or raises for the batches named."""

    def __init__(self, *, failing: frozenset[str] = frozenset(), raising: frozenset[str] = frozenset(),
                 identity: dict[str, str] | None = None) -> None:
        self.asked: list[str] = []
        self.in_flight = 0
        self.peak = 0
        self._failing = failing
        self._raising = raising
        self.identity = identity or {'asker': 'counting'}

    async def __call__(self, worklist: Any) -> Any:
        name = worklist.batch.name
        self.asked.append(name)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        if name in self._raising:
            raise RuntimeError(f'all accounts capped while asking {name}')
        if name in self._failing:
            return _mod().AskReply(report={}, cost_usd=0.25, error='CLI exited 1', model='m')
        reply = await _mod().FakeAsker()(worklist)
        return dataclasses.replace(reply, cost_usd=0.5, model='m')


def _run(worklists: list[Any], asker: Any, scratch: Path, **options: Any) -> Any:
    options.setdefault('concurrency', 2)
    return asyncio.run(_mod().run_replay(worklists, asker, scratch, **options))


def _batch_record(scratch: Path, name: str) -> dict[str, Any]:
    return json.loads((scratch / 'batches' / f'{name}.json').read_text())


class TestTheRunLoop:
    def test_it_writes_one_parsed_file_per_batch(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        scratch = tmp_path / 'scratch'
        statuses = _run(worklists, _CountingAsker(), scratch)
        assert statuses == {worklist.batch.name: _mod().BatchStatus.OK for worklist in worklists}
        assert sorted(path.name for path in (scratch / 'batches').iterdir()) == sorted(
            f'{worklist.batch.name}.json' for worklist in worklists)
        for worklist in worklists:
            record = _batch_record(scratch, worklist.batch.name)
            fake = asyncio.run(_mod().FakeAsker()(worklist))
            parsed = _mod().parse_batch_reply(worklist.batch, fake.report)
            assert record['status'] == 'ok'
            assert record['decisions'] == json.loads(json.dumps(list(parsed.rows)))
            assert (record['cost_usd'], record['model'], record['error']) == (0.5, 'm', '')
            assert record['payload_sha256'] == hashlib.sha256(worklist.payload.encode()).hexdigest()

    def test_it_asks_at_most_concurrency_batches_at_once(self, tmp_path: Path) -> None:
        asker = _CountingAsker()
        _run(_worklists(tmp_path), asker, tmp_path / 'scratch', concurrency=2)
        assert asker.peak == 2
        single = _CountingAsker()
        _run(_worklists(tmp_path), single, tmp_path / 'scratch-1', concurrency=1)
        assert single.peak == 1

    def test_a_failed_reply_is_recorded_with_its_reason_and_cost(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        failing = worklists[1].batch.name
        scratch = tmp_path / 'scratch'
        statuses = _run(worklists, _CountingAsker(failing=frozenset({failing})), scratch)
        assert statuses[failing] is _mod().BatchStatus.FAILED
        record = _batch_record(scratch, failing)
        assert record['status'] == 'failed'
        assert record['reason']
        assert record['decisions'] == []
        assert (record['error'], record['cost_usd']) == ('CLI exited 1', 0.25)

    def test_an_ask_that_raises_fails_only_its_batch(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        raising = worklists[0].batch.name
        scratch = tmp_path / 'scratch'
        statuses = _run(worklists, _CountingAsker(raising=frozenset({raising})), scratch)
        assert statuses[raising] is _mod().BatchStatus.FAILED
        assert 'all accounts capped' in _batch_record(scratch, raising)['error']
        assert sum(status is _mod().BatchStatus.OK for status in statuses.values()) == len(worklists) - 1

    def test_a_rerun_skips_parsed_batches_and_re_asks_failed_ones(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        failing = worklists[2].batch.name
        scratch = tmp_path / 'scratch'
        _run(worklists, _CountingAsker(failing=frozenset({failing})), scratch)
        again = _CountingAsker()
        statuses = _run(worklists, again, scratch)
        assert again.asked == [failing]
        assert set(statuses.values()) == {_mod().BatchStatus.OK}
        assert _batch_record(scratch, failing)['status'] == 'ok'

    def test_only_asks_the_batches_named(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        chosen = worklists[3].batch.name
        asker = _CountingAsker()
        statuses = _run(worklists, asker, tmp_path / 'scratch', only=frozenset({chosen}))
        assert asker.asked == [chosen]
        assert statuses == {chosen: _mod().BatchStatus.OK}

    def test_only_naming_no_planned_batch_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(_mod().ReplayRefused, match='p9-00'):
            _run(_worklists(tmp_path), _CountingAsker(), tmp_path / 'scratch', only=frozenset({'p9-00'}))

    def test_a_scratch_planned_with_other_batches_is_refused(self, tmp_path: Path) -> None:
        scratch = tmp_path / 'scratch'
        _run(_worklists(tmp_path), _CountingAsker(), scratch)
        asker = _CountingAsker()
        with pytest.raises(_mod().ReplayRefused, match='plan.json'):
            _run(_worklists(tmp_path, size=3), asker, scratch)
        assert asker.asked == []

    def test_a_scratch_planned_for_another_asker_is_refused(self, tmp_path: Path) -> None:
        scratch = tmp_path / 'scratch'
        _run(_worklists(tmp_path), _CountingAsker(), scratch)
        with pytest.raises(_mod().ReplayRefused, match='plan.json'):
            _run(_worklists(tmp_path), _CountingAsker(identity={'asker': 'other'}), scratch)


# ── collecting the decisions ───────────────────────────────────────────────

_DECISION_KEYS = {
    'entry_id', 'target_id', 'batch', 'model', 'decision', 'relabel_to', 'reason', 'evidence',
    'cost_usd',
}


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestCollecting:
    def test_it_writes_a_header_then_one_calibration_row_per_decision(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        scratch = tmp_path / 'scratch'
        _run(worklists, _CountingAsker(), scratch)
        out = tmp_path / 'decisions.jsonl'
        _mod().collect(scratch, out, tool_health={'search': 'ok'})
        header, *rows = _lines(out)
        assert header['header'] == 'recon_triage_replay_decisions'
        assert header['tool_health'] == {'search': 'ok'}
        assert (header['missing_batches'], header['failed_batches']) == ([], [])
        assert header['cost_usd'] == pytest.approx(0.5 * len(worklists))
        assert sorted(row['entry_id'] for row in rows) == sorted(
            row['entry_id'] for row in _worklist_rows())
        batch_of = {row['entry_id']: w.batch.name for w in worklists for row in w.batch.rows}
        for row in rows:
            assert set(row) == _DECISION_KEYS
            assert row['batch'] == batch_of[row['entry_id']]
            assert row['model'] == 'm'
        for worklist in worklists:
            spent = sum(row['cost_usd'] for row in rows if row['batch'] == worklist.batch.name)
            assert spent == pytest.approx(0.5)

    def test_it_is_byte_identical_for_the_same_scratch(self, tmp_path: Path) -> None:
        scratch = tmp_path / 'scratch'
        _run(_worklists(tmp_path), _CountingAsker(), scratch)
        _mod().collect(scratch, tmp_path / 'a.jsonl')
        _mod().collect(scratch, tmp_path / 'b.jsonl')
        assert (tmp_path / 'a.jsonl').read_bytes() == (tmp_path / 'b.jsonl').read_bytes()

    def test_it_refuses_a_failed_or_missing_batch_naming_them(self, tmp_path: Path) -> None:
        worklists = _worklists(tmp_path)
        failing, unasked = worklists[0].batch.name, worklists[-1].batch.name
        scratch = tmp_path / 'scratch'
        _run(worklists, _CountingAsker(failing=frozenset({failing})), scratch,
             only=frozenset(w.batch.name for w in worklists[:-1]))
        out = tmp_path / 'decisions.jsonl'
        with pytest.raises(_mod().ReplayRefused) as refused:
            _mod().collect(scratch, out)
        assert failing in str(refused.value)
        assert unasked in str(refused.value)
        assert not out.exists()

    def test_allow_missing_omits_the_undecided_children_and_names_their_batches(
        self, tmp_path: Path,
    ) -> None:
        worklists = _worklists(tmp_path)
        failing, unasked = worklists[0].batch.name, worklists[-1].batch.name
        scratch = tmp_path / 'scratch'
        _run(worklists, _CountingAsker(failing=frozenset({failing})), scratch,
             only=frozenset(w.batch.name for w in worklists[:-1]))
        out = tmp_path / 'decisions.jsonl'
        _mod().collect(scratch, out, allow_missing=True)
        header, *rows = _lines(out)
        assert header['failed_batches'] == [failing]
        assert header['missing_batches'] == [unasked]
        undecided = {row['entry_id'] for w in (worklists[0], worklists[-1]) for row in w.batch.rows}
        assert {row['entry_id'] for row in rows} == {
            row['entry_id'] for row in _worklist_rows()} - undecided

    def test_a_scratch_with_no_plan_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(_mod().ReplayRefused, match='plan.json'):
            _mod().collect(tmp_path / 'scratch', tmp_path / 'decisions.jsonl')


# ── the CLI, end to end ────────────────────────────────────────────────────

_E2E_PROJECTS = ('dark_factory', 'reify', 'solar_challenge', 'solar_challenge_platform')
_E2E_SHAPES = (
    (OUTCOME_CONTESTED, 21, {'conflict': 'YES'}),
    (OUTCOME_AMENDED, 11, {'home': 'NO', 'conflict': 'NO'}),
    (OUTCOME_RESTATED, 11, {'home': 'YES', 'conflict': 'NO', 'loss': 'NO'}),
)


def _e2e_inputs(root: Path) -> dict[str, Path]:
    """A corpus one Decided child over the default allocation per stratum, and its inputs."""
    items: list[dict[str, Any]] = []
    verdicts: list[dict[str, Any]] = []
    for project in _E2E_PROJECTS:
        for outcome, count, answers in _E2E_SHAPES:
            for index in range(count):
                entry = f'{outcome}-{project}-{index:02d}'
                items.append(_item(entry, outcome, project))
                verdicts.append(_vote(entry, **answers))
    root.mkdir(parents=True)
    paths = {
        'items': root / 'items.jsonl', 'verdicts': root / 'verdicts.jsonl',
        'strata': root / 'strata.json', 'texts': root / 'texts', 'touched': root / 'touched.json',
    }
    paths['items'].write_text(''.join(json.dumps(item) + '\n' for item in items))
    paths['verdicts'].write_text(''.join(json.dumps(vote) + '\n' for vote in verdicts))
    paths['strata'].write_text(json.dumps(
        {'contested-census': 473, 'amended-sample': 1224, 'restated-sample': 125}))
    paths['texts'].mkdir()
    (paths['texts'] / 'texts.json').write_text(json.dumps([
        {'basis': 'current', 'entry_id': item['entry_id'], 'target_id': item['target_id'],
         'child_text': f'child text of {item["entry_id"]}',
         'parent_text': f'parent text of {item["target_id"]}'}
        for item in items
    ]))
    paths['touched'].write_text(json.dumps([
        {'id': items[0]['entry_id'], 'state': 'missing', 'shape': 'contested'},
        {'id': 'not-in-corpus', 'state': 'relinked', 'shape': 'sighting'},
    ]))
    return paths


def _forbidden_stage(**_arguments: Any) -> Any:
    raise AssertionError('the fake arm must never reach the Stage-1 CLI')


def _env(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        'config_loader': lambda _path: FusedMemoryConfig(),
        'project_registry': lambda _config: {},
        'run_stage': _forbidden_stage,
        'usage_gate_for': lambda _config: None,
    }
    fields.update(overrides)
    return _mod().ReplayEnv(**fields)


_ROOTS = [arg for project in _E2E_PROJECTS for arg in ('--project-root', f'{project}=/srv/{project}')]


def _tree(root: Path) -> set[Path]:
    return {path for path in root.rglob('*') if path.is_file()}


class TestTheCli:
    def _main(self, *argv: Any, env: Any = None) -> int:
        return _mod().main([str(arg) for arg in argv], env=env or _env())

    def _sample(self, inputs: dict[str, Path], out: Path) -> int:
        return self._main(
            'sample', '--items', inputs['items'], '--verdicts', inputs['verdicts'],
            '--texts-root', inputs['texts'], '--touched', inputs['touched'], '--out', out,
        )

    def _run_fake(self, inputs: dict[str, Path], sample: Path, scratch: Path, *extra: Any) -> int:
        return self._main(
            'run', '--sample', sample, '--items', inputs['items'], '--texts-root', inputs['texts'],
            '--scratch', scratch, '--asker', 'fake', *extra,
        )

    def _score(self, inputs: dict[str, Path], sample: Path, decisions: Path, out: Path) -> int:
        return self._main(
            'score', '--sample', sample, '--decisions', decisions, '--items', inputs['items'],
            '--verdicts', inputs['verdicts'], '--strata', inputs['strata'],
            '--out-json', out / 'scores.json', '--out-md', out / 'scores.md',
        )

    def test_sample_run_collect_score_end_to_end_with_the_fake_arm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        out, scratch, cwd = tmp_path / 'out', tmp_path / 'scratch', tmp_path / 'cwd'
        out.mkdir()
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        before = _tree(tmp_path)
        sample, decisions = out / 'sample.jsonl', out / 'decisions.jsonl'

        assert self._sample(inputs, sample) == 0
        assert self._run_fake(inputs, sample, scratch, *_ROOTS) == 0
        assert self._main('collect', '--scratch', scratch, '--out', decisions) == 0
        assert self._score(inputs, sample, decisions, out) == 0

        header, *rows = _lines(sample)
        assert header['header'] == 'recon_triage_replay_sample'
        assert header['seed'] == 7161
        assert set(header['inputs']) == {'items', 'verdicts', 'texts', 'touched'}
        assert sum(row['headline'] for row in rows) == 160
        [touched] = [row for row in rows if row['live_recon_state'] is not None]
        assert (touched['entry_id'], touched['live_recon_state']) == (
            f'{OUTCOME_CONTESTED}-dark_factory-00', 'missing')

        _decisions_header, *decided = _lines(decisions)
        assert sorted(row['entry_id'] for row in decided) == sorted(row['entry_id'] for row in rows)
        assert {row['model'] for row in decided} == {'fake'}

        scores = json.loads((out / 'scores.json').read_text())
        assert scores['rubric'] == 'v2.2'
        assert scores['sample']['n_headline'] == 160
        assert (out / 'scores.md').read_text() == _scoring().render_markdown(scores)

        written = _tree(tmp_path) - before
        assert written
        assert all(path.is_relative_to(out) or path.is_relative_to(scratch) for path in written)
        assert not any(cwd.iterdir())

    def test_score_reruns_byte_identical_from_the_committed_files(self, tmp_path: Path) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        first, second, scratch = tmp_path / 'first', tmp_path / 'second', tmp_path / 'scratch'
        first.mkdir()
        second.mkdir()
        sample, decisions = first / 'sample.jsonl', first / 'decisions.jsonl'
        assert self._sample(inputs, sample) == 0
        assert self._run_fake(inputs, sample, scratch, *_ROOTS) == 0
        assert self._main('collect', '--scratch', scratch, '--out', decisions) == 0
        assert self._score(inputs, sample, decisions, first) == 0
        assert self._score(inputs, sample, decisions, second) == 0
        for name in ('scores.json', 'scores.md'):
            assert (first / name).read_bytes() == (second / name).read_bytes()

    def test_a_missing_input_exits_2_naming_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        inputs['items'] = tmp_path / 'in' / 'absent-items.jsonl'
        assert self._sample(inputs, tmp_path / 'sample.jsonl') == 2
        assert 'absent-items.jsonl' in capsys.readouterr().err
        assert not (tmp_path / 'sample.jsonl').exists()

    def test_a_batch_project_with_no_root_exits_2_naming_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        sample, scratch = tmp_path / 'sample.jsonl', tmp_path / 'scratch'
        assert self._sample(inputs, sample) == 0
        roots = [arg for arg in _ROOTS if not arg.startswith('reify=')]
        roots.remove('--project-root')
        assert self._run_fake(inputs, sample, scratch, *roots) == 2
        assert 'reify' in capsys.readouterr().err
        assert not scratch.exists()

    def test_collect_with_a_failed_batch_exits_2_unless_allowed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        sample, scratch = tmp_path / 'sample.jsonl', tmp_path / 'scratch'
        assert self._sample(inputs, sample) == 0
        assert self._run_fake(inputs, sample, scratch, *_ROOTS, '--only', 'reify-00') == 0
        decisions = tmp_path / 'decisions.jsonl'
        assert self._main('collect', '--scratch', scratch, '--out', decisions) == 2
        assert 'reify-01' in capsys.readouterr().err
        assert not decisions.exists()
        assert self._main('collect', '--scratch', scratch, '--out', decisions, '--allow-missing') == 0
        assert {row['batch'] for row in _lines(decisions)[1:]} == {'reify-00'}

    def test_the_stage_arm_runs_with_recons_configured_model_cwd_and_gate(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        sample, scratch = tmp_path / 'sample.jsonl', tmp_path / 'scratch'
        assert self._sample(inputs, sample) == 0
        stage = _RecordingStage(StageResult(cost_usd=0.3, error='CLI exited 1', model='sonnet'))
        gate = object()
        env = _env(run_stage=stage, usage_gate_for=lambda _config: gate,
                   mcp_candidates=[_mcp_json(tmp_path)])
        recon = FusedMemoryConfig().reconciliation
        rc = self._main(
            'run', '--sample', sample, '--items', inputs['items'], '--texts-root', inputs['texts'],
            '--scratch', scratch, '--asker', 'stage', '--only', 'reify-00', *_ROOTS, env=env,
        )
        assert rc == 1
        [call] = stage.calls
        assert call['model'] == recon.agent_llm_model
        assert call['cwd'] == Path(recon.explore_codebase_root)
        assert call['usage_gate'] is gate
        assert call['config_dir'].path.is_relative_to(scratch)
        assert 'Use project_root="/srv/reify"' in call['payload']
        assert _batch_record(scratch, 'reify-00')['error'] == 'CLI exited 1'
        assert 'reify-00' in capsys.readouterr().out

    def test_the_stage_arm_takes_the_model_and_cwd_given(self, tmp_path: Path) -> None:
        inputs = _e2e_inputs(tmp_path / 'in')
        sample, scratch = tmp_path / 'sample.jsonl', tmp_path / 'scratch'
        assert self._sample(inputs, sample) == 0
        stage = _RecordingStage(StageResult(error='CLI exited 1'))
        env = _env(run_stage=stage, mcp_candidates=[_mcp_json(tmp_path)])
        self._main(
            'run', '--sample', sample, '--items', inputs['items'], '--texts-root', inputs['texts'],
            '--scratch', scratch, '--asker', 'stage', '--only', 'reify-00', '--model', 'opus',
            '--cwd', tmp_path, *_ROOTS, env=env,
        )
        [call] = stage.calls
        assert (call['model'], call['cwd']) == ('opus', tmp_path)
