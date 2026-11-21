"""Tests for eval_recon_triage_replay.py — the read-only recon replay of write-triage attaches.

Reached only through the public surface. The scripts are loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``. Derivations come
from synthetic verdict rows through the scoring module's ``corpus_derivations``.
"""
from __future__ import annotations

import functools
import hashlib
import json
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.reconciliation.cli_stage_runner import (
    DISALLOW_MEMORY_WRITES,
    STAGE1_DISALLOWED,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.stage1_payload_format import format_memories
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
