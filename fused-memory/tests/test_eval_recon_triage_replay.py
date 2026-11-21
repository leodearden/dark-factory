"""Tests for eval_recon_triage_replay.py — the read-only recon replay of write-triage attaches.

Reached only through the public surface. The scripts are loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``. Derivations come
from synthetic verdict rows through the scoring module's ``corpus_derivations``.
"""
from __future__ import annotations

import functools
import hashlib
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

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
