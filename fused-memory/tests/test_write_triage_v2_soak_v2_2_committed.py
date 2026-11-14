"""The committed v2.2 soak report and its rating rounds cannot drift from their committed inputs.

Built by ``scripts/build_write_triage_v2_soak_baseline.py`` (``rerate
--brief-version v2.2`` then ``assemble``) from the rounds scratch
``calibration/write_triage_v2_soak_rounds/``, the v2.1 soak rounds of task
7174 with the soak's LOSS-reading pairs re-rated under rater brief v2.2, and
scored by ``scripts/score_write_triage_reader.py --rubric v2.2``. The
invariants below hold whatever the ratings say: no rate and no kappa value is
pinned. The classes run on one xdist worker, so the cached loads are paid once
per run.
"""
from __future__ import annotations

import functools
import hashlib
import json
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
CALIBRATION = Path(__file__).parent.parent / 'calibration'
STEM = 'write_triage_v2_soak'
ROUNDS = CALIBRATION / f'{STEM}_rounds'


@functools.cache
def _soak() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'build_write_triage_v2_soak_baseline.py',
                              mod_name='build_write_triage_v2_soak_baseline')


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_rating_rounds.py',
                              mod_name='write_triage_rating_rounds')


@functools.cache
def _score() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'score_write_triage_reader.py',
                              mod_name='score_write_triage_reader')


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _json(name: str) -> Any:
    return json.loads((CALIBRATION / f'{STEM}_{name}').read_text())


@functools.cache
def _items() -> list[dict[str, Any]]:
    return _jsonl_rows(CALIBRATION / f'{STEM}_items.jsonl')


@functools.cache
def _verdicts() -> list[dict[str, Any]]:
    return _jsonl_rows(CALIBRATION / f'{STEM}_verdicts_v2_2.jsonl')


@functools.cache
def _v2_1_verdicts() -> list[dict[str, Any]]:
    return _jsonl_rows(CALIBRATION / f'{STEM}_verdicts.jsonl')


@functools.cache
def _derivation() -> dict[str, Any]:
    return _json('derivation_v2_2.json')


@functools.cache
def _key() -> dict[str, Any]:
    return json.loads((ROUNDS / 'key.json').read_text())


@functools.cache
def _report() -> dict[str, Any]:
    return _json('baseline_v2_2.json')


def _rerate() -> dict[str, Any]:
    [rerate] = _key()['rerates']
    return rerate


def _order(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (row['entry_id'], row['target_id'], row['basis'], row['rater'])


def _current_rows_by_pair() -> dict[tuple[str, str], list[dict[str, Any]]]:
    rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _verdicts():
        if row['basis'] == 'current':
            rows[row['entry_id'], row['target_id']].append(row)
    return rows


@pytest.mark.xdist_group('committed_v2_soak_v2_2')
class TestTheCommittedV22SoakReport:

    def test_rescoring_the_inputs_under_v2_2_reproduces_the_report(self) -> None:
        rescored = _score().score_reader(_items(), _verdicts(), rubric='v2.2',
                                         strata=_json('strata.json'))
        assert rescored == _report()
        assert _report()['rubric'] == 'v2.2'

    def test_the_markdown_is_the_rendered_report(self) -> None:
        markdown = (CALIBRATION / f'{STEM}_baseline_v2_2.md').read_text()
        assert markdown == _score().render_markdown(_report())

    def test_the_rater_rows_are_the_effective_lines_of_the_committed_ratings(self) -> None:
        lines = _rounds().RatingScratch(ROUNDS).effective_lines(_key())
        rater_rows = [{key: value for key, value in row.items() if key not in ('digest', 'non_consumed')}
                      for row in _verdicts() if row['role'] != 'derived']
        assert sorted(rater_rows, key=_order) == sorted(lines, key=_order)

    def test_the_derived_rows_and_held_rows_are_the_v2_1_baselines(self) -> None:
        assert [row for row in _verdicts() if row['role'] == 'derived'] == [
            row for row in _v2_1_verdicts() if row['role'] == 'derived']
        v2_1 = _json('derivation.json')
        assert _derivation()['rows'] == v2_1['rows']
        assert _derivation()['held_rows'] == v2_1['held_rows']

    def test_every_current_pair_has_one_primary_and_one_second_row(self) -> None:
        roles = {pair: Counter(row['role'] for row in rows) for pair, rows in _current_rows_by_pair().items()}
        assert set(roles) == {(item['entry_id'], item['target_id']) for item in _items()}
        assert {pair: dict(counted) for pair, counted in roles.items()
                if counted['primary'] != 1 or counted['second'] != 1} == {}


@pytest.mark.xdist_group('committed_v2_soak_v2_2')
class TestTheCommittedV22Rerate:

    def test_every_round_0_double_shard_records_brief_v2_1(self) -> None:
        briefs = {spec['brief_sha256'] for spec in _key()['shards'].values()
                  if spec['round'] == 0 and spec['role'] in ('primary', 'second')}
        assert briefs == {_rounds().BRIEF_V2_1_SHA256}

    def test_one_v2_2_rerate_under_the_committed_brief_which_the_derivation_reports(self) -> None:
        rerate = _rerate()
        brief = CALIBRATION / 'write_triage_rater_brief_v2.md'
        assert rerate['version'] == 'v2.2'
        assert rerate['brief_sha256'] == hashlib.sha256(brief.read_bytes()).hexdigest()
        assert _derivation()['rerates'] == [
            {'version': 'v2.2', 'round': rerate['round'], 'pairs': len(rerate['pairs']),
             'brief_sha256': rerate['brief_sha256']}]
        assert _derivation()['check']['brief_sha256'] == _rounds().BRIEF_V2_1_SHA256

    def test_every_pair_reading_loss_was_rerated(self) -> None:
        reads_loss = _rounds().pairs_reading_loss(_soak().current_derivations(_items(), _verdicts()))
        assert reads_loss - {tuple(pair) for pair in _rerate()['pairs']} == set()

    def test_no_retired_third_rater_votes_on_a_rerated_pair(self) -> None:
        rerate, rows = _rerate(), _current_rows_by_pair()
        assert [(entry_id, target_id, row['rater'])
                for entry_id, target_id in rerate['pairs'] for row in rows[entry_id, target_id]
                if row['rater'] in rerate['retires_third_raters']] == []

    def test_a_rerated_pairs_double_rows_come_from_its_rerate_round_or_later(self) -> None:
        rerate, rows, shards = _rerate(), _current_rows_by_pair(), _key()['shards']
        assert [(entry_id, target_id, row['role'], row['batch'])
                for entry_id, target_id in rerate['pairs'] for row in rows[entry_id, target_id]
                if row['role'] in ('primary', 'second')
                and shards[row['batch']]['round'] < rerate['round']] == []
