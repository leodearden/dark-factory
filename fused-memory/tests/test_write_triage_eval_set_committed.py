"""The committed write-triage eval set cannot drift from its committed inputs.

Built by ``scripts/build_write_triage_eval_set.py`` (``assemble`` then ``split``)
from the rounds scratch ``calibration/write_triage_eval_set_v2_rounds/`` and the
committed live slates. The invariants below hold whatever the ratings say: no
rate and no kappa value is pinned. The class runs on one xdist worker, so the
cached loads and derivations are paid once per run.
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
STEM = 'write_triage_eval_set_v2'
OFFLINE_JUDGED_WRITES = 484
"""The production-judged writes of the 658-write offline write-time draw (no recon marker, no attach keys)."""


@functools.cache
def _build() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'build_write_triage_eval_set.py',
                              mod_name='build_write_triage_eval_set')


@functools.cache
def _rubric() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_reader_rubric.py',
                              mod_name='write_triage_reader_rubric')


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_rating_rounds.py',
                              mod_name='write_triage_rating_rounds')


@functools.cache
def _score() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'score_write_triage_reader.py',
                              mod_name='score_write_triage_reader')


@functools.cache
def _cross() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'rate_write_triage_cross_family.py',
                              mod_name='rate_write_triage_cross_family')


@functools.cache
def _pairs() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'score_write_triage_pairs.py',
                              mod_name='score_write_triage_pairs')


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@functools.cache
def _writes() -> list[dict[str, Any]]:
    return _jsonl_rows(CALIBRATION / f'{STEM}_writes.jsonl')


@functools.cache
def _verdicts() -> list[dict[str, Any]]:
    return _jsonl_rows(CALIBRATION / f'{STEM}_verdicts.jsonl')


@functools.cache
def _summary() -> dict[str, Any]:
    return json.loads((CALIBRATION / f'{STEM}_summary.json').read_text())


@functools.cache
def _split() -> dict[str, Any]:
    return json.loads((CALIBRATION / f'{STEM}_split.json').read_text())


@functools.cache
def _key() -> dict[str, Any]:
    return json.loads((CALIBRATION / f'{STEM}_rounds' / 'key.json').read_text())


@functools.cache
def _cross_records() -> list[dict[str, Any]]:
    return _jsonl_rows(CALIBRATION / f'{STEM}_cross_family.jsonl')


@functools.cache
def _cross_report() -> dict[str, Any]:
    return json.loads((CALIBRATION / f'{STEM}_cross_family.json').read_text())


@functools.cache
def _derivations() -> dict[tuple[str, str], Any]:
    return _build().pair_derivations(_writes(), _verdicts())


def _current_rows_by_pair() -> defaultdict[tuple[str, str], list[dict[str, Any]]]:
    rows: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in _verdicts():
        if row['basis'] == 'current':
            rows[row['entry_id'], row['target_id']].append(row)
    return rows


@pytest.mark.xdist_group('committed_write_triage_eval_set')
class TestTheCommittedEvalSet:

    def test_every_pair_is_decided_or_unclear(self) -> None:
        rubric = _rubric()
        derivations = _derivations()
        assert set(derivations) == {(row['entry_id'], target['target_id'])
                                    for row in _writes() for target in row['targets']}
        incomplete = {pair: derivation for pair, derivation in derivations.items()
                      if not isinstance(derivation, rubric.Decided | rubric.Unclear)}
        assert incomplete == {}

    def test_every_pair_has_one_primary_and_one_second_line(self) -> None:
        roles = {pair: Counter(row['role'] for row in rows)
                 for pair, rows in _current_rows_by_pair().items()}
        assert set(roles) == set(_derivations())
        assert {pair: dict(counted) for pair, counted in roles.items()
                if counted['primary'] != 1 or counted['second'] != 1} == {}
        assert {row['role'] for row in _verdicts()} <= {'primary', 'second', 'third'}
        assert {row['basis'] for row in _verdicts()} == {'current'}

    def test_the_verdicts_are_the_effective_lines_of_the_committed_ratings(self) -> None:
        def order(row: dict[str, Any]) -> tuple[str, str, str, str]:
            return (row['entry_id'], row['target_id'], row['basis'], row['rater'])

        lines = _rounds().RatingScratch(CALIBRATION / f'{STEM}_rounds').effective_lines(_key())
        verdicts = [{key: value for key, value in row.items() if key != 'digest'} for row in _verdicts()]
        assert sorted(lines, key=order) == sorted(verdicts, key=order)

    def test_every_loss_yes_line_carries_digest(self) -> None:
        assert [row for row in _verdicts()
                if row.get('loss') == 'YES' and row.get('digest') not in ('YES', 'NO')] == []

    def test_recomputing_kappa_labels_and_volume_reproduces_the_summary(self) -> None:
        rubric, summary = _rubric(), _summary()
        decided = {pair: d for pair, d in _derivations().items() if isinstance(d, rubric.Decided)}
        unclear = {pair: d for pair, d in _derivations().items() if isinstance(d, rubric.Unclear)}
        consumed = [
            (rubric.PairKey(entry_id, target_id, rubric.Basis.CURRENT), binary)
            for (entry_id, target_id), derivation in decided.items()
            for _, binary in derivation.consumed
        ]
        assert summary['kappa'] == json.loads(json.dumps(
            _score().kappa_over_consumed(consumed, rubric.resolve_judgements(_verdicts()))))
        assert summary['labels'] == {'decided': len(decided), 'unclear': len(unclear)}
        assert {(pair['entry_id'], pair['target_id']) for pair in summary['unclear_pairs']} == set(unclear)
        assert summary['rating_volume'] == _rounds().rating_volume(_key())
        assert summary['pairs'] == dict(Counter(
            row['source'] for row in _writes() for _ in row['targets']))

    def test_the_key_records_the_inputs_the_summary_was_assembled_from(self) -> None:
        assert _key()['inputs'] == _summary()['inputs']

    def test_the_summary_pins_the_committed_live_slates(self) -> None:
        body = (CALIBRATION / f'{STEM}_live_slates.jsonl').read_bytes()
        assert _summary()['inputs']['live_slates_sha256'] == hashlib.sha256(body).hexdigest()
        header = json.loads(body.decode().splitlines()[0])
        live = Counter(row['project'] for row in _writes() if row['source'] == 'live')
        assert dict(live) == header['allocation']

    def test_the_frozen_split_verifies_and_partitions_the_writes(self) -> None:
        build, split = _build(), _split()
        build.verify_split(split, _writes(), _verdicts())
        members = [entry_id for name in build.SPLITS for entry_id in split['splits'][name]]
        assert len(members) == len(set(members))
        assert set(members) == {row['entry_id'] for row in _writes()}
        test = set(split['splits']['test'])
        held_out = {row['entry_id'] for row in _writes()
                    if row['source'] == 'live' and row['project'] == build.HELD_OUT_PROJECT}
        assert held_out <= test
        assert [row['entry_id'] for row in _writes()
                if row['source'] == 'offline' and row['entry_id'] in test] == []

    def test_the_offline_writes_are_the_production_judged_write_time_draw(self) -> None:
        offline = [row for row in _writes() if row['source'] == 'offline']
        assert len(offline) == OFFLINE_JUDGED_WRITES
        assert {row['stratum'] for row in offline} == {_build().OFFLINE_STRATUM}
        population = json.loads((CALIBRATION / 'write_triage_population_write_time.json').read_text())
        assert _summary()['inputs']['snapshot_sha256'] == population['population']['snapshot_sha256']


@pytest.mark.xdist_group('committed_write_triage_eval_set')
class TestTheCommittedRerates:
    """Every re-rate the rounds key records is the one the summary reports and the verdicts rest on."""

    def test_the_summary_reports_each_keyed_rerate(self) -> None:
        assert _summary()['rerates'] == [
            {'version': rerate['version'], 'round': rerate['round'],
             'pairs': len(rerate['pairs']), 'brief_sha256': rerate['brief_sha256']}
            for rerate in _key().get('rerates', ())
        ]

    def test_the_latest_rerate_was_keyed_under_the_committed_brief(self) -> None:
        rerates = _key().get('rerates', ())
        if not rerates:
            pytest.skip('no re-rate is keyed')
        brief = CALIBRATION / 'write_triage_rater_brief_v2.md'
        assert rerates[-1]['brief_sha256'] == hashlib.sha256(brief.read_bytes()).hexdigest()

    def test_a_rerated_pairs_double_lines_come_from_its_rerate_round_or_later(self) -> None:
        shards = _key()['shards']
        rows = _current_rows_by_pair()
        earlier = []
        for rerate in _key().get('rerates', ()):
            for entry_id, target_id in rerate['pairs']:
                earlier += [
                    (row['entry_id'], row['target_id'], row['role'], row['batch'])
                    for row in rows[entry_id, target_id]
                    if row['role'] in ('primary', 'second')
                    and shards[row['batch']]['round'] < rerate['round']
                ]
        assert earlier == []

    def test_every_pair_that_reads_loss_was_rerated_under_the_latest_brief(self) -> None:
        rubric = _rubric()
        rerates = _key().get('rerates', ())
        if not rerates:
            pytest.skip('no re-rate is keyed')
        reads_loss = {
            pair for pair, derivation in _derivations().items()
            if isinstance(derivation, rubric.Decided)
            and any(binary == rubric.Binary.LOSS for _, binary in derivation.consumed)
            or isinstance(derivation, rubric.Unclear)
            and any(binary == rubric.Binary.LOSS for _, binary in derivation.binaries)
        }
        assert reads_loss - {tuple(pair) for pair in rerates[-1]['pairs']} == set()

    def test_no_retired_third_rater_votes_on_a_rerated_pair(self) -> None:
        rows = _current_rows_by_pair()
        retired = [
            (row['entry_id'], row['target_id'], row['rater'])
            for rerate in _key().get('rerates', ())
            for entry_id, target_id in rerate['pairs']
            for row in rows[entry_id, target_id]
            if row['rater'] in rerate['retires_third_raters']
        ]
        assert retired == []


@pytest.mark.xdist_group('committed_write_triage_eval_set')
class TestTheCommittedCrossFamilyRerate:
    """The committed cross-family records are the report's only input, and cover every discordant pair."""

    def test_reporting_the_records_reproduces_the_committed_report(self) -> None:
        v1 = _jsonl_rows(CALIBRATION / 'write_triage_pair_verdicts.jsonl')
        built = _cross().report(_cross_records(), _writes(), _verdicts(), v1)
        assert json.loads(json.dumps(built)) == _cross_report()

    def test_the_markdown_is_the_rendered_report(self) -> None:
        rendered = _cross().render_markdown(_cross_report())
        assert (CALIBRATION / f'{STEM}_cross_family.md').read_text() == rendered

    def test_every_discordant_pair_has_a_row_or_a_recorded_failure(self) -> None:
        cross = _cross()
        records = {(record['entry_id'], record['target_id']): record for record in _cross_records()}
        assert len(records) == len(_cross_records())
        discordant = set(cross.discordant_pairs(_writes(), _verdicts()))
        assert discordant - set(records) == set()
        assert {pair for pair, record in records.items()
                if record['selection'] == cross.SELECTION_DISCORDANT} == discordant
        controls = [pair for pair, record in records.items()
                    if record['selection'] == cross.SELECTION_CONTROL]
        assert set(controls) <= set(cross.control_pairs(_writes(), _verdicts(), seed=7160))
        assert [pair for pair, record in records.items()
                if (record['row'] is None) == (record['failure'] is None)] == []

    def test_every_row_is_cross_family_and_names_its_model_and_effort(self) -> None:
        for record in _cross_records():
            model, _, effort = record['rater'].partition(':')
            assert model in _pairs().LIST_PRICES and effort, record['rater']
            if record['row'] is not None:
                assert record['row']['role'] == 'cross_family'
                assert record['row']['rater'] == record['rater']
