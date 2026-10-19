"""Tests for build_write_triage_v2_soak_baseline.py — the one-shot v2 soak baseline builder.

Synthetic rows only: the soak bundle is gitignored and lives in the main
checkout, so no test reads it. The script is loaded by path (``scripts/`` is
not a package), lazily through ``_mod()``.
"""
from __future__ import annotations

import functools
import sqlite3
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

from fused_memory.server.grouped_read import _digest

SCRIPTS = Path(__file__).parent.parent / 'scripts'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'build_write_triage_v2_soak_baseline.py',
        mod_name='build_write_triage_v2_soak_baseline',
    )


@functools.cache
def _rubric() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_reader_rubric.py', mod_name='write_triage_reader_rubric',
    )


def _adjudication(
    word: str, *, outcome: str = 'contested', new_detail: bool = True,
    dispute: str | None = None, action_changing: bool | None = None, **extra: Any,
) -> dict[str, Any]:
    return {
        'child_memory_id': 'c1', 'parent_id': 'p1', 'project': 'df',
        'category': 'procedural_knowledge', 'outcome': outcome, 'rater_verdict': word,
        'new_detail': new_detail, 'dispute': dispute, 'action_changing': action_changing,
        'likely_via_sibling': False, 'parent_edited_after_attach': False, **extra,
    }


def _determined(row: dict[str, Any]) -> dict[str, str]:
    return {str(binary): str(answer) for binary, answer in _mod().v1_determined(row).items()}


class TestV1Determined:
    @pytest.mark.parametrize(('row', 'expected'), [
        (_adjudication('SAME', new_detail=False), {'home': 'YES', 'loss': 'NO', 'conflict': 'NO'}),
        (_adjudication('SAME'), {'home': 'YES', 'conflict': 'NO'}),
        (_adjudication('SUBSUMED'), {'home': 'YES', 'conflict': 'NO'}),
        (_adjudication('EXTENDS'), {'home': 'YES'}),
        (_adjudication('RELATED'), {'home': 'NO'}),
        (_adjudication('UNRELATED'), {'home': 'NO', 'conflict': 'NO'}),
        (_adjudication('CORRECTS', dispute='genuine_correction', action_changing=True),
         {'conflict': 'YES', 'scope': 'CENTRAL'}),
        (_adjudication('CORRECTS', dispute='one_detail', action_changing=True),
         {'conflict': 'YES', 'scope': 'SECONDARY'}),
        (_adjudication('CORRECTS', dispute='genuine_correction', action_changing=False),
         {'conflict': 'NO'}),
        (_adjudication('CORRECTS', dispute='noise', action_changing=True), {'conflict': 'NO'}),
        (_adjudication('UNCLEAR', new_detail=False), {}),
    ])
    def test_follows_the_derivation_table(
        self, row: dict[str, Any], expected: dict[str, str],
    ) -> None:
        assert _determined(row) == expected


def _item(**overrides: Any) -> dict[str, Any]:
    row = {
        'entry_id': 'c1', 'outcome': 'contested', 'target_id': 'p1', 'earlier_ids': [],
        'band': 'judge', 'project': 'df', 'stratum': 'contested-census',
        'via_sibling': False, 'cross_category': False, 'parent_edited_after_attach': False,
        'supersedes_target': False, 'superseded_marks': [],
    }
    return {**row, **overrides}


class TestNeedsRating:
    @pytest.mark.parametrize(('item', 'row', 'needed'), [
        (_item(), _adjudication('CORRECTS', dispute='genuine_correction', action_changing=True),
         False),
        (_item(outcome='restated', band='restated', stratum='restated-sample'),
         _adjudication('SAME', outcome='restated', new_detail=False), False),
        (_item(outcome='restated', band='restated', stratum='restated-sample'),
         _adjudication('EXTENDS', outcome='restated'), True),
        (_item(), _adjudication('RELATED'), True),
        (_item(), _adjudication('UNCLEAR'), True),
    ])
    def test_is_true_exactly_when_the_v1_labels_leave_a_decisive_binary_open(
        self, item: dict[str, Any], row: dict[str, Any], needed: bool,
    ) -> None:
        builder = _mod()
        assert builder.needs_rating(item, builder.v1_determined(row)) is needed


def _history_db(path: Path, rows: list[tuple[str, str | None, str, str, str]]) -> Path:
    connection = sqlite3.connect(path)
    connection.execute(
        'CREATE TABLE history (id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT,'
        ' new_memory TEXT, event TEXT, created_at DATETIME, updated_at DATETIME,'
        ' is_deleted INTEGER, actor_id TEXT, role TEXT)'
    )
    for index, (memory_id, old, new, event, updated_at) in enumerate(rows):
        connection.execute(
            'INSERT INTO history VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL)',
            (f'h{index}', memory_id, old, new, event, '2026-10-01T00:00:00+00:00', updated_at),
        )
    connection.commit()
    connection.close()
    return path


class TestAttachTimeText:
    @pytest.fixture
    def history(self, tmp_path: Path) -> Path:
        return _history_db(tmp_path / 'history.db', [
            ('p1', None, 'first', 'ADD', '2026-10-01T00:00:00+00:00'),
            ('p1', 'first', 'second', 'UPDATE', '2026-10-05T00:00:00+00:00'),
            ('p1', 'second', 'third', 'UPDATE', '2026-10-09T00:00:00+00:00'),
            ('p2', 'other', 'other v2', 'UPDATE', '2026-10-03T00:00:00+00:00'),
            ('p3', None, 'lost', 'UPDATE', '2026-10-03T00:00:00+00:00'),
        ])

    @pytest.mark.parametrize(('child_created_at', 'expected'), [
        ('2026-10-02T12:00:00.000001+00:00', 'first'),
        ('2026-10-07T00:00:00+00:00', 'second'),
        ('2026-10-10T00:00:00+00:00', 'third, current'),
    ])
    def test_is_the_text_before_the_first_edit_after_the_child(
        self, history: Path, child_created_at: str, expected: str,
    ) -> None:
        assert _mod().attach_time_text(history, 'p1', child_created_at, 'third, current') == expected

    def test_an_edit_without_its_old_text_is_unrecoverable(self, history: Path) -> None:
        assert _mod().attach_time_text(history, 'p3', '2026-10-02T00:00:00+00:00', 'lost') is None


class TestAssemble:
    def _assembly(self) -> object:
        determined = {
            ('e1', 'p1'): {'home': 'YES', 'conflict': 'NO'},
            ('e2', 'p2'): {},
            ('e3', 'p3'): {'home': 'YES', 'loss': 'NO', 'conflict': 'NO'},
        }
        rubric = _rubric()
        typed = {
            pair: {rubric.Binary(binary): answer for binary, answer in answers.items()}
            for pair, answers in determined.items()
        }

        def rated(entry: str, target: str, role: str, rater: str, **answers: str) -> dict[str, Any]:
            return {
                'entry_id': entry, 'target_id': target, 'basis': answers.pop('basis', 'current'),
                'rater': rater, 'role': role, 'batch': 'shard-0', 'reason': 'two claims', **answers,
            }

        return _mod().assemble(typed, [
            rated('e1', 'p1', 'primary', 'opus', home='YES', loss='YES', conflict='NO',
                  digest='NO'),
            rated('e1', 'p1', 'second', 'sonnet', home='NO', loss='NO', conflict='YES',
                  scope='CENTRAL'),
            rated('e2', 'p2', 'primary', 'opus', home='YES', loss='NO', conflict='NO'),
            rated('e1', 'p1', 'primary', 'opus', basis='attach_time', home='YES', loss='NO',
                  conflict='YES', scope='SECONDARY'),
            rated('e3', 'p3', 'primary', 'opus', home='YES', loss='NO', conflict='NO'),
        ])

    def _rows(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        return {
            (row['entry_id'], row['basis'], row['role']): row
            for row in self._assembly().verdicts
        }

    def _answers(self, row: dict[str, Any]) -> dict[str, str]:
        return {key: row[key] for key in ('home', 'loss', 'conflict', 'scope', 'digest') if key in row}

    def test_one_derived_row_carries_each_pairs_v1_binaries(self) -> None:
        rows = self._rows()
        assert self._answers(rows['e1', 'current', 'derived']) == {'home': 'YES', 'conflict': 'NO'}
        assert self._answers(rows['e3', 'current', 'derived']) == {
            'home': 'YES', 'loss': 'NO', 'conflict': 'NO',
        }
        assert ('e2', 'current', 'derived') not in rows

    def test_rated_rows_keep_only_what_v1_left_open(self) -> None:
        rows = self._rows()
        assert self._answers(rows['e1', 'current', 'primary']) == {'loss': 'YES', 'digest': 'NO'}
        assert self._answers(rows['e1', 'current', 'second']) == {'loss': 'NO'}
        assert self._answers(rows['e2', 'current', 'primary']) == {
            'home': 'YES', 'loss': 'NO', 'conflict': 'NO',
        }
        assert ('e3', 'current', 'primary') not in rows
        assert rows['e1', 'current', 'primary']['reason'] == 'two claims'

    def test_attach_time_rows_keep_every_answer(self) -> None:
        assert self._answers(self._rows()['e1', 'attach_time', 'primary']) == {
            'home': 'YES', 'loss': 'NO', 'conflict': 'YES', 'scope': 'SECONDARY',
        }

    def test_the_corpus_is_a_valid_verdict_corpus(self) -> None:
        _rubric().resolve_judgements(self._assembly().verdicts)

    def test_answers_on_v1_binaries_feed_only_the_derivation_check(self) -> None:
        assert self._assembly().derivation == {
            'home': {'derived': 2, 'rated': 1, 'n_compared': 3, 'agree': 2},
            'loss': {'derived': 1, 'rated': 3, 'n_compared': 1, 'agree': 1},
            'conflict': {'derived': 2, 'rated': 1, 'n_compared': 3, 'agree': 2},
            'scope': {'derived': 0, 'rated': 0, 'n_compared': 0, 'agree': 0},
            'digest': {'derived': 0, 'rated': 1, 'n_compared': 0, 'agree': 0},
        }


class TestItemRow:
    @pytest.mark.parametrize(('high_band', 'band'), [(True, 'restated'), (False, 'judge'),
                                                     (None, None)])
    def test_band_comes_from_the_high_band_flag(self, high_band: bool | None, band: str | None) -> None:
        row = _mod().item_row(
            _adjudication('SAME', outcome='restated'), high_band=high_band,
            parent_category='procedural_knowledge', child_metadata={},
        )
        assert row['band'] == band

    def test_carries_the_scorer_contract(self) -> None:
        adjudication = _adjudication(
            'EXTENDS', outcome='amended', likely_via_sibling=True, parent_edited_after_attach=True,
        )
        assert _mod().item_row(
            adjudication, high_band=False, parent_category='observations_and_summaries',
            child_metadata={'supersedes': ['p1', 'p9'], 'kind': 'amendment'},
        ) == {
            'entry_id': 'c1', 'outcome': 'amended', 'target_id': 'p1', 'earlier_ids': [],
            'band': 'judge', 'project': 'df', 'stratum': 'amended-sample',
            'via_sibling': True, 'cross_category': True, 'parent_edited_after_attach': True,
            'supersedes_target': True, 'superseded_marks': [],
        }

    @pytest.mark.parametrize(('outcome', 'stratum'), [
        ('restated', 'restated-sample'), ('amended', 'amended-sample'),
        ('contested', 'contested-census'),
    ])
    def test_a_stratum_per_outcome(self, outcome: str, stratum: str) -> None:
        row = _mod().item_row(
            _adjudication('SAME', outcome=outcome), high_band=False,
            parent_category=None, child_metadata={},
        )
        assert row['stratum'] == stratum
        assert row['cross_category'] is None
        assert row['supersedes_target'] is False

    def test_the_strata_are_the_soak_populations(self) -> None:
        assert _mod().STRATA == {
            'restated-sample': 125, 'amended-sample': 1224, 'contested-census': 473,
        }


class TestShard:
    def _shard(self) -> tuple[object, str, str]:
        child = 'c' * 300 + 'tail of the child ' * 300
        parent = 'p' * 4500
        builder = _mod()
        shard = builder.shard([
            builder.ShardEntry('entry-one', 'target-one', 'current', 'df', child, parent),
            builder.ShardEntry('entry-two', 'target-two', 'attach_time', 'reify', 'short', 'old'),
        ], name='primary_00')
        return shard, child, parent

    def test_caps_each_text_and_marks_the_cut(self) -> None:
        shard, child, parent = self._shard()
        assert parent[:4000] + '…[truncated, 4500 chars total]' in shard.text
        assert parent[:4001] not in shard.text
        assert child[:4000] + f'…[truncated, {len(child)} chars total]' in shard.text

    def test_shows_the_production_digest(self) -> None:
        shard, child, _ = self._shard()
        assert f'--- CHILD DIGEST:\n{_digest(child)}\n' in shard.text
        assert '--- CHILD DIGEST:\nshort\n' in shard.text

    def test_item_ids_are_opaque_and_the_key_maps_them_back(self) -> None:
        shard, _, _ = self._shard()
        assert 'entry-one' not in shard.text and 'target-one' not in shard.text
        assert sorted(shard.key.values(), key=lambda entry: entry['entry_id']) == [
            {'entry_id': 'entry-one', 'target_id': 'target-one', 'basis': 'current'},
            {'entry_id': 'entry-two', 'target_id': 'target-two', 'basis': 'attach_time'},
        ]
        for item_id in shard.key:
            assert f'=== ITEM id={item_id}' in shard.text

    def test_an_attach_time_item_says_so(self) -> None:
        shard, _, _ = self._shard()
        assert 'basis=attach_time' in shard.text


class TestTheCli:
    def test_refuses_to_write_inside_the_bundle(self, tmp_path: Path) -> None:
        bundle = tmp_path / 'bundle'
        bundle.mkdir()
        with pytest.raises(SystemExit):
            _mod().main([
                'shards', '--bundle', str(bundle), '--history-db', str(tmp_path / 'h.db'),
                '--scratch', str(bundle / 'scratch'),
            ])
        assert list(bundle.iterdir()) == []
