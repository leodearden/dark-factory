"""Tests for backfill_rating_shard_kinds.py — each legacy shard spec's kind and basis, derived.

Synthetic legacy keys in tmp only. The module is loaded by path
(``scripts/`` is not a package), lazily through ``_mod()``.
"""
from __future__ import annotations

import copy
import functools
import json
import re
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import load_script_module

SCRIPTS = Path(__file__).parent.parent / 'scripts'
SHA = 'a' * 64


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'backfill_rating_shard_kinds.py',
                              mod_name='backfill_rating_shard_kinds')


@functools.cache
def _rounds() -> types.ModuleType:
    return load_script_module(SCRIPTS / 'write_triage_rating_rounds.py',
                              mod_name='write_triage_rating_rounds')


def _spec(role: str, round_: int, *items: str) -> dict[str, Any]:
    rater = 'opus-third-00' if role == 'third' else f'opus-{role}'
    return {'role': role, 'rater': rater, 'round': round_, 'items': list(items), 'brief_sha256': SHA}


def _item(entry: str, basis: str = 'current') -> dict[str, str]:
    return {'entry_id': entry, 'target_id': 'p1', 'basis': basis}


def _legacy_key() -> dict[str, Any]:
    """Two initial shards (one attach-time only), a reask, a third and a rerate shard; no kind or basis."""
    return {
        'shards': {
            'primary_00': _spec('primary', 0, 'primary_00-000'),
            'attach_primary_00': _spec('primary', 0, 'attach_primary_00-000'),
            'reask_02_primary': _spec('primary', 2, 'reask_02_primary-000'),
            'third_00': _spec('third', 0, 'third_00-000'),
            'rerate_01_primary_00': _spec('primary', 1, 'rerate_01_primary_00-000'),
        },
        'items': {
            'primary_00-000': _item('e0'),
            'attach_primary_00-000': _item('e0', 'attach_time'),
            'reask_02_primary-000': _item('e1'),
            'third_00-000': _item('e2'),
            'rerate_01_primary_00-000': _item('e3'),
        },
        'rerates': [{'version': 'v2.2', 'round': 1, 'brief_sha256': SHA, 'pairs': [['e3', 'p1']],
                     'shards': ['rerate_01_primary_00'], 'retires_third_raters': ['opus-third-00']}],
    }


EXPECTED = {
    'primary_00': ('initial', 'current'),
    'attach_primary_00': ('initial', 'attach_time'),
    'reask_02_primary': ('reask', 'current'),
    'third_00': ('third', 'current'),
    'rerate_01_primary_00': ('rerate', 'current'),
}


def _kinds(key: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {name: (spec['kind'], spec['basis']) for name, spec in key['shards'].items()}


class TestBackfilled:
    def test_derives_each_kind_and_basis_from_structured_fields_leaving_the_input_unchanged(
        self,
    ) -> None:
        legacy = _legacy_key()
        before = copy.deepcopy(legacy)
        backfilled = _mod().backfilled(legacy)
        assert legacy == before
        assert _kinds(backfilled) == EXPECTED
        for name, spec in backfilled['shards'].items():
            assert {key: value for key, value in spec.items() if key not in ('kind', 'basis')} == (
                legacy['shards'][name])
        assert {key: value for key, value in backfilled.items() if key != 'shards'} == {
            key: value for key, value in legacy.items() if key != 'shards'}

    def test_the_derivation_reads_no_shard_name(self) -> None:
        legacy = _legacy_key()
        legacy['shards']['primary_07'] = legacy['shards'].pop('reask_02_primary')
        assert _kinds(_mod().backfilled(legacy))['primary_07'] == ('reask', 'current')

    def test_derived_kind_and_basis_answer_per_shard(self) -> None:
        legacy = _legacy_key()
        assert _mod().derived_kind(legacy, 'rerate_01_primary_00') == _rounds().ShardKind.RERATE
        assert _mod().derived_basis(legacy, 'attach_primary_00') == 'attach_time'

    def test_a_spec_whose_items_mix_bases_is_refused_naming_it_and_the_bases(self) -> None:
        legacy = _legacy_key()
        legacy['shards']['primary_00']['items'].append('attach_primary_00-000')
        with pytest.raises(ValueError, match=r"primary_00.*\['attach_time', 'current'\]"):
            _mod().backfilled(legacy)

    def test_a_role_outside_primary_second_and_third_is_refused_naming_the_shard(self) -> None:
        legacy = _legacy_key()
        legacy['shards']['cross_family_00'] = _spec('cross_family', 0, 'primary_00-000')
        with pytest.raises(ValueError, match='cross_family_00'):
            _mod().backfilled(legacy)

    def test_a_recorded_kind_or_basis_equal_to_its_derivation_is_kept(self) -> None:
        legacy = _legacy_key()
        legacy['shards']['third_00'].update(kind='third', basis='current')
        assert _kinds(_mod().backfilled(legacy)) == EXPECTED

    @pytest.mark.parametrize(('field', 'recorded', 'derived'), [
        ('kind', 'reask', 'third'), ('basis', 'attach_time', 'current'),
    ])
    def test_a_recorded_value_that_disagrees_is_refused_naming_shard_recorded_and_derived(
        self, field: str, recorded: str, derived: str,
    ) -> None:
        legacy = _legacy_key()
        legacy['shards']['third_00'][field] = recorded
        with pytest.raises(ValueError, match=rf'third_00.*{recorded}.*{derived}'):
            _mod().backfilled(legacy)


class TestTheCli:
    @staticmethod
    def _key_file(tmp_path: Path) -> Path:
        path = tmp_path / 'rounds' / 'key.json'
        path.parent.mkdir()
        path.write_text(json.dumps(_legacy_key()))
        return path

    def test_rewrites_each_key_in_place_as_the_scratch_saves_it(self, tmp_path: Path) -> None:
        path = self._key_file(tmp_path)
        expected = tmp_path / 'expected' / 'key.json'
        _rounds().RatingScratch(expected.parent).save_key(_mod().backfilled(_legacy_key()))
        assert _mod().main([str(path)]) == 0
        assert path.read_bytes() == expected.read_bytes()

    def test_check_reports_a_key_needing_a_change_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = self._key_file(tmp_path)
        before = path.read_bytes()
        assert _mod().main(['--check', str(path)]) == 1
        assert str(path) in capsys.readouterr().err
        assert path.read_bytes() == before

    @staticmethod
    def _mixed_key_file(tmp_path: Path) -> Path:
        mixed = _legacy_key()
        mixed['shards']['primary_00']['items'].append('attach_primary_00-000')
        path = tmp_path / 'mixed' / 'key.json'
        path.parent.mkdir()
        path.write_text(json.dumps(mixed))
        return path

    @pytest.mark.parametrize('check', [True, False])
    def test_a_refused_key_is_named_with_its_refusal_and_every_other_key_still_handled(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], check: bool,
    ) -> None:
        mixed, legacy = self._mixed_key_file(tmp_path), self._key_file(tmp_path)
        mixed_before, legacy_before = mixed.read_bytes(), legacy.read_bytes()
        assert _mod().main([*(['--check'] if check else []), str(mixed), str(legacy)]) == 1
        err = capsys.readouterr().err
        assert re.search(rf"{re.escape(str(mixed))}: shard primary_00: .*\['attach_time', 'current'\]",
                         err)
        assert mixed.read_bytes() == mixed_before
        assert (str(legacy) in err) is check
        assert (legacy.read_bytes() == legacy_before) is check

    def test_a_path_that_is_not_a_key_is_refused_before_any_key_is_written(
        self, tmp_path: Path,
    ) -> None:
        legacy = self._key_file(tmp_path)
        before = legacy.read_bytes()
        with pytest.raises(SystemExit) as exited:
            _mod().main([str(legacy), str(tmp_path / 'missing' / 'key.json')])
        assert exited.value.code == 2
        assert legacy.read_bytes() == before

    def test_check_passes_a_backfilled_key(self, tmp_path: Path) -> None:
        path = self._key_file(tmp_path)
        assert _mod().main([str(path)]) == 0
        after = path.read_bytes()
        assert _mod().main(['--check', str(path)]) == 0
        assert path.read_bytes() == after
