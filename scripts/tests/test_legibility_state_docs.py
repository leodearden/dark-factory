"""Tests for scripts/legibility/state_docs.py — the read/write seam every
legibility state document (trickle-state.json today) goes through.

Driven against real tmp files with a tiny codec, so the seam is pinned by
what a caller sees: which variant comes back, what it carries, and what was
logged.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import subprocess
from datetime import UTC, date, datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path

import pytest
from legibility import state_docs
from legibility.state_docs import Malformed, Missing, Ok

_LEGIBILITY_DIR = Path(state_docs.__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class _Doc:
    name: str
    count: int

    @classmethod
    def from_json(cls, raw: object) -> _Doc:
        doc = state_docs.json_object(raw, 'test document')
        return cls(
            name=state_docs.required(doc, 'name', state_docs.as_str),
            count=state_docs.required(doc, 'count', state_docs.as_count),
        )

    def to_json(self) -> dict:
        return {'name': self.name, 'count': self.count}


class _Colour(StrEnum):
    RED = 'red'
    BLUE = 'blue'


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


class TestReadState:
    def test_absent_path_is_missing_and_logs_nothing(self, tmp_path, caplog):
        path = tmp_path / 'absent.json'
        with caplog.at_level(logging.DEBUG):
            read = state_docs.read_state(path, _Doc.from_json)
        assert read == Missing(path)
        assert caplog.records == []

    def test_invalid_json_is_malformed_with_one_warning(self, tmp_path, caplog):
        path = tmp_path / 'broken.json'
        path.write_text('{"name": "x", ')
        with caplog.at_level(logging.WARNING):
            read = state_docs.read_state(path, _Doc.from_json)
        assert isinstance(read, Malformed)
        assert read.path == path
        assert 'JSON' in read.reason
        assert len(_warnings(caplog)) == 1

    def test_wrong_top_level_type_is_malformed_naming_it(self, tmp_path, caplog):
        path = tmp_path / 'list.json'
        path.write_text('[1, 2]')
        with caplog.at_level(logging.WARNING):
            read = state_docs.read_state(path, _Doc.from_json)
        assert isinstance(read, Malformed)
        assert 'list' in read.reason
        [warning] = _warnings(caplog)
        assert str(path) in warning.getMessage()
        assert read.reason in warning.getMessage()

    def test_bad_field_is_malformed_naming_field_and_value(self, tmp_path, caplog):
        path = tmp_path / 'bad-field.json'
        path.write_text(json.dumps({'name': 'x', 'count': 'seven'}))
        with caplog.at_level(logging.WARNING):
            read = state_docs.read_state(path, _Doc.from_json)
        assert isinstance(read, Malformed)
        assert 'count' in read.reason
        assert 'seven' in read.reason
        [warning] = _warnings(caplog)
        assert str(path) in warning.getMessage()

    def test_unreadable_path_is_malformed_never_raised(self, tmp_path, caplog):
        path = tmp_path / 'a-directory.json'
        path.mkdir()
        with caplog.at_level(logging.WARNING):
            read = state_docs.read_state(path, _Doc.from_json)
        assert isinstance(read, Malformed)
        assert read.path == path
        assert len(_warnings(caplog)) == 1

    def test_valid_document_is_ok_with_the_codec_value(self, tmp_path):
        path = tmp_path / 'good.json'
        path.write_text(json.dumps({'name': 'x', 'count': 3}))
        assert state_docs.read_state(path, _Doc.from_json) == Ok(_Doc('x', 3))


class TestWriteState:
    def test_creates_parents_and_writes_the_canonical_bytes(self, tmp_path):
        path = tmp_path / 'a' / 'b' / 'doc.json'
        state_docs.write_state(path, _Doc('x', 3))
        assert path.read_text() == (
            json.dumps({'count': 3, 'name': 'x'}, indent=2, sort_keys=True) + '\n'
        )

    def test_round_trips_through_read_state(self, tmp_path):
        path = tmp_path / 'doc.json'
        state_docs.write_state(path, _Doc('y', 0))
        assert state_docs.read_state(path, _Doc.from_json) == Ok(_Doc('y', 0))

    def test_overwrites_and_leaves_no_temp_residue(self, tmp_path):
        path = tmp_path / 'doc.json'
        state_docs.write_state(path, _Doc('first', 1))
        state_docs.write_state(path, _Doc('second', 2))
        assert sorted(p.name for p in tmp_path.iterdir()) == ['doc.json']
        assert state_docs.read_state(path, _Doc.from_json) == Ok(_Doc('second', 2))

    def test_a_fixed_name_tmp_sibling_is_neither_used_nor_consumed(self, tmp_path):
        path = tmp_path / 'doc.json'
        sentinel = tmp_path / 'doc.json.tmp'
        sentinel.write_text('someone else is writing this')
        state_docs.write_state(path, _Doc('x', 1))
        assert sentinel.read_text() == 'someone else is writing this'
        assert sorted(p.name for p in tmp_path.iterdir()) == ['doc.json', 'doc.json.tmp']
        assert state_docs.read_state(path, _Doc.from_json) == Ok(_Doc('x', 1))


class TestFieldReaders:
    def test_json_object_rejects_a_non_object_naming_what_and_type(self):
        with pytest.raises(ValueError, match=r'counters.*list'):
            state_docs.json_object([1], 'counters')

    def test_json_object_returns_the_mapping(self):
        assert state_docs.json_object({'a': 1}, 'doc') == {'a': 1}

    def test_required_field_absent_names_the_field(self):
        with pytest.raises(ValueError, match='count'):
            state_docs.required({}, 'count', state_docs.as_int)

    def test_required_field_null_names_the_field(self):
        with pytest.raises(ValueError, match='count'):
            state_docs.required({'count': None}, 'count', state_docs.as_int)

    def test_defaulted_field_absent_or_null_is_the_default(self):
        assert state_docs.defaulted({}, 'count', state_docs.as_count, 0) == 0
        assert state_docs.defaulted({'flag': None}, 'flag', state_docs.as_bool, None) is None

    def test_defaulted_field_present_is_parsed_strictly(self):
        assert state_docs.defaulted({'count': 4}, 'count', state_docs.as_count, 0) == 4
        with pytest.raises(ValueError, match=r'count.*-1'):
            state_docs.defaulted({'count': -1}, 'count', state_docs.as_count, 0)

    @pytest.mark.parametrize(
        ('parser', 'raw'),
        [
            (state_docs.as_int, True),
            (state_docs.as_int, '7'),
            (state_docs.as_int, 7.0),
            (state_docs.as_count, False),
            (state_docs.as_count, -1),
            (state_docs.as_bool, 1),
            (state_docs.as_bool, 'true'),
            (state_docs.as_str, 3),
            (state_docs.as_date, '2026-13-01'),
            (state_docs.as_date, '2026-10-01T04:00:00'),
            (state_docs.as_date, 20261001),
            (state_docs.as_datetime, 'yesterday'),
            (state_docs.as_datetime, 1_700_000_000),
        ],
    )
    def test_a_wrong_type_names_field_and_value(self, parser, raw):
        with pytest.raises(ValueError) as excinfo:
            state_docs.required({'the_field': raw}, 'the_field', parser)
        assert 'the_field' in str(excinfo.value)
        assert repr(raw) in str(excinfo.value)

    def test_valid_values_parse_to_their_types(self):
        doc = {
            'i': -3,
            'n': 0,
            'b': False,
            's': 'text',
            'd': '2026-10-01',
            't': '2026-10-01T04:00:00+02:00',
        }
        assert state_docs.required(doc, 'i', state_docs.as_int) == -3
        assert state_docs.required(doc, 'n', state_docs.as_count) == 0
        assert state_docs.required(doc, 'b', state_docs.as_bool) is False
        assert state_docs.required(doc, 's', state_docs.as_str) == 'text'
        assert state_docs.required(doc, 'd', state_docs.as_date) == date(2026, 10, 1)
        assert state_docs.required(doc, 't', state_docs.as_datetime) == datetime(
            2026, 10, 1, 4, tzinfo=timezone(timedelta(hours=2))
        )

    def test_as_member_accepts_only_the_enum_values(self):
        parse = state_docs.as_member(_Colour)
        assert state_docs.required({'c': 'red'}, 'c', parse) is _Colour.RED
        with pytest.raises(ValueError) as excinfo:
            state_docs.required({'c': 'mauve'}, 'c', parse)
        assert 'c' in str(excinfo.value)
        assert "'mauve'" in str(excinfo.value)
        assert "'red'" in str(excinfo.value)

    def test_a_naive_datetime_is_read_as_utc(self):
        stamp = state_docs.required(
            {'t': '2026-10-01T04:00:00'}, 't', state_docs.as_datetime
        )
        assert stamp == datetime(2026, 10, 1, 4, tzinfo=UTC)
        assert stamp.tzinfo is not None


def test_imports_under_a_bare_isolated_python3(tmp_path):
    """The bare-python3 predicate path (check_trickle_progress.py exec'd with
    no venv) must be able to import this module: stdlib plus shared.safe_io
    reached through the module's own sys.path bootstrap, nothing else."""
    completed = subprocess.run(
        [
            '/usr/bin/env',
            'python3',
            '-I',
            '-S',
            '-c',
            f'import sys; sys.path.insert(0, {str(_LEGIBILITY_DIR)!r}); import state_docs',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
