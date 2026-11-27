"""Tests for scripts/legibility/census_state.py — the census-state.json
document (plans/census-incremental-prd.md §4.2): its validated reader and its
one writer.

Driven against real tmp files, so each case is pinned by what a caller sees:
which state_docs variant comes back, the typed values it carries, and what
was logged.
"""
from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime

import pytest
from legibility import census_state, state_docs

_FIVE_KEYS = {
    'last_census_at',
    'last_census_run_id',
    'last_census_report',
    'last_census_as_of_sha',
    'session_watermark',
}

_LEGACY_STATE = {
    'last_census_at': '2026-10-03',
    'last_census_report': '/home/leo/src/dark-factory/plans/confusion-census-2026-10-03.md',
    'last_census_done_count': 2872,
}

_SIX_KEY_STATE = {
    'last_census_at': '2026-10-06',
    'last_census_run_id': 'census-dark_factory-20261006',
    'last_census_report': 'plans/confusion-census-2026-10-06.md',
    'last_census_as_of_sha': 'a' * 40,
    'session_watermark': '2026-10-06T00:00:00+00:00',
    'last_census_done_count': 2900,
}

_HAND_SEEDED_DATETIME_STATE = {
    'last_census_at': '2026-10-01T09:30:00+00:00',
    'last_census_report': 'x.md',
}

_NULL_WATERMARK_STATE = {**_SIX_KEY_STATE, 'session_watermark': None}


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def _load_text(tmp_path, caplog, text: str):
    path = tmp_path / 'census-state.json'
    path.write_text(text, encoding='utf-8')
    with caplog.at_level(logging.WARNING):
        return census_state.load(path)


def _load_json(tmp_path, caplog, document: object):
    return _load_text(tmp_path, caplog, json.dumps(document))


class TestLoad:
    def test_a_missing_file_is_missing_and_logs_nothing(self, tmp_path, caplog):
        path = tmp_path / 'census-state.json'
        with caplog.at_level(logging.WARNING):
            read = census_state.load(path)
        assert read == state_docs.Missing(path)
        assert _warnings(caplog) == []

    @pytest.mark.parametrize(
        'text', ['not json{', 'null', '[]'], ids=['invalid-json', 'json-null', 'json-list']
    )
    def test_a_document_that_is_not_a_json_object_is_malformed_with_one_warning(
        self, tmp_path, caplog, text
    ):
        read = _load_text(tmp_path, caplog, text)
        assert isinstance(read, state_docs.Malformed)
        assert len(_warnings(caplog)) == 1

    @pytest.mark.parametrize(
        ('field', 'value'),
        [
            ('last_census_at', 'not-a-date'),
            ('session_watermark', 'yesterday'),
            ('session_watermark', 5),
            ('last_census_run_id', 5),
            ('last_census_run_id', ''),
        ],
    )
    def test_a_bad_field_is_malformed_with_one_warning_naming_field_and_value(
        self, tmp_path, caplog, field, value
    ):
        read = _load_json(tmp_path, caplog, {**_SIX_KEY_STATE, field: value})
        assert isinstance(read, state_docs.Malformed)
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert field in warnings[0]
        assert repr(value) in warnings[0]

    def test_a_bad_last_census_at_names_the_date_shape_it_expected(self, tmp_path, caplog):
        read = _load_json(tmp_path, caplog, {**_SIX_KEY_STATE, 'last_census_at': 'not-a-date'})
        assert isinstance(read, state_docs.Malformed)
        assert 'expected an ISO date (or a legacy ISO datetime)' in read.reason

    @pytest.mark.parametrize(
        'document',
        [_LEGACY_STATE, _SIX_KEY_STATE, _HAND_SEEDED_DATETIME_STATE, _NULL_WATERMARK_STATE, {}],
        ids=['legacy', 'six-key', 'hand-seeded-datetime', 'null-watermark', 'empty'],
    )
    def test_every_supported_shape_is_ok_and_logs_nothing(self, tmp_path, caplog, document):
        read = _load_json(tmp_path, caplog, document)
        assert isinstance(read, state_docs.Ok)
        assert isinstance(read.value, census_state.CensusState)
        assert _warnings(caplog) == []

    def test_the_six_key_state_reads_as_typed_values(self, tmp_path, caplog):
        read = _load_json(tmp_path, caplog, _SIX_KEY_STATE)
        assert read == state_docs.Ok(
            census_state.CensusState(
                last_census_at=date(2026, 10, 6),
                last_census_run_id='census-dark_factory-20261006',
                last_census_report='plans/confusion-census-2026-10-06.md',
                last_census_as_of_sha='a' * 40,
                session_watermark=datetime(2026, 10, 6, tzinfo=UTC),
            )
        )

    def test_the_legacy_state_ignores_the_done_count_and_reads_absent_keys_as_none(
        self, tmp_path, caplog
    ):
        read = _load_json(tmp_path, caplog, {**_LEGACY_STATE, 'some_future_key': [1]})
        assert read == state_docs.Ok(
            census_state.CensusState(
                last_census_at=date(2026, 10, 3),
                last_census_run_id=None,
                last_census_report=_LEGACY_STATE['last_census_report'],
                last_census_as_of_sha=None,
                session_watermark=None,
            )
        )

    def test_a_hand_seeded_datetime_reads_as_its_utc_date(self, tmp_path, caplog):
        read = _load_json(
            tmp_path, caplog, {'last_census_at': '2026-10-01T23:30:00-05:00'}
        )
        assert isinstance(read, state_docs.Ok)
        assert read.value.last_census_at == date(2026, 10, 2)

        read = _load_json(tmp_path, caplog, _HAND_SEEDED_DATETIME_STATE)
        assert isinstance(read, state_docs.Ok)
        assert read.value.last_census_at == date(2026, 10, 1)
        assert type(read.value.last_census_at) is date

    def test_a_null_watermark_reads_as_none(self, tmp_path, caplog):
        read = _load_json(tmp_path, caplog, _NULL_WATERMARK_STATE)
        assert isinstance(read, state_docs.Ok)
        assert read.value.session_watermark is None

    def test_the_watermark_is_tz_aware(self, tmp_path, caplog):
        read = _load_json(tmp_path, caplog, _SIX_KEY_STATE)
        assert isinstance(read, state_docs.Ok)
        watermark = read.value.session_watermark
        assert watermark is not None
        assert watermark.tzinfo is not None


def _advance(path, session_watermark: datetime | None) -> census_state.CensusState:
    return census_state.advance(
        path,
        census_at=date(2026, 7, 14),
        run_id='census-dark_factory-20260714',
        report_path='plans/confusion-census-2026-07-14.md',
        as_of_sha='a' * 40,
        session_watermark=session_watermark,
    )


class TestAdvance:
    def test_writes_exactly_the_five_keys(self, tmp_path):
        path = tmp_path / 'census-state.json'
        _advance(path, datetime(2026, 7, 14, tzinfo=UTC))
        data = json.loads(path.read_text(encoding='utf-8'))
        assert set(data) == _FIVE_KEYS
        assert data == {
            'last_census_at': '2026-07-14',
            'last_census_run_id': 'census-dark_factory-20260714',
            'last_census_report': 'plans/confusion-census-2026-07-14.md',
            'last_census_as_of_sha': 'a' * 40,
            'session_watermark': '2026-07-14T00:00:00+00:00',
        }

    def test_a_none_watermark_is_written_as_json_null(self, tmp_path):
        """A run with no mining window (an injected batch source) has no
        watermark; the key is still written, as null."""
        path = tmp_path / 'census-state.json'
        _advance(path, None)
        data = json.loads(path.read_text(encoding='utf-8'))
        assert set(data) == _FIVE_KEYS
        assert data['session_watermark'] is None

    @pytest.mark.parametrize('watermark', [datetime(2026, 7, 14, tzinfo=UTC), None])
    def test_load_reads_back_the_state_advance_returned(self, tmp_path, watermark):
        path = tmp_path / 'census-state.json'
        written = _advance(path, watermark)
        assert census_state.load(path) == state_docs.Ok(written)
        assert written.last_census_at == date(2026, 7, 14)
        assert written.session_watermark == watermark

    def test_creates_missing_parent_directories(self, tmp_path):
        path = tmp_path / 'docs' / 'legibility' / 'census-state.json'
        assert not path.parent.exists()
        _advance(path, None)
        assert set(json.loads(path.read_text(encoding='utf-8'))) == _FIVE_KEYS

    def test_fully_replaces_a_previous_file_and_leaves_no_temp_sibling(self, tmp_path):
        path = tmp_path / 'census-state.json'
        path.write_text(json.dumps({'stale': 'data'}), encoding='utf-8')
        _advance(path, None)
        data = json.loads(path.read_text(encoding='utf-8'))
        assert set(data) == _FIVE_KEYS
        assert [p.name for p in tmp_path.iterdir()] == ['census-state.json']
