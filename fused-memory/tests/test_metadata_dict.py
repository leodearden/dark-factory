"""Unit tests for fused_memory.middleware.metadata_dict."""

from __future__ import annotations

import logging

import pytest
from shared.task_metadata_wire import coerce_task_metadata

from fused_memory.middleware.metadata_dict import raw_metadata_dict

_SCHEMA_WARNING = 'task_metadata.schema_warning'

_WIRE_SHAPES = [None, '', {}, {'a': 1}, '{"a": 1}', '{not json', '[1]', '42', '""', '   ', 42, 0, False, 1.5, [], ['a']]
_UNREADABLE = [shape for shape in _WIRE_SHAPES if coerce_task_metadata(shape) is None]


def _schema_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if _SCHEMA_WARNING in r.getMessage()]


class TestRawMetadataDict:
    def test_dict_is_returned_as_is(self):
        metadata = {'files': ['a.py'], 'x_private': 1}
        assert raw_metadata_dict(metadata, source='t') is metadata

    def test_json_object_string_is_decoded_raw(self):
        decoded = raw_metadata_dict('{"files": ["a.py"], "spawned_from": "7"}', source='t')
        assert decoded == {'files': ['a.py'], 'spawned_from': '7'}

    @pytest.mark.parametrize('metadata', [None, ''])
    def test_absent_metadata_is_an_empty_dict_without_a_warning(self, metadata, caplog):
        with caplog.at_level(logging.WARNING):
            assert raw_metadata_dict(metadata, source='t') == {}
        assert _schema_warnings(caplog) == []

    @pytest.mark.parametrize('metadata', ['{not json', '["a.py"]', '42'])
    def test_malformed_string_is_discarded_with_one_schema_warning(self, metadata, caplog):
        with caplog.at_level(logging.WARNING):
            assert raw_metadata_dict(metadata, source='my_guard') == {}
        warnings = _schema_warnings(caplog)
        assert len(warnings) == 1
        assert 'source=my_guard' in warnings[0]

    @pytest.mark.parametrize('metadata', _WIRE_SHAPES, ids=repr)
    def test_agrees_with_coerce_task_metadata_on_every_wire_shape(self, metadata, caplog):
        expected = coerce_task_metadata(metadata)
        with caplog.at_level(logging.WARNING):
            resolved = raw_metadata_dict(metadata, source='t')
        assert resolved == (expected if expected is not None else {})
        assert len(_schema_warnings(caplog)) == (1 if expected is None else 0)

    def test_empty_dict_is_returned_as_is(self):
        metadata: dict = {}
        assert raw_metadata_dict(metadata, source='t') is metadata

    @pytest.mark.parametrize('metadata', _UNREADABLE, ids=repr)
    def test_unreadable_warning_names_source_and_type_with_a_real_diagnosis(self, metadata, caplog):
        with caplog.at_level(logging.WARNING):
            raw_metadata_dict(metadata, source='reader_x')
        (warning,) = _schema_warnings(caplog)
        assert 'source=reader_x' in warning
        assert f'type={type(metadata).__name__}' in warning
        assert 'unrecognised shape' not in warning
