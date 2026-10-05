"""Unit tests for fused_memory.middleware.metadata_dict."""

from __future__ import annotations

import logging

import pytest

from fused_memory.middleware.metadata_dict import raw_metadata_dict

_SCHEMA_WARNING = 'task_metadata.schema_warning'


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
