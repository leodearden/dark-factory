"""Tests for shared.task_metadata_wire — the one rule for a task's ``metadata`` wire value.

The contract is a tri-state: ``{}`` for absent (``None`` or exactly ``''``),
the dict itself for readable metadata, and ``None`` only for metadata that is
present but unreadable. Every case below must return, never raise.
"""
from __future__ import annotations

import pytest

from shared.task_metadata_wire import coerce_task_metadata


class TestAbsent:
    @pytest.mark.parametrize('raw', [None, ''])
    def test_absent_is_empty_dict(self, raw):
        assert coerce_task_metadata(raw) == {}

    def test_absent_result_is_fresh_not_shared(self):
        first = coerce_task_metadata(None)
        second = coerce_task_metadata(None)
        assert first is not None
        assert first is not second
        first['k'] = 1
        assert coerce_task_metadata(None) == {}
        assert coerce_task_metadata('') == {}


class TestReadable:
    def test_dict_is_returned_as_the_same_object(self):
        d = {'a': 1}
        assert coerce_task_metadata(d) is d

    def test_empty_dict_is_a_dict(self):
        assert coerce_task_metadata({}) == {}

    def test_json_object_string_is_decoded(self):
        assert coerce_task_metadata('{"a": 1, "files": ["x.py"]}') == {
            'a': 1,
            'files': ['x.py'],
        }


class TestUnreadable:
    @pytest.mark.parametrize(
        'raw', ['{not json', '   ', '[1,2]', '"x"', '42', 'null', 'true']
    )
    def test_string_that_is_not_a_json_object_is_none(self, raw):
        assert coerce_task_metadata(raw) is None

    @pytest.mark.parametrize(
        'raw', [[], ['cross_repo'], (), 42, 3.5, True, False, object(), b'{}']
    )
    def test_non_string_non_dict_is_none(self, raw):
        assert coerce_task_metadata(raw) is None
