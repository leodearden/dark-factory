"""Unit tests for the ``CALL db.indexes()`` read seam in ``falkor_indices``."""

from __future__ import annotations

import pytest
from _falkor_index_doubles import LIVE_HEADER, rows_for

from fused_memory.backends import falkor_indices
from fused_memory.backends.falkor_indices import IndexHeaderShapeError, expected_index_set


class TestIndexRecordsFromRows:
    def test_maps_a_live_header_result_by_name(self):
        expected = expected_index_set()

        records = falkor_indices.index_records_from_rows(LIVE_HEADER, rows_for(expected))

        assert records
        for record in records:
            assert set(record) == {'label', 'field', 'type', 'entity_type', 'status'}
            assert record['entity_type'] in {'NODE', 'RELATIONSHIP'}
            assert record['status'] == 'OPERATIONAL'
        assert falkor_indices.normalize_index_records(records) == expected

    def test_a_reordered_header_yields_the_same_records(self):
        rows = rows_for(expected_index_set())
        order = list(reversed(range(len(LIVE_HEADER))))
        header = [LIVE_HEADER[i] for i in order]
        reordered_rows = [[row[i] for i in order] for row in rows]

        assert falkor_indices.index_records_from_rows(
            header, reordered_rows,
        ) == falkor_indices.index_records_from_rows(LIVE_HEADER, rows)

    def test_a_header_without_status_raises(self):
        header = [column for column in LIVE_HEADER if column[1] != 'status']

        with pytest.raises(IndexHeaderShapeError):
            falkor_indices.index_records_from_rows(header, [])

    def test_a_header_without_entitytype_raises_naming_it(self):
        header = [column for column in LIVE_HEADER if column[1] != 'entitytype']

        with pytest.raises(IndexHeaderShapeError, match='entitytype'):
            falkor_indices.index_records_from_rows(header, [])

    def test_no_rows_yields_no_records(self):
        assert falkor_indices.index_records_from_rows(LIVE_HEADER, None) == []
