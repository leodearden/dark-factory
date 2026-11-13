"""Unit tests for the ``CALL db.indexes()`` read seam in ``falkor_indices``."""

from __future__ import annotations

import itertools

import pytest
import redis.exceptions
from _falkor_index_doubles import LIVE_HEADER, rows_for, statements_read, statements_written

from fused_memory.backends import falkor_indices
from fused_memory.backends.falkor_indices import (
    INDEX_CATALOG_QUERY,
    IndexCatalogUnsettledError,
    IndexHeaderShapeError,
    IndexRecordShapeError,
    expected_index_set,
)

_UNDER_CONSTRUCTION = '[Indexing] 12/50: UNDER CONSTRUCTION'


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


@pytest.mark.asyncio
class TestReadIndexRecords:
    async def test_reads_the_catalog_on_the_read_only_path(self, make_graph_mock):
        rows = rows_for(expected_index_set())
        graph = make_graph_mock(rows, header=LIVE_HEADER)

        records = await falkor_indices.read_index_records(graph)

        assert records == falkor_indices.index_records_from_rows(LIVE_HEADER, rows)
        assert statements_read(graph) == [INDEX_CATALOG_QUERY]
        assert statements_written(graph) == []

    async def test_an_empty_catalog_yields_no_records(self, make_graph_mock):
        graph = make_graph_mock(header=LIVE_HEADER)

        assert await falkor_indices.read_index_records(graph) == []


def _settled_records():
    return falkor_indices.index_records_from_rows(LIVE_HEADER, rows_for(expected_index_set()))


def _unsettled_records():
    records = _settled_records()
    records[0] = {**records[0], 'status': _UNDER_CONSTRUCTION}
    return records


class _CountingReader:
    def __init__(self, reads):
        self._reads = iter(reads)
        self.calls = 0

    async def __call__(self):
        self.calls += 1
        read = next(self._reads)
        if isinstance(read, BaseException):
            raise read
        return read


async def _settle(reader, *, timeout_s=1.0, interval=0.001):
    return await falkor_indices.await_index_catalog_settled(
        reader, graph_name='scratch_graph', timeout_s=timeout_s, interval=interval,
    )


@pytest.mark.asyncio
class TestAwaitIndexCatalogSettled:
    async def test_a_settled_first_read_is_returned_after_one_read(self):
        settled = _settled_records()
        reader = _CountingReader([settled])

        assert await _settle(reader) == settled
        assert reader.calls == 1

    async def test_returns_the_certified_read_once_the_catalog_settles(self):
        unsettled, settled = _unsettled_records(), _settled_records()
        reader = _CountingReader([unsettled, settled])

        records = await _settle(reader)

        assert records is settled
        assert reader.calls == 2

    async def test_an_empty_catalog_counts_as_settled(self):
        reader = _CountingReader([[]])

        assert await _settle(reader) == []
        assert reader.calls == 1

    async def test_a_catalog_that_never_settles_raises_naming_what_is_unsettled(self):
        unsettled = _unsettled_records()
        reader = _CountingReader(itertools.repeat(unsettled))

        with pytest.raises(IndexCatalogUnsettledError) as excinfo:
            await _settle(reader, timeout_s=0.01, interval=0.001)

        message = str(excinfo.value)
        assert 'scratch_graph' in message
        assert '0.01' in message
        assert unsettled[0]['label'] in message
        assert _UNDER_CONSTRUCTION in message

    async def test_a_read_error_propagates_unchanged_after_one_read(self):
        error = redis.exceptions.ResponseError('Invalid graph operation on empty key')
        reader = _CountingReader([error, _settled_records()])

        with pytest.raises(redis.exceptions.ResponseError) as excinfo:
            await _settle(reader)

        assert excinfo.value is error
        assert reader.calls == 1

    async def test_a_record_without_status_raises_without_retrying(self):
        records = _settled_records()
        records[0] = {key: value for key, value in records[0].items() if key != 'status'}
        reader = _CountingReader([records, _settled_records()])

        with pytest.raises(IndexRecordShapeError):
            await _settle(reader)

        assert reader.calls == 1
