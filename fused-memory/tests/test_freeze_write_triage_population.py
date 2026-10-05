"""Tests for freeze_write_triage_population.py — the frozen π population (task 6151).

Driven through the script's public functions against a LOCAL store double: the
three reads the freeze makes (``mem0.scroll_all_by_metadata``, ``search`` and
``get_memory_by_id``) and nothing else. No Qdrant, no network.
"""
from __future__ import annotations

import functools
import types
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _fm_helpers import load_script_module

from fused_memory.models.enums import MEM0_PRIMARY
from fused_memory.reconciliation.prompts import (
    FLAG_FOR_STAGE2_MARKER_KIND,
    STAGE2_SUPPRESS_GUARD_KIND,
)
from fused_memory.server.write_triage import declares_attach_keys

SCRIPTS = Path(__file__).parent.parent / 'scripts'

SINCE = datetime(2026, 9, 29, tzinfo=UTC)
FROZEN_AT = datetime(2026, 10, 5, 12, tzinfo=UTC)
CATEGORIES = sorted(c.value for c in MEM0_PRIMARY)


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'freeze_write_triage_population.py', 'freeze_write_triage_population',
    )


class _Mem0:
    """``Mem0Backend.scroll_all_by_metadata``: a plain call returning an async stream."""

    def __init__(self, scrolled: dict[tuple[str, str], list[dict]]) -> None:
        self._scrolled = scrolled
        self.calls: list[tuple[str, dict]] = []

    def scroll_all_by_metadata(self, scope: Any, filters: dict) -> AsyncIterator[dict]:
        self.calls.append((scope.project_id, dict(filters)))
        rows = self._scrolled.get((scope.project_id, filters['category']), [])

        async def _stream() -> AsyncIterator[dict]:
            for row in rows:
                yield row

        return _stream()


class _Store:
    """The memory service the freeze reads through."""

    def __init__(
        self,
        scrolled: dict[tuple[str, str], list[dict]] | None = None,
        by_id: dict[tuple[str, str], dict] | None = None,
    ) -> None:
        self.mem0 = _Mem0(scrolled or {})
        self._by_id = by_id or {}
        self.search = AsyncMock()

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> dict | None:
        return self._by_id.get((project_id, memory_id))


def _scrolled(memory_id: str, created_at: str | None, category: str) -> dict:
    return {'id': memory_id, 'created_at': created_at, 'metadata': {'category': category}}


def _stored(memory_id: str, **metadata: Any) -> dict:
    return {'id': memory_id, 'content': f'content of {memory_id}', 'metadata': metadata}


def _enumerate(store: _Store, projects=('dark_factory', 'reify')):
    import asyncio  # noqa: PLC0415

    return asyncio.run(_mod().enumerate_population(
        store, projects=projects, since=SINCE, frozen_at=FROZEN_AT,
    ))


def _one_project_store(rows: list[tuple[str, str | None, dict]]) -> _Store:
    """``procedural_knowledge`` rows in dark_factory, each live with *metadata*."""
    return _Store(
        scrolled={('dark_factory', 'procedural_knowledge'): [
            _scrolled(memory_id, created_at, 'procedural_knowledge')
            for memory_id, created_at, _ in rows
        ]},
        by_id={
            ('dark_factory', memory_id): _stored(memory_id, **metadata)
            for memory_id, _, metadata in rows
        },
    )


def _ids(writes: list[dict]) -> list[str]:
    return [write['memory_id'] for write in writes]


class TestEnumeration:
    def test_every_project_scrolls_every_mem0_primary_category(self) -> None:
        store = _Store()
        _enumerate(store)
        assert store.mem0.calls == [
            (project, {'category': category})
            for project in ('dark_factory', 'reify')
            for category in CATEGORIES
        ]

    def test_the_window_is_compared_tz_aware_and_inclusive_of_since(self) -> None:
        store = _one_project_store([
            ('pacific-evening', '2026-09-28T18:30:00-07:00', {}),
            ('utc-before', '2026-09-28T23:59:59+00:00', {}),
            ('at-since', '2026-09-29T00:00:00+00:00', {}),
            ('after-frozen', '2026-10-05T12:00:01+00:00', {}),
            ('at-frozen', '2026-10-05T12:00:00+00:00', {}),
        ])
        writes, _ = _enumerate(store)
        assert set(_ids(writes)) == {'pacific-evening', 'at-since', 'at-frozen'}

    @pytest.mark.parametrize('created_at', [
        '2026-09-30T10:00:00', None, 'not-a-date', '',
    ])
    def test_an_undated_record_is_excluded_and_counted_never_guessed(
        self, created_at: str | None,
    ) -> None:
        store = _one_project_store([
            ('undated', created_at, {}),
            ('dated', '2026-09-30T10:00:00+00:00', {}),
        ])
        writes, excluded = _enumerate(store)
        assert _ids(writes) == ['dated']
        assert excluded['undated'] == 1

    def test_a_record_that_vanished_before_its_read_is_excluded_and_counted(self) -> None:
        store = _Store(
            scrolled={('reify', 'observations_and_summaries'): [
                _scrolled('gone', '2026-09-30T10:00:00+00:00', 'observations_and_summaries'),
                _scrolled('here', '2026-09-30T11:00:00+00:00', 'observations_and_summaries'),
            ]},
            by_id={('reify', 'here'): _stored('here')},
        )
        writes, excluded = _enumerate(store)
        assert _ids(writes) == ['here']
        assert excluded['vanished'] == 1

    def test_content_and_metadata_come_from_the_point_read(self) -> None:
        store = _one_project_store([
            ('m', '2026-09-30T10:00:00+00:00', {'category': 'procedural_knowledge', 'topic': 't'}),
        ])
        [write] = _enumerate(store)[0]
        assert write['content'] == 'content of m'
        assert write['metadata'] == {'category': 'procedural_knowledge', 'topic': 't'}

    @pytest.mark.parametrize(('metadata', 'marker'), [
        ({'flag_for_stage2': True}, True),
        ({'stage2_suppress': True}, True),
        ({'kind': FLAG_FOR_STAGE2_MARKER_KIND}, True),
        ({'kind': STAGE2_SUPPRESS_GUARD_KIND}, True),
        ({'flag_for_stage2': False}, False),
        ({'stage2_suppress': 'true'}, False),
        ({'kind': 'sighting'}, False),
        ({}, False),
    ])
    def test_a_recon_marker_is_recognised_by_flag_or_kind(
        self, metadata: dict, marker: bool,
    ) -> None:
        store = _one_project_store([('m', '2026-09-30T10:00:00+00:00', metadata)])
        [write] = _enumerate(store)[0]
        assert write['recon_marker'] is marker

    @pytest.mark.parametrize('metadata', [
        {}, {'kind': 'sighting'}, {'parent_id': 'p'}, {'x_contested': False}, {'topic': 't'},
    ])
    def test_declares_attach_keys_is_the_shipped_predicate(self, metadata: dict) -> None:
        store = _one_project_store([('m', '2026-09-30T10:00:00+00:00', metadata)])
        [write] = _enumerate(store)[0]
        assert write['declares_attach_keys'] is declares_attach_keys(metadata)

    def test_each_write_carries_its_identity(self) -> None:
        store = _Store(
            scrolled={('reify', 'preferences_and_norms'): [
                _scrolled('m', '2026-09-28T18:30:00-07:00', 'preferences_and_norms'),
            ]},
            by_id={('reify', 'm'): _stored('m')},
        )
        [write] = _enumerate(store)[0]
        assert (
            write['memory_id'], write['project_id'], write['category'], write['created_at'],
        ) == ('m', 'reify', 'preferences_and_norms', '2026-09-28T18:30:00-07:00')

    def test_the_order_is_project_then_instant_then_id(self) -> None:
        rows = {
            ('reify', 'procedural_knowledge'): [
                _scrolled('r1', '2026-09-30T00:00:00+00:00', 'procedural_knowledge'),
            ],
            ('dark_factory', 'procedural_knowledge'): [
                _scrolled('late', '2026-09-28T18:30:00-07:00', 'procedural_knowledge'),
                _scrolled('b-tie', '2026-09-30T00:00:00+00:00', 'procedural_knowledge'),
            ],
            ('dark_factory', 'observations_and_summaries'): [
                _scrolled('early', '2026-09-29T01:00:00+00:00', 'observations_and_summaries'),
                _scrolled('a-tie', '2026-09-30T00:00:00+00:00', 'observations_and_summaries'),
            ],
        }
        by_id = {
            (project, row['id']): _stored(row['id'])
            for (project, _), scrolled in rows.items() for row in scrolled
        }
        writes, _ = _enumerate(_Store(scrolled=rows, by_id=by_id))
        assert _ids(writes) == ['early', 'late', 'a-tie', 'b-tie', 'r1']
