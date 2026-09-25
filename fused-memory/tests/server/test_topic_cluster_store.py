"""Tests for the runtime topic-cluster store and its auto-seed (task 3135).

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf ζ: every
``consolidate_memories`` run teaches the write-time topic guard its topic, by
persisting a derived ``ProceduralTopicCluster`` that the guard merges with the
config seeds.

Covers:
- ``TopicClusterStore`` durable lifecycle, restart round-trip and project
  scoping (TestStoreLifecycle)
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from fused_memory.config.schema import ProceduralTopicCluster
from fused_memory.server.topic_cluster_store import TopicClusterStore

_PROJECT = 'dark_factory'


def _cluster(
    topic_id: str = 'pytest-xdist-serial-override',
    phrases: list[str] | None = None,
    hint: str = 'Consolidated topic; update the canonical instead.',
) -> ProceduralTopicCluster:
    return ProceduralTopicCluster(
        topic_id=topic_id,
        phrases=phrases if phrases is not None else ['--dist loadgroup', 'max-worker-restart'],
        min_phrase_hits=2,
        sufficient_phrases=[],
        hint=hint,
    )


def _upsert(store: TopicClusterStore, cluster: ProceduralTopicCluster, *, project_id: str = _PROJECT) -> None:
    store.upsert(
        cluster,
        source='consolidate_memories',
        project_id=project_id,
        canonical_id=str(uuid.uuid4()),
        category='procedural_knowledge',
        run_id='r1',
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / 'sub' / 'topic_clusters.db'


class TestStoreLifecycle:
    """The store's durable contract: a derived cluster survives a restart."""

    def test_open_creates_missing_parent_dir_and_starts_empty(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.open()
        try:
            assert db_path.parent.is_dir()
            assert store.db_path == db_path
            assert store.list_clusters(_PROJECT) == []
        finally:
            store.close()

    def test_upsert_round_trips_the_real_model(self, db_path: Path) -> None:
        cluster = _cluster(phrases=['zeta-phrase', '--dist loadgroup', 'max-worker-restart'])
        store = TopicClusterStore(db_path)
        store.open()
        try:
            _upsert(store, cluster)
            listed = store.list_clusters(_PROJECT)
        finally:
            store.close()
        assert listed == [cluster]
        assert listed[0].phrases == ['zeta-phrase', '--dist loadgroup', 'max-worker-restart']

    def test_restart_preserves_the_cluster(self, db_path: Path) -> None:
        cluster = _cluster()
        first = TopicClusterStore(db_path)
        first.open()
        _upsert(first, cluster)
        first.close()

        second = TopicClusterStore(db_path)
        second.open()
        try:
            assert second is not first
            assert second.list_clusters(_PROJECT) == [cluster]
            assert second.list_clusters('reify') == []
        finally:
            second.close()

    def test_double_open_raises(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.open()
        try:
            with pytest.raises(RuntimeError):
                store.open()
        finally:
            store.close()

    def test_close_is_idempotent(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.close()
        store.open()
        store.close()
        store.close()

    def test_upsert_before_open_raises(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        with pytest.raises(RuntimeError):
            _upsert(store, _cluster())
