"""Tests for the runtime topic-cluster store and its auto-seed (task 3135).

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf ζ: every
``consolidate_memories`` run teaches the write-time topic guard its topic, by
persisting a derived ``ProceduralTopicCluster`` that the guard merges with the
config seeds.

Covers:
- ``TopicClusterStore`` durable lifecycle, restart round-trip and project
  scoping (TestStoreLifecycle)
- Mistyped/duplicate rows fail loud at ``open()`` like the config path does at
  load (TestStoreFailsLoudLikeTheConfigPath)
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

import pytest

from fused_memory.config.schema import ProceduralTopicCluster
from fused_memory.server.topic_cluster_store import TopicClusterStore, TopicClusterStoreError

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


def _create_schema(db_path: Path) -> None:
    store = TopicClusterStore(db_path)
    store.open()
    store.close()


def _insert_raw_row(db_path: Path, *, project_id: str, topic_id: str, cluster_json: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO topic_clusters (project_id, topic_id, cluster_json, source, updated_at) '
            'VALUES (?, ?, ?, ?, ?)',
            (project_id, topic_id, cluster_json, 'consolidate_memories', 0.0),
        )
        conn.commit()
    finally:
        conn.close()


def _row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute('SELECT COUNT(*) FROM topic_clusters').fetchone()[0]
    finally:
        conn.close()


def _cluster_json(**overrides: object) -> str:
    payload = _cluster().model_dump(mode='json')
    payload.update(overrides)
    return json.dumps(payload)


_SNAKE_TOPIC = 'pytest_xdist_serial_override'

_CORRUPT_ROWS = {
    'snake_case_topic_id': (_SNAKE_TOPIC, _cluster_json(topic_id=_SNAKE_TOPIC)),
    'unknown_key': ('pytest-xdist-serial-override', _cluster_json(severity='high')),
    'non_json': ('pytest-xdist-serial-override', '{not json'),
}


class TestStoreFailsLoudLikeTheConfigPath:
    """A row that fails re-validation is a startup condition, never a silent skip."""

    @pytest.mark.parametrize('shape', sorted(_CORRUPT_ROWS))
    def test_a_corrupt_row_fails_open_naming_the_row_path_and_recovery(
        self, db_path: Path, shape: str
    ) -> None:
        topic_id, cluster_json = _CORRUPT_ROWS[shape]
        _create_schema(db_path)
        _insert_raw_row(db_path, project_id=_PROJECT, topic_id=topic_id, cluster_json=cluster_json)

        store = TopicClusterStore(db_path)
        with pytest.raises(TopicClusterStoreError) as excinfo:
            store.open()

        message = str(excinfo.value)
        assert _PROJECT in message
        assert topic_id in message
        assert str(db_path) in message
        assert 'machine-derived' in message
        assert 're-seed' in message

    def test_a_snake_case_topic_id_names_the_slug_rule(self, db_path: Path) -> None:
        _create_schema(db_path)
        _insert_raw_row(
            db_path, project_id=_PROJECT, topic_id=_SNAKE_TOPIC,
            cluster_json=_cluster_json(topic_id=_SNAKE_TOPIC),
        )
        with pytest.raises(TopicClusterStoreError, match=r'fused_memory\.topic_slug'):
            TopicClusterStore(db_path).open()

    def test_every_bad_row_is_named_in_one_error(self, db_path: Path) -> None:
        _create_schema(db_path)
        _insert_raw_row(
            db_path, project_id=_PROJECT, topic_id=_SNAKE_TOPIC,
            cluster_json=_cluster_json(topic_id=_SNAKE_TOPIC),
        )
        _insert_raw_row(db_path, project_id='reify', topic_id='broken-row', cluster_json='{not json')

        with pytest.raises(TopicClusterStoreError) as excinfo:
            TopicClusterStore(db_path).open()

        message = str(excinfo.value)
        assert _SNAKE_TOPIC in message
        assert 'broken-row' in message
        assert 'reify' in message

    def test_a_failed_open_leaks_no_half_open_state(self, db_path: Path) -> None:
        _create_schema(db_path)
        _insert_raw_row(db_path, project_id=_PROJECT, topic_id='broken-row', cluster_json='{not json')
        store = TopicClusterStore(db_path)
        with pytest.raises(TopicClusterStoreError):
            store.open()

        conn = sqlite3.connect(str(db_path))
        conn.execute('DELETE FROM topic_clusters')
        conn.commit()
        conn.close()

        store.open()
        try:
            assert store.list_clusters(_PROJECT) == []
        finally:
            store.close()

    def test_a_duplicate_upsert_replaces_the_row(self, db_path: Path) -> None:
        earlier = _cluster(phrases=['--dist loadgroup', 'max-worker-restart'])
        later = _cluster(phrases=['pytest-xdist', 'xdist_group'], hint='New canonical.')
        store = TopicClusterStore(db_path)
        store.open()
        try:
            _upsert(store, earlier)
            _upsert(store, later)
            listed = store.list_clusters(_PROJECT)
        finally:
            store.close()
        assert _row_count(db_path) == 1
        assert listed == [later]

    def test_the_same_slug_in_two_projects_is_two_rows(self, db_path: Path) -> None:
        ours = _cluster(phrases=['--dist loadgroup', 'max-worker-restart'])
        theirs = _cluster(phrases=['reify-only-phrase', 'reify_other_phrase'])
        store = TopicClusterStore(db_path)
        store.open()
        try:
            _upsert(store, ours, project_id=_PROJECT)
            _upsert(store, theirs, project_id='reify')
            assert store.list_clusters(_PROJECT) == [ours]
            assert store.list_clusters('reify') == [theirs]
        finally:
            store.close()
        assert _row_count(db_path) == 2

    def test_an_unvalidated_dict_is_rejected_before_any_sql(self, db_path: Path) -> None:
        store = TopicClusterStore(db_path)
        store.open()
        try:
            with pytest.raises(TypeError):
                store.upsert(
                    _cluster().model_dump(mode='json'),  # type: ignore[arg-type]
                    source='consolidate_memories',
                    project_id=_PROJECT,
                )
            assert store.list_clusters(_PROJECT) == []
        finally:
            store.close()
        assert _row_count(db_path) == 0
