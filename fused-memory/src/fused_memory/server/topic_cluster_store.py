"""Runtime store of machine-derived topic clusters for the write-time topic guard (task 3135).

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf ζ (contract C2,
decision D5): every ``consolidate_memories`` run teaches the guard its topic
by persisting a derived :class:`~fused_memory.config.schema.ProceduralTopicCluster`
here. ``near_duplicate_guard.resolve_topic_guard_clusters`` merges the writing
project's rows with the config seeds.

Rows are keyed ``(project_id, topic_id)`` because topic slugs are a
per-project namespace: canonical uniqueness is per (project, topic), so one
project's consolidation must never overwrite another project's cluster for
the same slug. ``source`` and ``gate_task`` identify the trigger that wrote a
row, so a second trigger adds rows rather than a migration.

Modelled on :class:`~fused_memory.server.recon_report_store.ReconReportStore`:
sync ``sqlite3`` on one persistent connection with
``shared.sqlite_sync_base.apply_full_durability_pragmas_sync``, and
``check_same_thread`` at its default so a cross-thread call fails loudly.
Sync is forced by the reader: the guard read runs synchronously on the
``add_memory`` hot path, so :meth:`TopicClusterStore.list_clusters` serves a
write-through in-memory cache and does no I/O.

Cross-process caveat: the cache is hydrated at :meth:`TopicClusterStore.open`.
Every sanctioned writer runs inside the server process, so a row written by
another process is not seen until the server restarts.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from fused_memory.config.schema import ProceduralTopicCluster

__all__ = ['TopicClusterStore', 'TopicClusterStoreError']

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS topic_clusters (
    project_id   TEXT NOT NULL,
    topic_id     TEXT NOT NULL,
    cluster_json TEXT NOT NULL,
    source       TEXT NOT NULL,
    canonical_id TEXT,
    gate_task    TEXT,
    category     TEXT,
    run_id       TEXT,
    updated_at   REAL NOT NULL,
    PRIMARY KEY (project_id, topic_id)
);
"""

_UPSERT_SQL = (
    'INSERT INTO topic_clusters '
    '(project_id, topic_id, cluster_json, source, canonical_id, gate_task, '
    'category, run_id, updated_at) '
    'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) '
    'ON CONFLICT(project_id, topic_id) DO UPDATE SET '
    'cluster_json = excluded.cluster_json, '
    'source = excluded.source, '
    'canonical_id = excluded.canonical_id, '
    'gate_task = excluded.gate_task, '
    'category = excluded.category, '
    'run_id = excluded.run_id, '
    'updated_at = excluded.updated_at'
)

_SELECT_ALL_SQL = (
    'SELECT project_id, topic_id, cluster_json FROM topic_clusters '
    'ORDER BY project_id, topic_id'
)


class TopicClusterStoreError(RuntimeError):
    """A persisted topic-cluster row failed re-validation at :meth:`TopicClusterStore.open`.

    Rows land only through :meth:`TopicClusterStore.upsert`, which accepts
    nothing but a validated :class:`ProceduralTopicCluster`. A row that fails
    re-validation therefore means tampering or an unmigrated model change.
    Both are startup conditions an operator must fix, which is exactly the
    config path's posture when a cluster fails validation at load.
    """


class TopicClusterStore:
    """Persistent-connection sync SQLite store of derived topic clusters.

    Lifecycle::

        store = TopicClusterStore(path)
        store.open()
        try:
            store.upsert(cluster, source=..., project_id=...)
            store.list_clusters(project_id)
        finally:
            store.close()
    """

    def __init__(self, db_path: Path, *, busy_timeout_ms: int = 30000) -> None:
        self._db_path = db_path
        self._busy_timeout_ms = busy_timeout_ms
        self._conn: sqlite3.Connection | None = None
        self._clusters: dict[str, dict[str, ProceduralTopicCluster]] = {}

    @property
    def db_path(self) -> Path:
        return self._db_path

    def open(self) -> None:
        """Open the connection, apply durability pragmas, ensure schema, hydrate.

        A failed ``open()`` closes its connection before raising, so it can be
        retried once the file is fixed.

        Raises:
            RuntimeError: if called while already open.
            TopicClusterStoreError: naming EVERY persisted row that fails
                re-validation through :class:`ProceduralTopicCluster`.
        """
        if self._conn is not None:
            raise RuntimeError(f'{type(self).__name__} already opened')
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        try:
            apply_full_durability_pragmas_sync(conn, busy_timeout_ms=self._busy_timeout_ms)
            conn.executescript(_SCHEMA)
            conn.commit()
            clusters = _hydrate(conn, self._db_path)
        except BaseException:
            conn.close()
            raise
        self._conn = conn
        self._clusters = clusters

    def close(self) -> None:
        """Close the connection. Idempotent — safe when already closed or never opened."""
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError(f'{type(self).__name__} not opened')
        return self._conn

    def upsert(
        self,
        cluster: ProceduralTopicCluster,
        *,
        source: str,
        project_id: str,
        canonical_id: str | None = None,
        gate_task: str | None = None,
        category: str | None = None,
        run_id: str | None = None,
    ) -> None:
        """Insert or replace the row for ``(project_id, cluster.topic_id)`` (one commit).

        Raises:
            TypeError: if ``cluster`` is not a validated ``ProceduralTopicCluster``.
            RuntimeError: if the store is not open.
        """
        if not isinstance(cluster, ProceduralTopicCluster):
            raise TypeError(
                f'upsert() takes a validated ProceduralTopicCluster, got {type(cluster).__name__}'
            )
        conn = self._require_conn()
        conn.execute(
            _UPSERT_SQL,
            (
                project_id,
                cluster.topic_id,
                json.dumps(cluster.model_dump(mode='json')),
                source,
                canonical_id,
                gate_task,
                category,
                run_id,
                time.time(),
            ),
        )
        conn.commit()
        self._clusters.setdefault(project_id, {})[cluster.topic_id] = cluster

    def list_clusters(self, project_id: str) -> list[ProceduralTopicCluster]:
        """Return ``project_id``'s clusters in ``topic_id`` order, from memory only."""
        by_topic = self._clusters.get(project_id, {})
        return [by_topic[topic_id] for topic_id in sorted(by_topic)]


def _hydrate(
    conn: sqlite3.Connection, db_path: Path
) -> dict[str, dict[str, ProceduralTopicCluster]]:
    clusters: dict[str, dict[str, ProceduralTopicCluster]] = {}
    offenders: list[str] = []
    for project_id, topic_id, cluster_json in conn.execute(_SELECT_ALL_SQL):
        try:
            cluster = ProceduralTopicCluster.model_validate(json.loads(cluster_json))
        except ValueError as exc:
            offenders.append(f'  ({project_id!r}, {topic_id!r}): {exc}')
            continue
        clusters.setdefault(project_id, {})[topic_id] = cluster
    if offenders:
        raise TopicClusterStoreError(_invalid_rows_message(db_path, offenders))
    return clusters


def _invalid_rows_message(db_path: Path, offenders: list[str]) -> str:
    listing = '\n'.join(offenders)
    return (
        f'topic-cluster store {db_path} holds {len(offenders)} row(s) that fail '
        f're-validation as ProceduralTopicCluster (topic_id is a slug per '
        f'fused_memory.topic_slug):\n{listing}\n'
        f'These rows are machine-derived: only a validated upsert writes them, so '
        f'this means tampering or an unmigrated model change. Delete the offending '
        f'row(s) or the file; that loses only derived clusters, which the next '
        f'consolidate_memories of each topic re-seeds.'
    )
