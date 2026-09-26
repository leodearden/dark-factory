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

:func:`derive_topic_cluster` turns a topic's member texts into the cluster
this store persists, and :func:`seed_topic_cluster` is the one non-raising
derive-and-persist call every trigger makes. Both are trigger-agnostic, so a
second trigger reuses them unchanged.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shared.sqlite_sync_base import apply_full_durability_pragmas_sync

from fused_memory.config.schema import ProceduralTopicCluster

__all__ = [
    'TopicClusterStore',
    'TopicClusterStoreError',
    'derive_topic_cluster',
    'seed_topic_cluster',
]

logger = logging.getLogger(__name__)

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


_MAX_NGRAM_TOKENS = 4
_MIN_SUPPORTING_TEXTS = 2
_MIN_DERIVED_PHRASES = 2
_MAX_DERIVED_PHRASES = 6
_DERIVED_MIN_PHRASE_HITS = 2
_IDENTIFIER_PUNCTUATION = frozenset('-_./:')
_LONG_WORD_CHARS = 10
_MIN_DISTINCTIVE_TOKEN_CHARS = 3
_TOKEN_RE = re.compile(r'[\w\-./:]+')
_SENTENCE_TRAILERS = '.:'

# Tokens whose SHAPE passes the distinctiveness test (identifier punctuation, a
# digit, or a long word) yet which carry no topic signal, so they must never
# qualify a phrase on their own.
_GENERIC_SHAPED_TOKENS = frozenset({
    'e.g', 'i.e', 'a.k.a', 'and/or', 'n/a', 'w/o',
    'additionally', 'alternatively', 'automatically', 'consistently',
    'especially', 'eventually', 'everything', 'furthermore', 'immediately',
    'information', 'nevertheless', 'particularly', 'previously', 'regardless',
    'specifically', 'successfully', 'understand',
})

# Closed-class English only (articles, prepositions, conjunctions, pronouns,
# determiners, auxiliaries, modals): a finite category, not a tuning knob.
_PHRASE_EDGE_FUNCTION_WORDS = frozenset({
    'a', 'an', 'the',
    'and', 'or', 'but', 'nor', 'so', 'if', 'then', 'than', 'when', 'while',
    'because', 'since', 'unless', 'until',
    'of', 'to', 'in', 'on', 'at', 'by', 'for', 'from', 'with', 'without',
    'into', 'onto', 'over', 'under', 'via', 'per', 'as',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am',
    'do', 'does', 'did', 'has', 'have', 'had',
    'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would',
    'it', 'its', 'this', 'that', 'these', 'those', 'there', 'here',
    'which', 'who', 'what', 'where', 'how',
    'not', 'no', 'all', 'any', 'each', 'every', 'some', 'one',
    'you', 'your', 'we', 'our', 'they', 'their', 'i', 'me', 'my',
    'he', 'she', 'him', 'her', 'them',
})


def derive_topic_cluster(
    texts: Sequence[str], *, topic_id: str, hint: str
) -> ProceduralTopicCluster | None:
    """Derive a conservative cluster from a topic's member texts, or ``None`` to abstain.

    The rules are motivated by the over-blocking MEASURED in the retirement
    notes on ``config.schema._default_topic_guard_clusters``, where a cluster
    built from ordinary subsystem vocabulary fired 13 off-topic blocks out of 14:

    * a phrase must occur in at least two DISTINCT texts, so it characterises
      the cluster rather than one member (a duplicated input adds no support);
    * a phrase is at least two words and never begins or ends with a
      closed-class word. With no background corpus, a token's shape cannot
      tell a topic identifier (``64kb``) from the project's everyday API
      vocabulary (``add_memory``), which recurs in every member of a topic
      about that API; a recurring multi-word construction is topic evidence,
      a lone identifier is not;
    * a phrase must hold a distinctive token, which keeps generic prose out;
    * no selected phrase nests inside another, because the matcher counts
      substring hits and one occurrence of a longer form would score twice;
    * at most six phrases, ``min_phrase_hits`` 2 and never any
      ``sufficient_phrases``: promoting a phrase to sufficient is a human
      judgement the schema reserves for identifier-shaped names;
    * fewer than two phrases abstains, since such a cluster can never fire.

    Residual: a multi-word construction common across the whole project can
    still qualify. Rejecting it needs a document-frequency check against the
    project's other memories, a read this zero-I/O derivation does not make.

    Phrases are lowercase, matching the matcher's own ``str.lower`` comparison.
    Ranking is a total order, so the result does not depend on input order.
    """
    corpus = _distinct_normalised_texts(texts)
    phrases = _select_unnested(_rank(_supported_candidates(corpus)))
    if len(phrases) < _MIN_DERIVED_PHRASES:
        return None
    return ProceduralTopicCluster(
        topic_id=topic_id,
        phrases=phrases,
        min_phrase_hits=_DERIVED_MIN_PHRASE_HITS,
        sufficient_phrases=[],
        hint=hint,
    )


def _distinct_normalised_texts(texts: Sequence[str]) -> list[str]:
    return sorted({' '.join(text.lower().split()) for text in texts} - {''})


def _tokenise(text: str) -> list[str]:
    tokens = (raw.rstrip(_SENTENCE_TRAILERS) for raw in _TOKEN_RE.findall(text))
    return [token for token in tokens if any(ch.isalnum() for ch in token)]


def _ngrams(tokens: list[str]) -> set[str]:
    return {
        ' '.join(tokens[start:start + size])
        for size in range(1, _MAX_NGRAM_TOKENS + 1)
        for start in range(len(tokens) - size + 1)
    }


def _is_distinctive(token: str) -> bool:
    if (
        token in _GENERIC_SHAPED_TOKENS
        or len(token) < _MIN_DISTINCTIVE_TOKEN_CHARS
        or not any(ch.isalpha() for ch in token)
    ):
        return False
    return (
        any(ch in _IDENTIFIER_PUNCTUATION for ch in token)
        or any(ch.isdigit() for ch in token)
        or len(token) >= _LONG_WORD_CHARS
    )


def _distinctive_token_count(phrase: str) -> int:
    return sum(_is_distinctive(token) for token in phrase.split(' '))


def _is_key_phrase(phrase: str) -> bool:
    tokens = phrase.split(' ')
    return (
        len(tokens) >= 2
        and tokens[0] not in _PHRASE_EDGE_FUNCTION_WORDS
        and tokens[-1] not in _PHRASE_EDGE_FUNCTION_WORDS
    )


def _supported_candidates(corpus: list[str]) -> dict[str, int]:
    """Map each distinctive shared key phrase to the number of texts that literally contain it.

    Sharing is counted on n-grams first (cheap), then confirmed with the
    matcher's own substring test, which also drops an n-gram that spans a
    sentence boundary and so never occurs verbatim.
    """
    shared = Counter(ngram for text in corpus for ngram in _ngrams(_tokenise(text)))
    support: dict[str, int] = {}
    for phrase, sharing_texts in shared.items():
        if (
            sharing_texts < _MIN_SUPPORTING_TEXTS
            or not _is_key_phrase(phrase)
            or not _distinctive_token_count(phrase)
        ):
            continue
        literal = sum(phrase in text for text in corpus)
        if literal >= _MIN_SUPPORTING_TEXTS:
            support[phrase] = literal
    return support


def _rank(support: dict[str, int]) -> list[str]:
    return sorted(
        support,
        key=lambda phrase: (
            -support[phrase],
            -_distinctive_token_count(phrase),
            -len(phrase.split(' ')),
            -len(phrase),
            phrase,
        ),
    )


def _select_unnested(ranked: list[str]) -> list[str]:
    selected: list[str] = []
    for phrase in ranked:
        if any(phrase in kept or kept in phrase for kept in selected):
            continue
        selected.append(phrase)
        if len(selected) == _MAX_DERIVED_PHRASES:
            break
    return selected


def seed_topic_cluster(
    store: TopicClusterStore,
    *,
    enabled: bool,
    texts: Sequence[str],
    topic: str,
    canonical_id: str,
    project_id: str,
    category: str | None,
    run_id: str | None,
    source: str,
) -> dict[str, Any]:
    """Derive *topic*'s cluster from *texts* and persist it; report the outcome, never raise.

    Returns ``{'outcome': 'disabled'}``, ``{'outcome': 'skipped', 'reason'}``,
    ``{'outcome': 'seeded', 'topic_id', 'phrases'}`` or
    ``{'outcome': 'failed', 'topic_id', 'error', 'error_type'}``. This is the
    only home of that vocabulary; every trigger calls this rather than
    re-deriving it.

    It cannot raise because it runs after an irreversible fold has completed:
    teaching the guard is a side effect, and a side effect must not veto a
    completed fold. A failure is logged at WARNING and disclosed in the return
    value instead.
    """
    if not enabled:
        return {'outcome': 'disabled'}
    try:
        cluster = derive_topic_cluster(
            texts, topic_id=topic, hint=_consolidated_topic_hint(canonical_id)
        )
        if cluster is None:
            return {
                'outcome': 'skipped',
                'reason': (
                    f'fewer than {_MIN_DERIVED_PHRASES} distinctive multi-word phrases are '
                    f'shared by at least {_MIN_SUPPORTING_TEXTS} of the {len(texts)} merged texts'
                ),
            }
        store.upsert(
            cluster,
            source=source,
            project_id=project_id,
            canonical_id=canonical_id,
            category=category,
            run_id=run_id,
        )
    except Exception as exc:
        logger.warning(
            'topic-cluster seed failed for topic %r in project %r',
            topic,
            project_id,
            exc_info=True,
        )
        return {
            'outcome': 'failed',
            'topic_id': topic,
            'error': str(exc),
            'error_type': type(exc).__name__,
        }
    return {'outcome': 'seeded', 'topic_id': cluster.topic_id, 'phrases': list(cluster.phrases)}


def _consolidated_topic_hint(canonical_id: str) -> str:
    """The derived cluster's hint, in the three-outcome shape of the seeded clusters.

    A per-cluster hint SHADOWS the guard's default hint, which is the only
    other place naming ``allow_near_duplicate``, so this hint must carry the
    override itself. Content amends are authz-gated to ``recon-stage-`` /
    ``curator-`` agent_ids, so ``update_memory`` is offered only to them.
    """
    return (
        f'Known-recurring topic, already consolidated into canonical memory '
        f'{canonical_id}. Do NOT add another entry. '
        f'(1) Your content is genuinely DISTINCT from that entry -- re-send this '
        f"write with metadata={{'allow_near_duplicate': True}}, which is open to "
        f'every agent. '
        f'(2) It duplicates or extends that entry -- SKIP the write. Only '
        f'recon-stage- / curator- agent_ids may fold it in, with '
        f"update_memory(memory_id='{canonical_id}', store='mem0', project_id=..., "
        f'content=<merged text>, reason=...). '
        f'(3) It CONTRADICTS that entry, or you are unsure -- escalate with '
        f'escalate_blocker (or escalate_info if you are merely unsure).'
    )
