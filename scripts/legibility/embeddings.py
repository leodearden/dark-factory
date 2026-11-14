"""The OpenAI embeddings client and the host-local vector cache the census
similarity stage reads (plans/census-incremental-prd.md D3).

Knows nothing of codebooks: it turns texts into vectors. Stdlib plus a lazy
``import httpx``.
"""
from __future__ import annotations

import hashlib
import math
import os
import sqlite3
from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import TypeGuard

from legibility import trickle_state

OPENAI_EMBEDDINGS_URL = 'https://api.openai.com/v1/embeddings'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'
CACHE_FILENAME = 'embeddings.sqlite'
REQUEST_BATCH = 256

_ERROR_TEXT_LIMIT = 200
_CONNECT_TIMEOUT_SECS = 30.0

_DDL = """
CREATE TABLE IF NOT EXISTS vectors (
    record_id TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    model TEXT NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (record_id, text_sha256, model)
)
"""
_HAS_TABLE = "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'vectors'"
_SELECT = 'SELECT record_id, text_sha256, vector FROM vectors WHERE model = ?'
_SELECT_KEYS = 'SELECT record_id, text_sha256, model FROM vectors'
_UPSERT = 'INSERT OR REPLACE INTO vectors (record_id, text_sha256, model, vector) VALUES (?, ?, ?, ?)'
_DELETE = 'DELETE FROM vectors WHERE record_id = ? AND text_sha256 = ? AND model = ?'

Vector = tuple[float, ...]
CacheKey = tuple[str, str, str]


class VectorsUnavailable(Exception):
    """No vectors this call. ``requests_made`` counts the embedding requests
    spent before the failure."""

    def __init__(self, message: str, *, requests_made: int = 0) -> None:
        super().__init__(message)
        self.requests_made = requests_made


class EmbeddingFailed(VectorsUnavailable):
    """The embedding request, or its response, failed. Never partial."""


class VectorCacheError(VectorsUnavailable):
    """The vector cache could not be read or written; the message names it."""


@dataclass(frozen=True)
class Text:
    """One text to embed. ``record_id`` None is ephemeral: never cached."""

    key: str
    record_id: str | None
    text: str


@dataclass(frozen=True)
class Embedder:
    """The census seam: how texts become vectors, and where they are kept."""

    embed_fn: Callable[..., Sequence[Sequence[float]]]
    cache_path: Path


@dataclass(frozen=True)
class Embedded:
    vectors: Mapping[str, Vector] = field(default_factory=lambda: MappingProxyType({}))
    requests: int = 0
    embedded: int = 0
    cached: int = 0
    pruned: int = 0


def cache_path(project_id: str) -> Path:
    return trickle_state.project_state_dir(project_id) / CACHE_FILENAME


def _is_vector(value: object) -> TypeGuard[list[Real]]:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(x, Real) and not isinstance(x, bool) for x in value)
    )


def _parsed_vectors(body: object, expected: int) -> list[list[float]]:
    data = body.get('data') if isinstance(body, dict) else None
    if not isinstance(data, list):
        raise EmbeddingFailed('embeddings response carries no data list')
    if len(data) != expected:
        raise EmbeddingFailed(
            f'embeddings response carries {len(data)} vectors for {expected} inputs',
        )
    by_index: dict[int, list[float]] = {}
    for datum in data:
        index = datum.get('index') if isinstance(datum, dict) else None
        vector = datum.get('embedding') if isinstance(datum, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or not _is_vector(vector):
            raise EmbeddingFailed('embeddings response holds a malformed datum')
        by_index[index] = [float(x) for x in vector]
    if sorted(by_index) != list(range(expected)):
        raise EmbeddingFailed('embeddings response indexes do not cover the inputs')
    return [by_index[index] for index in range(expected)]


def openai_embed(
    texts: Sequence[str], *, model: str, timeout: float = 60.0,
) -> list[list[float]]:
    """One embeddings request for *texts*; vectors in input order."""
    key = os.environ.get(OPENAI_API_KEY_ENV)
    if not key:
        raise EmbeddingFailed(f'{OPENAI_API_KEY_ENV} is not set')
    import httpx

    try:
        response = httpx.post(
            OPENAI_EMBEDDINGS_URL,
            headers={'Authorization': f'Bearer {key}'},
            json={'model': model, 'input': list(texts)},
            timeout=timeout,
        )
    except Exception as exc:
        raise EmbeddingFailed(
            f'embeddings request failed: {type(exc).__name__}: {exc}',
        ) from exc
    if response.status_code != 200:
        raise EmbeddingFailed(
            f'embeddings endpoint answered HTTP {response.status_code}: '
            f'{str(response.text)[:_ERROR_TEXT_LIMIT]}',
        )
    try:
        body = response.json()
    except Exception as exc:
        raise EmbeddingFailed(f'embeddings response is not JSON: {exc}') from exc
    return _parsed_vectors(body, len(texts))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _float32(values: Sequence[float]) -> Vector:
    return tuple(array('f', values))


def _blob(vector: Vector) -> bytes:
    return array('f', vector).tobytes()


def _unblob(blob: bytes) -> Vector:
    values = array('f')
    values.frombytes(blob)
    return tuple(values)


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        return sqlite3.connect(
            f'{path.absolute().as_uri()}?mode=ro', uri=True, timeout=_CONNECT_TIMEOUT_SECS,
        )
    return sqlite3.connect(path, timeout=_CONNECT_TIMEOUT_SECS, isolation_level=None)


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute('BEGIN IMMEDIATE')
    try:
        yield
    except BaseException:
        conn.execute('ROLLBACK')
        raise
    conn.execute('COMMIT')


def _read_cache(path: Path, model: str) -> dict[tuple[str, str], Vector]:
    if not path.exists():
        return {}
    try:
        with closing(_connect(path, read_only=True)) as conn:
            if conn.execute(_HAS_TABLE).fetchone() is None:
                return {}
            return {
                (record_id, sha): _unblob(blob)
                for record_id, sha, blob in conn.execute(_SELECT, (model,))
            }
    except (sqlite3.Error, OSError) as exc:
        raise VectorCacheError(f'vector cache {path}: {exc}') from exc


def _write_cache(
    path: Path, rows: Mapping[CacheKey, Vector], keep: frozenset[CacheKey], *, requests_made: int,
) -> int:
    """Insert *rows* and delete every row not in *keep*; returns the deleted count."""
    if not rows and not path.exists():
        return 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path, read_only=False)) as conn, _immediate_transaction(conn):
            conn.execute(_DDL)
            conn.executemany(
                _UPSERT, [(*key, _blob(vector)) for key, vector in rows.items()],
            )
            stale = [key for key in conn.execute(_SELECT_KEYS).fetchall() if key not in keep]
            conn.executemany(_DELETE, stale)
            return len(stale)
    except (sqlite3.Error, OSError) as exc:
        raise VectorCacheError(
            f'vector cache {path}: {exc}', requests_made=requests_made,
        ) from exc


def _checked(raw: Sequence[Sequence[float]], expected: int, *, requests_made: int) -> list[Vector]:
    def failed(why: str) -> EmbeddingFailed:
        return EmbeddingFailed(f'embedder returned unusable vectors: {why}', requests_made=requests_made)

    if len(raw) != expected:
        raise failed(f'{len(raw)} vectors for {expected} texts')
    vectors = [_float32(vector) for vector in raw]
    if len({len(vector) for vector in vectors}) > 1:
        raise failed('ragged dimensions')
    if not all(math.isfinite(x) for vector in vectors for x in vector):
        raise failed('a NaN or infinite component')
    if any(not math.fsum(x * x for x in vector) for vector in vectors):
        raise failed('an empty or all-zero vector')
    return vectors


def _embed_misses(
    misses: Sequence[Text], *, embedder: Embedder, model: str, batch_size: int,
) -> tuple[dict[str, Vector], int]:
    vectors: dict[str, Vector] = {}
    requests = 0
    for start in range(0, len(misses), batch_size):
        batch = misses[start:start + batch_size]
        requests += 1
        try:
            raw = embedder.embed_fn([t.text for t in batch], model=model)
        except EmbeddingFailed as exc:
            raise EmbeddingFailed(str(exc), requests_made=requests) from exc
        for text, vector in zip(
            batch, _checked(raw, len(batch), requests_made=requests), strict=True,
        ):
            vectors[text.key] = vector
    return vectors, requests


def vectors_for(
    texts: Sequence[Text], *, embedder: Embedder, model: str, batch_size: int = REQUEST_BATCH,
) -> Embedded:
    """A vector per ``Text.key``: cached ones read, the rest embedded in
    batches of *batch_size* with no retry. Every vector has one dimension.
    The cache is written only when every batch succeeded, and then holds
    exactly this call's cacheable texts. Raises EmbeddingFailed or
    VectorCacheError, never a partial result.
    """
    cached_rows = _read_cache(embedder.cache_path, model)
    vectors: dict[str, Vector] = {}
    misses: list[Text] = []
    for text in texts:
        hit = None if text.record_id is None else cached_rows.get((text.record_id, _sha256(text.text)))
        if hit is None:
            misses.append(text)
        else:
            vectors[text.key] = hit
    fresh, requests = _embed_misses(misses, embedder=embedder, model=model, batch_size=batch_size)
    vectors.update(fresh)
    dimensions = sorted({len(vector) for vector in vectors.values()})
    if len(dimensions) > 1:
        raise EmbeddingFailed(
            f'vectors of {", ".join(map(str, dimensions))} dimensions under {model} from the embedder '
            f'and the vector cache {embedder.cache_path}; one index needs one dimension',
            requests_made=requests,
        )
    keep = frozenset(
        (text.record_id, _sha256(text.text), model) for text in texts if text.record_id is not None
    )
    new_rows = {
        (text.record_id, _sha256(text.text), model): fresh[text.key]
        for text in misses
        if text.record_id is not None
    }
    pruned = _write_cache(embedder.cache_path, new_rows, keep, requests_made=requests)
    return Embedded(
        vectors=MappingProxyType(vectors),
        requests=requests,
        embedded=len(misses),
        cached=len(texts) - len(misses),
        pruned=pruned,
    )
