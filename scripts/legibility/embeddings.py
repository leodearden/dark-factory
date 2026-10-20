"""The OpenAI embeddings client and the host-local vector cache the census
similarity stage reads (plans/census-incremental-prd.md D3).

Knows nothing of codebooks: it turns texts into vectors. Stdlib plus a lazy
``import httpx``.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from numbers import Real

OPENAI_EMBEDDINGS_URL = 'https://api.openai.com/v1/embeddings'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'

_ERROR_TEXT_LIMIT = 200


class EmbeddingFailed(Exception):
    """No vectors: the request, or the response, failed. Never partial."""

    def __init__(self, message: str, *, requests_made: int = 0) -> None:
        super().__init__(message)
        self.requests_made = requests_made


def _is_vector(value: object) -> bool:
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
