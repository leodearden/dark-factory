"""Tests for scripts/legibility/embeddings.py — the OpenAI embeddings client
and the host-local vector cache the census similarity stage reads
(plans/census-incremental-prd.md D3)."""
from __future__ import annotations

import pytest
from legibility import embeddings

_KEY = 'sk-test-secret-key-0123456789'
_MODEL = 'text-embedding-3-small'


class _Response:
    def __init__(self, status_code: int, body: object, text: str = '') -> None:
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self) -> object:
        return self._body


def _ok_body(vectors: list[list[float]], order: list[int] | None = None) -> dict:
    indexes = order if order is not None else list(range(len(vectors)))
    return {
        'object': 'list',
        'data': [
            {'object': 'embedding', 'index': index, 'embedding': vectors[index]}
            for index in indexes
        ],
        'model': _MODEL,
    }


class _RecordingPost:
    def __init__(self, response: object = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._response = response
        self._raises = raises

    def __call__(self, url: str, **kwargs: object) -> object:
        self.calls.append((url, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._response


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv(embeddings.OPENAI_API_KEY_ENV, _KEY)


class TestOpenAIEmbed:
    def test_posts_once_and_orders_vectors_by_index(self, with_key, install_fake_httpx):
        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        post = _RecordingPost(_Response(200, _ok_body(vectors, order=[2, 0, 1])))
        install_fake_httpx(post)

        got = embeddings.openai_embed(['a', 'b', 'c'], model=_MODEL)

        assert got == vectors
        assert len(post.calls) == 1
        url, kwargs = post.calls[0]
        assert url == embeddings.OPENAI_EMBEDDINGS_URL == 'https://api.openai.com/v1/embeddings'
        assert kwargs['headers']['Authorization'] == f'Bearer {_KEY}'
        assert kwargs['json'] == {'model': _MODEL, 'input': ['a', 'b', 'c']}

    @pytest.mark.parametrize('value', [None, ''])
    def test_missing_key_raises_before_any_request(self, monkeypatch, install_fake_httpx, value):
        if value is None:
            monkeypatch.delenv(embeddings.OPENAI_API_KEY_ENV, raising=False)
        else:
            monkeypatch.setenv(embeddings.OPENAI_API_KEY_ENV, value)
        post = _RecordingPost(_Response(200, _ok_body([[1.0]])))
        install_fake_httpx(post)

        with pytest.raises(embeddings.EmbeddingFailed, match='OPENAI_API_KEY'):
            embeddings.openai_embed(['a'], model=_MODEL)
        assert post.calls == []

    def test_non_200_status_raises_naming_the_status(self, with_key, install_fake_httpx):
        post = _RecordingPost(_Response(503, {}, text='upstream unavailable'))
        install_fake_httpx(post)

        with pytest.raises(embeddings.EmbeddingFailed, match='503') as excinfo:
            embeddings.openai_embed(['a'], model=_MODEL)
        assert _KEY not in str(excinfo.value)
        assert len(post.calls) == 1

    def test_transport_error_raises_carrying_the_cause(self, with_key, install_fake_httpx):
        post = _RecordingPost(raises=RuntimeError('connect'))
        install_fake_httpx(post)

        with pytest.raises(embeddings.EmbeddingFailed, match='connect') as excinfo:
            embeddings.openai_embed(['a'], model=_MODEL)
        assert _KEY not in str(excinfo.value)
        assert len(post.calls) == 1

    @pytest.mark.parametrize('body', [
        pytest.param({'object': 'list'}, id='no-data'),
        pytest.param({'data': 'nope'}, id='data-not-a-list'),
        pytest.param(_ok_body([[0.1, 0.2]]), id='count-mismatch'),
        pytest.param(
            {'data': [{'index': 0, 'embedding': ['x', 'y']},
                      {'index': 1, 'embedding': [0.1, 0.2]}]},
            id='non-numeric-embedding',
        ),
        pytest.param(
            {'data': [{'index': 0, 'embedding': []},
                      {'index': 1, 'embedding': [0.1, 0.2]}]},
            id='empty-embedding',
        ),
        pytest.param(
            {'data': [{'index': 0, 'embedding': [True, False]},
                      {'index': 1, 'embedding': [0.1, 0.2]}]},
            id='boolean-embedding',
        ),
        pytest.param(
            {'data': [{'embedding': [0.1]}, {'index': 1, 'embedding': [0.1]}]},
            id='missing-index',
        ),
    ])
    def test_malformed_body_raises(self, with_key, install_fake_httpx, body):
        post = _RecordingPost(_Response(200, body))
        install_fake_httpx(post)

        with pytest.raises(embeddings.EmbeddingFailed) as excinfo:
            embeddings.openai_embed(['a', 'b'], model=_MODEL)
        assert _KEY not in str(excinfo.value)
        assert len(post.calls) == 1

    def test_unparseable_json_raises(self, with_key, install_fake_httpx):
        class _BadJson(_Response):
            def json(self) -> object:
                raise ValueError('Expecting value')

        post = _RecordingPost(_BadJson(200, None))
        install_fake_httpx(post)

        with pytest.raises(embeddings.EmbeddingFailed):
            embeddings.openai_embed(['a'], model=_MODEL)
        assert len(post.calls) == 1
