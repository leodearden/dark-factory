"""Tests for scripts/legibility/embeddings.py — the OpenAI embeddings client
and the host-local vector cache the census similarity stage reads
(plans/census-incremental-prd.md D3)."""
from __future__ import annotations

import math
import re

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


class _RecordingEmbed:
    """A fake embed_fn: a deterministic vector per (text, model), recording
    each call's texts. ``fail_on`` makes the call with that 1-based number
    raise EmbeddingFailed; ``reply`` overrides what a call returns."""

    def __init__(self, *, fail_on: int | None = None, reply=None) -> None:
        self.calls: list[list[str]] = []
        self._fail_on = fail_on
        self._reply = reply

    @staticmethod
    def vector(text: str, model: str) -> list[float]:
        seed = sum(map(ord, f'{model}:{text}'))
        return [float(seed % 7 + 1), float(seed % 5 + 1), float(seed % 3 + 1)]

    def __call__(self, texts, *, model):
        self.calls.append(list(texts))
        if self._fail_on == len(self.calls):
            raise embeddings.EmbeddingFailed('embeddings endpoint answered HTTP 503')
        if self._reply is not None:
            return self._reply(texts)
        return [self.vector(text, model) for text in texts]

    @property
    def embedded_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _texts(*pairs: tuple[str | None, str]) -> list[embeddings.Text]:
    return [
        embeddings.Text(key=f'k{i}', record_id=record_id, text=text)
        for i, (record_id, text) in enumerate(pairs)
    ]


def _embedder(tmp_path, embed_fn) -> embeddings.Embedder:
    return embeddings.Embedder(embed_fn=embed_fn, cache_path=tmp_path / 'embeddings.sqlite')


class TestVectorsFor:
    def test_first_call_embeds_everything_in_batches(self, tmp_path):
        fake = _RecordingEmbed()
        texts = _texts(*[(f'r{i}', f'text {i}') for i in range(5)])

        got = embeddings.vectors_for(
            texts, embedder=_embedder(tmp_path, fake), model=_MODEL, batch_size=2,
        )

        assert [len(call) for call in fake.calls] == [2, 2, 1]
        assert set(got.vectors) == {t.key for t in texts}
        for text in texts:
            assert got.vectors[text.key] == pytest.approx(fake.vector(text.text, _MODEL))
        assert (got.requests, got.embedded, got.cached, got.pruned) == (3, 5, 0, 0)

    def test_second_call_is_served_from_the_cache(self, tmp_path):
        texts = _texts(('r1', 'alpha'), ('r2', 'beta'))
        first = embeddings.vectors_for(
            texts, embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )
        fake = _RecordingEmbed()

        second = embeddings.vectors_for(texts, embedder=_embedder(tmp_path, fake), model=_MODEL)

        assert fake.calls == []
        assert (second.requests, second.embedded, second.cached, second.pruned) == (0, 0, 2, 0)
        assert dict(second.vectors) == dict(first.vectors)

    def test_changed_text_is_re_embedded_and_its_stale_row_pruned(self, tmp_path):
        embeddings.vectors_for(
            _texts(('r1', 'alpha'), ('r2', 'beta')),
            embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )
        fake = _RecordingEmbed()

        got = embeddings.vectors_for(
            _texts(('r1', 'alpha, reworded'), ('r2', 'beta')),
            embedder=_embedder(tmp_path, fake), model=_MODEL,
        )

        assert fake.embedded_texts == ['alpha, reworded']
        assert (got.embedded, got.cached, got.pruned) == (1, 1, 1)

    def test_changed_model_is_re_embedded(self, tmp_path):
        texts = _texts(('r1', 'alpha'))
        embeddings.vectors_for(
            texts, embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )
        fake = _RecordingEmbed()

        got = embeddings.vectors_for(
            texts, embedder=_embedder(tmp_path, fake), model='text-embedding-3-large',
        )

        assert fake.embedded_texts == ['alpha']
        assert got.vectors['k0'] == pytest.approx(fake.vector('alpha', 'text-embedding-3-large'))

    def test_rows_of_records_no_longer_indexed_are_pruned(self, tmp_path):
        embeddings.vectors_for(
            _texts(('r1', 'alpha'), ('r2', 'beta'), ('r3', 'gamma')),
            embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )

        got = embeddings.vectors_for(
            _texts(('r2', 'beta')), embedder=_embedder(tmp_path, _RecordingEmbed()),
            model=_MODEL,
        )
        assert got.pruned == 2

        fake = _RecordingEmbed()
        embeddings.vectors_for(
            _texts(('r1', 'alpha'), ('r2', 'beta')),
            embedder=_embedder(tmp_path, fake), model=_MODEL,
        )
        assert fake.embedded_texts == ['alpha']

    def test_ephemeral_text_is_embedded_every_call_and_never_cached(self, tmp_path):
        texts = _texts(('r1', 'alpha'), (None, 'a novel cluster'))
        embeddings.vectors_for(
            texts, embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )
        fake = _RecordingEmbed()

        got = embeddings.vectors_for(texts, embedder=_embedder(tmp_path, fake), model=_MODEL)

        assert fake.embedded_texts == ['a novel cluster']
        assert (got.embedded, got.cached, got.pruned) == (1, 1, 0)
        assert set(got.vectors) == {'k0', 'k1'}

    def test_failed_batch_propagates_and_writes_nothing(self, tmp_path):
        embeddings.vectors_for(
            _texts(('kept', 'kept text')),
            embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )
        failing = _RecordingEmbed(fail_on=2)
        texts = _texts(*[(f'r{i}', f'text {i}') for i in range(5)])

        with pytest.raises(embeddings.EmbeddingFailed, match='503') as excinfo:
            embeddings.vectors_for(
                texts, embedder=_embedder(tmp_path, failing), model=_MODEL, batch_size=2,
            )
        assert excinfo.value.requests_made == 2
        assert len(failing.calls) == 2

        fake = _RecordingEmbed()
        after = embeddings.vectors_for(
            _texts(('kept', 'kept text'), ('r0', 'text 0')),
            embedder=_embedder(tmp_path, fake), model=_MODEL,
        )
        assert fake.embedded_texts == ['text 0']
        assert after.cached == 1

    @pytest.mark.parametrize('reply', [
        pytest.param(lambda texts: [[1.0, 2.0]] * (len(texts) - 1), id='wrong-count'),
        pytest.param(
            lambda texts: [[1.0, 2.0]] + [[1.0, 2.0, 3.0]] * (len(texts) - 1),
            id='ragged-dimensions',
        ),
        pytest.param(lambda texts: [[0.0, 0.0]] * len(texts), id='all-zero'),
        pytest.param(lambda texts: [[]] * len(texts), id='empty'),
        pytest.param(lambda texts: [[1.0, math.nan]] * len(texts), id='nan'),
        pytest.param(lambda texts: [[1.0, -math.inf]] * len(texts), id='inf'),
        pytest.param(lambda texts: [[1.0, 1e300]] * len(texts), id='float32-overflow'),
    ])
    def test_unusable_vectors_raise(self, tmp_path, reply):
        fake = _RecordingEmbed(reply=reply)

        with pytest.raises(embeddings.EmbeddingFailed) as excinfo:
            embeddings.vectors_for(
                _texts(('r1', 'alpha'), ('r2', 'beta')),
                embedder=_embedder(tmp_path, fake), model=_MODEL,
            )
        assert excinfo.value.requests_made == 1

    def test_a_cached_vector_of_another_dimension_raises_naming_the_cache_and_writes_nothing(self, tmp_path):
        embeddings.vectors_for(
            _texts(('r1', 'alpha')), embedder=_embedder(tmp_path, _RecordingEmbed()), model=_MODEL,
        )
        before = (tmp_path / 'embeddings.sqlite').read_bytes()
        wider = _RecordingEmbed(reply=lambda texts: [[1.0, 2.0, 3.0, 4.0]] * len(texts))

        with pytest.raises(embeddings.EmbeddingFailed, match=re.escape(str(tmp_path / 'embeddings.sqlite'))) as excinfo:
            embeddings.vectors_for(
                _texts(('r1', 'alpha'), ('r2', 'beta')), embedder=_embedder(tmp_path, wider), model=_MODEL,
            )
        assert excinfo.value.requests_made == 1
        assert (tmp_path / 'embeddings.sqlite').read_bytes() == before

    def test_batches_of_different_dimensions_raise(self, tmp_path):
        widths = iter([2, 3])
        fake = _RecordingEmbed(reply=lambda texts: [[1.0] * next(widths)] * len(texts))

        with pytest.raises(embeddings.EmbeddingFailed) as excinfo:
            embeddings.vectors_for(
                _texts(('r1', 'alpha'), ('r2', 'beta')),
                embedder=_embedder(tmp_path, fake), model=_MODEL, batch_size=1,
            )
        assert excinfo.value.requests_made == 2
        assert not (tmp_path / 'embeddings.sqlite').exists()

    def test_non_sqlite_cache_file_raises_naming_the_path(self, tmp_path):
        path = tmp_path / 'embeddings.sqlite'
        path.write_bytes(b'this is not a sqlite database, not even close' * 20)
        fake = _RecordingEmbed()

        with pytest.raises(embeddings.VectorCacheError, match=re.escape(str(path))):
            embeddings.vectors_for(
                _texts(('r1', 'alpha')),
                embedder=embeddings.Embedder(embed_fn=fake, cache_path=path), model=_MODEL,
            )
        assert fake.calls == []

    def test_nothing_to_embed_creates_no_cache_file(self, tmp_path):
        fake = _RecordingEmbed()
        got = embeddings.vectors_for([], embedder=_embedder(tmp_path, fake), model=_MODEL)

        assert dict(got.vectors) == {}
        assert (got.requests, got.embedded, got.cached, got.pruned) == (0, 0, 0, 0)
        assert not (tmp_path / 'embeddings.sqlite').exists()


def test_cache_path_sits_in_the_project_state_dir(tmp_path, monkeypatch):
    from legibility import trickle_state

    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path))

    path = embeddings.cache_path('dark_factory')

    assert path.parent == trickle_state.project_state_dir('dark_factory')
    assert path.name == embeddings.CACHE_FILENAME == 'embeddings.sqlite'
    assert path.is_relative_to(tmp_path)
