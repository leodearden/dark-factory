"""Tests for the live-Qdrant scaffolding in _fm_qdrant_live.py."""

from unittest.mock import MagicMock

import httpx
import pytest

# ---------------------------------------------------------------------------
# Tests for shared Qdrant test scaffolding (task 2277)
# ---------------------------------------------------------------------------
# De-duplicates the _qdrant_available()/QDRANT_URL/skip-marker scaffolding that
# was previously copy-pasted verbatim in test_recon_dedup_premise.py and
# test_mem0_qdrant_integration.py (flagged by esc-2221-26). No real Qdrant is
# used here — qdrant_client.QdrantClient is monkeypatched throughout.

class TestQdrantHelpers:
    """Unit tests for QDRANT_URL, _qdrant_available(), and qdrant_skipif()."""

    def test_qdrant_url_constant(self):
        """QDRANT_URL matches the local-Qdrant default both integration files used."""
        from _fm_qdrant_live import QDRANT_URL

        assert QDRANT_URL == 'http://localhost:6333'

    def test_qdrant_available_true_when_client_ok(self, monkeypatch):
        """_qdrant_available() returns True when construction, get_collections, and close all succeed."""
        from _fm_qdrant_live import _qdrant_available

        fake_client = MagicMock()
        fake_client.get_collections.return_value = None
        fake_client.close.return_value = None
        monkeypatch.setattr('qdrant_client.QdrantClient', MagicMock(return_value=fake_client))

        assert _qdrant_available() is True
        fake_client.get_collections.assert_called_once()
        fake_client.close.assert_called_once()

    def test_qdrant_available_false_when_get_collections_raises(self, monkeypatch):
        """_qdrant_available() returns False (never raises) when get_collections() raises."""
        from _fm_qdrant_live import _qdrant_available

        fake_client = MagicMock()
        fake_client.get_collections.side_effect = Exception('boom')
        monkeypatch.setattr('qdrant_client.QdrantClient', MagicMock(return_value=fake_client))

        assert _qdrant_available() is False

    def test_qdrant_available_false_when_construct_raises(self, monkeypatch):
        """_qdrant_available() returns False (never raises) when QdrantClient(...) itself raises."""
        from _fm_qdrant_live import _qdrant_available

        def _boom(*args, **kwargs):
            raise Exception('connection refused')

        monkeypatch.setattr('qdrant_client.QdrantClient', _boom)

        assert _qdrant_available() is False

    def test_qdrant_skipif_default_reason(self):
        """qdrant_skipif() returns a pytest.mark.skipif marker with a non-empty default reason."""
        from _fm_qdrant_live import qdrant_skipif

        m = qdrant_skipif()

        assert m.name == 'skipif'
        assert isinstance(m.kwargs['reason'], str) and m.kwargs['reason']

    def test_qdrant_skipif_custom_reason(self):
        """qdrant_skipif(reason=...) propagates a caller-supplied reason string."""
        from _fm_qdrant_live import qdrant_skipif

        assert qdrant_skipif(reason='x').kwargs['reason'] == 'x'

    def test_qdrant_skipif_condition_tracks_probe(self, monkeypatch):
        """qdrant_skipif()'s skip condition is `not _qdrant_available()`, evaluated at call time.

        Pins the behaviour-preserving contract: the marker must track whatever
        _qdrant_available() currently reports, not a value cached at import time.
        """
        import _fm_qdrant_live

        monkeypatch.setattr(_fm_qdrant_live, '_qdrant_available', lambda: False)
        assert _fm_qdrant_live.qdrant_skipif().args[0] is True

        monkeypatch.setattr(_fm_qdrant_live, '_qdrant_available', lambda: True)
        assert _fm_qdrant_live.qdrant_skipif().args[0] is False


# ---------------------------------------------------------------------------
# Tests for the shared ensure_fresh_collection() helper (task 2773, item 3)
# ---------------------------------------------------------------------------
# Idempotent + 409-tolerant collection (re)creation, used by the qdrant
# integration fixture. No real Qdrant is used here — the client is a
# MagicMock and qdrant_client.http.exceptions.UnexpectedResponse is
# constructed directly — so this suite runs in the default merge-verify lane
# even though its one live consumer (the qdrant fixture) is integration-only.
# Assertions are on call counts and ordering only, never on live Qdrant.

class TestEnsureFreshCollection:
    """Unit tests for ensure_fresh_collection(client, collection_name, *, size, distance)."""

    def test_creates_once_when_absent(self):
        """collection_exists()->False: create_collection called exactly once, no delete."""
        from _fm_qdrant_live import ensure_fresh_collection

        client = MagicMock()
        client.collection_exists.return_value = False

        ensure_fresh_collection(client, 'c', size=8)

        client.create_collection.assert_called_once()
        client.delete_collection.assert_not_called()

    def test_deletes_then_recreates_when_present(self):
        """collection_exists()->True: delete_collection precedes create_collection."""
        from _fm_qdrant_live import ensure_fresh_collection

        client = MagicMock()
        client.collection_exists.return_value = True

        ensure_fresh_collection(client, 'c', size=8)

        names = [c[0] for c in client.method_calls]
        assert names == ['collection_exists', 'delete_collection', 'create_collection']

    def test_conflict_on_create_triggers_single_delete_and_retry(self):
        """A 409 on the first create_collection triggers exactly one delete then a second create.

        Models the swallowed-teardown stale-collection case the helper exists
        to self-heal: the collection was reported absent but a concurrent /
        prior run left it behind, so the create 409-conflicts; the helper
        deletes and recreates rather than propagating the conflict.
        """
        from _fm_qdrant_live import ensure_fresh_collection
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = [
            UnexpectedResponse(409, 'Conflict', b'{}', httpx.Headers()),
            None,
        ]

        # Must not raise — the 409 is tolerated.
        ensure_fresh_collection(client, 'c', size=8)

        assert client.create_collection.call_count == 2
        assert client.delete_collection.call_count == 1
        names = [c[0] for c in client.method_calls]
        assert names == [
            'collection_exists',
            'create_collection',
            'delete_collection',
            'create_collection',
        ]

    def test_non_conflict_create_error_propagates(self):
        """A non-409 UnexpectedResponse (e.g. 500) propagates — fail loud, no retry."""
        from _fm_qdrant_live import ensure_fresh_collection
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.collection_exists.return_value = False
        client.create_collection.side_effect = UnexpectedResponse(500, 'err', b'{}', httpx.Headers())

        with pytest.raises(UnexpectedResponse):
            ensure_fresh_collection(client, 'c', size=8)

        client.delete_collection.assert_not_called()
        assert client.create_collection.call_count == 1


# ---------------------------------------------------------------------------
# Tests for the shared collection_vector_size() helper (task 2773, item 4)
# ---------------------------------------------------------------------------
# Race-safe read of a collection's vector dimension: under `-n auto` load a
# just-created collection's get_collection() can transiently 404 (or the HTTP
# layer can raise ResponseHandlingException) before it is durably visible, so
# a single naive read flakes. The helper wraps the read in poll_until so the
# transient window is a bounded retry, while a non-404 error (e.g. a genuine
# 500) is re-raised immediately — fail loud. No real Qdrant is used here: the
# client is a MagicMock and get_collection is scripted. Following
# TestPollUntil's discipline, cases assert only on call counts and raises,
# never on elapsed wall-clock, so the suite stays load-independent.

class TestCollectionVectorSize:
    """Unit tests for async collection_vector_size(client, collection, *, timeout, interval)."""

    @staticmethod
    def _info(size: int):
        """A fake get_collection() return value exposing .config.params.vectors."""
        from qdrant_client.models import Distance, VectorParams

        info = MagicMock()
        info.config.params.vectors = VectorParams(size=size, distance=Distance.COSINE)
        return info

    @pytest.mark.asyncio
    async def test_returns_size_after_one_call_when_readable(self):
        """A readable collection returns its vector size after exactly 1 get_collection call."""
        from _fm_qdrant_live import collection_vector_size

        client = MagicMock()
        client.get_collection.return_value = self._info(8)

        size = await collection_vector_size(client, 'c', timeout=5.0, interval=0.001)

        assert size == 8
        assert client.get_collection.call_count == 1

    @pytest.mark.asyncio
    async def test_tolerates_transient_404_then_returns(self):
        """Two transient 404s then a readable collection: returns 8 after exactly 3 calls.

        Proves the transient-404 window is a bounded retry, not a single naive
        read — the call count is deterministic (scripted side_effect), so this
        assertion is load-independent.
        """
        from _fm_qdrant_live import collection_vector_size
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collection.side_effect = [
            UnexpectedResponse(404, 'Not Found', b'{}', httpx.Headers()),
            UnexpectedResponse(404, 'Not Found', b'{}', httpx.Headers()),
            self._info(8),
        ]

        size = await collection_vector_size(client, 'c', timeout=5.0, interval=0.001)

        assert size == 8
        assert client.get_collection.call_count == 3

    @pytest.mark.asyncio
    async def test_persistent_404_raises_assertion_error(self):
        """A get_collection that always 404s raises AssertionError once the deadline passes.

        Uses a tiny timeout and asserts only the raise and that the probe ran
        at least once (never elapsed wall-clock or an exact poll count) — a
        slow host makes this slower, never falsely passing or failing.
        """
        from _fm_qdrant_live import collection_vector_size
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collection.side_effect = UnexpectedResponse(404, 'Not Found', b'{}', httpx.Headers())

        with pytest.raises(AssertionError):
            await collection_vector_size(client, 'c', timeout=0.05, interval=0.01)

        assert client.get_collection.call_count >= 1

    @pytest.mark.asyncio
    async def test_non_404_error_propagates_immediately(self):
        """A non-404 UnexpectedResponse (e.g. 500) propagates on the first probe — fail loud."""
        from _fm_qdrant_live import collection_vector_size
        from qdrant_client.http.exceptions import UnexpectedResponse

        client = MagicMock()
        client.get_collection.side_effect = UnexpectedResponse(500, 'err', b'{}', httpx.Headers())

        with pytest.raises(UnexpectedResponse):
            await collection_vector_size(client, 'c', timeout=5.0, interval=0.001)

        assert client.get_collection.call_count == 1
