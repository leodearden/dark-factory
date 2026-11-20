"""What a test needs to reach a live Qdrant: URL, reachability gate, and
collection (re)create/read barriers.

Lives outside conftest.py, imported by name, under the convention stated in
_fm_helpers' module docstring.
"""

import contextlib
from collections.abc import Callable
from typing import Any

import pytest
from _fm_polling import poll_until, poll_until_sync

# ---------------------------------------------------------------------------
# Shared Qdrant test scaffolding (task 2277)
# ---------------------------------------------------------------------------
#
# De-duplicates the _qdrant_available()/QDRANT_URL/skip-marker trio that was
# previously copy-pasted verbatim in test_recon_dedup_premise.py and
# test_mem0_qdrant_integration.py (flagged by esc-2221-26).
# ---------------------------------------------------------------------------

QDRANT_URL = 'http://localhost:6333'

#: The mem0 collection_prefix whose collections scripts/cleanup_test_collections.py reaps (its PREFIX).
EPHEMERAL_COLLECTION_PREFIX = '_test_mem0_qdrant_integration'

#: Seconds an abandoned collection meta-op may take to show its effect (task 10599).
META_OP_SETTLE_TIMEOUT_S = 15.0


def _qdrant_available() -> bool:
    """Probe whether a Qdrant instance is reachable at QDRANT_URL.

    Imports QdrantClient lazily (rather than at module scope) so that
    importing this module never charges its importer for qdrant_client, and
    so tests can deterministically patch ``qdrant_client.QdrantClient``.
    """
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2)
    except Exception:
        return False

    try:
        client.get_collections()
        return True
    except Exception:
        return False
    finally:
        with contextlib.suppress(Exception):
            client.close()


def qdrant_skipif(reason: str = 'Qdrant not reachable'):
    """Return a ``pytest.mark.skipif`` marker gated on `_qdrant_available()`.

    A factory rather than a module-level marker constant: a constant would
    freeze whatever the probe reported when this module was first imported.
    Calling this from a module's ``pytestmark`` instead defers the probe to
    that module's own collection time — identical timing to each file's
    previous local copy.
    """
    return pytest.mark.skipif(not _qdrant_available(), reason=reason)


def _run_meta_op(
    request: Callable[[], Any],
    *,
    reached: Callable[[], bool],
    describe: str,
    expected: str,
    timeout: float,
) -> None:
    """Send one collection meta-op; if its answer is abandoned on a read timeout, wait for *reached*.

    A read timeout means the server has the request and may still be applying
    it, so the outcome is judged by the observed state, never by re-sending.
    Every other failure propagates unchanged.
    """
    import httpx
    from qdrant_client.http.exceptions import ResponseHandlingException

    try:
        request()
    except ResponseHandlingException as exc:
        if not isinstance(exc.source, httpx.ReadTimeout):
            raise
        poll_until_sync(
            reached,
            timeout=timeout,
            interval=0.05,
            message=(
                f'{describe}: the client stopped waiting for the answer '
                f'({type(exc.source).__name__}: {exc.source}), and the '
                f'collection was not {expected} {timeout}s later'
            ),
        )


def _is_present_and_empty(client: Any, collection_name: str) -> bool:
    """Whether *collection_name* exists and holds no points; a transient 404 on the count reads as not yet."""
    from qdrant_client.http.exceptions import UnexpectedResponse

    if not client.collection_exists(collection_name):
        return False
    try:
        return client.count(collection_name, exact=True).count == 0
    except UnexpectedResponse as exc:
        if getattr(exc, 'status_code', None) != 404:
            raise
        return False


def _create_collection(
    client: Any, collection_name: str, vectors_config: Any, *, settle_timeout: float,
) -> None:
    _run_meta_op(
        lambda: client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config,
        ),
        reached=lambda: _is_present_and_empty(client, collection_name),
        describe=f'create of collection {collection_name!r}',
        expected='present and empty',
        timeout=settle_timeout,
    )


def _delete_collection(client: Any, collection_name: str, *, settle_timeout: float) -> None:
    _run_meta_op(
        lambda: client.delete_collection(collection_name),
        reached=lambda: not client.collection_exists(collection_name),
        describe=f'delete of collection {collection_name!r}',
        expected='absent',
        timeout=settle_timeout,
    )


def ensure_fresh_collection(
    client: Any,
    collection_name: str,
    *,
    size: int,
    distance: Any = None,
    settle_timeout: float = META_OP_SETTLE_TIMEOUT_S,
) -> None:
    """Idempotently (re)create *collection_name* as an empty size-*size* collection.

    Replaces the fragile ``delete_collection(suppressed) + create_collection``
    pair each integration fixture rolled by hand. Under ``-n auto`` xdist load,
    a prior run's teardown can be swallowed (or a concurrent worker can win the
    create race), leaving a stale collection behind that ``collection_exists()``
    may still report absent — so a naive create then 409-conflicts and fails the
    next run. This helper self-heals that case: if the create raises a 409
    ``UnexpectedResponse`` it deletes the stale collection and creates once more.
    Any other ``UnexpectedResponse`` (e.g. a genuine 500) is re-raised — fail
    loud, no retry. A create or delete whose answer the client abandons on a
    read timeout is judged by whether the collection appears empty, or
    disappears, within *settle_timeout*. An abandoned create cannot tell its
    own collection from a stale one whose 409 it never read, so one that
    appears holding points fails rather than passing as fresh.

    Imports ``VectorParams``/``Distance``/``UnexpectedResponse`` lazily inside
    the body, mirroring ``_qdrant_available``'s convention (see its docstring):
    a module-scope qdrant import would charge every importing module for
    qdrant_client and break tests that monkeypatch ``qdrant_client.QdrantClient``.

    Args:
        client: A ``qdrant_client.QdrantClient`` (or test double) exposing
            ``collection_exists`` / ``delete_collection`` / ``create_collection``
            / ``count``.
        collection_name: The collection to (re)create.
        size: Vector dimensionality for the new collection.
        distance: Optional ``qdrant_client.models.Distance``; defaults to
            ``Distance.COSINE`` when None.
        settle_timeout: Seconds to wait for an abandoned meta-op's effect.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance, VectorParams

    vectors_config = VectorParams(size=size, distance=distance or Distance.COSINE)

    if client.collection_exists(collection_name):
        _delete_collection(client, collection_name, settle_timeout=settle_timeout)
    try:
        _create_collection(
            client, collection_name, vectors_config, settle_timeout=settle_timeout,
        )
    except UnexpectedResponse as exc:
        if getattr(exc, 'status_code', None) != 409:
            raise
        # Stale collection survived a swallowed teardown / lost create race:
        # delete it and create once more rather than propagating the conflict.
        _delete_collection(client, collection_name, settle_timeout=settle_timeout)
        _create_collection(
            client, collection_name, vectors_config, settle_timeout=settle_timeout,
        )


async def collection_vector_size(
    client: Any,
    collection: str,
    *,
    timeout: float = 15.0,
    interval: float = 0.05,
) -> int:
    """Read *collection*'s vector dimension, tolerating the transient-404 read race.

    Under ``-n auto`` xdist load a collection created moments earlier can
    transiently 404 (or the HTTP layer can raise ``ResponseHandlingException``)
    before it is durably readable, so a single naive ``get_collection`` flakes.
    This wraps the read in the shared :func:`poll_until` engine so the
    transient window becomes a bounded retry: a 404 ``UnexpectedResponse`` or a
    ``ResponseHandlingException`` maps to "keep polling", while any other
    ``UnexpectedResponse`` (e.g. a genuine 500) is re-raised immediately —
    fail loud, no retry. A collection that never becomes readable surfaces as
    the usual :func:`poll_until` ``AssertionError`` at the deadline.

    Imports the qdrant symbols lazily inside the body, mirroring
    :func:`_qdrant_available` / :func:`ensure_fresh_collection` (see their
    docstrings) so importing this module never pulls in qdrant_client.

    Args:
        client: A ``qdrant_client.QdrantClient`` (or test double) exposing
            ``get_collection``.
        collection: The collection to read.
        timeout: Seconds to keep retrying the transient-404 window before
            raising. Defaults to 15s — generous enough to absorb host CPU
            oversubscription while still catching a genuinely-missing collection.
        interval: Seconds to sleep between probes. Defaults to 0.05s.

    Returns:
        The collection's single-vector dimensionality.

    Raises:
        AssertionError: the collection never became readable within *timeout*.
        qdrant_client.http.exceptions.UnexpectedResponse: a non-404 server
            error on the read (surfaced immediately, not retried).
    """
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
    from qdrant_client.models import VectorParams

    def _probe():
        try:
            return client.get_collection(collection)
        except UnexpectedResponse as exc:
            if getattr(exc, 'status_code', None) == 404:
                return None  # transient: not yet durably visible — keep polling
            raise  # genuine server error — fail loud, no retry
        except ResponseHandlingException:
            return None  # transient HTTP-layer hiccup — keep polling

    info = await poll_until(
        _probe,
        timeout=timeout,
        interval=interval,
        message=f'collection {collection!r} not readable within {timeout}s',
    )
    vectors = info.config.params.vectors
    assert isinstance(vectors, VectorParams), (
        f'collection {collection!r} vectors config is not a single VectorParams: {vectors!r}'
    )
    return vectors.size
