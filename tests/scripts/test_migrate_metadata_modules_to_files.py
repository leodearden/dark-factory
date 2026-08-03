"""Tests for scripts/migrate_metadata_modules_to_files.py — the ``_client_name``
seam on :class:`FusedMemoryClient`.

This script had NO test coverage anywhere in the repo before task 3437. It gets
a file of its own now because 3437 changes its EXTENSION SURFACE, not its
behaviour: ``_initialize`` used to bake ``clientInfo.name`` in as a literal, so
a subclass that needed a different name had to restate the entire handshake —
both JSON-RPC posts, ``protocolVersion`` and the capabilities block — to vary
one leaf. ``scripts/repair_wiped_metadata_files.py`` did exactly that, and
recorded the clone as knowingly-drifting in its own class docstring.

WHY ``clientInfo.name`` IS LOAD-BEARING, NOT COSMETIC. fused-memory's
``_resolve_identity`` (fused-memory/src/fused_memory/server/tools.py) derives a
write's ``agent_id`` from ``ctx.session.client_params.clientInfo.name``. Every
memory write, journal entry and reconciliation event a client makes is filed
under that string, so a tool that handshakes under someone else's name makes its
own writes unattributable — which is the failure exactly when a human is trying
to work out who touched a historical record.

Every test here runs against a CONSTRUCTED CLIENT with ``_post`` stubbed out. No
server is dialled, no socket is opened and no ``httpx.AsyncClient`` is built (see
:func:`_record_handshake`), so this file costs nothing to run and cannot be made
flaky by the state of a live server.
"""
from __future__ import annotations

import asyncio
from typing import Any

# scripts/ is put on sys.path by tests/scripts/conftest.py at collection time,
# and — as of task 3456 — is ALSO listed (with scripts/legibility) in
# [tool.pyright] extraPaths in the root pyproject.toml, so this import resolves
# statically too and needs no ignore. Before 3456 this comment claimed scripts/
# was "deliberately absent" from that table; declaring the `scripts` module's
# type gate required the opposite. Same correction in
# tests/scripts/test_repair_wiped_metadata_files.py.
from migrate_metadata_modules_to_files import (
    FusedMemoryClient,
)


def _record_handshake(client: Any) -> list[dict]:
    """Run ``_initialize`` against a stubbed ``_post`` and return the payloads.

    NO SERVER, NO SOCKET, NO TRANSPORT. ``_initialize`` only ever calls
    ``self._post``, which is replaced here on the INSTANCE (an instance
    attribute shadows the class's method), so the real ``_post`` — and its
    ``assert self._client is not None`` — never runs. The client is constructed
    directly rather than via ``async with``, so ``__aenter__``, the only thing
    that builds an ``httpx.AsyncClient``, never fires either.

    Returning the raw JSON-RPC payloads makes the handshake itself observable,
    which is what lets the seam be asserted on behaviour rather than on source
    text.

    DELIBERATELY A TWIN of the helper in
    tests/scripts/test_repair_wiped_metadata_files.py rather than a shared
    import: importing one test module from another couples their collection, and
    a two-caller double does not belong in a conftest.py that ~20 unrelated
    modules in this directory also load.

    THE COPIES NEVER STRADDLE A SINGLE ASSERTION, which is what makes the
    duplication safe rather than merely cheap. Each file's tests record only
    through their own copy — the cross-client comparison over there
    (``test_repair_handshake_is_the_parents_with_only_the_name_substituted``)
    records BOTH clients through that file's copy — so the two drifting apart
    cannot silently change what any test means.
    """
    posts: list[dict] = []

    async def _fake_post(payload: dict) -> dict:
        posts.append(payload)
        return {}

    client._post = _fake_post
    asyncio.run(client._initialize())
    return posts


def test_initialize_honours_a_subclass_client_name():
    """THE SEAM: a subclass sets ``_client_name`` and the handshake follows it.

    This is the whole point of task 3437. Before it, the only way for a subclass
    to change one leaf of the handshake was to restate all of it, which is a
    silent-drift clone: bump the parent's ``protocolVersion`` and the copy keeps
    handshaking with the stale one, with nothing going red.

    The name has to reach the WIRE, not merely sit on the class — fused-memory
    reads ``clientInfo.name`` off the initialize params to derive the ``agent_id``
    it files a write under, so an attribute the handshake ignores would attribute
    nothing. Hence asserting on the recorded payload rather than on the attribute.
    """

    class _Named(FusedMemoryClient):
        _client_name = 'some-other-tool'

    posts = _record_handshake(_Named('http://127.0.0.1:9'))

    assert posts[0]['method'] == 'initialize'
    assert posts[0]['params']['clientInfo']['name'] == 'some-other-tool'


def test_initialize_defaults_to_migrate_metadata():
    """THE NON-REGRESSION HALF: an un-overridden client is still the migration.

    The seam must not change what THIS script puts on the wire. Its own writes
    are filed under ``agent_id='migrate-metadata'`` — a string that is already
    recorded in the journal of every project it has been run against — so a
    refactor that quietly renamed it would retroactively split one tool's history
    across two identities.

    Pinned on a bare ``FusedMemoryClient`` so the default lives at the class
    surface, not inside ``_initialize``: a future reader looking for what to
    override sees the documented attribute, and this test fails if the default is
    ever moved or renamed out from under it.
    """
    posts = _record_handshake(FusedMemoryClient('http://127.0.0.1:9'))

    assert posts[0]['params']['clientInfo']['name'] == 'migrate-metadata'


def test_initialize_posts_both_steps_of_the_mcp_handshake():
    """A dropped ``notifications/initialized`` is a real defect. Pin the steps.

    MCP requires that second post before any ``tools/call`` is accepted. Lose it
    and ``_initialize`` still returns cleanly — the failure surfaces later, as
    every subsequent call being refused by a live server — so nothing else in
    this file's stubbed, serverless setup would notice.

    ONLY THE TWO METHOD NAMES ARE PINNED, deliberately. This does not assert the
    ``protocolVersion``, the capabilities block or the ``clientInfo.version``:
    those literals exist nowhere but the one implementation this file tests, so
    restating them here cannot detect a defect — there is no second source of
    truth for them to disagree with — while still going red on a legitimate
    protocol bump. That is a pure change-detector: false positives, no true
    positives. It is also the same objection that deleted 3329's source-scraping
    guard at 57eb02b53f, one abstraction level up.

    Drift between the parent and the repair client is guarded where it is
    genuinely observable, by
    ``test_repair_handshake_is_the_parents_with_only_the_name_substituted``,
    which compares the two clients' RECORDED payloads against each other and
    needs no literal from this file.
    """
    posts = _record_handshake(FusedMemoryClient('http://127.0.0.1:9'))

    assert [p['method'] for p in posts] == ['initialize', 'notifications/initialized']
