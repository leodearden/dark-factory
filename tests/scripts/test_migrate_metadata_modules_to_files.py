"""Tests for scripts/migrate_metadata_modules_to_files.py — the ``_client_name``
seam on :class:`FusedMemoryClient`, and the migration's own branch behaviour.

TWO GROUPS, and they are deliberately different in kind. The first (task 3437)
pins the handshake EXTENSION SURFACE and is described immediately below. The
second (task 4528) pins ``_migrate_one_project``: what it writes, what it
counts, what it skips and what it refuses to call a success. Both run against
constructed doubles — no socket, no ``httpx.AsyncClient``, no live server.

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

Every test in the FIRST group runs against a CONSTRUCTED CLIENT with ``_post``
stubbed out (see :func:`_record_handshake`); every test in the second runs
against :class:`_CannedProject`, which replaces ``call_tool`` outright. No
server is dialled, no socket is opened and no ``httpx.AsyncClient`` is built in
either, so this file costs nothing to run and cannot be made flaky by the state
of a live server.
"""
from __future__ import annotations

import asyncio
import json
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
    _migrate_one_project,
)

# The repo root's conftest.py puts shared/src on sys.path for the whole suite,
# so this needs no bootstrap of its own. Imported to assert the WRITTEN list
# against the very predicate the server-side gate uses
# (``_reject_directory_locks_in_update_metadata`` calls ``directory_locks``),
# rather than against a literal transcribed into this file — a transcription
# would keep passing if the charter moved underneath it.
from shared.locking import directory_locks


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


# ---------------------------------------------------------------------------
# The migration branch itself (task 4528).
# ---------------------------------------------------------------------------


class _CannedProject:
    """Serverless double for :func:`_migrate_one_project`'s one collaborator.

    ``_migrate_one_project`` touches its client through exactly one method —
    ``call_tool`` — so the double replaces that and nothing else. No ``_post``,
    no ``httpx.AsyncClient``, no socket: this is not a subclass of
    :class:`FusedMemoryClient` with a stubbed transport, because subclassing
    would drag in the handshake the first group of tests already covers and
    couple these assertions to it.

    DELIBERATELY A LOCAL DOUBLE, not a conftest fixture. tests/scripts/
    conftest.py is loaded by ~30 unrelated modules in this directory, and a
    double that models ONE script's two tool calls has no business there — the
    same reasoning :func:`_record_handshake` records for its own duplication.

    ``update_reply`` is a callable rather than a value so a test can vary the
    reply per task (task 4528 step 7 needs one task rejected and its neighbour
    accepted in the same pass). It defaults to a plain success dict.
    """

    def __init__(self, tasks: list[dict], *, update_reply: Any = None) -> None:
        self._tasks = tasks
        self.updates: list[dict] = []
        self._update_reply = update_reply or (lambda args: {'success': True})

    async def call_tool(self, name: str, arguments: dict) -> dict:
        if name == 'get_tasks':
            return {'tasks': self._tasks}
        if name == 'update_task':
            self.updates.append(arguments)
            return self._update_reply(arguments)
        raise AssertionError(f'unexpected tool call: {name}')


def _run(client: Any, *, dry_run: bool = False, project_root: str = '/p') -> Any:
    """Drive one migration pass. ``asyncio.run``, matching :func:`_record_handshake`."""
    return asyncio.run(_migrate_one_project(client, project_root, dry_run=dry_run))


def _written(client: _CannedProject, task_id: str) -> dict:
    """Return the metadata dict the migration actually put on the wire for *task_id*.

    Reads it back through ``json.loads`` because the script serialises metadata
    into the ``metadata`` argument as a JSON STRING — asserting on the string
    would pin key order, which is not part of the contract.
    """
    for args in client.updates:
        if args.get('id') == task_id:
            return json.loads(args['metadata'])
    raise AssertionError(f'no update_task recorded for task {task_id}')


def _task(task_id: str, **meta: Any) -> dict:
    """A minimal in-progress task carrying *meta*."""
    return {'id': task_id, 'status': 'pending', 'metadata': dict(meta)}


def test_all_directory_modules_are_not_copied_into_files():
    """THE FIX. An all-directory ``modules`` must not be copied verbatim.

    This is the collision the task exists for.
    ``_reject_directory_locks_in_update_metadata``
    (fused-memory/src/fused_memory/middleware/task_interceptor.py) runs
    ``directory_locks(extract_files(metadata))`` BEFORE the write and rejects
    any ``metadata.files`` carrying a directory-shaped entry. The pre-fix copy
    branch wrote ``modules`` through untouched, so every one of the 11
    copy-branch tasks measured live — all 11 of them all-directory — would have
    been rejected.

    Both live shapes are covered: trailing-slash directories and extension-less
    directory names. The latter is the one a naive "does it end in /" predicate
    gets wrong.

    Asserting NO ``files`` KEY, not ``files == []``: the fix leaves ``new_meta``
    untouched when the sanitised list is empty, so a task whose ``files`` was
    absent stays absent. Writing an explicit ``[]`` would be a new, unasked-for
    write semantic on a task the migration has decided not to give scope to.
    """
    client = _CannedProject([
        _task('1', modules=['fused-memory/scripts/', 'fused-memory/tests/']),
        _task('2', modules=['crates/reify-core', 'crates/reify-ir']),
    ])

    _run(client)

    for task_id in ('1', '2'):
        meta = _written(client, task_id)
        assert 'files' not in meta, f'task {task_id} wrote {meta!r}'
        assert 'modules' not in meta


def test_an_already_empty_files_list_stays_empty_rather_than_being_filled():
    """The other half of "leaves files empty": a pre-existing ``[]`` survives as ``[]``.

    Same branch as above (``not files`` is true for both absent and ``[]``), but
    a different observable: the fix must not DELETE an existing key any more
    than it invents one. Pinned separately because a fix that unconditionally
    popped ``files`` when the sanitised list was empty would pass the test above
    and fail here.
    """
    client = _CannedProject([_task('3', modules=['orchestrator/tests/'], files=[])])

    _run(client)

    meta = _written(client, '3')
    assert meta['files'] == []
    assert 'modules' not in meta


def test_a_mixed_list_keeps_exactly_the_file_level_entries():
    """The surviving-copy case, and the #3248 behaviour a hand-rolled predicate loses.

    ``hooks/pre-commit`` carries no extension and IS a real file — it is in
    ``shared.locking.EXTENSIONLESS_FILENAMES``. A predicate hand-rolled here as
    "contains a dot in the last segment" would drop it, silently narrowing a
    task's scope. That is why the fix imports the charter rather than
    re-deriving it, and why this test uses a name from that frozenset.

    The final assertion runs the WRITTEN list back through ``directory_locks``,
    the exact predicate the server-side gate calls. It is the machine-checkable
    form of "this write would now be accepted", and it needs no literal.
    """
    client = _CannedProject([
        _task('4', modules=['orchestrator/tests/', 'scripts/x.py', 'hooks/pre-commit']),
    ])

    _run(client)

    meta = _written(client, '4')
    assert meta['files'] == ['scripts/x.py', 'hooks/pre-commit']
    assert 'modules' not in meta
    assert directory_locks(meta['files']) == []


def test_the_write_drops_done_provenance_and_preserves_every_other_key():
    """Only ``modules`` and ``done_provenance`` are removed; nothing else moves.

    ``metadata_mode='replace'`` means the dict this script builds IS the task's
    new metadata — anything it forgets to carry over is DESTROYED. So the
    preservation half is not incidental to the fix, it is the thing that makes a
    replace-mode write safe, and it is pinned here rather than assumed.

    ``done_provenance`` is the deliberate exception: ``update_task`` rejects a
    metadata write carrying it (``set_task_status`` is its only sanctioned
    writer), so an orphan stamp on a non-terminal task would make the whole
    write fail.
    """
    client = _CannedProject([
        _task(
            '5',
            modules=['crates/reify-eval'],
            done_provenance={'kind': 'merge', 'sha': 'deadbeef'},
            priority='high',
            milestone='2026-09-01',
            x_note='keep me',
        ),
    ])

    _run(client)

    assert _written(client, '5') == {
        'priority': 'high',
        'milestone': '2026-09-01',
        'x_note': 'keep me',
    }
