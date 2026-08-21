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

import argparse
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
import migrate_metadata_modules_to_files as migrate_mod
from migrate_metadata_modules_to_files import (
    FusedMemoryClient,
    _migrate_one_project,
    main_async,
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


class _CannedServer:
    """Stand in for the ``FusedMemoryClient`` CLASS so ``main_async`` can be driven.

    ``main_async`` builds its client with ``async with FusedMemoryClient(url)``,
    so the seam for a serverless end-to-end test is the class name in the
    module namespace, not the instance. This object is callable (standing in for
    the constructor) and is its own async context manager, yielding the
    :class:`_CannedProject` every root then shares.

    Driving the REAL ``main_async`` rather than re-deriving its arithmetic here
    is the point: the per-project line and the ``---- summary ----`` totals are
    the operator-facing surface this task has to change, and a test that
    recomputed them from the returned counts could not see them go wrong.
    """

    def __init__(self, project: _CannedProject) -> None:
        self._project = project

    def __call__(self, server_url: str) -> _CannedServer:
        return self

    async def __aenter__(self) -> _CannedProject:
        return self._project

    async def __aexit__(self, *exc: Any) -> None:
        return None


def _run_main(monkeypatch: Any, client: _CannedProject, *, dry_run: bool = False) -> int:
    """Drive ``main_async`` end to end against *client*, one project root."""
    monkeypatch.setattr(migrate_mod, 'FusedMemoryClient', _CannedServer(client))
    return asyncio.run(main_async(argparse.Namespace(
        server_url='http://127.0.0.1:9', dry_run=dry_run, project_roots=['/p'],
    )))


def _one_of_each() -> _CannedProject:
    """A canned project holding exactly one task per outcome, plus one no-op.

    Shared by the accounting tests so the expected counter tuple is read against
    a single, legible population rather than a per-test ad-hoc one.
    """
    return _CannedProject([
        _task('copy-file', modules=['scripts/a.py']),
        _task('copy-dirs', modules=['orchestrator/tests/']),
        _task('drop', modules=['crates/reify-core'], files=['src/b.py']),
        _task('untouched', priority='low'),
    ])


def test_a_sanitize_to_nothing_is_counted_apart_from_a_real_copy():
    """THE HONEST-ACCOUNTING FIX: three outcomes, not two.

    Before this, both copy-branch outcomes fell through the same
    ``if action == 'copy'`` tail, so an all-directory task that copied NOTHING
    was reported as a copy. On the live corpora that is 11 of 11 — the PR's
    before/after table would have claimed "copied: 11" describing zero copied
    files, which is precisely the number a reader would use to decide the
    migration had given those tasks their scope back.

    ``dropped`` and ``visited`` are asserted in the same tuple so a fix that
    invented ``sanitized_empty`` by cannibalising one of them goes red.
    """
    counts = _run(_one_of_each())

    assert (counts.visited, counts.copied, counts.sanitized_empty, counts.dropped) == (
        4, 1, 1, 1,
    )


def test_the_outcome_is_decided_by_the_sanitized_result_not_by_files_alone():
    """A mixed list that KEEPS something is a real copy; one that keeps nothing is not.

    The distinguishing input is identical on the ``files`` axis — both tasks
    have no ``files`` — so an implementation that still branches on ``files``
    alone cannot separate them. Only the post-charter result can, which is what
    this pins.
    """
    counts = _run(_CannedProject([
        _task('survives', modules=['orchestrator/tests/', 'hooks/pre-commit']),
        _task('vanishes', modules=['orchestrator/tests/', 'crates/reify-ir']),
    ]))

    assert (counts.copied, counts.sanitized_empty) == (1, 1)


def test_the_per_project_line_and_the_summary_both_name_sanitized_empty(monkeypatch, capsys):
    """The count has to reach the OPERATOR, not just the return value.

    The PR's before/after table is transcribed from this stdout, and the run
    evidence committed alongside it quotes the summary verbatim. A
    ``sanitized_empty`` that exists only on the result object would leave that
    table reporting two numbers for three outcomes.

    Both surfaces are asserted because they are separate call sites in
    ``main_async`` — the per-root line and the aggregate — and a fix that
    updated one and not the other is exactly the plausible half-done change.
    """
    _run_main(monkeypatch, _one_of_each())

    out = capsys.readouterr().out
    per_project, summary = out.split('---- summary ----')
    assert 'sanitized_empty' in per_project
    assert 'sanitized_empty: 1' in summary
    assert 'copied' in summary and 'dropped' in summary


def _terminal(task_id: str, status: str, **meta: Any) -> dict:
    """A task in one of the migration's deliberate skip statuses."""
    return {'id': task_id, 'status': status, 'metadata': dict(meta)}


def test_residual_carriers_in_skip_statuses_are_tallied_by_status():
    """The skipped tasks are LEFT ALONE but COUNTED, per status.

    PRD decision 1 keeps ``modules`` on terminal tasks deliberately — it is the
    only in-record trace of a finished task's original scope, and
    ``update_task`` will not write them anyway. But "we deliberately left some
    behind" is only a defensible claim if the number is stated, which is what
    the task requires in the PR and what the user-observable signal is judged
    on: remaining carriers must be provably confined to these statuses.

    Two things pinned at once, and both matter: the tally is BY STATUS (a
    single scalar could not distinguish 400 done carriers from 400 cancelled
    ones), and no ``update_task`` is recorded for any of them.
    """
    client = _CannedProject([
        _terminal('d1', 'done', modules=['a/b.py']),
        _terminal('d2', 'done', modules=['c/']),
        _terminal('c1', 'cancelled', modules=['e/f.py']),
        _terminal('f1', 'deferred', modules=['g/']),
    ])

    counts = _run(client)

    assert counts.residual_by_status == {'done': 2, 'cancelled': 1, 'deferred': 1}
    assert client.updates == []


def test_the_residual_tally_counts_carriers_not_every_skipped_task():
    """A skipped task with no ``modules`` is not a residual carrier.

    The number in the PR answers "how many records still carry the retired
    key", so counting every terminal task would inflate it by the entire
    history of the project — on dark-factory that is the difference between a
    handful and several hundred. A status with no carriers at all must be
    ABSENT rather than present-and-zero, so the printed report names only
    statuses that actually hold something.
    """
    client = _CannedProject([
        _terminal('d1', 'done', modules=['a/b.py']),
        _terminal('d2', 'done', priority='high'),
        _terminal('d3', 'done'),
        _terminal('c1', 'cancelled', files=['x/y.py']),
    ])

    counts = _run(client)

    assert counts.residual_by_status == {'done': 1}


def test_merge_deferred_is_processed_and_never_skipped():
    """``merge-deferred`` is NOT a skip status. Pin the exact-match semantics.

    ``deferred`` IS skipped and ``merge-deferred`` contains it as a substring,
    so a refactor of ``in skip_statuses`` to a ``startswith``/substring test —
    or to a "does the status mention deferred" heuristic — would silently stop
    migrating a whole live status class, with no test failing and no operator
    output changing except a number nobody has a baseline for. PRD open
    question 1 states merge-deferred IS processed.

    Asserted on the WRITE, not merely on ``visited``: being counted as visited
    is not the same as being migrated.
    """
    client = _CannedProject([_task('m1', modules=['scripts/a.py'])])
    client._tasks[0]['status'] = 'merge-deferred'

    counts = _run(client)

    assert _written(client, 'm1')['files'] == ['scripts/a.py']
    assert counts.visited == 1
    assert counts.residual_by_status == {}


def test_the_residual_counts_reach_the_per_project_line_and_the_summary(monkeypatch, capsys):
    """The by-status numbers have to be READ OFF the run, not reconstructed later.

    The run evidence committed with this migration records residual carriers
    per project and in aggregate; both come from this stdout. Ordering is
    pinned as sorted-by-status so two runs of the same corpus produce
    byte-comparable reports — an unordered dict repr would make the evidence
    file churn for no reason and defeat a diff.
    """
    _run_main(monkeypatch, _CannedProject([
        _terminal('d1', 'done', modules=['a/b.py']),
        _terminal('d2', 'done', modules=['c/']),
        _terminal('c1', 'cancelled', modules=['e/f.py']),
        _task('p1', modules=['scripts/a.py']),
    ]))

    out = capsys.readouterr().out
    per_project, summary = out.split('---- summary ----')
    assert 'cancelled=1' in per_project and 'done=2' in per_project
    assert per_project.index('cancelled=1') < per_project.index('done=2')
    assert 'cancelled=1' in summary and 'done=2' in summary


# The reply shapes THIS SCRIPT'S OWN TRANSPORT can actually deliver for a
# rejected write. Not hypothetical: `call_tool` raises only on a JSON-RPC-level
# `error`, while a fused-memory tool rejection comes back as an ORDINARY reply
# (`@mcp_tool_errors` converts every exception into one), and `_post` returns a
# bare `{}` for a 202 or an empty body. scripts/repair_wiped_metadata_files.py:
# 405-450 cites this module's `_post` (:89-90) and `call_tool` (:126) by line
# number as the source of the reachable `{}`.
REJECTION_REPLIES = [
    # A lock_charter_error, the gate this whole migration collides with: an
    # `error` marker and no `success` key at all.
    {'error': 'metadata.files carries directory locks', 'task_id': '9'},
    {'success': False, 'error': 'status_via_update_task', 'error_type': 'GuardError'},
    {},
    'not a dict at all',
]


def _reply_ids() -> list[str]:
    return ['lock_charter_error', 'success_false', 'empty_dict', 'non_dict']


def test_a_rejected_write_is_counted_as_a_failure_not_as_a_migration():
    """THE SILENT-SUCCESS BUG. Today all four of these print as successes.

    A migration whose entire subject is a collision with a server-side write
    gate MUST be able to see that gate fire. If it cannot, the numbers in the
    PR and in the committed run evidence are unverifiable — a run in which
    every single write was rejected produces output identical to a clean one.

    The rejected task is asserted absent from EVERY success tally, not merely
    present in ``failed``: double-counting it would keep the copied/dropped
    figures wrong in exactly the direction that hides the problem.
    """
    for reply, name in zip(REJECTION_REPLIES, _reply_ids(), strict=True):
        client = _CannedProject(
            [_task('r1', modules=['scripts/a.py'])],
            update_reply=lambda args, _r=reply: _r,
        )

        counts = _run(client)

        assert counts.failed == 1, name
        assert (counts.copied, counts.sanitized_empty, counts.dropped) == (0, 0, 0), name


def test_a_rejected_write_is_named_on_stderr(capsys):
    """An operator has to be told WHICH task, and why, not just a total.

    ``error_type`` is included when present because it is the machine-readable
    half of the reply and is what an operator greps for.
    """
    client = _CannedProject(
        [_task('r1', modules=['scripts/a.py'])],
        update_reply=lambda args: {
            'success': False, 'error': 'nope', 'error_type': 'GuardError',
        },
    )

    _run(client)

    err = capsys.readouterr().err
    assert 'r1' in err
    assert 'GuardError' in err


def test_a_rejection_does_not_abort_the_remaining_tasks():
    """One rejected task must not cost the rest of the corpus its migration.

    A partial run is the realistic failure — a single task tripping a guard the
    others do not — and stopping there would leave the corpus half-migrated
    with no record of where it stopped.
    """
    client = _CannedProject(
        [
            _task('bad', modules=['scripts/a.py']),
            _task('good', modules=['scripts/b.py']),
        ],
        update_reply=lambda args: (
            {'error': 'rejected'} if args['id'] == 'bad' else {'success': True}
        ),
    )

    counts = _run(client)

    assert (counts.failed, counts.copied) == (1, 1)
    assert _written(client, 'good')['files'] == ['scripts/b.py']


def test_a_success_shaped_reply_with_no_success_key_still_counts():
    """THE POSITIVE CONTROL: the classifier must not reject every real success.

    ``update_task`` answers with the updated task record, which carries an
    ``id`` and no ``success`` key at all. A classifier demanding an explicit
    ``success: True`` would report a completely clean run as a total failure —
    the opposite over-correction, and just as unusable.
    """
    client = _CannedProject(
        [_task('ok', modules=['scripts/a.py'])],
        update_reply=lambda args: {'id': 'ok', 'status': 'pending'},
    )

    counts = _run(client)

    assert (counts.failed, counts.copied) == (0, 1)


def test_a_nonzero_failed_total_makes_the_process_exit_nonzero(monkeypatch, capsys):
    """A partly-rejected run must not be mistakable for a clean one at the shell.

    ``main_async``'s return value becomes the process exit status. A migration
    that prints failures and still exits 0 will be recorded as a green run by
    anything driving it non-interactively — including the operator who reads
    only the last line.
    """
    rejected = _CannedProject(
        [_task('r1', modules=['scripts/a.py'])],
        update_reply=lambda args: {'error': 'rejected'},
    )
    assert _run_main(monkeypatch, rejected) != 0
    assert 'failed: 1' in capsys.readouterr().out.split('---- summary ----')[1]

    clean = _CannedProject([_task('ok', modules=['scripts/a.py'])])
    assert _run_main(monkeypatch, clean) == 0


#: One canned task per outcome, with the ACTION LABEL and RESULTING FILES the
#: live path is contracted to produce. The parity and idempotence tests both
#: walk this, so the three outcomes cannot be covered unevenly.
OUTCOME_CASES = [
    ('copy-file', {'modules': ['scripts/a.py']}, 'copy', ['scripts/a.py']),
    ('copy-dirs', {'modules': ['orchestrator/tests/']}, 'copy-sanitized-empty', None),
    (
        'drop',
        {'modules': ['crates/reify-core'], 'files': ['src/b.py']},
        'drop',
        ['src/b.py'],
    ),
]


def test_the_dry_run_reports_what_the_live_path_would_actually_write(capsys):
    """PARITY. Dry-run output IS the observable signal — it must not lie.

    The pre-4528 dry-run line printed ``modules=... files=...``: the INPUT,
    never the result. On the live corpora every copy-branch task is
    all-directory, so an operator reading that output would have seen 11 lines
    saying ``action=copy modules=['crates/reify-core', ...]`` and concluded
    those directories were about to be written into ``files`` — when in fact
    the write would have been rejected outright, and post-fix writes nothing at
    all.

    The label and the resulting files are compared against WHAT THE LIVE PASS
    ACTUALLY DID, not against literals restated here, so the two paths cannot
    drift apart while both still matching this test.
    """
    for task_id, meta, expected_label, expected_files in OUTCOME_CASES:
        dry = _CannedProject([{'id': task_id, 'status': 'pending', 'metadata': dict(meta)}])
        live = _CannedProject([{'id': task_id, 'status': 'pending', 'metadata': dict(meta)}])

        _run(dry, dry_run=True)
        dry_out = capsys.readouterr().out
        _run(live)
        capsys.readouterr()

        # The dry-run made no writes at all.
        assert dry.updates == [], task_id
        # ...and reported the live path's label and result.
        assert f'action={expected_label}' in dry_out, (task_id, dry_out)
        written = _written(live, task_id).get('files')
        assert written == expected_files, task_id
        assert f'files={written!r}' in dry_out, (task_id, dry_out)


def test_the_dry_run_counts_what_the_live_run_counts():
    """The same parity on the COUNTERS, which is what the summary reports.

    A dry-run whose per-task lines were honest but whose totals still lumped
    the two copy outcomes together would put the wrong before/after table in
    the PR — and the before table can only ever come from a dry run.
    """
    tasks = [
        {'id': tid, 'status': 'pending', 'metadata': dict(meta)}
        for tid, meta, _label, _files in OUTCOME_CASES
    ]

    dry = _run(_CannedProject(list(tasks)), dry_run=True)
    live = _run(_CannedProject(list(tasks)))

    assert dry == live
    assert (dry.copied, dry.sanitized_empty, dry.dropped) == (1, 1, 1)


def test_a_second_pass_over_the_migrated_metadata_is_a_no_op():
    """IDEMPOTENCE, measured against what the FIRST pass really wrote.

    The script's docstring has claimed "idempotent — safe to re-run" since it
    was written, and the run this task performs is a three-phase
    dry-run/live/dry-run whose final phase must report zero pending actions
    everywhere. That assertion is only meaningful if a second pass over the
    first pass's OWN OUTPUT is a no-op, which is what this feeds back — rather
    than a hand-written "already migrated" shape that could differ from
    reality.

    The sanitize-to-empty case is the one worth the care: that task ends with
    neither ``modules`` nor ``files``, so a re-run must not decide the empty
    ``files`` means it has scope to restore.
    """
    first = _CannedProject([
        {'id': tid, 'status': 'pending', 'metadata': dict(meta)}
        for tid, meta, _label, _files in OUTCOME_CASES
    ])
    _run(first)

    migrated = [
        {'id': args['id'], 'status': 'pending', 'metadata': json.loads(args['metadata'])}
        for args in first.updates
    ]
    assert len(migrated) == len(OUTCOME_CASES)

    second_client = _CannedProject(migrated)
    second = _run(second_client)

    assert second_client.updates == []
    assert (second.copied, second.sanitized_empty, second.dropped, second.failed) == (
        0, 0, 0, 0,
    )
    assert second.visited == len(OUTCOME_CASES)
