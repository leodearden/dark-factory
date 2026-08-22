#!/usr/bin/env python3
"""Migrate live tasks: ``metadata.modules`` → ``metadata.files``.

Run this BEFORE landing the principled-files-rename code change. Idempotent —
safe to re-run, and the sanctioned way to run it is dry-run → live → dry-run,
with the final dry-run reporting zero pending actions.

Behaviour per task carrying ``metadata.modules``:

- If ``metadata.files`` is missing or empty: copy ``modules`` into ``files``
  SANITIZED THROUGH THE LOCK CHARTER (``shared.locking.strip_directory_locks``),
  then drop ``modules``. The sanitization is not optional politeness: the
  server rejects any ``metadata.files`` write carrying a directory-shaped entry
  (``_reject_directory_locks_in_update_metadata``), so a verbatim copy of a
  directory-shaped ``modules`` is a guaranteed write failure. Measured across
  the seven live corpora, ALL of the copy-branch tasks were all-directory.
- If the sanitized list is EMPTY — nothing in ``modules`` was file-level —
  ``files`` is deliberately left empty (an absent key stays absent, an existing
  ``[]`` stays ``[]``) and the outcome is reported as ``sanitized_empty``, not
  as a copy. Scope is deferred to the architect, which widens it at plan time.
  Reporting this as a copy would claim tasks got their scope back when nothing
  was written.
- If ``metadata.files`` is non-empty: keep ``files`` as authoritative, drop
  ``modules``.

``done``/``cancelled``/``deferred`` are skipped BY POLICY, not by necessity:
PRD decision 1 keeps ``modules`` on terminal tasks as the only in-record trace
of their original scope. (``update_task`` would in fact accept the write —
there is no status-based immutability gate in ``SqliteTaskBackend.update_task``
or the interceptor — so this really is a choice, and the pre-4528 comment
claiming terminal tasks were "immutable" was wrong about why.) Their residual
carriers are
COUNTED BY STATUS and reported, so "we deliberately left some behind" is a
checkable claim. The skip set is an EXACT MATCH: ``merge-deferred`` is a live,
processed status and is not a member.

A write the server REFUSES is reported as a failure and never as a migration.
Nothing server-side crosses the wire as an exception — a rejection arrives as
an ordinary reply — so replies are classified by
:func:`write_failure_reason`, failures are named on stderr, and a non-zero
failed total makes the process exit non-zero. A record whose ``modules`` or
``files`` is not a list, and one whose pre-existing ``files`` carries a
directory-shaped entry the gate would reject, are counted as failures too:
both are unmigratable, and both would otherwise be reported as ordinary
successes having written nothing or having been refused.

A project root whose ``get_tasks`` never produced a readable task list is
reported as ``read_failed`` and also exits non-zero — an unread root's zeros
are indistinguishable from a migrated root's. Transport-level failures on
either call are retried (:data:`CALL_RETRY_ATTEMPTS`); a server REJECTION is
never retried, because it never raises.

Talks to the running fused-memory MCP server over JSON-RPC. Discovers project
roots from ``DASHBOARD_KNOWN_PROJECT_ROOTS`` (comma-separated). Falls back to
the dark-factory root.

Usage:
    python scripts/migrate_metadata_modules_to_files.py [--dry-run] \\
        [--server-url http://127.0.0.1:8002]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, NamedTuple

import httpx

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would silently evaluate the lock
# charter using the main checkout's predicate. Same reasoning and same form as
# scripts/repair_wiped_metadata_files.py:80-86 (which imports this module's
# exact inverse, `directory_locks`, for the very same gate).
_SHARED_SRC = Path(__file__).resolve().parent.parent / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

# WHY `shared.locking` AND NOT `orchestrator.module_charter.
# sanitize_files_for_persist`: `orchestrator` is not importable from this
# script's runtime context (measured — it is not a dependency of the `shared`
# project this script runs under, and a scripts/-rooted sys.path[0] gives no
# namespace-package fallback), and that function's ENTIRE body is
# `return strip_directory_locks(files)`. So this IS the charter predicate, one
# indirection lower — and it is the same module the server-side gate,
# `_reject_directory_locks_in_update_metadata`, calls `directory_locks` from.
# `directory_locks` itself is imported for the drop branch's pre-write check:
# that branch re-sends a task's PRE-EXISTING `files` verbatim, which the copy
# branch's sanitizer never sees, so the gate's own predicate is what decides
# whether that payload could ever be accepted.
from shared.locking import directory_locks, strip_directory_locks  # noqa: E402

DEFAULT_SERVER = 'http://127.0.0.1:8002'
DEFAULT_ROOTS = ['/home/leo/src/dark-factory']

#: Reserved keys :meth:`FusedMemoryClient.call_tool` stamps onto a reply that
#: is not a readable success shape. BOTH are read by
#: :func:`write_failure_reason` and neither can collide with a real server
#: field: they are set by this client, on the way out of the transport.
MCP_IS_ERROR_KEY = '_mcp_is_error'
RAW_REPLY_KEY = '_raw'

#: Transport-level retry for the two tool calls this script makes. BOTH are
#: idempotent — the read is a read, and the write is a COMPLETE replace-mode
#: payload computed from the task's own metadata, so re-sending it can only
#: ever re-do the same thing. The live run of this migration lost four writes
#: to `httpx.ReadTimeout` under 437 back-to-back calls and needed a hand-driven
#: targeted retry pass afterwards; this absorbs that class without operator
#: intervention. ONLY `httpx.TransportError` is retried, which is what keeps
#: the retry safe: a server-side REJECTION never crosses the wire as an
#: exception (see :func:`write_failure_reason`), so nothing retried here can be
#: a guard firing, and a guard can never be retried into submission.
CALL_RETRY_ATTEMPTS = 3
#: Seconds, multiplied by the attempt number for a linear backoff. Module-level
#: rather than a literal so a test can set it to 0 instead of sleeping.
CALL_RETRY_BACKOFF_S = 1.0


def _flatten_tasks(tasks_payload: Any) -> list[dict]:
    out: list[dict] = []

    def _walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for t in items:
            if isinstance(t, dict):
                out.append(t)
                _walk(t.get('subtasks') or [])

    if isinstance(tasks_payload, dict):
        _walk(tasks_payload.get('tasks'))
    return out


class FusedMemoryClient:
    """Minimal HTTP/JSON-RPC client for the fused-memory MCP server."""

    #: clientInfo.name sent at handshake. LOAD-BEARING, not cosmetic:
    #: fused_memory/server/tools.py `_resolve_identity` derives a write's
    #: agent_id from ctx.session.client_params.clientInfo.name, so a subclass
    #: that does not set this files its writes under THIS migration's identity.
    #: Override it in a subclass — do NOT restate `_initialize` to change it.
    _client_name: str = 'migrate-metadata'

    def __init__(self, server_url: str):
        self._url = server_url.rstrip('/')
        self._client: httpx.AsyncClient | None = None
        self._session_id: str | None = None

    async def __aenter__(self) -> FusedMemoryClient:
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._session_id = uuid.uuid4().hex
        await self._initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _post(self, payload: dict) -> dict:
        assert self._client is not None
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'mcp-session-id': self._session_id or '',
        }
        resp = await self._client.post(f'{self._url}/mcp/', json=payload, headers=headers)
        resp.raise_for_status()
        # 202 Accepted (notifications) returns no body.
        if resp.status_code == 202 or not resp.content:
            return {}
        ctype = resp.headers.get('content-type', '')
        if 'text/event-stream' in ctype:
            for line in resp.text.splitlines():
                if line.startswith('data:'):
                    return json.loads(line[5:].strip())
            raise RuntimeError(f'no SSE data line in response: {resp.text[:200]}')
        return resp.json()

    async def _initialize(self) -> None:
        await self._post({
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'clientInfo': {'name': self._client_name, 'version': '1.0'},
                'capabilities': {},
            },
        })
        await self._post({
            'jsonrpc': '2.0',
            'method': 'notifications/initialized',
            'params': {},
        })

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self._post({
            'jsonrpc': '2.0',
            'id': uuid.uuid4().hex,
            'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        })
        if 'error' in result:
            raise RuntimeError(f'{name} failed: {result["error"]}')
        # MCP tools/call returns content list; structuredContent is the parsed dict
        content = result.get('result', {})
        reply = self._reply_payload(content)
        # THE ENVELOPE'S OWN FAILURE FLAG, kept rather than discarded. FastMCP
        # sets `isError` on the tools/call result for failures that never enter
        # an `@mcp_tool_errors`-decorated body at all — argument validation, the
        # `_install_safe_tool_wrapper` backstop — and those replies carry no
        # `error` key inside the payload for :func:`write_failure_reason` to
        # see. Dropping the flag here made the classifier STRUCTURALLY unable to
        # notice that whole class, so it is stamped onto the payload under a
        # reserved key instead.
        if content.get('isError') and isinstance(reply, dict):
            reply[MCP_IS_ERROR_KEY] = True
        return reply

    @staticmethod
    def _reply_payload(content: dict) -> dict:
        """Pull the tool's own payload out of one MCP ``tools/call`` result.

        Split out of :meth:`call_tool` only so the ``isError`` stamp above has
        a single value to apply itself to; the extraction rules are unchanged.

        The ``json.JSONDecodeError`` fallback is NOT a success shape. It means
        the server answered with text this script cannot read as a structured
        reply at all — the FastMCP-level shapes above stringify that way — so it
        is returned under the reserved :data:`RAW_REPLY_KEY`, which
        :func:`write_failure_reason` treats as a failure rather than as a reply
        with no error marker in it.
        """
        if 'structuredContent' in content:
            return content['structuredContent']
        # Fall back to first text content
        for entry in content.get('content', []) or []:
            if entry.get('type') == 'text':
                try:
                    return json.loads(entry['text'])
                except json.JSONDecodeError:
                    return {RAW_REPLY_KEY: entry['text']}
        return content


def _clip(value: Any, limit: int = 200) -> str:
    """``repr`` *value*, truncated, for an operator-facing failure message.

    An unreadable reply can be a whole HTML error page; a failure line that
    scrolls the real ones off the terminal is its own kind of silence.
    """
    raw = value if isinstance(value, str) else repr(value)
    return raw if len(raw) <= limit else raw[:limit] + '…'


async def call_tool_with_retry(
    client: FusedMemoryClient, name: str, arguments: dict,
) -> dict:
    """Call *name*, retrying only TRANSPORT-level failures. See :data:`CALL_RETRY_ATTEMPTS`.

    A rejection is not an exception here — the server answers, and
    :func:`write_failure_reason` classifies the answer — so the only thing this
    can retry is a connection that never delivered a verdict. That is what
    makes retrying a WRITE safe: a guard cannot be retried into submission,
    because a guard firing never raises.

    Retries are announced on stderr rather than absorbed silently. A run that
    needed nine attempts to place 437 writes is a materially different run from
    one that needed 437, and the operator reading the evidence should be able
    to tell.
    """
    for attempt in range(1, CALL_RETRY_ATTEMPTS + 1):
        try:
            return await client.call_tool(name, arguments)
        except httpx.TransportError as exc:
            if attempt >= CALL_RETRY_ATTEMPTS:
                raise
            print(
                f'  [retry] {name} attempt {attempt}/{CALL_RETRY_ATTEMPTS} failed '
                f'with {type(exc).__name__}: {exc} — retrying',
                file=sys.stderr,
            )
            await asyncio.sleep(CALL_RETRY_BACKOFF_S * attempt)
    raise AssertionError('unreachable: the loop above either returns or re-raises')


def write_failure_reason(reply: Any) -> str | None:
    """Return why *reply* is not a successful write, or ``None`` if it is one.

    NOTHING SERVER-SIDE EVER CROSSES THE WIRE AS AN EXCEPTION.
    ``@mcp_tool_errors`` (fused-memory/src/fused_memory/server/tool_errors.py)
    converts every exception into an ordinary reply dict, and
    :meth:`FusedMemoryClient.call_tool` raises only on a JSON-RPC-LEVEL
    ``error``. So a ``lock_charter_error`` from
    ``_reject_directory_locks_in_update_metadata`` — the very gate this
    migration exists to stop colliding with — arrives here as a perfectly
    ordinary return value, and without this predicate is printed as a success.

    DELIBERATELY MIRRORS, RATHER THAN IMPORTS, two existing implementations of
    the same contract:

    * ``fused_memory.middleware.task_interceptor.interceptor_write_succeeded``,
      the canonical one. Not importable here — ``fused_memory`` fails to import
      from this script's runtime context (measured:
      ``ModuleNotFoundError: graphiti_core``).
    * ``scripts/repair_wiped_metadata_files.py:classify_reply``. Not imported
      because that module imports :class:`FusedMemoryClient` FROM this one
      (:1066); importing it back would make this base module depend on its
      subclass's module.

    THE CHECK ORDER IS LOAD-BEARING: not-a-dict first (nothing else can call
    ``.get`` on it), then emptiness — a ``{}`` has neither an ``error`` nor a
    ``success`` key and would otherwise fall through to the success branch,
    which is exactly the wedged-server-ACKs-with-a-202 case ``_post`` (:89-90)
    can produce — and only then the error/success probes.

    THE TWO TRANSPORT-STAMPED KEYS ARE CHECKED LAST, and that is deliberate
    too: a reply carrying BOTH a structured ``error`` and the envelope's
    ``isError`` flag gets the structured message, which names the
    ``error_type`` an operator greps for, rather than a generic
    envelope-said-no. They close the last silent-success hole in this
    predicate. ``{'_raw': 'Error calling tool update_task: ...'}`` is a
    NON-EMPTY dict with no ``error`` key and no ``success`` key, so without the
    :data:`RAW_REPLY_KEY` probe it reached the success branch — the very class
    of bug this function was added to close, surviving inside it. Both shapes
    are reachable for FastMCP-level failures that never enter an
    ``@mcp_tool_errors``-decorated body (argument validation, the
    ``_install_safe_tool_wrapper`` backstop).

    ``success`` defaults to True, not to False: ``update_task`` answers with
    the updated task record, which carries no ``success`` key at all. The
    falsy test rather than ``is False`` matches
    ``interceptor_write_succeeded``'s own ``bool(resp.get('success', True))``,
    so ``None``/``0``/``''`` are failures under this copy too.
    """
    if not isinstance(reply, dict):
        return f'server reply was not a dict: {type(reply).__name__}: {_clip(reply)}'
    if not reply:
        return 'server reply was empty ({}) — it carries no positive write signal'
    if reply.get('error'):
        error_type = reply.get('error_type')
        named = f'{error_type}: ' if error_type else ''
        return f'server returned an error: {named}{reply["error"]}'
    if not reply.get('success', True):
        detail = reply.get('error_type') or reply.get('message') or reply
        return f'server reported failure: {detail}'
    if reply.get(MCP_IS_ERROR_KEY):
        return f'server set the MCP isError flag on the reply: {_clip(reply)}'
    if RAW_REPLY_KEY in reply:
        return (
            'server reply was not structured JSON: '
            f'{_clip(reply[RAW_REPLY_KEY])}'
        )
    return None


class MigrationCounts(NamedTuple):
    """One project's migration outcome, split by what actually happened.

    A NamedTuple rather than the bare ``tuple[int, int, int]`` this replaced,
    matching the house style of the sibling script's ``ReplyVerdict`` /
    ``AuditCoverage``: the caller aggregates four fields whose ORDER is not
    self-evident, and a positional mix-up between ``copied`` and
    ``sanitized_empty`` would be silent and would land in the PR's table.

    ``copied`` and ``sanitized_empty`` are BOTH copy-branch outcomes and are
    deliberately not summed. ``copied`` means file-level entries survived the
    lock charter and were written; ``sanitized_empty`` means nothing survived,
    so ``files`` was left empty and the task's scope is deferred to the
    architect. Reporting the second as a copy would state that N tasks got
    their scope back when zero files were written — on the live corpora, all
    11 of them.
    """

    visited: int
    copied: int
    sanitized_empty: int
    dropped: int
    #: Tasks this run could NOT migrate: the server refused the write, the
    #: write never reached a verdict, or the record's own ``modules``/``files``
    #: is malformed and unmigratable. Counted apart from every success tally,
    #: never folded into one: a run whose writes were all rejected must not be
    #: able to report the same numbers as a clean one.
    failed: int
    #: Project roots whose ``get_tasks`` never produced a readable task list —
    #: 0 or 1 per root, summed across them. A ROOT-level tally rather than a
    #: task-level one, and deliberately not folded into ``failed``: an
    #: unreadable root yields ``visited=0 copied=0 dropped=0``, which is
    #: BYTE-IDENTICAL to a root that is genuinely fully migrated. This task's
    #: headline claim ("six of seven roots are at zero pending") is read off
    #: exactly that line, so "never read" has to be visible as its own thing.
    #: Precedent: the sibling script's ``FAILED_LIVE_READ`` /
    #: ``EXIT_LIVE_READ_FAILED`` (scripts/repair_wiped_metadata_files.py).
    read_failed: int
    #: ``{status: carriers}`` for tasks in a deliberate skip status that STILL
    #: carry ``metadata.modules``. Left untouched on purpose (PRD decision 1
    #: keeps ``modules`` on terminal tasks as the only in-record trace of their
    #: original scope) — but the number is what makes "we deliberately left
    #: some behind" a checkable claim rather than an assertion. Only statuses
    #: with at least one carrier appear.
    residual_by_status: dict[str, int]

    def merged_with(self, other: MigrationCounts) -> MigrationCounts:
        """Field-wise sum. Used to aggregate across project roots.

        Written out by NAME rather than as a positional ``zip`` so that adding
        a field whose merge is not integer addition cannot be silently folded
        into a wrong one.
        """
        residual = dict(self.residual_by_status)
        for status, count in other.residual_by_status.items():
            residual[status] = residual.get(status, 0) + count
        return MigrationCounts(
            visited=self.visited + other.visited,
            copied=self.copied + other.copied,
            sanitized_empty=self.sanitized_empty + other.sanitized_empty,
            dropped=self.dropped + other.dropped,
            failed=self.failed + other.failed,
            read_failed=self.read_failed + other.read_failed,
            residual_by_status=residual,
        )


def empty_counts() -> MigrationCounts:
    """A zeroed :class:`MigrationCounts`.

    A FACTORY, not a module-level constant: ``residual_by_status`` is a mutable
    dict, and one shared instance handed to every early return and to
    ``main_async``'s accumulator seed would let an unrelated caller's tally
    leak across project boundaries. ``merged_with`` already copies rather than
    mutates, but the constant would be one refactor away from being aliased
    into something that does not.
    """
    return MigrationCounts(
        visited=0, copied=0, sanitized_empty=0, dropped=0, failed=0,
        read_failed=0, residual_by_status={},
    )


def format_residuals(residual_by_status: dict[str, int]) -> str:
    """Render a residual tally for the operator, sorted by status.

    SORTED so two runs of the same corpus produce byte-comparable reports —
    the committed run evidence is diffed across the before/after passes, and an
    insertion-ordered dict repr would churn for no reason.
    """
    if not residual_by_status:
        return 'none'
    return ' '.join(f'{s}={n}' for s, n in sorted(residual_by_status.items()))

#: Printed action label per outcome. The two copy-branch outcomes get DISTINCT
#: labels so a per-task line is self-explaining without cross-referencing the
#: summary — `copy` alone cannot tell an operator whether anything was written.
ACTION_LABELS = {
    'copied': 'copy',
    'sanitized_empty': 'copy-sanitized-empty',
    'dropped': 'drop',
}


def malformed_shape_reason(modules: Any, raw_files: Any) -> str | None:
    """Return why a task's ``modules``/``files`` cannot be migrated, or ``None``.

    ``metadata`` IS FREE-FORM JSON AND MALFORMED SCALARS DO EXIST IN THESE
    CORPORA — this migration's own run hit reify 5050, whose
    ``metadata.milestone`` is the bool ``true`` where a mapping is required,
    and it is luck rather than design that the malformed key there was not this
    one.

    A bare-string ``modules`` is the dangerous shape, because every step
    downstream of it degrades QUIETLY instead of raising:
    ``strip_directory_locks`` iterates its argument, so ``'scripts/a.py'``
    explodes into single characters, none of which is a file path; the
    sanitised list comes back empty; the task is reported as the perfectly
    innocuous ``sanitized_empty``; and the replace-mode write drops the
    ``modules`` key outright. The scope is destroyed, the report is clean and
    ``failed`` is zero. A dict behaves identically (it iterates as keys).

    The house convention for this coercion is
    ``scripts/audit_wiped_metadata_files.py:_coerce_file_list``, which degrades
    a non-list to empty rather than raising. That is right for the AUDIT, which
    only reports; this script WRITES, so the same degradation would be the
    silent-destruction path above. Hence: name the record, refuse to write it.
    """
    if not isinstance(modules, list):
        return f'malformed modules ({type(modules).__name__}, expected a list)'
    if raw_files is not None and not isinstance(raw_files, list):
        return f'malformed files ({type(raw_files).__name__}, expected a list)'
    return None


async def _migrate_one_project(
    client: FusedMemoryClient, project_root: str, *, dry_run: bool,
) -> MigrationCounts:
    """Walk one project's tasks and migrate metadata. See :class:`MigrationCounts`."""
    # A ROOT THAT WAS NEVER READ MUST NOT LOOK LIKE A CLEAN ONE. Both failure
    # shapes are covered because both produce the same zeroed line otherwise:
    # the read RAISING (transport dead), and the read ANSWERING with something
    # that carries no task list — the same classifier the write path uses, for
    # the same reason (`{}` from a 202, a `_raw` blob, an `isError` envelope).
    # Without this, `_flatten_tasks` quietly returns [] and the root reports
    # `visited=0 ... failed=0`, byte-identical to a fully-migrated root — and
    # this task's headline claim is read off exactly that line.
    try:
        tasks_result = await call_tool_with_retry(
            client, 'get_tasks', {'project_root': project_root, 'with_subtasks': True},
        )
    except Exception as exc:
        print(
            f'  [error] get_tasks failed for {project_root}: '
            f'{type(exc).__name__}: {exc}',
            file=sys.stderr,
        )
        return empty_counts()._replace(read_failed=1)
    unreadable = write_failure_reason(tasks_result)
    if unreadable is not None:
        print(
            f'  [error] get_tasks returned no readable task list for '
            f'{project_root}: {unreadable}',
            file=sys.stderr,
        )
        return empty_counts()._replace(read_failed=1)

    tasks = _flatten_tasks(tasks_result)
    # Terminal tasks are skipped BY POLICY (PRD decision 1: `modules` is their
    # only in-record trace of original scope), NOT because they are immutable.
    # The pre-4528 comment here claimed update_task could not write them; that
    # is false — there is no status-based immutability gate in
    # SqliteTaskBackend.update_task or the interceptor. Keeping the wrong
    # reason on a correct behaviour is how the next reader "fixes" the skip.
    # EXACT MATCH, never a prefix or substring test: `merge-deferred` contains
    # `deferred` and is deliberately NOT a member — it is a live, processed
    # status, and a `startswith`/`in`-the-string refactor would silently stop
    # migrating the whole class.
    skip_statuses = {'done', 'cancelled', 'deferred'}
    visited = 0
    outcomes = {'copied': 0, 'sanitized_empty': 0, 'dropped': 0}
    failed = 0
    residual_by_status: dict[str, int] = {}

    for task in tasks:
        status = str(task.get('status', ''))
        meta = task.get('metadata')
        if status in skip_statuses:
            # Count the carrier, then leave the task ENTIRELY alone. Counting
            # only actual carriers, not every skipped task: the number answers
            # "how many records still hold the retired key", and tallying the
            # rest would inflate it by the project's whole finished history.
            if isinstance(meta, dict) and meta.get('modules'):
                residual_by_status[status] = residual_by_status.get(status, 0) + 1
            continue
        visited += 1
        if not isinstance(meta, dict):
            continue
        modules = meta.get('modules')
        if not modules:
            continue
        task_id = str(task.get('id', ''))
        raw_files = meta.get('files')

        # SHAPE GUARD, above every branch: an unmigratable record is NAMED, not
        # silently emptied. See :func:`malformed_shape_reason` for why a
        # non-list degrades quietly all the way to a clean-looking report.
        # Checked above the dry-run fork so a dry-run surfaces it too — a
        # "zero pending actions" reading must not be blocked by a record the
        # operator was never told about.
        malformed = malformed_shape_reason(modules, raw_files)
        if malformed is not None:
            print(
                f'  [error][{project_root}] task={task_id} {malformed} '
                f'— NOT migrated',
                file=sys.stderr,
            )
            failed += 1
            continue
        files = raw_files or []

        # Build a full new metadata dict (None-deletion via merge doesn't
        # work — a merge/append=True keeps existing keys). We copy the existing
        # metadata, mutate, and write the COMPLETE result under the explicit
        # metadata_mode='replace' co-signal — a bare append=False is now
        # rejected by the task-2180 metadata-wipe guard in
        # _resolve_metadata_mode.
        # Strip done_provenance — update_task rejects metadata writes that
        # carry it (set_task_status is the only sanctioned writer); we
        # already skip done/cancelled, but be defensive about orphan stamps.
        new_meta = {
            k: v for k, v in meta.items()
            if k not in ('modules', 'done_provenance')
        }
        if not files:
            # SANITIZE, never copy verbatim. The server rejects any
            # metadata.files carrying a directory-shaped entry
            # (_reject_directory_locks_in_update_metadata), so a verbatim copy
            # of a directory-shaped `modules` is a guaranteed write failure.
            # When nothing survives the charter, leave new_meta ALONE: an
            # absent `files` stays absent and an existing `[]` stays `[]`, and
            # the task's scope is deferred to the architect at plan time.
            sanitized = strip_directory_locks(modules)
            if sanitized:
                new_meta['files'] = sanitized
            # THE OUTCOME IS THE SANITIZED RESULT, not `files`. Both branches
            # here have an empty `files`; only the post-charter list separates
            # a copy that gave the task scope from one that gave it none.
            outcome = 'copied' if sanitized else 'sanitized_empty'
        else:
            outcome = 'dropped'
        action = ACTION_LABELS[outcome]

        # The RESULT, computed once above the fork so the dry-run reports what
        # the live path would actually write rather than what it was handed.
        # `.get`, not `[...]`: the sanitize-to-empty case deliberately leaves
        # no `files` key at all.
        resulting_files = new_meta.get('files')

        # WOULD THE SERVER GATE EVEN ACCEPT THIS PAYLOAD? The copy branch is
        # sanitised by construction, but the DROP branch re-sends the task's
        # PRE-EXISTING `files` verbatim, and `metadata_mode='replace'` means
        # `_reject_directory_locks_in_update_metadata` inspects the whole
        # payload's files rather than the delta. So a task whose own `files`
        # already carries a directory-shaped entry can NEVER converge: every
        # pass re-sends it and is rejected, forever. Checked here with the
        # gate's own predicate, above the dry-run fork, so the dry-run shows it
        # too — otherwise the copy branch would be sanitised against a gate the
        # drop branch can still trip, and a "zero pending" claim could be
        # silently blocked by a task nobody named.
        blocked = directory_locks(resulting_files or [])
        if blocked:
            print(
                f'  [error][{project_root}] task={task_id} action={action} '
                f'metadata.files carries directory-shaped entries {blocked!r} '
                f'that the server gate would reject — NOT migrated',
                file=sys.stderr,
            )
            failed += 1
            continue

        if dry_run:
            print(
                f'  [dry-run][{project_root}] task={task_id} action={action} '
                f'modules={modules!r} files={resulting_files!r}'
            )
        else:
            try:
                reply = await call_tool_with_retry(client, 'update_task', {
                    'id': task_id,
                    'project_root': project_root,
                    'metadata': json.dumps(new_meta),
                    'metadata_mode': 'replace',
                })
            except Exception as exc:
                # THE EXCEPTION CLASS, not just its message. `httpx.ReadTimeout`
                # stringifies to '', so the pre-amendment form printed a bare
                # `update_task raised:` — which is what the live run of this
                # migration actually emitted for its four transient failures,
                # forcing a hand re-measurement to tell "retry it" from
                # "escalate it" apart.
                print(
                    f'  [error][{project_root}] task={task_id} update_task '
                    f'raised: {type(exc).__name__}: {exc}',
                    file=sys.stderr,
                )
                failed += 1
                continue
            reason = write_failure_reason(reply)
            if reason is not None:
                print(
                    f'  [error][{project_root}] task={task_id} action={action} '
                    f'update_task rejected: {reason}',
                    file=sys.stderr,
                )
                failed += 1
                continue
            print(
                f'  [{project_root}] task={task_id} action={action}',
            )

        outcomes[outcome] += 1

    return MigrationCounts(
        visited=visited, failed=failed, read_failed=0,
        residual_by_status=residual_by_status, **outcomes,
    )


async def main_async(args: argparse.Namespace) -> int:
    roots_env = os.environ.get('DASHBOARD_KNOWN_PROJECT_ROOTS', '').strip()
    if args.project_roots:
        roots = list(args.project_roots)
    elif roots_env:
        roots = [p.strip() for p in roots_env.split(',') if p.strip()]
    else:
        roots = DEFAULT_ROOTS

    print(f'Server:   {args.server_url}')
    print(f'Roots:    {roots}')
    print(f'Dry-run:  {args.dry_run}')
    print()

    totals = empty_counts()
    async with FusedMemoryClient(args.server_url) as client:
        for root in roots:
            print(f'Project: {root}')
            counts = await _migrate_one_project(client, root, dry_run=args.dry_run)
            totals = totals.merged_with(counts)
            # `copied_modules→files` and `dropped_modules_only` keep their
            # pre-4528 spellings so an operator diffing against an older run's
            # output can still line the columns up; `sanitized_empty` is the
            # only new one.
            print(
                f'  visited={counts.visited} '
                f'copied_modules→files={counts.copied} '
                f'sanitized_empty={counts.sanitized_empty} '
                f'dropped_modules_only={counts.dropped} '
                f'failed={counts.failed} '
                f'read_failed={counts.read_failed}'
            )
            print(
                f'  residual_carriers_in_skip_statuses: '
                f'{format_residuals(counts.residual_by_status)}'
            )
            print()

    print('---- summary ----')
    for k, v in totals._asdict().items():
        if k == 'residual_by_status':
            print(f'  residual_carriers_in_skip_statuses: {format_residuals(v)}')
        else:
            print(f'  {k}: {v}')
    # A partly-rejected run must not be mistakable for a clean one at the
    # shell — this return becomes the process exit status. `read_failed` is in
    # the condition for the same reason it is a field at all: a root that was
    # never read reports the same zeros as a fully-migrated one, so exiting 0
    # on it would let "every root is at zero pending" be recorded from a run
    # that read nothing.
    problems = []
    if totals.failed:
        problems.append(
            f'{totals.failed} task(s) were REJECTED by the server or are '
            f'unmigratable'
        )
    if totals.read_failed:
        problems.append(
            f'{totals.read_failed} project root(s) could not be READ — their '
            f'zeros mean "unknown", not "clean"'
        )
    if problems:
        print(
            f'  [FAILED] {"; ".join(problems)}; see the [error] lines above.',
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Report changes without applying.')
    parser.add_argument(
        '--server-url', default=DEFAULT_SERVER,
        help=f'Fused-memory MCP server URL (default: {DEFAULT_SERVER})',
    )
    parser.add_argument(
        '--project-root', dest='project_roots', action='append',
        help='Project root to migrate. May be repeated. Falls back to '
        '$DASHBOARD_KNOWN_PROJECT_ROOTS or the dark-factory root.',
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == '__main__':
    main()
