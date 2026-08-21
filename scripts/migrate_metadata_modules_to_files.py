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
failed total makes the process exit non-zero.

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
from shared.locking import strip_directory_locks  # noqa: E402

DEFAULT_SERVER = 'http://127.0.0.1:8002'
DEFAULT_ROOTS = ['/home/leo/src/dark-factory']


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
        if 'structuredContent' in content:
            return content['structuredContent']
        # Fall back to first text content
        for entry in content.get('content', []) or []:
            if entry.get('type') == 'text':
                try:
                    return json.loads(entry['text'])
                except json.JSONDecodeError:
                    return {'_raw': entry['text']}
        return content


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

    ``success`` defaults to True, not to False: ``update_task`` answers with
    the updated task record, which carries no ``success`` key at all. The
    falsy test rather than ``is False`` matches
    ``interceptor_write_succeeded``'s own ``bool(resp.get('success', True))``,
    so ``None``/``0``/``''`` are failures under this copy too.
    """
    if not isinstance(reply, dict):
        raw = repr(reply)
        if len(raw) > 200:
            raw = raw[:200] + '…'
        return f'server reply was not a dict: {type(reply).__name__}: {raw}'
    if not reply:
        return 'server reply was empty ({}) — it carries no positive write signal'
    if reply.get('error'):
        error_type = reply.get('error_type')
        named = f'{error_type}: ' if error_type else ''
        return f'server returned an error: {named}{reply["error"]}'
    if not reply.get('success', True):
        detail = reply.get('error_type') or reply.get('message') or reply
        return f'server reported failure: {detail}'
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
    #: Tasks whose ``update_task`` the server REFUSED. Counted apart from every
    #: success tally, never folded into one: a run whose writes were all
    #: rejected must not be able to report the same numbers as a clean one.
    failed: int
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
        residual_by_status={},
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


async def _migrate_one_project(
    client: FusedMemoryClient, project_root: str, *, dry_run: bool,
) -> MigrationCounts:
    """Walk one project's tasks and migrate metadata. See :class:`MigrationCounts`."""
    try:
        tasks_result = await client.call_tool(
            'get_tasks', {'project_root': project_root, 'with_subtasks': True},
        )
    except Exception as exc:
        print(f'  [skip] get_tasks failed for {project_root}: {exc}', file=sys.stderr)
        return empty_counts()

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
        files = meta.get('files') or []
        task_id = str(task.get('id', ''))

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
            sanitized = strip_directory_locks(list(modules))
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

        if dry_run:
            print(
                f'  [dry-run][{project_root}] task={task_id} action={action} '
                f'modules={modules!r} files={resulting_files!r}'
            )
        else:
            try:
                reply = await client.call_tool('update_task', {
                    'id': task_id,
                    'project_root': project_root,
                    'metadata': json.dumps(new_meta),
                    'metadata_mode': 'replace',
                })
            except Exception as exc:
                print(
                    f'  [error][{project_root}] task={task_id} update_task '
                    f'raised: {exc}',
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
        visited=visited, failed=failed,
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
                f'failed={counts.failed}'
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
    # shell — this return becomes the process exit status.
    if totals.failed:
        print(
            f'  [FAILED] {totals.failed} task(s) were REJECTED by the server; '
            f'see the [error] lines above.',
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
