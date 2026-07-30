#!/usr/bin/env python3
"""Consume reify's closure-staleness re-dispatch requests (task 3102).

This is the dark-factory half of reify task 5321's cross-repo seam. reify's
``scripts/deterministic-gate-closure-staleness-sweep.sh`` is READ-ONLY on all
task state by design (its invariant L6): it adjudicates stranded rows and
EMITS one request file per confirmed hit under ``--emit-requests``, and
dark-factory performs the actual ``set_task_status`` / ``set_task_claimant``
write through the fused-memory MCP server, so the write goes through the
reconciliation-triggering path CLAUDE.md requires.

THE INVARIANT THIS WHOLE MODULE RESTS ON
----------------------------------------
This consumer acts on the sweep's emitted FILES and NEVER re-derives the
sweep's predicates. None of the L1 heartbeat/claimant liveness guard, the
escalation terminal-allowlist oracle, merge-verify ancestry, the dependency
roll-up, or the #5316 corruption signatures is reproduced here. Two things
follow, and both are load-bearing:

  * There is exactly one implementation of the adjudication, in reify, where
    the evidence lives. A second one here could disagree with it silently.
  * ``CORRUPT-HOLD`` rows are unreachable to this consumer *by construction*.
    The sweep emits only on ``verdict=STALE`` (its invariant L5), so a
    corrupt-hold row produces no file at all; declining to read the sweep's
    stdout report is therefore sufficient to honour L5. Those rows need the
    human git-history adjudication in reify's
    ``docs/notes/offline-lane-red-corruption-remediation.md`` §4.

The NORMATIVE contract is the reify script itself -- its ``--emit-requests
consumer contract`` header block and ``_write_request`` -- not this docstring
and not reify's ``docs/notes/deterministic-gate-closure-staleness-sweep.md``
digest. Where they disagree, the script wins. Read it before changing any
parsing code here.

What the contract guarantees, and what this module therefore relies on:

  * One file per confirmed hit, named ``redispatch-<task_id>-<class>.json``,
    body ``{schema_version, task_id, class, verdict, action, evidence,
    main_ref_sha, emitted_by}`` rendered ``sort_keys=True, indent=2`` with a
    trailing newline.
  * The body carries NO wall-clock field, deliberately, so re-emission is
    byte-idempotent. **Recency is the file's mtime** -- which is what the
    compare-and-swap guard reads.
  * Emission is atomic (mktemp in the same dir, then ``mv``), so a partial
    file is never observable. The ``redispatch-tmp-XXXXXX`` intermediates are
    skipped by the filename predicate regardless.
  * The directory is a SNAPSHOT, not a log: each run RETRACTS the requests
    that are no longer confirmed hits before emitting. Retraction globs
    ``redispatch-*.json`` at the TOP LEVEL only and touches only names the
    sweep could itself have emitted -- which is exactly what makes archiving
    into a ``consumed/`` subdirectory retraction-safe.

Stdlib-only on purpose (mirrors ``scripts/reclaim_orphaned_worktrees.py``): no
uv, no ``.env``, no ``fused_memory`` import. A nightly timer that can fail on a
lockfile or venv-resolution problem is a job that silently stops running.
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

# Identify ourselves explicitly on `initialize`. The TaskInterceptor derives an
# actor class from the MCP client identity, so a nightly automated mutator of
# the task store should be namable in that record rather than showing up as
# whatever default the transport happened to send.
CLIENT_NAME = 'reify-redispatch-consumer'
CLIENT_VERSION = '1.0'
AGENT_ID = 'reify-redispatch-consumer'

DEFAULT_SERVER = 'http://127.0.0.1:8002/mcp'
DEFAULT_TIMEOUT = 30.0

# The FIXED class -> action mapping, from the reify sweep's three classifiers
# (_classify_gate_closure, _classify_merge_verify_red,
# _classify_unmet_dependency). A file whose class is absent from this table, or
# whose action disagrees with it, is not a request in a dialect we understand:
# it is skipped and reported, never reconciled or guessed at.
CLASS_ACTION = {
    'gate_closure': 'close',
    'merge_verify_red': 'reverify',
    'unmet_dependency': 'redispatch',
}

# The only verdict the sweep ever emits on. Everything else -- CORRUPT-HOLD,
# LIVE, GATED, UNRESOLVED, NO-CLASS, unknown -- is filtered upstream, so seeing
# one here means the contract was violated somewhere above us. Hard skip.
STALE = 'STALE'

SCHEMA_VERSION = 1

# redispatch-<digits>-<known class>.json, anchored. Deliberately the same shape
# as the sweep's own retraction predicate, so the two agree on what counts as
# a request file this side could have been handed.
_REQUEST_NAME_RE = re.compile(
    r'^redispatch-(\d+)-(' + '|'.join(sorted(CLASS_ACTION)) + r')\.json$'
)

_REQUIRED_FIELDS = ('schema_version', 'task_id', 'class', 'verdict', 'action')


def log(message: str) -> None:
    """Emit one greppable line to stderr.

    Everything this consumer decides -- every skip, every apply, every failure
    -- is REPORTED. A request that is silently dropped is indistinguishable
    from one that was never emitted, which is exactly the confusion the
    loud-over-silent-degradation invariant exists to prevent.
    """
    print(f'consume_redispatch_requests: {message}', file=sys.stderr, flush=True)


@dataclass
class Request:
    """One parsed request file.

    ``skip_reason is None`` means the file validated and may be acted on
    (subject to the separate confirm-before-write guard). Otherwise it holds a
    human-readable reason, which is REPORTED rather than swallowed -- a request
    we cannot understand must be visible in the journal, not silently dropped.
    """

    path: str
    mtime: float
    task_id: int | None = None
    cls: str | None = None
    action: str | None = None
    verdict: str | None = None
    evidence: str = ''
    skip_reason: str | None = None


def discover_requests(requests_dir) -> list[str]:
    """Return the paths of candidate request files at the TOP LEVEL of *dir*.

    Only ``redispatch-<digits>-<known class>.json`` regular files qualify. That
    excludes the sweep's ``redispatch-tmp-XXXXXX`` intermediates (which may be
    mid-write) and this consumer's own ``consumed/`` archive subdirectory, so
    an already-actioned request is never picked up a second time.

    A missing or unreadable directory yields no requests -- a clean zero-request
    scan is the correct reading of "the sweep found nothing to emit", and an
    absent dir on a first run is normal.
    """
    requests_dir = str(requests_dir)
    try:
        names = os.listdir(requests_dir)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    out = []
    for name in names:
        if not _REQUEST_NAME_RE.match(name):
            continue
        path = os.path.join(requests_dir, name)
        if os.path.isfile(path):
            out.append(path)
    return sorted(out)


def load_request(path) -> Request:
    """Parse and STRICTLY validate one request file.

    Every check here answers "is this a file the sweep emitted, in the schema
    this consumer was written against?" -- never "does the sweep's verdict look
    right?". Anything that fails is returned with a ``skip_reason`` for the
    caller to report; nothing is inferred, repaired, or defaulted.
    """
    path = str(path)
    try:
        mtime = os.path.getmtime(path)
    except OSError as exc:
        return Request(path=path, mtime=0.0, skip_reason=f'unreadable: {exc}')

    name_match = _REQUEST_NAME_RE.match(os.path.basename(path))
    if name_match is None:
        return Request(path=path, mtime=mtime,
                       skip_reason='filename is not redispatch-<id>-<class>.json')
    name_task_id = int(name_match.group(1))
    name_cls = name_match.group(2)

    try:
        with open(path) as fh:
            body = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return Request(path=path, mtime=mtime, skip_reason=f'unparseable JSON: {exc}')
    if not isinstance(body, dict):
        return Request(path=path, mtime=mtime,
                       skip_reason=f'body is {type(body).__name__}, expected object')

    missing = [f for f in _REQUIRED_FIELDS if f not in body]
    if missing:
        return Request(path=path, mtime=mtime,
                       skip_reason=f'missing required field(s): {", ".join(missing)}')

    # schema_version is checked FIRST among the value checks: it is the field
    # whose whole job is to tell us whether the rest of the body means what we
    # think it means. `is not SCHEMA_VERSION`-style identity is avoided, but
    # bool is excluded explicitly since `True == 1` in Python.
    version = body['schema_version']
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        return Request(path=path, mtime=mtime,
                       skip_reason=f'unsupported schema_version {version!r} '
                                   f'(this consumer understands {SCHEMA_VERSION})')

    task_id = body['task_id']
    if isinstance(task_id, bool) or not isinstance(task_id, int):
        return Request(path=path, mtime=mtime,
                       skip_reason=f'task_id {task_id!r} is not an integer')
    if task_id != name_task_id:
        # The name and the body state the same fact twice. A disagreement means
        # one of them is wrong and there is no basis for choosing between them.
        return Request(path=path, mtime=mtime,
                       skip_reason=f'task_id {task_id} disagrees with filename '
                                   f'({name_task_id})')

    cls = body['class']
    if cls not in CLASS_ACTION:
        return Request(path=path, mtime=mtime,
                       skip_reason=f'unknown class {cls!r}')
    if cls != name_cls:
        return Request(path=path, mtime=mtime,
                       skip_reason=f'class {cls!r} disagrees with filename '
                                   f'({name_cls!r})')

    verdict = body['verdict']
    if verdict != STALE:
        # The sweep emits on STALE only (invariant L5 keeps CORRUPT-HOLD out of
        # this directory entirely). A non-STALE file is an upstream contract
        # violation, so refuse it rather than infer what was intended.
        return Request(path=path, mtime=mtime,
                       skip_reason=f'verdict {verdict!r} is not {STALE}')

    action = body['action']
    expected = CLASS_ACTION[cls]
    if action != expected:
        return Request(path=path, mtime=mtime,
                       skip_reason=f'action {action!r} disagrees with the fixed '
                                   f'mapping for class {cls!r} (expected '
                                   f'{expected!r})')

    evidence = body.get('evidence')
    return Request(
        path=path,
        mtime=mtime,
        task_id=task_id,
        cls=cls,
        action=action,
        verdict=verdict,
        evidence=evidence if isinstance(evidence, str) else '',
    )


class McpClient:
    """Minimal streamable-HTTP JSON-RPC MCP client on ``urllib.request``.

    A stdlib port of ``fused-memory/scripts/cgl_eta_scheduler_gate.py``'s httpx
    ``McpClient`` -- see the module docstring for why this consumer does not
    take the httpx/uv dependency. Four behaviours are ported deliberately, each
    with the hard-won rationale that motivated it, and each covered by a test
    against a real socket in
    ``scripts/tests/test_consume_redispatch_requests.py``:

      1. `initialize` is sent SESSION-LESS.
      2. The server-assigned `mcp-session-id` is captured BEFORE any early
         return, then echoed on every later request.
      3. A `text/event-stream` response is read by extracting its first
         `data:` line.
      4. `result.structuredContent` is unwrapped, with a text-content JSON
         fallback, and a JSON-RPC `error` envelope RAISES.
    """

    def __init__(self, url: str = DEFAULT_SERVER, timeout: float = DEFAULT_TIMEOUT):
        self._url = url
        self._timeout = timeout
        self._session_id: str | None = None

    def __enter__(self) -> 'McpClient':
        # No session id on the FIRST request: the MCP streamable-HTTP contract
        # requires `initialize` to be sent session-less. A STATEFUL server 404s
        # "Session not found" if a client invents its own id here. The
        # server-assigned id is captured from the initialize response in
        # `_post` below and reused from then on.
        self._post({
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {'protocolVersion': '2024-11-05',
                       'clientInfo': {'name': CLIENT_NAME,
                                      'version': CLIENT_VERSION},
                       'capabilities': {}},
        })
        self._post({'jsonrpc': '2.0', 'method': 'notifications/initialized',
                    'params': {}})
        return self

    def __exit__(self, *exc) -> None:
        return None

    def _post(self, payload: dict) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
        }
        if self._session_id is not None:
            headers['mcp-session-id'] = self._session_id
        req = urllib.request.Request(
            self._url, data=json.dumps(payload).encode(), headers=headers,
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            status = resp.status
            # Capture the server-assigned session id (returned on `initialize`)
            # so it can be reused on every subsequent request. Must happen
            # BEFORE the 202/empty-content early return below, since
            # `initialize` responses carry a JSON body but
            # `notifications/initialized` may not -- and a server may answer
            # `initialize` itself with a bodiless 202.
            sid = resp.headers.get('mcp-session-id')
            if sid:
                self._session_id = sid
            content_type = resp.headers.get('content-type', '') or ''
            raw = resp.read()
        if status == 202 or not raw:
            return {}
        text = raw.decode()
        if 'text/event-stream' in content_type:
            for line in text.splitlines():
                if line.startswith('data:'):
                    return json.loads(line[5:].strip())
            raise RuntimeError(f'no SSE data line: {text[:200]}')
        return json.loads(text)

    def call_tool(self, name: str, arguments: dict) -> dict:
        result = self._post({
            'jsonrpc': '2.0', 'id': uuid.uuid4().hex, 'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        })
        # An {"error": ...} envelope carries no result. Raising rather than
        # returning it is what keeps a failed write from being counted as an
        # applied one further up.
        if 'error' in result:
            raise RuntimeError(f'{name} failed: {result["error"]}')
        content = result.get('result', {})
        if 'structuredContent' in content:
            return content['structuredContent']
        for entry in content.get('content', []) or []:
            if entry.get('type') == 'text':
                try:
                    return json.loads(entry['text'])
                except json.JSONDecodeError:
                    return {'_raw': entry['text']}
        return content

    # ── the three tools this consumer needs, and no others ──────────────────

    def get_task(self, task_id: int, project_root: str) -> dict:
        return self.call_tool('get_task', {
            'id': str(task_id), 'project_root': project_root,
        })

    def set_task_status(self, task_id: int, status: str, project_root: str) -> dict:
        return self.call_tool('set_task_status', {
            'id': str(task_id), 'status': status, 'project_root': project_root,
        })

    def set_task_claimant(self, task_id: int, project_root: str, *,
                          claimant_run_id=None, heartbeat_at=None) -> dict:
        # The wire parameter is `id`, not `task_id` (fused-memory tools.py
        # `set_task_claimant(id, project_root, tag, ...)`) -- matching the
        # orchestrator's own call at scheduler.py:2409. Both claimant kwargs
        # are tri-state server-side against a '__unset__' STRING sentinel, so
        # an explicit JSON null is the CLEAR value; sending them is exactly
        # what this consumer wants.
        return self.call_tool('set_task_claimant', {
            'id': str(task_id), 'project_root': project_root,
            'claimant_run_id': claimant_run_id, 'heartbeat_at': heartbeat_at,
        })


# ── the confirm-before-write guard ──────────────────────────────────────────
#
# THIS IS A STALE-READ GUARD ON THE ADJUDICATED ROW, NOT A RE-EVALUATION OF
# THE SWEEP'S PREDICATES. It asks exactly two questions -- "is this still the
# row the sweep looked at?" and "is it still in a state where this action means
# anything?" -- and never re-derives escalation state, dependency roll-ups,
# merge-verify ancestry, or liveness. Re-deriving any of those here would put a
# second, drifting implementation of the adjudication on this side of the seam.
#
# The fail-safe direction throughout is DO NOTHING: every uncertainty (an
# unreadable row, an un-comparable timestamp, a status we cannot place) skips.
# A skipped request stays on disk, and the next sweep either re-confirms it or
# retracts it -- so a false skip costs one night, while a false write costs a
# wrongly-cancelled or wrongly-re-dispatched task.

# The status each action drives the row TO. Used for the idempotent-no-op
# check: a row already there has had this request applied.
ACTION_TARGET_STATUS = {
    'close': 'cancelled',
    'reverify': 'pending',
    'redispatch': 'pending',
}

# The sweep's invariant L4, transcribed: classes A (gate_closure) and C
# (unmet_dependency) are `blocked`-ONLY; only class B (merge_verify_red) spans
# `in-progress`. A row outside its class's scope is one the sweep would not
# have adjudicated this way today, so acting on it would be acting on an
# observation that no longer applies.
LEGAL_STATUSES = {
    'gate_closure': frozenset({'blocked'}),
    'merge_verify_red': frozenset({'blocked', 'in-progress'}),
    'unmet_dependency': frozenset({'blocked'}),
}

# The two row timestamps that say "this row moved". `updatedAt` is written by
# sqlite_task_backend._now() as `...THH:MM:SS.mmmZ`; `heartbeat_at` is stamped
# by the orchestrator as `datetime.now(UTC).isoformat()` (`...+00:00`). Both
# spellings must parse or the guard fails open on one of them.
_FRESHNESS_FIELDS = ('updatedAt', 'heartbeat_at')


def _parse_timestamp(value):
    """Epoch seconds for an ISO-8601 timestamp, or None if it cannot be read.

    Tolerant of both the `Z` suffix and an explicit `+00:00` offset. A naive
    timestamp (no zone) is read as UTC, which is what every writer in this
    system actually means. None is returned rather than raising so the caller
    can make the fail-safe decision explicitly.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def guard_row(req: Request, row: dict) -> str | None:
    """Return a skip reason for *row*, or None if the write may proceed.

    Pure: no I/O, no clock. Everything it compares against is either in the
    request file or in the row the caller just read.
    """
    status = row.get('status')
    if not isinstance(status, str) or not status:
        return f'row carries no readable status ({status!r})'

    target = ACTION_TARGET_STATUS.get(req.action)
    if target is None:
        # Unreachable via load_request (which validates the action against
        # CLASS_ACTION), but a missing entry here must never mean "no target,
        # so nothing to compare, so proceed".
        return f'no target status known for action {req.action!r}'
    if status == target:
        # Idempotent no-op: this request was already applied. The sweep re-emits
        # byte-identical files, so seeing one twice is expected, not an error.
        return f'row is already {target!r} — request already applied'

    legal = LEGAL_STATUSES.get(req.cls)
    if legal is None:
        return f'no legal-status scope known for class {req.cls!r}'
    if status not in legal:
        return (f'row status {status!r} is outside the legal scope for class '
                f'{req.cls!r} ({"/".join(sorted(legal))}) — out of scope')

    # Freshness: the request body deliberately carries no wall-clock field, so
    # the file's MTIME is the documented recency signal. A row timestamp that
    # post-dates it means the row moved after the sweep observed it -- most
    # sharply when a `heartbeat_at` landed, i.e. a runner came alive after the
    # sweep's own L1 liveness check. Let the next sweep re-adjudicate.
    for field in _FRESHNESS_FIELDS:
        raw = row.get(field)
        if raw is None and field == 'heartbeat_at':
            # A `blocked` row normally carries no heartbeat at all. Absence is
            # not staleness -- only an unreadable PRESENT value is.
            continue
        when = _parse_timestamp(raw)
        if when is None:
            return (f'row {field} {raw!r} could not be read as a timestamp — '
                    f'cannot show the row is unchanged')
        if when > req.mtime:
            return (f'row {field} {raw!r} post-dates the request file — the row '
                    f'moved after the sweep observed it')
    return None


def confirm_request(client, req: Request, project_root: str):
    """Re-read the row and decide whether *req*'s write may proceed.

    Returns ``(row, skip_reason)``; ``skip_reason is None`` means proceed.
    Issues exactly one read and never a write -- whatever it decides, it
    decides before any mutation.
    """
    try:
        row = client.get_task(req.task_id, project_root)
    except Exception as exc:
        return None, f'get_task failed: {type(exc).__name__}: {exc}'
    if not isinstance(row, dict) or row.get('error'):
        detail = row.get('error') if isinstance(row, dict) else row
        return None, f'get_task returned no usable row: {detail!r}'
    return row, guard_row(req, row)


# ── applying a confirmed request ────────────────────────────────────────────


def rejection_reason(response) -> str | None:
    """Return a rejection description embedded in *response*, or None.

    fused-memory answers its gates (terminal-exit, phantom-done,
    done_provenance validation) with a 200 carrying a structured error rather
    than raising -- so the absence of an exception is NOT evidence that a write
    landed. Mirrors ``orchestrator.scheduler.extract_rejection``, minus the
    envelope-unwrapping it does, since :meth:`McpClient.call_tool` has already
    unwrapped ``result.structuredContent`` by the time we see the payload.

    A no-op (``success: True, no_op: True``, returned for a same-status write)
    is a SUCCESS, not a rejection.
    """
    if not isinstance(response, dict):
        return None
    if response.get('error'):
        hint = response.get('hint') or response.get('error_type') or ''
        return f'{response["error"]}{" — " + hint if hint else ""}'
    if response.get('success') is False:
        return f'success=False payload={response!r}'
    return None


def _checked(response, what: str) -> None:
    """Raise unless *response* shows the write actually landed."""
    reason = rejection_reason(response)
    if reason is not None:
        raise RuntimeError(f'{what} rejected: {reason}')


def _apply_close(client, req: Request, project_root: str) -> None:
    """class A / gate_closure: the gate is already satisfied, so retire the row."""
    _checked(
        client.set_task_status(req.task_id, 'cancelled', project_root),
        f'set_task_status({req.task_id}, cancelled)',
    )


def _apply_repend(client, req: Request, project_root: str) -> None:
    """classes B and C: return the row to the dispatchable pool.

    ORDERING IS LOAD-BEARING, and is carried verbatim from the orchestrator's
    own stranded-blocked re-dispatch path
    (orchestrator/src/orchestrator/scheduler.py:5726-5736): clear the stale
    claimant BEFORE the status flip. The row is still `blocked`/`in-progress`
    here -- not yet dispatch-contestable -- so nothing should be racing a fresh
    claimant stamp. Clearing first means a concurrent orchestrator that
    dispatches into this task right after it goes `pending` can never have its
    fresh stamp clobbered by a late-landing clear.

    Unlike the scheduler's best-effort call, a rejected clear ABORTS here: this
    consumer's whole warrant is that the row is quiescent, and flipping to
    `pending` while a stale claimant is still stamped would hand a dispatcher a
    row that reads as live.
    """
    _checked(
        client.set_task_claimant(req.task_id, project_root,
                                 claimant_run_id=None, heartbeat_at=None),
        f'set_task_claimant({req.task_id}, clear)',
    )
    _checked(
        client.set_task_status(req.task_id, 'pending', project_root),
        f'set_task_status({req.task_id}, pending)',
    )


# The dispatch table. `reverify` and `redispatch` differ in WHY the sweep
# emitted them (a merge-verify red that main has since moved past, vs a
# dependency roll-up that is now satisfied), not in what the write must do:
# both return the row to the dispatchable pool. Keeping them as separate keys
# rather than collapsing them preserves the distinction the request carries,
# so the journal line names the class that was actually adjudicated.
ACTION_APPLIERS = {
    'close': _apply_close,
    'reverify': _apply_repend,
    'redispatch': _apply_repend,
}


def apply_request(client, req: Request, project_root: str) -> bool:
    """Perform *req*'s writes. Returns True only if every write LANDED.

    Never raises: a per-request failure is contained and reported so its
    siblings still get their chance. The outcome is definite -- checked against
    each tool response, not assumed from the absence of an exception -- because
    a request counted as applied is a request that gets archived and never
    retried.
    """
    applier = ACTION_APPLIERS.get(req.action)
    if applier is None:
        log(f'task {req.task_id}: no applier for action {req.action!r} — not applied')
        return False
    try:
        applier(client, req, project_root)
    except Exception as exc:
        log(f'task {req.task_id}: {req.action} FAILED: {type(exc).__name__}: {exc}')
        return False
    return True
