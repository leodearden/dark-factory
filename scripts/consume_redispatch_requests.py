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
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass

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
