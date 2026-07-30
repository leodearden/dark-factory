"""Tests for scripts/consume_redispatch_requests.py (task 3102).

The consumer drains the request files emitted by reify's
``scripts/deterministic-gate-closure-staleness-sweep.sh --emit-requests``.
That script is the NORMATIVE contract; this module's fixtures render request
files byte-faithfully to its ``_write_request`` so the parser under test is
exercised against the real on-disk shape rather than a paraphrase of it.

``scripts/tests/conftest.py`` already inserts ``scripts/`` onto ``sys.path``,
so ``import consume_redispatch_requests`` resolves with no packaging work.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import consume_redispatch_requests as crr

# The fixed class -> action mapping, transcribed from the reify sweep
# (_classify_gate_closure -> close, _classify_merge_verify_red -> reverify,
# _classify_unmet_dependency -> redispatch). Tests assert the consumer agrees
# with THIS table; the consumer must never infer an action from a class it
# does not recognise.
CLASS_ACTION = {
    'gate_closure': 'close',
    'merge_verify_red': 'reverify',
    'unmet_dependency': 'redispatch',
}


def request_body(task_id, cls, *, verdict='STALE', action=None,
                 evidence='dependency roll-up satisfied', schema_version=1,
                 main_ref_sha='0' * 40, emitted_by='sweep@host'):
    """Build a request body dict with the sweep's exact key set.

    Defaults produce a VALID request; every field is overridable so a test can
    corrupt exactly one thing at a time.
    """
    return {
        'schema_version': schema_version,
        'task_id': task_id,
        'class': cls,
        'verdict': verdict,
        'action': CLASS_ACTION.get(cls, 'close') if action is None else action,
        'evidence': evidence,
        'main_ref_sha': main_ref_sha,
        'emitted_by': emitted_by,
    }


def write_request(requests_dir, task_id, cls, *, mtime=None, name=None,
                  raw=None, **body_kwargs):
    """Write one request file byte-faithfully to the sweep's ``_write_request``.

    ``json.dump(..., indent=2, sort_keys=True)`` followed by a single trailing
    newline, named ``redispatch-<task_id>-<class>.json`` -- exactly what the
    reify script emits (via its embedded python3 renderer).

    ``mtime`` sets the file's modification time, which is the ONLY recency
    signal the contract provides (the body deliberately carries no wall-clock
    field, so re-emission is byte-idempotent). The compare-and-swap guard reads
    it, so tests must be able to drive it both ways.

    ``raw`` writes the given text verbatim instead of rendering a body, for
    the unparseable-JSON cases. ``name`` overrides the filename, for the
    discovery-predicate cases.
    """
    requests_dir = str(requests_dir)
    fname = name if name is not None else f'redispatch-{task_id}-{cls}.json'
    path = os.path.join(requests_dir, fname)
    if raw is not None:
        text = raw
    else:
        body = request_body(task_id, cls, **body_kwargs)
        text = json.dumps(body, indent=2, sort_keys=True) + '\n'
    with open(path, 'w') as fh:
        fh.write(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def requests_dir(tmp_path):
    """An empty requests directory, as the sweep would create it on demand."""
    d = tmp_path / 'redispatch-requests'
    d.mkdir()
    return d


def test_write_request_is_byte_faithful_to_the_sweep(requests_dir):
    """The fixture renders what the sweep's _write_request renders.

    Pins the helper itself: sorted keys, two-space indent, trailing newline,
    integer task_id, and the sweep's exact eight-key body. If reify's renderer
    ever changes shape, this is the test that should fail first.
    """
    path = write_request(requests_dir, 5321, 'unmet_dependency',
                         evidence='all deps terminal',
                         main_ref_sha='abc123', emitted_by='sweep@box')
    assert os.path.basename(path) == 'redispatch-5321-unmet_dependency.json'
    text = open(path).read()
    assert text == json.dumps({
        'action': 'redispatch',
        'class': 'unmet_dependency',
        'emitted_by': 'sweep@box',
        'evidence': 'all deps terminal',
        'main_ref_sha': 'abc123',
        'schema_version': 1,
        'task_id': 5321,
        'verdict': 'STALE',
    }, indent=2, sort_keys=True) + '\n'
    body = json.loads(text)
    assert set(body) == {
        'schema_version', 'task_id', 'class', 'verdict', 'action',
        'evidence', 'main_ref_sha', 'emitted_by',
    }
    assert body['task_id'] == 5321 and isinstance(body['task_id'], int)
    assert body['verdict'] == 'STALE'
    assert body['action'] == 'redispatch'


def test_write_request_mtime_is_settable(requests_dir):
    """The mtime seam the compare-and-swap guard depends on actually works."""
    path = write_request(requests_dir, 42, 'gate_closure', mtime=1_000_000)
    assert os.path.getmtime(path) == pytest.approx(1_000_000, abs=1)


# ── step-1: request-file discovery and validation ────────────────────────────
#
# The consumer's whole safety posture rests on it acting ONLY on files the
# sweep actually emitted, and never guessing at a body it does not fully
# recognise. These tests pin both halves: the filename predicate (what is even
# looked at) and the strict validator (what is accepted once read).


def _discovered(requests_dir):
    """Basenames the consumer would consider, in whatever order it scans."""
    return sorted(os.path.basename(p) for p in crr.discover_requests(requests_dir))


def test_discovery_picks_up_only_well_formed_request_names(requests_dir):
    write_request(requests_dir, 5321, 'unmet_dependency')
    write_request(requests_dir, 77, 'gate_closure')
    write_request(requests_dir, 88, 'merge_verify_red')
    assert _discovered(requests_dir) == [
        'redispatch-5321-unmet_dependency.json',
        'redispatch-77-gate_closure.json',
        'redispatch-88-merge_verify_red.json',
    ]


def test_discovery_ignores_sweep_tmp_intermediates(requests_dir):
    """redispatch-tmp-XXXXXX is the sweep's mktemp intermediate, mid-write.

    Acting on one would be reading a partial file the atomic-emit contract
    exists to prevent a consumer from ever seeing.
    """
    write_request(requests_dir, 1, 'gate_closure', name='redispatch-tmp-AbCdEf')
    write_request(requests_dir, 1, 'gate_closure', name='redispatch-tmp-XXXXXX.json')
    assert _discovered(requests_dir) == []


def test_discovery_ignores_the_consumed_subdirectory(requests_dir):
    """The consumer's own archive must not be re-consumed on the next run."""
    consumed = requests_dir / 'consumed'
    consumed.mkdir()
    write_request(consumed, 5321, 'gate_closure')
    assert _discovered(requests_dir) == []


def test_discovery_ignores_unrelated_and_malformed_names(requests_dir):
    for name in (
        'README.md',
        'redispatch.json',
        'redispatch-5321.json',                      # no class
        'redispatch--gate_closure.json',             # no id
        'redispatch-abc-gate_closure.json',          # non-numeric id
        'redispatch-5321-unknown_class.json',        # class not in the table
        'redispatch-5321-gate_closure.json.bak',     # not .json
        'sweep-report.txt',
    ):
        write_request(requests_dir, 1, 'gate_closure', name=name)
    assert _discovered(requests_dir) == []


def test_missing_requests_dir_discovers_nothing(tmp_path):
    """An absent dir is a clean zero-request scan, not an error."""
    assert crr.discover_requests(tmp_path / 'nope') == []


def _load(path):
    return crr.load_request(path)


def test_valid_request_loads_into_a_typed_record(requests_dir):
    path = write_request(requests_dir, 5321, 'unmet_dependency',
                         evidence='all deps terminal')
    req = _load(path)
    assert req.skip_reason is None
    assert req.task_id == 5321
    assert req.cls == 'unmet_dependency'
    assert req.action == 'redispatch'
    assert req.verdict == 'STALE'
    assert req.path == path
    assert req.mtime == pytest.approx(os.path.getmtime(path), abs=1)


@pytest.mark.parametrize('cls,action', sorted(CLASS_ACTION.items()))
def test_every_known_class_loads_with_its_fixed_action(requests_dir, cls, action):
    req = _load(write_request(requests_dir, 9, cls))
    assert req.skip_reason is None and req.action == action


def test_wrong_schema_version_is_skipped_not_guessed(requests_dir):
    for bad in (0, 2, '1', None):
        req = _load(write_request(requests_dir, 5, 'gate_closure',
                                  schema_version=bad))
        assert req.skip_reason is not None
        assert 'schema_version' in req.skip_reason


def test_unknown_class_in_body_is_skipped(requests_dir):
    """A body class the consumer has no action table entry for is never acted on."""
    path = write_request(requests_dir, 5, 'gate_closure')
    body = json.load(open(path))
    body['class'] = 'some_future_class'
    open(path, 'w').write(json.dumps(body, indent=2, sort_keys=True) + '\n')
    req = _load(path)
    assert req.skip_reason is not None and 'class' in req.skip_reason


def test_non_integer_task_id_is_skipped(requests_dir):
    for bad in ('5321', 5321.5, None, [5321]):
        req = _load(write_request(requests_dir, bad, 'gate_closure',
                                  name='redispatch-5321-gate_closure.json'))
        assert req.skip_reason is not None
        assert 'task_id' in req.skip_reason


def test_body_task_id_disagreeing_with_the_filename_is_skipped(requests_dir):
    """The name and the body are two statements of the same fact; a mismatch
    means one of them is wrong and there is no basis for choosing."""
    req = _load(write_request(requests_dir, 999, 'gate_closure',
                              name='redispatch-5321-gate_closure.json'))
    assert req.skip_reason is not None and 'task_id' in req.skip_reason


def test_body_class_disagreeing_with_the_filename_is_skipped(requests_dir):
    path = write_request(requests_dir, 5321, 'gate_closure')
    body = json.load(open(path))
    body['class'] = 'unmet_dependency'
    body['action'] = 'redispatch'
    open(path, 'w').write(json.dumps(body, indent=2, sort_keys=True) + '\n')
    req = _load(path)
    assert req.skip_reason is not None and 'class' in req.skip_reason


def test_unparseable_json_is_skipped(requests_dir):
    for raw in ('{not json', '', '[]', 'null', '"a string"'):
        req = _load(write_request(requests_dir, 5, 'gate_closure', raw=raw))
        assert req.skip_reason is not None


def test_missing_required_field_is_skipped(requests_dir):
    for field in ('schema_version', 'task_id', 'class', 'verdict', 'action'):
        path = write_request(requests_dir, 5, 'gate_closure')
        body = json.load(open(path))
        del body[field]
        open(path, 'w').write(json.dumps(body, indent=2, sort_keys=True) + '\n')
        req = _load(path)
        assert req.skip_reason is not None, field


@pytest.mark.parametrize('cls,wrong_action', [
    ('gate_closure', 'redispatch'),
    ('gate_closure', 'reverify'),
    ('merge_verify_red', 'close'),
    ('unmet_dependency', 'close'),
    ('unmet_dependency', 'reverify'),
    ('gate_closure', 'human_gate'),
    ('gate_closure', 'none'),
])
def test_class_action_disagreement_is_skipped(requests_dir, cls, wrong_action):
    """The mapping is FIXED in the sweep. A file that disagrees with it is not
    a request in a dialect we understand -- it is skipped, never reconciled."""
    req = _load(write_request(requests_dir, 5, cls, action=wrong_action))
    assert req.skip_reason is not None and 'action' in req.skip_reason


@pytest.mark.parametrize('verdict', [
    'CORRUPT-HOLD', 'LIVE', 'GATED', 'UNRESOLVED', 'NO-CLASS', 'unknown',
    'stale', 'STALE ',
])
def test_non_stale_verdict_is_a_hard_skip(requests_dir, verdict):
    """The sweep emits ONLY on verdict=STALE, CORRUPT-HOLD included (its
    invariant L5). Anything else in a file is a contract violation upstream,
    so the consumer refuses it rather than inferring intent."""
    req = _load(write_request(requests_dir, 5, 'gate_closure', verdict=verdict))
    assert req.skip_reason is not None and 'verdict' in req.skip_reason


# ── step-3: the stdlib MCP client ────────────────────────────────────────────
#
# The consumer talks to the fused-memory MCP server over streamable HTTP with
# urllib.request (stdlib-only posture -- see the module docstring). That means
# re-implementing four behaviours the httpx McpClient in
# fused-memory/scripts/cgl_eta_scheduler_gate.py already handles, so each one
# gets a test here against a real socket rather than a mock.


class _FakeMcpServer:
    """A real HTTP server on an ephemeral port, scripted per-request.

    Records every received (headers, payload) so the client's session-id
    handling can be asserted on the wire, not on internal attributes.
    """

    def __init__(self, responder):
        self.received = []
        self._responder = responder
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self):
                length = int(self.headers.get('Content-Length') or 0)
                raw = self.rfile.read(length) if length else b''
                try:
                    payload = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    payload = {'_raw': raw.decode()}
                outer.received.append((dict(self.headers), payload))
                status, headers, body = outer._responder(payload, len(outer.received))
                data = body.encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                if data:
                    self.wfile.write(data)

            def log_message(self, *a):
                pass

        self._httpd = HTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self._httpd.server_port}/mcp'

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _json_rpc_ok(payload, result):
    return json.dumps({'jsonrpc': '2.0', 'id': payload.get('id'), 'result': result})


def _default_responder(session_id='sess-abc123', tool_result=None):
    """initialize -> assigns a session id; notifications/initialized -> 202;
    tools/call -> structuredContent."""
    tool_result = tool_result if tool_result is not None else {'ok': True}

    def responder(payload, n):
        method = payload.get('method')
        if method == 'initialize':
            return (200, {'Content-Type': 'application/json',
                          'mcp-session-id': session_id},
                    _json_rpc_ok(payload, {'protocolVersion': '2024-11-05'}))
        if method == 'notifications/initialized':
            return (202, {}, '')
        return (200, {'Content-Type': 'application/json'},
                _json_rpc_ok(payload, {'structuredContent': tool_result}))

    return responder


def test_initialize_is_sent_session_less(requests_dir):
    """The MCP streamable-HTTP contract requires the FIRST request to carry no
    session id. A stateful server 404s "Session not found" on an invented one."""
    with _FakeMcpServer(_default_responder()) as srv:
        with crr.McpClient(srv.url) as client:
            client.call_tool('get_task', {'id': '1'})
    init_headers, init_payload = srv.received[0]
    assert init_payload['method'] == 'initialize'
    assert not any(k.lower() == 'mcp-session-id' for k in init_headers)


def test_server_assigned_session_id_is_echoed_on_every_later_request():
    with _FakeMcpServer(_default_responder(session_id='sess-xyz')) as srv:
        with crr.McpClient(srv.url) as client:
            client.call_tool('get_task', {'id': '1'})
            client.call_tool('get_task', {'id': '2'})
    assert len(srv.received) >= 4  # initialize, initialized, 2x tools/call
    for headers, payload in srv.received[1:]:
        got = {k.lower(): v for k, v in headers.items()}.get('mcp-session-id')
        assert got == 'sess-xyz', payload.get('method')


def test_session_id_is_captured_before_the_202_empty_body_early_return():
    """`notifications/initialized` may answer 202 with NO body. Capturing the
    header only after an empty-body early return loses the id entirely."""
    def responder(payload, n):
        if payload.get('method') == 'initialize':
            # 202 + empty body + a session id, all at once: the exact shape the
            # capture-before-early-return ordering exists for.
            return (202, {'mcp-session-id': 'sess-from-202'}, '')
        if payload.get('method') == 'notifications/initialized':
            return (202, {}, '')
        return (200, {'Content-Type': 'application/json'},
                _json_rpc_ok(payload, {'structuredContent': {'ok': True}}))

    with _FakeMcpServer(responder) as srv:
        with crr.McpClient(srv.url) as client:
            client.call_tool('get_task', {'id': '1'})
    tool_headers = {k.lower(): v for k, v in srv.received[-1][0].items()}
    assert tool_headers.get('mcp-session-id') == 'sess-from-202'


def test_sse_response_is_parsed_by_extracting_the_first_data_line():
    def responder(payload, n):
        if payload.get('method') == 'initialize':
            return (200, {'Content-Type': 'application/json',
                          'mcp-session-id': 's'},
                    _json_rpc_ok(payload, {}))
        if payload.get('method') == 'notifications/initialized':
            return (202, {}, '')
        inner = _json_rpc_ok(payload, {'structuredContent': {'id': '5321'}})
        return (200, {'Content-Type': 'text/event-stream'},
                f'event: message\ndata: {inner}\n\n')

    with _FakeMcpServer(responder) as srv:
        with crr.McpClient(srv.url) as client:
            assert client.call_tool('get_task', {'id': '5321'}) == {'id': '5321'}


def test_structured_content_is_preferred_and_text_content_is_the_fallback():
    with _FakeMcpServer(_default_responder(tool_result={'id': '7'})) as srv:
        with crr.McpClient(srv.url) as client:
            assert client.call_tool('get_task', {'id': '7'}) == {'id': '7'}

    def text_responder(payload, n):
        if payload.get('method') == 'initialize':
            return (200, {'Content-Type': 'application/json', 'mcp-session-id': 's'},
                    _json_rpc_ok(payload, {}))
        if payload.get('method') == 'notifications/initialized':
            return (202, {}, '')
        return (200, {'Content-Type': 'application/json'},
                _json_rpc_ok(payload, {
                    'content': [{'type': 'text', 'text': json.dumps({'id': '9'})}]}))

    with _FakeMcpServer(text_responder) as srv:
        with crr.McpClient(srv.url) as client:
            assert client.call_tool('get_task', {'id': '9'}) == {'id': '9'}


def test_json_rpc_error_raises_rather_than_reading_as_success():
    """An {"error": ...} envelope carries no result. Treating it as success is
    how a failed write gets counted as applied -- exactly the confusion the
    apply path's definite applied/failed outcome exists to prevent."""
    def responder(payload, n):
        if payload.get('method') == 'initialize':
            return (200, {'Content-Type': 'application/json', 'mcp-session-id': 's'},
                    _json_rpc_ok(payload, {}))
        if payload.get('method') == 'notifications/initialized':
            return (202, {}, '')
        return (200, {'Content-Type': 'application/json'},
                json.dumps({'jsonrpc': '2.0', 'id': payload.get('id'),
                            'error': {'code': -32602, 'message': 'boom'}}))

    with _FakeMcpServer(responder) as srv:
        with crr.McpClient(srv.url) as client:
            with pytest.raises(RuntimeError, match='boom'):
                client.call_tool('set_task_status', {'id': '1', 'status': 'pending'})


def test_client_identifies_itself_explicitly():
    """The interceptor derives an actor class from clientInfo, so naming
    ourselves is a deliberate choice rather than an incidental default."""
    with _FakeMcpServer(_default_responder()) as srv:
        with crr.McpClient(srv.url) as client:
            client.call_tool('get_task', {'id': '1'})
    _, init_payload = srv.received[0]
    assert init_payload['params']['clientInfo']['name'] == crr.CLIENT_NAME
    assert crr.CLIENT_NAME


# ── step-5: the confirm-before-write compare-and-swap guard ──────────────────
#
# The guard is a STALE-READ check on the adjudicated row, NOT a re-evaluation
# of the sweep's predicates. It asks only "is this still the row the sweep
# looked at, and is it still in a state where this action means anything?" --
# never "was the sweep right?". The fail-safe direction is always "do nothing".

REIFY = '/home/leo/src/reify'

# A fixed epoch for the request-file mtime, so every timestamp in these tests
# is stated relative to the moment the sweep is pretended to have emitted.
SWEEP_MTIME = 1_760_000_000.0


def _iso(offset_seconds, style='Z'):
    """An ISO-8601 UTC timestamp *offset_seconds* from SWEEP_MTIME.

    Two styles, because the store really does write both: `updatedAt` comes
    from sqlite_task_backend._now() (`...THH:MM:SS.mmmZ`) while `heartbeat_at`
    is stamped by the orchestrator as `datetime.now(UTC).isoformat()`
    (`...+00:00`). A guard that parses only one of them would silently fail
    open on the other.
    """
    import datetime as _dt
    ts = _dt.datetime.fromtimestamp(SWEEP_MTIME + offset_seconds, _dt.timezone.utc)
    if style == 'Z':
        return ts.strftime('%Y-%m-%dT%H:%M:%S.') + f'{ts.microsecond // 1000:03d}Z'
    return ts.isoformat()


def _row(status='blocked', *, task_id=5321, updated_at=None, heartbeat_at=None,
         claimant_run_id=None):
    """A get_task wire row, in fused-memory's _row_to_task shape."""
    return {
        'id': task_id,
        'status': status,
        'updatedAt': _iso(-3600) if updated_at is None else updated_at,
        'heartbeat_at': heartbeat_at,
        'claimant_run_id': claimant_run_id,
        'dependencies': [],
    }


class RecordingClient:
    """An McpClient stand-in that records the exact call sequence.

    Assertions are made on `.calls` -- an ORDERED list of (tool, arguments) --
    because the ordering of the two re-dispatch writes is load-bearing, so a
    test that only checked both calls happened would pass on the wrong code.
    """

    def __init__(self, rows=None, *, get_task_raises=None, raises_on=(),
                 rejects_on=()):
        self.rows = dict(rows or {})
        self.calls = []
        self._get_task_raises = get_task_raises
        self._raises_on = set(raises_on)
        self._rejects_on = set(rejects_on)

    @property
    def tools_called(self):
        return [name for name, _ in self.calls]

    def _maybe_fail(self, tool):
        if tool in self._raises_on:
            raise RuntimeError(f'{tool} exploded')
        if tool in self._rejects_on:
            # A rejection EMBEDDED in a 200 response -- fused-memory's
            # terminal-exit / phantom-done gates answer this way without
            # raising, which is why the apply path checks the payload.
            return {'success': False, 'error': f'{tool} refused'}
        return {'success': True}

    def get_task(self, task_id, project_root):
        self.calls.append(('get_task', {'id': task_id, 'project_root': project_root}))
        if self._get_task_raises:
            raise RuntimeError(self._get_task_raises)
        row = self.rows.get(task_id)
        if row is None:
            return {'error': f'No tasks found for ID(s): {task_id}'}
        return row

    def set_task_status(self, task_id, status, project_root):
        self.calls.append(('set_task_status', {
            'id': task_id, 'status': status, 'project_root': project_root}))
        return self._maybe_fail('set_task_status')

    def set_task_claimant(self, task_id, project_root, *, claimant_run_id=None,
                          heartbeat_at=None):
        self.calls.append(('set_task_claimant', {
            'id': task_id, 'project_root': project_root,
            'claimant_run_id': claimant_run_id, 'heartbeat_at': heartbeat_at}))
        return self._maybe_fail('set_task_claimant')


def _req(requests_dir, task_id, cls, **kw):
    kw.setdefault('mtime', SWEEP_MTIME)
    return crr.load_request(write_request(requests_dir, task_id, cls, **kw))


def _confirm(client, req):
    return crr.confirm_request(client, req, REIFY)


def test_guard_proceeds_on_an_unchanged_in_scope_row(requests_dir):
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked')})
    row, skip = _confirm(client, req)
    assert skip is None
    assert row['status'] == 'blocked'
    assert client.tools_called == ['get_task']


def test_guard_skips_when_the_row_is_already_at_the_target_status(requests_dir):
    """Consuming an already-applied request is a NO-OP, not a second transition.

    This is the idempotency the brief requires: the sweep re-emits byte-identical
    files, and a re-run must not re-transition a row that already moved.
    """
    for cls, already in (('gate_closure', 'cancelled'),
                         ('unmet_dependency', 'pending'),
                         ('merge_verify_red', 'pending')):
        req = _req(requests_dir, 5321, cls)
        client = RecordingClient({5321: _row(already)})
        _, skip = _confirm(client, req)
        assert skip is not None, cls
        assert 'already' in skip


@pytest.mark.parametrize('cls,status', [
    # Classes A and C are `blocked`-ONLY (the sweep's invariant L4).
    ('gate_closure', 'in-progress'),
    ('gate_closure', 'review'),
    ('gate_closure', 'done'),
    ('gate_closure', 'deferred'),
    ('unmet_dependency', 'in-progress'),
    ('unmet_dependency', 'done'),
    # Class B spans blocked + in-progress and nothing else.
    ('merge_verify_red', 'review'),
    ('merge_verify_red', 'done'),
    ('merge_verify_red', 'deferred'),
])
def test_guard_skips_a_row_outside_its_class_legal_scope(requests_dir, cls, status):
    req = _req(requests_dir, 5321, cls)
    client = RecordingClient({5321: _row(status)})
    _, skip = _confirm(client, req)
    assert skip is not None and 'scope' in skip


@pytest.mark.parametrize('cls,status', [
    ('gate_closure', 'blocked'),
    ('unmet_dependency', 'blocked'),
    ('merge_verify_red', 'blocked'),
    ('merge_verify_red', 'in-progress'),
])
def test_guard_accepts_every_in_scope_class_status_pair(requests_dir, cls, status):
    req = _req(requests_dir, 5321, cls)
    client = RecordingClient({5321: _row(status)})
    _, skip = _confirm(client, req)
    assert skip is None


@pytest.mark.parametrize('style', ['Z', 'offset'])
def test_guard_skips_when_updated_at_post_dates_the_request(requests_dir, style):
    """The row moved AFTER the sweep observed it -- let the next sweep re-adjudicate.

    The request body deliberately carries no wall-clock field, so the file's
    mtime is the documented recency signal to compare against.
    """
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked', updated_at=_iso(+60, style))})
    _, skip = _confirm(client, req)
    assert skip is not None and 'updatedAt' in skip


@pytest.mark.parametrize('style', ['Z', 'offset'])
def test_guard_skips_when_a_heartbeat_landed_after_the_sweep(requests_dir, style):
    """The live-agent case: a runner came alive after the sweep's L1 check.

    This is the 2026-07-26 ten-live-in-progress-rows hazard caught a second
    time, at the last possible moment before the write.
    """
    req = _req(requests_dir, 5321, 'merge_verify_red')
    client = RecordingClient({5321: _row(
        'in-progress', heartbeat_at=_iso(+30, style), claimant_run_id='run-abc')})
    _, skip = _confirm(client, req)
    assert skip is not None and 'heartbeat_at' in skip


def test_guard_proceeds_when_both_timestamps_pre_date_the_request(requests_dir):
    req = _req(requests_dir, 5321, 'merge_verify_red')
    client = RecordingClient({5321: _row(
        'in-progress', updated_at=_iso(-600), heartbeat_at=_iso(-300))})
    _, skip = _confirm(client, req)
    assert skip is None


@pytest.mark.parametrize('bad', ['', 'not-a-date', '2026-13-45T99:99:99Z', 12345, {}])
def test_guard_skips_on_an_unparseable_timestamp(requests_dir, bad):
    """Fail-safe: an un-comparable timestamp means we cannot show the row is
    unchanged, and the safe direction for a re-dispatch decision is do-nothing."""
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked', updated_at=bad)})
    _, skip = _confirm(client, req)
    assert skip is not None


def test_guard_skips_when_updated_at_is_missing(requests_dir):
    req = _req(requests_dir, 5321, 'unmet_dependency')
    row = _row('blocked')
    del row['updatedAt']
    client = RecordingClient({5321: row})
    _, skip = _confirm(client, req)
    assert skip is not None


def test_guard_tolerates_an_absent_heartbeat(requests_dir):
    """No heartbeat at all is the normal shape for a `blocked` row -- absence is
    not staleness, so it must not be treated as an unparseable timestamp."""
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked', heartbeat_at=None)})
    _, skip = _confirm(client, req)
    assert skip is None


def test_guard_skips_when_get_task_raises(requests_dir):
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked')}, get_task_raises='server down')
    row, skip = _confirm(client, req)
    assert row is None and skip is not None and 'server down' in skip
    assert client.tools_called == ['get_task']


def test_guard_skips_when_the_row_does_not_exist(requests_dir):
    req = _req(requests_dir, 9999, 'unmet_dependency')
    client = RecordingClient({})
    row, skip = _confirm(client, req)
    assert row is None and skip is not None


def test_guard_skips_when_the_row_has_no_status(requests_dir):
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: {'id': 5321, 'updatedAt': _iso(-60)}})
    _, skip = _confirm(client, req)
    assert skip is not None


def test_guard_never_issues_a_write(requests_dir):
    """The guard is a READ. Whatever it decides, it decides before any mutation."""
    for status in ('blocked', 'done', 'cancelled', 'in-progress'):
        req = _req(requests_dir, 5321, 'gate_closure')
        client = RecordingClient({5321: _row(status)})
        _confirm(client, req)
        assert client.tools_called == ['get_task'], status
