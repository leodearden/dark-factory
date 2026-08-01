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

import consume_redispatch_requests as crr
import pytest

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
    with open(path) as f:
        text = f.read()
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
    with open(path) as f:
        body = json.load(f)
    body['class'] = 'some_future_class'
    with open(path, 'w') as f:
        f.write(json.dumps(body, indent=2, sort_keys=True) + '\n')
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
    with open(path) as f:
        body = json.load(f)
    body['class'] = 'unmet_dependency'
    body['action'] = 'redispatch'
    with open(path, 'w') as f:
        f.write(json.dumps(body, indent=2, sort_keys=True) + '\n')
    req = _load(path)
    assert req.skip_reason is not None and 'class' in req.skip_reason


def test_unparseable_json_is_skipped(requests_dir):
    for raw in ('{not json', '', '[]', 'null', '"a string"'):
        req = _load(write_request(requests_dir, 5, 'gate_closure', raw=raw))
        assert req.skip_reason is not None


def test_missing_required_field_is_skipped(requests_dir):
    for field in ('schema_version', 'task_id', 'class', 'verdict', 'action'):
        path = write_request(requests_dir, 5, 'gate_closure')
        with open(path) as f:
            body = json.load(f)
        del body[field]
        with open(path, 'w') as f:
            f.write(json.dumps(body, indent=2, sort_keys=True) + '\n')
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
    with _FakeMcpServer(_default_responder()) as srv, crr.McpClient(srv.url) as client:
        client.call_tool('get_task', {'id': '1'})
    init_headers, init_payload = srv.received[0]
    assert init_payload['method'] == 'initialize'
    assert not any(k.lower() == 'mcp-session-id' for k in init_headers)


def test_server_assigned_session_id_is_echoed_on_every_later_request():
    with (
        _FakeMcpServer(_default_responder(session_id='sess-xyz')) as srv,
        crr.McpClient(srv.url) as client,
    ):
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

    with _FakeMcpServer(responder) as srv, crr.McpClient(srv.url) as client:
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

    with _FakeMcpServer(responder) as srv, crr.McpClient(srv.url) as client:
        assert client.call_tool('get_task', {'id': '5321'}) == {'id': '5321'}


def test_structured_content_is_preferred_and_text_content_is_the_fallback():
    with (
        _FakeMcpServer(_default_responder(tool_result={'id': '7'})) as srv,
        crr.McpClient(srv.url) as client,
    ):
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

    with _FakeMcpServer(text_responder) as srv, crr.McpClient(srv.url) as client:
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

    with _FakeMcpServer(responder) as srv, crr.McpClient(srv.url) as client:  # noqa: SIM117 -- keep pytest.raises scoped to the CALL: merging it here would let it swallow a server/client setup failure as a pass
        with pytest.raises(RuntimeError, match='boom'):
            client.call_tool('set_task_status', {'id': '1', 'status': 'pending'})


def test_client_identifies_itself_explicitly():
    """The interceptor derives an actor class from clientInfo, so naming
    ourselves is a deliberate choice rather than an incidental default."""
    with _FakeMcpServer(_default_responder()) as srv, crr.McpClient(srv.url) as client:
        client.call_tool('get_task', {'id': '1'})
    _, init_payload = srv.received[0]
    assert init_payload['params']['clientInfo']['name'] == crr.CLIENT_NAME
    assert crr.CLIENT_NAME


def _is_error_result(payload, message):
    """A FastMCP tool-level error: a 200 whose result carries isError."""
    return (200, {'Content-Type': 'application/json'},
            _json_rpc_ok(payload, {
                'content': [{'type': 'text', 'text': message}],
                'isError': True}))


def test_a_tool_level_is_error_result_raises():
    """A tool-level failure arrives INSIDE a 200 as `isError`, not as a
    JSON-RPC error envelope.

    fused-memory's `@mcp_tool_errors()` turns in-tool exceptions into an
    `{'error': ...}` payload, but a FastMCP-level failure -- argument
    validation, an unknown or RENAMED tool -- bypasses that decorator and
    surfaces only this way. A client that returned it would hand the apply
    path a payload carrying neither `error` nor `success: False`, i.e. a
    failed write indistinguishable from a landed one.
    """
    def responder(payload, n):
        if payload.get('method') == 'initialize':
            return (200, {'Content-Type': 'application/json', 'mcp-session-id': 's'},
                    _json_rpc_ok(payload, {}))
        if payload.get('method') == 'notifications/initialized':
            return (202, {}, '')
        return _is_error_result(
            payload, "Input validation error: unexpected keyword 'task_id'")

    with _FakeMcpServer(responder) as srv, crr.McpClient(srv.url) as client:  # noqa: SIM117 -- keep pytest.raises scoped to the CALL: merging it here would let it swallow a server/client setup failure as a pass
        with pytest.raises(RuntimeError, match='Input validation error'):
            client.call_tool('set_task_status', {'id': '1', 'status': 'pending'})


def test_an_unparsed_text_payload_is_read_as_a_rejection_not_a_success():
    """`call_tool` returns {'_raw': text} when a text content block is not JSON.

    That shape is what a bare error string looks like coming back, so treating
    it as a success (no `error` key, no `success: False`) would count a failed
    write as applied and archive the request."""
    assert crr.rejection_reason({'_raw': 'Input validation error: nope'}) is not None
    assert 'Input validation error' in crr.rejection_reason(
        {'_raw': 'Input validation error: nope'})
    # The shapes that really are successes stay successes.
    assert crr.rejection_reason({'success': True}) is None
    assert crr.rejection_reason({'success': True, 'no_op': True}) is None


# ── the wire arguments the three tool wrappers actually send ─────────────────
#
# Every other test in this module either hand-writes an arguments dict or
# substitutes RecordingClient, whose signatures mirror the CONSUMER's rather
# than the SERVER's. These three drive the real McpClient against a socket and
# assert on `params.arguments` as received, so a regression in the wire names
# (the seam the source comment flags as historically error-prone -- the param
# is `id`, not `task_id`) cannot pass the suite.


def _tool_call_params(srv):
    return [p['params'] for _, p in srv.received if p.get('method') == 'tools/call']


def test_get_task_sends_the_servers_wire_parameter_names():
    with _FakeMcpServer(_default_responder()) as srv, crr.McpClient(srv.url) as client:
        client.get_task(5321, REIFY)
    params = _tool_call_params(srv)[-1]
    assert params['name'] == 'get_task'
    assert params['arguments'] == {'id': '5321', 'project_root': REIFY}


def test_set_task_status_sends_the_servers_wire_parameter_names():
    with _FakeMcpServer(_default_responder()) as srv, crr.McpClient(srv.url) as client:
        client.set_task_status(5321, 'cancelled', REIFY)
    params = _tool_call_params(srv)[-1]
    assert params['name'] == 'set_task_status'
    assert params['arguments'] == {
        'id': '5321', 'status': 'cancelled', 'project_root': REIFY}


def test_set_task_claimant_sends_both_clear_kwargs_as_explicit_json_null():
    """PRESENT-with-null is the clear; OMITTED means "leave untouched".

    Both claimant kwargs are tri-state server-side against a '__unset__'
    STRING sentinel, so a dropped key does not clear the field -- it silently
    preserves the stale claimant this consumer exists to remove, and the row
    would then go `pending` still reading as live to a dispatcher."""
    with _FakeMcpServer(_default_responder()) as srv, crr.McpClient(srv.url) as client:
        client.set_task_claimant(5321, REIFY,
                                 claimant_run_id=None, heartbeat_at=None)
    params = _tool_call_params(srv)[-1]
    assert params['name'] == 'set_task_claimant'
    assert params['arguments'] == {
        'id': '5321', 'project_root': REIFY,
        'claimant_run_id': None, 'heartbeat_at': None}
    # Stated separately from the == above: presence is the load-bearing half.
    assert 'claimant_run_id' in params['arguments']
    assert 'heartbeat_at' in params['arguments']


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
    ts = _dt.datetime.fromtimestamp(SWEEP_MTIME + offset_seconds, _dt.UTC)
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


# ── step-7: action -> MCP-call mapping ───────────────────────────────────────
#
# Asserted on the recorded call SEQUENCE, not on call presence. The
# clear-claimant-then-flip-to-pending ordering is load-bearing (see
# orchestrator/src/orchestrator/scheduler.py:5726-5736), and a test that only
# checked "both calls happened" would pass on the reversed, wrong code.


def _apply(client, req, **kw):
    return crr.apply_request(client, req, REIFY, **kw)


def test_close_issues_exactly_one_status_write_and_no_claimant_call(requests_dir):
    req = _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('blocked')})
    assert _apply(client, req) is True
    assert client.calls == [
        ('set_task_status', {'id': 5321, 'status': 'cancelled',
                             'project_root': REIFY}),
    ]


@pytest.mark.parametrize('cls', ['unmet_dependency', 'merge_verify_red'])
def test_redispatch_and_reverify_clear_the_claimant_before_flipping_to_pending(
        requests_dir, cls):
    """Ordering is the assertion. Once the row reads `pending` a competing
    dispatcher may stamp a fresh claimant that a late-landing clear would
    clobber -- so the clear goes FIRST, while the row is still uncontestable."""
    req = _req(requests_dir, 5321, cls)
    client = RecordingClient({5321: _row('blocked')})
    assert _apply(client, req) is True
    assert client.tools_called == ['set_task_claimant', 'set_task_status']
    assert client.calls[0] == ('set_task_claimant', {
        'id': 5321, 'project_root': REIFY,
        'claimant_run_id': None, 'heartbeat_at': None})
    assert client.calls[1] == ('set_task_status', {
        'id': 5321, 'status': 'pending', 'project_root': REIFY})


def test_every_call_targets_reifys_project_root(requests_dir):
    for cls in CLASS_ACTION:
        req = _req(requests_dir, 5321, cls)
        client = RecordingClient({5321: _row('blocked')})
        _apply(client, req)
        assert client.calls, cls
        for _, args in client.calls:
            assert args['project_root'] == REIFY


def test_no_done_provenance_is_ever_sent(requests_dir):
    """No action targets `done`, so the evidence field that only a done
    transition needs must never appear on the wire."""
    for cls in CLASS_ACTION:
        req = _req(requests_dir, 5321, cls)
        client = RecordingClient({5321: _row('blocked')})
        _apply(client, req)
        for _, args in client.calls:
            assert 'done_provenance' not in args
            assert args.get('status') != 'done'


def test_a_failed_claimant_clear_aborts_before_the_status_flip(requests_dir):
    """Flipping to `pending` with a stale claimant still stamped would hand the
    row to a dispatcher that then reads a live-looking claimant. Abort instead."""
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked')}, raises_on={'set_task_claimant'})
    assert _apply(client, req) is False
    assert client.tools_called == ['set_task_claimant']


def test_an_embedded_rejection_on_the_claimant_clear_also_aborts(requests_dir):
    """fused-memory answers its gates with a 200 carrying success=False rather
    than raising, so the absence of an exception is not evidence of a write."""
    req = _req(requests_dir, 5321, 'unmet_dependency')
    client = RecordingClient({5321: _row('blocked')}, rejects_on={'set_task_claimant'})
    assert _apply(client, req) is False
    assert client.tools_called == ['set_task_claimant']


@pytest.mark.parametrize('cls', sorted(CLASS_ACTION))
def test_a_raised_status_write_is_reported_as_failed(requests_dir, cls):
    req = _req(requests_dir, 5321, cls)
    client = RecordingClient({5321: _row('blocked')}, raises_on={'set_task_status'})
    assert _apply(client, req) is False


@pytest.mark.parametrize('cls', sorted(CLASS_ACTION))
def test_an_embedded_rejection_on_the_status_write_is_reported_as_failed(
        requests_dir, cls):
    """The definite applied/failed outcome is checked against the RESPONSE, not
    inferred from the absence of an exception."""
    req = _req(requests_dir, 5321, cls)
    client = RecordingClient({5321: _row('blocked')}, rejects_on={'set_task_status'})
    assert _apply(client, req) is False


def test_a_no_op_success_payload_counts_as_applied(requests_dir):
    """fused-memory returns success=True, no_op=True for a same-status write.
    That is a success, not a rejection."""
    req = _req(requests_dir, 5321, 'gate_closure')

    class _NoOpClient(RecordingClient):
        def _maybe_fail(self, tool):
            return {'success': True, 'no_op': True, 'task_id': 5321}

    client = _NoOpClient({5321: _row('blocked')})
    assert _apply(client, req) is True


# ── step-9: archival, failure containment and blast radius ───────────────────
#
# Ordering is the safety property here: archive ON SUCCESS ONLY. Archiving
# before a confirmed write would drop a request whose transition never landed,
# and leaving a failed request in place makes the retry immediate and local
# rather than dependent on the next sweep re-emitting it.


def _run(requests_dir, client, **kw):
    kw.setdefault('project_root', REIFY)
    return crr.consume(str(requests_dir), client, **kw)


def _top_level(requests_dir):
    return sorted(p.name for p in requests_dir.iterdir() if p.is_file())


def _archived(requests_dir):
    d = requests_dir / 'consumed'
    return sorted(p.name for p in d.iterdir()) if d.is_dir() else []


def test_an_applied_request_is_archived_into_consumed(requests_dir):
    _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('blocked')})
    summary = _run(requests_dir, client)
    assert summary.applied == 1
    assert _top_level(requests_dir) == []
    assert _archived(requests_dir) == ['redispatch-5321-gate_closure.json']


def test_a_failed_apply_leaves_the_file_in_place_for_the_next_run(requests_dir):
    _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('blocked')}, raises_on={'set_task_status'})
    summary = _run(requests_dir, client)
    assert summary.failed == 1 and summary.applied == 0
    assert _top_level(requests_dir) == ['redispatch-5321-gate_closure.json']
    assert _archived(requests_dir) == []


def test_a_guard_skipped_request_is_left_in_place(requests_dir):
    _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('done')})   # out of scope
    summary = _run(requests_dir, client)
    assert summary.skipped == 1 and summary.applied == 0
    assert _top_level(requests_dir) == ['redispatch-5321-gate_closure.json']
    assert _archived(requests_dir) == []


def test_an_invalid_request_is_skipped_and_left_in_place(requests_dir):
    write_request(requests_dir, 5321, 'gate_closure', verdict='CORRUPT-HOLD',
                  mtime=SWEEP_MTIME)
    client = RecordingClient({5321: _row('blocked')})
    summary = _run(requests_dir, client)
    assert summary.skipped == 1
    assert client.calls == []            # never even read the row
    assert _top_level(requests_dir) == ['redispatch-5321-gate_closure.json']


def test_the_archive_destination_is_retraction_safe(requests_dir):
    """The sweep's retraction globs "$REQUESTS_DIR"/redispatch-*.json at the
    TOP LEVEL only, so a subdirectory is invisible to it -- which is what makes
    the archive a durable audit trail rather than something the next sweep
    deletes."""
    import glob
    _req(requests_dir, 5321, 'gate_closure')
    _run(requests_dir, RecordingClient({5321: _row('blocked')}))
    assert _archived(requests_dir) == ['redispatch-5321-gate_closure.json']
    assert glob.glob(str(requests_dir / 'redispatch-*.json')) == []


def test_an_archive_failure_still_counts_the_write_as_applied(requests_dir, capsys):
    """The write LANDED; only the bookkeeping failed. Saying otherwise would
    invite a retry of a transition that already happened."""
    _req(requests_dir, 5321, 'gate_closure')
    # A regular FILE where the archive directory needs to be: os.makedirs()
    # raises FileExistsError even with exist_ok=True, since the path is not a
    # directory.
    (requests_dir / 'consumed').write_text('not a directory\n')

    client = RecordingClient({5321: _row('blocked')})
    summary = _run(requests_dir, client)
    assert summary.applied == 1 and summary.failed == 0
    err = capsys.readouterr().err
    assert 'archiving failed' in err          # loud, never silent
    # Left at the top level, so the next run will see it again.
    assert 'redispatch-5321-gate_closure.json' in _top_level(requests_dir)


def test_a_request_left_behind_by_a_failed_archive_is_a_no_op_next_run(requests_dir):
    """The load-bearing claim in archive_request's failure branch: the next
    run's guard skips the file as already-applied rather than re-transitioning
    the row. Pinned here so the branch's comment cannot quietly become false."""
    _req(requests_dir, 5321, 'gate_closure')
    (requests_dir / 'consumed').write_text('not a directory\n')
    _run(requests_dir, RecordingClient({5321: _row('blocked')}))

    # Second run, with the row now where the first run put it.
    client = RecordingClient({5321: _row('cancelled')})
    summary = _run(requests_dir, client)
    assert summary.skipped == 1 and summary.applied == 0
    assert client.tools_called == ['get_task']     # the guard read, no write


def test_max_writes_stops_after_n_applied_and_defers_the_rest(requests_dir):
    rows = {}
    for tid in range(1, 6):
        _req(requests_dir, tid, 'gate_closure')
        rows[tid] = _row('blocked', task_id=tid)
    client = RecordingClient(rows)
    summary = _run(requests_dir, client, max_writes=2)
    assert summary.applied == 2
    assert summary.deferred == 3
    assert len(_archived(requests_dir)) == 2
    assert len(_top_level(requests_dir)) == 3


def test_deferred_requests_are_reported_not_silently_truncated(requests_dir, capsys):
    for tid in (1, 2, 3):
        _req(requests_dir, tid, 'gate_closure')
    rows = {tid: _row('blocked', task_id=tid) for tid in (1, 2, 3)}
    _run(requests_dir, RecordingClient(rows), max_writes=1)
    err = capsys.readouterr().err
    assert 'deferred' in err


def test_max_writes_counts_attempts_so_a_systematically_failing_run_stops(
        requests_dir):
    """The cap must engage on the run it exists for.

    `_apply_repend` clears the claimant BEFORE flipping the status, so a run
    whose flips are all rejected still lands a real mutation on every row while
    counting each as `failed`. A cap keyed on `applied` alone would sit at zero
    and clear the claimant on the WHOLE directory -- the store-wide event the
    cap's own docstring says it exists to prevent.
    """
    rows = {}
    for tid in range(1, 6):
        _req(requests_dir, tid, 'unmet_dependency')
        rows[tid] = _row('blocked', task_id=tid)
    client = RecordingClient(rows, rejects_on={'set_task_status'})

    summary = _run(requests_dir, client, max_writes=2)
    assert summary.applied == 0
    assert summary.failed == 2
    assert summary.deferred == 3
    # The blast radius that actually reached the store: two claimant clears,
    # not five.
    assert client.tools_called.count('set_task_claimant') == 2


def test_a_skip_does_not_consume_the_max_writes_budget(requests_dir):
    """The cap bounds WRITES, not requests looked at -- otherwise a directory
    of already-applied no-ops would starve the requests that still need work."""
    _req(requests_dir, 1, 'gate_closure')          # already cancelled -> skip
    _req(requests_dir, 2, 'gate_closure')          # applies
    client = RecordingClient({1: _row('cancelled', task_id=1),
                              2: _row('blocked', task_id=2)})
    summary = _run(requests_dir, client, max_writes=1)
    assert summary.applied == 1 and summary.skipped == 1 and summary.deferred == 0


def test_dry_run_guards_and_reports_but_writes_and_archives_nothing(requests_dir):
    _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('blocked')})
    summary = _run(requests_dir, client, dry_run=True)
    assert summary.planned == 1 and summary.applied == 0
    assert client.tools_called == ['get_task']     # the guard read, nothing else
    assert _top_level(requests_dir) == ['redispatch-5321-gate_closure.json']
    assert _archived(requests_dir) == []


def test_dry_run_still_reports_skips(requests_dir):
    _req(requests_dir, 5321, 'gate_closure')
    summary = _run(requests_dir, RecordingClient({5321: _row('done')}), dry_run=True)
    assert summary.skipped == 1 and summary.planned == 0


def test_one_failing_request_does_not_stop_its_siblings(requests_dir):
    _req(requests_dir, 1, 'gate_closure')
    _req(requests_dir, 2, 'gate_closure')
    _req(requests_dir, 3, 'gate_closure')

    class _FailsOnTwo(RecordingClient):
        def set_task_status(self, task_id, status, project_root):
            if task_id == 2:
                self.calls.append(('set_task_status', {'id': task_id}))
                raise RuntimeError('nope')
            return super().set_task_status(task_id, status, project_root)

    client = _FailsOnTwo({tid: _row('blocked', task_id=tid) for tid in (1, 2, 3)})
    summary = _run(requests_dir, client)
    assert summary.applied == 2 and summary.failed == 1
    assert _top_level(requests_dir) == ['redispatch-2-gate_closure.json']


def test_summary_line_reports_every_bucket(requests_dir, capsys):
    _req(requests_dir, 1, 'gate_closure')                       # applies
    _req(requests_dir, 2, 'gate_closure')                       # skipped
    _req(requests_dir, 3, 'gate_closure')                       # fails

    class _FailsOnThree(RecordingClient):
        def set_task_status(self, task_id, status, project_root):
            if task_id == 3:
                raise RuntimeError('nope')
            return super().set_task_status(task_id, status, project_root)

    client = _FailsOnThree({1: _row('blocked', task_id=1),
                            2: _row('done', task_id=2),
                            3: _row('blocked', task_id=3)})
    _run(requests_dir, client)
    line = [ln for ln in capsys.readouterr().err.splitlines() if 'SUMMARY' in ln]
    assert len(line) == 1
    for token in ('applied=1', 'skipped=1', 'failed=1', 'deferred=0'):
        assert token in line[0]


def test_the_applied_line_carries_the_sweeps_evidence(requests_dir, capsys):
    """`evidence` is the only operator-facing statement of WHY a row was
    cancelled, and an applied request is archived out of the top level
    immediately -- so a journal line without it forces whoever reads it later
    to go cross-reference the sweep's own stdout."""
    _req(requests_dir, 5321, 'gate_closure',
         evidence='escalation 3102-1 resolved at 2026-07-29T04:11Z')
    _run(requests_dir, RecordingClient({5321: _row('blocked')}))
    line = [ln for ln in capsys.readouterr().err.splitlines() if 'applied' in ln
            and 'SUMMARY' not in ln]
    assert len(line) == 1
    assert 'escalation 3102-1 resolved at 2026-07-29T04:11Z' in line[0]


def test_the_dry_run_line_carries_the_sweeps_evidence(requests_dir, capsys):
    _req(requests_dir, 5321, 'unmet_dependency', evidence='all deps terminal')
    _run(requests_dir, RecordingClient({5321: _row('blocked')}), dry_run=True)
    line = [ln for ln in capsys.readouterr().err.splitlines() if 'WOULD' in ln]
    assert len(line) == 1 and 'all deps terminal' in line[0]


def test_evidence_is_truncated_and_flattened_to_one_line(requests_dir, capsys):
    """One line per decision is the log format's whole shape; a multi-line or
    unbounded evidence string would break it."""
    _req(requests_dir, 5321, 'gate_closure',
         evidence='dep 1 done\ndep 2 done\n' + 'x' * 500)
    _run(requests_dir, RecordingClient({5321: _row('blocked')}))
    lines = [ln for ln in capsys.readouterr().err.splitlines()
             if 'applied' in ln and 'SUMMARY' not in ln]
    assert len(lines) == 1
    assert 'dep 1 done dep 2 done' in lines[0]      # newlines collapsed
    assert lines[0].endswith('...]')                # truncated, visibly
    assert len(lines[0]) < 300


def test_an_absent_evidence_string_adds_nothing_to_the_line(requests_dir, capsys):
    _req(requests_dir, 5321, 'gate_closure', evidence='')
    _run(requests_dir, RecordingClient({5321: _row('blocked')}))
    line = [ln for ln in capsys.readouterr().err.splitlines() if 'applied' in ln
            and 'SUMMARY' not in ln][0]
    assert line.endswith('close applied -> cancelled')


def _tool_responder(rows, *, is_error_on=()):
    """A fake fused-memory: real rows for get_task, isError for named tools."""
    def responder(payload, n):
        method = payload.get('method')
        if method == 'initialize':
            return (200, {'Content-Type': 'application/json', 'mcp-session-id': 's'},
                    _json_rpc_ok(payload, {}))
        if method == 'notifications/initialized':
            return (202, {}, '')
        name = payload['params']['name']
        if name in is_error_on:
            return _is_error_result(payload, f'Input validation error: {name}')
        if name == 'get_task':
            row = rows[int(payload['params']['arguments']['id'])]
            return (200, {'Content-Type': 'application/json'},
                    _json_rpc_ok(payload, {'structuredContent': row}))
        return (200, {'Content-Type': 'application/json'},
                _json_rpc_ok(payload, {'structuredContent': {'success': True}}))

    return responder


def test_a_tool_level_is_error_is_counted_as_failed_not_applied(requests_dir):
    """End-to-end over a real socket: a FastMCP-level rejection (a renamed tool
    or a bad wire param) must show up as `failed`, with the request left on
    disk. Counting it as applied would archive the file and report a green
    `applied=N` every night while nothing actually transitioned."""
    _req(requests_dir, 5321, 'gate_closure')
    responder = _tool_responder({5321: _row('blocked')},
                                is_error_on={'set_task_status'})
    with _FakeMcpServer(responder) as srv, crr.McpClient(srv.url) as client:
        summary = _run(requests_dir, client)
    assert summary.failed == 1 and summary.applied == 0
    assert _top_level(requests_dir) == ['redispatch-5321-gate_closure.json']
    assert _archived(requests_dir) == []


def test_a_tool_level_is_error_on_the_guard_read_is_a_skip(requests_dir):
    """Fail-safe: an unreadable row cannot show the row is unchanged, so the
    guard declines rather than writing on an unconfirmed observation."""
    _req(requests_dir, 5321, 'gate_closure')
    responder = _tool_responder({5321: _row('blocked')}, is_error_on={'get_task'})
    with _FakeMcpServer(responder) as srv, crr.McpClient(srv.url) as client:
        summary = _run(requests_dir, client)
    assert summary.skipped == 1 and summary.applied == 0 and summary.failed == 0


def test_an_empty_requests_dir_is_a_clean_zero_report(requests_dir):
    summary = _run(requests_dir, RecordingClient({}))
    assert (summary.applied, summary.skipped, summary.failed) == (0, 0, 0)


def test_a_missing_requests_dir_is_a_clean_zero_report(tmp_path):
    summary = crr.consume(str(tmp_path / 'absent'), RecordingClient({}),
                          project_root=REIFY)
    assert (summary.applied, summary.skipped, summary.failed) == (0, 0, 0)


# ── exit codes ──────────────────────────────────────────────────────────────


def _main(argv, client):
    return crr.main(argv, client_factory=lambda server: client)


def test_main_exits_zero_when_individual_requests_fail(requests_dir):
    """A recurring `oneshot` that can fail enters systemd `failed` state and
    STAYS there, so a single unactionable request would silently stop the whole
    nightly job (the lesson already written into
    scripts/fused-memory-flag-marker-sweep.sh)."""
    _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('blocked')}, raises_on={'set_task_status'})
    rc = _main(['--requests-dir', str(requests_dir), '--project-root', REIFY], client)
    assert rc == 0


def test_main_exits_zero_on_a_clean_run(requests_dir):
    _req(requests_dir, 5321, 'gate_closure')
    client = RecordingClient({5321: _row('blocked')})
    assert _main(['--requests-dir', str(requests_dir), '--project-root', REIFY],
                 client) == 0


def test_main_exits_non_zero_on_a_usage_error(requests_dir):
    with pytest.raises(SystemExit) as exc:
        _main(['--not-a-flag'], RecordingClient({}))
    assert exc.value.code != 0


def test_main_rejects_a_non_positive_max_writes(requests_dir):
    """A cap of 0 would silently do nothing every night while reporting success."""
    with pytest.raises(SystemExit) as exc:
        _main(['--requests-dir', str(requests_dir), '--max-writes', '0'],
              RecordingClient({}))
    assert exc.value.code != 0


def _summary_lines(capsys):
    return [ln for ln in capsys.readouterr().err.splitlines() if 'SUMMARY' in ln]


def test_a_clean_run_through_main_emits_exactly_one_summary_line(
        requests_dir, capsys):
    _req(requests_dir, 5321, 'gate_closure')
    _main(['--requests-dir', str(requests_dir), '--project-root', REIFY],
          RecordingClient({5321: _row('blocked')}))
    assert len(_summary_lines(capsys)) == 1


def test_an_unreachable_server_still_emits_a_summary_line(requests_dir, capsys):
    """OPERATIONS.md tells operators "a red run is found by reading the summary
    line, not the unit state". A night that died before consume() could report
    must therefore still emit one -- otherwise a green unit plus a silent
    journal is indistinguishable from a clean no-op run."""
    _req(requests_dir, 5321, 'gate_closure')

    def refuse(server):
        raise ConnectionRefusedError('[Errno 111] Connection refused')

    rc = crr.main(['--requests-dir', str(requests_dir)], client_factory=refuse)
    assert rc == 0                                   # a failed night, not a failed job
    err = capsys.readouterr().err
    assert len([ln for ln in err.splitlines() if 'SUMMARY' in ln]) == 1
    assert 'RUN FAILED' in err
    assert 'could not reach the MCP server' in err
    assert _top_level(requests_dir) == ['redispatch-5321-gate_closure.json']


def test_an_unexpected_error_is_not_mislabelled_as_an_unreachable_server(
        requests_dir, capsys):
    """`consume` is written never to raise, so anything escaping it is a bug in
    THIS consumer. Reporting that as "could not reach <server>" sends an
    operator to check a healthy server instead of reading the traceback."""
    def explode(server):
        raise ValueError('bug in the consumer')

    assert crr.main(['--requests-dir', str(requests_dir)],
                    client_factory=explode) == 0
    err = capsys.readouterr().err
    assert 'RUN FAILED' in err
    assert 'could not reach the MCP server' not in err
    assert 'unexpected error' in err and 'ValueError' in err
    assert len([ln for ln in err.splitlines() if 'SUMMARY' in ln]) == 1


def test_main_default_max_writes_is_the_blast_radius_cap(requests_dir):
    """Default 5: bounds a first-night surprise to a handful of transitions an
    operator can read in the journal and undo."""
    assert crr.DEFAULT_MAX_WRITES == 5
