"""``/api/v2/dashboard/tasks?terminal=<project>`` — the on-demand terminal window, end to end.

The default render reads no terminal row at all. A client that wants them
asks with ``?terminal=<project>`` and is answered with ONE ``Datum`` in the
``lower_bound`` state, under the flat key ``TASKS_TERMINAL:<project>``.

Every test drives the real handler over the shared substrate fake
(``tests/_canned_mcp.py``), patched at ``dashboard.data.tasks.mcp_tool_call``,
so the real ``fetch_tasks`` / ``fetch_task_page`` / ``fetch_statuses`` / unit
cache run underneath and every read is asserted at the wire. The tab's half
of the contract, which projects it asks for, is
``test_tab_tasks_terminal_window.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
from unittest.mock import AsyncMock, patch

from _canned_mcp import CannedMCP
from _dashboard_helpers import bug_warnings, node_path
from shared.task_statuses import TERMINAL

# ---------------------------------------------------------------------------
# /api/v2/dashboard/tasks?terminal=<project> — the on-demand window (task 5587)
# ---------------------------------------------------------------------------


def _terminal_row(task_id, status='done'):
    """One raw MCP ``get_tasks`` row in a terminal status, distinguishable by id."""
    return {
        'id': task_id,
        'title': f'task {task_id}',
        'status': status,
        'dependencies': [],
        'metadata': {},
        'updatedAt': f'2026-01-0{task_id % 9 + 1}T00:00:00+00:00',
    }


def _install_terminal_roots(client, tmp_path, labels):
    """One real directory per label under *tmp_path*, installed as the running roster.

    ``dataclasses.replace`` of the running config rather than a fresh one, so
    the ``client`` fixture's dead-port fused-memory URL is kept: a seam the
    canned substrate does not cover fails fast instead of dialing a live
    service.
    """
    roots = [tmp_path / label for label in labels]
    for root in roots:
        root.mkdir(parents=True, exist_ok=True)
    client.app.state.config = dataclasses.replace(
        client.app.state.config,
        project_root=roots[0], known_project_roots=list(roots[1:]),
    )
    return roots


@contextlib.contextmanager
def _over_the_substrate(monkeypatch, canned):
    """Requests made inside run with only ``mcp_tool_call`` canned, from a cold unit.

    Everything above the substrate is real: the handler, the collectors,
    ``acquire_snapshot`` and its cache. Requests made inside one block share
    that cache, as two renders inside one TTL do. The runtime probe is stubbed
    empty because the default render's full collector runs it, and it must
    dial no orchestrator. The unit cache and the root rotation start, and are
    left, empty.
    """
    import dashboard.data.active_tasks as active_tasks_mod
    import dashboard.data.task_snapshot as snapshot_mod

    monkeypatch.setattr(
        'dashboard.data.active_tasks.fetch_task_runtime', AsyncMock(return_value={}),
    )
    snapshot_mod._snapshot_cache_clear()
    active_tasks_mod._reset_root_rotation()
    try:
        with patch('dashboard.data.tasks.mcp_tool_call', new=canned):
            yield
    finally:
        snapshot_mod._snapshot_cache_clear()
        active_tasks_mod._reset_root_rotation()


def _terminal_render(
    client, monkeypatch, tmp_path, *, terminal, labels=('dark-factory',),
    terminal_rows=(), window=None, count_unknown=(),
):
    """``GET /tasks?terminal=<terminal>`` over the canned substrate; ``(body, calls)``.

    The CENSUS the window is positioned from is measured from *terminal_rows*
    themselves, so the census and the rows cannot disagree — the window's
    offset is a function of the terminal population, so a test that cannot
    state that population cannot check the offset. Every *count_unknown*
    root's status-map read times out, which leaves its census UNKNOWN.

    The terminal read is asserted at the wire: "the handler discards the rows
    afterwards" is precisely the defect, and which rows the server was asked
    for is observable nowhere else. ``conftest.py::client`` runs with the
    samplers off, so the call log holds only this request's own reads.
    """
    import dashboard.data.task_snapshot as snapshot_mod

    if window is not None:
        monkeypatch.setattr(snapshot_mod, '_TERMINAL_FETCH_WINDOW', window)
    _install_terminal_roots(client, tmp_path, labels)
    canned = CannedMCP(
        rows=terminal_rows,
        status_map={int(row['id']): row['status'] for row in terminal_rows},
    )
    unmeasured = {str(tmp_path / label) for label in count_unknown}
    canned.fail_when = lambda call: (
        call['tool'] == 'get_statuses'
        and call['args'].get('project_root') in unmeasured
    )
    with _over_the_substrate(monkeypatch, canned):
        resp = client.get(f'/api/v2/dashboard/tasks?terminal={terminal}')
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # data.js::ON_DEMAND_KEYS.terminal.key names the body key AND the DF_DATA
    # key, and refreshOne applies body[key] verbatim, so the window must answer
    # under exactly one flat per-project key. A nested TASKS_TERMINAL map would
    # pass every read below and never reach the browser.
    assert set(body) == {f'TASKS_TERMINAL:{terminal}', 'served_at'}, (
        'the client applies exactly the one key it built plus the served_at '
        'receipt (data.js::startOnDemand/refreshOne), so any other key is '
        f'bytes nobody reads; got {sorted(body)}'
    )
    return body, canned.calls


def _terminal_keys(body):
    """Every key of *body* naming a terminal window, bare or per-project."""
    return sorted(key for key in body if key.startswith('TASKS_TERMINAL'))


def _reads_terminal_rows(statuses):
    """Whether a task read filtered to *statuses* can return a terminal row.

    ``None`` is the whole tree, terminal rows included.
    """
    return statuses is None or bool(TERMINAL & set(statuses))


def _terminal_get_tasks(calls):
    """The ``get_tasks`` calls that read terminal rows: the window, never the unit's ACTIVE read."""
    return [
        call for call in calls
        if call['tool'] == 'get_tasks'
        and _reads_terminal_rows(call['args'].get('statuses'))
    ]


# The REAL client's on-demand path, run over a body handed in on stdin. The
# three scripts load in index.html's order because each destructures the one
# before it at module scope, and `document` stays undefined on purpose: data.js
# starts polling only in a real browser document, so without one it loads
# inert (dashboard/tests/js/_data_loader.mjs::loadDataJs records why).
_ON_DEMAND_CLIENT = r"""
const fs = require('fs');
const path = require('path');
const [redux, project] = process.argv.slice(1);
const body = JSON.parse(fs.readFileSync(0, 'utf8'));
globalThis.window = { dispatchEvent() {} };
require(path.join(redux, 'endpoint_staleness.js'));
require(path.join(redux, 'datum.js'));
const api = require(path.join(redux, 'data.js'));
api.requestOnDemand('terminal', project, {
  state: api.createPollState(),
  deps: { fetchImpl: () => Promise.resolve({ ok: true, json: async () => body }), now: () => 1 },
}).then(outcome => {
  const datum = api.datumFor(`TASKS_TERMINAL:${project}`);
  process.stdout.write(JSON.stringify({ outcome, datum }));
});
"""


def _request_on_demand(body, project):
    """``(outcome, datum)`` from data.js's REAL ``requestOnDemand`` over *body*."""
    import json
    import subprocess
    from pathlib import Path

    redux = Path(__file__).parent.parent / 'src' / 'dashboard' / 'static' / 'redux'
    result = subprocess.run(
        [node_path(), '-e', _ON_DEMAND_CLIENT, str(redux), project],
        input=json.dumps(body), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f'the client driver exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    applied = json.loads(result.stdout)
    return applied['outcome'], applied['datum']


class TestTerminalWindow:
    """``?terminal=<project>`` — the only render that pays for terminal rows.

    The default render stopped fetching them, so every fact the retired
    ``TestShapeOneProjectNarrowing`` terminal cases protected — the window
    reaches the HIGH-id end, it goes through ``fetch_task_page``, truncation
    WARNS rather than capping silently, and an unpositionable window is not
    fetched at all — has to hold HERE or it holds nowhere.
    """

    def test_the_answer_carries_only_the_window_and_its_receipt(
        self, client, monkeypatch, tmp_path
    ):
        """The answer is the window and ``served_at``, and nothing else.

        The only client applies exactly one key and reads ``served_at`` as its
        receipt (``data.js::startOnDemand`` and ``refreshOne``). Every
        project's snapshot and every banner list used to ride along with it:
        task 5598 measured 3.38 MB for this request, 0.32 MB of it the window.
        """
        from datetime import datetime

        body, _calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            labels=('dark-factory', 'other'),
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        default_render_keys = {
            'TASKS_SNAPSHOT', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS',
            'TASKS_DEGRADED_PROJECTS', 'TASKS_COUNT_UNKNOWN_PROJECTS',
            'TASKS_PROJECT_COUNT',
        }
        assert not default_render_keys & set(body), (
            'no project snapshot and no banner list belongs on the terminal '
            f'answer: {sorted(default_render_keys & set(body))}'
        )
        served_at = datetime.fromisoformat(body['served_at'])
        assert served_at.utcoffset() is not None, (
            'served_at must name one moment, not a local-clock reading'
        )
        window = body['TASKS_TERMINAL:dark-factory']
        assert window['state'] == 'lower_bound', window
        assert datetime.fromisoformat(window['as_of']) <= served_at, (
            'the window claims to be measured after it was served'
        )

    def test_the_answer_reads_only_the_named_roots_unit(
        self, client, monkeypatch, tmp_path
    ):
        """One root asked for, one root read — asserted at the MCP wire.

        The answer carries no other root's facts, so reading another root
        would spend that root's share of the fused-memory budget for nothing.
        The named root's own status map IS read: the window is positioned from
        that root's census.
        """
        body, calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            labels=('dark-factory', 'other'),
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        named = str(tmp_path / 'dark-factory')
        read_roots = {call['args'].get('project_root') for call in calls}
        assert read_roots == {named}, (
            f'only {named} may be read for its terminal answer, got {sorted(read_roots)}'
        )
        assert [call for call in calls if call['tool'] == 'get_statuses'], (
            'the window is positioned from the named root\'s own census, so its '
            f'status map must be read: {calls}'
        )
        assert body['TASKS_TERMINAL:dark-factory']['state'] == 'lower_bound'

    def test_the_window_is_positioned_from_the_cached_unit(
        self, client, monkeypatch, tmp_path
    ):
        """Inside one TTL, the terminal answer reuses the unit the default render cached.

        The census the window is positioned from is the one the default render
        just measured, not a second walk of the status map.
        """
        rows = [_terminal_row(i) for i in range(100, 110)]
        canned = CannedMCP(
            rows=rows, status_map={int(row['id']): row['status'] for row in rows},
        )
        _install_terminal_roots(client, tmp_path, ['dark-factory'])

        with _over_the_substrate(monkeypatch, canned):
            default = client.get('/api/v2/dashboard/tasks')
            answer = client.get('/api/v2/dashboard/tasks?terminal=dark-factory')

        assert default.status_code == 200, default.text
        assert answer.status_code == 200, answer.text
        status_walks = canned.calls_to('get_statuses')
        assert len(status_walks) == 1, (
            'the terminal answer re-walked the status map instead of reading '
            f'the cached unit: {status_walks}'
        )
        assert answer.json()['TASKS_TERMINAL:dark-factory']['state'] == 'lower_bound'

    def test_the_window_is_a_lower_bound_datum_naming_the_bound(
        self, client, monkeypatch, tmp_path
    ):
        """(a) The disclosure IS the state: a windowed read under-reports, and says so.

        The window size is parsed OUT of ``reason`` and compared to the module
        constant rather than matched against a fixed string, so a reworded
        message cannot silently change the bound the payload discloses.
        """
        import re
        from datetime import datetime

        from dashboard.data.task_snapshot import _TERMINAL_FETCH_WINDOW

        body, _calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        entry = body['TASKS_TERMINAL:dark-factory']
        assert entry['state'] == 'lower_bound', (
            'a windowed read is a measured value known to under-report — the '
            f'state is the disclosure, got {entry["state"]!r}'
        )
        assert len(entry['value']) == 10
        assert all(row['started'] == 0 for row in entry['value']), (
            'a finished task has no elapsed runtime to report'
        )
        assert all(row['completed'] for row in entry['value']), (
            'a terminal row carries the completion instant the substrate offers'
        )
        as_of = datetime.fromisoformat(entry['as_of'])
        assert as_of.utcoffset() is not None, (
            'as_of must name one moment, not a local-clock reading'
        )
        assert as_of <= datetime.fromisoformat(body['served_at'])
        assert _TERMINAL_FETCH_WINDOW in [
            int(number) for number in re.findall(r'\d+', entry['reason'] or '')
        ], (
            'the reason must disclose the window the rows were read under, so '
            'a consumer can tell "these are all of them" from "these are the '
            f'newest {_TERMINAL_FETCH_WINDOW}"; got {entry["reason"]!r}'
        )

    def test_the_window_reaches_the_high_id_end(
        self, client, monkeypatch, tmp_path
    ):
        """(b) ``offset`` must select the NEWEST terminal rows, not the oldest.

        Relocated from ``TestShapeOneProjectNarrowing``. ``page_size``/
        ``offset`` slice an ASCENDING-id list, so a naive ``offset=0`` returns
        the oldest terminal rows — the opposite of what the tab renders.
        """
        body, calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            window=3,
        )

        page = _terminal_get_tasks(calls)
        assert len(page) == 1, f'exactly one windowed read, got {page}'
        assert page[0]['args'].get('statuses') == ['cancelled', 'done']
        assert page[0]['args'].get('page_size') == 3
        assert page[0]['args'].get('offset') == 7, 'max(0, n_terminal - window)'

        emitted = sorted(
            int(row['id'].rsplit('T-', 1)[-1])
            for row in body['TASKS_TERMINAL:dark-factory']['value']
        )
        assert emitted == [107, 108, 109], (
            f'the window must reach the high-id end, got {emitted}'
        )

    def test_the_terminal_window_goes_through_fetch_task_page(
        self, client, monkeypatch, tmp_path
    ):
        """(b) The read that wants a PARTIAL answer names itself one.

        Relocated from ``TestShapeOneProjectNarrowing``. After task 5018 the
        partial-answer intent lives in the function name rather than in an
        argument combination, so a reader cannot mistake this for a whole-tree
        read — and ``fetch_tasks``, the whole-set read, must not be asked for
        a terminal row on this path. The unit's ACTIVE read is a whole-set
        read of its own, and may go through it.
        """
        import dashboard.data.task_snapshot as snapshot_mod

        whole: list[dict] = []
        paged: list[dict] = []

        async def _whole_set(client_, config, project_root, **kwargs):
            whole.append(kwargs)
            return []

        async def _one_page(client_, config, project_root, **kwargs):
            paged.append(kwargs)
            return []

        monkeypatch.setattr(snapshot_mod, 'fetch_tasks', _whole_set)
        monkeypatch.setattr(snapshot_mod, 'fetch_task_page', _one_page)

        _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        assert len(paged) == 1, f'exactly one windowed read, got {paged}'
        assert paged[0]['statuses'] == sorted(TERMINAL)
        assert paged[0]['page_size'] == snapshot_mod._TERMINAL_FETCH_WINDOW
        assert paged[0]['offset'] == max(
            0, 10 - snapshot_mod._TERMINAL_FETCH_WINDOW
        )
        terminal_whole = [
            kwargs for kwargs in whole if _reads_terminal_rows(kwargs.get('statuses'))
        ]
        assert terminal_whole == [], (
            'the terminal window is a PARTIAL read and must not borrow the '
            f'whole-set read to make it, got {terminal_whole}'
        )

    def test_no_truncation_warning_when_the_population_fits(
        self, client, monkeypatch, tmp_path, caplog
    ):
        """(c) n_terminal <= window → offset 0, every terminal row, no WARNING."""
        import logging

        body, calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 106)],
            window=10,
        )

        assert _terminal_get_tasks(calls)[0]['args'].get('offset') == 0
        assert len(body['TASKS_TERMINAL:dark-factory']['value']) == 6
        assert not [
            record for record in caplog.records
            if record.name == 'dashboard.data.task_snapshot'
            and record.levelno >= logging.WARNING
        ], 'no truncation WARNING may fire when the population fits the window'

    def test_truncation_warns_naming_project_and_counts(
        self, client, monkeypatch, tmp_path, caplog
    ):
        """(c) n_terminal > window → a WARNING naming project, count and window.

        Relocated from ``TestShapeOneProjectNarrowing``. No silent cap: the
        window selects by descending id while a reader expects recency, so
        when the gap can bite it has to be visible in the log.
        """
        import logging

        with caplog.at_level(logging.WARNING, logger='dashboard.data.task_snapshot'):
            _terminal_render(
                client, monkeypatch, tmp_path, terminal='dark-factory',
                terminal_rows=[_terminal_row(i) for i in range(100, 112)],
                window=3,
            )

        messages = [
            record.getMessage() for record in caplog.records
            if record.name == 'dashboard.data.task_snapshot'
            and record.levelno >= logging.WARNING
        ]
        assert any(
            'dark-factory' in message and '12' in message and '3' in message
            for message in messages
        ), f'expected a truncation WARNING naming project/count/window, got {messages}'

    def test_an_unpositionable_window_is_unknown_not_the_oldest_rows(
        self, client, monkeypatch, tmp_path
    ):
        """(d) No census, no offset — and offset 0 would serve the OLDEST rows.

        Relocated from ``test_offline_status_map_never_emits_the_oldest_
        terminal_rows``. The offset is ``n_terminal - window`` and
        ``n_terminal`` comes from the census; with the census unmeasured the
        offset collapses to ``max(0, 0 - window) == 0``, which slices the
        ASCENDING-id list at its OLDEST end — months-old rows presented as the
        newest. Emitting nothing and saying why is the honest failure.
        """
        body, calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
            count_unknown=['dark-factory'], window=3,
        )

        entry = body['TASKS_TERMINAL:dark-factory']
        assert entry['state'] == 'unknown'
        assert entry['value'] is None, (
            'omitting the rows is honest; showing the OLDEST as the newest is not'
        )
        assert entry['as_of'] is None
        assert (entry['reason'] or '').strip(), (
            'an unknown datum must say why it is unknown'
        )
        assert _terminal_get_tasks(calls) == [], (
            'a window that cannot be positioned must not be fetched at all, '
            f'got {_terminal_get_tasks(calls)}'
        )
        # The unknown window IS the whole answer, still carrying served_at.
        assert body['served_at'], body

    def test_an_unconfigured_project_is_unknown_not_a_500(
        self, client, monkeypatch, tmp_path
    ):
        """(e) A name that resolves to no root is answered, not crashed on.

        An empty success would be the worse failure of the two: it reads as
        "this project has no terminal tasks", which is a measurement nobody
        made.
        """
        body, calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='no-such-project',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        entry = body['TASKS_TERMINAL:no-such-project']
        assert entry['state'] == 'unknown'
        assert entry['value'] is None
        assert 'no-such-project' in (entry['reason'] or ''), (
            f'the reason must name what could not be resolved, got {entry["reason"]!r}'
        )
        assert calls == [], (
            f'a name that resolves to no root must read no root at all, got {calls}'
        )

    def test_no_terminal_query_carries_no_terminal_key(
        self, client, monkeypatch, tmp_path
    ):
        """(f) No terminal key at all, bare or per-project — an empty one is a claim."""
        _install_terminal_roots(client, tmp_path, ['dark-factory'])
        with _over_the_substrate(monkeypatch, CannedMCP()):
            resp = client.get('/api/v2/dashboard/tasks')

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert _terminal_keys(body) == []
        assert 'dark-factory' in body['TASKS_SNAPSHOT'], (
            f'no parameter is the default render: {sorted(body)}'
        )

    def test_the_terminal_fetch_spends_the_third_roster_slot(
        self, client, monkeypatch, tmp_path
    ):
        """(g) The window is bounded by the budget the roster reserved for it.

        The roster names three bounded OPERATIONS and the default render spends
        two; this render spends the third. The arithmetic
        ``PER_CALL_TIMEOUT * len(PER_PROJECT_MCP_CALLS) <=
        _TASKS_PER_PROJECT_BUDGET`` is what makes that affordable, and the
        ``timeout=`` at the wire is what makes it true rather than aspirational.
        """
        from dashboard.data.active_tasks import _TASKS_PER_PROJECT_BUDGET
        from dashboard.data.task_snapshot import (
            PER_CALL_TIMEOUT,
            PER_PROJECT_MCP_CALLS,
        )

        _body, calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        page = _terminal_get_tasks(calls)
        assert len(page) == 1
        assert page[0]['kwargs'].get('timeout') == PER_CALL_TIMEOUT, (
            'the terminal read must carry the per-operation budget its roster '
            f'slot reserves, got {page[0]["kwargs"]}'
        )
        assert any('terminal' in slot for slot in PER_PROJECT_MCP_CALLS), (
            'the roster must name the slot this read spends, or the budget '
            f'arithmetic is not about it: {PER_PROJECT_MCP_CALLS}'
        )
        assert PER_CALL_TIMEOUT * len(PER_PROJECT_MCP_CALLS) <= _TASKS_PER_PROJECT_BUDGET

    def test_the_real_client_applies_the_window_this_endpoint_serves(
        self, client, monkeypatch, tmp_path
    ):
        """(h) Both halves of the key contract, run against each other.

        ``data.js::ON_DEMAND_KEYS.terminal.key`` names the body key AND the
        DF_DATA key, and ``refreshOne`` applies ``body[key]`` verbatim. Each
        side once pinned only its own literal, both suites were green, and the
        pair still disagreed: this endpoint nested the window under
        ``TASKS_TERMINAL`` while the client read ``TASKS_TERMINAL:<project>``,
        so the window would never have applied — and nothing would have said
        so, because ``applied`` is reported for a body that merely LACKS the
        key. Feeding this endpoint's own JSON through the REAL
        ``requestOnDemand`` fails on a rename on either side.

        The body the real client is fed is the narrowed one — the window and
        ``served_at``, nothing else — and it still stores the window with its
        receipt.
        """
        body, _calls = _terminal_render(
            client, monkeypatch, tmp_path, terminal='dark-factory',
            terminal_rows=[_terminal_row(i) for i in range(100, 110)],
        )

        outcome, stored = _request_on_demand(body, 'dark-factory')

        assert outcome == 'applied'
        assert stored['state'] == 'lower_bound', (
            'the client found no window under the key it reads, so it still '
            f'holds its pre-request placeholder: {stored}'
        )
        assert stored['_served_at'] == body['served_at'], (
            'the receipt must come from the body the window arrived in'
        )
        assert stored['value'] == body['TASKS_TERMINAL:dark-factory']['value'], (
            'the client must store the row list the server served, unchanged'
        )


def test_a_terminal_window_that_breaks_the_envelope_is_unknown_not_a_500(
    client, monkeypatch, tmp_path, caplog,
):
    """The window degrades to unknown by the same rule as a root's snapshot.

    A ``lower_bound`` datum with no reason breaks the envelope. The key still
    answers, as an explained unknown, with its receipt.
    """
    import logging
    from datetime import UTC, datetime

    from dashboard.data.datum import Datum, DatumInvariant, DatumState
    from dashboard.data.task_snapshot import FRESHNESS_BOUND_SECONDS

    label = 'p0'
    unexplained = Datum(
        [], datetime.now(UTC), DatumState.LOWER_BOUND, None, FRESHNESS_BOUND_SECONDS,
    )
    _install_terminal_roots(client, tmp_path, [label])
    with patch(
        'dashboard.api.tasks.acquire_terminal_window',
        new=AsyncMock(return_value=unexplained),
    ), caplog.at_level(logging.WARNING), _over_the_substrate(monkeypatch, CannedMCP()):
        resp = client.get(f'/api/v2/dashboard/tasks?terminal={label}')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    key = f'TASKS_TERMINAL:{label}'
    window = body[key]
    assert window['state'] == 'unknown', window
    assert window['value'] is None and window['as_of'] is None
    assert DatumInvariant.REASON_REQUIRED.value in window['reason']
    assert set(body) == {key, 'served_at'}, sorted(body)
    assert len(bug_warnings(caplog)) == 1, bug_warnings(caplog)


def test_a_bug_in_the_window_read_is_unknown_not_a_500(
    client, monkeypatch, tmp_path, caplog,
):
    """Our own code raising under the window read costs that one key, never the response.

    Every I/O failure is already a degraded half by the time it gets here, so
    an exception that reaches the handler is a bug. The key still answers, as
    an explained unknown with its receipt, and the bug is logged as one.
    """
    import logging

    label = 'p0'
    _install_terminal_roots(client, tmp_path, [label])
    canned = CannedMCP()
    with patch(
        'dashboard.api.tasks.acquire_snapshot',
        new=AsyncMock(side_effect=RuntimeError('a shaping bug')),
    ), caplog.at_level(logging.WARNING), _over_the_substrate(monkeypatch, canned):
        resp = client.get(f'/api/v2/dashboard/tasks?terminal={label}')

    assert resp.status_code == 200, resp.text
    body = resp.json()
    key = f'TASKS_TERMINAL:{label}'
    assert set(body) == {key, 'served_at'}, sorted(body)
    window = body[key]
    assert window['state'] == 'unknown', window
    assert window['value'] is None and window['as_of'] is None
    assert (window['reason'] or '').strip(), 'an unknown datum must say why'
    assert canned.calls == [], (
        f'a window that cannot be positioned must not be fetched: {canned.calls}'
    )
    assert len(bug_warnings(caplog)) == 1, bug_warnings(caplog)
