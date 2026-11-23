"""No dashboard endpoint may hang when the fetch_tasks MCP seam hangs.

This is the ACCEPTANCE PROBE for task 4788 — verification by probe, not by
reading. Three endpoints (``/api/v2/dashboard/orchestrators``,
``/merge-queue``, ``/escalations``) once wedged for 19.8 h behind a hung
fused-memory seam, and ``/tasks`` — which reaches the SAME ``fetch_tasks``
through ``collect_tasks_with_counts`` — survived, because it alone wrapped
the call in ``asyncio.wait_for``. That asymmetry is what this file pins.

``/orchestrators`` has since been taken out of the population altogether
(task 5587): it reads no task tree, so it has no seam to hang behind. It is
still swept for a 200 — the route list is derived and nothing can opt out of
it — but it is no longer one of the endpoints that must reach the hang.

Why the per-request timeout was never enough: ``fetch_tasks``' *timeout* is
threaded into ``mcp_tool_call`` and thence ``client.post``, so it bounds
connect/read/write and pool acquisition ONLY. The incident's hang was inside
httpcore's connection lock, where no outbound socket is ever opened and that
timeout therefore never fires. Only an enclosing ``asyncio.wait_for`` — whose
cancellation IS delivered into an awaiting lock — bounds it.

Three properties keep this probe from quietly stopping checking:

* the route list is DERIVED from ``app.routes``, never hard-coded, so a new
  endpoint is swept automatically and a shrinking sweep fails loudly;
* the hang stub is ``await asyncio.Event().wait()`` on an event nothing ever
  sets — it has no duration at all, so no budget value can make pre-fix code
  pass, unlike a slow-sleep stub;
* a NON-VACUITY assertion requires each task-lookup endpoint's OWN request,
  and ``/prds``' lineage refresh, to reach the hang for its own root, so the
  sweep cannot pass by never exercising one, nor by one endpoint's reach
  standing in for another's.

A lookup whose deadline would enclose a cold snapshot read is served from a
pre-warmed unit instead, so its cold read is checked only by the warm-up
finishing within its bound; ``/escalations`` makes its cold read in its own
request, deadline-free, and the sweep requires it to.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Collection, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _dashboard_routes import dashboard_get_paths
from escalation.models import Escalation

# The task-prose probe's route budget: the stub never returns, so the route
# pays exactly this and then degrades.
_PROSE_BUDGET = 0.05

# Each task-lookup endpoint pays this once, against the hung fetch_task seam.
_LOOKUP_BUDGET = 0.5

# /prds waits on its lineage refresh, and the refresh on the hung projected
# read, for this and no longer.
_LINEAGE_BUDGET = 0.05

# The warm-up must finish within this many snapshot budgets, or the read is unbounded.
_WARM_UP_BUDGETS = 10

_SNAPSHOT = 'dashboard.data.task_snapshot'
_LOOKUP = 'dashboard.data.task_lookup'
_LINEAGE = 'dashboard.data.task_lineage'

# Every task-read binding the sweep hangs, keyed by the module that imported
# it by value — patching dashboard.data.tasks.fetch_tasks alone would miss
# them all. ``dashboard.app`` is absent: its /healthz probe reads
# ``fetch_task_page`` (task 6321), and this sweep does not reach /healthz
# anyway. ``dashboard.api.escalations`` binds no task read at all (task 5596):
# its owner probe reads the snapshot unit's active rows and its cards go
# through task_lookup. ``dashboard.data.orchestrator`` binds none since task 5587.
_HUNG_BINDINGS: Mapping[str, str] = {
    # The Tasks tab's read travelled to the snapshot unit (task 5587);
    # active_tasks holds no fetch name at all now.
    _SNAPSHOT: 'fetch_tasks',
    # /merge-queue and /escalations read their rows' tasks one id at a time
    # (tasks 5595, 5596).
    _LOOKUP: 'fetch_task',
    # /prds' lineage index reads one projected get_tasks per root; the
    # index's background refresh is that binding's one caller.
    _LINEAGE: 'fetch_metadata_projection',
}


@dataclass(frozen=True)
class HungSeams:
    """What ``hung_mcp`` hung, and which seams each endpoint's own request must reach.

    ``reached`` is every ``(module_name, project_root)`` the hang stub was
    entered with, in order. ``must_reach`` maps each task-reading endpoint's
    path to the ``(module_name, project_root)`` seams its own request must
    reach; ``_HUNG_BINDINGS`` names the binding each module's seam hangs.
    """

    reached: list[tuple[str, str]]
    must_reach: Mapping[str, frozenset[tuple[str, str]]]


@pytest.fixture()
def hung_mcp(request, monkeypatch, tmp_path, client):
    """Hang every task-read binding; bound every budget; warm the timed lookup roots.

    Returns :class:`HungSeams` once every lookup root whose read would sit
    inside its lookup's deadline is cached, so no lookup deadline encloses a
    snapshot read and a caller can assert each endpoint reached the hang for
    its own root. Requests ``client`` so the configured root it seeds is the
    one the running app reads.
    """
    import dashboard.api.escalations as _esc
    from dashboard.data import (
        escalation_corpus,
        task_lineage,
        task_lookup,
        task_snapshot,
    )

    reached = _hang_task_reads(monkeypatch)

    # A snapshot read outlasts any lookup's whole deadline, so none may enclose one.
    monkeypatch.setattr(task_lookup, 'LOOKUP_BUDGET_SECONDS', _LOOKUP_BUDGET)
    monkeypatch.setattr(task_snapshot, 'PER_CALL_TIMEOUT', 2 * _LOOKUP_BUDGET)
    # The request waits on the lineage refresh, and the refresh on the hang,
    # for these and no longer.
    monkeypatch.setattr(task_lineage, 'LINEAGE_REQUEST_WAIT_SECONDS', _LINEAGE_BUDGET)
    monkeypatch.setattr(task_lineage, 'LINEAGE_READ_TIMEOUT', _LINEAGE_BUDGET)

    def _clear_caches() -> None:
        task_snapshot.clear_snapshot_cache()
        task_lineage.clear_lineage_state()
        task_lookup._lookup_cache_clear()
        escalation_corpus._corpus_cache_clear()
        _esc._analytics_memo_clear()

    # A warm entry would be served without ever reaching the hang.
    _clear_caches()
    # Leave no hang-stubbed entry behind for the next test, even if setup fails.
    request.addfinalizer(_clear_caches)

    probe_root = str(_seed_probe_root(tmp_path, monkeypatch))
    config_root = str(client.app.state.config.project_root)
    must_reach = {
        # /merge-queue looks up its rows' tasks in the configured root.
        '/api/v2/dashboard/merge-queue': frozenset({(_LOOKUP, config_root)}),
        # /escalations reads its root's unit itself, deadline-free, before its
        # lookup, so its request keeps that cold read against the hang.
        '/api/v2/dashboard/escalations': frozenset({
            (_SNAPSHOT, probe_root), (_LOOKUP, probe_root),
        }),
        # /prds reads the lineage index of every configured root; the client has one.
        '/api/v2/dashboard/prds': frozenset({(_LINEAGE, config_root)}),
    }
    monkeypatch.setattr(task_snapshot, 'SNAPSHOT_TTL_SECONDS', 3600.0)
    assert client.portal is not None, 'the warm-up runs inside a live TestClient'
    client.portal.call(
        _warm_snapshot_units,
        client.app.state.http_client,
        client.app.state.config,
        _timed_lookup_roots(must_reach),
    )
    # /merge-queue looks up only the ids its rows name, so it needs a row.
    request.addfinalizer(_seed_merge_attempt(client.app.state.config.project_root))

    return HungSeams(reached=reached, must_reach=must_reach)


def _hang_task_reads(monkeypatch) -> list[tuple[str, str]]:
    """Hang every binding in ``_HUNG_BINDINGS``; return the log of reaches it fills."""
    reached: list[tuple[str, str]] = []

    def _make_stub(module_name: str):
        # **kwargs absorbs each binding's own narrowing arguments — the
        # snapshot unit threads statuses/timeout/cached, the others do not —
        # so one stub can stand in for every call shape without being laxer
        # about the thing under test, which is that the call HANGS.
        async def _hang(client, config, project_root, *_args, **_kwargs):
            reached.append((module_name, str(project_root)))
            await asyncio.Event().wait()  # nothing ever sets it

        return _hang

    for module_name, binding in _HUNG_BINDINGS.items():
        monkeypatch.setattr(f'{module_name}.{binding}', _make_stub(module_name))
    return reached


def _seed_probe_root(tmp_path, monkeypatch) -> Path:
    """Make the probe root that a running orchestrator and the one escalation queue name.

    These preconditions make the target endpoints actually REACH their call
    site — otherwise the probe would pass vacuously.
    """
    import dashboard.api.escalations as _esc
    from dashboard.data import orchestrator
    from dashboard.data.escalation_corpus import QueueKind, QueueRef

    proj = tmp_path / 'probe_root'
    (proj / '.taskmaster').mkdir(parents=True)
    # The orchestrator stub stays even though that endpoint no longer fetches:
    # it is what makes the sweep exercise discovery's grouping path rather
    # than its empty one.
    monkeypatch.setattr(
        orchestrator,
        'find_running_orchestrators',
        lambda: [{
            'pid': 4788, 'prd': str(proj / 'prd.md'), 'config_path': None,
            'running': True, 'started': 'Aug27',
        }],
    )
    # /escalations looks up only the task ids its rows name, so its one queue
    # holds a real record naming one.
    escalation_dir = proj / 'data' / 'escalations'
    escalation_dir.mkdir(parents=True)
    (escalation_dir / 'esc-4788-1.json').write_text(json.dumps(Escalation(
        id='esc-4788-1', task_id='4788', agent_role='implementer',
        severity='blocking', category='infra_issue', summary='hung seam probe',
    ).to_dict()))
    monkeypatch.setattr(
        _esc,
        'corpus_queues',
        lambda config: (QueueRef(
            id=str(proj), label='probe', kind=QueueKind.ORCHESTRATOR,
            directory=escalation_dir, runs_db=proj / 'data' / 'orchestrator' / 'runs.db',
        ),),
    )
    return proj


def _timed_lookup_roots(must_reach: Mapping[str, frozenset[tuple[str, str]]]) -> frozenset[str]:
    """Each root a request must look up without itself reaching that root's snapshot read.

    Left cold, such a lookup would make the snapshot read inside its own
    deadline, so these roots are warmed before the sweep; a request that must
    reach the snapshot read keeps its root cold.
    """
    return frozenset(
        root
        for seams in must_reach.values()
        for module_name, root in seams
        if module_name == _LOOKUP and (_SNAPSHOT, root) not in seams
    )


def _describe_seam(seam: tuple[str, str]) -> str:
    module_name, root = seam
    return f'{module_name}.{_HUNG_BINDINGS[module_name]} for root {root}'


async def _warm_snapshot_units(http_client, config, roots: Collection[str]) -> None:
    """Cache each root's snapshot unit; failing, not hanging, if its read is unbounded."""
    from dashboard.data import task_snapshot
    from dashboard.data.utils import resolve_now

    now = resolve_now(None)
    bound = _WARM_UP_BUDGETS * task_snapshot.PER_CALL_TIMEOUT
    try:
        async with asyncio.timeout(bound):
            await asyncio.gather(*(
                task_snapshot.acquire_snapshot(http_client, config, root, now=now)
                for root in roots
            ))
    except TimeoutError:
        raise AssertionError(
            f'warming the snapshot units of {sorted(roots)} outlived {bound}s, '
            f'{_WARM_UP_BUDGETS}x the {task_snapshot.PER_CALL_TIMEOUT}s '
            'PER_CALL_TIMEOUT, against the hung fetch_tasks — the snapshot '
            'read is no longer bounded by its own budget'
        ) from None


def _seed_merge_attempt(project_root):
    """Put one titled-by-lookup merge_attempt in *project_root*'s runs.db.

    The root is the session-scoped one every ``client`` shares, so the return
    value undoes exactly what this wrote: the file if it created it, else the
    one row.
    """
    runs_db = project_root / 'data' / 'orchestrator' / 'runs.db'
    created = not runs_db.exists()
    runs_db.parent.mkdir(parents=True, exist_ok=True)
    attempted_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    with closing(sqlite3.connect(runs_db)) as conn:
        conn.execute(
            'CREATE TABLE IF NOT EXISTS events ('
            ' id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,'
            ' run_id TEXT NOT NULL, task_id TEXT, event_type TEXT NOT NULL,'
            " phase TEXT, role TEXT, data TEXT DEFAULT '{}', cost_usd REAL,"
            ' duration_ms INTEGER)'
        )
        row_id = conn.execute(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, phase,'
            ' data, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (attempted_at, 'run-4788', '4788', 'merge_attempt', 'merge',
             json.dumps({'outcome': 'done', 'branch': 'task/4788'}), 1200),
        ).lastrowid
        conn.commit()

    def _undo() -> None:
        if created:
            runs_db.unlink(missing_ok=True)
            return
        with closing(sqlite3.connect(runs_db)) as conn:
            conn.execute('DELETE FROM events WHERE id = ?', (row_id,))
            conn.commit()

    return _undo


def test_every_dashboard_endpoint_survives_a_hung_fetch_tasks(client, hung_mcp):
    """Every GET endpoint answers 200 while the fetch_tasks seam hangs.

    Only the LITERAL routes are swept. A templated one
    (``/api/v2/dashboard/task/{project}/T-{task_id}``) would be fetched as the
    literal brace string and answer 404/422 — a false failure about path
    construction, reported under a message about hung dependencies. A
    templated route needs its own probe that binds real parameter values; see
    :func:`test_templated_dashboard_routes_are_not_silently_unswept`, which
    fails loudly if one is ever added rather than letting it go unswept.
    """
    paths = dashboard_get_paths().literal

    # The sweep is derived, and may not silently shrink. 14 is the count at
    # the time this probe was written; a NEW endpoint raises it, and a route
    # that disappears must be noticed rather than quietly dropped.
    assert len(paths) >= 14, (
        f'only {len(paths)} GET routes discovered under /api/v2/dashboard/ '
        f'({paths}) — the sweep has shrunk below the 14 that existed when '
        'this probe was written; a check that quietly stops checking is '
        'indistinguishable from a passing one'
    )
    unswept = set(hung_mcp.must_reach) - set(paths)
    assert not unswept, (
        f'task-reading endpoint(s) {sorted(unswept)} '
        'are no longer swept, so their reach of the hang cannot be checked'
    )

    reached_by_path: dict[str, set[tuple[str, str]]] = {}
    for path in paths:
        before = len(hung_mcp.reached)
        # TestClient.get is SYNCHRONOUS and blocks for as long as the handler
        # does, so a regression here shows up as a hung or failed call — the
        # 19.8 h symptom in miniature.
        resp = client.get(path)
        assert resp.status_code == 200, (
            f'{path} returned {resp.status_code} while the fetch_tasks MCP '
            'seam was hung — a hung dependency must degrade this endpoint, '
            'never wedge or 500 it'
        )
        reached_by_path[path] = set(hung_mcp.reached[before:])

    # NON-VACUITY: each task-reading endpoint's own request must have waited on
    # every hung seam it names, for that seam's root, or that bound is unexercised.
    for path, seams in hung_mcp.must_reach.items():
        missed = sorted(seams - reached_by_path[path])
        assert not missed, (
            f'{path} never reached the hung {", ".join(map(_describe_seam, missed))} '
            f'(that request reached: {sorted(reached_by_path[path])}) — that read '
            'went unexercised, so the sweep proves nothing about its bound. Check '
            'the endpoint preconditions in the hung_mcp fixture, and that neither '
            'the warm-up nor an earlier request in the sweep caches a root this '
            'request must read cold.'
        )


def test_task_prose_route_survives_a_hung_fetch_task_prose(client, monkeypatch):
    """The templated prose route, probed with REAL parameter values.

    The sweep above cannot request ``/task/{project:path}/T-{task_id}`` verbatim, so
    this binds the configured root's own label and a real id, hangs the route
    module's ``fetch_task_prose`` binding, and requires a bounded answer.
    """
    import dashboard.api.task_prose as task_prose

    reached: list[tuple[str, int]] = []

    async def _hang(http_client, config, project_root, task_id):
        reached.append((str(project_root), task_id))
        await asyncio.Event().wait()  # nothing ever sets it

    monkeypatch.setattr(task_prose, 'fetch_task_prose', _hang)
    monkeypatch.setattr(task_prose, '_TASK_PROSE_BUDGET', _PROSE_BUDGET)
    root = client.app.state.config.project_root

    resp = client.get(f'/api/v2/dashboard/task/{root.name}/T-1')

    assert resp.status_code == 504, (
        f'the prose route returned {resp.status_code} while fetch_task_prose '
        'was hung — a hung dependency must degrade this endpoint, never wedge '
        'or 500 it'
    )
    # NON-VACUITY: a 504 that never reached the seam proves nothing.
    assert reached == [(str(root), 1)], (
        f'the hang was reached as {reached}, not exactly once with the '
        f'configured root {root} and task id 1 — the probe passed without '
        'exercising the seam'
    )


def test_templated_dashboard_routes_are_not_silently_unswept():
    """A templated route must be noticed, not quietly dropped from the sweep.

    The sweep requests only ``dashboard_get_paths().literal``, because it
    cannot request a templated path verbatim. That filter is the right call — but a silent
    filter is how coverage rots, so this test fails the moment one appears and
    says what to do about it. Extend the expected set below ONLY together with
    a dedicated probe that exercises the new route with real parameter values
    against a hung dependency, as
    :func:`test_task_prose_route_survives_a_hung_fetch_task_prose` does.
    """
    templated = list(dashboard_get_paths().templated)

    assert templated == ['/api/v2/dashboard/task/{project:path}/T-{task_id}'], (
        f'templated GET route(s) {templated} exist under /api/v2/dashboard/; '
        'the only one with a dedicated hung-dependency probe is the task-prose '
        'route (test_task_prose_route_survives_a_hung_fetch_task_prose). The '
        'sweep requests every path verbatim and would fetch the literal brace '
        'string, so a new templated route needs its own probe that binds real '
        'parameter values, listed here together with it.'
    )
