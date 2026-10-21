"""The synthetic fleet the payload budget sweep serves every budgeted request over.

``test_payload_budget_sweep.py`` asks the REAL app for each request in
``payload_budgets.json`` and compares the wire bytes with the request's budget.
That comparison means something only if the app is serving a population the
size of the one the budget was measured against, so this module builds one:
:data:`CENSUS` describes it, :class:`FleetMcp` answers every MCP read the
handlers make from it, and :func:`open_fleet_client` points the real app at
it. Only the seams are faked: the ``mcp_tool_call`` binding in every module
that imported it, and the ``ps`` scan that discovers running orchestrators.
Every shaper above them is the production one, so a field added to a row
shows up here as bytes.

Content is deterministic ASCII filler at the census lengths. Two runs serve
byte-identical rows; only the instants the app stamps itself vary, and those
have a fixed width.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import pytest
import yaml
from _canned_mcp import ByRoot, CannedMCP
from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS
from fastapi.testclient import TestClient
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot
from shared.task_statuses import TERMINAL, TaskStatus

import dashboard.api.escalations as api_escalations
import dashboard.app
from dashboard.config import DashboardConfig
from dashboard.data import active_tasks, escalation_corpus, task_lookup, task_snapshot
from dashboard.data.memory import mcp_tool_call
from dashboard.data.orchestrator import find_running_orchestrators
from dashboard.data.scheduler import _scheduler_cache_clear


@dataclass(frozen=True)
class RootCensus:
    """One project root's share of the fleet.

    ``active`` and ``terminal`` count rows per status. ``modules`` is the number
    of distinct lock modules the active rows' footprints cover, and
    ``files_per_row`` how many each footprint names. ``deps_per_100_rows`` is
    the dependency density. ``parked``, ``skipped`` and ``pinned`` count the
    pending tasks the scheduler parks, skips and pins; ``recovery_events``
    counts the sweep events its last hour holds.
    """

    label: str
    active: Mapping[TaskStatus, int]
    terminal: Mapping[TaskStatus, int]
    files_per_row: int = 0
    modules: int = 0
    deps_per_100_rows: int = 0
    orchestrator_running: bool = True
    parked: int = 0
    skipped: int = 0
    pinned: int = 0
    recovery_events: int = 0


@dataclass(frozen=True)
class FleetCensus:
    """The population the payload budgets were measured against.

    PROVENANCE. The budgets in ``payload_budgets.json`` were set on 2026-10-15
    from live measurements (task 5824, pre-1). The ACTIVE row counts here are
    that day's: 4,951 rows, pending 4,304, deferred 527, in-progress 88,
    blocked 30, merge-deferred 2, and 8,278 scheduler modules. The pre-1
    iteration log did not survive, so the per-root split, the terminal totals
    and the per-row shapes were re-measured against the live dashboard on
    2026-10-21 (5,335 active rows):

    * titles average 120 characters, ``meta_files`` 5.5 paths of 43;
    * 1.12 dependencies per row, 92% of them on another active row (titled);
    * 36% of rows name a PRD of 45 characters; 1.1% carry an external dep;
    * 480 held modules, 9 parked tasks, 8 skipped, 17 pinned, 7 recovery
      events in the last hour;
    * the 2026-09-23 per-field accounting put a task description at about
      2,600 characters (``description`` is not shaped into a task row, but
      escalation cards embed it).

    Re-grow this census, rather than loosening a budget, whenever a budget
    raised for real growth would otherwise trip the sweep's slack rule.
    """

    roots: tuple[RootCensus, ...]
    title_len: int
    description_len: int
    file_path_len: int
    prd_path_len: int
    prd_rows_per_25: int
    external_dep_every: int
    untitled_dep_every: int
    claimant_len: int
    lock_depth: int
    skip_events_per_task: int


CENSUS = FleetCensus(
    roots=(
        RootCensus(
            'dark-factory',
            {
                TaskStatus.PENDING: 1465, TaskStatus.DEFERRED: 293, TaskStatus.IN_PROGRESS: 24,
                TaskStatus.BLOCKED: 6, TaskStatus.MERGE_DEFERRED: 2,
            },
            {TaskStatus.DONE: 4929, TaskStatus.CANCELLED: 396},
            files_per_row=5, modules=2792, deps_per_100_rows=25,
            parked=5, skipped=4, pinned=17, recovery_events=1,
        ),
        RootCensus(
            'reify',
            {TaskStatus.PENDING: 2095, TaskStatus.DEFERRED: 223, TaskStatus.IN_PROGRESS: 23},
            {TaskStatus.DONE: 6076, TaskStatus.CANCELLED: 772},
            files_per_row=5, modules=3775, deps_per_100_rows=131, parked=4, skipped=4,
        ),
        RootCensus(
            'solar-challenge-platform',
            {TaskStatus.PENDING: 499, TaskStatus.IN_PROGRESS: 18, TaskStatus.BLOCKED: 9},
            {TaskStatus.DONE: 715, TaskStatus.CANCELLED: 94},
            files_per_row=8, modules=1255, deps_per_100_rows=247,
        ),
        RootCensus(
            'solar-challenge',
            {TaskStatus.PENDING: 200, TaskStatus.IN_PROGRESS: 20, TaskStatus.BLOCKED: 3},
            {TaskStatus.DONE: 658, TaskStatus.CANCELLED: 133},
            files_per_row=8, modules=354, deps_per_100_rows=270, recovery_events=6,
        ),
        RootCensus(
            'scoping-intake',
            {TaskStatus.PENDING: 45, TaskStatus.IN_PROGRESS: 3},
            {TaskStatus.DONE: 50, TaskStatus.CANCELLED: 3},
            files_per_row=5, modules=88, deps_per_100_rows=181,
        ),
        RootCensus(
            'know-live',
            {TaskStatus.DEFERRED: 10, TaskStatus.BLOCKED: 6},
            {TaskStatus.DONE: 594, TaskStatus.CANCELLED: 2},
            files_per_row=2, modules=10, deps_per_100_rows=214,
        ),
        RootCensus(
            'pump-web-ui', {TaskStatus.BLOCKED: 5}, {TaskStatus.DONE: 14, TaskStatus.CANCELLED: 1},
            files_per_row=1, modules=2,
        ),
        RootCensus(
            'autopilot-video',
            {TaskStatus.DEFERRED: 1, TaskStatus.BLOCKED: 1},
            {TaskStatus.DONE: 615, TaskStatus.CANCELLED: 37},
            files_per_row=1, modules=2,
        ),
        RootCensus('autotrade', {}, {}, orchestrator_running=False),
        RootCensus('mission-control', {}, {}, orchestrator_running=False),
    ),
    title_len=120,
    description_len=2600,
    file_path_len=43,
    prd_path_len=45,
    prd_rows_per_25=9,
    external_dep_every=90,
    untitled_dep_every=12,
    claimant_len=40,
    lock_depth=4,
    skip_events_per_task=10,
)

TERMINAL_WINDOW_ROOT = 'dark-factory'
"""The root the budgeted ``?terminal=`` request names; it must be in the census."""

_PRIORITY_CYCLE = ('low', 'low', 'medium', 'high')
"""Effective priorities in the live mix: about half low, a quarter each medium and high."""

_ESCALATION_HOST = 'payload-fleet.invalid'
_ESCALATION_BASE_PORT = 47000
_TASK_TOOLS = frozenset({'get_tasks', 'get_statuses', 'get_task'})

_KNOWN_MCP_IMPORTERS = frozenset({
    'dashboard.data.memory',
    'dashboard.data.tasks',
    'dashboard.data.scheduler',
    'dashboard.data.merge_queue',
    'dashboard.data.escalations',
    'dashboard.data.task_runtime',
    'dashboard.data.metrics',
    'dashboard.data.merge_halt',
})
_KNOWN_PS_IMPORTERS = frozenset({
    'dashboard.data.orchestrator',
    'dashboard.data.metrics',
    'dashboard.data.burndown',
})


def _text(seed: str, length: int) -> str:
    """*seed*, padded with ASCII filler to exactly *length* characters."""
    if len(seed) >= length:
        raise ValueError(f'seed {seed!r} does not fit in {length} characters')
    return seed + ' ' + 'x' * (length - len(seed) - 1)


def _module_path(index: int, length: int) -> str:
    """The *index*-th lock module of a root's pool: three components, *length* characters."""
    head = f'src/m{index:05d}/'
    return head + 'f' * (length - len(head) - len('.py')) + '.py'


def _spread(count: int, total: int) -> set[int]:
    """*count* distinct indices in ``range(total)``, evenly spaced."""
    return {(2 * j + 1) * total // (2 * count) for j in range(count)}


def _share(index: int, per_100: int) -> int:
    """How many of a ``per_100``-per-hundred quantity row *index* carries; sums exactly."""
    return (index + 1) * per_100 // 100 - index * per_100 // 100


@dataclass(frozen=True)
class _Tree:
    """One root's generated task tree and the scheduler state over it."""

    census: RootCensus
    rows: tuple[dict[str, Any], ...]
    scheduler_state: Mapping[str, Any]
    runtime: Mapping[str, Any]

    def status_map(self) -> dict[int, str]:
        return {int(row['id']): row['status'] for row in self.rows}

    def ids(self, status: TaskStatus) -> list[int]:
        return [int(row['id']) for row in self.rows if row['status'] == status]


class _TreeBuilder:
    """Builds one root's :class:`_Tree` from the census, at the instant *now*.

    Terminal tasks take the lowest ids and active tasks the ones above them,
    as in a real tree, so the newest terminal window and the active rows'
    dependencies on finished work both land on real ids.
    """

    def __init__(self, fleet: FleetCensus, root: RootCensus, now: datetime) -> None:
        self._fleet = fleet
        self._root = root
        self._now = now
        self._pool = [_module_path(i, fleet.file_path_len) for i in range(root.modules)]
        self._terminal_total = sum(root.terminal.values())
        self._description = _text('fleet task description', fleet.description_len)
        self._prd = _text('plans/fleet', fleet.prd_path_len - len('.md')) + '.md'
        self._dep_count = 0

    def build(self) -> _Tree:
        rows = [*self._terminal_rows(), *self._active_rows()]
        return _Tree(
            census=self._root,
            rows=tuple(rows),
            scheduler_state=self._scheduler_state(rows),
            runtime=self._runtime(rows),
        )

    def _terminal_rows(self) -> Iterator[dict[str, Any]]:
        cancelled = _spread(self._root.terminal.get(TaskStatus.CANCELLED, 0), self._terminal_total)
        for index in range(self._terminal_total):
            status = TaskStatus.CANCELLED if index in cancelled else TaskStatus.DONE
            yield self._row(index + 1, status, files=self._files(index), dependencies=[])

    def _active_rows(self) -> Iterator[dict[str, Any]]:
        statuses = [status for status, n in self._root.active.items() for _ in range(n)]
        for index, status in enumerate(statuses):
            task_id = self._terminal_total + 1 + index
            yield self._row(
                task_id, status,
                files=self._files(index),
                dependencies=self._dependencies(index),
                prd=index % 25 < self._fleet.prd_rows_per_25,
                external=index % self._fleet.external_dep_every == 0,
            )

    def _row(
        self, task_id: int, status: TaskStatus, *,
        files: list[str], dependencies: list[int], prd: bool = False, external: bool = False,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {'files': files}
        if prd:
            metadata['prd'] = self._prd
        if external:
            metadata['external_deps'] = [f'fleet-external:{task_id}']
        row: dict[str, Any] = {
            'id': str(task_id),
            'title': _text(f'{self._root.label} task {task_id}', self._fleet.title_len),
            'description': self._description,
            'status': status.value,
            'dependencies': dependencies,
            'metadata': metadata,
            'updatedAt': self._now.isoformat(),
        }
        if status is TaskStatus.IN_PROGRESS:
            row['claimant_run_id'] = _text(f'run-fleet/{task_id}', self._fleet.claimant_len)
            row['heartbeat_at'] = self._now.isoformat()
        return row

    def _files(self, index: int) -> list[str]:
        if not self._pool:
            return []
        per_row = self._root.files_per_row
        return [self._pool[(index * per_row + j) % len(self._pool)] for j in range(per_row)]

    def _dependencies(self, index: int) -> list[int]:
        """Row *index*'s deps: mostly earlier active rows, every Nth a finished task."""
        deps: list[int] = []
        for j in range(_share(index, self._root.deps_per_100_rows)):
            self._dep_count += 1
            on_active = index - j - 1
            if on_active >= 0 and self._dep_count % self._fleet.untitled_dep_every:
                deps.append(self._terminal_total + 1 + on_active)
            elif self._terminal_total:
                deps.append(self._terminal_total - self._dep_count % self._terminal_total)
        return deps

    def _scheduler_state(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        active = [row for row in rows if row['status'] not in TERMINAL]
        pending = [row['id'] for row in active if row['status'] == TaskStatus.PENDING]
        root = self._root
        parked = pending[:root.parked]
        skipped = pending[root.parked:root.parked + root.skipped]
        pinned = pending[root.parked + root.skipped:root.parked + root.skipped + root.pinned]
        files_by_id = {row['id']: row['metadata']['files'] for row in active}
        installed_at = (self._now - timedelta(hours=2)).isoformat()
        return {
            'lock_depth': self._fleet.lock_depth,
            'current_holders': {
                module: row['id']
                for row in active if row['status'] == TaskStatus.IN_PROGRESS
                for module in files_by_id[row['id']]
            },
            'parks': {tid: {'modules': files_by_id[tid], 'installed_at': installed_at} for tid in parked},
            'park_stacks': {
                module: [{
                    'owner': tid, 'rank': 1, 'shadowed': False, 'installed_at': installed_at,
                    'source': 'fairness', 'tier': 'high',
                }]
                for tid in parked for module in files_by_id[tid]
            },
            'skip_counts': {tid: self._fleet.skip_events_per_task for tid in skipped},
            'effective_priorities': {
                row['id']: _PRIORITY_CYCLE[i % len(_PRIORITY_CYCLE)] for i, row in enumerate(active)
            },
            'overrides': {
                tid: {'pinned': True, 'reserve_now': False, 'boost_tier': None, 'ttl_until': None}
                for tid in pinned
            },
            'pin_queue': [{'task_id': tid, 'order': order} for order, tid in enumerate(pinned, start=1)],
            'is_paused': False,
        }

    def _runtime(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        entries = [
            TaskRuntimeEntry(
                task_id=int(row['id']), has_worktree=True, loops=2, attempts=0,
                started=self._now.isoformat(), lane='_lane-1', phase='EXECUTE', lane_state='assigned',
            )
            for row in rows if row['status'] == TaskStatus.IN_PROGRESS
        ]
        return TaskRuntimeSnapshot(tasks=entries).model_dump(mode='json')


class FleetMcp:
    """A stand-in for ``mcp_tool_call`` that answers every read from the fleet.

    Task reads go to one :class:`CannedMCP` per root through :class:`ByRoot`,
    so paging, status narrowing and lookups by id are the faithful ones. Every
    other tool is answered from the root's generated tree. A tool the fleet
    does not answer, or a root it does not hold, RAISES rather than answering
    ``{}``: an empty answer would let a handler serve a near-empty payload that
    only the slack rule would ever notice.
    """

    def __init__(
        self,
        fleet: FleetCensus,
        escalation_urls: Mapping[str, str],
        *,
        now: datetime,
    ) -> None:
        self._trees = {root.label: _TreeBuilder(fleet, root, now).build() for root in fleet.roots}
        self._skip_events_per_task = fleet.skip_events_per_task
        self._label_by_mcp_endpoint = {url: label for label, url in escalation_urls.items()}
        self._tasks = ByRoot({
            label: CannedMCP(tree.rows, tree.status_map(), status_page_size=2000)
            for label, tree in self._trees.items()
        })
        self._handlers = {
            'get_task_runtime_state': self._runtime_state,
            'get_scheduler_state': self._scheduler_state,
            'get_scheduler_events': self._scheduler_events,
            'get_external_statuses': self._external_statuses,
        }

    async def __call__(self, client, url, tool, args, **kwargs):
        if tool in _TASK_TOOLS:
            return await self._tasks(client, url, tool, args, **kwargs)
        handler = self._handlers.get(tool)
        if handler is None:
            raise AssertionError(
                f'the payload fleet answers no {tool!r}; answer it from the census in '
                'dashboard/tests/_payload_fleet.py::FleetMcp',
            )
        return handler(url, args)

    def _tree(self, label: str) -> _Tree:
        try:
            return self._trees[label]
        except KeyError:
            raise LookupError(f'the payload fleet holds no root labelled {label!r}') from None

    def _root_tree(self, args: Mapping[str, Any]) -> _Tree:
        return self._tree(Path(args['project_root']).name)

    def _runtime_state(self, base_url: str, _args: Mapping[str, Any]) -> Mapping[str, Any]:
        # mcp_tool_call is handed a BASE url and posts to its /mcp endpoint,
        # which is the form the config's escalation URLs are written in.
        endpoint = f'{base_url}/mcp'
        label = self._label_by_mcp_endpoint.get(endpoint)
        if label is None:
            raise LookupError(f'the payload fleet has no orchestrator at {endpoint!r}')
        return self._tree(label).runtime

    def _scheduler_state(self, _url: str, args: Mapping[str, Any]) -> dict[str, Any]:
        return dict(self._root_tree(args).scheduler_state)

    def _scheduler_events(self, _url: str, args: Mapping[str, Any]) -> list[dict[str, Any]]:
        tree = self._root_tree(args)
        since = datetime.fromisoformat(args['since'])
        if 'task_skipped' in args['event_types']:
            events = self._skip_events(tree, since)
        else:
            events = self._recovery_events(tree, since)
        return events[:args['limit']]

    def _skip_events(self, tree: _Tree, since: datetime) -> list[dict[str, Any]]:
        skipped = tree.scheduler_state['skip_counts']
        return [
            _event(since + timedelta(minutes=5 * n, seconds=30), tid, 'task_skipped', {})
            for tid in skipped for n in range(self._skip_events_per_task)
        ]

    def _recovery_events(self, tree: _Tree, since: datetime) -> list[dict[str, Any]]:
        blocked = tree.ids(TaskStatus.BLOCKED) or tree.ids(TaskStatus.IN_PROGRESS)
        return [
            _event(
                since + timedelta(minutes=7 * n, seconds=45), str(blocked[n % len(blocked)]),
                'recovery_vetoed',
                {'site': 'reconcile_sweep', 'reason': 'escalation_pinned', 'streak': 3},
            )
            for n in range(tree.census.recovery_events)
        ]

    def _external_statuses(self, _url: str, args: Mapping[str, Any]) -> dict[str, str]:
        return dict.fromkeys(args['deps'], 'done')


def _event(at: datetime, task_id: str, event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'timestamp': at.isoformat(), 'run_id': 'run-payload-fleet', 'task_id': task_id,
        'event_type': event_type, 'data': {'task_id': task_id, **data},
    }


def _write_roots(workdir: Path, fleet: FleetCensus) -> list[Path]:
    """One project dir per census root, each with ``.taskmaster/`` and its orchestrator config."""
    roots: list[Path] = []
    for index, root in enumerate(fleet.roots):
        path = workdir / root.label
        (path / '.taskmaster').mkdir(parents=True)
        (path / 'dark-factory-orchestrator.yaml').write_text(yaml.safe_dump({
            'project_root': str(path),
            'escalation': {'host': _ESCALATION_HOST, 'port': _ESCALATION_BASE_PORT + index},
        }))
        roots.append(path.resolve())
    return roots


def _running_orchestrators(fleet: FleetCensus, roots: Sequence[Path]) -> list[dict[str, Any]]:
    """What the ``ps`` scan would report for the fleet: one process per running root."""
    return [
        {
            'pid': 60000 + index, 'prd': None,
            'config_path': str(path / 'dark-factory-orchestrator.yaml'),
            'running': True, 'started': 'Oct15',
        }
        for index, (root, path) in enumerate(zip(fleet.roots, roots, strict=True))
        if root.orchestrator_running
    ]


def _bindings_of(target: object, name: str, known: frozenset[str]) -> list[ModuleType]:
    """Every ``dashboard.*`` module binding *target* as *name*.

    Derived from ``sys.modules`` so a module that newly imports the seam by
    value is faked without anyone listing it. *known* must all be found, so a
    scan that silently finds nothing cannot pass.
    """
    found = [
        module for module_name, module in sorted(sys.modules.items())
        if module_name.startswith('dashboard.') and getattr(module, name, None) is target
    ]
    missing = known - {module.__name__ for module in found}
    assert not missing, f'the {name} binding scan missed {sorted(missing)}; the fleet would leak'
    return found


def _clear_caches() -> None:
    """Every cache a served payload could come from, so no other test's entry is served here."""
    task_snapshot._snapshot_cache_clear()
    task_lookup._lookup_cache_clear()
    escalation_corpus._corpus_cache_clear()
    api_escalations._analytics_memo_clear()
    _scheduler_cache_clear()
    active_tasks._reset_root_rotation()


@contextmanager
def open_fleet_client(fleet: FleetCensus = CENSUS) -> Iterator[TestClient]:
    """The real app, configured over the fleet's roots and answered from the fleet.

    The config is installed inside the ``TestClient`` context, as
    ``_boundary_payloads._canned_roots`` does, so escalation URLs and every
    data dir come from the REAL ``DashboardConfig`` derivation over the roots.
    """
    if TERMINAL_WINDOW_ROOT not in {root.label for root in fleet.roots}:
        raise ValueError(f'the fleet must hold the {TERMINAL_WINDOW_ROOT!r} root the budget table names')
    app = dashboard.app.app
    with TemporaryDirectory(prefix='payload-fleet-') as workdir:
        roots = _write_roots(Path(workdir), fleet)
        config = DashboardConfig(
            project_root=roots[0],
            known_project_roots=roots[1:],
            fused_memory_urls=list(HERMETIC_FUSED_MEMORY_URLS),
        )
        substrate = FleetMcp(fleet, config.escalation_urls, now=datetime.now(UTC).replace(microsecond=0))
        processes = _running_orchestrators(fleet, roots)
        with pytest.MonkeyPatch.context() as mp:
            for module in _bindings_of(mcp_tool_call, 'mcp_tool_call', _KNOWN_MCP_IMPORTERS):
                mp.setattr(module, 'mcp_tool_call', substrate)
            for module in _bindings_of(
                find_running_orchestrators, 'find_running_orchestrators', _KNOWN_PS_IMPORTERS,
            ):
                mp.setattr(module, 'find_running_orchestrators', lambda: [dict(p) for p in processes])
            _clear_caches()
            try:
                with TestClient(app) as client:
                    app.state.config = config
                    yield client
            finally:
                _clear_caches()
