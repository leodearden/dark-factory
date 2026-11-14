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

import asyncio
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any

import pytest
import yaml
from _canned_mcp import ByRoot, CannedMCP
from _dashboard_helpers import HERMETIC_FUSED_MEMORY_URLS, build_dual_escalation_tree
from _payload_fleet_history import (
    BurndownShape,
    MemoryOpsShape,
    MemoryShape,
    ReconShape,
    seed_burndown_history,
    seed_metrics_history,
    seed_reconciliation,
    seed_write_journal,
)
from _payload_fleet_seed import (
    CostShape,
    EscalationShape,
    seed_account_events,
    seed_invocations,
    seed_merge_events,
    seed_pending_escalations,
    seed_resolved_history,
    seed_task_results,
    text,
)
from _quality_rows import NO_COST, NO_OUTCOMES, history, write_quality_db
from _served_reads import forget_served_reads
from _tasks_db_fixture import task_row, write_tasks_db
from fastapi.testclient import TestClient
from shared.project_id import normalize_project_id, project_id_for_root
from shared.quality_series.git_lines import ModuleLines
from shared.quality_series.landings import GitCrossCheck
from shared.task_runtime_state import TaskRuntimeEntry, TaskRuntimeSnapshot
from shared.task_statuses import TERMINAL, TaskStatus

import dashboard.app
from dashboard.config import DashboardConfig
from dashboard.data import task_lookup
from dashboard.data.escalation_corpus import corpus_queues, reconciliation_queue, serve_corpus
from dashboard.data.memory import mcp_tool_call
from dashboard.data.orchestrator import find_running_orchestrators
from dashboard.data.quality_store import QualitySample
from dashboard.data.scheduler import _scheduler_cache_clear


@dataclass(frozen=True)
class RootCensus:
    """One project root's share of the fleet.

    ``active`` and ``terminal`` count rows per status. ``modules`` is the number
    of distinct lock modules the active rows' footprints cover, and
    ``files_per_row`` how many each footprint names. ``deps_per_100_rows`` is
    the dependency density. ``parked``, ``skipped`` and ``pinned`` count the
    pending tasks the scheduler parks, skips and pins; ``recovery_events``
    counts the sweep events its last hour holds. ``prds`` is how many distinct
    PRDs the root's PRD-naming rows spread across.
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
    pending_escalations: int = 0
    resolved_escalations: int = 0
    completed_hours: int = 0
    claimed_terminal: int = 0
    merges: int = 0
    trains: int = 0
    prds: int = 1


@dataclass(frozen=True)
class QualityShape:
    """The quality sampler's store for the budgeted root: its landings and the modules they touch."""

    landings: int
    modules: int


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

    The sampled and billed histories were measured live on 2026-10-21, and
    the fleet serves each within a few percent of it:

    * costs (110,596 bytes at ``window=all``): spend on 189 of the 3,650
      days, 23 (model, role) pairs, 9 accounts, the 200-event cap full;
    * burndown (989,413 at ``window=90d``): the store reaches back 31 days
      but holds gap rows, so a root's series averages 1,328 MEASURED rows;
      the payload scales with that count, not the span;
    * recon (35,216): 50 runs listed, 24 writing agents, 400 buffered events;
    * memory (41,065): 173 memory projects, 18 WAL stores, a day of samples;
    * curator (17,057): no pending tickets, every account capped all day;
    * memory-graphs (1,722): 17 operations hourly; the journal's counts are
      five digits a bucket live, a few rows here, a difference of bytes.

    The pending escalations were re-measured live on 2026-11-14, when main's
    cards had shrunk /escalations to 713,606 bytes: 138 pending, 51 of them in
    the reconciliation queue (the dual tree adds five to the 46 seeded here).

    The quality store was measured live on 2026-11-14 (3,072,421 bytes at
    ``?project=dark-factory&window=all``): 3,736 landings, each a point on
    14 metric series, across 338 modules. Only that root is budgeted.

    The PRD population was measured live on 2026-11-14 (590,357 bytes at
    /prds, every PRD-holding root fresh): dark-factory 119 PRDs, reify 110,
    solar-challenge-platform 37, solar-challenge 23, scoping-intake 2. The
    fleet spreads each root's PRD-naming rows across that many PRDs and
    serves about 481,000 bytes; it seeds no blocker or followup links.

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
    metadata_extra_len: int
    escalation: EscalationShape
    reconciliation_escalations: int
    merge_history_days: int
    live_queue_depth: int
    costs: CostShape
    burndown: BurndownShape
    memory: MemoryShape
    recon: ReconShape
    memory_ops: MemoryOpsShape
    quality: QualityShape


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
            pending_escalations=12, resolved_escalations=1968,
            completed_hours=1555, claimed_terminal=40, merges=600, trains=50, prds=119,
        ),
        RootCensus(
            'reify',
            {TaskStatus.PENDING: 2095, TaskStatus.DEFERRED: 223, TaskStatus.IN_PROGRESS: 23},
            {TaskStatus.DONE: 6076, TaskStatus.CANCELLED: 772},
            files_per_row=5, modules=3775, deps_per_100_rows=131, parked=4, skipped=4,
            pending_escalations=4, resolved_escalations=8658,
            completed_hours=2123, claimed_terminal=37, merges=600, trains=4, prds=110,
        ),
        RootCensus(
            'solar-challenge-platform',
            {TaskStatus.PENDING: 499, TaskStatus.IN_PROGRESS: 18, TaskStatus.BLOCKED: 9},
            {TaskStatus.DONE: 715, TaskStatus.CANCELLED: 94},
            files_per_row=8, modules=1255, deps_per_100_rows=247,
            pending_escalations=31, resolved_escalations=1380,
            completed_hours=529, merges=400, prds=37,
        ),
        RootCensus(
            'solar-challenge',
            {TaskStatus.PENDING: 200, TaskStatus.IN_PROGRESS: 20, TaskStatus.BLOCKED: 3},
            {TaskStatus.DONE: 658, TaskStatus.CANCELLED: 133},
            files_per_row=8, modules=354, deps_per_100_rows=270, recovery_events=6,
            pending_escalations=8, resolved_escalations=633,
            completed_hours=397, merges=400, prds=23,
        ),
        RootCensus(
            'scoping-intake',
            {TaskStatus.PENDING: 45, TaskStatus.IN_PROGRESS: 3},
            {TaskStatus.DONE: 50, TaskStatus.CANCELLED: 3},
            files_per_row=5, modules=88, deps_per_100_rows=181,
            pending_escalations=0, resolved_escalations=67,
            completed_hours=32, merges=51, prds=2,
        ),
        RootCensus(
            'know-live',
            {TaskStatus.DEFERRED: 10, TaskStatus.BLOCKED: 6},
            {TaskStatus.DONE: 594, TaskStatus.CANCELLED: 2},
            files_per_row=2, modules=10, deps_per_100_rows=214,
            pending_escalations=14, resolved_escalations=13,
            completed_hours=113, merges=300,
        ),
        RootCensus(
            'pump-web-ui', {TaskStatus.BLOCKED: 5}, {TaskStatus.DONE: 14, TaskStatus.CANCELLED: 1},
            files_per_row=1, modules=2,
            pending_escalations=11,
            completed_hours=13, merges=15,
        ),
        RootCensus(
            'autopilot-video',
            {TaskStatus.DEFERRED: 1, TaskStatus.BLOCKED: 1},
            {TaskStatus.DONE: 615, TaskStatus.CANCELLED: 37},
            files_per_row=1, modules=2,
            pending_escalations=4, resolved_escalations=7,
            completed_hours=58, merges=250,
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
    metadata_extra_len=3000,
    escalation=EscalationShape(
        detail_len=1950, summary_len=150, triage_note_len=250, option_len=110, history_days=30,
    ),
    reconciliation_escalations=46,
    merge_history_days=196,
    live_queue_depth=5,
    costs=CostShape(
        spend_days=189,
        invocations_per_day=3,
        model_roles=(
            ('opus', 'architect'), ('opus', 'implementer'), ('opus', 'reviewer_comprehensive'),
            ('opus', 'debugger'), ('opus', 'steward'), ('opus', 'merger'), ('opus', 'deep_reviewer'),
            ('opus', 'escalation-watcher-auto'), ('sonnet', 'implementer'), ('sonnet', 'debugger'),
            ('sonnet', 'escalation-watcher-auto'), ('sonnet', 'simple_task'), ('sonnet', 'unblock_auto'),
            ('sonnet', 'resettled_adjudicator'), ('sonnet', 'module_tagger'),
            ('sonnet', 'reviewer_test_analyst'), ('sonnet', 'reviewer_robustness'),
            ('sonnet', 'reviewer_architect_reviewer'), ('sonnet', 'reviewer_reuse_auditor'),
            ('sonnet', 'reviewer_performance'), ('haiku', 'module_tagger'),
            ('claude-fable-5-1', 'merger'), ('claude-fable-5-1', 'steward'),
        ),
        curator_roles=('task_curator', 'task_curator_batch'),
        accounts=('max-a', 'max-b', 'max-c', 'max-d', 'max-e', 'max-f', 'max-g', 'max-h', 'default'),
        account_events=30,
        cap_reason_len=80,
    ),
    burndown=BurndownShape(samples_per_root=1330, concurrency_cap=24),
    memory=MemoryShape(
        projects=173, project_id_len=21, graphiti_nodes=132_790, mem0_memories=85_425,
        wal_stores=(
            'task_backend', 'recon_journal', 'event_buffer', 'ticket_store', 'recon_ledger',
            'write_journal', 'durable_queue', 'planned_episode_registry',
        ),
    ),
    recon=ReconShape(runs=200, buffered_events=400, agents=24),
    memory_ops=MemoryOpsShape(operations=(
        ('get_task', 'read'), ('get_statuses', 'read'), ('get_tasks', 'read'),
        ('get_external_statuses', 'read'), ('update_task', 'write'), ('search', 'read'),
        ('add_memory', 'write'), ('set_task_status', 'write'), ('get_entity', 'read'),
        ('add_dependency', 'write'), ('task_created', 'write'), ('delete_memory', 'write'),
        ('add_system_record', 'write'), ('find_tasks_by_metadata', 'read'), ('task_combined', 'write'),
        ('update_edge', 'write'), ('remove_dependency', 'write'),
    )),
    quality=QualityShape(landings=3736, modules=338),
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
        self._description = text('fleet task description', fleet.description_len)
        self._prds = [
            text(f'plans/fleet-{i}', fleet.prd_path_len - len('.md')) + '.md' for i in range(root.prds)
        ]
        self._notes = text('fleet task notes', fleet.metadata_extra_len)
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
                prd=self._prds[index % len(self._prds)] if index % 25 < self._fleet.prd_rows_per_25 else None,
                external=index % self._fleet.external_dep_every == 0,
            )

    def _row(
        self, task_id: int, status: TaskStatus, *,
        files: list[str], dependencies: list[int], prd: str | None = None, external: bool = False,
    ) -> dict[str, Any]:
        # 'notes' stands for every metadata key no row shaper reads; escalation
        # cards embed the whole metadata, so its size is theirs.
        metadata: dict[str, Any] = {'files': files, 'notes': self._notes}
        if prd:
            metadata['prd'] = prd
        if external:
            metadata['external_deps'] = [f'fleet-external:{task_id}']
        row: dict[str, Any] = {
            'id': str(task_id),
            'title': text(f'{self._root.label} task {task_id}', self._fleet.title_len),
            'description': self._description,
            'status': status.value,
            'dependencies': dependencies,
            'metadata': metadata,
            'updatedAt': self._now.isoformat(),
        }
        if status is TaskStatus.IN_PROGRESS:
            row['claimant_run_id'] = text(f'run-fleet/{task_id}', self._fleet.claimant_len)
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
    other tool is answered from the root's generated tree, or from
    :class:`_ServerStatus` for fused-memory's reports of itself. A tool the
    fleet does not answer, or a root it does not hold, RAISES rather than
    answering ``{}``: an empty answer would let a handler serve a near-empty
    payload that only the slack rule would ever notice. A fan-out that
    swallows the raise as an offline server is still caught: every refused
    tool is kept, and :func:`open_fleet_client` fails on any.
    """

    def __init__(
        self,
        fleet: FleetCensus,
        escalation_urls: Mapping[str, str],
        fused_memory_urls: Sequence[str],
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
        server = _ServerStatus(fleet, fused_memory_urls, now=now)
        self._handlers = {
            'get_task_runtime_state': self._runtime_state,
            'get_scheduler_state': self._scheduler_state,
            'get_scheduler_events': self._scheduler_events,
            'get_external_statuses': self._external_statuses,
            'get_pending_escalations': self._pending_escalations,
            'get_merge_queue': self._merge_queue,
            'get_merge_halt_status': self._merge_halt_status,
            'list_tickets': self._pending_tickets,
            'get_status': server.status,
            'get_queue_stats': server.queue_stats,
            'get_wal_status': server.wal_status,
            'get_curator_state': server.curator_state,
        }
        self._live_queue_depth = fleet.live_queue_depth
        self._refused: list[str] = []

    async def __call__(self, client, url, tool, args, **kwargs):
        if tool in _TASK_TOOLS:
            return await self._tasks(client, url, tool, args, **kwargs)
        handler = self._handlers.get(tool)
        if handler is None:
            self._refused.append(tool)
            raise AssertionError(
                f'the payload fleet answers no {tool!r}; answer it from the census in '
                'dashboard/tests/_payload_fleet.py::FleetMcp',
            )
        return handler(url, args)

    def refused(self) -> tuple[str, ...]:
        """Every tool asked for that the fleet does not answer, in the order asked."""
        return tuple(self._refused)

    def tree(self, label: str) -> _Tree:
        try:
            return self._trees[label]
        except KeyError:
            raise LookupError(f'the payload fleet holds no root labelled {label!r}') from None

    def _root_tree(self, args: Mapping[str, Any]) -> _Tree:
        return self.tree(Path(args['project_root']).name)

    def _tree_at(self, base_url: str) -> _Tree:
        """The tree of the orchestrator whose escalation server is at *base_url*."""
        # mcp_tool_call is handed a BASE url and posts to its /mcp endpoint,
        # which is the form the config's escalation URLs are written in.
        endpoint = f'{base_url}/mcp'
        label = self._label_by_mcp_endpoint.get(endpoint)
        if label is None:
            raise LookupError(f'the payload fleet has no orchestrator at {endpoint!r}')
        return self.tree(label)

    def _runtime_state(self, base_url: str, _args: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._tree_at(base_url).runtime

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

    def _merge_queue(self, base_url: str, _args: Mapping[str, Any]) -> dict[str, Any]:
        """The orchestrator's live merge queue: its newest pending tasks, waiting."""
        tree = self._tree_at(base_url)
        waiting = tree.ids(TaskStatus.PENDING)[-self._live_queue_depth:]
        return {
            'entries': [
                {'task_id': str(tid), 'branch': f'task/{tid}', 'state': 'queued',
                 'age_secs': 60.0 * (n + 1), 'position': n, 'waiter_alive': True}
                for n, tid in enumerate(waiting)
            ],
            'metrics': {},
        }

    def _merge_halt_status(self, base_url: str, _args: Mapping[str, Any]) -> dict[str, Any]:
        self._tree_at(base_url)
        return {'wired': True, 'halted': False, 'owner_esc_id': None}

    def _pending_escalations(self, _url: str, _args: Mapping[str, Any]) -> list[dict[str, Any]]:
        """No record carries a pins_recovery annotation: a read that succeeded and pins nothing."""
        return []

    def _pending_tickets(self, _url: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """The root's curator queue: empty, as every live root's was when the census was taken."""
        tree = self._root_tree(args)
        return {'count': 0, 'tickets': [], 'project_id': normalize_project_id(tree.census.label)}


class _ServerStatus:
    """What the fleet's one fused-memory server reports of itself: stores, write queue, WAL, curator gate.

    A read addressed anywhere but the fleet's fused-memory URLs raises.
    """

    _UPTIME = timedelta(seconds=12_269)

    def __init__(self, fleet: FleetCensus, urls: Sequence[str], *, now: datetime) -> None:
        self._memory = fleet.memory
        self._urls = frozenset(urls)
        self._now = now
        self._project_ids = [normalize_project_id(root.label) for root in fleet.roots]
        self._account_count = len(fleet.costs.accounts)

    def _check(self, url: str) -> None:
        if url not in self._urls:
            raise LookupError(f'the payload fleet runs no fused-memory server at {url!r}')

    def status(self, url: str, _args: Mapping[str, Any]) -> dict[str, Any]:
        self._check(url)
        memory = self._memory
        return {
            'graphiti': {
                'connected': True, 'node_count': memory.graphiti_nodes,
                'edge_count': 3 * memory.graphiti_nodes, 'episode_count': memory.graphiti_nodes // 4,
            },
            'mem0': {'connected': True, 'memory_count': memory.mem0_memories},
            'projects': memory.project_counts(),
            'uptime_seconds': int(self._UPTIME.total_seconds()),
            'started_at': (self._now - self._UPTIME).isoformat(),
        }

    def queue_stats(self, url: str, _args: Mapping[str, Any]) -> dict[str, Any]:
        self._check(url)
        return {'counts': {'pending': 0, 'retry': 0, 'dead': 1}, 'oldest_pending_age_seconds': None}

    def wal_status(self, url: str, _args: Mapping[str, Any]) -> dict[str, Any]:
        """One checkpoint row per store: the shared ones, then each project's overrides store."""
        self._check(url)
        stores = [*self._memory.wal_stores, *(f'overrides:{pid}' for pid in self._project_ids)]
        checked = (self._now - timedelta(seconds=117)).isoformat()
        return {'stores': {
            store: {'ts': checked, 'busy': 0, 'log': 0, 'checkpointed': 0,
                    'detail': f'{len(self._project_ids)} project(s)'}
            for store in stores
        }}

    def curator_state(self, url: str, _args: Mapping[str, Any]) -> dict[str, Any]:
        self._check(url)
        return {'paused': False, 'paused_reason': None, 'soonest_open_at': None, 'account_count': self._account_count}


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


def _escalated_task_ids(tree: _Tree, count: int) -> list[str]:
    """The task each of *count* escalations names: two in three a task of *tree*, the rest no task.

    A non-numeric id is how a record about no single task (a disk alarm, a
    fleet-wide incident) is filed; its row carries no card.
    """
    rows = tree.rows
    return [
        f'{tree.census.label}-concern-{n}' if n % 3 == 2 or not rows
        else rows[(n * 37) % len(rows)]['id']
        for n in range(count)
    ]


def _seed_escalations(
    config: DashboardConfig, fleet: FleetCensus, substrate: FleetMcp, *, now: datetime,
) -> None:
    """Every orchestrator queue's live records and archive, and the reconciliation queue's.

    Reconciliation records name the primary root's active tasks and carry a
    worktree under it, which is how that queue's rows find their owner.
    """
    queues = {queue.label: queue for queue in corpus_queues(config)}
    for root in fleet.roots:
        queue_dir = queues[root.label].directory
        tree = substrate.tree(root.label)
        seed_pending_escalations(
            queue_dir, fleet.escalation, _escalated_task_ids(tree, root.pending_escalations), now=now,
        )
        seed_resolved_history(queue_dir, fleet.escalation, root.resolved_escalations, now=now)
    primary = substrate.tree(config.project_root.name)
    active = [row['id'] for row in primary.rows if row['status'] not in TERMINAL]
    seed_pending_escalations(
        reconciliation_queue(config).directory, fleet.escalation,
        [active[(n * 13) % len(active)] for n in range(fleet.reconciliation_escalations)],
        now=now, worktree_root=config.project_root,
    )


def _seed_ledgers_and_stores(
    config: DashboardConfig, fleet: FleetCensus, substrate: FleetMcp, *, now: datetime,
) -> None:
    """Each root's run ledger and task store, the curator's ledger and the memory-eval artifacts.

    A task store holds the root's whole tree. Its first ``claimed_terminal``
    finished tasks still carry a claimant: the violations /invariants lists.
    Only a root whose orchestrator runs bills invocations and account events.
    """
    stale = (now - timedelta(hours=6)).isoformat()
    paths = [config.project_root, *config.known_project_roots]
    for index, (root, path) in enumerate(zip(fleet.roots, paths, strict=True)):
        rows = substrate.tree(root.label).rows
        runs_db = path / 'data' / 'orchestrator' / 'runs.db'
        project_id = normalize_project_id(root.label)
        seed_task_results(runs_db, project_id, root.completed_hours, now=now)
        if root.orchestrator_running:
            seed_invocations(runs_db, project_id, fleet.costs, fleet.costs.model_roles, root.completed_hours, now=now)
            seed_account_events(runs_db, project_id, fleet.costs, offset=index, now=now)
        if rows:
            seed_merge_events(
                runs_db, [row['id'] for row in rows], root.merges, root.trains,
                history_days=fleet.merge_history_days, now=now,
            )
        finished = [row for row in rows if row['status'] in TERMINAL]
        claimed = {row['id'] for row in finished[:root.claimed_terminal]}
        write_tasks_db(path / '.taskmaster' / 'tasks' / 'tasks.db', [
            task_row(int(row['id']), row['status'], *((f'run-fleet/{row["id"]}', stale) if row['id'] in claimed else ()))
            for row in rows
        ])
    curator_pairs = [('opus', role) for role in fleet.costs.curator_roles]
    seed_invocations(config.curator_events_db, 'dark_factory', fleet.costs, curator_pairs, 0, now=now)
    build_dual_escalation_tree(config.project_root)


def _seed_histories(
    config: DashboardConfig, fleet: FleetCensus, substrate: FleetMcp, *, now: datetime,
) -> None:
    """The sampled stores: burndown series, the metrics loop's samples, the recon store, the write journal."""
    paths = [str(path) for path in (config.project_root, *config.known_project_roots)]
    trees = {
        path: Counter(TaskStatus(status) for status in substrate.tree(root.label).status_map().values())
        for root, path in zip(fleet.roots, paths, strict=True)
    }
    seed_burndown_history(config.burndown_db, trees, fleet.burndown, now=now)
    seed_metrics_history(
        config.metrics_db, memory=fleet.memory, recon=fleet.recon,
        running_roots=[path for root, path in zip(fleet.roots, paths, strict=True) if root.orchestrator_running],
        merge_depth=fleet.live_queue_depth, now=now,
    )
    seed_reconciliation(
        config.reconciliation_db, fleet.recon, [normalize_project_id(root.label) for root in fleet.roots], now=now,
    )
    seed_write_journal(config.write_journal_db, fleet.memory_ops, now=now)
    _seed_quality(config, fleet.quality, now=now)


def _seed_quality(config: DashboardConfig, shape: QualityShape, *, now: datetime) -> None:
    """The budgeted root's landings in quality.db, the newest an hour old, and the sampler tick
    that measured them, so /quality serves every series fresh."""
    root = next(path for path in (config.project_root, *config.known_project_roots)
                if path.name == TERMINAL_WINDOW_ROOT)
    project_id = project_id_for_root(root)
    step = timedelta(hours=2)
    rows = history(
        shape.landings, start=now - timedelta(hours=1) - step * (shape.landings - 1), step=step,
        project_id=project_id,
        outcome=lambda i: replace(NO_OUTCOMES, reviewed=True, rejected=i % 5 == 0,
                                  rejection_events=int(i % 5 == 0), review_blocking=i % 4 == 0),
        cost=lambda i: replace(NO_COST, all_role_usd=2.0 + (i % 7) / 3, by_role=(('implementer', 2.0),),
                               implementer_invocations=1 + i % 3, turns=40 + i % 11),
        modules=lambda i: (ModuleLines(f'module-{i % shape.modules:03d}', 40 + i % 97),),
    )
    sample = QualitySample.value(
        project_id, now, cursor_sha=rows[-1].landing.merge_sha, landings_sampled=len(rows),
        landings_deferred=0, git_check=GitCrossCheck(0, 0, 0),
    )
    asyncio.run(write_quality_db(config.quality_db, rows, [sample]))


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


def _land_corpus_walk(client: TestClient, config: DashboardConfig) -> None:
    """Walk the escalation corpus once in the app's own loop, unbounded, as a running dashboard has.

    ``/prds`` waits
    ``dashboard/src/dashboard/api/prds.py::REQUEST_READS_WAIT_SECONDS`` on the
    walk and serves every root UNKNOWN when it misses. The fleet's first walk takes
    several seconds, so without one landed walk the request's measured size
    would depend on whether an earlier request had happened to walk it.
    """
    assert client.portal is not None, 'the corpus is walked inside a live TestClient'
    corpus = client.portal.call(partial(
        serve_corpus, corpus_queues(config), now=datetime.now(UTC), wait_seconds=None,
    ))
    assert corpus.value is not None, f'the fleet\'s escalation corpus walk answered no value: {corpus.reason}'


def _clear_caches() -> None:
    """Every cache a served payload could come from, so no other test's entry is served here."""
    forget_served_reads()
    task_lookup._lookup_cache_clear()
    _scheduler_cache_clear()


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
        now = datetime.now(UTC).replace(microsecond=0)
        substrate = FleetMcp(fleet, config.escalation_urls, config.fused_memory_urls, now=now)
        _seed_escalations(config, fleet, substrate, now=now)
        _seed_ledgers_and_stores(config, fleet, substrate, now=now)
        _seed_histories(config, fleet, substrate, now=now)
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
                    _land_corpus_walk(client, config)
                    yield client
            finally:
                _clear_caches()
        assert not substrate.refused(), (
            f'the app asked the payload fleet for {sorted(set(substrate.refused()))}, which it does '
            'not answer; a fan-out swallowed the refusal as an offline server, so some payload '
            'was measured empty. Answer each from the census in dashboard/tests/_payload_fleet.py::FleetMcp'
        )
