"""A recorded ordinary-pass scenario: the live candidate dump plus tasks that make every split step act.

``build_scenario`` lays out a rig in a directory: a git repo with a ``main``
branch as the project root, the committed dump ``RECORDED_DUMP`` of 1,480
recorded pending tasks with the wall clock fixed at its capture time, and the
synthetic tasks named
below, each of which makes one read/write step of the ordinary pass act (task
10476, plans/early-start-on-predecessor-branch-prd.md §5.4).
``record_passes`` runs ordinary passes over it and returns what they decided.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from _delivered_check_fixtures import install_delivered_check_script, script_check
from _orch_helpers import HermeticMcpSession
from _scheduler_candidate_fixture import (
    fixture_as_tasks,
    fixture_captured_at,
    load_candidate_fixture,
)
from shared.task_read import TASK_NOT_FOUND_ERROR_TYPE

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.overrides import OverrideRow, OverrideStore
from orchestrator.scheduler import Scheduler, SchedulerCallbacks, TaskAssignment

PASSES = 4
WRITE_TOOLS = frozenset({'set_task_status', 'update_task', 'set_task_claimant'})
_UNRECORDED_WRITE_ARGS = frozenset({'client_op_id', 'project_root', 'agent_id'})
_TIMING_EVENT_TYPES = frozenset({EventType.scheduler_tick.value})
_TIMING_DATA_SUFFIXES = ('_at', '_ms')
_SEED_COMMIT_DATE = '2026-10-12T00:00:00+00:00'
# The golden was recorded on this dump; a newer one would change the population.
RECORDED_DUMP = '2026-10-12.json'
# The seed commit's sha is in the golden, so no user or system git config or hook may touch it.
_GIT_ENV = {
    'GIT_AUTHOR_DATE': _SEED_COMMIT_DATE,
    'GIT_COMMITTER_DATE': _SEED_COMMIT_DATE,
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
}

# Synthetic tasks: ids above the dump's, one role each.
GREP_PASS_DEP = '9001'
GREP_PASS_DEPENDENT = '9002'
GREP_FAIL_DEP = '9003'
GREP_FAIL_DEPENDENT = '9004'
UNKNOWN_DEP_DEPENDENT = '9005'
UNKNOWN_DEP = '9099'
SENTINEL_EXT_DEPENDENT = '9006'
SENTINEL_EXT_DEP = 'elsewhere:1'
DONE_EXT_DEPENDENT = '9007'
DONE_EXT_DEP = 'elsewhere:2'
SCRIPT_DEP = '9008'
SCRIPT_DEPENDENT = '9009'
INHERITING_LOW = '9010'
HIGH_DEPENDENT = '9011'
BOOSTED = '9012'
PINNED = '9013'
EXPIRED_ROW = '9015'
WATCHDOG_PINNED = '9016'
PLAIN_DONE_DEP = '9017'
PLAIN_DONE_DEPENDENT = '9018'

SYNTHETIC_IDS = (
    GREP_PASS_DEP, GREP_PASS_DEPENDENT, GREP_FAIL_DEP, GREP_FAIL_DEPENDENT,
    UNKNOWN_DEP_DEPENDENT, SENTINEL_EXT_DEPENDENT, DONE_EXT_DEPENDENT,
    SCRIPT_DEP, SCRIPT_DEPENDENT, INHERITING_LOW, HIGH_DEPENDENT, BOOSTED,
    PINNED, EXPIRED_ROW, WATCHDOG_PINNED, PLAIN_DONE_DEP, PLAIN_DONE_DEPENDENT,
)

_PRESENT_PATTERN = 'DeliveredCapabilityMarker'
_ABSENT_PATTERN = 'NeverLandedCapability'
_SCRIPT_REL_PATH = 'checks/count_runs.sh'


class TaskStoreMcpSession(HermeticMcpSession):
    """A fused-memory double backed by a full task table, logging every call.

    ``get_tasks`` honours its ``statuses`` filter, so done dependencies are
    absent from the active fetch as in production; ``get_statuses`` omits ids
    the table lacks; ``get_task`` returns the full record. Write tools are
    answered benignly by the parent and logged like every other call.
    A tool named in ``failing`` answers with an MCP error reply.
    """

    def __init__(self, tasks: list[dict], external_statuses: dict[str, str]) -> None:
        super().__init__(tasks=[], external_statuses=external_statuses)
        self._table: dict[str, dict] = {str(t['id']): t for t in tasks}
        self.calls: list[tuple[str, dict]] = []
        self.failing: set[str] = set()

    def set_status(self, task_id: str, status: str) -> None:
        self._table[task_id] = {**self._table[task_id], 'status': status}

    def count(self, tool: str) -> int:
        return sum(1 for name, _ in self.calls if name == tool)

    @property
    def writes(self) -> list[tuple[str, dict]]:
        return [(name, arguments) for name, arguments in self.calls if name in WRITE_TOOLS]

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        self.calls.append((name, json.loads(json.dumps(arguments))))
        if name in self.failing:
            return {'jsonrpc': '2.0', 'id': self._next_id(),
                    'result': {'isError': True, 'content': [{'type': 'text', 'text': f'{name} failed'}]}}
        if name == 'get_tasks':
            wanted = arguments.get('statuses')
            rows = [t for t in self._table.values() if wanted is None or t['status'] in wanted]
            return self._envelope(json.dumps({'tasks': rows}))
        if name == 'get_statuses':
            ids = arguments.get('ids')
            known = self._table if ids is None else [i for i in ids if i in self._table]
            return self._envelope(json.dumps({'statuses': {i: self._table[i]['status'] for i in known}}))
        if name == 'get_task':
            task = self._table.get(str(arguments.get('id')))
            if task is None:
                return self._envelope(json.dumps({'error': 'no such task', 'error_type': TASK_NOT_FOUND_ERROR_TYPE}))
            return self._envelope(json.dumps(task))
        return await super().call_tool(name, arguments, timeout)


@dataclass(frozen=True)
class Scenario:
    scheduler: Scheduler
    session: TaskStoreMcpSession
    event_store: EventStore
    override_store: OverrideStore
    callback_calls: list[dict]
    script_counter: Path

    @property
    def project_root(self) -> str:
        return str(self.scheduler.config.project_root)


@dataclass(frozen=True)
class PassRecord:
    """What K ordinary passes decided, and the reads each pass made.

    ``decisions`` is JSON-shaped and holds behaviour only; the per-pass read
    counts are kept apart because they describe how the pass read, not what
    it decided.
    """

    decisions: dict
    get_task_calls: tuple[int, ...]
    script_runs: tuple[int, ...]


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ['git', '-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false', '-C', str(repo), *args],
        check=True, capture_output=True,
        env={**os.environ, **_GIT_ENV},
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, 'init', '-q', '-b', 'main')
    (repo / 'capability.py').write_text(f'{_PRESENT_PATTERN} = True\n')
    _git(repo, 'add', 'capability.py')
    _git(repo, 'commit', '-q', '--no-verify', '-m', 'seed')


def _grep_check(name: str, pattern: str) -> dict:
    return {'name': name, 'kind': 'grep', 'pattern': pattern, 'expect': 'present'}


def _task(task_id: str, status: str, priority: str = 'medium', *, deps: tuple[str, ...] = (),
          metadata: dict | None = None) -> dict:
    return {
        'id': task_id, 'title': f'synthetic {task_id}', 'status': status, 'priority': priority,
        'dependencies': list(deps), 'metadata': dict(metadata or {}),
    }


def _synthetic_tasks() -> list[dict]:
    return [
        _task(GREP_PASS_DEP, 'done', metadata={'delivered_checks': [_grep_check('lands', _PRESENT_PATTERN)]}),
        _task(GREP_PASS_DEPENDENT, 'pending', 'critical', deps=(GREP_PASS_DEP,)),
        _task(GREP_FAIL_DEP, 'done', metadata={'delivered_checks': [_grep_check('missing', _ABSENT_PATTERN)]}),
        _task(GREP_FAIL_DEPENDENT, 'pending', 'critical', deps=(GREP_FAIL_DEP,)),
        _task(UNKNOWN_DEP_DEPENDENT, 'pending', 'high', deps=(UNKNOWN_DEP,)),
        _task(SENTINEL_EXT_DEPENDENT, 'pending', 'critical', metadata={'external_deps': [SENTINEL_EXT_DEP]}),
        _task(DONE_EXT_DEPENDENT, 'pending', 'high', metadata={'external_deps': [DONE_EXT_DEP]}),
        _task(SCRIPT_DEP, 'done', metadata={'delivered_checks': [script_check('counts', _SCRIPT_REL_PATH)]}),
        _task(SCRIPT_DEPENDENT, 'pending', 'high', deps=(SCRIPT_DEP,)),
        _task(INHERITING_LOW, 'pending', 'low'),
        _task(HIGH_DEPENDENT, 'pending', 'high', deps=(INHERITING_LOW,)),
        _task(BOOSTED, 'pending', 'low'),
        _task(PINNED, 'pending', 'medium'),
        _task(EXPIRED_ROW, 'pending', 'low'),
        _task(WATCHDOG_PINNED, 'in-progress', 'medium'),
        _task(PLAIN_DONE_DEP, 'done'),
        _task(PLAIN_DONE_DEPENDENT, 'pending', 'low', deps=(PLAIN_DONE_DEP,)),
    ]


def _seed_overrides(store: OverrideStore, project_root: str, now: datetime) -> None:
    store.set_override(project_root, BOOSTED, boost_tier='high')
    store.set_override(project_root, PINNED, pinned=True)
    store.set_override(project_root, GREP_PASS_DEP, boost_tier='critical')
    store.set_override(project_root, EXPIRED_ROW, boost_tier='critical', ttl_until=now - timedelta(hours=1))
    store.create_watchdog_pin(project_root, WATCHDOG_PINNED, ttl_until=now + timedelta(days=1))


def build_scenario(root: Path, *, delivered_checks_enabled: bool = True) -> Scenario:
    """A fresh rig under *root*. Needs the ``code_default_config`` fixture active."""
    fixture = load_candidate_fixture(RECORDED_DUMP)
    now = fixture_captured_at(fixture)
    repo = root / 'repo'
    _init_repo(repo)
    script_counter = root / 'script_runs.log'
    install_delivered_check_script(repo, _SCRIPT_REL_PATH, f'#!/bin/sh\necho run >> {script_counter}\nexit 0\n')

    config = OrchestratorConfig(project_root=repo, max_external_dep_unresolved_cycles=2)
    config.delivered_checks.enabled = delivered_checks_enabled
    config.delivered_checks.grace_cycles = 2
    project_root = str(config.project_root)

    override_store = OverrideStore(root / 'scheduler_overrides.db')
    _seed_overrides(override_store, project_root, now)
    session = TaskStoreMcpSession(
        fixture_as_tasks(fixture) + _synthetic_tasks(),
        external_statuses={SENTINEL_EXT_DEP: 'unknown_task', DONE_EXT_DEP: 'done'},
    )
    callback_calls: list[dict] = []

    def recorder(name: str):
        async def _record(task_id: str, *, summary: str, category: str | None = None, **_: Any) -> None:
            callback_calls.append({'callback': name, 'task_id': task_id, 'category': category, 'summary': summary})
        return _record

    event_store = EventStore(root / 'runs.db', 'run-1')
    scheduler = Scheduler(
        config,
        event_store,
        callbacks=SchedulerCallbacks(
            on_external_dep_block=recorder('on_external_dep_block'),
            on_delivered_check_block=recorder('on_delivered_check_block'),
            on_starvation_warn=recorder('on_starvation_warn'),
        ),
        mcp_session=session,
        time_source=lambda: 1000.0,
        monotonic_clock_source=lambda: 1000.0,
        wall_time_source=lambda: now,
        override_store=override_store,
        state_snapshot_path=root / 'scheduler_state.json',
        branch_work_reader=dict,
    )
    scheduler.finish_startup()
    return Scenario(scheduler, session, event_store, override_store, callback_calls, script_counter)


def all_events(store: EventStore) -> list[dict]:
    rows = [row for event_type in EventType for row in store.fetch_events_by_type(event_type)]
    return sorted(rows, key=lambda row: row['id'])


def _event_decision(row: dict) -> dict:
    data = {k: v for k, v in (row['data'] or {}).items() if not k.endswith(_TIMING_DATA_SUFFIXES)}
    return {'type': row['event_type'], 'task_id': row['task_id'], 'data': data}


def _write_decision(name: str, arguments: dict) -> dict:
    return {'tool': name, 'arguments': {k: v for k, v in arguments.items() if k not in _UNRECORDED_WRITE_ARGS}}


def _assignment_decision(assignment: TaskAssignment | None) -> dict | None:
    if assignment is None:
        return None
    return {'task_id': assignment.task_id, 'modules': assignment.modules}


def _override_decision(row: OverrideRow) -> dict:
    fields = dataclasses.asdict(row)
    fields['ttl_until'] = None if row.ttl_until is None else row.ttl_until.astimezone(UTC).isoformat()
    fields['source'] = str(row.source)
    return fields


def _priority_decision(scheduler: Scheduler, task_id: str) -> dict:
    priority = scheduler.task_priority(task_id)
    return {'tier': priority.tier, 'base_tier': priority.base_tier, 'pinned': priority.pinned,
            'pin_order': priority.pin_order}


def count_script_runs(counter: Path) -> int:
    return len(counter.read_text().splitlines()) if counter.exists() else 0


async def record_passes(scenario: Scenario, passes: int = PASSES) -> PassRecord:
    """Run *passes* ordinary passes, marking each assigned task in-progress, and record them."""
    scheduler, session = scenario.scheduler, scenario.session
    seen_events = {row['id'] for row in all_events(scenario.event_store)}
    calls_before, callbacks_before = len(session.calls), len(scenario.callback_calls)
    get_task_calls: list[int] = []
    script_runs: list[int] = []
    records: list[dict] = []
    for _ in range(passes):
        get_tasks_before = session.count('get_task')
        runs_before = count_script_runs(scenario.script_counter)
        assignment = await scheduler.acquire_next()
        if assignment is not None:
            session.set_status(assignment.task_id, 'in-progress')
        new_events = [row for row in all_events(scenario.event_store) if row['id'] not in seen_events]
        seen_events.update(row['id'] for row in new_events)
        records.append({
            'assignment': _assignment_decision(assignment),
            'writes': [_write_decision(n, a) for n, a in session.calls[calls_before:] if n in WRITE_TOOLS],
            'callbacks': scenario.callback_calls[callbacks_before:],
            'events': [_event_decision(row) for row in new_events if row['event_type'] not in _TIMING_EVENT_TYPES],
        })
        calls_before, callbacks_before = len(session.calls), len(scenario.callback_calls)
        get_task_calls.append(session.count('get_task') - get_tasks_before)
        script_runs.append(count_script_runs(scenario.script_counter) - runs_before)
    decisions = {
        'passes': records,
        'overrides': {
            tid: _override_decision(row)
            for tid, row in sorted(scenario.override_store.get_overrides(scenario.project_root).items())
        },
        'task_priority': {tid: _priority_decision(scheduler, tid) for tid in SYNTHETIC_IDS},
    }
    return PassRecord(json.loads(json.dumps(decisions)), tuple(get_task_calls), tuple(script_runs))
