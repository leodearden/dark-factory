"""The payload fleet's on-disk half: the files the routes read beside the MCP reads.

``_payload_fleet.py`` answers every MCP read from its census; the routes this
module feeds read files instead: escalation queues and their archives, and
each root's run ledger (task results, events, invocations and account
events), and the memory-eval artifact tree. The sampled histories live in
``_payload_fleet_history.py``. Each
seeder writes one family through the production writer or schema wherever one
is importable, at the sizes its ``*Shape`` names, and knows nothing of MCP.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path

from _dashboard_helpers import dump_artifact
from _runs_db_fixture import EVENTS_DDL, TASK_RESULTS_DDL
from escalation.archive import archive_dir_for_date
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cost_store import CostStore

FLEET_RUN_ID = 'run-payload-fleet'
"""The one orchestrator run every ledger row belongs to, so invocations join their task results."""


def text(seed: str, length: int) -> str:
    """*seed*, padded with ASCII filler to exactly *length* characters."""
    if len(seed) >= length:
        raise ValueError(f'seed {seed!r} does not fit in {length} characters')
    return seed + ' ' + 'x' * (length - len(seed) - 1)


@dataclass(frozen=True)
class EscalationShape:
    """The sizes an escalation record is written at, and how far back resolved history reaches."""

    detail_len: int
    summary_len: int
    triage_note_len: int
    option_len: int
    history_days: int


_RESOLVERS = ('interactive', 'escalation-watcher-auto', 'harness-orphan-reaper', 'claude-task-1-steward')
"""One resolver per tier the analytics split on: human, auto-watcher, reaper sweep, steward."""
_ROLES = ('implementer', 'steward', 'escalation-watcher-auto', 'merger')
_CATEGORIES = ('design_concern', 'infra_issue', 'risk_identified', 'cleanup_needed')


def _record(shape: EscalationShape, esc_id: str, task_id: str, filed: datetime, n: int) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id=task_id,
        agent_role=_ROLES[(n // 7) % len(_ROLES)],
        severity='blocking',
        category=_CATEGORIES[n % len(_CATEGORIES)],
        summary=text(f'fleet escalation {esc_id}', shape.summary_len),
        detail=text(f'fleet escalation detail {esc_id}', shape.detail_len),
        timestamp=filed.isoformat(),
        level=n % 3,
        options=[text('A: fleet option', shape.option_len)],
        triage_note=text('fleet triage note', shape.triage_note_len),
    )


def seed_pending_escalations(
    queue_dir: Path,
    shape: EscalationShape,
    task_ids: Sequence[str],
    *,
    now: datetime,
    worktree_root: Path | None = None,
) -> None:
    """File one pending escalation per entry of *task_ids* into the live queue at *queue_dir*.

    *worktree_root*, when given, puts each record's worktree under it: how a
    reconciliation record names the root that owns its task.
    """
    queue = EscalationQueue(queue_dir)
    for n, task_id in enumerate(task_ids):
        record = _record(shape, f'esc-{task_id}-{n + 1}', task_id, now - timedelta(hours=n + 1), n)
        if worktree_root is not None:
            record = replace(record, worktree=str(worktree_root / '.worktrees' / task_id))
        queue.submit(record)


def seed_resolved_history(queue_dir: Path, shape: EscalationShape, count: int, *, now: datetime) -> None:
    """Write *count* resolved escalations into *queue_dir*'s dated archive, spread over the history.

    Records pair up on one task a day apart, so every second filing re-files
    a task resolved the day before: the churn the workflow view counts.
    """
    for n in range(count):
        filed = now - timedelta(days=n % shape.history_days, hours=6)
        resolved_at = filed + timedelta(minutes=30 + 7 * (n % 300))
        record = replace(
            _record(shape, f'esc-h{n}-1', f'h{n // 2}', filed, n),
            status='resolved',
            resolved_at=resolved_at.isoformat(),
            resolved_by=_RESOLVERS[(n // 3) % len(_RESOLVERS)],
            resolution='fleet resolution',
        )
        archive = archive_dir_for_date(queue_dir, resolved_at.isoformat())
        archive.mkdir(parents=True, exist_ok=True)
        (archive / f'{record.id}.json').write_text(record.to_json())


_TASK_RESULT_COLUMNS = (
    'run_id', 'task_id', 'project_id', 'title', 'outcome', 'cost_usd', 'duration_ms',
    'agent_invocations', 'execute_iterations', 'verify_attempts', 'review_cycles',
    'steward_cost_usd', 'steward_invocations', 'completed_at',
)


def seed_task_results(runs_db: Path, project_id: str, completed_hours: int, *, now: datetime) -> None:
    """One finished task per hour over the last *completed_hours* hours, in *runs_db*'s ledger.

    Every completion lands in its own hour, so the performance histories,
    which bucket by hour, hold one bucket per completion.
    """
    runs_db.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            FLEET_RUN_ID, str(hour), project_id, None, 'done', 0.5,
            600_000 + (997 * hour) % 3_600_000, 3, 2, 1 + hour % 2, hour % 2, 0.0, int(hour % 9 == 0),
            (now - timedelta(hours=hour)).isoformat(),
        )
        for hour in range(1, completed_hours + 1)
    ]
    with closing(sqlite3.connect(runs_db)) as conn, conn:
        conn.executescript(TASK_RESULTS_DDL)
        conn.executemany(
            f'INSERT INTO task_results ({", ".join(_TASK_RESULT_COLUMNS)}) '
            f'VALUES ({", ".join("?" * len(_TASK_RESULT_COLUMNS))})',
            rows,
        )


_MERGE_OUTCOMES = ('done', 'done', 'done', 'conflict', 'blocked', 'cas_retry')


def seed_merge_events(
    runs_db: Path, task_ids: Sequence[str], merges: int, trains: int, *, history_days: int, now: datetime,
) -> None:
    """*merges* merge attempts spread evenly over *history_days*, and the *trains* newest train merges.

    Each attempt names a task of *task_ids* in turn, so the recent table's
    title lookups resolve against the fleet.
    """
    runs_db.parent.mkdir(parents=True, exist_ok=True)
    span = timedelta(days=history_days)
    attempts = [
        (
            (now - span * (n + 1) / (merges + 1)).isoformat(), FLEET_RUN_ID, task_ids[n % len(task_ids)],
            'merge_attempt', 'merge', json.dumps({'outcome': _MERGE_OUTCOMES[n % len(_MERGE_OUTCOMES)]}),
            60_000 + (7_919 * n) % 1_800_000,
        )
        for n in range(merges)
    ]
    train_merges = [
        (
            (now - timedelta(hours=6 * (n + 1))).isoformat(), FLEET_RUN_ID, task_ids[n % len(task_ids)],
            'train_merged', 'merge', json.dumps({
                'train_id': f'coalesce-{task_ids[n % len(task_ids)]}-fleet',
                'member_task_ids': [task_ids[(n + k) % len(task_ids)] for k in range(2)],
                'merge_commit_sha': text('fleet-merge-sha', 40), 'base_sha': text('fleet-base-sha', 40),
            }),
            None,
        )
        for n in range(trains)
    ]
    with closing(sqlite3.connect(runs_db)) as conn, conn:
        conn.executescript(EVENTS_DDL)
        conn.executemany(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, phase, data, duration_ms) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            [*attempts, *train_merges],
        )


@dataclass(frozen=True)
class CostShape:
    """What the invocation ledgers hold: days of spend, who spent it, and the account events beside it.

    ``model_roles`` are the (model, role) pairs orchestrator invocations bill
    under; ``curator_roles`` bill the fused-memory curator's own ledger.
    ``account_events`` is per orchestrator ledger, newest first.
    """

    spend_days: int
    invocations_per_day: int
    model_roles: tuple[tuple[str, str], ...]
    curator_roles: tuple[str, ...]
    accounts: tuple[str, ...]
    account_events: int
    cap_reason_len: int


_ACCOUNT_EVENT_CYCLE = (
    'failover', 'cap_hit', 'failover', 'cap_hit', 'failover', 'cap_hit', 'failover', 'cap_hit', 'resumed',
)
"""The live mix of account events: about four failovers and four cap hits to each resume."""

_INVOCATION_COLUMNS = (
    'run_id', 'task_id', 'project_id', 'account_name', 'model', 'role', 'cost_usd',
    'input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_create_tokens',
    'duration_ms', 'capped', 'started_at', 'completed_at', 'model_id', 'capped_reason',
)


def _create_cost_tables(ledger: Path) -> None:
    """Create *ledger*'s tables through its own writer, so its schema and migrations are the production ones."""
    async def _open_and_close() -> None:
        store = CostStore(ledger)
        await store.open()
        await store.close()

    ledger.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(_open_and_close())


def _invocation(
    n: int, project_id: str, shape: CostShape, pair: tuple[str, str], task_id: str | None, completed: datetime,
) -> tuple[object, ...]:
    model, role = pair
    capped = n % 50 == 0
    return (
        FLEET_RUN_ID, task_id, project_id, shape.accounts[n % len(shape.accounts)], model, role,
        round(20 + (7_919 * n) % 40_000 / 1_000, 4),
        6_000 + n % 997, 40_000 + n % 9_973, 4_000_000 + n % 99_991, 150_000 + n % 9_967,
        600_000 + (997 * n) % 3_600_000, int(capped),
        (completed - timedelta(minutes=10)).isoformat(), completed.isoformat(),
        f'claude-{model}-fleet', 'turns' if capped else None,
    )


def seed_invocations(
    ledger: Path,
    project_id: str,
    shape: CostShape,
    pairs: Sequence[tuple[str, str]],
    completed_tasks: int,
    *,
    now: datetime,
) -> None:
    """``invocations_per_day`` invocations a day over the newest ``spend_days`` days, cycling *pairs*.

    Each names one of the run's *completed_tasks* finished tasks (none when
    it has none), so the model/role rollup joins it to a done outcome.
    """
    _create_cost_tables(ledger)
    rows = []
    for day in range(shape.spend_days):
        for j in range(shape.invocations_per_day):
            n = day * shape.invocations_per_day + j
            task_id = str(1 + n % completed_tasks) if completed_tasks else None
            completed = now - timedelta(days=day, minutes=5 + 37 * j)
            rows.append(_invocation(n, project_id, shape, pairs[n % len(pairs)], task_id, completed))
    with closing(sqlite3.connect(ledger)) as conn, conn:
        conn.executemany(
            f'INSERT INTO invocations ({", ".join(_INVOCATION_COLUMNS)}) '
            f'VALUES ({", ".join("?" * len(_INVOCATION_COLUMNS))})',
            rows,
        )


def _account_event_details(kind: str, shape: CostShape, account: int, now: datetime) -> str:
    if kind == 'failover':
        return json.dumps({'from': shape.accounts[account - 1], 'to': shape.accounts[account]})
    if kind == 'cap_hit':
        return json.dumps({
            'reason': text('fleet usage limit reached', shape.cap_reason_len),
            'resets_at': (now + timedelta(hours=11 + account)).isoformat(),
        })
    return json.dumps({'label': f'fleet probe #{account} confirmed'})


def seed_account_events(ledger: Path, project_id: str, shape: CostShape, *, offset: int, now: datetime) -> None:
    """``account_events`` account events, four minutes apart, newest first; *offset* staggers ledgers.

    Every third cap hit names no project, as an account-wide cap does.
    """
    _create_cost_tables(ledger)
    rows = []
    for k in range(shape.account_events):
        kind = _ACCOUNT_EVENT_CYCLE[k % len(_ACCOUNT_EVENT_CYCLE)]
        account = (k + offset) % len(shape.accounts)
        account_wide = kind == 'cap_hit' and k % 3 == 0
        rows.append((
            shape.accounts[account], kind, None if account_wide else project_id, FLEET_RUN_ID,
            _account_event_details(kind, shape, account, now),
            (now - timedelta(minutes=4 * k, seconds=23 * offset)).isoformat(),
        ))
    with closing(sqlite3.connect(ledger)) as conn, conn:
        conn.executemany(
            'INSERT INTO account_events (account_name, event_type, project_id, run_id, details, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?)',
            rows,
        )


@dataclass(frozen=True)
class MemoryEvalShape:
    """One eval's artifacts: its window metrics, its tripwire's items (0: no tripwire), its run
    depth, the lengths of an item key and of a window verdict's detail, and its corpus counters."""

    eval_id: str
    window_metrics: int
    tripwire_items: int
    runs: int
    item_key_len: int
    detail_len: int
    corpus_counts: int


_EVAL_STAMP_FORMAT = '%Y%m%dT%H%M%SZ'
_EVAL_ALPHA = 0.002777777777777778
_TRIPWIRE_METRIC = 'tripwire-items-present'
# Two items in five are grandfathered, the live share (43 of 107).
_GRANDFATHERED_PER_5 = 2


def _eval_stamps(runs: int, now: datetime) -> list[str]:
    """*runs* stamps a day apart, oldest first, the newest three hours before *now*."""
    newest = now.replace(microsecond=0) - timedelta(hours=3)
    return [(newest - timedelta(days=age)).strftime(_EVAL_STAMP_FORMAT) for age in reversed(range(runs))]


def _window_ids(shape: MemoryEvalShape) -> list[str]:
    return [f'window-metric-{index:02d}' for index in range(shape.window_metrics)]


def _item_keys(shape: MemoryEvalShape) -> list[str]:
    return [text(f't-{index:03d}', shape.item_key_len) for index in range(shape.tripwire_items)]


def _grandfathered(index: int) -> bool:
    return index % 5 < _GRANDFATHERED_PER_5


def _fingerprint(*identity: str) -> str:
    return hashlib.sha256('|'.join(identity).encode()).hexdigest()


def _metrics_run(shape: MemoryEvalShape, stamp: str) -> dict:
    """One M1 run: the window metrics as proportions, then the tripwire valued at its failing items."""
    metrics: list[dict] = [
        {
            'metric_id': metric_id, 'kind': 'proportion', 'value': 0.9, 'n': 27, 'denominator': 30,
            'direction': 'lower_is_worse',
        }
        for metric_id in _window_ids(shape)
    ]
    if shape.tripwire_items:
        failing = sum(_grandfathered(index) for index in range(shape.tripwire_items))
        metrics.append({
            'metric_id': _TRIPWIRE_METRIC, 'kind': 'tripwire', 'value': float(failing),
            'n': shape.tripwire_items,
        })
    return {
        'schema_version': 1,
        'eval_id': shape.eval_id,
        'run_stamp': stamp,
        'corpus': {
            'project_id': 'dark_factory',
            'counts': {text(f'counter-{index:02d}', 24): 1000 + index for index in range(shape.corpus_counts)},
        },
        'metrics': metrics,
    }


def _limits(shape: MemoryEvalShape, stamps: list[str], grandfather_hash: str) -> dict:
    rule_kinds = [(metric_id, 'proportion') for metric_id in _window_ids(shape)]
    if shape.tripwire_items:
        rule_kinds.append((_TRIPWIRE_METRIC, 'tripwire'))
    return {
        'schema_version': 1,
        'eval_id': shape.eval_id,
        'run_stamp': stamps[-1],
        'alpha': _EVAL_ALPHA,
        'false_alarm_budget': 1.0,
        'runs_per_quarter': 90,
        'min_samples': 10,
        'baseline_window': 3,
        'baseline_run_stamps': stamps[:-1][-3:],
        'grandfather_set_hash': grandfather_hash,
        'generator': 'shared.memory_eval_limits',
        'verdicts': [
            {'metric_id': metric_id, 'rule_kind': rule_kind, 'status': 'ok', 'alarms': []}
            for metric_id, rule_kind in rule_kinds
        ],
    }


def _verdict_entries(shape: MemoryEvalShape, stamp: str, grandfather_hash: str) -> list[dict]:
    """The eval's M2 entries: one per window metric, and one per tripwire item shaped like
    shared/src/shared/memory_eval_verdicts.py::_tripwire_entries makes them."""
    entries = [
        {
            'eval_id': shape.eval_id, 'metric_id': metric_id, 'window': 'trailing-3',
            'verdict': 'no_alarm', 'fingerprint': _fingerprint(shape.eval_id, metric_id, 'trailing-3'),
            'detail': text(f'{metric_id} against a pooled baseline', shape.detail_len),
            'value': 0.9, 'limit_ref': f'alpha={_EVAL_ALPHA}', 'run_stamp': stamp,
            'rule_kind': 'proportion', 'metric_status': 'ok',
        }
        for metric_id in _window_ids(shape)
    ]
    failing = float(sum(_grandfathered(index) for index in range(shape.tripwire_items)))
    for index, item in enumerate(_item_keys(shape)):
        grandfathered = _grandfathered(index)
        entries.append({
            'eval_id': shape.eval_id, 'metric_id': _TRIPWIRE_METRIC, 'item': item,
            'verdict': 'grandfathered' if grandfathered else 'no_alarm',
            'fingerprint': _fingerprint(shape.eval_id, _TRIPWIRE_METRIC, item),
            'detail': f"Item '{item}' fails and is grandfathered (known-bad)." if grandfathered else f"Item '{item}' passes.",
            'value': failing, 'limit_ref': f'grandfather_set_hash={grandfather_hash}', 'run_stamp': stamp,
            'rule_kind': 'tripwire', 'metric_status': 'ok',
        })
    return entries


def seed_memory_evals(
    root: Path, shapes: Sequence[MemoryEvalShape], *, runless_evals: int, now: datetime,
) -> None:
    """Write each shape's eval dir under the memory-eval *root*, plus *runless_evals* dirs holding
    no run, and APPEND every shape's verdict entries to the root verdicts artifact already there.

    Appending keeps the escalation reach-path records an earlier
    ``_dashboard_helpers.build_dual_escalation_tree`` wrote. Content depends on
    *now* only through the stamps, which are fixed-width, so every build serves
    the same number of bytes.
    """
    verdicts_path = root / 'verdicts-current.json'
    verdicts = json.loads(verdicts_path.read_text())
    for shape in shapes:
        stamps = _eval_stamps(shape.runs, now)
        grandfather_hash = _fingerprint(shape.eval_id, 'grandfather-set')
        for stamp in stamps:
            dump_artifact(root / shape.eval_id / f'metrics-{stamp}.json', _metrics_run(shape, stamp))
        dump_artifact(root / shape.eval_id / 'limits-current.json', _limits(shape, stamps, grandfather_hash))
        verdicts['entries'].extend(_verdict_entries(shape, stamps[-1], grandfather_hash))
    for index in range(runless_evals):
        dump_artifact(root / f'runless-eval-{index}' / 'notes.json', {'note': 'no metrics run here'})
    dump_artifact(verdicts_path, verdicts)
