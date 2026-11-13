#!/usr/bin/env python3
"""The offline-lane soak report: ζ's mechanical checks and its re-measurement figures (task 6867).

PRD ``plans/offline-lane-red-path-prd.md`` §9 ζ and ζ0.  Read-only.  Reads the lane's
run records, its episode records, the flake ledger, the escalation queue and the task
store through their owners (``offline_lane_soak_inputs.py``), then judges them here.

Exit codes, the contract ζ's runbook reads:
    0  every check holds (a check that examined nothing reports no_data, and passes the run)
    1  any check fails
    2  nothing to judge: a store is missing or unreadable, or no pass record is in the window

Usage:
    python scripts/offline_lane_soak_report.py [--project-root ROOT] [--since ISO]
        [--command NAME ...] [--json] [--server-url URL] [--recheck-secs N]

Verdict lines go to stderr; stdout carries the figures, or with --json the full report.
Run it directly, never via `python -m`: it imports flat siblings in scripts/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import _checkout as _checkout
from migrate_metadata_modules_to_files import DEFAULT_SERVER
from offline_lane_soak_inputs import (
    SoakInputs,
    TaskReader,
    Unjudgeable,
    WindowPass,
    episode_bindings,
    load_inputs,
    load_inputs_from_server,
)
from orchestrator.flake_debt_drain import REGRESSED_AFTER_RESOLUTION
from orchestrator.flake_ledger import (
    DUE_ITEM_STALE_AFTER,
    DUE_ITEM_STALE_MINUTES,
    UNKNOWN_TEST_ID,
    DebtTrigger,
    FlakeCallSite,
    FlakeVerdict,
)
from orchestrator.offline_lane_confirm import Confirmed, Outage
from orchestrator.offline_lane_episode_state import (
    CommandRecord,
    EndReason,
    Episode,
    LanePolicy,
    OfferedDebt,
    unconfirmable_node,
)
from orchestrator.offline_lane_escalations import PENDING_ESCALATION
from orchestrator.offline_lane_filing import LANE_METADATA_KEY, covering_lane_task
from orchestrator.offline_lane_records import FailureRecord, PassRecord
from shared.project_layout import default_project_root

# The de-flake owner's metadata keys, by their PRD section 5.6 names: flake_ledger spells them
# inline (orchestrator/src/orchestrator/flake_ledger.py::_deflake_metadata) with no public constant.
DEBT_CALL_SITE_KEY = 'flake_debt_call_site'
DEBT_TRIGGER_KEY = 'flake_debt_trigger'


class Verdict(StrEnum):
    PASS = 'pass'
    FAIL = 'fail'
    NO_DATA = 'no_data'


class Check(StrEnum):
    RECORDS_WELL_FORMED = 'records_well_formed'
    FILINGS_RESOLVE = 'filings_resolve'
    FILINGS_BOUNDED_BY_THRESHOLD = 'filings_bounded_by_threshold'
    LEDGER_ROWS_PER_PASS = 'ledger_rows_per_pass'
    DEFLAKE_TASKS_CARRY_EVIDENCE = 'deflake_tasks_carry_evidence'
    NO_STRANDED_FILING = 'no_stranded_filing'
    OFFERED_DEBTS_OWNED_OR_DUE = 'offered_debts_owned_or_due'


@dataclass(frozen=True)
class CheckResult:
    name: Check
    verdict: Verdict
    examined: int
    rule: str
    failures: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Figure:
    name: str
    rule: str
    value: Any


@dataclass(frozen=True)
class Settings:
    recheck_secs: float


def _rules(settings: Settings) -> dict[Check, str]:
    return {
        Check.RECORDS_WELL_FORMED: 'every pass directory in the window holds a record.json, outcome block '
        'and effect block their owners parse, and each finished sub-run output file exists; a pass with no '
        'detect record, or with sub-runs and no effect block, is interrupted and does not fail',
        Check.FILINGS_RESOLVE: 'every task id a windowed effect block lists in tasks_filed resolves through '
        f'get_task with a non-empty details digest and an object at metadata.{LANE_METADATA_KEY} (PRD 5.5)',
        Check.FILINGS_BOUNDED_BY_THRESHOLD: 'the distinct task ids the windowed effect blocks list in tasks_filed '
        'number no more than the (pass, test) entries they list in threshold_reached',
        Check.LEDGER_ROWS_PER_PASS: 'every test a windowed confirmed outcome lists as reproduced has a '
        'fails_in_isolation row, and every cleared test a passes_in_isolation row, at call site offline_lane '
        "observed at the pass's start (PRD 5.8); the unconfirmable pseudo-test and the unknown id are exempt",
        Check.DEFLAKE_TASKS_CARRY_EVIDENCE: 'every owner task of a debt a windowed effect block offered, whose '
        'metadata says flake_debt_call_site offline_lane, carries a flake_debt_trigger and, in its description '
        "or details, the kind and first line of one of that test's failures in the command's pass records; "
        'an owner whose metadata names no flake_debt_call_site is reported',
        Check.NO_STRANDED_FILING: 'a test listed in threshold_reached at pass P is owned when either of two lane '
        'tasks reads back through get_task covering it: a task that a same-command effect block at or after P '
        'lists in tasks_filed, or the task its episode (live or ended) is bound to now; it is also owned when its '
        "filing is due and the command's stuck-filing record is pending; a task an earlier episode was bound to "
        f'(prior_task_ids) owns nothing; a filing due for less than {settings.recheck_secs:g}s is pending, '
        'not stranded',
        Check.OFFERED_DEBTS_OWNED_OR_DUE: 'every test a windowed effect block lists in debts_offered has a debt '
        'row with an owner task, or is still on the due list; due items older than '
        f'{DUE_ITEM_STALE_MINUTES} minutes are reported',
    }


def _judged(
    check: Check, rule: str, examined: int, failures: Sequence[str] = (), notes: Sequence[str] = (),
) -> CheckResult:
    return CheckResult(check, _verdict(examined, failures), examined, rule, tuple(failures), tuple(notes))


def _verdict(examined: int, failures: Sequence[str]) -> Verdict:
    if examined == 0:
        return Verdict.NO_DATA
    return Verdict.FAIL if failures else Verdict.PASS


CheckFn = Callable[[SoakInputs, str, Settings], CheckResult]


def _label(window_pass: WindowPass) -> str:
    return f'{window_pass.stored.command_key}/{window_pass.pass_id}'


def _interrupted(window_pass: WindowPass) -> bool:
    """A header with no sub-run, or sub-runs and no effect block: a restart cut the pass short."""
    record = window_pass.stored.record
    return record is not None and (record.detect is None or record.effect is None)


def _records_well_formed(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    failures = [f'{_label(p)}: {problem}' for p in inputs.passes for problem in p.problems]
    failures += [
        f'{_label(p)}: the {phase} sub-run output file is missing' for p in inputs.passes for phase in p.missing_outputs
    ]
    notes = [f'interrupted: {_label(p)}' for p in inputs.passes if _interrupted(p)]
    return _judged(Check.RECORDS_WELL_FORMED, rule, len(inputs.passes), failures, notes)


def _filing_problem(task: Mapping[str, Any] | None) -> str | None:
    if task is None:
        return 'get_task found no such task'
    details = task.get('details')
    if not isinstance(details, str) or not details.strip():
        return 'its details digest is empty'
    metadata = task.get('metadata')
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get(LANE_METADATA_KEY), Mapping):
        return f'it carries no metadata.{LANE_METADATA_KEY}'
    return None


def _first_filings(inputs: SoakInputs) -> dict[str, WindowPass]:
    """Each task id the windowed effect blocks list in tasks_filed, and the first pass that lists it."""
    filed_by: dict[str, WindowPass] = {}
    for window_pass, effect in inputs.effects():
        for task_id in effect.tasks_filed:
            filed_by.setdefault(task_id, window_pass)
    return filed_by


def _filings_resolve(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    filed_by = _first_filings(inputs)
    failures = [
        f'task {task_id} (filed by {_label(window_pass)}): {problem}'
        for task_id, window_pass in filed_by.items()
        if (problem := _filing_problem(inputs.tasks.get(task_id))) is not None
    ]
    return _judged(Check.FILINGS_RESOLVE, rule, len(filed_by), failures)


def _filings_bounded_by_threshold(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    blocks = [effect for _, effect in inputs.effects() if effect.tasks_filed or effect.threshold_reached]
    filed = len({task_id for effect in blocks for task_id in effect.tasks_filed})
    crossings = sum(len(set(effect.threshold_reached)) for effect in blocks)
    counts = f'{filed} distinct task(s) filed; {crossings} (pass, test) threshold crossing(s)'
    if filed > crossings:
        return _judged(Check.FILINGS_BOUNDED_BY_THRESHOLD, rule, len(blocks), failures=[counts])
    return _judged(Check.FILINGS_BOUNDED_BY_THRESHOLD, rule, len(blocks), notes=[counts])


def _lane_rows(inputs: SoakInputs) -> set[tuple[str, datetime, str]]:
    return {
        (row.test_id, datetime.fromisoformat(row.observed_at), row.verdict)
        for row in inputs.ledger.occurrences
        if row.call_site == FlakeCallSite.offline_lane
    }


def _owed_rows(record: PassRecord, outcome: Confirmed) -> list[tuple[str, FlakeVerdict]]:
    exempt = {unconfirmable_node(record.command), UNKNOWN_TEST_ID}
    owed = [(node, FlakeVerdict.fails_in_isolation) for node in outcome.reproduced]
    owed += [(node, FlakeVerdict.passes_in_isolation) for node in outcome.cleared]
    return [(node, verdict) for node, verdict in owed if node not in exempt]


def _ledger_rows_per_pass(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    rows = _lane_rows(inputs)
    examined = 0
    failures: list[str] = []
    for window_pass in inputs.passes:
        record = window_pass.stored.record
        if record is None or not isinstance(window_pass.outcome, Confirmed):
            continue
        owed = _owed_rows(record, window_pass.outcome)
        examined += len(owed)
        missing = [f'{node} ({verdict})' for node, verdict in owed if (node, record.started_at, verdict) not in rows]
        if missing:
            failures.append(f'{_label(window_pass)}: no offline_lane row for ' + ', '.join(missing))
    return _judged(Check.LEDGER_ROWS_PER_PASS, rule, examined, failures)


def _live_episodes(inputs: SoakInputs) -> Iterator[Episode]:
    for record in inputs.episodes.values():
        yield from (episode for episode in record.episodes.values() if episode.live)


@dataclass(frozen=True)
class _Crossing:
    """One (pass, test) entry of a windowed effect block's threshold_reached, and the episode,
    live or ended, and stuck-filing record state the lane holds for it now."""

    window_pass: WindowPass
    command: str
    node: str
    episode: Episode | None
    stuck_record_state: str | None


def _crossings(inputs: SoakInputs) -> Iterator[_Crossing]:
    for window_pass, effect in inputs.effects():
        record = window_pass.stored.record
        if record is None:
            continue
        held = inputs.episodes.get(record.command) or CommandRecord.empty(record.command)
        stuck_id = held.stuck_filing_escalation_id
        state = None if stuck_id is None else inputs.escalations.stuck_filing_states.get(stuck_id)
        for node in dict.fromkeys(effect.threshold_reached):
            yield _Crossing(window_pass, record.command, node, held.episodes.get(node), state)


def _lane_task_covers(inputs: SoakInputs, task_id: str, command: str, node: str) -> bool:
    """Whether *task_id* reads back as a lane task covering *node* of *command*."""
    task = inputs.tasks.get(task_id)
    return task is not None and covering_lane_task({task_id: task}, command, node) == task_id


def _owner_candidates(
    inputs: SoakInputs, crossing: _Crossing, bindings: Mapping[tuple[str, str], str],
) -> Iterator[str]:
    """The tasks a same-command effect block at or after the crossing filed, then the episode's binding now."""
    crossed = crossing.window_pass
    for window_pass, effect in inputs.effects():
        if window_pass.stored.command_key == crossed.stored.command_key and window_pass.started_at >= crossed.started_at:
            yield from effect.tasks_filed
    bound = bindings.get((crossing.command, crossing.node))
    if bound is not None:
        yield bound


def _owned(inputs: SoakInputs, crossing: _Crossing, bindings: Mapping[tuple[str, str], str]) -> bool:
    candidates = _owner_candidates(inputs, crossing, bindings)
    if any(_lane_task_covers(inputs, task_id, crossing.command, crossing.node) for task_id in candidates):
        return True
    episode = crossing.episode
    return episode is not None and episode.filing_due and crossing.stuck_record_state == PENDING_ESCALATION


def _due_within(crossing: _Crossing, now: datetime, recheck: timedelta) -> datetime | None:
    """When the crossing's filing fell due, if that is less than *recheck* before *now*."""
    episode = crossing.episode
    if episode is None or not episode.filing_due or episode.filing_due_since is None:
        return None
    return episode.filing_due_since if now - episode.filing_due_since < recheck else None


def _no_stranded_filing(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    bindings = episode_bindings(inputs.episodes)
    recheck = timedelta(seconds=settings.recheck_secs)
    crossings = list(_crossings(inputs))
    failures: list[str] = []
    notes: list[str] = []
    for crossing in crossings:
        if _owned(inputs, crossing, bindings):
            continue
        what = f'{crossing.command}: {crossing.node} (threshold reached at {_label(crossing.window_pass)})'
        due_since = _due_within(crossing, inputs.now, recheck)
        if due_since is not None:
            notes.append(f'pending: {what}, filing due since {due_since.isoformat()}')
        else:
            failures.append(f'{what} has no covering lane task and no pending stuck-filing record')
    return _judged(Check.NO_STRANDED_FILING, rule, len(crossings), failures, notes)


def _offered(inputs: SoakInputs) -> Iterator[tuple[WindowPass, PassRecord, OfferedDebt]]:
    for window_pass, effect in inputs.effects():
        record = window_pass.stored.record
        if record is not None:
            for debt in effect.debts_offered:
                yield window_pass, record, debt


def _metadata(task: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = task.get('metadata')
    return metadata if isinstance(metadata, Mapping) else {}


def _text(task: Mapping[str, Any]) -> str:
    return '\n'.join(value for key in ('description', 'details') if isinstance(value := task.get(key), str))


def _evidence_problem(owner: Mapping[str, Any], failures: Sequence[FailureRecord]) -> str | None:
    trigger = _metadata(owner).get(DEBT_TRIGGER_KEY)
    if not isinstance(trigger, str) or not trigger:
        return f'it carries no metadata.{DEBT_TRIGGER_KEY}'
    text = _text(owner)
    if not any(failure.kind in text and failure.first_line in text for failure in failures):
        return "its text carries no failure kind and first line of the test from the lane's pass records"
    return None


def _offered_owners(inputs: SoakInputs) -> Iterator[tuple[str, Mapping[str, Any], str, str]]:
    """(owner id, owner task, command, node) for each debt a windowed effect block offered whose owner read back."""
    for _, record, debt in _offered(inputs):
        row = inputs.ledger.debts.get(debt.node)
        owner_id = None if row is None else row.owner_task_id
        owner = None if owner_id is None else inputs.tasks.get(owner_id)
        if owner is not None:
            yield str(owner_id), owner, record.command, debt.node


def _deflake_tasks_carry_evidence(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    examined: dict[str, tuple[Mapping[str, Any], str, str]] = {}
    unplaced: dict[str, str] = {}
    for owner_id, owner, command, node in _offered_owners(inputs):
        call_site = _metadata(owner).get(DEBT_CALL_SITE_KEY)
        if call_site == FlakeCallSite.offline_lane:
            examined.setdefault(owner_id, (owner, command, node))
        elif not call_site:
            unplaced.setdefault(owner_id, node)
    failures = [
        f'owner task {owner_id} of {node}: {problem}'
        for owner_id, (owner, command, node) in examined.items()
        if (problem := _evidence_problem(owner, inputs.failures.get((command, node), ()))) is not None
    ]
    notes = [
        f'owner task {owner_id} of {node} carries no metadata.{DEBT_CALL_SITE_KEY}, so its trigger and evidence '
        'are not examined'
        for owner_id, node in unplaced.items()
    ]
    return _judged(Check.DEFLAKE_TASKS_CARRY_EVIDENCE, rule, len(examined), failures, notes)


def _stale_notes(inputs: SoakInputs) -> list[str]:
    stale_before = inputs.now - DUE_ITEM_STALE_AFTER
    return [
        f'stale due item: {item.test_id} at {item.call_site}, enqueued {item.enqueued_at}, '
        f'{item.attempts} attempt(s), last error: {item.last_error or "none"}'
        for item in inputs.ledger.due_items
        if datetime.fromisoformat(item.enqueued_at) <= stale_before
    ]


def _offered_debts_owned_or_due(inputs: SoakInputs, rule: str, settings: Settings) -> CheckResult:
    pending = inputs.ledger.due_summary.pending_since
    offered = list(_offered(inputs))
    failures = [
        f'{_label(window_pass)}: {debt.node} ({debt.trigger.value}) has no debt owner and is not on the due list'
        for window_pass, _, debt in offered
        if debt.node not in pending
        and ((row := inputs.ledger.debts.get(debt.node)) is None or row.owner_task_id is None)
    ]
    return _judged(Check.OFFERED_DEBTS_OWNED_OR_DUE, rule, len(offered), failures, _stale_notes(inputs))


_CHECKS: dict[Check, CheckFn] = {
    Check.RECORDS_WELL_FORMED: _records_well_formed,
    Check.FILINGS_RESOLVE: _filings_resolve,
    Check.FILINGS_BOUNDED_BY_THRESHOLD: _filings_bounded_by_threshold,
    Check.LEDGER_ROWS_PER_PASS: _ledger_rows_per_pass,
    Check.DEFLAKE_TASKS_CARRY_EVIDENCE: _deflake_tasks_carry_evidence,
    Check.NO_STRANDED_FILING: _no_stranded_filing,
    Check.OFFERED_DEBTS_OWNED_OR_DUE: _offered_debts_owned_or_due,
}


def run_checks(inputs: SoakInputs | Unjudgeable, settings: Settings) -> tuple[CheckResult, ...]:
    rules = _rules(settings)
    if isinstance(inputs, Unjudgeable):
        return tuple(_judged(check, rules[check], 0) for check in Check)
    return tuple(_CHECKS[check](inputs, rules[check], settings) for check in Check)


def _passes_per_command_per_day(inputs: SoakInputs) -> Figure:
    days: dict[str, Counter[str]] = defaultdict(Counter)
    for p in inputs.passes:
        days[p.stored.command_key][p.started_at.date().isoformat()] += 1
    return Figure(
        'passes_per_command_per_day',
        'pass directories in the window by command key and the UTC date of their pass-id start; '
        "completeness against an independent count is the resolver's judgment",
        {command: dict(sorted(count.items())) for command, count in sorted(days.items())},
    )


def _interrupted_passes(inputs: SoakInputs) -> Figure:
    return Figure(
        'interrupted_passes',
        'passes in the window with a header and no detect record, or sub-runs and no effect block',
        [{'command': p.stored.command_key, 'pass_id': p.pass_id} for p in inputs.passes if _interrupted(p)],
    )


def _by_count(counter: Counter[int]) -> dict[str, int]:
    return {str(count): counter[count] for count in sorted(counter)}


def _episodes_by_length(inputs: SoakInputs) -> Figure:
    ended: dict[str, Counter[int]] = defaultdict(Counter)
    for _, effect in inputs.effects():
        for episode in effect.episodes_ended:
            ended[episode.reason][episode.counted_passes] += 1
    live = Counter(len(episode.counted_passes) for episode in _live_episodes(inputs))
    return Figure(
        'episodes_by_length',
        'ended: the episodes_ended of windowed effect blocks, by reason and counted passes; '
        'live: the live episodes of the selected commands, by counted passes',
        {'ended': {reason: _by_count(counts) for reason, counts in sorted(ended.items())}, 'live': _by_count(live)},
    )


def _episode_start(
    inputs: SoakInputs, filing: WindowPass, task_id: str, task: Mapping[str, Any] | None,
) -> WindowPass | None:
    """The latest pass of the filing's command, at or before it, that started an episode the task covers."""
    record = filing.stored.record
    if task is None or record is None:
        return None
    covers = {task_id: task}
    starts = [
        window_pass
        for window_pass, effect in inputs.effects()
        if window_pass.stored.command_key == filing.stored.command_key
        and window_pass.started_at <= filing.started_at
        and any(covering_lane_task(covers, record.command, node) == task_id for node in effect.episodes_started)
    ]
    return max(starts, key=lambda window_pass: window_pass.started_at, default=None)


def _filing_delay(inputs: SoakInputs) -> Figure:
    rows = []
    for task_id, filing in _first_filings(inputs).items():
        start = _episode_start(inputs, filing, task_id, inputs.tasks.get(task_id))
        rows.append({
            'task_id': task_id,
            'command': filing.stored.command_key,
            'episode_pass': None if start is None else start.pass_id,
            'filing_pass': filing.pass_id,
            'delay_secs': None if start is None else (filing.started_at - start.started_at).total_seconds(),
        })
    return Figure(
        'filing_delay',
        'per filed task, from the start of the latest windowed pass at or before its filing pass whose '
        'episodes_started holds a test the task covers, to the start of the filing pass (None: no such pass)',
        rows,
    )


def _lane_debts(inputs: SoakInputs) -> Figure:
    rows = []
    for window_pass, _, debt in _offered(inputs):
        row = inputs.ledger.debts.get(debt.node)
        owner = None if row is None else row.owner_task_id
        task = None if owner is None else inputs.tasks.get(owner)
        rows.append({
            'pass_id': window_pass.pass_id,
            'test': debt.node,
            'trigger': debt.trigger.value,
            'evidence_head': debt.head,
            'owner': owner,
            'owner_status': None if task is None else task.get('status'),
        })
    return Figure(
        'lane_debts',
        "each debt a windowed effect block offered, with its debt row's owner task and that task's status now",
        rows,
    )


def _due_list(inputs: SoakInputs) -> Figure:
    items = [
        {
            'test': item.test_id,
            'call_site': item.call_site,
            'age_secs': (inputs.now - datetime.fromisoformat(item.enqueued_at)).total_seconds(),
            'attempts': item.attempts,
            'last_error': item.last_error,
        }
        for item in inputs.ledger.due_items
    ]
    return Figure(
        'due_list',
        f'every item on the flake ledger due list, with its age; stale: items older than {DUE_ITEM_STALE_MINUTES} '
        'minutes',
        {'items': items, 'stale': inputs.ledger.due_summary.stale},
    )


def _undecided_tests(inputs: SoakInputs) -> Figure:
    return Figure(
        'undecided_tests',
        'per windowed pass, the tests its effect block lists in debts_undecided',
        [
            {'command': window_pass.stored.command_key, 'pass_id': window_pass.pass_id,
             'tests': list(effect.debts_undecided)}
            for window_pass, effect in inputs.effects()
            if effect.debts_undecided
        ],
    )


def _due_offers_per_pass(inputs: SoakInputs) -> Figure:
    return Figure(
        'due_offers_per_pass',
        'how many windowed effect blocks offered each number of debts',
        _by_count(Counter(len(effect.debts_offered) for _, effect in inputs.effects())),
    )


def _clearing_passes_per_day(inputs: SoakInputs) -> Figure:
    days: dict[str, Counter[str]] = defaultdict(Counter)
    for p in inputs.passes:
        if isinstance(p.outcome, Confirmed) and p.outcome.cleared:
            days[p.stored.command_key][p.started_at.date().isoformat()] += 1
    return Figure(
        'clearing_passes_per_day',
        'windowed passes whose confirmed outcome cleared a test, by command key and UTC date',
        {command: dict(sorted(count.items())) for command, count in sorted(days.items())},
    )


def _held_reds_ended_elsewhere(inputs: SoakInputs) -> Figure:
    rows = []
    for window_pass, effect in inputs.effects():
        record = window_pass.stored.record
        if record is None:
            continue
        recovered = {debt.node for debt in effect.debts_offered if debt.trigger is DebtTrigger.recovered}
        rows += [
            {
                'command': window_pass.stored.command_key,
                'pass_id': window_pass.pass_id,
                'test': ended.node,
                'head': record.head,
                'recovered_debt_offered': ended.node in recovered,
            }
            for ended in effect.episodes_ended
            if ended.reason == EndReason.PASSED and record.head not in ended.reproduced_heads
        ]
    return Figure(
        'held_reds_ended_elsewhere',
        "episodes a windowed pass ended as passed on a head that never reproduced them, and whether "
        'that pass offered a recovered debt for the test',
        rows,
    )


def _regression_escalations(inputs: SoakInputs) -> Figure:
    return Figure(
        'regression_escalations',
        f'{REGRESSED_AFTER_RESOLUTION} escalations filed in the window whose evidence names call site '
        'offline_lane; unread: escalation files the scan could not read',
        {
            'escalations': [
                {'id': e.id, 'task_id': e.task_id, 'status': e.status, 'timestamp': e.timestamp, 'summary': e.summary}
                for e in inputs.escalations.regressions
            ],
            'unread': [{'path': path, 'reason': reason} for path, reason in inputs.escalations.unread],
        },
    )


def _outage_records(inputs: SoakInputs, outage_passes: Sequence[WindowPass]) -> list[dict[str, Any]]:
    rows = []
    for window_pass in outage_passes:
        for escalation_id in () if window_pass.effect is None else window_pass.effect.escalations_raised:
            escalation = inputs.escalations.outages.get(escalation_id)
            rows.append({
                'id': escalation_id,
                'command': window_pass.stored.command_key,
                'pass_id': window_pass.pass_id,
                'status': None if escalation is None else escalation.status,
                'resolved_by': None if escalation is None else escalation.resolved_by,
                'resolution': None if escalation is None else escalation.resolution,
            })
    return rows


def _outages(inputs: SoakInputs) -> Figure:
    outage_passes = [p for p in inputs.passes if isinstance(p.outcome, Outage)]
    probes: dict[str, Counter[str]] = defaultdict(Counter)
    for p in inputs.passes:
        detect = None if p.stored.record is None else p.stored.record.detect
        if detect is not None and detect.probe is not None:
            probes[p.stored.command_key][detect.probe.result] += 1
    return Figure(
        'outages',
        "windowed outage passes; probe results by kind per command from the detect records' probe; and each "
        'record an outage pass raised, with its status, resolver and resolution (None: not in the queue)',
        {
            'passes': [{'command': p.stored.command_key, 'pass_id': p.pass_id} for p in outage_passes],
            'probe_results': {command: dict(sorted(count.items())) for command, count in sorted(probes.items())},
            'escalations': _outage_records(inputs, outage_passes),
        },
    )


def _record_directory_bytes(inputs: SoakInputs) -> Figure:
    return Figure('record_directory_bytes', 'the size of every file under the run-record root', inputs.record_bytes)


_FIGURES: tuple[Callable[[SoakInputs], Figure], ...] = (
    _passes_per_command_per_day,
    _episodes_by_length,
    _filing_delay,
    _lane_debts,
    _due_list,
    _undecided_tests,
    _due_offers_per_pass,
    _clearing_passes_per_day,
    _held_reds_ended_elsewhere,
    _regression_escalations,
    _outages,
    _interrupted_passes,
    _record_directory_bytes,
)


def figures(inputs: SoakInputs | Unjudgeable) -> tuple[Figure, ...]:
    if isinstance(inputs, Unjudgeable):
        return ()
    return tuple(figure(inputs) for figure in _FIGURES)


def exit_code(inputs: SoakInputs | Unjudgeable, results: Sequence[CheckResult]) -> int:
    if isinstance(inputs, Unjudgeable):
        return 2
    return 1 if any(result.verdict is Verdict.FAIL for result in results) else 0


def verdict_lines(results: Sequence[CheckResult]) -> list[str]:
    lines: list[str] = []
    for result in results:
        lines.append(f'[{result.verdict.upper()}] {result.name}: examined {result.examined}; rule: {result.rule}')
        lines.extend(f'  - {failure}' for failure in result.failures)
    return lines


def _report_json(
    inputs: SoakInputs | Unjudgeable, now: datetime, results: Sequence[CheckResult], shown: Sequence[Figure],
) -> dict[str, Any]:
    judged = not isinstance(inputs, Unjudgeable)
    return {
        'exit_code': exit_code(inputs, results),
        'unjudgeable': None if judged else inputs.reason,
        'since': inputs.since.isoformat() if judged else None,
        'now': now.isoformat(),
        'commands': list(inputs.commands) if judged else [],
        'checks': [
            {
                'name': r.name.value, 'verdict': r.verdict.value, 'examined': r.examined, 'rule': r.rule,
                'failures': list(r.failures), 'notes': list(r.notes),
            }
            for r in results
        ],
        'figures': [{'name': f.name, 'rule': f.rule, 'value': f.value} for f in shown],
    }


def render_figures(shown: Sequence[Figure]) -> str:
    return ''.join(f'{f.name} ({f.rule}):\n  {json.dumps(f.value, sort_keys=True)}\n' for f in shown)


def _instant(text: str) -> datetime:
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'not an ISO-8601 instant: {text!r}') from exc
    if instant.tzinfo is None:
        raise argparse.ArgumentTypeError(f'--since needs a UTC offset: {text!r}')
    return instant


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The offline-lane soak report: zeta's checks and figures.")
    parser.add_argument('--project-root', type=Path, default=None,
                        help='project whose lane to judge (default: the main checkout)')
    parser.add_argument('--since', type=_instant, default=None,
                        help='window start, ISO-8601 with an offset (default: the first effect-carrying pass)')
    parser.add_argument('--command', action='append', default=[], dest='commands',
                        help='a lane command name to judge; repeatable (default: every command with records)')
    parser.add_argument('--json', action='store_true', help='print the full report as JSON on stdout')
    parser.add_argument('--server-url', default=DEFAULT_SERVER,
                        help=f'fused-memory MCP server for task reads (default: {DEFAULT_SERVER})')
    parser.add_argument('--recheck-secs', type=float, default=LanePolicy().recheck_secs,
                        help="how long a due filing may stay due before it is stranded (default: the lane's)")
    return parser


def main(
    argv: Sequence[str] | None = None, *, task_reader: TaskReader | None = None, now: datetime | None = None,
) -> int:
    args = _parser().parse_args(argv)
    now = now or datetime.now(UTC)
    project_root: Path = args.project_root or default_project_root()
    settings = Settings(recheck_secs=args.recheck_secs)
    if task_reader is None:
        loading = load_inputs_from_server(
            project_root, server_url=args.server_url, since=args.since, commands=args.commands, now=now,
        )
    else:
        loading = load_inputs(project_root, since=args.since, commands=args.commands, now=now, task_reader=task_reader)
    inputs = asyncio.run(loading)
    results = run_checks(inputs, settings)
    shown = figures(inputs)
    if isinstance(inputs, Unjudgeable):
        print(f'offline-lane soak: nothing to judge: {inputs.reason}', file=sys.stderr)
    print('\n'.join(verdict_lines(results)), file=sys.stderr)
    if args.json:
        print(json.dumps(_report_json(inputs, now, results, shown), indent=2, sort_keys=True))
    else:
        print(render_figures(shown), end='')
    return exit_code(inputs, results)


if __name__ == '__main__':
    sys.exit(main())
