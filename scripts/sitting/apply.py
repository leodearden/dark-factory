#!/usr/bin/env python3
"""The store-writing half of the sitting tooling: apply Leo's answers to the stores of record.

``prepare_sitting.py`` reads the stores and never writes them; every mutation
it prepares is an ``ApplyPayload``. This module executes those payloads, so
the read-only contract over there stands.

``apply`` resolves Leo's numbered answers against the sitting ledger, then
pre-flights every answered escalation on the server serving its queue
(``sitting.ports``) BEFORE anything is written: one refused item refuses the
whole batch. Each item then runs the skill's order (stamp the markers,
``resolve_issue`` as the coordinator, stamp ``x_ruling``, close the
DecisionRecord, write two memories), and a failed step skips the rest of that
item only.

``audit`` records the sitting's ``ok`` / ``wrong`` verdict on each drawn
executor close (``sitting.audits``). A ``wrong`` re-counts the class's audits on
that queue from the registry and, under ``authority.audit_escape_target``,
moves the class to shadow in the queue's policy-table state.

``policy-state`` reads or moves one queue's class switches; the moves it may
make are ``authority.CLASS_SWITCH_TRANSITIONS``, enforced by
``escalation.policy_state.set_class_state``, its only write path.

Exit codes: 0 applied; 1 a step failed after pre-flight; 2 a configuration or
validation error, nothing written; 3 ask back, a token did not resolve and
nothing was applied; 4 refused, either at pre-flight (nothing written, the
ledger included) or by the server mid-apply.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

# Bind scripts/ and orchestrator/ to THIS checkout, never an editable install's (tasks 2881/2882).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / 'orchestrator' / 'src', _REPO_ROOT / 'scripts'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dashboard.config import DEFAULT_FUSED_MEMORY_URLS  # noqa: E402
from escalation.authority import (  # noqa: E402
    COORDINATOR_IDENTITY,
    ClassSwitch,
    IllegalClassTransition,
    audit_escape_target,
    coordinator_denied,
)
from escalation.models import Escalation  # noqa: E402
from orchestrator.session_registry import (  # noqa: E402
    AuditVerdict,
    DecisionAuditRefused,
    DecisionRecord,
    DecisionState,
    decision_path_for_id,
    fleet_root,
    normalize_escalations_dir,
    record_audit_outcome,
)
from shared.mcp_session import McpCallError, SyncMcpSession  # noqa: E402
from sitting import audits, brief, payloads, ports  # noqa: E402
from sitting import ledger as ledgers  # noqa: E402
from sitting.inventory import AUDIT_KIND, index_queue, key_str, parse_key  # noqa: E402
from sitting.payloads import AgreedMarker, ApplyPayload, PreparedMarker, Ruling  # noqa: E402
from sitting.ports import PortUnknown, QueuePort  # noqa: E402
from sitting.preparation import (  # noqa: E402
    NIGHTLY_CONFINEMENT_ENV,
    NoLean,
    Preparation,
    Recommended,
)
from sitting.prepare_sitting import (  # noqa: E402
    AnswerTurn,
    Sources,
    answer_turn,
    clock_arguments,
    parse_answer_token,
    sources_from_args,
    stores_arguments,
)

from escalation import policy_state  # noqa: E402

EXIT_OK, EXIT_STEP_FAILED, EXIT_CONFIG, EXIT_ASK_BACK, EXIT_REFUSED = 0, 1, 2, 3, 4

IDENTITY_HEADER = 'X-Escalation-Identity'
CLIENT_NAME = 'sitting-apply'
MEMORY_AGENT_ID = 'sitting-apply'
CLOSING_ACTIONS: tuple[str, ...] = tuple(action for action in payloads.RESOLVE_ACTIONS if action != 'park')
"""The resolve actions that close a record; ``park`` leaves it open, so it is no answer to apply."""

ESCALATION_TOOLS = frozenset({'stamp_triage', 'resolve_issue', 'get_escalation'})
FUSED_MEMORY_TOOLS = frozenset({'get_task', 'update_task', 'add_memory'})
STEP_NAMES: Mapping[str, str] = {payloads.SESSION_REGISTRY_TOOL: 'close_decision'}

StepOutcome = Literal['ok', 'failed', 'refused', 'skipped']


class ToolCaller(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict: ...


Connect = Callable[[QueuePort], AbstractContextManager[ToolCaller]]


@dataclass(frozen=True)
class ApplyReport:
    """What ``apply`` did: its exit code, one JSON-ready row per item, the echo table, and stderr messages."""

    exit_code: int
    rows: tuple[dict[str, Any], ...] = ()
    echo: str = ''
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Target:
    """An answered escalation as pre-flight reads it: its number, its queue, and the server serving that queue."""

    number: int
    queue_dir: str
    escalation_id: str
    port: QueuePort | PortUnknown
    still_open: bool


@dataclass(frozen=True)
class _Skip:
    reason: str


def coordinator_session(port: QueuePort) -> SyncMcpSession:
    """A session on *port*'s server asserting the coordinator identity."""
    return SyncMcpSession(port.base_url, client_name=CLIENT_NAME, headers={IDENTITY_HEADER: COORDINATOR_IDENTITY})


def run_apply(
    sources: Sources,
    ledger_path: Path,
    answers: Sequence[ledgers.Answer],
    actions: Mapping[str, str],
    *,
    now: datetime,
    fused_memory: ToolCaller,
    connect: Connect,
    fleet_root: Path,
) -> ApplyReport:
    """Resolve *answers*, pre-flight every answered escalation, then apply each item; see the module docstring."""
    loaded = ledgers.load(ledger_path)
    if loaded is None:
        return ApplyReport(EXIT_CONFIG, messages=(f'no sitting at {ledger_path}; run new-sitting first',))
    turn = answer_turn(sources, loaded, answers, now=now)
    gone = _gone_escalations(loaded, turn)
    ask_back = tuple(u for u in turn.resolved.unresolved if ledgers.key_for_ref(loaded, u.answer.item_ref) not in gone)
    if ask_back:
        ledgers.save(ledger_path, turn.persisted)
        echo = ledgers.render_echo_table(ledgers.ResolvedAnswers(turn.resolved.rows, ask_back))
        return ApplyReport(EXIT_ASK_BACK, echo=echo)
    targets = [*_open_targets(turn), *_gone_targets(loaded, gone, answers)]
    problems = _action_problems(turn, targets, actions)
    if problems:
        return ApplyReport(EXIT_CONFIG, messages=tuple(problems))
    with contextlib.ExitStack() as stack:
        sessions = _Sessions(connect, stack)
        refusals, records = _preflight(targets, sessions, now)
        if refusals:
            return ApplyReport(EXIT_REFUSED, rows=tuple(refusals), messages=(
                f'nothing applied: {len(refusals)} item(s) refused at pre-flight',))
        context = _ApplyContext(turn, sessions, fused_memory, fleet_root, now)
        rows = tuple(
            context.apply_item(row, actions.get(str(row.number)), records.get(row.number))
            for row in turn.resolved.rows
        )
    ledgers.save(ledger_path, turn.persisted)
    return ApplyReport(_exit_code(rows), rows=rows, echo=ledgers.render_echo_table(turn.resolved))


class _Sessions:
    """One escalation session per queue for the whole run, opened on first use."""

    def __init__(self, connect: Connect, stack: contextlib.ExitStack) -> None:
        self._connect = connect
        self._stack = stack
        self._open: dict[str, ToolCaller] = {}

    def for_port(self, port: QueuePort) -> ToolCaller:
        if port.queue_dir not in self._open:
            self._open[port.queue_dir] = self._stack.enter_context(self._connect(port))
        return self._open[port.queue_dir]


def _gone_escalations(loaded: ledgers.Ledger, turn: AnswerTurn) -> frozenset[str]:
    """Escalation keys open when Leo read the ledger whose record has since left the open set."""
    return frozenset(
        key for key, entry in loaded.entries.items()
        if parse_key(key)[0] == 'esc' and entry.state != 'done' and turn.sitting.ledger.entries[key].state == 'done'
    )


def _open_targets(turn: AnswerTurn) -> list[_Target]:
    targets = []
    for row in turn.resolved.rows:
        item = _entry(turn, row.key).item
        if item.escalation_id:
            targets.append(_target(row.number, item.queue_dir, item.escalation_id, still_open=True))
    return targets


def _gone_targets(loaded: ledgers.Ledger, gone: frozenset[str], answers: Sequence[ledgers.Answer]) -> list[_Target]:
    keys = {ledgers.key_for_ref(loaded, answer.item_ref) for answer in answers} & gone
    targets = []
    for key in sorted(keys, key=lambda key: loaded.entries[key].number):
        _, queue_dir, esc_id = parse_key(key)
        targets.append(_target(loaded.entries[key].number, queue_dir, esc_id, still_open=False))
    return targets


def _target(number: int, queue_dir: str, escalation_id: str, *, still_open: bool) -> _Target:
    return _Target(number, queue_dir, escalation_id, ports.queue_port(queue_dir), still_open)


def _action_problems(turn: AnswerTurn, targets: Sequence[_Target], actions: Mapping[str, str]) -> list[str]:
    answered = [row.number for row in turn.resolved.rows] + [t.number for t in targets if not t.still_open]
    needing = {t.number for t in targets}
    problems = [f'item {n} is answered more than once in this turn' for n in sorted(set(answered))
                if answered.count(n) > 1]
    problems += [f'--action {ref}={action}: item {ref} is not answered in this turn'
                 for ref, action in actions.items() if not ref.isdigit() or int(ref) not in answered]
    problems += [f'--action {ref}={action}: item {ref} is a decision, not an escalation; it takes no action'
                 for ref, action in actions.items() if ref.isdigit() and int(ref) in set(answered) - needing]
    problems += [f'--action {ref}={action}: {action!r} is not one of {list(CLOSING_ACTIONS)}'
                 for ref, action in actions.items() if action not in CLOSING_ACTIONS]
    problems += [f'item {n} is an escalation; give its resolve action with --action {n}=ACTION '
                 f'(one of {list(CLOSING_ACTIONS)})' for n in sorted(needing) if str(n) not in actions]
    return problems


def _preflight(
    targets: Sequence[_Target], sessions: _Sessions, now: datetime,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    refusals: list[dict[str, Any]] = []
    records: dict[int, dict[str, Any]] = {}
    for target in targets:
        record, refusal = _preflight_one(target, sessions, now)
        if refusal is not None:
            refusals.append({'number': target.number, 'escalation_id': target.escalation_id,
                             'queue_dir': target.queue_dir, 'refusal': refusal})
        elif record is not None:
            records[target.number] = record
    return refusals, records


def _preflight_one(
    target: _Target, sessions: _Sessions, now: datetime,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    port = target.port
    if isinstance(port, PortUnknown):
        return None, {'code': 'no_port', 'reason': port.reason}
    try:
        record = sessions.for_port(port).call_tool('get_escalation', _get_escalation_args(port, target))
    except McpCallError as exc:
        return None, {'code': 'unreachable', 'reason': str(exc)}
    if 'error' in record:
        return None, {'code': record.get('code', 'not_found'), 'reason': record['error']}
    status = record.get('status')
    if status != 'pending':
        return None, {'code': 'not_pending', 'status': status}
    if not target.still_open:
        return None, {'code': 'not_open', 'status': status}
    esc = Escalation.from_dict(record)
    if coordinator_denied(esc, now):
        return None, {'code': 'claimed_by_executor', 'policy_class': esc.policy_class, 'claimed_at': esc.claimed_at}
    return record, None


def _get_escalation_args(port: QueuePort, target: _Target) -> dict[str, Any]:
    scoped = {'project_root': str(port.project_root)} if port.kind == 'project' and port.project_root else {}
    return {'escalation_id': target.escalation_id, **scoped}


class _ApplyContext:
    """Everything one item's steps need; builds each step's payload and runs it through one dispatcher."""

    def __init__(
        self, turn: AnswerTurn, sessions: _Sessions, fused_memory: ToolCaller, fleet_root: Path, now: datetime,
    ) -> None:
        self._turn = turn
        self._sessions = sessions
        self._fused = fused_memory
        self._fleet_root = fleet_root
        self._now = now

    def apply_item(self, row: ledgers.EchoRow, action: str | None, record: Mapping[str, Any] | None) -> dict[str, Any]:
        entry = _entry(self._turn, row.key)
        item = entry.item
        port = ports.queue_port(item.queue_dir) if item.escalation_id else None
        ruling_text = payloads.ruling_resolution(row.option_label, row.option_text, row.note)
        steps: list[dict[str, Any]] = []
        halted = False
        for name, build in self._plan(entry, row, action, record, port, ruling_text):
            if halted:
                steps.append(_step(name, 'skipped', 'an earlier step did not complete'))
                continue
            outcome = self._run(name, build, port)
            steps.append(outcome)
            halted = outcome['outcome'] in ('failed', 'refused')
        return {'number': row.number, 'escalation_id': item.escalation_id, 'decision_id': item.decision_id,
                'queue_dir': item.queue_dir, 'steps': steps}

    def _plan(
        self,
        entry: brief.BriefEntry,
        row: ledgers.EchoRow,
        action: str | None,
        record: Mapping[str, Any] | None,
        port: QueuePort | PortUnknown | None,
        ruling_text: str,
    ) -> list[tuple[str, Callable[[], ApplyPayload | _Skip]]]:
        item = entry.item
        turns = ledgers.resolution_turns(self._turn.persisted, row.key)
        no_escalation = _Skip('a decision item has no escalation record')

        def stamp() -> ApplyPayload | _Skip:
            if not item.escalation_id or record is None:
                return no_escalation
            prepared = _prepared_marker(entry.preparation, self._turn.sitting.ledger.sitting_id)
            agreed = (AgreedMarker.for_answer(prepared, row.option_label, answer_rounds=turns, at=row.answered_at)
                      if prepared is not None else AgreedMarker(row.option_label, None, turns, row.answered_at))
            return payloads.stamp_markers_payload(item.escalation_id, record.get('triage_note') or '', prepared, agreed)

        def resolve() -> ApplyPayload | _Skip:
            if not item.escalation_id or action is None:
                return no_escalation
            return payloads.resolve_issue_payload(item.escalation_id, ruling_text, action,
                                                  resolved_by=COORDINATOR_IDENTITY, resolution_turns=turns)

        def x_ruling() -> ApplyPayload | _Skip:
            if not item.escalation_id or action is None:
                return no_escalation
            if not isinstance(port, QueuePort) or port.kind != 'project' or port.project_root is None:
                return _Skip('a reconciliation queue names no task of its own project')
            if not item.task_id:
                return _Skip('the escalation names no task')
            root = str(port.project_root)
            task = self._fused.call_tool('get_task', {'id': item.task_id, 'project_root': root})
            existing = (task.get('metadata') or {}).get(payloads.X_RULING_KEY)
            ruling = Ruling(item.escalation_id, action, ruling_text, COORDINATOR_IDENTITY, self._now.isoformat())
            return payloads.update_task_payload(item.task_id, root, ruling, existing=existing)

        def close() -> ApplyPayload:
            evidence = (f'{ruling_text}\nAnswered at the sitting: item {row.number} = option {row.option_label} '
                        f'on {row.record_id} at {row.answered_at}')
            return payloads.close_decision_argv(item, DecisionState.ANSWERED, evidence)

        decision, why = payloads.memory_payloads(
            item, option_label=row.option_label, option_text=row.option_text, note=row.note,
            rationale=_rationale(entry.preparation, row), agent_id=MEMORY_AGENT_ID,
        )
        return [
            ('stamp_triage', stamp), ('resolve_issue', resolve), ('update_task', x_ruling),
            ('close_decision', close), ('add_memory', lambda: decision), ('add_memory', lambda: why),
        ]

    def _run(
        self, name: str, build: Callable[[], ApplyPayload | _Skip], port: QueuePort | PortUnknown | None,
    ) -> dict[str, Any]:
        try:
            payload = build()
            if isinstance(payload, _Skip):
                return _step(name, 'skipped', payload.reason)
            return self._execute(name, payload, port)
        except McpCallError as exc:
            return _step(name, 'failed', str(exc))

    def _execute(self, name: str, payload: ApplyPayload, port: QueuePort | PortUnknown | None) -> dict[str, Any]:
        if payload.tool == payloads.SESSION_REGISTRY_TOOL:
            return _run_registry(name, payload, self._fleet_root)
        if payload.tool in ESCALATION_TOOLS and isinstance(port, QueuePort):
            reply = self._sessions.for_port(port).call_tool(payload.tool, payload.args)
        elif payload.tool in FUSED_MEMORY_TOOLS:
            reply = self._fused.call_tool(payload.tool, payload.args)
        else:
            raise ValueError(f'{payload.tool!r} has no executor in apply')
        if reply.get('code') == 'claimed_by_executor':
            return {**_step(name, 'refused', str(reply.get('error', ''))), 'code': 'claimed_by_executor'}
        if 'error' in reply:
            return _step(name, 'failed', str(reply['error']))
        return _step(name, 'ok')


def _run_registry(name: str, payload: ApplyPayload, fleet_root: Path) -> dict[str, Any]:
    env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(fleet_root)}
    for argv in payload.args:
        done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=120, check=False)
        if done.returncode != 0:
            return _step(name, 'failed', f'{argv[2]} exited {done.returncode}: {done.stderr.strip()[-400:]}')
    return _step(name, 'ok')


def _step(name: str, outcome: StepOutcome, detail: str = '') -> dict[str, Any]:
    return {'step': name, 'outcome': outcome, **({'detail': detail} if detail else {})}


def _exit_code(rows: Sequence[Mapping[str, Any]]) -> int:
    outcomes = {step['outcome'] for row in rows for step in row['steps']}
    if 'refused' in outcomes:
        return EXIT_REFUSED
    return EXIT_STEP_FAILED if 'failed' in outcomes else EXIT_OK


def _entry(turn: AnswerTurn, key: str) -> brief.BriefEntry:
    return next(entry for entry in turn.sitting.numbered if key_str(entry.item.key) == key)


def _prepared_marker(prep: Preparation | None, sitting_id: str) -> PreparedMarker | None:
    if prep is None:
        return None
    rec = prep.recommendation
    return PreparedMarker(
        recommendation=rec.option_label if isinstance(rec, Recommended) else '',
        no_lean_reason=rec.reason if isinstance(rec, NoLean) else '',
        sitting_id=sitting_id,
        prepared_at=prep.prepared_at,
    )


def _rationale(prep: Preparation | None, row: ledgers.EchoRow) -> str:
    parts = [row.note.strip()] if row.note.strip() else []
    if prep is not None:
        parts += [option.ramification for option in prep.options if option.label == row.option_label]
    return '; '.join(parts) or 'Leo answered at the sitting without a note'


@dataclass(frozen=True)
class _AuditTarget:
    row: ledgers.EchoRow
    record: DecisionRecord
    close: audits.ExecutorClose


def run_audit(
    ledger_path: Path, answers: Sequence[ledgers.Answer], *, decisions_root: Path,
) -> ApplyReport:
    """Record each ``audit N=ok|wrong[:note]`` verdict; a ``wrong`` may flip its class to shadow on its queue.

    The ledger is read, never saved: an audited close leaves the next draw
    because its DecisionRecord now carries an outcome.
    """
    loaded = ledgers.load(ledger_path)
    if loaded is None:
        return ApplyReport(EXIT_CONFIG, messages=(f'no sitting at {ledger_path}',))
    options = {key: audits.AUDIT_OPTIONS for key, entry in loaded.entries.items()
               if parse_key(key)[0] == AUDIT_KIND and entry.state == 'open'}
    _, resolved = ledgers.resolve_answers(loaded, answers, options)
    if resolved.unresolved:
        return ApplyReport(EXIT_ASK_BACK, echo=ledgers.render_echo_table(resolved))
    numbers = [row.number for row in resolved.rows]
    repeated = sorted({n for n in numbers if numbers.count(n) > 1})
    if repeated:
        return ApplyReport(EXIT_CONFIG, messages=tuple(f'audit {n} is answered more than once' for n in repeated))
    targets, refusals = _audit_preflight(resolved.rows, decisions_root)
    if refusals:
        return ApplyReport(EXIT_REFUSED, rows=tuple(refusals), messages=(
            f'nothing audited: {len(refusals)} item(s) refused at pre-flight',))
    try:
        for queue_dir in {target.close.queue_dir for target in targets}:
            policy_state.load_class_states(Path(queue_dir))
    except policy_state.PolicyStateUnreadable as exc:
        return ApplyReport(EXIT_CONFIG, messages=(f'nothing audited: {exc}',))
    rows = tuple(_audit_one(target, decisions_root) for target in targets)
    failed = any(row['outcome'] != 'ok' for row in rows)
    return ApplyReport(EXIT_STEP_FAILED if failed else EXIT_OK, rows=rows,
                       echo=ledgers.render_echo_table(resolved))


def _audit_preflight(
    rows: Sequence[ledgers.EchoRow], decisions_root: Path,
) -> tuple[list[_AuditTarget], list[dict[str, Any]]]:
    targets: list[_AuditTarget] = []
    refusals: list[dict[str, Any]] = []
    for row in rows:
        target = _audit_target(row, decisions_root)
        if isinstance(target, _AuditTarget):
            targets.append(target)
        else:
            refusals.append({'number': row.number, 'decision_id': row.record_id, 'refusal': target})
    return targets, refusals


def _audit_target(row: ledgers.EchoRow, decisions_root: Path) -> _AuditTarget | dict[str, Any]:
    try:
        record = DecisionRecord.from_json(decision_path_for_id(row.record_id, root=decisions_root).read_text())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {'code': 'decision_unreadable', 'reason': f'{type(exc).__name__}: {exc}'}
    if record.audit_outcome is not None:
        return {'code': 'already_audited', 'verdict': str(record.audit_outcome.verdict),
                'audited_at': record.audit_outcome.audited_at}
    queue_dir = normalize_escalations_dir(record.escalations_dir)
    close = audits.executor_close_of(record, {queue_dir: index_queue(queue_dir)})
    if not isinstance(close, audits.ExecutorClose):
        reason = close.reason if close is not None else 'its escalation was not closed by the executor under a class'
        return {'code': 'not_an_executor_close', 'reason': reason}
    return _AuditTarget(row, record, close)


def _audit_one(target: _AuditTarget, decisions_root: Path) -> dict[str, Any]:
    close, row = target.close, target.row
    queue = Path(close.queue_dir)
    verdict = AuditVerdict(row.option_label)
    before = policy_state.load_class_states(queue).get(close.policy_class)
    base = {'number': row.number, 'decision_id': close.decision_id, 'escalation_id': close.escalation_id,
            'policy_class': close.policy_class, 'queue_dir': close.queue_dir, 'verdict': str(verdict),
            'class_state_before': None if before is None else str(before)}
    try:
        written = record_audit_outcome(
            close.decision_id, verdict, row.note, root=decisions_root, expected_project=target.record.project,
            expected_escalations_dir=target.record.escalations_dir, audited_at=row.answered_at,
        )
    except DecisionAuditRefused as exc:
        return {**base, 'outcome': 'failed', 'detail': str(exc), 'class_state_after': base['class_state_before']}
    if written is None:
        return {**base, 'outcome': 'failed', 'detail': 'the decision registry did not take the write',
                'class_state_after': base['class_state_before']}
    after = before
    if verdict == AuditVerdict.WRONG and before is not None:
        index = {close.queue_dir: index_queue(close.queue_dir)}
        audited, wrong = audits.class_tally(decisions_root, index, close.queue_dir, close.policy_class)
        target_state = audit_escape_target(before, audited, wrong)
        if target_state is not None:
            policy_state.set_class_state(queue, close.policy_class, target_state)
            after = target_state
    return {**base, 'outcome': 'ok', 'class_state_after': None if after is None else str(after)}


def run_policy_state(queue_dir: Path, policy_class: str | None, target: ClassSwitch | None) -> ApplyReport:
    """Every class's switch on *queue_dir*'s queue, or one class moved to *target* through set_class_state."""
    if not queue_dir.is_dir():
        return ApplyReport(EXIT_CONFIG, messages=(f'{queue_dir} is not an existing queue directory; '
                                                  'policy-state never creates one',))
    try:
        if policy_class is None or target is None:
            states = policy_state.load_class_states(queue_dir)
            return ApplyReport(EXIT_OK, rows=({name: str(switch) for name, switch in states.items()},))
        before = policy_state.set_class_state(queue_dir, policy_class, target)
    except IllegalClassTransition as exc:
        return ApplyReport(EXIT_REFUSED, messages=(f'refused, nothing written: {exc}',))
    except ValueError as exc:
        return ApplyReport(EXIT_CONFIG, messages=(f'nothing written: {exc}',))
    return ApplyReport(EXIT_OK, rows=({'queue_dir': normalize_escalations_dir(queue_dir), 'policy_class': policy_class,
                                       'before': str(before), 'after': str(target)},))


def parse_action_token(token: str) -> tuple[str, str]:
    """One ``N=ACTION`` token as ``(item number text, action)``; anything else is a ValueError."""
    ref, has_action, action = token.partition('=')
    if not (has_action and ref.strip().isdigit() and action.strip()):
        raise ValueError(f'--action takes N=ACTION, got {token!r}')
    return str(int(ref)), action.strip()


def parse_actions(tokens: Sequence[str]) -> dict[str, str]:
    actions: dict[str, str] = {}
    for token in tokens:
        ref, action = parse_action_token(token)
        if ref in actions:
            raise ValueError(f'--action names item {ref} more than once')
        actions[ref] = action
    return actions


class _LazyMcpSession:
    """A session that handshakes on its first call, so a run refused before any write never connects."""

    def __init__(self, base_url: str, client_name: str) -> None:
        self._base_url = base_url
        self._client_name = client_name
        self._stack = contextlib.ExitStack()
        self._session: SyncMcpSession | None = None

    def __enter__(self) -> _LazyMcpSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stack.close()

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict:
        if self._session is None:
            self._session = self._stack.enter_context(SyncMcpSession(self._base_url, client_name=self._client_name))
        return self._session.call_tool(name, arguments)


def _cmd_apply(args: argparse.Namespace) -> int:
    now = args.now or datetime.now(UTC)
    try:
        actions = parse_actions(args.action)
    except ValueError as exc:
        print(f'apply refused, nothing written: {exc}', file=sys.stderr)
        return EXIT_CONFIG
    answers = [parse_answer_token(token, args.source, now.isoformat()) for token in args.answer]
    with _LazyMcpSession(DEFAULT_FUSED_MEMORY_URLS[0], CLIENT_NAME) as fused:
        report = run_apply(
            sources_from_args(args), args.ledger, answers, actions, now=now, fused_memory=fused,
            connect=coordinator_session, fleet_root=fleet_root(args.decisions_root),
        )
    _print_report(report)
    return report.exit_code


def _cmd_audit(args: argparse.Namespace) -> int:
    now = args.now or datetime.now(UTC)
    answers = [parse_answer_token(token, args.source, now.isoformat()) for token in args.answer]
    report = run_audit(args.ledger, answers, decisions_root=fleet_root(args.decisions_root))
    _print_report(report)
    return report.exit_code


def _cmd_policy_state(args: argparse.Namespace) -> int:
    if (args.policy_class is None) != (args.state is None):
        print('--class and --state are given together or not at all', file=sys.stderr)
        return EXIT_CONFIG
    report = run_policy_state(args.queue_dir, args.policy_class, args.state)
    _print_report(report)
    return report.exit_code


def _print_report(report: ApplyReport) -> None:
    print(report.echo, end='')
    for row in report.rows:
        print(json.dumps(row, sort_keys=True))
    for message in report.messages:
        print(message, file=sys.stderr)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='apply.py', description="Apply a sitting's answers to the stores of record.")
    commands = parser.add_subparsers(dest='command', required=True)
    apply_cmd = commands.add_parser('apply', parents=[stores_arguments()], help="apply Leo's numbered answers")
    apply_cmd.add_argument('--ledger', type=Path, required=True)
    apply_cmd.add_argument('--answer', action='append', required=True, help="repeatable 'N=OPTION[:note]' token")
    apply_cmd.add_argument('--action', action='append', default=[],
                           help=f"repeatable 'N=ACTION' per escalation item; ACTION one of {list(CLOSING_ACTIONS)}")
    apply_cmd.add_argument('--source', choices=sorted(ledgers.ANSWER_SOURCES), default='terminal')
    apply_cmd.set_defaults(handler=_cmd_apply)

    audit_cmd = commands.add_parser('audit', parents=[clock_arguments()], help='record verdicts on drawn audits')
    audit_cmd.add_argument('--ledger', type=Path, required=True, help='the ledger that numbered the audits')
    audit_cmd.add_argument('--answer', action='append', required=True, help="repeatable 'N=ok|wrong[:note]' token")
    audit_cmd.add_argument('--decisions-root', type=Path, help='fleet root holding decisions/ (default: fleet_root())')
    audit_cmd.add_argument('--source', choices=sorted(ledgers.ANSWER_SOURCES), default='terminal')
    audit_cmd.set_defaults(handler=_cmd_audit)

    policy = commands.add_parser('policy-state', help="read or move a policy class's switch on one queue")
    policy.add_argument('--queue-dir', type=Path, required=True, help='an existing escalation queue directory')
    policy.add_argument('--class', dest='policy_class', help='a POLICY_TABLE class; given with --state')
    policy.add_argument('--state', type=ClassSwitch, choices=list(ClassSwitch), help='the switch to move it to')
    policy.set_defaults(handler=_cmd_policy_state)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get(NIGHTLY_CONFINEMENT_ENV):
        print(f'refused under {NIGHTLY_CONFINEMENT_ENV}: apply.py writes stores of record, '
              'which the unattended run never may', file=sys.stderr)
        return EXIT_CONFIG
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except ledgers.LedgerCorrupt as exc:
        print(exc, file=sys.stderr)
        return EXIT_CONFIG


if __name__ == '__main__':
    sys.exit(main())
