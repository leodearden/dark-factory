#!/usr/bin/env python3
"""The store-writing half of the sitting tooling: apply Leo's answers to the stores of record.

``prepare_sitting.py`` reads the stores and never writes them; every mutation
it prepares is an ``ApplyPayload``. This module executes those payloads, so
the read-only contract over there stands.

``apply`` resolves Leo's numbered answers against the sitting ledger, then
pre-flights every answered escalation item on the server serving its queue
(``sitting.ports``) BEFORE anything is written: one refused item refuses the
whole batch. Each item then runs the skill's order (stamp the markers,
``resolve_issue`` as the coordinator, stamp ``x_ruling``, close the
DecisionRecord, write two memories), and a failed step skips the rest of that
item only. A decision item acts on no escalation, even one its DecisionRecord
links to: it closes its own DecisionRecord and writes the two memories.

The markers go on first because the server stamps only a pending record. So
a resolve that is then refused or fails leaves them on the still-pending
record. That is deliberate: they record Leo's answer against the
preparation, which is what the preparer trial counts, whether or not the
answer then applied. A re-run appends a newer pair, and the last pair is the
one read.

``audit`` records the sitting's ``ok`` / ``wrong`` verdict on each drawn
executor close (``sitting.audits``). A ``wrong`` re-counts the class's audits on
that queue and, under ``authority.audit_escape_target``, moves the class to
shadow in the queue's policy-table state.

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
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

# Bind scripts/ and orchestrator/ to THIS checkout, never an editable install's (tasks 2881/2882).
# Bare sys.path calls, the one statement ruff's E402 lets an import follow.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'orchestrator' / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))

from dashboard.config import DEFAULT_FUSED_MEMORY_URLS
from escalation.authority import (
    COORDINATOR_IDENTITY,
    ClassSwitch,
    IllegalClassTransition,
    audit_escape_target,
    coordinator_denied,
)
from escalation.models import Escalation
from orchestrator.session_registry import (
    AuditVerdict,
    DecisionAuditRefused,
    DecisionRecord,
    DecisionState,
    decision_path_for_id,
    fleet_root,
    normalize_escalations_dir,
    record_audit_outcome,
)
from shared.mcp_session import McpCallError, SyncMcpSession
from sitting import audits, brief, payloads, ports
from sitting import ledger as ledgers
from sitting.inventory import AUDIT_KIND, ESCALATION_KIND, index_queue, key_str, parse_key
from sitting.payloads import AgreedMarker, ApplyPayload, PreparedMarker, Ruling
from sitting.ports import PortUnknown, QueuePort
from sitting.preparation import (
    NIGHTLY_CONFINEMENT_ENV,
    NoLean,
    Preparation,
    Recommended,
)
from sitting.prepare_sitting import (
    AnswerTurn,
    Sources,
    answer_turn,
    clock_arguments,
    parse_answer_token,
    sources_from_args,
    stores_arguments,
)

from escalation import policy_state

EXIT_OK, EXIT_STEP_FAILED, EXIT_CONFIG, EXIT_ASK_BACK, EXIT_REFUSED = 0, 1, 2, 3, 4

IDENTITY_HEADER = 'X-Escalation-Identity'
CLIENT_NAME = 'sitting-apply'
MEMORY_AGENT_ID = 'sitting-apply'
REGISTRY_TIMEOUT_SECS = 120
CLOSING_ACTIONS: tuple[str, ...] = tuple(action for action in payloads.RESOLVE_ACTIONS if action != 'park')
"""The resolve actions that close a record; ``park`` leaves it open, so it is no answer to apply."""

ESCALATION_TOOLS = frozenset({'stamp_triage', 'resolve_issue', 'get_escalation'})
FUSED_MEMORY_TOOLS = frozenset({'get_task', 'update_task', 'add_memory'})


class StepOutcome(StrEnum):
    OK = 'ok'
    FAILED = 'failed'
    REFUSED = 'refused'
    SKIPPED = 'skipped'

    @property
    def halts(self) -> bool:
        """A failed or refused step skips the rest of its item."""
        return self in (StepOutcome.FAILED, StepOutcome.REFUSED)


class RefusalCode(StrEnum):
    """Why an item was refused before anything was written, or a step refused by the server."""

    NO_PORT = 'no_port'
    UNREACHABLE = 'unreachable'
    NOT_FOUND = 'not_found'
    NOT_PENDING = 'not_pending'
    NOT_OPEN = 'not_open'
    CLAIMED_BY_EXECUTOR = 'claimed_by_executor'
    DECISION_UNREADABLE = 'decision_unreadable'
    ALREADY_AUDITED = 'already_audited'
    ESCALATION_AGED_OUT = 'escalation_aged_out'
    NOT_AN_EXECUTOR_CLOSE = 'not_an_executor_close'


@dataclass(frozen=True)
class Refusal:
    """A refusal's code, which decides the exit, and the facts the report prints beside it."""

    code: RefusalCode
    facts: Mapping[str, str | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'facts', MappingProxyType(dict(self.facts)))

    def to_json(self) -> dict[str, Any]:
        return {'code': str(self.code), **self.facts}


@dataclass(frozen=True)
class StepResult:
    """One step of one item; ``supersedes`` names the value an ``ok`` write replaced."""

    step: str
    outcome: StepOutcome
    detail: str = ''
    code: RefusalCode | None = None
    supersedes: str = ''

    def to_json(self) -> dict[str, Any]:
        optional = {'detail': self.detail, 'code': self.code and str(self.code), 'supersedes': self.supersedes}
        return {'step': self.step, 'outcome': str(self.outcome), **{k: v for k, v in optional.items() if v}}


@dataclass(frozen=True)
class ItemResult:
    number: int
    escalation_id: str | None
    decision_id: str | None
    queue_dir: str
    steps: tuple[StepResult, ...]

    def to_json(self) -> dict[str, Any]:
        return {'number': self.number, 'escalation_id': self.escalation_id, 'decision_id': self.decision_id,
                'queue_dir': self.queue_dir, 'steps': [step.to_json() for step in self.steps]}


@dataclass(frozen=True)
class RefusedItem:
    """An item refused at pre-flight, named by whichever of its record ids the verb acts on."""

    number: int
    refusal: Refusal
    ids: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'ids', MappingProxyType(dict(self.ids)))

    def to_json(self) -> dict[str, Any]:
        return {'number': self.number, **self.ids, 'refusal': self.refusal.to_json()}


@dataclass(frozen=True)
class AuditResult:
    number: int
    close: audits.ExecutorClose
    verdict: AuditVerdict
    class_state_before: ClassSwitch | None
    class_state_after: ClassSwitch | None
    outcome: StepOutcome = StepOutcome.OK
    detail: str = ''

    def to_json(self) -> dict[str, Any]:
        close = self.close
        row = {
            'number': self.number, 'decision_id': close.decision_id, 'escalation_id': close.escalation_id,
            'policy_class': close.policy_class, 'queue_dir': close.queue_dir, 'verdict': str(self.verdict),
            'class_state_before': _switch_text(self.class_state_before),
            'class_state_after': _switch_text(self.class_state_after), 'outcome': str(self.outcome),
        }
        return {**row, 'detail': self.detail} if self.detail else row


@dataclass(frozen=True)
class ClassStates:
    states: Mapping[str, ClassSwitch]

    def to_json(self) -> dict[str, Any]:
        return {name: str(switch) for name, switch in self.states.items()}


@dataclass(frozen=True)
class ClassMove:
    queue_dir: str
    policy_class: str
    before: ClassSwitch
    after: ClassSwitch

    def to_json(self) -> dict[str, Any]:
        return {'queue_dir': self.queue_dir, 'policy_class': self.policy_class,
                'before': str(self.before), 'after': str(self.after)}


ReportRow = ItemResult | RefusedItem | AuditResult | ClassStates | ClassMove


@dataclass(frozen=True)
class ApplyReport:
    """What a verb did: its exit code, one row per item, the echo table, and stderr messages."""

    exit_code: int
    rows: tuple[ReportRow, ...] = ()
    echo: str = ''
    messages: tuple[str, ...] = ()

    def json_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(row.to_json() for row in self.rows)


class ToolCaller(Protocol):
    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict: ...


Connect = Callable[[QueuePort], AbstractContextManager[ToolCaller]]
ConnectFusedMemory = Callable[[], AbstractContextManager[ToolCaller]]


@dataclass(frozen=True)
class _Target:
    """An answered escalation item as pre-flight reads it: its number, its queue, and the server serving that queue."""

    number: int
    queue_dir: str
    escalation_id: str
    port: QueuePort | PortUnknown
    still_open: bool


@dataclass(frozen=True)
class _Cleared:
    """An escalation pre-flight cleared: the server serving its queue and the record as that server returned it."""

    port: QueuePort
    escalation_id: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class _Skip:
    reason: str


@dataclass(frozen=True)
class _Failed:
    """A step builder's own read failed, so the step fails without being sent."""

    reason: str


def coordinator_session(port: QueuePort) -> SyncMcpSession:
    """A session on *port*'s server asserting the coordinator identity."""
    return SyncMcpSession(port.base_url, client_name=CLIENT_NAME, headers={IDENTITY_HEADER: COORDINATOR_IDENTITY})


def fused_memory_session() -> SyncMcpSession:
    return SyncMcpSession(DEFAULT_FUSED_MEMORY_URLS[0], client_name=CLIENT_NAME)


def run_apply(
    sources: Sources,
    ledger_path: Path,
    answers: Sequence[ledgers.Answer],
    actions: Mapping[str, str],
    *,
    now: datetime,
    fused_memory: ConnectFusedMemory,
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
        sessions = _Sessions(connect, fused_memory, stack)
        cleared, refused = _preflight(targets, sessions, now)
        if refused:
            return ApplyReport(EXIT_REFUSED, rows=tuple(refused), messages=(
                f'nothing applied: {len(refused)} item(s) refused at pre-flight',))
        context = _ApplyContext(turn, sessions, fleet_root, now)
        rows = tuple(
            context.apply_item(row, actions.get(str(row.number)), cleared.get(row.number))
            for row in turn.resolved.rows
        )
    ledgers.save(ledger_path, turn.persisted)
    return ApplyReport(_exit_code(rows), rows=rows, echo=ledgers.render_echo_table(turn.resolved))


class _Sessions:
    """One session per server for the whole run, each opened on its first call, so a refused run opens none it need not."""

    def __init__(self, connect: Connect, connect_fused_memory: ConnectFusedMemory, stack: contextlib.ExitStack) -> None:
        self._connect = connect
        self._connect_fused_memory = connect_fused_memory
        self._stack = stack
        self._open: dict[str | None, ToolCaller] = {}

    def for_port(self, port: QueuePort) -> ToolCaller:
        return self._opened(port.queue_dir, lambda: self._connect(port))

    def fused_memory(self) -> ToolCaller:
        return self._opened(None, self._connect_fused_memory)

    def _opened(self, key: str | None, connect: ConnectFusedMemory) -> ToolCaller:
        if key not in self._open:
            self._open[key] = self._stack.enter_context(connect())
        return self._open[key]


def _gone_escalations(loaded: ledgers.Ledger, turn: AnswerTurn) -> frozenset[str]:
    """Escalation keys open when Leo read the ledger whose record has since left the open set."""
    return frozenset(
        key for key, entry in loaded.entries.items()
        if parse_key(key)[0] == ESCALATION_KIND and entry.state != 'done'
        and turn.sitting.ledger.entries[key].state == 'done'
    )


def _open_targets(turn: AnswerTurn) -> list[_Target]:
    targets = []
    for row in turn.resolved.rows:
        item = _entry(turn, row.key).item
        if item.kind == ESCALATION_KIND and item.escalation_id:
            targets.append(_target(row.number, item.queue_dir, item.escalation_id, still_open=True))
    return targets


def _gone_targets(loaded: ledgers.Ledger, gone: frozenset[str], answers: Sequence[ledgers.Answer]) -> list[_Target]:
    keys = gone.intersection(ledgers.key_for_ref(loaded, answer.item_ref) for answer in answers)
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
) -> tuple[dict[int, _Cleared], list[RefusedItem]]:
    cleared: dict[int, _Cleared] = {}
    refused: list[RefusedItem] = []
    for target in targets:
        outcome = _preflight_one(target, sessions, now)
        if isinstance(outcome, Refusal):
            ids = {'escalation_id': target.escalation_id, 'queue_dir': target.queue_dir}
            refused.append(RefusedItem(target.number, outcome, ids))
        else:
            cleared[target.number] = outcome
    return cleared, refused


def _preflight_one(target: _Target, sessions: _Sessions, now: datetime) -> _Cleared | Refusal:
    port = target.port
    if isinstance(port, PortUnknown):
        return Refusal(RefusalCode.NO_PORT, {'reason': port.reason})
    try:
        record = sessions.for_port(port).call_tool('get_escalation', _get_escalation_args(port, target))
    except McpCallError as exc:
        return Refusal(RefusalCode.UNREACHABLE, {'reason': str(exc)})
    if 'error' in record:
        return Refusal(RefusalCode.NOT_FOUND, {'reason': str(record['error'])})
    status = record.get('status')
    if status != 'pending':
        return Refusal(RefusalCode.NOT_PENDING, {'status': None if status is None else str(status)})
    if not target.still_open:
        return Refusal(RefusalCode.NOT_OPEN, {'status': status})
    esc = Escalation.from_dict(record)
    if coordinator_denied(esc, now):
        return Refusal(RefusalCode.CLAIMED_BY_EXECUTOR, {'policy_class': esc.policy_class, 'claimed_at': esc.claimed_at})
    return _Cleared(port, target.escalation_id, MappingProxyType(record))


def _get_escalation_args(port: QueuePort, target: _Target) -> dict[str, Any]:
    scoped = {'project_root': str(port.project_root)} if port.kind == 'project' and port.project_root else {}
    return {'escalation_id': target.escalation_id, **scoped}


StepBuilder = Callable[[], ApplyPayload | _Skip | _Failed]


class _ApplyContext:
    """Everything one item's steps need; builds each step's payload and runs it through one dispatcher."""

    def __init__(self, turn: AnswerTurn, sessions: _Sessions, fleet_root: Path, now: datetime) -> None:
        self._turn = turn
        self._sessions = sessions
        self._fleet_root = fleet_root
        self._now = now

    def apply_item(self, row: ledgers.EchoRow, action: str | None, cleared: _Cleared | None) -> ItemResult:
        """Run *row*'s steps; *cleared* is its escalation as pre-flight read it, None for a decision item."""
        item = _entry(self._turn, row.key).item
        steps: list[StepResult] = []
        for name, build in self._plan(row, action, cleared):
            if any(step.outcome.halts for step in steps):
                steps.append(StepResult(name, StepOutcome.SKIPPED, 'an earlier step did not complete'))
            else:
                steps.append(self._run(name, build, cleared))
        return ItemResult(row.number, item.escalation_id, item.decision_id, item.queue_dir, tuple(steps))

    def _plan(
        self, row: ledgers.EchoRow, action: str | None, cleared: _Cleared | None,
    ) -> list[tuple[str, StepBuilder]]:
        entry = _entry(self._turn, row.key)
        item = entry.item
        turns = ledgers.resolution_turns(self._turn.persisted, row.key)
        ruling_text = payloads.ruling_resolution(row.option_label, row.option_text, row.note)
        no_escalation = _Skip('a decision item acts on no escalation; it closes its own DecisionRecord')

        def stamp() -> ApplyPayload | _Skip:
            if cleared is None:
                return no_escalation
            prepared = _prepared_marker(entry.preparation, self._turn.sitting.ledger.sitting_id)
            agreed = (AgreedMarker.for_answer(prepared, row.option_label, answer_rounds=turns, at=row.answered_at)
                      if prepared is not None else AgreedMarker(row.option_label, None, turns, row.answered_at))
            note = cleared.record.get('triage_note') or ''
            return payloads.stamp_markers_payload(cleared.escalation_id, note, prepared, agreed)

        def resolve() -> ApplyPayload | _Skip:
            if cleared is None or action is None:
                return no_escalation
            return payloads.resolve_issue_payload(cleared.escalation_id, ruling_text, action,
                                                  resolved_by=COORDINATOR_IDENTITY, resolution_turns=turns)

        def x_ruling() -> ApplyPayload | _Skip | _Failed:
            if cleared is None or action is None:
                return no_escalation
            task_id = item.store_task_id
            if task_id is None:
                return _Skip('the escalation names no task of its queue\'s project: none, a sentinel, '
                             'or a reconciliation gate')
            if cleared.port.project_root is None:
                return _Skip('the queue names no project root')
            root = str(cleared.port.project_root)
            task = self._sessions.fused_memory().call_tool('get_task', {'id': task_id, 'project_root': root})
            if 'error' in task:
                return _Failed(f'get_task {task_id} could not read the prior ruling: {task["error"]}')
            existing = (task.get('metadata') or {}).get(payloads.X_RULING_KEY)
            ruling = Ruling(cleared.escalation_id, action, ruling_text, COORDINATOR_IDENTITY, self._now.isoformat())
            return payloads.update_task_payload(task_id, root, ruling, existing=existing)

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

    def _run(self, name: str, build: StepBuilder, cleared: _Cleared | None) -> StepResult:
        try:
            payload = build()
            if isinstance(payload, _Skip):
                return StepResult(name, StepOutcome.SKIPPED, payload.reason)
            if isinstance(payload, _Failed):
                return StepResult(name, StepOutcome.FAILED, payload.reason)
            return self._execute(name, payload, cleared)
        except McpCallError as exc:
            return StepResult(name, StepOutcome.FAILED, str(exc))

    def _execute(self, name: str, payload: ApplyPayload, cleared: _Cleared | None) -> StepResult:
        """Run *payload*; an ``ok`` result names the value it superseded, so nothing is replaced silently."""
        result = self._dispatch(name, payload, cleared)
        if result.outcome == StepOutcome.OK and payload.supersedes:
            return replace(result, supersedes=payload.supersedes)
        return result

    def _dispatch(self, name: str, payload: ApplyPayload, cleared: _Cleared | None) -> StepResult:
        if payload.tool == payloads.SESSION_REGISTRY_TOOL:
            return _run_registry(name, payload, self._fleet_root)
        if not isinstance(payload.args, dict):
            raise ValueError(f'{payload.tool!r} carries argv lists, which only {payloads.SESSION_REGISTRY_TOOL!r} runs')
        if payload.tool in ESCALATION_TOOLS and cleared is not None:
            reply = self._sessions.for_port(cleared.port).call_tool(payload.tool, payload.args)
        elif payload.tool in FUSED_MEMORY_TOOLS:
            reply = self._sessions.fused_memory().call_tool(payload.tool, payload.args)
        else:
            raise ValueError(f'{payload.tool!r} has no executor in apply')
        if reply.get('code') == RefusalCode.CLAIMED_BY_EXECUTOR:
            return StepResult(name, StepOutcome.REFUSED, str(reply.get('error', '')),
                              code=RefusalCode.CLAIMED_BY_EXECUTOR)
        if 'error' in reply:
            return StepResult(name, StepOutcome.FAILED, str(reply['error']))
        return StepResult(name, StepOutcome.OK)


def _run_registry(name: str, payload: ApplyPayload, fleet_root: Path) -> StepResult:
    """Run the registry verbs in order; one that exits non-zero, times out, or cannot start fails the step."""
    env = {**os.environ, 'CLAUDE_FLEET_ROOT': str(fleet_root)}
    for argv in payload.args:
        try:
            done = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=REGISTRY_TIMEOUT_SECS,
                                  check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            return StepResult(name, StepOutcome.FAILED, f'{argv[2]} did not complete: {type(exc).__name__}: {exc}')
        if done.returncode != 0:
            return StepResult(name, StepOutcome.FAILED, f'{argv[2]} exited {done.returncode}: {done.stderr.strip()[-400:]}')
    return StepResult(name, StepOutcome.OK)


def _exit_code(rows: Sequence[ItemResult]) -> int:
    outcomes = {step.outcome for row in rows for step in row.steps}
    if StepOutcome.REFUSED in outcomes:
        return EXIT_REFUSED
    return EXIT_STEP_FAILED if StepOutcome.FAILED in outcomes else EXIT_OK


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


def _switch_text(switch: ClassSwitch | None) -> str | None:
    return None if switch is None else str(switch)


@dataclass(frozen=True)
class _AuditTarget:
    row: ledgers.EchoRow
    record: DecisionRecord
    close: audits.ExecutorClose


def run_audit(
    ledger_path: Path, answers: Sequence[ledgers.Answer], *, decisions_root: Path,
) -> ApplyReport:
    """Record each ``N=ok|wrong[:note]`` verdict on a numbered ``audit N`` item; a ``wrong`` may flip its class.

    *N* is the bare item number, as for every other answer: the token names
    the number and the ledger knows the item is an audit. A ``wrong`` flips
    the class to shadow on its queue under ``authority.audit_escape_target``.
    The ledger is read, never saved: an audited close leaves the next draw
    because its DecisionRecord now carries an outcome. Each queue is indexed
    once and the registry is read once per run.
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
    targets, refused = _audit_preflight(resolved.rows, decisions_root)
    if refused:
        return ApplyReport(EXIT_REFUSED, rows=tuple(refused), messages=(
            f'nothing audited: {len(refused)} item(s) refused at pre-flight',))
    try:
        for queue_dir in {target.close.queue_dir for target in targets}:
            policy_state.load_class_states(Path(queue_dir))
    except policy_state.PolicyStateUnreadable as exc:
        return ApplyReport(EXIT_CONFIG, messages=(f'nothing audited: {exc}',))
    recorded = audits.RecordedAudits.read(decisions_root)
    rows: list[AuditResult] = []
    for target in targets:
        row, recorded = _audit_one(target, decisions_root, recorded)
        rows.append(row)
    failed = any(row.outcome != StepOutcome.OK for row in rows)
    return ApplyReport(EXIT_STEP_FAILED if failed else EXIT_OK, rows=tuple(rows),
                       echo=ledgers.render_echo_table(resolved))


def _audit_preflight(
    rows: Sequence[ledgers.EchoRow], decisions_root: Path,
) -> tuple[list[_AuditTarget], list[RefusedItem]]:
    targets: list[_AuditTarget] = []
    refused: list[RefusedItem] = []
    indexes: dict[str, Mapping[str, Path]] = {}
    for row in rows:
        target = _audit_target(row, decisions_root, indexes)
        if isinstance(target, _AuditTarget):
            targets.append(target)
        else:
            refused.append(RefusedItem(row.number, target, {'decision_id': row.record_id}))
    return targets, refused


def _audit_target(
    row: ledgers.EchoRow, decisions_root: Path, indexes: dict[str, Mapping[str, Path]],
) -> _AuditTarget | Refusal:
    """*row*'s executor close, joined through *indexes*, which gains each queue the first time one is needed."""
    try:
        record = DecisionRecord.from_json(decision_path_for_id(row.record_id, root=decisions_root).read_text())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return Refusal(RefusalCode.DECISION_UNREADABLE, {'reason': f'{type(exc).__name__}: {exc}'})
    current = record.current_audit
    if current is not None:
        return Refusal(RefusalCode.ALREADY_AUDITED, {'verdict': str(current.verdict), 'audited_at': current.audited_at})
    queue_dir = normalize_escalations_dir(record.escalations_dir)
    if queue_dir not in indexes:
        indexes[queue_dir] = index_queue(queue_dir)
    close = audits.executor_close_of(record, indexes)
    if isinstance(close, audits.AgedOut):
        return Refusal(RefusalCode.ESCALATION_AGED_OUT,
                       {'escalation_id': close.escalation_id, 'queue_dir': close.queue_dir})
    if not isinstance(close, audits.ExecutorClose):
        reason = close.reason if close is not None else 'its escalation was not closed by the executor under a class'
        return Refusal(RefusalCode.NOT_AN_EXECUTOR_CLOSE, {'reason': reason})
    return _AuditTarget(row, record, close)


def _audit_one(
    target: _AuditTarget, decisions_root: Path, recorded: audits.RecordedAudits,
) -> tuple[AuditResult, audits.RecordedAudits]:
    """Write *target*'s verdict; return its result and *recorded* with the new audit folded in."""
    close, row = target.close, target.row
    queue = Path(close.queue_dir)
    verdict = AuditVerdict(row.option_label)
    before = policy_state.load_class_states(queue).get(close.policy_class)
    result = AuditResult(row.number, close, verdict, before, before)
    try:
        written = record_audit_outcome(
            close.decision_id, verdict, row.note, root=decisions_root, expected_project=target.record.project,
            expected_escalations_dir=target.record.escalations_dir, audited_at=row.answered_at,
            policy_class=close.policy_class, closed_at=close.closed_at,
        )
    except DecisionAuditRefused as exc:
        return replace(result, outcome=StepOutcome.FAILED, detail=str(exc)), recorded
    if written is None:
        return replace(result, outcome=StepOutcome.FAILED,
                       detail='the decision registry did not take the write'), recorded
    recorded = recorded.recording(written)
    if verdict == AuditVerdict.WRONG and before is not None:
        audited, wrong = recorded.tally(close.queue_dir, close.policy_class)
        target_state = audit_escape_target(before, audited, wrong)
        if target_state is not None:
            policy_state.set_class_state(queue, close.policy_class, target_state)
            return replace(result, class_state_after=target_state), recorded
    return result, recorded


def run_policy_state(queue_dir: Path, policy_class: str | None, target: ClassSwitch | None) -> ApplyReport:
    """Every class's switch on *queue_dir*'s queue, or one class moved to *target* through set_class_state."""
    if not queue_dir.is_dir():
        return ApplyReport(EXIT_CONFIG, messages=(f'{queue_dir} is not an existing queue directory; '
                                                  'policy-state never creates one',))
    try:
        if policy_class is None or target is None:
            return ApplyReport(EXIT_OK, rows=(ClassStates(policy_state.load_class_states(queue_dir)),))
        before = policy_state.set_class_state(queue_dir, policy_class, target)
    except IllegalClassTransition as exc:
        return ApplyReport(EXIT_REFUSED, messages=(f'refused, nothing written: {exc}',))
    except ValueError as exc:
        return ApplyReport(EXIT_CONFIG, messages=(f'nothing written: {exc}',))
    return ApplyReport(EXIT_OK, rows=(ClassMove(normalize_escalations_dir(queue_dir), policy_class, before, target),))


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


def _cmd_apply(args: argparse.Namespace) -> int:
    now = args.now or datetime.now(UTC)
    try:
        actions = parse_actions(args.action)
    except ValueError as exc:
        print(f'apply refused, nothing written: {exc}', file=sys.stderr)
        return EXIT_CONFIG
    answers = [parse_answer_token(token, args.source, now.isoformat()) for token in args.answer]
    report = run_apply(
        sources_from_args(args), args.ledger, answers, actions, now=now, fused_memory=fused_memory_session,
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
    for row in report.json_rows():
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
    audit_cmd.add_argument('--answer', action='append', required=True, help="repeatable 'N=ok|wrong[:note]' token, N the bare number of an 'audit N' item")
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
