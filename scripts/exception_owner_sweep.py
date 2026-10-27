"""INV-12's nightly owner sweep: every cited debt owner is still alive, or someone hears.

Reads the exception register's ``--json`` for every debt owner a governed list
or an inline marker cites, asks ``exception_owner_liveness`` whether each owner
still owns its debt, and turns a dead owner into exactly one finding, keyed so
that the next night does not repeat it.  Contract:
``plans/inv12-exceptions-owned-or-ratified-prd.md`` D11 and "Sweep".

Three of the PRD's open questions, as decided:

* Which tree (1): the register measures the WORKING TREE of ``--root``, not
  HEAD blobs.  Every owner is re-read from the store just before it is acted
  on, so a mid-merge tree shifts at most one night's citations, and key
  idempotence settles the next.
* Where a finding's key lives (2): a filed task carries it in
  ``metadata.x_inv12_finding``; an escalation carries it in the caller-chosen
  ``task_id`` slot, checked against one ``get_pending_escalations`` per run.
  The stamp keeps no ledger of escalated keys, which would be a second home
  of what is pending.
* Self-citation (4): not detected.  Every dead-owner finding states both
  remedies: re-own the site, or fix the suppression the owner's work introduced.

The stamp, ``exception_register.SWEEP_STATE_PATH`` under ``--project-root``, is
written after every completed run.  A failed run rewrites a prior stamp with
``ok`` false and that stamp's ``last_run``, so the stamp's age is the liveness
signal ``exception_register.py --report`` reads.  ``--dry-run`` writes,
files and escalates nothing.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import enum
import functools
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

try:
    import _checkout as _checkout
    import exception_owner_liveness as liveness
    import exception_register
    from _task_db_scan import UNREADABLE_STORE_ERRORS, tasks_db_path
    from dashboard.config import DEFAULT_FUSED_MEMORY_URLS, DashboardConfig
    from shared.governed_exceptions import (
        MalformedDisposition,
        TaskRef,
        TicketRef,
        owner_from_record,
        owner_record,
    )
    from shared.mcp_session import SyncMcpSession
    from shared.project_layout import default_project_root
    from shared.safe_io import atomic_write_text
    from shared.task_statuses import OwnerLiveness
except ImportError as exc:
    print(
        f'exception_owner_sweep.py: {exc} -- run it as '
        '`uv run --project shared python scripts/exception_owner_sweep.py`',
        file=sys.stderr,
    )
    raise SystemExit(2) from exc

_SCRIPTS = Path(__file__).resolve().parent

REGISTER_SCHEMA_VERSION_READ = 1
"""The register document version this sweep reads: the consumer-side pin of
``exception_register.py::REGISTER_SCHEMA_VERSION``, as that module pins
``INLINE_REPORT_SCHEMA_VERSION``.  The two change together."""

REGISTER_TIMEOUT_SECS = 300
"""The register measured 10.6 s over this repository on 2026-10-16."""


class FailureStage(enum.StrEnum):
    """Which dependency the sweep could not read, so nothing it saw can be trusted."""

    REGISTER = 'register'
    STORE = 'store'
    FUSED_MEMORY = 'fused_memory'
    ESCALATION = 'escalation'
    CONFIG = 'config'


class SweepFailure(Exception):
    """The sweep could not take a measurement: no finding is filed on it.

    Attributes:
        stage: Where it failed.
        detail: What failed, naming the tool, file or field at fault.
    """

    def __init__(self, stage: FailureStage, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f'{stage}: {detail}')


# ---------------------------------------------------------------------------
# The register input.

INLINE_SCOPE = 'inline'
"""The scope of every inline marker; a governed list's scope is its list id."""

Owner = TaskRef | TicketRef


@dataclass(frozen=True)
class Citation:
    """One owner cited in one scope, with every site that cites it.

    Attributes:
        sites: ``path:line`` for an inline marker; ``default`` or the override
            key for a governed list's entry.
        files: The repository files those sites live in.
    """

    scope: str
    owner: Owner
    sites: tuple[str, ...]
    files: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegisterView:
    """What the sweep reads from the register: who is cited, and the baseline slack."""

    citations: tuple[Citation, ...]
    slack_total: int


_T = TypeVar('_T')


def _field(block: Mapping[str, object], key: str, kind: type[_T], where: str) -> _T:
    """``block[key]``, refused unless present and of *kind* (``bool`` is not an int)."""
    if key not in block:
        raise SweepFailure(FailureStage.REGISTER, f'register field `{where}{key}` is missing')
    value = block[key]
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise SweepFailure(
            FailureStage.REGISTER, f'register field `{where}{key}` is {value!r}, not {kind.__name__}'
        )
    return value


def _entries(block: Mapping[str, object], key: str, where: str) -> list[Mapping[str, object]]:
    entries = _field(block, key, list, where)
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SweepFailure(FailureStage.REGISTER, f'register `{where}{key}[{index}]` is not an object')
    return entries


def _owner(record: object, where: str) -> Owner:
    try:
        return owner_from_record(record)
    except MalformedDisposition as exc:
        raise SweepFailure(FailureStage.REGISTER, f'register `{where}`: {exc}') from exc


def _debt_owner(record: object, where: str) -> Owner | None:
    """The owner of a debt disposition record; None for a policy or no disposition."""
    if record is None or (isinstance(record, dict) and record.get('type') == 'policy'):
        return None
    if isinstance(record, dict) and record.get('type') == 'debt':
        return _owner(record.get('owner'), f'{where}.owner')
    raise SweepFailure(FailureStage.REGISTER, f'register `{where}` is not a disposition: {record!r}')


@dataclass(frozen=True)
class _Cite:
    """One citing site: its sort key renders as the site, ``:``-joined."""

    scope: str
    owner: Owner
    site: tuple[str] | tuple[str, int]
    file: str


def _declaration_cites(declaration: Mapping[str, object], where: str) -> list[_Cite]:
    list_id = _field(declaration, 'list_id', str, where)
    path = _field(declaration, 'path', str, where)
    owned = [('default', _debt_owner(declaration.get('default'), f'{where}default'))]
    for index, override in enumerate(_entries(declaration, 'overrides', where)):
        key = _field(override, 'key', str, f'{where}overrides[{index}].')
        owned.append((key, _debt_owner(override.get('disposition'), f'{where}overrides[{index}].disposition')))
    return [_Cite(list_id, owner, (key,), path) for key, owner in owned if owner is not None]


def _inline_cite(debt: Mapping[str, object], where: str) -> _Cite:
    path = _field(debt, 'path', str, where)
    line = _field(debt, 'line', int, where)
    return _Cite(INLINE_SCOPE, _owner(debt.get('owner'), f'{where}owner'), (path, line), path)


def _citation_order(cited: tuple[str, Owner]) -> tuple[str, int, int, str]:
    """Scope, then task owners by number before ticket owners by id."""
    scope, owner = cited
    return (scope, 0, owner.id, '') if isinstance(owner, TaskRef) else (scope, 1, 0, owner.id)


def parse_register(document: object) -> RegisterView:
    """The register's ``--json`` document as citations, or :class:`SweepFailure` (REGISTER)."""
    if not isinstance(document, dict):
        raise SweepFailure(FailureStage.REGISTER, f'register document is not an object: {document!r}')
    version = document.get('schema_version')
    if version != REGISTER_SCHEMA_VERSION_READ:
        raise SweepFailure(
            FailureStage.REGISTER,
            f'register published schema_version {version!r}, but this sweep reads '
            f'{REGISTER_SCHEMA_VERSION_READ}; the two change together',
        )
    cites = [
        cite
        for index, declaration in enumerate(_entries(document, 'declarations', ''))
        for cite in _declaration_cites(declaration, f'declarations[{index}].')
    ]
    inline = _field(document, 'inline', dict, '')
    cites += [
        _inline_cite(debt, f'inline.debt[{index}].')
        for index, debt in enumerate(_entries(inline, 'debt', 'inline.'))
    ]
    baseline = _field(document, 'baseline', dict, '')
    slack = baseline.get('slack_total')
    return RegisterView(
        citations=_citations(cites),
        slack_total=0 if slack is None else _field(baseline, 'slack_total', int, 'baseline.'),
    )


def _citations(cites: list[_Cite]) -> tuple[Citation, ...]:
    """One citation per (scope, owner), every list sorted."""
    grouped: dict[tuple[str, Owner], list[_Cite]] = {}
    for cite in cites:
        grouped.setdefault((cite.scope, cite.owner), []).append(cite)
    return tuple(
        Citation(
            scope,
            owner,
            sites=tuple(':'.join(map(str, site)) for site in sorted({c.site for c in grouped[scope, owner]})),
            files=tuple(sorted({c.file for c in grouped[scope, owner]})),
        )
        for scope, owner in sorted(grouped, key=_citation_order)
    )


def read_register(root: Path, sweep_state: Path) -> Mapping[str, object]:
    """Run ``exception_register.py --json`` over *root*; any fault is a REGISTER failure."""
    command = [
        sys.executable,
        str(_SCRIPTS / 'exception_register.py'),
        '--json',
        '--root',
        str(root),
        '--sweep-state',
        str(sweep_state),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=REGISTER_TIMEOUT_SECS, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise SweepFailure(
            FailureStage.REGISTER, f'exception_register.py ran past {REGISTER_TIMEOUT_SECS} s'
        ) from exc
    if completed.returncode != 0:
        raise SweepFailure(
            FailureStage.REGISTER,
            f'exception_register.py exited {completed.returncode}: {completed.stderr.strip()}',
        )
    try:
        document = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise SweepFailure(FailureStage.REGISTER, f'exception_register.py printed no JSON: {exc}') from exc
    if not isinstance(document, dict):
        raise SweepFailure(FailureStage.REGISTER, 'exception_register.py printed no JSON object')
    return document


# ---------------------------------------------------------------------------
# The finding key: what a finding is about, carried in task metadata and in
# the escalation task_id slot so the next run can see it was already raised.

FINDING_KEY_FIELD = 'x_inv12_finding'
"""Task metadata field carrying a filed finding's :meth:`FindingKey.record`."""

OWNS_NOTE_FIELD = 'x_inv12_owns'
"""Task metadata field carrying the pointer note on a live cited owner."""


class FindingKind(enum.StrEnum):
    DEAD_OWNER = 'dead_owner'
    TIGHTEN = 'tighten'
    OVERFLOW = 'overflow'

    @property
    def escalates_when_closed(self) -> bool:
        """A closed task carrying this kind's key means escalate, never re-file.

        Only a dead owner: slack is a recurring chore, so a closed tighten
        task with slack back above zero is simply filed again.
        """
        return self is FindingKind.DEAD_OWNER


@dataclass(frozen=True)
class FindingKey:
    """A finding's identity: a dead owner in one scope, or one of two constant findings."""

    kind: FindingKind
    scope: str | None = None
    owner: Owner | None = None

    def __post_init__(self) -> None:
        scoped = self.kind is FindingKind.DEAD_OWNER
        if scoped != (self.scope is not None) or scoped != (self.owner is not None):
            raise ValueError(
                f'FindingKey: kind={self.kind!s} with scope={self.scope!r} and owner={self.owner!r} '
                '-- a dead-owner key names a scope and an owner, any other kind names neither'
            )

    def record(self) -> dict[str, object]:
        if self.scope is None or self.owner is None:
            return {'kind': str(self.kind)}
        return {'kind': str(self.kind), 'scope': self.scope, 'owner': owner_record(self.owner)}

    @classmethod
    def from_record(cls, value: object) -> FindingKey | None:
        """The key *value* records, or None for anything that records no key."""
        if not isinstance(value, dict) or value.get('kind') not in set(FindingKind):
            return None
        scope = value.get('scope')
        owner = value.get('owner')
        if scope is not None and not isinstance(scope, str):
            return None
        try:
            return cls(
                FindingKind(value['kind']), scope, None if owner is None else owner_from_record(owner)
            )
        except (ValueError, MalformedDisposition):
            return None

    def escalation_task_id(self) -> str:
        """An opaque id for the escalation ``task_id`` slot, compared and never parsed.

        Precedent: ``fused-memory/src/fused_memory/reconciliation/index_drift_detector.py``.
        """
        if self.scope is None or self.owner is None:
            return f'inv12-finding:{self.kind}'
        owner = f'task-{self.owner.id}' if isinstance(self.owner, TaskRef) else self.owner.id
        return f'inv12-finding:{self.kind}:{self.scope}:{owner}'


TIGHTEN_KEY = FindingKey(FindingKind.TIGHTEN)
OVERFLOW_KEY = FindingKey(FindingKind.OVERFLOW)

# ---------------------------------------------------------------------------
# Ports: every store and MCP interaction is one call carrying what it needs.

ToolCall = Callable[[str, dict[str, object]], Mapping[str, object]]
"""One MCP tool call: tool name and arguments in, the unwrapped result out."""

AGENT_ROLE = 'exception-owner-sweep'

REPORT_COMMAND = 'uv run --project shared python scripts/exception_register.py --report'

SUBMIT_RESOLVE_WAIT_SECS = 115
"""How long a filing waits for the curator: under the server's 120 s MCP call limit."""


@dataclass(frozen=True)
class Ports:
    """Where the sweep reads and writes: the main checkout's store and both MCP servers."""

    project_root: Path
    db: Path
    fused_memory: ToolCall
    escalation: ToolCall


def _call(port: ToolCall, stage: FailureStage, tool: str, args: dict[str, object]) -> Mapping[str, object]:
    """The one MCP boundary: a transport fault or an error answer is a :class:`SweepFailure`."""
    try:
        result = port(tool, args)
    except Exception as exc:
        raise SweepFailure(stage, f'{tool} failed: {type(exc).__name__}: {exc}') from exc
    if not isinstance(result, Mapping) or 'error' in result:
        raise SweepFailure(stage, f'{tool} answered {result!r}')
    return result


def _from_store(db: Path, read: Callable[[], _T]) -> _T:
    try:
        return read()
    except UNREADABLE_STORE_ERRORS as exc:
        raise SweepFailure(FailureStage.STORE, f'{db}: {exc}') from exc


def _resolve_ticket(ports: Ports, ticket: str, wait_secs: float) -> liveness.TicketOutcome:
    args: dict[str, object] = {
        'ticket': ticket,
        'project_root': str(ports.project_root),
        'timeout_seconds': wait_secs,
    }
    result = _call(ports.fused_memory, FailureStage.FUSED_MEMORY, 'resolve_ticket', args)
    try:
        return liveness.ticket_outcome(result)
    except liveness.TicketUnreadable as exc:
        raise SweepFailure(FailureStage.FUSED_MEMORY, f'resolve_ticket {ticket}: {exc}') from exc


def owner_resolver(ports: Ports) -> Callable[[Owner], liveness.OwnerVerdict]:
    """``resolve_owner`` bound to this store and to a never-waiting ``resolve_ticket``.

    Zero wait: a ticket still being curated answers the unpersisted ``timeout``,
    which reads as pending, so the owner is live.
    """
    read_tasks = functools.partial(liveness.read_tasks, ports.db)

    def resolve_ticket(ticket: TicketRef) -> liveness.TicketOutcome:
        return _resolve_ticket(ports, ticket.id, wait_secs=0)

    def resolve(owner: Owner) -> liveness.OwnerVerdict:
        return _from_store(
            ports.db,
            lambda: liveness.resolve_owner(owner, read_tasks=read_tasks, resolve_ticket=resolve_ticket),
        )

    return resolve


def _carriers(ports: Ports, field: str) -> tuple[liveness.TaskRow, ...]:
    return _from_store(ports.db, lambda: liveness.read_tasks_carrying(ports.db, field))


def pending_escalation_ids(escalation: ToolCall) -> frozenset[str]:
    """The ``task_id`` of every pending escalation: one ``get_pending_escalations`` call."""
    answer = _call(escalation, FailureStage.ESCALATION, 'get_pending_escalations', {'compact': True})
    records = answer.get('result')
    if not isinstance(records, list):
        raise SweepFailure(
            FailureStage.ESCALATION, f'get_pending_escalations answered {answer!r}, not a result list'
        )
    return frozenset(
        record['task_id']
        for record in records
        if isinstance(record, dict) and isinstance(record.get('task_id'), str)
    )


# ---------------------------------------------------------------------------
# Scan (read-only), decide (pure), act (re-read, then write).


@dataclass(frozen=True)
class Scan:
    """Every cited owner's verdict and the pending escalations, as measured at one moment."""

    register: RegisterView
    verdicts: Mapping[tuple[str, Owner], liveness.OwnerVerdict]
    pending: frozenset[str]
    measured_at: datetime.datetime


def scan(register: RegisterView, ports: Ports, *, now: datetime.datetime) -> Scan:
    """Resolve every cited owner once; reads only."""
    pending = pending_escalation_ids(ports.escalation)
    resolve = owner_resolver(ports)
    by_owner = {owner: resolve(owner) for owner in dict.fromkeys(c.owner for c in register.citations)}
    verdicts = {(c.scope, c.owner): by_owner[c.owner] for c in register.citations}
    return Scan(register, MappingProxyType(verdicts), pending, now)


class Action(enum.Enum):
    """What :func:`decide` plans for one finding, before the output bound or any output."""

    SKIP_LIVE_TASK = 'skip_live_task'
    SKIP_PENDING_ESCALATION = 'skip_pending_escalation'
    ESCALATE_CLOSED_FINDING = 'escalate_closed_finding'
    FILE_TASK = 'file_task'
    FILE_ESCALATION = 'file_escalation'

    @property
    def label(self) -> str:
        return f'planned {self.value}'


_SKIPS = frozenset({Action.SKIP_LIVE_TASK, Action.SKIP_PENDING_ESCALATION})


def decide(
    key: FindingKey, *, carriers: Sequence[liveness.TaskRow], pending: frozenset[str]
) -> Action:
    """What to do about *key*, given the tasks carrying a finding key and the pending escalations."""
    mine = [row for row in carriers if FindingKey.from_record(row.metadata.get(FINDING_KEY_FIELD)) == key]
    if any(row.liveness() is OwnerLiveness.OPEN for row in mine):
        return Action.SKIP_LIVE_TASK
    if key.escalation_task_id() in pending:
        return Action.SKIP_PENDING_ESCALATION
    if mine and key.kind.escalates_when_closed:
        return Action.ESCALATE_CLOSED_FINDING
    if key.scope in (None, INLINE_SCOPE):
        return Action.FILE_TASK
    return Action.FILE_ESCALATION


@dataclass(frozen=True)
class ActOutcome:
    checked: int = 0
    findings: int = 0
    filed: int = 0
    escalated: int = 0
    skipped_existing: int = 0
    overflow: int = 0
    via_successor: int = 0


@dataclass(frozen=True)
class _Body:
    """A finding as both outputs render it: task title/description or escalation summary/detail."""

    summary: str
    detail: str
    remedy: str
    observation: str
    files: tuple[str, ...] = ()


SITES_SHOWN = 10

_DEAD_OWNER_REMEDY = (
    'Re-own every listed site with a live task or ticket; or, if this owner\'s own work '
    'introduced the suppression, fix the suppression rather than re-citing it. '
    'Never cite your own task.'
)


def _owner_label(owner: Owner) -> str:
    return f'task {owner.id}' if isinstance(owner, TaskRef) else f'ticket {owner.id}'


def _death_facts(verdict: liveness.OwnerVerdict) -> str:
    facts = [f'cause {verdict.cause}']
    if verdict.closed_status is not None:
        facts.append(f'status {verdict.closed_status!r}')
    if len(verdict.path) > 1:
        facts.append('successors followed ' + ' -> '.join(map(str, verdict.path)))
    if verdict.ticket is not None:
        facts.append(f'ticket {verdict.ticket.status} ({verdict.ticket.reason})')
    return ', '.join(facts)


def _bounded(items: Sequence[str]) -> str:
    rest = len(items) - SITES_SHOWN
    shown = ', '.join(items[:SITES_SHOWN])
    return shown if rest <= 0 else f'{shown} and {rest} more'


def _dead_owner_body(citation: Citation, verdict: liveness.OwnerVerdict) -> _Body:
    owner = _owner_label(citation.owner)
    facts = _death_facts(verdict)
    return _Body(
        summary=f'INV-12: dead debt owner {owner} still cited by {len(citation.sites)} site(s) in {citation.scope}',
        detail='\n'.join(
            [
                f'{owner} no longer owns the debt it is cited for ({facts}).',
                f'Scope: {citation.scope}. Sites: {_bounded(citation.sites)}.',
                f'Remedy: {_DEAD_OWNER_REMEDY}',
                f'Report: {REPORT_COMMAND}',
            ]
        ),
        remedy=_DEAD_OWNER_REMEDY,
        observation=f'{owner} cited in {citation.scope} is dead: {facts}',
        files=citation.files,
    )


@dataclass(frozen=True)
class FilingOutcome:
    """What one submit yielded; ``confirmed`` only for a live task carrying the finding's key."""

    confirmed: bool
    ticket: str
    outcome: liveness.TicketOutcome
    task_row: liveness.TaskRow | None

    @property
    def still_curating(self) -> bool:
        """The curator had not answered within the wait: a task carrying the key may yet appear."""
        return self.outcome.state is liveness.TicketState.PENDING


def _submit(key: FindingKey, body: _Body, ports: Ports) -> str:
    args: dict[str, object] = {
        'project_root': str(ports.project_root),
        'title': body.summary,
        'description': body.detail,
        'priority': 'medium',
        'agent_id': AGENT_ROLE,
        'metadata': {FINDING_KEY_FIELD: key.record(), 'source': AGENT_ROLE, 'files': list(body.files)},
    }
    submitted = _call(ports.fused_memory, FailureStage.FUSED_MEMORY, 'submit_task', args)
    ticket = submitted.get('ticket')
    if not isinstance(ticket, str):
        raise SweepFailure(FailureStage.FUSED_MEMORY, f'submit_task answered {submitted!r}, no ticket')
    return ticket


def _read_row(ports: Ports, task_id: int) -> liveness.TaskRow | None:
    return _from_store(ports.db, lambda: liveness.read_tasks(ports.db, (task_id,))).get(task_id)


def _file_task(key: FindingKey, body: _Body, ports: Ports) -> FilingOutcome:
    """Submit *key*'s task, wait for the curator, and read back what it made.

    The curator is not the dedup mechanism (D11): a combine or a refusal
    yields no task carrying the key, which is not filed and must be loud.
    """
    ticket = _submit(key, body, ports)
    outcome = _resolve_ticket(ports, ticket, wait_secs=SUBMIT_RESOLVE_WAIT_SECS)
    row = None if outcome.task_id is None else _read_row(ports, outcome.task_id)
    confirmed = (
        row is not None
        and row.liveness() is OwnerLiveness.OPEN
        and FindingKey.from_record(row.metadata.get(FINDING_KEY_FIELD)) == key
    )
    return FilingOutcome(confirmed=confirmed, ticket=ticket, outcome=outcome, task_row=row)


def _unlanded(filing: FilingOutcome) -> str:
    outcome = filing.outcome
    reason = '' if outcome.reason is None else f' ({outcome.reason})'
    if outcome.task_id is None:
        target = 'no task'
    elif filing.task_row is None:
        target = f'task {outcome.task_id}, which the store does not hold'
    else:
        target = f'task {outcome.task_id}, status {filing.task_row.status!r}, not carrying this finding'
    return f'Filing did not land: ticket {filing.ticket} resolved {outcome.status}{reason} to {target}.'


def _curating(filing: FilingOutcome) -> str:
    return (
        f'Filing still being curated: ticket {filing.ticket} had not resolved after '
        f'{SUBMIT_RESOLVE_WAIT_SECS} s. If the curator creates a task carrying this finding, '
        'that task is its home and this escalation can be resolved.'
    )


def _unconfirmed_body(body: _Body, filing: FilingOutcome) -> _Body:
    """*body* plus why its filing is not confirmed, for the escalation that replaces it."""
    why = _curating(filing) if filing.still_curating else _unlanded(filing)
    return dataclasses.replace(body, detail=f'{body.detail}\n{why}', observation=f'{body.observation}; {why}')


def _escalate(key: FindingKey, body: _Body, ports: Ports, now: datetime.datetime) -> None:
    args: dict[str, object] = {
        'task_id': key.escalation_task_id(),
        'agent_role': AGENT_ROLE,
        'category': 'design_concern',
        'summary': body.summary,
        'detail': body.detail,
        'suggested_action': body.remedy,
        'evidence': [
            {'observation': body.observation, 'measured_at': now.isoformat(), 'ref': key.escalation_task_id()}
        ],
    }
    _call(ports.escalation, FailureStage.ESCALATION, 'escalate_info', args)


TIGHTEN_COMMAND = 'uv run --project shared python scripts/inline_suppressions.py --tighten'

MAX_OUTPUTS_PER_RUN = 10
"""INV-4's storm bound: filings and escalations per run, plus one overflow roll-up."""


def _tighten_body(slack: int) -> _Body:
    remedy = f'Run `{TIGHTEN_COMMAND}` and commit the tightened baseline.'
    return _Body(
        summary=f'INV-12: tighten the inline-suppression baseline ({slack} slack)',
        detail='\n'.join(
            [
                f'The inline-suppression baseline allows {slack} more unowned sites than the tree holds.',
                f'Remedy: {remedy}',
                f'Report: {REPORT_COMMAND}',
            ]
        ),
        remedy=remedy,
        observation=f'baseline slack_total {slack}',
    )


def _overflow_body(count: int) -> _Body:
    remedy = f'Run `{REPORT_COMMAND}` to see every finding now; the sweep retries them next run.'
    return _Body(
        summary=f'INV-12 owner sweep: {count} finding(s) over the per-run output bound',
        detail='\n'.join(
            [
                f'{count} finding(s) were neither filed nor escalated this run: the sweep sends '
                f'at most {MAX_OUTPUTS_PER_RUN} outputs per run (INV-4).',
                f'Remedy: {remedy}',
            ]
        ),
        remedy=remedy,
        observation=f'{count} finding(s) over the bound of {MAX_OUTPUTS_PER_RUN}',
    )


class Performed(enum.Enum):
    """What :func:`act` did about one finding, which can differ from what was planned."""

    SKIPPED_EXISTING = 'skipped_existing'
    FILED = 'filed'
    ESCALATED = 'escalated'
    ESCALATED_FILING_UNLANDED = 'escalated_filing_unlanded'
    ESCALATED_FILING_CURATING = 'escalated_filing_still_curating'
    OVERFLOW = 'overflow'

    @property
    def escalated(self) -> bool:
        return self in (
            Performed.ESCALATED,
            Performed.ESCALATED_FILING_UNLANDED,
            Performed.ESCALATED_FILING_CURATING,
        )

    @property
    def label(self) -> str:
        return self.value


Step = Action | Performed
"""A finding's step: planned on a dry run, performed on a real one."""


def _verdict_phrase(verdict: liveness.OwnerVerdict) -> str:
    if verdict.liveness is liveness.Liveness.DEAD:
        return f'dead ({_death_facts(verdict)})'
    if verdict.liveness is liveness.Liveness.LIVE_VIA_SUCCESSOR:
        chain = ' -> '.join(map(str, verdict.path))
        return f'live via successor task {verdict.owning_task} ({chain})'
    if verdict.owning_task is None:
        return 'live (ticket still being curated)'
    return f'live (task {verdict.owning_task})'


@dataclass(frozen=True)
class Report:
    """One citation as measured, and the step taken on it (None: there was none to take).

    The step's label says whether it was planned or performed, so a journal
    line never claims an output the run did not send.
    """

    scope: str
    owner: Owner
    verdict: liveness.OwnerVerdict
    step: Step | None

    def line(self) -> str:
        head = f'{self.scope} {_owner_label(self.owner)}: {_verdict_phrase(self.verdict)}'
        return head if self.step is None else f'{head} -> {self.step.label}'


@dataclass(frozen=True)
class Tally:
    """One sweep's steps, which its counts derive from.

    Attributes:
        reports: One per citation, in register order.
        tighten: The tighten finding's step; None while there is no slack.
        rolled_up: Whether the overflow roll-up was sent.
    """

    reports: tuple[Report, ...] = ()
    tighten: Step | None = None
    rolled_up: bool = False

    def counts(self) -> ActOutcome:
        verdicts = collections.Counter(report.verdict.liveness for report in self.reports)
        steps = [*(report.step for report in self.reports), self.tighten]
        performed = collections.Counter(step for step in steps if isinstance(step, Performed))
        return ActOutcome(
            checked=len(self.reports),
            findings=verdicts[liveness.Liveness.DEAD],
            filed=performed[Performed.FILED],
            escalated=sum(n for step, n in performed.items() if step.escalated) + int(self.rolled_up),
            skipped_existing=performed[Performed.SKIPPED_EXISTING],
            overflow=performed[Performed.OVERFLOW],
            via_successor=verdicts[liveness.Liveness.LIVE_VIA_SUCCESSOR],
        )


class _Outputs:
    """One act() run's outputs: the steps taken so far, and the output budget left."""

    def __init__(self, ports: Ports, pending: frozenset[str], now: datetime.datetime) -> None:
        self.ports = ports
        self.pending = set(pending)
        self.now = now
        self.budget = MAX_OUTPUTS_PER_RUN
        self.reports: list[Report] = []
        self.tighten: Performed | None = None
        self.rolled_up = False
        self._carriers: tuple[liveness.TaskRow, ...] | None = None

    def tally(self) -> Tally:
        return Tally(tuple(self.reports), self.tighten, self.rolled_up)

    def _decide(self, key: FindingKey, *, reread: bool) -> Action:
        if reread or self._carriers is None:
            self._carriers = _carriers(self.ports, FINDING_KEY_FIELD)
        return decide(key, carriers=self._carriers, pending=frozenset(self.pending))

    def raise_finding(self, key: FindingKey, body: _Body) -> Performed:
        """Perform at most one budgeted output for *key*, returning the step taken.

        The carriers are re-read only where an output can follow (INV-3), so a
        finding that ends skipped or held back costs no further store scan.
        """
        if self._decide(key, reread=False) in _SKIPS:
            return Performed.SKIPPED_EXISTING
        if self.budget == 0:
            return Performed.OVERFLOW
        action = self._decide(key, reread=True)
        if action in _SKIPS:
            return Performed.SKIPPED_EXISTING
        self.budget -= 1
        if action is Action.FILE_TASK:
            return self._file(key, body)
        self._escalate(key, body)
        return Performed.ESCALATED

    def _file(self, key: FindingKey, body: _Body) -> Performed:
        filing = _file_task(key, body, self.ports)
        if filing.confirmed:
            return Performed.FILED
        self._escalate(key, _unconfirmed_body(body, filing))
        if filing.still_curating:
            return Performed.ESCALATED_FILING_CURATING
        return Performed.ESCALATED_FILING_UNLANDED

    def _escalate(self, key: FindingKey, body: _Body) -> None:
        _escalate(key, body, self.ports, self.now)
        self.pending.add(key.escalation_task_id())

    def roll_up(self) -> None:
        """The one unbudgeted message: how many findings the bound held back."""
        held_back = self.tally().counts().overflow
        if held_back and OVERFLOW_KEY.escalation_task_id() not in self.pending:
            self._escalate(OVERFLOW_KEY, _overflow_body(held_back))
            self.rolled_up = True


def _desired_notes(reports: Sequence[Report]) -> dict[int, set[str]]:
    """Owning task -> the scopes citing it, for each owner live in its own right.

    That is a directly cited task, or the task a cited ticket resolved to;
    never a successor, which no citation names.
    """
    desired: dict[int, set[str]] = {}
    for report in reports:
        verdict = report.verdict
        if verdict.liveness is liveness.Liveness.LIVE and verdict.owning_task is not None:
            desired.setdefault(verdict.owning_task, set()).add(report.scope)
    return desired


def _write_note(ports: Ports, task_id: int, note: Mapping[str, object] | None) -> None:
    args: dict[str, object] = {
        'id': str(task_id),
        'project_root': str(ports.project_root),
        'metadata': {OWNS_NOTE_FIELD: None if note is None else dict(note)},
        'metadata_mode': 'merge',
        'agent_id': AGENT_ROLE,
    }
    _call(ports.fused_memory, FailureStage.FUSED_MEMORY, 'update_task', args)


def _reconcile_notes(reports: Sequence[Report], ports: Ports, now: datetime.datetime) -> None:
    """Point every live cited owner at the register, and clear notes nothing names (INV-9).

    A note is written only when its scopes change, so ``as_of`` is the day
    these scopes were first seen.  A carrier any citation still names (its
    owner, or its ticket's task) keeps its note even when dead.
    """
    desired = _desired_notes(reports)
    named = {report.verdict.path[0] for report in reports if report.verdict.path}
    current = {row.id: row.metadata[OWNS_NOTE_FIELD] for row in _carriers(ports, OWNS_NOTE_FIELD)}
    for task_id, scopes in sorted(desired.items()):
        held = current.get(task_id)
        if not (isinstance(held, Mapping) and held.get('scopes') == sorted(scopes)):
            note = {'scopes': sorted(scopes), 'as_of': now.date().isoformat(), 'report': REPORT_COMMAND}
            _write_note(ports, task_id, note)
    for task_id in sorted(current.keys() - named):
        _write_note(ports, task_id, None)


class ActStopped(SweepFailure):
    """:func:`act` failed part-way; ``done`` counts the outputs that had already landed."""

    def __init__(self, failure: SweepFailure, done: ActOutcome) -> None:
        super().__init__(failure.stage, failure.detail)
        self.done = done


def _act_on(scan: Scan, outputs: _Outputs) -> None:
    resolve = functools.cache(owner_resolver(outputs.ports))
    for citation in scan.register.citations:
        verdict = resolve(citation.owner)
        step: Performed | None = None
        if verdict.liveness is liveness.Liveness.DEAD:
            key = FindingKey(FindingKind.DEAD_OWNER, citation.scope, citation.owner)
            step = outputs.raise_finding(key, _dead_owner_body(citation, verdict))
        outputs.reports.append(Report(citation.scope, citation.owner, verdict, step))
    if scan.register.slack_total > 0:
        outputs.tighten = outputs.raise_finding(TIGHTEN_KEY, _tighten_body(scan.register.slack_total))
    outputs.roll_up()
    _reconcile_notes(outputs.reports, outputs.ports, outputs.now)


def act(scan: Scan, ports: Ports, *, now: datetime.datetime) -> Tally:
    """Re-resolve each cited owner immediately before acting on it, then act (INV-3).

    An owner cited in several scopes is re-read once, before its first
    citation is acted on.  Any failure raises :class:`ActStopped`, counting
    what had already landed.
    """
    outputs = _Outputs(ports, scan.pending, now)
    try:
        _act_on(scan, outputs)
    except SweepFailure as exc:
        raise ActStopped(exc, outputs.tally().counts()) from exc
    return outputs.tally()


# ---------------------------------------------------------------------------
# One run: measure, plan, act, stamp; or fail loud.

FAILURE_TASK_ID = 'inv12-owner-sweep'
"""The escalation ``task_id`` of the sweep's report that it could not run."""


@dataclass(frozen=True)
class RunConfig:
    """Which tree is measured, and which main checkout's store and stamp are used."""

    root: Path
    project_root: Path
    dry_run: bool = False

    @property
    def stamp_path(self) -> Path:
        return self.project_root / exception_register.SWEEP_STATE_PATH


def plan(scan: Scan, ports: Ports) -> Tally:
    """Each citation's planned step as of *scan*, ignoring the output bound; reads only."""
    carriers = _carriers(ports, FINDING_KEY_FIELD)

    def planned(key: FindingKey) -> Action:
        return decide(key, carriers=carriers, pending=scan.pending)

    reports = []
    for citation in scan.register.citations:
        verdict = scan.verdicts[citation.scope, citation.owner]
        dead = verdict.liveness is liveness.Liveness.DEAD
        key = FindingKey(FindingKind.DEAD_OWNER, citation.scope, citation.owner)
        reports.append(Report(citation.scope, citation.owner, verdict, planned(key) if dead else None))
    tighten = planned(TIGHTEN_KEY) if scan.register.slack_total > 0 else None
    return Tally(tuple(reports), tighten)


@dataclass(frozen=True)
class Failure:
    stage: FailureStage
    detail: str


@dataclass(frozen=True)
class SweepResult(ActOutcome):
    """One run: its counts, what its stamp says, and the corroborated list.

    ``slack`` is None when the run failed; ``last_run`` is the stamp's, so it
    is the PRIOR run's after a failure and None after a dry run.  The steps
    in ``reports`` and ``tighten`` are planned on a dry run, performed otherwise.
    """

    ok: bool = False
    slack: int | None = None
    last_run: datetime.datetime | None = None
    failure: Failure | None = None
    reports: tuple[Report, ...] = ()
    tighten: Step | None = None

    def record(self) -> dict[str, object]:
        """The stamp and stdout object: the Contract's nine keys, plus ``failure`` and ``via_successor``."""
        counts = {field.name: getattr(self, field.name) for field in dataclasses.fields(ActOutcome)}
        failure = self.failure
        return {
            **counts,
            'slack': self.slack,
            'last_run': None if self.last_run is None else self.last_run.isoformat(),
            'ok': self.ok,
            'failure': None if failure is None else {'stage': str(failure.stage), 'detail': failure.detail},
        }

    def lines(self) -> list[str]:
        lines = [report.line() for report in self.reports]
        if self.tighten is not None:
            lines.append(f'baseline slack {self.slack} -> {self.tighten.label}')
        return lines


def _write_stamp(config: RunConfig, result: SweepResult) -> None:
    atomic_write_text(config.stamp_path, json.dumps(result.record(), indent=2) + '\n', mkdir=True)


def _failure_report_pending(escalation: ToolCall) -> bool:
    """Whether an earlier run's failure report is still pending (INV-4).

    An unreadable queue answers False: a duplicate report beats a silent outage.
    """
    try:
        return FAILURE_TASK_ID in pending_escalation_ids(escalation)
    except SweepFailure:
        return False


def _report_failure(escalation: ToolCall | None, failure: SweepFailure, now: datetime.datetime) -> str:
    """Send the one best-effort infra_issue escalation, unless one is pending.

    Returns the failure's detail, plus why no escalation was sent.
    """
    if escalation is None:
        return f'{failure.detail}; no escalation server to report it to'
    if _failure_report_pending(escalation):
        return f'{failure.detail}; an earlier failure report ({FAILURE_TASK_ID}) is still pending, so none was sent'
    args: dict[str, object] = {
        'task_id': FAILURE_TASK_ID,
        'agent_role': AGENT_ROLE,
        'category': 'infra_issue',
        'summary': f'INV-12 owner sweep failed at {failure.stage}: no finding past it was filed',
        'detail': '\n'.join(
            [
                failure.detail,
                'The stamp keeps its previous last_run, so its age shows how long the sweep has been down.',
                f'Report: {REPORT_COMMAND}',
            ]
        ),
        'evidence': [
            {'observation': f'{failure.stage}: {failure.detail}', 'measured_at': now.isoformat(), 'ref': str(failure.stage)}
        ],
    }
    try:
        _call(escalation, FailureStage.ESCALATION, 'escalate_info', args)
    except SweepFailure as unsent:
        return f'{failure.detail}; the failure escalation was not sent: {unsent}'
    return failure.detail


def _failed(
    config: RunConfig, failure: SweepFailure, *, escalation: ToolCall | None, now: datetime.datetime
) -> SweepResult:
    """Fail loud: report once (never on a dry run) and leave the prior stamp's last_run unadvanced."""
    done = failure.done if isinstance(failure, ActStopped) else ActOutcome()
    if config.dry_run:
        return SweepResult(**dataclasses.asdict(done), failure=Failure(failure.stage, failure.detail))
    detail = _report_failure(escalation, failure, now)
    prior = exception_register.read_sweep(config.stamp_path, now=now)
    result = SweepResult(
        **dataclasses.asdict(done), last_run=prior.last_run, failure=Failure(failure.stage, detail)
    )
    if prior.last_run is not None:
        _write_stamp(config, result)
    return result


def run(
    config: RunConfig,
    ports: Ports,
    *,
    read_register: Callable[[], Mapping[str, object]],
    now: datetime.datetime,
) -> SweepResult:
    """Measure, then plan (a dry run) or act and stamp.  A :class:`SweepFailure` is returned, never raised."""
    try:
        measured = scan(parse_register(read_register()), ports, now=now)
        tally = plan(measured, ports) if config.dry_run else act(measured, ports, now=now)
    except SweepFailure as exc:
        return _failed(config, exc, escalation=ports.escalation, now=now)
    result = SweepResult(
        **dataclasses.asdict(tally.counts()),
        ok=True,
        slack=measured.register.slack_total,
        last_run=None if config.dry_run else now,
        reports=tally.reports,
        tighten=tally.tighten,
    )
    if not config.dry_run:
        _write_stamp(config, result)
    return result


# ---------------------------------------------------------------------------
# Ports over the real servers, and the CLI.

FUSED_MEMORY_URL_ENV = 'FUSED_MEMORY_MCP_URL'

ESCALATION_TIMEOUT_SECS = 30


def escalation_url(project_root: Path) -> str:
    """The escalation server's MCP URL from *project_root*'s orchestrator config, or a CONFIG failure."""
    config = DashboardConfig(project_root=project_root)
    url = config.escalation_urls.get(config.project_root.name)
    if url is None:
        raise SweepFailure(
            FailureStage.CONFIG, f'{config.project_root} configures no escalation server (escalation.port)'
        )
    return url


def _call_tool(base_url: str, tool: str, arguments: dict[str, object], *, timeout: float) -> dict:
    with SyncMcpSession(base_url, client_name='exception-owner-sweep', timeout=timeout) as session:
        return session.call_tool(tool, arguments)


def _server_base(endpoint_url: str) -> str:
    """The server base of an MCP endpoint URL, the form the dashboard config and --escalation-url speak."""
    return endpoint_url.removesuffix('/mcp')


def build_ports(project_root: Path, *, fused_memory_url: str | None, escalation_mcp_url: str | None) -> Ports:
    """Both MCP servers over the real transport; a URL left None takes its default."""
    fused_memory = fused_memory_url or os.environ.get(FUSED_MEMORY_URL_ENV, DEFAULT_FUSED_MEMORY_URLS[0])
    return Ports(
        project_root=project_root,
        db=tasks_db_path(project_root),
        fused_memory=functools.partial(_call_tool, fused_memory, timeout=SUBMIT_RESOLVE_WAIT_SECS + 15),
        escalation=functools.partial(
            _call_tool,
            _server_base(escalation_mcp_url or escalation_url(project_root)),
            timeout=ESCALATION_TIMEOUT_SECS,
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="INV-12's owner sweep: find every cited debt owner that no longer owns its debt.",
        epilog=(
            'Exit 0: the sweep ran and stamped its result. Exit 2: it could not measure or act '
            '(the result names the stage), or it could not start; nothing past that point was filed.'
        ),
    )
    parser.add_argument('--dry-run', action='store_true', help='measure and report; file, escalate and stamp nothing')
    parser.add_argument(
        '--root',
        type=Path,
        default=_checkout.ROOT,
        help='the tree whose citations are measured (default: the checkout holding this script)',
    )
    parser.add_argument(
        '--project-root',
        type=Path,
        default=None,
        help="the main checkout whose tasks.db and stamp are used (default: this checkout's main checkout)",
    )
    parser.add_argument(
        '--fused-memory-url',
        default=None,
        help=f'fused-memory base URL (default: ${FUSED_MEMORY_URL_ENV} or {DEFAULT_FUSED_MEMORY_URLS[0]})',
    )
    parser.add_argument(
        '--escalation-url',
        default=None,
        help="escalation MCP URL (default: from the project root's dark-factory-orchestrator.yaml)",
    )
    return parser


def _sweep(args: argparse.Namespace, now: datetime.datetime) -> SweepResult:
    project_root = args.project_root or default_project_root()
    config = RunConfig(root=args.root, project_root=project_root, dry_run=args.dry_run)
    try:
        ports = build_ports(
            project_root, fused_memory_url=args.fused_memory_url, escalation_mcp_url=args.escalation_url
        )
    except SweepFailure as exc:
        return _failed(config, exc, escalation=None, now=now)
    read = functools.partial(read_register, config.root, config.stamp_path)
    return run(config, ports, read_register=read, now=now)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = _sweep(args, datetime.datetime.now(datetime.UTC))
    except Exception as exc:
        print(f'exception_owner_sweep.py: could not run: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    for line in result.lines():
        print(line)
    if result.failure is not None:
        print(f'exception_owner_sweep.py: failed at {result.failure.stage}: {result.failure.detail}', file=sys.stderr)
    print(json.dumps(result.record(), sort_keys=True))
    return 0 if result.ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
