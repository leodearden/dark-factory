"""INV-12's nightly owner sweep: every cited debt owner is still alive, or someone hears.

Reads the exception register's ``--json`` for every debt owner a governed list
or an inline marker cites, asks ``exception_owner_liveness`` whether each owner
still owns its debt, and turns a dead owner into exactly one finding, keyed so
that the next night does not repeat it.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import enum
import functools
import json
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
    from _task_db_scan import UNREADABLE_STORE_ERRORS
    from shared.governed_exceptions import (
        MalformedDisposition,
        TaskRef,
        TicketRef,
        owner_from_record,
        owner_record,
    )
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


def pending_escalation_ids(ports: Ports) -> frozenset[str]:
    """The ``task_id`` of every pending escalation: one ``get_pending_escalations`` call."""
    answer = _call(
        ports.escalation, FailureStage.ESCALATION, 'get_pending_escalations', {'compact': True}
    )
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
    pending = pending_escalation_ids(ports)
    resolve = owner_resolver(ports)
    by_owner = {owner: resolve(owner) for owner in dict.fromkeys(c.owner for c in register.citations)}
    verdicts = {(c.scope, c.owner): by_owner[c.owner] for c in register.citations}
    return Scan(register, MappingProxyType(verdicts), pending, now)


class Action(enum.Enum):
    SKIP_LIVE_TASK = 'skip_live_task'
    SKIP_PENDING_ESCALATION = 'skip_pending_escalation'
    ESCALATE_CLOSED_FINDING = 'escalate_closed_finding'
    FILE_TASK = 'file_task'
    FILE_ESCALATION = 'file_escalation'


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


def _unconfirmed_body(body: _Body, filing: FilingOutcome) -> _Body:
    """*body* plus why its filing did not land, for the escalation that replaces it."""
    outcome = filing.outcome
    reason = '' if outcome.reason is None else f' ({outcome.reason})'
    if outcome.task_id is None:
        target = 'no task'
    elif filing.task_row is None:
        target = f'task {outcome.task_id}, which the store does not hold'
    else:
        target = f'task {outcome.task_id}, status {filing.task_row.status!r}, not carrying this finding'
    landed = f'Filing did not land: ticket {filing.ticket} resolved {outcome.status}{reason} to {target}.'
    return dataclasses.replace(
        body, detail=f'{body.detail}\n{landed}', observation=f'{body.observation}; {landed}'
    )


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


class _Outputs:
    """One act() run's outputs: what it has filed, escalated and skipped so far."""

    def __init__(self, ports: Ports, pending: frozenset[str], now: datetime.datetime) -> None:
        self.ports = ports
        self.pending = set(pending)
        self.now = now
        self.counts: collections.Counter[str] = collections.Counter()

    def raise_finding(self, key: FindingKey, body: _Body) -> None:
        """Re-read the carriers, decide, and perform at most one output for *key*."""
        action = decide(
            key, carriers=_carriers(self.ports, FINDING_KEY_FIELD), pending=frozenset(self.pending)
        )
        if action in (Action.SKIP_LIVE_TASK, Action.SKIP_PENDING_ESCALATION):
            self.counts['skipped_existing'] += 1
        elif action is Action.FILE_TASK:
            self._file(key, body)
        else:
            self._escalate(key, body)

    def _file(self, key: FindingKey, body: _Body) -> None:
        filing = _file_task(key, body, self.ports)
        if filing.confirmed:
            self.counts['filed'] += 1
        else:
            self._escalate(key, _unconfirmed_body(body, filing))

    def _escalate(self, key: FindingKey, body: _Body) -> None:
        _escalate(key, body, self.ports, self.now)
        self.pending.add(key.escalation_task_id())
        self.counts['escalated'] += 1


def act(scan: Scan, ports: Ports, *, now: datetime.datetime) -> ActOutcome:
    """Re-resolve each cited owner immediately before acting on it, then act (INV-3)."""
    resolve = owner_resolver(ports)
    outputs = _Outputs(ports, scan.pending, now)
    for citation in scan.register.citations:
        verdict = resolve(citation.owner)
        outputs.counts['checked'] += 1
        outputs.counts['via_successor'] += verdict.liveness is liveness.Liveness.LIVE_VIA_SUCCESSOR
        if verdict.liveness is liveness.Liveness.DEAD:
            outputs.counts['findings'] += 1
            key = FindingKey(FindingKind.DEAD_OWNER, citation.scope, citation.owner)
            outputs.raise_finding(key, _dead_owner_body(citation, verdict))
    return ActOutcome(**outputs.counts)
