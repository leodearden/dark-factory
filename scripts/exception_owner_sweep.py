"""INV-12's nightly owner sweep: every cited debt owner is still alive, or someone hears.

Reads the exception register's ``--json`` for every debt owner a governed list
or an inline marker cites, asks ``exception_owner_liveness`` whether each owner
still owns its debt, and turns a dead owner into exactly one finding, keyed so
that the next night does not repeat it.
"""

from __future__ import annotations

import enum
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

try:
    import _checkout as _checkout
    from shared.governed_exceptions import (
        MalformedDisposition,
        TaskRef,
        TicketRef,
        owner_from_record,
        owner_record,
    )
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
    """One owner cited in one scope, with every site that cites it."""

    scope: str
    owner: Owner
    sites: tuple[str, ...]


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


def _declaration_cites(declaration: Mapping[str, object], where: str) -> list[tuple[str, Owner, str]]:
    list_id = _field(declaration, 'list_id', str, where)
    cites = [(list_id, _debt_owner(declaration.get('default'), f'{where}default'), 'default')]
    for index, override in enumerate(_entries(declaration, 'overrides', where)):
        key = _field(override, 'key', str, f'{where}overrides[{index}].')
        owner = _debt_owner(override.get('disposition'), f'{where}overrides[{index}].disposition')
        cites.append((list_id, owner, key))
    return [(scope, owner, site) for scope, owner, site in cites if owner is not None]


def _inline_cites(debt: Mapping[str, object], where: str) -> tuple[str, Owner, tuple[str, int]]:
    path = _field(debt, 'path', str, where)
    line = _field(debt, 'line', int, where)
    return INLINE_SCOPE, _owner(debt.get('owner'), f'{where}owner'), (path, line)


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
    sites: dict[tuple[str, Owner], set[tuple[str, int] | tuple[str]]] = {}
    for index, declaration in enumerate(_entries(document, 'declarations', '')):
        for scope, owner, key in _declaration_cites(declaration, f'declarations[{index}].'):
            sites.setdefault((scope, owner), set()).add((key,))
    inline = _field(document, 'inline', dict, '')
    for index, debt in enumerate(_entries(inline, 'debt', 'inline.')):
        scope, owner, site = _inline_cites(debt, f'inline.debt[{index}].')
        sites.setdefault((scope, owner), set()).add(site)
    baseline = _field(document, 'baseline', dict, '')
    slack = baseline.get('slack_total')
    return RegisterView(
        citations=tuple(
            Citation(scope, owner, tuple(':'.join(map(str, site)) for site in sorted(sites[scope, owner])))
            for scope, owner in sorted(sites, key=_citation_order)
        ),
        slack_total=0 if slack is None else _field(baseline, 'slack_total', int, 'baseline.'),
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
