"""Program state for a /review-all Phase 6: the one library every program script imports.

The design-time statement is ``plans/review-all-program-tooling-prd.md`` §4
(H1–H3); this module is the machine home it names.
"""
from __future__ import annotations

import enum
import importlib
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CHECKOUT_ROOT = Path(__file__).resolve().parents[4]
_SHARED_SRC = str(CHECKOUT_ROOT / 'shared' / 'src')


def _first_party(name: str) -> types.ModuleType:
    """Import a stdlib-only module from this checkout's ``shared/src``, whatever the cwd."""
    if _SHARED_SRC not in sys.path:
        sys.path.insert(0, _SHARED_SRC)
    return importlib.import_module(name)


_task_statuses = _first_party('shared.task_statuses')

RETURN_SCHEMA = 'review-all-prd-return/1'
PROGRAM_SCHEMA = 'review-all-program/1'
NON_TERMINAL_STATUSES: tuple[str, ...] = tuple(sorted(str(s) for s in _task_statuses.ACTIVE))
TERMINAL_STATUSES: tuple[str, ...] = tuple(sorted(str(s) for s in _task_statuses.TERMINAL))


@dataclass(frozen=True)
class SeamKeys:
    key: str
    fields: tuple[str, ...]


SEAM_WAIT_KEYS = SeamKeys('x_seam_wait', ('stream', 'contract', 'release_owner'))
SEAM_RELEASED_KEYS = SeamKeys('x_seam_released', ('by', 'at', 'onto', 'contract_as_delivered'))


class Pass(enum.StrEnum):
    AUTHOR = 'author'
    DECOMPOSE = 'decompose'
    INVESTIGATE = 'investigate'


class SessionKind(enum.StrEnum):
    AGENT = 'agent'
    SPAWN = 'spawn'


class QuestionSource(enum.StrEnum):
    AUTHOR = 'author'
    CRITIC = 'critic'


class LeadActionOp(enum.StrEnum):
    CANCEL = 'cancel'
    EDGE = 'edge'
    NOTE = 'note'
    STAMP = 'stamp'
    REPATH = 'repath'
    RELEASE = 'release'
    AMEND = 'amend'


class StreamMode(enum.StrEnum):
    AGENT = 'agent'
    SPAWN = 'spawn'


class StreamSize(enum.StrEnum):
    S = 'S'
    M = 'M'
    L = 'L'


class StreamRisk(enum.StrEnum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'


class StreamStatus(enum.StrEnum):
    BRIEFED = 'briefed'
    SPAWNED = 'spawned'
    AUTHORED = 'authored'
    COMMITTED = 'committed'
    DECOMPOSED = 'decomposed'
    RECORDED = 'recorded'


class StreamField(enum.StrEnum):
    """The stream-row fields an ``amend_stream`` ledger line may change."""

    KEYS = 'keys'
    DEPS = 'deps'
    DELETABLE = 'deletable'
    MODE = 'mode'
    REMEDY = 'remedy'


class AmendChange(enum.StrEnum):
    ADD = 'add'
    REMOVE = 'remove'
    SET = 'set'


class ProgramSection(enum.StrEnum):
    SEAMS_EXTRA = 'seams_extra'
    INVARIANTS = 'invariants'
    ROUTED = 'routed'
    CONVENTIONS_EXTRA = 'conventions_extra'
    TRIAGE = 'triage'


class ActionResultState(enum.StrEnum):
    APPLIED = 'applied'
    SKIPPED = 'skipped'
    REJECTED = 'rejected'
    KEPT_FOR_HUMAN = 'kept-for-human'


class RuledBy(enum.StrEnum):
    HUMAN = 'human'
    LEAD = 'lead'
    LEAD_YES = 'lead (--yes)'


class TriggerChainRole(enum.StrEnum):
    HUMAN_GATE = 'human_gate'
    COMPLETION_GATE = 'completion_gate'


class RejectionClass(enum.StrEnum):
    FORM = 'form'
    CONTRADICTION = 'contradiction'


class Priority(enum.StrEnum):
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'


class LedgerOp(enum.StrEnum):
    STREAM_STATUS = 'stream_status'
    AMEND_STREAM = 'amend_stream'
    AMEND_PROGRAM = 'amend_program'
    XREF = 'xref'
    ACTION = 'action'
    ACTION_RESULT = 'action_result'
    SCANNED = 'scanned'
    NOTE = 'note'
    COMMIT = 'commit'


@dataclass(frozen=True)
class LedgerOpSpec:
    """The fields one ledger op carries beyond ``LEDGER_COMMON_FIELDS``, and their vocabularies."""

    required: frozenset[str]
    optional: frozenset[str]
    enums: Mapping[str, frozenset[str]]


def _values(vocabulary: type[enum.StrEnum]) -> frozenset[str]:
    return frozenset(member.value for member in vocabulary)


def _ledger_op(
    required: Sequence[str],
    optional: Sequence[str] = (),
    **enums: type[enum.StrEnum],
) -> LedgerOpSpec:
    return LedgerOpSpec(
        required=frozenset(required),
        optional=frozenset(optional),
        enums=types.MappingProxyType({field: _values(e) for field, e in enums.items()}),
    )


LEDGER_COMMON_FIELDS = frozenset({'seq', 'at', 'by', 'op'})
LEDGER_OPS: Mapping[LedgerOp, LedgerOpSpec] = types.MappingProxyType({
    LedgerOp.STREAM_STATUS: _ledger_op(
        ('stream', 'status'), ('prd_path', 'return'), status=StreamStatus,
    ),
    LedgerOp.AMEND_STREAM: _ledger_op(
        ('stream', 'field', 'change', 'value', 'ruling'),
        field=StreamField, change=AmendChange,
    ),
    LedgerOp.AMEND_PROGRAM: _ledger_op(('section', 'value'), ('ruling',), section=ProgramSection),
    LedgerOp.XREF: _ledger_op(('from', 'to', 'text', 'return')),
    LedgerOp.ACTION: _ledger_op(
        ('id', 'op', 'task', 'target', 'text', 'needs_human', 'return'), op=LeadActionOp,
    ),
    LedgerOp.ACTION_RESULT: _ledger_op(
        ('id', 'state', 'batch', 'detail'), state=ActionResultState,
    ),
    LedgerOp.SCANNED: _ledger_op(('return', 'repaths')),
    LedgerOp.NOTE: _ledger_op(('text',)),
    LedgerOp.COMMIT: _ledger_op(('sha', 'paths', 'measured_at')),
})


class StateError(Exception):
    """A program-state fact is malformed or contradicts another; the message names it."""


class ProgramInvalid(StateError):
    def __init__(self, violations: Sequence[str]) -> None:
        self.violations = tuple(violations)
        super().__init__('program.json is invalid: ' + '; '.join(self.violations))


class LedgerRefused(StateError):
    """An append-only JSONL line was refused; nothing was written."""


class StoreReaderError(StateError):
    """The task store could not be read; the message names the server and the failure."""


class StoreReaderUnsupported(StoreReaderError):
    """The server lacks a capability the reader requires (INV-11); there is no fallback."""

    def __init__(self, missing: str, detail: str) -> None:
        self.missing = missing
        super().__init__(f'the task store server does not support {missing!r}: {detail}')


class AmbiguousStream(StateError):
    """A task's finding keys belong to more than one stream."""
