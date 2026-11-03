"""Compare-and-swap and readback for task prose writes (task 4650).

Two defects on the ``update_task`` path: a ``description``/``details``
amendment could be reported as applied without the stored text ever being
read back, and two writers composing against the same read could silently
clobber each other. The fingerprint is the shared task-prose one; the append
and details-over-prompt semantics this module relies on live in
``backends/sqlite_task_backend.py::SqliteTaskBackend.update_task``.

A leaf ``middleware`` module: it imports nothing from ``reconciliation/`` or
``server/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from shared.task_text_fingerprint import task_text_fingerprint

TASK_TEXT_FP_KEY = 'task_text_fp'
TEXT_BASE_STALE = 'TaskTextBaseStale'
TEXT_BASE_REQUIRED = 'ReconTaskTextBaseRequired'
TEXT_VERIFIED_KEY = 'text_verified'


def text_fp_of(task: object) -> str:
    if not isinstance(task, Mapping):
        return task_text_fingerprint(None, None)
    return task_text_fingerprint(task.get('description'), task.get('details'))


@dataclass(frozen=True)
class PlannedTextWrite:
    description: str | None
    details: str | None
    details_appends: bool

    @property
    def writes_text(self) -> bool:
        return self.description is not None or self.details is not None

    @property
    def replaces_text(self) -> bool:
        return self.description is not None or (
            self.details is not None and not self.details_appends
        )

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, payload in (('description', self.description), ('details', self.details))
            if payload is not None
        )


def plan_text_write(update_kwargs: Mapping[str, Any]) -> PlannedTextWrite:
    details = update_kwargs.get('details')
    return PlannedTextWrite(
        description=update_kwargs.get('description'),
        details=details if details is not None else update_kwargs.get('prompt'),
        details_appends=bool(update_kwargs.get('append')),
    )


_REREAD_HINT = (
    'Re-read the task with get_task, recompose the amendment against the '
    'description/details that read returns, and resend it with expected_text_fp '
    f'set to that response\'s {TASK_TEXT_FP_KEY}. Never resend the same text.'
)


@dataclass(frozen=True)
class TextBaseRejection:
    error_type: str
    task_id: str
    agent_id: str | None
    fields: tuple[str, ...]
    expected_text_fp: str | None
    reason: str
    hint: str = _REREAD_HINT

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'error': self.reason,
            'error_type': self.error_type,
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'op': 'update_task',
            'fields': list(self.fields),
            'expected_text_fp': self.expected_text_fp,
            'hint': self.hint,
        }


def check_text_base(
    *,
    before: object,
    write: PlannedTextWrite,
    expected_text_fp: str | None,
    fp_required: bool,
    task_id: str,
    agent_id: str | None,
) -> TextBaseRejection | None:
    if not write.writes_text:
        return None
    if expected_text_fp is not None and expected_text_fp != text_fp_of(before):
        return TextBaseRejection(
            error_type=TEXT_BASE_STALE,
            task_id=task_id,
            agent_id=agent_id,
            fields=write.fields,
            expected_text_fp=expected_text_fp,
            reason=(
                'the task text changed since the read this write was composed '
                'against; writing it would overwrite that change'
            ),
        )
    if expected_text_fp is None and fp_required and write.replaces_text:
        return TextBaseRejection(
            error_type=TEXT_BASE_REQUIRED,
            task_id=task_id,
            agent_id=agent_id,
            fields=write.fields,
            expected_text_fp=None,
            reason=(
                'a reconciliation-stage write that replaces description or details '
                'must name the text it was composed against via expected_text_fp'
            ),
        )
    return None


@dataclass(frozen=True)
class TextFieldVerification:
    field: str
    confirmed: bool
    appended: bool


@dataclass(frozen=True)
class TextWriteVerification:
    fields: tuple[TextFieldVerification, ...]

    @property
    def confirmed(self) -> bool:
        return all(field.confirmed for field in self.fields)

    @property
    def unconfirmed_fields(self) -> tuple[str, ...]:
        return tuple(field.field for field in self.fields if not field.confirmed)

    def to_result_keys(self) -> dict[str, Any]:
        return {
            TEXT_VERIFIED_KEY: self.confirmed,
            'text_verification': {
                field.field: {'confirmed': field.confirmed, 'appended': field.appended}
                for field in self.fields
            },
        }


def _collapse_ws(text: object) -> str:
    return ' '.join(text.split()) if isinstance(text, str) else ''


def verify_text_write(after: object, write: PlannedTextWrite) -> TextWriteVerification | None:
    if not write.writes_text:
        return None
    stored = after if isinstance(after, Mapping) else {}
    written = (
        ('description', write.description, False),
        ('details', write.details, write.details_appends),
    )
    return TextWriteVerification(
        fields=tuple(
            TextFieldVerification(
                field=field,
                confirmed=_collapse_ws(payload) in _collapse_ws(stored.get(field)),
                appended=appended,
            )
            for field, payload, appended in written
            if payload is not None
        ),
    )
