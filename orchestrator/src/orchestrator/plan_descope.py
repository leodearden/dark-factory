"""The descope record and done-step integrity invariant: ``plans/task-amendment-delivery-prd.md`` C4/C5."""
from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping

__all__ = [
    'DESCOPED_STEPS_KEY',
    'DESCOPE_DISPOSITIONS',
    'DESCOPE_PLAN_KEYS',
    'RESCOPED_AT_KEY',
    'RESCOPED_BY_KEY',
    'RESCOPED_BY_UNKNOWN',
    'RESCOPED_NOTE_KEY',
    'DescopedStep',
    'DoneRecord',
    'DoneStepIntegrityVerdict',
    'DuplicateDoneCommit',
    'InvalidDescopeRecord',
    'check_done_step_integrity',
    'commits_match',
    'descope_record_of',
    'done_step_items',
    'done_step_records',
    'same_done_record',
]

RESCOPED_AT_KEY = '_rescoped_at'
RESCOPED_BY_KEY = '_rescoped_by'
RESCOPED_NOTE_KEY = '_rescoped_note'
DESCOPED_STEPS_KEY = '_descoped_steps'
DESCOPE_PLAN_KEYS: tuple[str, ...] = (
    RESCOPED_AT_KEY,
    RESCOPED_BY_KEY,
    RESCOPED_NOTE_KEY,
    DESCOPED_STEPS_KEY,
)
RESCOPED_BY_UNKNOWN = 'unknown'

DESCOPE_DISPOSITIONS: frozenset[str] = frozenset({'orphan', 'revert', 'ride-along'})

_MIN_PREFIX_LEN = 7

DoneRecord = tuple[str, str]


class InvalidDescopeRecord(ValueError):
    def __init__(self, field: str, value: object, requirement: str) -> None:
        self.field = field
        self.value = value
        super().__init__(f'descope record field {field!r} {requirement}; got {value!r}')


def _non_blank_str(field: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidDescopeRecord(field, value, 'must be a non-blank string')
    return value


@dataclasses.dataclass(frozen=True, kw_only=True)
class DescopedStep:
    """One ``_descoped_steps`` entry: a done step a revalidation dropped, and why."""

    step_id: str
    commit: str
    disposition: str
    authority: str
    note: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            _non_blank_str(field.name, getattr(self, field.name))
        if self.disposition not in DESCOPE_DISPOSITIONS:
            allowed = ', '.join(sorted(DESCOPE_DISPOSITIONS))
            raise InvalidDescopeRecord(
                'disposition', self.disposition, f'must be one of: {allowed}',
            )

    @classmethod
    def from_mapping(cls, obj: object) -> DescopedStep:
        if not isinstance(obj, Mapping):
            raise InvalidDescopeRecord('record', obj, 'must be a mapping')
        return cls(
            step_id=_non_blank_str('step_id', obj.get('step_id')),
            commit=_non_blank_str('commit', obj.get('commit')),
            disposition=_non_blank_str('disposition', obj.get('disposition')),
            authority=_non_blank_str('authority', obj.get('authority')),
            note=_non_blank_str('note', obj.get('note')),
        )

    @property
    def done_record(self) -> DoneRecord:
        return (self.step_id, self.commit)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DuplicateDoneCommit:
    commit: str
    step_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {'commit': self.commit, 'step_ids': list(self.step_ids)}


@dataclasses.dataclass(frozen=True)
class DoneStepIntegrityVerdict:
    undisposed_drops: tuple[DoneRecord, ...]
    introduced_duplicates: tuple[DuplicateDoneCommit, ...]
    invalid_descope_records: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.undisposed_drops and not self.introduced_duplicates

    def to_dict(self) -> dict:
        return {
            'undisposed_drops': [
                {'step_id': step_id, 'commit': commit}
                for step_id, commit in self.undisposed_drops
            ],
            'introduced_duplicates': [dup.to_dict() for dup in self.introduced_duplicates],
            'invalid_descope_records': list(self.invalid_descope_records),
        }

    def reason(self) -> str:
        return (
            f'done-step integrity: {len(self.undisposed_drops)} done step(s) dropped '
            f'without a descope record, {len(self.introduced_duplicates)} commit(s) '
            f'newly shared by done steps, {len(self.invalid_descope_records)} '
            f'invalid descope record(s)'
        )


def done_step_items(plan: Mapping) -> list[dict]:
    """Every done prerequisite/step that records a commit, as the plan's own dicts."""
    return [
        item
        for column in ('prerequisites', 'steps')
        for item in plan.get(column) or []
        if isinstance(item, dict) and item.get('status') == 'done' and item.get('commit')
    ]


def done_step_records(plan: Mapping) -> tuple[DoneRecord, ...]:
    """``(id, commit)`` of every :func:`done_step_items` entry."""
    return tuple((str(item.get('id')), str(item['commit'])) for item in done_step_items(plan))


def commits_match(a: str, b: str) -> bool:
    """Equal shas, or one a prefix of the other when the shorter has >= 7 chars."""
    shorter, longer = sorted((a.lower(), b.lower()), key=len)
    if shorter == longer:
        return True
    return len(shorter) >= _MIN_PREFIX_LEN and longer.startswith(shorter)


def descope_record_of(plan: Mapping) -> dict:
    return {key: plan[key] for key in DESCOPE_PLAN_KEYS if key in plan}


def check_done_step_integrity(before: Mapping, after: Mapping) -> DoneStepIntegrityVerdict:
    """C5: every dropped done record is disposed, and no done commit is newly shared."""
    before_records = done_step_records(before)
    after_records = done_step_records(after)
    descoped, invalid = _parse_descoped_steps(after)
    dropped = [record for record in before_records if not _has_match(record, after_records)]
    added = [record for record in after_records if not _has_match(record, before_records)]
    return DoneStepIntegrityVerdict(
        undisposed_drops=tuple(record for record in dropped if not _is_disposed(record, descoped)),
        introduced_duplicates=_introduced_duplicates(added, after_records),
        invalid_descope_records=invalid,
    )


def same_done_record(a: DoneRecord, b: DoneRecord) -> bool:
    """The same step id, and commits that :func:`commits_match`."""
    a_id, a_commit = a
    b_id, b_commit = b
    return a_id == b_id and commits_match(a_commit, b_commit)


def _has_match(record: DoneRecord, candidates: Iterable[DoneRecord]) -> bool:
    return any(same_done_record(record, other) for other in candidates)


def _is_disposed(record: DoneRecord, descoped: Iterable[DescopedStep]) -> bool:
    return _has_match(record, (entry.done_record for entry in descoped))


def _parse_descoped_steps(plan: Mapping) -> tuple[tuple[DescopedStep, ...], tuple[str, ...]]:
    entries = plan.get(DESCOPED_STEPS_KEY) or []
    if not isinstance(entries, list):
        return (), (str(InvalidDescopeRecord(DESCOPED_STEPS_KEY, entries, 'must be a list')),)
    valid: list[DescopedStep] = []
    invalid: list[str] = []
    for entry in entries:
        try:
            valid.append(DescopedStep.from_mapping(entry))
        except InvalidDescopeRecord as exc:
            invalid.append(str(exc))
    return tuple(valid), tuple(invalid)


def _introduced_duplicates(
    added: Iterable[DoneRecord], after_records: tuple[DoneRecord, ...],
) -> tuple[DuplicateDoneCommit, ...]:
    groups: dict[tuple[str, ...], DuplicateDoneCommit] = {}
    for _step_id, commit in added:
        sharing = tuple(sorted({
            other_id for other_id, other_commit in after_records
            if commits_match(other_commit, commit)
        }))
        if len(sharing) > 1 and sharing not in groups:
            groups[sharing] = DuplicateDoneCommit(commit=commit, step_ids=sharing)
    return tuple(groups.values())
