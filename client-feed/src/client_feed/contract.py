"""The one spelling of the client feed's shape; plans/client-feed-prd.md §7 and §14.3 are the normative prose."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from enum import StrEnum
from itertools import chain
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, TypeAlias, TypeVar, get_args

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)

FEED_VERSION: Final = '1.0'


def feed_major(version: str) -> int:
    """The major component of a `<major>.<minor>` feed version."""
    return int(version.split('.', 1)[0])


FEED_MAJOR: Final = feed_major(FEED_VERSION)

FEED_FILENAME: Final = 'feed.jsonl'
STATE_FILENAME: Final = 'state.json'
INBOX_FILENAME: Final = 'inbox.jsonl'
ENGAGEMENT_FILENAME: Final = 'engagement.json'

NOT_RECORDED: Final = 'not_recorded'
NotRecorded = Literal['not_recorded']


class FeedError(Exception):
    """A feed, inbox or engagement text a reader must refuse."""


class ContractViolation(FeedError):
    """A text that is not a valid record of the contract."""

    def __init__(self, *, line_no: int | None, detail: str) -> None:
        self.line_no = line_no
        self.detail = detail
        where = 'the whole file' if line_no is None else f'line {line_no}'
        super().__init__(f'contract violation at {where}: {detail}')


class UnsupportedFeedVersion(FeedError):
    """A feed line whose major version this reader does not implement."""

    def __init__(self, *, line_no: int, found: str, supported_major: int) -> None:
        self.line_no = line_no
        self.found = found
        self.supported_major = supported_major
        super().__init__(
            f'unsupported feed version at line {line_no}: found {found!r}, this reader'
            f' supports major {supported_major} only'
        )


def _require_utc(value: datetime) -> datetime:
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f'timestamp must be UTC, got offset {offset} in {value.isoformat()}')
    return value


def _require_repo_relative(path: str) -> str:
    segments = path.split('/')
    if path.startswith('/') or any(segment in ('', '.', '..') for segment in segments):
        raise ValueError(
            f'repo path must be relative with no empty, "." or ".." segment, got {path!r}'
        )
    return path


UtcTimestamp = Annotated[AwareDatetime, AfterValidator(_require_utc)]
CommitSha = Annotated[str, StringConstraints(pattern=r'^(?:[0-9a-f]{40}|[0-9a-f]{64})$')]
RepoPath = Annotated[str, AfterValidator(_require_repo_relative)]
ProjectId = Annotated[str, StringConstraints(pattern=r'^[a-z0-9][a-z0-9_-]{0,63}$')]
FeedVersion = Annotated[str, StringConstraints(pattern=r'^\d+\.\d+$')]
Seq = Annotated[StrictInt, Field(ge=1)]
TaskId = Annotated[StrictInt, Field(ge=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
CriterionId = Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')]
Token = Annotated[str, StringConstraints(pattern=r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')]
_DefectText = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
_DefectStep = Annotated[str, StringConstraints(min_length=1, max_length=500)]


class Source(StrEnum):
    TASKS = 'tasks'
    EVENTS = 'events'
    GIT = 'git'
    ARCHIVE = 'archive'
    PRDS = 'prds'
    CRITERIA = 'criteria'
    ENGAGEMENT = 'engagement'
    INBOX = 'inbox'


class SchedulerState(StrEnum):
    RUNNING = 'running'
    PAUSED = 'paused'


class WaitingOn(StrEnum):
    FACTORY = 'factory'
    LEO = 'leo'


class TaskStatus(StrEnum):
    """shared.task_statuses.TaskStatus re-declared so the contract stays vendorable; parity is tested."""

    PENDING = 'pending'
    IN_PROGRESS = 'in-progress'
    BLOCKED = 'blocked'
    DEFERRED = 'deferred'
    REVIEW = 'review'
    MERGE_DEFERRED = 'merge-deferred'
    INFRA_HOLD = 'infra-hold'
    DONE = 'done'
    CANCELLED = 'cancelled'


class VerifyKind(StrEnum):
    TASK = 'task'
    MERGE = 'merge'


class CaseOutcome(StrEnum):
    PASSED = 'passed'
    FAILED = 'failed'
    SKIPPED = 'skipped'
    ERRORED = 'errored'


class FileKind(StrEnum):
    PRODUCTION = 'production'
    TEST = 'test'
    DOCS = 'docs'
    GENERATED = 'generated'


class RefusalReason(StrEnum):
    NOT_IN_TREE = 'not_in_tree'
    NOT_VISIBLE = 'not_visible'
    NOT_REVEALABLE = 'not_revealable'
    OVER_BUDGET = 'over_budget'
    SYMLINK = 'symlink'


class RunKind(StrEnum):
    VERIFY = 'verify'
    STAGING = 'staging'


class CriterionOutcome(StrEnum):
    PASSED = 'passed'
    FAILED = 'failed'
    NOT_RUN = 'not_run'


class CriterionKind(StrEnum):
    SUCCESS = 'success'
    FAILURE = 'failure'


class QuarantineReason(StrEnum):
    SUSPECTED_INJECTION = 'suspected_injection'
    NOT_RESTATABLE = 'not_restatable'
    RESTATER_DECLINED = 'restater_declined'
    RESTATER_FAILED = 'restater_failed'


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')
    __test__ = False  # pytest never collects a model imported into a test module


def _duplicates(values: Iterable[str]) -> list[str]:
    """The values that occur more than once, sorted."""
    return sorted(value for value, count in Counter(values).items() if count > 1)


class _Envelope(_Record):
    """The §7.1 fields every feed line carries; `at` is when the fact occurred."""

    seq: Seq
    at: UtcTimestamp
    project_id: ProjectId
    feed_version: FeedVersion

    @property
    def natural_key(self) -> str | int | None:
        """The key under which a re-emission replaces its predecessor; None for unkeyed events."""
        return None


class Degradation(_Record):
    """A source the publisher could not read on this pass."""

    source: Source
    reason: str


class Heartbeat(_Envelope):
    """The publisher's health, every pass."""

    type: Literal['heartbeat'] = 'heartbeat'
    head: CommitSha | None
    as_of: UtcTimestamp
    engagement_version: NonNegativeInt | None
    watermark_seq: NonNegativeInt
    degraded: tuple[Degradation, ...]


class FactoryState(_Envelope):
    """Whether the factory's scheduler runs, and how long it has been quiet."""

    type: Literal['factory_state'] = 'factory_state'
    scheduler: SchedulerState
    since: UtcTimestamp | None
    last_event_at: UtcTimestamp | None
    idle_seconds: NonNegativeInt | None


class PrdUpserted(_Envelope):
    """A PRD and its tasks' status counts."""

    type: Literal['prd_upserted'] = 'prd_upserted'
    prd_path: RepoPath
    title: str
    excerpt: str
    task_counts: dict[TaskStatus, NonNegativeInt]

    @property
    def natural_key(self) -> str:
        return self.prd_path


class TaskUpserted(_Envelope):
    """A task's allowlisted fields."""

    type: Literal['task_upserted'] = 'task_upserted'
    task_id: TaskId
    title: str
    status: TaskStatus
    waiting_on: WaitingOn | None
    updated_at: UtcTimestamp
    depends_on: tuple[TaskId, ...]
    prd_path: RepoPath | None
    prd_task_label: str | None
    landed_commit: CommitSha | None

    @property
    def natural_key(self) -> int:
        return self.task_id


class TaskPhase(_Envelope):
    """A task entering or leaving a workflow phase at the envelope's `at`."""

    type: Literal['task_phase'] = 'task_phase'
    task_id: TaskId
    phase: NonEmptyStr
    entered: StrictBool


class VerifyLeg(_Record):
    """One command of a verify run and its exit code."""

    name: str
    category: str
    rc: StrictInt


class VerifyRun(_Envelope):
    """A task or merge verify and whether it passed."""

    type: Literal['verify_run'] = 'verify_run'
    run_key: NonEmptyStr
    task_id: TaskId | None
    kind: VerifyKind
    passed: StrictBool
    duration_ms: NonNegativeInt | None
    commit: CommitSha | None
    legs: tuple[VerifyLeg, ...] | NotRecorded

    @property
    def natural_key(self) -> str:
        return self.run_key


class CaseCounts(_Record):
    """Case totals of a run, including the cases cut from the listing."""

    passed: NonNegativeInt
    failed: NonNegativeInt
    skipped: NonNegativeInt
    errored: NonNegativeInt
    truncated: NonNegativeInt


class CaseResult(_Record):
    """One test case's outcome, never its message or output."""

    id: NonEmptyStr
    outcome: CaseOutcome
    duration_ms: NonNegativeInt | None
    criterion: str | None
    exception_type: str | None


class VerifyTestResults(_Envelope):
    """The junit cases of one verify run."""

    type: Literal['test_results'] = 'test_results'
    run_key: NonEmptyStr
    counts: CaseCounts
    duration_ms: NonNegativeInt | None
    cases: tuple[CaseResult, ...]

    @property
    def natural_key(self) -> str:
        return self.run_key

    @model_validator(mode='after')
    def _counts_cover_the_cases(self) -> Self:
        counts = self.counts
        counted = counts.passed + counts.failed + counts.skipped + counts.errored
        listed = len(self.cases) + counts.truncated
        if counted != listed:
            raise ValueError(
                f'test_results counts must cover the cases: passed+failed+skipped+errored is'
                f' {counted} but len(cases)+truncated is {len(self.cases)}+{counts.truncated}'
                f' = {listed}'
            )
        return self


class ReviewCounts(_Record):
    """The last review cycle's counts, never its text."""

    cycles: NonNegativeInt
    blocking_count: NonNegativeInt
    suggestion_count: NonNegativeInt
    by_category: dict[str, NonNegativeInt]


class MergeLanded(_Envelope):
    """A merge onto main, landed at the envelope's `at`."""

    type: Literal['merge_landed'] = 'merge_landed'
    commit: CommitSha
    task_ids: tuple[TaskId, ...]
    titles: tuple[str, ...]
    files_changed: NonNegativeInt
    insertions: NonNegativeInt
    deletions: NonNegativeInt
    review: ReviewCounts | NotRecorded

    @property
    def natural_key(self) -> str:
        return self.commit

    @model_validator(mode='after')
    def _one_title_per_task(self) -> Self:
        if len(self.titles) != len(self.task_ids):
            raise ValueError(
                f'merge_landed pairs titles with task_ids: {len(self.titles)} titles for'
                f' {len(self.task_ids)} task_ids'
            )
        return self


class Branch(_Record):
    """A task branch and how far it is ahead of main."""

    name: NonEmptyStr
    task_id: TaskId | None
    ahead: NonNegativeInt
    last_commit_at: UtcTimestamp


class BranchSnapshot(_Envelope):
    """Every task branch."""

    type: Literal['branch_snapshot'] = 'branch_snapshot'
    branches: tuple[Branch, ...]


class TreeTotals(_Record):
    """Line totals of the visible tree."""

    files: NonNegativeInt
    production_lines: NonNegativeInt
    test_lines: NonNegativeInt
    docs_lines: NonNegativeInt


class TreeFile(_Record):
    """One visible file of the tree."""

    path: RepoPath
    kind: FileKind
    lines: NonNegativeInt
    bytes: NonNegativeInt
    last_changed_at: UtcTimestamp | None


class RevealBudget(_Record):
    """The reveal budget at this tree."""

    proportion: Annotated[float, Field(ge=0, le=1)]
    counted_lines: NonNegativeInt
    budget_lines: NonNegativeInt
    spent_lines: NonNegativeInt
    floor_lines: NonNegativeInt


class TreeSnapshot(_Envelope):
    """The visible tree at a main commit."""

    type: Literal['tree_snapshot'] = 'tree_snapshot'
    commit: CommitSha
    totals: TreeTotals
    files: tuple[TreeFile, ...]
    budget: RevealBudget


class FileRevealed(_Envelope):
    """A granted file's content at a commit, or its deletion."""

    type: Literal['file_revealed'] = 'file_revealed'
    path: RepoPath
    commit: CommitSha
    lines: NonNegativeInt
    content: str | None
    deleted: StrictBool
    renamed_from: RepoPath | None

    @property
    def natural_key(self) -> str:
        return self.path

    @model_validator(mode='after')
    def _content_present_unless_deleted(self) -> Self:
        if self.deleted == (self.content is not None):
            raise ValueError(
                f'file_revealed carries content exactly when not deleted: deleted={self.deleted}'
                f' with content={"None" if self.content is None else "<str>"}'
            )
        return self


class RevealRefused(_Envelope):
    """A refused reveal request; `path` is the raw requested text."""

    type: Literal['reveal_refused'] = 'reveal_refused'
    path: str
    reason: RefusalReason


class CriterionResult(_Record):
    """One acceptance criterion's outcome and the tests bound to it."""

    id: NonEmptyStr
    outcome: CriterionOutcome
    tests: tuple[str, ...]


class CriteriaResults(_Envelope):
    """The newest criteria outcomes of one run kind."""

    type: Literal['criteria_results'] = 'criteria_results'
    run_kind: RunKind
    commit: CommitSha | None
    run_at: UtcTimestamp
    results: tuple[CriterionResult, ...]


class Criterion(_Record):
    """One acceptance criterion as acceptance/criteria.yaml declares it."""

    id: NonEmptyStr
    prd: RepoPath
    capability: NonEmptyStr
    statement: NonEmptyStr
    kind: CriterionKind


class CriteriaDeclared(_Envelope):
    """Every acceptance criterion the project declares; each declaration replaces the last."""

    type: Literal['criteria_declared'] = 'criteria_declared'
    criteria: tuple[Criterion, ...]

    @model_validator(mode='after')
    def _ids_are_unique(self) -> Self:
        duplicates = _duplicates(criterion.id for criterion in self.criteria)
        if duplicates:
            raise ValueError(
                f'criteria_declared ids must be unique: {", ".join(duplicates)} declared more'
                f' than once'
            )
        return self


FeedEvent = Annotated[
    Heartbeat
    | FactoryState
    | PrdUpserted
    | TaskUpserted
    | TaskPhase
    | VerifyRun
    | VerifyTestResults
    | MergeLanded
    | BranchSnapshot
    | TreeSnapshot
    | FileRevealed
    | RevealRefused
    | CriteriaResults
    | CriteriaDeclared,
    Field(discriminator='type'),
]
FEED_EVENT_ADAPTER: Final = TypeAdapter(FeedEvent)

EVENT_MODELS: Final[Mapping[str, type[_Envelope]]] = MappingProxyType(
    {model.model_fields['type'].default: model for model in get_args(get_args(FeedEvent)[0])}
)


class EventHeader(_Envelope):
    """A line of a type this reader does not know: its envelope only, payload unread."""

    type: NonEmptyStr


FeedLine: TypeAlias = FeedEvent | EventHeader
_HEADER_KEYS: Final = frozenset(EventHeader.model_fields)


class InFlightTask(_Record):
    """A task inside a phase, since the phase's entry."""

    task_id: TaskId
    phase: NonEmptyStr
    since: UtcTimestamp


class RevealedFile(_Record):
    """A file currently revealed; its content stays in the feed."""

    path: RepoPath
    commit: CommitSha
    lines: NonNegativeInt


class FeedState(_Record):
    """state.json: the fold of the feed up to watermark_seq."""

    project_id: ProjectId
    feed_version: FeedVersion
    watermark_seq: NonNegativeInt
    as_of: UtcTimestamp | None
    head: CommitSha | None
    engagement_version: NonNegativeInt | None
    degraded: tuple[Degradation, ...]
    factory: FactoryState | None
    prds: dict[str, PrdUpserted]
    tasks: dict[int, TaskUpserted]
    in_flight: tuple[InFlightTask, ...]
    branches: BranchSnapshot | None
    tree: TreeSnapshot | None
    revealed: tuple[RevealedFile, ...]
    criteria: dict[RunKind, CriteriaResults]
    criteria_declared: dict[str, Criterion]
    recent: tuple[FeedEvent, ...]

    @model_validator(mode='after')
    def _keys_name_their_records(self) -> Self:
        keyed_records = chain(
            (('prds', key, prd.prd_path) for key, prd in self.prds.items()),
            (('tasks', key, task.task_id) for key, task in self.tasks.items()),
            (('criteria', key, run.run_kind) for key, run in self.criteria.items()),
            (('criteria_declared', key, c.id) for key, c in self.criteria_declared.items()),
        )
        for section, key, record_key in keyed_records:
            if key != record_key:
                raise ValueError(f"{section} key '{key}' holds the record of '{record_key}'")
        return self


class RevealRequested(_Record):
    """inbox.jsonl: the portal asking for a file."""

    seq: Seq
    at: UtcTimestamp
    type: Literal['reveal_requested'] = 'reveal_requested'
    path: str


class Restated(_Record):
    """A client's defect report in the restater's words."""

    kind: Literal['restated'] = 'restated'
    observed: _DefectText
    expected: _DefectText
    steps: Annotated[tuple[_DefectStep, ...], Field(max_length=20)]


class Quarantined(_Record):
    """A client's defect report the restater would not restate, and why."""

    kind: Literal['quarantined'] = 'quarantined'
    reasons: Annotated[tuple[QuarantineReason, ...], Field(min_length=1)]

    @model_validator(mode='after')
    def _reasons_are_unique(self) -> Self:
        duplicates = _duplicates(self.reasons)
        if duplicates:
            raise ValueError(
                f'quarantined reasons must be unique: {", ".join(duplicates)} given more than once'
            )
        return self


class Restater(_Record):
    """The restater that read the client's words."""

    prompt_version: Token


DefectOutcome = Annotated[Restated | Quarantined, Field(discriminator='kind')]


class DefectReported(_Record):
    """inbox.jsonl: the portal reporting a client defect, restated or quarantined; never the client's own words."""

    seq: Seq
    at: UtcTimestamp
    type: Literal['defect_reported'] = 'defect_reported'
    block: PositiveInt
    report: PositiveInt
    criterion_id: CriterionId
    restater: Restater
    outcome: DefectOutcome


InboxLine = Annotated[RevealRequested | DefectReported, Field(discriminator='type')]
_INBOX_LINE_ADAPTER: Final = TypeAdapter(InboxLine)


class EngagementProjection(_Record):
    """engagement.json: the portal's projection of the engagement record."""

    engagement_version: NonNegativeInt
    project_id: ProjectId
    stage: NonEmptyStr
    delivered_commit: CommitSha | None


_Parsed = TypeVar('_Parsed')


def _json_object(text: str, *, line_no: int | None) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ContractViolation(
            line_no=line_no, detail=f'not JSON: {exc.msg} at column {exc.colno}'
        ) from exc
    if not isinstance(value, dict):
        raise ContractViolation(
            line_no=line_no, detail=f'expected a JSON object, got {type(value).__name__}'
        )
    return value


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    first = errors[0]
    location = '.'.join(str(part) for part in first['loc']) or '<record>'
    more = f' (and {len(errors) - 1} more)' if len(errors) > 1 else ''
    return f'{location}: {first["msg"]}{more}'


def _validated(
    validate: Callable[[Any], _Parsed], value: Any, *, line_no: int | None
) -> _Parsed:
    try:
        return validate(value)
    except ValidationError as exc:
        raise ContractViolation(line_no=line_no, detail=_first_error(exc)) from exc


def _require_supported_major(header: EventHeader, *, line_no: int) -> None:
    if feed_major(header.feed_version) != FEED_MAJOR:
        raise UnsupportedFeedVersion(
            line_no=line_no, found=header.feed_version, supported_major=FEED_MAJOR
        )


def parse_feed_line(text: str, *, line_no: int) -> FeedLine:
    """One feed.jsonl line as its event, or as an EventHeader when its type is unknown (PRD §4.9)."""
    raw = _json_object(text, line_no=line_no)
    envelope = {key: value for key, value in raw.items() if key in _HEADER_KEYS}
    header = _validated(EventHeader.model_validate, envelope, line_no=line_no)
    _require_supported_major(header, line_no=line_no)
    if header.type not in EVENT_MODELS:
        return header
    return _validated(FEED_EVENT_ADAPTER.validate_python, raw, line_no=line_no)


def parse_inbox_line(text: str, *, line_no: int) -> InboxLine:
    """One inbox.jsonl line as the line type its `type` names."""
    return _validated(
        _INBOX_LINE_ADAPTER.validate_python, _json_object(text, line_no=line_no), line_no=line_no
    )


def split_jsonl(text: str) -> tuple[str, ...]:
    """The record texts of a JSONL file, split on '\\n' alone (JSONL's only record separator), so a U+2028, U+2029 or U+0085 held raw inside a JSON string never splits a record."""
    texts = text.split('\n')
    if texts[-1] == '':
        texts.pop()
    return tuple(texts)


def parse_engagement(text: str) -> EngagementProjection:
    """The whole of engagement.json."""
    return _validated(
        EngagementProjection.model_validate, _json_object(text, line_no=None), line_no=None
    )


def parse_state(text: str) -> FeedState:
    """The whole of state.json."""
    return _validated(FeedState.model_validate, _json_object(text, line_no=None), line_no=None)
