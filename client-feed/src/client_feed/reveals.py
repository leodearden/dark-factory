"""Each reveal request answered, and each revealed file followed to main's head, as feed events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from client_feed.budget import AddedLineCounts, CountedSet, Granted, counted_set, judge, lines_since
from client_feed.contract import (
    EngagementProjection,
    EventHeader,
    FeedEvent,
    FeedLine,
    FeedState,
    FileRevealed,
    RefusalReason,
    RevealBudget,
    RevealedFile,
    RevealRefused,
    RevealRequested,
    TreeFile,
)
from client_feed.envelope import Envelope, Stamp
from client_feed.policy import Policy
from client_feed.sources.git import (
    ChangeStatus,
    PathChange,
    TreeEntry,
    blob_reader,
    first_parent_changes,
    is_ancestor,
    tree_changes,
    tree_entries,
)

AnswerEvent = FileRevealed | RevealRefused


@dataclass(frozen=True)
class RevealAnswer:
    """What a file_revealed says, apart from the envelope that places it in the feed."""

    path: str
    commit: str
    lines: int
    content: str | None
    deleted: bool
    renamed_from: str | None

    @classmethod
    def of(cls, event: FileRevealed) -> RevealAnswer:
        return cls(
            path=event.path,
            commit=event.commit,
            lines=event.lines,
            content=event.content,
            deleted=event.deleted,
            renamed_from=event.renamed_from,
        )

    def placed(self, envelope: Envelope) -> FileRevealed:
        return FileRevealed(
            **envelope,
            path=self.path,
            commit=self.commit,
            lines=self.lines,
            content=self.content,
            deleted=self.deleted,
            renamed_from=self.renamed_from,
        )


@dataclass(frozen=True)
class RefusalAnswer:
    """What a reveal_refused says, apart from the envelope that places it in the feed."""

    path: str
    reason: RefusalReason

    @classmethod
    def of(cls, event: RevealRefused) -> RefusalAnswer:
        return cls(path=event.path, reason=event.reason)

    def placed(self, envelope: Envelope) -> RevealRefused:
        return RevealRefused(**envelope, path=self.path, reason=self.reason)


Answer = RevealAnswer | RefusalAnswer


def answer_of(event: FeedEvent) -> Answer | None:
    """What an answer event says wherever it lands, its identity; None for any other event."""
    if isinstance(event, FileRevealed):
        return RevealAnswer.of(event)
    if isinstance(event, RevealRefused):
        return RefusalAnswer.of(event)
    return None


def answered_in(lines: Iterable[FeedLine]) -> Counter[Answer]:
    """The answers a stopped writer appended but never recorded in its cursor, counted by what they say."""
    answers = (answer_of(line) for line in lines if not isinstance(line, EventHeader))
    return Counter(answer for answer in answers if answer is not None)


@dataclass(frozen=True)
class RevealOutput:
    """One pass's reveal events, the budget after them, the positions they answered through, and the added-line counts to carry."""

    events: tuple[AnswerEvent, ...]
    budget: RevealBudget
    inbox_offset: int
    reveals_head: str
    added_line_counts: AddedLineCounts | None


@dataclass(frozen=True)
class _Head:
    """What every judgement at head shares: its tree entries, the counted set, and one blob reader."""

    commit: str
    policy: Policy
    entries: Mapping[str, TreeEntry]
    counted: CountedSet
    read: Callable[[str], bytes]

    def judge(self, path: str, revealed: set[str]) -> Granted | RefusalReason:
        return judge(
            path,
            entries=self.entries,
            policy=self.policy,
            counted=self.counted,
            revealed=revealed,
            read=self.read,
        )


def _revealed(path: str, head: str, granted: Granted, *, renamed_from: str | None) -> RevealAnswer:
    return RevealAnswer(
        path=path,
        commit=head,
        lines=granted.lines,
        content=granted.text,
        deleted=False,
        renamed_from=renamed_from,
    )


def _ended(origin: str, head: str) -> RevealAnswer:
    return RevealAnswer(
        path=origin, commit=head, lines=0, content=None, deleted=True, renamed_from=None
    )


@dataclass(frozen=True)
class _Followed:
    """A revealed path followed commit by commit: where it is now, its blob before its first change and after its last, or its end."""

    origin: str
    path: str
    first_blob: str | None = None
    blob: str | None = None
    ended: bool = False

    @property
    def untouched(self) -> bool:
        return not self.ended and (
            self.first_blob is None or (self.path == self.origin and self.blob == self.first_blob)
        )

    def through(self, change: PathChange) -> _Followed:
        first = self.first_blob or change.old_blob
        if change.status is ChangeStatus.DELETED or change.new_path is None:
            return replace(self, first_blob=first, ended=True)
        return replace(self, path=change.new_path, first_blob=first, blob=change.new_blob)


def _follow(
    origins: Iterable[str], change_sets: Iterable[Sequence[PathChange]]
) -> tuple[_Followed, ...]:
    """Each origin through the change sets in order; a deleted path ends, and a path added later is never picked up."""
    tracks = [_Followed(origin=origin, path=origin) for origin in origins]
    for changes in change_sets:
        by_old = {change.old_path: change for change in changes if change.old_path is not None}
        tracks = [
            track if track.ended or track.path not in by_old else track.through(by_old[track.path])
            for track in tracks
        ]
    return tuple(tracks)


def _tracks(
    root: Path, revealed: Sequence[RevealedFile], *, since: str | None, head: str
) -> tuple[_Followed, ...]:
    """Revealed files not already at head, through main's first-parent commits since `since`; else each from its own commit."""
    behind = [file for file in revealed if file.commit != head]
    if not behind or since == head:
        return ()
    if since is not None and is_ancestor(root, since, head):
        return _follow(
            (file.path for file in behind), first_parent_changes(root, since=since, head=head)
        )
    by_commit: dict[str, list[str]] = {}
    for file in behind:
        by_commit.setdefault(file.commit, []).append(file.path)
    return tuple(
        track
        for commit, paths in by_commit.items()
        for track in _follow(paths, (tree_changes(root, commit, head),))
    )


def _follow_events(
    tracks: Iterable[_Followed], at_head: _Head, revealed: set[str]
) -> list[Answer]:
    """A moved or changed file re-emitted at head while it stays revealable; otherwise its reveal ends at its origin, ends first."""
    touched = sorted((track for track in tracks if not track.untouched), key=lambda t: t.origin)
    verdicts = [
        (track, None if track.ended else at_head.judge(track.path, revealed | {track.path}))
        for track in touched
    ]
    revealed.difference_update(track.origin for track in touched)
    ended: list[Answer] = []
    moved: list[Answer] = []
    for track, verdict in verdicts:
        if isinstance(verdict, Granted):
            revealed.add(track.path)
            renamed_from = None if track.path == track.origin else track.origin
            moved.append(_revealed(track.path, at_head.commit, verdict, renamed_from=renamed_from))
        else:
            ended.append(_ended(track.origin, at_head.commit))
    return [*ended, *moved]


def _answer(request: RevealRequested, at_head: _Head, revealed: set[str]) -> Answer | None:
    """The request judged at head; None when it is granted for a path already revealed, since _follow_events put head's blob in the feed."""
    verdict = at_head.judge(request.path, revealed)
    if not isinstance(verdict, Granted):
        return RefusalAnswer(path=request.path, reason=verdict)
    if request.path in revealed:
        return None
    revealed.add(request.path)
    return _revealed(request.path, at_head.commit, verdict, renamed_from=None)


def _counted(
    root: Path,
    policy: Policy,
    files: Sequence[TreeFile],
    engagement: EngagementProjection,
    *,
    head: str,
    cached: AddedLineCounts | None,
) -> tuple[CountedSet, AddedLineCounts | None]:
    """The counted set at head, and with a delivered commit the added-line counts it was charged by."""
    delivered = engagement.delivered_commit
    if delivered is None:
        return counted_set(policy, files), None
    since = lines_since(root, policy, files, delivered=delivered, head=head, cached=cached)
    return counted_set(policy, files, since=since.by_path), since.counts


def _unanswered(
    decided: Iterable[Answer], answered: Counter[Answer], *, stamp: Stamp, now: datetime
) -> tuple[AnswerEvent, ...]:
    """The decided answers stamped in order, but each one the tail already holds consumed there, spending no seq."""
    unspent = Counter(answered)
    events: list[AnswerEvent] = []
    for answer in decided:
        if unspent[answer] > 0:
            unspent[answer] -= 1
        else:
            events.append(answer.placed(stamp(now)))
    return tuple(events)


def reveal_events(
    root: Path,
    policy: Policy,
    *,
    head: str,
    since: str | None,
    files: Sequence[TreeFile],
    engagement: EngagementProjection,
    state: FeedState,
    requests: Sequence[RevealRequested],
    inbox_offset: int,
    answered: Counter[Answer],
    added_line_counts: AddedLineCounts | None,
    stamp: Stamp,
    now: datetime,
) -> RevealOutput:
    """Every revealed file followed from `since` to head, then each request answered in inbox order; the budget is what is spent after them.

    A request for a path already revealed appends nothing, because _follow_events has already put head's blob in the
    feed for every revealed path. An answer already in `answered`, the uncommitted tail, is consumed there instead of
    emitted again. A blob pair `added_line_counts` already counted is not diffed again.
    """
    counted, counts = _counted(
        root, policy, files, engagement, head=head, cached=added_line_counts
    )
    revealed = {file.path for file in state.revealed}
    tracks = _tracks(root, state.revealed, since=since, head=head)
    with blob_reader(root) as read:
        at_head = _Head(
            commit=head,
            policy=policy,
            entries={entry.path: entry for entry in tree_entries(root, head)},
            counted=counted,
            read=read,
        )
        decided = _follow_events(tracks, at_head, revealed)
        for request in requests:
            answer = _answer(request, at_head, revealed)
            if answer is not None:
                decided.append(answer)
    return RevealOutput(
        events=_unanswered(decided, answered, stamp=stamp, now=now),
        budget=counted.budget(revealed),
        inbox_offset=inbox_offset,
        reveals_head=head,
        added_line_counts=counts,
    )
