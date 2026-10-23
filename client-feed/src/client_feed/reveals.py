"""Each reveal request answered, and each revealed file followed to main's head, as feed events."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from client_feed.budget import AddedLineCounts, CountedSet, Granted, counted_set, judge, lines_since
from client_feed.contract import (
    FEED_VERSION,
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

Answer = FileRevealed | RevealRefused
AnswerIdentity = tuple[str, str]

_UNPLACED_SEQ = 1


def answer_identity(event: FeedEvent) -> AnswerIdentity | None:
    """What an answer says wherever it lands: its type and its JSON without seq and at; None for any other event."""
    if not isinstance(event, FileRevealed | RevealRefused):
        return None
    return (event.type, event.model_dump_json(exclude={'seq', 'at'}))


def answered_in(lines: Iterable[FeedLine]) -> Counter[AnswerIdentity]:
    """The answers a stopped writer appended but never recorded in its cursor, counted by identity."""
    keys = (answer_identity(line) for line in lines if not isinstance(line, EventHeader))
    return Counter(key for key in keys if key is not None)


@dataclass(frozen=True)
class RevealOutput:
    """One pass's reveal events, the budget after them, the positions they answered through, and the added-line counts to carry."""

    events: tuple[Answer, ...]
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


@dataclass(frozen=True)
class _Unstamped:
    """An answer decided but not yet placed in the feed."""

    model: type[Answer]
    fields: Mapping[str, Any]

    def placed(self, envelope: Envelope) -> Answer:
        return self.model(**envelope, **self.fields)


def _revealed(path: str, head: str, granted: Granted, *, renamed_from: str | None) -> _Unstamped:
    return _Unstamped(
        FileRevealed,
        {
            'path': path,
            'commit': head,
            'lines': granted.lines,
            'content': granted.text,
            'deleted': False,
            'renamed_from': renamed_from,
        },
    )


def _ended(origin: str, head: str) -> _Unstamped:
    return _Unstamped(
        FileRevealed,
        {
            'path': origin,
            'commit': head,
            'lines': 0,
            'content': None,
            'deleted': True,
            'renamed_from': None,
        },
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
) -> list[_Unstamped]:
    """A moved or changed file re-emitted at head while it stays revealable; otherwise its reveal ends at its origin, ends first."""
    touched = sorted((track for track in tracks if not track.untouched), key=lambda t: t.origin)
    verdicts = [
        (track, None if track.ended else at_head.judge(track.path, revealed | {track.path}))
        for track in touched
    ]
    revealed.difference_update(track.origin for track in touched)
    ended: list[_Unstamped] = []
    moved: list[_Unstamped] = []
    for track, verdict in verdicts:
        if isinstance(verdict, Granted):
            revealed.add(track.path)
            renamed_from = None if track.path == track.origin else track.origin
            moved.append(_revealed(track.path, at_head.commit, verdict, renamed_from=renamed_from))
        else:
            ended.append(_ended(track.origin, at_head.commit))
    return [*ended, *moved]


def _answer(request: RevealRequested, at_head: _Head, revealed: set[str]) -> _Unstamped:
    verdict = at_head.judge(request.path, revealed)
    if isinstance(verdict, Granted):
        revealed.add(request.path)
        return _revealed(request.path, at_head.commit, verdict, renamed_from=None)
    return _Unstamped(RevealRefused, {'path': request.path, 'reason': verdict})


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
    decided: Iterable[_Unstamped],
    answered: Counter[AnswerIdentity],
    *,
    project_id: str,
    stamp: Stamp,
    now: datetime,
) -> tuple[Answer, ...]:
    """The decided answers stamped in order, but each one the tail already holds consumed there, spending no seq."""
    unplaced = Envelope(seq=_UNPLACED_SEQ, at=now, project_id=project_id, feed_version=FEED_VERSION)
    unspent = Counter(answered)
    events: list[Answer] = []
    for answer in decided:
        key = answer_identity(answer.placed(unplaced))
        if key is not None and unspent[key] > 0:
            unspent[key] -= 1
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
    answered: Counter[AnswerIdentity],
    added_line_counts: AddedLineCounts | None,
    stamp: Stamp,
    now: datetime,
) -> RevealOutput:
    """Every revealed file followed from `since` to head, then each request answered in inbox order; the budget is what is spent after them.

    An answer already in `answered`, the uncommitted tail, is consumed there instead of emitted again. A blob
    pair `added_line_counts` already counted is not diffed again.
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
        decided += [_answer(request, at_head, revealed) for request in requests]
    return RevealOutput(
        events=_unanswered(decided, answered, project_id=state.project_id, stamp=stamp, now=now),
        budget=counted.budget(revealed),
        inbox_offset=inbox_offset,
        reveals_head=head,
        added_line_counts=counts,
    )
