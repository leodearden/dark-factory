"""Each reveal request answered, and each revealed file followed to main's head, as feed events."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from client_feed.budget import CountedSet, Granted, counted_set, judge, lines_since
from client_feed.contract import (
    EngagementProjection,
    FeedState,
    FileRevealed,
    RevealBudget,
    RevealRefused,
    RevealRequested,
    TreeFile,
)
from client_feed.envelope import Stamp
from client_feed.policy import Policy
from client_feed.sources.git import TreeEntry, blob_reader, tree_entries

Answer = FileRevealed | RevealRefused


@dataclass(frozen=True)
class RevealOutput:
    """One pass's reveal events, the budget after them, and the positions they answered through."""

    events: tuple[Answer, ...]
    budget: RevealBudget
    inbox_offset: int
    reveals_head: str


@dataclass(frozen=True)
class _Head:
    """What every judgement at head shares: its tree entries, the counted set, and one blob reader."""

    commit: str
    policy: Policy
    entries: Mapping[str, TreeEntry]
    counted: CountedSet
    read: Callable[[str], bytes]


def _counted(
    root: Path, policy: Policy, files: Sequence[TreeFile], engagement: EngagementProjection, head: str
) -> CountedSet:
    delivered = engagement.delivered_commit
    if delivered is None:
        return counted_set(policy, files)
    return counted_set(
        policy, files, since=lines_since(root, policy, files, delivered=delivered, head=head)
    )


def _answer(
    request: RevealRequested, head: _Head, revealed: set[str], stamp: Stamp, now: datetime
) -> Answer:
    verdict = judge(
        request.path,
        entries=head.entries,
        policy=head.policy,
        counted=head.counted,
        revealed=revealed,
        read=head.read,
    )
    if isinstance(verdict, Granted):
        revealed.add(request.path)
        return FileRevealed(
            **stamp(now),
            path=request.path,
            commit=head.commit,
            lines=verdict.lines,
            content=verdict.text,
            deleted=False,
            renamed_from=None,
        )
    return RevealRefused(**stamp(now), path=request.path, reason=verdict)


def reveal_events(
    root: Path,
    policy: Policy,
    *,
    head: str,
    files: Sequence[TreeFile],
    engagement: EngagementProjection,
    state: FeedState,
    requests: Sequence[RevealRequested],
    inbox_offset: int,
    stamp: Stamp,
    now: datetime,
) -> RevealOutput:
    """Each request answered in inbox order against the revealed set as it grows; the budget is what is spent after them."""
    counted = _counted(root, policy, files, engagement, head)
    revealed = {file.path for file in state.revealed}
    events: list[Answer] = []
    with blob_reader(root) as read:
        at_head = _Head(
            commit=head,
            policy=policy,
            entries={entry.path: entry for entry in tree_entries(root, head)},
            counted=counted,
            read=read,
        )
        events.extend(_answer(request, at_head, revealed, stamp, now) for request in requests)
    return RevealOutput(
        events=tuple(events),
        budget=counted.budget(revealed),
        inbox_offset=inbox_offset,
        reveals_head=head,
    )
