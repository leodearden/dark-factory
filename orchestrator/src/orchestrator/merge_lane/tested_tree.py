"""The tree an in-flight entry verifies: its base, its root and its members, in order.

The value behind ``InflightEntry.tree`` (plans/merge-pool-batches-prd.md §6.1).
T1 (each request at most once; the first own member is the owner) is held at
construction; T2 (the last own member's merge commit is the tree the verify
ran on) is checked where the tree is recorded, through ``require_verifies``.
A base resolves by commit through the three-row table of
``resolve_tested_tree``: main first, then the live trees, else no root.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    'TestedTree',
    'TestedTreeViolation',
    'TreeMember',
    'resolve_tested_tree',
]


class TestedTreeViolation(ValueError):
    """A tested tree breaks an invariant; the message names it and the values."""

    def __init__(self, invariant: str, what: str, values: str) -> None:
        super().__init__(f'{invariant}: {what} ({values})')


@dataclass(frozen=True)
class TreeMember:
    """One request merged into a tested tree.

    ``merge_commit`` is the tree with this member and everything before it;
    ``tip`` is the branch tip merged into that tree, None for an owner that
    is a rebased head; ``own`` is True for a member of the verifying entry
    and False for one of an entry it was built on.
    """

    request_id: str
    task_id: str
    merge_commit: str
    tip: str | None
    own: bool

    def __post_init__(self) -> None:
        if not self.request_id or not self.merge_commit:
            raise TestedTreeViolation(
                'member',
                'a member needs a request id and a merge commit',
                f'request_id={self.request_id!r}, merge_commit={self.merge_commit!r}',
            )

    def as_built_on(self) -> TreeMember:
        return dataclasses.replace(self, own=False)

    def to_wire(self) -> dict[str, Any]:
        return {
            'request_id': self.request_id,
            'task_id': self.task_id,
            'merge_commit': self.merge_commit,
            'tip': self.tip,
            'own': self.own,
        }


@dataclass(frozen=True)
class TestedTree:
    """The commit an owner was built on, the main commit under it, and its members.

    ``members`` lists the built-on members first, then the own members in
    land order. ``root_commit`` is None when the base cannot be resolved.
    """

    base_commit: str
    root_commit: str | None
    members: tuple[TreeMember, ...]

    def __post_init__(self) -> None:
        _check_t1(self.members)

    @property
    def owner(self) -> TreeMember:
        return next(m for m in self.members if m.own)

    @property
    def verified_commit(self) -> str:
        return self.members[-1].merge_commit

    def members_through(self, merge_commit: str) -> tuple[TreeMember, ...] | None:
        """The members up to and including the one with *merge_commit*, as built on."""
        commits = [m.merge_commit for m in self.members]
        if merge_commit not in commits:
            return None
        through = self.members[: commits.index(merge_commit) + 1]
        return tuple(m.as_built_on() for m in through)

    def require_verifies(self, owner_request_id: str, verified_commit: str) -> None:
        """Refuse a tree whose owner or verified commit is not the entry's (T1, T2)."""
        if self.owner.request_id != owner_request_id:
            raise TestedTreeViolation(
                'T1',
                "the tree's owner is not the entry's",
                f'tree owner={self.owner.request_id!r}, entry={owner_request_id!r}',
            )
        if self.verified_commit != verified_commit:
            raise TestedTreeViolation(
                'T2',
                "the last own member's merge commit is not the tree the entry verifies",
                f'last own={self.verified_commit!r}, verifies={verified_commit!r}',
            )

    def to_wire(self) -> dict[str, Any]:
        return {
            'base_commit': self.base_commit,
            'root_commit': self.root_commit,
            'members': [m.to_wire() for m in self.members],
        }


def _check_t1(members: Sequence[TreeMember]) -> None:
    seen: set[str] = set()
    for position, member in enumerate(members):
        if member.request_id in seen:
            raise TestedTreeViolation(
                'T1',
                'a request is listed twice',
                f'request_id={member.request_id!r} at position {position}',
            )
        seen.add(member.request_id)
    own_positions = [i for i, m in enumerate(members) if m.own]
    if not own_positions:
        raise TestedTreeViolation(
            'T1',
            'the tree has no own member, so no owner',
            f'request_ids={[m.request_id for m in members]!r}',
        )
    for position in range(own_positions[0], len(members)):
        if not members[position].own:
            raise TestedTreeViolation(
                'T1',
                'a built-on member follows an own member',
                f'request_id={members[position].request_id!r} at position {position}',
            )


def resolve_tested_tree(
    *,
    base_commit: str,
    base_on_main: bool,
    live_trees: Iterable[TestedTree],
    own_members: Sequence[TreeMember],
) -> TestedTree:
    """Resolve *base_commit* by commit (PRD §6.1's base table) and build the tree."""
    own = tuple(own_members)
    if base_on_main:
        return TestedTree(base_commit=base_commit, root_commit=base_commit, members=own)
    for live in live_trees:
        built_on = live.members_through(base_commit)
        if built_on is not None:
            return TestedTree(
                base_commit=base_commit,
                root_commit=live.root_commit,
                members=built_on + own,
            )
    return TestedTree(base_commit=base_commit, root_commit=None, members=own)
