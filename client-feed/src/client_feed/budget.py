"""What a reveal costs and whether a request is granted (plans/client-feed-prd.md §4.12, §4.13)."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from client_feed.contract import FileKind, RevealBudget, TreeFile
from client_feed.policy import Policy, matches_any
from client_feed.sources.git import EntryKind, PathChange, added_lines, tree_changes


@dataclass(frozen=True)
class CountedSet:
    """The paths a reveal is charged for, each with its chargeable lines (all above 0), and the policy's proportion and floor."""

    lines: Mapping[str, int]
    proportion: float
    floor_lines: int

    @classmethod
    def of(cls, policy: Policy, lines: Mapping[str, int]) -> CountedSet:
        return cls(
            lines=MappingProxyType(dict(lines)),
            proportion=policy.reveal_proportion,
            floor_lines=policy.reveal_floor_lines,
        )

    @property
    def counted_lines(self) -> int:
        return sum(self.lines.values())

    @property
    def budget_lines(self) -> int:
        return math.floor(self.proportion * self.counted_lines)

    def cost(self, path: str) -> int:
        """A counted path's lines, at least the floor; nothing outside the counted set."""
        lines = self.lines.get(path)
        return 0 if lines is None else max(lines, self.floor_lines)

    def spent(self, revealed: Collection[str]) -> int:
        return sum(self.cost(path) for path in set(revealed))

    def grantable(self, path: str, revealed: Collection[str]) -> bool:
        """A revealed path stays granted; any other must fit what is left of the budget."""
        return path in revealed or self.spent(revealed) + self.cost(path) <= self.budget_lines

    def budget(self, revealed: Collection[str]) -> RevealBudget:
        return RevealBudget(
            proportion=self.proportion,
            counted_lines=self.counted_lines,
            budget_lines=self.budget_lines,
            spent_lines=self.spent(revealed),
            floor_lines=self.floor_lines,
        )


def counted_set(
    policy: Policy, files: Sequence[TreeFile], *, since: Mapping[str, int] | None = None
) -> CountedSet:
    """Production files outside the free and never-reveal globs, charged their lines, or with `since` their lines since delivery."""
    uncounted = (*policy.free_globs, *policy.never_reveal_globs)
    chargeable = {
        file.path: file.lines if since is None else since.get(file.path, 0)
        for file in files
        if file.kind is FileKind.PRODUCTION and not matches_any(uncounted, file.path)
    }
    return CountedSet.of(policy, {path: lines for path, lines in chargeable.items() if lines > 0})


def _lines_changed(root: Path, policy: Policy, change: PathChange, file: TreeFile) -> int:
    if change.old_kind is not EntryKind.FILE or change.old_blob is None or change.new_blob is None:
        return file.lines
    if change.old_blob == change.new_blob:
        return 0
    return policy.counted_lines(added_lines(root, change.old_blob, change.new_blob))


def lines_since(
    root: Path, policy: Policy, files: Sequence[TreeFile], *, delivered: str, head: str
) -> dict[str, int]:
    """Each file of `files` that changed since `delivered`, with the counted lines it added or changed; a pure rename adds none."""
    by_path = {file.path: file for file in files}
    since: dict[str, int] = {}
    for change in tree_changes(root, delivered, head):
        file = None if change.new_path is None else by_path.get(change.new_path)
        if file is not None:
            since[file.path] = _lines_changed(root, policy, change, file)
    return since
