"""What a reveal costs and whether a request is granted (plans/client-feed-prd.md §4.12, §4.13)."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict

from client_feed.contract import FileKind, NonNegativeInt, RefusalReason, RevealBudget, TreeFile
from client_feed.policy import Policy, PositiveInt, matches_any
from client_feed.sources.git import EntryKind, PathChange, TreeEntry, added_lines, tree_changes

_LINKED_KINDS: Final = frozenset({EntryKind.SYMLINK, EntryKind.SUBMODULE})


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

    @functools.cached_property
    def counted_lines(self) -> int:
        return sum(self.lines.values())

    @functools.cached_property
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


def _countable(policy: Policy, files: Sequence[TreeFile]) -> Iterator[TreeFile]:
    """The production files outside the free and never-reveal globs."""
    uncounted = (*policy.free_globs, *policy.never_reveal_globs)
    return (
        file
        for file in files
        if file.kind is FileKind.PRODUCTION and not matches_any(uncounted, file.path)
    )


def counted_set(
    policy: Policy, files: Sequence[TreeFile], *, since: Mapping[str, int] | None = None
) -> CountedSet:
    """Production files outside the free and never-reveal globs, charged their lines, or with `since` their lines since delivery."""
    chargeable = {
        file.path: file.lines if since is None else since.get(file.path, 0)
        for file in _countable(policy, files)
    }
    return CountedSet.of(policy, {path: lines for path, lines in chargeable.items() if lines > 0})


class AddedLineCounts(BaseModel):
    """Counted lines a new blob adds or changes against an old one, by old then new blob sha.

    Valid only under the max_line_bytes they were counted with.
    """

    model_config = ConfigDict(frozen=True, extra='forbid')

    max_line_bytes: PositiveInt
    by_blob_pair: dict[str, dict[str, NonNegativeInt]]

    def lines_of(self, old: str, new: str) -> int:
        return self.by_blob_pair[old][new]


@dataclass(frozen=True)
class LinesSince:
    """Each counted file changed since delivery with the lines it added or changed, and the counts to carry to the next pass."""

    by_path: Mapping[str, int]
    counts: AddedLineCounts


def _diffed_pair(change: PathChange) -> tuple[str, str] | None:
    """The old and new blob a changed file is charged by; None when the path held no file before, so every line is new."""
    if change.old_kind is not EntryKind.FILE or change.old_blob is None or change.new_blob is None:
        return None
    return change.old_blob, change.new_blob


def _added_line_counts(
    root: Path, policy: Policy, pairs: Collection[tuple[str, str]], cached: AddedLineCounts | None
) -> AddedLineCounts:
    """Each pair's counted added lines: from `cached` when it was counted under this max_line_bytes, else diffed now."""
    known = {} if cached is None or cached.max_line_bytes != policy.max_line_bytes else cached.by_blob_pair
    by_blob_pair: dict[str, dict[str, int]] = {}
    for old, new in sorted(pairs):
        lines = known.get(old, {}).get(new)
        if lines is None:
            lines = 0 if old == new else policy.counted_lines(added_lines(root, old, new))
        by_blob_pair.setdefault(old, {})[new] = lines
    return AddedLineCounts(max_line_bytes=policy.max_line_bytes, by_blob_pair=by_blob_pair)


def lines_since(
    root: Path,
    policy: Policy,
    files: Sequence[TreeFile],
    *,
    delivered: str,
    head: str,
    cached: AddedLineCounts | None,
) -> LinesSince:
    """Each countable file of `files` changed since `delivered`, with the counted lines it added or changed.

    A file added since is charged all its lines and a pure rename none; a blob pair `cached` already counted is not diffed again.
    """
    by_path = {file.path: file for file in _countable(policy, files)}
    changed: list[tuple[TreeFile, tuple[str, str] | None]] = []
    for change in tree_changes(root, delivered, head):
        file = None if change.new_path is None else by_path.get(change.new_path)
        if file is not None:
            changed.append((file, _diffed_pair(change)))
    counts = _added_line_counts(
        root, policy, {pair for _, pair in changed if pair is not None}, cached
    )
    return LinesSince(
        by_path={
            file.path: file.lines if pair is None else counts.lines_of(*pair)
            for file, pair in changed
        },
        counts=counts,
    )


@dataclass(frozen=True)
class Granted:
    """A granted file's text at head and its counted lines."""

    text: str
    lines: int


@dataclass(frozen=True)
class _Asked:
    """One request under judgement: head's entry at its raw path, and the entry's blob, read once and only when first needed."""

    path: str
    entry: TreeEntry | None
    policy: Policy
    counted: CountedSet
    revealed: Collection[str]
    read: Callable[[str], bytes]

    @property
    def kind(self) -> EntryKind | None:
        return None if self.entry is None else self.entry.kind

    @functools.cached_property
    def blob(self) -> bytes:
        if self.entry is None:
            raise ValueError(f'{self.path!r} is no entry of the tree and has no blob')
        return self.read(self.entry.object_sha)


def _not_in_tree(asked: _Asked) -> bool:
    return asked.entry is None


def _not_visible(asked: _Asked) -> bool:
    return not asked.policy.is_visible(asked.path)


def _symlink(asked: _Asked) -> bool:
    return asked.kind in _LINKED_KINDS


def _never_revealed(asked: _Asked) -> bool:
    return matches_any(asked.policy.never_reveal_globs, asked.path)


def _not_text(asked: _Asked) -> bool:
    try:
        asked.blob.decode('utf-8')
    except UnicodeDecodeError:
        return True
    return False


def _over_budget(asked: _Asked) -> bool:
    return not asked.counted.grantable(asked.path, asked.revealed)


_REFUSALS: Final[tuple[tuple[RefusalReason, Callable[[_Asked], bool]], ...]] = (
    (RefusalReason.NOT_IN_TREE, _not_in_tree),
    (RefusalReason.NOT_VISIBLE, _not_visible),
    (RefusalReason.SYMLINK, _symlink),
    (RefusalReason.NOT_REVEALABLE, _never_revealed),
    (RefusalReason.NOT_REVEALABLE, _not_text),
    (RefusalReason.OVER_BUDGET, _over_budget),
)


def judge(
    path: str,
    *,
    entries: Mapping[str, TreeEntry],
    policy: Policy,
    counted: CountedSet,
    revealed: Collection[str],
    read: Callable[[str], bytes],
) -> Granted | RefusalReason:
    """The first refusal in precedence order that holds for the raw `path` at head, else its grant; content only through `read`."""
    asked = _Asked(
        path=path, entry=entries.get(path), policy=policy, counted=counted, revealed=revealed, read=read
    )
    for reason, refuses in _REFUSALS:
        if refuses(asked):
            return reason
    return Granted(text=asked.blob.decode('utf-8'), lines=policy.counted_lines(asked.blob))
