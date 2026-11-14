"""The client-visible tree at a main commit under a policy: which files, their kinds, counted lines and last change."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from client_feed.contract import FileKind, NonNegativeInt, TreeFile, TreeSnapshot, TreeTotals
from client_feed.policy import Policy, PositiveInt
from client_feed.sources.git import (
    EntryKind,
    TreeEntry,
    blob_reader,
    generated_paths,
    is_ancestor,
    last_changes,
    tree_entries,
)


class LineCounts(BaseModel):
    """Counted lines per blob sha; valid only under the max_line_bytes they were counted with."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    max_line_bytes: PositiveInt
    by_blob: dict[str, NonNegativeInt]


@dataclass(frozen=True)
class VisibleTree:
    """The visible files of one commit, their totals, and the line counts to carry to the next snapshot."""

    commit: str
    files: tuple[TreeFile, ...]
    totals: TreeTotals
    line_counts: LineCounts


def _visible_files(root: Path, policy: Policy, commit: str) -> tuple[TreeEntry, ...]:
    return tuple(
        entry
        for entry in tree_entries(root, commit)
        if entry.kind is EntryKind.FILE and policy.is_visible(entry.path)
    )


def _line_counts(
    root: Path, policy: Policy, entries: Sequence[TreeEntry], cached: LineCounts | None
) -> LineCounts:
    reusable = cached is not None and cached.max_line_bytes == policy.max_line_bytes
    known = cached.by_blob if reusable and cached is not None else {}
    blobs = {entry.object_sha for entry in entries}
    by_blob = {sha: known[sha] for sha in blobs if sha in known}
    missing = sorted(blobs - by_blob.keys())
    if missing:
        with blob_reader(root) as read:
            by_blob.update((sha, policy.counted_lines(read(sha))) for sha in missing)
    return LineCounts(max_line_bytes=policy.max_line_bytes, by_blob=by_blob)


def _last_changed(
    root: Path, commit: str, previous: TreeSnapshot | None, paths: Collection[str]
) -> Mapping[str, datetime | None]:
    """Walk main only since the previous snapshot when it is an ancestor; unchanged paths keep its dates."""
    if previous is None or not is_ancestor(root, previous.commit, commit):
        full = last_changes(root, commit, since=None)
        return {path: full.get(path) for path in paths}
    walked = last_changes(root, commit, since=previous.commit)
    carried = {file.path: file.last_changed_at for file in previous.files}
    dates = {path: walked.get(path, carried.get(path)) for path in paths}
    if all(date is not None for date in dates.values()):
        return dates
    full = last_changes(root, commit, since=None)
    return {path: full.get(path) if date is None else date for path, date in dates.items()}


def _blob_size(entry: TreeEntry) -> int:
    if entry.size is None:
        raise ValueError(f'tree entry {entry.path!r} has no blob size')
    return entry.size


def _lines_of(files: Sequence[TreeFile], kind: FileKind) -> int:
    return sum(file.lines for file in files if file.kind is kind)


def _totals(files: Sequence[TreeFile]) -> TreeTotals:
    return TreeTotals(
        files=len(files),
        production_lines=_lines_of(files, FileKind.PRODUCTION),
        test_lines=_lines_of(files, FileKind.TEST),
        docs_lines=_lines_of(files, FileKind.DOCS),
    )


def visible_tree(
    root: Path,
    policy: Policy,
    commit: str,
    *,
    previous: TreeSnapshot | None,
    line_counts: LineCounts | None,
) -> VisibleTree:
    """The files of `commit` the policy shows, read from git objects only, sorted by path."""
    entries = _visible_files(root, policy, commit)
    paths = [entry.path for entry in entries]
    generated = generated_paths(root, commit, paths)
    counts = _line_counts(root, policy, entries, line_counts)
    dates = _last_changed(root, commit, previous, paths)
    files = tuple(
        TreeFile(
            path=entry.path,
            kind=policy.kind_of(entry.path, linguist_generated=entry.path in generated),
            lines=counts.by_blob[entry.object_sha],
            bytes=_blob_size(entry),
            last_changed_at=dates[entry.path],
        )
        for entry in sorted(entries, key=lambda entry: entry.path)
    )
    return VisibleTree(commit=commit, files=files, totals=_totals(files), line_counts=counts)
