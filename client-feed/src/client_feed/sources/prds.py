"""A PRD's title and excerpt, read from its regular-file blob at a main commit, never from the working tree."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from client_feed.redact import redact_paths
from client_feed.sources.git import EntryKind, blob_reader, tree_entries

_OPENING_FENCE: Final = re.compile(r'`{3,}|~{3,}')
_HEADING: Final = re.compile(r' {0,3}(?P<marks>#{1,6})(?:[ \t]+(?P<text>.*))?')


@dataclass(frozen=True)
class PrdText:
    prd_path: str
    title: str
    excerpt: str


@dataclass(frozen=True)
class PrdsRead:
    """The PRDs found as regular files at the commit, and every other requested path; both sorted."""

    found: tuple[PrdText, ...]
    absent: tuple[str, ...]


def _opening_fence(line: str) -> str | None:
    """The run of three or more backticks or tildes that opens a fenced block on this line, if any."""
    match = _OPENING_FENCE.match(line.strip())
    return None if match is None else match[0]


def _closes(line: str, fence: str) -> bool:
    """CommonMark's closing fence: the opening's character, at least as many times, and nothing else."""
    stripped = line.strip()
    return len(stripped) >= len(fence) and stripped == fence[0] * len(stripped)


def _unfenced(markdown: str) -> Iterator[str]:
    """Every line outside a fenced block; a whole block reads as one blank line, so it ends a paragraph."""
    fence: str | None = None
    for line in markdown.splitlines():
        if fence is None:
            fence = _opening_fence(line)
            yield line if fence is None else ''
        elif _closes(line, fence):
            fence = None


def _heading(line: str) -> tuple[int, str]:
    """A heading's level and stripped text; level 0 for a line that is not a heading."""
    match = _HEADING.fullmatch(line)
    if match is None:
        return 0, ''
    return len(match['marks']), (match['text'] or '').strip()


def _title(lines: list[str]) -> str | None:
    for line in lines:
        level, text = _heading(line)
        if level == 1 and text:
            return text
    return None


def _first_paragraph(lines: Iterable[str]) -> str:
    paragraph: list[str] = []
    for line in lines:
        if _heading(line)[0] or not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line.strip())
    return ' '.join(paragraph)


def _excerpt(lines: list[str]) -> str:
    """The first paragraph after the first level-2 heading, skipping any heading before it."""
    start = next((index for index, line in enumerate(lines) if _heading(line)[0] == 2), None)
    return '' if start is None else _first_paragraph(lines[start + 1 :])


def summarize_prd(prd_path: str, markdown: str) -> PrdText:
    """The first H1 as the title (else the path), and the first paragraph after the first H2 as the excerpt."""
    lines = list(_unfenced(markdown))
    title = _title(lines)
    return PrdText(
        prd_path=prd_path,
        title=prd_path if title is None else redact_paths(title),
        excerpt=redact_paths(_excerpt(lines)),
    )


def read_prds(root: Path, *, commit: str, prd_paths: Iterable[str]) -> PrdsRead:
    """Each requested PRD summarized from its blob at `commit`; a symlink, directory or missing path is absent."""
    requested = sorted(set(prd_paths))
    if not requested:
        return PrdsRead(found=(), absent=())
    files = {
        entry.path: entry.object_sha
        for entry in tree_entries(root, commit)
        if entry.kind is EntryKind.FILE
    }
    present = [path for path in requested if path in files]
    found: tuple[PrdText, ...] = ()
    if present:
        with blob_reader(root) as read:
            found = tuple(
                summarize_prd(path, read(files[path]).decode(errors='replace')) for path in present
            )
    return PrdsRead(found=found, absent=tuple(path for path in requested if path not in files))
