"""The publisher's factory-side policy (plans/client-feed-prd.md §7.6): what a client sees and how its lines count."""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Final, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictInt, model_validator

from client_feed.contract import FileKind, NonNegativeInt, ProjectId, RepoPath

DEFAULT_TEST_GLOBS: Final = ('**/tests/**', '**/test_*.py', '**/*_test.py', '**/conftest.py')
DEFAULT_DOCS_GLOBS: Final = ('**/*.md', 'docs/**')

_ANY_SEGMENTS: Final = '**'


def _class_end(segment: str, start: int) -> int | None:
    """The index of the ']' closing the class opened at `start`; a ']' first in the class is a member."""
    index = start + 1
    if index < len(segment) and segment[index] == '!':
        index += 1
    if index < len(segment) and segment[index] == ']':
        index += 1
    close = segment.find(']', index)
    return None if close == -1 else close


def _class_regex(body: str) -> str:
    negated = body.startswith('!')
    members = body[1:] if negated else body
    parts: list[str] = []
    index = 0
    while index < len(members):
        if index + 2 < len(members) and members[index + 1] == '-':
            parts.append(f'{re.escape(members[index])}-{re.escape(members[index + 2])}')
            index += 3
        else:
            parts.append(re.escape(members[index]))
            index += 1
    joined = ''.join(parts)
    return f'[^/{joined}]' if negated else f'(?!/)[{joined}]'


def _segment_regex(segment: str) -> str:
    parts: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        close = _class_end(segment, index) if char == '[' else None
        if char == '*':
            parts.append('[^/]*')
        elif char == '?':
            parts.append('[^/]')
        elif close is not None:
            parts.append(_class_regex(segment[index + 1 : close]))
            index = close
        else:
            parts.append(re.escape(char))
        index += 1
    return ''.join(parts)


@functools.cache
def _glob_regex(glob: str) -> re.Pattern[str]:
    """A gitignore-like glob as a regex over a repo-relative path: only a whole '**' segment crosses '/'."""
    segments = glob.split('/')
    last = len(segments) - 1
    parts: list[str] = []
    for index, segment in enumerate(segments):
        if segment == _ANY_SEGMENTS:
            parts.append('.+' if index == last else '(?:[^/]+/)*')
        else:
            parts.append(_segment_regex(segment) + ('' if index == last else '/'))
    return re.compile(''.join(parts), re.DOTALL)


def matches_any(globs: Sequence[str], path: str) -> bool:
    """Whether a repo-relative path matches any of the globs; the one home of the policy's glob semantics."""
    return any(_glob_regex(glob).fullmatch(path) for glob in globs)


def _require_compilable(glob: str) -> str:
    try:
        _glob_regex(glob)
    except re.error as exc:
        raise ValueError(f'glob {glob!r} does not compile: {exc}') from exc
    return glob


def _require_absolute(path: Path) -> Path:
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError(f'path must be absolute with no ".." part, got {path}')
    return path


Glob = Annotated[RepoPath, AfterValidator(_require_compilable)]
Globs = tuple[Glob, ...]
AbsolutePath = Annotated[Path, AfterValidator(_require_absolute)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]


class Policy(BaseModel):
    """One publisher's configuration: its root, its output and inbox, and what the client may see."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    project_id: ProjectId
    project_root: AbsolutePath
    out_dir: AbsolutePath
    in_dir: AbsolutePath
    interval_seconds: PositiveInt = 5
    visible_globs: Annotated[Globs, Field(min_length=1)]
    reveal_proportion: Annotated[float, Field(ge=0, le=1)] = 0.25
    reveal_floor_lines: NonNegativeInt = 50
    test_globs: Globs = DEFAULT_TEST_GLOBS
    docs_globs: Globs = DEFAULT_DOCS_GLOBS
    free_globs: Globs
    never_reveal_globs: Globs = ()
    generated_globs: Globs = ()
    max_line_bytes: PositiveInt = 400
    max_cases_per_run: PositiveInt = 5000

    @model_validator(mode='before')
    @classmethod
    def _free_globs_default_to_the_test_and_docs_globs(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or 'free_globs' in data:
            return data
        tests = data.get('test_globs', DEFAULT_TEST_GLOBS)
        docs = data.get('docs_globs', DEFAULT_DOCS_GLOBS)
        if not (isinstance(tests, list | tuple) and isinstance(docs, list | tuple)):
            return data
        return {**data, 'free_globs': (*tests, *docs)}

    @model_validator(mode='after')
    def _out_and_in_dirs_stand_apart(self) -> Self:
        if self.out_dir == self.in_dir:
            raise ValueError(f'out_dir and in_dir must differ, both are {self.out_dir}')
        for name, path in (('out_dir', self.out_dir), ('in_dir', self.in_dir)):
            if path.is_relative_to(self.project_root):
                raise ValueError(
                    f'{name} {path} must lie outside project_root {self.project_root}'
                )
        return self

    def is_visible(self, path: str) -> bool:
        return matches_any(self.visible_globs, path)

    def kind_of(self, path: str, *, linguist_generated: bool) -> FileKind:
        """A visible file's kind, ranked generated > test > docs > production."""
        if linguist_generated or matches_any(self.generated_globs, path):
            return FileKind.GENERATED
        if matches_any(self.test_globs, path):
            return FileKind.TEST
        if matches_any(self.docs_globs, path):
            return FileKind.DOCS
        return FileKind.PRODUCTION

    def counted_lines(self, content: bytes) -> int:
        """THE counting rule: non-blank lines, a line longer than max_line_bytes counting ceil(bytes / max)."""
        return sum(
            -(-len(line) // self.max_line_bytes) for line in content.split(b'\n') if line.strip()
        )
