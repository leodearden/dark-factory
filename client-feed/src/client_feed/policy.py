"""The publisher's factory-side policy (plans/client-feed-prd.md §7.6, §14.3): what a client sees, how its lines count, and the defect filer's block."""

from __future__ import annotations

import functools
import os
import re
import stat
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Self

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    StrictBool,
    ValidationError,
    model_validator,
)

from client_feed.contract import FileKind, NonNegativeInt, PositiveInt, ProjectId, RepoPath
from client_feed.sources.git import GitError, repository_toplevel

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


def _require_outside(what: str, path: Path, *, of: str, container: Path) -> None:
    if path.is_relative_to(container):
        raise ValueError(f'{what} {path} must lie outside {of} {container}')


class TaskPriority(StrEnum):
    """The factory's task priority tiers, re-declared from orchestrator/src/orchestrator/config.py::PRIORITY_TIERS because client_feed may not import orchestrator."""

    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    POLISH = 'polish'


class FilerPolicy(BaseModel):
    """The client defect filer's block (§14.3)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    fused_memory_url: HttpUrl = HttpUrl('http://127.0.0.1:8002')
    state_dir: AbsolutePath
    auto_dispatch: StrictBool = False
    interval_seconds: PositiveInt = 10
    fix_priority: TaskPriority = TaskPriority.HIGH


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
    filer_grace_seconds: PositiveInt = 600
    filer: FilerPolicy | None = None

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
            _require_outside(name, path, of='project_root', container=self.project_root)
        return self

    @model_validator(mode='after')
    def _filer_state_dir_stands_apart(self) -> Self:
        if self.filer is None:
            return self
        for name, path in (
            ('out_dir', self.out_dir),
            ('in_dir', self.in_dir),
            ('project_root', self.project_root),
        ):
            _require_outside('filer.state_dir', self.filer.state_dir, of=name, container=path)
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


class PolicyProblem(StrEnum):
    UNREADABLE = 'unreadable'
    INVALID = 'invalid'
    NOT_A_REPOSITORY_ROOT = 'not_a_repository_root'
    OUT_DIR_UNUSABLE = 'out_dir_unusable'
    OUT_DIR_NOT_OWNED = 'out_dir_not_owned'


class PolicyError(Exception):
    """A policy file the publisher refuses to start on."""

    def __init__(self, *, config: Path, problem: PolicyProblem, detail: str) -> None:
        self.config = config
        self.problem = problem
        self.detail = detail
        super().__init__(f'policy {config} refused ({problem}): {detail}')


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    first = errors[0]
    location = '.'.join(str(part) for part in first['loc']) or '<policy>'
    more = f' (and {len(errors) - 1} more)' if len(errors) > 1 else ''
    return f'{location}: {first["msg"]}{more}'


def _read_mapping(config: Path) -> Mapping[str, Any]:
    try:
        text = config.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        raise PolicyError(config=config, problem=PolicyProblem.UNREADABLE, detail=str(exc)) from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError(
            config=config, problem=PolicyProblem.INVALID, detail=f'not YAML: {exc}'
        ) from exc
    if not isinstance(data, dict):
        raise PolicyError(
            config=config,
            problem=PolicyProblem.INVALID,
            detail=f'expected a mapping, got {type(data).__name__}',
        )
    return data


def _validated(config: Path, data: Mapping[str, Any]) -> Policy:
    try:
        return Policy.model_validate(data)
    except ValidationError as exc:
        raise PolicyError(
            config=config, problem=PolicyProblem.INVALID, detail=_first_error(exc)
        ) from exc


def _require_repository_root(config: Path, policy: Policy) -> None:
    """The root must BE a work tree's top: from a subdirectory, every git call would act on the parent repository."""
    root = policy.project_root
    try:
        toplevel = repository_toplevel(root)
    except GitError as exc:
        raise PolicyError(
            config=config,
            problem=PolicyProblem.NOT_A_REPOSITORY_ROOT,
            detail=f'project_root {root} is not a usable git work tree: {exc.reason}',
        ) from exc
    if toplevel != root.resolve():
        raise PolicyError(
            config=config,
            problem=PolicyProblem.NOT_A_REPOSITORY_ROOT,
            detail=f'project_root {root} is inside the work tree {toplevel}, not at its top',
        )


class OutDirRefused(Exception):
    """An out_dir the publisher must not write to: missing, not a real directory, or another user's."""

    def __init__(self, *, problem: PolicyProblem, out_dir: Path, detail: str) -> None:
        self.problem = problem
        self.out_dir = out_dir
        self.detail = detail
        super().__init__(f'out_dir refused ({problem}): {detail}')


def require_usable_out_dir(out_dir: Path) -> None:
    """The one check that out_dir is a directory, not a symlink, owned by the running user; at policy load and at writer open."""
    try:
        status = os.lstat(out_dir)
    except OSError as exc:
        raise OutDirRefused(
            problem=PolicyProblem.OUT_DIR_UNUSABLE,
            out_dir=out_dir,
            detail=f'out_dir {out_dir} cannot be inspected: {exc.strerror}',
        ) from exc
    if not stat.S_ISDIR(status.st_mode):
        raise OutDirRefused(
            problem=PolicyProblem.OUT_DIR_UNUSABLE,
            out_dir=out_dir,
            detail=f'out_dir {out_dir} is not a directory (a symlink is refused)',
        )
    if status.st_uid != os.geteuid():
        raise OutDirRefused(
            problem=PolicyProblem.OUT_DIR_NOT_OWNED,
            out_dir=out_dir,
            detail=f'out_dir {out_dir} is owned by uid {status.st_uid}, not the running uid {os.geteuid()}',
        )


def _require_owned_out_dir(config: Path, policy: Policy) -> None:
    try:
        require_usable_out_dir(policy.out_dir)
    except OutDirRefused as exc:
        raise PolicyError(config=config, problem=exc.problem, detail=exc.detail) from exc


def load_policy(config: Path) -> Policy:
    """The policy in `config`, or a PolicyError naming why the publisher must not start on it."""
    policy = _validated(config, _read_mapping(config))
    _require_repository_root(config, policy)
    _require_owned_out_dir(config, policy)
    return policy
