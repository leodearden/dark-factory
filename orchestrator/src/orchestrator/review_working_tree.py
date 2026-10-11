"""Describe the uncommitted state of the checkout a review's Phase 1 measured."""

from __future__ import annotations

import enum
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shared.git_async import run_git

from orchestrator.verify import VerifyResult

logger = logging.getLogger(__name__)

SECTION_NAMED_PATH_CAP = 20
_SUMMARY_NAMED_PATH_CAP = 3

_GIT_QUERY_TIMEOUT_SECS = 30.0

_PATH_START = r'''(?<![^\s'"`(\[{<=,])'''
_PATH_END = r'(?![\w./-])'

_UNTRACKED_QUERY = (
    'git', '--no-optional-locks', 'ls-files', '--others', '--exclude-standard', '-z',
)
_UNCOMMITTED_QUERY = (
    'git', '--no-optional-locks', 'diff', '--name-only', '--no-renames', '-z', 'HEAD', '--',
)


class TreeChange(enum.Enum):
    UNTRACKED = 'untracked'
    UNCOMMITTED = 'uncommitted'


def _reject(problem: str, paths: Iterable[str]) -> None:
    offending = sorted(set(paths))
    if offending:
        raise ValueError(f'{problem}: {", ".join(offending)}')


@dataclass(frozen=True)
class WorkingTreeState:
    """Repo-relative POSIX paths, sorted, that make the checkout differ from HEAD."""

    untracked: tuple[str, ...]
    uncommitted: tuple[str, ...]

    def __post_init__(self) -> None:
        _reject(
            'working-tree paths must be repo-relative',
            (p for p in self.untracked + self.uncommitted if PurePosixPath(p).is_absolute()),
        )
        _reject(
            'a path cannot be both untracked and uncommitted',
            set(self.untracked) & set(self.uncommitted),
        )

    @property
    def is_clean(self) -> bool:
        return not self.untracked and not self.uncommitted

    def changes(self) -> Iterator[tuple[str, TreeChange]]:
        for path in self.untracked:
            yield path, TreeChange.UNTRACKED
        for path in self.uncommitted:
            yield path, TreeChange.UNCOMMITTED


async def read_working_tree_state(project_root: Path) -> WorkingTreeState | None:
    """The checkout's state, or None (logged) when git cannot describe it."""
    untracked = await _query_paths(project_root, _UNTRACKED_QUERY)
    if untracked is None:
        return None
    uncommitted = await _query_paths(project_root, _UNCOMMITTED_QUERY)
    if uncommitted is None:
        return None
    return WorkingTreeState(
        untracked=tuple(sorted(untracked - uncommitted)),
        uncommitted=tuple(sorted(uncommitted)),
    )


async def _query_paths(project_root: Path, query: tuple[str, ...]) -> set[str] | None:
    try:
        result = await run_git(query, cwd=project_root, timeout=_GIT_QUERY_TIMEOUT_SECS)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            'Working-tree read skipped for %s: `%s` raised %r',
            project_root, ' '.join(query), exc,
        )
        return None
    if not result.ok:
        logger.warning(
            'Working-tree read skipped for %s: `%s` rc=%s stderr=%s',
            project_root, ' '.join(query), result.returncode, result.stderr,
        )
        return None
    return {path for path in result.stdout.split('\0') if path}


@dataclass(frozen=True)
class NamedChange:
    path: str
    change: TreeChange


@dataclass(frozen=True)
class Phase1WorkingTree:
    """A checkout's state plus the changes a failing Phase 1 output names."""

    state: WorkingTreeState
    named_in_failure: tuple[NamedChange, ...]

    def __post_init__(self) -> None:
        kinds = dict(self.state.changes())
        _reject(
            'a named path must be a working-tree change of the same kind',
            (n.path for n in self.named_in_failure if kinds.get(n.path) is not n.change),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            'untracked_count': len(self.state.untracked),
            'uncommitted_count': len(self.state.uncommitted),
            'named_in_failure': [
                {'path': n.path, 'change': n.change.value} for n in self.named_in_failure
            ],
        }


def attribute_phase1(
    phase1: VerifyResult,
    state: WorkingTreeState,
    *,
    root: Path,
    run_dirs: Iterable[str],
) -> Phase1WorkingTree:
    """A change is named only by a printed form that starts at a delimiter.

    A relative form counts only when no other file in the checkout could be
    what it denotes.
    """
    if phase1.passed:
        return Phase1WorkingTree(state, ())
    output = '\n'.join((phase1.test_output, phase1.lint_output, phase1.type_output))
    prefixes = tuple(f'{d.rstrip("/")}/' for d in run_dirs if d)
    roots = tuple(dict.fromkeys((root, root.resolve())))
    named = tuple(
        NamedChange(path, change)
        for path, change in state.changes()
        if _named_relatively(output, path, root=root, prefixes=prefixes)
        or _named_absolutely(output, path, roots=roots)
    )
    return Phase1WorkingTree(state, named)


def _named_relatively(
    output: str, path: str, *, root: Path, prefixes: tuple[str, ...],
) -> bool:
    return any(
        _output_names(output, form, relative=True)
        and _denotes_only(root, path, form, prefixes)
        for form in _relative_forms(path, prefixes)
    )


def _named_absolutely(output: str, path: str, *, roots: tuple[Path, ...]) -> bool:
    return any(_output_names(output, str(r / path), relative=False) for r in roots)


def _relative_forms(path: str, prefixes: tuple[str, ...]) -> Iterator[str]:
    yield path
    for prefix in prefixes:
        if path.startswith(prefix):
            yield path[len(prefix):]


def _output_names(output: str, form: str, *, relative: bool) -> bool:
    if form not in output:
        return False
    dot_slash = r'(?:\./)?' if relative else ''
    pattern = _PATH_START + dot_slash + re.escape(form) + _PATH_END
    return re.search(pattern, output) is not None


def _denotes_only(root: Path, path: str, form: str, prefixes: tuple[str, ...]) -> bool:
    referents = {form, *(prefix + form for prefix in prefixes)} - {path}
    return not any((root / ref).exists() for ref in referents)


def qualified_phase1_summary(
    phase1: VerifyResult, working_tree: Phase1WorkingTree | None,
) -> str:
    """*phase1*'s summary, qualified when the checkout's state bears on it."""
    tree = _qualifying_tree(phase1, working_tree)
    if tree is None:
        return phase1.summary
    return f'{phase1.summary} — {_summary_qualifier(phase1, tree)}'


def render_working_tree_section(
    phase1: VerifyResult, working_tree: Phase1WorkingTree | None,
) -> str:
    """A '### Working-tree state' prompt subsection, or '' when unqualified."""
    tree = _qualifying_tree(phase1, working_tree)
    if tree is None:
        return ''
    lines = [
        '',
        '### Working-tree state',
        '',
        'Phase 1 ran in the live checkout, which differs from HEAD by '
        f'{_counts(tree.state)}.',
    ]
    lines += [f'{_pass_caveat(tree.state)}.'] if phase1.passed else _failed_lines(tree)
    return '\n'.join(lines) + '\n'


def _qualifying_tree(
    phase1: VerifyResult, working_tree: Phase1WorkingTree | None,
) -> Phase1WorkingTree | None:
    if working_tree is None:
        return None
    state = working_tree.state
    bears_on_result = bool(state.uncommitted) if phase1.passed else not state.is_clean
    return working_tree if bears_on_result else None


def _summary_qualifier(phase1: VerifyResult, tree: Phase1WorkingTree) -> str:
    if phase1.passed:
        return _pass_caveat(tree.state)
    named = tree.named_in_failure
    if not named:
        return f'the failing output names none of the {_counts(tree.state)}'
    shown, hidden = _capped(named, _SUMMARY_NAMED_PATH_CAP)
    more = f' +{hidden} more' if hidden else ''
    return (
        f'the failing output names {len(named)} file(s) that differ from the '
        f'committed tree: {", ".join(_describe(n) for n in shown)}{more}'
    )


def _pass_caveat(state: WorkingTreeState) -> str:
    return (
        f'{len(state.uncommitted)} tracked file(s) carry uncommitted edits, so this '
        'PASS may not hold for the committed versions of those files'
    )


_FAILURE_GUIDANCE = {
    TreeChange.UNTRACKED: (
        'Failures located in an untracked file come from a file absent from the '
        'committed tree: do NOT report them as code findings.'
    ),
    TreeChange.UNCOMMITTED: (
        'Failures located in an uncommitted file may come from its local edit or from '
        'committed code it still carries: compare the committed version '
        '(`git show HEAD:<path>`) before dismissing them.'
    ),
}


def _failed_lines(tree: Phase1WorkingTree) -> list[str]:
    named = tree.named_in_failure
    unattributed = 'are not attributed to any local change.'
    if not named:
        return [f'The failing output names none of those files, so its failures {unattributed}']
    shown, hidden = _capped(named, SECTION_NAMED_PATH_CAP)
    listed = [f'- `{n.path}` ({n.change.value})' for n in shown]
    if hidden:
        listed.append(f'- and {hidden} more')
    kinds = {n.change for n in named}
    return [
        'The failing output names these files, which differ from the committed tree:',
        '',
        *listed,
        '',
        *(_FAILURE_GUIDANCE[kind] for kind in TreeChange if kind in kinds),
        f'Failures located elsewhere in the output {unattributed}',
    ]


def _capped(
    named: tuple[NamedChange, ...], cap: int,
) -> tuple[tuple[NamedChange, ...], int]:
    return named[:cap], max(len(named) - cap, 0)


def _counts(state: WorkingTreeState) -> str:
    return f'{len(state.untracked)} untracked and {len(state.uncommitted)} uncommitted file(s)'


def _describe(named: NamedChange) -> str:
    return f'{named.path} ({named.change.value})'
