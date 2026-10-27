"""Every merge-marker search a skill presents must select by SUBJECT, not by newest hit.

``git log --grep`` matches the whole commit message, so a newer commit whose
BODY quotes a merge subject shadows the genuine merge. The selecting form in
``skills/_shared/deriving-landed-sha.md`` step 1 scans past such a body match;
a ``--max-count=1`` stops at it. This guard extracts every marker search in
``skills/**/*.md`` (any fenced block or inline code span in which a
``git log`` or ``git rev-list`` greps for ``$S`` or a ``Merge task/…`` /
``Merge branch 'task/…'`` subject, whatever its other flags), RUNS it against a
fixture history where a body match shadows the genuine marker, and requires it
to print the genuine marker and nothing else.

It executes rather than greps for a flag because INV-10
``guards-exercise-behaviour`` (``docs/legibility/design-invariants.md``) asks a
guard to exercise the behaviour it protects: any newest-hit shortcut fails the
same way, and the prose around a snippet stays free to change.

A snippet that quotes a non-selecting form on purpose, to show what not to do,
is not executed when it carries ``<!-- marker-search-guard: counterexample -->``:
an inline span right after its closing backtick, a fenced block on one of its
lines.

This module reads real ``skills/`` paths, so it is registered in
``tests/scripts/test_skills_module_config_decision.py::SKILLS_CONSUMING_TESTS``.
Nothing here skips: a missing git, bash or skill file fails.
"""
from __future__ import annotations

import os
import re
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pytest
from git_listing import listed_files
from shared.merge_subject import format_merge_subject

REPO_ROOT = Path(__file__).parents[2]

COUNTEREXAMPLE_MARKER = 'marker-search-guard: counterexample'

_COUNTEREXAMPLE_TAG = re.compile(rf'[ \t]*<!--[ \t]*{re.escape(COUNTEREXAMPLE_MARKER)}[ \t]*-->')
_HISTORY_SEARCH_COMMAND = re.compile(r'\bgit\s+(?:log|rev-list)\b')
_MARKER_GREP_OPTION = re.compile(r'''--grep[= ]['"]?\^?(?:\$\{?S\b|Merge (?:branch ')?task/[<\d])''')
_FENCE_OPEN = re.compile(r'^\s*(`{3,})[^`]*$')
_FENCE_CLOSE = re.compile(r'^\s*(`{3,})\s*$')
_INLINE_SPAN = re.compile(r'(?<!`)`([^`]+)`(?!`)')
_BLANK_LINE = re.compile(r'\n\s*\n')
_SOFT_WRAP = re.compile(r'\s*\n\s*')


class MarkerSearch(NamedTuple):
    source: str
    line: int
    shell: str


class _Fence(NamedTuple):
    opener: int
    closer: int


class _Snippet(NamedTuple):
    first_line: int
    shell: str
    counterexample: bool


def marker_searches(markdown: str, *, source: str) -> list[MarkerSearch]:
    lines = markdown.split('\n')
    fences = _fences(lines)
    snippets = [
        *(_fenced_snippet(lines, fence) for fence in fences),
        *_inline_snippets(_blank_fences(lines, fences)),
    ]
    return [
        MarkerSearch(source, snippet.first_line, snippet.shell)
        for snippet in sorted(snippets, key=lambda snippet: snippet.first_line)
        if _is_marker_search(snippet.shell) and not snippet.counterexample
    ]


def _fences(lines: list[str]) -> list[_Fence]:
    fences: list[_Fence] = []
    index = 0
    while index < len(lines):
        opener = _FENCE_OPEN.match(lines[index])
        if opener is None:
            index += 1
            continue
        closer = _closing_fence(lines, index, len(opener.group(1)))
        fences.append(_Fence(index, closer))
        index = closer + 1
    return fences


def _closing_fence(lines: list[str], opener: int, run_length: int) -> int:
    for index in range(opener + 1, len(lines)):
        closer = _FENCE_CLOSE.match(lines[index])
        if closer is not None and len(closer.group(1)) >= run_length:
            return index
    return len(lines)


def _fenced_snippet(lines: list[str], fence: _Fence) -> _Snippet:
    body = textwrap.dedent('\n'.join(lines[fence.opener + 1 : fence.closer]))
    occupied = lines[fence.opener : fence.closer + 1]
    return _Snippet(
        fence.opener + 1, body, any(COUNTEREXAMPLE_MARKER in line for line in occupied),
    )


def _blank_fences(lines: list[str], fences: list[_Fence]) -> str:
    fenced = {index for fence in fences for index in range(fence.opener, fence.closer + 1)}
    return '\n'.join('' if index in fenced else line for index, line in enumerate(lines))


def _inline_snippets(prose: str) -> list[_Snippet]:
    return [
        _Snippet(
            _line_number(prose, span.start()),
            _SOFT_WRAP.sub(' ', span.group(1)),
            _COUNTEREXAMPLE_TAG.match(prose, span.end()) is not None,
        )
        for span in _INLINE_SPAN.finditer(prose)
        if _BLANK_LINE.search(span.group(1)) is None
    ]


def _line_number(text: str, offset: int) -> int:
    return text.count('\n', 0, offset) + 1


def _is_marker_search(shell: str) -> bool:
    return (
        _HISTORY_SEARCH_COMMAND.search(shell) is not None
        and _MARKER_GREP_OPTION.search(shell) is not None
    )


_SYNTHETIC_SKILL = r"""# Synthetic skill

1. Find the marker:

   ```bash
   S="Merge task/<TASK_ID> into main"
   git log main --fixed-strings --grep="$S" --format='%H%x09%s' \
     | awk -F'\t' -v s="$S" '$2==s && !seen {print $1; seen=1}'
   ```

   - Or re-derive it inline (`git log main
     --fixed-strings --grep="$S" --format='%H%x09%s' | awk -F'\t' -v s="$S" '$2==s && !seen {print $1; seen=1}'`), which selects by subject.

Never `git log main --fixed-strings --grep="Merge task/1 into main" --max-count=1 --format=%H` <!-- marker-search-guard: counterexample -->

Prefer `git log main --fixed-strings --grep="$S" --format='%H%x09%s' | awk -F'\t' -v s="$S" '$2==s && !seen {print $1; seen=1}'` to `git log main --fixed-strings --grep="$S" --max-count=1 --format=%H` <!-- marker-search-guard: counterexample -->.

A careless rewrite: `git log main --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H`, `git log -1 --format=%H --grep "${S}"` or `git rev-list -1 main --grep="^Merge branch 'task/42'"`.

```bash
git log main --extended-regexp --format='%H%x09%s' --grep='^Merge task/[0-9]+ into main$' | awk -F '\t' '{print $1}'
```

Split with `awk -F'\t' '{print $1}'`, never use the bare `--grep ... --max-count=1`, and note that `--grep="Merge task/<task-id> into main"` alone is no command.
"""

_SYNTHETIC_FENCED_SHELL = r"""S="Merge task/<TASK_ID> into main"
git log main --fixed-strings --grep="$S" --format='%H%x09%s' \
  | awk -F'\t' -v s="$S" '$2==s && !seen {print $1; seen=1}'"""

_SYNTHETIC_INLINE_SHELL = (
    r"""git log main --fixed-strings --grep="$S" --format='%H%x09%s' | """
    r"""awk -F'\t' -v s="$S" '$2==s && !seen {print $1; seen=1}'"""
)


def test_marker_search_extractor_reads_every_snippet_shape() -> None:
    found = marker_searches(_SYNTHETIC_SKILL, source='skills/synthetic.md')

    assert found == [
        MarkerSearch('skills/synthetic.md', 5, _SYNTHETIC_FENCED_SHELL),
        MarkerSearch('skills/synthetic.md', 11, _SYNTHETIC_INLINE_SHELL),
        MarkerSearch('skills/synthetic.md', 16, _SYNTHETIC_INLINE_SHELL),
        MarkerSearch(
            'skills/synthetic.md', 18,
            'git log main --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H',
        ),
        MarkerSearch('skills/synthetic.md', 18, 'git log -1 --format=%H --grep "${S}"'),
        MarkerSearch(
            'skills/synthetic.md', 18, '''git rev-list -1 main --grep="^Merge branch 'task/42'"''',
        ),
    ]


FIXTURE_TASK_ID = '7001'
_FIXTURE_BRANCH = f'task/{FIXTURE_TASK_ID}'
_MARKER_SUBJECT = format_merge_subject(_FIXTURE_BRANCH, 'main')
_HAND_MERGE_SUBJECT = f"Merge branch '{_FIXTURE_BRANCH}'"
_FIXTURE_IDENTITY = (
    '-c', 'user.name=Marker Search Guard',
    '-c', 'user.email=marker-search-guard@example.invalid',
    '-c', 'commit.gpgsign=false',
)
_PLACEHOLDER = re.compile(r'<[A-Za-z]+[-_](?:ID|id)>')
_SHA = re.compile(r'\b[0-9a-f]{40}\b')


def _skill_marker_searches() -> list[MarkerSearch]:
    return [
        search
        for path in listed_files(REPO_ROOT, 'skills/*.md')
        for search in marker_searches(
            path.read_text(encoding='utf-8'), source=path.relative_to(REPO_ROOT).as_posix(),
        )
    ]


_LIVE_MARKER_SEARCHES = _skill_marker_searches()


def _search_id(search: MarkerSearch) -> str:
    return f'{search.source}:{search.line}'


@dataclass(frozen=True)
class ShadowedHistory:
    root: Path
    genuine: frozenset[str]
    shadow: str


def _isolated_git_env(root: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    env['GIT_CONFIG_GLOBAL'] = os.devnull
    env['GIT_CONFIG_NOSYSTEM'] = '1'
    env['GIT_CEILING_DIRECTORIES'] = str(root.parent.resolve())
    return env


def _fixture_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ['git', *_FIXTURE_IDENTITY, *args],
        cwd=root, env=_isolated_git_env(root), capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, (
        f'building the fixture history: {proc.args} failed in {root} with '
        f'rc={proc.returncode}; stderr: {proc.stderr!r}'
    )
    return proc.stdout.strip()


def _commit_file(root: Path, name: str, *messages: str) -> str:
    (root / name).write_text(f'{messages[0]}\n', encoding='utf-8')
    _fixture_git(root, 'add', name)
    _fixture_git(root, 'commit', *(arg for message in messages for arg in ('-m', message)))
    return _fixture_git(root, 'rev-parse', 'HEAD')


def _merge_into_main(root: Path, subject: str) -> str:
    _fixture_git(root, 'checkout', '-q', 'main')
    _fixture_git(root, 'merge', '--no-ff', '-m', subject, _FIXTURE_BRANCH)
    return _fixture_git(root, 'rev-parse', 'HEAD')


@pytest.fixture(scope='module')
def shadowed_history(tmp_path_factory: pytest.TempPathFactory) -> ShadowedHistory:
    root = tmp_path_factory.mktemp('shadowed-history') / 'repo'
    root.mkdir()
    _fixture_git(root, 'init', '-q', '-b', 'main')
    assert (root / '.git').is_dir(), f'git init did not make {root} a repository root'
    _commit_file(root, 'README', 'initial commit')
    _fixture_git(root, 'checkout', '-q', '-b', _FIXTURE_BRANCH)
    _commit_file(root, 'work-1', 'task work, part one')
    _fixture_git(root, 'checkout', '-q', 'main')
    _commit_file(root, 'main-1', 'main moves on')
    orchestrator_marker = _merge_into_main(root, _MARKER_SUBJECT)
    _fixture_git(root, 'checkout', '-q', _FIXTURE_BRANCH)
    _commit_file(root, 'work-2', 'task work, part two')
    hand_merge = _merge_into_main(root, _HAND_MERGE_SUBJECT)
    shadow = _commit_file(
        root, 'docs',
        'docs: explain the marker search',
        f'A marker search for "{_MARKER_SUBJECT}" or "{_HAND_MERGE_SUBJECT}" '
        'also matches this body, even one anchored to a whole line:',
        f'{_MARKER_SUBJECT}\n{_HAND_MERGE_SUBJECT}',
    )
    return ShadowedHistory(root, frozenset({orchestrator_marker, hand_merge}), shadow)


def _snippet_env(history: ShadowedHistory) -> dict[str, str]:
    return {**_isolated_git_env(history.root), 'S': _MARKER_SUBJECT}


def _resolve_placeholders(shell: str) -> str:
    return _PLACEHOLDER.sub(FIXTURE_TASK_ID, shell)


def _violated_condition(
    history: ShadowedHistory, returncode: int, printed: set[str],
) -> str | None:
    if returncode != 0:
        return f'the snippet exited rc={returncode} instead of running cleanly'
    if not printed:
        return 'the snippet printed no commit sha: the genuine marker behind a newer body match was lost'
    if history.shadow in printed:
        return (
            f'the snippet printed {history.shadow}, a non-merge commit that only '
            'quotes the subject in its body'
        )
    if not printed <= history.genuine:
        return f'the snippet printed {sorted(printed - history.genuine)}, which are not genuine markers'
    return None


def _selection_defect(history: ShadowedHistory, search: MarkerSearch) -> str | None:
    shell = _resolve_placeholders(search.shell)
    proc = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', shell],
        cwd=history.root, env=_snippet_env(history),
        capture_output=True, text=True, timeout=60, check=False,
    )
    defect = _violated_condition(history, proc.returncode, set(_SHA.findall(proc.stdout)))
    if defect is None:
        return None
    return (
        f'{_search_id(search)}: a merge-marker search must select the newest commit whose '
        f'SUBJECT is the marker, but {defect}.\n'
        f'genuine markers: {sorted(history.genuine)}\n'
        f'shell (placeholders resolved to {FIXTURE_TASK_ID}, S={_MARKER_SUBJECT!r}):\n{shell}\n'
        f'rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    )


def _with_max_count(search: MarkerSearch) -> MarkerSearch:
    mutant = _MARKER_GREP_OPTION.sub(
        lambda grep: f'--max-count=1 {grep.group(0)}', search.shell, count=1,
    )
    return search._replace(shell=mutant)


def test_skills_present_marker_searches() -> None:
    assert _LIVE_MARKER_SEARCHES, (
        'no merge-marker search (a code snippet whose git log or git rev-list greps for '
        '$S or a merge subject) was found in skills/**/*.md, so the parametrised guards below collected '
        'nothing and would pass without running a single snippet'
    )


@pytest.mark.parametrize('search', _LIVE_MARKER_SEARCHES, ids=_search_id)
def test_documented_marker_search_selects_the_subject_match(
    shadowed_history: ShadowedHistory, search: MarkerSearch,
) -> None:
    defect = _selection_defect(shadowed_history, search)
    assert defect is None, defect


@pytest.mark.parametrize('search', _LIVE_MARKER_SEARCHES, ids=_search_id)
def test_max_count_reintroduction_is_caught(
    shadowed_history: ShadowedHistory, search: MarkerSearch,
) -> None:
    mutant = _with_max_count(search)
    assert _selection_defect(shadowed_history, mutant) is not None, (
        f'{_search_id(search)} with --max-count=1 injected still selected the genuine '
        'marker: the fixture history does not shadow this snippet\'s subject, so the '
        'guard could not go red for it. Extend shadowed_history to cover it.\n'
        f'mutant:\n{mutant.shell}'
    )
