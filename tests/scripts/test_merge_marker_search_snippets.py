"""Every merge-marker search a skill presents must select by SUBJECT, not by newest hit.

``git log --grep`` matches the whole commit message, so a newer commit whose
BODY quotes a merge subject shadows the genuine merge. The selecting form in
``skills/_shared/deriving-landed-sha.md`` step 1 scans past such a body match;
a ``--max-count=1`` stops at it. This guard extracts every marker search in
``skills/**/*.md`` (any fenced block or inline code span carrying ``--grep``
plus a fixed-strings flag), RUNS it against a fixture history where a body
match shadows the genuine marker, and requires it to print the genuine marker
and nothing else.

It executes rather than greps for a flag because INV-10
``guards-exercise-behaviour`` (``docs/legibility/design-invariants.md``) asks a
guard to exercise the behaviour it protects: any newest-hit shortcut fails the
same way, and the prose around a snippet stays free to change.

A snippet that quotes a non-selecting form on purpose, to show what not to do,
carries ``<!-- marker-search-guard: counterexample -->`` on one of its lines
and is not executed.

This module reads real ``skills/`` paths, so it is registered in
``tests/scripts/test_skills_module_config_decision.py::SKILLS_CONSUMING_TESTS``.
Nothing here skips: a missing git, bash or skill file fails.
"""
from __future__ import annotations

import re
import textwrap
from typing import NamedTuple

COUNTEREXAMPLE_MARKER = 'marker-search-guard: counterexample'

_FIXED_STRINGS_FLAG = re.compile(r'--fixed-strings|(?<=\s)-F(?=\s)')
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
    last_line: int
    shell: str


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
        if _is_marker_search(snippet.shell) and not _opts_out(lines, snippet)
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
    return _Snippet(fence.opener + 1, min(fence.closer + 1, len(lines)), body)


def _blank_fences(lines: list[str], fences: list[_Fence]) -> str:
    fenced = {index for fence in fences for index in range(fence.opener, fence.closer + 1)}
    return '\n'.join('' if index in fenced else line for index, line in enumerate(lines))


def _inline_snippets(prose: str) -> list[_Snippet]:
    return [
        _Snippet(
            _line_number(prose, span.start()),
            _line_number(prose, span.end() - 1),
            _SOFT_WRAP.sub(' ', span.group(1)),
        )
        for span in _INLINE_SPAN.finditer(prose)
        if _BLANK_LINE.search(span.group(1)) is None
    ]


def _line_number(text: str, offset: int) -> int:
    return text.count('\n', 0, offset) + 1


def _is_marker_search(shell: str) -> bool:
    return '--grep' in shell and _FIXED_STRINGS_FLAG.search(shell) is not None


def _opts_out(lines: list[str], snippet: _Snippet) -> bool:
    occupied = lines[snippet.first_line - 1 : snippet.last_line]
    return any(COUNTEREXAMPLE_MARKER in line for line in occupied)


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

```bash
git log main --extended-regexp --format='%H %s' --grep='^Merge task/[0-9]+ into main$'
```

Split with `awk -F'\t' '{print $1}'`, and never use the bare `--grep ... --max-count=1`.
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
    ]
