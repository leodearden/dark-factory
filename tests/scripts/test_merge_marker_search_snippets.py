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

from typing import NamedTuple


class MarkerSearch(NamedTuple):
    source: str
    line: int
    shell: str


def marker_searches(markdown: str, *, source: str) -> list[MarkerSearch]:
    raise NotImplementedError


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
