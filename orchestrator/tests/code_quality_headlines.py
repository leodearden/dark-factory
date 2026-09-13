"""Parser for the fourteen code-quality heuristic HEADLINES (task 5225).

Test-support module — NOT production code. Nothing under any ``src/`` tree
reads ``docs/code-quality.md``: an orchestrator operating a project whose
checkout lacks that file is unaffected, because the role prompts carry the
headlines inline and only this guard compares the two.

One parser serves BOTH sides of the comparison — the doc and the rendered role
prompt — against one anchor string, :data:`HEADLINE_SECTION_HEADING`. That is
the point: two parsers, or two anchors, would reintroduce the class of bug
where a drift test's halves read subtly different shapes and agree by accident.

Every failure mode here — a missing doc, a renamed heading, a hole in the
numbering — RAISES, naming what it looked for and what it found. It never
returns an empty list and never ``pytest.skip``s: a tolerant reader would let
exactly the drift this guard exists to catch pass unnoticed (same discipline as
``shared/tests/architecture_doc_transitions.py``).

Consumed by ``orchestrator/tests/test_code_quality_guidance_parity.py``.
Importable bare (``from code_quality_headlines import ...``) because
``orchestrator/tests/conftest.py`` inserts this ``tests/`` directory onto
``sys.path`` — the same mechanism ``_orch_helpers.py`` relies on.
"""

from __future__ import annotations

import re
from pathlib import Path

# Resolved from THIS FILE, never from the process CWD: merge-verify runs pytest
# from orchestrator/ while a plain run starts at the repo root, and both must
# find the same doc. Correct inside a ``.worktrees/<id>`` checkout too, which is
# where task worktrees run.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / 'docs' / 'code-quality.md'

#: The ONE anchor, used unchanged on the doc side and the prompt side.
HEADLINE_SECTION_HEADING = '## The fourteen heuristics'

_NEXT_HEADING_RE = re.compile(r'^## ', re.MULTILINE)

#: A headline is ``<n>. **<text>**`` at COLUMN 0. Anchoring there is what keeps
#: an indented continuation line's own bold run (the real doc's heuristic 14
#: carries ``**No cheating**``) from being read as a further headline.
_HEADLINE_RE = re.compile(r'^(\d{1,2})\. \*\*(.+?)\*\*', re.MULTILINE)


def numbered_headlines(text: str, heading: str) -> list[str]:
    """Return the bold headlines of the numbered list under *heading*, in order.

    Slices *text* from the literal *heading* line to the next line starting
    ``## ``, then reads every column-0 ``<n>. **<text>**`` line in that slice.

    Raises :class:`ValueError` naming *heading* when it is absent, and
    :class:`ValueError` naming the observed numbers when they are not exactly
    ``1..n``. The numbering check is what makes "an unbolded numbered line is
    not a headline" safe: an item silently dropped from the middle of the list
    leaves a hole, and a hole is an error rather than a shorter answer.
    """
    heading_idx = text.find(heading)
    if heading_idx == -1:
        raise ValueError(
            f'heading {heading!r} not found — code_quality_headlines.py '
            '(task 5225) cannot locate the numbered heuristic list to compare. '
            'Was the section renamed or removed?'
        )
    section_start = heading_idx + len(heading)
    terminator = _NEXT_HEADING_RE.search(text, section_start)
    section = text[section_start:terminator.start() if terminator else len(text)]

    matches = _HEADLINE_RE.findall(section)
    numbers = [int(number) for number, _ in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(
            f'the numbered list under {heading!r} is not numbered 1..n — '
            f'observed {numbers} (task 5225). An item was dropped, renumbered, '
            'or lost its bold headline.'
        )
    return [headline for _, headline in matches]


def doc_headlines(path: Path = DOC_PATH) -> list[str]:
    """Return the fourteen headlines of ``docs/code-quality.md``.

    A missing file raises :class:`OSError` from the read — loudly, naming the
    path — rather than degrading to an empty list.
    """
    return numbered_headlines(path.read_text(encoding='utf-8'), HEADLINE_SECTION_HEADING)
