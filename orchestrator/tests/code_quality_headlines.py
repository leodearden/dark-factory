"""Parsers for the code-quality lists the role prompts duplicate (task 5225).

Test-support module — NOT production code. Nothing under any ``src/`` tree
reads ``docs/code-quality.md``: an orchestrator operating a project whose
checkout lacks that file is unaffected, because the role prompts carry the
lists inline and only this guard compares the two.

Two parsers, both reading COLUMN-0 list items out of a markdown section:
:func:`numbered_headlines` for the fourteen ``<n>. **<headline>**`` heuristics
and :func:`bold_item_labels` for a ``- **<label>**`` bullet list. They share one
section slicer, so both halves of every comparison read the same shape — two
independently-written slicers would reintroduce the class of bug where a drift
test's halves agree by accident.

The section heading is matched as a WHOLE LINE, the same way the section's
terminator is. An unanchored substring search would match a DEMOTED heading
(``### The fourteen heuristics``) that the ``^## `` terminator then fails to
close, silently running the slice on into later sections — reporting success on
exactly the structural drift this guard exists to catch.

Every failure mode here — a missing doc, a renamed heading, a hole in the
numbering — RAISES, naming what it looked for and what it found. It never
returns an empty list and never ``pytest.skip``s: a tolerant reader would let
exactly the drift this guard exists to catch pass unnoticed (same discipline as
``shared/tests/architecture_doc_transitions.py``).

Consumed by ``orchestrator/tests/test_code_quality_guidance_parity.py``, which
owns WHICH sections are compared against which. Importable bare (``from
code_quality_headlines import ...``) because ``orchestrator/tests/conftest.py``
inserts this ``tests/`` directory onto ``sys.path`` — the same mechanism
``_orch_helpers.py`` relies on.
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

#: The heuristics anchor, used unchanged on the doc side and the prompt side.
HEADLINE_SECTION_HEADING = '## The fourteen heuristics'

_NEXT_HEADING_RE = re.compile(r'^## ', re.MULTILINE)

#: A headline is ``<n>. **<text>**`` at COLUMN 0. Anchoring there is what keeps
#: an indented continuation line's own bold run (the real doc's heuristic 14
#: carries ``**No cheating**``) from being read as a further headline.
_HEADLINE_RE = re.compile(r'^(\d{1,2})\. \*\*(.+?)\*\*', re.MULTILINE)

#: A bullet label is ``- **<text>**`` at COLUMN 0, for the same reason: a
#: bullet's continuation lines carry their own inline bold and are not items.
_BULLET_LABEL_RE = re.compile(r'^- \*\*(.+?)\*\*', re.MULTILINE)


def _section(text: str, heading: str) -> str:
    """Return the body between the *heading* LINE and the next ``## `` line.

    Raises :class:`ValueError` naming *heading* when no line of *text* is
    exactly that heading — including when it appears only as a substring, e.g.
    demoted to ``###``, which is drift rather than a match.
    """
    start = re.search(rf'^{re.escape(heading)}[ \t]*$', text, re.MULTILINE)
    if start is None:
        raise ValueError(
            f'heading {heading!r} not found as a whole line — '
            'code_quality_headlines.py (task 5225) cannot locate the list to '
            'compare. Was the section renamed, removed, or demoted below `## `?'
        )
    terminator = _NEXT_HEADING_RE.search(text, start.end())
    return text[start.end():terminator.start() if terminator else len(text)]


def numbered_headlines(text: str, heading: str) -> list[str]:
    """Return the bold headlines of the numbered list under *heading*, in order.

    Raises :class:`ValueError` naming *heading* when it is absent, and
    :class:`ValueError` naming the observed numbers when they are not exactly
    ``1..n``. The numbering check is what makes "an unbolded numbered line is
    not a headline" safe: an item silently dropped from the middle of the list
    leaves a hole, and a hole is an error rather than a shorter answer.
    """
    matches = _HEADLINE_RE.findall(_section(text, heading))
    numbers = [int(number) for number, _ in matches]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(
            f'the numbered list under {heading!r} is not numbered 1..n — '
            f'observed {numbers} (task 5225). An item was dropped, renumbered, '
            'or lost its bold headline.'
        )
    return [headline for _, headline in matches]


def bold_item_labels(text: str, heading: str) -> list[str]:
    """Return the bold LABELS of the bullet list under *heading*, in order.

    Only the label — the leading bold run — is returned, never the prose after
    it. That is the point: the doc and the prompts word the bodies differently
    on purpose (the prompt drops the doc's repo-specific measurements), so only
    the labels are comparable, and comparing them catches a bullet renamed,
    reordered, added or dropped on either side.

    Raises :class:`ValueError` naming *heading* when it is absent.
    """
    return _BULLET_LABEL_RE.findall(_section(text, heading))


def doc_headlines(path: Path = DOC_PATH) -> list[str]:
    """Return the fourteen headlines of ``docs/code-quality.md``.

    A missing file raises :class:`OSError` from the read — loudly, naming the
    path — rather than degrading to an empty list.
    """
    return numbered_headlines(path.read_text(encoding='utf-8'), HEADLINE_SECTION_HEADING)


def doc_bold_item_labels(heading: str, path: Path = DOC_PATH) -> list[str]:
    """Return the bullet labels under *heading* in ``docs/code-quality.md``.

    Same loud-on-missing-file discipline as :func:`doc_headlines`.
    """
    return bold_item_labels(path.read_text(encoding='utf-8'), heading)
