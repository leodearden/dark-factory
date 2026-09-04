"""Doc/table parity helpers for ARCHITECTURE.md section 3.1 (task 4535).

Test-support module — NOT production code; nothing under ``shared/src/``
changes for this. Pins ARCHITECTURE.md section 3.1's mermaid
``stateDiagram-v2`` diagram against the real transition authority,
``shared/src/shared/task_transitions.py::TRANSITIONS``: parses the diagram
out of the doc, reads the table, and reports the two as a set-difference so
a future edit to either side that the other doesn't mirror is a loud
failure instead of silent drift.

Consumed by ``shared/tests/test_architecture_doc_transition_parity.py``.
Importable bare (``from architecture_doc_transitions import ...``) because
``shared/tests/conftest.py`` inserts this ``tests/`` directory onto
``sys.path`` — the same mechanism ``silent_fallthrough_scan.py`` and
``capability_manifest_corpus.py`` already rely on.

Cross-file references below are cited as ``path/to/module.py::symbol``
(never a bare line number), per this repo's convention (CLAUDE.md).

Every failure mode here — a missing ``ARCHITECTURE.md``, a renamed section
heading, a section with zero or more than one mermaid fence — RAISES rather
than ``pytest.skip``s or returns an empty/default result. A tolerant reader
would let exactly the drift this guard exists to catch (a renamed heading,
a relocated diagram) pass through unnoticed, mirroring
``silent_fallthrough_scan.py::iter_first_party_files``'s sentinel
validation, done "to prevent a mis-resolved root from producing a
silently-empty scan".
"""

from __future__ import annotations

import re
from pathlib import Path

# shared/tests/architecture_doc_transitions.py -> parents[0]=shared/tests,
# parents[1]=shared, parents[2]=repo root. Same idiom as
# shared/tests/conftest.py:23 (REPO_ROOT) and every sibling scanner module in
# this directory; correct inside a `.worktrees/<id>` checkout too.
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / 'ARCHITECTURE.md'

#: The section heading that anchors the lifecycle diagram. ARCHITECTURE.md
#: has THREE mermaid fences (process topology, this one, the escalation
#: ladder) — extraction must anchor to this literal heading text rather than
#: grabbing "the first fence in the doc", or it would silently read the
#: wrong diagram.
SECTION_HEADING = '### 3.1 Status vocabulary'

_NEXT_HEADING_RE = re.compile(r'^### ', re.MULTILINE)
_MERMAID_FENCE_RE = re.compile(r'```mermaid\n(.*?)```', re.DOTALL)


def read_architecture_doc(root: Path = REPO_ROOT) -> str:
    """Return the full text of ``<root>/ARCHITECTURE.md``.

    Raises :class:`FileNotFoundError` — loudly, naming the resolved path —
    when the file is absent. Never ``pytest.skip``s and never falls back to
    an empty string: a caller with no doc to check against has no basis for
    asserting parity, and silently treating that as "nothing to check"
    would defeat the guard exactly when its precondition breaks.
    """
    doc_path = Path(root) / 'ARCHITECTURE.md'
    if not doc_path.is_file():
        raise FileNotFoundError(
            f'ARCHITECTURE.md not found at {doc_path!r} -- cannot check the '
            'task-status diagram/table parity guard (task 4535). Pass the real '
            'repo root, or restore the file.'
        )
    return doc_path.read_text(encoding='utf-8')


def extract_lifecycle_mermaid(text: str) -> str:
    """Return the body of section 3.1's mermaid fence (no backticks/prose).

    Slices *text* from :data:`SECTION_HEADING` to the next ``### `` heading,
    then requires that slice to contain EXACTLY one ```` ```mermaid ```` fence.
    Raises :class:`RuntimeError` — naming what was expected and what was
    found — when the heading is absent, or the section holds zero or more
    than one mermaid fence. A tolerant "best effort" extractor here would
    let a renamed heading or a relocated diagram silently produce an empty
    or wrong result, which is precisely the drift this guard exists to
    catch.
    """
    heading_idx = text.find(SECTION_HEADING)
    if heading_idx == -1:
        raise RuntimeError(
            f'heading {SECTION_HEADING!r} not found in ARCHITECTURE.md -- '
            'architecture_doc_transitions.py (task 4535) cannot locate the '
            'status-lifecycle diagram to check against '
            'shared/src/shared/task_transitions.py::TRANSITIONS. Was the section '
            'renamed or removed?'
        )
    section_start = heading_idx + len(SECTION_HEADING)
    next_heading_match = _NEXT_HEADING_RE.search(text, section_start)
    section_end = next_heading_match.start() if next_heading_match else len(text)
    section_text = text[section_start:section_end]

    fences = _MERMAID_FENCE_RE.findall(section_text)
    if len(fences) != 1:
        raise RuntimeError(
            f'expected exactly 1 mermaid fence under {SECTION_HEADING!r}, found '
            f'{len(fences)} (task 4535) -- architecture_doc_transitions.py cannot '
            'unambiguously locate the status-lifecycle diagram.'
        )
    return fences[0]
