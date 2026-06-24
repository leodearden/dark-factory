"""Doc-content and doc/code-consistency tests for the two-layer merge queue.

This file encodes the required content of the operator-facing and
developer-facing two-layer merge-queue documentation as executable
assertions (TDD-for-docs pattern, mirroring test_skill_prompt.py).

Each ``test_skill_*`` test asserts that stable phrases are present in
``skills/merge-queue/SKILL.md``; ``test_design_doc_*`` tests target
``skills/merge-queue/references/two-layer-model.md``.  A doc/code-
consistency guard checks that every code symbol the docs cite exists in
``orchestrator/src/orchestrator/merge_queue.py`` (MERGE_QUEUE_SRC).

Authoritative source for required content: the λ=1895 integration-test
header (test_merge_queue_two_layer_integration.py:1-55) and the real
public symbols in merge_queue.py.
"""

from __future__ import annotations

import pathlib


def _repo_root() -> pathlib.Path:
    """Return the repository root (two levels above this test file)."""
    # __file__ → orchestrator/tests/test_merge_queue_docs.py
    # parents[0] → orchestrator/tests/
    # parents[1] → orchestrator/
    # parents[2] → <repo root>
    return pathlib.Path(__file__).parents[2]


def _read(relpath: str) -> str:
    """Return the text of a repo-root-relative file."""
    return (_repo_root() / relpath).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Doc/code consistency guard — read once; all tests share this constant.
# Every code symbol the docs cite must exist in this source text.
# ---------------------------------------------------------------------------
MERGE_QUEUE_SRC: str = _read("orchestrator/src/orchestrator/merge_queue.py")
