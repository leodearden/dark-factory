"""The normative code-quality doc's texts that the legibility coder tests
expect in, or out of, the coder prompt. They are computed from the doc at test
time, never restated (plans/census-incremental-prd.md §4.8 row 17).

This module binds this checkout's orchestrator/src itself, the way
scripts/legibility/coder.py does, so a test file importing it does not depend
on having imported coder.py first.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_ORCH_SRC = Path(__file__).resolve().parents[2] / "orchestrator" / "src"
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from orchestrator.agents import code_quality  # noqa: E402

__all__ = ["code_quality", "definition_body", "heuristic_headlines"]

_HEURISTIC_HEADLINE_RE = re.compile(r"^\d+\. \*\*(.+?)\*\*", re.MULTILINE)


def _doc_text() -> str:
    return code_quality.NORMATIVE_DOC.read_text(encoding="utf-8")


def definition_body() -> str:
    return code_quality.section(_doc_text(), "## Definition").strip()


def heuristic_headlines() -> list[str]:
    headlines = _HEURISTIC_HEADLINE_RE.findall(
        code_quality.section(_doc_text(), "## The fourteen heuristics")
    )
    assert headlines, "no numbered bold heuristic headline parsed from the normative doc"
    return headlines
