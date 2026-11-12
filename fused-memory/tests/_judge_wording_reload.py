"""A fresh import of ``write_triage_judge_wording`` under another shipped prompt.

The committed π and π2 traceability tests use it to show they read the prompt
each artifact measured, not the live ``JUDGE_SYSTEM_PROMPT``: the wording
module captures the shipped prompt at import, so only a fresh import sees a
substitute.
"""
from __future__ import annotations

import contextlib
import sys
import types
from collections.abc import Iterator
from pathlib import Path

from _fm_helpers import load_script_module

from fused_memory.server import write_triage_judge

_SCRIPT = Path(__file__).parent.parent / 'scripts' / 'write_triage_judge_wording.py'
_THROWAWAY = 'write_triage_judge_wording_under_another_shipped_prompt'


@contextlib.contextmanager
def wording_imported_under(shipped_prompt: str) -> Iterator[types.ModuleType]:
    """Yield the wording module as imported while *shipped_prompt* is shipped."""
    live = write_triage_judge.JUDGE_SYSTEM_PROMPT
    write_triage_judge.JUDGE_SYSTEM_PROMPT = shipped_prompt
    try:
        yield load_script_module(_SCRIPT, _THROWAWAY)
    finally:
        sys.modules.pop(_THROWAWAY, None)
        write_triage_judge.JUDGE_SYSTEM_PROMPT = live
