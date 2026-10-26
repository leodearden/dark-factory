"""The prompts name the dangling-citation handoff by interpolation (task 5749).

These pin WIRING, not prose: the constants the ``dangling_citation`` leaf owns
must appear in the assembled prompts, so the prompt text and the code that
emits the finding cannot drift apart.
"""

from __future__ import annotations

import re

import pytest

from fused_memory.reconciliation.dangling_citation import (
    DANGLING_CITATION_LINE_LABEL,
    DANGLING_CROSS_RUN_CITATION_FLAG_TYPE,
)
from fused_memory.reconciliation.prompts import CITATION_REPAIR_TOOL_BLOCK
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

STAGE3_HANDOFF_HEADING = '## Dangling Cross-Run Citations — Detected For You'


def _names_report_tool(text: str, tool_name: str) -> bool:
    """Is *tool_name* named in *text* as a whole recon-report tool identifier?

    Copied, not imported, from
    ``test_recon_report_stage_gated_tool_advertisement.py::_names_report_tool``.
    """
    pattern = (
        r'(?<![A-Za-z0-9_])(?:mcp__recon-report__)?'
        + re.escape(tool_name)
        + r'(?![A-Za-z0-9_])'
    )
    return re.search(pattern, text) is not None


def _stage3_handoff_section() -> str:
    start = STAGE3_SYSTEM_PROMPT.find(STAGE3_HANDOFF_HEADING)
    assert start != -1, f'Stage 3 prompt lacks the {STAGE3_HANDOFF_HEADING!r} section'
    end = STAGE3_SYSTEM_PROMPT.find('\n## ', start + len(STAGE3_HANDOFF_HEADING))
    return STAGE3_SYSTEM_PROMPT[start:] if end == -1 else STAGE3_SYSTEM_PROMPT[start:end]


def test_stage3_is_told_the_flag_type_the_scan_files() -> None:
    assert DANGLING_CROSS_RUN_CITATION_FLAG_TYPE in _stage3_handoff_section()


@pytest.mark.parametrize(
    'constant', [DANGLING_CITATION_LINE_LABEL, DANGLING_CROSS_RUN_CITATION_FLAG_TYPE],
)
def test_shared_repair_block_names_the_handoff(constant: str) -> None:
    assert constant in CITATION_REPAIR_TOOL_BLOCK


@pytest.mark.parametrize(
    ('label', 'prompt'),
    [
        ('stage1', STAGE1_SYSTEM_PROMPT),
        ('stage2:dark_factory', build_stage2_system_prompt('dark_factory')),
        ('stage2:autopilot_video', build_stage2_system_prompt('autopilot_video')),
    ],
)
def test_repair_block_reaches_each_holding_stage_once(label: str, prompt: str) -> None:
    assert prompt.count(CITATION_REPAIR_TOOL_BLOCK) == 1, label


def test_stage3_handoff_section_does_not_name_the_repair_tool() -> None:
    assert not _names_report_tool(_stage3_handoff_section(), 'repair_memory_citation')
