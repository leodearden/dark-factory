"""Which recon system prompts can write memory, read off each prompt itself.

Shared by the drift pins that require write-time guidance to reach exactly the
recon prompts able to write (fused-memory/tests/test_referent_guidance_prompt_drift.py,
fused-memory/tests/test_metadata_vocabulary_prompt_pinning.py), so they agree
on one scope rule by construction rather than by copy.

SCOPE IS READ OFF EACH PROMPT. A prompt can write iff its own
``## Available Tools`` section names ``mcp__fused-memory__add_memory``. The
prompts themselves are found with ``dir()`` rather than listed here: a
hand-maintained list in a test file the prompt author never opens goes stale
unnoticed, as task 3878 found in ``test_recon_report_guidance_drift.py``.
Consumers pin the split in both directions. A stage that gains the write tool
goes red until it carries the guidance, and a read-only stage cannot quietly
pick it up.

Lives outside conftest.py, for the same reason ``_fm_helpers`` does, so test
files can ``from _recon_prompt_write_scope import X``.
"""

from __future__ import annotations

from collections.abc import Mapping

from fused_memory.reconciliation.prompts import judge, stage1, stage2, stage3
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt

WRITE_TOOL = 'mcp__fused-memory__add_memory'
TOOL_LISTING_HEADING = '## Available Tools\n'


def recon_system_prompts() -> dict[str, str]:
    prompts = {
        f'{module.__name__.rsplit(".", 1)[-1]}.{name}': getattr(module, name)
        for module in (stage1, stage2, stage3, judge)
        for name in dir(module)
        if name.endswith('_SYSTEM_PROMPT') and isinstance(getattr(module, name), str)
    }
    for project_id in ('dark_factory', 'autopilot_video'):
        prompts[f'build_stage2_system_prompt({project_id!r})'] = build_stage2_system_prompt(
            project_id
        )
    return prompts


def tool_listing(prompt: str) -> str:
    """The prompt's ``## Available Tools`` section, up to the next ``## `` heading.

    Empty when the prompt has no such section. Prose elsewhere that merely
    mentions a tool, such as a read-only stage describing what Stage 1 wrote,
    is not a grant of that tool.
    """
    start = prompt.find(TOOL_LISTING_HEADING)
    if start < 0:
        return ''
    end = prompt.find('\n## ', start + len(TOOL_LISTING_HEADING))
    return prompt[start : end if end >= 0 else None]


def split_by_write_grant(prompts: Mapping[str, str]) -> tuple[list[str], list[str]]:
    """(sorted names that can write, sorted names that cannot)."""
    writing = sorted(name for name, prompt in prompts.items() if WRITE_TOOL in tool_listing(prompt))
    return writing, sorted(set(prompts) - set(writing))
