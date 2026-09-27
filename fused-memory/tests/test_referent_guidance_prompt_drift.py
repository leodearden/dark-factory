"""Drift pin: the referent-declaration fragment reaches every recon prompt that writes.

Task 3675. ``fused_memory.utils.referent_resolution::render_referent_declaration_guidance``
is the one source of the ``entities`` teaching. The recon stage prompts
interpolate it into their f-strings, so ``render_*() in <prompt>`` is a
byte-identical containment pin, the shape ``test_standing_decision_prompt_drift.py``
uses. A re-hardcoded or reworded copy fails here, while rewording the source
never does.

SCOPE IS READ OFF EACH PROMPT. A prompt is in scope iff its own
``## Available Tools`` section names ``mcp__fused-memory__add_memory``. The prompts themselves are found with
``dir()`` rather than listed here: a hand-maintained list in a test file the
prompt author never opens goes stale unnoticed, as task 3878 found in
``test_recon_report_guidance_drift.py``. The split is pinned in both
directions. A stage that gains the write tool goes red until it carries the
fragment, and a read-only stage cannot quietly pick it up.

Whether the fragment's worked example is one the live gate accepts is certified
separately, in ``test_referent_declaration_examples.py``.
"""

from __future__ import annotations

import pytest

from fused_memory.reconciliation.prompts import judge, stage1, stage2, stage3
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt
from fused_memory.utils.referent_resolution import render_referent_declaration_guidance

_WRITE_TOOL = 'mcp__fused-memory__add_memory'
_TOOL_LISTING_HEADING = '## Available Tools\n'


def _system_prompts() -> dict[str, str]:
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


def _tool_listing(prompt: str) -> str:
    """The prompt's ``## Available Tools`` section, up to the next ``## `` heading.

    Empty when the prompt has no such section. Prose elsewhere that merely
    mentions a tool, such as a read-only stage describing what Stage 1 wrote,
    is not a grant of that tool.
    """
    start = prompt.find(_TOOL_LISTING_HEADING)
    if start < 0:
        return ''
    end = prompt.find('\n## ', start + len(_TOOL_LISTING_HEADING))
    return prompt[start : end if end >= 0 else None]


_PROMPTS = _system_prompts()
_WRITING = sorted(name for name, prompt in _PROMPTS.items() if _WRITE_TOOL in _tool_listing(prompt))
_NOT_WRITING = sorted(set(_PROMPTS) - set(_WRITING))


def test_scope_split_is_not_vacuous_on_either_side():
    assert _WRITING, f'no discovered prompt lists {_WRITE_TOOL}: {sorted(_PROMPTS)}'
    assert _NOT_WRITING, f'every discovered prompt lists {_WRITE_TOOL}: {sorted(_PROMPTS)}'


@pytest.mark.parametrize('name', _WRITING)
def test_every_prompt_that_can_write_carries_the_guidance(name):
    assert render_referent_declaration_guidance() in _PROMPTS[name], (
        f'{name} lists {_WRITE_TOOL} but does not carry the rendered referent '
        'declaration guidance verbatim. Interpolate the prompts package constant '
        'bound from render_referent_declaration_guidance(); do not restate it.'
    )


@pytest.mark.parametrize('name', _NOT_WRITING)
def test_no_prompt_that_cannot_write_carries_the_guidance(name):
    assert render_referent_declaration_guidance() not in _PROMPTS[name], (
        f'{name} carries write-time referent guidance, but its '
        f'{_TOOL_LISTING_HEADING.strip()!r} section does not name {_WRITE_TOOL}.'
    )
