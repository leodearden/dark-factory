"""Drift pin: the referent-declaration fragment reaches every recon prompt that writes.

Task 3675. ``fused_memory.utils.referent_resolution::render_referent_declaration_guidance``
is the one source of the ``entities`` teaching. The recon stage prompts
interpolate it into their f-strings, so ``render_*() in <prompt>`` is a
byte-identical containment pin, the shape ``test_standing_decision_prompt_drift.py``
uses. A re-hardcoded or reworded copy fails here, while rewording the source
never does.

Which prompts can write is decided by ``fused-memory/tests/_recon_prompt_write_scope.py``.

Whether the fragment's worked example is one the live gate accepts is certified
separately, in ``test_referent_declaration_examples.py``.
"""

from __future__ import annotations

import pytest
from _recon_prompt_write_scope import (
    TOOL_LISTING_HEADING,
    WRITE_TOOL,
    recon_system_prompts,
    split_by_write_grant,
)

from fused_memory.utils.referent_resolution import render_referent_declaration_guidance

_PROMPTS = recon_system_prompts()
_WRITING, _NOT_WRITING = split_by_write_grant(_PROMPTS)


def test_scope_split_is_not_vacuous_on_either_side():
    assert _WRITING, f'no discovered prompt lists {WRITE_TOOL}: {sorted(_PROMPTS)}'
    assert _NOT_WRITING, f'every discovered prompt lists {WRITE_TOOL}: {sorted(_PROMPTS)}'


@pytest.mark.parametrize('name', _WRITING)
def test_every_prompt_that_can_write_carries_the_guidance(name):
    assert render_referent_declaration_guidance() in _PROMPTS[name], (
        f'{name} lists {WRITE_TOOL} but does not carry the rendered referent '
        'declaration guidance verbatim. Interpolate the prompts package constant '
        'bound from render_referent_declaration_guidance(); do not restate it.'
    )


@pytest.mark.parametrize('name', _NOT_WRITING)
def test_no_prompt_that_cannot_write_carries_the_guidance(name):
    assert render_referent_declaration_guidance() not in _PROMPTS[name], (
        f'{name} carries write-time referent guidance, but its '
        f'{TOOL_LISTING_HEADING.strip()!r} section does not name {WRITE_TOOL}.'
    )
