"""Drift pin: every metadata example embedded in a PROMPT must agree with the
registry that defines the vocabulary (task 3202, PRD
``docs/prds/memory-metadata-vocabulary.md`` leaf ι, invariant INV-5 — one
normative copy, everything else a pointer).

``fused_memory.memory_metadata`` (leaf β / task 3195) is that normative copy.
Two prompt surfaces quote it, and both are pinned here:

ARM 1 — the writer instructions. ``METADATA_VOCABULARY_INSTRUCTIONS`` in
``orchestrator/src/orchestrator/agents/roles.py`` documents the reserved keys to
every memory-writing agent role. Adding a key to ``RESERVED_VOCABULARY_KEYS``
without documenting it, or renaming one, fails here.

ARM 2 — the reconciliation stage prompts, whose ``'kind': '<literal>'`` filter
examples must name kinds that are in ``KIND_REGISTRY``, so a kind rename cannot
silently strand recon agents emitting filters that match nothing.

WHY THIS FILE LIVES IN fused-memory/tests: this suite's pytest config already
declares ``pythonpath = ["src", "../orchestrator/src"]``, so BOTH
``fused_memory.memory_metadata`` and ``orchestrator.agents.roles`` are hard
imports here. The orchestrator suite cannot import ``fused_memory``, and a
guard written there would need a ``pytest.importorskip`` — i.e. a drift test
that silently becomes a no-op the day the path breaks.

SCOPE DISCIPLINE (review, task 3202). This file pins REGISTRY-DERIVED facts
only. It does NOT assert on prompt WORDING: no bullet/em-dash/backtick shape,
no verbatim substring of surrounding prose, no non-emptiness of a literal, no
``endswith`` splice-order check. Those all go red on a cosmetic reflow while
catching no functional regression. Reflowing, reordering, or rewording the
vocabulary section is expected to keep this file green — only a genuine
registry-vs-prompt divergence turns it red. Do not add wording assertions, and
do not add tests that exercise a helper defined in this file rather than
production code.

SEQUENCING SEAM — task 3131 (dep-gated behind 3169) rewrites the
write-eagerness prose inside ``_MEMORY_INSTRUCTIONS``. Nothing here pins any of
those sentences, so that edit and this pin cannot collide.
"""

from __future__ import annotations

import re

import pytest
from orchestrator.agents.roles import _MEMORY_INSTRUCTIONS, METADATA_VOCABULARY_INSTRUCTIONS, ROLES

from fused_memory.memory_metadata import (
    KIND_REGISTRY,
    RESERVED_VOCABULARY_KEYS,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_pool_map import CYCLE_SUMMARY_KIND
from fused_memory.reconciliation.recon_self_model import render_cycle_summary_section

# Roles are DERIVED from the splice, never restated as a literal tuple: adding a
# tenth role must not fail a test about the metadata vocabulary.
_MEMORY_ROLES = sorted(
    name for name, role in ROLES.items() if _MEMORY_INSTRUCTIONS in role.system_prompt
)


class TestReservedKeysReachTheWriters:
    """ARM 1: the writer-facing text names every key the write path validates.

    Registry-derived and reflow-proof: each reserved key is looked up as a
    plain substring, so the section can be reworded or relaid-out freely. What
    it does catch is the PRD §6 gap this task closes — a key that the writer
    path enforces but no agent was ever told about.
    """

    @pytest.mark.parametrize('key', sorted(RESERVED_VOCABULARY_KEYS))
    def test_every_reserved_key_is_documented(self, key: str) -> None:
        assert key in METADATA_VOCABULARY_INSTRUCTIONS, (
            f'reserved key {key!r} is validated on write but never named in the '
            f'writer instructions — writers cannot use a key they are not told about'
        )

    @pytest.mark.parametrize('role_name', _MEMORY_ROLES)
    @pytest.mark.parametrize('key', sorted(RESERVED_VOCABULARY_KEYS))
    def test_every_reserved_key_reaches_each_memory_role(self, key: str, role_name: str) -> None:
        """The task's user-observable signal, asserted end-to-end on the
        RENDERED prompt: a memory-writing role's assembled briefing names every
        reserved key (including ``supersedes``)."""
        assert key in ROLES[role_name].system_prompt

    def test_some_role_actually_splices_the_memory_block(self) -> None:
        """Guards the derivation above: if the splice were renamed away, the
        parametrized role checks would vacuously collapse to zero cases."""
        assert _MEMORY_ROLES


_KIND_LITERAL_RE = re.compile(r"'kind':\s*'([^']+)'")

_PROMPT_SOURCES = {
    'STAGE1_SYSTEM_PROMPT': STAGE1_SYSTEM_PROMPT,
    'STAGE2_SYSTEM_PROMPT': STAGE2_SYSTEM_PROMPT,
    "build_stage2_system_prompt('dark_factory')": build_stage2_system_prompt('dark_factory'),
    'STAGE3_SYSTEM_PROMPT': STAGE3_SYSTEM_PROMPT,
    'render_cycle_summary_section()': render_cycle_summary_section(),
}


def _kind_literals(text: str) -> set[str]:
    """Extract the ``'kind': '<value>'`` filter literals a prompt shows an agent.

    This matches the literal dict syntax the agent is being told to emit — not
    prose formatting — which is why it is a fair thing to pin.
    """
    return set(_KIND_LITERAL_RE.findall(text))


class TestReconPromptKindLiteralsPinned:
    """ARM 2: every ``'kind'`` filter example a recon prompt shows is REGISTERED.

    This is the INV-5 guard for ``render_cycle_summary_section``: if it re-typed
    the literal instead of interpolating ``CYCLE_SUMMARY_KIND``, a rename in
    ``recon_pool_map`` would leave the rendered text on the old value and
    strand every recon agent that copied the filter.

    An ``inspect.getsource`` pin on the *identifier text* was removed in review
    (task 3202): it passed for a function that merely MENTIONS the name in a
    comment, and failed correct refactors — ``import ... as CSK``, a
    module-qualified ``recon_pool_map.CYCLE_SUMMARY_KIND``, or extracting the
    f-string into a helper. Do NOT re-add it; assert on rendered output instead.
    """

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_every_kind_literal_is_registered(self, source_name: str) -> None:
        literals = _kind_literals(_PROMPT_SOURCES[source_name])
        assert literals, (
            'no kind literal extracted — the filter-example syntax changed, so '
            'this guard would silently pass against nothing'
        )
        outside = literals - KIND_REGISTRY
        assert not outside, (
            f'{source_name} shows kind literals that are not in KIND_REGISTRY: '
            f'{sorted(outside)} — recon agents copying these filters would match nothing'
        )

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_cycle_summary_survives_rendering(self, source_name: str) -> None:
        """Each pinned surface really does carry the shared constant's value."""
        assert CYCLE_SUMMARY_KIND in _kind_literals(_PROMPT_SOURCES[source_name])

    def test_cycle_summary_constant_is_registered(self) -> None:
        """The constant<->registry edge itself: the shared kind constant the
        writers use must be a member of the closed registry."""
        assert CYCLE_SUMMARY_KIND in KIND_REGISTRY
