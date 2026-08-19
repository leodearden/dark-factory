"""Drift pin: every metadata example embedded in a PROMPT must agree with the
registry that defines the vocabulary (task 3202, PRD
``docs/prds/memory-metadata-vocabulary.md`` leaf ι, invariant INV-5 — one
normative copy, everything else a pointer).

``fused_memory.memory_metadata`` (leaf β / task 3195) is that normative copy.
Two prompt surfaces quote it, and both are pinned here:

ARM 1 — the writer instructions.  ``METADATA_VOCABULARY_INSTRUCTIONS`` in
``orchestrator/src/orchestrator/agents/roles.py`` documents the reserved keys to
every memory-writing agent role.  Adding a key to ``RESERVED_VOCABULARY_KEYS``
without documenting it, documenting a key that is not reserved, or renaming
either side, all fail here.

ARM 2 — the reconciliation stage prompts, whose
``'kind': '<literal>'`` filter examples must name kinds that are in
``KIND_REGISTRY``.

WHY THIS FILE LIVES IN fused-memory/tests: this suite's pytest config already
declares ``pythonpath = ["src", "../orchestrator/src"]``, so BOTH
``fused_memory.memory_metadata`` and ``orchestrator.agents.roles`` are hard
imports here.  The orchestrator suite cannot import ``fused_memory``, and a
guard written there would need a ``pytest.importorskip`` — i.e. a drift test
that silently becomes a no-op the day the path breaks.  The orchestrator side
keeps only its own presence/brace/role-split anchor
(``orchestrator/tests/test_roles_metadata_vocabulary.py``).

The registry-facing assertions are deliberately DERIVED, never hardcoded: no
test here restates the five key names or the slug regex, because that would be
the very second copy INV-5 forbids.
"""

from __future__ import annotations

import re

import pytest
from orchestrator.agents.roles import METADATA_VOCABULARY_INSTRUCTIONS

from fused_memory.memory_metadata import (
    EXPERIMENTAL_KEY_PREFIX,
    KIND_REGISTRY,
    RESERVED_VOCABULARY_KEYS,
    TOPIC_SLUG_MAX_LEN,
    is_valid_topic_slug,
)
from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)
from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT
from fused_memory.reconciliation.recon_pool_map import CYCLE_SUMMARY_KIND
from fused_memory.reconciliation.recon_self_model import render_cycle_summary_section

# The documented-key bullet shape the writer instructions use:
#     - `topic` — kebab-case slug naming ...
# `[^`]+` rather than a character class of legal key names, so a RENAMED key is
# still extracted (and then fails the comparison) instead of vanishing from the
# extracted set and passing as "not documented, not required".
_KEY_BULLET_RE = re.compile(r'^- `([^`]+)` —', re.MULTILINE)

# Slug exemplars advertised to writers. Pinned SEMANTICALLY: the test asserts
# both strings appear in the instruction text AND that `is_valid_topic_slug`
# still agrees about them, so a change to TOPIC_SLUG_RE that invalidates the
# advertised example fails here. Pinning `TOPIC_SLUG_RE.pattern` as a string
# instead would force the literal `\Z` into agent-facing prose.
_GOOD_SLUG = 'memory-write-path'
_BAD_SLUG = 'Memory Write Path'


def _documented_vocabulary_keys(text: str) -> set[str]:
    """Extract the metadata keys documented as bullets in `text`."""
    return set(_KEY_BULLET_RE.findall(text))


def _pin_keys(text: str, registry: set[str]) -> set[str]:
    """Assert `text` documents exactly `registry`; raise AssertionError if not.

    Factored out of the tests so the DRIFT DEMONSTRATION below can run the very
    same checker against a doctored registry — proving the pin actually fires
    on divergence rather than merely passing today.
    """
    documented = _documented_vocabulary_keys(text)
    expected = set(registry)
    missing = expected - documented
    undocumented_extras = documented - expected
    assert not missing and not undocumented_extras, (
        f'writer instructions drifted from the registry: '
        f'reserved-but-undocumented={sorted(missing)}, '
        f'documented-but-not-reserved={sorted(undocumented_extras)}'
    )
    return documented


class TestReservedKeysPinnedToRegistry:
    """ARM 1: the documented keys ARE `RESERVED_VOCABULARY_KEYS`."""

    def test_documented_keys_match_registry(self) -> None:
        assert _documented_vocabulary_keys(METADATA_VOCABULARY_INSTRUCTIONS) == set(
            RESERVED_VOCABULARY_KEYS
        )

    def test_extraction_is_non_empty(self) -> None:
        """Guards the regex itself: if the bullet shape changes, the extractor
        silently matches nothing and every set comparison above would pass
        vacuously against an empty registry."""
        assert _documented_vocabulary_keys(METADATA_VOCABULARY_INSTRUCTIONS)

    def test_pin_helper_passes_against_the_real_registry(self) -> None:
        assert _pin_keys(METADATA_VOCABULARY_INSTRUCTIONS, set(RESERVED_VOCABULARY_KEYS))


class TestEscapeHatchAndSlugRulesPinned:
    """The non-key parts of the vocabulary the instructions also advertise."""

    def test_experimental_prefix_is_verbatim(self) -> None:
        """The `x_` escape agents are told to use is the registry's actual
        prefix, not a lookalike."""
        assert EXPERIMENTAL_KEY_PREFIX in METADATA_VOCABULARY_INSTRUCTIONS

    def test_slug_exemplars_are_present_and_semantically_correct(self) -> None:
        """Both exemplars appear verbatim in the prose AND the validator still
        agrees with what the prose claims about them."""
        assert _GOOD_SLUG in METADATA_VOCABULARY_INSTRUCTIONS
        assert _BAD_SLUG in METADATA_VOCABULARY_INSTRUCTIONS
        assert is_valid_topic_slug(_GOOD_SLUG) is True
        assert is_valid_topic_slug(_BAD_SLUG) is False

    def test_slug_length_cap_is_verbatim(self) -> None:
        """A plain integer is safe to string-pin (unlike the regex)."""
        assert str(TOPIC_SLUG_MAX_LEN) in METADATA_VOCABULARY_INSTRUCTIONS


class TestPinFiresOnDrift:
    """DRIFT DEMONSTRATION — the user-observable signal this task delivers.

    Same checker, doctored registry: if the pin could not fail, every assertion
    above would be decorative.
    """

    def test_added_registry_key_fails_the_pin(self) -> None:
        with pytest.raises(AssertionError):
            _pin_keys(
                METADATA_VOCABULARY_INSTRUCTIONS,
                set(RESERVED_VOCABULARY_KEYS) | {'invented_key'},
            )

    def test_removed_registry_key_fails_the_pin(self) -> None:
        with pytest.raises(AssertionError):
            _pin_keys(
                METADATA_VOCABULARY_INSTRUCTIONS,
                set(RESERVED_VOCABULARY_KEYS) - {'supersedes'},
            )


# --------------------------------------------------------------------------
# ARM 2 — recon prompt-embedded `'kind'` filter examples <-> KIND_REGISTRY
# --------------------------------------------------------------------------

# The shape a metadata filter example takes in RENDERED prompt text. The stage
# prompts are f-strings that write `{{ }}` for a literal brace, so by the time
# the constant exists the text carries single braces and ordinary dict syntax.
_KIND_LITERAL_RE = re.compile(r"'kind':\s*'([^']+)'")

# The prompt surfaces that show an agent a `'kind'` filter. Built at import
# time, exactly as the stage prompts themselves are assembled.
_PROMPT_SOURCES = {
    'STAGE1_SYSTEM_PROMPT': STAGE1_SYSTEM_PROMPT,
    'STAGE2_SYSTEM_PROMPT': STAGE2_SYSTEM_PROMPT,
    "build_stage2_system_prompt('dark_factory')": build_stage2_system_prompt('dark_factory'),
    'STAGE3_SYSTEM_PROMPT': STAGE3_SYSTEM_PROMPT,
    'render_cycle_summary_section()': render_cycle_summary_section(),
}

# DELIBERATE NON-ASSERTION — do not "complete" this file by adding
# `set(recon_self_model.MARKER_KINDS) <= KIND_REGISTRY`. MARKER_KINDS contains
# `stage2_persistence_marker`, which is absent from KIND_REGISTRY and CORRECTLY
# so: `_STAGE2_PERSISTENCE_MARKER_SOURCE` (stages/task_knowledge_sync.py) shows
# it is a `metadata.source` value and a SQLite ledger `record_kind` — the
# WRITER-PROVENANCE axis the PRD's V1 explicitly separates from `metadata.kind`.
# That subset relation is a false premise; asserting it would fail on correct
# code and pressure a future reader into polluting the kind registry with a
# provenance token.


def _kind_literals(text: str) -> set[str]:
    """Extract the `'kind': '<value>'` literals a prompt shows to an agent."""
    return set(_KIND_LITERAL_RE.findall(text))


def _pin_kinds(text: str, registry: frozenset[str] | set[str]) -> set[str]:
    """Assert every kind literal in `text` is in `registry`.

    Same factoring rationale as `_pin_keys`: the drift demonstration below runs
    this exact checker against a doctored registry.
    """
    literals = _kind_literals(text)
    assert literals, 'no kind literal extracted — the regex stopped matching, so this guard is a no-op'
    outside = literals - set(registry)
    assert not outside, (
        f'prompt names kind literals that are not in the registry: {sorted(outside)}'
    )
    return literals


class TestReconPromptKindLiteralsPinned:
    """Every `'kind'` filter example a recon prompt shows is a REGISTERED kind.

    These RENDERED-OUTPUT assertions are the INV-5 guard for
    ``render_cycle_summary_section``: if it re-typed the literal instead of
    interpolating ``CYCLE_SUMMARY_KIND``, a rename in ``recon_pool_map`` would
    leave the rendered text on the old value and
    ``test_kind_literals_extracted_and_include_cycle_summary`` would go RED.

    An ``inspect.getsource`` pin on the *identifier text* was removed in review
    (task 3202): it passed for a function that merely MENTIONS the name in a
    comment, and failed correct refactors -- ``import ... as CSK``, a
    module-qualified ``recon_pool_map.CYCLE_SUMMARY_KIND``, or extracting the
    f-string into a helper.  Do NOT re-add it, and do not replace it with a
    stricter source-text regex; assert on rendered output instead.
    """

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_kind_literals_extracted_and_include_cycle_summary(self, source_name: str) -> None:
        """Non-emptiness is the load-bearing half: an extractor that silently
        stops matching would turn the registry check below into a no-op."""
        literals = _kind_literals(_PROMPT_SOURCES[source_name])
        assert literals
        assert CYCLE_SUMMARY_KIND in literals

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_every_kind_literal_is_registered(self, source_name: str) -> None:
        assert _pin_kinds(_PROMPT_SOURCES[source_name], KIND_REGISTRY)

    def test_cycle_summary_constant_is_registered(self) -> None:
        """The constant<->registry edge itself: the shared kind constant the
        writers use must be a member of the closed registry."""
        assert CYCLE_SUMMARY_KIND in KIND_REGISTRY


class TestKindPinFiresOnDrift:
    """DRIFT DEMONSTRATION — a `cycle_summary` rename in the registry provably
    strands every recon filter example, loudly."""

    @pytest.mark.parametrize('source_name', sorted(_PROMPT_SOURCES))
    def test_renamed_kind_fails_the_pin(self, source_name: str) -> None:
        with pytest.raises(AssertionError):
            _pin_kinds(_PROMPT_SOURCES[source_name], KIND_REGISTRY - {CYCLE_SUMMARY_KIND})
