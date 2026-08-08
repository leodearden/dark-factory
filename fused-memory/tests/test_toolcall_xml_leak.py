"""Tests for ``fused_memory.utils.toolcall_xml_leak`` — the SHARED detector
for leaked serialized tool-call XML fragments.

Task 3083. This module is the promotion of
``scripts/scan_task_toolcall_leaks.py``'s ``LEAK_TAIL``/``detect_leak``
(task 2939) into an importable home so that the task-DB scanner, the live
write-boundary guard, and the Mem0 corpus sweep all share ONE definition of
"what a leak looks like". A second, independently-written regex would drift
immediately and re-open the exact ambiguity this task exists to close.

Two classes of assertion live here:

1. BACKWARD COMPATIBILITY — the four live fixtures from
   ``scripts/tests/test_scan_task_toolcall_leaks.py`` (originating tasks
   992, 1068, 1067, 2691) are still detected, and the ``\\s+`` real-whitespace
   discriminator that excludes prose quoting the ESCAPED literal backslash-n
   (the tasks 2938/2939 false-positive shape) is preserved EXACTLY. That
   conservatism is the reason the detector is trustworthy; a naive substring
   scan over-reports on precisely that shape.

2. NEW MEM0 SHAPES — the two 2026-07-27 Mem0 specimens (9f2d2ae6, c759c53b)
   which the task-DB-era regex misses: it lacks ``content`` in the
   closing-tag alternation, and it requires an opening-parameter
   continuation, whereas c759c53b's tail is a closing content tag, a real
   newline, and a bare closing invoke tag with nothing after it.

SENTINEL-LITERAL HAZARD (plan pre-1): every tool-call sentinel below is
spelled with the ``\\x3c`` escape for ``<``. Writing one verbatim into this
file would force the authoring agent to emit that literal inside its own
tool-call XML — reproducing the exact bug under repair and truncating the
file being written. The escaped form is byte-identical at runtime.
"""
from __future__ import annotations

import re

import pytest
from shared import toolcall_markup

from fused_memory.utils import toolcall_xml_leak
from fused_memory.utils.toolcall_xml_leak import (
    LEAK_TAIL,
    LeakHit,
    detect_leak,
    find_toolcall_xml_leak,
    has_toolcall_xml_leak,
)

# ---------------------------------------------------------------------------
# Sentinel literals, built with the \x3c escape (see module docstring).
# ---------------------------------------------------------------------------

_CLOSE_DESCRIPTION = '\x3c/description>'
_CLOSE_PARAMETER = '\x3c/parameter>'
_CLOSE_DETAILS = '\x3c/details>'
_CLOSE_CONTENT = '\x3c/content>'
_CLOSE_INVOKE = '\x3c/invoke>'


def _open_param(name: str) -> str:
    """Return a serialized opening parameter tag for *name*."""
    return '\x3cparameter name="' + name + '">'


# ---------------------------------------------------------------------------
# (A) Backward-compatibility fixtures — copied from
# scripts/tests/test_scan_task_toolcall_leaks.py:59,67,74,83-90, which models
# them on real live-DB leak shapes (tasks 992, 1068, 1067, 2691) rather than
# invented ones. THREE of the four are a leaked `priority` parameter — direct
# corroboration of the sibling-argument-loss vector.
# ---------------------------------------------------------------------------

GENUINE_DESCRIPTION_LEAK_FRAGMENT = _CLOSE_DESCRIPTION + '\n' + _open_param('priority') + 'low'
GENUINE_DESCRIPTION_LEAK = (
    'Add a new test that writes >limit records and asserts only the newest '
    '`limit` are materialised.'
) + GENUINE_DESCRIPTION_LEAK_FRAGMENT

CHAINED_PARAMETER_LEAK_FRAGMENT = _CLOSE_PARAMETER + '\n' + _open_param('priority') + 'high'
CHAINED_PARAMETER_LEAK = (
    'Root cause is documented in a code comment; a regression test exercises '
    'the orphaned-commit path.'
) + CHAINED_PARAMETER_LEAK_FRAGMENT

DETAILS_LEAK_FRAGMENT = _CLOSE_DETAILS + '\n' + _open_param('priority') + 'polish'
DETAILS_LEAK_TEXT = (
    'Verify with `pytest fused-memory/tests/test_bulk_reset_guard.py -q`.'
) + DETAILS_LEAK_FRAGMENT

SWALLOWED_DETAILS_FRAGMENT = (
    _CLOSE_DESCRIPTION + '\n' + _open_param('details') + 'Run fused-memory/scripts/'
    'audit_duplicate_memories.py or a direct Qdrant admin query against '
    'memory_id=028edb1f-299c-4755-9438-deadbeefcafe (null category, '
    'unretrievable content matching the fingerprint) to confirm-safe the '
    'deletion.\n\nClose out this one-off cleanup once resolved so it stops '
    'recurring in Stage 1/2 payloads every cycle.'
)
SWALLOWED_DETAILS_LEAK = (
    'The dashboard shows an orphaned memory row with a null agent_id and '
    'unretrievable content matching the fingerprint.'
) + SWALLOWED_DETAILS_FRAGMENT

_LIVE_FIXTURES = {
    'task_992_description': (GENUINE_DESCRIPTION_LEAK, GENUINE_DESCRIPTION_LEAK_FRAGMENT),
    'task_1068_chained_parameter': (CHAINED_PARAMETER_LEAK, CHAINED_PARAMETER_LEAK_FRAGMENT),
    'task_1067_details': (DETAILS_LEAK_TEXT, DETAILS_LEAK_FRAGMENT),
    'task_2691_swallowed_details': (SWALLOWED_DETAILS_LEAK, SWALLOWED_DETAILS_FRAGMENT),
}

# ---------------------------------------------------------------------------
# (B) False-positive fixtures — prose that MENTIONS the fragment, quoting the
# ESCAPED literal backslash-n (two real characters, "\" then "n" — NOT a real
# whitespace character). Exact tasks 2938/2939 shape.
# ---------------------------------------------------------------------------

PROSE_MENTION_FALSE_POSITIVE = (
    "Stage 1 found that task 2865's description had a leaked plain-text "
    'tool-call/XML fragment appended: `' + _CLOSE_DESCRIPTION + '\\n' + _open_param('priority')
    + 'low`. Stage 2 (this run) verified the fragment '
    'verbatim via live get_task and stripped it from the description via '
    'update_task.'
)

# A closing tag with the continuation IMMEDIATELY adjacent — zero whitespace
# in between. The \s+ requirement means this is not a leak.
ZERO_WHITESPACE_ADJACENT = (
    'Some ordinary body text.' + _CLOSE_DESCRIPTION + _open_param('priority') + 'low'
)

# ---------------------------------------------------------------------------
# (C) New Mem0 specimen shapes (2026-07-27), which the task-DB-era regex
# cannot see.
# ---------------------------------------------------------------------------

_MEM0_BODY = (
    'The reconciliation Stage 2 judge treats an empty backlog as a halt '
    'candidate only after the grace window elapses.'
)

# Shape 9f2d2ae6 — closing content tag, real newline, an opening `content`
# parameter, then a VERBATIM duplicate of the same body.
MEM0_DUPLICATE_FRAGMENT = (
    _CLOSE_CONTENT + '\n' + _open_param('content') + _MEM0_BODY
)
MEM0_DUPLICATE_LEAK = _MEM0_BODY + MEM0_DUPLICATE_FRAGMENT

# Shape c759c53b — closing content tag, real newline, bare closing invoke tag,
# with NO opening-parameter continuation at all. The task-DB-era LEAK_TAIL
# misses this on BOTH counts (no `content` in the alternation, and it demands
# an opening parameter tag after the whitespace).
MEM0_TAIL_FRAGMENT = _CLOSE_CONTENT + '\n' + _CLOSE_INVOKE
MEM0_TAIL_LEAK = _MEM0_BODY + MEM0_TAIL_FRAGMENT


# ---------------------------------------------------------------------------
# (A) Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibilityWithLiveFixtures:
    """The four live task-DB fixtures must survive the promotion unchanged."""

    @pytest.mark.parametrize('name', sorted(_LIVE_FIXTURES))
    def test_live_fixture_is_detected(self, name: str) -> None:
        text, _fragment = _LIVE_FIXTURES[name]
        assert has_toolcall_xml_leak(text) is True
        assert find_toolcall_xml_leak(text), f'{name} must be detected'

    @pytest.mark.parametrize('name', sorted(_LIVE_FIXTURES))
    def test_live_fixture_reports_the_exact_fragment(self, name: str) -> None:
        text, fragment = _LIVE_FIXTURES[name]
        hits = find_toolcall_xml_leak(text)
        assert len(hits) == 1
        assert hits[0].fragment == fragment

    @pytest.mark.parametrize('name', sorted(_LIVE_FIXTURES))
    def test_detect_leak_shim_preserves_the_task_2939_contract(self, name: str) -> None:
        text, fragment = _LIVE_FIXTURES[name]
        assert detect_leak(text) == fragment

    def test_hit_index_points_at_the_start_of_the_fragment(self) -> None:
        text, fragment = _LIVE_FIXTURES['task_992_description']
        hits = find_toolcall_xml_leak(text)
        assert hits[0].index == text.index(fragment)
        assert text[hits[0].index:] == fragment

    def test_trailing_whitespace_does_not_defeat_detection(self) -> None:
        text, fragment = _LIVE_FIXTURES['task_1067_details']
        assert detect_leak(text + '  \n\n') == fragment


# ---------------------------------------------------------------------------
# (B) The \s+ discriminator
# ---------------------------------------------------------------------------


class TestRealWhitespaceDiscriminatorIsPreserved:
    """The hard-won tasks 2938/2939 conservatism must carry across verbatim."""

    def test_prose_quoting_the_escaped_newline_is_not_a_leak(self) -> None:
        assert find_toolcall_xml_leak(PROSE_MENTION_FALSE_POSITIVE) == []
        assert has_toolcall_xml_leak(PROSE_MENTION_FALSE_POSITIVE) is False
        assert detect_leak(PROSE_MENTION_FALSE_POSITIVE) is None

    def test_zero_whitespace_between_tag_and_continuation_is_not_a_leak(self) -> None:
        assert find_toolcall_xml_leak(ZERO_WHITESPACE_ADJACENT) == []
        assert detect_leak(ZERO_WHITESPACE_ADJACENT) is None

    def test_escaped_newline_before_a_bare_closing_invoke_is_not_a_leak(self) -> None:
        # Same discriminator, applied to the newly-added continuation branch:
        # generalizing the regex must not weaken the whitespace requirement.
        text = _MEM0_BODY + _CLOSE_CONTENT + '\\n' + _CLOSE_INVOKE
        assert find_toolcall_xml_leak(text) == []

    def test_zero_whitespace_before_a_bare_closing_invoke_is_not_a_leak(self) -> None:
        text = _MEM0_BODY + _CLOSE_CONTENT + _CLOSE_INVOKE
        assert find_toolcall_xml_leak(text) == []


# ---------------------------------------------------------------------------
# (C) New Mem0 shapes
# ---------------------------------------------------------------------------


class TestMem0SpecimenShapes:
    """The 2026-07-27 Mem0 specimens the task-DB-era regex could not see."""

    def test_content_self_duplicate_shape_is_detected(self) -> None:
        hits = find_toolcall_xml_leak(MEM0_DUPLICATE_LEAK)
        assert len(hits) == 1
        assert hits[0].fragment == MEM0_DUPLICATE_FRAGMENT
        assert hits[0].index == len(_MEM0_BODY)

    def test_bare_closing_invoke_tail_shape_is_detected(self) -> None:
        hits = find_toolcall_xml_leak(MEM0_TAIL_LEAK)
        assert len(hits) == 1
        assert hits[0].fragment == MEM0_TAIL_FRAGMENT

    def test_closing_content_tag_is_in_the_alternation(self) -> None:
        # The task-DB-era alternation was description|parameter|details only.
        text = 'body text' + _CLOSE_CONTENT + '\n' + _open_param('priority') + 'high'
        assert has_toolcall_xml_leak(text) is True

    @pytest.mark.parametrize(
        'closing_tag',
        [_CLOSE_DESCRIPTION, _CLOSE_PARAMETER, _CLOSE_DETAILS, _CLOSE_CONTENT],
    )
    def test_every_closing_tag_pairs_with_a_bare_closing_invoke(self, closing_tag: str) -> None:
        assert has_toolcall_xml_leak('body text' + closing_tag + '\n' + _CLOSE_INVOKE) is True


# ---------------------------------------------------------------------------
# (D) API surface
# ---------------------------------------------------------------------------


class TestPublicApi:
    def test_leak_hit_carries_fragment_and_index(self) -> None:
        hit = find_toolcall_xml_leak(MEM0_TAIL_LEAK)[0]
        assert isinstance(hit, LeakHit)
        assert hit.fragment == MEM0_TAIL_FRAGMENT
        assert hit.index == len(_MEM0_BODY)

    def test_hits_are_ordered_by_index(self) -> None:
        hits = find_toolcall_xml_leak(MEM0_DUPLICATE_LEAK)
        assert [h.index for h in hits] == sorted(h.index for h in hits)

    def test_has_toolcall_xml_leak_mirrors_find(self) -> None:
        for text in (MEM0_TAIL_LEAK, GENUINE_DESCRIPTION_LEAK, 'clean prose', ''):
            assert has_toolcall_xml_leak(text) is bool(find_toolcall_xml_leak(text))

    def test_detect_leak_mirrors_the_first_hit(self) -> None:
        for text in (MEM0_TAIL_LEAK, DETAILS_LEAK_TEXT, 'clean prose'):
            hits = find_toolcall_xml_leak(text)
            assert detect_leak(text) == (hits[0].fragment if hits else None)

    def test_leak_tail_is_a_compiled_pattern(self) -> None:
        assert LEAK_TAIL.search(MEM0_TAIL_LEAK) is not None
        assert LEAK_TAIL.search('clean prose') is None


# ---------------------------------------------------------------------------
# (E) Degenerate input
# ---------------------------------------------------------------------------


class TestDegenerateInput:
    @pytest.mark.parametrize('value', [None, '', 0, 12, 3.5, {'a': 1}, ['x'], b'bytes', True])
    def test_non_str_and_empty_input_never_raises(self, value: object) -> None:
        assert find_toolcall_xml_leak(value) == []
        assert has_toolcall_xml_leak(value) is False
        assert detect_leak(value) is None

    def test_whitespace_only_input_is_clean(self) -> None:
        assert find_toolcall_xml_leak('   \n\t ') == []


# ---------------------------------------------------------------------------
# (F) Clean prose and near misses
# ---------------------------------------------------------------------------


class TestCleanTextAndNearMisses:
    @pytest.mark.parametrize(
        'text',
        [
            'the parameter name is foo',
            'invoke the handler and return',
            'content ends here',
            'A description of the details of the content.',
            'Use </> as a shorthand in the docs.',
            'priority: low',
            'A multi-line\nbody with\nreal newlines but no tags at all.',
        ],
    )
    def test_ordinary_text_is_clean(self, text: str) -> None:
        assert find_toolcall_xml_leak(text) == []
        assert detect_leak(text) is None

    def test_a_closing_tag_alone_is_not_a_leak(self) -> None:
        # Without a continuation after real whitespace there is no evidence of
        # early argument termination — the discriminator requires both halves.
        assert find_toolcall_xml_leak('body text' + _CLOSE_CONTENT) == []
        assert find_toolcall_xml_leak('body text' + _CLOSE_CONTENT + '\n') == []

    def test_a_continuation_alone_is_not_a_leak(self) -> None:
        assert find_toolcall_xml_leak('body\n' + _open_param('priority') + 'low') == []
        assert find_toolcall_xml_leak('body\n' + _CLOSE_INVOKE) == []


# ---------------------------------------------------------------------------
# (G) INV-5 — the enumeration lives in shared.toolcall_markup, and moving it
#     there changed no behaviour.
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    def test_prefilter_needles_is_the_shared_object(self) -> None:
        """IDENTITY, not equality — a copied tuple would re-open the drift.

        ``==`` is satisfied by a tuple re-spelled here with the same value, and
        a re-spelled tuple is exactly how this list and ``markup_tripwire``'s
        write-time list drifted apart while both were documented as the single
        source of truth. Identity is the one assertion a duplicate cannot pass.
        """
        assert toolcall_xml_leak.PREFILTER_NEEDLES is toolcall_markup.PREFILTER_NEEDLES

    def test_prefilter_needles_order_is_load_bearing(self) -> None:
        """``tests/test_mem0_client.py`` zips this against the Qdrant clauses
        with ``strict=True``, so the ORDER is part of the contract, not just the
        membership. Pinned from the consumer side because that is where a
        reorder would actually break.
        """
        expected = tuple(
            toolcall_markup.closer_for(name)
            for name in toolcall_markup.PARAMETER_CLOSER_NAMES
        )
        assert list(toolcall_xml_leak.PREFILTER_NEEDLES) == list(expected)

    def test_leak_tail_pattern_string_is_unchanged(self) -> None:
        """BEHAVIOUR PRESERVED: the regex is now BUILT from the shared name
        tuple rather than re-spelled, which would otherwise make it a third
        enumeration site. Pinning the compiled pattern string proves the
        rewrite is byte-for-byte the same detector — the only evidence that
        matters for a consolidation that must change nothing.
        """
        assert toolcall_xml_leak.LEAK_TAIL.pattern == (
            r'\x3c/(?:description|parameter|details|content)>'
            r'\s+'
            r'(\x3c(?:parameter\s+name="[^"]*">|/invoke>).*)$'
        )
        assert toolcall_xml_leak.LEAK_TAIL.flags & re.DOTALL

    def test_leak_tail_stays_a_single_compiled_object(self) -> None:
        """``scripts/tests/test_scan_task_toolcall_leaks.py`` asserts
        ``scan_task_toolcall_leaks.LEAK_TAIL is toolcall_xml_leak.LEAK_TAIL``.
        Rebuilding the pattern behind a module ``__getattr__`` or a property
        would satisfy every value assertion above and still break that `is`.
        """
        assert isinstance(toolcall_xml_leak.LEAK_TAIL, re.Pattern)
        assert toolcall_xml_leak.LEAK_TAIL is toolcall_xml_leak.LEAK_TAIL
