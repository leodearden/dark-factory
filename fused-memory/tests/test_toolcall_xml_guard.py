"""Tests for ``fused_memory.middleware.toolcall_xml_guard`` (task 3083).

The pre-existing detector (task 2939) has one fatal weakness: it is wired
nowhere. It reports leaks out-of-band, after the corrupted text has already
been persisted. This guard is the live rejection at the fused-memory MCP
write boundary, mirroring ``middleware/premise_lint_guard.py``'s
``-> dict | None`` contract and its per-field-independence invariant.

Two design commitments are pinned here as behaviour, not just prose:

* It REJECTS rather than sanitizes. Silently stripping the fragment would be
  a SECOND silent mutation of user content layered on the first — the exact
  failure class this task exists to kill. Rejection also carries information
  the caller needs, so the message must say that SIBLING parameters (e.g.
  ``priority``) may have been silently dropped by the harness. That sentence
  is the root-cause payoff: it converts the invisible sibling-argument-loss
  vector into a visible one.
* It is NOT scoped to recon callers the way ``premise_lint_error`` is. The
  harness leak is caller-agnostic, so scoping it would leave every
  non-recon write path unguarded.

SENTINEL-LITERAL HAZARD (plan pre-1): every sentinel is spelled with the
``\\x3c`` escape for ``<``; writing one verbatim would reproduce the bug
under repair inside this file's own authoring tool call.
"""
from __future__ import annotations

import pytest

from fused_memory.middleware.toolcall_xml_guard import toolcall_xml_error

_CLOSE_DESCRIPTION = '\x3c/description>'
_CLOSE_CONTENT = '\x3c/content>'
_CLOSE_INVOKE = '\x3c/invoke>'
_OPEN_PRIORITY = '\x3cparameter name="priority">'

# The task 992 live shape: a body, a stray closing tag, a real newline, and a
# leaked `priority` argument that never reached the MCP boundary.
LEAK_FRAGMENT = _CLOSE_DESCRIPTION + '\n' + _OPEN_PRIORITY + 'low'
DIRTY_TEXT = 'Add a regression test for the truncation path.' + LEAK_FRAGMENT

# The c759c53b Mem0 shape: a bare closing invoke tag as the continuation.
TAIL_FRAGMENT = _CLOSE_CONTENT + '\n' + _CLOSE_INVOKE
DIRTY_CONTENT = 'A memory about the reconciliation grace window.' + TAIL_FRAGMENT

CLEAN_TEXT = 'An ordinary task description mentioning priority and content.'


class TestCleanInputPassesThrough:
    def test_all_clean_fields_return_none(self) -> None:
        assert toolcall_xml_error(
            title='A clean title',
            description=CLEAN_TEXT,
            prompt='Do the thing.',
            details='Some implementation notes.',
        ) is None

    def test_no_fields_at_all_returns_none(self) -> None:
        assert toolcall_xml_error() is None

    def test_none_and_empty_fields_return_none(self) -> None:
        assert toolcall_xml_error(title=None, description='', details=None, prompt='') is None

    def test_non_str_fields_do_not_raise(self) -> None:
        assert toolcall_xml_error(title=7, description={'a': 1}, details=['x']) is None

    def test_prose_quoting_the_escaped_newline_is_not_rejected(self) -> None:
        """The tasks 2938/2939 false-positive shape must still pass."""
        quoted = 'The leak looks like `' + _CLOSE_DESCRIPTION + '\\n' + _OPEN_PRIORITY + 'low`.'
        assert toolcall_xml_error(description=quoted) is None


class TestRejection:
    def test_dirty_field_returns_the_structured_error(self) -> None:
        err = toolcall_xml_error(description=DIRTY_TEXT)

        assert err is not None
        assert err['error_type'] == 'ToolCallXmlLeakError'

    def test_error_names_the_offending_field(self) -> None:
        err = toolcall_xml_error(title='clean', description=DIRTY_TEXT, details='clean')

        assert err is not None
        assert err['field'] == 'description'
        assert err['fields'] == ['description']
        assert 'description' in err['error']

    def test_error_carries_the_matched_fragment_and_offset(self) -> None:
        err = toolcall_xml_error(description=DIRTY_TEXT)

        assert err is not None
        assert err['fragment'] == LEAK_FRAGMENT
        assert err['offset'] == DIRTY_TEXT.index(LEAK_FRAGMENT)

    def test_error_warns_that_sibling_parameters_may_be_lost(self) -> None:
        """The root-cause payoff — this sentence makes the invisible vector
        visible. A leak in `description` is positive evidence that arguments
        which FOLLOWED it never reached the boundary at all."""
        err = toolcall_xml_error(description=DIRTY_TEXT)

        assert err is not None
        message = err['error'] + ' ' + err['hint']
        assert 'priority' in message
        assert 'sibling' in message.lower()
        assert 'silently' in message.lower()

    def test_hint_documents_the_opt_out(self) -> None:
        err = toolcall_xml_error(description=DIRTY_TEXT)

        assert err is not None
        assert 'allow_toolcall_xml' in err['hint']

    def test_content_field_tail_shape_is_rejected(self) -> None:
        err = toolcall_xml_error(content=DIRTY_CONTENT)

        assert err is not None
        assert err['field'] == 'content'
        assert err['fragment'] == TAIL_FRAGMENT

    def test_every_offending_field_is_reported(self) -> None:
        err = toolcall_xml_error(title=DIRTY_TEXT, description='clean', details=DIRTY_TEXT)

        assert err is not None
        assert err['fields'] == ['title', 'details']
        assert err['field'] == 'title'

    @pytest.mark.parametrize('field', ['title', 'description', 'prompt', 'details', 'content'])
    def test_each_field_is_guarded(self, field: str) -> None:
        err = toolcall_xml_error(**{field: DIRTY_TEXT})

        assert err is not None, f'{field} must be guarded'
        assert err['field'] == field


class TestFieldsAreScannedIndependently:
    """A leak must be wholly contained in one field.

    Mirrors premise_lint_guard's documented cross-field invariant: scanning a
    concatenation would let a match be assembled by straddling the join
    between two otherwise-clean fields, rejecting a legitimate write.
    """

    def test_a_leak_straddling_two_clean_fields_does_not_match(self) -> None:
        # `description` ends at the closing tag; `details` opens with the
        # whitespace + continuation. Concatenated they would match; separately
        # neither does.
        first = 'Body prose that ends here.' + _CLOSE_DESCRIPTION
        second = '\n' + _OPEN_PRIORITY + 'low'

        assert toolcall_xml_error(description=first, details=second) is None

    def test_the_halves_really_would_match_when_concatenated(self) -> None:
        """Guards the test above from decaying into a tautology."""
        first = 'Body prose that ends here.' + _CLOSE_DESCRIPTION
        second = '\n' + _OPEN_PRIORITY + 'low'

        assert toolcall_xml_error(description=first + second) is not None


class TestNotScopedToReconCallers:
    """Unlike premise_lint_error, this guard fires for every caller."""

    @pytest.mark.parametrize('agent_id', [None, 'claude-interactive', 'recon-stage-1', '', 42])
    def test_fires_identically_regardless_of_agent_id(self, agent_id: object) -> None:
        err = toolcall_xml_error(agent_id=agent_id, description=DIRTY_TEXT)

        assert err is not None
        assert err['error_type'] == 'ToolCallXmlLeakError'
        assert err['field'] == 'description'

    @pytest.mark.parametrize('agent_id', [None, 'claude-interactive', 'recon-stage-1'])
    def test_clean_text_passes_regardless_of_agent_id(self, agent_id: object) -> None:
        assert toolcall_xml_error(agent_id=agent_id, description=CLEAN_TEXT) is None

    def test_agent_id_is_not_itself_scanned_as_a_field(self) -> None:
        # It is an identity, not caller content: a leak-shaped agent_id must
        # not be reported as an offending text field.
        assert toolcall_xml_error(agent_id=DIRTY_TEXT, description=CLEAN_TEXT) is None


class TestOptOut:
    def test_allow_toolcall_xml_true_suppresses_the_rejection(self) -> None:
        """Load-bearing, not decorative: this task's own description quotes
        every sentinel verbatim and would otherwise be unfileable, as would
        any memory documenting the leak."""
        assert toolcall_xml_error(
            metadata={'allow_toolcall_xml': True}, description=DIRTY_TEXT
        ) is None

    @pytest.mark.parametrize(
        'metadata',
        [
            {},
            {'allow_toolcall_xml': False},
            {'allow_toolcall_xml': None},
            {'allow_toolcall_xml': 0},
            {'allow_near_duplicate': True},
            {'other': 'value'},
        ],
    )
    def test_absent_or_falsy_flag_does_not_suppress(self, metadata: dict) -> None:
        err = toolcall_xml_error(metadata=metadata, description=DIRTY_TEXT)

        assert err is not None
        assert err['error_type'] == 'ToolCallXmlLeakError'

    @pytest.mark.parametrize('truthy', ['true', 'yes', 1, ['x']])
    def test_only_the_literal_true_opts_out(self, truthy: object) -> None:
        """Mirrors the established `allow_near_duplicate` strictness
        (`metadata.get(...) is True`), so a stringly-typed value cannot
        accidentally disable the guard."""
        assert toolcall_xml_error(
            metadata={'allow_toolcall_xml': truthy}, description=DIRTY_TEXT
        ) is not None

    @pytest.mark.parametrize(
        'metadata', [None, 'a json string', 42, ['allow_toolcall_xml'], object()]
    )
    def test_non_dict_metadata_does_not_raise(self, metadata: object) -> None:
        """update_task json.dumps-es metadata before persisting, and callers
        may pass it through in either shape — the guard must tolerate both."""
        err = toolcall_xml_error(metadata=metadata, description=DIRTY_TEXT)

        assert err is not None
        assert err['error_type'] == 'ToolCallXmlLeakError'

    def test_non_dict_metadata_on_clean_input_is_still_none(self) -> None:
        assert toolcall_xml_error(metadata='a json string', description=CLEAN_TEXT) is None
