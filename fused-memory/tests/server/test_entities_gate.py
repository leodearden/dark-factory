"""Unit tests for the declared-referent write gate (task 3669, PRD leaf delta).

The gate is the POLICY half of the split leaf gamma drew deliberately:
``utils/referent_resolution`` REPORTS a conflict and never rejects, and
:func:`fused_memory.server.entities_gate.entities_gate` decides that a
conflict blocks the write.

Two arms live here; the third (CONFLICT) arrives in the sibling class below
once the gate can express it.  All of them are exercised against the PURE
function rather than through FastMCP, which is the whole reason the gate is a
module-level function instead of a ``create_mcp_server`` closure: it closes
over no server state, so it needs no server to test.

PRD resolved decision 3 — "reject on CONFLICT, never on ABSENCE" — is the
property the absence arm pins.  Rejecting on absence would lose memories from
every agent that does not retry (``/reflect`` at session end is the named
case), so the undeclared majority must flow through untouched.
"""

from __future__ import annotations

import json

import pytest

from fused_memory.server.entities_gate import entities_gate
from fused_memory.utils.referent_resolution import _DECLARED_REFERENT_HINT

GROUP = 'dark_factory'

#: Content whose prose names exactly one own-project referent, Task 3127.
_CITES_3127 = 'the fix for Task 3127 landed'
#: Content naming no referent of any kind — the scan comes back empty, which
#: gamma treats as UNINFORMATIVE rather than contradictory.
_CITES_NOTHING = 'the merge-lane hardening work'


class TestAbsenceIsNeverRejected:
    """The tri-state's two non-declaring arms, and the corroborating one.

    ``None`` short-circuits before any scan; ``[]`` is a REAL declaration
    ("considered, none apply") that still must not reject; and a declaration
    the scan cannot contradict is honoured.
    """

    def test_none_with_a_referent_in_the_prose_returns_none(self):
        assert entities_gate(None, content=_CITES_3127, group_id=GROUP) is None

    def test_none_with_no_referent_in_the_prose_returns_none(self):
        assert entities_gate(None, content=_CITES_NOTHING, group_id=GROUP) is None

    def test_the_empty_declaration_does_not_reject_even_when_the_scan_found_one(self):
        """``[]`` is tri-state arm 2, not a collapsed ``None``.

        The scan DID find Task 3127 and the declaration omits it — and that is
        still not a conflict.  Resolved decision 3 rejects on conflict, never
        on absence, and gamma's ``_conflicting_referents`` iterates the
        DECLARED entries, so an empty declaration has nothing to contradict.
        """
        assert entities_gate([], content=_CITES_3127, group_id=GROUP) is None

    def test_a_declaration_the_scan_cannot_speak_to_returns_none(self):
        """An empty per-kind scan is uninformative, never contradictory.

        The scanner's documented blind spots (bare-digit node names, title-only
        references, Greek-letter aliases) mean silence is silence.  Rejecting
        here would bounce an honest write whose prose simply named its task in
        a shape the scanner cannot see.
        """
        assert entities_gate(
            [{'kind': 'task', 'id': 3129}], content=_CITES_NOTHING, group_id=GROUP,
        ) is None

    def test_a_corroborated_declaration_returns_none(self):
        assert entities_gate(
            [{'kind': 'task', 'id': 3127}], content=_CITES_3127, group_id=GROUP,
        ) is None

    def test_agent_id_is_optional_on_the_accepted_path(self):
        """The parameter defaults, so a caller with no identity still gates."""
        assert entities_gate([], content=_CITES_3127, group_id=GROUP) is None


class TestMalformedDeclarationsAreRejected:
    """Gamma's ``InputValidationError`` contract is TOTAL, and lands here.

    Gamma states it wrote that contract so "delta's ``except
    InputValidationError`` gate catches all of them".  Each shape below leaves
    ``_declared_referents`` as exactly ``InputValidationError``, and the gate
    turns it into the flat house-shape block the sibling pre-service guards in
    ``server/tools.py`` already return.
    """

    @pytest.mark.parametrize(
        'declared',
        [
            pytest.param('task 3127', id='a-str-not-a-list'),
            pytest.param([{'id': 'abc'}], id='non-digit-id'),
            pytest.param([{'id': 3127, 'projectId': 'reify'}], id='unrecognized-key'),
            pytest.param([{'id': True}], id='bool-id'),
            pytest.param([{'id': 3127, 'project_id': 'task'}], id='task-vocabulary-qualifier'),
        ],
    )
    def test_each_malformed_shape_blocks_the_write(self, declared):
        block = entities_gate(
            declared, content=_CITES_3127, group_id=GROUP, agent_id='claude-interactive',
        )

        assert isinstance(block, dict), f'expected a block, got {block!r}'
        assert block['error_type'] == 'ValidationError', f'{block!r}'
        assert isinstance(block['error'], str) and block['error'], f'{block!r}'
        assert block['agent_id'] == 'claude-interactive', f'{block!r}'
        assert block['content_excerpt'] == _CITES_3127[:200], f'{block!r}'

    @pytest.mark.parametrize(
        'declared',
        [
            pytest.param('task 3127', id='a-str-not-a-list'),
            pytest.param([{'id': 'abc'}], id='non-digit-id'),
            pytest.param([{'id': 3127, 'projectId': 'reify'}], id='unrecognized-key'),
            pytest.param([{'id': True}], id='bool-id'),
            pytest.param([{'id': 3127, 'project_id': 'task'}], id='task-vocabulary-qualifier'),
        ],
    )
    def test_the_remediation_reaches_the_agent_inside_the_message(self, declared):
        """Gamma single-sources the accepted shape into the exception message.

        Imported from production rather than re-spelled here: a second copy of
        the hint text is exactly the lockstep duplication this batch keeps
        guarding against, and it would let the two drift silently.
        """
        block = entities_gate(declared, content=_CITES_3127, group_id=GROUP)

        assert _DECLARED_REFERENT_HINT in block['error'], f'{block!r}'

    def test_the_malformed_block_carries_no_separate_hint_key(self):
        """An exception has no room for a structured key, so gamma folds the
        remediation into the message.  A second copy under ``hint`` would be
        the same text maintained at two sites."""
        block = entities_gate([{'id': 'abc'}], content=_CITES_3127, group_id=GROUP)

        assert 'hint' not in block, f'{block!r}'

    def test_a_falsy_non_list_is_rejected_rather_than_read_as_absence(self):
        """The short-circuit is ``entities is None``, never ``not entities``.

        ``''``/``{}``/``0`` are wiring bugs, not declarations.  Treating them
        as absence would silently downgrade the write's referent source and
        lose the caller's intent invisibly — the silent degradation this
        repo's loud-over-silent norm forbids.
        """
        for falsy in ('', {}, 0):
            block = entities_gate(falsy, content=_CITES_3127, group_id=GROUP)
            assert isinstance(block, dict), f'{falsy!r} was read as absence: {block!r}'
            assert block['error_type'] == 'ValidationError', f'{falsy!r}: {block!r}'

    def test_the_malformed_block_is_json_safe(self):
        block = entities_gate([{'id': 'abc'}], content=_CITES_3127, group_id=GROUP)

        assert json.loads(json.dumps(block)) == block
