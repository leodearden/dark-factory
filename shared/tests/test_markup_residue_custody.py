"""Tests for ``shared.markup_residue_custody`` — the words every residue sink
states about a residue record's call and about who still holds its payload.
"""

from __future__ import annotations

import pytest

from shared.markup_residue_custody import (
    MARKUP_REFUSAL_CUSTODY_CAVEAT,
    MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT,
    MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT,
    ResidueCall,
    residue_prose,
)
from shared.mcp_markup_middleware import (
    MARKUP_RESIDUE_ERROR_TYPES,
    MARKUP_UNRECOVERED_PARAM_ERROR_TYPE,
    MARKUP_UNREPAIRABLE_ERROR_TYPE,
)

_FUTURE_KIND = 'mcp_markup_something_new'


class TestTheRefusedKind:
    def test_it_states_a_refused_call(self) -> None:
        prose = residue_prose({'error_type': MARKUP_UNREPAIRABLE_ERROR_TYPE})

        assert prose.call is ResidueCall.REFUSED
        assert prose.outcome.startswith('THE CALL WAS REFUSED')
        assert 'FORWARDED' not in prose.outcome
        assert 'landed' not in prose.outcome
        assert prose.custody == MARKUP_REFUSAL_CUSTODY_CAVEAT


class TestTheDroppedSliceKind:
    def test_it_states_a_forwarded_call_missing_the_value(self) -> None:
        prose = residue_prose({'error_type': MARKUP_UNRECOVERED_PARAM_ERROR_TYPE})

        assert prose.call is ResidueCall.FORWARDED_WITHOUT_PARAM
        assert 'THE CALL WAS FORWARDED WITHOUT THIS PARAMETER' in prose.outcome
        assert MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT in prose.outcome
        assert 'REFUSED' not in prose.outcome
        assert prose.custody == MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT


@pytest.mark.parametrize('error_type', sorted(MARKUP_RESIDUE_ERROR_TYPES))
def test_every_kind_the_middleware_emits_has_text(error_type: str) -> None:
    assert residue_prose({'error_type': error_type}).call is not ResidueCall.UNRECOGNISED


class TestAnUnrecognisedKind:
    @pytest.mark.parametrize(
        'record',
        [
            {
                'error_type': _FUTURE_KIND,
                'summary': 'A kind from the future.',
                'suggested_action': 'Recover it.',
            },
            {'summary': 'A kind from the future.', 'suggested_action': 'Recover it.'},
        ],
        ids=['unknown-kind', 'no-kind'],
    )
    def test_it_claims_no_outcome_and_renders_the_middleware_words(
        self, record: dict[str, str]
    ) -> None:
        prose = residue_prose(record)

        assert prose.call is ResidueCall.UNRECOGNISED
        assert f'UNRECOGNISED RESIDUE KIND {record.get("error_type")!r}' in prose.outcome
        assert "summary: 'A kind from the future.'" in prose.outcome
        assert "suggested_action: 'Recover it.'" in prose.outcome
        assert 'THE CALL WAS REFUSED' not in prose.outcome
        assert 'THE CALL WAS FORWARDED' not in prose.outcome
        assert prose.custody
        assert prose.custody not in {
            MARKUP_REFUSAL_CUSTODY_CAVEAT,
            MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT,
        }

    def test_the_unknown_kind_is_named_by_its_repr(self) -> None:
        prose = residue_prose({'error_type': _FUTURE_KIND})

        assert "UNRECOGNISED RESIDUE KIND 'mcp_markup_something_new'" in prose.outcome
