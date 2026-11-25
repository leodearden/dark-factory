"""Tests for ``shared.markup_residue_custody`` — the words every residue sink
states about a residue record's call and about who still holds its payload.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping

import pytest

from shared.markup_residue_custody import (
    MARKUP_REFUSAL_CUSTODY_CAVEAT,
    MARKUP_UNRECOVERED_PARAM_CUSTODY_CAVEAT,
    MARKUP_UNRECOVERED_PARAM_OUTCOME_CAVEAT,
    RESIDUE_UNPRESERVED_LOG_EVENT,
    ResidueCall,
    log_unpreserved_residue,
    residue_prose,
)
from shared.mcp_markup_middleware import (
    MARKUP_RECURRENCE_ROUTING,
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

    def test_it_names_where_the_missing_text_belongs(self) -> None:
        prose = residue_prose({'error_type': _FUTURE_KIND})

        assert 'shared/src/shared/markup_residue_custody.py::residue_prose' in prose.outcome
        assert 'this module' not in prose.outcome


_LOGGER_NAME = 'test.markup_residue_custody'
_RAW = 'THE ONLY COPY\n"quoted" tail'
_WHY = 'the test queue could not take it'


def _residue(error_type: str) -> dict[str, str]:
    return {'error_type': error_type, 'tool': 'tool', 'field': 'field', 'raw_value': _RAW}


def _log(
    record: dict[str, str],
    *,
    exc_info: bool = False,
    extra: Mapping[str, object] | None = None,
) -> None:
    log_unpreserved_residue(
        logging.getLogger(_LOGGER_NAME), record, why=_WHY, exc_info=exc_info, extra=extra
    )


def _captured(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER_NAME]


class TestTheUnpreservedResidueLogLine:
    @pytest.mark.parametrize('error_type', sorted(MARKUP_RESIDUE_ERROR_TYPES))
    def test_one_error_line_carries_the_whole_record(
        self, caplog: pytest.LogCaptureFixture, error_type: str
    ) -> None:
        record = _residue(error_type)

        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            _log(record)

        [line] = _captured(caplog)
        assert line.levelno == logging.ERROR
        assert line.__dict__['residue_record'] == dict(record)
        assert line.__dict__['residue_record']['raw_value'] == _RAW
        message = line.getMessage()
        assert RESIDUE_UNPRESERVED_LOG_EVENT in message
        assert repr(_RAW) in message
        assert repr(error_type) in message
        assert 'tool.field' in message
        assert _WHY in message
        assert MARKUP_RECURRENCE_ROUTING in message

    def test_caller_extras_ride_beside_the_record(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        record = _residue(MARKUP_UNREPAIRABLE_ERROR_TYPE)

        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            _log(record, extra={'residue_tier': 'unpreserved'})

        [line] = _captured(caplog)
        assert line.__dict__['residue_tier'] == 'unpreserved'
        assert line.__dict__['residue_record'] == dict(record)

    def test_the_failure_that_lost_the_record_is_attached(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        record = _residue(MARKUP_UNREPAIRABLE_ERROR_TYPE)

        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            try:
                raise OSError(28, 'No space left on device')
            except OSError:
                _log(record, exc_info=True)

        [line] = _captured(caplog)
        assert line.exc_info is not None
        assert line.exc_info[0] is OSError

    def test_the_record_is_not_mutated(self, caplog: pytest.LogCaptureFixture) -> None:
        record = _residue(MARKUP_UNRECOVERED_PARAM_ERROR_TYPE)
        before = copy.deepcopy(record)

        with caplog.at_level(logging.ERROR, logger=_LOGGER_NAME):
            _log(record, extra={'residue_tier': 'unpreserved'})

        assert record == before
