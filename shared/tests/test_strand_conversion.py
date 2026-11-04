"""The typed ``metadata.strand_converted`` stamp and its write-boundary registration."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from shared.strand_conversion import StrandConverted
from shared.task_metadata import parse_metadata

_AT = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
_SHAPE = 'in-progress|false|on_main|true|-'
_IDS = ('esc-7-1', 'esc-7-3')


def _stamp() -> StrandConverted:
    return StrandConverted(at=_AT, shape=_SHAPE, pinning_escalation_ids=_IDS)


def _slice(**overrides: object) -> dict:
    return {'at': _AT.isoformat(), 'shape': _SHAPE, 'pinning_escalation_ids': list(_IDS), **overrides}


def test_to_metadata_round_trips() -> None:
    stamp = _stamp()

    assert stamp.to_metadata() == {'strand_converted': _slice()}
    assert StrandConverted.model_validate(stamp.to_metadata()['strand_converted']) == stamp


@pytest.mark.parametrize(
    ('field', 'overrides'),
    [
        ('pinning_escalation_ids', {'pinning_escalation_ids': ()}),
        ('at', {'at': datetime(2026, 10, 13, 12, 0)}),
        ('shape', {'shape': ''}),
    ],
)
def test_invariant_breach_names_the_field(field: str, overrides: dict) -> None:
    fields = {'at': _AT, 'shape': _SHAPE, 'pinning_escalation_ids': _IDS, **overrides}

    with pytest.raises(ValidationError) as exc_info:
        StrandConverted(**fields)

    assert field in str(exc_info.value)


@pytest.mark.parametrize('field', ['at', 'shape', 'pinning_escalation_ids'])
def test_is_frozen(field: str) -> None:
    stamp = _stamp()

    with pytest.raises(ValidationError):
        setattr(stamp, field, 'x')


def test_a_valid_stamp_passes_the_write_boundary_without_warnings() -> None:
    _, warnings = parse_metadata(_stamp().to_metadata(), direction='write', enforce=True)

    assert warnings == []


def test_a_slice_nothing_pins_is_refused_at_the_write_boundary() -> None:
    malformed = {'strand_converted': _slice(pinning_escalation_ids=[])}

    with pytest.raises(ValidationError):
        parse_metadata(malformed, direction='write', enforce=True)

    _, warnings = parse_metadata(malformed, direction='write', enforce=False)
    slice_codes = [w.code for w in warnings if w.field == 'strand_converted']
    assert slice_codes
    assert 'unknown_key' not in slice_codes


def test_registration_holds_in_a_process_that_imports_only_task_metadata() -> None:
    script = textwrap.dedent(
        f"""
        from shared.task_metadata import parse_metadata
        _, warnings = parse_metadata(
            {{'strand_converted': {_slice()!r}}}, direction='write', enforce=True,
        )
        print(len(warnings))
        """
    )

    result = subprocess.run(
        [sys.executable, '-c', script], capture_output=True, text=True, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '0'
